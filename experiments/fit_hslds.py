import os, pickle, copy

import numpy as np
np.random.seed(0)


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.gridspec as gridspec
import matplotlib.animation as manimation

from hips.plotting.colormaps import gradient_cmap
from hips.plotting.sausage import sausage_plot
from hips.plotting.layout import create_axis_at_location

# Come up with a set of colors
import seaborn as sns
sns.set_style("white")
# sns.set_context("paper")

color_names = ["windows blue",
               "red",
               "amber",
               "faded green",
               "dusty purple",
               "orange",
               "clay",
               "pink",
               "greyish",
               "mint",
               "light cyan",
               "steel blue",
               "forest green",
               "pastel purple",
               "salmon",
               "dark brown"]

colors = sns.xkcd_palette(color_names)
cmap = gradient_cmap(colors)

import importlib
from sklearn.decomposition import PCA

from pybasicbayes.distributions import Gaussian, Regression, DiagonalRegression
from pybasicbayes.util.text import progprint_xrange
from pyhsmm.util.general import relabel_by_usage, relabel_by_permutation
from pyslds.util import get_empirical_ar_params



import zimmer.io
importlib.reload(zimmer.io)
from zimmer.io import WormData, N_worms, find_shared_neurons, load_key

import zimmer.states
importlib.reload(zimmer.states)

import zimmer.models
importlib.reload(zimmer.models)
from zimmer.models import HierarchicalWeakLimitStickyHDPHMMSLDS

from zimmer.emissions import HierarchicalDiagonalRegressionFixedScale
from zimmer.emissions import HierarchicalDiagonalRegressionTruncatedScale

import zimmer.plotting
importlib.reload(zimmer.plotting)
from zimmer.plotting import plot_1d_continuous_states, plot_3d_continuous_states, plot_vector_field_3d


from zimmer.util import states_to_changepoints

# IO
run_num = 1
results_dir = os.path.join("results", "01_27_17", "run{:03d}".format(run_num))
assert os.path.exists(results_dir)

# Hyperparameters
Nmax = 15      # number of latent discrete states
D_latent = 3   # latent linear dynamics' dimension
D_in = 1       # number of input dimensions

alpha = 3.     # Transition matrix concentration
gamma = 3.0    # Base state concentration
kappa = 100.   # Stickiness parameter

# Inference parameters
N_samples = 500

### Helper functions
def cached(results_name):
    def _cache(func):
        def func_wrapper(*args, **kwargs):
            results_file = os.path.join(results_dir, results_name)
            if not results_file.endswith(".pkl"):
                results_file += ".pkl"

            if os.path.exists(results_file):
                with open(results_file, "rb") as f:
                    results = pickle.load(f)
            else:
                assert os.path.exists(results_dir)
                results = func(*args, **kwargs)
                with open(results_file, "wb") as f:
                    pickle.dump(results, f)

            return results
        return func_wrapper
    return _cache


### IO
def load_data():
    # Load the data
    worm_datas = [WormData(i, name="worm{}".format(i)) for i in range(N_worms)]
    Ns = np.array([wd.N for wd in worm_datas])
    Ts = np.array([wd.T for wd in worm_datas])
    Ys = [wd.dff_deriv for wd in worm_datas]

    # Get the "true" discrete states as labeled by Zimmer
    z_true = [wd.zimmer_states for wd in worm_datas]
    perm_z_true, newlabels = relabel_by_usage(z_true, return_mapping=True)

    # Get the key
    z_key = load_key()
    perm_z_key = [z_key[i] for i in np.argsort(newlabels)]

    # Get the names of the neurons
    all_neuron_names = np.unique(np.concatenate([wd.neuron_names for wd in worm_datas]))
    N_neurons = all_neuron_names.size
    print("{} neurons across all {} worms".format(N_neurons, N_worms))

    # Find the shared neurons
    shared_neurons = find_shared_neurons(worm_datas)
    print("{} shared neurons".format(len(shared_neurons)))
    Ys_shared = []
    for wd, Y in zip(worm_datas, Ys):
        shared_indices = wd.find_neuron_indices(shared_neurons)
        Ys_shared.append(Y[:, shared_indices])

    # Construct a big dataset with all neurons for each worm
    datasets = []
    masks = []
    for wd in worm_datas:
        this_dataset = np.zeros((wd.T, N_neurons))
        this_mask = np.zeros((wd.T, N_neurons), dtype=bool)
        indices = wd.find_neuron_indices(all_neuron_names)
        for n, index in enumerate(indices):
            if index is not None:
                this_dataset[:,n] = wd.dff_deriv[:, index]
                this_mask[:,n] = True

        datasets.append(this_dataset)
        masks.append(this_mask)

    return perm_z_true, perm_z_key, N_neurons, Ts, all_neuron_names, \
           datasets, masks, \
           Ys, Ys_shared, shared_neurons


### Fitting
@cached("pca")
def fit_pca(Ys_shared):
    # Try to reproduce their plot of PC's over time
    pca = PCA(n_components=3, whiten=True)
    pca.fit(np.vstack(Ys_shared))

    x_inits = [pca.transform(Y) for Y in Ys_shared]
    C_init = pca.components_.T
    return x_inits, C_init

def make_hslds(N_neurons, datasets, masks, Ts, z_inits=None, x_inits=None, fixed_scale=True):
    dynamics_hypparams = \
        dict(nu_0=D_latent + D_in + 2,
             S_0=np.eye(D_latent),
             M_0=np.zeros((D_latent, D_latent + D_in)),
             K_0=np.eye(D_latent + D_in),
             affine=False)

    # dynamics_hypparams = get_empirical_ar_params(
    #     [np.hstack((np.vstack(pca_trajs)[:,:P], np.ones((sum(Ts), 1))))],
    #     dynamics_hypparams)

    dynamics_distns = [
        Regression(
            A=np.hstack((0.99 * np.eye(D_latent), np.zeros((D_latent, D_in)))),
            sigma=np.eye(D_latent),
            **dynamics_hypparams)
        for _ in range(Nmax)]

    # One emission distribution per "neuron," where some neurons
    # are observed in one worm but not another.
    if fixed_scale:
        emission_distns = \
            HierarchicalDiagonalRegressionFixedScale(
                N_neurons, D_latent + D_in, N_worms)
    else:
        emission_distns = \
            HierarchicalDiagonalRegressionTruncatedScale(
                N_neurons, D_latent + D_in, N_worms, smin=0.75, smax=1.25)



    init_dynamics_distns = [
        Gaussian(nu_0=D_latent + 2, sigma_0=3. * np.eye(D_latent), mu_0=np.zeros(D_latent), kappa_0=0.01)
        for _ in range(Nmax)]

    model = HierarchicalWeakLimitStickyHDPHMMSLDS(
        init_dynamics_distns=init_dynamics_distns,
        dynamics_distns=dynamics_distns,
        emission_distns=emission_distns,
        alpha=alpha,
        gamma=gamma,
        kappa=kappa,
        init_state_distn='uniform')

    # Add the data
    for worm, (dataset, mask) in enumerate(zip(datasets, masks)):
        T = Ts[worm]
        inputs = np.ones((T, D_in))
        model.add_data(data=dataset, mask=mask, group=worm, inputs=inputs)

        # Initialize continuous latent states
        if x_inits is not None:
            model.states_list[-1].gaussian_states = x_inits[worm][:, :D_latent]

        # Initialize discrete latent states
        if z_inits is not None:
            model.states_list[-1].stateseq = z_inits[worm]
        else:
            runlen = 10
            z0 = np.random.choice(Nmax, size=(T//10))
            z0 = np.repeat(z0, runlen)
            z0 = z0[:T] if len(z0) > T else z0
            z0 = np.concatenate((z0, z0[-1] * np.ones(T-len(z0))))
            z0 = z0.astype(np.int32)
            assert len(z0) == T
            model.states_list[-1].stateseq = z0

    # Resample parameters once to be consistent with x_init
    model.resample_parameters()

    return model

@cached("fit_hslds")
def fit_hslds(model):
    # Fit the model with MCMC
    def evaluate(model):
        ll = model.log_likelihood()
        stateseqs = copy.deepcopy(model.stateseqs)
        return ll, stateseqs

    def update(model):
        model.resample_model()
        return evaluate(model)

    smpls = [update(model) for itr in progprint_xrange(N_samples)]

    # Convert the samples into numpy arrays for faster pickling
    lls, raw_z_smpls = zip(*smpls)
    lls = np.array(lls)

    # stateseqs is a list of lists of arrays of "shape" (N_samples x N_worms x T_worm)
    # get one list of arrays for each worm
    z_smpls = []
    for w in range(N_worms):
        z_smpls_w = np.array([z_smpl[w] for z_smpl in raw_z_smpls])
        z_smpls.append(z_smpls_w)

    perm_z_smpls, newlabels = relabel_by_usage(z_smpls, return_mapping=True)
    perm_dynamics_distns = [model.dynamics_distns[i] for i in np.argsort(newlabels)]

    # TODO: Permute the transition matrix

    # Compute the smoothed continuous state trajectories
    for states in model.states_list:
        states.info_E_step()

    z_finals = [relabel_by_permutation(s.stateseq, newlabels) for s in model.states_list]
    x_finals = [s.smoothed_mus for s in model.states_list]
    sigma_x_finals = [s.smoothed_sigmas for s in model.states_list]

    return model, lls, perm_z_smpls, perm_dynamics_distns, z_finals, x_finals, sigma_x_finals

### Plotting
def plot_identified_neurons(datasets):
    dsizes = [
        list(map(lambda d: np.size(d) > 1, dataset))
        for dataset in datasets
        ]
    dsizes = np.array(dsizes)

    # Show the 60 labeled neurons that show up in one or more worms
    plt.figure(figsize=(10, 5))
    plt.imshow(dsizes[:, :60], aspect="auto", interpolation="nearest")
    plt.grid("on")
    plt.xticks(np.arange(60) + 0.5, all_neuron_names[:60], rotation=90)
    plt.yticks(np.arange(N_worms) + 0.5, np.arange(N_worms) + 1)
    plt.ylabel("Worm")
    plt.xlabel("Neuron")
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "identified_neurons.pdf"))

def plot_continuous_states(x, z=None):
    if z is None:
        z = np.zeros(x.shape[0], dtype=int)
    cps = states_to_changepoints(z)

    plt.figure(figsize=(10, 5))
    for cp_start, cp_stop in zip(cps[:-1], cps[1:]):
        plt.subplot(121)
        plt.plot(x[cp_start:cp_stop + 1, 0],
                 x[cp_start:cp_stop + 1, 1],
                 '-', color=colors[z[cp_start]])
        plt.subplot(122)
        plt.plot(x[cp_start:cp_stop + 1, 1],
                 x[cp_start:cp_stop + 1, 2],
                 '-', color=colors[z[cp_start]])

    plt.subplot(121)
    plt.xlabel("$x_1$", fontsize=15)
    plt.ylabel("$x_2$", fontsize=15)

    plt.subplot(122)
    plt.xlabel("$x_2$", fontsize=15)
    plt.ylabel("$x_3$", fontsize=15)


def plot_discrete_state_samples(z_smpls, z_true):
    # Plot the true and inferred state sequences
    plt_slice = (0, 3000)
    fig = plt.figure(figsize=(10, 5))
    gs = gridspec.GridSpec(7, 1)

    ax1 = fig.add_subplot(gs[:-2])
    ax2 = fig.add_subplot(gs[-2])
    ax3 = fig.add_subplot(gs[-1])

    assert len(colors) > Nmax

    im = ax1.matshow(z_smpls, aspect='auto', cmap=cmap, vmin=0, vmax=len(colors) - 1)
    ax1.autoscale(False)
    ax1.set_xticks([])
    ax1.set_yticks([0, N_samples])
    ax1.set_ylabel("Iteration")
    ax1.set_xlim(plt_slice)
    ax1.set_xticks(plt_slice)

    ax2.matshow(z_smpls[-1][None, :], aspect='auto', cmap=cmap, vmin=0, vmax=len(colors) - 1)
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.set_ylabel("Final")
    ax2.set_xlim(plt_slice)

    ax3.matshow(z_true[None, :], aspect='auto', cmap=cmap, vmin=0, vmax=len(colors) - 1)
    ax3.set_xticks([])
    ax3.set_yticks([])
    ax3.set_ylabel("Manual")
    ax3.set_xlabel("Labels per time bin")
    ax3.set_xlim(plt_slice)

    plt.savefig(os.path.join(results_dir, "discrete_state_samples.pdf"))

def plot_changepoint_prs(z_smpls, z_true, title=None, plt_slice=(0, 3239)):
    # Plot the true and inferred state sequences

    fig = plt.figure(figsize=(5.5, 3))

    ax1 = create_axis_at_location(fig, 1.0, 1.45, 4.25, 1.25)
    ax2 = create_axis_at_location(fig, 1.0, .95, 4.25, 0.40)
    ax3 = create_axis_at_location(fig, 1.0, 0.45, 4.25, 0.40)

    # gs = gridspec.GridSpec(5, 1)
    # ax1 = fig.add_subplot(gs[:-2])
    # ax2 = fig.add_subplot(gs[-2])
    # ax3 = fig.add_subplot(gs[-1])

    im = ax1.matshow(z_smpls, aspect='auto', cmap=cmap, vmin=0, vmax=len(colors) - 1)
    ax1.autoscale(False)
    ax1.set_xticks([])
    ax1.set_yticks([0, N_samples])
    ax1.set_ylabel("Iteration", labelpad=13)
    ax1.set_xlim(plt_slice)
    # ax1.set_xticks(plt_slice)
    ax1.set_xticks([])

    if title is not None:
        ax1.set_title(title)

    # Compute changepoint probability
    ischangepoint = lambda z: np.concatenate(([0], np.diff(z) != 0))
    sampled_changepoints = np.array([ischangepoint(z) for z in z_smpls[-250:]])
    changepoint_pr = np.mean(sampled_changepoints, axis=0)
    ax2.plot(changepoint_pr, '-k', lw=0.5)
    ax2.set_xticks([])
    ax2.set_yticks([0, .5, 1])
    ax2.set_ylabel("CP Pr.", labelpad=20, rotation=0)
    ax2.set_xlim(plt_slice)

    ax3.matshow(z_true[None, :], aspect='auto', cmap=cmap, vmin=0, vmax=len(colors) - 1)
    ax3.set_yticks([])
    ax3.set_ylabel("Kato et al", labelpad=35, rotation=0)
    ax3.set_xlabel("Time")
    ax3.set_xlim(plt_slice)
    # ax3.set_xticks(plt_slice)
    ax3.xaxis.tick_bottom()

    plt.savefig(os.path.join(results_dir, "discrete_state_cps_worm1.pdf"))

# def plot_all_continuous_states_3d(x, z, figsize=(4,4)):
#         fig = plt.figure(figsize=figsize)
#         ax = fig.add_subplot(111, projection="3d")
#         plot_3d_continuous_states(x_finals[w], z[w], colors=colors, ax=ax)
#         fig.savefig(os.path.join(results_dir, "x_3d_worm{}.pdf".format(w)))
#         plt.close(fig)

def plot_pca_trajectories(worm, Y):
    # Try to reproduce their plot of PC's over time
    n_comps = 25
    pca = PCA(n_components=n_comps, whiten=True)
    pca.fit(Y)

    x_pca = pca.transform(Y)

    fig = plt.figure(figsize=(1.5,3.))
    ax = create_axis_at_location(fig, 0.2, 1.5, 1.1, 1.25, projection="3d")
    # Color denotes our inferred latent discrete state
    ax.plot(x_pca[:,0], x_pca[:,1], x_pca[:,2], lw=0.5, alpha=0.75, ls='-', color=colors[0])
    ax.set_xlabel("$x_1$", labelpad=-12)
    ax.set_ylabel("$x_2$", labelpad=-12)
    ax.set_zlabel("$x_3$", labelpad=-12)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    ax.set_title("Worm {}".format(worm+1))

    # Plot percent variance explained
    ax = create_axis_at_location(fig, .45, 0.4, .9, .9, box=False)
    ax.patch.set_alpha(0.0)
    # Color denotes our inferred latent discrete state
    ax.bar(np.arange(n_comps),  100 * pca.explained_variance_ratio_, color=colors[0])
    ax.set_xlabel("PC")
    ax.set_ylabel("% Variance")
    # ax.set_title("Worm {} Explained Variance".format(worm + 1))

    # plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "pca_trajectory_worm{}.pdf".format(worm)))


def plot_3d_dynamics(dynamics_distns, z, x):
    for k in range(Nmax):
        fig = plt.figure(figsize=(2.5, 2.5))
        # ax = fig.add_subplot(111, projection='3d')
        ax = create_axis_at_location(fig, 0.025, 0.025, 2.35, 2.35, projection="3d")
        plot_vector_field_3d(k, z, x, dynamics_distns, colors,
                             arrow_length_ratio=0.5, pivot="middle",
                             affine=(D_in == 1), ax=ax, lims=(-6, 6), alpha=0.8, N_plot=200, length=0.5, lw=0.75)
        ax.set_title("State {}".format(k+1))
        fig.savefig(os.path.join(results_dir, "dynamics_3d_{}.pdf".format(k)))
        plt.close(fig)

def make_dynamics_3d_movie(worm, z_finals, x_finals, perm_dynamics_distns):
    FFMpegWriter = manimation.writers['ffmpeg']
    metadata = dict(title='3d vector field')
    writer = FFMpegWriter(fps=15, bitrate=1024, metadata=metadata)

    # overlay = False
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    for ii in range(Nmax):
        plot_vector_field_3d(ii, z_finals[worm], x_finals[worm], perm_dynamics_distns, colors,
                             affine=(D_in == 1), ax=ax, lims=(-4, 4), alpha=0.75, N_plot=150, length=.2)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_zlabel("")
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])

    plt.title("Worm {}".format(worm + 1))

    def update_frame(i):
        # Rotate the xy plane
        ax.view_init(elev=30., azim=i)

        # Plot the trajectories
        #         plot_trajectories(i, lns)

    filename = os.path.join(results_dir, "dynamics_overlay_3d_worm{}.mp4".format(worm))
    with writer.saving(fig, filename, 150):
        for i in progprint_xrange(360):
            update_frame(i)
            writer.grab_frame()

def plot_state_dependent_neural_activity(zs, Ys_shared, shared_neurons):
    # Look at the average activity of the individual neurons
    # in each of the discrete states
    fs = 3.0
    Y = np.vstack(Ys_shared)
    z = np.concatenate(zs)
    cps = states_to_changepoints(z)
    cps_start, cps_stop = cps[:-1], cps[1:]
    N_shared = Ys_shared[0].shape[1]
    for kk in range(Nmax):
        # for kk in range(1):
        # Find all the instances of this state
        kk_start = cps_start[z[cps_start] == kk]
        kk_stop = cps_stop[z[cps_start] == kk]
        if len(kk_start) == 0:
            continue

        durs = kk_stop - kk_start
        max_dur = durs.max()
        tt = np.arange(max_dur) / fs

        # Find the mean of Y
        # aligned to entry into this state
        Ysmpls = []
        Ysum = np.zeros((max_dur, N_shared))
        count = np.zeros(max_dur)
        for start, stop in zip(kk_start, kk_stop):
            Ysum[:stop - start] += Y[start:stop]
            Ysmpls.append(Y[start:stop])
            count[:stop - start] += 1

        Ymean = Ysum / count[:, None]

        # Find the variance of Y
        # aligned to entry into this state
        Ysumsq = np.zeros((max_dur, N_shared))
        for start, stop in zip(kk_start, kk_stop):
            Ysumsq[:stop - start] += (Y[start:stop] - Ymean[:stop - start]) ** 2

        Yvar = Ysumsq / count[:, None]
        Ystd = np.sqrt(Yvar)

        # Compute limits
        lim = 1.1 * (abs(Ymean) + Ystd).max()

        # Plot the average response
        fig = plt.figure(figsize=(10, 6))
        for n, neuron_name in enumerate(shared_neurons):
            ax = fig.add_subplot(3, 5, n + 1)
            # Plot examples
            iis = np.random.choice(len(Ysmpls), size=min(25, len(Ysmpls)), replace=False)
            for ii in iis:
                yy = Ysmpls[ii]
                ax.plot(tt[:yy.shape[0]], yy[:, n], '-', color="gray", lw=0.5, alpha=0.5)

            # Plot the x axis
            plt.plot(tt, np.zeros_like(tt), '-k', lw=0.5)

            sausage_plot(tt, Ymean[:, n], Ystd[:, n], ax, color=colors[kk], alpha=0.4)
            ax.plot(tt, Ymean[:, n], '-', color=colors[kk], lw=2)

            ax.set_ylim(-lim, lim)
            ax.set_xlim(0, 5.)
            ax.set_title(neuron_name, fontsize=15)
            ax.set_xlabel("time (s)")
            ax.set_ylabel("$\\Delta F/F'$")
            fig.suptitle("State {}".format(kk + 1), fontsize=20)

        plt.tight_layout()
        plt.subplots_adjust(top=0.85)

        plt.savefig(os.path.join(results_dir, "neural_responses_{}.pdf".format(kk)))

def plot_subset_of_state_dependent_neural_activity(zs, Ys_shared):
    # Look at the average activity of the individual neurons
    # in each of the discrete states
    fs = 3.0
    Y = np.vstack(Ys_shared)
    z = np.concatenate(zs)
    cps = states_to_changepoints(z)
    cps_start, cps_stop = cps[:-1], cps[1:]
    N_shared = Ys_shared[0].shape[1]
    neurons_to_plot = ["AVAL", "AVAR", "AVBL", "AVER", "RIML", "RIMR"]
    for kk in range(Nmax):
        # for kk in range(1):
        # Find all the instances of this state
        kk_start = cps_start[z[cps_start] == kk]
        kk_stop = cps_stop[z[cps_start] == kk]
        if len(kk_start) == 0:
            continue

        durs = kk_stop - kk_start
        max_dur = durs.max()
        tt = np.arange(max_dur) / fs

        # Find the mean of Y
        # aligned to entry into this state
        Ysmpls = []
        Ysum = np.zeros((max_dur, N_shared))
        count = np.zeros(max_dur)
        for start, stop in zip(kk_start, kk_stop):
            Ysum[:stop - start] += Y[start:stop]
            Ysmpls.append(Y[start:stop])
            count[:stop - start] += 1

        Ymean = Ysum / count[:, None]

        # Find the variance of Y
        # aligned to entry into this state
        Ysumsq = np.zeros((max_dur, N_shared))
        for start, stop in zip(kk_start, kk_stop):
            Ysumsq[:stop - start] += (Y[start:stop] - Ymean[:stop - start]) ** 2

        Yvar = Ysumsq / count[:, None]
        Ystd = np.sqrt(Yvar)

        # Compute limits
        lim = 1.1 * (abs(Ymean) + Ystd).max()

        # Plot the average response
        fig = plt.figure(figsize=(5.5, 1))
        for jj, neuron_name in enumerate(neurons_to_plot):
            n = shared_neurons.index(neuron_name)
            ax = fig.add_subplot(1, 6, jj + 1)
            # Plot examples
            iis = np.random.choice(len(Ysmpls), size=min(25, len(Ysmpls)), replace=False)
            for ii in iis:
                yy = Ysmpls[ii]
                ax.plot(tt[:yy.shape[0]], yy[:, n], '-', color="gray", lw=0.5, alpha=0.5)

            # Plot the x axis
            plt.plot(tt, np.zeros_like(tt), '-k', lw=0.5)

            sausage_plot(tt, Ymean[:, n], Ystd[:, n], ax, color=colors[kk], alpha=0.4)
            ax.plot(tt, Ymean[:, n], '-', color=colors[kk], lw=2)

            # ax.set_ylim(-lim, lim)
            ax.set_ylim(-0.15, 0.15)
            ax.set_xlim(0, 5.)
            ax.set_title(neuron_name, fontsize=8)
            # ax.set_xlabel("time (s)")
            # ax.set_ylabel("$\\Delta F/F'$")
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            # fig.suptitle("State {}".format(kk + 1), fontsize=20)

        plt.tight_layout()
        plt.subplots_adjust(top=0.85)

        plt.savefig(os.path.join(results_dir, "neural_responses_subset_{}.pdf".format(kk)))


def plot_smoothed_neural_activity(neurons, worm, hslds, x_finals, sigma_x_finals,
                                  Ys_shared, shared_neurons, all_neuron_names,
                                  z=None):
    fig = plt.figure(figsize=(5.5,4.))
    N = len(neurons)
    N_subplots = 2*N if z is None else 2*N + 1
    gs = gridspec.GridSpec(N_subplots,1)

    if z is not None:
        ax = fig.add_subplot(gs[0,0])
        ax.matshow(z[None, :], aspect='auto', cmap=cmap, vmin=0, vmax=len(colors) - 1)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_ylabel("$z_{1:T}$", labelpad=25, rotation=0)
        ax.set_title("Worm: {} Reconstructed Activity".format(worm + 1))
        offset = 1
    else:
        offset = 0

    for n,neuron in enumerate(neurons):
        ax = fig.add_subplot(gs[offset+2*n:offset+2*(n+1),0])

        # Plot the neuron's activity
        i_shared = shared_neurons.index(neuron)
        y = Ys_shared[worm][:,i_shared]
        T = y.size

        # Plot the smoothed activity
        x = x_finals[worm]
        i_all = np.where(all_neuron_names == neuron)[0][0]
        if D_in > 0:
            c = hslds.emission_distns[i_all].A[worm, 0, :-1]
            d = hslds.emission_distns[i_all].A[worm, 0, -1]
        else:
            c = hslds.emission_distns[i_all].A[worm, 0, :]
            d = 0
        sig_y = hslds.emission_distns[i_all].sigma[worm, 0, 0]

        y_smooth = x.dot(c) + d
        sigma_y_smooth = np.array([c.dot(sigma_x).dot(c) + sig_y for sigma_x in sigma_x_finals[worm]])
        # sigma_y_smooth = np.array([c.dot(sigma_x).dot(c) for sigma_x in sigma_x_finals[worm]])
        assert sigma_y_smooth.shape == (T,)

        sausage_plot(np.arange(T), y_smooth, np.sqrt(sigma_y_smooth), ax, color=colors[0], alpha=0.4)
        ax.plot(y_smooth, color=colors[0])
        ax.plot(y, '-k', lw=.5)

        ax.set_xlim(0, T)

        if n == N-1:
            ax.set_xlabel("Time")
        else:
            ax.set_xticklabels([])

        ax.set_ylabel(neuron)
        ax.set_ylim(-.2, .2)
        ax.set_yticks([-.1, 0, .1])

        if n == 0 and z is None:
            ax.set_title("Worm: {} Reconstructed Activity".format(worm+1, neuron))

        sns.despine()

    # plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "neural_activity_worm_{}.pdf".format(worm)))
    plt.close(fig)

def plot_all_neural_activity(worm, Ys_shared, shared_neurons):
    Y = Ys_shared[worm]
    T, N = Y.shape
    # N = 10
    ylim = abs(Y).max()

    sns.set_style("white")
    fig = plt.figure(figsize=(5.5,5))
    for n in range(N):
        ax = fig.add_subplot(N, 1, n+1)
        ax.plot(Y[:,n], '-k', lw=0.5)

        ax.set_xticks(np.arange(T, step=500))
        ax.set_xlim(0,T)

        ax.set_ylabel(shared_neurons[n], rotation=0, labelpad=10)
        # ax.yaxis.set_label_coords(-0.1, -0.5 * ylim)
        # ax.set_ylabel(" ", rotation=0)
        # ax.text(-100, 0, shared_neurons[n])
        ax.set_ylim(-1.1*ylim, 1.1*ylim)
        ax.set_yticklabels([])
        if n == N-1:
            ax.set_xlabel("Time")
        else:
            ax.set_xticklabels([])

        if n == 0:
            ax.set_title("Worm {} Recorded Activity".format(worm+1))

        sns.despine()

    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "shared_activity_worm{}.pdf".format(worm)))


def plot_neural_embedding(hslds, all_neuron_names, N_to_plot=60, worm=1):

    # Read in the true 1D locations
    import pandas as pd
    df = pd.read_csv("wormatlas_locations.csv")
    true_locs = np.array([df.location[df.name==name].values[0] for name in all_neuron_names[:N_to_plot]])
    true_locs = np.column_stack((true_locs, np.zeros(N_to_plot)))

    # Jitter
    true_locs[:,0] += 0.01 * np.random.randn(N_to_plot)
    true_locs[:,1] += 0.5 * np.random.randn(N_to_plot)
    # print(true_locs)

    fig = plt.figure(figsize=(4,4))
    gs = gridspec.GridSpec(5,1)
    ax1 = fig.add_subplot(gs[0,0])
    for name, true_loc in zip(all_neuron_names, true_locs):
        ax1.plot(true_loc[0], true_loc[1], 'o', markersize=5, markerfacecolor=cmap((true_loc[0] - 0.07) / 0.3 * 0.35))
    ax1.set_xlim(0.05,0.3)
    ax1.set_ylim(-3,3)
    ax1.set_yticks([])
    ax1.set_title("Soma Location (Head=0, Tail=1)")
    sns.despine(ax=ax1,top=True, right=True, left=True)

    # Get the embedding
    C = np.array([hslds.emission_distns[n]._A[0,:3]
                  for n in range(N_to_plot)])

    ax2 = fig.add_subplot(gs[1:,0], projection="3d")
    ax2.patch.set_alpha(0.0)
    for n in range(N_to_plot):
        cn = hslds.emission_distns[n]._A[0,:3]
        ax2.plot([C[n,0]], [C[n,1]], [C[n,2]], 'o', markersize=5,
                 markerfacecolor=cmap((true_locs[n,0] - 0.07) / 0.3 * 0.35))

    # Annotate a few neurons
    neurons_to_annotate = ["AVAL", "AVAR", "AVBL", "AVER", "RIML", "RIMR"]
    n_to_annotate = [np.where(all_neuron_names==name)[0][0] for name in neurons_to_annotate]
    # n_to_annotate = np.argsort(-C[:,0])[:4]
    for n in n_to_annotate:
        # print("annotating", all_neuron_names[n])
        ax2.text(C[n,0]+0.001, C[n,1]+0.001, C[n,2]-0.001,
                 all_neuron_names[n],
                 size=8, zorder=1000,
                 )

    lims = (-0.025, 0.025)
    ax2.set_xlim(lims)
    ax2.set_ylim(lims)
    ax2.set_zlim(lims)
    ax2.set_xticklabels([])
    ax2.set_yticklabels([])
    ax2.set_zticklabels([])
    ax2.set_xlabel("$x_1$", labelpad=-10)
    ax2.set_ylabel("$x_2$", labelpad=-10)
    ax2.set_zlabel("$x_3$", labelpad=-10)
    ax2.set_title("Inferred Embedding")



    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "neuron_embedding.pdf"))

if __name__ == "__main__":
    # Load the data
    z_trues, z_key, N_neurons, Ts, \
    all_neuron_names, datasets, masks, \
    Ys, Ys_shared, shared_neurons = \
        load_data()

    # for worm in range(N_worms):
    #     plot_all_neural_activity(worm, Ys_shared, shared_neurons)

    # Initialize with PCA
    x_inits, C_init = fit_pca(Ys_shared)

    # for worm in range(N_worms):
    #     plot_pca_trajectories(worm, Ys[worm])

    # Make and fit the hierarchical SLDS
    # hslds = make_hslds(N_neurons, datasets, Ts, x_inits=x_inits)
    # hslds, lls, z_smpls, dynamics_distns, z_finals, x_finals, sigma_x_finals = fit_hslds(hslds)

    hslds = make_hslds(N_neurons, datasets, masks, Ts, x_inits=x_inits, fixed_scale=False)
    hslds, lls, z_smpls, dynamics_distns, z_finals, x_finals, sigma_x_finals = fit_hslds(hslds)

    # plot_discrete_state_samples(z_smpls[1], z_trues[1])
    # plot_changepoint_prs(z_smpls[1], z_trues[1], title="Worm 2 Discrete States")

    # for worm in range(N_worms):
    #     fig = plt.figure(figsize=(2.5,2.5))
    #     ax = create_axis_at_location(fig, 0.025, 0.025, 2.35, 2.35, projection="3d")
    #     plot_3d_continuous_states(x_finals[worm], z_finals[worm],
    #                               ax=ax,
    #                               title="Worm {} Latent States".format(worm+1),
    #                               colors=colors, lw=0.5, alpha=0.75, figsize=(3,3),
    #                               results_dir=results_dir, filename="xs_3d_worm{}.pdf".format(worm))

    # Plot the state dynamics
    # plot_3d_dynamics(dynamics_distns, np.concatenate(z_finals), np.vstack(x_finals))

    # Make 3D animations of worm dynamics
    for w in range(N_worms):
        make_dynamics_3d_movie(w, z_finals, x_finals, dynamics_distns)


    # neurons_to_plot = ["AVAL", "AVAR", "AVBL", "AVER", "RIML", "RIMR"]
    # for worm in range(N_worms):
    #     plot_smoothed_neural_activity(neurons_to_plot, worm, hslds,
    #                                   x_finals, sigma_x_finals,
    #                                   Ys_shared, shared_neurons, all_neuron_names,
    #                                   z= z_finals[worm])


    # plot_neural_embedding(hslds, all_neuron_names)

    # plot_state_dependent_neural_activity(z_finals, Ys_shared, shared_neurons)
    # plot_subset_of_state_dependent_neural_activity(z_finals, Ys_shared)
    plt.show()