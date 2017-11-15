import os
import pickle

import numpy as np
import numpy.random as npr
npr.seed(1)

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.gridspec as gridspec
from matplotlib.font_manager import FontProperties

from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid1 import make_axes_locatable


import seaborn as sns
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
sns.set_style("white")
sns.set_context("paper")

from hips.plotting.colormaps import gradient_cmap
cmap = gradient_cmap(colors)
from hips.plotting.layout import create_axis_at_location

from pybasicbayes.util.text import progprint_xrange
from pybasicbayes.models import FactorAnalysis
from pybasicbayes.distributions import \
    Regression, Gaussian, DiagonalRegression, AutoRegression

from pybasicbayes.util.text import progprint_xrange

from pyhsmm.util.general import relabel_by_permutation, relabel_by_usage
from autoregressive.models import ARWeakLimitStickyHDPHMM
from pyslds.util import get_empirical_ar_params
from pylds.util import random_rotation

from pinkybrain.models import MixedEmissionHMMSLDS
from pyslds.models import WeakLimitStickyHDPHMMSLDS
from rslds.rslds import RecurrentSLDS
from rslds.nonconj_rslds import SoftmaxRecurrentOnlySLDS
from rslds.util import compute_psi_cmoments

from zimmer.io import WormData, N_worms, find_shared_neurons, load_kato_key
from zimmer.util import states_to_changepoints

### Global parameters
K = 5
D_latent = 3

# Specifying the number of iterations of the Gibbs sampler
N_iters = 1000

# Save / cache the outputs
runnum = 4
results_dir = os.path.join("results", "02_22_17", "run{:03d}".format(runnum))

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
    z_key = load_kato_key()
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
                this_dataset[:, n] = wd.dff_deriv[:, index]
                this_mask[:, n] = True

        datasets.append(this_dataset)
        masks.append(this_mask)

    return perm_z_true, perm_z_key, N_neurons, Ts, all_neuron_names, \
           datasets, masks, \
           Ys, Ys_shared, shared_neurons


### Plotting Code
def plot_dynamics(A, b=None, ax=None, plot_center=True,
                  xlim=(-4,4), ylim=(-3,3), npts=20,
                  color='r'):
    b = np.zeros((A.shape[0], 1)) if b is None else b
    x = np.linspace(*xlim, npts)
    y = np.linspace(*ylim, npts)
    X,Y = np.meshgrid(x,y)
    xy = np.column_stack((X.ravel(), Y.ravel()))

    # dydt_m = xy.dot(A.T) + b.T - xy
    dydt_m = xy.dot(A.T) + b.T - xy

    if ax is None:
        fig = plt.figure(figsize=(6,6))
        ax = fig.add_subplot(111)

    ax.quiver(xy[:, 0], xy[:, 1],
              dydt_m[:, 0], dydt_m[:, 1],
              color=color, alpha=1.0,
              headwidth=5.)

    # Plot the stable point
    if plot_center:
        try:
            center = -np.linalg.solve(A-np.eye(D_latent), b)
            ax.plot(center[0], center[1], 'o', color=color, markersize=8)
        except:
            print("Dynamics are not invertible!")

    ax.set_xlabel('$x_1$', fontsize=12, labelpad=10)
    ax.set_ylabel('$x_2$', fontsize=12, labelpad=10)

    return ax

def plot_all_dynamics(dynamics_distns,
                      filename=None):

    fig = plt.figure(figsize=(12,3))
    for k in range(K):
        ax = fig.add_subplot(1,K,k+1)
        plot_dynamics(dynamics_distns[k].A[:,:D_latent],
                      b=dynamics_distns[k].A[:,D_latent:],
                      plot_center=False,
                      color=colors[k % len(colors)], ax=ax)

    if filename is not None:
        fig.savefig(os.path.join(results_dir, filename))


def plot_most_likely_dynamics(
        reg, dynamics_distns,
        xlim=(-4, 4), ylim=(-3, 3),  nxpts=20, nypts=10,
        alpha=0.8,
        ax=None, figsize=(3,3)):

    x = np.linspace(*xlim, nxpts)
    y = np.linspace(*ylim, nypts)
    X, Y = np.meshgrid(x, y)
    xy = np.column_stack((X.ravel(), Y.ravel()))

    # Get the probability of each state at each xy location
    Ts = reg.get_trans_matrices(xy)
    prs = Ts[:,0,:]
    z = np.argmax(prs, axis=1)


    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)

    for k in range(K):
        A = dynamics_distns[k].A[:, :D_latent]
        b = dynamics_distns[k].A[:, D_latent:]
        dydt_m = xy.dot(A.T) + b.T - xy

        zk = z == k
        if zk.sum(0) > 0:
            ax.quiver(xy[zk, 0], xy[zk, 1],
                      dydt_m[zk, 0], dydt_m[zk, 1],
                      color=colors[k], alpha=alpha)

    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')

    plt.tight_layout()

    return ax

def plot_trans_probs(reg,
                     xlim=(-4,4), ylim=(-3,3), n_pts=50,
                     ax=None,
                     filename=None):
    XX,YY = np.meshgrid(np.linspace(*xlim,n_pts),
                        np.linspace(*ylim,n_pts))
    XY = np.column_stack((np.ravel(XX), np.ravel(YY)))

    W, b = reg.expected_W, reg.expected_b

    test_logpi = XY.dot(W) + b
    test_prs = np.exp(test_logpi)
    test_prs = test_prs / test_prs.sum(axis=1, keepdims=True)

    if ax is None:
        fig = plt.figure(figsize=(10,6))
        ax = fig.add_subplot(111)

    for k in range(K):
        start = np.array([1., 1., 1., 0.])
        end = np.concatenate((colors[k % len(colors)], [0.5]))
        cmap = gradient_cmap([start, end])
        im1 = ax.imshow(test_prs[:,k].reshape(*XX.shape),
                         extent=xlim + tuple(reversed(ylim)),
                         vmin=0, vmax=1, cmap=cmap)

        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        # ax.set_title("State {}".format(k+1))

    plt.tight_layout()
    return ax

def plot_trajectory(zhat, x, ax=None, ls="-", filename=None):
    zcps = np.concatenate(([0], np.where(np.diff(zhat))[0] + 1, [zhat.size]))
    if ax is None:
        fig = plt.figure(figsize=(4, 4))
        ax = fig.gca()
    for start, stop in zip(zcps[:-1], zcps[1:]):
        ax.plot(x[start:stop + 1, 0],
                x[start:stop + 1, 1],
                lw=1, ls=ls,
                color=colors[zhat[start] % len(colors)],
                alpha=1.0)

    # ax.set_xlabel('$x_1$', fontsize=12, labelpad=10)
    # ax.set_ylabel('$x_2$', fontsize=12, labelpad=10)
    if filename is not None:
        plt.savefig(filename)

    return ax

def plot_trajectory_and_probs(z, x,
                              ax=None,
                              trans_distn=None,
                              title=None,
                              filename=None,
                              **trargs):
    if ax is None:
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111)

    if trans_distn is not None:
        xlim = abs(x[:, 0]).max()
        xlim = (-xlim, xlim)
        ylim = abs(x[:, 0]).max()
        ylim = (-ylim, ylim)
        ax = plot_trans_probs(trans_distn, ax=ax,
                              xlim=xlim, ylim=ylim)
    plot_trajectory(z, x, ax=ax, **trargs)
    plt.tight_layout()
    plt.title(title)
    if filename is not None:
        plt.savefig(os.path.join(results_dir, filename))

    return ax


def plot_data(zhat, y, ax=None, ls="-", filename=None):
    zcps = np.concatenate(([0], np.where(np.diff(zhat))[0] + 1, [zhat.size]))
    if ax is None:
        fig = plt.figure(figsize=(4, 4))
        ax = fig.gca()
    for start, stop in zip(zcps[:-1], zcps[1:]):
        stop = min(y.shape[0], stop+1)
        ax.plot(np.arange(start, stop),
                y[start:stop ],
                lw=1, ls=ls,
                color=colors[zhat[start]],
                alpha=1.0)

    # ax.set_xlabel('$x_1$', fontsize=12, labelpad=10)
    # ax.set_ylabel('$x_2$', fontsize=12, labelpad=10)
    if filename is not None:
        plt.savefig(filename)

    return ax

def plot_separate_trans_probs(reg,
                              xlim=(-4,4), ylim=(-3,3), n_pts=100,
                              ax=None,
                              filename=None):
    XX,YY = np.meshgrid(np.linspace(*xlim,n_pts),
                        np.linspace(*ylim,n_pts))
    XY = np.column_stack((np.ravel(XX), np.ravel(YY)))

    D_reg = reg.D_in
    inputs = np.hstack((np.zeros((n_pts**2, D_reg-2)), XY))
    test_prs = reg.pi(inputs)

    if ax is None:
        fig = plt.figure(figsize=(12,3))

    for k in range(K):
        ax = fig.add_subplot(1,K,k+1)
        cmap = gradient_cmap([np.ones(3), colors[k]])
        im1 = ax.imshow(test_prs[:,k].reshape(*XX.shape),
                         extent=xlim + tuple(reversed(ylim)),
                         vmin=0, vmax=1, cmap=cmap)

        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im1, cax=cax, ax=ax)
        # ax.set_title("State {}".format(k+1))

    plt.tight_layout()
    return ax

def plot_discrete_state_samples(z_smpls, z_true):
    # Plot the true and inferred state sequences
    plt_slice = (0, 3000)
    fig = plt.figure(figsize=(10, 5))
    gs = gridspec.GridSpec(7, 1)

    ax1 = fig.add_subplot(gs[:-2])
    ax2 = fig.add_subplot(gs[-2])
    ax3 = fig.add_subplot(gs[-1])

    assert len(colors) > K

    im = ax1.matshow(z_smpls, aspect='auto', cmap=cmap, vmin=0, vmax=len(colors) - 1)
    ax1.autoscale(False)
    ax1.set_xticks([])
    ax1.set_yticks([0, len(z_smpls)])
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

def plot_3d_continuous_states(x, z, colors,
                              ax=None,
                              figsize=(2.5,2.5),
                              inds=(0,1,2),
                              title=None,
                              results_dir=".", filename=None,
                              **kwargs):

    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection="3d")

    cps = states_to_changepoints(z)

    # Color denotes our inferred latent discrete state
    for cp_start, cp_stop in zip(cps[:-1], cps[1:]):
        ax.plot(x[cp_start:cp_stop + 1, inds[0]],
                x[cp_start:cp_stop + 1, inds[1]],
                x[cp_start:cp_stop + 1, inds[2]],
                '-', marker='.', markersize=3,
                color=colors[z[cp_start]],
                **kwargs)

    ax.set_xlabel("$x_1$", labelpad=-10)
    ax.set_ylabel("$x_2$", labelpad=-10)
    ax.set_zlabel("$x_3$", labelpad=-10)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])

    if title is not None:
        ax.set_title(title)

    if filename is not None:
        plt.savefig(os.path.join(results_dir, filename))


def plot_3d_dynamics(dynamics_distns, z, x):
    for k in range(len(dynamics_distns)):
        fig = plt.figure(figsize=(2.5, 2.5))
        # ax = fig.add_subplot(111, projection='3d')
        ax = create_axis_at_location(fig, 0.025, 0.025, 2.35, 2.35, projection="3d")
        plot_vector_field_3d(k, z, x, dynamics_distns, colors,
                             arrow_length_ratio=0.5, pivot="middle",
                             affine=True, ax=ax, lims=(-6, 6), alpha=0.8,
                             N_plot=200, length=0.5, lw=0.75)
        ax.set_title("State {}".format(k+1))
        fig.savefig(os.path.join(results_dir, "dynamics_3d_{}.pdf".format(k)))
        plt.close(fig)

def plot_vector_field_3d(ii, z, x, perm_dynamics_distns, colors,
                         ax=None, affine=False, lims=(-3, 3), N_plot=500,
                         **kwargs):

    qargs = dict(arrow_length_ratio=0.25,
                 length=0.1,
                 alpha=0.5)
    qargs.update(kwargs)

    D = x.shape[1]
    ini = np.where(z == ii)[0]

    # Look at the projected dynamics under each model
    # Subsample accordingly
    if ini.size > N_plot:
        ini_inds = np.random.choice(ini.size, replace=False, size=N_plot)
        ini = ini[ini_inds]

    Ai = perm_dynamics_distns[ii].A[:, :D]
    bi = perm_dynamics_distns[ii].A[:, D] if affine else 0
    dxdt = x.dot(Ai.T) + bi - x

    # Create axis if not given
    if ax is None:
        fig = plt.figure(figsize=(5, 5))
        ax = fig.add_subplot(111, projection='3d')

    ax.quiver(x[ini, 0], x[ini, 1], x[ini, 2],
              dxdt[ini, 0], dxdt[ini, 1], dxdt[ini, 2],
              color=colors[ii],
              **qargs)
    ax.set_xlabel('$x_1$', labelpad=-10)
    ax.set_ylabel('$x_2$', labelpad=-10)
    ax.set_zlabel('$x_3$', labelpad=-10)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_zlim(lims)


### Factor Analysis and PCA for dimensionality reduction
@cached("factor_analysis")
def fit_factor_analysis(y, mask=None, N_iters=100):
    print("Fitting Factor Analysis")
    model = FactorAnalysis(D_obs, D_latent)

    if mask is None:
        mask = np.ones_like(y, dtype=bool)

    # Center the data
    b = y.mean(0)
    data = model.add_data(y-b, mask=mask)
    for _ in progprint_xrange(N_iters):
        model.resample_model()

    C_init = np.column_stack((model.W, b))
    return data.Z, C_init

@cached("pca")
def fit_pca(y, whiten=True):
    print("Fitting PCA")
    from sklearn.decomposition import PCA
    model = PCA(n_components=D_latent, whiten=whiten)
    x_init = model.fit_transform(y)
    C_init = model.components_.T
    b_init = model.mean_[:,None]
    sigma = np.sqrt(model.explained_variance_)

    # inverse transform is given by
    # X.dot(sigma * C_init.T) + b_init.T
    if whiten:
        C_init = sigma * C_init

    return x_init, np.column_stack((C_init, b_init))

### Make an ARHMM for initialization
@cached("arhmm")
def fit_arhmm(x, affine=True):
    print("Fitting Sticky ARHMM")
    dynamics_hypparams = \
        dict(nu_0=D_latent + 2,
             S_0=np.eye(D_latent),
             M_0=np.hstack((np.eye(D_latent), np.zeros((D_latent, int(affine))))),
             K_0=np.eye(D_latent + affine),
             affine=affine)
    dynamics_hypparams = get_empirical_ar_params([x], dynamics_hypparams)

    dynamics_distns = [
        AutoRegression(
            A=np.column_stack((0.99 * np.eye(D_latent),
                               np.zeros((D_latent, int(affine))))),
            sigma=np.eye(D_latent),
            **dynamics_hypparams)
        for _ in range(K)]

    init_distn = Gaussian(nu_0=D_latent + 2,
                          sigma_0=np.eye(D_latent),
                          mu_0=np.zeros(D_latent),
                          kappa_0=1.0)

    arhmm = ARWeakLimitStickyHDPHMM(
        init_state_distn='uniform',
        init_emission_distn=init_distn,
        obs_distns=dynamics_distns,
        alpha=3.0, kappa=10.0, gamma=3.0)

    arhmm.add_data(x)

    lps = []
    for _ in progprint_xrange(1000):
        arhmm.resample_model()
        lps.append(arhmm.log_likelihood())

    z_init = arhmm.states_list[0].stateseq
    z_init = np.concatenate(([0], z_init))

    return arhmm, z_init

def make_rslds_parameters(C_init):
    init_dynamics_distns = [
        Gaussian(
            mu=np.zeros(D_latent),
            sigma=3*np.eye(D_latent),
            nu_0=D_latent + 2, sigma_0=3. * np.eye(D_latent),
            mu_0=np.zeros(D_latent), kappa_0=1.0,
        )
        for _ in range(K)]

    dynamics_hypparams = \
        dict(nu_0=D_latent + 1 + 2,
             S_0=np.eye(D_latent),
             M_0=np.zeros((D_latent, D_latent + 1)),
             K_0=np.eye(D_latent + 1),
             affine=False)

    dynamics_distns = [
        Regression(
            A=np.hstack((0.99 * np.eye(D_latent), np.zeros((D_latent, 1)))),
            sigma=np.eye(D_latent),
            **dynamics_hypparams)
        for _ in range(K)]


    if C_init is not None:
        emission_distns = \
            DiagonalRegression(D_obs, D_latent + 1,
                               A=C_init.copy(), sigmasq=np.ones(D_obs),
                               alpha_0=2.0, beta_0=2.0)
    else:
        emission_distns = \
            DiagonalRegression(D_obs, D_latent + 1,
                               alpha_0=2.0, beta_0=2.0)

    return init_dynamics_distns, dynamics_distns, emission_distns


@cached("slds")
def fit_slds(inputs, z_init, x_init, y, mask, C_init,
              N_iters=10000):
    print("Fitting standard SLDS")
    init_dynamics_distns, dynamics_distns, emission_distns = \
        make_rslds_parameters(C_init)

    # slds = MixedEmissionHMMSLDS(
    #     init_state_distn='uniform',
    #     init_dynamics_distns=init_dynamics_distns,
    #     dynamics_distns=dynamics_distns,
    #     emission_distns=[emission_distns],
    #     alpha=3.)

    slds = WeakLimitStickyHDPHMMSLDS(
        init_state_distn='uniform',
        init_dynamics_distns=init_dynamics_distns,
        dynamics_distns=dynamics_distns,
        emission_distns=emission_distns,
        alpha=3., gamma=3.0, kappa=100.)

    slds.add_data(y, inputs=inputs, mask=mask)

    # Initialize states
    if z_init is not None:
        slds.states_list[0].stateseq = z_init.copy().astype(np.int32)

    if x_init is not None:
        slds.states_list[0].gaussian_states = x_init.copy()

    # Initialize dynamics
    print("Initializing dynamics with Gibbs sampling")
    for _ in progprint_xrange(100):
        slds.resample_dynamics_distns()

    # Fit the model
    lps = []
    z_smpls = []
    for _ in progprint_xrange(N_iters):
        slds.resample_model()
        lps.append(slds.log_likelihood())
        z_smpls.append(slds.stateseqs[0].copy())

    x_test = slds.states_list[0].gaussian_states
    z_smpls = np.array(z_smpls)
    lps = np.array(lps)

    return slds, lps, z_smpls, x_test

@cached("rslds")
def fit_rslds(inputs, z_init, x_init, y, mask, C_init,
              true_model=None, initialization="none", N_iters=10000):
    print("Fitting rSLDS")
    init_dynamics_distns, dynamics_distns, emission_distns = \
        make_rslds_parameters(C_init)

    rslds = SoftmaxRecurrentOnlySLDS(
        init_state_distn='uniform',
        init_dynamics_distns=init_dynamics_distns,
        dynamics_distns=dynamics_distns,
        emission_distns=emission_distns,
        fixed_emission=False,
        alpha=3.)

    rslds.add_data(y, inputs=inputs, mask=mask)

    # Initialize dynamics
    # print("Initializing dynamics with Gibbs sampling")
    for _ in progprint_xrange(100):
        rslds.resample_dynamics_distns()
        rslds.resample_trans_distn()
        rslds.resample_emission_distns()

    # if true_model is not None:
    #     rslds.trans_distn.W = true_model.trans_distn.W.copy()
    #     rslds.trans_distn.b = true_model.trans_distn.b.copy()
    #     for rdd, tdd in zip(rslds.dynamics_distns, true_model.dynamics_distns):
    #         rdd.A = tdd.A.copy()
    #         rdd.sigma = tdd.sigma.copy()
    #     rslds.emission_distns[0].A = true_model.emission_distns[0].A.copy()
    #     rslds.emission_distns[0].sigmasq_flat = true_model.emission_distns[0].sigmasq_flat.copy()

    # Fit the model
    lps = []
    z_smpls = []
    for _ in progprint_xrange(N_iters):
        rslds.resample_model()
        lps.append(rslds.log_likelihood())
        z_smpls.append(rslds.stateseqs[0].copy())

    x_test = rslds.states_list[0].gaussian_states
    z_smpls = np.array(z_smpls)
    lps = np.array(lps)

    print("Inf W_markov:\n{}".format(rslds.trans_distn.logpi))
    print("Inf W_input:\n{}".format(rslds.trans_distn.W))

    return rslds, lps, z_smpls, x_test

@cached("rslds_variational")
def fit_rslds_variational(inputs, z_init, x_init, y, mask, C_init,
              true_model=None, initialization="gibbs",  N_gibbs=100, N_iters=10000):
    print("Fitting rSLDS")
    init_dynamics_distns, dynamics_distns, emission_distns = \
        make_rslds_parameters(C_init)

    rslds = SoftmaxRecurrentOnlySLDS(
        init_state_distn='uniform',
        init_dynamics_distns=init_dynamics_distns,
        dynamics_distns=dynamics_distns,
        emission_distns=emission_distns,
        fixed_emission=False,
        alpha=3.)

    # Set the prior precision for the dynamics params
    rslds.add_data(y, inputs=inputs, mask=mask)

    if initialization == "true":
        print("Initializing with true model")
        rslds.emission_distns[0].J_0 = 1e2 * np.eye(D_latent + 1)
        rslds.trans_distn.W = true_model.trans_distn.W.copy()
        rslds.trans_distn.b = true_model.trans_distn.b.copy()
        rslds.trans_distn._initialize_mean_field()

        for rdd, tdd in zip(rslds.dynamics_distns, true_model.dynamics_distns):
            rdd.A = tdd.A.copy()
            rdd.sigma = tdd.sigma.copy()

        rslds.emission_distns[0].A = true_model.emission_distns[0].A.copy()
        rslds.emission_distns[0].sigmasq_flat = true_model.emission_distns[0].sigmasq_flat.copy()
        rslds.emission_distns[0].J_0 = 1e2 * np.eye(D_latent+1)

        rslds.states_list[0].stateseq = true_model.states_list[0].stateseq.copy()
        rslds.states_list[0].gaussian_states = true_model.states_list[0].gaussian_states.copy()

    elif initialization == "given":
        print("Initializing with given states")
        rslds.emission_distns[0].J_0 = 1e2 * np.eye(D_latent + 1)
        rslds.states_list[0].stateseq = z_init.copy()
        rslds.states_list[0].gaussian_states = x_init.copy()
        for _ in progprint_xrange(N_gibbs):
            rslds.resample_model()

    elif initialization == "given_regression":
        print("Initializing with given states and logistic regression")
        rslds.emission_distns[0].J_0 = 1e2 * np.eye(D_latent + 1)
        rslds.states_list[0].stateseq = z_init.astype(np.int32).copy()
        rslds.states_list[0].gaussian_states = x_init.copy()

        # Initialize the trans matrix with softmax regression
        from sklearn.linear_model import LogisticRegression
        lr = LogisticRegression()
        lr.fit(x_init, z_init.astype(np.int))
        rslds.trans_distn.b = lr.intercept_
        rslds.trans_distn.W = lr.coef_.T

    elif initialization == "gibbs":
        print("Initializing with Gibbs sampling")
        for _ in progprint_xrange(N_gibbs):
            rslds.resample_model()

    rslds._init_mf_from_gibbs()

    # Fit the model
    vlbs = []
    z_smpls = [np.argmax(rslds.states_list[0].expected_states, axis=1)]
    for _ in progprint_xrange(N_iters):
        vlbs.append(rslds.meanfield_coordinate_descent_step(compute_vlb=True))
        z_smpls.append(np.argmax(rslds.states_list[0].expected_states, axis=1))

    x_smpl = rslds.states_list[0].smoothed_mus
    z_smpls = np.array(z_smpls)
    vlbs = np.array(vlbs)


    if true_model is not None:
        print("True logpi:\n{}".format(true_model.trans_distn.logpi))
        print("True W:\n{}".format(true_model.trans_distn.W))

    print("Inf logpi:\n{}".format(rslds.trans_distn.logpi))
    print("Inf W:\n{}".format(rslds.trans_distn.W))

    return rslds, vlbs, z_smpls, x_smpl


if __name__ == "__main__":
    # Load the data
    z_trues, z_key, N_neurons, Ts, \
    all_neuron_names, datasets, masks, \
    Ys, _, shared_neurons = \
        load_data()

    # Only fit a single worm
    worm = 0
    z_true = z_trues[worm]
    y = Ys[worm]
    T = Ts[worm]
    N = y.shape[1]
    D_obs = N
    inputs = np.ones((T,1))
    mask = None

    # Initialize with PCA
    x_init, C_init = fit_pca(Ys[worm])

    ## Fit an ARHMM for initialization
    arhmm, z_init = fit_arhmm(x_init)
    # plot_trajectory_and_probs(
    #     z_init[1:], x_init[1:],
    #     title="Sticky ARHMM")
    # plt.show()

    # plot_all_dynamics(arhmm.obs_distns,
    #                   filename="sticky_arhmm_dynamics.png")

    ## Fit a standard SLDS
    slds, slds_lps, slds_z_smpls, slds_x = \
        fit_slds(inputs, None, x_init, y, mask, C_init, N_iters=1000)

    plot_discrete_state_samples(slds_z_smpls, z_true)
    plt.show()

    ## Fit a recurrent SLDS
    # rslds, rslds_lps, rslds_z_smpls, rslds_x = \
    #     fit_rslds(inputs, z_init, x_init, y, mask, C_init,
    #               true_model=true_model, N_iters=1000)

    rslds_z_init = slds_z_smpls[-1].copy()
    rslds_z_init[-K:] = np.arange(K, dtype=np.int32)
    rslds, rslds_vlbs, rslds_z_smpls, rslds_x = \
        fit_rslds_variational(inputs, rslds_z_init, slds_x, y, mask, C_init=C_init,
                  N_iters=10, initialization="given_regression")

    plot_discrete_state_samples(rslds_z_smpls, z_true)
    # plot_changepoint_prs(z_smpls[0], z_trues[0], title="Worm 1 Discrete States")

    fig = plt.figure(figsize=(2.5,2.5))
    ax = create_axis_at_location(fig, 0.025, 0.025, 2.35, 2.35, projection="3d")
    plot_3d_continuous_states(rslds_x, rslds_z_smpls[-1],
                              ax=ax,
                              title="Worm {} Latent States".format(worm+1),
                              colors=colors, lw=0.5, alpha=0.75, figsize=(3,3),
                              results_dir=results_dir, filename="xs_3d_worm{}.pdf".format(worm))


    plot_3d_dynamics(rslds.dynamics_distns,
                     rslds_z_smpls[-1],
                     rslds_x)
    plt.show()
    #
    # plot_all_dynamics(rslds.dynamics_distns)
    #
    # plot_z_samples(rslds_z_smpls,
    #                plt_slice=(0,1000),
    #                filename="rslds_zsamples.png")

    ## Generate from the model
    # T_gen = 2000
    # inputs = np.ones((T_gen, 1))
    # (rslds_y_gen, rslds_x_gen), rslds_z_gen = rslds.generate(T=T_gen, inputs=inputs)
    #
    # (slds_ys_gen, slds_x_gen), slds_z_gen = slds.generate(T=T_gen, inputs=inputs)
    # slds_y_gen = slds_ys_gen[0]

    plt.show()
