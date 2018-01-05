import os
import pickle

from functools import partial
import itertools as it

import numpy as np
np.random.seed(0)

from tqdm import tqdm

# Plotting stuff
import matplotlib.pyplot as plt
from hips.plotting.colormaps import gradient_cmap
import seaborn as sns
sns.set_style("white")
sns.set_context("paper")
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

from matplotlib.cm import get_cmap
cm = get_cmap("cubehelix")
zimmer_colors = [cm(i) for i in np.linspace(0.05, 0.95, 8)]

# Modeling stuff
from pyhsmm.util.general import relabel_by_usage, relabel_by_permutation

from autoregressive.models import ARWeakLimitStickyHDPHMM
from pyslds.util import get_empirical_ar_params
from pybasicbayes.distributions import AutoRegression, RobustAutoRegression
from rslds.models import SoftmaxRecurrentARHMM
from zimmer.models import HierarchicalARWeakLimitStickyHDPHMM, HierarchicalRecurrentARHMM
from zimmer.dynamics import HierarchicalAutoRegression, HierarchicalRobustAutoRegression

import importlib
import zimmer.plotting
importlib.reload(zimmer.plotting)
from zimmer.plotting import plot_3d_continuous_states, plot_2d_continuous_states, plot_expected_states, \
    plot_3d_dynamics, plot_2d_dynamics, plot_state_overlap, plot_state_usage_by_worm, plot_all_transition_matrices, \
    plot_simulated_trajectories, make_state_predictions_3d_movie, plot_simulated_trajectories2, plot_simulated_trajectories3,\
    plot_recurrent_transitions, plot_x_at_changepoints, plot_latent_trajectories_vs_time, \
    plot_state_usage_by_worm_matrix, plot_duration_histogram, plot_duration_cdfs

# LDS Results
lds_dir = os.path.join("results", "2017-11-03-hlds", "run003")
# lds_dir = os.path.join("results", "2017-11-03-hlds", "run003_dff_bc")
assert os.path.exists(lds_dir)

# AR-HMM RESULTS
results_dir = os.path.join("results", "2017-11-04-arhmm", "run004")
# results_dir = os.path.join("results", "2017-11-04-arhmm", "run004_dff_bc")
assert os.path.exists(results_dir)
fig_dir = os.path.join(results_dir, "figures")


# Fitting parameters
N_lags = 1
N_samples = 1000
N_worms = 5


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


def _fit_model_wrapper(K, alpha=3, gamma=100., kappa=100.,
                       is_hierarchical=True,
                       is_robust=True,
                       is_recurrent=True,
                       use_all_data=False,
                       init_with_kmeans=True):

    model_class = \
        ARWeakLimitStickyHDPHMM if (not is_hierarchical and not is_recurrent) else \
        SoftmaxRecurrentARHMM if (not is_hierarchical and is_recurrent) else \
        HierarchicalARWeakLimitStickyHDPHMM if (is_hierarchical and not is_recurrent) else \
        HierarchicalRecurrentARHMM if (is_hierarchical and is_recurrent) else None

    model_kwargs = \
        dict(alpha=alpha, gamma=gamma, kappa=kappa) if not is_recurrent else \
        dict(alpha=alpha, D_in=D_latent)

    obs_class = \
        AutoRegression if (not is_hierarchical and not is_robust) else \
        RobustAutoRegression if (not is_hierarchical and is_robust) else \
        HierarchicalAutoRegression if (is_hierarchical and not is_robust) else \
        HierarchicalRobustAutoRegression if (is_hierarchical and is_robust) else None

    obs_kwargs = \
        ar_params if not is_hierarchical else hier_ar_params

    model = model_class(
        init_state_distn='uniform',
        obs_distns=[obs_class(**obs_kwargs)
                    for _ in range(K)],
        **model_kwargs
    )

    datas = xs if use_all_data else xtrains

    if init_with_kmeans:
        # Initialize by clustering the x's (good for recurrent models
        from sklearn.cluster import KMeans
        km = KMeans(n_clusters=K)
        km.fit(np.vstack(datas))
        zs = km.labels_
        zs = np.split(zs, np.cumsum([d.shape[0] for d in datas])[:-1])
        assert len(zs) == len(datas)
    else:
        T = sum([d.shape[0] for d in datas])
        runlen = 25
        zs = np.random.randint(K, size=int(np.ceil(T / float(runlen))))
        zs = zs.repeat(runlen)[:T]
        zs = np.split(zs, np.cumsum([d.shape[0] for d in datas])[:-1])
        assert len(zs) == len(datas)

    # Initialize discrete states with runs
    for i in range(N_worms):
        x = datas[i]
        z = zs[i][N_lags:]

        data_kwargs = dict(group=i) if is_hierarchical else dict()
        model.add_data(x, stateseq=z.astype(np.int32), **data_kwargs)

    # Fit the model with Gibbs
    lls = []
    raw_z_smpls = []
    for itr in tqdm(range(N_samples)):
        model.resample_model()
        lls.append(model.log_likelihood())
        raw_z_smpls.append(model.stateseqs)

    z_smpls = []
    for i in range(N_worms):
        z_smpls_i = np.array([z_smpl[i] for z_smpl in raw_z_smpls])
        z_smpls.append(z_smpls_i)

    # Compute heldout likelihood
    hll = 0
    for i in range(N_worms):
        data_kwargs = dict(group=i) if is_hierarchical else dict()
        hll += model.log_likelihood(xtests[i], **data_kwargs)

    return model, np.array(lls), hll, None


def plot_likelihoods(group, Ks, final_lls, hlls, best_index,
                     color, name, axs=None):

    # Plot results of searching over models
    if axs is None:
        plt.figure(figsize=(2.4,1.8))
        # ax1 = plt.subplot(211)
        # ax2 = plt.subplot(212)
        ax = plt.subplot(111)
    else:
        ax = axs

    M = len(Ks)
    T_te = T_tests[:N_worms].sum()

    cmap = gradient_cmap([np.ones(3), colors[4]])
    for m in range(M):
        label = "K={}".format(Ks[m]) \
            if group == 0 and \
               (Ks[m] == 4 or Ks[m] == 12 or Ks[m] == 20) \
            else None
        xx = group + m / (M+2)
        ax.bar(xx, hlls[m] / T_te,
               width=1. / (M + 2),
               color=cmap((m+1) / M),
               edgecolor='k',
               label=label)

    if group == 0:
        ax.plot(best_index / (M+2), 12.65, 'k*', markersize=4)

    # ax.set_xlabel("Model", fontsize=6)
    ax.set_title("model selection", fontsize=8)
    ax.set_ylabel("test log likelihood\n(nats per time bin)", fontsize=6)
    ax.tick_params(labelsize=6)
    ax.set_ylim(11.25, 12.75)

    # update xticks
    xticks = ax.get_xticks()
    if group > 0:
        xticklabels = list(ax.get_xticklabels())
        xticklabels.append(name)
    else:
        xticklabels = [name]

    ax.set_xticks(np.arange(group + 1) + 0.5)
    ax.set_xticklabels(xticklabels)

    return ax


def fit_all_models(Ks=np.arange(4, 21, 2)):
    axs = None
    for index, (is_hierarchical, is_robust, is_recurrent) in \
            enumerate(it.product(*([(True, False)] * 3))):

        models = []
        llss = []
        hlls = []
        z_smplss = []

        group_name = "{}\n{}\n{}".format(
            "hier." if is_hierarchical else "std.",
            "rob." if is_robust else "std.",
            "rec." if is_recurrent else "std."
        )
        for K in Ks:
            name = "{}_{}_{}_{}".format(
                "hier" if is_hierarchical else "nohier",
                "rob" if is_robust else "norob",
                "rec" if is_recurrent else "norec",
                K
            )
            print("Fitting model: {}".format(name))

            fit = cached(name)(
                partial(_fit_model_wrapper,
                        is_hierarchical=is_hierarchical,
                        is_robust=is_robust,
                        is_recurrent=is_recurrent))
            mod, lls, hll, z_smpls = fit(K)

            # Append results
            models.append(mod)
            llss.append(lls)
            hlls.append(hll)
            z_smplss.append(z_smpls)

        final_lls = np.array([lls[-1] for lls in llss])
        hlls = np.array(hlls)
        best_index = np.argmax(hlls)
        print("Best number of states: {}".format(Ks[best_index]))

        axs = plot_likelihoods(index, Ks, final_lls, hlls, best_index,
                               name=group_name, color=colors[index], axs=axs)

    plt.tight_layout()
    plt.legend(loc="upper right", fontsize=4, handlelength=1, labelspacing=.5)
    # plt.savefig(os.path.join(fig_dir, "dimensionality.pdf".format(name)))


def fit_best_model(K=8,
                   is_hierarchical=True,
                   is_robust=True,
                   is_recurrent=True):
    name = "{}_{}_{}_{}_full".format(
        "hier" if is_hierarchical else "nohier",
        "rob" if is_robust else "norob",
        "rec" if is_recurrent else "norec",
        K
    )
    fit = cached(name)(
        partial(_fit_model_wrapper,
                K=K,
                is_hierarchical=is_hierarchical,
                is_robust=is_robust,
                is_recurrent=is_recurrent,
                use_all_data=True,
                init_with_kmeans=True))

    best_model, lls, hll, _ = fit()
    return best_model, lls, hll


def simulate_trajectories(N_trajs=100, T_sim=30, N_sims=4, worm=4, min_sim_dur=6):
    from pyhsmm.util.general import rle
    z_finals_rles = [rle(z) for z in z_finals]

    x_trajss = []
    x_simss = []
    for k in range(best_model.num_states):
        print("Simulating state ", k)
        x_trajs = []
        for worm in range(N_worms):
            x = xs[worm][N_lags:]
            offset = 0
            for z, dur in zip(*z_finals_rles[worm]):
                if z == k:
                    x_trajs.append(x[offset:offset+dur])
                offset += dur

        # Only show N_trajs traces at most
        if len(x_trajs) > N_trajs:
            inds = np.random.choice(len(x_trajs), replace=False, size=N_trajs)
            x_trajss.append([x_trajs[i] for i in inds])
        else:
            x_trajss.append(x_trajs)

        # Simulate dynamics starting from this state
        x_sims = []
        j = 0
        while j < N_sims:
            # Find the starting point of a non-trivial trajectory
            ind = np.random.choice(len(x_trajs))
            if x_trajs[ind].shape[0] < 2:
                continue
            start = x_trajs[ind][0]

            # Simulate a trajectory from this starting point
            # keep the segment that is using state k
            x_sim, z_sim = best_model.generate(T=T_sim, init_z=perm[k], init_x=start, group=worm, with_noise=False)
            is_k = z_sim == perm[k]
            if np.any(~is_k):
                dur_k = np.min(np.arange(T_sim)[~is_k])
            else:
                dur_k = T_sim

            # Make sure the simulation is long enough
            if dur_k < min_sim_dur:
                continue

            # If the simulation passes, keep it
            x_sims.append(x_sim[:dur_k])
            j += 1

            if j % 50 == 0:
                print("{} / {}".format(j, N_sims))

        x_simss.append(x_sims)

    return x_trajss, x_simss


def plot_best_model_results(do_plot_expected_states=True,
                            do_plot_x_2d=True,
                            do_plot_x_3d=True,
                            do_plot_dynamics_3d=True,
                            do_plot_dynamics_2d=True,
                            do_plot_state_overlap=True,
                            do_plot_state_usage=True,
                            do_plot_transition_matrices=True,
                            do_plot_simulated_trajs=True,
                            do_plot_recurrent_weights=True,
                            do_plot_x_at_changepoints=True,
                            do_plot_latent_trajectories_vs_time=True,
                            do_plot_duration_histogram=True,
                            do_plot_eigenspectrum=True,
                            T_sim=10*3):
    # Plot the expected states and changepoint probabilities
    if do_plot_expected_states:
        for i in range(N_worms):
            plot_expected_states(E_zs[i][:,perm],
                                 cp_prs[i],
                                 np.concatenate((z_true_trains[i], z_true_tests[i])),
                                 colors=zimmer_colors,
                                 title="Worm {} Discrete States".format(i + 1),
                                 # plt_slice=(0, E_zs[i].shape[0]),
                                 plt_slice=(0, 1000),
                                 filepath=os.path.join(fig_dir, "z_cps_worm{}.pdf".format(i)))

    plt.close("all")

    # Plot inferred states in 2d
    if do_plot_x_2d:
        for i in range(N_worms):
            plot_2d_continuous_states(xtrains[i], z_finals[i], colors,
                                      figsize=(4, 4),
                                      results_dir=fig_dir,
                                      filename="x_2d_{}.pdf".format(i + 1))
        plt.close("all")


    # Plot inferred states in 3d
    if do_plot_x_3d:
        for i in range(N_worms):
            plot_3d_continuous_states(xtrains[i], z_finals[i], colors,
                                      figsize=(1.2, 1.2),
                                      # title="LDS Worm {} States (ARHMM Labels)".format(i + 1),
                                      title="worm {}".format(i + 1),
                                      results_dir=fig_dir,
                                      filename="x_3d_{}.pdf".format(i + 1),
                                      lim=3,
                                      lw=.5,
                                      inds=(0,1,2))
            plot_3d_continuous_states(xtrains[i], z_finals[i], colors,
                                      figsize=(1.2, 1.2),
                                      # title="LDS Worm {} States (ARHMM Labels)".format(i + 1),
                                      title="worm {}".format(i + 1),
                                      results_dir=fig_dir,
                                      filename="x_3d_345_{}.pdf".format(i + 1),
                                      lim=3,
                                      lw=.5,
                                      inds=(3,4,5))
            plt.close("all")

    if do_plot_dynamics_3d:
        plot_3d_dynamics(
            dynamics_distns,
            np.concatenate(z_finals),
            np.vstack(xs),
            colors=colors,
            lim=3,
            filepath=os.path.join(fig_dir, "dynamics_123.pdf"))
        plt.close("all")

    if do_plot_dynamics_2d:
        plot_2d_dynamics(
            dynamics_distns,
            np.concatenate(z_finals),
            np.vstack(xs),
            colors=colors,
            lim=3,
            inds=(0,1),
            filepath=os.path.join(fig_dir, "dynamics_12.pdf"))
        plt.close("all")

        plot_2d_dynamics(
            dynamics_distns,
            np.concatenate(z_finals),
            np.vstack(xs),
            colors=colors,
            lim=3,
            inds=(0, 2),
            filepath=os.path.join(fig_dir, "dynamics_13.pdf"))
        plt.close("all")

        plot_2d_dynamics(
            dynamics_distns,
            np.concatenate(z_finals),
            np.vstack(xs),
            colors=colors,
            lim=3,
            inds=(1, 2),
            filepath=os.path.join(fig_dir, "dynamics_23.pdf"))
        plt.close("all")

    if do_plot_state_overlap:
        plot_state_overlap(z_finals, [ztr[N_lags:] for ztr in z_trues],
                           z_key=z_key,
                           z_colors=zimmer_colors,
                           results_dir=fig_dir)
        plt.close("all")

    if do_plot_state_usage:
        # plot_state_usage_by_worm(z_finals,
        #                          results_dir=fig_dir)
        plot_state_usage_by_worm_matrix(z_finals,
                                 results_dir=fig_dir)
        plt.close("all")

    if do_plot_transition_matrices:
        plot_all_transition_matrices(z_finals,
                                     results_dir=fig_dir)
        plt.close("all")

    if do_plot_simulated_trajs:
        for k in range(best_model.num_states):
        # for k in range(1):
            long_sims = [x_sim for x_sim in x_simss[k] if x_sim.shape[0] >= 6]
            stable_sims = [x_sim for x_sim in long_sims if abs(x_sim).max() < 3]
            inds = np.random.choice(len(stable_sims), size=4, replace=False)

            # plot_simulated_trajectories2(
            #     k, x_trajss[k], [stable_sims[i] for i in inds], C_clusters, d_clusters, T_sim,
            #     lim=3,
            #     results_dir=fig_dir)

            plot_simulated_trajectories3(
                k, [stable_sims[i] for i in inds], C_clusters, d_clusters, T_sim,
                results_dir=fig_dir)

            plt.close("all")

    if do_plot_recurrent_weights:
        plot_recurrent_transitions(best_model.trans_distn,
                                   [x[1:] for x in xs],
                                   z_finals,
                                   results_dir=fig_dir)
        # plt.close("all")
        plt.show()

    if do_plot_x_at_changepoints:
        # plot_x_at_changepoints(z_finals, xs,
        #                        results_dir=fig_dir)

        plot_x_at_changepoints(z_trues, xs,
                               colors=zimmer_colors,
                               basename="x_cp_zimmer",
                               results_dir=fig_dir)

    if do_plot_latent_trajectories_vs_time:
        plot_slice = (9 * 60 * 3, 12 * 60 * 3)
        plot_latent_trajectories_vs_time(xs, z_finals,
                                         plot_slice=plot_slice,
                                         show_xticks=False,
                                         title="inferred segmentation",
                                         basename="x_segmentation",
                                         colors=colors,
                                         results_dir=fig_dir)

        plot_latent_trajectories_vs_time(xs, z_trues,
                                         plot_slice=plot_slice,
                                         title="manual segmentation",
                                         basename="x_segmentation_zimmer",
                                         colors=zimmer_colors,
                                         results_dir=fig_dir)
        plt.close("all")

    if do_plot_duration_histogram:
        durss = [np.array([x_sim.shape[0] for x_sim in x_sims]) for x_sims in x_simss]

        # plot_duration_histogram(best_model.trans_distn,
        #                         z_finals,
        #                         durss,
        #                         perm=perm,
        #                         results_dir=fig_dir)

        plot_duration_cdfs(best_model.trans_distn,
                                z_finals,
                                durss,
                                perm=perm,
                                results_dir=fig_dir)

        plt.close("all")

    # if do_plot_eigenspectrum:
    #     markers = ['o', '^', 's', 'p', 'h']
    #     for i, hdd in enumerate(hier_dynamics_distns):
    #         width = 1.0 if i == 0 else 0.7
    #         left = 0.3 if i == 0 else 0.05
    #         fig = plt.figure(figsize=(width, 1.0))
    #
    #         # ax = fig.add_subplot(111, aspect="equal")
    #         from hips.plotting.layout import create_axis_at_location
    #         ax = create_axis_at_location(fig, left, 0.2, 0.6, 0.6)
    #         for w, dd in enumerate(hdd.regressions):
    #             evs = np.linalg.eigvals(dd.A[:,:-1])
    #             assert np.all(evs.real >= 0.45)
    #             assert np.all(evs.real <= 1.2)
    #             assert np.all(evs.imag >= -0.3)
    #             assert np.all(evs.imag <= 0.3)
    #
    #             ax.plot(np.real(evs), np.imag(evs),
    #                     ls='',
    #                     marker=markers[w],
    #                     # marker='o',
    #                     markerfacecolor=colors[i],
    #                     mec='k',
    #                     mew=.5,
    #                     markersize=3,
    #                     alpha=0.75,
    #                     label="{}".format(w+1))
    #
    #         ax.plot([-2.1, 1.2], [0, 0], ':k', lw=0.5)
    #         ax.plot([0, 0], [-1.2, 1.2], ':k', lw=0.5)
    #         ths = np.linspace(0, 2*np.pi, 100)
    #         ax.plot(np.cos(ths), np.sin(ths), '-k', lw=0.5)
    #         ax.set_xlim(0.4, 1.2)
    #         ax.set_ylim(-0.25, 0.25)
    #         ax.set_xlabel("re($\\lambda$)", labelpad=0, fontsize=6)
    #
    #         if i == 0:
    #             ax.set_ylabel("im($\\lambda$)", labelpad=0, fontsize=6)
    #             ax.set_yticks([-0.2, 0, 0.2])
    #         else:
    #             ax.set_yticks([])
    #
    #         ax.tick_params(labelsize=4)
    #
    #         if i == 0:
    #             ax.legend(loc="lower right", fontsize=4,
    #                       ncol=3, labelspacing=0.5, columnspacing=.5,
    #                       handletextpad=.5)
    #
    #         ax.set_title("state {}".format(i+1), fontsize=6)
    #         # plt.tight_layout(pad=0.05)
    #
    #         plt.savefig(os.path.join(fig_dir, "eigenspectrum_{}.pdf".format(i+1)))

    if do_plot_eigenspectrum:
        fig = plt.figure(figsize=(6, 2.5))

        for i, hdd in enumerate(hier_dynamics_distns):
            for w, dd in enumerate(hdd.regressions):
                ax = fig.add_subplot(N_worms, 8, w * 8 + i + 1)
                evs = np.linalg.eigvals(dd.A[:, :-1])
                assert np.all(evs.real >= 0.45)
                assert np.all(evs.real <= 1.2)
                assert np.all(evs.imag >= -0.3)
                assert np.all(evs.imag <= 0.3)

                ax.plot(np.real(evs), np.imag(evs),
                        ls='',
                        # marker=markers[w],
                        marker='o',
                        markerfacecolor=colors[i],
                        mec='k',
                        mew=.5,
                        markersize=3,
                        alpha=1.0,
                        label="{}".format(w + 1))

                ax.plot([-2.1, 1.2], [0, 0], ':k', lw=0.5)
                ax.plot([0, 0], [-1.2, 1.2], ':k', lw=0.5)
                ths = np.linspace(0, 2 * np.pi, 100)
                ax.plot(np.cos(ths), np.sin(ths), '-k', lw=0.5)
                ax.set_xlim(0.4, 1.2)
                ax.set_ylim(-0.25, 0.25)

                if w == N_worms - 1:
                    ax.set_xlabel("re($\\lambda$)", labelpad=0, fontsize=6)
                    ax.set_xticks([0.5, 1.0])
                else:
                    ax.set_xticks([])

                if i == 0:
                    ax.set_ylabel("worm {}\nim($\\lambda$)".format(w+1), labelpad=0, fontsize=6)
                    ax.set_yticks([-0.2, 0, 0.2])
                else:
                    ax.set_yticks([])

                ax.tick_params(labelsize=4)

                # if i == 0:
                #     ax.legend(loc="lower right", fontsize=4,
                #               ncol=3, labelspacing=0.5, columnspacing=.5,
                #               handletextpad=.5)

                if w == 0:
                    ax.set_title("state {}".format(i + 1), fontsize=6)

        plt.tight_layout(pad=0.2)
        plt.savefig(os.path.join(fig_dir, "eigenspectrum_all.pdf".format(i + 1)))


# Using the smoothed states, run each model forward making predictions
# and see how well the predictions align with the smoothed states
def rolling_predictions(T_pred, worm, N_sim=50):

    z_filt = best_model.states_list[worm].filter()
    x = xs[worm][N_lags:]
    T = min(x.shape[0], np.inf)

    z_preds = np.empty((N_sim, T - T_pred, T_pred), dtype=np.int)
    x_preds = np.empty((N_sim, T - T_pred, T_pred, D_latent))
    for t in tqdm(range(T - T_pred)):
        for i in range(N_sim):
            z0 = np.random.choice(best_model.num_states, p=z_filt[t])
            x0 = x[t]
            x_gen, z_gen = best_model.generate(T=T_pred, init_z=z0, init_x=x0, group=worm)
            z_preds[i, t, :] = z_gen
            x_preds[i, t, :, :] = x_gen

    return z_preds, x_preds


if __name__ == "__main__":
    # Load the continuous states found with the LDS
    with open(os.path.join(lds_dir, "lds_data.pkl"), "rb") as f:
        lds_results = pickle.load(f)

    D_latent = lds_results['D_latent']
    xtrains = lds_results['xtrains']
    xtests = lds_results['xtests']
    xs = [np.vstack((xtr, xte)) for xtr, xte in zip(xtrains, xtests)]

    z_true_trains = lds_results['z_true_trains']
    z_true_tests = lds_results['z_true_tests']
    z_trues = [np.concatenate((ztr, zte)) for ztr, zte in zip(z_true_trains, z_true_tests)]
    z_key = lds_results['z_key']

    T_trains = np.array([xtr.shape[0] for xtr in xtrains])
    T_tests = np.array([xte.shape[0] for xte in xtests])
    Ts = [Ttr + Tte for Ttr, Tte in zip(T_trains, T_tests)]

    C = lds_results['best_model'].C[:,lds_results['perm']]
    d = lds_results['best_model'].D[:,0]
    N_clusters = lds_results['N_clusters']
    neuron_clusters = lds_results['neuron_clusters']
    C_norm = C / np.linalg.norm(C, axis=1)[:, None]
    C_clusters = np.array([C[neuron_clusters == c].mean(0) for c in range(N_clusters)])
    d_clusters = np.array([d[neuron_clusters == c].mean(0) for c in range(N_clusters)])

    # Set the AR-HMM hyperparameters
    ar_params = dict(nu_0=D_latent + 2,
                     S_0=np.eye(D_latent),
                     M_0=np.hstack((np.eye(D_latent), np.zeros((D_latent, D_latent * (N_lags - 1) + 1)))),
                     K_0=np.eye(D_latent * N_lags + 1),
                     affine=True)
    ar_params = get_empirical_ar_params(xtrains, ar_params)

    # We need a few more parameters for the hierarchical ar models
    hier_ar_params = dict(N_groups=N_worms,
                          M_0=ar_params['M_0'],
                          Sigma_0=np.eye(ar_params['M_0'].size),
                          nu_0=ar_params['nu_0'],
                          Q_0=ar_params['S_0'],
                          etasq=0.01,
                          affine=True)

    Ks = np.arange(4, 21, 2)
    fit_all_models(Ks)

    # Fit the best model
    best_model, lls, hll =\
        fit_best_model(K=8,
                       is_hierarchical=True,
                       is_recurrent=True,
                       is_robust=True)

    # Compute the expected states
    cp_prs = []
    E_zs = []
    for s in best_model.states_list:
        s.E_step()
        E_zs.append(s.expected_states)
        cp_prs.append(s.changepoint_probability())

    # Relabel by usage
    z_finals, iperm = \
        relabel_by_usage([np.argmax(E_z, axis=1) for E_z in E_zs],
                         return_mapping=True)
    perm = np.argsort(iperm)
    hier_dynamics_distns = [best_model.obs_distns[i] for i in perm]
    dynamics_distns = [dd.regressions[-1] for dd in hier_dynamics_distns]
    print("State usage:")
    print(best_model.state_usages[perm])

    # x_trajss, x_simss = simulate_trajectories()
    sim = cached("simulations")(simulate_trajectories)
    x_trajss, x_simss = sim(min_sim_dur=0, N_sims=1000, T_sim=100 * 3)

    # Plot results
    plot_best_model_results(
        do_plot_expected_states=False,
        do_plot_x_2d=False,
        do_plot_x_3d=False,
        do_plot_dynamics_3d=True,
        do_plot_dynamics_2d=False,
        do_plot_state_overlap=False,
        do_plot_state_usage=False,
        do_plot_transition_matrices=False,
        do_plot_simulated_trajs=False,
        do_plot_recurrent_weights=False,
        do_plot_x_at_changepoints=False,
        do_plot_latent_trajectories_vs_time=False,
        do_plot_duration_histogram=False,
        do_plot_eigenspectrum=False
    )

    # Rolling predictions
    # z_preds, x_preds = rolling_predictions(T_pred=6, worm=0)
    # z_preds = relabel_by_permutation(z_preds, iperm)
    #
    # make_state_predictions_3d_movie(
    #     best_model.states_list[0].stateseq,
    #     xs[0],
    #     z_preds, x_preds,
    #     title="Rolling Predictions (Worm 1)",
    #     lim=3,
    #     colors=colors,
    #     figsize=(4, 4),
    #     filepath=os.path.join(results_dir, "predictions_0.mp4")
    # )

    # Generate some data
    # x_sim, z_sim = best_model.generate(T=200)
    # z_sim = relabel_by_permutation(z_sim, iperm)
    # plot_3d_continuous_states(x_sim, z_sim, colors,
    #                           figsize=(4, 4),
    #                           title="Simulation",
    #                           # results_dir=results_dir,
    #                           # filename="x_3d_{}.pdf".format(i + 1),
    #                           lim=3,
    #                           lw=1)
    #
    # plt.show()
