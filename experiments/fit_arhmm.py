import os
import pickle

from functools import partial
import itertools as it

import numpy as np
np.random.seed(0)
from scipy.ndimage import gaussian_filter1d

from tqdm import tqdm

# Modeling stuff
from pyhsmm.util.general import relabel_by_usage, relabel_by_permutation

from autoregressive.models import ARWeakLimitStickyHDPHMM, ARHMM
from pyslds.util import get_empirical_ar_params
from pybasicbayes.distributions import AutoRegression, RobustAutoRegression
from rslds.models import SoftmaxRecurrentARHMM
from zimmer.models import HierarchicalARWeakLimitStickyHDPHMM, HierarchicalRecurrentARHMM, HierarchicalRecurrentARHMMWithNN
from zimmer.dynamics import HierarchicalAutoRegression, HierarchicalRobustAutoRegression

from zimmer.io import WormData, load_kato_key, load_kato_data

# LDS Results
lds_dir = os.path.join("results", "kato", "2018-03-16-hlds", "run001")
signal = "dff_diff"
assert os.path.exists(lds_dir)

# AR-HMM RESULTS
results_dir = os.path.join("results", "kato", "2018-03-16-arhmm", "run001")

assert os.path.exists(results_dir)
fig_dir = os.path.join(results_dir, "figures")
mov_dir = os.path.join(results_dir, "movies")


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


def compute_baseline_likelihood(xtrains, xtests, alpha=3.):
    model = ARHMM(init_state_distn="uniform",
                  obs_distns=[AutoRegression(**ar_params)],
                  alpha=alpha)

    for xtr in xtrains:
        model.add_data(xtr)

    # Fit the model with Gibbs
    lls = []
    for itr in tqdm(range(N_samples)):
        model.resample_model()
        lls.append(model.log_likelihood())

    # Compute heldout likelihood
    hll = 0
    for xte in xtests:
        hll += model.log_likelihood(xte)

    return hll


def _fit_model_wrapper(K, alpha=3, gamma=100., kappa=100.,
                       is_hierarchical=True,
                       is_robust=True,
                       is_recurrent=True,
                       is_nn=False,
                       use_all_data=False,
                       init_with_kmeans=True,
                       ):

    model_class = \
        ARWeakLimitStickyHDPHMM if (not is_hierarchical and not is_recurrent) else \
        SoftmaxRecurrentARHMM if (not is_hierarchical and is_recurrent) else \
        HierarchicalARWeakLimitStickyHDPHMM if (is_hierarchical and not is_recurrent) else \
        HierarchicalRecurrentARHMM if (is_hierarchical and is_recurrent and not is_nn) else \
        HierarchicalRecurrentARHMMWithNN if (is_hierarchical and is_recurrent and is_nn) else None

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

    return model, np.array(lls), hll, z_smpls


def fit_all_models(Ks=np.arange(4, 21, 2)):

    # Compute the baseline probability
    fit_baseline = cached("baseline")(compute_baseline_likelihood)
    baseline_hll = fit_baseline(xtrains[:N_worms], xtests[:N_worms])
    baseline_hll /= T_tests[:N_worms].sum()
    print("baseline test ll: ", baseline_hll)

    results = dict(baseline_hll=baseline_hll)

    for index, (is_hierarchical, is_robust, is_recurrent, is_nn) in \
            enumerate(it.product(*([(False, True)] * 4))):

        models = []
        llss = []
        hlls = []
        z_smplss = []

        group_name = "{}_{}_{}_{}".format(
            "hier" if is_hierarchical else "nohier",
            "rob" if is_robust else "norob",
            "rec" if is_recurrent else "norec",
            "nn" if is_nn else "linear"
        )
        for K in Ks:
            name = "{}_{}_{}_{}_{}".format(
                "hier" if is_hierarchical else "nohier",
                "rob" if is_robust else "norob",
                "rec" if is_recurrent else "norec",
                "nn" if is_nn else "linear",
                K
            )
            print("Fitting model: {}".format(name))

            if (is_nn and K == 1) or \
               (is_nn and not is_recurrent) or \
               (is_nn and not is_hierarchical):
                models.append(None)
                llss.append(np.array([-np.inf]))
                hlls.append(np.array([-np.inf]))
                z_smplss.append(None)
                continue
            
            fit = cached(name)(
                partial(_fit_model_wrapper,
                        is_hierarchical=is_hierarchical,
                        is_robust=is_robust,
                        is_recurrent=is_recurrent,
                        is_nn=is_nn))
            mod, lls, hll, z_smpls = fit(K)

            # Append results
            models.append(mod)
            llss.append(lls)
            hlls.append(hll)
            z_smplss.append(z_smpls)

            print("test ll: ", hll / T_tests[:N_worms].sum())

        final_lls = np.array([lls[-1] for lls in llss])
        hlls = np.array(hlls)
        best_index = np.argmax(hlls)
        print("Best number of states: {}".format(Ks[best_index]))

        results[group_name] = \
            models[best_index], best_index, models, llss, final_lls, hlls

    return results


def fit_best_model(K=8,
                   is_hierarchical=True,
                   is_robust=True,
                   is_recurrent=True,
                   init_with_kmeans=True):
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
                init_with_kmeans=init_with_kmeans,
                use_all_data=True))

    best_model, lls, hll, z_smpls = fit()
    return best_model, lls, hll, z_smpls


def fit_best_model_with_nn(K=8,
                   is_hierarchical=True,
                   is_robust=True,
                   is_recurrent=True,
                   init_with_kmeans=True):
    name = "{}_{}_{}_{}_nn_full".format(
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
                init_with_kmeans=init_with_kmeans,
                is_nn=True,
                use_all_data=True
                ))

    best_model, lls, hll, z_smpls = fit()
    print("best model test ll: ", hll / np.sum([xte.shape[0] for xte in xtests]))
    return best_model, lls, hll, z_smpls


def simulate_trajectories(model, N_trajs=100, T_sim=30, N_sims=4, group=4, min_sim_dur=6):
    from pyhsmm.util.general import rle
    z_finals_rles = [rle(z) for z in rslds_zs]

    x_trajss = []
    x_simss = []
    for k in range(model.num_states):
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
            x_sim, z_sim = model.generate(T=T_sim, init_z=perm[k], init_x=start, group=group, with_noise=False)
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


# def plot_best_model_results(do_plot_expected_states=True,
#                             do_plot_x_2d=True,
#                             do_plot_x_3d=True,
#                             do_plot_dynamics_3d=True,
#                             do_plot_dynamics_2d=True,
#                             do_plot_state_overlap=True,
#                             do_plot_state_usage=True,
#                             do_plot_transition_matrices=True,
#                             do_plot_simulated_trajs=True,
#                             do_plot_recurrent_weights=True,
#                             do_plot_x_at_changepoints=True,
#                             do_plot_latent_trajectories_vs_time=True,
#                             do_plot_duration_histogram=True,
#                             do_plot_eigenspectrum=True,
#                             T_sim=10*3):
#     # Plot the expected states and changepoint probabilities
#     if do_plot_expected_states:
#         for i in range(N_worms):
#             plot_expected_states(rslds_E_zs[i][:, perm],
#                                  cp_prs[i],
#                                  np.concatenate((z_true_trains[i], z_true_tests[i])),
#                                  colors=zimmer_colors,
#                                  title="Worm {} Discrete States".format(i + 1),
#                                  # plt_slice=(0, E_zs[i].shape[0]),
#                                  plt_slice=(0, 1000),
#                                  filepath=os.path.join(fig_dir, "z_cps_worm{}.pdf".format(i)))
#
#     plt.close("all")
#
#     # Plot inferred states in 2d
#     if do_plot_x_2d:
#         for i in range(N_worms):
#             plot_2d_continuous_states(xtrains[i], rslds_zs[i], colors,
#                                       figsize=(4, 4),
#                                       results_dir=fig_dir,
#                                       filename="x_2d_{}.pdf".format(i + 1))
#
#             plot_2d_continuous_states(xtrains[i], rslds_zs[i], colors,
#                                       inds=(0, 2),
#                                       figsize=(4, 4),
#                                       results_dir=fig_dir,
#                                       filename="x_2d_13_{}.pdf".format(i + 1))
#
#             plot_2d_continuous_states(xtrains[i], z_trues[i], zimmer_colors,
#                                       figsize=(4, 4),
#                                       results_dir=fig_dir,
#                                       filename="x_2d_zimmer_{}.pdf".format(i + 1))
#
#             plot_2d_continuous_states(xtrains[i], z_trues[i], zimmer_colors,
#                                       inds=(0, 2),
#                                       figsize=(4, 4),
#                                       results_dir=fig_dir,
#                                       filename="x_2d_13_zimmer_{}.pdf".format(i + 1))
#         plt.close("all")
#
#
#     # Plot inferred states in 3d
#     if do_plot_x_3d:
#         for i in range(N_worms):
#             plot_3d_continuous_states(xtrains[i], rslds_zs[i], colors,
#                                       figsize=(1.2, 1.2),
#                                       # title="LDS Worm {} States (ARHMM Labels)".format(i + 1),
#                                       title="worm {}".format(i + 1),
#                                       results_dir=fig_dir,
#                                       filename="x_3d_{}.pdf".format(i + 1),
#                                       lim=3,
#                                       lw=.5,
#                                       inds=(0,1,2))
#
#             # make_states_3d_movie(z_finals[i], xtrains[i],
#             #                      title=None,
#             #                      lim=None,
#             #                      inds=(0, 1, 2),
#             #                      colors=colors,
#             #                      figsize=(1.2, 1.2),
#             #                      filepath=os.path.join(mov_dir, "x_3d_{}.mp4".format(i+1)),
#             #                      lw=.5)
#
#             # plot_3d_continuous_states(xtrains[i], z_finals[i], colors,
#             #                           figsize=(1.2, 1.2),
#             #                           # title="LDS Worm {} States (ARHMM Labels)".format(i + 1),
#             #                           title="worm {}".format(i + 1),
#             #                           results_dir=fig_dir,
#             #                           filename="x_3d_345_{}.pdf".format(i + 1),
#             #                           lim=3,
#             #                           lw=.5,
#             #                           inds=(3,4,5))
#
#             plot_3d_continuous_states(xtrains[i], z_true_trains[i], zimmer_colors,
#                                       figsize=(1.2, 1.2),
#                                       # title="LDS Worm {} States (ARHMM Labels)".format(i + 1),
#                                       title="worm {}".format(i + 1),
#                                       results_dir=fig_dir,
#                                       filename="x_3d_zimmer_{}.pdf".format(i + 1),
#                                       lim=3,
#                                       lw=.5,
#                                       inds=(0, 1, 2))
#
#             # make_states_3d_movie(z_true_trains[i], xtrains[i],
#             #                      title=None,
#             #                      lim=None,
#             #                      inds=(0, 1, 2),
#             #                      colors=zimmer_colors,
#             #                      figsize=(1.2, 1.2),
#             #                      filepath=os.path.join(mov_dir, "x_3d_zimmer_{}.mp4".format(i + 1)),
#             #                      lw=.5)
#
#             # make_states_3d_movie(slds_z_finals[i], xtrains[i],
#             #                      title=None,
#             #                      lim=None,
#             #                      inds=(0, 1, 2),
#             #                      colors=colors,
#             #                      figsize=(1.2, 1.2),
#             #                      filepath=os.path.join(mov_dir, "x_3d_slds_{}.mp4".format(i + 1)),
#             #                      lw=.5)
#
#             plt.close("all")
#
#     if do_plot_dynamics_3d:
#         plot_3d_dynamics(
#             dynamics_distns,
#             np.concatenate(rslds_zs),
#             np.vstack(xs),
#             simss=[stable_sims[2:3] for stable_sims in stable_simss],
#             colors=colors,
#             lim=3,
#             filepath=os.path.join(fig_dir, "dynamics_123.pdf"))
#
#         # make_states_dynamics_movie(
#         #     hier_dynamics_distns,
#         #     np.concatenate(z_finals),
#         #     np.vstack(xs),
#         #     simss=[stable_sims[2:3] for stable_sims in stable_simss],
#         #     colors=colors,
#         #     lim=3,
#         #     filepath=os.path.join(mov_dir, "dynamics_123.pdf"))
#
#         # make_states_dynamics_movie(
#         #     hier_dynamics_distns,
#         #     np.concatenate(z_finals),
#         #     np.vstack(xs),
#         #     simss=[stable_sims[:25] for stable_sims in stable_simss],
#         #     sims_lw=.5, sims_ms=2,
#         #     colors=colors,
#         #     lim=3,
#         #     filepath=os.path.join(mov_dir, "dynamics_123_many.pdf"))
#
#         plt.close("all")
#
#     if do_plot_dynamics_2d:
#         plot_2d_dynamics(
#             dynamics_distns,
#             np.concatenate(rslds_zs),
#             np.vstack(xs),
#             colors=colors,
#             lim=3,
#             inds=(0,1),
#             filepath=os.path.join(fig_dir, "dynamics_12.pdf"))
#         plt.close("all")
#
#         # plot_2d_dynamics(
#         #     dynamics_distns,
#         #     np.concatenate(z_finals),
#         #     np.vstack(xs),
#         #     colors=colors,
#         #     lim=3,
#         #     inds=(0, 2),
#         #     filepath=os.path.join(fig_dir, "dynamics_13.pdf"))
#         # plt.close("all")
#         #
#         # plot_2d_dynamics(
#         #     dynamics_distns,
#         #     np.concatenate(z_finals),
#         #     np.vstack(xs),
#         #     colors=colors,
#         #     lim=3,
#         #     inds=(1, 2),
#         #     filepath=os.path.join(fig_dir, "dynamics_23.pdf"))
#         plt.close("all")
#
#     if do_plot_state_overlap:
#         plot_state_overlap(rslds_zs, [ztr[N_lags:] for ztr in z_trues],
#                            z_key=z_key,
#                            z_colors=zimmer_colors,
#                            results_dir=fig_dir)
#         plt.close("all")
#
#     if do_plot_state_usage:
#         # plot_state_usage_by_worm(z_finals,
#         #                          results_dir=fig_dir)
#         plot_state_usage_by_worm_matrix(rslds_zs,
#                                         results_dir=fig_dir)
#         plt.close("all")
#
#     if do_plot_transition_matrices:
#         plot_all_transition_matrices(rslds_zs,
#                                      results_dir=fig_dir)
#         plt.close("all")
#
#     if do_plot_simulated_trajs:
#         # for k in range(best_model.num_states):
#         #     long_sims = [x_sim for x_sim in x_simss[k] if x_sim.shape[0] >= 6]
#         #     stable_sims = [x_sim for x_sim in long_sims if abs(x_sim).max() < 3]
#         #     inds = np.random.choice(len(stable_sims), size=4, replace=False)
#         #
#         #     plot_simulated_trajectories2(
#         #         k, x_trajss[k], [stable_sims[i] for i in inds], C_clusters, d_clusters, T_sim,
#         #         lim=3,
#         #         results_dir=fig_dir)
#         #
#         #     plot_simulated_trajectories3(
#         #         k, [stable_sims[i] for i in inds], C_clusters, d_clusters, T_sim,
#         #         results_dir=fig_dir)
#         #
#         #     plt.close("all")
#         # plot_simulated_trajectories4(
#         #     [stable_sims[2:3] for stable_sims in stable_simss], C_clusters, d_clusters, T_sim,
#         #     results_dir=fig_dir)
#
#         plot_simulated_cluster_activation(stable_simss, C_clusters, d_clusters, results_dir=fig_dir)
#         plt.show()
#
#     if do_plot_recurrent_weights:
#         plot_recurrent_transitions(best_model.trans_distn,
#                                    [x[1:] for x in xs],
#                                    rslds_zs,
#                                    results_dir=fig_dir)
#         # plt.close("all")
#         plt.show()
#
#     if do_plot_x_at_changepoints:
#         # plot_x_at_changepoints(z_finals, xs,
#         #                        results_dir=fig_dir)
#
#         plot_x_at_changepoints(z_trues, xs,
#                                colors=zimmer_colors,
#                                basename="x_cp_zimmer",
#                                results_dir=fig_dir)
#
#     if do_plot_latent_trajectories_vs_time:
#         plot_slice = (9 * 60 * 3, 12 * 60 * 3)
#         plot_latent_trajectories_vs_time(xs, rslds_zs,
#                                          plot_slice=plot_slice,
#                                          show_xticks=False,
#                                          title="inferred segmentation",
#                                          basename="x_segmentation",
#                                          colors=colors,
#                                          results_dir=fig_dir)
#
#         plot_latent_trajectories_vs_time(xs, z_trues,
#                                          plot_slice=plot_slice,
#                                          title="manual segmentation",
#                                          basename="x_segmentation_zimmer",
#                                          colors=zimmer_colors,
#                                          results_dir=fig_dir)
#         plt.close("all")
#
#     if do_plot_duration_histogram:
#         durss = [np.array([x_sim.shape[0] for x_sim in x_sims]) for x_sims in x_simss]
#
#         plot_duration_histogram(best_model.trans_distn,
#                                 rslds_zs,
#                                 durss,
#                                 perm=perm,
#                                 results_dir=fig_dir)
#
#         # plot_duration_cdfs(best_model.trans_distn,
#         #                         z_finals,
#         #                         durss,
#         #                         perm=perm,
#         #                         results_dir=fig_dir)
#
#         plt.close("all")
#
#     # if do_plot_eigenspectrum:
#     #     markers = ['o', '^', 's', 'p', 'h']
#     #     for i, hdd in enumerate(hier_dynamics_distns):
#     #         width = 1.0 if i == 0 else 0.7
#     #         left = 0.3 if i == 0 else 0.05
#     #         fig = plt.figure(figsize=(width, 1.0))
#     #
#     #         # ax = fig.add_subplot(111, aspect="equal")
#     #         from hips.plotting.layout import create_axis_at_location
#     #         ax = create_axis_at_location(fig, left, 0.2, 0.6, 0.6)
#     #         for w, dd in enumerate(hdd.regressions):
#     #             evs = np.linalg.eigvals(dd.A[:,:-1])
#     #             assert np.all(evs.real >= 0.45)
#     #             assert np.all(evs.real <= 1.2)
#     #             assert np.all(evs.imag >= -0.3)
#     #             assert np.all(evs.imag <= 0.3)
#     #
#     #             ax.plot(np.real(evs), np.imag(evs),
#     #                     ls='',
#     #                     marker=markers[w],
#     #                     # marker='o',
#     #                     markerfacecolor=colors[i],
#     #                     mec='k',
#     #                     mew=.5,
#     #                     markersize=3,
#     #                     alpha=0.75,
#     #                     label="{}".format(w+1))
#     #
#     #         ax.plot([-2.1, 1.2], [0, 0], ':k', lw=0.5)
#     #         ax.plot([0, 0], [-1.2, 1.2], ':k', lw=0.5)
#     #         ths = np.linspace(0, 2*np.pi, 100)
#     #         ax.plot(np.cos(ths), np.sin(ths), '-k', lw=0.5)
#     #         ax.set_xlim(0.4, 1.2)
#     #         ax.set_ylim(-0.25, 0.25)
#     #         ax.set_xlabel("re($\\lambda$)", labelpad=0, fontsize=6)
#     #
#     #         if i == 0:
#     #             ax.set_ylabel("im($\\lambda$)", labelpad=0, fontsize=6)
#     #             ax.set_yticks([-0.2, 0, 0.2])
#     #         else:
#     #             ax.set_yticks([])
#     #
#     #         ax.tick_params(labelsize=4)
#     #
#     #         if i == 0:
#     #             ax.legend(loc="lower right", fontsize=4,
#     #                       ncol=3, labelspacing=0.5, columnspacing=.5,
#     #                       handletextpad=.5)
#     #
#     #         ax.set_title("state {}".format(i+1), fontsize=6)
#     #         # plt.tight_layout(pad=0.05)
#     #
#     #         plt.savefig(os.path.join(fig_dir, "eigenspectrum_{}.pdf".format(i+1)))
#
#     if do_plot_eigenspectrum:
#         fig = plt.figure(figsize=(6, 2.5))
#
#         for i, hdd in enumerate(hier_dynamics_distns):
#             for w, dd in enumerate(hdd.regressions):
#                 ax = fig.add_subplot(N_worms, 8, w * 8 + i + 1)
#                 evs = np.linalg.eigvals(dd.A[:, :-1])
#                 assert np.all(evs.real >= 0.45)
#                 assert np.all(evs.real <= 1.2)
#                 assert np.all(evs.imag >= -0.3)
#                 assert np.all(evs.imag <= 0.3)
#
#                 ax.plot(np.real(evs), np.imag(evs),
#                         ls='',
#                         # marker=markers[w],
#                         marker='o',
#                         markerfacecolor=colors[i],
#                         mec='k',
#                         mew=.5,
#                         markersize=3,
#                         alpha=1.0,
#                         label="{}".format(w + 1))
#
#                 ax.plot([-2.1, 1.2], [0, 0], ':k', lw=0.5)
#                 ax.plot([0, 0], [-1.2, 1.2], ':k', lw=0.5)
#                 ths = np.linspace(0, 2 * np.pi, 100)
#                 ax.plot(np.cos(ths), np.sin(ths), '-k', lw=0.5)
#                 ax.set_xlim(0.4, 1.2)
#                 ax.set_ylim(-0.25, 0.25)
#
#                 if w == N_worms - 1:
#                     ax.set_xlabel("re($\\lambda$)", labelpad=0, fontsize=6)
#                     ax.set_xticks([0.5, 1.0])
#                 else:
#                     ax.set_xticks([])
#
#                 if i == 0:
#                     ax.set_ylabel("worm {}\nim($\\lambda$)".format(w+1), labelpad=0, fontsize=6)
#                     ax.set_yticks([-0.2, 0, 0.2])
#                 else:
#                     ax.set_yticks([])
#
#                 ax.tick_params(labelsize=4)
#
#                 # if i == 0:
#                 #     ax.legend(loc="lower right", fontsize=4,
#                 #               ncol=3, labelspacing=0.5, columnspacing=.5,
#                 #               handletextpad=.5)
#
#                 if w == 0:
#                     ax.set_title("state {}".format(i + 1), fontsize=6)
#
#         plt.tight_layout(pad=0.2)
#         plt.savefig(os.path.join(fig_dir, "eigenspectrum_all.pdf".format(i + 1)))
#
#
#     if True:
#         for i in range(N_worms):
#             plot_changepoint_prs(
#                 np.array([relabel_by_permutation(z, rslds_iperm) for z in z_smpls[i]]),
#                 z_trues[i],
#                 true_colors=zimmer_colors,
#                 true_cmap=gradient_cmap(zimmer_colors),
#                 filepath=os.path.join(fig_dir, "z_smpls_{}.pdf".format(i)))


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


def predict_from_z():
    """
    Compute the implied distribution p(x | z) and then combine this with
    p(y | x) to get p(y | z).
    """
    # We can compute the mean and covariance of x in feedforward fashion
    # starting with the mean and covariance of x[0] and the given
    # discrete state sequence z.
    # for worm in range(N_worms):
    for worm in range(1):
        z, x, T = rslds_zs[worm], xs[worm], Ts[worm]
        dds = [dd.regressions[worm] for dd in hier_dynamics_distns]
        As = [dd.A[:,:-1] for dd in dds]
        bs = [dd.A[:,-1] for dd in dds]
        Qs = [dd.sigma for dd in dds]

        mus = np.zeros((T, D_latent))
        sigmas = np.zeros((T, D_latent, D_latent))

        # Start with a small variance around x[0]
        mus[0] = x[0]
        sigmas[0] = 1e-4 * np.eye(D_latent)

        for t in range(T-1):
            At, bt, Qt = As[z[t]], bs[z[t]], Qs[z[t]]
            mus[t+1] = At.dot(mus[t]) + bt
            sigmas[t+1] = At.dot(sigmas[t]).dot(At.T) + Qt

        # Project this into the space of neural activity
        mus_y = mus.dot(C.T) + d
        sigmas_y = np.array([(C.dot(sigmat) * C).sum(axis=1) for sigmat in sigmas])

        plt.figure()
        plt.plot(mus[:,0])
        plt.plot(x[:,0])
        plt.show()

def predict_from_z_given_partial_obs(xlim=(0, 18)):
    """
    Compute the implied distribution p(x | z) and then combine this with
    p(y | x) to get p(y | z).
    """
    from pylds.lds_messages_interface import rts_smoother
    for worm in range(N_worms):
        z, x, T = rslds_zs[worm], xs[worm], Ts[worm]
        tt = np.arange(T) / 60. / 3.0

        # Extend z with a dummy state at the end to get the right time scale
        z = np.concatenate((z, [0]))

        mu_init = x[0].copy('C')
        sigma_init = 1e-4 * np.eye(D_latent)

        dds = [dd.regressions[worm] for dd in hier_dynamics_distns]
        As = np.array([dd.A[:,:-1] for dd in dds])[z]
        bs = np.array([dd.A[:,-1:] for dd in dds])[z]
        Qs = np.array([dd.sigma for dd in dds])[z]

        # Make a dummy observation to constrain the scale of the latent states
        C = np.ones((1, D_latent))
        d = np.zeros((1, 1))
        R = 9 * np.eye(1)
        y = np.zeros((T, 1))
        u = np.ones((T, 1))

        # Use RTS smoother to get posterior over x
        _, mus, sigmasqs = rts_smoother(mu_init, sigma_init, As, bs, Qs, C, d, R, u, y)
        sigmasqs = sigmasqs[:, np.arange(D_latent), np.arange(D_latent)]
        sigmas = np.sqrt(sigmasqs)

        # Plot the latent states and their expectations under z and soft constraint
        from hips.plotting.layout import create_axis_at_location
        fig = plt.figure(figsize=(6, 5))
        ax1 = create_axis_at_location(fig, 0.5, 1.0, 5.25, 3.75)
        spc = 5
        for d in range(D_latent):
            plt.fill_between(tt, spc * d + mus[:, d] - 2 * sigmas[:,d], spc * d + mus[:, d] + 2 * sigmas[:,d], color=colors[0], alpha=0.25)
            plt.plot(tt, spc * d + mus[:, d], label='$\\mathbb{E}[x \\mid z]$' if d == 0 else None, lw=3, color=colors[0])
            plt.plot(tt, spc * d + x[:, d], label='$x$' if d == 0 else None, ls='-', lw=2, color='k')
        plt.yticks(spc * np.arange(D_latent), ["$x_{{{}}}$".format(d+1) for d in range(D_latent)])
        plt.ylim(-spc, spc*(D_latent + 1))
        plt.xlim(xlim)
        plt.xticks([])
        plt.legend(loc="upper right", ncol=2)

        # Plot the discrete latent states
        K = best_model.num_states
        ax2 = create_axis_at_location(fig, 0.5, .5, 5.25, .375)
        plt.imshow(z[None,:], cmap=gradient_cmap(colors[:K]), interpolation="none", vmin=0, vmax=K-1, aspect="auto", extent=(0, tt[-1], 0, 1))
        plt.yticks([])
        plt.ylabel("$z$", labelpad=10, rotation=0)
        plt.xlabel("time (min)")
        plt.xlim(xlim)
        plt.savefig(os.path.join(fig_dir, "E_x_given_z_{}.png".format(worm)))
        plt.savefig(os.path.join(fig_dir, "E_x_given_z_{}.pdf".format(worm)))
    plt.close("all")



def hacky_simulation(n_iter=100, T=1000, worm=0):
    """
    Compute the implied distribution p(x | z) and then combine this with
    p(y | x) to get p(y | z).
    """
    from pylds.lds_messages_interface import rts_smoother
    model = best_model
    D = model.D
    x = np.zeros((T, D))
    # z = np.random.randint(model.num_states, size=(T,))
    z = rslds_zs[worm][:T]

    def sample_x_given_z(z):
        # Extend z with a dummy state at the end to get the right time scale
        mu_init = x[0].copy('C')
        sigma_init = 1e-4 * np.eye(D)

        dds = [o.regressions[worm] for o in best_model.obs_distns]
        # dds = [dd.regressions[worm] for dd in hier_dynamics_distns]
        As = np.array([dd.A[:,:-1] for dd in dds])[z]
        bs = np.array([dd.A[:,-1:] for dd in dds])[z]
        Qs = np.array([dd.sigma for dd in dds])[z]

        # Make a dummy observation to constrain the scale of the latent states
        C = np.ones((1, D))
        d = np.zeros((1, 1))
        R = 9 * np.eye(1)
        y = np.zeros((T, 1))
        u = np.ones((T, 1))

        # Use RTS smoother to get posterior over x
        _, mus, _ = rts_smoother(mu_init, sigma_init, As, bs, Qs, C, d, R, u, y)
        return mus

    def sample_z_given_x(x):
        model.add_data(x, group=worm)
        states = model.states_list.pop()
        states.resample()
        return np.concatenate((np.zeros(N_lags, dtype=int), states.stateseq))

    for _ in range(n_iter):
        print(".")
        x = sample_x_given_z(z)
        # z = sample_z_given_x(x)

    return x, z


def hack_simulation_2(model, data, T=100, init_x=None, init_z=None, with_noise=True, group=None):
    from pybasicbayes.util.stats import sample_discrete
    # Generate from the prior and raise exception if unstable
    K, n = model.num_states, model.D

    # Initialize discrete state sequence
    zs = np.empty(T, dtype=np.int32)
    if init_z is None:
        zs[0] = sample_discrete(model.init_state_distn.pi_0.ravel())
    else:
        zs[0] = init_z

    xs = np.empty((T, n), dtype='double')
    if init_x is None:
        xs[0] = np.random.randn(n)
    else:
        xs[0] = init_x

    for t in range(1, T):
        # Sample discrete state given previous continuous state
        A = model.trans_distn.get_trans_matrices(xs[t - 1:t])[0]
        zs[t] = sample_discrete(A[zs[t - 1], :])

        # Sample continuous state given current discrete state
        # mu = model.obs_distns[zs[t]].predict(xs[t - 1][None, :], group=group)[0]
        # sigma = model.obs_distns[zs[t]].sigmas[group]
        # x_prop = np.random.multivariate_normal(mu, sigma)
        x_prop = model.obs_distns[zs[t]].rvs(xs[t-1][None, :], return_xy=False, group=group)

        # Find the nearest data point in the dataset
        nearest = np.argmin(np.sum((data - x_prop)**2, axis=1))

        xs[t] = data[nearest]

        assert np.all(np.isfinite(xs[t])), "RARHMM appears to be unstable!"

    # TODO:
    # if keep:
    #     ...

    return xs, zs
#
#
# def plot_neural_activity_plus_segmentation():
#     from matplotlib.gridspec import GridSpec
#     from pyhsmm.util.general import rle
#
#     for i in range(N_worms):
#
#         y = ys[i].copy()
#         z = rslds_zs[i]
#         x = xs[i]
#         z_rle = rle(z)
#         ysm = xs[i].dot(C.T) + d
#         spc = 5
#
#         fig = plt.figure(figsize=(15, 12))
#         gs = GridSpec(2, 1, height_ratios=(5, 1))
#         ax = fig.add_subplot(gs[0,0])
#
#         all_ys = np.vstack(ys)
#         all_ms = np.vstack(ms)
#         all_ys[~all_ms] = np.nan
#         scales = np.nanstd(all_ys, axis=0)
#
#         n_start = 0
#         # n_frames = 18 * 60 * 3 + 1
#         n_frames = y.shape[0]
#         t = n_start / 180.0 + np.arange(n_frames) / 180.0
#
#         offset = 0
#         ticks = []
#         for c in range(N_clusters):
#             for n in neuron_perm:
#                 if neuron_clusters[n] != c:
#                     continue
#
#                 if ms[i][0, n]:
#                     plt.plot(t, -y[n_start:n_start + n_frames, n] / scales[n] + spc * offset, '-', color=colors[3],
#                              lw=2)
#                 else:
#                     # plt.plot(t, np.zeros_like(t) + spc * offset, ':', color='k', lw=.5)
#                     pass
#                 plt.plot(t, -ysm[n_start:n_start + n_frames, n] / scales[n] + spc * offset, '-', color='k', lw=2)
#
#                 ticks.append(offset * spc)
#                 offset += 1
#
#             # Add an extra space between clusters
#             offset += 2
#
#         # Remove last space
#         offset -= 2
#
#         if i == 0:
#             plt.yticks(ticks, neuron_names[neuron_perm])
#         else:
#             plt.yticks([])
#         yl = (-spc, offset * spc)
#
#         offset = 0
#         for k, dur in zip(*z_rle):
#             ax.fill_between([offset / 60 / 3, (offset + dur) / 60 / 3],
#                              [yl[0], yl[0]], [yl[1], yl[1]],
#                              color=colors[k], alpha=.5)
#             offset += dur
#
#         plt.ylim(reversed(yl))
#         plt.xlim(t[0], t[-1])
#         plt.xlabel("time (min)")
#
#
#
#         plt.title("worm {} differenced Ca++".format(i + 1))
#
#         # Plot the neural activity
#         ax2 = fig.add_subplot(gs[1,0])
#         lim = 1.1 * abs(x).max()
#         xsc = x / (2 * lim)
#         D_latent = x.shape[1]
#
#         # Plot z in background
#         offset = 0
#         for k, dur in zip(*z_rle):
#             ax2.fill_between([offset / 60 / 3, (offset + dur) / 60 / 3],
#                              [-D_latent, -D_latent], [0, 0],
#                             color=colors[k], alpha=.5)
#             offset += dur
#
#                 # Plot x
#         for j in range(D_latent):
#             ax2.plot(t, xsc[:, j] - j - 0.5, '-k', lw=2)
#             # ax2plot(plot_slice, (-d - 0.5) * np.ones(2), ':k', lw=1)
#
#         ax2.set_xlim([0, t[-1]])
#         ax2.set_xlabel("time (min)")
#
#
#         ax2.set_yticks(-1 * np.arange(D_latent) - 0.5)
#         ax2.set_yticklabels(1 + np.arange(D_latent))
#         # ax2set_yticklabels(["$x_{{{}}}$".format(d + 1) for d in range(D_latent)])
#         ax2.set_ylabel("latent dimension")
#         ax2.set_ylim(-D_latent, 0)
#
#
#         plt.tight_layout()
#         plt.savefig(os.path.join(fig_dir, "y_{}.pdf".format(i)))
#         plt.close("all")
#
#
# def plot_neural_activity_and_trans_probs(x, z, model, T=None):
#
#     K = model.num_states
#     T = T if T is not None else x.shape[0]
#     fig = plt.figure(figsize=(6, 3))
#     ax1 = fig.add_subplot(121, projection="3d")
#     plot_3d_continuous_states(x[:T], z[:T], colors, ax=ax1)
#     point = ax1.plot([x[0,0]], [x[0, 1]], [x[0,2]], 'ko')[0]
#
#     # plot probabilities
#     ax2 = fig.add_subplot(122, aspect=4)
#     circle = ax2.plot(K, 0.9, 'o', color=colors[z[0]], mec='k', mew=1, ms=10)[0]
#     ax2.text(K-.5, 0.85, "current state", horizontalalignment="right")
#     bars = [ax2.bar(k, 0, color=colors[k], width=0.8, edgecolor='k', linewidth=1)[0] for k in range(K)]
#     ax2.set_xticklabels([])
#     ax2.set_ylim(0, 1)
#     ax2.set_ylabel("probability")
#     plt.tight_layout(pad=.1)
#
#     trans_matrices = model.trans_distn.get_trans_matrices(x)
#
#     def _draw(t):
#         point.set_data(x[t:t+1, 0], x[t:t+1, 1])
#         point.set_3d_properties(x[t:t+1, 2])
#
#         orig_zt = perm[z[t]]
#         pi = trans_matrices[t, orig_zt, perm]
#
#         for k in range(K):
#             bars[k].set_height(pi[k])
#         circle.set_color(colors[z[t]])
#
#     import matplotlib.animation as manimation
#     FFMpegWriter = manimation.writers['ffmpeg']
#     metadata = dict(title='probability vs space')
#     writer = FFMpegWriter(fps=6, bitrate=1024, metadata=metadata)
#     with writer.saving(fig, os.path.join(mov_dir, "trans_probs.mp4"), 300):
#         # for t in tqdm(range(x.shape[0])):
#         for t in tqdm(range(T)):
#             _draw(t)
#             writer.grab_frame()
#
#
# def permute_to_match_zimmer(z_infs):
#
#     # Compute total overlap from all worms
#     K_zimmer = 8
#     K_model = best_model.num_states
#     assert K_model == K_zimmer
#
#     overlap = np.zeros((K_zimmer, K_model), dtype=float)
#     for z_true, z_inf in zip(z_trues, z_infs):
#         for k1 in range(K_zimmer):
#             for k2 in range(K_model):
#                 overlap[k1, k2] += np.sum((z_true[1:] == k1) & (z_inf == k2))
#
#     # Use the Hungarian algorithm to find a permutation of states that
#     # yields the highest overlap
#     from scipy.optimize import linear_sum_assignment
#     _, perm = linear_sum_assignment(-overlap)
#     print("overlap: ", overlap[np.arange(K_model), perm].sum())
#     # _ = linear_sum_assignment(-overlap)
#
#     # Add any unused inferred states
#     # if K_model > K_zimmer:
#     #     unused = np.array(list(set(np.arange(K_model)) - set(perm)))
#     #     perm = np.concatenate((perm, unused))
#
#     # z_trues_perm = [relabel_by_permutation(z_true, perm) for z_true in z_trues]
#     iperm = np.argsort(perm)
#     return iperm
#
#
# def make_movies(z_infs, name="", slc=(9, 12)):
#     iperm = permute_to_match_zimmer(z_infs)
#     z_infs_perm = [relabel_by_permutation(z_inf, iperm) for z_inf in z_infs]
#
#     # z_trues_perm = z_trues
#
#     # make_states_3d_movie(z_finals[i], xtrains[i],
#     #                      title=None,
#     #                      lim=None,
#     #                      inds=(0, 1, 2),
#     #                      colors=colors,
#     #                      figsize=(1.2, 1.2),
#     #                      filepath=os.path.join(mov_dir, "x_3d_{}.mp4".format(i+1)),
#     #                      lw=.5)
#     slc = slice(slc[0] * 60 * 3, slc[1] * 60 * 3)
#     for i in range(N_worms):
#         plot_3d_continuous_states(xs_smoothed[i][slc], z_infs_perm[i][slc],
#                                   colors,
#                                   figsize=(1.2, 1.2),
#                                   # title="worm {}".format(i + 1),
#                                   results_dir=fig_dir,
#                                   filename="x{}_3d_{}.pdf".format(name, i + 1),
#                                   lim=3,
#                                   lw=.75,
#                                   inds=(0, 1, 2))
#
#         plot_3d_continuous_states(xs_smoothed[i][slc], z_trues[i][slc],
#                                   np.array(colors),
#                                   figsize=(1.2, 1.2),
#                                   # title="worm {}".format(i + 1),
#                                   results_dir=fig_dir,
#                                   filename="x_3d_zimmerb_{}.pdf".format(i + 1),
#                                   lim=3,
#                                   lw=.75,
#                                   inds=(0, 1, 2))
#
#     # make_states_3d_movie(z_true_trains[i], xtrains[i],
#     #                      title=None,
#     #                      lim=None,
#     #                      inds=(0, 1, 2),
#     #                      colors=zimmer_colors,
#     #                      figsize=(1.2, 1.2),
#     #                      filepath=os.path.join(mov_dir, "x_3d_zimmer_{}.mp4".format(i + 1)),
#     #                      lw=.5)
#
#     # make_states_3d_movie(slds_z_finals[i], xtrains[i],
#     #                      title=None,
#     #                      lim=None,
#     #                      inds=(0, 1, 2),
#     #                      colors=colors,
#     #                      figsize=(1.2, 1.2),
#     #                      filepath=os.path.join(mov_dir, "x_3d_slds_{}.mp4".format(i + 1)),
#     #                      lw=.5)
#
#     plt.close("all")
#
#     # Plot the overlap
#     plot_state_overlap(z_infs_perm, [ztr[N_lags:] for ztr in z_trues],
#                        z_key=["" for _ in range(8)],
#                        z_colors=colors,
#                        fig_name=name,
#                        results_dir=fig_dir,
#                        permute=True)
#     plt.close("all")
#
#
#
#     # plot_3d_dynamics(
#     #     dynamics_distns,
#     #     np.concatenate(z_finals),
#     #     np.vstack(xs),
#     #     simss=[stable_sims[2:3] for stable_sims in stable_simss],
#     #     colors=colors,
#     #     lim=3,
#     #     filepath=os.path.join(fig_dir, "dynamics_123.pdf"))
#
#     # make_states_dynamics_movie(
#     #     hier_dynamics_distns,
#     #     np.concatenate(z_finals),
#     #     np.vstack(xs),
#     #     simss=[stable_sims[2:3] for stable_sims in stable_simss],
#     #     colors=colors,
#     #     lim=3,
#     #     filepath=os.path.join(mov_dir, "dynamics_123.pdf"))
#
#     # make_states_dynamics_movie(
#     #     hier_dynamics_distns,
#     #     np.concatenate(z_finals),
#     #     np.vstack(xs),
#     #     simss=[stable_sims[:25] for stable_sims in stable_simss],
#     #     sims_lw=.5, sims_ms=2,
#     #     colors=colors,
#     #     lim=3,
#     #     filepath=os.path.join(mov_dir, "dynamics_123_many.pdf"))
#
#
#
# # def make_latent_states_movie(x, z, filename):
# #     T = x.shape[0]
# #     fig = plt.figure(figsize=(1.2, 1.2))
# #     ax1 = fig.add_subplot(111, projection="3d")
# #     point = ax1.plot([x[0,0]], [x[0, 1]], [x[0,2]], 'ko')[0]
# #
# #     def _draw(t):
# #         ax1.cla()
# #         plot_3d_continuous_states(x[:t], z[:t],
# #                                   colors=colors,
# #                                   ax=ax1,
# #                                   lim=3,
# #                                   lw=.5,
# #                                   inds=(0, 1, 2))
# #
# #         # Update point
# #         # point.set_data(x[t:t+1, 0], x[t:t+1, 1])
# #         # point.set_3d_properties(x[t:t+1, 2])
# #         ax1.plot([x[t, 0]], [x[t, 1]], [x[t, 2]], 'ko', markersize=2)[0]
# #
# #     import matplotlib.animation as manimation
# #     FFMpegWriter = manimation.writers['ffmpeg']
# #     metadata = dict(title='probability vs space')
# #     writer = FFMpegWriter(fps=30, bitrate=1024, metadata=metadata)
# #     with writer.saving(fig, os.path.join(mov_dir, filename), 300):
# #         # for t in tqdm(range(x.shape[0])):
# #         for t in tqdm(range(1, T)):
# #             _draw(t)
# #             writer.grab_frame()
#
#
# def make_latent_states_movie(x, z, filename, fps=60):
#     T = x.shape[0]
#     lw = 0.75
#     lim = 3
#
#     fig = plt.figure(figsize=(1.2, 1.2))
#     ax = fig.add_subplot(111, projection="3d")
#
#     point = ax.plot([x[0,0]], [x[0, 1]], [x[0,2]], 'ko', markersize=3)[0]
#     point.set_zorder(1000)
#     paths = [x[:1]]
#     path_handles = [ax.plot([x[0,0]], [x[0, 1]], [x[0,2]], '-', color=colors[z[0]], lw=lw)[0]]
#
#     ax.set_xlabel("dim 1", labelpad=-18, fontsize=6)
#     ax.set_ylabel("dim 2", labelpad=-18, fontsize=6)
#     ax.set_zlabel("dim 3", labelpad=-18, fontsize=6)
#     ax.set_xticklabels([])
#     ax.set_yticklabels([])
#     ax.set_zticklabels([])
#     if lim is not None:
#         ax.set_xlim(-lim, lim)
#         ax.set_ylim(-lim, lim)
#         ax.set_zlim(-lim, lim)
#
#     plt.tight_layout(pad=0.1)
#
#     def _draw(t):
#         assert t > 0
#         # ax1.cla()
#         # plot_3d_continuous_states(x[:t], z[:t],
#         #                           colors=colors,
#         #                           ax=ax1,
#         #                           lim=3,
#         #                           lw=.5,
#         #                           inds=(0, 1, 2))
#
#         # Update point
#         point.set_data(x[t:t+1, 0], x[t:t+1, 1])
#         point.set_3d_properties(x[t:t+1, 2])
#
#         # Update paths
#         if z[t] == z[t-1]:
#             paths[-1] = np.row_stack((paths[-1], x[t:t+1]))
#             path_handles[-1].set_data(paths[-1][:,0], paths[-1][:, 1])
#             path_handles[-1].set_3d_properties(paths[-1][:, 2])
#         else:
#             paths.append(x[t-1:t+1])
#             path_handles.append(ax.plot(paths[-1][:, 0],
#                                         paths[-1][:, 1],
#                                         paths[-1][:, 2],
#                                         '-',
#                                         color=colors[z[t]],
#                                         lw=lw)[0])
#
#     import matplotlib.animation as manimation
#     FFMpegWriter = manimation.writers['ffmpeg']
#     metadata = dict(title='probability vs space')
#     writer = FFMpegWriter(fps=fps, bitrate=1024, metadata=metadata)
#     with writer.saving(fig, os.path.join(mov_dir, filename), 300):
#         # for t in tqdm(range(x.shape[0])):
#         for t in tqdm(range(1, T)):
#             _draw(t)
#             writer.grab_frame()
#
#
#
# def make_data_movie(y, m, z, filename, fps=60):
#     sns.set_style("ticks")
#     T = z.shape[0]
#     # T = 100
#
#     # from pyhsmm.util.general import rle
#     # z_rle, durs = rle(z)
#
#
#     fig = plt.figure(figsize=(6, 6))
#     ax = fig.add_subplot(111)
#
#     lw = 2
#     window = 60 * 3     # window size in frames
#
#     # mask off unseen neurons
#     y = y.copy()
#     scales = y.std(axis=0) + 1e-8
#     y /= scales
#
#     # perm = neuron_perm
#     perm = np.arange(y.shape[1])
#     yp = y[:, perm]
#     mp = m[:, perm]
#     visible = mp[0]
#     yp = yp[:, visible]
#     N = yp.shape[1]
#
#     spc = 6
#     offset = -np.arange(N) * spc
#
#     start = 1
#     stop = T
#
#     cps, = np.where(np.diff(z[start:stop + 1]) != 0)
#     cps = np.concatenate(([start], start + cps + 1, [stop]))
#     zcps = z[cps[:-1]]
#
#     for cp0, cpf, zcp in zip(cps[:-1], cps[1:], zcps):
#         ax.plot(np.arange(cp0 - 1, cpf), offset + yp[cp0 - 1:cpf], color=colors[zcp])
#
#     ax.set_xticks(np.arange(T, step=3 * 15))
#     ax.set_xticklabels(np.arange(T, step=3 * 15) / 3.0 / 60.)
#     ax.set_xlim(-window, 0)
#     ax.set_xlabel("time (min)")
#     ax.set_yticks(offset)
#     ax.set_yticklabels(neuron_names[perm][visible])
#     ax.set_ylim(-N * spc, spc)
#
#     # ax.set_xticks(np.arange(T, step=3*15))
#     # ax.set_xticklabels(np.arange(T, step=3*15) / 3.0 / 60.)
#     # ax.set_xlim(0, window)
#     # ax.set_xlabel("time (min)")
#     # ax.set_yticks(offset)
#     # ax.set_yticklabels(neuron_names[perm][visible])
#     # ax.set_ylim(-N*spc, spc)
#
#     plt.tight_layout(pad=0.1)
#
#     def _draw(t):
#         assert t > 0
#         # start = max(1, t-window)
#         # stop = t
#         #
#         # cps, = np.where(np.diff(z[start:stop+1]) != 0)
#         # cps = np.concatenate(([start], start + cps + 1, [stop]))
#         # zcps = z[cps]
#         #
#         # for cp0, cpf, zcp in zip(cps[:-1], cps[1:], zcps[:-1]):
#         #     ax.plot(np.arange(cp0-1, cpf), offset + yp[cp0-1:cpf], color=colors[zcp])
#         #
#         # ax.set_xticks(np.arange(T, step=3 * 15))
#         # ax.set_xticklabels(np.arange(T, step=3 * 15) / 3.0 / 60.)
#         # ax.set_xlim(start, max(t, window))
#         # ax.set_xlabel("time (min)")
#         # ax.set_yticks(offset)
#         # ax.set_yticklabels(neuron_names[perm][visible])
#         # ax.set_ylim(-N * spc, spc)
#         ax.set_xlim(t-window, t)
#
#     import matplotlib.animation as manimation
#     FFMpegWriter = manimation.writers['ffmpeg']
#     metadata = dict()
#     writer = FFMpegWriter(fps=fps, bitrate=1024, metadata=metadata)
#     with writer.saving(fig, os.path.join(mov_dir, filename), 300):
#         # for t in tqdm(range(x.shape[0])):
#         for t in tqdm(range(1, T)):
#             _draw(t)
#             writer.grab_frame()
#
#
# def plot_simulated_data(y, y_mean, worm, neuron, slc=(9, 12)):
#     all_ys = np.vstack(ys)
#     all_ms = np.vstack(ms)
#     all_ys[~all_ms] = np.nan
#     scales = np.nanstd(all_ys, axis=0)
#
#     fig = plt.figure(figsize=(8, 1))
#     fig.patch.set_alpha(0.0)
#     ax = fig.add_subplot(111)
#     ax.patch.set_alpha(0.0)
#
#     index = np.where(neuron_names == neuron)[0][0]
#
#     T = y.shape[0]
#     t = np.arange(T) / 180.0
#
#     # ax.plot(t, np.zeros(T), ':k', lw=2)
#     # ax.plot(t, np.zeros(T), ':', color='k', lw=4)
#     ax.plot(t, y[:, index] / scales[index], '-', color=colors[3], lw=4)
#     ax.plot(t, y_mean[:, index] / scales[index], '-', color='k', lw=4)
#
#     ax.set_ylim(-3, 3)
#
#     ax.spines["left"].set_visible(False)
#     ax.spines["top"].set_visible(False)
#     ax.spines["right"].set_visible(False)
#     ax.spines["bottom"].set_visible(False)
#
#     plt.xlim(slc)
#     # plt.ylabel(neuron, rotation=0)
#     plt.yticks([])
#     plt.xticks([])
#     plt.tight_layout()
#     plt.savefig(os.path.join(fig_dir, "ysim_{}_{}.pdf".format(worm, neuron)))
#


if __name__ == "__main__":
    # Load the continuous states found with the LDS
    ys, ms, z_trues, z_true_key, neuron_names = load_kato_data(include_unnamed=True)
    D_obs = ys[0].shape[1]
    with open(os.path.join(lds_dir, "lds_data.pkl"), "rb") as f:
        lds_results = pickle.load(f)

    D_latent = lds_results['D_latent']
    xtrains = lds_results['xtrains']
    xtests = lds_results['xtests']
    xs = [np.vstack((xtr, xte)) for xtr, xte in zip(xtrains, xtests)]
    xs_smoothed = [gaussian_filter1d(x, 1.0, axis=0) for x in xs]

    z_true_trains = lds_results['z_true_trains']
    z_true_tests = lds_results['z_true_tests']
    z_trues = [np.concatenate((ztr, zte)) for ztr, zte in zip(z_true_trains, z_true_tests)]
    z_key = lds_results['z_key']

    T_trains = np.array([xtr.shape[0] for xtr in xtrains])
    T_tests = np.array([xte.shape[0] for xte in xtests])
    Ts = [Ttr + Tte for Ttr, Tte in zip(T_trains, T_tests)]

    C = lds_results['best_model'].C[:, lds_results['perm']]
    d = lds_results['best_model'].D[:, 0]
    ys_preds = [x.dot(C.T) + d for x in xs]

    # N_clusters = lds_results['N_clusters']
    # neuron_clusters = lds_results['neuron_clusters']
    # neuron_perm = lds_results['neuron_perm']
    # C_norm = C / np.linalg.norm(C, axis=1)[:, None]
    # C_clusters = np.array([C[neuron_clusters == c].mean(0) for c in range(N_clusters)])
    # d_clusters = np.array([d[neuron_clusters == c].mean(0) for c in range(N_clusters)])

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
                          etasq=1.0,
                          affine=True)

    Ks = np.concatenate(([1], np.arange(2, 21, 2)))
    results = fit_all_models(Ks)

    # Fit the best model with linear decision boundaries
    # best_model, lls, hll =\
    #     fit_best_model(K=8,
    #                    is_hierarchical=True,
    #                    is_recurrent=True,
    #                    is_robust=True)

    # Fit the best model with neural net decision boundaries
    best_model, lls, hll, z_smpls = \
        fit_best_model_with_nn(K=8,
                               is_hierarchical=True,
                               is_recurrent=True,
                               is_robust=True,
                               init_with_kmeans=True
                               )
    # Compute the expected states
    rslds_E_zs = []
    for s in best_model.states_list:
        s.E_step()
        rslds_E_zs.append(s.expected_states)

    # Relabel by usage
    rslds_zs, rslds_iperm = \
    relabel_by_usage([np.argmax(E_z, axis=1) for E_z in rslds_E_zs],
                     return_mapping=True)

    perm = np.argsort(rslds_iperm)
    hier_dynamics_distns = [best_model.obs_distns[i] for i in perm]
    dynamics_distns = [dd.regressions[-1] for dd in hier_dynamics_distns]
    print("State usage:")
    print(best_model.state_usages[perm])

    # Simulate the best model
    # assert False, "This needs to be updated with new permutation stuff!"
    # sim = cached("simulations")(simulate_trajectories)
    # x_trajss, x_simss = sim(best_model, min_sim_dur=0, N_sims=1000, T_sim=100 * 3, group=0)
    # long_simss = [[x_sim for x_sim in x_sims if x_sim.shape[0] >= 6] for x_sims in x_simss]
    # stable_simss = [[x_sim for x_sim in long_sims if abs(x_sim).max() < 3] for long_sims in long_simss]

    # plot_neural_activity_plus_segmentation()

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

    # Make a movie of the real data with rSLDS labels
    # for worm in range(1, N_worms):
    #     make_latent_states_movie(xs[worm][1:], rslds_zs[worm],
    #                              "real_data_rslds_labels_{}.mp4".format(worm))

    # make_data_movie(np.cumsum(ys[1], axis=0), ms[1], rslds_zs[1], "data_int_{}.mp4".format(1))

    # Generate some data
    # for group in range(2):
    #     x_sim, z_sim = best_model.generate(T=3000, with_noise=True, group=group, tau=0, noise_scale=0.05)
    #     z_sim = relabel_by_permutation(z_sim, rslds_iperm)
    #     # plot_3d_continuous_states(x_sim, z_sim, colors,
    #     #                           figsize=(1.2, 1.2),
    #     #                           results_dir=fig_dir,
    #     #                           filename="x_rslds_gen_3d_{}.pdf".format(group + 1),
    #     #                           lim=3,
    #     #                           lw=.5)
    #     #
    #     # make_latent_states_movie(x_sim, z_sim, "rslds_sample_{}.mp4".format(group), fps=90)
    #     y_sim = x_sim.dot(C.T) + d
    #     sigma_obs = np.diag(lds_results['best_model'].sigma_obs[group])
    #
    #
    #     y_sim_noise = y_sim + np.sqrt(sigma_obs) * np.random.randn(3000, 61)
    #
    #     for neuron in neuron_names:
    #         plot_simulated_data(y_sim_noise, y_sim, group, neuron)
    #         plt.close("all")
    #

    # Plot results
    # plot_best_model_results(
    #     do_plot_expected_states=False,
    #     do_plot_x_2d=False,
    #     do_plot_x_3d=False,
    #     do_plot_dynamics_3d=False,
    #     do_plot_dynamics_2d=False,
    #     do_plot_state_overlap=False,
    #     do_plot_state_usage=False,
    #     do_plot_transition_matrices=False,
    #     do_plot_simulated_trajs=False,
    #     do_plot_recurrent_weights=False,
    #     do_plot_x_at_changepoints=False,
    #     do_plot_latent_trajectories_vs_time=False,
    #     do_plot_duration_histogram=False,
    #     do_plot_eigenspectrum=False
    # )

    ### Now do the same with the hierarchical robust SLDS
    # slds, lls, hll = \
    #     fit_best_model(K=8,
    #                    is_hierarchical=True,
    #                    is_recurrent=False,
    #                    is_robust=True)
    #
    # # Compute the expected states
    # cp_prs = []
    # slds_E_zs = []
    # for s in slds.states_list:
    #     s.E_step()
    #     slds_E_zs.append(s.expected_states)
    #     # cp_prs.append(s.changepoint_probability())

    # Relabel by usage
    # slds_z_finals, slds_iperm = \
    #     relabel_by_usage([np.argmax(E_z, axis=1) for E_z in slds_E_zs],
    #                      return_mapping=True)
    # slds_perm = np.argsort(slds_iperm)
    # slds_dynamics_distns = [slds.obs_distns[i] for i in slds_perm]


    # slds_zs = [np.argmax(E_z, axis=1) for E_z in slds_E_zs]
    # slds_iperm = permute_to_match_zimmer(slds_zs)

    # slds_zs, slds_iperm = \
    #     relabel_by_usage([np.argmax(E_z, axis=1) for E_z in slds_E_zs],
    #                  return_mapping=True)
    #
    # plot_state_overlap(slds_zs, [ztr[N_lags:] for ztr in z_trues],
    #                    z_key=z_key,
    #                    z_colors=zimmer_colors,
    #                    results_dir=fig_dir,
    #                    fig_name="_slds")
    #
    # for i in range(N_worms):
    #     plot_3d_continuous_states(xtrains[i], slds_zs[i], colors,
    #                               figsize=(1.2, 1.2),
    #                               # title="LDS Worm {} States (ARHMM Labels)".format(i + 1),
    #                               title="worm {}".format(i + 1),
    #                               results_dir=fig_dir,
    #                               filename="x_3d_slds_{}.pdf".format(i + 1),
    #                               lim=3,
    #                               lw=.5,
    #                               inds=(0, 1, 2))
    #
    # plt.close("all")



    # rslds_zs = [np.argmax(E_z, axis=1) for E_z in rslds_E_zs]
    # rslds_iperm = permute_to_match_zimmer(rslds_zs)

    # Make a movie of the real data with SLDS labels
    # for worm in range(1, 2):
    #     make_latent_states_movie(xs[worm][1:], slds_zs[worm],
    #                              "real_data_slds_labels_{}.mp4".format(worm))
    #
    #     plot_3d_continuous_states(xs[worm][1:], slds_zs[worm], colors,
    #                               figsize=(1.2, 1.2),
    #                               # title="Simulation (Worm {})".format(group+1),
    #                               results_dir=fig_dir,
    #                               filename="x_slds_3d_{}.pdf".format(worm + 1),
    #                               lim=3,
    #                               lw=.5)
    #

    # Generate some data
    # for group in range(1, 2):
    #     x_sim, z_sim = slds.generate(T=3000, with_noise=True, group=group, tau=0, noise_scale=0.05)
    #     # x_sim, z_sim = hacky_simulation(T=1000, worm=4, n_iter=10)
    #     # start = np.random.randint(Ts[group])
    #     # x, z = xs[group], best_model.states_list[group].stateseq
    #     # x_sim, z_sim = hack_simulation_2(best_model, T=3239, data=x, group=group, init_x=x[start], init_z=z[start])
    #     z_sim = relabel_by_permutation(z_sim, slds_iperm)
    #     plot_3d_continuous_states(x_sim, z_sim, colors,
    #                               figsize=(1.2, 1.2),
    #                               # title="Simulation (Worm {})".format(group+1),
    #                               results_dir=fig_dir,
    #                               filename="x_slds_gen_3d_{}.pdf".format(group + 1),
    #                               lim=3,
    #                               lw=.5)
    #
    #     make_latent_states_movie(x_sim, z_sim, "slds_sample_{}.mp4".format(group), fps=90)

    # plot_neural_activity_and_trans_probs(xs_smoothed[0], z_finals[0], best_model, T=1000)


    # Now just cluster the data and see what it looks like
    # from sklearn.cluster import KMeans
    #
    # km = KMeans(n_clusters=8)
    # km.fit(np.vstack(xs_smoothed))
    # kmeans_zs = km.labels_
    # kmeans_zs = np.split(kmeans_zs, np.cumsum([d.shape[0] for d in xs_smoothed])[:-1])
    # assert len(kmeans_zs) == len(xs_smoothed)

    # make_movies([kmz[1:] for kmz in kmeans_zs], name="_km")
    # make_movies(slds_z_finals, name="_slds")
    # make_movies(z_finals, name="_nnscratch")
