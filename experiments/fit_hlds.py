import os
import pickle

import numpy as np
np.random.seed(1234)

from tqdm import tqdm
from functools import partial

# Plotting stuff
# import matplotlib.pyplot as plt
# from mpl_toolkits.axes_grid1 import make_axes_locatable
# from hips.plotting.colormaps import gradient_cmap
# from hips.plotting.layout import create_axis_at_location
# import seaborn as sns
# sns.set_style("white")
# sns.set_context("paper")
# color_names = ["windows blue",
#                "red",
#                "amber",
#                "faded green",
#                "dusty purple",
#                "orange",
#                "clay",
#                "pink",
#                "greyish",
#                "mint",
#                "light cyan",
#                "steel blue",
#                "forest green",
#                "pastel purple",
#                "salmon",
#                "dark brown"]
# colors = sns.xkcd_palette(color_names)
# cmap = gradient_cmap(colors)


# Modeling stuff
from pyhsmm.util.general import relabel_by_usage

# Load worm modeling specific stuff
from zimmer.io import WormData, load_kato_key

from zimmer.models import HierarchicalLDS
from zimmer.emissions import HierarchicalDiagonalRegression

# from zimmer.plotting import plot_3d_continuous_states
# from zimmer.plotting import make_states_3d_movie

from pybasicbayes.distributions import Regression, DiagonalRegression
from pylds.models import MissingDataLDS

# IO
# run_num = 3
# results_dir = os.path.join("results", "2017-11-03-hlds", "run{:03d}".format(run_num))
# signal = "dff_diff"

# run_num = 1
# results_dir = os.path.join("results", "2018-01-17-hlds", "run{:03d}".format(run_num))
# signal = "dff_deriv"

run_num = 1
results_dir = os.path.join("results", "kato", "2018-02-26-hlds", "run{:03d}".format(run_num))
signal = "dff_diff"

assert os.path.exists(results_dir)
fig_dir = os.path.join(results_dir, "figures")

N_worms = 5
N_clusters = 12


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


def load_data(include_unnamed=True):
    # Load the data
    worm_datas = [WormData(i, name="worm{}".format(i), version="kato") for i in range(N_worms)]

    # Get the "true" discrete states as labeled by Zimmer
    z_trues = [wd.zimmer_states for wd in worm_datas]
    z_trues, newlabels = relabel_by_usage(z_trues, return_mapping=True)

    # Get the key
    z_key = load_kato_key()
    z_key = [z_key[i] for i in np.argsort(newlabels)]

    # Get the names of the neurons
    neuron_names = np.unique(np.concatenate([wd.neuron_names for wd in worm_datas]))
    if not include_unnamed:
        print("Only including named neurons.")
        neuron_names = neuron_names[:61]
    else:
        print("Including all neurons, regardless of whether they were identified.")

    N_neurons = neuron_names.size
    print("{} neurons across all {} worms".format(N_neurons, N_worms))

    # Construct a big dataset with all neurons for each worm
    ys = []
    masks = []
    for wd in worm_datas:
        y_indiv = getattr(wd, signal)
        y = np.zeros((wd.T, N_neurons))
        mask = np.zeros((wd.T, N_neurons), dtype=bool)
        indices = wd.find_neuron_indices(neuron_names)
        for n, index in enumerate(indices):
            if index is not None:
                y[:, n] = y_indiv[:, index]
                mask[:, n] = True

        ys.append(y)
        masks.append(mask)

    return ys, masks, z_trues, z_key, neuron_names


def comput_xcorr(ys, masks):
    N = D_obs
    xcorr = np.nan * np.ones((N, N))
    for n1 in range(N):
        for n2 in range(n1+1, N):
            valid = [(np.all(ms[i][:,n1]) and np.all(ms[i][:,n2])) for i in range(N_worms)]
            if np.any(valid):
                y1s = np.concatenate([ys[i][:, n1] for i in range(N_worms) if valid[i]])
                y2s = np.concatenate([ys[i][:, n2] for i in range(N_worms) if valid[i]])
                xcorr[n1, n2] = np.corrcoef(y1s, y2s)[0, 1]
                xcorr[n2, n1] = xcorr[n1, n2]
    return xcorr

def _split_test_train(y, train=None, train_frac=0.8):
    T = y.shape[0]
    train = train if train is not None else np.arange(T) < train_frac * T
    assert train.shape == (T,) and train.dtype == bool
    return y[train], y[~train], train


# def make_hlds(D_latent, D_obs, D_in=1, alpha_0=1.0, beta_0=1.0):
#     # Don't resample the dynamics distribution
#     dynamics_distn = \
#         Regression(
#             A=np.hstack((0.99 * np.eye(D_latent), np.zeros((D_latent, D_in)))),
#             sigma=0.1 * np.eye(D_latent))
#
#     emission_distn = \
#         HierarchicalDiagonalRegression(
#             D_obs, D_latent + D_in, N_groups=N_worms,
#             alpha_0=alpha_0, beta_0=beta_0)
#
#     return HierarchicalLDS(dynamics_distn, emission_distn)


def _fit_lds(D_latent, D_in=1, alpha_0=1.0, beta_0=1.0,
             is_hierarchical=True,
             datas=None,
             masks=None,
             compute_hll=True):

    print("Fitting LDS with {} latent dimensions".format(D_latent))
    # model = make_hlds(D_latent=D_latent, D_obs=D_obs, D_in=1)

    # Don't resample the dynamics distribution
    dynamics_distn = \
        Regression(
            A=np.hstack((0.99 * np.eye(D_latent), np.zeros((D_latent, D_in)))),
            sigma=0.1 * np.eye(D_latent))

    if is_hierarchical:
        emission_distn = \
            HierarchicalDiagonalRegression(
                D_obs, D_latent + D_in, N_groups=N_worms,
                alpha_0=alpha_0, beta_0=beta_0)

        model = HierarchicalLDS(dynamics_distn, emission_distn)
    else:
        emission_distn = \
            DiagonalRegression(
                D_obs, D_latent + D_in,
                alpha_0=alpha_0, beta_0=beta_0)

        model = MissingDataLDS(dynamics_distn, emission_distn)

    datas = ytrains if datas is None else datas
    masks = mtrains if masks is None else masks
    for i in range(len(datas)):
        data_kwargs = dict(group=i) if is_hierarchical else dict()
        model.add_data(datas[i], mask=masks[i],
                       inputs=np.ones((datas[i].shape[0], 1)),
                       **data_kwargs)

    # Fit the model
    lls = []
    for _ in tqdm(range(50)):
        model.resample_states()
        model.resample_emission_distn()
        lls.append(model.log_likelihood())

    # Evaluate heldout likelihood for this model
    hll = 0
    if compute_hll:
        for i in range(N_worms):
            data_kwargs = dict(group=i) if is_hierarchical else dict()
            hll += model.log_likelihood(ytests[i], mask=mtests[i],
                                        inputs=np.ones((ytests[i].shape[0], 1)),
                                        **data_kwargs)

    return model, np.array(lls), hll


def fit_all_models(D_latents=np.arange(2, 21, 2)):

    results = {}

    for index, is_hierarchical in enumerate([True, False]):

        models = []
        llss = []
        hlls = []

        for D_latent in D_latents:
            name = "{}_{}".format(
                "hlds" if is_hierarchical else "lds",
                D_latent
            )
            print("Fitting model: {}".format(name))

            fit = cached(name)(
                partial(_fit_lds,
                        is_hierarchical=is_hierarchical))
            model, lls, hll = fit(D_latent)

            # Append results
            models.append(model)
            llss.append(lls)
            hlls.append(hll)

        final_lls = np.array([lls[-1] for lls in llss])
        hlls = np.array(hlls)
        best_index = np.argmax(hlls)
        print("Best dimension: {}".format(D_latents[best_index]))

        results["hier" if is_hierarchical else "no_hier"] = \
            models[best_index], best_index, models, llss, final_lls, hlls

    return results

#
# def _plot_likelihoods(D_latents, final_lls, hlls, best_index,
#                      color, name, axs=None):
#
#     # Plot results of searching over models
#     if axs is None:
#         plt.figure(figsize=(6,3))
#         ax1 = plt.subplot(121)
#         ax2 = plt.subplot(122)
#     else:
#         ax1, ax2 = axs
#
#     # for D_latent, lls in zip(D_latents, llss):
#     #     plt.plot(lls, label=D_latent)
#     # plt.legend(loc="lower right")
#     ax1.plot(D_latents, final_lls / n_trains.sum(),
#              '-', markersize=6, color=color)
#     for index in range(len(D_latents)):
#         if index != best_index:
#             ax1.plot(D_latents[index], final_lls[index] / n_trains.sum(),
#                      'o', markersize=6, color=color)
#
#     ax1.plot(D_latents[best_index], final_lls[best_index] / n_trains.sum(),
#              '*', markersize=10, color=color)
#     ax1.set_xlabel("Latent Dimension")
#     ax1.set_ylabel("Train Log Likelihood")
#
#     ax2.plot(D_latents, hlls / n_tests.sum(),
#              '-', markersize=6, color=color, label=name)
#
#     for index in range(len(D_latents)):
#         if index != best_index:
#             ax2.plot(D_latents[index], hlls[index] /  n_tests.sum(),
#                      'o', markersize=6, color=color)
#
#     ax2.plot(D_latents[best_index], hlls[best_index] / n_tests.sum(),
#              '*', markersize=10, color=color)
#     ax2.set_xlabel("Latent Dimension")
#     ax2.set_ylabel("Test Log Likelihood")
#
#     return ax1, ax2
#
#
# def plot_likelihoods(results):
#     axs = None
#     for index, key in enumerate(["hier", "no_hier"]):
#         best_model, best_index, models, llss, final_lls, hlls = results[key]
#         axs = _plot_likelihoods(D_latents, final_lls, hlls, best_index,
#                                 name=key, color=colors[index], axs=axs)
#     plt.tight_layout()
#     plt.legend(loc="lower right")
#     plt.savefig(os.path.join(fig_dir, "dimensionality.pdf"))


def order_latent_dims(xtrains, C, ytrains, mtrains):

    # Sort latent dimensions by how much variance they account for
    D_latent = xtrains[0].shape[1]
    corrcoeffs = np.zeros(D_latent)
    for d in range(D_latent):
        yobss = []
        yhats = []
        for i in range(N_worms):
            mask = mtrains[i][0]
            yobss.append(ytrains[i][:, mask].ravel())
            yhats.append(np.outer(xtrains[i][:, d], C[mask, d]).ravel())
        yobss = np.concatenate(yobss)
        yhats = np.concatenate(yhats)

        corrcoeffs[d] = np.corrcoef(yobss.ravel(), yhats.ravel())[0, 1]

    perm = np.argsort(corrcoeffs)[::-1]
    return perm


def cluster_neruons(best_model, seed=0):
    from pyhsmm.util.general import relabel_by_permutation
    from sklearn.cluster import KMeans
    # C_true = best_model.emission_distn.A[:, :-1].copy()
    # C_true /= np.linalg.norm(C_true, axis=1)[:, None]
    C_norm = C[:,:-1].copy()
    C_norm /= np.linalg.norm(C_norm, axis=1)[:, None]

    np.random.seed(seed)
    cluster = KMeans(n_clusters=N_clusters)
    cluster.fit(C_norm)
    neuron_clusters = cluster.labels_

    avg_C = np.zeros((N_clusters, best_model.D_latent))
    for c in range(N_clusters):
        if not np.any(neuron_clusters == c):
            continue
        avg_C[c] = np.mean(C_norm[neuron_clusters == c], axis=0)

    # Permute the cluster labels by doing PCA on the average C and sorting
    from sklearn.decomposition import PCA
    pca = PCA(n_components=1, random_state=0, svd_solver="full")
    pca.fit(avg_C)
    weights = pca.transform(avg_C)[:, 0]
    weights *= np.sign(weights[0])

    # weights = avg_C[:,0]
    labels_perm = np.argsort(weights)
    neuron_clusters = relabel_by_permutation(neuron_clusters, np.argsort(labels_perm))

    # Lex sort by label then by emission vector
    # perm = np.lexsort(np.row_stack((C_true.T, labels)))
    neuron_perm = np.lexsort((neuron_names[:D_obs], neuron_clusters))

    return neuron_perm, neuron_clusters

#
# def plot_best_model_results(best_model,
#                             do_plot_pca=True,
#                             do_plot_x_3d=True,
#                             do_plot_x_2d=True,
#                             do_plot_sigmasq=True,
#                             do_plot_xcorr=True,
#                             do_plot_similarity=True,
#                             do_plot_cluster_embedding=True,
#                             do_plot_cluster_locations=True,
#                             do_plot_data=True,
#                             do_plot_data_wide=True,
#                             do_plot_data_zoom=True,
#                             do_plot_data_as_matrix=True,
#                             do_plot_smoothed_data_as_matrix=True,
#                             do_plot_cluster_types=True):
#
#     if do_plot_pca:
#         from sklearn.decomposition import PCA
#         for i in range(N_worms):
#             # plot_3d_continuous_states(xtrains[i], np.zeros(xtrains[i].shape[0], dtype=int), colors,
#             #                           figsize=(1.2, 1.2),
#             #                           # title="LDS Worm {} States (Zimmer Lables)".format(i + 1),
#             #                           results_dir=fig_dir,
#             #                           filename="xtr_{}.pdf".format(i + 1),
#             #                           # lim=1.5,
#             #                           lw=1, alpha=0.85)
#             # pca = PCA(3)
#             # yi = ytrains[i][:, mtrains[i][0]]
#             # assert False
#             # x = pca.fit_transform(yi)
#             model, _, _ = _fit_lds(3, datas=[ytrains[i]], masks=[mtrains[i]], compute_hll=False)
#             x = model.states_list[0].gaussian_states
#             make_states_3d_movie(np.zeros(1000, dtype=int),
#                                  x[:1000],
#                                  title=None,
#                                  lim=None,
#                                  inds=(0, 1, 2),
#                                  colors=colors,
#                                  figsize=(1.2, 1.2),
#                                  filepath=os.path.join(fig_dir, "x_pca_{}.mp4".format(i+1)),
#                                  lw=0.25)
#
#         plt.close("all")
#
#
#     # False color with the "true" label from Zimmer
#     if do_plot_x_3d:
#         for i in range(N_worms):
#             plot_3d_continuous_states(xtrains[i], np.zeros(xtrains[i].shape[0], dtype=int), colors,
#                                       figsize=(1.2, 1.2),
#                                       # title="LDS Worm {} States (Zimmer Lables)".format(i + 1),
#                                       results_dir=fig_dir,
#                                       filename="xtr_{}.pdf".format(i + 1),
#                                       # lim=1.5,
#                                       lw=1, alpha=0.85)
#
#             # make_states_3d_movie(np.zeros(1000, dtype=int),
#             #                      xtrains[i][:1000],
#             #                      title=None,
#             #                      lim=None,
#             #                      inds=(0, 1, 2),
#             #                      colors=colors,
#             #                      figsize=(1.2, 1.2),
#             #                      filepath=os.path.join(fig_dir, "xtr_{}.mp4".format(i+1)),
#             #                      lw=0.25)
#
#         plt.close("all")
#
#
#     # if do_plot_x_3d:
#     #     for i in range(N_worms):
#     #         plot_3d_continuous_states(xtrains[i], z_true_trains[i], colors,
#     #                                   figsize=(4, 4),
#     #                                   title="LDS Worm {} States (Zimmer Lables)".format(i + 1),
#     #                                   results_dir=fig_dir,
#     #                                   filename="xtr_zimmer_{}.pdf".format(i + 1),
#     #                                   # lim=1.5,
#     #                                   lw=1)
#     #     plt.close("all")
#     #
#     #     # for i in range(N_worms):
#     #     #     plot_3d_continuous_states(xtests[i], z_true_tests[i], colors,
#     #     #                               figsize=(4, 4),
#     #     #                               title="LDS Worm {} States (Zimmer Lables)".format(i + 1),
#     #     #                               results_dir=fig_dir,
#     #     #                               filename="xte_zimmer_{}.pdf".format(i + 1),
#     #     #                               lim=1.5,
#     #     #                               lw=1)
#     #     # plt.close("all")
#
#     if do_plot_x_2d:
#         pass
#
#     if do_plot_sigmasq:
#
#         # Look at the observation variance across worms
#         sigma_obs = best_model.emission_distn.sigmasq_flat.copy()
#         sigma_obs_mask = np.array([np.any(~m, axis=0) for m in ms])
#         sigma_obs[sigma_obs_mask] = np.nan
#
#         cmap = gradient_cmap([np.ones(3), colors[0]])
#         cmap.set_bad(0.7 * np.ones(3))
#
#         fig = plt.figure(figsize=(1.9, 1.0))
#         # ax = fig.add_subplot(111)
#         ax = create_axis_at_location(fig, 0.4, 0.3, 1.15, .5)
#         im = ax.imshow(np.sqrt(sigma_obs)[:,neuron_perm], vmin=0, aspect="auto", cmap=cmap)
#
#         for o in cluster_offsets[:-1]:
#             ax.plot([o+.5, o+.5], [-.5, N_worms-.5], '-', lw=1, color='k')
#
#         ax.set_xlim(-0.5, D_obs - 0.5)
#         plt.xticks(cluster_offsets, cluster_offsets + 1, fontsize=6)
#         ax.set_xlabel("neuron", fontsize=8, labelpad=-1)
#         ax.set_ylim(N_worms -0.5, -0.5)
#         # ax.set_xticks(np.arange(D_obs))
#         # ax.set_xticklabels(neuron_names[neuron_perm], rotation="90", fontsize=6)
#         ax.set_yticks(np.arange(N_worms))
#         ax.set_yticklabels(np.arange(N_worms) + 1, fontsize=6)
#         ax.set_ylabel("worm", fontsize=8)
#
#         ax.set_title("$\sigma_{\mathsf{obs}}$", fontsize=8)
#
#         cax = create_axis_at_location(fig, 1.6, 0.3, .075, .5)
#         cbar = plt.colorbar(im, cax=cax, ticks=[0, 0.3, 0.6])
#         cbar.ax.tick_params(labelsize=6)
#         # plt.tight_layout(pad=0.1)
#         plt.savefig(os.path.join(fig_dir, "observation_variance.pdf"))
#
#     if do_plot_xcorr:
#
#         # Plot the correlation matrix
#         fig = plt.figure(figsize=(2.5, 3.0))
#         ax = create_axis_at_location(fig, 0.3, 0.6, 2.1, 2.1)
#
#         cmap = gradient_cmap([colors[0], np.ones(3), colors[1]])
#         cmap.set_bad(0.5 * np.ones(3))
#         np.fill_diagonal(xcorr, np.nan)
#         lim = max(abs(np.nanmax(xcorr)), abs(np.nanmin(xcorr)))
#         im = ax.imshow(xcorr[np.ix_(neuron_perm, neuron_perm)], cmap=cmap, vmin=-lim, vmax=lim)
#
#         for o in cluster_offsets[:-1]:
#             ax.plot([o + .5, o + .5], [-0.5, D_obs + 0.5], '-', lw=1, color='k')
#             ax.plot([-.5, D_obs + 0.5], [o + .5, o + .5], '-', lw=1, color='k')
#         plt.xlim(-0.5, D_obs - 0.5)
#         plt.ylim(D_obs - 0.5, -0.5)
#         plt.xlabel("neuron", fontsize=6)
#         plt.ylabel("neuron", fontsize=6)
#         # plt.xticks(np.arange(D_obs), neuron_names[neuron_perm], rotation=90, fontsize=3)
#         # plt.yticks(np.arange(D_obs), neuron_names[neuron_perm], fontsize=3)
#         plt.xticks([])
#         plt.yticks([])
#         plt.title("empirical correlation matrix", fontsize=8)
#
#         cax = create_axis_at_location(fig, 0.3, 0.3, 2.1, 0.075)
#         cbar = plt.colorbar(mappable=im, cax=cax, orientation="horizontal")
#         cbar.ax.tick_params(labelsize=4)
#         cbar.ax.set_xlabel("correlation coefficient", fontsize=6)
#
#         plt.savefig(os.path.join(fig_dir, "permuted_correlation.pdf"))
#         plt.savefig(os.path.join(fig_dir, "permuted_correlation.png"), dpi=300)
#
#         plt.close("all")
#
#     if do_plot_similarity:
#
#         S_true = C.dot(C.T)
#         S_true /= np.linalg.norm(C, axis=1)[:, None]
#         S_true /= np.linalg.norm(C, axis=1)[None, :]
#
#         # Plot the similarity matrix
#         fig = plt.figure(figsize=(2.5, 3.0))
#         ax = create_axis_at_location(fig, 0.3, 0.6, 2.1, 2.1)
#
#         cmap = gradient_cmap([colors[0], np.ones(3), colors[1]])
#         cmap.set_bad(0.5 * np.ones(3))
#         np.fill_diagonal(S_true, np.nan)
#         im = ax.imshow(S_true[np.ix_(neuron_perm, neuron_perm)], cmap=cmap, vmin=-1, vmax=1)
#
#         for o in cluster_offsets[:-1]:
#             ax.plot([o + .5, o + .5], [-0.5, D_obs + 0.5], '-', lw=1, color='k')
#             ax.plot([-.5, D_obs + 0.5], [o + .5, o + .5], '-', lw=1, color='k')
#         plt.xlim(-0.5, D_obs - 0.5)
#         plt.ylim(D_obs - 0.5, -0.5)
#         plt.xlabel("neuron", fontsize=6)
#         plt.ylabel("neuron", fontsize=6)
#         # plt.xticks(np.arange(D_obs), neuron_names[neuron_perm], rotation=90, fontsize=3)
#         # plt.yticks(np.arange(D_obs), neuron_names[neuron_perm], fontsize=3)
#         plt.xticks([])
#         plt.yticks([])
#         plt.title("embedding similarity", fontsize=8)
#
#         cax = create_axis_at_location(fig, 0.3, 0.3, 2.1, 0.075)
#         cbar = plt.colorbar(mappable=im, cax=cax, orientation="horizontal")
#         cbar.ax.tick_params(labelsize=4)
#         cbar.ax.set_xlabel("cosine similarity", fontsize=6)
#
#         plt.savefig(os.path.join(fig_dir, "permuted_similarity.pdf"))
#         plt.savefig(os.path.join(fig_dir, "permuted_similarity.png"), dpi=300)
#
#         plt.close("all")
#
#     if do_plot_cluster_embedding:
#
#         # fig = plt.figure(figsize=(1, 1))
#         # ax = fig.add_subplot(111, projection="3d")
#         # for cluster in range(N_clusters):
#         #     for i in np.where(neuron_clusters == cluster)[0]:
#         #         ci = C[i]
#         #         ci /= np.linalg.norm(ci)
#         #         ax.plot([0, ci[0]], [0, ci[1]], [0, ci[2]],
#         #                  '-k', lw=.5)
#         #         ax.plot([ci[0]], [ci[1]], [ci[2]],
#         #                 'ok', markersize=2)
#         #         # ax.text(1.1 * ci[0], 1.1 * ci[1], 1.1 * ci[2], neuron_names[i], fontsize=6)
#         #
#         #     ax.set_xlabel("dim 1", labelpad=-17, fontsize=6)
#         #     ax.set_ylabel("dim 2", labelpad=-17, fontsize=6)
#         #     ax.set_zlabel("dim 3", labelpad=-17, fontsize=6)
#         #     ax.set_xticklabels([])
#         #     ax.set_yticklabels([])
#         #     ax.set_zticklabels([])
#         #     ax.set_xlim(-.75, .75)
#         #     ax.set_ylim(-.75, .75)
#         #     ax.set_zlim(-.75, .75)
#         #
#         # plt.tight_layout(pad=0.05)
#         # plt.savefig(os.path.join(fig_dir, "embedding.pdf"))
#
#         fig = plt.figure(figsize=(1., 3.0))
#         ax = create_axis_at_location(fig, 0.4, 0.6, 0.5, 2.1)
#         C_norm = C / np.linalg.norm(C, axis=1, keepdims=True)
#         lim = abs(C_norm).max()
#         cmap = gradient_cmap([colors[0], np.ones(3), colors[1]])
#         ax.imshow(C_norm[neuron_perm][:,:-1], aspect="auto", vmin=-lim, vmax=lim, cmap=cmap)
#
#         for o in cluster_offsets[:-1]:
#             ax.plot([-.5, 10-.5], [o+.5, o+.5], '-', lw=1, color='k')
#
#         ax.set_ylabel("neuron", labelpad=1, fontsize=6)
#         plt.yticks(np.arange(D_obs), neuron_names[neuron_perm], fontsize=3)
#         ax.set_xlabel("latent\ndimension", labelpad=-1, fontsize=6)
#         ax.set_xticks([0, 9])
#         ax.set_xticklabels([1, 10], fontsize=4)
#         ax.set_title("emission\nmatrix", fontsize=8)
#
#         plt.savefig(os.path.join(fig_dir, "C.pdf"))
#         plt.close("all")
#
#     if do_plot_data:
#         all_ys = np.vstack(ys)
#         all_ms = np.vstack(ms)
#         all_ys[~all_ms] = np.nan
#         scales = np.nanstd(all_ys, axis=0)
#
#         for i in range(N_worms):
#             ysm = best_model.smooth(ys[i], mask=ms[i],
#                               inputs=np.ones((ys[i].shape[0], 1)),
#                               group=i)
#
#             y = ys[i].copy()
#             spc = 5
#
#             n_start = 9 * 60 * 3
#             n_frames = 3 * 60 * 3 + 1
#             t = n_start / 180.0 + np.arange(n_frames) / 180.0
#
#             fig = plt.figure(figsize=(3, 5))
#             left = .5 if i == 0 else .1
#             ax = create_axis_at_location(fig, left, .4, 2.4, 4.43)
#             offset = 0
#             ticks = []
#             for c in range(N_clusters):
#                 for n in neuron_perm:
#                     if neuron_clusters[n] != c:
#                         continue
#
#                     if ms[i][0, n]:
#                         plt.plot(t, -y[n_start:n_start+n_frames, n] / scales[n] + spc * offset, '-', color=colors[3], lw=.5)
#                     else:
#                         # plt.plot(t, np.zeros_like(t) + spc * offset, ':', color='k', lw=.5)
#                         pass
#                     plt.plot(t, -ysm[n_start:n_start+n_frames, n] / scales[n] + spc * offset, '-', color='k', lw=.5)
#
#                     ticks.append(offset * spc)
#                     offset += 1
#
#                 # Add an extra space between clusters
#                 offset += 2
#
#             # Remove last space
#             offset -= 2
#
#             if i == 0:
#                 plt.yticks(ticks, neuron_names[neuron_perm], fontsize=5)
#             else:
#                 plt.yticks([])
#
#             plt.ylim(offset * spc, -spc)
#             plt.ylim(offset * spc, -spc)
#             plt.xlim(t[0], t[-1])
#             plt.xticks(fontsize=6)
#             plt.xlabel("time (min)", fontsize=8)
#
#             plt.title("worm {} differenced Ca++".format(i+1), fontsize=8)
#
#             plt.savefig(os.path.join(fig_dir, "y_{}.pdf".format(i)))
#
#         plt.close("all")
#
#     if do_plot_data_wide:
#         all_ys = np.vstack(ys)
#         all_ms = np.vstack(ms)
#         all_ys[~all_ms] = np.nan
#         scales = np.nanstd(all_ys, axis=0)
#
#         for i in range(N_worms):
#             ysm = best_model.smooth(ys[i], mask=ms[i],
#                               inputs=np.ones((ys[i].shape[0], 1)),
#                               group=i)
#
#             y = ys[i].copy()
#             spc = 5
#
#             n_start = 9 * 60 * 3
#             n_frames = 3 * 60 * 3 + 1
#             t = n_start / 180.0 + np.arange(n_frames) / 180.0
#
#             fig = plt.figure(figsize=(6, 5))
#             ax = create_axis_at_location(fig, 0.5, 0.4, 5.4, 4.43)
#             offset = 0
#             ticks = []
#             for c in range(1):
#                 for n in neuron_perm:
#                     if neuron_clusters[n] != c:
#                         continue
#
#                     if ms[i][0, n]:
#                         plt.plot(t, -y[n_start:n_start+n_frames, n] / scales[n] + spc * offset, '-', color=colors[3], lw=.5)
#                     else:
#                         # plt.plot(t, np.zeros_like(t) + spc * offset, ':', color='k', lw=.5)
#                         pass
#                     plt.plot(t, -ysm[n_start:n_start+n_frames, n] / scales[n] + spc * offset, '-', color='k', lw=.5)
#
#                     ticks.append(offset * spc)
#                     offset += 1
#
#                 # Add an extra space between clusters
#                 offset += 2
#
#             # Remove last space
#             offset -= 2
#
#             plt.yticks(ticks, neuron_names[neuron_perm], fontsize=5)
#             plt.ylim(offset * spc, -spc)
#             plt.ylim(offset * spc, -spc)
#             plt.xlim(t[0], t[-1])
#             plt.xticks(fontsize=6)
#             plt.xlabel("time (min)", fontsize=8)
#
#             plt.title("Worm {} Differenced Ca++".format(i+1), fontsize=8)
#
#             plt.savefig(os.path.join(fig_dir, "y_wide_{}.pdf".format(i)))
#
#         plt.close("all")
#
#     if do_plot_data_zoom:
#         all_ys = np.vstack(ys)
#         all_ms = np.vstack(ms)
#         all_ys[~all_ms] = np.nan
#         scales = np.nanstd(all_ys, axis=0)
#
#         for i in range(N_worms):
#             ysm = best_model.smooth(ys[i], mask=ms[i],
#                                     inputs=np.ones((ys[i].shape[0], 1)),
#                                     group=i)
#
#             y = ys[i].copy()
#             spc = 5
#
#             n_start = 10 * 60 * 3
#             n_frames = int(.5 * 60) * 3 + 1
#             t = (n_start + np.arange(n_frames)) / 180.0
#
#             fig = plt.figure(figsize=(2.15, 5))
#             ax = create_axis_at_location(fig, 0.5, 0.4, 1.5, 4.43)
#             offset = 0
#             ticks = []
#             for c in range(N_clusters):
#                 for n in neuron_perm:
#                     if neuron_clusters[n] != c:
#                         continue
#
#                     if ms[i][0, n]:
#                         plt.plot(t, -y[n_start:n_start + n_frames, n] / scales[n] + spc * offset, '-', color=colors[3],
#                                  lw=.5)
#                     else:
#                         # plt.plot(t, np.zeros_like(t) + spc * offset, ':', color='k', lw=.5)
#                         pass
#                     plt.plot(t, -ysm[n_start:n_start + n_frames, n] / scales[n] + spc * offset, '-', color='k', lw=.5)
#
#                     ticks.append(offset * spc)
#                     offset += 1
#
#                 # Add an extra space between clusters
#                 offset += 2
#
#             # Remove last space
#             offset -= 2
#
#             plt.yticks(ticks, neuron_names[neuron_perm], fontsize=5)
#             plt.ylim(offset * spc, -spc)
#             plt.ylim(offset * spc, -spc)
#             plt.xlim(t[0], t[-1])
#             plt.xticks([t[0], t[n_frames//2], t[-1]], fontsize=6)
#             plt.xlabel("time (min)", fontsize=8)
#
#             plt.title("Worm {} Differenced Ca++".format(i + 1), fontsize=8)
#
#             plt.savefig(os.path.join(fig_dir, "y_zoom_{}.pdf".format(i)))
#
#         plt.close("all")
#
#     if do_plot_data_as_matrix:
#         all_ys = np.vstack(ys)
#         all_ms = np.vstack(ms)
#         all_ys[~all_ms] = np.nan
#         scales = np.nanstd(all_ys, axis=0)
#         all_ys /= scales[None, :]
#         # ylim = 1.05 * np.nanmax(abs(all_ys))
#         ylim = 8
#         cticks = [-8, -4, 0, 4, 8]
#         cticklabels = ["-8", "-4", " 0", " 4", " 8"]
#
#         fig = plt.figure(figsize=(3.85, 5))
#
#         for i in range(N_worms):
#             ax = create_axis_at_location(fig, 0.4, 5 - (i+1) * 0.92, 1.5, 0.75)
#
#             # Plot the observed data
#             y = ys[i].copy()
#             y /= scales[None, :]
#             y[~ms[i]] = np.nan
#
#             cmap = gradient_cmap([colors[0], np.ones(3), colors[1]])
#             cmap.set_bad(0.75 * np.ones(3))
#             plt.imshow(y[:,neuron_perm].T, cmap=cmap, vmin=-ylim, vmax=ylim,
#                        aspect="auto", interpolation="nearest")
#
#             # Plot cluster dividers
#             for o in cluster_offsets[:-1]:
#                 plt.plot([0, 3240], [o + .5, o + .5], '-', lw=.5, color='k')
#
#             plt.yticks(cluster_offsets, cluster_offsets+1, fontsize=6)
#             plt.ylabel("neurons", fontsize=8)
#
#             plt.xlim(0, 3240)
#             if i == N_worms - 1:
#                 plt.xticks(np.arange(19, step=3) * 60 * 3, np.arange(19, step=3), fontsize=6)
#                 plt.xlabel("time (min)", fontsize=8)
#             else:
#                 plt.xticks([])
#
#             plt.title("Worm {} Differenced Ca++".format(i+1), y=.95, fontsize=8)
#
#             # Plot the smoothed activity
#             ax = create_axis_at_location(fig, 2.0, 5 - (i + 1) * 0.92, 1.5, 0.75)
#             ysm = best_model.smooth(ys[i], mask=ms[i],
#                                     inputs=np.ones((ys[i].shape[0], 1)),
#                                     group=i)
#             ysm /= scales[None, :]
#             im = plt.imshow(ysm[:, neuron_perm].T, cmap=cmap, vmin=-ylim, vmax=ylim,
#                             aspect="auto", interpolation="nearest")
#
#             # Plot cluster dividers
#             for o in cluster_offsets[:-1]:
#                 plt.plot([0, 3240], [o + .5, o + .5], '-', lw=.5, color='k')
#
#             plt.yticks([])
#             plt.xlim(0, 3240)
#             if i == N_worms - 1:
#                 plt.xticks(np.arange(19, step=3) * 60 * 3, np.arange(19, step=3), fontsize=6)
#                 plt.xlabel("time (min)", fontsize=8)
#             else:
#                 plt.xticks([])
#
#             plt.title("Worm {} Smoothed".format(i + 1), y=.95, fontsize=8)
#
#             # Make a colorbar
#             cax = create_axis_at_location(fig, 3.55, 5 - (i + 1) * 0.92, 0.075, 0.75)
#             cbar = plt.colorbar(im, cax=cax, ticks=cticks)
#             cbar.ax.tick_params(labelsize=6)
#             cbar.ax.set_yticklabels(cticklabels)
#
#         plt.savefig(os.path.join(fig_dir, "y_matrix.pdf"))
#         plt.close("all")
#
#     if do_plot_cluster_locations:
#         import pandas
#         locs = pandas.read_csv("wormatlas_locations.csv").values
#         l_values = np.unique(locs[:,1])
#
#         # for lv in l_values:
#         #     print("{} - {}".format(lv, np.sum(locs[:,1] == lv)))
#
#         fig = plt.figure(figsize=(2.75, 1.0))
#         # ax = fig.add_subplot(111)
#         ax = create_axis_at_location(fig, 0.3, 0.05, 2.1, .9)
#         spc = 1
#
#         sizes = dict([(lv, 2) for lv in l_values])
#         ax.plot([0.08, 0.25], np.zeros(2), ':k', lw=0.5)
#         for name in neuron_names:
#             i2 = np.where(locs[:, 0] == name)[0][0]
#             l = locs[i2, 1]
#             plt.plot(l, 0, 'ko', markersize=sizes[l], alpha=1.0)
#             sizes[l] += .75
#         ax.text(0.07, 0, "all neurons", fontsize=6, style="italic", horizontalalignment="right")
#
#         for cluster in range(N_clusters):
#             sizes = dict([(lv, 2) for lv in l_values])
#             yoff = spc + spc * (cluster+1)
#             ax.plot([0.08, 0.25], yoff * np.ones(2), ':k', lw=0.5)
#             for i1 in np.where(neuron_clusters == cluster)[0]:
#                 i2 = np.where(locs[:,0] == neuron_names[i1])[0][0]
#                 l = locs[i2, 1]
#                 plt.plot(l, yoff, 'o', color=colors[0], markersize=sizes[l], alpha=1.0)
#                 sizes[l] += .75
#
#             if cluster % 2 == 0:
#                 ax.text(0.07, yoff+.5, "cluster {}".format(cluster+1), fontsize=6, style="italic", horizontalalignment="right")
#
#
#             print("cluster ", cluster)
#             print(sizes)
#         # Make a legend
#         ax.plot(.273, 2, 'ko', markersize=2)
#         ax.text(.285, 2.5, "1 neuron", fontsize=6)
#         ax.plot(.273, 4, 'ko', markersize=3.5)
#         ax.text(.285, 4.5, "3 neurons", fontsize=6)
#         ax.plot(.273, 6, 'ko', markersize=5)
#         ax.text(.285, 6.5, "5 neurons", fontsize=6)
#         ax.plot(.273, 8, 'ko', markersize=6.5)
#         ax.text(.285, 8.5, "7 neurons", fontsize=6)
#         ax.plot(.273, 10, 'ko', markersize=8)
#         ax.text(.285, 10.5, "9 neurons", fontsize=6)
#
#         ax.spines["left"].set_visible(False)
#         ax.spines["top"].set_visible(False)
#         ax.spines["right"].set_visible(False)
#         ax.spines["bottom"].set_visible(False)
#         ax.set_xlim([0.01, 0.30])
#         ax.set_yticks([])
#         ax.set_ylim(spc * (N_clusters + 3), -3 * spc)
#         plt.xticks([0.08, 0.15, 0.25], fontsize=4)
#
#         # Draw lines for the head ganglia
#         # Reference http://www.wormatlas.org/ver1/MoW_built0.92/art/intro_figs/figure2.jpg
#         # Anterior ganglion is from OLGDL (0.09) to BAGL (0.1)
#         # Dorsal is from RID (0.11) to ASKL (0.13)
#         # Ventral ganglion is from ASKL (.13) to SABVL (.2)
#         # RVG is from VB02 (0.19) to VD02 (0.26)
#         # ax.plot(0.095, spc * (N_clusters + 2), 'k^', markersize=3)
#         ax.plot([.09, .1], spc * (N_clusters + 2) * np.ones(2), '-k', lw=1)
#         # ax.plot(0.1, spc * (N_clusters + 2), 'k^', markersize=3)
#         ax.plot([.11, .13], spc * (N_clusters + 2) * np.ones(2), '-k', lw=1)
#         # ax.plot(0.15, spc * (N_clusters + 2), 'k^', markersize=3)
#         ax.plot([.14, .17], spc * (N_clusters + 2) * np.ones(2), '-k', lw=1)
#         # ax.plot(.225, spc * (N_clusters + 2), 'k^', markersize=3)
#         ax.plot([.19, .25], (spc * (N_clusters + 2)) * np.ones(2), '-k', lw=1)
#
#
#         # ax.set_xlabel("location along head-tail axis", fontsize=6)
#         plt.savefig(os.path.join(fig_dir, "cluster_locs.pdf"))
#
#         plt.close("all")
#
#     if do_plot_cluster_types:
#         import pandas
#         all_types = pandas.read_csv("wormatlas_celltypes.csv", header=0,
#                                  dtype=dict(Name=str, S=int, I=int, M=int, P=int, U=int))
#
#         type_names = ["S", "I", "M", "P", "U"]
#         type_colors = [colors[7], colors[1], colors[4], colors[6], "gray"]
#
#         fig = plt.figure(figsize=(1.6, 1.0))
#         # ax = create_axis_at_location(fig, 0.3, 0.05, 2.1, .9)
#         ax = fig.add_subplot(111)
#
#         # Break down cells into 5 types: sensory, inter, motor, poly, unknown
#         for cluster in range(N_clusters):
#             inds = np.where(neuron_clusters == cluster)[0]
#             cluster_types = np.zeros(5)
#             for i, key in enumerate(type_names):
#                 cluster_types[i] = all_types[key].values[inds].sum()
#             assert cluster_types.sum() == len(inds)
#             bottom = np.concatenate(([0], np.cumsum(cluster_types)[:-1]))
#
#             for i in range(5):
#                 ax.bar(cluster, cluster_types[i], width=0.8, bottom=bottom[i],
#                        color=type_colors[i],
#                        edgecolor='k', linewidth=.5,
#                        label=type_names[i] if cluster == 0 else None
#                        )
#
#         plt.xticks(np.arange(N_clusters), np.arange(N_clusters) + 1)
#         plt.xlabel("cluster", fontsize=6, labelpad=-1)
#         plt.ylabel("cell count", fontsize=6)
#         plt.ylim(0, 15)
#
#         ax.tick_params(labelsize=6)
#         # ax.spines["left"].set_visible(False)
#         ax.spines["top"].set_visible(False)
#         ax.spines["right"].set_visible(False)
#         # ax.spines["bottom"].set_visible(False)
#         ax.legend(loc="upper left", ncol=3, fontsize=6, borderpad=0, handlelength=1, labelspacing=.5, columnspacing=.5, handletextpad=.5)
#
#         plt.tight_layout(pad=.1)
#
#         plt.savefig(os.path.join(fig_dir, "cluster_types.pdf"))
#         plt.show()



def heldout_neuron_identification(N_heldout=10, D_latent=10, is_hierarchical=True, seed=0):
    np.random.seed(seed)

    # Artificially hold out some neurons for identification test
    n_observed = np.array([hm[0] for hm in ms]).sum(0)
    heldout_neurons = []
    heldout_masks = []
    for i,m in enumerate(ms):
        observed = np.where(m[0] & (n_observed > 1))[0]
        hn = np.random.choice(observed, size=N_heldout, replace=False)
        print("worm {}. holding out: {}. Number observed: {}".format(i, hn, m[0].sum() - N_heldout))
        hm = m.copy()
        hm[:,hn] = False
        n_observed[hn] -= 1

        heldout_neurons.append(hn)
        heldout_masks.append(hm)

    # Make sure that we haven't thrown out the only instance of a neuron
    assert (np.all(np.vstack(heldout_masks).sum(0) > 0))
    print("number of observations per neuron: ")
    print(n_observed)

    # Fit an lds to this artificially heldout data
    @cached("heldout_hlds")
    def _fit():
        print("Fitting hlds to heldout data")
        model, lls, _ = _fit_lds(D_latent,
                                  is_hierarchical=is_hierarchical,
                                  datas=ys,
                                  masks=heldout_masks,
                                  compute_hll=False)

        return model, lls
    model, lls = _fit()

    # Now use model.C to predict neuron identities
    # Fit a linear regression to get the mapping from X to Y for all heldout neurons
    xs = [s.gaussian_states for s in model.states_list]
    C = model.C
    ypreds = [x.dot(C.T) for x in xs]

    # Now the coup de grâce -- use all the unlabeled neurons
    all_ys, all_ms, _, _, all_neuron_names = load_data(include_unnamed=True)

    # Compute similarity between heldout neuron and true identity
    # and between other unlabeled neurons and true identity
    s_heldouts = []
    s_others = []
    rankss = []
    n_unlabeleds = []
    accs = []
    for worm in range(N_worms):
        possible = ~heldout_masks[worm][0]
        possible_rows = np.where(possible)[0]
        others = np.where([name.startswith('worm{}'.format(worm)) for name in all_neuron_names])[0]
        n_unlabeled = N_heldout + len(others)

        S_heldout = np.zeros((possible.sum(), N_heldout))
        S_other = np.zeros((possible.sum(), len(others)))

        for i, n in enumerate(others):
            assert np.all(np.isfinite(xs[worm]))
            assert np.all(np.isfinite(all_ys[worm][:, n]))
            X = xs[worm]
            yn = all_ys[worm][:,n]
            c_reg = np.linalg.solve(X.T.dot(X) + 1e-8 * np.eye(D_latent), X.T.dot(yn))

            # Compute cosine similarity between the inferred C and the hrslds emission matrix
            S_other[:,i] = np.dot(C[possible], c_reg) / np.linalg.norm(C[possible], axis=1) / np.linalg.norm(c_reg)

        for i, n in enumerate(heldout_neurons[worm]):
            assert np.all(np.isfinite(xs[worm]))
            assert np.all(np.isfinite(ys[worm][:, n]))
            X = xs[worm]
            yn = ys[worm][:, n]
            c_reg = np.linalg.solve(X.T.dot(X) + 1e-8 * np.eye(D_latent), X.T.dot(yn))

            # Compute cosine similarity between the inferred C and the hrslds emission matrix
            S_heldout[:,i] = np.dot(C[possible], c_reg) / (np.linalg.norm(C[possible], axis=1) * np.linalg.norm(c_reg) + 1e-8)

        S_full = np.hstack((S_heldout, S_other))

        # Match the neurons with Hungarian algorithm
        from scipy.optimize import linear_sum_assignment
        rows, cols = linear_sum_assignment(-S_full)
        assert np.all(np.diff(rows) == 1), "All rows should have been matched!"

        # Count number of correct assignments and ranks of similarity to true neuron
        num_correct = 0
        ranks = []
        for i,n in enumerate(heldout_neurons[worm]):
            r = np.where(possible_rows == n)[0][0]
            perm_n = np.argsort(S_full[r])[::-1]
            rank = np.where(perm_n == i)[0][0]
            print("True rank {}: {} / {}".format(i+1, rank + 1, n_unlabeled))

            # Count how many correct assignments we made
            if cols[r] == i:
                num_correct += 1

            ranks.append(rank)
            s_heldouts.append([S_heldout[r, i]])
            s_others.append(S_heldout[r, :i])
            s_others.append(S_heldout[r, i+1:])
            s_others.append(S_other[r, :])

        n_unlabeleds.append(n_unlabeled)
        accs.append(num_correct / N_heldout)
        print("Fraction correct: {:.2f}".format(accs[-1]))
        rankss.append(ranks)


    # Plot results
    plt.figure(figsize=(2, 2))
    lim = 1
    bins = 20
    plt.hist(np.concatenate(s_others), np.linspace(-lim, lim, bins + 1),
             color=colors[0], alpha=0.75, normed=True, label="Incorrect candidates")
    plt.hist(np.concatenate(s_heldouts), np.linspace(-lim, lim, bins + 1),
             color=colors[1], alpha=0.75, normed=True, label="Correct candidates")
    plt.legend(loc="upper left", fontsize=6)
    plt.title("")
    plt.xlabel("cosine similarity", fontsize=8)
    plt.ylabel("probability density", fontsize=8)
    plt.tick_params(labelsize=6)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "similarity_comparison.pdf"))
    plt.savefig(os.path.join(fig_dir, "similarity_comparison.png"))

    print("Table of results")
    table = ""
    header = "$N_{\\mathsf{labeled}$ & "
    header += "$N_{\\mathsf{unlabeled}$ & "
    for i in range(N_heldout):
        header += "{} & ".format(i+1)
    header += "Matching Acc. \\\\"
    table += header + "\n"

    for worm in range(N_worms):
        row = "{} & ".format(heldout_masks[worm][0].sum())
        row += "{} & ".format(n_unlabeleds[worm])

        total = heldout_masks[worm][0].sum() + n_unlabeleds[worm]
        assert total == all_ms[worm][0].sum()

        for i in range(N_heldout):
            row += "{} & ".format(rankss[worm][i]+1)
        row += "{:.2f} \\\\".format(accs[worm])
        table += row + "\n"

    print(table)
    plt.close("all")


def heldout_neuron_identification_corr(N_heldout=10, D_latent=10, is_hierarchical=True, seed=0):
    np.random.seed(seed)

    # Artificially hold out some neurons for identification test
    n_observed = np.array([hm[0] for hm in ms]).sum(0)
    heldout_neurons = []
    heldout_masks = []
    for i, m in enumerate(ms):
        observed = np.where(m[0] & (n_observed > 1))[0]
        hn = np.random.choice(observed, size=N_heldout, replace=False)
        print("worm {}. holding out: {}. Number observed: {}".format(i, hn, m[0].sum() - N_heldout))
        hm = m.copy()
        hm[:, hn] = False
        n_observed[hn] -= 1

        heldout_neurons.append(hn)
        heldout_masks.append(hm)

    # Make sure that we haven't thrown out the only instance of a neuron
    assert (np.all(np.vstack(heldout_masks).sum(0) > 0))
    print("number of observations per neuron: ")
    print(n_observed)

    # Fit an lds to this artificially heldout data
    @cached("heldout_hlds")
    def _fit():
        print("Fitting hlds to heldout data")
        model, lls, _ = _fit_lds(D_latent,
                                 is_hierarchical=is_hierarchical,
                                 datas=ys,
                                 masks=heldout_masks,
                                 compute_hll=False)

        return model, lls

    model, lls = _fit()

    # Now use model.C to predict neuron identities
    # Fit a linear regression to get the mapping from X to Y for all heldout neurons
    xs = [s.gaussian_states for s in model.states_list]
    C = model.C
    ypreds = [x.dot(C.T) for x in xs]

    # Now the coup de grâce -- use all the unlabeled neurons
    all_ys, all_ms, _, _, all_neuron_names = load_data(include_unnamed=True)

    # Compute similarity between heldout neuron and true identity
    # and between other unlabeled neurons and true identity
    s_heldouts = []
    s_others = []
    rankss = []
    n_unlabeleds = []
    accs = []
    for worm in range(N_worms):
        possible = ~heldout_masks[worm][0]
        possible_rows = np.where(possible)[0]
        others = np.where([name.startswith('worm{}'.format(worm)) for name in all_neuron_names])[0]
        n_unlabeled = N_heldout + len(others)

        # Predict the activity of unlabeled neurons under the model
        yp = ypreds[worm][:, possible]

        S_heldout = np.zeros((possible.sum(), N_heldout))
        S_other = np.zeros((possible.sum(), len(others)))

        for i, n in enumerate(others):
            assert np.all(np.isfinite(xs[worm]))
            assert np.all(np.isfinite(all_ys[worm][:, n]))
            yn = all_ys[worm][:, n]

            # Compute cosine similarity between the inferred C and the hrslds emission matrix
            S_other[:, i] = np.corrcoef(yp, yn, rowvar=False)[:-1, -1]

        for i, n in enumerate(heldout_neurons[worm]):
            assert np.all(np.isfinite(xs[worm]))
            assert np.all(np.isfinite(ys[worm][:, n]))
            yn = ys[worm][:, n]

            # Compute cosine similarity between the inferred C and the hrslds emission matrix
            S_heldout[:, i] = np.corrcoef(yp, yn, rowvar=False)[:-1, -1]

        S_full = np.hstack((S_heldout, S_other))

        # Match the neurons with Hungarian algorithm
        from scipy.optimize import linear_sum_assignment
        rows, cols = linear_sum_assignment(-S_full)
        assert np.all(np.diff(rows) == 1), "All rows should have been matched!"

        # Count number of correct assignments and ranks of similarity to true neuron
        num_correct = 0
        ranks = []
        for i, n in enumerate(heldout_neurons[worm]):
            r = np.where(possible_rows == n)[0][0]
            perm_n = np.argsort(S_full[r])[::-1]
            rank = np.where(perm_n == i)[0][0]
            print("True rank {}: {} / {}".format(i + 1, rank + 1, n_unlabeled))

            # Count how many correct assignments we made
            if cols[r] == i:
                num_correct += 1

            ranks.append(rank)
            s_heldouts.append([S_heldout[r, i]])
            s_others.append(S_heldout[r, :i])
            s_others.append(S_heldout[r, i + 1:])
            s_others.append(S_other[r, :])

        n_unlabeleds.append(n_unlabeled)
        accs.append(num_correct / N_heldout)
        print("Fraction correct: {:.2f}".format(accs[-1]))
        rankss.append(ranks)

    # Plot results
    plt.figure(figsize=(2, 1.9))
    lim = 1
    bins = np.linspace(-lim, lim, 26)
    p_other, _ = np.histogram(np.concatenate(s_others), bins)
    p_other = p_other.astype(np.float) / np.concatenate(s_others).size
    p_heldouts, _ = np.histogram(np.concatenate(s_heldouts), bins)
    p_heldouts = p_heldouts.astype(np.float) / np.concatenate(s_heldouts).size

    plt.bar(bins[:-1], p_other, width=bins[1]-bins[0],
            color=colors[0], alpha=0.75, label="incorrect")
    plt.bar(bins[:-1], p_heldouts, width=bins[1]-bins[0],
            color=colors[1], alpha=0.75, label="correct")

    plt.legend(loc="upper left", fontsize=6)
    plt.title("")
    plt.xlabel("correlation coefficient", fontsize=6)
    plt.ylabel("probability", fontsize=6)
    plt.tick_params(labelsize=6)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "similarity_comparison_corr.pdf"))
    plt.savefig(os.path.join(fig_dir, "similarity_comparison_corr.png"))

    print("Table of results")
    table = ""
    header = "$N_{\\mathsf{labeled}$ & "
    header += "$N_{\\mathsf{unlabeled}$ & "
    for i in range(N_heldout):
        header += "{} & ".format(i + 1)
    header += "Matching Acc. \\\\"
    table += header + "\n"

    for worm in range(N_worms):
        row = "{} & ".format(heldout_masks[worm][0].sum())
        row += "{} & ".format(n_unlabeleds[worm])

        total = heldout_masks[worm][0].sum() + n_unlabeleds[worm]
        assert total == all_ms[worm][0].sum()

        for i in range(N_heldout):
            row += "{} & ".format(rankss[worm][i] + 1)
        row += "{:.2f} \\\\".format(accs[worm])
        table += row + "\n"

    print(table)
    plt.close("all")

    # Make a schematic of how the matching works
    nho = heldout_neurons[0][2]
    plt.figure(figsize=(1.75, 1.75))
    tt = np.arange(60*3)
    N_plot = 5
    yticklabels = []
    for i,nobs in enumerate(np.where(heldout_masks[0][0])[0][:N_plot-1]):
        plt.plot(tt / 3.0, -i + ys[0][tt, nobs], '-', color=colors[3])
        yticklabels.append(neuron_names[nobs])

    plt.plot(tt / 3.0, -N_plot + 1 + ypreds[0][tt, nho], '-k')
    yticklabels.append(neuron_names[nho])

    plt.yticks(-np.arange(N_plot), yticklabels)
    plt.xlabel("time (sec)", fontsize=6)
    plt.tick_params(labelsize=6)
    for sp in ["top", "left", "right"]:
        plt.gca().spines[sp].set_visible(False)
    plt.tight_layout(0.1)
    plt.savefig(os.path.join(fig_dir, "cartoon_observed.pdf"))

    # Plot the heldout activity
    plt.figure(figsize=(1.75, 1.75))
    tt = np.arange(60 * 3)
    N_plot = 5
    yticklabels = []

    plt.plot(tt / 3.0, ys[0][tt, nho], '-', color=colors[1])
    yticklabels.append("correct")
    print(np.corrcoef(ypreds[0][:, nho], ys[0][:, nho])[0, 1])
    i = 0
    for nobs in heldout_neurons[0]:
        if nobs == nho:
            continue

        plt.plot(tt / 3.0, -1 - i + ys[0][tt, nobs], '-', color=colors[0])
        # yticklabels.append(neuron_names[nobs])
        yticklabels.append("incorrect {}".format(i+1))
        print(np.corrcoef(ypreds[0][:, nho], ys[0][:, nobs])[0, 1])
        i += 1
        if i == N_plot - 1:
            break

    plt.yticks(-np.arange(N_plot), yticklabels)
    # plt.gca().yaxis.tick_right()
    plt.xlabel("time (sec)", fontsize=6)
    plt.tick_params(labelsize=6)
    for sp in ["top", "left", "right"]:
        plt.gca().spines[sp].set_visible(False)
    plt.tight_layout(0.1)
    plt.savefig(os.path.join(fig_dir, "cartoon_heldout.pdf"))
    plt.close("all")


if __name__ == "__main__":
    ys, ms, z_trues, z_true_key, neuron_names = load_data(include_unnamed=False)
    D_obs = ys[0].shape[1]

    # compute the correlation coefficients for each pair of neurons
    xcorr = comput_xcorr(ys, ms)

    # Split test train
    ytrains, ytests, train_inds = list(zip(*[_split_test_train(y, train_frac=0.8) for y in ys]))
    mtrains, mtests, _ = list(zip(*[_split_test_train(m, train=train) for m, train in zip(ms, train_inds)]))
    z_true_trains, z_true_tests, _ = list(zip(*[_split_test_train(z, train=train) for z, train in zip(z_trues, train_inds)]))
    n_trains = np.array([mtr.sum() for mtr in mtrains])
    n_tests = np.array([mte.sum() for mte in mtests])

    D_latents = np.arange(2, 21, 2)
    fit_results = fit_all_models(D_latents)
    best_model = fit_results["hier"][0]

    # Do an E step to smooth the latent states
    C = best_model.emission_distn.A
    xtrains = []
    xtests = []
    for i in range(N_worms):
        best_model.states_list[i].E_step()
        xtrains.append(best_model.states_list[i].smoothed_mus)

        best_model.add_data(ytests[i], mask=mtests[i],
                            inputs=np.ones((ytests[i].shape[0], 1)), group=i)
        s = best_model.states_list.pop()
        s.E_step()
        xtests.append(s.smoothed_mus)

    # Sort the states based on the correlation coefficient between their
    # 1D reconstruction of the data and the actual data
    dim_perm = order_latent_dims(xtrains, C, ytrains, mtrains)
    C = np.hstack((C[:, :-1][:, dim_perm], C[:, -1:]))
    xtrains = [x[:, dim_perm] for x in xtrains]
    xtests = [x[:, dim_perm] for x in xtests]

    # Cluster the neurons based on C
    neuron_perm, neuron_clusters = cluster_neruons(best_model)
    cluster_sizes = np.bincount(neuron_clusters, minlength=N_clusters)
    cluster_offsets = np.cumsum(cluster_sizes) - 1

    # Save out the results
    results = dict(
        xtrains=xtrains,
        xtests=xtests,
        ytrains=ytrains,
        ytests=ytests,
        mtrains=mtrains,
        mtests=mtests,
        z_true_trains=z_true_trains,
        z_true_tests=z_true_tests,
        z_key=z_true_key,
        best_model=best_model,
        D_latent=best_model.D_latent,
        C=C,
        perm=dim_perm,
        N_clusters=N_clusters,
        neuron_clusters=neuron_clusters,
        neuron_perm=neuron_perm
    )

    with open(os.path.join(results_dir, "lds_data.pkl"), "wb") as f:
        pickle.dump(results, f)

    # Plotting
    # plot_likelihoods(fit_results)
    # plot_best_model_results(best_model,
    #                         do_plot_pca=False,
    #                         do_plot_x_3d=True,
    #                         do_plot_x_2d=False,
    #                         do_plot_sigmasq=False,
    #                         do_plot_xcorr=False,
    #                         do_plot_similarity=False,
    #                         do_plot_cluster_embedding=False,
    #                         do_plot_cluster_locations=False,
    #                         do_plot_data=False,
    #                         do_plot_data_wide=False,
    #                         do_plot_data_zoom=False,
    #                         do_plot_data_as_matrix=False,
    #                         do_plot_cluster_types=False
    #                         )

    # heldout_neuron_identification_corr()
