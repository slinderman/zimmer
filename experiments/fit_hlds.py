import os
import pickle

import numpy as np
np.random.seed(1234)

from tqdm import tqdm
from functools import partial

# Plotting stuff
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
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


# Modeling stuff
from pyhsmm.util.general import relabel_by_usage

# Load worm modeling specific stuff
from zimmer.io import WormData, load_kato_key

from zimmer.models import HierarchicalLDS
from zimmer.emissions import HierarchicalDiagonalRegression
from zimmer.plotting import plot_3d_continuous_states
from pybasicbayes.distributions import Regression, DiagonalRegression
from pylds.models import MissingDataLDS

# IO
# run_num = 2
# results_dir = os.path.join("results", "2017-11-03-hlds", "run{:03d}".format(run_num))
# signal = "dff_diff"
results_dir = os.path.join("results", "2017-11-03-hlds", "run003_dff_bc")
signal = "dff_bc"

assert os.path.exists(results_dir)
fig_dir = os.path.join(results_dir, "figures")

N_worms = 5
N_clusters = 5


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
    worm_datas = [WormData(i, name="worm{}".format(i)) for i in range(N_worms)]

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
    for i in range(N_worms):
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


def plot_likelihoods(D_latents, final_lls, hlls, best_index,
                     color, name, axs=None):

    # Plot results of searching over models
    if axs is None:
        plt.figure(figsize=(6,3))
        ax1 = plt.subplot(121)
        ax2 = plt.subplot(122)
    else:
        ax1, ax2 = axs

    # for D_latent, lls in zip(D_latents, llss):
    #     plt.plot(lls, label=D_latent)
    # plt.legend(loc="lower right")
    ax1.plot(D_latents, final_lls / n_trains.sum(),
             '-', markersize=6, color=color)
    for index in range(len(D_latents)):
        if index != best_index:
            ax1.plot(D_latents[index], final_lls[index] / n_trains.sum(),
                     'o', markersize=6, color=color)

    ax1.plot(D_latents[best_index], final_lls[best_index] / n_trains.sum(),
             '*', markersize=10, color=color)
    ax1.set_xlabel("Latent Dimension")
    ax1.set_ylabel("Train Log Likelihood")

    ax2.plot(D_latents, hlls / n_tests.sum(),
             '-', markersize=6, color=color, label=name)

    for index in range(len(D_latents)):
        if index != best_index:
            ax2.plot(D_latents[index], hlls[index] /  n_tests.sum(),
                     'o', markersize=6, color=color)

    ax2.plot(D_latents[best_index], hlls[best_index] / n_tests.sum(),
             '*', markersize=10, color=color)
    ax2.set_xlabel("Latent Dimension")
    ax2.set_ylabel("Test Log Likelihood")

    return ax1, ax2


def order_latent_dims(xs, C, ytrains, mtrains):

    # Sort latent dimensions by how much variance they account for
    D_latent = xs[0].shape[1]
    corrcoeffs = np.zeros(D_latent)
    for d in range(D_latent):
        yobss = []
        yhats = []
        for i in range(N_worms):
            mask = mtrains[i][0]
            yobss.append(ytrains[i][:, mask].ravel())
            yhats.append(np.outer(xs[i][:,d], C[mask,d]).ravel())
        yobss = np.concatenate(yobss)
        yhats = np.concatenate(yhats)
        corrcoeffs[d] = np.corrcoef(yobss.ravel(), yhats.ravel())[0,1]
    return np.argsort(corrcoeffs)[::-1]


def cluster_neruons(best_model, seed=0):
    from pyhsmm.util.general import relabel_by_permutation
    from sklearn.cluster import SpectralClustering, KMeans
    C_true = best_model.emission_distn.A[:, :-1].copy()
    C_true /= np.linalg.norm(C_true, axis=1)[:, None]

    np.random.seed(seed)
    # cluster = SpectralClustering(n_clusters=N_clusters, affinity="precomputed")
    # cluster.fit((1 + S_true) / 2.0)
    # labels = cluster.labels_
    cluster = KMeans(n_clusters=N_clusters)
    cluster.fit(C_true)
    neuron_clusters = cluster.labels_

    avg_C = np.zeros((N_clusters, best_model.D_latent))
    for c in range(N_clusters):
        if not np.any(neuron_clusters == c):
            continue
        avg_C[c] = np.mean(C_true[neuron_clusters == c], axis=0)

    # Permute the cluster labels by doing PCA on the average C and sorting
    from sklearn.decomposition import PCA
    pca = PCA(n_components=1)
    weights = pca.fit_transform(avg_C)[:, 0]
    labels_perm = np.argsort(weights)
    neuron_clusters = relabel_by_permutation(neuron_clusters, np.argsort(labels_perm))

    # Lex sort by label then by emission vector
    # perm = np.lexsort(np.row_stack((C_true.T, labels)))
    neuron_perm = np.lexsort((neuron_names[:D_obs], neuron_clusters))

    return neuron_perm, neuron_clusters


def plot_best_model_results(best_model,
                            do_plot_x_3d=True,
                            do_plot_x_2d=True,
                            do_plot_sigmasq=True,
                            do_plot_similarity=True,
                            do_plot_cluster_embedding=True,
                            do_plot_data=True):

    # False color with the "true" label from Zimmer
    if do_plot_x_3d:
        for i in range(N_worms):
            plot_3d_continuous_states(xtrains[i], z_true_trains[i], colors,
                                      figsize=(4, 4),
                                      title="LDS Worm {} States (Zimmer Lables)".format(i + 1),
                                      results_dir=fig_dir,
                                      filename="xtr_zimmer_{}.pdf".format(i + 1),
                                      # lim=1.5,
                                      lw=1)
        plt.close("all")

        # for i in range(N_worms):
        #     plot_3d_continuous_states(xtests[i], z_true_tests[i], colors,
        #                               figsize=(4, 4),
        #                               title="LDS Worm {} States (Zimmer Lables)".format(i + 1),
        #                               results_dir=fig_dir,
        #                               filename="xte_zimmer_{}.pdf".format(i + 1),
        #                               lim=1.5,
        #                               lw=1)
        # plt.close("all")

    if do_plot_x_2d:
        pass

    if do_plot_sigmasq:

        # Look at the observation variance across worms
        sigma_obs = best_model.emission_distn.sigmasq_flat.copy()
        sigma_obs_mask = np.array([np.any(~m, axis=0) for m in ms])
        sigma_obs[sigma_obs_mask] = np.nan

        cmap = gradient_cmap([np.ones(3), colors[0]])
        cmap.set_bad(0.7 * np.ones(3))

        fig = plt.figure(figsize=(5.5, 2.))
        ax = fig.add_subplot(111)
        im = ax.imshow(np.sqrt(sigma_obs), vmin=0, aspect="auto", cmap=cmap)
        ax.set_xticks(np.arange(D_obs))
        ax.set_xticklabels(neuron_names, rotation="90", fontsize=6)
        ax.set_yticks(np.arange(N_worms))
        ax.set_yticklabels(np.arange(N_worms) + 1)
        ax.set_ylabel("Worm")

        ax.set_title("$\sigma_{\mathsf{obs}}$")

        plt.colorbar(im)
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, "observation_variance.pdf"))

    if do_plot_similarity:

        S_true = C.dot(C.T)
        S_true /= np.linalg.norm(C, axis=1)[:, None]
        S_true /= np.linalg.norm(C, axis=1)[None, :]

        # Plot the similarity matrix
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111)
        im = ax.imshow(S_true[np.ix_(neuron_perm, neuron_perm)], cmap="RdBu", vmin=-1, vmax=1)

        cluster_sizes = np.bincount(neuron_clusters, minlength=N_clusters)
        print("cluster sizes: ", cluster_sizes)

        offsets = np.cumsum(cluster_sizes) - 1
        for o in offsets[:-1]:
            ax.plot([o + .5, o + .5], [-0.5, D_obs + 0.5], '-', lw=2, color=0 * np.ones(3))
            ax.plot([-.5, D_obs + 0.5], [o + .5, o + .5], '-', lw=2, color=0 * np.ones(3))
        plt.xlim(-0.5, D_obs - 0.5)
        plt.ylim(D_obs - 0.5, -0.5)
        plt.xlabel("neuron")
        plt.ylabel("neuron")
        plt.xticks(np.arange(D_obs), neuron_names[neuron_perm], rotation=90, fontsize=6)
        plt.yticks(np.arange(D_obs), neuron_names[neuron_perm], fontsize=6)
        plt.title("similarity between neural embeddings")

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="3%", pad=0.1)
        plt.colorbar(mappable=im, cax=cax, label="cosine similarity")

        plt.tight_layout()

        plt.savefig(os.path.join(fig_dir, "permuted_similarity.pdf"))
        plt.savefig(os.path.join(fig_dir, "permuted_similarity.png"), dpi=300)

        plt.show()

    # if do_plot_cluster_embedding:
    #     for cluster in range(N_clusters):
    #         fig = plt.figure(figsize=(1.5, 1.5))
    #         ax = fig.add_subplot(111, projection="3d")
    #         for i in np.where(neuron_clusters == cluster)[0]:
    #             ci = C[i]
    #             ci /= np.linalg.norm(ci)
    #             ax.plot([0, ci[0]],
    #                      [0, ci[1]],
    #                      [0, ci[2]],
    #                      '-k', lw=1)
    #             ax.plot([ci[0]],
    #                     [ci[1]],
    #                     [ci[2]],
    #                     'ok', markersize=4)
    #             # ax.text(1.1 * ci[0], 1.1 * ci[1], 1.1 * ci[2], neuron_names[i], fontsize=6)
    #
    #         ax.set_xlabel("$x_1$", labelpad=-10)
    #         ax.set_ylabel("$x_2$", labelpad=-10)
    #         ax.set_zlabel("$x_3$", labelpad=-10)
    #         ax.set_xticklabels([])
    #         ax.set_yticklabels([])
    #         ax.set_zticklabels([])
    #         ax.set_xlim(-.75, .75)
    #         ax.set_ylim(-.75, .75)
    #         ax.set_zlim(-.75, .75)
    #
    #     plt.show()

    if do_plot_data:
        all_ys = np.vstack(ys)
        all_ms = np.vstack(ms)
        all_ys[~all_ms] = np.nan
        scales = np.nanstd(all_ys, axis=0)

        for i in range(N_worms):
            ysm = best_model.smooth(ys[i], mask=ms[i],
                              inputs=np.ones((ys[i].shape[0], 1)),
                              group=i)

            y = ys[i].copy()
            # y[~ms[i]] = np.nan
            # ysm[~ms[i]
            # spc = .75 * abs(y).max()
            spc = 5

            n_frames = 3 * 120 + 1
            # n_frames = 3 * 60 + 1
            # n_plot = min(20, D_obs)
            n_plot = D_obs
            t = np.arange(n_frames) / 3.0

            plt.figure(figsize=(5.5, 7))
            offset = 0
            ticks = []
            for c in range(N_clusters):
                for n in neuron_perm:
                    if neuron_clusters[n] != c:
                        continue

                    if ms[i][0, n]:
                        plt.plot(t, y[:n_frames, n] / scales[n] + spc * offset, '-', color='k', lw=1.5)
                    else:
                        plt.plot(t, np.zeros_like(t) + spc * offset, ':', color='k', lw=1)
                    plt.plot(t, ysm[:n_frames, n] / scales[n] + spc * offset, '-', color=colors[0], lw=1)

                    ticks.append(offset * spc)
                    offset += 1

                # Add an extra space between clusters
                offset += 2

            # Remove last space
            offset -= 2

            # for d,n in enumerate(neuron_perm):
            #     if ms[i][0, n]:
            #         plt.plot(t, y[:n_frames, n] / scales[n] + spc * d, '-', color='k', lw=1)
            #     else:
            #         plt.plot(t, np.zeros_like(t) + spc * d, ':', color='k', lw=1)
            #     plt.plot(t, ysm[:n_frames, n] / scales[n] + spc * d, '-', color=colors[0], lw=1.5)

            plt.yticks(ticks, neuron_names[neuron_perm], fontsize=6)
            plt.ylim(offset * spc, -spc)
            plt.ylim(offset * spc, -spc)
            plt.xlim(0, t[-1])
            plt.xticks(fontsize=6)
            plt.xlabel("time (s)")

            plt.title("Worm {} Activity".format(i+1))
            plt.tight_layout()

            # plt.savefig(os.path.join(results_dir, "y_{}.png".format(i)), dpi=300)
            plt.savefig(os.path.join(fig_dir, "y_{}.pdf".format(i)))

        plt.close("all")
        # plt.show()

def fit_all_models(D_latents=np.arange(2, 21, 2)):
    axs = None
    best_models = []
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
        best_models.append(models[best_index])
        print("Best dimension: {}".format(D_latents[best_index]))

        axs = plot_likelihoods(D_latents, final_lls, hlls, best_index,
                               name=name, color=colors[index], axs=axs)

    plt.tight_layout()
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(fig_dir, "dimensionality.pdf".format(name)))

    return best_models


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
    from sklearn.linear_model import LinearRegression
    xs = [s.gaussian_states for s in model.states_list]
    C = model.C

    # Now the coup de grâce -- use all the unlabeled neurons
    all_ys, all_ms, _, _, all_neuron_names = load_data(include_unnamed=True)
    print(all_ys[0].shape[1])

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
            reg = LinearRegression().fit(xs[worm], all_ys[worm][:, n])
            c_reg = reg.coef_

            # Compute cosine similarity between the inferred C and the hrslds emission matrix
            S_other[:,i] = np.dot(C[possible], c_reg) / np.linalg.norm(C[possible], axis=1) / np.linalg.norm(c_reg)

        for i, n in enumerate(heldout_neurons[worm]):
            reg = LinearRegression().fit(xs[worm], ys[worm][:, n])
            c_reg = reg.coef_

            # Compute cosine similarity between the inferred C and the hrslds emission matrix
            S_heldout[:,i] = np.dot(C[possible], c_reg) / np.linalg.norm(C[possible], axis=1) / np.linalg.norm(c_reg)

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
    plt.figure(figsize=(3,3))
    lim = 1
    bins = 20
    plt.hist(np.concatenate(s_others), np.linspace(-lim, lim, bins + 1),
             color=colors[0], alpha=0.5, normed=True, label="Incorrect candidates")
    plt.hist(np.concatenate(s_heldouts), np.linspace(-lim, lim, bins + 1),
             color=colors[1], alpha=0.5, normed=True, label="Correct candidates")
    plt.legend(loc="upper left")
    plt.title("")
    plt.xlabel("cosine similarity")
    plt.ylabel("probability density")
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
    plt.show()


if __name__ == "__main__":
    ys, ms, z_trues, z_true_key, neuron_names = load_data(include_unnamed=False)
    D_obs = ys[0].shape[1]

    # Split test train
    ytrains, ytests, train_inds = list(zip(*[_split_test_train(y, train_frac=0.8) for y in ys]))
    mtrains, mtests, _ = list(zip(*[_split_test_train(m, train=train) for m, train in zip(ms, train_inds)]))
    z_true_trains, z_true_tests, _ = list(zip(*[_split_test_train(z, train=train) for z, train in zip(z_trues, train_inds)]))
    n_trains = np.array([mtr.sum() for mtr in mtrains])
    n_tests = np.array([mte.sum() for mte in mtests])

    D_latents = np.arange(2, 21, 2)
    best_models = fit_all_models(D_latents)
    best_model = best_models[0]

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

    # plot_likelihoods(final_lls, hlls, best_index)
    #
    plot_best_model_results(best_model,
                            do_plot_x_3d=True,
                            do_plot_x_2d=False,
                            do_plot_sigmasq=False,
                            do_plot_similarity=False,
                            do_plot_cluster_embedding=False,
                            do_plot_data=False)

    # heldout_neuron_identification()
    #
    # # Save out the results
    # results = dict(
    #     xtrains=xtrains,
    #     xtests=xtests,
    #     ytrains=ytrains,
    #     ytests=ytests,
    #     mtrains=mtrains,
    #     mtests=mtests,
    #     z_true_trains=z_true_trains,
    #     z_true_tests=z_true_tests,
    #     z_key=z_true_key,
    #     best_model=best_model,
    #     D_latent=best_model.D_latent,
    #     C=C,
    #     perm=dim_perm,
    #     N_clusters=N_clusters,
    #     neuron_clusters=neuron_clusters,
    #     neuron_perm=neuron_perm
    # )
    #
    # with open(os.path.join(results_dir, "lds_data.pkl"), "wb") as f:
    #     pickle.dump(results, f)
