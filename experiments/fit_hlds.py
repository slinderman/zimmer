import os
import pickle

import numpy as np
np.random.seed(1234)

from tqdm import tqdm
from functools import partial

# Load worm modeling specific stuff
from zimmer.io import load_kato_data
from zimmer.models import HierarchicalLDS
from zimmer.emissions import HierarchicalDiagonalRegression


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

def _split_test_train(y, train=None, train_frac=0.8):
    T = y.shape[0]
    train = train if train is not None else np.arange(T) < train_frac * T
    assert train.shape == (T,) and train.dtype == bool
    return y[train], y[~train], train


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
    ys, ms, z_trues, z_true_key, neuron_names = load_kato_data(include_unnamed=False)
    D_obs = ys[0].shape[1]

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

    # heldout_neuron_identification_corr()
