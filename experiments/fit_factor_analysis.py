import os
import pickle
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from ssm.preprocessing import factor_analysis_with_imputation
from zimmer.util import cached

np.random.seed(1234)

# Parse command line arguments
parser = argparse.ArgumentParser(description='Fit an SLDS to worm data')
parser.add_argument('dataset', 
                    help='which dataset to run on')
parser.add_argument('--D_min', type=int, default=1,
                    help='minimum dimension')
parser.add_argument('--D_max', type=int, default=10,
                    help='maximum dimension')
parser.add_argument('--N_iter', type=int, default=50,
                    help='number of EM iterations')
parser.add_argument('--N_repeats', type=int, default=3,
                    help='number of EM iterations')
parser.add_argument('-d', '--data_dir', default=os.path.join("data", "processed"),
                    help='where the processed data is stored')
parser.add_argument('-o', '--results_dir', default='results',
                    help='where to store the results')
args = parser.parse_args()


def fit_factor_analysis(D, y_trains, m_trains, y_tests, m_tests):
    fa, xs, lls = factor_analysis_with_imputation(D, y_trains, m_trains, num_iters=args.N_iter)
    
    # Evaluate heldout likelihood
    hll = 0
    for val_y, val_m in zip(y_tests, m_tests):
        fa.add_data(val_y, mask=val_m)
        states = fa.data_list.pop()
        hll += states.log_likelihood().sum()
    
    return fa, xs, lls, hll


if __name__ == "__main__":
    # Load the preprocessed data
    with open(os.path.join(args.data_dir, args.dataset + ".pkl"), "rb") as f:
        train_datas, val_datas, test_datas, full_datas, tags = pickle.load(f)

    train_ys = [d['y'] for d in train_datas]
    train_ms = [d['m'] for d in train_datas]
    val_ys = [d['y'] for d in val_datas]
    val_ms = [d['m'] for d in val_datas]
    test_ys = [d['y'] for d in test_datas]
    test_ms = [d['m'] for d in test_datas]

    fas = []
    llss = []
    hlls = []
    all_hlls = []

    Ds = np.arange(args.D_min, args.D_max + 1)
    for D in Ds:
        results = []
        for itr in range(args.N_repeats):
            print("D = ", D, " repeat = ", itr)
            fit = partial(fit_factor_analysis, D, train_ys + val_ys, train_ms + val_ms, test_ys, test_ms)
            cached_fit = cached(args.results_dir, "fa_D{}_i{}".format(D, itr))(fit)
            results.append(cached_fit())
        
        # Pick the iteration with the best heldout likelihood
        best_results = results[np.argmax([r[3] for r in results])]

        # Save results
        fas.append(best_results[0])
        llss.append(best_results[2])
        hlls.append(best_results[3])
        all_hlls.append([r[3] for r in results])

    # Plot the likelihoods
    plt.figure(figsize=(3, 2))
    plt.plot(Ds, all_hlls, '-ko')
    plt.xlabel("latent dimension")
    plt.ylabel("heldout likelihood")
    plt.savefig(os.path.join(args.results_dir, "test_ll.pdf"))

    plt.figure(figsize=(3, 2))
    for D, lls in zip(Ds, llss): 
        plt.plot(np.arange(2, len(lls)), lls[2:], '-', label="D={}".format(D))
    plt.xlabel("Iteration")
    plt.legend()
    plt.savefig(os.path.join(args.results_dir, "train_ll.pdf"))

    # Get the continuous latent states with the best dimension
    D = Ds[np.argmax(hlls)]
    print("Best D = ", D)
    fa = fas[np.where(Ds == D)[0][0]]

    def get_xs(ys, ms):
        xs = []
        for y, m in zip(ys, ms):
            fa.add_data(y, mask=m)
            states = fa.data_list.pop()
            states.E_step()
            xs.append(states.Z.copy('C'))
        return xs

    xs = get_xs(ys, ms)
    train_xs = get_xs(train_ys, train_ms)
    val_xs = get_xs(val_ys, val_ms)
    test_xs = get_xs(test_ys, test_ms)

    with open(os.path.join(args.results_dir, args.dataset + "_xs.pkl"), "wb") as f:
        pickle.dump((train_xs, val_xs, test_xs), f)


