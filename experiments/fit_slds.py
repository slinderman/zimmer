import os
import copy
import pickle
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import zimmer.plotting as zplt
from zimmer.observations import HierarchicalAutoRegressiveObservations,  \
    HierarchicalRobustAutoRegressiveObservations
from zimmer.transitions import HierarchicalRecurrentTransitions, \
    HierarchicalStationaryTransitions, HierarchicalRBFRecurrentTransitions
from zimmer.util import cached
from ssm.core import _SwitchingLDS
from ssm.init_state_distns import InitialStateDistribution
from ssm.transitions import RecurrentTransitions, RBFRecurrentTransitions, StationaryTransitions
from ssm.observations import AutoRegressiveObservations, \
    RobustAutoRegressiveObservations
from ssm.emissions import GaussianEmissions
from ssm.variational import SLDSTriDiagVariationalPosterior
from ssm.util import find_permutation

# You may want to limit the number of threads used by numpy
# To do that, set the following:
# export MKL_NUM_THREADS=1
# export NUMEXPR_NUM_THREADS=1
# export OMP_NUM_THREADS=1

np.random.seed(1234)

# Parse command line arguments
parser = argparse.ArgumentParser(description='Fit an SLDS to worm data')
parser.add_argument('dataset', 
                    help='which dataset to run on')
parser.add_argument('K', type=int, 
                    help='number of discrete states')
parser.add_argument('D', type=int,
                    help='continuous latent state dim')
parser.add_argument('--no_hierarchical', dest="hierarchical", 
                    action="store_false", default=True,
                    help='do not fit a hierarchical model')
parser.add_argument('--transitions', dest="transitions", default="recurrent",
                    help='type of transition model')
# parser.add_argument('--no_recurrent', dest="recurrent", 
#                     action="store_false", default=True,
#                     help='do not fit recurrent model')
parser.add_argument('--no_robust', dest="robust", 
                    action="store_false", default=True,
                    help='do not fit robust model')
parser.add_argument('--eta', type=float, default=1e-3,
                    help='variance of hierarchical prior')
parser.add_argument('--method', default="vem",
                    help='training method')
parser.add_argument('--N_train_iter', type=int, default=5000,
                    help='number of training VI iterations')
parser.add_argument('--N_val_iter', type=int, default=1000,
                    help='number of validation VI iterations')
parser.add_argument('--N_full_iter', type=int, default=1000,
                    help='number of VI iterations on the full data')
parser.add_argument('-d', '--data_dir', default=os.path.join("data", "processed"),
                    help='where the processed data is stored')
parser.add_argument('-o', '--results_dir', default='results',
                    help='where to store the results')
args = parser.parse_args()


def make_slds(N, M, tags):
    K, D, eta = args.K, args.D, args.eta
    # Make the SLDS
    initial_state = InitialStateDistribution(K, D, M)
    if args.hierarchical:
        # Hierarchical transition model
        if args.transitions.lower() == "recurrent":
            transitions = HierarchicalRecurrentTransitions(K, D, tags, eta=eta)
        elif args.transitions.lower() == "rbf":
            transitions = HierarchicalRBFRecurrentTransitions(K, D, tags, eta=eta)
        elif args.transitions.lower() == "standard":
            transitions = HierarchicalStationaryTransitions(K, D, tags, eta=eta)
        else:
            raise Exception("Invalid transition model: {}".format(args.transitions))

        # Hierarchicalynamcis model
        if args.robust:
            dynamics = HierarchicalRobustAutoRegressiveObservations(K, D, tags, M, eta=eta)
        else:
            dynamics = HierarchicalAutoRegressiveObservations(K, D, tags, M, eta=eta)

    else:
        # Shared transitions
        if args.transitions.lower() == "recurrent":
            transitions = RecurrentTransitions(K, D)
        elif args.transitions.lower() == "rbf":
            transitions = RBFRecurrentTransitions(K, D)
        elif args.transitions.lower() == "standard":
            transitions = StationaryTransitions(K, D)
        else:
            raise Exception("Invalid transition model: {}".format(args.transitions))

        # Shared dynamics
        if args.robust:
            dynamics = RobustAutoRegressiveObservations(K, D)
        else:
            dynamics = AutoRegressiveObservations(K, D)

    emissions = GaussianEmissions(N, K, D, M)
    return _SwitchingLDS(N, K, D, M, initial_state, transitions, dynamics, emissions)


def train_slds(rslds, train_datas):

    # Initialize with the training data
    rslds.initialize([data['y'] for data in train_datas],
                     masks=[data['m'] for data in train_datas],
                     tags=[data['tag'] for data in train_datas])

    # Fit the rSLDS on the training data
    q_train = SLDSTriDiagVariationalPosterior(
        rslds,
        datas=[data['y'] for data in train_datas],
        masks=[data['m'] for data in train_datas],
        tags=[data['tag'] for data in train_datas])

    train_elbos = rslds.fit(
        q_train,
        datas=[data['y'] for data in train_datas],
        masks=[data['m'] for data in train_datas],
        tags=[data['tag'] for data in train_datas],
        method=args.method,
        initialize=False,
        num_iters=args.N_train_iter)

    return q_train, train_elbos


def validate_slds(rslds, val_datas):

    # Evaluate the rSLDS on the validation data
    q_val = SLDSTriDiagVariationalPosterior(
        rslds,
        datas=[data['y'] for data in val_datas],
        masks=[data['m'] for data in val_datas],
        tags=[data['tag'] for data in val_datas])

    val_elbos = rslds.approximate_posterior(
        q_val,
        datas=[data['y'] for data in val_datas],
        masks=[data['m'] for data in val_datas],
        tags=[data['tag'] for data in val_datas],
        num_iters=args.N_val_iter)

    return q_val, val_elbos


def full_slds(rslds, full_datas):

    q_full = SLDSTriDiagVariationalPosterior(
        rslds,
        datas=[data['y'] for data in full_datas],
        masks=[data['m'] for data in full_datas],
        tags=[data['tag'] for data in full_datas])

    full_elbos = rslds.approximate_posterior(
        q_full,
        datas=[data['y'] for data in full_datas],
        masks=[data['m'] for data in full_datas],
        tags=[data['tag'] for data in full_datas],
        num_iters=args.N_full_iter)

    # Find the most likely discrete states that are
    # best aligned with the Kato states
    xs = q_full.mean
    zs = [rslds.most_likely_states(x, data['y'], tag=data['tag'])
          for x, data in zip(xs, full_datas)]
    rslds.permute(find_permutation(
        np.concatenate([data['z_true'] for data in full_datas]),
        np.concatenate(zs)))
    zs = [rslds.most_likely_states(x, data['y'], tag=data['tag'])
          for x, data in zip(xs, full_datas)]

    return q_full, full_elbos, xs, zs


def plot_elbos(figures_dir, train_elbos, val_elbos, full_elbos):
    plt.figure(figsize=(12, 4))

    plt.subplot(131)
    plt.plot(train_elbos)
    plt.xlabel("Training Iteration")
    plt.ylabel("ELBO")

    plt.subplot(132)
    plt.plot(val_elbos)
    plt.xlabel("Validation Iteration")
    plt.ylabel("ELBO")

    plt.subplot(133)
    plt.plot(full_elbos)
    plt.xlabel("Test Iteration")
    plt.ylabel("ELBO")

    plt.savefig(os.path.join(figures_dir, "elbos.png"), dpi=300)


def plot_latent_trajectories(figures_dir, xs, zs, lim=(-3, 3)):
    zplt.plot_latent_trajectories_vs_time(xs, zs, plot_slice=(0, 1000))
    plt.savefig(
        os.path.join(figures_dir, "continuous_states_v_time.png"),
        dpi=300)

    plt.figure(figsize=(12, 3))
    for w, (x, z) in enumerate(zip(xs, zs)):
        ax = plt.subplot(1, len(xs), w + 1)
        zplt.plot_2d_continuous_states(
            x, z, xlims=lim, ylims=lim, inds=(0, 1), ax=ax)
        plt.ylabel("PC 2" if w == 0 else "")
        plt.xlabel("PC 1")
        plt.title("worm {}".format(w + 1))
        plt.savefig(
            os.path.join(figures_dir, "continuous_traj_{}.png".format(w + 1)), dpi=300)


def plot_discrete_states(K, zs, ztrues):
    zplt.plot_state_overlap(z_infs, z_trues)
    plt.savefig(os.path.join(results_dir, "figures", "discrete_state_overlap.png"), dpi=300)

    # # Helper function to find overlap percentages
    # def compute_pct_overlap(zi, ztr):
    #     overlap = np.zeros((K, K_true))
    #     for k in range(K):
    #         overlap[k] = np.bincount(ztr[zi == k], minlength=K_true).astype(float)
    #         overlap[k] /= (overlap[k].sum() + 1e-3)
    #     return overlap

    # # Find a permutation so that the bar codes look progressive
    # total_overlap = compute_pct_overlap(np.concatenate(z_infs), np.concatenate(z_trues))
    # overlap_perm = np.argsort(np.argmax(total_overlap, axis=1))

    # # Helper function to plot "barcodes"
    # from matplotlib.cm import get_cmap
    # zimmer_colors = get_cmap("cubehelix")(np.linspace(0, 1, K_true))
    # def plot_overlap_barcode(ax, overlap):
    #     for i,k in enumerate(overlap_perm):        
    #         for ktr in range(K_true):
    #             plt.bar(i, overlap[k, ktr], bottom=np.sum(overlap[k, :ktr]), color=zimmer_colors[ktr], width=0.8)
    #     ax.set_xlim(-.5, K-.5)
        
    # # Plot all overlaps as bar codes
    # plt.figure(figsize=(12, 4))

    # # Plot the total overlap first
    # ax = plt.subplot(1, W+1, 1)
    # plot_overlap_barcode(ax, total_overlap)
    # plt.ylabel("Pct of manual state")
    # plt.yticks([0, .25, .5, .75, 1], [0, 25, 50, 75, 100])
    # plt.ylim(0, 1)
    # plt.xlabel("Inferred state")
    # plt.xticks(np.arange(K), np.arange(K)+1)
    # plt.title("All worms")

    # for w in range(W):
    #     ax = plt.subplot(1, W+1, w+2)
    #     overlap_w = compute_pct_overlap(z_infs[w], z_trues[w])
    #     plot_overlap_barcode(ax, overlap_w)
    #     plt.yticks([])        
    #     plt.ylim(0, 1)
    #     plt.xlabel("Inferred state")
    #     plt.xticks(np.arange(K), np.arange(K)+1)
    #     plt.title("Worm {}".format(w+1))
    # plt.tight_layout()

    # # Print key
    # for color_name, state_name in zip(zplt.color_names, z_true_key):
    #     print("{} : {}".format(color_name, state_name))


def simulate_rslds(rslds, pad=3, noise_reduction=-4):
    rslds_low_noise = copy.deepcopy(rslds)
    rslds_low_noise.dynamics.inv_sigmas += noise_reduction

    zsmpls = []
    xsmpls = []
    ysmpls = []

    for w in range(W):
        # Sample data
        Tsmpl = Ts[w]
        zpre, xpre, ypre = z_infs[w][-pad:], xs[w][-pad:], ys[w][-pad:]
        zsmpl, xsmpl, ysmpl = rslds_low_noise.sample(Tsmpl, prefix=(zpre, xpre, ypre), tag=w, with_noise=False)

        zsmpl = np.concatenate((zpre, zsmpl))
        xsmpl = np.concatenate((xpre, xsmpl))
        ysmpl = np.concatenate((ypre, ysmpl))
        
        # Truncate to stable region
        unstable = np.arange(Tsmpl+pad)[np.any(abs(xsmpl) > 5, axis=1)]
        T_stable = np.min(np.concatenate(([Tsmpl+pad], unstable)))
        zsmpl = zsmpl[:T_stable]
        xsmpl = xsmpl[:T_stable]
        ysmpl = ysmpl[:T_stable]
        
        # Append
        zsmpls.append(zsmpl)
        xsmpls.append(xsmpl)
        ysmpls.append(ysmpl)

    # Plot continuous latent states
    for w, (zsmpl, xsmpl) in enumerate(zip(zsmpls, xsmpls)):
        plt.figure(figsize=(12, 6))
        ax = plt.subplot(1, 2, 1, aspect="equal")
        zplt.plot_2d_continuous_states(xsmpl, zsmpl, xlims=(-lim, lim), ylims=(-lim, lim), inds=(0, 1), ax=ax)
        plt.plot(xsmpl[pad-1,0], xsmpl[pad-1,1], 'k*')
        plt.xlabel("PC 1")
        plt.ylabel("PC 2")
        plt.title("Worm {}".format(w + 1))

        ax = plt.subplot(1, 2, 2, aspect="equal")
        zplt.plot_2d_continuous_states(xsmpl, zsmpl, xlims=(-lim, lim), ylims=(-lim, lim), inds=(0, 2), ax=ax)
        plt.plot(xsmpl[pad-1,0], xsmpl[pad-1,2], 'k*')
        plt.xlabel("PC 1")
        plt.ylabel("PC 3")
        plt.title("Worm {}".format(w + 1))

    for w, ysmpl in enumerate(ysmpls):
        plt.figure(figsize=(12, 8))
        plt.plot(np.arange(ysmpl.shape[0]) / 3.0, ysmpl - np.arange(N), '-k')
        plt.yticks(-np.arange(N), neuron_names)
        plt.ylim(-N,1)
        plt.xlim(0, Ts[w] / 3.0)
        plt.xlabel("time (s)")
        plt.title("Simulated Worm {}".format(w + 1))
        
        # Plot real data for comparison
        plt.figure(figsize=(12, 8))
        plt.plot(np.arange(Ts[w]) / 3.0, (ys[w] - np.arange(N)) * ms[w], '-', color=zplt.default_colors[3])
        plt.yticks(-np.arange(N), neuron_names)
        plt.ylim(-N,1)
        plt.xlim(0, Ts[w] / 3.0)
        plt.xlabel("time (s)")
        plt.title("Real Worm {}".format(w + 1))
        
    return zsmpls, xsmpls, ysmpls


if __name__ == "__main__":
    experiment_name = "{}{}{}SLDS_K{}D{}eta{:.0e}".format(
        'h' if args.hierarchical else '',
        'r' if args.transitions == "recurrent" else 'rbf' if args.transitions == "rbf" else '',
        'b' if args.robust else '',
        args.K, args.D, args.eta
        )
    print("Fitting model {} on {} data".format(experiment_name, args.dataset))

    experiment_dir = os.path.join(args.results_dir, experiment_name)
    if not os.path.exists(experiment_dir):
        os.mkdir(experiment_dir)
    
    # Load the preprocessed data
    with open(os.path.join(args.data_dir, args.dataset + ".pkl"), "rb") as f:
        train_datas, val_datas, test_datas, full_datas, tags = pickle.load(f)

    # Identify some constants
    K_true = full_datas[0]['z_true'].max() + 1  # Number of states
    N = full_datas[0]['y'].shape[1]              # Number of neurons
    W = len(tags)                                # Number of worms
    M = full_datas[0]['u'].shape[1]              # Number of input dims

    # Make the SLDS
    slds = make_slds(N, M, tags)

    # Fit the model and evaluate it
    q_train, train_elbos = cached(experiment_dir, "train")(train_slds)(slds, train_datas)
    q_val, val_elbos = cached(experiment_dir, "validate")(validate_slds)(slds, val_datas)
    q_full, full_elbos, xs, zs = cached(experiment_dir, "full")(full_slds)(slds, full_datas)

    # Plot some basic results
    plot_elbos(experiment_dir, train_elbos, val_elbos, full_elbos)
    plot_latent_trajectories(experiment_dir, xs, zs)
