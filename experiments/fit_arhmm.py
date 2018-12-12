import os
import sys
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
from ssm.core import _HMM
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
parser.add_argument('--method', default="em",
                    help='training method')
parser.add_argument('--N_train_iter', type=int, default=500,
                    help='number of training VI iterations')
parser.add_argument('-d', '--data_dir', default=os.path.join("data", "processed"),
                    help='where the processed data is stored')
parser.add_argument('-o', '--results_dir', default='results',
                    help='where to store the results')
args = parser.parse_args()


def make_arhmm(D, M, tags):
    K, eta = args.K, args.eta
    
    # Make the ARHMM
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

    return _HMM(K, D, M, initial_state, transitions, dynamics)


def initialize_arhmm(arhmm, train_datas):
    """
    Initialize with a non-hierarchical model if possible
    """
    if args.hierarchical:
        base_exp_name = "{}{}ARHMM_K{}eta{:.0e}".format(
            'r' if args.transitions == "recurrent" else 'rbf' if args.transitions == "rbf" else '',
            'b' if args.robust else '',
            args.K, args.eta
            )
        print("Loading non-hierarchical model {} on {} data".format(base_exp_name, args.dataset))

        base_exp_dir = os.path.join(args.results_dir, base_exp_name)
        if not os.path.exists(base_exp_dir):
            raise Exception("Could not find non-hierarchical model for initialization.")

        with open(os.path.join(base_exp_dir, "train.pkl"), "rb") as f:
            base, _ = pickle.load(f)

        # Initialize the hierarchical model with the regular model
        arhmm.init_state_distn.params = copy.deepcopy(base.init_state_distn.params)
        arhmm.transitions.initialize_from_standard(base.transitions)
        arhmm.observations.initialize_from_standard(base.observations)

    else:
        # Initialize with the training data
        arhmm.initialize([data['x'] for data in train_datas],
                         tags=[data['tag'] for data in train_datas])

    return arhmm

def _train_arhmm_chunk(arhmm, train_datas, N_iter):

    train_lls = arhmm.fit(
        datas=[data['x'] for data in train_datas],
        tags=[data['tag'] for data in train_datas],
        method=args.method,
        initialize=False,
        num_em_iters=N_iter)

    return arhmm, train_lls


def train_arhmm(arhmm, train_datas, chunk_size=500):
    # Initialize the model and variational posterior with training data
    _init = cached(experiment_dir, "_init")(initialize_arhmm)
    arhmm = _init(arhmm, train_datas)

    # Train in chunks so that we don't lose everything if job halts
    train_lls = []
    for chunk, start in enumerate(np.arange(0, args.N_train_iter, chunk_size)):
        this_chunk_size = min(chunk_size, args.N_train_iter - start)
        print("Train chunk: {} -- {}".format(start, start + this_chunk_size))

        _train = cached(experiment_dir, "_train_{}".format(chunk))(_train_arhmm_chunk)
        arhmm, chunk_lls = _train(arhmm, train_datas, this_chunk_size)
        train_lls.append(chunk_lls)
    train_lls = np.concatenate(train_lls)

    return arhmm, train_lls


def validate_arhmm(arhmm, val_datas):
    val_elbos = arhmm.log_likelihood(
        datas=[data['x'] for data in val_datas],
        tags=[data['tag'] for data in val_datas])

    return val_elbos


def full_arhmm(arhmm, full_datas):
    full_lls = arhmm.log_likelihood(
        datas=[data['x'] for data in full_datas],
        tags=[data['tag'] for data in full_datas])

    # Find the most likely discrete states that are
    # best aligned with the Kato states
    zs = [arhmm.most_likely_states(data['x'], tag=data['tag']) 
          for data in full_datas]

    try:
        arhmm.permute(find_permutation(
            np.concatenate([data['z_true'] for data in full_datas]),
            np.concatenate(zs)))

        zs = [arhmm.most_likely_states(x, data['x'], tag=data['tag'])
              for data in full_datas]
    except:
        pass

    return full_lls, xs, zs


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

    plt.savefig(os.path.join(figures_dir, "continuous_states.png"), dpi=300)


def plot_discrete_states(figures_dir, zs, ztrues):
    zplt.plot_state_overlap(z_infs, z_trues)
    plt.savefig(os.path.join(figures_dir, "discrete_state_overlap.png"), dpi=300)

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
    experiment_name = "{}{}{}ARHMM_K{}eta{:.0e}".format(
        'h' if args.hierarchical else '',
        'r' if args.transitions == "recurrent" else 'rbf' if args.transitions == "rbf" else '',
        'b' if args.robust else '',
        args.K, args.eta
        )
    print("Fitting model {} on {} data".format(experiment_name, args.dataset))

    experiment_dir = os.path.join(args.results_dir, experiment_name)
    if not os.path.exists(experiment_dir):
        os.mkdir(experiment_dir)
    
    # Load the preprocessed data
    with open(os.path.join(args.data_dir, args.dataset + ".pkl"), "rb") as f:
        train_datas, val_datas, test_datas, full_datas, tags = pickle.load(f)

    # Load the preprocessed continuous states
    with open(os.path.join("data", "processed", "kato_xs.pkl"), "rb") as f:
        train_xs, val_xs, test_xs, full_xs = pickle.load(f)

    # Insert x into datas
    for ds, xs in [(train_datas, train_xs), 
                 (val_datas,   val_xs), 
                 (test_datas,  test_xs), 
                 (full_datas,  full_xs)]:

        assert len(ds) == len(xs)
        for d, x in zip(ds, xs):
            d['x'] = x

    # Identify some constants
    K_true = full_datas[0]['z_true'].max() + 1   # Number of manually labeled states
    N = full_datas[0]['y'].shape[1]              # Number of neurons
    D = full_datas[0]['x'].shape[1]              # Number of neurons
    M = full_datas[0]['u'].shape[1]              # Number of input dims
    W = len(tags)                                # Number of worms
    
    # Make the SLDS
    arhmm = make_arhmm(D, M, tags)

    # Fit the model and evaluate it
    arhmm, train_lls = cached(experiment_dir, "train")(train_arhmm)(arhmm, train_datas)
    val_lls = cached(experiment_dir, "validate")(validate_arhmm)(arhmm, val_datas)
    full_lls, xs, zs = cached(experiment_dir, "full")(full_arhmm)(arhmm, full_datas)

    # Plot some basic results
    plot_elbos(experiment_dir, train_lls, val_lls, full_lls)
    plot_latent_trajectories(experiment_dir, xs, zs)
    plot_discrete_states(experiment_dir, zs, [d['z_true'] for d in full_datas])
