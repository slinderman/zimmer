import numpy as np
np.random.seed(1)

from pybasicbayes.util.text import progprint_xrange
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from pybasicbayes.util.stats import sample_mniw

from zimmer.models import HierarchicalFactorAnalysis

N = 2000
D_obs = 20
D_latent = 2

def principal_angle(A,B):
    """
    Find the principal angle between two subspaces
    spanned by columns of A and B
    """
    from numpy.linalg import qr, svd
    qA, _ = qr(A)
    qB, _ = qr(B)
    U,S,V = svd(qA.T.dot(qB))
    return np.arccos(min(S.min(), 1.0))


def generate_synth_data(N_groups=5):

    # Create a true model and sample from it
    mask = np.random.rand(N,D_obs) < 0.9
    true_model = HierarchicalFactorAnalysis(D_obs, D_latent, N_groups)
    true_datas = []
    for group in range(N_groups):
        true_datas.append(true_model.generate(N=N, mask=mask, group=group, keep=True))
    return true_model, true_datas


def plot_results(lls, angles, Ztrue, Zinf, sigmasq_true, sigmasq_inf):
    # Plot log probabilities
    plt.figure()
    plt.plot(lls)
    plt.ylabel("Log Likelihood")
    plt.xlabel("Iteration")

    plt.figure()
    plt.plot(np.array(angles) / np.pi * 180.)
    plt.ylabel("Principal Angle")
    plt.xlabel("Iteration")

    # Plot locations, color by angle
    N = Ztrue.shape[0]
    inds_to_plot = np.random.randint(0, N, min(N, 500))
    th = np.arctan2(Ztrue[:,1], Ztrue[:,0])
    nperm = np.argsort(np.argsort(th))
    cmap = get_cmap("jet")

    plt.figure()
    plt.subplot(121)
    for n in inds_to_plot:
        plt.plot(Ztrue[n,0], Ztrue[n,1], 'o', markerfacecolor=cmap(nperm[n] / float(N)), markeredgecolor="none")
    plt.title("True Embedding")
    plt.xlim(-4,4)
    plt.ylim(-4,4)

    plt.subplot(122)
    for n in inds_to_plot:
        plt.plot(Zinf[n,0], Zinf[n,1], 'o', markerfacecolor=cmap(nperm[n] / float(N)), markeredgecolor="none")
    plt.title("Inferred Embedding")
    plt.xlim(-4,4)
    plt.ylim(-4,4)

    plt.figure()
    vmax = max(sigmasq_true.max(), sigmasq_inf.max())
    plt.subplot(311)
    plt.imshow(sigmasq_true, vmin=0, vmax=vmax, interpolation="none")
    plt.title("True $\\sigma^2$")
    plt.colorbar()
    plt.subplot(312)
    plt.imshow(sigmasq_inf, vmin=0, vmax=vmax, interpolation="none")
    plt.title("Inferred $\\sigma^2$")
    plt.colorbar()
    plt.subplot(313)
    vmax = abs(sigmasq_true - sigmasq_inf).max()
    plt.imshow(sigmasq_true-sigmasq_inf, vmin=-vmax, vmax=vmax, cmap="RdBu", interpolation="none")
    plt.title("True minus Inferred $\\sigma^2$")
    plt.colorbar()
    plt.tight_layout()
    plt.show()


def gibbs_example(true_model, true_datas):
    # Fit a test model
    model = HierarchicalFactorAnalysis(D_obs, D_latent, len(true_datas))
    inf_datas = []
    for group, data in enumerate(true_datas):
        inf_datas.append(model.add_data(data.X, mask=data.mask, group=group))

    lps = []
    angles = []
    N_iters = 100
    for _ in progprint_xrange(N_iters):
        model.resample_model()
        lps.append(model.log_likelihood())
        angles.append(principal_angle(true_model.W, model.W))

    plot_results(lps, angles, true_datas[0].Z, inf_datas[0].Z,
                 true_model.sigmasq, model.sigmasq)


def em_example(true_model, true_datas):
    # Fit a test model
    model = HierarchicalFactorAnalysis(D_obs, D_latent, len(true_datas))
    inf_datas = []
    for group, data in enumerate(true_datas):
        inf_datas.append(model.add_data(data.X, mask=data.mask, group=group))

    lps = []
    angles = []
    N_iters = 100
    for _ in progprint_xrange(N_iters):
        model.EM_step()
        lps.append(model.log_likelihood())
        angles.append(principal_angle(true_model.W, model.W))

    plot_results(lps, angles, true_datas[0].Z, inf_datas[0].E_Z,
                 true_model.sigmasq, model.sigmasq)

if __name__ == "__main__":
    true_model, true_datas = generate_synth_data()
    # gibbs_example(true_model, true_datas)
    em_example(true_model, true_datas)
