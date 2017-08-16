import os
import pickle

import numpy as np
import numpy.random as npr
npr.seed(0)

from pybasicbayes.util.text import progprint_xrange
from pybasicbayes.models import FactorAnalysis
from pybasicbayes.distributions import \
    Regression, Gaussian, AutoRegression

from autoregressive.models import ARWeakLimitStickyHDPHMM
from pyslds.util import get_empirical_ar_params
from pylds.util import random_rotation

from pyslds.models import HMMSLDS
from zimmer.emissions import HierarchicalDiagonalRegression
from zimmer.models import HierarchicalRecurrentOnlySLDS, HierarchicalHMMSLDS

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.font_manager import FontProperties

from rslds.plotting import plot_most_likely_dynamics, plot_trajectory, plot_z_samples, \
    plot_data, plot_trajectory_and_probs, plot_all_dynamics

from rslds.util import cached

### Global parameters
T, K, K_true, D_obs, D_latent = 1000, 4, 4, 10, 2
N_groups = 3
N_trials_per_group = 1
mask_start, mask_stop = 0, 0
N_iters = 1000

# Save / cache the outputs
RUN_NUMBER = 1
RESULTS_DIR = os.path.join("results", "nascar", "run{:03d}".format(RUN_NUMBER))


### Plotting code
def make_figure(true_model, z_true, x_true, y,
                rslds, zs_rslds, x_rslds,
                z_rslds_gen, x_rslds_gen, y_rslds_gen,
                slds, zs_slds, x_slds,
                z_slds_gen, x_slds_gen, y_slds_gen):
    """
    Show the following:
     - True latent dynamics (for most likely state)
     - Segment of trajectory in latent space
     - A few examples of observations in 10D space
     - ARHMM segmentation of factors
     - rSLDS segmentation of factors
     - ARHMM synthesis
     - rSLDS synthesis
    """
    # fig = plt.figure(figsize=(6.5,3.5))
    fig = plt.figure(figsize=(13,7))
    gs = gridspec.GridSpec(2,3)

    fp = FontProperties()
    fp.set_weight("bold")

    # True dynamics
    ax1 = fig.add_subplot(gs[0,0])
    plot_most_likely_dynamics(true_model.trans_distn,
                              true_model.dynamics_distns,
                              xlim=(-3,3), ylim=(-2,2),
                              ax=ax1)

    # Overlay a partial trajectory
    plot_trajectory(z_true[1:1000], x_true[1:1000], ax=ax1, ls="-")
    ax1.set_title("True Latent Dynamics")
    plt.figtext(.025, 1-.075, '(a)', fontproperties=fp)

    # Plot a few output dimensions
    ax2 = fig.add_subplot(gs[1, 0])
    for n in range(D_obs):
        plot_data(z_true[1:1000], y[1:1000, n], ax=ax2, ls="-")
    ax2.set_xlabel("Time")
    ax2.set_ylabel("$y$")
    ax2.set_title("Observed Data")
    plt.figtext(.025, .5 - .075, '(b)', fontproperties=fp)

    # Plot the inferred dynamics under the rSLDS
    ax3 = fig.add_subplot(gs[0, 1])
    ax3_lim = 1.05 * abs(x_rslds[1:1000]).max(axis=0)
    plot_most_likely_dynamics(rslds.trans_distn,
                              rslds.dynamics_distns,
                              xlim=(-ax3_lim[0], ax3_lim[0]),
                              ylim=(-ax3_lim[1], ax3_lim[1]),
                              ax=ax3)

    # Overlay a partial trajectory
    plot_trajectory(zs_rslds[-1][1:1000], x_rslds[1:1000], ax=ax3, ls="-")
    ax3.set_title("Inferred Dynamics (rSLDS)")
    plt.figtext(.33 + .025, 1. - .075, '(c)', fontproperties=fp)

    # Plot something... z samples?
    ax4 = fig.add_subplot(gs[1,1])
    plot_z_samples(K, zs_rslds, zref=z_true, plt_slice=(0,1000), ax=ax4)
    ax4.set_title("Discrete State Samples")
    plt.figtext(.33 + .025, .5 - .075, '(d)', fontproperties=fp)

    # Plot simulated SLDS data
    ax5 = fig.add_subplot(gs[0, 2])
    # for n, ls in enumerate(["-", ":", "-."]):
    #     plot_data(z_slds_gen[-1000:], y_slds_gen[-1000:, n], ax=ax5, ls=ls)
    plot_trajectory(z_slds_gen[-1000:], x_slds_gen[-1000:], ax=ax5, ls="-")
    # ax5.set_xlabel("Time")
    # ax5.set_ylabel("$y$")
    plt.grid(True)
    ax5.set_title("Generated States (SLDS)")
    plt.figtext(.66 + .025, 1. - .075, '(e)', fontproperties=fp)

    # Plot simulated rSLDS data
    ax6 = fig.add_subplot(gs[1, 2])
    # for n, ls in enumerate(["-", ":", "-."]):
    #     plot_data(z_rslds_gen[-1000:], y_rslds_gen[-1000:, n], ax=ax6, ls=ls)
    # ax6.set_xlabel("Time")
    # ax6.set_ylabel("$y$")
    plot_trajectory(z_rslds_gen[-1000:], x_rslds_gen[-1000:], ax=ax6, ls="-")
    ax6.set_title("Generated States (rSLDS)")
    plt.grid(True)
    plt.figtext(.66 + .025, .5 - .075, '(f)', fontproperties=fp)



    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "nascar.png"), dpi=200)
    plt.savefig(os.path.join(RESULTS_DIR, "nascar.pdf"))
    plt.show()


### Make an example with 2D latent states and 4 discrete states
# @cached(RESULTS_DIR, "simulated_data")
def simulate_nascar():
    assert K_true == 4
    As = [random_rotation(D_latent, np.pi/24.),
          random_rotation(D_latent, np.pi/48.)]

    # Set the center points for each system
    centers = [np.array([+2.0, 0.]),
               np.array([-2.0, 0.])]
    bs = [-(A - np.eye(D_latent)).dot(center) for A, center in zip(As, centers)]

    # Add a "right" state
    As.append(np.eye(D_latent))
    bs.append(np.array([+0.1, 0.]))

    # Add a "right" state
    As.append(np.eye(D_latent))
    bs.append(np.array([-0.25, 0.]))

    # Construct multinomial regression to divvy up the space
    w1, b1 = np.array([+1.0, 0.0]), np.array([-2.0])   # x + b > 0 -> x > -b
    w2, b2 = np.array([-1.0, 0.0]), np.array([-2.0])   # -x + b > 0 -> x < b
    w3, b3 = np.array([0.0, +1.0]), np.array([0.0])    # y > 0
    w4, b4 = np.array([0.0, -1.0]), np.array([0.0])    # y < 0

    reg_W = np.column_stack((100*w1, 100*w2, 10*w3,10*w4))
    reg_b = np.concatenate((100*b1, 100*b2, 10*b3, 10*b4))

    # Make a recurrent SLDS with these params #
    dynamics_distns = [
        Regression(
            A=np.column_stack((A,b)),
            sigma=1e-4 * np.eye(D_latent),
            nu_0=D_latent + 2,
            S_0=1e-4 * np.eye(D_latent),
            M_0=np.zeros((D_latent, D_latent + 1)),
            K_0=np.eye(D_latent + 1),
        )
        for A,b in zip(As, bs)]

    init_dynamics_distns = [
        Gaussian(
            mu=np.array([0.0, 1.0]),
            sigma=1e-3 * np.eye(D_latent))
        for _ in range(K)]

    C = np.hstack((npr.randn(D_obs, D_latent), np.zeros((D_obs, 1))))
    R = np.tile(np.logspace(-2, -2 + N_groups, N_groups, endpoint=False)[:,None], (1, D_obs))
    emission_distns = \
        HierarchicalDiagonalRegression(D_obs, D_latent+1, N_groups,
                                       A=C, sigmasq=R,
                                       alpha_0=2.0, beta_0=2.0)

    model = HierarchicalRecurrentOnlySLDS(
        trans_params=dict(W=reg_W, b=reg_b),
        init_state_distn='uniform',
        init_dynamics_distns=init_dynamics_distns,
        dynamics_distns=dynamics_distns,
        emission_distns=emission_distns,
        alpha=3.)

    #########################
    # Sample from the model #
    #########################
    inputs = np.ones((T, 1))
    groups, ys, xs, zs, masks = [], [], [], [], []
    for g in range(N_groups):
        for t in range(N_trials_per_group):
            y, x, z = model.generate(T=T, inputs=inputs, group=g)
            groups.append(g)
            ys.append(y)
            xs.append(x)
            zs.append(z)

            # Maks off some data
            if mask_start == mask_stop:
                mask = None
            else:
                mask = np.ones((T,D_obs), dtype=bool)
                mask[mask_start:mask_stop] = False
            masks.append(mask)

    # Print the true parameters
    np.set_printoptions(precision=2)
    print("True W:\n{}".format(model.trans_distn.W))
    print("True logpi:\n{}".format(model.trans_distn.logpi))

    return model, inputs, zs, xs, ys, masks, groups

### Factor Analysis and PCA for dimensionality reduction
# @cached(RESULTS_DIR, "factor_analysis")
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

# @cached(RESULTS_DIR, "pca")
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
# @cached(RESULTS_DIR, "arhmm")
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

    ths = np.random.uniform(np.pi/30., 1.0, size=K)
    As = [random_rotation(D_latent, th) for th in ths]
    As = [np.hstack((A, np.ones((D_latent,1)))) for A in As]
    dynamics_distns = [
        Regression(
            A=As[k],
            sigma=np.eye(D_latent),
            nu_0=D_latent + 1000,
            S_0=np.eye(D_latent),
            M_0=np.hstack((np.eye(D_latent), np.zeros((D_latent, 1)))),
            K_0=np.eye(D_latent + 1),
        )
        for k in range(K)]

    if C_init is not None:
        emission_distns = \
            HierarchicalDiagonalRegression(
                D_obs, D_latent + 1, N_groups,
                A=C_init.copy(), sigmasq=np.ones((N_groups, D_obs)),
                alpha_0=2.0, beta_0=2.0)
    else:
        emission_distns = \
            HierarchicalDiagonalRegression(
                D_obs, D_latent + 1, N_groups,
                alpha_0=2.0, beta_0=2.0)

    return init_dynamics_distns, dynamics_distns, emission_distns


# @cached(RESULTS_DIR, "slds")
def fit_slds(inputs, z_inits, x_inits, ys, masks, groups, C_init,
              N_iters=1000):
    print("Fitting standard SLDS")
    init_dynamics_distns, dynamics_distns, emission_distns = \
        make_rslds_parameters(C_init)

    slds = HierarchicalHMMSLDS(
        init_state_distn='uniform',
        init_dynamics_distns=init_dynamics_distns,
        dynamics_distns=dynamics_distns,
        emission_distns=emission_distns,
        alpha=3.)

    for z, x, y, mask, group in zip(z_inits, x_inits, ys, masks, groups):
        slds.add_data(y, inputs=inputs, mask=mask, group=group)

        # Initialize states
        slds.states_list[-1].stateseq = z.copy().astype(np.int32)
        slds.states_list[-1].gaussian_states = x.copy()

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
        z_smpls.append([z.copy() for z in slds.stateseqs])

    x_test = np.array([s.gaussian_states for s in slds.states_list])
    z_smpls = np.array(z_smpls)
    lps = np.array(lps)

    return slds, lps, z_smpls, x_test


# @cached(RESULTS_DIR, "rslds_vbem")
def fit_rslds_vbem(
        inputs, ys, masks, groups,
        x_inits=None, z_inits=None, C_init=None, initialization="none",
        true_model=None,
        N_iters=10000):

    print("Fitting rSLDS")
    init_dynamics_distns, dynamics_distns, emission_distns = \
        make_rslds_parameters(C_init)

    rslds = HierarchicalRecurrentOnlySLDS(
        init_state_distn='uniform',
        init_dynamics_distns=init_dynamics_distns,
        dynamics_distns=dynamics_distns,
        emission_distns=emission_distns,
        fixed_emission=False,
        alpha=3.)

    # Set the prior precision for the dynamics params
    for y, mask, group in zip(ys, masks, groups):
        rslds.add_data(y, inputs=inputs, mask=mask, group=group)

    if initialization == "true":
        print("Initializing with true model")
        rslds.emission_distns[0].J_0 = 1e2 * np.eye(D_latent + 1)
        rslds.trans_distn.W = true_model.trans_distn.W.copy()
        rslds.trans_distn.b = true_model.trans_distn.b.copy()

        for rdd, tdd in zip(rslds.dynamics_distns, true_model.dynamics_distns):
            rdd.A = tdd.A.copy()
            rdd.sigma = tdd.sigma.copy()

        rslds.emission_distns[0].A = true_model.emission_distns[0].A.copy()
        rslds.emission_distns[0].sigmasq_flat = true_model.emission_distns[0].sigmasq_flat.copy()
        rslds.emission_distns[0].J_0 = 1e2 * np.eye(D_latent+1)

        rslds.states_list[0].stateseq = true_model.states_list[0].stateseq.copy()
        rslds.states_list[0].gaussian_states = true_model.states_list[0].gaussian_states.copy()

        rslds._init_mf_from_gibbs()
        rslds._vb_E_step()

    elif initialization == "given":
        print("Initializing with given states")
        rslds.emission_distns[0].J_0 = 1e2 * np.eye(D_latent + 1)
        for i in range(len(ys)):
            rslds.states_list[i].stateseq = z_inits[i].astype(np.int32).copy()
            rslds.states_list[i].gaussian_states = x_inits[i].copy()
            rslds.states_list[i]._init_vbem_from_gibbs()
        rslds._vb_M_step()
        rslds._vb_E_step()

    else:
        print("no initialization")
        rslds._init_mf_from_gibbs()

    # Fit the model
    vlbs = []
    z_smpls = [np.array([z.copy() for z in rslds.stateseqs])]
    for _ in progprint_xrange(N_iters):
        rslds.VBEM_step()
        vlbs.append(rslds.VBEM_ELBO())
        z_smpls.append(np.array([z.copy() for z in rslds.stateseqs]))
        print(rslds.emission_distns[0].sigmasq_flat)

    x_smpl = np.array([s.smoothed_mus.copy() for s in rslds.states_list])
    z_smpls = np.array(z_smpls)
    vlbs = np.array(vlbs)

    if true_model is not None:
        print("True logpi:\n{}".format(true_model.trans_distn.logpi))
        print("True W:\n{}".format(true_model.trans_distn.W))

    print("Inf logpi:\n{}".format(rslds.trans_distn.logpi))
    print("Inf W:\n{}".format(rslds.trans_distn.W))

    return rslds, vlbs, z_smpls, x_smpl

if __name__ == "__main__":
    ## Simulate NASCAR data
    true_model, inputs, z_trues, x_trues, ys, masks, groups = simulate_nascar()

    ## Run PCA to get 2D dynamics
    x_inits, C_init = fit_pca(np.vstack(ys))

    ## Fit an ARHMM for initialization
    arhmm, z_inits = fit_arhmm(x_inits)

    ## Split the initial states
    split_inds = np.cumsum([y.shape[0] for y in ys[:-1]])
    z_inits = np.split(z_inits, split_inds)
    x_inits = np.split(x_inits, split_inds)

    ## Fit a standard SLDS
    slds, slds_lps, slds_z_smpls, slds_x = \
        fit_slds(inputs, z_inits, x_inits, ys, masks, groups, C_init, N_iters=1000)

    ## Fit a recurrent SLDS
    # rslds, rslds_lps, rslds_z_smpls, rslds_x = \
    #     fit_rslds_variational(inputs, y, mask,
    #                           z_init=z_init, x_init=x_init, C_init=C_init,
    #                           initialization="given",
    #                           true_model=true_model, N_iters=100)

    # rslds, rslds_lps, rslds_z_smpls, rslds_x = \
    #     fit_rslds_vbem(inputs, z_init, x_init, y, mask, C_init=C_init,
    #                    true_model=true_model, N_iters=100,
    #                    initialization="true")

    rslds, rslds_lps, rslds_z_smpls, rslds_x = \
        fit_rslds_vbem(inputs, ys, masks, groups,
                       z_inits=slds_z_smpls[-1], x_inits=slds_x, C_init=C_init,
                       initialization="given",
                       true_model=true_model, N_iters=500)

    # rslds, rslds_lps, rslds_z_smpls, rslds_x = \
    #     fit_rslds_vbem(inputs, y, mask,
    #                    initialization="none",
    #                    N_iters=100)

    plot_trajectory_and_probs(
        rslds_z_smpls[-1][0][1:], rslds_x[0][1:],
        trans_distn=rslds.trans_distn,
        title="Recurrent SLDS")

    plot_all_dynamics(rslds.dynamics_distns)

    plot_z_samples(K, [s[0] for s in rslds_z_smpls], plt_slice=(0,1000))

    ## Generate from the model
    T_gen = 2000
    inputs = np.ones((T_gen, 1))
    rslds_y_gen, rslds_x_gen, rslds_z_gen = rslds.generate(T=T_gen, inputs=inputs, group=0)
    slds_y_gen, slds_x_gen, slds_z_gen = slds.generate(T=T_gen, inputs=inputs, group=0)

    make_figure(true_model, z_trues[0], x_trues[0], ys[0],
                rslds, [s[0] for s in rslds_z_smpls], rslds_x[0],
                rslds_z_gen, rslds_x_gen, rslds_y_gen,
                slds, [s[0] for s in slds_z_smpls], slds_x[0],
                slds_z_gen, slds_x_gen, slds_y_gen,
                )

    plt.figure()
    plt.plot(rslds_lps)
    plt.xlabel("Iteration")
    plt.ylabel("ELBO")

    plt.show()
