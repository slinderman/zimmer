from __future__ import division
import numpy as np
import numpy.random as npr
npr.seed(0)

import matplotlib
# matplotlib.use("macosx")  # might be necessary for animation to work
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from hips.plotting.colormaps import gradient_cmap

sns.set_style("white")
sns.set_context("paper")

color_names = ["red",
               "windows blue",
               "medium green",
               "dusty purple",
               "orange",
               "amber",
               "clay",
               "pink",
               "greyish",
               "light cyan",
               "steel blue",
               "forest green",
               "pastel purple",
               "mint",
               "salmon",
               "dark brown"]
colors = sns.xkcd_palette(color_names)
cmap = gradient_cmap(colors)

from pybasicbayes.distributions import Regression, Gaussian
from pyhsmm.util.general import rle
from pylds.util import random_rotation

from zimmer.models import HierarchicalHMMSLDS
from zimmer.emissions import HierarchicalDiagonalRegression


###################
#  generate data  #
###################
N_groups = 4
K = 5                               # number of latent discrete states
D_latent = 2                        # latent linear dynamics' dimension
D_obs = 2                           # data dimension
D_input = 1                         # input dimension
T = 1000                            # duration of data

true_mu_inits = [np.ones(D_latent) for _ in range(K)]
true_sigma_inits = [np.eye(D_latent) for _ in range(K)]
true_As = [.99 * random_rotation(D_latent, theta=np.pi/((k+1) * 4)) for k in range(K)]
true_Bs = [3 * npr.randn(D_latent, D_input) for k in range(K)]
true_Qs = [np.eye(D_latent) for _ in range(K)]
true_C = npr.randn(D_obs, D_latent + D_input)
true_Rs = np.tile(np.logspace(-1, -1 + N_groups, N_groups, endpoint=False)[:,None], (1, D_obs))

init_dynamics_distns = [Gaussian(mu=mu, sigma=sigma)
    for mu, sigma in zip(true_mu_inits, true_sigma_inits)]

dynamics_distns = [Regression(A=np.hstack((A, B)), sigma=Q)
                   for A,B,Q in zip(true_As, true_Bs, true_Qs)]

emission_distns = HierarchicalDiagonalRegression(
    D_obs, D_latent + D_input, N_groups,
    A=true_C, sigmasq=true_Rs)

true_model = HierarchicalHMMSLDS(
    init_dynamics_distns=init_dynamics_distns,
    dynamics_distns=dynamics_distns,
    emission_distns=emission_distns,
    init_state_distn='uniform',
    alpha=3.0)

datas = []
inputs = np.ones((T, D_input))
z = np.arange(K).repeat(T // K)
z_rle = rle(z)
for g in range(N_groups):
    y, x, _ = true_model.generate(T, inputs=inputs, stateseq=z, group=g)
    datas.append(y)

## Plot the data
plt.figure(figsize=(8,4))
for g in range(N_groups):
    plt.subplot(1, N_groups, g+1)
    data = datas[g]
    offset = 0
    for k, dur in zip(*z_rle):
        plt.plot(data[offset:offset + dur, 0], data[offset:offset + dur, 1], color=colors[k])
        offset += dur

    plt.xlabel("$y_1$")
    plt.ylabel("$y_2$")
    plt.title("Group {}".format(g))
plt.tight_layout()

#################
#  test model  #
#################
init_dynamics_distns = \
    [Gaussian(nu_0=D_latent+3,
              sigma_0=3.*np.eye(D_latent),
              mu_0=np.zeros(D_latent),
              kappa_0=0.01)
     for _ in range(2*K)]

dynamics_distns = [Regression(
    nu_0=D_latent + 1,
    S_0=D_latent * np.eye(D_latent),
    M_0=np.hstack((.99 * np.eye(D_latent), np.zeros((D_latent, D_input)))),
    K_0=D_latent * np.eye(D_latent + D_input),
    A=np.hstack((0.99 * np.eye(D_latent), np.zeros((D_latent, D_input)))),
    sigma=np.eye(D_latent))
    for _ in range(2*K)]

emission_distns = HierarchicalDiagonalRegression(
    D_obs, D_latent + D_input, N_groups,
    A=true_C, sigmasq=true_Rs)

model = HierarchicalHMMSLDS(
    init_dynamics_distns=init_dynamics_distns,
    dynamics_distns=dynamics_distns,
    emission_distns=emission_distns,
    init_state_distn='uniform',
    alpha=3.0)

for g, data in enumerate(datas):
    model.add_data(data, inputs=inputs, group=g)
    model.states_list[-1].resample()
    model.states_list[-1]._init_mf_from_gibbs()

##################
#  run sampling  #
##################
n_show = 50
samples = np.empty((n_show, T))
samples[:n_show] = model.stateseqs[0]

fig = plt.figure(figsize=(8,3))
gs = gridspec.GridSpec(6,1)
ax1 = fig.add_subplot(gs[:-1])
ax2 = fig.add_subplot(gs[-1], sharex=ax1)

im = ax1.matshow(samples[::-1], aspect='auto', cmap=cmap, vmax=len(colors))
ax1.autoscale(False)
ax1.set_xticks([])
ax1.set_yticks([])
ax1.set_ylabel("Discrete State")
xo, yo, w, ht = ax1.bbox.bounds
h = ht / n_show

ax2.matshow(z[None,:], aspect='auto', cmap=cmap, vmax=len(colors))
ax2.set_xticks([])
ax2.set_xlabel("Time")
ax2.set_yticks([])

plt.draw()
plt.ion()
plt.show()


print("Press Ctrl-C to stop...")
from itertools import count
for itr in count():
    model.resample_model()
    # model.VBEM_step()

    samples[itr % n_show] = model.stateseqs[0]
    im.set_array(samples[::-1])
    plt.pause(0.001)
