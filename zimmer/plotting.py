# Plot the vector field showing inferred (discrete-state-dependent)
# dynamics at each point in continuous latent space.
# Sum over time bins and smoothed continuous latent states.
# so that each point in continuous latent space is
#    E_{z} E_x_t [A x_{t} + b]
#
# We would ideally sample over A as well, but to start, just take
# a single sample of A.
import os
import numpy as np
from scipy.stats import multivariate_normal

import matplotlib.pyplot as plt

from zimmer.util import states_to_changepoints

def plot_1d_continuous_states(x_inf, z_inf, z_zimmer, colors,
                              x_index=0, plt_slice=(0,1000),
                              results_dir=".", filename="xs_1d.pdf"):
    # Plot one continuous latent state at a time
    cps_inf = states_to_changepoints(z_inf)
    cps_zimmer = states_to_changepoints(z_zimmer)

    plt.figure(figsize=(10, 5))

    # Inferred states
    ax1 = plt.subplot(211)
    for cp_start, cp_stop in zip(cps_inf[:-1], cps_inf[1:]):
        ax1.plot(np.arange(cp_start, cp_stop + 1),
                 x_inf[cp_start:cp_stop + 1, x_index],
                 '-', lw=3,
                 color=colors[z_inf[cp_start]])

    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.set_ylabel("$x_%d$" % (x_index + 1), fontsize=15)
    ax1.set_title("Inferred labels", fontsize=15)

    # Zimmer states
    ax2 = plt.subplot(212)
    for cp_start, cp_stop in zip(cps_zimmer[:-1], cps_zimmer[1:]):
        plt.subplot(212)
        plt.plot(np.arange(cp_start, cp_stop + 1),
                 x_inf[cp_start:cp_stop + 1, x_index],
                 '-', lw=3,
                 color=colors[z_zimmer[cp_start]])

    ax2.set_yticks([])
    ax2.set_xlabel("Frame", fontsize=15)
    ax2.set_ylabel("$x_%d$" % (x_index + 1), fontsize=15)
    ax2.set_title("Kato et. al.", fontsize=15)

    ax1.set_xlim(plt_slice)
    ax2.set_xlim(plt_slice)

    plt.savefig(os.path.join(results_dir, filename))

def plot_2d_continuous_states(ax, x, z, colors,
                              inds=(0,1)):

    cps = states_to_changepoints(z)

    # Color denotes our inferred latent discrete state
    for cp_start, cp_stop in zip(cps[:-1], cps[1:]):
        ax.plot(x[cp_start:cp_stop + 1, inds[0]],
                x[cp_start:cp_stop + 1, inds[1]],
                 '-', color=colors[z[cp_start]])


def plot_vector_field(ax, dds, zs, xs, sigma_xs, kk, inds=(0, 1), P=3, P_in=0,
                      title="",
                      color=np.zeros(4), alpha_max=1.0, alpha_offset=-3, n_pts=50,
                      xmin=-5, xmax=5, ymin=-5, ymax=5):
    dx = (xmax - xmin) / float(n_pts)
    dy = (ymax - ymin) / float(n_pts)

    t_kk = np.where(zs == kk)[0]
    if len(t_kk) == 0:
        return

    # Make a grid for the two dimensions of interest
    omitted = list(set(range(P)) - set(inds))[0]
    if omitted == 0:
        XX, YY, ZZ = np.meshgrid(
            np.linspace(0, 1, 1),
            np.linspace(xmin, xmax, n_pts),
            np.linspace(xmin, xmax, n_pts)
        )
    elif omitted == 1:
        XX, YY, ZZ = np.meshgrid(
            np.linspace(xmin, xmax, n_pts),
            np.linspace(0, 1, 1),
            np.linspace(xmin, xmax, n_pts)
        )
    else:
        XX, YY, ZZ = np.meshgrid(
            np.linspace(xmin, xmax, n_pts),
            np.linspace(xmin, xmax, n_pts),
            np.linspace(0, 1, 1),
        )

    xx, yy, zz = np.ravel(XX), np.ravel(YY), np.ravel(ZZ)
    xyz = np.column_stack((xx, yy, zz))

    # Figure out where the dynamics take each point
    A = dds[kk].A[:, :P]
    if P_in > 0:
        b = dds[kk].A[:, P:]
    else:
        b = 0
    d_xyz = xyz.dot(A.T) + b - xyz

    pr = np.zeros(n_pts ** 2)
    UU, VV = np.zeros(n_pts ** 2), np.zeros(n_pts ** 2)
    for t in t_kk:
        pr_t = multivariate_normal.pdf(
            xyz[:, inds], mean=xs[t][np.ix_(inds)],
            cov=sigma_xs[t][np.ix_(inds, inds)])
        UU += d_xyz[:, inds[0]]
        VV += d_xyz[:, inds[1]]
        pr += pr_t * dx * dy

    UU /= len(t_kk)
    VV /= len(t_kk)
    pr_t /= len(t_kk)

    UU = UU.reshape((n_pts, n_pts))
    VV = VV.reshape((n_pts, n_pts))

    # Make the plot
    XYZ = map(np.squeeze, [XX, YY, ZZ])
    C = np.ones((n_pts ** 2, 1)) * color[None, :]

    logistic = lambda x: 1. / (1 + np.exp(-x))
    pr_to_alpha = lambda pr: alpha_max * logistic((pr - pr.mean()) / pr.std() + alpha_offset)
    C[:, -1] = pr_to_alpha(pr)

    ax.quiver(XYZ[inds[0]], XYZ[inds[1]], UU, VV, color=C,
              scale=1.0, scale_units="inches",
              headwidth=5.,
              )

    ax.set_xlabel("$x_%d$" % (inds[0] + 1), fontsize=15)
    ax.set_ylabel("$x_%d$" % (inds[1] + 1), fontsize=15)

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(xmin, xmax)

    ax.set_title(title, fontsize=15)

