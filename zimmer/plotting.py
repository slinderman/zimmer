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
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as manimation

from hips.plotting.colormaps import gradient_cmap
from hips.plotting.sausage import sausage_plot
from hips.plotting.layout import create_axis_at_location

# Come up with a set of colors
import seaborn as sns
sns.set_style("white")
# sns.set_context("paper")

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

default_colors = sns.xkcd_palette(color_names)
default_cmap = gradient_cmap(default_colors)

from zimmer.util import states_to_changepoints
from tqdm import tqdm


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

    ax1.set_xticklabels([])
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
    ax2.set_title("Zimmer labels", fontsize=15)

    ax1.set_xlim(plt_slice)
    ax2.set_xlim(plt_slice)

    plt.savefig(os.path.join(results_dir, filename))

def plot_2d_continuous_states(x, z, colors,
                              ax=None,
                              inds=(0,1),
                              figsize=(2.5, 2.5),
                              results_dir=".", filename=None,):

    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)

    cps = states_to_changepoints(z)

    # Color denotes our inferred latent discrete state
    for cp_start, cp_stop in zip(cps[:-1], cps[1:]):
        ax.plot(x[cp_start:cp_stop + 1, inds[0]],
                x[cp_start:cp_stop + 1, inds[1]],
                 '-', color=colors[z[cp_start]])

    if filename is not None:
        plt.savefig(os.path.join(results_dir, filename))


def plot_3d_continuous_states(x, z, colors,
                              ax=None,
                              figsize=(2.5,2.5),
                              inds=(0,1,2),
                              title=None,
                              lim=None,
                              results_dir=".", filename=None,
                              **kwargs):

    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection="3d")

    cps = states_to_changepoints(z)

    # Color denotes our inferred latent discrete state
    for cp_start, cp_stop in zip(cps[:-1], cps[1:]):
        ax.plot(x[cp_start:cp_stop + 1, inds[0]],
                x[cp_start:cp_stop + 1, inds[1]],
                x[cp_start:cp_stop + 1, inds[2]],
                '-', marker='.', markersize=1,
                color=colors[z[cp_start]],
                **kwargs)

    # ax.set_xlabel("$x_1$", labelpad=-10)
    # ax.set_ylabel("$x_2$", labelpad=-10)
    # ax.set_zlabel("$x_3$", labelpad=-10)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    if lim is not None:
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_zlim(-lim, lim)

    if title is not None:
        ax.set_title(title)

    plt.tight_layout(pad=0.1)

    if filename is not None:
        plt.savefig(os.path.join(results_dir, filename))


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
    if P_in == 1:
        b = dds[kk].A[:, P:].T
    elif P_in == 0:
        b = 0
    else:
        raise Exception

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
    XYZ = list(map(np.squeeze, [XX, YY, ZZ]))
    C = np.ones((n_pts ** 2, 1)) * color[None, :]

    logistic = lambda x: 1. / (1 + np.exp(-x))
    pr_to_alpha = lambda pr: alpha_max * logistic((pr - pr.mean()) / pr.std() + alpha_offset)
    C[:, -1] = pr_to_alpha(pr)

    ax.quiver(XYZ[inds[0]], XYZ[inds[1]], UU, VV, color=C,
              scale=1., scale_units="inches",
              headwidth=5.,
              )

    ax.set_xlabel("$x_%d$" % (inds[0] + 1), fontsize=15)
    ax.set_ylabel("$x_%d$" % (inds[1] + 1), fontsize=15)

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(xmin, xmax)

    ax.set_title(title, fontsize=15)


def plot_vector_field_2d(ii, z, x, perm_dynamics_distns, colors,
                         inds=(0,1), ax=None, lims=(-3,3), N_plot=500,
                         **kwargs):

    # qargs = dict(arrow_length_ratio=0.25,
    #              length=0.1,
    #              alpha=0.5)
    # qargs.update(kwargs)
    qargs = kwargs

    D = x.shape[1]
    ini = np.where(z == ii)[0]

    # Look at the projected dynamics under each model
    # Subsample accordingly
    if ini.size > N_plot:
        ini_inds = np.random.choice(ini.size, replace=False, size=N_plot)
        ini = ini[ini_inds]

    Ai_full = perm_dynamics_distns[ii].A \
        if hasattr(perm_dynamics_distns[ii], 'A') \
        else perm_dynamics_distns[ii].A_0

    Ai = Ai_full[:, :D]
    bi = Ai_full[:, D] if Ai_full.shape[1] == D+1 else 0
    dxdt = x.dot(Ai.T) + bi - x

    # Create axis if not given
    if ax is None:
        fig = plt.figure(figsize=(5, 5))
        ax = fig.add_subplot(111)

    # ax.quiver(xy[:, 0], xy[:, 1],
    #           dydt_m[:, 0], dydt_m[:, 1],
    #           color=color, alpha=1.0,
    #           headwidth=5.)

    ax.quiver(x[ini, inds[0]], x[ini, inds[1]],
              dxdt[ini, inds[0]], dxdt[ini, inds[1]],
              color=colors[ii],
              **qargs)

    ax.plot(lims, [0, 0], ':k', lw=0.5)
    ax.plot([0, 0], lims, ':k', lw=0.5)

    ax.set_xlabel('$x_{}$'.format(inds[0]+1))
    ax.set_ylabel('$x_{}$'.format(inds[1]+1))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_xlim(lims)
    ax.set_ylim(lims)


def plot_vector_field_3d(ii, z, x, perm_dynamics_distns, colors,
                         ax=None, lims=(-3,3), N_plot=500,
                         **kwargs):

    qargs = dict(arrow_length_ratio=0.5,
                 alpha=1.0,
                 length=1.5,
                 lw=1,
                 pivot="middle",)
    qargs.update(kwargs)

    D = x.shape[1]
    ini = np.where(z == ii)[0]

    # Look at the projected dynamics under each model
    # Subsample accordingly
    if ini.size > N_plot:
        ini_inds = np.random.choice(ini.size, replace=False, size=N_plot)
        ini = ini[ini_inds]

    Ai_full = perm_dynamics_distns[ii].A \
        if hasattr(perm_dynamics_distns[ii], 'A') \
        else perm_dynamics_distns[ii].A_0

    Ai = Ai_full[:, :D]
    bi = Ai_full[:, D] if Ai_full.shape[1] == D+1 else 0
    dxdt = x.dot(Ai.T) + bi - x

    # Create axis if not given
    if ax is None:
        fig = plt.figure(figsize=(5, 5))
        ax = fig.add_subplot(111, projection='3d')

    ax.quiver(x[ini, 0], x[ini, 1], x[ini, 2],
              dxdt[ini, 0], dxdt[ini, 1], dxdt[ini, 2],
              color=colors[ii],
              **qargs)

    ax.set_xlabel('$x_1$', labelpad=-10)
    ax.set_ylabel('$x_2$', labelpad=-10)
    ax.set_zlabel('$x_3$', labelpad=-10)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_zlim(lims)
    ax.w_xaxis.set_pane_color((.9, .9, .9, 0.1))
    ax.w_yaxis.set_pane_color((.9, .9, .9, 0.1))
    ax.w_zaxis.set_pane_color((.9, .9, .9, 0.1))


def plot_3d_dynamics(dynamics_distns, z, x,
                     colors=None,
                     lim=None,
                     figsize=(2.7, 2.7),
                     filepath=None):
    colors = default_colors if colors is None else colors
    for k in range(len(dynamics_distns)):
        fig = plt.figure(figsize=figsize)
        # ax = fig.add_subplot(111, projection='3d')
        ax = create_axis_at_location(fig, 0.025, 0.025, 2.55, 2.55, projection="3d")

        plot_vector_field_3d(k, z, x, dynamics_distns, colors,
                             N_plot=200,
                             ax=ax,
                             lims=(-lim, lim), )
        ax.set_title("State {}".format(k+1))

        if filepath is not None:
            if filepath.endswith('.pdf'):
                filepath = filepath[:-4]
            fig.savefig(filepath + "_{}.pdf".format(k))


def plot_2d_dynamics(dynamics_distns, z, x,
                     inds=(0,1),
                     colors=None,
                     lim=None,
                     filepath=None):
    colors = default_colors if colors is None else colors
    for k in range(len(dynamics_distns)):
        fig = plt.figure(figsize=(2.5, 2.5))
        # ax = fig.add_subplot(111, projection='3d')
        ax = create_axis_at_location(fig, 0.4, 0.4, 1.8, 1.8)
        plot_vector_field_2d(k, z, x, dynamics_distns, colors,
                             inds=inds, ax=ax, lims=(-lim, lim),
                             N_plot=200, alpha=0.8, headwidth=4., lw=.5,
                             units='inches', scale=2.0)
        ax.set_title("State {}".format(k+1))

        if filepath is not None:
            if filepath.endswith('.pdf'):
                filepath = filepath[:-4]
            fig.savefig(filepath + "_{}.pdf".format(k))


def make_states_dynamics_movie(
        dynamics_distns, z, x,
        lim=None,
        colors=None,
        filepath=None):

    colors = default_colors if colors is None else colors

    if filepath.endswith('.mp4'):
        filepath = filepath[:-4]

    for k in range(len(dynamics_distns)):
        FFMpegWriter = manimation.writers['ffmpeg']
        metadata = dict(title='3d dynamics {}'.format(k))
        writer = FFMpegWriter(fps=15, bitrate=1024, metadata=metadata)

        fig = plt.figure(figsize=(4, 4))
        # ax = fig.add_subplot(111, projection='3d')
        ax = create_axis_at_location(fig, 0.05, 0.05, 3.9, 3.9, projection="3d")
        plot_vector_field_3d(k, z, x, dynamics_distns, colors,
                             arrow_length_ratio=0.5, pivot="middle",
                             ax=ax, lims=(-6, 6), alpha=0.8, N_plot=200, length=0.5, lw=1.0)
        ax.set_title("State {}".format(k + 1))

        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_zlabel("")
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])

        if lim is not None:
            ax.set_xlim(-lim,lim)
            ax.set_ylim(-lim,lim)
            ax.set_zlim(-lim,lim)

        def update_frame(i):
            # Rotate the xy plane
            ax.view_init(elev=30., azim=i)

            # Plot the trajectories
            #         plot_trajectories(i, lns)

        with writer.saving(fig, filepath + "_{}.mp4".format(k), 150):
            for i in tqdm(range(360)):
                update_frame(i)
                writer.grab_frame()


def plot_transition_matrix(P, colors, cmap,
                           results_dir=".", filename="trans_matrix.pdf"):

    K = P.shape[0]
    # Look at the transition matrix
    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(121)
    im = ax.imshow(P, interpolation="nearest", cmap="Greys", vmin=0, vmax=1.0)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("$z_{t+1}$", fontsize=15, labelpad=30)
    ax.set_ylabel("$z_{t}$", fontsize=15, labelpad=30)
    ax.set_title("Transition Matrix", fontsize=18)

    divider = make_axes_locatable(ax)
    lax = divider.append_axes("left", size="5%", pad=0.05)
    lax.imshow(np.arange(K)[:, None], cmap=cmap, vmin=0, vmax=len(colors) - 1, aspect="auto",
               interpolation="nearest")
    lax.set_xticks([])
    lax.set_yticks([])

    bax = divider.append_axes("bottom", size="5%", pad=0.05)
    bax.imshow(np.arange(K)[None, :], cmap=cmap, vmin=0, vmax=len(colors) - 1, aspect="auto",
               interpolation="nearest")
    bax.set_xticks([])
    bax.set_yticks([])

    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)

    # View the off diagonal elements
    Pod = P - np.diag(np.diag(P))
    vmax = np.max(Pod)
    ax = fig.add_subplot(122)
    im = ax.imshow(Pod, interpolation="nearest", cmap="Greys", vmin=0, vmax=vmax)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("$z_{t+1}$", fontsize=15, labelpad=30)
    ax.set_ylabel("$z_{t}$", fontsize=15, labelpad=30)
    ax.set_title("Off Diagonal Only", fontsize=18)

    divider = make_axes_locatable(ax)
    lax = divider.append_axes("left", size="5%", pad=0.05)
    lax.imshow(np.arange(K)[:, None], cmap=cmap, vmin=0, vmax=len(colors) - 1, aspect="auto",
               interpolation="nearest")
    lax.set_xticks([])
    lax.set_yticks([])

    bax = divider.append_axes("bottom", size="5%", pad=0.05)
    bax.imshow(np.arange(K)[None, :], cmap=cmap, vmin=0, vmax=len(colors) - 1, aspect="auto",
               interpolation="nearest")
    bax.set_xticks([])
    bax.set_yticks([])

    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)

    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, filename))


def plot_discrete_state_samples(z_smpls, z_true, Kmax,
                                plt_slice=(0, 2500),
                                colors=None, cmap=None,
                                filepath=None):

    colors = default_colors if colors is None else colors
    cmap = default_cmap if cmap is None else cmap

    # Plot the true and inferred state sequences
    fig = plt.figure(figsize=(10, 5))
    gs = gridspec.GridSpec(7, 1)

    ax1 = fig.add_subplot(gs[:-2])
    ax2 = fig.add_subplot(gs[-2])
    ax3 = fig.add_subplot(gs[-1])

    assert len(colors) > Kmax

    im = ax1.matshow(z_smpls, aspect='auto', cmap=cmap, vmin=0, vmax=len(colors) - 1)
    ax1.autoscale(False)
    ax1.set_xticks([])
    ax1.set_yticks([0, z_smpls.shape[0]])
    ax1.set_ylabel("Iteration")
    ax1.set_xlim(plt_slice)
    ax1.set_xticks(plt_slice)

    ax2.matshow(z_smpls[-1][None, :], aspect='auto', cmap=cmap, vmin=0, vmax=len(colors) - 1)
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.set_ylabel("Final")
    ax2.set_xlim(plt_slice)

    ax3.matshow(z_true[None, :], aspect='auto', cmap=cmap, vmin=0, vmax=len(colors) - 1)
    ax3.set_xticks([])
    ax3.set_yticks([])
    ax3.set_ylabel("Manual")
    ax3.set_xlabel("Labels per time bin")
    ax3.set_xlim(plt_slice)

    if filepath is not None:
        plt.savefig(filepath)

def plot_changepoint_prs(z_smpls, z_true, title=None, plt_slice=(0, 2500),
                         cmap=None, colors=None,
                         filepath=None):

    colors = default_colors if colors is None else colors
    cmap = default_cmap if cmap is None else cmap

    # Plot the true and inferred state sequences
    N_samples = z_smpls.shape[0]
    fig = plt.figure(figsize=(5.5, 3))

    ax1 = create_axis_at_location(fig, 1.0, 1.45, 4.25, 1.25)
    ax2 = create_axis_at_location(fig, 1.0, .95, 4.25, 0.40)
    ax3 = create_axis_at_location(fig, 1.0, 0.45, 4.25, 0.40)

    im = ax1.matshow(z_smpls, aspect='auto', cmap=cmap, vmin=0, vmax=len(colors) - 1)
    ax1.autoscale(False)
    ax1.set_xticks([])
    ax1.set_yticks([0, N_samples])
    ax1.set_ylabel("Iteration", labelpad=13)
    ax1.set_xlim(plt_slice)
    # ax1.set_xticks(plt_slice)
    ax1.set_xticks([])

    if title is not None:
        ax1.set_title(title)

    # Compute changepoint probability
    ischangepoint = lambda z: np.concatenate(([0], np.diff(z) != 0))
    sampled_changepoints = np.array([ischangepoint(z) for z in z_smpls[-N_samples//2:]])
    changepoint_pr = np.mean(sampled_changepoints, axis=0)
    ax2.plot(changepoint_pr, '-k', lw=0.5)
    ax2.set_xticks([])
    ax2.set_yticks([0, .5, 1])
    ax2.set_ylabel("CP Pr.", labelpad=20, rotation=0)
    ax2.set_xlim(plt_slice)

    ax3.matshow(z_true[None, :], aspect='auto', cmap=cmap, vmin=0, vmax=len(colors) - 1)
    ax3.set_yticks([])
    ax3.set_ylabel("Kato et al", labelpad=35, rotation=0)
    ax3.set_xlabel("Time")
    ax3.set_xlim(plt_slice)
    # ax3.set_xticks(plt_slice)
    ax3.xaxis.tick_bottom()

    if filepath is not None:
        plt.savefig(filepath)


from pyhsmm.util.general import rle


def plot_latent_trajectories_vs_time(xs, zs,
                                     colors=None,
                                     plot_slice=(0, 500),
                                     alpha=0.5,
                                     title=None,
                                     basename="x_segmentation",
                                     results_dir=None):

    colors = default_colors if colors is None else colors
    for i, (x, z) in enumerate(zip(xs, zs)):
        z_rle = rle(z)
        lim = 1.1 * abs(x).max()
        x = x / (2 * lim)
        D_latent  = x.shape[1]

        plt.figure(figsize=(6, 4))
        ax = plt.subplot(111)

        # Plot z in background
        offset = 0
        for k, dur in zip(*z_rle):
            ax.fill_between([offset, offset + dur], [-D_latent, -D_latent], [0, 0],
                            color=colors[k], alpha=alpha)
            offset += dur
            if offset > plot_slice[1]:
                break

                # Plot x
        for d in range(D_latent):
            ax.plot(x[:, d] - d - 0.5, '-k', lw=2)
            ax.plot(plot_slice, (-d - 0.5) * np.ones(2), ':k', lw=1)

        ax.set_xlim(plot_slice)
        ax.set_xlabel("Time")
        if title is None:
            ax.set_title("segmentation of $x$ (Worm {})".format(i+1))
        else:
            ax.set_title(title + " (Worm {})".format(i+1))

        ax.set_yticks(-1 * np.arange(D_latent) - 0.5)
        ax.set_yticklabels(["$x_{{{}}}$".format(d + 1) for d in range(D_latent)])
        ax.set_ylim(-D_latent, 0)
        plt.tight_layout()

        if results_dir is not None:
            plt.savefig(os.path.join(results_dir, basename + "_{}.pdf".format(i)))


def plot_expected_states(E_z, cp_pr, z_true,
                         title=None, plt_slice=(0, 2500),
                         colors=None,
                         filepath=None):

    colors = default_colors if colors is None else colors
    K = E_z.shape[1]

    # Plot the true and inferred state sequences
    fig = plt.figure(figsize=(5.5, 3))

    ax1 = create_axis_at_location(fig, 1.0, 1.45, 4.25, 1.25)
    ax2 = create_axis_at_location(fig, 1.0, .95, 4.25, 0.40)
    ax3 = create_axis_at_location(fig, 1.0, 0.45, 4.25, 0.40)

    im = ax1.matshow(E_z.T, aspect='auto', cmap="Greys", vmin=0, vmax=1)
    ax1.autoscale(False)
    ax1.set_xticks([])
    ax1.set_yticks(np.arange(K))
    ax1.set_yticklabels(np.arange(K)+1)
    ax1.set_ylim([K-0.5, -0.5])
    ax1.set_ylabel("Discrete State", labelpad=13)
    ax1.set_xlim(plt_slice)
    # ax1.set_xticks(plt_slice)
    ax1.set_xticks([])

    if title is not None:
        ax1.set_title(title)

    # Compute changepoint probability
    ax2.plot(cp_pr, '-k', lw=0.5)
    ax2.set_xticks([])
    ax2.set_yticks([0, .5, 1])
    ax2.set_ylabel("CP Pr.", labelpad=20, rotation=0)
    ax2.set_xlim(plt_slice)

    ax3.matshow(z_true[None, :], aspect='auto', cmap=gradient_cmap(colors), vmin=0, vmax=len(colors) - 1)
    ax3.set_yticks([])
    ax3.set_ylabel("Kato et al", labelpad=35, rotation=0)
    ax3.set_xlabel("Time")
    ax3.set_xlim(plt_slice)
    # ax3.set_xticks(plt_slice)
    ax3.xaxis.tick_bottom()

    if filepath is not None:
        plt.savefig(filepath)


def plot_state_overlap(z_finals, z_trues,
                       z_key=None,
                       colors=None,
                       z_colors=None,
                       results_dir=None):
    colors = default_colors if colors is None else colors
    z_colors = default_colors if z_colors is None else z_colors
    Kmax = np.max(np.concatenate(z_finals)) + 1
    K_zimmer = np.concatenate(z_trues).max() + 1

    # Use the Hungarian algorithm to find a permutation of states that
    # yields the highest overlap
    overlap = np.zeros((K_zimmer, Kmax), dtype=float)
    for z_true, z_inf in zip(z_trues, z_finals):
        for k1 in range(K_zimmer):
            for k2 in range(Kmax):
                overlap[k1, k2] = np.sum((z_true == k1) & (z_inf == k2))

    from scipy.optimize import linear_sum_assignment
    _, perm = linear_sum_assignment(-overlap)


    # Compare zimmer labels to inferred labels
    for worm, (z_true, z_inf) in enumerate(zip(z_trues, z_finals)):
        overlap = np.zeros((K_zimmer, Kmax), dtype=float)
        for k1 in range(K_zimmer):
            for k2 in range(Kmax):
                overlap[k1, k2] = np.sum((z_true == k1) & (z_inf == k2))

        # Normalize the rows
        overlap /= overlap.sum(1)[:, None]

        fig = plt.figure(figsize=(2., 2))
        ax1 = create_axis_at_location(fig, .6, .5, 1., 1)
        im = ax1.imshow(overlap[:, perm], vmin=0, vmax=.75, interpolation="nearest", aspect="auto")
        ax1.set_xticks([])
        if z_key is None:
            ax1.set_yticks([])
        else:
            ax1.set_yticks(np.arange(K_zimmer))
            ax1.set_yticklabels(z_key, fontdict=dict(size=6))
            ax1.tick_params(axis='y', which='major', pad=11)
        ax1.set_title("State Overlap (Worm {})".format(worm + 1))

        lax = create_axis_at_location(fig, .5, .5, .06, 1)
        lax.imshow(np.arange(K_zimmer)[:, None], cmap=gradient_cmap(z_colors[:K_zimmer]), interpolation="nearest",
                   aspect="auto")
        lax.set_xticks([])
        lax.set_yticks([])

        if z_key is None:
            lax.set_ylabel("Zimmer State", fontsize=8)

        bax = create_axis_at_location(fig, .6, .4, 1., .06)
        bax.imshow(np.arange(Kmax)[perm][None, :], cmap=gradient_cmap(colors[:Kmax]), interpolation="nearest", aspect="auto")
        bax.set_xticks([])
        bax.set_yticks([])
        bax.set_xlabel("Inferred State", fontsize=8)

        axcb = create_axis_at_location(fig, 1.65, .5, .1, 1)
        plt.colorbar(im, cax=axcb)

        if results_dir is not None:
            plt.savefig(os.path.join(results_dir, "overlap_{}.pdf".format(worm)))


def plot_state_usage_by_worm(z_finals,
                             colors=None,
                             results_dir=None):
    colors = default_colors if colors is None else colors
    N_worms = len(z_finals)
    Kmax = np.max(np.concatenate(z_finals)) + 1

    usage = np.zeros((N_worms, Kmax))
    for worm in range(N_worms):
        usage[worm] = np.bincount(z_finals[worm], minlength=Kmax)

    usage = usage / usage.sum(axis=1)[:, None]

    sns.set_context("paper")
    fig = plt.figure(figsize=(5, 1.25))
    gs = gridspec.GridSpec(1, Kmax)
    for k in range(Kmax):
        ax = fig.add_subplot(gs[0, k])
        ax.bar(np.arange(N_worms), usage[:, k], width=1, color=colors[k], edgecolor="k")
        ax.set_title("State {}".format(k+1), fontsize=6, y=0.8)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.set_xticks(np.arange(N_worms))
        ax.set_xticklabels(np.arange(1, N_worms + 1), fontsize=6)
        ax.set_ylim(0, 0.45)
        if k > 0:
            ax.set_yticks([])
            ax.spines['left'].set_visible(False)
        else:
            ax.set_yticks([0, .2, .4])
            ax.set_yticklabels([0, 20, 40], fontsize=6)
            ax.set_ylabel("% Usage", fontsize=8)

        # if k == Kmax // 2:
        #     ax.set_xlabel("Worm", fontsize=6)
        ax.set_xlabel("Worm", fontsize=6)

    plt.tight_layout(rect=(0.01, 0.1, 0.98, 0.98), pad=0.01)

    if results_dir is not None:
        plt.savefig(os.path.join(results_dir, "state_usage_per_worm.pdf"))



def plot_state_usage_by_worm_matrix(z_finals,
                             colors=None,
                             results_dir=None):
    colors = default_colors if colors is None else colors
    N_worms = len(z_finals)
    Kmax = np.max(np.concatenate(z_finals)) + 1

    usage = np.zeros((N_worms, Kmax))
    for worm in range(N_worms):
        usage[worm] = np.bincount(z_finals[worm], minlength=Kmax)

    usage = usage / usage.sum(axis=1)[:, None]

    sns.set_context("paper")

    fig = plt.figure(figsize=(3, 2))
    ax = fig.add_subplot(111)
    im = ax.imshow(usage, cmap="Greys")
    ax.set_xlabel("State")
    ax.set_xticks(np.arange(Kmax))
    ax.set_xticklabels(np.arange(Kmax) + 1)
    ax.set_ylabel("Worm")
    ax.set_yticks(np.arange(N_worms))
    ax.set_yticklabels(np.arange(N_worms)+1)
    plt.title("State usage by worm")

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)

    plt.tight_layout()

    if results_dir is not None:
        plt.savefig(os.path.join(results_dir, "state_usage_per_worm2.pdf"))


def _count_transitions(z1s, z2s):
    Kmax = z1s.max() + 1
    assert Kmax == z2s.max() + 1

    trans_matrix = np.zeros((Kmax, Kmax))
    for z1, z2 in zip(z1s, z2s):
        trans_matrix[z1, z2] += 1
    trans_matrix /= (1e-8 + trans_matrix.sum(1)[:, None])
    return trans_matrix


def _permute_feedforward(zs):
    assert isinstance(zs, list)
    Kmax = np.max(np.concatenate(zs)) + 1

    # Find a permutation to make the matrix the most "feed forward"
    # Use a simple greedy search to do so
    trans_matrix = _count_transitions(
        np.concatenate([zf[:-1] for zf in zs]),
        np.concatenate([zf[1:] for zf in zs]))
    np.fill_diagonal(trans_matrix, 0)

    perm = [0]
    remaining = list(range(1, Kmax))
    for level in range(1, Kmax):
        next = np.argmax(trans_matrix[perm[-1], remaining])
        perm.append(remaining[next])
        del remaining[next]
        print("Select {}. Remaining {}".format(next, remaining))

    return perm

def plot_all_transition_matrices(z_finals,
                                 colors=None,
                                 results_dir=None):
    colors = default_colors if colors is None else colors
    N_worms = len(z_finals)

    from pyhsmm.util.general import relabel_by_permutation
    perm = _permute_feedforward(z_finals)
    z_finals_perm = [relabel_by_permutation(zf, np.argsort(perm)) for zf in z_finals]

    def _plot_transition_matrix(ax, P, colors, cmap, vmax=None, plot_colorbar=False):

        K = P.shape[0]
        # View the off diagonal elements
        Pod = P - np.diag(np.diag(P))

        # Renormalize rows
        Pod = Pod / Pod.sum(1, keepdims=True)

        vmax = np.max(Pod) if vmax is None else vmax
        im = ax.imshow(Pod, interpolation="nearest", cmap="Greys", vmin=0, vmax=vmax)
        ax.set_xticks([])
        ax.set_yticks([])

        divider = make_axes_locatable(ax)
        lax = divider.append_axes("left", size="5%", pad=0.01)
        lax.imshow(np.arange(K)[perm][:, None], cmap=cmap, vmin=0, vmax=len(colors) - 1, aspect="auto",
                   interpolation="nearest")
        lax.set_xticks([])
        lax.set_yticks([])

        bax = divider.append_axes("bottom", size="5%", pad=0.01)
        bax.imshow(np.arange(K)[perm][None, :], cmap=cmap, vmin=0, vmax=len(colors) - 1, aspect="auto",
                   interpolation="nearest")
        bax.set_xticks([])
        bax.set_yticks([])

        if plot_colorbar:
            cax = divider.append_axes("right", size="5%", pad=0.05)
            plt.colorbar(im, cax=cax)

    fig = plt.figure(figsize=(5.5, 1.5))
    for worm in range(N_worms):
        ax = fig.add_subplot(1, N_worms, worm + 1)
        trans_matrix = _count_transitions(z_finals_perm[worm][:-1], z_finals_perm[worm][1:])
        _plot_transition_matrix(ax, trans_matrix, colors=colors, cmap=gradient_cmap(colors), vmax=1)
        ax.set_title("Worm {}".format(worm + 1))
        ax.set_xlabel("$z_{t+1}$", labelpad=10)
        if worm == 0:
            ax.set_ylabel("$z_t$", labelpad=10)

    plt.tight_layout(pad=0.5)

    if results_dir is not None:
        plt.savefig(os.path.join(results_dir, "trans_matrices.pdf"))


def plot_simulated_trajectories(k, x_trajs, x_sim, C, d, T,
                                colors=None,
                                results_dir=None,
                                lim=3,
                                ylim=.3):
    colors = default_colors if colors is None else colors
    N_clusters = C.shape[0]

    # Find and plot the partial trajectories of this state
    fig = plt.figure(figsize=(5, 2.5))
    gs = gridspec.GridSpec(N_clusters, 2, width_ratios=[1.0, 1.0])
    ax = fig.add_subplot(gs[:, 0], projection="3d")

    ax.set_title("State {} Trajectories".format(k+1))

    amax = 0.5
    amin = 0.
    for x_traj in x_trajs:
        t_traj = min(x_traj.shape[0], T)

        # plt.plot(x_traj[:, 0], x_traj[:, 1], x_traj[:, 2],
        #          '-', lw=1.0, color=colors[k], alpha=amin)

        for t in range(t_traj-1):
            ax.plot(x_traj[t:t+2, 0], x_traj[t:t+2, 1], x_traj[t:t+2, 2], '-',
                    color=colors[k], lw=1,
                    alpha=(amax - amin) * (1 - t/float(t_traj)) + amin
                    )


    for i,c_unnorm in enumerate(C):
        c = c_unnorm / np.linalg.norm(c_unnorm)
        ax.plot([0, 3 * c[0]], [0, 3 * c[1]], [0, 3 * c[2]], '-k')
        # ax.plot([3 * c[0]], [3 * c[1]], [3 * c[2]], 'ko')
        ax.text(4 * c[0], 4 * c[1], 4 * c[2], '{}'.format(i+1), fontsize=6)

    # plt.plot(x_sim[:1, 0], x_sim[:1, 1], x_sim[:1, 2], 'o', color=colors[k])
    # plt.plot(x_sim[:, 0], x_sim[:, 1], x_sim[:, 2], '.', markersize=4, color=colors[k], lw=1)

    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(-lim, lim)
    ax.set_xticks([-lim, -lim/2, 0, lim/2, lim])
    ax.set_yticks([-lim, -lim/2, 0, lim/2, lim])
    ax.set_zticks([-lim, -lim/2, 0, lim/2, lim])
    # ax.set_xticklabels([-lim, None, 0, None, lim])
    # ax.set_yticklabels([-lim, None, 0, None, lim])
    # ax.set_zticklabels([-lim, None, 0, None, lim])

    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    ax.set_xlabel("$x_1$", labelpad=-15)
    ax.set_ylabel("$x_2$", labelpad=-15)
    ax.set_zlabel("$x_3$", labelpad=-15)

    # Plot the neural activity
    # Y_sim = x_sim.dot(Cs.T)
    Y_trajs = [x_traj.dot(C.T) + d for x_traj in x_trajs]

    # Concatenate all the Y_trajs
    all_Y_trajs = np.nan * np.zeros((len(Y_trajs), T, N_clusters))
    for i, y_traj in enumerate(Y_trajs):
        ti = min(T, y_traj.shape[0])
        all_Y_trajs[i, :ti] = y_traj[:ti]
    y_mean = np.nanmean(all_Y_trajs, axis=0)
    y_std = np.nanstd(all_Y_trajs, axis=0)
    y_count = np.sum(np.isfinite(all_Y_trajs[:,:,0]), axis=0)

    T_valid = np.where(y_count > 10)[0].max() + 1
    t_sim = np.arange(T_valid) / 3.0

    for c in range(N_clusters):
        axc = fig.add_subplot(gs[c, 1])
        # axc.plot(t_sim, Y_sim[:, c], '-', color=colors[k])
        # for y_traj in Y_trajs:
        #     axc.plot(np.arange(y_traj.shape[0]) / 3.0, y_traj[:, c], '-', alpha=0.5, color=colors[k], lw=1)
        # axc.fill_between(t_sim,
        #                  y_mean[:T_valid, c] - 2 * y_std[:T_valid, c],
        #                  y_mean[:T_valid, c] + 2 * y_std[:T_valid, c],
        #                  color=colors[k], alpha=0.25)
        # axc.plot(t_sim, y_mean[:T_valid, c], '-', color=colors[k])

        # axc.plot(all_Y_trajs[:, :T_valid, c].T, lw=0.5, color=colors[k], alpha=0.25)
        for ytr in all_Y_trajs:
            # axc.plot(t_sim, np.cumsum(ytr[:T_valid, c]), lw=0.5, color=colors[k], alpha=0.25)
            axc.plot(t_sim, ytr[:T_valid, c], lw=0.5, color=colors[k], alpha=0.25)

        axc.plot(t_sim, np.zeros_like(t_sim), ':k', lw=1)
        if c == 0:
            axc.set_title("Neural Cluster Activity")

        axc.set_ylabel("cluster {}".format(c + 1), fontsize=6)
        axc.set_ylim(-ylim, ylim)
        axc.set_yticks([-ylim, 0, ylim])
        axc.set_yticklabels([-ylim, 0, ylim], fontsize=6)

        if c == N_clusters - 1:
            axc.set_xlabel("Time (sec)", fontsize=6)
            axc.set_xticks([0, 5, 10, 15, 20])
            axc.set_xticklabels([0, 5, 10, 15, 20], fontsize=6)
        else:
            axc.set_xticks([])

        from matplotlib.patches import Rectangle
        if T_valid < T:
            axc.add_patch(Rectangle(((T_valid - 1)/ 3.0, -ylim), 20 - (T_valid - 1)/ 3.0, 2*ylim,
                                    edgecolor='lightgray',
                                    fill=False, hatch='//'))

        axc.set_xlim(0, (T-1) / 3.0)

    plt.tight_layout(pad=1.0)

    if results_dir is not None:
        plt.savefig(os.path.join(results_dir, "canonical_cluster_activity_{}.pdf".format(k)))
        plt.savefig(os.path.join(results_dir, "canonical_cluster_activity_{}.png".format(k)), dpi=300)


def plot_simulated_trajectories2(k, x_trajs, x_sims, C, d, T,
                                colors=None,
                                results_dir=None,
                                lim=3,
                                ylim=.15,
                                alpha=0.25,
                                markers=('o', '^',  's', 'p', 'h')):
    """
    Similar to above but with two-dimensional views as well
    """
    colors = default_colors if colors is None else colors
    N_clusters, D = C.shape

    # Find and plot the partial trajectories of this state
    fig = plt.figure(figsize=(5.5, 3))
    gs = gridspec.GridSpec(2, 4, height_ratios=[2., 1.])
    ax = fig.add_subplot(gs[0, 0], projection="3d", aspect="equal")

    for x_traj in x_trajs:
        ax.plot(x_traj[:1, 0], x_traj[:1, 1], x_traj[:1, 2],
                'o', markersize=3, markeredgecolor='k', mew=1, color=colors[k], alpha=alpha)
        ax.plot(x_traj[:, 0], x_traj[:, 1], x_traj[:, 2],
                 '-', lw=1.0, color=colors[k], alpha=alpha)

    # for i,c_unnorm in enumerate(C):
    #     c = c_unnorm / np.linalg.norm(c_unnorm)
    #     ax.plot([0, 3 * c[0]], [0, 3 * c[1]], [0, 3 * c[2]], ':', color='k')
    #     # ax.plot([3 * c[0]], [3 * c[1]], [3 * c[2]], 'ko')
    #     ax.text(4 * c[0], 4 * c[1], 4 * c[2], '{}'.format(i+1), fontsize=8)
    #

    for i, x_sim in enumerate(x_sims):
        plt.plot(x_sim[:, 0], x_sim[:, 1], x_sim[:, 2], '-', color='k', lw=1.5)
        plt.plot(x_sim[:1, 0], x_sim[:1, 1], x_sim[:1, 2], 'o',
                 marker=markers[i % len(markers)],
                 markerfacecolor=colors[k],
                 markeredgecolor='k',
                 markeredgewidth=1,
                 markersize=5,
                 alpha=0.75)

    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(-lim, lim)
    ax.set_xticks([-lim, -lim/2, 0, lim/2, lim])
    ax.set_yticks([-lim, -lim/2, 0, lim/2, lim])
    ax.set_zticks([-lim, -lim/2, 0, lim/2, lim])

    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    ax.set_xlabel("$x_1$", fontsize=6, labelpad=-15)
    ax.set_ylabel("$x_2$", fontsize=6, labelpad=-15)
    ax.set_zlabel("$x_3$", fontsize=6, labelpad=-15)

    def _plot_2d_trajectories(ax, dims, legend=False):
        ax.plot([-lim, lim], [0, 0], ':k', lw=0.5)
        ax.plot([0, 0], [-lim, lim], ':k', lw=0.5)
        for x_traj in x_trajs:
            ax.plot(x_traj[0, dims[0]], x_traj[0, dims[1]],
                    'o', markersize=3, markeredgecolor='k', mew=1, color=colors[k], alpha=alpha)
            ax.plot(x_traj[:, dims[0]], x_traj[:, dims[1]],
                    ls='-', lw=1.0, color=colors[k], alpha=alpha)

        for i, x_sim in enumerate(x_sims):
            plt.plot(x_sim[:, dims[0]], x_sim[:, dims[1]], '-', color='k', lw=1.5)
            plt.plot(x_sim[0, dims[0]], x_sim[0, dims[1]],
                     ls='',
                     marker=markers[i % len(markers)],
                     markerfacecolor=colors[k],
                     markeredgecolor='k',
                     markeredgewidth=1,
                     markersize=5,
                     alpha=0.75,
                     label=i+1)

        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_xticks([-lim, -lim / 2, 0, lim / 2, lim])
        ax.set_yticks([-lim, -lim / 2, 0, lim / 2, lim])
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_xlabel("$x_{}$".format(dims[0]+1), fontsize=6, labelpad=-1)
        ax.set_ylabel("$x_{}$".format(dims[1]+1), fontsize=6, labelpad=-1)

        if legend:
            ax.legend(loc="lower right",
                      fontsize=6,
                      labelspacing=.15,
                      borderpad=.1,
                      handletextpad=.1)

    for i, dims in enumerate([(0,1), (0,2), (1,2)]):
        ax = fig.add_subplot(gs[0,i+1], aspect="equal")
        _plot_2d_trajectories(ax, dims, legend=(i == 0))

    # Plot the neural activity
    for i, x_sim in enumerate(x_sims):
        ax = fig.add_subplot(gs[1, i])
        y_sim = x_sim.dot(C.T) + d
        t_sim = np.arange(x_sim.shape[0]) / 3.0

        for c in range(N_clusters):
            ax.plot([0, T / 3.], np.ones(2) * c * 2 * ylim, ':k', lw=1)
            ax.plot(t_sim, y_sim[:, c] + c * 2 * ylim, '-', color=colors[k], lw=2)

        # Plot the marker for convenience
        ax.plot(T / 3.0 - 1, 0,
                marker=markers[i % len(markers)],
                markerfacecolor=colors[k],
                markeredgecolor='k',
                markeredgewidth=1,
                markersize=6,
                label=i + 1)

        ax.set_ylabel("cluster activity", fontsize=6)
        ax.set_ylim((2 * N_clusters - 1) * ylim, -ylim)
        ax.set_yticks(np.arange(N_clusters) * 2 * ylim)
        ax.set_yticklabels(np.arange(N_clusters) + 1, size=6)
        ax.set_xlabel("time (sec)", fontsize=6)
        ax.set_xticks(np.linspace(0, T /3.0, 5))
        ax.set_xticklabels(np.linspace(0, T /3.0, 5), size=6)
        ax.set_xlim(0, T/3.0)
        ax.set_title("Trajectory {}".format(i+1))

    plt.tight_layout(pad=.3)

    if results_dir is not None:
        plt.savefig(os.path.join(results_dir, "cluster_activity_{}.pdf".format(k)))
        # plt.savefig(os.path.join(results_dir, "cluster_activity_{}.png".format(k)), dpi=300)


def plot_recurrent_transitions(trans_distn, xs, zs,
                               perm=None,
                               results_dir=None):
    logpi = trans_distn.logpi
    W = trans_distn.W
    lim = abs(np.vstack((logpi, W))).max()
    D, K = W.shape

    xs = np.vstack(xs) if isinstance(xs, list) else xs
    zs = np.concatenate(zs) if isinstance(zs, list) else zs
    zs = zs.astype(int)
    N, D2 = xs.shape
    assert D == D2
    assert zs.shape == (N,)

    if perm is None:
        perm = _permute_feedforward([zs])

    # Multiply w by the standard deviation of xs,
    # since Wx is that we actually care about

    fig = plt.figure(figsize=(5.5, 2.5))
    ax1 = fig.add_subplot(131, aspect="equal")
    ax1.imshow(logpi[np.ix_(perm, perm)], vmin=-lim, vmax=lim, cmap="RdBu_r")
    ax1.set_ylabel("$z_t$")
    ax1.set_xlabel("$z_{t+1}$")
    ax1.set_xticks(np.arange(K))
    ax1.set_xticklabels(np.arange(K)+1)
    ax1.set_yticks(np.arange(K))
    ax1.set_yticklabels(np.arange(K) + 1)
    ax1.set_title("Markov weights")

    ax2 = fig.add_subplot(132, aspect="equal")
    im = ax2.imshow(W[:,perm] * np.std(xs), vmin=-lim, vmax=lim, cmap="RdBu_r")
    ax2.set_ylabel("$x_t$")
    ax2.set_xlabel("$z_{t+1}$")
    ax2.set_xticks(np.arange(K))
    ax2.set_xticklabels(np.arange(K) + 1)
    ax2.set_yticks(np.arange(D))
    ax2.set_yticklabels(np.arange(D) + 1)
    ax2.set_title("Recurrent weights")

    divider = make_axes_locatable(ax2)
    rax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=rax)

    # Compute the log weights under the two components
    cps = np.where(np.diff(zs) != 0)[0]
    trm1 = logpi[zs].ravel()
    trm1_cps = logpi[zs[cps]].ravel()
    trm2 = xs.dot(W).ravel()
    trm2_cps = xs[cps].dot(W).ravel()
    lim = np.percentile(np.concatenate((abs(trm1), abs(trm2))), 99.5)
    bins = np.linspace(-lim, lim, 30)

    ax3 = fig.add_subplot(133, aspect=20)
    # ax3.hist(trm1, bins, color=default_colors[2], alpha=0.5, edgecolor='k', label="Mkv", normed=True)
    # ax3.hist(trm2, bins, color=default_colors[3], alpha=0.5, edgecolor='k', label="Rec", normed=True)
    ax3.hist(trm1_cps, bins, color=default_colors[2], alpha=0.5, edgecolor='k', label="Mkv", normed=True)
    ax3.hist(trm2_cps, bins, color=default_colors[3], alpha=0.5, edgecolor='k', label="Rec", normed=True)
    # sns.kdeplot(trm1, color=default_colors[2], alpha=0.5, shade=True, label="Mkv")
    # sns.kdeplot(trm2, color=default_colors[3], alpha=0.5, shade=True, label="Mkv")
    # sns.kdeplot(trm2_cps, color=default_colors[4], alpha=0.5, shade=True, label="Mkv")
    ax3.legend(loc="upper left",
               fontsize=6,
               labelspacing=.15,
               borderpad=.1,
               # handletextpad=.25
               )
    ax3.set_xlabel("weight")
    ax3.set_xlim(-lim, lim)
    ax3.set_ylim(0, 0.5)
    ax3.set_yticks([0, 0.25, 0.5])
    ax3.set_title("weight distribution")

    plt.tight_layout(pad=1)

    if results_dir is not None:
        plt.savefig(os.path.join(results_dir, "recurrent_weights.pdf"))



def plot_duration_histogram(trans_distn, zs,
                            colors=None,
                            results_dir=None):

    from pyhsmm.util.general import rle
    from scipy.stats import geom
    colors = default_colors if colors is None else colors

    states, durs = list(zip(*[rle(z) for z in zs]))
    states = np.concatenate(states)
    durs = np.concatenate(durs)
    K = np.max(states) + 1

    logpi = trans_distn.logpi
    P = np.exp(logpi)
    P /= P.sum(axis=1, keepdims=True)

    for k in range(K):
    # for k in range(1):
        p_stay = P[k, k]
        g = geom(1-p_stay)
        dk = durs[states == k]
        dmax = dk.max()
        bins = np.linspace(0, dmax+1, 15)

        print(len(dk))

        plt.figure(figsize=(1.8, 1.8))
        plt.hist(dk / 3.0, bins/ 3.0, color=colors[k], edgecolor='k', normed=True, label="Empirical")
        plt.plot(np.arange(1, dmax) / 3.0, g.pmf(np.arange(1, dmax)) * 3.0, '-k', label="Markov")
        plt.xlabel("duration (s)")
        plt.ylabel("probability")
        plt.legend(loc="upper right", fontsize=6)
        plt.title("State {}".format(k+1))

        plt.tight_layout(pad=1)

        if results_dir is not None:
            plt.savefig(os.path.join(results_dir, "durations_{}.pdf".format(k)))


def plot_x_at_changepoints(zs, xs, window=9, colors=None,
                           basename="x_cp_avg",
                           results_dir=None):
    colors = default_colors if colors is None else colors

    xs = np.vstack(xs) if isinstance(xs, list) else xs
    N, D = xs.shape
    zs = np.concatenate(zs) if isinstance(zs, list) else zs
    K = zs.max() + 1

    cps = np.where(np.diff(zs) != 0)[0]
    z_pres = zs[cps]
    z_posts = zs[cps+1]

    tt = np.arange(-window, window+1) / 3.0
    for k in range(K):
        to_k = cps[z_posts == k]
        assert np.all(zs[to_k+1] == k)

        fig = plt.figure(figsize=(2.5, 2.5))
        ax = fig.add_subplot(111)

        X = []
        for t in to_k:
            l = t - max(0, t-window)
            r = min(xs.shape[0], t+window+1) - t

            if r < window+1 or not np.all(zs[t+1:t+r+1] == k):
                continue

            xx = np.nan * np.zeros((2*window+1, D))
            xx[window-l:window+r] = xs[t-l:t+r]
            X.append(xx)

        X_mean = np.nanmean(X, axis=0)
        X_std = np.nanstd(X, axis=0)

        spc = 2.5 * X_std.max()
        # spc = 2
        for d in range(D):
            ax.fill_between(tt,
                            d * spc + X_mean[:,d]-X_std[:,d],
                            d * spc + X_mean[:,d]+X_std[:,d],
                            color=colors[k],
                            alpha=0.25
                            )

            ax.plot(tt, d * spc + X_mean[:,d], color=colors[k], lw=2)
            ax.plot([tt[0], tt[-1]], [d * spc, d*spc], ':k', lw=.5)

        yl = ax.get_ylim()
        ax.plot([0, 0], yl, ':k', lw=0.5)
        ax.set_ylim(reversed(yl))
        ax.set_yticks(np.arange(D) * spc)
        ax.set_yticklabels(np.arange(D) + 1)
        ax.set_ylabel("latent dimension")
        ax.set_xlim(tt[0], tt[-1])
        ax.set_xlabel("time around changepoint")

        ax.set_title("$x$ at entry to state ${}$".format(k+1))
        plt.tight_layout(pad=0.25)

        if results_dir is not None:
            plt.savefig(os.path.join(results_dir, basename + "_{}.pdf".format(k)))


def make_states_3d_movie(z_finals, x_finals, title=None, lim=None,
                         colors=None,
                         filepath=None):

    colors = default_colors if colors is None else colors
    FFMpegWriter = manimation.writers['ffmpeg']
    metadata = dict(title='3d states field')
    writer = FFMpegWriter(fps=15, bitrate=1024, metadata=metadata)

    # overlay = False
    fig = plt.figure(figsize=(4, 4))
    ax = create_axis_at_location(fig, 0, 0, 4, 4, projection="3d")
    # ax = fig.add_subplot(111, projection='3d')

    plot_3d_continuous_states(x_finals, z_finals, colors,
                              ax=ax,
                              lw=1)

    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_zlabel("")
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])

    if lim is not None:
        ax.set_xlim(-lim,lim)
        ax.set_ylim(-lim,lim)
        ax.set_zlim(-lim,lim)

    plt.title(title)

    def update_frame(i):
        # Rotate the xy plane
        ax.view_init(elev=30., azim=i)

        # Plot the trajectories
        #         plot_trajectories(i, lns)

    with writer.saving(fig, filepath, 150):
        for i in tqdm(range(360)):
            update_frame(i)
            writer.grab_frame()



# Make a movie of the rolling predictions
def make_state_predictions_3d_movie(
        z_smooth, x_smooth,
        z_pred, x_pred,
        title=None,
        lim=None,
        colors=None,
        figsize=(4, 4),
        filepath=None):
    # colors = default_colors if colors is None else colors
    FFMpegWriter = manimation.writers['ffmpeg']
    metadata = dict(title='3d states field')
    writer = FFMpegWriter(fps=15, bitrate=1024, metadata=metadata)

    # overlay = False
    fig = plt.figure(figsize=figsize)
    ax = create_axis_at_location(fig, 0, 0, figsize[0], figsize[1], projection="3d")

    # Initialize the plots
    N_pred, T, T_pred, _ = x_pred.shape
    # h_time = ax.text(15, -15, -20, "T=0")

    h_preds = []
    for n in range(N_pred):
        h_preds.append(
            plt.plot(x_pred[n, 0, :, 0],
                     x_pred[n, 0, :, 1],
                     x_pred[n, 0, :, 2],
                     color=colors[z_pred[n, 0, 0]], lw=0.5)[0])

    h_prev = plt.plot(x_smooth[:1, 0], x_smooth[:1, 1], x_smooth[:1, 2], lw=1, color='gray')
    h_curs = plt.plot([x_smooth[0, 0]], [x_smooth[0, 1]], [x_smooth[0, 2]], 'ko')[0]
    h_true = plt.plot(x_smooth[:T_pred, 0],
                      x_smooth[:T_pred, 1],
                      x_smooth[:T_pred, 2],
                      '-k', lw=2)[0]

    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_zlabel("")
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])

    if lim is not None:
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_zlim(-lim, lim)

    ax.set_title(title)

    def update_frame(t):
        # Rotate the xy plane
        # ax.view_init(elev=30., azim=i)

        # Update the trajectories
        # h_time.set_text("T={}".format(t))
        h_prev[0].remove()
        t_prev = max(t - T_pred, 0)
        h_prev[0] = plt.plot(x_smooth[t_prev:t + 1, 0],
                             x_smooth[t_prev:t + 1, 1],
                             x_smooth[t_prev:t + 1, 2],
                             lw=1, color='gray')[0]
        h_curs.set_data(([x_smooth[t, 0]], [x_smooth[t, 1]]))
        h_curs.set_3d_properties([x_smooth[t, 2]])

        h_true.set_data((x_smooth[t:t + T_pred, 0],
                         x_smooth[t:t + T_pred, 1]))
        h_true.set_3d_properties(x_smooth[t:t + T_pred, 2])

        for n, h in enumerate(h_preds):
            h.set_data((x_pred[n, t, :, 0],
                        x_pred[n, t, :, 1]))
            h.set_color(colors[z_pred[n, t, 0]])
            h.set_3d_properties(x_pred[n, t, :, 2])

    with writer.saving(fig, filepath, 150):
        for i in tqdm(range(T)):
            update_frame(i)
            writer.grab_frame()