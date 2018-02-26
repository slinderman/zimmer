import os
import pickle

import numpy as np
np.random.seed(1234)
from scipy.ndimage import gaussian_filter1d


# Plotting stuff
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from hips.plotting.colormaps import gradient_cmap
from hips.plotting.layout import create_axis_at_location
import seaborn as sns
sns.set_style("white")
sns.set_context("talk")
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
from pyhsmm.util.general import relabel_by_usage, rle

# Load worm modeling specific stuff
from zimmer.io import WormData, load_kato_key
import zimmer.plotting as zplt

# LDS Results
lds_dir = os.path.join("results", "2017-11-03-hlds", "run003")
assert os.path.exists(lds_dir)

results_dir = os.path.join("results", "2018-01-19-arhmm", "run001")

# Figure dir
fig_dir = os.path.join("results", "talk")

N_worms = 5
N_clusters = 12


def load_data(include_unnamed=True):
    # Load the data
    worm_datas = [WormData(i, name="worm{}".format(i), version="kato") for i in range(N_worms)]

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
    y_raws = []
    masks = []
    for wd in worm_datas:
        y_indiv = getattr(wd, "dff_diff")
        y_raw_indiv = getattr(wd, "dff_bc_zscored")

        y = np.zeros((wd.T, N_neurons))
        y_raw = np.zeros((wd.T, N_neurons))
        mask = np.zeros((wd.T, N_neurons), dtype=bool)
        indices = wd.find_neuron_indices(neuron_names)
        for n, index in enumerate(indices):
            if index is not None:
                y[:, n] = y_indiv[:, index]
                y_raw[:, n] = y_raw_indiv[:, index]
                mask[:, n] = True

        ys.append(y)
        y_raws.append(y_raw)
        masks.append(mask)

    return ys, y_raws, masks, z_trues, z_key, neuron_names


def plot_raw_data(worm, neuron, slc=(9, 12)):

    fig = plt.figure(figsize=(8, 1))
    fig.patch.set_alpha(0.0)
    ax = fig.add_subplot(111)
    ax.patch.set_alpha(0.0)

    index = np.where(neuron_names == neuron)[0][0]

    T = Ts[worm]
    t = np.arange(T) / 180.0

    # ax.plot(t, np.zeros(T), ':k', lw=2)
    if np.allclose(y_raws[worm][:, index], 0):
        ax.plot(t, np.zeros(T), ':', color='k', lw=4)
    else:
        ax.plot(t, y_raws[worm][:, index], '-', color=colors[3], lw=4)

    ax.spines["left"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)

    plt.xlim(slc)
    # plt.ylabel(neuron, rotation=0)
    plt.yticks([])
    plt.xticks([])
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "yraw_{}_{}.pdf".format(worm, neuron)))


def plot_diff_data(worm, neuron, slc=(9, 12)):

    fig = plt.figure(figsize=(8, 1))
    fig.patch.set_alpha(0.0)
    ax = fig.add_subplot(111)
    ax.patch.set_alpha(0.0)

    index = np.where(neuron_names == neuron)[0][0]

    T = Ts[worm]
    t = np.arange(T) / 180.0

    # ax.plot(t, np.zeros(T), ':k', lw=2)
    # ax.plot(t, np.zeros(T), ':', color='k', lw=4)
    if np.allclose(ys[worm][:, index], 0):
        ax.plot(t, np.zeros(T), ':', color='k', lw=4)
    else:
        ax.plot(t, ys[worm][:, index], '-', color=colors[3], lw=4)

    ax.spines["left"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)

    plt.xlim(slc)
    # plt.ylabel(neuron, rotation=0)
    plt.yticks([])
    plt.xticks([])
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "y_{}_{}.pdf".format(worm, neuron)))


def plot_recon_data(worm, neuron, slc=(9, 12)):
    all_ys = np.vstack(ys)
    all_ms = np.vstack(ms)
    all_ys[~all_ms] = np.nan
    scales = np.nanstd(all_ys, axis=0)

    fig = plt.figure(figsize=(8, 1))
    fig.patch.set_alpha(0.0)
    ax = fig.add_subplot(111)
    ax.patch.set_alpha(0.0)

    index = np.where(neuron_names == neuron)[0][0]

    T = Ts[worm]
    t = np.arange(T) / 180.0

    # ax.plot(t, np.zeros(T), ':k', lw=2)
    # ax.plot(t, np.zeros(T), ':', color='k', lw=4)
    if np.allclose(ys[worm][:, index], 0):
        ax.plot(t, np.zeros(T), ':', color='k', lw=2)
    else:
        ax.plot(t, ys[worm][:, index] / scales[index], '-', color=colors[3], lw=4)
    ax.plot(t, ys_recon[worm][:, index] /scales[index], '-', color='k', lw=4)

    ax.set_ylim(-3, 3)

    ax.spines["left"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)

    plt.xlim(slc)
    # plt.ylabel(neuron, rotation=0)
    plt.yticks([])
    plt.xticks([])
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "y_recon_{}_{}.pdf".format(worm, neuron)))


def plot_latent_states(worm, slc=(9, 12)):
    T = Ts[worm]
    t = np.arange(T) / 180.0

    for dim in range(D_latent):

        fig = plt.figure(figsize=(8, 1))
        fig.patch.set_alpha(0.0)
        ax = fig.add_subplot(111)
        ax.patch.set_alpha(0.0)

        # ax.plot(t, np.zeros(T), ':k', lw=2)
        # ax.plot(t, np.zeros(T), ':', color='k', lw=4)
        ax.plot(t, xs[worm][:, dim], '-', color=colors[0], lw=4)

        ax.spines["left"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_visible(False)

        plt.xlim(slc)
        # plt.ylabel(neuron, rotation=0)
        plt.yticks([])
        plt.xticks([])
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, "x_{}_{}.pdf".format(worm, dim)))


def plot_latent_states_colored(worm, slc=(9, 12)):
    # xx = xs_smooth[worm]
    xx = xs[worm]
    istart= slc[0] * 60 * 3
    istop = slc[1] * 60 * 3
    xslc = xx[istart:istop]

    zslc = z_trues[worm][istart:istop]
    zs, durs = rle(zslc)
    T = zslc.size


    for dim in range(D_latent):

        fig = plt.figure(figsize=(8, 1))
        fig.patch.set_alpha(0.0)
        ax = fig.add_subplot(111)
        ax.patch.set_alpha(0.0)

        # ax.plot(t, np.zeros(T), ':k', lw=2)
        # ax.plot(t, np.zeros(T), ':', color='k', lw=4)
        offset = 0
        for z, dur in zip(zs, durs):
            end = min(offset+dur+1, T)
            ax.plot(np.arange(offset, end), xslc[offset:end, dim], '-', color=colors[z+1], lw=4)
            offset += dur

        ax.spines["left"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_visible(False)

        plt.xlim(0, T)
        # plt.ylabel(neuron, rotation=0)
        plt.yticks([])
        plt.xticks([])
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, "xz_{}_{}.pdf".format(worm, dim)))


def plot_discrete_states(worm, slc=(9, 12)):
    istart = slc[0] * 60 * 3
    istop = slc[1] * 60 * 3
    zslc = z_trues[worm][istart:istop]


    fig = plt.figure(figsize=(8, 1))
    fig.patch.set_alpha(0.0)
    ax = fig.add_subplot(111)
    ax.patch.set_alpha(0.0)

    ax.imshow(zslc[None, :], cmap=gradient_cmap(colors[1:9]), vmin=0, vmax=7, aspect="auto")

    # ax.spines["left"].set_visible(False)
    # ax.spines["top"].set_visible(False)
    # ax.spines["right"].set_visible(False)
    # ax.spines["bottom"].set_visible(False)

    plt.yticks([])
    plt.xticks([])
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "z_{}.pdf".format(worm)))


def plot_empty_3d_axes():
    zplt.plot_3d_continuous_states(np.zeros((1, 3)), np.zeros(1, dtype=int),
                                   colors,
                                   figsize=(1.2, 1.2),
                                   title=None,
                                   filename=None,
                                   lim=3,
                                   lw=.5)

    plt.savefig(os.path.join(fig_dir, "empty_3d_axes_{}.pdf".format(worm + 1)))


def plot_3d_axes(worm, slc=(9,12),):
    xx = xs_smooth[worm]
    istart = slc[0] * 60 * 3
    istop = slc[1] * 60 * 3
    xslc = xx[istart:istop]

    zplt.plot_3d_continuous_states(np.zeros((1, 3)), np.zeros(1, dtype=int),
                                   colors,
                                   figsize=(1.2, 1.2),
                                   title=None,
                                   filename=None,
                                   lim=3,
                                   lw=.5)

    pts = [np.argmax(xslc[:,2]),
           np.argmin(xslc[:,1])]
    markers = ['^', 's']

    print("worm: {} markers {}".format(worm, (np.array(pts) + istart) / 60 / 3))
    for m, pt in zip(markers, pts):
        # mi = int(m * 60 * 3) + istart
        plt.plot([xslc[pt, 0]], [xslc[pt, 1]], [xslc[pt, 2]],
                 marker=m, color='k', markersize=4, mec='k', mew=0)

    plt.savefig(os.path.join(fig_dir, "x_3d_axes_{}.pdf".format(worm + 1)))



def plot_3d_trajectories(worm, slc=(9,12)):

    xx = xs_smooth[worm]
    istart= slc[0] * 60 * 3
    istop = slc[1] * 60 * 3
    xslc = xx[istart:istop]

    zplt.plot_3d_continuous_states(xslc, np.zeros(istop-istart, dtype=int),
                                   colors,
                                   figsize=(1.2, 1.2),
                                   title=None,
                                   filename=None,
                                   lim=3,
                                   lw=.5)

    # markers = [np.argmax(xslc[:, 2]),
    #            np.argmin(xslc[:, 1])]
    # print("worm: {} markers {}".format(worm, (np.array(markers) + istart) / 60 / 3))
    # for j, m in enumerate(markers):
    #     # mi = int(m * 60 * 3) + istart
    #     plt.plot([xslc[m, 0]], [xslc[m, 1]], [xslc[m, 2]],
    #              'o', color=colors[j + 1], markersize=3,
    #              mec='k', mew=1)

    # pts = [np.argmax(xslc[:,2]),
    #        np.argmin(xslc[:,1])]
    # markers = ['^', 's']
    #
    # print("worm: {} markers {}".format(worm, (np.array(pts) + istart) / 60 / 3))
    # for m, pt in zip(markers, pts):
    #     # mi = int(m * 60 * 3) + istart
    #     plt.plot([xslc[pt, 0]], [xslc[pt, 1]], [xslc[pt, 2]],
    #              marker=m, color='k', markersize=4, mec='k', mew=0)

    plt.savefig(os.path.join(fig_dir, "x_3d_{}.pdf".format(worm + 1)))


def plot_3d_pca_trajectories(worm, slc=(9,12)):

    y = ys[worm][:,ms[worm][0]]
    assert y.ndim == 2

    from sklearn.decomposition import PCA
    pca = PCA(n_components=3)
    pcs = pca.fit_transform(y)

    xx = gaussian_filter1d(pcs, 3.0, axis=0)

    istart= slc[0] * 60 * 3
    istop = slc[1] * 60 * 3
    xslc = xx[istart:istop]
    zslc = z_trues[worm][istart:istop]

    zplt.plot_3d_continuous_states(xslc, zslc,
                                   colors,
                                   figsize=(1.2, 1.2),
                                   title=None,
                                   filename=None,
                                   lim=1,
                                   lw=.5)

    plt.savefig(os.path.join(fig_dir, "pca_3d_{}.pdf".format(worm + 1)))


def plot_3d_trajectories_colored(worm, slc=(9,12), rotate=False):

    xx = xs_smooth[worm]

    if rotate:
        from pylds.util import random_rotation
        R = random_rotation(3, theta=np.pi/2.0)
        from scipy.linalg import block_diag
        R = block_diag(R, np.eye(xx.shape[1] - 3))
        xx = xx.dot(R.T)

    istart= slc[0] * 60 * 3
    istop = slc[1] * 60 * 3
    xslc = xx[istart:istop]
    zslc = z_trues[worm][istart:istop]

    zplt.plot_3d_continuous_states(xslc, zslc,
                                   colors[1:],
                                   figsize=(1.2, 1.2),
                                   title=None,
                                   filename=None,
                                   lim=3,
                                   lw=.5)

    # markers = [np.argmax(xslc[:, 2]),
    #            np.argmin(xslc[:, 1])]
    # print("worm: {} markers {}".format(worm, (np.array(markers) + istart) / 60 / 3))
    # for j, m in enumerate(markers):
    #     # mi = int(m * 60 * 3) + istart
    #     plt.plot([xslc[m, 0]], [xslc[m, 1]], [xslc[m, 2]],
    #              'o', color=colors[j + 1], markersize=3,
    #              mec='k', mew=1)

    # pts = [np.argmax(xslc[:,2]),
    #        np.argmin(xslc[:,1])]
    # markers = ['^', 's']
    #
    # print("worm: {} markers {}".format(worm, (np.array(pts) + istart) / 60 / 3))
    # for m, pt in zip(markers, pts):
    #     # mi = int(m * 60 * 3) + istart
    #     plt.plot([xslc[pt, 0]], [xslc[pt, 1]], [xslc[pt, 2]],
    #              marker=m, color='k', markersize=4, mec='k', mew=0)

    plt.savefig(os.path.join(fig_dir, "xz_3dtraj_{}{}.pdf".format("rotated_" if rotate else "", worm + 1)))



def plot_3d_trajectory_pieces(worm, slc=(9,12)):

    xx = xs_smooth[worm]
    istart= slc[0] * 60 * 3
    istop = slc[1] * 60 * 3
    xslc = xx[istart:istop]

    zslc = z_trues[worm][istart:istop]

    zs, durs = rle(zslc)


    for k in range(8):
        ax = None
        offset = 0
        for z, dur in zip(zs, durs):
            if z == k:
                ax = zplt.plot_3d_continuous_states(
                    xslc[offset:offset+dur],
                    k * np.ones(dur, dtype=int),
                    colors[1:],
                    figsize=(1.2, 1.2),
                    title=None,
                    filename=None,
                    lim=3,
                    lw=1,
                    ax=ax)
                assert ax is not None
            offset += dur

            # markers = [np.argmax(xslc[:, 2]),
            #            np.argmin(xslc[:, 1])]
            # print("worm: {} markers {}".format(worm, (np.array(markers) + istart) / 60 / 3))
            # for j, m in enumerate(markers):
            #     # mi = int(m * 60 * 3) + istart
            #     plt.plot([xslc[m, 0]], [xslc[m, 1]], [xslc[m, 2]],
            #              'o', color=colors[j + 1], markersize=3,
            #              mec='k', mew=1)

            # pts = [np.argmax(xslc[:,2]),
            #        np.argmin(xslc[:,1])]
            # markers = ['^', 's']
            #
            # print("worm: {} markers {}".format(worm, (np.array(pts) + istart) / 60 / 3))
            # for m, pt in zip(markers, pts):
            #     # mi = int(m * 60 * 3) + istart
            #     plt.plot([xslc[pt, 0]], [xslc[pt, 1]], [xslc[pt, 2]],
            #              marker=m, color='k', markersize=4, mec='k', mew=0)

            plt.savefig(os.path.join(fig_dir, "xz_3d_{}_{}.pdf".format(worm + 1, k)))


def plot_trans_matrix():
    P = np.eye(8) + np.random.gamma(1, .2, size=(8, 8))
    P /= P.sum(axis=1, keepdims=True)
    zplt.plot_transition_matrix(P, colors, cmap, results_dir=fig_dir)

    for k in range(8):
        P = np.eye(8) + np.random.gamma(1, .2, size=(8, 8))
        P[:, k] += 1
        P /= P.sum(axis=1, keepdims=True)
        zplt.plot_transition_matrix(P, colors, cmap, results_dir=fig_dir, filename="trans_matrix_{}.pdf".format(k))


def make_latent_states_movie(x, z, filename, slc=(9, 12), fps=60):
    istart = slc[0] * 60 * 3
    istop = slc[1] * 60 * 3
    x = x[istart:istop]

    T = x.shape[0]
    lw = 0.75
    lim = 3


    fig = plt.figure(figsize=(1.2, 1.2))
    ax = fig.add_subplot(111, projection="3d")

    point = ax.plot([x[0,0]], [x[0, 1]], [x[0,2]], 'ko', markersize=3)[0]
    point.set_zorder(1000)
    paths = [x[:1]]
    path_handles = [ax.plot([x[0,0]], [x[0, 1]], [x[0,2]], '-', color=colors[z[0]], lw=lw)[0]]

    ax.set_xlabel("dim 1", labelpad=-18, fontsize=6)
    ax.set_ylabel("dim 2", labelpad=-18, fontsize=6)
    ax.set_zlabel("dim 3", labelpad=-18, fontsize=6)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    if lim is not None:
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_zlim(-lim, lim)

    plt.tight_layout(pad=0.1)

    def _draw(t):
        assert t > 0
        # ax1.cla()
        # plot_3d_continuous_states(x[:t], z[:t],
        #                           colors=colors,
        #                           ax=ax1,
        #                           lim=3,
        #                           lw=.5,
        #                           inds=(0, 1, 2))

        # Update point
        point.set_data(x[t:t+1, 0], x[t:t+1, 1])
        point.set_3d_properties(x[t:t+1, 2])

        # Update paths
        if z[t] == z[t-1]:
            paths[-1] = np.row_stack((paths[-1], x[t:t+1]))
            path_handles[-1].set_data(paths[-1][:,0], paths[-1][:, 1])
            path_handles[-1].set_3d_properties(paths[-1][:, 2])
        else:
            paths.append(x[t-1:t+1])
            path_handles.append(ax.plot(paths[-1][:, 0],
                                        paths[-1][:, 1],
                                        paths[-1][:, 2],
                                        '-',
                                        color=colors[z[t]],
                                        lw=lw)[0])

    from tqdm import tqdm
    import matplotlib.animation as manimation
    FFMpegWriter = manimation.writers['ffmpeg']
    metadata = dict(title='probability vs space')
    writer = FFMpegWriter(fps=fps, bitrate=1024, metadata=metadata)
    with writer.saving(fig, os.path.join(fig_dir, filename), 300):
        # for t in tqdm(range(x.shape[0])):
        for t in tqdm(range(1, T)):
            _draw(t)
            writer.grab_frame()


if __name__ == "__main__":
    ys, y_raws, ms, z_trues, z_true_key, neuron_names = load_data(include_unnamed=False)
    D_obs = ys[0].shape[1]
    Ts = [y.shape[0] for y in ys]

    # Load the continuous states found with the LDS
    with open(os.path.join(lds_dir, "lds_data.pkl"), "rb") as f:
        lds_results = pickle.load(f)

    xtrains = lds_results["xtrains"]
    xtests = lds_results["xtests"]
    xs = [np.concatenate((xtr, xte), axis=0) for xtr, xte in zip(xtrains, xtests)]
    ytrains = lds_results["ytrains"]
    ytests = lds_results["ytests"]
    mtrains = lds_results["mtrains"]
    mtests = lds_results["mtests"]
    z_true_trains = lds_results["z_true_trains"]
    z_true_tests = lds_results["z_true_tests"]
    z_trues = [np.concatenate((ztr, zte)) for ztr, zte in zip(z_true_trains, z_true_tests)]
    z_key = lds_results["z_key"]
    best_model = lds_results["best_model"]
    D_latent = lds_results["D_latent"]
    C = lds_results['best_model'].C[:, lds_results['perm']]
    d = lds_results['best_model'].D[:, 0]
    perm = lds_results["perm"]
    N_clusters = lds_results["N_clusters"]
    neuron_clusters = lds_results["neuron_clusters"]
    neuron_perm = lds_results["neuron_perm"]

    # Smooth the latent trajectories for visualization
    xs_smooth = [gaussian_filter1d(x, 1.0, axis=0) for x in xs]

    # Reconstruct the data
    ys_recon = [x.dot(C.T) + d for x in xs]

    for worm in range(N_worms):
        # for neuron in neuron_names:
        #     # plot_raw_data(worm, neuron)
        #     # plot_diff_data(worm, neuron)
        #     plot_recon_data(worm, neuron)
        #     plt.close("all")
        #
        plot_latent_states(worm)
        # plot_3d_trajectories(worm)
        # plot_3d_trajectories_colored(worm)
        # plot_3d_trajectories_colored(worm, rotate=True)
        # plot_3d_pca_trajectories(worm)
        # plot_3d_axes(worm)
        # plot_discrete_states(worm)
        pass

    make_latent_states_movie(xs_smooth[0],
                             np.zeros(xs_smooth[0].shape[0], dtype=int),
                             "x_3d_1.mp4")

    # plot_3d_trajectory_pieces(0)
    # plot_latent_states_colored(0)

    # plot_empty_3d_axes()
    # plot_trans_matrix()

    plt.close("all")
