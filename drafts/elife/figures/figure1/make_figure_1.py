import numpy as np
np.random.seed(0)

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import seaborn as sns
sns.set_style("white")
sns.set_context("paper")

color_names = ["windows blue",
               "red",
               "amber",
               "faded green",
               "dusty purple",
               "orange"]

colors = sns.xkcd_palette(color_names)

W = 3    # number of worm panels
N = 10   # number of neurons
T = 50   # number of time bins
D = 2    # dimension of latent states
K = 3    # number of latent states


def make_dynamics_library_2D():
    # First state is return to zero
    assert D == 2
    As = np.zeros((K, D, D))
    bs = np.zeros((K, D))

    # Remaining states are rotations in a plane
    ths = np.pi / 8 + np.pi / 8 * np.random.rand(K)
    # ths *= (-1 + 2 * (np.arange(K) % 2 == 0))
    # ths = np.pi / 8 * np.ones(K)

    for k in range(K):
        As[k] = np.array([
            [np.cos(ths[k]), -np.sin(ths[k])],
            [np.sin(ths[k]),  np.cos(ths[k])]
        ])

        phase = 2 * np.pi / K * k
        fp = np.array(
            [np.cos(phase),
             np.sin(phase)])
        bs[k] = (np.eye(D) - As[k]).dot(fp)

    return As, bs


def plot_vector_field_2d(k, As, bs,
                         worm=0,
                         ax=None,
                         lims=(-3, 3),
                         n_pts=7):
    X, Y = np.meshgrid(np.linspace(lims[0], lims[1], n_pts),
                     np.linspace(lims[0], lims[1], n_pts))
    x = np.column_stack((X.ravel(), Y.ravel()))

    Ai, bi = As[k], bs[k]
    dxdt = x.dot(Ai.T) + bi - x

    # Find the probability at each of the points
    W = np.array([np.linalg.solve(np.eye(D) - A, b) for A, b in zip(As, bs)])
    p = np.exp(x.dot(W.T))
    p /= p.sum(axis=1, keepdims=True)

    # Use p to set alpha
    C = np.zeros((x.shape[0], 4))
    C[:,:3] = colors[k]
    C[:, 3] = 0. + 1 * p[:,k]

    # Create axis if not given
    if ax is None:
        fig = plt.figure(figsize=(.5, .5))
        ax = fig.add_subplot(111)

    ax.quiver(x[:,0], x[:,1],
              dxdt[:,0], dxdt[:, 1],
              color=C,
              # headlength=3,
              minlength=.1,
              scale=10,
              width=0.05
              )

    ax.plot(W[k,0], W[k,1], 'o', markersize=2, color=colors[k])

    ax.plot(lims, [0, 0], ':k', lw=0.5)
    ax.plot([0, 0], lims, ':k', lw=0.5)

    # ax.set_xlabel('$x_1$', labelpad=-5)
    # ax.set_ylabel('$x_2$', labelpad=-5)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_xlim(lims)
    ax.set_ylim(lims)

    plt.tight_layout(pad=0.1)
    plt.savefig("fig1_worm{}_dyn{}.pdf".format(worm + 1, k + 1))


def plot_neural_tuning(C, lims=(-3, 3), ):
    fig = plt.figure(figsize=(.5, .5))
    ax = fig.add_subplot(111)

    for n in range(N):
        # plt.arrow(0, 0, C[n, 0], C[n, 1],
        #          color='k',
        #          head_width=0.5,)
        plt.plot([0, C[n, 0]], [0, C[n, 1]],
                  color='k',
                  lw=1,
                  alpha=0.75)

    ax.plot(lims, [0, 0], ':k', lw=0.5)
    ax.plot([0, 0], lims, ':k', lw=0.5)

    # ax.set_xlabel('$x_1$', labelpad=-5)
    # ax.set_ylabel('$x_2$', labelpad=-5)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_xlim(lims)
    ax.set_ylim(lims)

    plt.tight_layout(pad=0.1)
    plt.savefig("fig1_tuning.pdf")


def simulate_latent_states(As, bs):
    # find the fixed points
    W = np.array([np.linalg.solve(np.eye(D) - A, b) for A, b in zip(As, bs)])

    # continuous states
    x = np.zeros((T, D))
    x[0] = np.random.randn(D)

    # discrete states
    z = np.zeros(T, dtype=int)
    z[0] = np.argmax(W.dot(x[0]))

    # Set some stickiness for the current state
    kappa = 1
    onehot = lambda k: np.arange(K) == k

    for t in range(1, T):
        # Compute probability based on location with softmax and stickiness
        p = np.exp(W.dot(x[t-1]) + kappa * onehot(z[t-1]))
        p /= p.sum()
        P = np.repeat(p[None, :], K, axis=0)

        # Sample z given previous x location
        z[t] = np.random.choice(K, p=P[z[t-1]])

        # Simulate x given z
        x[t] = As[z[t-1]].dot(x[t-1]) + bs[z[t-1]] + 0.**2 * np.random.randn(D)

    return x, z


def plot_discrete_latent_states():
    # Make panels for each worm
    for w in range(W):
        fig = plt.figure(figsize=(1.5, .125))
        ax = fig.add_subplot(111)

        z = zs[w]
        cps = np.concatenate(([0], np.where(np.diff(z) != 0)[0] + 1, [T - 1]))
        for i,j in zip(cps[:-1], cps[1:]):
            # ax.plot(np.arange(i, j+1), d + xnorm[i:j+1, d], color=colors[z[i]])
            ax.fill_between((i, j+1), np.zeros(2), np.ones(2), color=colors[z[i]])

        ax.set_ylim(0, 1)
        ax.set_yticks([])
        ax.set_xlim(0, T)
        ax.set_xticks([])

        plt.tight_layout(pad=0.1)
        plt.savefig("fig1_z{}.pdf".format(w + 1))

    plt.close("all")


def plot_continuous_latent_states():
    # Make panels for each worm
    for w in range(W):
        fig = plt.figure(figsize=(1.5, .5))
        ax = fig.add_subplot(111)

        x, z = xs[w], zs[w]
        xnorm = x / (2.1 * np.max(abs(x)))
        cps = np.concatenate(([0], np.where(np.diff(z) != 0)[0] + 1, [T - 1]))

        for d in range(D):
            ax.plot(np.arange(T), d + np.zeros(T), '-k', lw=0.5)
            for i,j in zip(cps[:-1], cps[1:]):
                ax.plot(np.arange(i, j+1), d + xnorm[i:j+1, d], color=colors[z[i]])

        ax.set_ylim(D, -1)
        ax.set_yticks([])
        ax.set_xlim(-1, T)
        ax.set_xticks([])

        plt.tight_layout(pad=0.1)
        plt.savefig("fig1_x{}.pdf".format(w + 1))

    plt.close("all")


def plot_neural_activity():
    # Make panels for each worm
    for w in range(W):
        fig = plt.figure(figsize=(1.5, 2))
        ax = fig.add_subplot(111)

        mask = np.zeros(N, dtype=bool)
        mask[np.random.choice(N, size=6, replace=False)] = True

        for n in range(N):
            ax.plot(np.arange(T), n + np.zeros(T), '-k', lw=0.5)
            if mask[n]:
                ax.plot(np.arange(T), n + ys[w][:,n], color=colors[3])

        ax.set_ylim(N, -1)
        ax.set_yticks([])
        ax.set_xlim(-1, T)
        ax.set_xticks([])

        plt.tight_layout(pad=0.1)
        plt.savefig("fig1_y{}.pdf".format(w + 1))

    plt.close("all")


if __name__ == "__main__":
    # Simulate data
    As, bs = make_dynamics_library_2D()
    C = np.random.randn(N, D)

    # Perturb the states for each worm
    eta = 0.1
    Ahats = [As + eta * np.random.randn(*As.shape) for _ in range(W)]
    bhats = [bs + eta * np.random.randn(*bs.shape) for _ in range(W)]

    # Simulate data for each worm with its specific dynamics
    xzs = [simulate_latent_states(Ahat, bhat) for Ahat, bhat in zip(Ahats, bhats)]
    xs, zs = list(zip(*xzs))

    # simulate observations
    sigma = 0.1
    ys = [x.dot(C.T) + sigma * np.random.randn(T, N) for x in xs]
    for y in ys:
        y /= (2.1 * np.max(abs(y)))

    # Plot the canonical dynamics
    for k in range(K):
        plot_vector_field_2d(k, As, bs, worm=-1)
    plt.close("all")

    # Plot the individual worm dynamics
    for k in range(K):
        for w in range(W):
            plot_vector_field_2d(k, Ahats[w], bhats[w], worm=w)
    plt.close("all")

    # Plot the latent states of each worm
    # plot_neural_tuning(C)
    # plot_discrete_latent_states()
    # plot_continuous_latent_states()
    # plot_neural_activity()

