"""
More sophisticated emission models than just simple regressions.
"""
import numpy as np

from pybasicbayes.distributions import DiagonalRegression, Regression
from pybasicbayes.util.stats import sample_gaussian, sample_invgamma

class HierarchicalDiagonalRegression(DiagonalRegression):
    """
    y_i^g ~ N(A \dot x_i, eta^g)

    where
    x_i, y_i:   covariates and observations
    A:          shared regression weights
    eta^g       group specific variance
    """
    def __init__(self, D_out, D_in, N_groups,
                 mu_0=None, Sigma_0=None, alpha_0=3.0, beta_0=2.0,
                 A=None, sigmasq=None, niter=1):

        # Initialize the scaling factors for each group
        self._D_out = D_out
        self._D_in = D_in
        self.N_groups = N_groups

        if A is not None:
            assert A.shape == (D_out, D_in)
            self.A = A
        else:
            self.A = np.zeros((D_out, D_in))

        if sigmasq is not None:
            assert sigmasq.shape == (N_groups, D_out) and np.all(sigmasq > 0)
            self.sigmasq_flat = sigmasq
        else:
            self.sigmasq_flat = np.ones((N_groups, D_out))

        # Affine support must be done manually
        self.affine = False

        mu_0 = np.zeros(D_in) if mu_0 is None else mu_0
        Sigma_0 = np.eye(D_in) if Sigma_0 is None else Sigma_0
        assert mu_0.shape == (D_in,)
        assert Sigma_0.shape == (D_in, D_in)
        self.h_0 = np.linalg.solve(Sigma_0, mu_0)
        self.J_0 = np.linalg.inv(Sigma_0)
        self.alpha_0 = alpha_0
        self.beta_0 = beta_0

        self.niter = niter

        if A is None or sigmasq is None:
            self.resample(data=[], groups=[], mask=[])  # initialize from prior

    @property
    def sigma(self):
        return np.array([np.diag(s) for s in self.sigmasq_flat])

    def log_likelihood(self, xy, group=None, mask=None):
        assert group is not None

        if isinstance(xy, tuple):
            x,y = xy
        else:
            x,y = xy[:,:self.D_in], xy[:,self.D_in:]
            assert y.shape[1] == self.D_out

        if mask is None:
            mask = np.ones_like(y)
        else:
            assert mask.shape == y.shape

        sqerr = -0.5 * (y-x.dot(self.A.T))**2 * mask
        ll = np.sum(sqerr / self.sigmasq_flat[group], axis=1)

        # Add normalizer
        ll += np.sum(-0.5*np.log(2*np.pi*self.sigmasq_flat[group]) * mask, axis=1)

        return ll

    ### Gibbs
    def resample(self, data, groups=None, stats=None, mask=None, niter=None):
        """
        data is a list of tuples [(x1, y1), (x2, y2), ..., (xN,yN)]
        groups is a list of thelength specifying which group the
        corresponding dataset get
        """
        assert groups is not None, "HierarchicalRegression requires each data to have a group"
        assert isinstance(data, list), "HierarchicalRegression requires a list of data matrices"

        if mask is None:
            mask = [None] * len(data)

        niter = niter if niter else self.niter
        for itr in range(niter):

            # Resample A using shared statistics
            all_group_stats = []
            for group in range(self.N_groups):
                group_data = [d for d,g in zip(data, groups) if g == group]
                group_mask = [m for m,g in zip(mask, groups) if g == group]
                all_group_stats.append(self._get_statistics(group_data,
                                                            mask=group_mask,
                                                            D_out=self.D_out,
                                                            D_in=self.D_in))

            # Resample the shared emission matrix
            self._resample_A(all_group_stats)

            # Resample the group specific variance
            for group in range(self.N_groups):
                self._resample_sigma(all_group_stats[group], group)


    def _resample_A(self, group_stats):
        assert len(group_stats) == self.N_groups

        # Sample each row of W
        for d in range(self.D_out):
            Jd = self.J_0.copy()
            hd = self.h_0.copy()

            for group, stats in enumerate(group_stats):
                _, yxT, xxT, _ = stats
                Jd += xxT[d] / self.sigmasq_flat[group, d]
                hd += yxT[d] / self.sigmasq_flat[group, d]

            self.A[d] = sample_gaussian(J=Jd, h=hd)

    def _resample_sigma(self, stats, group):
        ysq, yxT, xxT, n = stats
        AAT = np.array([np.outer(a, a) for a in self.A])

        alpha = self.alpha_0 + n / 2.0

        beta = self.beta_0
        beta += 0.5 * ysq
        beta += -1.0 * np.sum(yxT * self.A, axis=1)
        beta += 0.5 * np.sum(AAT * xxT, axis=(1, 2))

        self.sigmasq_flat[group] = np.reshape(sample_invgamma(alpha, beta), (self.D_out,))
        assert np.all(self.sigmasq_flat > 0)

    ### EM
    def max_likelihood(self, stats, groups):
        stats = [self._stats_ensure_array(stat) for stat in stats]
        self._max_likelihood_A(stats, groups)
        self._max_likelihood_sigma(stats, groups)

    def _max_likelihood_A(self, stats, groups):
        # Update A | sigma
        for d in range(self.D_out):
            Jd = self.J_0.copy()
            hd = self.h_0.copy()

            for group, stat in zip(groups, stats):
                _, yxT, xxT, _ = stat
                Jd += xxT[d] / self.sigmasq_flat[group, d]
                hd += yxT[d] / self.sigmasq_flat[group, d]

            self.A[d] = np.linalg.solve(Jd, hd)

    def _max_likelihood_sigma(self, stats, groups):
        # Update sigmasq | A for each group
        AAT = np.array([np.outer(ad, ad) for ad in self.A])
        for g in range(self.N_groups):
            alpha, beta = self.alpha_0, self.beta_0

            # Sum statistics for this group
            for group, stat in zip(groups, stats):
                if group == g:
                    ysq, yxT, xxT, n = stat
                    alpha += n / 2.0
                    beta += 0.5 * ysq
                    beta += -1.0 * np.sum(yxT * self.A, axis=1)
                    beta += 0.5 * np.sum(AAT * xxT, axis=(1, 2))

            self.sigmasq_flat[g] = beta / (alpha + 1.0)
            assert np.all(self.sigmasq_flat[g] >= 0)

    ### Prediction and generation
    def predict(self, x, group=0):
        A, sigma = self.A, self.sigma[group]
        y = x.dot(A.T)
        return y

    def rvs(self,x=None,size=1,return_xy=True, group=0):
        x = np.random.normal(size=(size, self.D_in)) if x is None else x
        y = self.predict(x, group=group)
        y += np.random.normal(size=(x.shape[0], self.D_out)) \
            .dot(np.linalg.cholesky(self.sigma[group]).T)

        return np.hstack((x,y)) if return_xy else y

