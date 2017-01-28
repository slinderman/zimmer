"""
More sophisticated emission models than just simple regressions.
"""
import numpy as np

from pybasicbayes.distributions import DiagonalRegression
from pybasicbayes.util.stats import sample_gaussian, sample_invgamma

class HierarchicalDiagonalRegression(DiagonalRegression):
    """
    y_i^g ~ N(a^g (c \dot x_i), eta^g)

    where
    x_i, y_i:   covariates and observations
    c:          shared regression weights
    a^g:        group specific scaling
    eta^g       group specific variance
    """
    def __init__(self, D_out, D_in, N_groups,
                 mu_0=None, Sigma_0=None, alpha_0=3.0, beta_0=2.0,
                 A=None, sigmasq=None, niter=1):

        # Initialize the scaling factors for each group
        self._D_out = D_out
        self._D_in = D_in
        self.N_groups = N_groups
        self._A = np.zeros((D_out, D_in))
        self.scale = np.ones((N_groups, D_out))
        self.sigmasq_flat = np.ones((N_groups, D_out))


        self.affine = False  # We do not yet support affine

        mu_0 = np.zeros(D_in) if mu_0 is None else mu_0
        Sigma_0 = np.eye(D_in) if Sigma_0 is None else Sigma_0
        assert mu_0.shape == (D_in,)
        assert Sigma_0.shape == (D_in, D_in)
        self.h_0 = np.linalg.solve(Sigma_0, mu_0)
        self.J_0 = np.linalg.inv(Sigma_0)
        self.alpha_0 = alpha_0
        self.beta_0 = beta_0

        self.niter = niter
        self.resample(data=[], groups=[], mask=[])  # initialize from prior

    @property
    def A(self):
        return self.scale[:, :, None] * self._A[None, :, :]

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

        sqerr = -0.5 * (y-x.dot(self.A[group].T))**2 * mask
        ll = np.sum(sqerr / self.sigmasq_flat[group], axis=1)

        # Add normalizer
        ll += np.sum(-0.5*np.log(2*np.pi*self.sigmasq_flat[group]) * mask, axis=1)

        return ll

    def _get_scale_statistics(self, data, mask=None):
        D_out = self.D_out

        if data is None:
            return (np.zeros((D_out,)),
                    np.zeros((D_out,)),
                    np.zeros((D_out,)),
                    np.zeros((D_out,)))

        # Make sure data is a list
        if not isinstance(data, list):
            datas = [data]
        else:
            datas = data

        # Make sure mask is also a list if given
        if mask is not None:
            if not isinstance(mask, list):
                masks = [mask]
            else:
                masks = mask
        else:
            masks = [None] * len(datas)

        # Sum sufficient statistics from each dataset
        ysq = np.zeros(D_out)
        yxT = np.zeros(D_out)
        xxT = np.zeros(D_out)
        n = np.zeros(D_out)

        for data, mask in zip(datas, masks):
            x, y = data
            assert x.shape[1] == D_out
            assert y.shape[1] == D_out

            if mask is None:
                mask = np.ones_like(y, dtype=bool)

            ysq += np.sum(y**2 * mask, axis=0)
            yxT += np.sum(y * x * mask, axis=0)
            xxT += np.sum(x**2 * mask, axis=0)
            n += np.sum(mask, axis=0)

        return ysq, yxT, xxT, n


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

            # Resample the group specific scale
            # accounting for the shared emission matrix
            for group in range(self.N_groups):
                group_data = [d for d, g in zip(data, groups) if g == group]
                group_data = [(x.dot(self._A.T), y) for (x,y) in group_data]
                group_mask = [m for m, g in zip(mask, groups) if g == group]

                this_group_stats = self._get_scale_statistics(group_data,
                                                              mask=group_mask)

                # Resample the group scale
                self._resample_scale(this_group_stats, group)

            # Resample the group specific variance
            # for group in range(self.N_groups):
            #     group_data = [d for d, g in zip(data, groups) if g == group]
            #     this_group_stats = self._get_statistics(group_data, D_out=self.D_out, D_in=self.D_in)
            #
            #     # Resample the group variance
            #     self._resample_sigma(this_group_stats, group)

    def _resample_A(self, group_stats):
        assert len(group_stats) == self.N_groups

        # Sample each row of W
        for d in range(self.D_out):
            Jd = self.J_0.copy()
            hd = self.h_0.copy()

            for group, stats in enumerate(group_stats):
                _, yxT, xxT, _ = stats
                Jd += self.scale[group, d]**2 * xxT[d] / self.sigmasq_flat[group, d]
                hd += self.scale[group, d]    * yxT[d] / self.sigmasq_flat[group, d]

            self._A[d] = sample_gaussian(J=Jd, h=hd)

    def _resample_scale(self, stats, group):

        _, yxT, xxT, _ = stats

        # Sample each row of W
        for d in range(self.D_out):
            # Get sufficient statistics from the data
            Jd = np.ones((1,1)) + xxT[d] / self.sigmasq_flat[group, d]
            hd = np.zeros(1,) + yxT[d] / self.sigmasq_flat[group, d]
            assert Jd.size == 1
            assert hd.size == 1
            self.scale[group, d] = sample_gaussian(J=Jd, h=hd)


    def _resample_sigma(self, stats, group):
        ysq, yxT, xxT, n = stats
        A = self.A[group]
        AAT = np.array([np.outer(a, a) for a in A])

        alpha = self.alpha_0 + n / 2.0

        beta = self.beta_0
        beta += 0.5 * ysq
        beta += -1.0 * np.sum(yxT * A, axis=1)
        beta += 0.5 * np.sum(AAT * xxT, axis=(1, 2))

        self.sigmasq_flat[group] = np.reshape(sample_invgamma(alpha, beta), (self.D_out,))

    ### TODO: Implement meanfield stuff
    def meanfieldupdate(self, data=None, weights=None, stats=None, mask=None):
        raise NotImplementedError

    def max_likelihood(self, data, weights=None, stats=None, mask=None):
        raise NotImplementedError

    @property
    def mf_expectations(self):
        raise NotImplementedError

    def meanfield_expectedstats(self):
        raise NotImplementedError


class HierarchicalDiagonalRegressionFixedScale(HierarchicalDiagonalRegression):
    """
    Same as above but with scale fixed to one
    """
    def __init__(self, D_out, D_in, N_groups, **kwargs):

        super(HierarchicalDiagonalRegressionFixedScale, self).\
            __init__(D_out, D_in, N_groups, **kwargs)

        # Fix the scale to one
        self.scale = np.ones((N_groups, D_out))

    def _resample_scale(self, stats, group):
        pass

class HierarchicalDiagonalRegressionTruncatedScale(HierarchicalDiagonalRegression):
    """
    Same as above but with scale drawn from a truncated
    normal distribution
    """
    def __init__(self, D_out, D_in, N_groups, smin=0.5, smax=2.0, **kwargs):

        # Set the scale limits
        self.smin = smin
        self.smax = smax

        super(HierarchicalDiagonalRegressionTruncatedScale, self).\
            __init__(D_out, D_in, N_groups, **kwargs)


    def _resample_scale(self, stats, group):
        from pybasicbayes.util.stats import sample_truncated_gaussian
        _, yxT, xxT, _ = stats

        # Sample each row of W
        for d in range(self.D_out):
            # Get sufficient statistics from the data
            Jd = np.ones((1, 1)) + xxT[d] / self.sigmasq_flat[group, d]
            hd = np.ones(1, )    + yxT[d] / self.sigmasq_flat[group, d]
            assert Jd.size == 1
            assert hd.size == 1

            sigma = np.sqrt(1. / Jd)
            mu = hd / Jd
            self.scale[group, d] = sample_truncated_gaussian(mu, sigma, self.smin, self.smax)

