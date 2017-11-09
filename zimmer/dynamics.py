"""
Hierarchical model for dynamics distributions
"""
import numpy as np

from pybasicbayes.distributions import RegressionNonconj, AutoRegression, RobustAutoRegression, GaussianFixedCov
from pybasicbayes.distributions.regression import _ARMixin
from pybasicbayes.util.stats import sample_gaussian


class NonConjAutoRegression(_ARMixin, RegressionNonconj):
    pass


class _HierarchicalAutoRegressionMixin(object):

    def __init__(self, N_groups, M_0, Sigma_0, nu_0, Q_0, etasq, affine=True, **kwargs):
        self.N_groups, self.M_0, self.Sigma_0, self.nu_0, self.Q_0, self.etasq, self.affine = \
            N_groups, M_0, Sigma_0, nu_0, Q_0, etasq, affine

        assert isinstance(N_groups, int) and N_groups >= 1

        assert M_0.ndim == 2
        self.D_out, self.D_in = M_0.shape

        if np.isscalar(Sigma_0):
            self.Sigma_0 = Sigma_0 * np.eye(self.D_out * self.D_in)
        assert Q_0.shape == (self.D_out, self.D_out)
        assert np.isscalar(etasq) and etasq > 0

        self.A_0 = M_0.copy()
        self.regressions = []
        self._initialize_regressions(**kwargs)

    def _initialize_regressions(self, **kwargs):
        raise NotImplementedError

    def _resample_A(self, groups):
        raise NotImplementedError

    @property
    def D(self):
        return self.regressions[0].D

    @property
    def nlags(self):
        return self.regressions[0].nlags

    @property
    def As(self):
        return np.array([r.A for r in self.regressions])

    @property
    def sigmas(self):
        return np.array([r.sigma for r in self.regressions])


    def predict(self, x, group=None):
        A = self.A_0 if group is None else self.As[group]

        if self.affine:
            A, b = A[:, :-1], A[:, -1]
            y = x.dot(A.T) + b.T
        else:
            y = x.dot(A.T)

        return y


    def rvs(self, x=None, size=1, return_xy=True, group=None):
        if group is None:
            A, sigma = self.A_0, self.Q_0
        else:
            A, sigma = self.As[group], self.sigmas[group]

        if self.affine:
            A, b = A[:,:-1], A[:,-1]

        x = np.random.normal(size=(size,A.shape[1])) if x is None else x
        y = self.predict(x)
        y += np.random.normal(size=(x.shape[0], self.D_out)) \
            .dot(np.linalg.cholesky(sigma).T)

        return np.hstack((x,y)) if return_xy else y

    def resample(self, data=(), groups=()):
        """
        Resample the group-specific parameters
        :param data:
        :param groups:
        :return:
        """
        if isinstance(data, np.ndarray) and np.isscalar(groups):
            data = [data]
            groups = [groups]

        # Resample the individual-level regression parameters
        for group in range(self.N_groups):
            group_data = [d for d, g in zip(data, groups) if g == group]
            self.regressions[group].resample(group_data)

        self._resample_A(groups)


class HierarchicalNonConjAutoRegression(_HierarchicalAutoRegressionMixin):
    def _initialize_regressions(self, **kwargs):
        self.regressions = \
            [NonConjAutoRegression(self.M_0, self.etasq * np.eye(self.D_out * self.D_in),
                                   self.nu_0, self.Q_0, affine=self.affine, **kwargs)
             for _ in range(self.N_groups)]

    def _resample_A(self, groups):
        # Resample global mean
        used = np.unique(groups)
        As = [self.regressions[u].A.ravel() for u in used]
        gaussian = GaussianFixedCov(
            sigma=self.etasq * np.eye(self.D_out * self.D_in),
            mu_0=self.M_0.ravel(),
            sigma_0=self.Sigma_0)
        gaussian.resample(np.array(As))
        self.A_0 = gaussian.mu.reshape((self.D_out, self.D_in))

        # Update the regression objects with the new global mean
        h_0 = self.A_0 / self.etasq
        J_0 = np.eye(self.D_out * self.D_in) / self.etasq
        for r in self.regressions:
            r.h_0, r.J_0 = h_0, J_0


class HierarchicalAutoRegression(_HierarchicalAutoRegressionMixin):
    def __init__(self, N_groups, M_0, Sigma_0, nu_0, Q_0, etasq, affine=True, **kwargs):
        self.K_0 = etasq * np.eye(M_0.shape[1])
        super(HierarchicalAutoRegression, self).__init__(
            N_groups, M_0, Sigma_0, nu_0, Q_0, etasq, affine=affine, **kwargs)

    def _initialize_regressions(self, **kwargs):
        # Set small K so that individual A_i's are close to the global mean
        self.regressions = \
            [AutoRegression(M_0=self.M_0, K_0=self.K_0,
                            nu_0=self.nu_0, S_0=self.Q_0,
                            affine=self.affine, **kwargs)
             for _ in range(self.N_groups)]

    def _resample_A(self, groups):
        # Resample global mean
        used = np.unique(groups)

        # Compute sufficient statistics for A
        J = np.linalg.inv(self.Sigma_0)
        h = J.dot(self.M_0.ravel())

        for u in used:
            A_i, U, V = self.regressions[u].A, self.regressions[u].sigma, self.K_0

            # vec(Au) ~ N(vec(A), kron(U, V)
            # vectorize rasters across rows and down (i.e. in C order)
            # so the first p entries of vec(A) are the first row of A
            #
            # U (n x n) is the covariance of the rows
            # V (p x p) is the covariance of the columns
            #
            # kron(U, V) = [[u11 * V,  u12 * V, ..., u1p * V],
            #                ...
            #               [un1 * V,  un2 * V, ..., unp * V]]
            Sigma_u = np.kron(U, V)
            J_i = np.linalg.inv(Sigma_u)
            h_i = J_i.dot(A_i.ravel())
            J += J_i
            h += h_i

        self.A_0 = sample_gaussian(J=J, h=h).reshape((self.D_out, self.D_in))

        # Update the regression objects with the new global mean
        for r in self.regressions:
            r.natural_hypparam = \
                r._standard_to_natural(
                    self.nu_0, self.Q_0, self.A_0, self.K_0)


class HierarchicalRobustAutoRegression(_HierarchicalAutoRegressionMixin):
    def __init__(self, N_groups, M_0, Sigma_0, nu_0, Q_0, etasq, affine=True, **kwargs):
        self.K_0 = etasq * np.eye(M_0.shape[1])
        super(HierarchicalRobustAutoRegression, self).__init__(
            N_groups, M_0, Sigma_0, nu_0, Q_0, etasq, affine=affine, **kwargs)


    def _initialize_regressions(self, **kwargs):
        # Set small K so that individual A_i's are close to the global mean
        self.regressions = \
            [RobustAutoRegression(M_0=self.M_0, K_0=self.K_0,
                                  nu_0=self.nu_0, S_0=self.Q_0,
                                  affine=self.affine, **kwargs)
             for _ in range(self.N_groups)]

    def _resample_A(self, groups):
        # TODO: we should resample the precisions to get the conditional Gaussian
        # TODO: model for A_i | A.  If we don't do that, it's like assuming tau = 1.
        # Resample global mean
        used = np.unique(groups)

        # Compute sufficient statistics for A
        J = np.linalg.inv(self.Sigma_0)
        h = J.dot(self.M_0.ravel())

        for u in used:
            A_i, U, V = self.regressions[u].A, self.regressions[u].sigma, self.K_0

            # vec(Au) ~ N(vec(A), kron(U, V)
            # vectorize rasters across rows and down (i.e. in C order)
            # so the first p entries of vec(A) are the first row of A
            #
            # U (n x n) is the covariance of the rows
            # V (p x p) is the covariance of the columns
            #
            # kron(U, V) = [[u11 * V,  u12 * V, ..., u1p * V],
            #                ...
            #               [un1 * V,  un2 * V, ..., unp * V]]
            Sigma_u = np.kron(U, V)
            J_i = np.linalg.inv(Sigma_u)
            h_i = J_i.dot(A_i.ravel())
            J += J_i
            h += h_i

        self.A_0 = sample_gaussian(J=J, h=h).reshape((self.D_out, self.D_in))

        # Update the regression objects with the new global mean
        for r in self.regressions:
            r.natural_hypparam = \
                r._standard_to_natural(
                    self.nu_0, self.Q_0, self.A_0, self.K_0)

