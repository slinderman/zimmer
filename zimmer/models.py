import numpy as np
from future.utils import iteritems

from zimmer.emissions import HierarchicalDiagonalRegression
from zimmer.dynamics import _HierarchicalAutoRegressionMixin
from zimmer.states import HierarchicalFactorAnalysisStates, HierarchicalSLDSStates, \
    HierarchicalRecurrentSLDSStates, HierarchicalLDSStates, HierarchicalARHMMStates, \
    HierarchicalRecurrentARHMMStates, HierarchicalRecurrentARHMMSeparateTransStates
from pybasicbayes.models.factor_analysis import FactorAnalysis
from pybasicbayes.distributions import Regression
from pybasicbayes.util.general import AR_striding
from pyhsmm.models import _HMMGibbsSampling, _SeparateTransMixin
from pylds.models import MissingDataLDS
from autoregressive.models import ARWeakLimitStickyHDPHMM, ARHMM
from pyslds.models import HMMSLDS, WeakLimitStickyHDPHMMSLDS
from rslds.models import SoftmaxRecurrentSLDS, SoftmaxRecurrentOnlySLDS, _RecurrentARHMMMixin


class HierarchicalFactorAnalysis(FactorAnalysis):
    _states_class = HierarchicalFactorAnalysisStates

    def __init__(self, D_obs, D_latent, N_groups,
                 W=None, sigmasq=None,
                 sigmasq_W_0=1.0, mu_W_0=0.0,
                 alpha_0=3.0, beta_0=2.0):

        self.D_obs, self.D_latent, self.N_groups = D_obs, D_latent, N_groups

        self.regression = \
            HierarchicalDiagonalRegression(
                self.D_obs, self.D_latent, self.N_groups,
                mu_0=mu_W_0 * np.ones(self.D_latent),
                Sigma_0=sigmasq_W_0 * np.eye(self.D_latent),
                alpha_0=alpha_0, beta_0=beta_0,
                A=W, sigmasq=sigmasq)

        self.data_list = []

    def generate(self, keep=True, N=1, mask=None, group=0, **kwargs):
        # Sample from the factor analysis model
        W, sigmasq = self.W, self.sigmasq[group]
        Z = np.random.randn(N, self.D_latent)
        X = np.dot(Z, W.T) + np.sqrt(sigmasq) * np.random.randn(N, self.D_obs)

        data = self._states_class(self, X, mask=mask, group=group, **kwargs)
        data.Z = Z
        if keep:
            self.data_list.append(data)
        return data


    def resample_model(self):
        for data in self.data_list:
            data.resample()

        data = [(d.Z, d.X) for d in self.data_list]
        mask = [d.mask for d in self.data_list]
        groups = [d.group for d in self.data_list]
        self.regression.resample(data, mask=mask, groups=groups)

    def EM_step(self):
        for data in self.data_list:
            data.E_step()

        stats = [d.E_emission_stats for d in self.data_list]
        groups = [d.group for d in self.data_list]
        self.regression.max_likelihood(stats=stats, groups=groups)



class HierarchicalLDS(MissingDataLDS):
    _states_class = HierarchicalLDSStates

    def resample_emission_distn(self):
        xys = [(np.hstack((s.gaussian_states, s.inputs)), s.data) for s in self.states_list]
        mask = [s.mask for s in self.states_list]
        groups = [s.group for s in self.states_list]
        self.emission_distn.resample(data=xys, mask=mask, groups=groups)


class HierarchicalLDSWithLocalAR(HierarchicalLDS):
    """
    A linear dynamical system with the extra feature that observations
    for each neuron are conditioned on that neuron's preceding values.

    y_{tn} = \sum_{p=1}^P a_n^{(p)} y_{t-p, n} + c_n^T x_t + d_n + noise
    """
    def __init__(self, dynamics_distn, emission_distn, nlags=1):
        super(HierarchicalLDSWithLocalAR, self).__init__(dynamics_distn, emission_distn)
        self.nlags = nlags

        # Initialize autoregressions for each output dimension
        A0 = np.zeros((1,nlags))
        A0[0,-1] = 1
        self.autoregressions = [
            Regression(nu_0=2,
                       S_0=np.eye(1),
                       M_0=np.zeros((1, nlags)),
                       K_0=np.eye(nlags),
                       A=A0,
                       sigma=np.eye(1))
            for _ in range(self.D_obs)
        ]

    def add_data(self, data, inputs=None, **kwargs):
        # First preprocess the data with an AR model
        yhats = []
        for n in range(self.D_obs):
            yn_lag = np.concatenate((np.repeat(data[0,n], self.nlags), data[:,n]))
            yn_lag = AR_striding(yn_lag[:, None], self.nlags)
            yhats.append(self.autoregressions[n].predict(yn_lag[:,:-1]))
        yhats = np.hstack(yhats)

        assert yhats.shape == data.shape
        residual = data - yhats

        super(HierarchicalLDSWithLocalAR, self).add_data(residual, inputs=inputs, **kwargs)
        self.states_list[-1].raw_data = data

    def _resample_autoregressions(self):
        nlags = self.nlags
        for n, ar in enumerate(self.autoregressions):
            # Subtract off the LDS component then refit the autoregression
            xs = []
            ys = []
            for s in self.states_list:
                if s.mask[0,n]:
                    xs.append(AR_striding(s.raw_data[:,n:n+1], nlags)[:,:-1])
                    ys.append((s.raw_data[:,n]
                               - s.gaussian_states.dot(self.C[n])
                               - s.inputs.dot(self.D[n]))[nlags:, None])

            # Fit the regression with these (x,y) pairs
            ar.max_likelihood(list(zip(xs, ys)))

            # Subtract off the autoregression part and reset the residuals
            for s in self.states_list:
                if s.mask[0, n]:
                    x = AR_striding(
                        np.concatenate((
                            np.repeat(s.raw_data[0, n], nlags),
                            s.raw_data[:,n])),
                        nlags)[:,:-1]
                    ypred = ar.predict(x)
                    s.data[:, n:n+1] = s.raw_data[:, n:n+1] - ypred

    def resample_emission_distn(self):
        super(HierarchicalLDSWithLocalAR, self).resample_emission_distn()
        self._resample_autoregressions()


class _HierarchicalARHMMMixin(object):
    _states_class = HierarchicalARHMMStates

    def resample_obs_distns(self):
        for state, distn in enumerate(self.obs_distns):
            assert isinstance(distn, _HierarchicalAutoRegressionMixin)
            distn.resample([s.data[s.stateseq == state] for s in self.states_list],
                           [s.group for s in self.states_list])
        self._clear_caches()

    # todo: implement generate function


    def generate(self, T=100, keep=True, init_x=None, init_z=None, covariates=None, with_noise=True, group=None,
                 tau=0, noise_scale=1):
        from pybasicbayes.util.stats import sample_discrete, sample_gaussian
        # Generate from the prior and raise exception if unstable
        K, n = self.num_states, self.D

        # Initialize discrete state sequence
        zs = np.empty(T, dtype=np.int32)
        if init_z is None:
            zs[0] = sample_discrete(self.init_state_distn.pi_0.ravel())
        else:
            zs[0] = init_z

        xs = np.empty((T, n), dtype='double')
        if init_x is None:
            xs[0] = np.random.randn(n)
        else:
            xs[0] = init_x

        for t in range(1, T):
            # Sample discrete state given previous continuous state
            A = self.trans_distn.trans_matrix
            zs[t] = sample_discrete(A[zs[t-1], :])

            # Sample continuous state given current discrete state
            mu = self.obs_distns[zs[t]].predict(xs[t - 1][None, :], group=group)[0]
            sigma = self.obs_distns[zs[t]].sigmas[group] * noise_scale
            J_dyn = np.linalg.inv(sigma)
            h = J_dyn.dot(mu)

            # Combine predicted mean with stability penalty from
            #   p(1^T x | 0, \tau)
            # where \tau = stable_prec. Thisintroduces potential of the form
            #   exp( -\tau/2 (1^T x)^2 )
            # = exp( -\tau/2 x^T 1 1^T x)
            #
            # In information form, this is N(x | J = \tau 11^T, h = 0)
            J_pen = tau * np.eye(self.D)

            if with_noise:
                # xs[t] = self.obs_distns[zs[t]].rvs(xs[t-1][None, :], return_xy=False, group=group)
                xs[t] = sample_gaussian(J=J_dyn+J_pen, h=h)
            else:
                xs[t] = np.linalg.solve(J_dyn + J_pen, h)

            # assert np.all(abs(xs[t]) < 5), "RARHMM appears to be unstable!"

        # TODO:
        # if keep:
        #     ...

        return xs, zs


class HierarchicalARHMM(_HierarchicalARHMMMixin, ARHMM):
    pass


class HierarchicalARWeakLimitStickyHDPHMM(_HierarchicalARHMMMixin, ARWeakLimitStickyHDPHMM):
    pass


class HierarchicalRecurrentARHMM(_HierarchicalARHMMMixin, _RecurrentARHMMMixin, _HMMGibbsSampling):
    from rslds.transitions import SoftmaxInputHMMTransitions
    _trans_class = SoftmaxInputHMMTransitions
    _states_class = HierarchicalRecurrentARHMMStates

    def generate(self, T=100, keep=True, init_x=None, init_z=None, covariates=None, with_noise=True, group=None,
                 tau=0, noise_scale=1):
        from pybasicbayes.util.stats import sample_discrete, sample_gaussian
        # Generate from the prior and raise exception if unstable
        K, n = self.num_states, self.D

        # Initialize discrete state sequence
        zs = np.empty(T, dtype=np.int32)
        if init_z is None:
            zs[0] = sample_discrete(self.init_state_distn.pi_0.ravel())
        else:
            zs[0] = init_z

        xs = np.empty((T, n), dtype='double')
        if init_x is None:
            xs[0] = np.random.randn(n)
        else:
            xs[0] = init_x

        for t in range(1, T):
            # Sample discrete state given previous continuous state
            A = self.trans_distn.get_trans_matrices(xs[t-1:t])[0]
            zs[t] = sample_discrete(A[zs[t-1], :])

            # Sample continuous state given current discrete state
            mu = self.obs_distns[zs[t]].predict(xs[t - 1][None, :], group=group)[0]
            sigma = self.obs_distns[zs[t]].sigmas[group] * noise_scale
            J_dyn = np.linalg.inv(sigma)
            h = J_dyn.dot(mu)

            # Combine predicted mean with stability penalty from
            #   p(1^T x | 0, \tau)
            # where \tau = stable_prec. Thisintroduces potential of the form
            #   exp( -\tau/2 (1^T x)^2 )
            # = exp( -\tau/2 x^T 1 1^T x)
            #
            # In information form, this is N(x | J = \tau 11^T, h = 0)
            J_pen = tau * np.eye(self.D)

            if with_noise:
                # xs[t] = self.obs_distns[zs[t]].rvs(xs[t-1][None, :], return_xy=False, group=group)
                xs[t] = sample_gaussian(J=J_dyn+J_pen, h=h)
            else:
                xs[t] = np.linalg.solve(J_dyn + J_pen, h)

            # assert np.all(abs(xs[t]) < 5), "RARHMM appears to be unstable!"

        # TODO:
        # if keep:
        #     ...

        return xs, zs


class HierarchicalRecurrentARHMMWithNN(HierarchicalRecurrentARHMM):
    from rslds.transitions import NNInputHMMTransitions
    _trans_class = NNInputHMMTransitions


class HierarchicalRecurrentARHMMSeparateTrans(_SeparateTransMixin, HierarchicalRecurrentARHMM):
    """
    SeparateTrans mixin allows each state to have a 'group_id' associated with it.
    Each group gets its own transition object.  We need to update this a bit though
    since the trans distributions take both stateseqs and covseqs.
    """
    _states_class = HierarchicalRecurrentARHMMSeparateTransStates
    def resample_trans_distn(self):
        for trans_group, trans_distn in iteritems(self.trans_distns):
            group_states = [s for s in self.states_list if hash(s.trans_group) == hash(trans_group)]
            trans_distn.resample(
                stateseqs=[s.stateseq for s in group_states],
                covseqs=[s.covariates for s in group_states]
            )
        self._clear_caches()

    def resample_init_state_distn(self):
        for trans_group, init_state_distn in iteritems(self.init_state_distns):
            init_state_distn.resample([s.stateseq[0] for s in self.states_list
                if hash(s.trans_group) == hash(trans_group)])
        self._clear_caches()


class _HierarchicalSLDSMixin(object):
    _states_class = HierarchicalSLDSStates

    def resample_emission_distns(self):
        assert self._single_emission
        data = [(np.hstack((s.gaussian_states, s.inputs)), s.data)
                for s in self.states_list]
        mask = [s.mask for s in self.states_list] if self.has_missing_data else None
        groups = [s.group for s in self.states_list]

        if self.has_missing_data:
            self._emission_distn.resample(data=data, mask=mask, groups=groups)
        else:
            self._emission_distn.resample(data=data, groups=groups)

        self._clear_caches()

    # Max likelihood
    def _M_step_emission_distn(self):
        assert self._single_emission
        E_emi_stats = lambda s: \
            tuple(np.sum(stat, axis=0) for stat in s.E_emission_stats)
        stats = [E_emi_stats(s) for s in self.states_list]
        groups = [s.group for s in self.states_list]
        self._emission_distn.max_likelihood(stats=stats, groups=groups)

    def VBEM_ELBO(self):
        # log p(theta)
        from pyslds.util import gaussian_logprior, regression_logprior
        elbo = self.trans_distn.log_prior() if hasattr(self.trans_distn, 'log_prior') else 0
        elbo += np.sum([gaussian_logprior(id) for id in self.init_dynamics_distns])
        elbo += np.sum([regression_logprior(dd) for dd in self.dynamics_distns])

        # Handle the hierarchical emission distn log prior
        # elbo += regression_logprior(self.emission_distns[0])
        from scipy.stats import multivariate_normal, gamma
        ed = self._emission_distn
        A = ed.A
        J, h = ed.J_0, ed.h_0
        Sigma = np.linalg.inv(J)
        mu = Sigma.dot(h)

        for d in range(ed.D_out):
            elbo += multivariate_normal(mu, Sigma).logpdf(A[d])

        # log p(sigmasq)
        sigmasq = ed.sigmasq_flat
        alpha, beta = ed.alpha_0, ed.beta_0
        for g in range(ed.N_groups):
            for d in range(ed.D_out):
                elbo += gamma(alpha, scale=1. / beta).logpdf(1. / sigmasq[g, d])

        # E_q [log p(z, x, y, theta)]
        elbo += sum(s.vb_elbo() for s in self.states_list)
        return elbo


class HierarchicalHMMSLDS(_HierarchicalSLDSMixin, HMMSLDS):
    pass


class HierarchicalWeakLimitStickyHDPHMMSLDS(_HierarchicalSLDSMixin, WeakLimitStickyHDPHMMSLDS):
    pass


class HierarchicalRecurrentSLDS(_HierarchicalSLDSMixin, SoftmaxRecurrentSLDS):
    _states_class = HierarchicalRecurrentSLDSStates


class HierarchicalRecurrentOnlySLDS(_HierarchicalSLDSMixin, SoftmaxRecurrentOnlySLDS):
    _states_class = HierarchicalRecurrentSLDSStates
