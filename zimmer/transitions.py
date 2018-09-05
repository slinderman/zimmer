import copy
import warnings

import autograd.numpy as np
import autograd.numpy.random as npr
from autograd.scipy.misc import logsumexp
from autograd.scipy.special import gammaln
from autograd.scipy.stats import norm, gamma
from autograd.misc.optimizers import sgd, adam
from autograd import grad

from ssm.transitions import _Transitions
from ssm.util import random_rotation, ensure_args_are_lists, ensure_args_not_none, \
    logistic, logit, adam_with_convergence_check, one_hot
from ssm.preprocessing import interpolate_data


class HierarchicalStationaryTransitions(_Transitions):
    """
    Only allow the past observations and inputs to influence the
    next state.  Get rid of the transition matrix and replace it
    with a constant bias r.
    """
    def __init__(self, K, D, G=1, M=0, eta=0.1):
        super(HierarchicalRecurrentTransitions, self).__init__(K, D, M)

        
        # Global recurrence parameters
        self.shared_log_Ps = npr.randn(K, K)

        # Per-group parameters
        self.G = G
        self.eta = eta
        self.log_Ps = self.shared_log_Ps + np.sqrt(eta) * npr.randn(G, K, K)
        
    @property
    def params(self):
        return self.shared_log_Ps, self.log_Ps
    
    @params.setter
    def params(self, value):
        self.shared_log_Ps, self.log_Ps = value

    def permute(self, perm):
        """
        Permute the discrete latent states.
        """
        self.shared_log_Ps = self.shared_log_Ps[np.ix_(perm, perm)]
        for g in range(self.G):
            self.log_Ps[g] = self.log_Ps[g][np.ix_(perm, perm)]
        
    def initialize_from_standard(self, tr):
        # Copy the transition parameters
        self.shared_log_Ps = tr.log_Ps.copy()
        for g in range(self.G):
            self.log_Ps[g] = tr.log_Ps.copy()
                        
    def log_prior(self):
        lp = 0
        for g in range(self.G):
            lp += np.sum(norm.logpdf(self.log_Ps[g], self.shared_log_Ps, np.sqrt(self.eta)))
        return lp

    def log_transition_matrices(self, data, input, mask, tag):
        T, D = data.shape
        log_Ps = np.tile(self.log_Ps[tag][None, :, :], (T-1, 1, 1)) 
        return log_Ps - logsumexp(log_Ps, axis=2, keepdims=True)
            
    def m_step(self, expectations, datas, inputs, masks, tags, optimizer="adam", num_iters=10, **kwargs):
        """
        Use SGD to fit the model parameters
        """
        optimizer = dict(sgd=sgd, adam=adam, adam_with_convergence_check=adam_with_convergence_check)[optimizer]
        
        # Define the EM objective
        def _expected_log_joint(expectations):
            elbo = 0
            for data, input, mask, tag, (expected_states, expected_joints) \
                in zip(datas, inputs, masks, tags, expectations):
                log_Ps = self.log_transition_matrices(data, input, mask, tag)
                elbo += np.sum(expected_joints * log_Ps)
            return elbo

        # define optimization target
        T = sum([data.shape[0] for data in datas])
        def _objective(params, itr):
            self.params = params
            obj = _expected_log_joint(expectations) + self.log_prior()
            return -obj / T

        self.params = optimizer(grad(_objective), self.params, num_iters=num_iters, **kwargs)


class HierarchicalRecurrentTransitions(_Transitions):
    """
    Only allow the past observations and inputs to influence the
    next state.  Get rid of the transition matrix and replace it
    with a constant bias r.
    """
    def __init__(self, K, D, G=1, M=0, eta=0.1):
        super(HierarchicalRecurrentTransitions, self).__init__(K, D, M)

        
        # Global recurrence parameters
        self.shared_log_Ps = npr.randn(K, K)
        self.shared_Ws = npr.randn(K, M)
        self.shared_Rs = npr.randn(K, D)
        
        # Per-group parameters
        self.G = G
        self.eta = eta
        self.log_Ps = self.shared_log_Ps + np.sqrt(eta) * npr.randn(G, K, K)
        self.Ws = self.shared_Ws + np.sqrt(eta) * npr.randn(G, K, M)
        self.Rs = self.shared_Rs + np.sqrt(eta) * npr.randn(G, K, D)
        
    @property
    def params(self):
        return self.shared_log_Ps, self.shared_Ws, self.shared_Rs, \
               self.log_Ps, self.Ws, self.Rs
    
    @params.setter
    def params(self, value):
        self.shared_log_Ps, self.shared_Ws, self.shared_Rs, \
        self.log_Ps, self.Ws, self.Rs = value

    def permute(self, perm):
        """
        Permute the discrete latent states.
        """
        self.shared_log_Ps = self.shared_log_Ps[np.ix_(perm, perm)]
        self.shared_Ws = self.shared_Ws[perm]
        self.shared_Rs = self.shared_Rs[perm]
        
        for g in range(self.G):
            self.log_Ps[g] = self.log_Ps[g][np.ix_(perm, perm)]
            self.Ws[g] = self.Ws[g, perm]
            self.Rs[g] = self.Rs[g, perm]

    def initialize_from_standard(self, tr):
        # Copy the transition parameters
        self.shared_log_Ps = tr.log_Ps.copy()
        self.shared_Ws = tr.Ws.copy()
        self.shared_Rs = tr.Rs.copy()

        for g in range(self.G):
            self.log_Ps[g] = tr.log_Ps.copy()
            self.Ws[g] = tr.Ws.copy()
            self.Rs[g] = tr.Rs.copy()
                        
    def log_prior(self):
        lp = 0
        for g in range(self.G):
            lp += np.sum(norm.logpdf(self.log_Ps[g], self.shared_log_Ps, np.sqrt(self.eta)))
            lp += np.sum(norm.logpdf(self.Ws[g], self.shared_Ws, np.sqrt(self.eta)))
            lp += np.sum(norm.logpdf(self.Rs[g], self.shared_Rs, np.sqrt(self.eta)))
        return lp

    def log_transition_matrices(self, data, input, mask, tag):
        T, D = data.shape
        # Previous state effect
        log_Ps = np.tile(self.log_Ps[tag][None, :, :], (T-1, 1, 1)) 
        # Input effect
        log_Ps = log_Ps + np.dot(input[1:], self.Ws[tag].T)[:, None, :]
        # Past observations effect
        log_Ps = log_Ps + np.dot(data[:-1], self.Rs[tag].T)[:, None, :]
        return log_Ps - logsumexp(log_Ps, axis=2, keepdims=True)
            
    def m_step(self, expectations, datas, inputs, masks, tags, optimizer="adam", num_iters=10, **kwargs):
        """
        Use SGD to fit the model parameters
        """
        optimizer = dict(sgd=sgd, adam=adam, adam_with_convergence_check=adam_with_convergence_check)[optimizer]
        
        # Define the EM objective
        def _expected_log_joint(expectations):
            elbo = 0
            for data, input, mask, tag, (expected_states, expected_joints) \
                in zip(datas, inputs, masks, tags, expectations):
                log_Ps = self.log_transition_matrices(data, input, mask, tag)
                elbo += np.sum(expected_joints * log_Ps)
            return elbo

        # define optimization target
        T = sum([data.shape[0] for data in datas])
        def _objective(params, itr):
            self.params = params
            obj = _expected_log_joint(expectations) + self.log_prior()
            return -obj / T

        self.params = optimizer(grad(_objective), self.params, num_iters=num_iters, **kwargs)


class HierarchicalRecurrentOnlyTransitions(_Transitions):
    """
    Only allow the past observations and inputs to influence the
    next state.  Get rid of the transition matrix and replace it
    with a constant bias r.
    """
    def __init__(self, K, D, G=1, M=0, eta=0.1):
        super(HierarchicalRecurrentOnlyTransitions, self).__init__(K, D, M)

        # Global recurrence parameters
        self.shared_Ws = npr.randn(K, M)
        self.shared_Rs = npr.randn(K, D)
        self.shared_r = npr.randn(K)

        # Per-group parameters
        self.G = G
        self.eta = eta
        self.Ws = self.shared_Ws + np.sqrt(eta) * npr.randn(G, K, M)
        self.Rs = self.shared_Rs + np.sqrt(eta) * npr.randn(G, K, D)
        self.r = self.shared_r + np.sqrt(eta) * npr.randn(G, K)

    @property
    def params(self):
        return self.shared_Ws, self.shared_Rs, self.shared_r, \
               self.Ws, self.Rs, self.r
    
    @params.setter
    def params(self, value):
        self.shared_Ws, self.shared_Rs, self.shared_r, \
        self.Ws, self.Rs, self.r = value

    def permute(self, perm):
        """
        Permute the discrete latent states.
        """
        self.shared_Ws = self.shared_Ws[perm]
        self.shared_Rs = self.shared_Rs[perm]
        self.shared_r = self.shared_r[perm]

        for g in range(self.G):
            self.Ws[g] = self.Ws[g, perm]
            self.Rs[g] = self.Rs[g, perm]
            self.r[g] = self.r[g, perm]

    def log_prior(self):
        lp = 0
        for g in range(self.G):
            lp += np.sum(norm.logpdf(self.Ws[g], self.shared_Ws, np.sqrt(self.eta)))
            lp += np.sum(norm.logpdf(self.Rs[g], self.shared_Rs, np.sqrt(self.eta)))
            lp += np.sum(norm.logpdf(self.r[g], self.shared_r, np.sqrt(self.eta)))
        return lp
            
    def log_transition_matrices(self, data, input, mask, tag):
        T, D = data.shape
        log_Ps = np.dot(input[1:], self.Ws[tag].T)[:, None, :]              # inputs
        log_Ps = log_Ps + np.dot(data[:-1], self.Rs[tag].T)[:, None, :]     # past observations
        log_Ps = log_Ps + self.r[tag]                                       # bias
        log_Ps = np.tile(log_Ps, (1, self.K, 1))                            # expand
        return log_Ps - logsumexp(log_Ps, axis=2, keepdims=True)            # normalize

    def m_step(self, expectations, datas, inputs, masks, tags, optimizer="adam", num_iters=10, **kwargs):
        """
        Use SGD to fit the model parameters
        """
        optimizer = dict(sgd=sgd, adam=adam, adam_with_convergence_check=adam_with_convergence_check)[optimizer]
        
        # Define the EM objective
        def _expected_log_joint(expectations):
            elbo = 0
            for data, input, mask, tag, (expected_states, expected_joints) \
                in zip(datas, inputs, masks, tags, expectations):
                log_Ps = self.log_transition_matrices(data, input, mask, tag)
                elbo += np.sum(expected_joints * log_Ps)
            return elbo

        # define optimization target
        T = sum([data.shape[0] for data in datas])
        def _objective(params, itr):
            self.params = params
            obj = _expected_log_joint(expectations) + self.log_prior()
            return -obj / T

        self.params = optimizer(grad(_objective), self.params, num_iters=num_iters, **kwargs)
