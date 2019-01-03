import copy
import warnings

import autograd.numpy as np
import autograd.numpy.random as npr
from autograd.scipy.misc import logsumexp
from autograd.scipy.special import gammaln
from autograd.scipy.stats import norm, gamma, dirichlet
from autograd import grad

from ssm.transitions import _Transitions
from ssm.util import random_rotation, ensure_args_are_lists, ensure_args_not_none, \
    logistic, logit, one_hot, relu, batch_mahalanobis, fit_multiclass_logistic_regression
from ssm.preprocessing import interpolate_data
from ssm.optimizers import bfgs


class HierarchicalStationaryTransitions(_Transitions):
    """
    Only allow the past observations and inputs to influence the
    next state.  Get rid of the transition matrix and replace it
    with a constant bias r.
    """
    def __init__(self, K, D, tags=(None,), M=0, eta=0.1):
        super(HierarchicalStationaryTransitions, self).__init__(K, D, M)
        
        # Global recurrence parameters
        self.shared_log_Ps = npr.randn(K, K)
        
        # Per-group parameters
        self.tags = tags
        self.tags_to_indices = dict([(tag, i) for i, tag in enumerate(tags)])
        self.G = len(tags)
        assert self.G > 0
        
        self.eta = eta
        self.log_Ps = self.shared_log_Ps + np.sqrt(eta) * npr.randn(self.G, K, K)
        
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
        g = self.tags_to_indices[tag]
        T, D = data.shape
        log_Ps = np.tile(self.log_Ps[g][None, :, :], (T-1, 1, 1)) 
        return log_Ps - logsumexp(log_Ps, axis=2, keepdims=True)


class HierarchicalRecurrentTransitions(_Transitions):
    """
    Only allow the past observations and inputs to influence the
    next state.  Get rid of the transition matrix and replace it
    with a constant bias r.
    """
    def __init__(self, K, D, tags=(None,), M=0, eta=0.1, alpha=1.0, kappa=0.0):
        super(HierarchicalRecurrentTransitions, self).__init__(K, D, M)

        # Prior on log Ps
        self.alpha = alpha
        self.kappa = kappa

        # Global recurrence parameters
        self.shared_log_Ps = kappa * np.eye(K)
        self.shared_Ws = npr.randn(K, M)
        self.shared_Rs = np.zeros((K, D))
        
        # Per-group parameters
        self.tags = tags
        self.tags_to_indices = dict([(tag, i) for i, tag in enumerate(tags)])
        self.G = len(tags)
        assert self.G > 0
        
        self.eta = eta
        self.log_Ps = self.shared_log_Ps + np.sqrt(eta) * npr.randn(self.G, K, K)
        self.Ws = self.shared_Ws + np.sqrt(eta) * npr.randn(self.G, K, M)
        self.Rs = self.shared_Rs + np.sqrt(eta) * npr.randn(self.G, K, D)
        
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
        
        # Log prior on the shared transition matrix
        # shared_Ps = np.exp(self.shared_log_Ps - logsumexp(self.shared_log_Ps, axis=1, keepdims=True))
        # for k in range(self.K):
        #     alpha = self.alpha * np.ones(self.K) + self.kappa * (np.arange(self.K) == k)
        #     lp += dirichlet.logpdf(shared_Ps[k], alpha)
        lp += np.sum(norm.logpdf(self.shared_log_Ps, self.kappa * np.eye(self.K), 1/np.sqrt(self.alpha)))

        # Penalty on difference from shared matrix
        for g in range(self.G):
            lp += np.sum(norm.logpdf(self.log_Ps[g], self.shared_log_Ps, np.sqrt(self.eta)))
            lp += np.sum(norm.logpdf(self.Ws[g], self.shared_Ws, np.sqrt(self.eta)))
            lp += np.sum(norm.logpdf(self.Rs[g], self.shared_Rs, np.sqrt(self.eta)))
        return lp

    def log_transition_matrices(self, data, input, mask, tag):
        g = self.tags_to_indices[tag]

        T, D = data.shape
        # Previous state effect
        log_Ps = np.tile(self.log_Ps[g][None, :, :], (T-1, 1, 1)) 
        # Input effect
        log_Ps = log_Ps + np.dot(input[1:], self.Ws[g].T)[:, None, :]
        # Past observations effect
        log_Ps = log_Ps + np.dot(data[:-1], self.Rs[g].T)[:, None, :]
        return log_Ps - logsumexp(log_Ps, axis=2, keepdims=True)

    def m_step(self, expectations, datas, inputs, masks, tags, **kwargs):
        G, K, M, D = self.G, self.K, self.M, self.D

        # Update each group's parameters, then update the global parameters
        for g in range(G):
            Xs = []
            ys = []
            for tag, (Ez, _, _), data, input, in zip(tags, expectations, datas, inputs):
                if tag == g:
                    z = np.array([np.random.choice(K, p=p) for p in Ez])
                    Xs.append(np.hstack((one_hot(z[:-1], K), input[1:], data[:-1])))
                    ys.append(z[1:])
                    
            # Combine regressors and labels
            X = np.vstack(Xs)
            y = np.concatenate(ys)

            # Fit the logistic regression
            W0 = np.column_stack([self.log_Ps[g].T, self.Ws[g], self.Rs[g]])
            mu0 = np.column_stack([self.shared_log_Ps.T, self.shared_Ws, self.shared_Rs])
            coef_ = fit_multiclass_logistic_regression(X, y, K=K, W0=W0, mu0=mu0, sigmasq0=self.eta)

            # Extract the coefficients
            self.log_Ps[g] = coef_[:, :K].T
            self.Ws[g] = coef_[:, K:K+M]
            self.Rs[g] = coef_[:, K+M:]
            
        # Update the shared transition weights, incorporating sticky prior
        J_prior = self.alpha                            # alpha is the precision
        h_prior = self.alpha * self.kappa * np.eye(K)
        J_lkhd  = 1/self.eta * self.G
        h_lkhd  = np.sum(self.log_Ps, axis=0) / self.eta 
        Sigma_post = 1 / (J_prior + J_lkhd)
        mu_post = Sigma_post * (h_prior + h_lkhd)
        self.shared_log_Ps = mu_post

        # Update the input and recurrent weights (no prior here)
        self.shared_Ws = np.mean(self.Ws, axis=0)
        self.shared_Rs = np.mean(self.Rs, axis=0)


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

    def m_step(self, expectations, datas, inputs, masks, tags, **kwargs):
        G, K, M, D = self.G, self.K, self.M, self.D

        # Update each group's parameters, then update the global parameters
        for g in range(G):
            zps, zns = [], []
            for tag, (Ez, _, _) in zip(tags, expectations):
                if tag == g:
                    z = np.array([np.random.choice(K, p=p) for p in Ez])
                    zps.append(z[:-1])
                    zns.append(z[1:])

            X = np.vstack([np.hstack((input[1:], data[:-1], np.ones(data.shape[0]-1))) 
                           for zp, input, data, tag in zip(zps, inputs, datas, tags)
                           if tag == g])
            y = np.concatenate(zns)

            # Fit the logistic regression
            W0 = np.column_stack([self.Ws[g], self.Rs[g], self.r[g]])
            mu0 = np.column_stack([self.shared_Ws, self.shared_Rs, self.shared_r])
            coef_ = fit_multiclass_logistic_regression(X, y, K=K, W0=W0, mu0=mu0, sigmasq0=self.eta)

            # Extract the coefficients
            self.Ws[g] = coef_[:, :M]
            self.Rs[g] = coef_[:, M:M+D]
            self.r[g] = coef_[:, -1]
            
        # Update the shared parameters
        self.shared_Ws = np.mean(self.Ws, axis=0)
        self.shared_Rs = np.mean(self.Rs, axis=0)
        self.shared_r = np.mean(self.r, axis=0)


class HierarchicalRBFRecurrentTransitions(_Transitions):
    def __init__(self, K, D, tags=(None,), M=0, eta=0.1, alpha=1.0, kappa=0.0):
        super(HierarchicalRBFRecurrentTransitions, self).__init__(K, D, M)

        # Prior on log Ps
        self.alpha = alpha
        self.kappa = kappa

        # Global recurrence parameters
        self.shared_log_Ps = npr.randn(K, K)
        self.shared_mus = npr.randn(K, D)
        self._shared_sqrt_Sigmas = npr.randn(K, D, D)
        self.shared_Ws = npr.randn(K, M)
        
        # Per-group parameters
        self.tags = tags
        self.tags_to_indices = dict([(tag, i) for i, tag in enumerate(tags)])
        self.G = len(tags)
        assert self.G > 0
        
        self.eta = eta
        self.log_Ps = self.shared_log_Ps + np.sqrt(eta) * npr.randn(self.G, K, K)
        self.mus = self.shared_mus + np.sqrt(eta) * npr.randn(self.G, K, D)
        self._sqrt_Sigmas = self._shared_sqrt_Sigmas + np.sqrt(eta) * npr.randn(self.G, K, D, D)
        self.Ws = self.shared_Ws + np.sqrt(eta) * npr.randn(self.G, K, M)
        
    @property
    def params(self):
        return self.shared_log_Ps, self.shared_mus, self._shared_sqrt_Sigmas, self.shared_Ws, \
            self.log_Ps, self.mus, self._sqrt_Sigmas, self.Ws
    
    @params.setter
    def params(self, value):
        self.shared_log_Ps, self.shared_mus, self._shared_sqrt_Sigmas, self.shared_Ws, \
            self.log_Ps, self.mus, self._sqrt_Sigmas, self.Ws = value

    @property
    def Sigmas(self):
        return np.matmul(self._sqrt_Sigmas, np.swapaxes(self._sqrt_Sigmas, -1, -2))

    def initialize(self, datas, inputs, masks, tags):
        # Fit a GMM to the data to set the means and covariances
        from sklearn.mixture import GaussianMixture
        gmm = GaussianMixture(self.K, covariance_type="full")
        gmm.fit(np.vstack(datas))
        self.shared_mus = gmm.means_
        self._shared_sqrt_Sigmas = np.linalg.cholesky(gmm.covariances_)
        self.mus = np.repeat(self.shared_mus[None, ...], self.G, axis=0)
        self._sqrt_Sigmas = np.repeat(self._shared_sqrt_Sigmas[None, ...], self.G, axis=0)
        
    def permute(self, perm):
        self.shared_log_Ps = self.shared_log_Ps[np.ix_(perm, perm)]
        self.shared_mus = self.shared_mus[perm]
        self._shared_sqrt_Sigmas = self._shared_sqrt_Sigmas[perm]
        self.shared_Ws = self.shared_Ws[perm]

        for g in range(self.G):
            self.log_Ps[g] = self.log_Ps[g][np.ix_(perm, perm)]
            self.mus[g] = self.mus[g, perm]
            self._sqrt_Sigmas[g] = self._sqrt_Sigmas[g, perm]
            self.Ws[g] = self.Ws[g, perm]

    def initialize_from_standard(self, tr):
        # Copy the transition parameters
        self.shared_log_Ps = tr.log_Ps.copy()
        self.shared_mus = tr.mus.copy()
        self._shared_sqrt_Sigmas = tr._sqrt_Sigmas.copy()
        self.shared_Ws = tr.Ws.copy()

        for g in range(self.G):
            self.log_Ps[g] = tr.log_Ps.copy()
            self.mus[g] = tr.mus.copy()
            self._sqrt_Sigmas[g] = tr._sqrt_Sigmas.copy()
            self.Ws[g] = tr.Ws.copy()
                        
    def log_prior(self):
        lp = 0
        
        # Log prior on the shared transition matrix
        shared_Ps = np.exp(self.shared_log_Ps - logsumexp(self.shared_log_Ps, axis=1, keepdims=True))
        for k in range(self.K):
            alpha = self.alpha * np.ones(self.K) + self.kappa * (np.arange(self.K) == k)
            lp += dirichlet.logpdf(shared_Ps[k], alpha)

        # Penalty on difference from shared matrix
        for g in range(self.G):
            lp += np.sum(norm.logpdf(self.log_Ps[g], self.shared_log_Ps, np.sqrt(self.eta)))
            lp += np.sum(norm.logpdf(self.mus[g], self.shared_mus, np.sqrt(self.eta)))
            lp += np.sum(norm.logpdf(self._sqrt_Sigmas[g], self._shared_sqrt_Sigmas, np.sqrt(self.eta)))
            lp += np.sum(norm.logpdf(self.Ws[g], self.shared_Ws, np.sqrt(self.eta)))
        return lp

    def log_transition_matrices(self, data, input, mask, tag):
        K, D = self.K, self.D
        T = data.shape[0]
        assert np.all(mask), "Recurrent models require that all data are present."
        assert input.shape[0] == T

        g = self.tags_to_indices[tag]
        
        # Previous state effect
        log_Ps = np.tile(self.log_Ps[g, None, :, :], (T-1, 1, 1)) 

        # RBF function, quadratic term
        Ls = np.linalg.cholesky(self.Sigmas[g])                          # (K, D, D)
        diff = data[:-1, None, :] - self.mus[g]                          # (T-1, K, D)
        M = batch_mahalanobis(Ls, diff)                                  # (T-1, K)
        log_Ps = log_Ps + -0.5 * M[:, None, :]

        # RBF function, Gaussian normalizer
        # L_diag = np.reshape(Ls, Ls.shape[:-2] + (-1,))[..., ::D + 1]
        L_diag = np.array([np.diag(L) for L in Ls])                      # (K, D)
        half_log_det = np.sum(np.log(abs(L_diag)), axis=-1)              # (K,)
        log_normalizer = -0.5 * D * np.log(2 * np.pi) - half_log_det     # (K,)
        log_Ps = log_Ps + log_normalizer[None, None, :]

        # Input effect
        log_Ps = log_Ps + np.dot(input[1:], self.Ws[g].T)[:, None, :]
        return log_Ps - logsumexp(log_Ps, axis=2, keepdims=True)


class HierarchicalNeuralNetworkRecurrentTransitions(_Transitions):
    def __init__(self, K, D, G=1, M=0, eta=0.1, hidden_layer_sizes=(50,), nonlinearity="relu"):
        super(HierarchicalNeuralNetworkRecurrentTransitions, self).__init__(K, D, M)

        # Global recurrence parameters
        Ps = .95 * np.eye(K) + .05 * npr.rand(K, K)
        Ps /= Ps.sum(axis=1, keepdims=True)
        self.shared_log_Ps = np.log(Ps)

        # Initialize the NN weights
        layer_sizes = (D + M,) + hidden_layer_sizes + (K,)
        self.shared_weights = [npr.randn(m, n) for m, n in zip(layer_sizes[:-1], layer_sizes[1:])]
        self.shared_biases = [npr.randn(n) for n in layer_sizes[1:]]

        nonlinearities = dict(
            relu=relu,
            tanh=np.tanh,
            sigmoid=logistic)
        self.nonlinearity = nonlinearities[nonlinearity]

        # Per-group parameters
        self.G = G
        self.eta = eta
        self.log_Ps = self.shared_log_Ps + np.sqrt(eta) * npr.randn(G, K, K)
        self.weights = [[w + np.sqrt(eta) * npr.randn(*w.shape) for w in self.shared_weights] for g in range(G)]
        self.biases = [[b + np.sqrt(eta) * npr.randn(*b.shape) for b in self.shared_biases] for g in range(G)]

    @property
    def params(self):
        return self.shared_log_Ps, self.shared_weights, self.shared_biases, \
               self.log_Ps, self.weights, self.biases
    
    @params.setter
    def params(self, value):
        self.shared_log_Ps, self.shared_weights, self.shared_biases, \
        self.log_Ps, self.weights, self.biases = value

    def initialize_from_standard(self, tr):
        # Copy the transition parameters
        self.shared_log_Ps = tr.log_Ps.copy()
        self.shared_weights = copy.deepcopy(tr.weights)
        self.shared_biaes = copy.deepcopy(tr.biases)

        for g in range(self.G):
            self.log_Ps[g] = tr.log_Ps.copy()
            self.weights[g] = copy.deepcopy(tr.weights)
            self.biases[g] = copy.deepcopy(tr.biases)

    def permute(self, perm):
        """
        Permute the discrete latent states.
        """
        self.shared_log_Ps = self.shared_log_Ps[np.ix_(perm, perm)]
        self.shared_weights[-1] = self.shared_weights[-1][:,perm]
        self.shared_biases[-1] = self.shared_biases[-1][perm]

        for g in range(self.G):
            tmp = copy.deepcopy(self.weights[g])
            tmp[-1] = tmp[-1][:, perm]
            self.weights[g] = tmp

            tmp = copy.deepcopy(self.biases[g])
            tmp[-1] = tmp[-1][perm]
            self.biases[g] = tmp

    def log_prior(self):
        lp = 0
        for g in range(self.G):
            lp += np.sum(norm.logpdf(self.log_Ps[g], self.shared_log_Ps, np.sqrt(self.eta)))

            for w1, w2 in zip(self.shared_weights, self.weights[g]):
                lp += np.sum(norm.logpdf(w1, w2, np.sqrt(self.eta)))

            for b1, b2 in zip(self.shared_biases, self.biases[g]):
                lp += np.sum(norm.logpdf(b1, b2, np.sqrt(self.eta)))

        return lp
            
    def log_transition_matrices(self, data, input, mask, tag):
        T, D = data.shape

        # Pass the data and inputs through the neural network 
        x = np.hstack((data[:-1], input[1:]))
        for W, b in zip(self.weights[tag], self.biases[tag]):
            y = np.dot(x, W) + b
            x = self.nonlinearity(y)
        
        # Add the baseline transition biases
        log_Ps = self.log_Ps[tag][None, :, :] + y[:, None, :]

        # Normalize
        return log_Ps - logsumexp(log_Ps, axis=2, keepdims=True)


class GroupRecurrentTransitions(_Transitions):
    """
    Assume tags are of the form (group, index).  This model only
    shares across the same group; not individuals.
    """
    def __init__(self, K, D, groups=(None,), M=0, eta=0.1):
        super(GroupRecurrentTransitions, self).__init__(K, D, M)

        
        # Global recurrence parameters
        self.shared_log_Ps = npr.randn(K, K)
        self.shared_Ws = npr.randn(K, M)
        self.shared_Rs = npr.randn(K, D)
        
        # Per-group parameters
        # Per-group AR parameters
        self.groups = groups
        self.groups_to_indices = dict([(group, i) for i, group in enumerate(groups)])
        self.G = len(groups)

        self.eta = eta
        self.log_Ps = self.shared_log_Ps + np.sqrt(eta) * npr.randn(self.G, K, K)
        self.Ws = self.shared_Ws + np.sqrt(eta) * npr.randn(self.G, K, M)
        self.Rs = self.shared_Rs + np.sqrt(eta) * npr.randn(self.G, K, D)
        
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
        _, group = tag
        ind = self.groups_to_indices[group]

        # Previous state effect
        log_Ps = np.tile(self.log_Ps[ind][None, :, :], (T-1, 1, 1)) 
        # Input effect
        log_Ps = log_Ps + np.dot(input[1:], self.Ws[ind].T)[:, None, :]
        # Past observations effect
        log_Ps = log_Ps + np.dot(data[:-1], self.Rs[ind].T)[:, None, :]
        return log_Ps - logsumexp(log_Ps, axis=2, keepdims=True)


class ElaborateGroupRecurrentTransitions(_Transitions):
    """
    Assume tags are of the form (group, index).  This model allows
    each individual to have its own transition matrix and recurrent weights.
    Each group, however, is forced to share the input weights.
    """
    def __init__(self, K, D, tags=(None,), M=0, eta1=1e-4, eta2=1, alpha=1, kappa=0):
        super(ElaborateGroupRecurrentTransitions, self).__init__(K, D, M)

        # Hyperparameters
        self.eta1 = eta1
        self.eta2 = eta2
        self.alpha = alpha
        self.kappa = kappa

        # Global recurrence parameters
        self.shared_log_Ps = kappa * np.eye(K) + 1 / np.sqrt(alpha) * npr.randn(K, K)
        self.shared_Ws = npr.randn(K, M)
        self.shared_Rs = npr.randn(K, D)

        # Per-individual parameters
        self.tags = tags
        self.tags_to_indices = dict([(tag, i) for i, tag in enumerate(tags)])
        self.T = len(tags)
        assert self.T > 0

        self.log_Ps = self.shared_log_Ps + np.sqrt(eta1) * npr.randn(self.T, K, K)
        self.Rs = self.shared_Rs + np.sqrt(eta1) * npr.randn(self.T, K, D)

        # Per-group AR parameters
        self.groups = []
        for (i, group) in tags:
            if group not in self.groups:
                self.groups.append(group)
        self.groups_to_indices = dict([(group, i) for i, group in enumerate(self.groups)])
        self.G = len(self.groups)
        assert self.G > 0

        self.Ws = self.shared_Ws + np.sqrt(eta2) * npr.randn(self.G, K, M)
        
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

        for t in range(self.T):
            self.log_Ps[t] = self.log_Ps[t][np.ix_(perm, perm)]
            self.Rs[t] = self.Rs[t, perm]
        
        for g in range(self.G):
            self.Ws[g] = self.Ws[g, perm]
            
    def initialize_from_standard(self, tr):
        # Copy the transition parameters
        self.shared_log_Ps = tr.log_Ps.copy()
        self.shared_Ws = tr.Ws.copy()
        self.shared_Rs = tr.Rs.copy()

        for t in range(self.T):
            self.log_Ps[t] = tr.log_Ps.copy()
            self.Rs[t] = tr.Rs.copy()

        for g in range(self.G):
            self.Ws[g] = tr.Ws.copy()
                        
    def log_prior(self):
        lp = 0
        # Log prior on the shared transition matrix
        lp += np.sum(norm.logpdf(self.shared_log_Ps, self.kappa * np.eye(self.K), 1/np.sqrt(self.alpha)))

        # Penalize differences from the global mean
        for t in range(self.T):
            lp += np.sum(norm.logpdf(self.log_Ps[t], self.shared_log_Ps, np.sqrt(self.eta1)))
            lp += np.sum(norm.logpdf(self.Rs[t], self.shared_Rs, np.sqrt(self.eta1)))

        for g in range(self.G):
            lp += np.sum(norm.logpdf(self.Ws[g], self.shared_Ws, np.sqrt(self.eta2)))
        return lp

    def log_transition_matrices(self, data, input, mask, tag):
        T, D = data.shape
        _, group = tag
        tind = self.tags_to_indices[tag]
        gind = self.groups_to_indices[group]

        # Previous state effect
        log_Ps = np.tile(self.log_Ps[tind][None, :, :], (T-1, 1, 1)) 
        log_Ps = log_Ps + np.dot(data[:-1], self.Rs[tind].T)[:, None, :]
        log_Ps = log_Ps + np.dot(input[1:], self.Ws[gind].T)[:, None, :]
        return log_Ps - logsumexp(log_Ps, axis=2, keepdims=True)

    def m_step(self, expectations, datas, inputs, masks, tags, **kwargs):
        """
        Stochastic M-step. Sample discrete states and solve a multiclass
        logistic regression for each set of weights.
        """
        import pdb; pdb.set_trace()
        self._m_step_per_tag(expectations, datas, inputs, masks, tags, **kwargs)
        self._m_step_per_group(expectations, datas, inputs, masks, tags, **kwargs)
        self._m_step_global(expectations, datas, inputs, masks, tags, **kwargs)

    def _m_step_per_tag(self, expectations, datas, inputs, masks, tags, **kwargs):
        T, G, K, M, D = self.T, self.G, self.K, self.M, self.D

        # Update each tag's weights, holding group-level weights fixed
        for t in range(T):
            # Maximize the expected log joint
            def _expected_log_joint(params):
                log_Pt, Rt = params

                elbo = np.sum(norm.logpdf(log_Pt, self.shared_log_Ps, np.sqrt(self.eta1)))
                elbo += np.sum(norm.logpdf(Rt, self.shared_Rs, np.sqrt(self.eta1)))

                for data, input, mask, tag, (expected_states, expected_joints, _) \
                    in zip(datas, inputs, masks, tags, expectations):
                    
                    T, D = data.shape
                    _, group = tag
                    tind = self.tags_to_indices[tag]
                    gind = self.groups_to_indices[group]
                    
                    if self.tags_to_indices[tag] == t:
                        # log_Ps = np.tile(log_Pt[np.newaxis, :, :], (T-1, 1, 1)) 
                        log_Ps = log_Pt[np.newaxis, :, :]
                        log_Ps = log_Ps + np.dot(data[:-1], Rt.T)[:, np.newaxis, :]
                        log_Ps = log_Ps + np.dot(input[1:], self.Ws[gind].T)[:, np.newaxis, :]
                        log_Ps = log_Ps - logsumexp(log_Ps, axis=2, keepdims=True)
                        elbo += np.sum(expected_joints * log_Ps)

                return elbo

            # Normalize and negate for minimization
            T = sum([data.shape[0] for data, tag in zip(datas, tags) if self.tags_to_indices[tag] == t])
            _objective = lambda params, itr: -_expected_log_joint(params) / T

            # Fit the parameters for this group
            self.log_Ps[t], self.Rs[t] = bfgs(_objective, (self.log_Ps[t], self.Rs[t]))

    def _m_step_per_group(self, expectations, datas, inputs, masks, tags, **kwargs):
        T, G, K, M, D = self.T, self.G, self.K, self.M, self.D

        if M == 0:
            return

        # Update each group's weights, holding per-tag weights fixed
        for g in range(G):
            # Maximize the expected log joint
            def _expected_log_joint(Wg):
                elbo = np.sum(norm.logpdf(Wg, self.shared_Ws, np.sqrt(self.eta2)))
                for data, input, mask, tag, (expected_states, expected_joints, _) \
                    in zip(datas, inputs, masks, tags, expectations):

                    T, D = data.shape
                    _, group = tag
                    tind = self.tags_to_indices[tag]
                    gind = self.groups_to_indices[group]

                    if self.groups_to_indices[group] == g:
                        log_Ps = self.log_Ps[tind][np.newaxis, :, :]
                        log_Ps = log_Ps + np.dot(data[:-1], self.Rs[tind].T)[:, np.newaxis, :]
                        log_Ps = log_Ps + np.dot(input[1:], Wg.T)[:, np.newaxis, :]
                        log_Ps = log_Ps - logsumexp(log_Ps, axis=2, keepdims=True)
                        elbo += np.sum(expected_joints * log_Ps)

                return elbo

            # Normalize and negate for minimization
            T = sum([data.shape[0] for data, tag in zip(datas, tags) if self.groups_to_indices[tag[1]] == g])
            _objective = lambda Wg, itr: -_expected_log_joint(Wg) / T

            # Fit the logistic regression with different precision for each set of weights
            self.Ws[g] = bfgs(_objective, self.Ws[g])

    def _m_step_global(self, expectations, datas, inputs, masks, tags, **kwargs):
        # Update the global weights. alpha is the precision.
        J_prior = self.alpha                            
        h_prior = self.alpha * self.kappa * np.eye(self.K)
        J_lkhd  = 1/self.eta1 * self.T
        h_lkhd  = np.sum(self.log_Ps, axis=0) / self.eta1
        Sigma_post = 1 / (J_prior + J_lkhd)
        self.shared_log_Ps = Sigma_post * (h_prior + h_lkhd)
        
        # Update the input and recurrent weights (no prior here)
        self.shared_Ws = np.mean(self.Ws, axis=0)
        self.shared_Rs = np.mean(self.Rs, axis=0)
