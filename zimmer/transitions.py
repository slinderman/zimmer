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
    logistic, logit, adam_with_convergence_check, one_hot, relu
from ssm.preprocessing import interpolate_data


class HierarchicalStationaryTransitions(_Transitions):
    """
    Only allow the past observations and inputs to influence the
    next state.  Get rid of the transition matrix and replace it
    with a constant bias r.
    """
    def __init__(self, K, D, G=1, M=0, eta=0.1):
        super(HierarchicalStationaryTransitions, self).__init__(K, D, M)

        
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
    def __init__(self, K, D, tags=(None,), M=0, eta1=1e-4, eta2=1):
        super(ElaborateGroupRecurrentTransitions, self).__init__(K, D, M)

        
        # Global recurrence parameters
        self.shared_log_Ps = npr.randn(K, K)
        self.shared_Ws = npr.randn(K, M)
        self.shared_Rs = npr.randn(K, D)
        
        self.eta1 = eta1
        self.eta2 = eta2

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
            if group not in groups:
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
