import copy
import warnings
from functools import partial

import autograd.numpy as np
import autograd.numpy.random as npr
from autograd.scipy.misc import logsumexp
from autograd.misc.optimizers import sgd, adam
from autograd.extend import notrace_primitive
from autograd import grad

from ssm.init_state_distns import InitialStateDistribution
from ssm.transitions import StationaryTransitions, RecurrentTransitions
from ssm.observations import AutoRegressiveObservations
from ssm.primitives import hmm_normalizer, hmm_expected_states, hmm_filter, hmm_sample
from ssm.util import ensure_args_are_lists, ensure_args_not_none, \
    ensure_slds_args_not_none, ensure_elbo_args_are_lists, one_hot

class SuperHMM(object):
    """
    Hierarchical Hidden Markov Model with per-neuron states.

    
    super states  s_t in {1, ..., K}
    neuron states z_tn in {1, ..., C}
    observations  y_tn in R^N
    inputs        u_t in R^M           [optional]
    
    dynamics:

    s_t ~ p(s_t | s_{t-1}, u_t)
    z_tn ~ p(z_tn | z_{t-1,n}, y_{t-1, n}, s_t)
    
    observations:

    y_tn ~ N(a_{z_tn} y_{t-1,n} + b_{z_tn}, sigma_n)

    fit with stochastic EM:
    - Gibbs sample the posterior distribution over latent states
    - Maximize complete data log likelihood with sample
    
    """
    def __init__(self, K, C, N, M):
        self.K, self.C, self.N, self.M = K, C, N, M
        self.init_state_distn = InitialStateDistribution(K, N, M)
        self.super_transitions = StationaryTransitions(K, N, M)
        self.neuron_transitions = [RecurrentTransitions(C, 1, K) for n in range(N)]
        self.neuron_observations = [AutoRegressiveObservations(C, 1, M) for n in range(N)]
        
        self._fitting_methods = \
            dict(stochastic_em=self._fit_stochastic_em)

    @property
    def params(self):
        return self.init_state_distn.params, \
               self.super_transitions.params, \
               [nt.params for nt in self.neuron_transitions], \
               [no.params for no in self.neuron_observations]
    
    @params.setter
    def params(self, value):
        self.init_state_distn.params = value[0]
        self.super_transitions.params = value[1]
        for nt, prms in zip(self.neuron_transitions, value[2]):
            nt.params = prms
        for no, prms in zip(self.neuron_observations, value[3]):
            no.params = prms

    @ensure_args_are_lists
    def initialize(self, datas, inputs=None, masks=None, tags=None):
        """
        Initialize parameters given data.
        """
        # Initialize the neuron observations parameters with a simple HMM
        from ssm.models import HMM
        from copy import deepcopy
        
        print("Initializing observation models...")
        for n in range(self.N):
            arhmm = HMM(self.C, 1, self.M, observations="autoregressive", transitions="recurrent")
            arhmm.fit([d[:,n:n+1] for d in datas], 
                      inputs, 
                      [m[:, n:n+1] for m in masks],
                      tags,
                      print_intvl=np.inf)
            self.neuron_observations[n].params = deepcopy(arhmm.observations.params)
        print("Done.")
        print("")

    def permute(self, perm):
        """
        Permute the discrete latent states.
        """
        assert np.all(np.sort(perm) == np.arange(self.K))
        self.init_state_distn.permute(perm)
        self.super_transitions.permute(perm)

    def log_prior(self):
        """
        Compute the log prior probability of the model parameters
        """  
        return 0

    def sample(self, T, prefix=None, input=None, tag=None, with_noise=True):
        K, C, N, M = self.K, self.C, self.N, self.M

        # Initialize inputs
        input = np.zeros((T, M)) if input is None else input
        mask = np.ones((T, N), dtype=bool)

        # Initialize outputs
        s = np.zeros(T, dtype=int)
        z = np.zeros((T, N), dtype=int)
        y = np.zeros((T, N))

        # Sample first time step
        log_pi0 = self.init_state_distn.log_initial_state_distn(y, input, mask, tag)
        s[0] = npr.choice(K, p=np.exp(log_pi0))
        for n in range(N):
            Pn = np.exp(self.neuron_transitions[n].\
                    log_transition_matrices(y[:1, n:n+1].repeat(2, axis=0), 
                                            one_hot(s[:1], K).repeat(2, axis=0), 
                                            mask=mask[:1].repeat(2, axis=0), 
                                            tag=tag))[0]
            z[0,n] = npr.choice(C, p=Pn[0])
            y[0,n] = self.neuron_observations[n].sample_x(
                            z[0,n], y[:0,n:n+1], input=input[0], tag=tag, with_noise=with_noise)

        # Sample subsequent time steps
        for t in range(1, T):
            Ps = np.exp(self.super_transitions.\
                    log_transition_matrices(y[t-1:t+1], 
                                            input[t-1:t+1], 
                                            mask=mask[t-1:t+1], tag=tag))[0]
            s[t] = npr.choice(K, p=Ps[s[t-1]])

            for n in range(N):
                Pn = np.exp(self.neuron_transitions[n].\
                        log_transition_matrices(y[t-1:t+1, n:n+1], 
                                                one_hot(s[t-1:t+1], K), 
                                                mask=mask[t-1:t+1], tag=tag))[0]
                z[t,n] = npr.choice(C, p=Pn[z[t-1,n]])

                y[t,n] = self.neuron_observations[n].sample_x(
                            z[t,n], y[:t,n:n+1], input=input[t], tag=tag, with_noise=with_noise)

        return s, z, y

    def log_probability(self, states, datas, inputs, masks, tags):
        """
        Compute the complete data log likelihood p(s, z, y | theta)
        """
        K, C, N = self.K, self.C, self.N
        
        lp = 0
        for (s, z), data, input, mask, tag in zip(states, datas, inputs, masks, tags):
            T = data.shape[0]

            log_pi0 = self.init_state_distn.log_initial_state_distn(data, input, mask, tag)
            lp += log_pi0[s[0]]

            log_Ps = self.super_transitions.log_transition_matrices(data, input, mask, tag)
            lp += np.sum(log_Ps[np.arange(T-1), s[:-1], s[1:]])

            for n in range(self.N):
                log_pi0 = -np.log(C) * np.ones(C)
                lp += log_pi0[z[0, n]]

                log_Ps = self.neuron_transitions[n].log_transition_matrices(data[:, n:n+1], one_hot(s, K), mask[:, n:n+1], tag)
                lp += np.sum(log_Ps[np.arange(T-1), z[:-1, n], z[1:, n]])

                lls = self.neuron_observations[n].log_likelihoods(data[:, n:n+1], input, mask[:, n:n+1], tag)
                lp += np.sum(lls[np.arange(T), z[:, n]])

        return lp 

    def _gibbs(self, data, input, mask, tag, current_sample=None, num_gibbs_iters=10, verbose=True):
        # Initialize the local and global states
        unbox = lambda x: x if isinstance(x, np.ndarray) else x._value
        K, C = self.K, self.C
        T, N = data.shape

        if current_sample is None:
            # Sample super state from the prior
            log_pi0 = unbox(self.init_state_distn.log_initial_state_distn(data, input, mask, tag))
            log_Ps = unbox(self.super_transitions.log_transition_matrices(data, input, mask, tag))
            s = hmm_sample(log_pi0, log_Ps, np.zeros((T, K)))

            # Local states will be filled in below
            z = np.zeros((T, N), dtype=int)

        else:
            s, z = current_sample
            assert s.shape == (T,) and s.dtype == int
            assert z.shape == (T, N) and z.dtype == int

        # Run the Gibbs sampler
        for i in range(num_gibbs_iters):

            # Sample the local states given the global states
            log_pi0 = -np.log(C) * np.ones(C)
            for n in range(self.N):
                log_Ps = unbox(self.neuron_transitions[n].log_transition_matrices(data[:, n:n+1], one_hot(s, K), mask[:, n:n+1], tag))
                lls = unbox(self.neuron_observations[n].log_likelihoods(data[:, n:n+1], input, mask[:, n:n+1], tag))
                z[:, n] = hmm_sample(log_pi0, log_Ps, lls)

            # Sample the global states given the local states
            log_pi0 = unbox(self.init_state_distn.log_initial_state_distn(data, input, mask, tag))
            log_Ps = unbox(self.super_transitions.log_transition_matrices(data, input, mask, tag))

            # Construct log likelihoods for each discrete state
            lls = np.zeros((T, K))
            for k in range(K):
                k_oh = np.repeat(one_hot(k, K), T, axis=0)
                for n in range(N):
                    lPn = unbox(self.neuron_transitions[n].log_transition_matrices(data[:, n:n+1], k_oh, mask[:, n:n+1], tag))
                    lls[1:, k] += lPn[np.arange(T-1), z[:-1, n], z[1:, n]]
            s = hmm_sample(log_pi0, log_Ps, lls)

        return s, z

    def _fit_stochastic_em(self, datas, inputs, masks, tags, print_intvl=1, num_samples=10, optimizer="adam", **kwargs):
        """
        Fit the model by maximizing the expected log probability 

        theta = argmax E_{p(s, z | y, theta')} [log p(s, z, y | theta)],
        
        where theta' are the parameters from the previous iteration. 
        
        This can be seen as variational inference with the true posterior
        p(s, z | y, theta').  Unfortunately, we can't compute exact expectations
        wrt this true posterior so instead we'll use Gibbs sampling to approximate it.
        
        Alternatively, we can call it stochastic EM.
        """
        T = sum([data.shape[0] for data in datas])
        
        current_sample = [None] * len(datas)
        def _objective(params, itr):
            # collect samples from the posterior distribution of latent states
            # (use the old parameters when doing so)
            states = [self._gibbs(data, input, mask, tag, current_sample=smpl) 
                      for data, input, mask, tag, smpl in zip(datas, inputs, masks, tags, current_sample)]
            # states = [self._gibbs(data, input, mask, tag, current_sample=None) 
            #           for data, input, mask, tag, smpl in zip(datas, inputs, masks, tags, current_sample)]

            for i, smpl in enumerate(states):
                current_sample[i] = smpl

            # Compute (approximate) expected log probability under new parameters
            self.params = params
            obj = self.log_probability(states, datas, inputs, masks, tags)
            return -obj / T

        lls = []
        def _print_progress(params, itr, g):
            lls.append(self.log_probability(current_sample, datas, inputs, masks, tags)._value)
            # lls.append(-T * _objective(params, itr))
            if itr % print_intvl == 0:
                print("Iteration {}.  LL: {}".format(itr, lls[-1]))
        
        optimizers = dict(sgd=sgd, adam=adam)
        self.params = \
            optimizers[optimizer](grad(_objective), self.params, callback=_print_progress, **kwargs)

        return lls

    @ensure_args_are_lists
    def fit(self, datas, inputs=None, masks=None, tags=None, method="stochastic_em", initialize=True, **kwargs):
        if method not in self._fitting_methods:
            raise Exception("Invalid method: {}. Options are {}".\
                            format(method, self._fitting_methods.keys()))

        if initialize:
            self.initialize(datas, inputs=inputs, masks=masks, tags=tags)

        return self._fitting_methods[method](datas, inputs=inputs, masks=masks, tags=tags, **kwargs)

