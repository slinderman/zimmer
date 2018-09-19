import copy
import warnings

import autograd.numpy as np
import autograd.numpy.random as npr
from autograd.scipy.misc import logsumexp
from autograd.scipy.special import gammaln
from autograd.scipy.stats import norm, gamma
from autograd.misc.optimizers import sgd, adam
from autograd import grad

from ssm.observations import _Observations
from ssm.util import random_rotation, ensure_args_are_lists, ensure_args_not_none, \
    logistic, logit, adam_with_convergence_check, one_hot
from ssm.preprocessing import interpolate_data


class HierarchicalIndependentAutoRegressiveObservations(_Observations):
    def __init__(self, K, D, G=1, M=0, lags=1, eta=0.1):
        super(HierarchicalIndependentAutoRegressiveObservations, self).__init__(K, D, M)
        
        # Distribution over initial point
        self.mu_init = np.zeros(D)
        self.inv_sigma_init = np.zeros(D)
        
        # Global AR parameters
        assert lags > 0 
        self.lags = lags
        self.shared_As = .95 * np.ones((K, D, lags))
        self.shared_bs = npr.randn(K, D)
        self.shared_Vs = npr.randn(K, D, M)
        # self.shared_inv_sigmas = -4 + npr.randn(K, D)

        # Per-group AR parameters
        self.G = G
        self.eta = eta
        self.As = .95 * np.ones((G, K, D, lags))
        self.bs = npr.randn(G, K, D)
        self.Vs = npr.randn(G, K, D, M)
        self.inv_sigmas = -4 + npr.randn(G, K, D)

    @property
    def params(self):
        return self.shared_As, self.shared_bs, self.shared_Vs, \
               self.As, self.bs, self.Vs, self.inv_sigmas
        
    @params.setter
    def params(self, value):
        self.shared_As, self.shared_bs, self.shared_Vs, \
        self.As, self.bs, self.Vs, self.inv_sigmas = value
        
    def permute(self, perm):
        self.shared_As = self.shared_As[perm]
        self.shared_bs = self.shared_bs[perm]
        self.shared_Vs = self.shared_Vs[perm]

        for g in range(self.G):
            self.As[g] = self.As[g, perm]
            self.bs[g] = self.bs[g, perm]
            self.Vs[g] = self.Vs[g, perm]
            self.inv_sigmas[g] = self.inv_sigmas[g, perm]

    def initialize(self, datas, inputs=None, masks=None, tags=None):
        # Initialize with linear regressions
        from sklearn.linear_model import LinearRegression
        data = np.concatenate(datas) 
        input = np.concatenate(inputs)
        T = data.shape[0]

        for k in range(self.K):
            for d in range(self.D):
                ts = npr.choice(T-self.lags, replace=False, size=(T-self.lags)//self.K)
                x = np.column_stack([data[ts + l, d:d+1] for l in range(self.lags)] + [input[ts, :self.M]])
                y = data[ts+self.lags, d:d+1]
                lr = LinearRegression().fit(x, y)

                self.shared_As[k, d] = lr.coef_[:, :self.lags]
                self.shared_Vs[k, d] = lr.coef_[:, self.lags:self.lags+self.M]
                self.shared_bs[k, d] = lr.intercept_
                
                for g in range(self.G):
                    self.As[g, k, d] = self.shared_As[k, d]
                    self.bs[g, k, d] = self.shared_bs[k, d]
                    self.Vs[g, k, d] = self.shared_Vs[k, d]

                resid = y - lr.predict(x)
                sigmas = np.var(resid, axis=0)
                for g in range(self.G):
                    self.inv_sigmas[g, k, d] = np.log(sigmas + 1e-8)

    def initialize_from_standard(self, ar):
        # Copy the observation parameters
        self.shared_As = ar.As.copy()
        self.shared_Vs = ar.Vs.copy()
        self.shared_bs = ar.bs.copy()

        for g in range(self.G):
            self.As[g] = ar.As.copy()
            self.Vs[g] = ar.Vs.copy()
            self.bs[g] = ar.bs.copy()
            self.inv_sigmas[g] = ar.inv_sigmas.copy()
                    
    def log_prior(self):
        lp = 0
        for g in range(self.G):
            lp += np.sum(norm.logpdf(self.As[g], self.shared_As, np.sqrt(self.eta)))
            lp += np.sum(norm.logpdf(self.bs[g], self.shared_bs, np.sqrt(self.eta)))
            lp += np.sum(norm.logpdf(self.Vs[g], self.shared_Vs, np.sqrt(self.eta)))
        return lp
                    
    def _compute_mus(self, data, input, mask, tag):
        T, D = data.shape
        As, bs, Vs = self.As[tag], self.bs[tag], self.Vs[tag]

        # Instantaneous inputs
        mus = np.matmul(Vs[None, ...], input[self.lags:, None, :self.M, None])[:, :, :, 0]

        # Lagged data
        for l in range(self.lags):
            mus = mus + As[:, :, l] * data[self.lags-l-1:-l-1, None, :]

        # Bias
        mus = mus + bs

        # Pad with the initial condition
        mus = np.concatenate((self.mu_init * np.ones((self.lags, self.K, self.D)), mus))

        assert mus.shape == (T, self.K, D)
        return mus

    def _compute_sigmas(self, data, input, mask, tag):
        T, D = data.shape
        inv_sigmas = self.inv_sigmas[tag]
        
        sigma_init = np.exp(self.inv_sigma_init) * np.ones((self.lags, self.K, self.D))
        sigma_ar = np.repeat(np.exp(inv_sigmas)[None, :, :], T-self.lags, axis=0)
        sigmas = np.concatenate((sigma_init, sigma_ar))
        assert sigmas.shape == (T, self.K, D)
        return sigmas

    def log_likelihoods(self, data, input, mask, tag):
        mus = self._compute_mus(data, input, mask, tag)
        sigmas = self._compute_sigmas(data, input, mask, tag)
        ll = -0.5 * (np.log(2 * np.pi * sigmas) + (data[:, None, :] - mus)**2 / sigmas) 
        return np.sum(ll * mask[:, None, :], axis=2)

    def _m_step_group(self, g, expectations, datas, inputs, masks, tags):
        G, K, D, M = self.G, self.K, self.D, self.M
        for d in range(D):
            # Collect data for this dimension
            xs, ys, weights = [], [], []
            for (Ez, _), data, input, mask, tag in zip(expectations, datas, inputs, masks, tags):
                # Only use data from current group
                if tag != g:
                    continue 

                # Only use data if it is complete
                if not np.all(mask[:, d]):
                    continue 

                xs.append(
                    np.hstack([data[self.lags-l-1:-l-1, d:d+1] for l in range(self.lags)] 
                              + [input[self.lags:, :self.M], np.ones((data.shape[0]-self.lags, 1))]))
                ys.append(data[self.lags:, d])
                weights.append(Ez[self.lags:])

            # Combine observations
            if len(xs) == 0:
                self.As[g, :, d] = 1.0
                self.Vs[g, :, d] = 0
                self.bs[g, :, d] = 0
                self.inv_sigmas[g, :, d] = 0
                continue

            xs = np.concatenate(xs)
            ys = np.concatenate(ys)
            weights = np.concatenate(weights)

            # Otherwise, fit a weighted linear regression for each discrete state
            for k in range(K):
                # Check for zero weights (singular matrix)
                if np.sum(weights[:, k]) < self.lags + M + 1:
                    self.As[g, k, d] = 1.0
                    self.Vs[g, k, d] = 0
                    self.bs[g, k, d] = 0
                    self.inv_sigmas[g, k, d] = 0
                    continue

                Jk = 1 / self.eta * np.eye(self.lags + self.M + 1)
                hk = 1 / self.eta * np.concatenate((self.shared_As[k, d], 
                                                    self.shared_Vs[k, d], 
                                                    [self.shared_bs[k, d]]))

                sigma = np.exp(self.inv_sigmas[g, k, d])
                Jk += np.sum(weights[:, k][:, None, None] * xs[:,:,None] * xs[:, None,:], axis=0) / sigma
                hk += np.sum(weights[:, k][:, None] * xs * ys[:, None], axis=0) / sigma

                muk = np.linalg.solve(Jk, hk)
                self.As[g, k, d] = muk[:self.lags]
                self.Vs[g, k, d] = muk[self.lags:self.lags+M]
                self.bs[g, k, d] = muk[-1]

                # Update the variances
                yhats = xs.dot(muk)
                sqerr = (ys - yhats)**2
                self.inv_sigmas[g, k, d] = np.log(np.average(sqerr, weights=weights[:,k], axis=0) + 1e-16)

    def _m_step_shared(self, expectations, datas, inputs, masks, tags):
        G, K, D, M = self.G, self.K, self.D, self.M
        for d in range(D):
            valid = [np.all(mask[:,d]) for mask in masks]
            valid_tags = [tag for tag, valid in zip(tags, valid) if valid]
            used = np.bincount(valid_tags, minlength=G) > 0
            self.shared_As[:, d, :] = np.mean(self.As[used, :, d, :], axis=0)
            self.shared_Vs[:, d, :] = np.mean(self.Vs[used, :, d, :], axis=0)
            self.shared_bs[:, d] = np.mean(self.bs[used, :, d], axis=0)

    def m_step(self, expectations, datas, inputs, masks, tags, num_iter=1, **kwargs):
        G, K, D, M = self.G, self.K, self.D, self.M

        for itr in range(num_iter):
            # Update the per-group weights
            for g in range(G):
                self._m_step_group(g, expectations, datas, inputs, masks, tags)

            # Update the shared weights
            self._m_step_shared(expectations, datas, inputs, masks, tags)

    def sample_x(self, z, xhist, input=None, tag=0, with_noise=True):
        D, As, bs, sigmas = self.D, self.As, self.bs, np.exp(self.inv_sigmas)
        if xhist.shape[0] < self.lags:
            sigma_init = np.exp(self.inv_sigma_init) if with_noise else 0
            return self.mu_init + np.sqrt(sigma_init) * npr.randn(D)
        else:
            mu = bs[tag, z].copy()
            for l in range(self.lags):
                mu += As[tag, z,:,l] * xhist[-l-1]

            sigma = sigmas[tag, z] if with_noise else 0
            return mu + np.sqrt(sigma) * npr.randn(D)

    def smooth(self, expectations, data, input, tag):
        """
        Compute the mean observation under the posterior distribution
        of latent discrete states.
        """
        T = expectations.shape[0]
        mask = np.ones((T, self.D), dtype=bool) 
        mus = self._compute_mus(data, input, mask, tag)
        return (expectations[:, :, None] * mus).sum(1)


class HierarchicalAutoRegressiveObservations(_Observations):
    def __init__(self, K, D, G=1, M=0, lags=1, eta=0.1):
        super(HierarchicalAutoRegressiveObservations, self).__init__(K, D, M)
        
        # Distribution over initial point
        self.mu_init = np.zeros(D)
        self.inv_sigma_init = np.zeros(D)
        
        # Global AR parameters
        assert lags > 0 
        self.lags = lags
        self.shared_As = .95 * np.array([
            np.column_stack([random_rotation(D), np.zeros((D, (lags-1) * D))]) 
            for _ in range(K)])
        self.shared_bs = npr.randn(K, D)
        self.shared_Vs = npr.randn(K, D, M)
        
        # Per-group AR parameters
        self.G = G
        self.eta = eta
        self.As = .95 * np.array([
                [np.column_stack([random_rotation(D), np.zeros((D, (lags-1) * D))]) for _ in range(K)]
                for _ in range(G)]
            )
        self.bs = npr.randn(G, K, D)
        self.Vs = npr.randn(G, K, D, M)
        self.inv_sigmas = -4 + npr.randn(G, K, D)

    @property
    def params(self):
        return self.shared_As, self.shared_bs, self.shared_Vs, \
               self.As, self.bs, self.Vs, self.inv_sigmas
        
    @params.setter
    def params(self, value):
        self.shared_As, self.shared_bs, self.shared_Vs, \
        self.As, self.bs, self.Vs, self.inv_sigmas = value
        
    def permute(self, perm):
        self.shared_As = self.shared_As[perm]
        self.shared_bs = self.shared_bs[perm]
        self.shared_Vs = self.shared_Vs[perm]

        for g in range(self.G):
            self.As[g] = self.As[g, perm]
            self.bs[g] = self.bs[g, perm]
            self.Vs[g] = self.Vs[g, perm]
            self.inv_sigmas[g] = self.inv_sigmas[g, perm]

    def initialize(self, datas, inputs=None, masks=None, tags=None):
        # Initialize with linear regressions
        from sklearn.linear_model import LinearRegression
        data = np.concatenate(datas) 
        input = np.concatenate(inputs)
        T = data.shape[0]

        # Initialize with linear regressions
        from sklearn.linear_model import LinearRegression
        data = np.concatenate(datas) 
        input = np.concatenate(inputs)
        T = data.shape[0]

        for k in range(self.K):
            ts = npr.choice(T-self.lags, replace=False, size=(T-self.lags)//self.K)
            x = np.column_stack([data[ts + l] for l in range(self.lags)] + [input[ts]])
            y = data[ts+self.lags]
            lr = LinearRegression().fit(x, y)
            self.shared_As[k] = lr.coef_[:, :self.D * self.lags]
            self.shared_Vs[k] = lr.coef_[:, self.D * self.lags:]
            self.shared_bs[k] = lr.intercept_
            
            for g in range(self.G):
                    self.As[g, k] = self.shared_As[k]
                    self.bs[g, k] = self.shared_bs[k]
                    self.Vs[g, k] = self.shared_Vs[k]

            resid = y - lr.predict(x)
            sigmas = np.var(resid, axis=0)
            for g in range(self.G):
                self.inv_sigmas[g, k] = np.log(sigmas + 1e-8)

    def initialize_from_standard(self, ar):
        # Copy the observation parameters
        self.shared_As = ar.As.copy()
        self.shared_Vs = ar.Vs.copy()
        self.shared_bs = ar.bs.copy()

        for g in range(self.G):
            self.As[g] = ar.As.copy()
            self.Vs[g] = ar.Vs.copy()
            self.bs[g] = ar.bs.copy()
            self.inv_sigmas[g] = ar.inv_sigmas.copy()

    def log_prior(self):
        lp = 0
        for g in range(self.G):
            lp += np.sum(norm.logpdf(self.As[g], self.shared_As, np.sqrt(self.eta)))
            lp += np.sum(norm.logpdf(self.bs[g], self.shared_bs, np.sqrt(self.eta)))
            lp += np.sum(norm.logpdf(self.Vs[g], self.shared_Vs, np.sqrt(self.eta)))
        return lp
                    
    def _compute_mus(self, data, input, mask, tag):
        T, D = data.shape
        As, bs, Vs = self.As[tag], self.bs[tag], self.Vs[tag]

        # Instantaneous inputs
        mus = np.matmul(Vs[None, ...], input[self.lags:, None, :self.M, None])[:, :, :, 0]

        # Lagged data
        for l in range(self.lags):
            mus = mus + np.matmul(As[None, :, :, l*D:(l+1)*D], 
                                  data[self.lags-l-1:-l-1, None, :, None])[:, :, :, 0]

        # Bias
        mus = mus + bs

        # Pad with the initial condition
        mus = np.concatenate((self.mu_init * np.ones((self.lags, self.K, self.D)), mus))

        assert mus.shape == (T, self.K, D)
        return mus

    def _compute_sigmas(self, data, input, mask, tag):
        T, D = data.shape
        inv_sigmas = self.inv_sigmas[tag]
        
        sigma_init = np.exp(self.inv_sigma_init) * np.ones((self.lags, self.K, self.D))
        sigma_ar = np.repeat(np.exp(inv_sigmas)[None, :, :], T-self.lags, axis=0)
        sigmas = np.concatenate((sigma_init, sigma_ar))
        assert sigmas.shape == (T, self.K, D)
        return sigmas

    def log_likelihoods(self, data, input, mask, tag):
        mus = self._compute_mus(data, input, mask, tag)
        sigmas = self._compute_sigmas(data, input, mask, tag)
        ll = -0.5 * (np.log(2 * np.pi * sigmas) + (data[:, None, :] - mus)**2 / sigmas) 
        return np.sum(ll * mask[:, None, :], axis=2)

    def _m_step_group(self, g, expectations, datas, inputs, masks, tags):
        G, K, D, M, lags, eta = self.G, self.K, self.D, self.M, self.lags, self.eta
        # Collect data for this dimension
        xs, ys, weights = [], [], []
        for (Ez, _), data, input, mask, tag in zip(expectations, datas, inputs, masks, tags):
            # Only use data from current group
            if tag != g:
                continue 

            # Only use data if it is complete
            if not np.all(mask):
                raise Exception("Encountered missing data in HierarchicalAutoRegressiveObservations!") 

            xs.append(
                np.hstack([data[self.lags-l-1:-l-1] for l in range(self.lags)] 
                          + [input[self.lags:, :self.M], np.ones((data.shape[0]-self.lags, 1))]))
            ys.append(data[self.lags:])
            weights.append(Ez[self.lags:])

        # Combine observations
        if len(xs) == 0:
            self.As[g] = 0
            self.Vs[g] = 0
            self.bs[g] = 0
            self.inv_sigmas[g] = 0

        xs = np.concatenate(xs)
        ys = np.concatenate(ys)
        weights = np.concatenate(weights)

        # Otherwise, fit a weighted linear regression for each discrete state
        for k in range(K):
            # Check for zero weights (singular matrix)
            if np.sum(weights[:, k]) < D * lags + M + 1:
                self.As[g, k] = self.shared_As[k]
                self.Vs[g, k] = self.shared_Vs[k]
                self.bs[g, k] = self.shared_bs[k]
                self.inv_sigmas[g, k] = 0
                continue

            # Update each row of the AR matrix
            for d in range(D):
                Jk = 1 / eta * np.eye(D * lags + M + 1)
                hk = 1 / eta * np.concatenate((self.shared_As[k, d], self.shared_Vs[k, d], [self.shared_bs[k, d]]))

                sigma = np.exp(self.inv_sigmas[g, k, d])
                Jk += 1 / sigma * np.sum(weights[:, k][:, None, None] * xs[:,:,None] * xs[:, None,:], axis=0)
                hk += 1 / sigma * np.sum(weights[:, k][:, None] * xs * ys[:, d:d+1], axis=0)

                muk = np.linalg.solve(Jk, hk)
                self.As[g, k, d] = muk[:D*lags]
                self.Vs[g, k, d] = muk[D*lags:D*lags+M]
                self.bs[g, k, d] = muk[-1]

                # Update the variances
                yhats = xs.dot(muk)
                sqerr = (ys[:, d] - yhats)**2
                self.inv_sigmas[g, k, d] = np.log(np.average(sqerr, weights=weights[:,k], axis=0) + 1e-16)

    def _m_step_shared(self, expectations, datas, inputs, masks, tags):
        G, K, D, M = self.G, self.K, self.D, self.M
        for d in range(D):
            valid = [np.all(mask[:,d]) for mask in masks]
            valid_tags = [tag for tag, valid in zip(tags, valid) if valid]
            used = np.bincount(valid_tags, minlength=G) > 0
            self.shared_As[:, d, :] = np.mean(self.As[used, :, d, :], axis=0)
            self.shared_Vs[:, d, :] = np.mean(self.Vs[used, :, d, :], axis=0)
            self.shared_bs[:, d] = np.mean(self.bs[used, :, d], axis=0)

    def m_step(self, expectations, datas, inputs, masks, tags, num_iter=1, **kwargs):
        G, K, D, M = self.G, self.K, self.D, self.M

        for itr in range(num_iter):
            # Update the per-group weights
            for g in range(G):
                self._m_step_group(g, expectations, datas, inputs, masks, tags)

            # Update the shared weights
            self._m_step_shared(expectations, datas, inputs, masks, tags)

    def sample_x(self, z, xhist, input=None, tag=0, with_noise=True):
        D, As, bs, sigmas = self.D, self.As, self.bs, np.exp(self.inv_sigmas)
        if xhist.shape[0] < self.lags:
            sigma_init = np.exp(self.inv_sigma_init) if with_noise else 0
            return self.mu_init + np.sqrt(sigma_init) * npr.randn(D)
        else:
            mu = bs[tag, z].copy()
            for l in range(self.lags):
                mu += As[tag, z][:,l*D:(l+1)*D].dot(xhist[-l-1])

            sigma = sigmas[tag, z] if with_noise else 0
            return mu + np.sqrt(sigma) * npr.randn(D)

    def smooth(self, expectations, data, input, tag):
        """
        Compute the mean observation under the posterior distribution
        of latent discrete states.
        """
        T = expectations.shape[0]
        mask = np.ones((T, self.D), dtype=bool) 
        mus = self._compute_mus(data, input, mask, tag)
        return (expectations[:, :, None] * mus).sum(1)


class HierarchicalRobustAutoRegressiveObservations(_Observations):
    def __init__(self, K, D, G=1, M=0, lags=1, eta=0.1):
        super(HierarchicalRobustAutoRegressiveObservations, self).__init__(K, D, M)
        
        # Distribution over initial point
        self.mu_init = np.zeros(D)
        self.inv_sigma_init = np.zeros(D)
        
        # Global AR parameters
        assert lags > 0 
        self.lags = lags
        self.shared_As = .95 * np.array([
            np.column_stack([random_rotation(D), np.zeros((D, (lags-1) * D))]) 
            for _ in range(K)])
        self.shared_bs = npr.randn(K, D)
        self.shared_Vs = npr.randn(K, D, M)
        
        # Per-group AR parameters
        self.G = G
        self.eta = eta
        self.As = .95 * np.array([
                [np.column_stack([random_rotation(D), np.zeros((D, (lags-1) * D))]) for _ in range(K)]
                for _ in range(G)]
            )
        self.bs = npr.randn(G, K, D)
        self.Vs = npr.randn(G, K, D, M)
        self.inv_sigmas = -4 + npr.randn(G, K, D)
        self.inv_nus = np.log(4) * np.ones((G, K))

    @property
    def params(self):
        return self.shared_As, self.shared_bs, self.shared_Vs, \
               self.As, self.bs, self.Vs, self.inv_sigmas, self.inv_nus
        
    @params.setter
    def params(self, value):
        self.shared_As, self.shared_bs, self.shared_Vs, \
        self.As, self.bs, self.Vs, self.inv_sigmas, self.inv_nus = value
        
    def permute(self, perm):
        self.shared_As = self.shared_As[perm]
        self.shared_bs = self.shared_bs[perm]
        self.shared_Vs = self.shared_Vs[perm]

        for g in range(self.G):
            self.As[g] = self.As[g, perm]
            self.bs[g] = self.bs[g, perm]
            self.Vs[g] = self.Vs[g, perm]
            self.inv_sigmas[g] = self.inv_sigmas[g, perm]
            self.inv_nus[g] = self.inv_nus[g, perm]

    def initialize(self, datas, inputs=None, masks=None, tags=None):
        # Initialize with linear regressions
        from sklearn.linear_model import LinearRegression
        data = np.concatenate(datas) 
        input = np.concatenate(inputs)
        T = data.shape[0]

        # Initialize with linear regressions
        from sklearn.linear_model import LinearRegression
        data = np.concatenate(datas) 
        input = np.concatenate(inputs)
        T = data.shape[0]

        for k in range(self.K):
            ts = npr.choice(T-self.lags, replace=False, size=(T-self.lags)//self.K)
            x = np.column_stack([data[ts + l] for l in range(self.lags)] + [input[ts]])
            y = data[ts+self.lags]
            lr = LinearRegression().fit(x, y)
            self.shared_As[k] = lr.coef_[:, :self.D * self.lags]
            self.shared_Vs[k] = lr.coef_[:, self.D * self.lags:]
            self.shared_bs[k] = lr.intercept_
            
            for g in range(self.G):
                    self.As[g, k] = self.shared_As[k]
                    self.bs[g, k] = self.shared_bs[k]
                    self.Vs[g, k] = self.shared_Vs[k]

            resid = y - lr.predict(x)
            sigmas = np.var(resid, axis=0)
            for g in range(self.G):
                self.inv_sigmas[g, k] = np.log(sigmas + 1e-8)

    def initialize_from_standard(self, ar):
        # Copy the observation parameters
        self.shared_As = ar.As.copy()
        self.shared_Vs = ar.Vs.copy()
        self.shared_bs = ar.bs.copy()

        for g in range(self.G):
            self.As[g] = ar.As.copy()
            self.Vs[g] = ar.Vs.copy()
            self.bs[g] = ar.bs.copy()
            self.inv_sigmas[g] = ar.inv_sigmas.copy()
            self.inv_nus[g] = ar.inv_nus.copy()

    def log_prior(self):
        lp = 0
        for g in range(self.G):
            lp += np.sum(norm.logpdf(self.As[g], self.shared_As, np.sqrt(self.eta)))
            lp += np.sum(norm.logpdf(self.bs[g], self.shared_bs, np.sqrt(self.eta)))
            lp += np.sum(norm.logpdf(self.Vs[g], self.shared_Vs, np.sqrt(self.eta)))
        return lp
                    
    def _compute_mus(self, data, input, mask, tag):
        T, D = data.shape
        As, bs, Vs = self.As[tag], self.bs[tag], self.Vs[tag]

        # Instantaneous inputs
        mus = np.matmul(Vs[None, ...], input[self.lags:, None, :self.M, None])[:, :, :, 0]

        # Lagged data
        for l in range(self.lags):
            mus = mus + np.matmul(As[None, :, :, l*D:(l+1)*D], 
                                  data[self.lags-l-1:-l-1, None, :, None])[:, :, :, 0]

        # Bias
        mus = mus + bs

        # Pad with the initial condition
        mus = np.concatenate((self.mu_init * np.ones((self.lags, self.K, self.D)), mus))

        assert mus.shape == (T, self.K, D)
        return mus

    # def _compute_sigmas(self, data, input, mask, tag):
    #     T, D = data.shape
    #     inv_sigmas = self.inv_sigmas[tag]
        
    #     sigma_init = np.exp(self.inv_sigma_init) * np.ones((self.lags, self.K, self.D))
    #     sigma_ar = np.repeat(np.exp(inv_sigmas)[None, :, :], T-self.lags, axis=0)
    #     sigmas = np.concatenate((sigma_init, sigma_ar))
    #     assert sigmas.shape == (T, self.K, D)
    #     return sigmas

    # def log_likelihoods(self, data, input, mask, tag):
    #     D = self.D
    #     mus = self._compute_mus(data, input, mask, tag)
    #     sigmas = self._compute_sigmas(data, input, mask, tag)
    #     nus = np.exp(self.inv_nus)[tag]

    #     resid = data[:, None, :] - mus
    #     z = resid / sigmas
    #     return -0.5 * (nus + D) * np.log(1.0 + (resid * z).sum(axis=2) / nus) + \
    #         gammaln((nus + D) / 2.0) - gammaln(nus / 2.0) - D / 2.0 * np.log(nus) \
    #         -D / 2.0 * np.log(np.pi) - 0.5 * np.sum(np.log(sigmas), axis=-1)

    def log_likelihoods(self, data, input, mask, tag):
        D = self.D
        mus = self._compute_mus(data, input, mask, tag)
        nus = np.exp(self.inv_nus)[tag]
        sigma_init = np.exp(self.inv_sigma_init)
        sigma_ar = np.exp(self.inv_sigmas[tag])
        
        resid = data[:, None, :] - mus

        # Handle the initial datapoints separate from the rest
        lls_init = -0.5 * (nus + D) * np.log(1.0 + (resid[:self.lags]**2 / sigma_init).sum(axis=2) / nus) + \
            gammaln((nus + D) / 2.0) - gammaln(nus / 2.0) - D / 2.0 * np.log(nus) \
            -D / 2.0 * np.log(np.pi) - 0.5 * np.sum(np.log(sigma_init), axis=-1)

        lls_ar = -0.5 * (nus + D) * np.log(1.0 + (resid[self.lags:]**2 / sigma_ar).sum(axis=2) / nus) + \
            gammaln((nus + D) / 2.0) - gammaln(nus / 2.0) - D / 2.0 * np.log(nus) \
            -D / 2.0 * np.log(np.pi) - 0.5 * np.sum(np.log(sigma_ar), axis=-1)

        return np.vstack((lls_init, lls_ar))

    def sample_x(self, z, xhist, input=None, tag=0, with_noise=True):
        D, As, bs, sigmas = self.D, self.As, self.bs, np.exp(self.inv_sigmas)
        if xhist.shape[0] < self.lags:
            sigma_init = np.exp(self.inv_sigma_init) if with_noise else 0
            return self.mu_init + np.sqrt(sigma_init) * npr.randn(D)
        else:
            mu = bs[tag, z].copy()
            for l in range(self.lags):
                mu += As[tag, z][:,l*D:(l+1)*D].dot(xhist[-l-1])

            sigma = sigmas[tag, z] if with_noise else 0
            return mu + np.sqrt(sigma) * npr.randn(D)

    def smooth(self, expectations, data, input, tag):
        """
        Compute the mean observation under the posterior distribution
        of latent discrete states.
        """
        T = expectations.shape[0]
        mask = np.ones((T, self.D), dtype=bool) 
        mus = self._compute_mus(data, input, mask, tag)
        return (expectations[:, :, None] * mus).sum(1)
