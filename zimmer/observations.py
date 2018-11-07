import copy
import warnings

import autograd.numpy as np
import autograd.numpy.random as npr
from autograd.scipy.misc import logsumexp
from autograd.scipy.special import gammaln, digamma
from autograd.scipy.stats import norm, gamma
from autograd import grad

from ssm.observations import _Observations
from ssm.util import random_rotation, ensure_args_are_lists, ensure_args_not_none, \
    logistic, logit, one_hot, generalized_newton_studentst_dof
from ssm.preprocessing import interpolate_data
from ssm.cstats import robust_ar_statistics


class HierarchicalAutoRegressiveObservations(_Observations):
    def __init__(self, K, D, M=0, tags=(None,), lags=1, eta=0.1):
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
        self.tags = tags
        self.tags_to_indices = dict([(tag, i) for i, tag in enumerate(tags)])
        self.G = len(tags)
        assert self.G > 0

        # Per-group AR parameters
        self.eta = eta
        self.As = .95 * np.array([
                [np.column_stack([random_rotation(D), np.zeros((D, (lags-1) * D))]) for _ in range(K)]
                for _ in range(self.G)]
            )
        self.bs = npr.randn(self.G, K, D)
        self.Vs = npr.randn(self.G, K, D, M)
        self.inv_sigmas = -4 + npr.randn(self.G, K, D)

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
        g = self.tags_to_indices[tag]
        As, bs, Vs = self.As[g], self.bs[g], self.Vs[g]

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
        g = self.tags_to_indices[tag]
        inv_sigmas = self.inv_sigmas[g]
        
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
        for (Ez, _, _), data, input, mask, tag in zip(expectations, datas, inputs, masks, tags):
            # Only use data from current group
            if self.tags_to_indices[tag] != g:
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
            valid_tags = [self.tags_to_indices[tag] for tag, valid in zip(tags, valid) if valid]
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
            g = self.tags_to_indices[tag]
            mu = bs[g, z].copy()
            for l in range(self.lags):
                mu += As[g, z][:,l*D:(l+1)*D].dot(xhist[-l-1])

            sigma = sigmas[g, z] if with_noise else 0
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
    def __init__(self, K, D, tags=(None,), M=0, lags=1, eta=0.1):
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
        self.tags = tags
        self.tags_to_indices = dict([(tag, i) for i, tag in enumerate(tags)])
        self.G = len(tags)
        assert self.G > 0

        self.eta = eta
        self.As = .95 * np.array([
                [np.column_stack([random_rotation(D), np.zeros((D, (lags-1) * D))]) for _ in range(K)]
                for _ in range(self.G)]
            )
        self.bs = npr.randn(self.G, K, D)
        self.Vs = npr.randn(self.G, K, D, M)
        self.inv_sigmas = -4 + npr.randn(self.G, K, D)
        self.inv_nus = np.log(4) * np.ones((self.G, K))

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
        ind = self.tags_to_indices[tag]
        As, bs, Vs = self.As[ind], self.bs[ind], self.Vs[ind]

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
        ind = self.tags_to_indices[tag]
        inv_sigmas = self.inv_sigmas[ind]
        
        sigma_init = np.exp(self.inv_sigma_init) * np.ones((self.lags, self.K, self.D))
        sigma_ar = np.repeat(np.exp(inv_sigmas)[None, :, :], T-self.lags, axis=0)
        sigmas = np.concatenate((sigma_init, sigma_ar))
        assert sigmas.shape == (T, self.K, D)
        return sigmas

    def log_likelihoods(self, data, input, mask, tag):
        D = self.D
        ind = self.tags_to_indices[tag]
        mus = self._compute_mus(data, input, mask, tag)
        nus = np.exp(self.inv_nus)[ind]
        sigma_init = np.exp(self.inv_sigma_init)
        sigma_ar = np.exp(self.inv_sigmas[ind])
        
        resid = data[:, None, :] - mus

        # Handle the initial datapoints separate from the rest
        lls_init = -0.5 * (nus + D) * np.log(1.0 + (resid[:self.lags]**2 / sigma_init).sum(axis=2) / nus) + \
            gammaln((nus + D) / 2.0) - gammaln(nus / 2.0) - D / 2.0 * np.log(nus) \
            -D / 2.0 * np.log(np.pi) - 0.5 * np.sum(np.log(sigma_init), axis=-1)

        lls_ar = -0.5 * (nus + D) * np.log(1.0 + (resid[self.lags:]**2 / sigma_ar).sum(axis=2) / nus) + \
            gammaln((nus + D) / 2.0) - gammaln(nus / 2.0) - D / 2.0 * np.log(nus) \
            -D / 2.0 * np.log(np.pi) - 0.5 * np.sum(np.log(sigma_ar), axis=-1)

        return np.vstack((lls_init, lls_ar))

    def _m_step_ar(self, g, expectations, datas, inputs, masks, tags):
        K, D, M, lags, eta = self.K, self.D, self.M, self.lags, self.eta

        # Collect data for this group
        xs, ys, Ezs = [], [], []
        for (Ez, _, _), data, input, mask, tag in zip(expectations, datas, inputs, masks, tags):
            if self.tags_to_indices[tag] != g:
                continue

            # Only use data if it is complete
            if not np.all(mask):
                raise Exception("Encountered missing data in AutoRegressiveObservations!") 

            xs.append(
                np.hstack([data[self.lags-l-1:-l-1] for l in range(self.lags)] 
                          + [input[self.lags:, :self.M], np.ones((data.shape[0]-self.lags, 1))]))
            ys.append(data[self.lags:])
            Ezs.append(Ez[self.lags:])

        # Compute expected precision for each data point given current parameters
        taus = []
        for x, y in zip(xs, ys):
            Afull = np.concatenate((self.As[g], self.Vs[g], self.bs[g, :, :, None]), axis=2)
            mus = np.matmul(Afull[None, :, :, :], x[:, None, :, None])[:, :, :, 0]
            sigmas = np.exp(self.inv_sigmas[g])
            nus = np.exp(self.inv_nus[g, :, None])

            # nu: (K,)  mus: (T, K, D)  sigmas: (K, D)  y: (T, D)  -> tau: (T, K, D)
            alpha = nus/2 + 1/2
            beta = nus/2 + 1/2 * (y[:, None, :] - mus)**2 / sigmas
            taus.append(alpha / beta)

        # Fit the weighted linear regressions for each K and D
        J = 1 / eta * np.tile((np.eye(D * lags + M + 1))[None, None, :, :], (K, D, 1, 1))
        h = 1 / eta * np.concatenate((self.shared_As, self.shared_Vs, self.shared_bs[:, :, None]), axis=2)
        for x, y, Ez, tau in zip(xs, ys, Ezs, taus):
            robust_ar_statistics(Ez, tau, x, y, J, h)

        mus = np.linalg.solve(J, h)
        self.As[g] = mus[:, :, :D*lags]
        self.Vs[g] = mus[:, :, D*lags:D*lags+M]
        self.bs[g] = mus[:, :, -1]

        # Fit the variance
        sqerr = 0
        weight = 0
        for x, y, Ez, tau in zip(xs, ys, Ezs, taus):
            yhat = np.matmul(x[None, :, :], np.swapaxes(mus, -1, -2))
            sqerr += np.einsum('tk, tkd, ktd -> kd', Ez, tau, (y - yhat)**2)
            weight += np.sum(Ez, axis=0)
        self.inv_sigmas[g] = np.log(sqerr / weight[:, None] + 1e-16)

    def _m_step_nu(self, g, expectations, datas, inputs, masks, tags):
        K, D = self.K, self.D
        E_taus = np.zeros(K)
        E_logtaus = np.zeros(K)
        weights = np.zeros(K)
        for (Ez, _, _,), data, input, mask, tag in zip(expectations, datas, inputs, masks, tags):
            if self.tags_to_indices[tag] != g:
                continue 

            # nu: (K,)  mus: (K, D)  sigmas: (K, D)  y: (T, D)  -> w: (T, K, D)
            mus = self._compute_mus(data, input, mask, tag)
            sigmas = self._compute_sigmas(data, input, mask, tag)
            nus = np.exp(self.inv_nus[g, :, None])

            alpha = nus/2 + 1/2
            beta = nus/2 + 1/2 * (data[:, None, :] - mus)**2 / sigmas
            
            E_taus += np.sum(Ez[:, :, None] * alpha / beta, axis=(0, 2))
            E_logtaus += np.sum(Ez[:, :, None] * (digamma(alpha) - np.log(beta)), axis=(0, 2))
            weights += np.sum(Ez, axis=0) * D

        E_taus /= weights
        E_logtaus /= weights

        for k in range(K):
            self.inv_nus[g, k] = np.log(generalized_newton_studentst_dof(E_taus[k], E_logtaus[k]) + 1e-16)

    def _m_step_shared(self, expectations, datas, inputs, masks, tags):
        G, K, D, M = self.G, self.K, self.D, self.M
        for d in range(D):
            valid = [np.all(mask[:,d]) for mask in masks]
            valid_indss = [self.tags_to_indices[tag] for tag, valid in zip(tags, valid) if valid]
            used = np.bincount(valid_indss, minlength=G) > 0
            self.shared_As[:, d, :] = np.mean(self.As[used, :, d, :], axis=0)
            self.shared_Vs[:, d, :] = np.mean(self.Vs[used, :, d, :], axis=0)
            self.shared_bs[:, d] = np.mean(self.bs[used, :, d], axis=0)

    def m_step(self, expectations, datas, inputs, masks, tags, num_iter=1, **kwargs):
        G, K, D, M = self.G, self.K, self.D, self.M

        for itr in range(num_iter):
            # Update the per-group weights
            for g in range(G):
                self._m_step_ar(g, expectations, datas, inputs, masks, tags)
                self._m_step_nu(g, expectations, datas, inputs, masks, tags)
            
            # Update the shared weights
            self._m_step_shared(expectations, datas, inputs, masks, tags)

    def sample_x(self, z, xhist, input=None, tag=0, with_noise=True):
        D, As, bs, sigmas = self.D, self.As, self.bs, np.exp(self.inv_sigmas)
        if xhist.shape[0] < self.lags:
            sigma_init = np.exp(self.inv_sigma_init) if with_noise else 0
            return self.mu_init + np.sqrt(sigma_init) * npr.randn(D)
        else:
            ind = self.tags_to_indices[tag]
            mu = bs[ind, z].copy()
            for l in range(self.lags):
                mu += As[ind, z][:,l*D:(l+1)*D].dot(xhist[-l-1])

            sigma = sigmas[ind, z] if with_noise else 0
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
