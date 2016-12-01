# Build specific models for joint observations
import numpy as np

from pybasicbayes.distributions import DiagonalRegression

from pyhsmm.internals.hmm_states import HMMStatesPython, HMMStatesEigen
from pyslds.states import _SLDSStatesGibbs

from pinkybrain.distributions import _PGLogisticRegressionBase


class _MultiEmissionSLDSStates(_SLDSStatesGibbs):
    """
    Incorporating two separate Gaussian observation streams.
    We just have to combine the two observations and compute
    the effective information form potential to give to the
    latent state sampler.

    As long as both the observations are linear and Gaussian,
    this should just be a matter of adding potentials.

    The major concern is: how do we weight the likelihoods so
    that the two streams are treated as equally important? If
    the behavioral video stream is 2000 pixels and the neural
    signal is 2 dimensional, then unless the precision is 1000
    times larger for the neural signal, the model is only
    incentivized to capture the behavioral video.

    NOTES:

        1. We assume the model emission_distns list is a set
           of distributions for each observation modality.
           These are shared by all discrete latent states.

        2. We need to handle missing data here too. Does each
           modality have its own mask?

        3. We can either take in the data as a hstack'ed array
           or as a list of arrays. I'm partial to the latter
           since it is more transparent; however, it also
           breaks with the pySLDS convention.

    """

    def __init__(self, model, data=None, T=None, inputs=None, mask=None, group=None, fixed_stateseq=None, **kwargs):
        # TODO: Fix up this initialization code.
        # TODO: It is way too complicated.

        self.model = model
        self.fixed_stateseq = fixed_stateseq

        if data is None:
            assert isinstance(T, int)
            self.T = T
            self.data = data
        else:
            if isinstance(data, list):
                self.data = data
            elif isinstance(data, np.ndarray) and len(self.emission_distns) == 1:
                self.data = [data]
            else:
                raise Exception("data must be a list, one array "
                                "for each observation modality")

            Ts = np.array(list(map(lambda d: d.shape[0] if d is not None else 0, self.data)))
            assert np.all((Ts == 0) | (np.isclose(Ts, np.max(Ts))))
            self.T = np.max(Ts)

        self.inputs = np.zeros((self.T, 0)) if inputs is None else inputs

        # Check for masks
        if mask is None:
            self.mask = [np.ones((self.T, ed.D_out), dtype=bool)
                         for ed in self.emission_distns]
        elif isinstance(mask, list):
            self.mask = mask
        elif isinstance(mask, np.ndarray) and len(self.emission_distns) == 1:
            self.mask = [mask]
        else:
            raise Exception("mask must be a list, one array "
                            "for each observation modality")

        if data is not None:
            assert isinstance(self.mask, list) and len(self.mask) == len(self.data)
            for m,d in zip(self.mask, self.data):
                if d is not None:
                    assert m.shape == d.shape
                    assert m.dtype == bool

        # Store the group -- important for hierarchical models
        assert group is None or isinstance(group, int)
        self.group = group

        # TODO: Allow for given states
        self.generate_states()

    @property
    def diagonal_noise(self):
        return True

    @property
    def N_output(self):
        return len(self.emission_distns)

    @property
    def Cs(self):
        raise Exception("Dual observation model does not have Cs")

    @property
    def DDTs(self):
        raise Exception("Dual observation model does not have DDTs")

    def generate_obs(self, with_noise=True):
        # Go through each time bin, get the discrete latent state,
        # use that to index into the emission_distns to get samples
        T= self.T
        dss, gss = self.stateseq, self.gaussian_states

        datas = [np.empty((T, ed.D_out)) for ed in self.emission_distns]
        for data, ed in zip(datas, self.emission_distns):
            for t in range(self.T):
                if with_noise:
                    data[t] = \
                        ed.rvs(x=np.hstack((gss[t][None, :], self.inputs[t][None,:])),
                               return_xy=False)
                else:
                    data[t] = \
                        ed.predict(np.hstack((gss[t][None,:], self.inputs[t][None,:])))
        return datas


    @property
    def info_emission_params(self):
        """
        Override the base class's emission params property.
        Here, we sum the potentials from each one of the observations.
        """
        J_node = np.zeros((self.T, self.D_latent**2))
        h_node = np.zeros((self.T, self.D_latent))
        for ed, d, m in zip(self.emission_distns, self.data, self.mask):
            if d is None:
                continue
            # Compute potential and add it
            J, h = self._compute_emission_params(ed, d, m)
            J_node += J
            h_node += h

        J_node = J_node.reshape((self.T, self.D_latent, self.D_latent))
        return J_node, h_node, 0

    def _compute_emission_params(self, emission_distn, data, mask):
        # TODO: Include inputs!
        if isinstance(emission_distn, DiagonalRegression):
            if self.group is not None:
                sigmasq = emission_distn.sigmasq_flat[self.group]
                C = emission_distn.A[self.group][:,:self.D_latent]
                D = emission_distn.A[self.group][:,self.D_latent:]
            else:
                sigmasq = emission_distn.sigmasq_flat
                C = emission_distn.A[:,:self.D_latent]
                D = emission_distn.A[:,self.D_latent:]

            J_obs = mask / sigmasq
            CCT = np.array([np.outer(cp, cp) for cp in C]). \
                reshape((emission_distn.D_out, self.D_latent ** 2))

            J_node = np.dot(J_obs, CCT)
            h_node = (data * J_obs).dot(C)
            if self.D_input > 0:
                h_node -= (self.inputs.dot(D.T) * J_obs).dot(C)

        else:
            raise Exception("Emission distribution class %s is not supported"
                            % emission_distn.__class__)
        return J_node, h_node

    @property
    def aBl(self):
        if self._aBl is None:
            aBl = self._aBl = np.zeros((self.T, self.num_states))
            ids, dds, eds = self.init_dynamics_distns, self.dynamics_distns, \
                            self.emission_distns

            # Dynamics distributions
            for idx, (d1, d2) in enumerate(zip(ids, dds)):
                aBl[0, idx] = d1.log_likelihood(self.gaussian_states[0])

                xs = np.hstack((self.gaussian_states[:-1], self.inputs[:-1]))
                aBl[:-1, idx] += d2.log_likelihood((xs, self.gaussian_states[1:]))

            # Emission distributions
            xs = np.hstack((self.gaussian_states, self.inputs))
            for ed, d, m in zip(self.emission_distns, self.data, self.mask):
                if d is None:
                    continue

                # TODO: Fix up this hackery
                if self.group is not None:
                    aBl += ed.log_likelihood((xs, d), group=self.group, mask=m)[:, None]
                else:
                    aBl += ed.log_likelihood((xs, d), mask=m)[:, None]

            aBl[np.isnan(aBl).any(1)] = 0.

        return self._aBl


    # def _set_gaussian_expected_stats(self, smoothed_mus, smoothed_sigmas, E_xtp1_xtT):
    #     if self.mask is None:
    #         return super(_SLDSStatesMaskedData, self). \
    #             _set_gaussian_expected_stats(smoothed_mus, smoothed_sigmas, E_xtp1_xtT)
    #
    #     assert not np.isnan(E_xtp1_xtT).any()
    #     assert not np.isnan(smoothed_mus).any()
    #     assert not np.isnan(smoothed_sigmas).any()
    #
    #     # Same as in parent class
    #     # this is like LDSStates._set_expected_states but doesn't sum over time
    #     T = self.T
    #     ExxT = self.ExxT = smoothed_sigmas \
    #                        + self.smoothed_mus[:, :, None] * self.smoothed_mus[:, None, :]
    #
    #     # Initial state stats
    #     self.E_init_stats = (self.smoothed_mus[0], ExxT[0], 1.)
    #
    #     # Dynamics stats
    #     # TODO avoid memory instantiation by adding to Regression (2TD vs TD^2)
    #     # TODO only compute EyyT once
    #     E_xtp1_xtp1T = self.E_xtp1_xtp1T = ExxT[1:]
    #     E_xt_xtT = self.E_xt_xtT = ExxT[:-1]
    #
    #     self.E_dynamics_stats = \
    #         (E_xtp1_xtp1T, E_xtp1_xtT, E_xt_xtT, np.ones(T - 1))
    #
    #     # Emission stats
    #     masked_data = self.data * self.mask if self.mask is not None else self.data
    #     if self.diagonal_noise:
    #         Eysq = self.EyyT = masked_data ** 2
    #         EyxT = self.EyxT = masked_data[:, :, None] * self.smoothed_mus[:, None, :]
    #         ExxT = self.mask[:, :, None, None] * ExxT[:, None, :, :]
    #         self.E_emission_stats = (Eysq, EyxT, ExxT, self.mask)
    #     else:
    #         raise Exception("Only DiagonalRegression currently supports missing data")
    #
    #     self._mf_aBl = None  # TODO

    def smooth(self):
        self.info_E_step()
        # TODO: Handle inputs
        return [self.smoothed_mus.dot(ed.A.T) for ed in self.emission_distns]


class MultiEmissionSLDSStatesPython(
    _MultiEmissionSLDSStates,
    HMMStatesPython):
    pass


class MultiEmissionSLDSStatesEigen(
    _MultiEmissionSLDSStates,
    HMMStatesEigen):
    pass
