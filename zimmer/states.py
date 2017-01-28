# Build specific models for joint observations
import numpy as np

from pyslds.states import _SLDSStatesMaskedData, _SLDSStatesGibbs, HMMStatesEigen


class HierarchicalSLDSStates(_SLDSStatesMaskedData,
                             _SLDSStatesGibbs,
                             HMMStatesEigen):
    """
    Let's try a new approach in which hierarchical states just have a tag
    to indicate which group they come from.  The data will always be a big
    matrix with all possible observation dimensions.  Some (most) of the
    observations will be masked off though.

    All the smarts will go into the emission distribution.

    This approach will support all Gaussian or all count observations, but
    not a mix.


    """

    def __init__(self, model, group=None, **kwargs):
        self.group = group
        super(HierarchicalSLDSStates, self).__init__(model, **kwargs)

    @property
    def _info_emission_params_diag(self):
        if self.model._single_emission:

            C = self.emission_distns[0].A[self.group, :, :self.D_latent]
            D = self.emission_distns[0].A[self.group, :, self.D_latent:]
            CCT = np.array([np.outer(cp, cp) for cp in C]). \
                reshape((self.D_emission, self.D_latent ** 2))

            sigmasq = self.emission_distns[0].sigmasq_flat[self.group]
            J_obs = self.mask / sigmasq
            centered_data = self.data - self.inputs.dot(D.T)

            J_node = np.dot(J_obs, CCT)

            # h_node = y^T R^{-1} C - u^T D^T R^{-1} C
            h_node = (centered_data * J_obs).dot(C)

            log_Z_node = -self.mask.sum(1) / 2. * np.log(2 * np.pi)
            log_Z_node -= 1. / 2 * np.sum(self.mask * np.log(sigmasq), axis=1)
            log_Z_node -= 1. / 2 * np.sum(centered_data ** 2 * J_obs, axis=1)


        else:
            expand = lambda a: a[None, ...]
            stack_set = lambda x: np.concatenate(list(map(expand, x)))

            sigmasq_set = [d.sigmasq_flat[self.group] for d in self.emission_distns]
            sigmasq = stack_set(sigmasq_set)[self.stateseq]
            J_obs = self.mask / sigmasq

            C_set = [d.A[self.group, :, :self.D_latent] for d in self.emission_distns]
            D_set = [d.A[self.group, :, self.D_latent:] for d in self.emission_distns]
            CCT_set = [np.array([np.outer(cp, cp) for cp in C]).
                           reshape((self.D_emission, self.D_latent ** 2))
                       for C in C_set]

            J_node = np.zeros((self.T, self.D_latent ** 2))
            h_node = np.zeros((self.T, self.D_latent))
            log_Z_node = -self.mask.sum(1) / 2. * np.log(2 * np.pi) * np.ones(self.T)

            for i in range(len(self.emission_distns)):
                ti = np.where(self.stateseq == i)[0]
                centered_data_i = self.data[ti] - self.inputs[ti].dot(D_set[i].T)

                J_node[ti] = np.dot(J_obs[ti], CCT_set[i])
                h_node[ti] = (centered_data_i * J_obs[ti]).dot(C_set[i])

                log_Z_node[ti] -= 1. / 2 * np.sum(np.log(sigmasq_set[i]))
                log_Z_node[ti] -= 1. / 2 * np.sum(centered_data_i ** 2 * J_obs[ti], axis=1)

        J_node = J_node.reshape((self.T, self.D_latent, self.D_latent))
        return J_node, h_node, log_Z_node

    @property
    def _info_emission_params_dense(self):
        raise NotImplementedError

    @property
    def aBl(self):
        if self._aBl is None:
            aBl = self._aBl = np.zeros((self.T, self.num_states))
            ids, dds, eds = self.init_dynamics_distns, self.dynamics_distns, \
                            self.emission_distns

            for idx, (d1, d2) in enumerate(zip(ids, dds)):
                # Initial state distribution
                aBl[0, idx] = d1.log_likelihood(self.gaussian_states[0])

                # Dynamics
                xs = np.hstack((self.gaussian_states[:-1], self.inputs[:-1]))
                aBl[:-1, idx] = d2.log_likelihood((xs, self.gaussian_states[1:]))

            # Emissions
            xs = np.hstack((self.gaussian_states, self.inputs))
            if self.model._single_emission:
                d3 = self.emission_distns[0]
                if self.mask is None:
                    aBl += d3.log_likelihood((xs, self.data), group=self.group)[:,None]
                else:
                    aBl += d3.log_likelihood((xs, self.data), mask=self.mask, group=self.group)[:,None]
            else:
                for idx, d3 in enumerate(eds):
                    if self.mask is None:
                        aBl[:,idx] += d3.log_likelihood((xs, self.data), group=self.group)
                    else:
                        aBl[:,idx] += d3.log_likelihood((xs, self.data), mask=self.mask, group=self.group)

            aBl[np.isnan(aBl).any(1)] = 0.

        return self._aBl

from rslds.rslds import RecurrentSLDSStates
class HierarchicalRecurrentSLDSStates(HierarchicalSLDSStates, RecurrentSLDSStates):
    pass


