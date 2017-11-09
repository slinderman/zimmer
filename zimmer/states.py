# Build specific models for joint observations
import numpy as np

from pybasicbayes.models.factor_analysis import FactorAnalysisStates
from pylds.states import LDSStatesMissingData
from pyslds.states import _SLDSStatesMaskedData, _SLDSStatesGibbs, _SLDSStatesVBEM
from pyhsmm.internals.hmm_states import HMMStatesEigen
from rslds.states import InputHMMStates


class HierarchicalFactorAnalysisStates(FactorAnalysisStates):
    def __init__(self, model, data, group=0, mask=None):
        self.group = group
        super(HierarchicalFactorAnalysisStates, self).__init__(model=model, data=data, mask=mask)

    @property
    def sigmasq(self):
        return self.model.regression.sigmasq_flat[self.group]


class HierarchicalLDSStates(LDSStatesMissingData):
    def __init__(self, model, data=None, mask=None, group=None, **kwargs):
        self.group = group
        super(HierarchicalLDSStates, self).__init__(model,data=data, mask=mask, **kwargs)

    @property
    def info_emission_params(self):
        if self.mask is None:
            return super(LDSStatesMissingData, self).info_emission_params

        assert self.diagonal_noise
        return self._info_emission_params_diag

    @property
    def _info_emission_params_diag(self):
        C, D = self.C, self.D
        sigmasq = self.emission_distn.sigmasq_flat[self.group]
        J_obs = self.mask / sigmasq
        centered_data = self.data - self.inputs.dot(D.T)

        CCT = np.array([np.outer(cp, cp) for cp in C]). \
            reshape((self.D_emission, self.D_latent ** 2))

        J_node = np.dot(J_obs, CCT)
        J_node = J_node.reshape((self.T, self.D_latent, self.D_latent))

        # h_node = y^T R^{-1} C - u^T D^T R^{-1} C
        h_node = (centered_data * J_obs).dot(C)

        log_Z_node = -self.mask.sum(1) / 2. * np.log(2 * np.pi)
        log_Z_node -= 1. / 2 * np.sum(self.mask * np.log(sigmasq), axis=1)
        log_Z_node -= 1. / 2 * np.sum(centered_data ** 2 * J_obs, axis=1)

        return J_node, h_node, log_Z_node


class _HierarchicalARHMMStatesMixin(object):
    def __init__(self, group=None, *args, **kwargs):
        self.group = group
        super(_HierarchicalARHMMStatesMixin, self).__init__(*args, **kwargs)

    @property
    def obs_distns(self):
        # Assume the model's obs_distns is a list of Hierarchical Regressions
        return [o.regressions[self.group] for o in self.model.obs_distns]

    def changepoint_probability(self):
        trans_potential = self.trans_matrix
        init_potential = self.pi_0
        likelihood_log_potential = self.aBl

        alphal = self._messages_forwards_log(
            trans_potential, init_potential,
            likelihood_log_potential)
        betal = self._messages_backwards_log(
            trans_potential, likelihood_log_potential)

        expected_states = alphal + betal
        expected_states -= expected_states.max(1)[:, None]
        np.exp(expected_states, out=expected_states)
        expected_states /= expected_states.sum(1)[:, None]

        Al = np.log(trans_potential)
        if Al.ndim == 2:
            Al = Al[None, :, :]

        log_joints = alphal[:-1, :, None] + \
                     (betal[1:, None, :] + \
                     likelihood_log_potential[1:, None, :]) + \
                     Al
        log_joints -= log_joints.max((1, 2))[:, None, None]
        joints = np.exp(log_joints)
        joints /= joints.sum((1, 2))[:, None, None]  # NOTE: renormalizing each isnt really necessary

        K = self.num_states
        stay_prob = joints[:, np.arange(K), np.arange(K)].sum(axis=1)
        return 1 - stay_prob

    def filter(self):
        trans_potential = self.trans_matrix
        init_potential = self.pi_0
        likelihood_log_potential = self.aBl

        alphal = self._messages_forwards_log(
            trans_potential, init_potential,
            likelihood_log_potential)

        expected_states = alphal
        expected_states -= expected_states.max(1)[:, None]
        np.exp(expected_states, out=expected_states)
        expected_states /= expected_states.sum(1)[:, None]
        return expected_states


class HierarchicalARHMMStates(_HierarchicalARHMMStatesMixin, HMMStatesEigen):
    pass


class HierarchicalRecurrentARHMMStates(_HierarchicalARHMMStatesMixin, InputHMMStates):
    pass


class _HierarchicalSLDSStatesMixin(object):
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
        super(_HierarchicalSLDSStatesMixin, self).__init__(model=model, **kwargs)

    ### Override properties with group-specific covariance
    @property
    def R_set(self):
        return np.concatenate([np.diag(d.sigmasq_flat[self.group])[None, ...]
                               for d in self.emission_distns])

    @property
    def Rinv_set(self):
        return np.concatenate([np.diag(1. / d.sigmasq_flat[self.group])[None, ...]
                               for d in self.emission_distns])

    def heldout_log_likelihood(self, test_mask=None):
        """
        Compute the log likelihood of the masked data given the latent
        discrete and continuous states.
        """
        if test_mask is None:
            # If a test mask is not supplied, use the negation of this object's mask
            if self.mask is None:
                return 0
            else:
                test_mask = ~self.mask

        xs = np.hstack((self.gaussian_states, self.inputs))
        if self.single_emission:
            return self.emission_distns[0].\
                log_likelihood((xs, self.data), mask=test_mask, group=self.group).sum()
        else:
            hll = 0
            z = self.stateseq
            for idx, ed in enumerate(self.emission_distns):
                hll += ed.log_likelihood((xs[z == idx], self.data[z == idx]),
                                         mask=test_mask[z == idx],
                                         group=self.group).sum()


    # Use group specific variances to calculate likelihoods
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

    @property
    def vbem_aBl(self):
        """
        These are the expected log likelihoods (node potentials)
        as seen from the discrete states.  In other words,
        E_{q(x)} [log p(y, x | z)]
        """
        from pyslds.util import expected_gaussian_logprob, expected_diag_regression_log_prob, \
            expected_regression_log_prob
        vbem_aBl = np.zeros((self.T, self.num_states))
        ids, dds, eds = self.init_dynamics_distns, self.dynamics_distns, self.emission_distns

        for k, (id, dd) in enumerate(zip(ids, dds)):
            vbem_aBl[0, k] = expected_gaussian_logprob(id.mu, id.sigma, self.E_init_stats)
            vbem_aBl[:-1, k] += expected_regression_log_prob(dd, self.E_dynamics_stats)

        # Override to use the group-specific sigma
        if self.single_emission:
            ed = self.emission_distns[0]
            vbem_aBl += expected_diag_regression_log_prob(ed.A, ed.sigmasq_flat[self.group], self.E_emission_stats)[:,None]
        else:
            for k, ed in enumerate(self.emission_distns):
                vbem_aBl[:, k] += expected_diag_regression_log_prob(ed.A, ed.sigmasq_flat[self.group], self.E_emission_stats)

        vbem_aBl[np.isnan(vbem_aBl).any(1)] = 0.
        return vbem_aBl

    def generate_obs(self):
        # Go through each time bin, get the discrete latent state,
        # use that to index into the emission_distns to get samples
        T, p = self.T, self.D_emission
        dss, gss = self.stateseq, self.gaussian_states
        data = np.empty((T,p),dtype='double')

        for t in range(self.T):
            ed = self.emission_distns[0] if self.model._single_emission \
                else self.emission_distns[dss[t]]
            data[t] = \
                ed.rvs(x=np.hstack((gss[t][None, :], self.inputs[t][None,:])),
                       return_xy=False, group=self.group)

        return data


class HierarchicalSLDSStates(_HierarchicalSLDSStatesMixin,
                             _SLDSStatesMaskedData,
                             _SLDSStatesGibbs,
                             _SLDSStatesVBEM,
                             HMMStatesEigen):
    pass


from rslds.states import SoftmaxRecurrentSLDSStates
class HierarchicalRecurrentSLDSStates(_HierarchicalSLDSStatesMixin, SoftmaxRecurrentSLDSStates):

    # @property
    # def vbem_info_emission_params(self):
    #     J_node, h_node, log_Z_node = \
    #         super(HierarchicalRecurrentSLDSStates, self).vbem_info_emission_params
    #
    #     J_rec, h_rec = self.vbem_info_rec_params
    #     return J_node + J_rec, h_node + h_rec, log_Z_node

    @property
    def vbem_aBl(self):
        aBl = super(HierarchicalRecurrentSLDSStates, self).vbem_aBl

        # Add in node potentials from transitions
        aBl += self._vbem_aBl_rec
        return aBl
