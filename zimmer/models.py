import numpy as np

from pyhsmm.models import HMMPython, HMM, WeakLimitHDPHMM, WeakLimitStickyHDPHMM

from zimmer.states import MultiEmissionSLDSStatesEigen, MultiEmissionSLDSStatesPython
from pyslds.models import _SLDSMixin

class _MultiEmissionSLDS(_SLDSMixin):
    def __init__(self,dynamics_distns,emission_distns,init_dynamics_distns,**kwargs):
        self.init_dynamics_distns = init_dynamics_distns
        self.dynamics_distns = dynamics_distns
        self.emission_distns = emission_distns

        super(_SLDSMixin, self).__init__(
            obs_distns=self.dynamics_distns, **kwargs)

    def generate(self, T=100, keep=True, **kwargs):
        s = self._states_class(model=self, T=T, initialize_from_prior=True, **kwargs)
        s.generate_states()
        data = self._generate_obs(s)
        if keep:
            self.states_list.append(s)
        return data, s.stateseq


class _MultiEmissionSLDSGibbsMixin(_MultiEmissionSLDS):
    def resample_parameters(self):
        self.resample_lds_parameters()
        self.resample_hmm_parameters()

    def resample_lds_parameters(self):
        self.resample_init_dynamics_distns()
        self.resample_dynamics_distns()
        self.resample_emission_distns()

    def resample_hmm_parameters(self):
        super(_MultiEmissionSLDSGibbsMixin, self).resample_parameters()

    def resample_init_dynamics_distns(self):
        for state, d in enumerate(self.init_dynamics_distns):
            d.resample(
                [s.gaussian_states[0] for s in self.states_list
                 if s.stateseq[0] == state])
        self._clear_caches()

    def resample_dynamics_distns(self):
        zs = [s.stateseq[:-1] for s in self.states_list]
        xs = [np.hstack((s.gaussian_states[:-1], s.inputs[:-1]))
              for s in self.states_list]
        ys = [s.gaussian_states[1:] for s in self.states_list]

        for state, d in enumerate(self.dynamics_distns):
            d.resample(
                [(x[z == state], y[z == state])
                 for x, y, z in zip(xs, ys, zs)])
        self._clear_caches()

    def resample_emission_distns(self):
        for i, ed in enumerate(self.emission_distns):
            # TODO: Fix up this hacky group support!
            data = []
            mask = []
            groups = []
            for s in self.states_list:
                if s.data[i] is not None:
                    data.append((np.hstack((s.gaussian_states,
                                            s.inputs)),
                                 s.data[i]))
                    mask.append(s.mask[i])

                    if hasattr(s, "group") and s.group is not None:
                        groups.append(s.group)

            if len(groups) > 0:
                ed.resample(data=data, groups=groups)
            else:
                ed.resample(data=data, mask=mask)

        self._clear_caches()

    def resample_obs_distns(self):
        pass  # handled in resample_parameters

    def _joblib_resample_states(self,states_list,num_procs):
        raise NotImplementedError


### Multiple Gaussian emission models

class MultiEmissionHMMSLDSPython(
    _MultiEmissionSLDSGibbsMixin,
    _MultiEmissionSLDS,
    HMMPython):
    _states_class = MultiEmissionSLDSStatesPython


class MultiEmissionHMMSLDS(
    _MultiEmissionSLDSGibbsMixin,
    _MultiEmissionSLDS,
    HMM):
    _states_class = MultiEmissionSLDSStatesEigen


class MultiEmissionWeakLimitHDPHMMSLDS(
    _MultiEmissionSLDSGibbsMixin,
    _MultiEmissionSLDS,
    WeakLimitHDPHMM):
    _states_class = MultiEmissionSLDSStatesEigen


class MultiEmissionWeakLimitStickyHDPHMMSLDS(
        _MultiEmissionSLDSGibbsMixin,
        _MultiEmissionSLDS,
        WeakLimitStickyHDPHMM):
    _states_class = MultiEmissionSLDSStatesEigen


from zimmer.states import HierarchicalInputSLDSStates
from rslds.rslds import InputHMMTransitions, InputHMM, \
    InputOnlyHMMTransitions, StickyInputHMMTransitions, StickyInputOnlyHMMTransitions


class HierarchicalInputSLDS(_MultiEmissionSLDSGibbsMixin, InputHMM):

    _states_class = HierarchicalInputSLDSStates
    _trans_class = InputHMMTransitions

    def __init__(self, dynamics_distns, emission_distns, init_dynamics_distns,
                 fixed_emission=False, **kwargs):

        self.fixed_emission = fixed_emission

        super(HierarchicalInputSLDS, self).__init__(
            dynamics_distns, emission_distns, init_dynamics_distns,
            D_in=dynamics_distns[0].D_out, **kwargs)

    def resample_trans_distn(self):
        # Include the auxiliary variables used for state resampling
        self.trans_distn.resample(
            stateseqs=[s.stateseq for s in self.states_list],
            covseqs=[s.covariates for s in self.states_list],
            omegas=[s.trans_omegas for s in self.states_list]
        )
        self._clear_caches()

    def add_data(self, data, covariates=None, **kwargs):
        self.states_list.append(
                self._states_class(
                    model=self, data=data,
                    **kwargs))

    def generate(self, T=100, keep=True, with_noise=True, **kwargs):
        s = self._states_class(model=self, T=T, initialize_from_prior=True, **kwargs)
        s.generate_states(with_noise=with_noise)
        data = self._generate_obs(s, with_noise=with_noise)
        if keep:
            self.states_list.append(s)
        return data, s.stateseq

    def _generate_obs(self, s, with_noise=True):
        if s.data is None:
            s.data = s.generate_obs()
        else:
            # TODO: Handle missing data
            raise NotImplementedError

        return s.data, s.gaussian_states

    def resample_emission_distns(self):
        if self.fixed_emission:
            return
        super(HierarchicalInputSLDS, self).resample_emission_distns()


class HierarchicalStickyInputSLDS(HierarchicalInputSLDS):
    _trans_class = StickyInputHMMTransitions


class HierarchicalInputOnlySLDS(HierarchicalInputSLDS):
    _trans_class = InputOnlyHMMTransitions


class HierarchicalStickyInputOnlySLDS(HierarchicalInputSLDS):
    _trans_class = StickyInputOnlyHMMTransitions
