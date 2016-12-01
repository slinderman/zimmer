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


class _MultiEmissionSLDSGibbsMixin(object):
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

