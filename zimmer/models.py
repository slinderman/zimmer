import numpy as np

from zimmer.states import HierarchicalSLDSStates, HierarchicalRecurrentSLDSStates
from pyslds.models import HMMSLDS, WeakLimitStickyHDPHMMSLDS
from rslds.models import SoftmaxRecurrentSLDS, SoftmaxRecurrentOnlySLDS

class _HierarchicalSLDSMixin(object):
    _states_class = HierarchicalSLDSStates

    def resample_emission_distns(self):
        assert self._single_emission
        data = [(np.hstack((s.gaussian_states, s.inputs)), s.data)
                for s in self.states_list]
        mask = [s.mask for s in self.states_list] if self.has_missing_data else None
        groups = [s.group for s in self.states_list]

        if self.has_missing_data:
            self._emission_distn.resample(data=data, mask=mask, groups=groups)
        else:
            self._emission_distn.resample(data=data, groups=groups)

        self._clear_caches()

    # Max likelihood
    def _M_step_emission_distn(self):
        assert self._single_emission
        E_emi_stats = lambda s: \
            tuple(np.sum(stat, axis=0) for stat in s.E_emission_stats)
        stats = [E_emi_stats(s) for s in self.states_list]
        groups = [s.group for s in self.states_list]
        self._emission_distn.max_likelihood(stats=stats, groups=groups)


class HierarchicalHMMSLDS(_HierarchicalSLDSMixin, HMMSLDS):
    pass


class HierarchicalWeakLimitStickyHDPHMMSLDS(_HierarchicalSLDSMixin, WeakLimitStickyHDPHMMSLDS):
    pass


class HierarchicalRecurrentSLDS(_HierarchicalSLDSMixin, SoftmaxRecurrentSLDS):
    _states_class = HierarchicalRecurrentSLDSStates


class HierarchicalRecurrentOnlySLDS(_HierarchicalSLDSMixin, SoftmaxRecurrentOnlySLDS):
    _states_class = HierarchicalRecurrentSLDSStates
