import numpy as np

from zimmer.states import HierarchicalSLDSStates, HierarchicalRecurrentSLDSStates
from pyslds.models import WeakLimitStickyHDPHMMSLDS
from rslds.rslds import RecurrentSLDS, RecurrentOnlySLDS, StickyRecurrentSLDS, StickyRecurrentOnlySLDS

class _HierarchicalSLDSMixin(object):
    _states_class = HierarchicalSLDSStates

    def resample_emission_distns(self):
        if self.has_count_data:
            raise NotImplementedError

        if self._single_emission:
            data = [(np.hstack((s.gaussian_states, s.inputs)), s.data)
                    for s in self.states_list]
            mask = [s.mask for s in self.states_list] if self.has_missing_data else None
            groups = [s.group for s in self.states_list]

            if self.has_missing_data:
                self._emission_distn.resample(data=data, mask=mask, groups=groups)
            else:
                self._emission_distn.resample(data=data, groups=groups)
        else:
            for state, d in enumerate(self.emission_distns):
                data = [(np.hstack((s.gaussian_states[s.stateseq == state],
                                    s.inputs[s.stateseq == state])),
                         s.data[s.stateseq == state])
                        for s in self.states_list]

                mask = [s.mask[s.stateseq == state] for s in self.states_list] \
                    if self.has_missing_data else None
                groups = [s.group for s in self.states_list]

                if self.has_missing_data:
                    d.resample(data=data, mask=mask, groups=groups)
                else:
                    d.resample(data=data, groups=groups)

        self._clear_caches()


class HierarchicalWeakLimitStickyHDPHMMSLDS(_HierarchicalSLDSMixin, WeakLimitStickyHDPHMMSLDS):
    pass

class HierarchicalRecurrentSLDS(_HierarchicalSLDSMixin, RecurrentSLDS):
    _states_class = HierarchicalRecurrentSLDSStates

class HierarchicalStickyRecurrentSLDS(_HierarchicalSLDSMixin, StickyRecurrentSLDS):
    _states_class = HierarchicalRecurrentSLDSStates

class HierarchicalRecurrentOnlySLDS(_HierarchicalSLDSMixin, RecurrentOnlySLDS):
    _states_class = HierarchicalRecurrentSLDSStates

class HierarchicalStickyRecurrentOnlySLDS(_HierarchicalSLDSMixin, StickyRecurrentOnlySLDS):
    _states_class = HierarchicalRecurrentSLDSStates
