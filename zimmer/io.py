import os
import numpy as np
from scipy.io import loadmat
from scipy.interpolate import interp1d

from zimmer.util import states_to_changepoints

# Load the data
data_dir = "whole brain data Kato et al 2015"

worm_files = ["TS20140715e_lite-1_punc-31_NLS3_2eggs_56um_1mMTet_basal_1080s.mat",
              "TS20140715f_lite-1_punc-31_NLS3_3eggs_56um_1mMTet_basal_1080s.mat",
              "TS20140905c_lite-1_punc-31_NLS3_AVHJ_0eggs_1mMTet_basal_1080s.mat",
              "TS20140926d_lite-1_punc-31_NLS3_RIV_2eggs_1mMTet_basal_1080s.mat",
              "TS20141221b_THK178_lite-1_punc-31_NLS3_6eggs_1mMTet_basal_1080s.mat"
              ]

N_worms = len(worm_files)

def load_labels():
    zimmer_state_labels = \
        loadmat(os.path.join(
            data_dir,
            "sevenStateColoring.mat"))
    return zimmer_state_labels

def load_key():
    data = load_labels()
    key = data["sevenStateColoring"]["key"][0,0][0]
    key = [str(k)[2:-2] for k in key]
    return key

class WormData(object):
    """
    Wrapper for basic worm dataset
    """
    def __init__(self, index, name, sample_rate=3.0):
        self.worm_name = name
        self.worm_index = index


        filename = os.path.join(data_dir, "wbdata", worm_files[index])
        zimmer_data = loadmat(filename)

        # Get the neuron names
        neuron_ids = zimmer_data["wbData"]['NeuronIds'][0, 0][0]
        self._neuron_ids_1 = np.array(
            list(map(lambda x: None if len(x[0]) == 0
                               else str(x[0][0][0]),
                neuron_ids)))

        self._neuron_ids_2 = np.array(
            list(map(lambda x: None if x.size < 2 or x[0, 1].size == 0
                               else str(x[0, 1][0]),
                neuron_ids)))

        self.neuron_names = self._get_neuron_names()

        # Get the calcium trace (corrected for bleaching)
        t_smpl = np.ravel(zimmer_data["wbData"]['tv'][0, 0])
        self.sample_rate = sample_rate
        self.t_start = t_smpl[0]
        self.t_stop = t_smpl[-1]
        self.tt = np.arange(self.t_start, self.t_stop, step=1./sample_rate)
        def interp_data(xx, kind="linear"):
            f = interp1d(t_smpl, xx, axis=0, kind=kind)
            return f(self.tt)
            # return np.interp(self.tt, t_smpl, xx, axis=0)

        self.dff = interp_data(zimmer_data["wbData"]['deltaFOverF'][0, 0])
        self.dff_bc = interp_data(zimmer_data["wbData"]['deltaFOverF_bc'][0, 0])
        self.dff_deriv = interp_data(zimmer_data["wbData"]['deltaFOverF_deriv'][0, 0])

        # Kato et al smoothed the derivative.  Let's just work with the first differences
        # of the bleaching corrected and normalized dF/F
        self.dff_bc_zscored = (self.dff_bc - self.dff_bc.mean(0)) / self.dff_bc.std(0)
        self.dff_diff = np.vstack((np.zeros((1, self.dff_bc_zscored.shape[1])),
                                   np.diff(self.dff_bc_zscored, axis=0)))

        # # Let's try our hand at our own smoothing of the derivative
        # from pylds.models import DefaultLDS
        # lds = DefaultLDS(D_obs=1, D_latent=1,
        #                  A=np.eye(1), sigma_states=0.01 * np.eye(1),
        #                  C=np.eye(1), sigma_obs=np.eye(1))
        #
        # self.smoothed_diff = np.zeros_like(self.dff_diff)
        # self.smoothed_y = np.zeros_like(self.dff_bc_zscored)
        # for n in range(self.dff.shape[1]):
        #     self.smoothed_diff[:,n] = lds.smooth(self.dff_diff[:, n:n + 1]).ravel()
        #     self.smoothed_y[:,n] = self.dff_bc_zscored[0,n] + np.cumsum(self.smoothed_diff[:,n])

        # Get the state sequence as labeled in Kato et al
        # Interpolate to get at new time points
        zimmer_state_labels = load_labels()
        zimmer_state_time_series = zimmer_state_labels["sevenStateColoring"]["dataset"][0, 0]['stateTimeSeries']
        self.zimmer_states = interp_data(zimmer_state_time_series[0, index].ravel() - 1, kind="nearest")
        self.zimmer_states = np.clip(np.round(self.zimmer_states), 0, 7).astype(np.int)
        self.zimmer_cps = np.concatenate(([0],
                                          1 + np.where(np.diff(self.zimmer_states))[0],
                                          [self.zimmer_states.size - 1]))

        # expose number of neurons and number of time bins
        self.T, self.N = self.dff.shape

    def _get_neuron_names(self):
        # Remove the neurons that are not uniquely identified
        def check_label(neuron_name):
            if neuron_name is None:
                return False
            if neuron_name == "---":
                return False

            neuron_index = np.where(self._neuron_ids_1 == neuron_name)[0]
            if len(neuron_index) != 1:
                return False

            if self._neuron_ids_2[neuron_index[0]] is not None:
                return False

            # Make sure it doesn't show up in the second neuron list
            if len(np.where(self._neuron_ids_2 == neuron_name)[0]) > 0:
                return False

            return True

        final_neuron_names = []
        for i, neuron_name in enumerate(self._neuron_ids_1):
            if check_label(neuron_name):
                final_neuron_names.append(neuron_name)
            else:
                final_neuron_names.append("{}_neuron{}".format(self.worm_name, i))

        return final_neuron_names

    def find_neuron_indices(self, target_list):
        # Find their indices
        indices = []
        for target in target_list:
            index = np.where(np.array(self.neuron_names) == target)[0]
            if len(index) > 0:
                indices.append(index[0])
            else:
                indices.append(None)
        return indices


# Find neurons that were identified in each worm
def find_shared_neurons(worm_datas):
    from functools import reduce

    all_first_neuron_ids = [[id if id is not None else "---"
                             for id in wd._neuron_ids_1]
                            for wd in worm_datas]
    shared_neurons = reduce(np.intersect1d, all_first_neuron_ids)
    print("Potentially shared neurons:\n {0}".format(shared_neurons))

    truly_shared_neurons = []
    for neuron_name in shared_neurons:
        is_shared = True
        for worm_data in worm_datas:
            is_shared = is_shared and neuron_name in worm_data.neuron_names
        if is_shared:
            truly_shared_neurons.append(neuron_name)

    shared_neurons = truly_shared_neurons
    N_shared = len(shared_neurons)
    print("Found {} truly shared neurons:".format(N_shared))
    print(shared_neurons)

    return shared_neurons


