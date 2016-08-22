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

class WormData(object):
    """
    Wrapper for basic worm dataset
    """
    def __init__(self, worm_index=None, sample_rate=3.0):
        filename = os.path.join(data_dir, "wbdata", worm_files[worm_index])
        zimmer_data = loadmat(filename)

        # Get the neuron names
        neuron_ids = zimmer_data["wbData"]['NeuronIds'][0, 0][0]
        self.neuron_ids_1 = np.array(
            map(lambda x: None if len(x[0]) == 0
                               else str(x[0][0][0]),
                neuron_ids))

        self.neuron_ids_2 = np.array(
            map(lambda x: None if x.size < 2 or x[0, 1].size == 0
                               else str(x[0, 1][0]),
                neuron_ids))

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


        # Get the state sequence as labeled in Kato et al
        # Interpolate to get at new time points
        zimmer_state_labels = load_labels()
        zimmer_state_time_series = zimmer_state_labels["sevenStateColoring"]["dataset"][0, 0]['stateTimeSeries']
        self.zimmer_states = interp_data(zimmer_state_time_series[0, worm_index].ravel() - 1, kind="nearest")
        self.zimmer_states = np.clip(np.round(self.zimmer_states), 0, 7).astype(np.int)
        self.zimmer_cps = np.concatenate(([0],
                                          1 + np.where(np.diff(self.zimmer_states))[0],
                                          [self.zimmer_states.size - 1]))

        # expose number of neurons and number of time bins
        self.T, self.N = self.dff.shape

    def find_neurons(self, neuron_names):
        # Find their indices
        indices = []
        for neuron_name in neuron_names:
            neuron_index = np.where(self.neuron_ids_1 == neuron_name)[0]
            indices.append(neuron_index[0])

        # Find the indices of the neurons unique to each worm
        other_indices = set(np.arange(self.N))
        other_indices -= set(indices)

        return indices


    # Find neurons that were identified in each worm
def find_shared_neurons(worm_datas):
    from functools import reduce

    all_first_neuron_ids = [wd.neuron_ids_1 for wd in worm_datas]
    all_second_neuron_ids = [wd.neuron_ids_2 for wd in worm_datas]
    shared_neurons = reduce(np.intersect1d, all_first_neuron_ids)
    print("Potentially shared neurons:\n {0}".format(shared_neurons))

    # Remove the neurons that are not uniquely identified
    def check_label(neuron_name, first_neuron_ids, second_neuron_ids):
        if neuron_name is None:
            return False
        neuron_index = np.where(first_neuron_ids == neuron_name)[0]
        if len(neuron_index) != 1:
            return False

        if second_neuron_ids[neuron_index[0]] is not None:
            return False

        # Make sure it doesn't show up in the second neuron list
        if len(np.where(second_neuron_ids == neuron_name)[0]) > 0:
            return False

        return True


    truly_shared_neurons = []
    for neuron_name in shared_neurons:
        is_shared = True
        for worm_index in range(5):
            first_neuron_ids = all_first_neuron_ids[worm_index]
            second_neuron_ids = all_second_neuron_ids[worm_index]
            is_shared = is_shared and check_label(neuron_name, first_neuron_ids, second_neuron_ids)
        if is_shared:
            truly_shared_neurons.append(neuron_name)

    shared_neurons = truly_shared_neurons
    N_shared = len(shared_neurons)
    print("Found {} truly shared neurons:".format(N_shared))
    print(shared_neurons)

    return shared_neurons


