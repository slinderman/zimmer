import os
import glob
from functools import partial
import numpy as np
from scipy.io import loadmat
from scipy.interpolate import interp1d

# Load the data
kato_dir = os.path.join("data", "kato2015")
kato_files = ["TS20140715e_lite-1_punc-31_NLS3_2eggs_56um_1mMTet_basal_1080s.mat",
              "TS20140715f_lite-1_punc-31_NLS3_3eggs_56um_1mMTet_basal_1080s.mat",
              "TS20140905c_lite-1_punc-31_NLS3_AVHJ_0eggs_1mMTet_basal_1080s.mat",
              "TS20140926d_lite-1_punc-31_NLS3_RIV_2eggs_1mMTet_basal_1080s.mat",
              "TS20141221b_THK178_lite-1_punc-31_NLS3_6eggs_1mMTet_basal_1080s.mat"]

nichols_dir = os.path.join("data", "nichols2017")


def load_kato_labels():
    zimmer_state_labels = \
        loadmat(os.path.join(
            kato_dir,
            "sevenStateColoring.mat"))
    return zimmer_state_labels


def load_kato_key():
    data = load_kato_labels()
    key = data["sevenStateColoring"]["key"][0,0][0]
    key = [str(k)[2:-2] for k in key]
    return key


def _get_neuron_names(neuron_ids_1, neuron_ids_2, worm_name):
    # Remove the neurons that are not uniquely identified
    def check_label(neuron_name):
        if neuron_name is None:
            return False
        if neuron_name == "---":
            return False

        neuron_index = np.where(neuron_ids_1 == neuron_name)[0]
        if len(neuron_index) != 1:
            return False

        if neuron_ids_2[neuron_index[0]] is not None:
            return False

        # Make sure it doesn't show up in the second neuron list
        if len(np.where(neuron_ids_2 == neuron_name)[0]) > 0:
            return False

        return True

    final_neuron_names = []
    for i, neuron_name in enumerate(neuron_ids_1):
        if check_label(neuron_name):
            final_neuron_names.append(neuron_name)
        else:
            final_neuron_names.append("{}_neuron{}".format(worm_name, i))

    return final_neuron_names


def _load_kato(index, name, sample_rate):
        filename = os.path.join(kato_dir, "wbdata", kato_files[index])
        zimmer_data = loadmat(filename)

        # Get the neuron names
        neuron_ids = zimmer_data["wbData"]['NeuronIds'][0, 0][0]
        neuron_ids_1 = np.array(
            list(map(lambda x: None if len(x[0]) == 0
                               else str(x[0][0][0]),
                neuron_ids)))

        neuron_ids_2 = np.array(
            list(map(lambda x: None if x.size < 2 or x[0, 1].size == 0
                               else str(x[0, 1][0]),
                neuron_ids)))

        # Fix labels of SMB neurons per Manuel's instructions on Jan 16, 2018
        mapping = dict(SMBDL="SMDDL", SMBDR="SMDDR", SMBVL="SMDDL", SMBVR="SMDDR")
        def translate(ids):
            for j, id in enumerate(ids):
                if id in mapping:
                    ids[j] = mapping[id]
                    
        translate(neuron_ids_1)
        translate(neuron_ids_2)

        neuron_names = _get_neuron_names(neuron_ids_1, neuron_ids_2, name)

        # Get the calcium trace (corrected for bleaching)
        t_smpl = np.ravel(zimmer_data["wbData"]['tv'][0, 0])
        sample_rate = sample_rate
        t_start = t_smpl[0]
        t_stop = t_smpl[-1]
        tt = np.arange(t_start, t_stop, step=1./sample_rate)
        def interp_data(xx, kind="linear"):
            f = interp1d(t_smpl, xx, axis=0, kind=kind)
            return f(tt)
            # return np.interp(tt, t_smpl, xx, axis=0)

        dff = interp_data(zimmer_data["wbData"]['deltaFOverF'][0, 0])
        dff_bc = interp_data(zimmer_data["wbData"]['deltaFOverF_bc'][0, 0])
        dff_deriv = interp_data(zimmer_data["wbData"]['deltaFOverF_deriv'][0, 0])

        # Kato et al smoothed the derivative.  Let's just work with the first differences
        # of the bleaching corrected and normalized dF/F
        dff_bc_zscored = (dff_bc - dff_bc.mean(0)) / dff_bc.std(0)
        dff_diff = np.vstack((np.zeros((1, dff_bc_zscored.shape[1])),
                                   np.diff(dff_bc_zscored, axis=0)))

        # # Let's try our hand at our own smoothing of the derivative
        # from pylds.models import DefaultLDS
        # lds = DefaultLDS(D_obs=1, D_latent=1,
        #                  A=np.eye(1), sigma_states=0.01 * np.eye(1),
        #                  C=np.eye(1), sigma_obs=np.eye(1))
        #
        # smoothed_diff = np.zeros_like(dff_diff)
        # smoothed_y = np.zeros_like(dff_bc_zscored)
        # for n in range(dff.shape[1]):
        #     smoothed_diff[:,n] = lds.smooth(dff_diff[:, n:n + 1]).ravel()
        #     smoothed_y[:,n] = dff_bc_zscored[0,n] + np.cumsum(smoothed_diff[:,n])

        # Get the state sequence as labeled in Kato et al
        # Interpolate to get at new time points
        zimmer_state_labels = load_kato_labels()
        zimmer_state_time_series = zimmer_state_labels["sevenStateColoring"]["dataset"][0, 0]['stateTimeSeries']
        zimmer_states = interp_data(zimmer_state_time_series[0, index].ravel() - 1, kind="nearest")
        zimmer_states = np.clip(np.round(zimmer_states), 0, 7).astype(np.int)

        return neuron_names, dff, dff_bc, dff_deriv, dff_bc_zscored, dff_diff, zimmer_states


def _load_nichols(index, name, condition, sample_rate):
    # Get the condition-specific subdirectory of the nichols data
    condition_to_dir = dict(
        n2_1_prelet="N2_1_PreLet",
        n2_2_let="N2_2_Let",
        npr1_1_prelet="npr1_1_PreLet",
        npr1_2_let="npr1_2_Let"
    )
    dir1 = os.path.join(nichols_dir, condition_to_dir[condition.lower()])
    dir2 = sorted(glob.glob(os.path.join(dir1, "AN*")))[index]
    zimmer_data = loadmat(os.path.join(dir2, "wbdataset.mat"))

    # Get the neuron names
    def _get_name(x, pos):
        if len(x[0]) < pos + 1:
            return None
        elif x[0][pos].size == 0:
            return None
        else:
            return str(x[0][pos][0])

    neuron_ids = zimmer_data["wbdataset"]['IDs'][0, 0][0]
    neuron_ids_1 = np.array(list(map(partial(_get_name, pos=0), neuron_ids)))
    neuron_ids_2 = np.array(list(map(partial(_get_name, pos=1), neuron_ids)))
    neuron_names = _get_neuron_names(neuron_ids_1, neuron_ids_2, name)

    # Get the calcium trace (corrected for bleaching)
    t_smpl = np.ravel(zimmer_data["wbdataset"]['timeVectorSeconds'][0, 0])
    sample_rate = sample_rate
    t_start = t_smpl[0]
    t_stop = t_smpl[-2]
    t_interp = np.arange(t_start, t_stop, step=1. / sample_rate)

    def interp_data(tt, xx, kind="linear"):
        f = interp1d(tt, xx, axis=0, kind=kind)
        return f(t_interp)
        # return np.interp(tt, t_smpl, xx, axis=0)

    dff = interp_data(t_smpl, zimmer_data["wbdataset"]['traces'][0, 0])
    dff_deriv = interp_data(t_smpl[:-1], zimmer_data["wbdataset"]['tracesDif'][0, 0])
    dff_zscored = (dff - dff.mean(0)) / dff.std(0)
    dff_diff = np.vstack((np.zeros((1, dff_zscored.shape[1])), np.diff(dff_zscored, axis=0)))

    # Extract the states
    states_obj = zimmer_data['wbdataset']['FiveStates'][0,0]
    zimmer_states = states_obj['fiveStates'][0,0].ravel().astype(int)
    zimmer_states = interp_data(t_smpl, zimmer_states, kind="nearest")
    zimmer_states = np.clip(np.round(zimmer_states), 0, 5).astype(np.int)
    zimmer_state_names = states_obj['fiveStateNames'][0,0][0]
    zimmer_state_names = [str(x[0]) for x in zimmer_state_names]

    # Extract the simulus
    stimulus_obj = zimmer_data["wbdataset"]['stimulus'][0,0]
    stimulus_id = str(stimulus_obj['identity'][0,0][0])
    stimulus_type = str(stimulus_obj['type'][0,0][0])
    stimulus_switchtimes = stimulus_obj['switchtimes'][0,0].ravel().astype(int)
    stimulus_initialstate = stimulus_obj['initialstate'][0,0][0,0].astype(int) - 1
    stimulus_conc = stimulus_obj['conc'][0,0].ravel().astype(int)
    stimulus_concunits = str(stimulus_obj['concunits'][0,0][0])

    # convert stimulus into a vector
    stimulus = np.zeros(t_interp.size, dtype=int)
    z_stim = stimulus_initialstate
    t_start = 0
    for t_stop in stimulus_switchtimes:
        stimulus[(t_interp >= t_start) & (t_interp < t_stop)] = stimulus_conc[z_stim]
        z_stim = 1 - z_stim
        t_start = t_stop
    # Handle last iteration
    stimulus[t_interp >= t_start] = stimulus_conc[z_stim]

    # I'm assuming all the stimuli are binary steps
    assert stimulus_id == "O2"
    assert stimulus_type == "binarysteps"
    assert np.all(stimulus_conc == np.array([10, 21]))

    return neuron_names, dff, dff_deriv, dff_zscored, dff_diff, zimmer_states, zimmer_state_names, stimulus


class WormData(object):
    """
    Wrapper for basic worm dataset.
    """
    def __init__(self, index, name, version, condition=None, sample_rate=3.0):
        self.worm_name = name
        self.worm_index = index
        self.version = version
        self.condition = condition
        self.sample_rate = sample_rate

        if version.lower() == "kato":
            assert condition is None, "condition has no influence for kato data."
            self.neuron_names, self.dff, self.dff_bc, self.dff_deriv, \
            self.dff_bc_zscored, self.dff_diff, self.zimmer_states = \
                _load_kato(index, name, sample_rate)
        elif version.lower() == "nichols":
            assert condition is not None, "must specify condition for nichols data"
            self.neuron_names, self.dff, self.dff_deriv, self.dff_zscored, \
            self.dff_diff, self.zimmer_states, self.zimmer_state_names, self.stimulus = \
                _load_nichols(index, name, condition, sample_rate)
        else:
            raise Exception("version must be either 'kato' or 'nichols'.")

        # expose number of neurons and number of time bins
        self.T, self.N = self.dff.shape

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


if __name__ == "__main__":
    # for ind in range(11):
    #     _load_nichols(ind, "test", "n2_1_prelet", 3.0)
    # for ind in range(12):
    #     _load_nichols(ind, "test", "n2_2_let", 3.0)
    # for ind in range(10):
    #     _load_nichols(ind, "test", "npr1_1_prelet", 3.0)
    for ind in range(11):
        _load_nichols(ind, "test", "npr1_2_let", 3.0)

