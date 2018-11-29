import os
import copy
import pickle
import numpy as np
from zimmer.io import load_kato_data, load_nichols_data
from ssm.preprocessing import trend_filter, pca_with_imputation

np.random.seed(1234)
data_dir = os.path.join("data", "processed")


def process_data(version="kato", chunk=250, train_frac=0.7, val_frac=0.15):
    # Load the raw data
    if version == "kato":
        ys, ms, z_trues, z_true_key, neuron_names = \
            load_kato_data(include_unnamed=False, signal="dff")
        us = [np.zeros((y.shape[0], 0)) for y in ys]
        tags = np.arange(len(ys))

    elif version == "nichols":
        groups = ["n2_1_prelet", 
          "n2_2_let",
          "npr1_1_prelet",
          "npr1_2_let"]

        tags = [(i, "n2_1_prelet") for i in range(11)] + \
               [(i, "n2_2_let") for i in range(12)] + \
               [(i, "npr1_1_prelet") for i in range(10)] + \
               [(i, "npr1_2_let") for i in range(11)]
        worm_names = ["{} worm {}".format(group, i) for (i, group) in tags]

        ys, ms, us, z_trues, z_true_key, neuron_names = \
            load_nichols_data(tags, worm_names, include_unnamed=False, signal="dff")

    else:
        raise Exception("Invalid version {}".format(version))

    # Trend filter the data
    ys = [trend_filter(y) for y in ys]

    # Run PCA to get a 3d projection of the data
    _pca = cached(results_dir, "pca")(pca_with_imputation)
    pca, xs = pca_with_imputation(D, ys, ms)
    xs = [x.copy('C') for x in xs]

    
    # Split into training and test data
    all_ys = []
    all_ms = []
    all_us = []
    all_tags = []
    all_z_trues = []
    all_choices = []
    for (y, m, u, tag, ztr) in zip(ys, ms, us, tags, z_trues):
        T = y.shape[0]
        C = 0
        for start in range(0, T, chunk):
            stop = min(start+chunk, T)
            all_ys.append(y[start:stop])
            all_ms.append(m[start:stop])
            all_us.append(u[start:stop])
            all_z_trues.append(ztr[start:stop])
            all_tags.append(tag)
            C += 1
            
        # assign some of the data to train, val, and test
        choices = -1 * np.ones(C)
        choices[:int(train_frac * C)] = 0
        choices[int(train_frac * C):int((train_frac + val_frac) * C)] = 1
        choices[int((train_frac + val_frac) * C):] = 2
        choices = choices[np.random.permutation(C)]
        all_choices.append(choices)

    all_choices = np.concatenate(all_choices)
    get = lambda arr, chc: [x for x, c in zip(arr, all_choices) if c == chc]

    train_ys = get(all_ys, 0)
    train_ms = get(all_ms, 0)
    train_us = get(all_us, 0)
    train_zs = get(all_z_trues, 0)
    train_tags = get(all_tags, 0)

    val_ys = get(all_ys, 1)
    val_ms = get(all_ms, 1)
    val_us = get(all_us, 1)
    val_zs = get(all_z_trues, 1)
    val_tags = get(all_tags, 1)

    test_ys = get(all_ys, 2)
    test_ms = get(all_ms, 2)
    test_us = get(all_us, 2)
    test_zs = get(all_z_trues, 2)
    test_tags = get(all_tags, 2)

    # print("Training chunks per worm:   ", np.bincount(train_tags))
    # print("Validation chunks per worm: ", np.bincount(val_tags))
    # print("Testing chunks per worm:    ", np.bincount(test_tags))

    # Wrap in dictionaries objects
    train_datas = []
    for y, m, u, tag, z in zip(train_ys, train_ms, train_us, train_tags, train_zs):
        train_datas.append(dict(y=y, m=m, u=u, tag=tag, z_true=z, 
                                z_true_key=z_true_key, neuron_names=neuron_names))

    val_datas = []
    for y, m, u, tag, z in zip(val_ys, val_ms, val_us, val_tags, val_zs):
        val_datas.append(dict(y=y, m=m, u=u, tag=tag, z_true=z, 
                              z_true_key=z_true_key, neuron_names=neuron_names))

    test_datas = []
    for y, m, u, tag, z in zip(test_ys, test_ms, test_us, test_tags, test_zs):
        test_datas.append(dict(y=y, m=m, u=u, tag=tag, z_true=z, 
                               z_true_key=z_true_key, neuron_names=neuron_names))
    
    full_datas = []
    for y, m, u, tag, z in zip(ys, ms, us, tags, z_trues):
        full_datas.append(dict(y=y, m=m, u=u, tag=tag, z_true=z, 
                               z_true_key=z_true_key, neuron_names=neuron_names))
    
    return train_datas, val_datas, test_datas, full_datas, tags


if __name__ == "__main__":    
    # Preprocess the data
    for version in ["kato", "nichols"]:
        processed_datas = process_data(version=version)
        with open(os.path.join("data", "processed", version + ".pkl"), "wb") as f:
            pickle.dump(processed_datas, f)
