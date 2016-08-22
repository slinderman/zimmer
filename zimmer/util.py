import numpy as np

def states_to_changepoints(z):
    assert z.ndim == 1
    return np.concatenate(([0], 1 + np.where(np.diff(z))[0], [z.size - 1]))