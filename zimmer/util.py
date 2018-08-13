import os
import pickle
import numpy as np

def states_to_changepoints(z):
    assert z.ndim == 1
    return np.concatenate(([0], 1 + np.where(np.diff(z))[0], [z.size - 1]))

def cached(results_dir, results_name):
    def _cache(func):
        def func_wrapper(*args, **kwargs):
            results_file = os.path.join(results_dir, results_name)
            if not results_file.endswith(".pkl"):
                results_file += ".pkl"

            if os.path.exists(results_file):
                with open(results_file, "rb") as f:
                    results = pickle.load(f)
            else:
                results = func(*args, **kwargs)
                with open(results_file, "wb") as f:
                    pickle.dump(results, f)

            return results
        return func_wrapper

    return _cache