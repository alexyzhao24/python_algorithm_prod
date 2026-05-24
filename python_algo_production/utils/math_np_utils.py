import numpy as np

# find the sum of a vectorthat has the sign of the maximal absolute item
def vec_sqrt_sign(vec):
    idx = np.argmax(np.abs(vec))

    value = np.sign(vec[idx])*np.sqrt(sum(vec**2))
    return value


def get_numpy_array_from_matrix(x):
    """
    get build-in list from matrix
    """
    return np.array(x).flatten()
