import numpy as np
from chainer import cuda


def sample_without_replacement(p):
    """
    Samples a permutation with probabilities scaled to p

    :param p: The probabilities (each row needs to sum up to one)
    :return: A random permutation, sampled without replacement with proportional
             probabilities
    """

    # Switch to CPU since xp.random.choice with replace=False is not
    # currently implemented in cupy, so we resort to numpy
    xp = cuda.get_array_module(p)
    a = cuda.to_cpu(p.data)

    # Return stacked version of random choice
    results = []
    for i in range(a.shape[0]):
        probabilities = a[i, :]
        sample = np.random.choice(probabilities.shape[0],
                                  np.count_nonzero(probabilities),
                                  replace=False,
                                  p=probabilities)
        uniform = np.arange(probabilities.shape[0])
        uniform = np.setdiff1d(uniform, sample, assume_unique=True)
        uniform = np.random.permutation(uniform)
        results.append(np.hstack((sample, uniform)))
    result = np.stack(results)

    return xp.array(result, copy=False)
