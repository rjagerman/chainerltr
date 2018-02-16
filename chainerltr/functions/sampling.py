from chainer import cuda


def sample_without_replacement(p):
    """
    Samples a permutation with probabilities scaled to p

    :param p: The probabilities (each row needs to sum up to one)
    :return: A random permutation, sampled without replacement with proportional
             probabilities
    """
    xp = cuda.get_array_module(p)

    # We use reservoir sampling, which resorts to doing Uniform(0, 1) ^ (1 / p)
    # and then sorting by the resulting values
    r = xp.random.uniform(0.0, 1.0, p.shape) ** (1. / p.data)
    s = xp.flip(xp.argsort(r, axis=1), axis=1)
    return s
