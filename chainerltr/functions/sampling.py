from chainer import cuda


def sample_without_replacement(p_log, rng=None, seed=None):
    """
    Samples a permutation with log-probabilities

    :param p_log: The log-probabilities (each row's exp(log_p) needs to sum up
                  to one)
    :type p_log: chainer.Variable

    :param rng: The random number generator (default is to use global random
                number generator in either cupy or numpy)
    :type rng: numpy.random.RandomState|cupy.random.RandomState|None

    :return: A random permutation, sampled without replacement with proportional
             probabilities
    :rtype: chainer.Variable
    """
    xp = cuda.get_array_module(p_log)

    # If a specific random seed is set, generate a random number generator with
    # that seed to sample from, otherwise use the global random library
    if rng is None:
        rng = xp.random
    if seed is not None:
        rng.seed(seed)

    # This uses reservoir sampling, which comes down to doing
    # Uniform(0, 1) ^ (1 / p) and then sorting by the resulting values. The
    # following implementation is a numerically stable variant that operates in
    # log-space and uses GPU-accelerated operations.
    u = rng.uniform(0.0, 1.0, p_log.shape)
    r = xp.log(-xp.log(u)) - p_log.data
    s = xp.argsort(r, axis=1)
    return s
