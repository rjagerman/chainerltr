from chainer import cuda, functions as F


def sample_without_replacement(p_log):
    """
    Samples a permutation with log-probabilities

    :param p: The log-probabilities (each row's exp(log_p) needs to sum up to
              one)
    :return: A random permutation, sampled without replacement with proportional
             probabilities
    """
    xp = cuda.get_array_module(p_log)

    # We use reservoir sampling, which resorts to doing Uniform(0, 1) ^ (1 / p)
    # and then sorting by the resulting values. The following implementation is
    # a numerically stable variant that operates in log-space and uses
    # GPU-accelerated operations.
    u = xp.random.uniform(0.0, 1.0, p_log.shape)
    r = xp.log(-xp.log(u)) - p_log.data
    s = xp.argsort(r, axis=1)
    return s
