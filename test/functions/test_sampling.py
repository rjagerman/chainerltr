import numpy as np
from chainer import Variable
from chainer.testing import assert_allclose
from chainerltr.functions import sample_without_replacement


def test_sample_without_replacement_statistics_top1():
    np.random.seed(42)
    samples_per_run = 1000
    runs = 10
    p = Variable(np.repeat(np.array([[0.2, 0.025, 0.725, 0.05]]),
                           samples_per_run, axis=0))
    res = np.zeros(4)
    for i in range(runs):
        idx = sample_without_replacement(p)
        res += np.histogram(idx[:,0], bins=4)[0]
    assert_allclose(res / (samples_per_run * runs), p[0, :].data,
                    atol=1e-2, rtol=1e-2)
