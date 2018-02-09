import numpy as np
from chainer import Variable
from chainer.testing import assert_allclose
from chainerltr.functions import argsort


def test_argsort_axis1():
    x = Variable(np.array([[3.0, 1.0, 2.0],
                           [1.0, 2.0, 5.0],
                           [6.6, 2.3, 2.3001]]))
    res = argsort(x, axis=1)
    expected = np.array([[1, 2, 0],
                         [0, 1, 2],
                         [1, 2, 0]], 'i')
    assert_allclose(res.data, expected)


def test_argsort_axis0():
    x = Variable(np.array([[2.0, 2.0, 2.3],
                           [0.5, 0.9, 9.8],
                           [10.3, 0.9001, 9.7999]]))
    res = argsort(x, axis=0)
    expected = np.array([[1, 1, 0],
                         [0, 2, 2],
                         [2, 0, 1]], 'i')
    assert_allclose(res.data, expected)
