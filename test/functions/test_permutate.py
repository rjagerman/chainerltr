import numpy as np
from chainer.testing import assert_allclose
from chainer import Variable
from chainerltr.functions import permutate2d


def test_permutate2d_basic():
    x = Variable(np.arange(6).reshape((3, 2)).astype('f'))
    indices = Variable(np.array([[1, 0], [0, 1], [1, 0]], 'i'))
    y = permutate2d(x, indices)
    expected = Variable(np.array([[1, 0], [2, 3], [5, 4]], 'f'))
    assert_allclose(y.data, expected.data)


def test_permutate2d_original():
    x = Variable(np.arange(6).reshape((2, 3)).astype('f'))
    indices = Variable(np.array([[0, 1, 2], [0, 1, 2]], 'i'))
    y = permutate2d(x, indices)
    expected = Variable(np.array([[0, 1, 2], [3, 4, 5]], 'f'))
    assert_allclose(y.data, expected.data)


def test_permutate2d_single_row():
    x = Variable(np.arange(9).reshape((1, 9)).astype('f'))
    indices = Variable(np.array([[8, 1, 0, 2, 3, 5, 6, 4, 7]], 'i'))
    y = permutate2d(x, indices)
    expected = Variable(np.array([[8, 1, 0, 2, 3, 5, 6, 4, 7]], 'f'))
    assert_allclose(y.data, expected.data)


def test_permutate2d_single_column():
    x = Variable(np.arange(4).reshape((4, 1)).astype('f'))
    indices = Variable(np.array([[0], [0], [0], [0]], 'i'))
    y = permutate2d(x, indices)
    expected = Variable(np.array([[0], [1], [2], [3]], 'f'))
    assert_allclose(y.data, expected.data)

