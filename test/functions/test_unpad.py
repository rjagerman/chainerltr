import numpy as np
from chainer.testing import assert_allclose
from chainer import as_variable
from chainerltr.functions import unpad


def test_unpad_basic():
    x = as_variable(np.array([[3, 2, 0, 1], [0, 1, 3, 2]]))
    print(x)
    nr_docs = as_variable(np.array([2, 3]))
    y = unpad(x, nr_docs)
    expected = as_variable(np.array([[0, 1, 3, 2], [0, 1, 2, 3]], 'f'))
    assert_allclose(y.data, expected.data)


def test_unpad_original():
    x = as_variable(np.array([[3, 2, 0, 1], [0, 1, 3, 2]]))
    print(x)
    nr_docs = as_variable(np.array([4, 4]))
    y = unpad(x, nr_docs)
    expected = as_variable(np.array([[3, 2, 0, 1], [0, 1, 3, 2]], 'f'))
    assert_allclose(y.data, expected.data)


def test_unpad_single_row():
    x = as_variable(np.array([[1, 2, 4, 0, 3]]))
    print(x)
    nr_docs = as_variable(np.array([3]))
    y = unpad(x, nr_docs)
    expected = as_variable(np.array([[1, 2, 0, 4, 3]], 'f'))
    assert_allclose(y.data, expected.data)


def test_unpad_single_column():
    x = as_variable(np.array([[1], [2], [4], [0], [3]]))
    print(x)
    nr_docs = as_variable(np.array([1, 1, 1, 1, 1]))
    y = unpad(x, nr_docs)
    expected = as_variable(np.array([[1], [2], [4], [0], [3]], 'f'))
    assert_allclose(y.data, expected.data)
