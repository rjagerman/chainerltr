import numpy as np
from nose.tools import raises
from chainer import Variable
from chainer.testing import assert_allclose
from chainer.gradient_check import check_backward
from chainerltr.functions import loginvcumsumexp


def test_loginvcumsumexp_forward_2d():
    x = np.array([[-3.2, 1.9, 0.01],
                  [0.5, 1.2, 3.5]])
    expected = np.array([[2.04597612, 2.04069352, 0.01],
                         [3.63980187, 3.59554546, 3.5]])
    res = loginvcumsumexp(Variable(x))
    assert_allclose(res.data, expected)


def test_loginvcumsumexp_backward_2d():
    x = np.array([[-3.2, 1.9, 0.01],
                  [0.5, 1.2, 3.5]])
    check_backward(loginvcumsumexp, x, np.ones(x.shape))


def test_loginvcumsumexp_forward_1d():
    x = np.array([-3.2, 1.9, 0.01])
    expected = np.array([2.04597612, 2.04069352, 0.01])
    res = loginvcumsumexp(Variable(x))
    assert_allclose(res.data, expected)


def test_loginvcumsumexp_backward_1d():
    x = np.array([-3.2, 1.9, 0.01])
    check_backward(loginvcumsumexp, x, np.ones(x.shape))


@raises(TypeError)
def test_loginvcumsumexp_typeerror_0d():
    x = np.array(0.2718)
    loginvcumsumexp(x)


@raises(TypeError)
def test_loginvcumsumexp_typeerror_3d():
    x = np.array([[[0.1, 0.2, 0.3]]])
    loginvcumsumexp(x)
