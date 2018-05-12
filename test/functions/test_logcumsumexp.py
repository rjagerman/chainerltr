import numpy as np
from nose.tools import raises
from chainer import Variable
from chainer.testing import assert_allclose
from chainer.gradient_check import check_backward
from chainer.utils.type_check import InvalidType
from chainerltr.functions.logcumsumexp import logcumsumexp


def test_logcumsumexp_forward_2d():
    x = np.array([[-3.2, 1.9, 0.01],
                  [0.5, 1.2, 3.5]])
    expected = np.array([[2.04597612, 2.04069352, 0.01],
                         [3.63980187, 3.59554546, 3.5]])
    res = logcumsumexp(Variable(x))
    assert_allclose(res.data, expected)


def test_logcumsumexp_backward_2d():
    x = np.array([[-3.2, 1.9, 0.01],
                  [0.5, 1.2, 3.5]])
    check_backward(logcumsumexp, x, np.ones(x.shape))


def test_logcumsumexp_backward_2d_2():
    x = np.array([[5.6, 6.6, 7.6, 0.1],
                  [0.1, 0.5, 0.8, 1.2],
                  [0.9, 0.9, 0.9, 0.9]])
    check_backward(logcumsumexp, x, np.ones(x.shape))


@raises(InvalidType)
def test_logcumsumexp_typeerror_0d():
    x = np.array(0.2718)
    logcumsumexp(x)


@raises(InvalidType)
def test_logcumsumexp_typeerror_3d():
    x = np.array([[[0.1, 0.2, 0.3]]])
    logcumsumexp(x)
