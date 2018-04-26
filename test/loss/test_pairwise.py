import numpy as np
from nose.tools import assert_almost_equal
from chainer import Variable, gradient_check
from chainerltr.loss.pairwise import ranknet


def test_ranknet_large_loss():
    x = Variable(np.array([[3.5, 1.0, 0.0, -5.0],
                           [0.0, 0.8, 0.9, 2.0]]))
    t = Variable(np.array([[0, 0, 1, 2],
                           [2, 2, 0, 0]], dtype='i'))
    nr_docs = Variable(np.array([4, 4], dtype='i'))

    result = ranknet(x, t, nr_docs)
    assert_almost_equal(result.data, 1.928377743940217)


def test_ranknet_small_loss():
    x = Variable(np.array([[3.5, 1.0, 0.0, -5.0],
                           [0.0, 0.8, 0.9, 2.0]]))
    t = Variable(np.array([[2, 1, 0, 0],
                           [0, 1, 1, 2]], dtype='i'))
    nr_docs = Variable(np.array([4, 4], dtype='i'))

    result = ranknet(x, t, nr_docs)
    assert_almost_equal(result.data, 0.8179770090422878)


def test_ranknet_backward():
    x = Variable(np.array([[3.5, 1.0, 0.0, -5.0],
                           [0.0, 0.8, 0.9, 2.0]]))
    t = Variable(np.array([[2, 1, 0, 0],
                           [0, 1, 1, 2]], dtype='i'))
    nr_docs = Variable(np.array([4, 4], dtype='i'))

    gradient_check.check_backward(ranknet, (x.data, t.data, nr_docs.data), None)
