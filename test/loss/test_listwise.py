import numpy as np
from nose.tools import assert_almost_equal
from chainer import Variable, gradient_check
from chainerltr.loss.listwise import listnet, listmle, listpl


def test_listmle():
    x = Variable(np.array([[0.5, 1.0, 0.3, 0.5],
                           [0.5, 1.0, 0.3, 0.5]]))
    t = Variable(np.array([[3, 0, 2, 0],
                           [2, 3, 2, 0]], dtype='i'))
    nr_docs = Variable(np.array([4, 4], dtype='i'))

    result = listmle(x, t, nr_docs)
    assert_almost_equal(result.data, 3.420283678027906)


def test_listmle_backward():
    np.random.seed(42)
    x = Variable(np.array([[0.5, 1.0, 0.3, 0.5],
                           [0.5, 1.0, 0.3, 0.5]]))
    t = Variable(np.array([[3, 0, 2, 0],
                           [2, 3, 2, 0]], dtype='i'))
    nr_docs = Variable(np.array([4, 4], dtype='i'))

    gradient_check.check_backward(listmle, (x.data, t.data, nr_docs.data), None)


def test_listnet():
    x = Variable(np.array([[0.5, 1.0, 0.3, 0.5],
                           [0.5, 1.0, 0.3, 0.5]]))
    t = Variable(np.array([[3, 0, 2, 0],
                           [2, 3, 2, 0]], dtype='i'))
    nr_docs = Variable(np.array([4, 4], dtype='i'))

    result = listnet(x, t, nr_docs)
    assert_almost_equal(result.data, 0.34849889467596895)


def test_listnet_backward():
    np.random.seed(42)
    x = Variable(np.array([[0.5, 1.0, 0.3, 0.5],
                           [0.5, 1.0, 0.3, 0.5]]))
    t = Variable(np.array([[3, 0, 2, 0],
                           [2, 3, 2, 0]], dtype='i'))
    nr_docs = Variable(np.array([4, 4], dtype='i'))

    gradient_check.check_backward(listnet, (x.data, t.data, nr_docs.data), None)


def test_listpl():
    x = Variable(np.array([[0.5, 1.0, 0.3, 0.5],
                           [0.5, 1.0, 0.3, 0.5]]))
    t = Variable(np.array([[3, 0, 2, 0],
                           [2, 3, 2, 0]], dtype='i'))
    nr_docs = Variable(np.array([4, 4], dtype='i'))

    result = listpl(x, t, nr_docs)
    assert_almost_equal(result.data, 3.1702836780279053)


def test_listpl_backward():
    np.random.seed(42)
    x = Variable(np.array([[0.5, 1.0, 0.3, 0.5],
                           [0.5, 1.0, 0.3, 0.5]]))
    t = Variable(np.array([[3, 0, 2, 0],
                           [2, 3, 2, 0]], dtype='i'))
    nr_docs = Variable(np.array([4, 4], dtype='i'))

    # Modify listpl call so it uses the same random seed (and thus samples
    # identically) on every call, this is necessary for gradient checking
    def loss_fn(x, t, n):
        np.random.seed(42)
        return listpl(x, t, n)
    gradient_check.check_backward(loss_fn, (x.data, t.data, nr_docs.data), None)
