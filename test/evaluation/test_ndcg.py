import numpy as np
from chainer.testing import assert_allclose
from chainer import Variable
from chainerltr.evaluation import ndcg
from nose.tools import raises


def test_perfect_ndcg():
    p = Variable(np.array([4., 3., 2., 1., 0.]))
    y = Variable(np.array([2,  2,  1,  0,  0]))
    assert_allclose(ndcg(p, y).data, 1.0)


def test_worst_ndcg():
    p = Variable(np.array([1., 0., 2., 3., 4.]))
    y = Variable(np.array([2,  2,  1,  0,  0]))
    assert_allclose(ndcg(p, y).data, 0.5475066710714894)


def test_perfect_ndcg_no_exp():
    p = Variable(np.array([4., 3., 2., 1., 0.]))
    y = Variable(np.array([2, 2, 1, 0, 0]))
    assert_allclose(ndcg(p, y, exp=False).data, 1.0)


def test_worst_ndcg_no_exp():
    p = Variable(np.array([1., 0., 2., 3., 4.]))
    y = Variable(np.array([2, 2, 1, 0, 0]))
    assert_allclose(ndcg(p, y, exp=False).data, 0.567554085037434)

def test_perfect_ndcg_at_1():
    p = Variable(np.array([4., 3., 2., 1., 0.]))
    y = Variable(np.array([2, 2, 1, 0, 0]))
    assert_allclose(ndcg(p, y, k=1).data, 1.0)


def test_worst_ndcg_at_1():
    p = Variable(np.array([1., 0., 2., 3., 4.]))
    y = Variable(np.array([2, 2, 1, 0, 0]))
    assert_allclose(ndcg(p, y, k=1).data, 0.0)


def test_perfect_ndcg_at_3():
    p = Variable(np.array([4., 3., 2., 1., 0.]))
    y = Variable(np.array([2, 2, 1, 0, 0]))
    assert_allclose(ndcg(p, y, k=3).data, 1.0)


def test_worst_ndcg_at_3():
    p = Variable(np.array([1., 0., 2., 3., 4.]))
    y = Variable(np.array([2, 2, 1, 0, 0]))
    assert_allclose(ndcg(p, y, k=3).data, 0.09271639884807328)


def test_perfect_ndcg_at_5():
    p = Variable(np.array([4., 3., 2., 1., 0.]))
    y = Variable(np.array([2, 2, 1, 0, 0]))
    assert_allclose(ndcg(p, y, k=5).data, 1.0)


def test_worst_ndcg_at_5():
    p = Variable(np.array([1., 0., 2., 3., 4.]))
    y = Variable(np.array([2, 2, 1, 0, 0]))
    assert_allclose(ndcg(p, y, k=5).data, 0.5475066710714894)


def test_zero_ndcg():
    p = Variable(np.array([0., 1., 2., 3., 4.]))
    y = Variable(np.array([0, 0, 0, 0, 0]))
    assert_allclose(ndcg(p, y).data, 0.0)


def test_empty_ndcg():
    p = Variable(np.array([]))
    y = Variable(np.array([]))
    assert_allclose(ndcg(p, y).data, 0.0)


@raises(ValueError)
def test_wrongsize_ndcg():
    p = Variable(np.array([0., 1., 2.]))
    y = Variable(np.array([]))
    assert_allclose(ndcg(p, y).data, 0.0)


@raises(ValueError)
def test_wrongsize2_ndcg():
    p = Variable(np.array([]))
    y = Variable(np.array([1, 1, 0]))
    assert_allclose(ndcg(p, y).data, 0.0)


@raises(ValueError)
def test_wrongsize3_ndcg():
    p = Variable(np.array([1., 0.]))
    y = Variable(np.array([1, 1, 0]))
    assert_allclose(ndcg(p, y).data, 0.0)


def test_perfect_ndcg_2d():
    p = Variable(np.array([[4, 3, 2, 1, 0],
                           [3, 2, 1, 0, 0]]))
    y = Variable(np.array([[2, 2, 1, 0, 0],
                           [2, 1, 0, 0, 0]]))
    nr_docs = Variable(np.array([5, 3]))
    expected = np.array([1.0, 1.0])
    assert_allclose(ndcg(p, y, nr_docs=nr_docs).data, expected)


def test_worst_ndcg_2d():
    p = Variable(np.array([[0, 1, 2, 3, 4],
                           [1, 2, 3, 0, 0]]))
    y = Variable(np.array([[2, 2, 1, 0, 0],
                           [2, 1, 0, 0, 0]]))
    nr_docs = Variable(np.array([5, 3]))
    expected = np.array([0.54750667, 0.58688267])
    assert_allclose(ndcg(p, y, nr_docs=nr_docs).data, expected)


def test_random_ndcg_2d():
    p = Variable(np.array([[0, 1, 2, 3, 4],
                           [3, 2, 1, 0, 0]]))
    y = Variable(np.array([[0, 0, 1, 1, 0],
                           [0, 1, 2, 0, 0]]))
    nr_docs = Variable(np.array([5, 3]))
    expected = np.array([0.6934264, 0.58688267])
    assert_allclose(ndcg(p, y, nr_docs=nr_docs).data, expected)


def test_random_ndcg_2d_no_nrdocs():
    p = Variable(np.array([[0, 1, 2, 3, 4],
                           [3, 2, 1, 2, 0]]))
    y = Variable(np.array([[0, 0, 1, 1, 0],
                           [0, 1, 2, 0, 0]]))
    expected = np.array([0.6934264, 0.4935456744811716])
    assert_allclose(ndcg(p, y).data, expected)


@raises(TypeError)
def test_ndcg_wrongdim():
    p = Variable(np.array([[[0, 1, 2, 3, 4],
                            [3, 2, 1, 0, 0]]]))
    y = Variable(np.array([[[0, 0, 1, 1, 0],
                            [0, 1, 2, 0, 0]]]))
    nr_docs = Variable(np.array([5]))
    expected = np.array([1.13092975, 2.13092975])
    assert_allclose(ndcg(p, y, nr_docs=nr_docs).data, expected)


@raises(TypeError)
def test_ndcg_wrongdim():
    p = Variable(np.array([[[0, 1, 2, 3, 4],
                            [3, 2, 1, 0, 0]]]))
    y = Variable(np.array([[[0, 0, 1, 1, 0],
                            [0, 1, 2, 0, 0]]]))
    nr_docs = Variable(np.array([5]))
    expected = np.array([1.13092975, 2.13092975])
    assert_allclose(ndcg(p, y, nr_docs=nr_docs).data, expected)
