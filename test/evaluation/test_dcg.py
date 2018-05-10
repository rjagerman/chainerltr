import numpy as np
from chainer.testing import assert_allclose
from chainer import Variable
from chainerltr.evaluation import dcg
from nose.tools import raises


def test_perfect_dcg():
    p = Variable(np.array([[0, 1, 2, 3, 4]]))
    y = Variable(np.array([[2,  2,  1,  0,  0]]))
    assert_allclose(dcg(p, y).data, 5.392789)


def test_worst_dcg():
    p = Variable(np.array([[4, 3, 2, 0, 1]]))
    y = Variable(np.array([[2,  2,  1,  0,  0]]))
    assert_allclose(dcg(p, y).data, 2.9525881)


def test_perfect_dcg_no_exp():
    p = Variable(np.array([[0, 1, 2, 3, 4]]))
    y = Variable(np.array([[2, 2, 1, 0, 0]]))
    assert_allclose(dcg(p, y, exp=False).data, 3.76185951)


def test_worst_dcg_no_exp():
    p = Variable(np.array([[4, 3, 2, 0, 1]]))
    y = Variable(np.array([[2, 2, 1, 0, 0]]))
    assert_allclose(dcg(p, y, exp=False).data, 2.13505873)

def test_perfect_dcg_at_1():
    p = Variable(np.array([[0, 1, 2, 3, 4]]))
    y = Variable(np.array([[2, 2, 1, 0, 0]]))
    assert_allclose(dcg(p, y, k=1).data, 3.0)


def test_worst_dcg_at_1():
    p = Variable(np.array([[4, 3, 2, 0, 1]]))
    y = Variable(np.array([[2, 2, 1, 0, 0]]))
    assert_allclose(dcg(p, y, k=1).data, 0.0)


def test_perfect_dcg_at_3():
    p = Variable(np.array([[0, 1, 2, 3, 4]]))
    y = Variable(np.array([[2, 2, 1, 0, 0]]))
    assert_allclose(dcg(p, y, k=3).data, 5.392789260714372)


def test_worst_dcg_at_3():
    p = Variable(np.array([[4, 3, 2, 1, 0]]))
    y = Variable(np.array([[2, 2, 1, 0, 0]]))
    assert_allclose(dcg(p, y, k=3).data, 0.5)


def test_perfect_dcg_at_5():
    p = Variable(np.array([[0, 1, 2, 3, 4]]))
    y = Variable(np.array([[2, 2, 1, 0, 0]]))
    assert_allclose(dcg(p, y, k=5).data, 5.392789260714372)


def test_worst_dcg_at_5():
    p = Variable(np.array([[4, 3, 2, 0, 1]]))
    y = Variable(np.array([[2, 2, 1, 0, 0]]))
    assert_allclose(dcg(p, y, k=5).data, 2.9525880959238044)


def test_zero_dcg():
    p = Variable(np.array([[4, 3, 2, 1, 0]]))
    y = Variable(np.array([[0, 0, 0, 0, 0]]))
    assert_allclose(dcg(p, y).data, 0.0)


def test_empty_dcg():
    p = Variable(np.array([[]]))
    y = Variable(np.array([[]]))
    assert_allclose(dcg(p, y).data, 0.0)


def test_perfect_dcg_at_all():
    p = Variable(np.array([[0, 1, 2, 3, 4]]))
    y = Variable(np.array([[2, 2, 1, 1, 0]]))
    expected = np.array([[3., 4.89278926, 5.39278926, 5.82346582, 5.82346582]])
    assert_allclose(dcg(p, y, k=-1).data, expected)


def test_worst_dcg_at_all():
    p = Variable(np.array([[4, 3, 2, 1, 0]]))
    y = Variable(np.array([[2, 2, 1, 1, 0]]))
    expected = np.array([[0., 0.63092975, 1.13092975, 2.42295943, 3.58351785]])
    assert_allclose(dcg(p, y, k=-1).data, expected)


@raises(ValueError)
def test_wrongsize_dcg():
    p = Variable(np.array([[2, 1, 0]]))
    y = Variable(np.array([[]]))
    assert_allclose(dcg(p, y).data, 0.0)


@raises(ValueError)
def test_wrongsize2_dcg():
    p = Variable(np.array([[]]))
    y = Variable(np.array([[1, 1, 0]]))
    assert_allclose(dcg(p, y).data, 0.0)


@raises(ValueError)
def test_wrongsize3_dcg():
    p = Variable(np.array([[0, 1]]))
    y = Variable(np.array([[1, 1, 0]]))
    assert_allclose(dcg(p, y).data, 0.0)


def test_perfect_dcg_2d():
    p = Variable(np.array([[0, 1, 2, 3, 4],
                           [0, 1, 2, 4, 3]]))
    y = Variable(np.array([[2, 2, 1, 0, 0],
                           [2, 1, 0, 0, 0]]))
    nr_docs = Variable(np.array([5, 3]))
    expected = np.array([5.39278926, 3.63092975])
    assert_allclose(dcg(p, y, nr_docs=nr_docs).data, expected)


def test_worst_dcg_2d():
    p = Variable(np.array([[4, 3, 2, 1, 0],
                           [2, 1, 0, 4, 3]]))
    y = Variable(np.array([[2, 2, 1, 0, 0],
                           [2, 1, 0, 0, 0]]))
    nr_docs = Variable(np.array([5, 3]))
    expected = np.array([2.9525881, 2.13092975])
    assert_allclose(dcg(p, y, nr_docs=nr_docs).data, expected)


def test_random_dcg_2d():
    p = Variable(np.array([[4, 3, 2, 1, 0],
                           [0, 1, 2, 3, 4]]))
    y = Variable(np.array([[0, 0, 1, 1, 0],
                           [0, 1, 2, 0, 0]]))
    nr_docs = Variable(np.array([5, 3]))
    expected = np.array([1.13092975, 2.13092975])
    assert_allclose(dcg(p, y, nr_docs=nr_docs).data, expected)


def test_random_dcg_2d_no_nrdocs():
    p = Variable(np.array([[4, 3, 2, 1, 0],
                           [0, 3, 1, 2, 4]]))
    y = Variable(np.array([[0, 0, 1, 1, 0],
                           [0, 1, 2, 0, 0]]))
    expected = np.array([1.1309297535714575, 1.7920296742201793])
    assert_allclose(dcg(p, y).data, expected)


@raises(ValueError)
def test_dcg_wrongdim():
    p = Variable(np.array([[[4, 3, 2, 1, 0],
                            [0, 1, 2, 4, 3]]]))
    y = Variable(np.array([[[0, 0, 1, 1, 0],
                            [0, 1, 2, 0, 0]]]))
    nr_docs = Variable(np.array([5]))
    expected = np.array([1.13092975, 2.13092975])
    assert_allclose(dcg(p, y, nr_docs=nr_docs).data, expected)
