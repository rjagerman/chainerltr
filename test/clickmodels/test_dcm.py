import numpy as np
from nose.tools import assert_not_almost_equals, assert_not_equals
from chainer.testing import assert_allclose
from chainer import as_variable
from chainerltr.clickmodels import DependentClickModel
from chainerltr.clickmodels.behavior import PerfectBehavior, \
    NavigationalBehavior, InformationalBehavior


def test_dcm_perfect():

    ranking = as_variable(np.array([[0, 1, 2, 3], [0, 3, 2, 1]], dtype=np.int32))
    labels = as_variable(np.array([[1, 2, 1, 0], [2, 1, 0, 0]], dtype='f'))
    nr_docs = as_variable(np.array([4, 4], dtype=np.int32))

    dcm = DependentClickModel(PerfectBehavior(maximum_relevance=2))

    # Try clicks with different seeds and check results
    np.random.seed(42)
    assert_allclose(dcm(ranking, labels, nr_docs).data,
                    np.array([[1, 1, 0, 0], [1, 0, 0, 0]]))
    np.random.seed(52)
    assert_allclose(dcm(ranking, labels, nr_docs).data,
                    np.array([[0, 1, 1, 0], [1, 0, 0, 0]]))
    np.random.seed(53)
    assert_allclose(dcm(ranking, labels, nr_docs).data,
                    np.array([[0, 1, 1, 0], [1, 0, 0, 1]]))


def test_dcm_navigational():

    ranking = as_variable(np.array([[0, 1, 2, 3], [0, 3, 2, 1]], dtype=np.int32))
    labels = as_variable(np.array([[1, 2, 1, 0], [2, 1, 0, 0]], dtype='f'))
    nr_docs = as_variable(np.array([4, 4], dtype=np.int32))

    dcm = DependentClickModel(NavigationalBehavior(maximum_relevance=2))

    # Try clicks with different seeds and check results
    np.random.seed(42)
    assert_allclose(dcm(ranking, labels, nr_docs).data,
                    np.array([[1, 0, 0, 0], [1, 0, 0, 0]]))
    np.random.seed(52)
    assert_allclose(dcm(ranking, labels, nr_docs).data,
                    np.array([[0, 1, 0, 0], [1, 0, 0, 0]]))
    np.random.seed(53)
    assert_allclose(dcm(ranking, labels, nr_docs).data,
                    np.array([[0, 1, 0, 0], [1, 0, 0, 0]]))


def test_dcm_informational():

    ranking = as_variable(np.array([[0, 1, 2, 3], [0, 3, 2, 1]], dtype=np.int32))
    labels = as_variable(np.array([[1, 2, 1, 0], [2, 1, 0, 0]], dtype='f'))
    nr_docs = as_variable(np.array([4, 4], dtype=np.int32))

    dcm = DependentClickModel(InformationalBehavior(maximum_relevance=2))

    # Try clicks with different seeds and check results
    np.random.seed(42)
    assert_allclose(dcm(ranking, labels, nr_docs).data,
                    np.array([[1, 0, 0, 0], [1, 1, 1, 0]]))
    np.random.seed(52)
    assert_allclose(dcm(ranking, labels, nr_docs).data,
                    np.array([[0, 1, 0, 0], [1, 0, 1, 0]]))
    np.random.seed(53)
    assert_allclose(dcm(ranking, labels, nr_docs).data,
                    np.array([[0, 1, 1, 1], [1, 0, 0, 1]]))

