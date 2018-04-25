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
    assert_allclose(dcm(ranking, labels, nr_docs, 42).data,
                    np.array([[1, 1, 0, 0], [1, 0, 0, 0]]))
    assert_allclose(dcm(ranking, labels, nr_docs, 52).data,
                    np.array([[0, 1, 1, 0], [1, 0, 0, 0]]))
    assert_allclose(dcm(ranking, labels, nr_docs, 53).data,
                    np.array([[0, 1, 1, 0], [1, 0, 0, 1]]))


def test_dcm_navigational():

    ranking = as_variable(np.array([[0, 1, 2, 3], [0, 3, 2, 1]], dtype=np.int32))
    labels = as_variable(np.array([[1, 2, 1, 0], [2, 1, 0, 0]], dtype='f'))
    nr_docs = as_variable(np.array([4, 4], dtype=np.int32))

    dcm = DependentClickModel(NavigationalBehavior(maximum_relevance=2))

    # Try clicks with different seeds and check results
    assert_allclose(dcm(ranking, labels, nr_docs, 42).data,
                    np.array([[1, 0, 0, 0], [1, 0, 0, 0]]))
    assert_allclose(dcm(ranking, labels, nr_docs, 52).data,
                    np.array([[0, 1, 0, 0], [1, 0, 0, 0]]))
    assert_allclose(dcm(ranking, labels, nr_docs, 53).data,
                    np.array([[0, 1, 0, 0], [1, 0, 0, 0]]))


def test_dcm_informational():

    ranking = as_variable(np.array([[0, 1, 2, 3], [0, 3, 2, 1]], dtype=np.int32))
    labels = as_variable(np.array([[1, 2, 1, 0], [2, 1, 0, 0]], dtype='f'))
    nr_docs = as_variable(np.array([4, 4], dtype=np.int32))

    dcm = DependentClickModel(InformationalBehavior(maximum_relevance=2))

    # Try clicks with different seeds and check results
    assert_allclose(dcm(ranking, labels, nr_docs, 42).data,
                    np.array([[1, 0, 0, 0], [1, 1, 1, 0]]))
    assert_allclose(dcm(ranking, labels, nr_docs, 52).data,
                    np.array([[0, 1, 0, 0], [1, 0, 1, 0]]))
    assert_allclose(dcm(ranking, labels, nr_docs, 53).data,
                    np.array([[0, 1, 1, 1], [1, 0, 0, 1]]))


def test_dcm_seed():

    ranking = as_variable(np.array([[0, 1, 2, 3], [0, 3, 2, 1]], dtype=np.int32))
    labels = as_variable(np.array([[1, 2, 1, 0], [2, 1, 0, 0]], dtype='f'))
    nr_docs = as_variable(np.array([4, 4], dtype=np.int32))

    dcm = DependentClickModel(PerfectBehavior(maximum_relevance=2))

    np.random.seed(42)
    different_without_seed = dcm(ranking, labels, nr_docs).data
    different_with_seed = dcm(ranking, labels, nr_docs, 42).data
    assert_allclose(different_without_seed, different_with_seed)

    np.random.seed(42)
    different_without_seed = dcm(ranking, labels, nr_docs).data
    different_with_seed = dcm(ranking, labels, nr_docs, 43).data

    assert not np.array_equal(different_without_seed, different_with_seed)

