import numpy as np

from chainer import as_variable
from chainer.testing import assert_allclose
from nose.tools import raises

from chainerltr.clickmodels.behavior import PerfectBehavior
from chainerltr.clickmodels.clickmodel import ClickModel


class MockClickModel(ClickModel):
    def _click_vector(self, relevance, nr_docs, rng):
        return relevance


@raises(NotImplementedError)
def test_clickmodel_unimplemented_error():

    ranking = as_variable(np.array([[0, 1, 2, 3], [0, 3, 2, 1]], dtype=np.int32))
    labels = as_variable(np.array([[1, 2, 1, 0], [2, 1, 0, 0]], dtype='f'))
    nr_docs = as_variable(np.array([4, 4], dtype=np.int32))

    cm = ClickModel(PerfectBehavior())

    cm(ranking, labels, nr_docs)


def test_clickmodel_top_1():

    ranking = as_variable(np.array([[0, 1, 2, 3], [0, 3, 2, 1]], dtype=np.int32))
    labels = as_variable(np.array([[1, 2, 1, 0], [2, 1, 0, 0]], dtype='f'))
    nr_docs = as_variable(np.array([4, 4], dtype=np.int32))
    k = 1

    cm = MockClickModel(PerfectBehavior(), k=k)

    assert_allclose(cm(ranking, labels, nr_docs).data,
                    np.array([[1], [2]], dtype='f'))


def test_clickmodel_top_2():

    ranking = as_variable(np.array([[0, 1, 2, 3], [0, 3, 2, 1]], dtype=np.int32))
    labels = as_variable(np.array([[1, 2, 1, 0], [2, 1, 0, 0]], dtype='f'))
    nr_docs = as_variable(np.array([4, 4], dtype=np.int32))
    k = 2

    cm = MockClickModel(PerfectBehavior(), k=k)

    assert_allclose(cm(ranking, labels, nr_docs).data,
                    np.array([[1, 2], [2, 0]], dtype='f'))


def test_clickmodel_top_3():

    ranking = as_variable(np.array([[0, 1, 2, 3], [0, 3, 2, 1]], dtype=np.int32))
    labels = as_variable(np.array([[1, 2, 1, 0], [2, 1, 0, 0]], dtype='f'))
    nr_docs = as_variable(np.array([4, 4], dtype=np.int32))
    k = 3

    cm = MockClickModel(PerfectBehavior(), k=k)

    assert_allclose(cm(ranking, labels, nr_docs).data,
                    np.array([[1, 2, 1], [2, 0, 0]], dtype='f'))
