import numpy as np

from chainer import as_variable
from chainer.testing import assert_allclose
from nose.tools import raises

from chainerltr.clickmodels.behavior import UserBehavior, NavigationalBehavior, \
    InformationalBehavior


@raises(NotImplementedError)
def test_behavior_rel_notimplementederror():
    labels = as_variable(np.array([[1, 2, 1, 0], [2, 1, 0, 0]], dtype='f'))
    ub = UserBehavior()
    ub.relevance_probability(labels)


@raises(NotImplementedError)
def test_behavior_stop_notimplementederror():
    labels = as_variable(np.array([[1, 2, 1, 0], [2, 1, 0, 0]], dtype='f'))
    ub = UserBehavior()
    ub.stop_probability(labels)


def test_behavior_navigational_relevance():
    labels = as_variable(np.array([[4, 4, 3, 0, 0, 1, 2]], dtype='f'))
    nb = NavigationalBehavior(maximum_relevance=4, minimum_relevance=0)
    result = nb.relevance_probability(labels)
    assert_allclose(result.data, np.array([[0.95, 0.95, 0.7, 0.05, 0.05, 0.3, 0.5]]))


def test_behavior_navigational_stop():
    labels = as_variable(np.array([[4, 4, 3, 0, 0, 1, 2]], dtype='f'))
    nb = NavigationalBehavior(maximum_relevance=4, minimum_relevance=0)
    result = nb.stop_probability(labels)
    assert_allclose(result.data, np.array([[0.9, 0.9, 0.7, 0.2, 0.2, 0.3, 0.5]]))


def test_behavior_informational_relevance():
    labels = as_variable(np.array([[4, 4, 3, 0, 0, 1, 2]], dtype='f'))
    nb = InformationalBehavior(maximum_relevance=4, minimum_relevance=0)
    result = nb.relevance_probability(labels)
    assert_allclose(result.data, np.array([[0.9, 0.9, 0.8, 0.4, 0.4, 0.6, 0.7]]))


def test_behavior_informational_stop():
    labels = as_variable(np.array([[4, 4, 3, 0, 0, 1, 2]], dtype='f'))
    nb = InformationalBehavior(maximum_relevance=4, minimum_relevance=0)
    result = nb.stop_probability(labels)
    assert_allclose(result.data, np.array([[0.5, 0.5, 0.4, 0.1, 0.1, 0.2, 0.3]]))
