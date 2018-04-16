import numpy as np

from chainer import as_variable
from nose.tools import raises

from chainerltr.clickmodels.behavior import UserBehavior


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
