import numpy as np

from chainer import as_variable
from nose.tools import raises

from chainerltr.clickmodels.behavior import PerfectBehavior
from chainerltr.clickmodels.clickmodel import ClickModel


@raises(NotImplementedError)
def test_clickmodel_unimplemented_error():

    ranking = as_variable(np.array([[0, 1, 2, 3], [0, 3, 2, 1]], dtype='i'))
    labels = as_variable(np.array([[1, 2, 1, 0], [2, 1, 0, 0]], dtype='f'))
    nr_docs = as_variable(np.array([4, 4]))

    cm = ClickModel(PerfectBehavior())

    cm(ranking, labels, nr_docs)
