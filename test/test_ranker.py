import numpy as np
from chainer import links as L
from chainer.testing import assert_allclose
from chainerltr import Ranker
from nose.tools import raises


def test_ranker():
    xs = np.array([[[0.5, 0.9, -0.3, 0.034]],
                   [[-0.1, 0.5, -0.4, 0.99]]], dtype=np.float32)
    W = np.array([0.01, 0.05, 0.03, 0.02], dtype=np.float32)
    predictor = L.Linear(None, 1, initialW=W)
    ranker = Ranker(predictor)

    expected = np.array([[0.04168], [0.0318]])
    assert_allclose(ranker(xs).data, expected)


@raises(TypeError)
def test_ranker_typeerror_2d():
    xs = np.array([[0.5, 0.9, -0.3, 0.034],
                   [-0.1, 0.5, -0.4, 0.99]], dtype=np.float32)
    W = np.array([0.01, 0.05, 0.03, 0.02], dtype=np.float32)
    predictor = L.Linear(None, 1, initialW=W)
    ranker = Ranker(predictor)
    ranker(xs)


@raises(TypeError)
def test_ranker_typeerror_4d():
    xs = np.array([[[[0.5, 0.9, -0.3, 0.034]],
                    [[-0.1, 0.5, -0.4, 0.99]]]], dtype=np.float32)
    W = np.array([0.01, 0.05, 0.03, 0.02], dtype=np.float32)
    predictor = L.Linear(None, 1, initialW=W)
    ranker = Ranker(predictor)
    ranker(xs)
