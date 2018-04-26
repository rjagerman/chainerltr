import numpy as np
from chainer import as_variable
from chainer.testing import assert_allclose
from chainerltr.functions import select_items_per_row, inverse_select_items_per_row


def test_select_items_identity():
    idx = as_variable(np.array([[0, 1, 2, 3], [0, 1, 2, 3]]))
    val = as_variable(np.array([[0.5, 3.14, 0.0, -9.9], [1.0, -1.0, 1.0, 4.0]]))

    out = select_items_per_row(val, idx)

    assert_allclose(out.data, val.data)


def test_select_items_none():
    idx = as_variable(np.array([[], []], dtype=np.int32))
    val = as_variable(np.array([[0.5, 3.14, 0.0, -9.9], [1.0, -1.0, 1.0, 4.0]]))

    out = select_items_per_row(val, idx)

    assert_allclose(out.data, np.array([[], []], dtype=np.int32))


def test_select_items_permuted():
    idx = as_variable(np.array([[3, 1, 0, 2], [1, 0, 3, 2]]))
    val = as_variable(np.array([[0.5, 3.14, 0.0, -9.9], [1.0, -1.0, 1.0, 4.0]]))
    exp = as_variable(np.array([[-9.9, 3.14, 0.5, 0.0], [-1.0, 1.0, 4.0, 1.0]]))

    out = select_items_per_row(val, idx)

    assert_allclose(out.data, exp.data)


def test_select_items_less_idx():
    idx = as_variable(np.array([[3, 1], [1, 3]]))
    val = as_variable(np.array([[0.5, 3.14, 0.0, -9.9], [1.0, -1.0, 1.0, 4.0]]))
    exp = as_variable(np.array([[-9.9, 3.14], [-1.0, 4.0]]))

    out = select_items_per_row(val, idx)

    assert_allclose(out.data, exp.data)


def test_inv_select_items_identity():
    idx = as_variable(np.array([[], []], dtype=np.int32))
    val = as_variable(np.array([[0.5, 3.14, 0.0, -9.9], [1.0, -1.0, 1.0, 4.0]]))

    out = inverse_select_items_per_row(val, idx)

    assert_allclose(out.data, val.data)


def test_inv_select_items_none():
    idx = as_variable(np.array([[0, 1, 2, 3], [0, 1, 2, 3]], dtype=np.int32))
    val = as_variable(np.array([[0.5, 3.14, 0.0, -9.9], [1.0, -1.0, 1.0, 4.0]]))

    out = inverse_select_items_per_row(val, idx)

    assert_allclose(out.data, np.array([[], []], dtype=np.int32))


def test_inv_select_items_less_idx():
    idx = as_variable(np.array([[3, 1], [1, 3]]))
    val = as_variable(np.array([[0.5, 3.14, 0.0, -9.9], [1.0, -1.0, 1.0, 4.0]]))
    exp = as_variable(np.array([[0.5, 0.0], [1.0, 1.0]]))

    out = inverse_select_items_per_row(val, idx)

    assert_allclose(out.data, exp.data)
