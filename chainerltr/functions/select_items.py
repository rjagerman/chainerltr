from chainer import cuda
import chainer.functions as F


def select_items_per_row(values2d, idx2d):
    """
    Selects items from 2-dimensional tensors (matrices) per row by given indices

    :param values2d: The values to choose from
    :type values2d: chainer.Variable

    :param idx2d: The indices to select
    :type idx2d: chainer.Variable

    :return: A matrix with the same shape as values2d but with values selected
    :rtype: chainer.Variable
    """
    xp = cuda.get_array_module(values2d, idx2d)
    rows = values2d.shape[0]
    cols = values2d.shape[1]

    flattened_idx = xp.ravel(idx2d.data) + xp.repeat(
        xp.arange(0, rows * cols, cols), cols)

    flattened_values = F.flatten(values2d)
    flattened_values = flattened_values[flattened_idx]
    return F.reshape(flattened_values, values2d.shape)
