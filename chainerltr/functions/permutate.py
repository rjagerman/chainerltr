import chainer.functions as F
from chainer import cuda


def permutate2d(x, indices):
    """
    Permutates a given variable for each row according to the indices provided.

    @param x: The parameter
    @type x: chainer.Variable
    @param indices: The indices to permute (same shape as ``x``)
    @type x: chainer.Variable
    @return: The permuted version of x
    @rtype: chainer.Variable

    .. admonition:: Example

        >>> x = np.arange(6).reshape((3, 2)).astype('f')
        >>> x
        array([[ 0.,  1.],
               [ 2.,  3.],
               [ 4.,  5.]], dtype=float32)
        >>> indices = np.array([[1, 0], [0, 1], [1, 0]], 'i')
        >>> y = permutate2d(x, indices)
        >>> y.data
        array([[ 1.,  0.],
               [ 2.,  3.],
               [ 5.,  4.]], dtype=float32)

    """
    xp = cuda.get_array_module(x, indices)
    rows = x.shape[0]
    cols = x.shape[1]
    flatten_row_range = F.reshape(xp.arange(0, rows, dtype=indices.dtype),
                                  (rows, 1))
    flatten_addition = F.broadcast_to(flatten_row_range, indices.shape) * cols
    flattened_indices = F.flatten(indices) + F.flatten(flatten_addition)
    flattened_x = F.flatten(x)
    flattened_x = flattened_x[flattened_indices.data]
    return F.reshape(flattened_x, x.shape)
