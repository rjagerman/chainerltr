import chainer
from chainer.backends import cuda
from chainer import function_node
from chainer.utils import type_check


class LogCumSumExp(function_node.FunctionNode):

    def check_type_forward(self, in_types):
        type_check.expect(
            in_types.size() == 1,
            in_types[0].dtype.kind == 'f',
        )

        type_check.expect(
            in_types[0].ndim == 2,
        )

    def forward(self, inputs):
        self.retain_inputs((0,))
        self.retain_outputs((0,))
        xp = cuda.get_array_module(*inputs)

        x, = inputs
        m = x.max(axis=1, keepdims=True)
        y = x - m
        xp.exp(y, out=y)
        y = xp.fliplr(y)
        y_sum = y.cumsum(axis=1)
        y_sum = xp.fliplr(y_sum)
        y = xp.asarray(xp.log(y_sum) + xp.broadcast_to(m, y_sum.shape))
        return y,

    def backward(self, indexes, grads):
        x, = self.get_retained_inputs()
        y, = self.get_retained_outputs()
        gy, = grads

        xp = cuda.get_array_module(x)

        xs = xp.reshape(x.data, (x.shape[0], 1, x.shape[1]))
        xs = xp.broadcast_to(xs, (x.shape[0], x.shape[1], x.shape[1]))

        ys = xp.reshape(y.data, (y.shape[0], y.shape[1], 1))
        ys = xp.broadcast_to(ys, (y.shape[0], y.shape[1], y.shape[1]))

        triu = xp.broadcast_to(xp.triu(xp.ones((ys.shape[1], ys.shape[2]))),
                               ys.shape)

        gx = gy * xp.sum(xp.exp(xs - ys) * triu, axis=1)
        return gx,


def logcumsumexp(x):
    """Log-cumsum-exp of array elements over a given axis.

    This function calculates logarithm of sum of exponential of array elements.

    .. math::

       y_i = \\log\\left(\\sum_j \\exp(x_{ij})\\right)

    Args:
        x (~chainer.Variable): Elements to log-sum-exp.
        axis (None, int, or tuple of int): Axis which a sum is performed.
            The default (axis = None) is perform a sum over all the dimensions
            of the input array.

    Returns:
        ~chainer.Variable: Output variable.

    """
    return LogCumSumExp().apply((x,))[0]
