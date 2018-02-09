from chainer import cuda
from chainer import function
from chainer.utils import type_check


class ArgSort(function.Function):

    def __init__(self, axis=0):
        self.axis = axis

    def check_type_forward(self, in_types):
        type_check.expect(
            in_types.size() == 1,
        )

    def forward(self, inputs):
        xp = cuda.get_array_module(*inputs)
        x, = inputs
        out = xp.argsort(x, axis=self.axis)
        return xp.array(out, copy=False),


def argsort(x, axis=0):
    """Arg-sort of array elements.

    This function calculates the argsort of the given array

    Args:
        x (~chainer.Variable): Elements to argsort.

    Returns:
        ~chainer.Variable: Output variable.

    """
    return ArgSort(axis)(x)
