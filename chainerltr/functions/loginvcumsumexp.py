from chainer.functions import logsumexp, reshape, hstack


def loginvcumsumexp(x):
    """

    This function calculates logarithm of an inverted cumulative sum of
    exponential of array elements.

    .. math::

       y_i = \\log\\left(\\sum_{j=i\\ldots} \\exp(x_{ij})\\right)

    Args:
        x (~chainer.Variable): Elements to log-cumsum-exp.

    Returns:
        ~chainer.Variable: Output variable.

    """
    if x.ndim == 1:
        return reshape(
            hstack(logsumexp(x[i:]) for i in range(x.shape[0])),
            x.shape
        )
    elif x.ndim == 2:
        return reshape(
            hstack(reshape(logsumexp(x[:, i:], axis=1), (x.shape[0], 1))
                   for i in range(x.shape[1])),
            x.shape
        )
    else:
        raise TypeError("loginvcumsumexp can only be applied to 1 or 2 "
                        "dimensional tensors")
