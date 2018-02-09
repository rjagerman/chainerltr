from chainer import Variable, functions as cf, as_variable


def ranknet(x, t, nr_docs):
    """
    The RankNet loss as in Burges et al (2005), Learning to Rank using Gradient
    Descent

    :param x: The activation of the previous layer
    :type x: chainer.Variable

    :param t: The target labels
    :type t: chainer.Variable

    :return: The RankNet loss
    :rtype: chainer.Variable
    """
    x, t, nr_docs = as_variable(x), as_variable(t), as_variable(nr_docs)
    t = as_variable(t.data.astype(x.dtype))

    x_ij = _tiled_diff(x)
    t_ij = _tiled_diff(t)
    p_t_ij = cf.sigmoid(t_ij)

    # This loss is a simplified sigmoid cross entropy described in the paper
    c_ij = -p_t_ij * x_ij + cf.log(1.0 + cf.exp(x_ij))
    loss = cf.mean(c_ij)

    return loss


def _tiled_diff(x):
    """
    This function computes a (back-propable) matrix r_ij = x_i - x_j
    with minibatch support (that is the first axis is left unchanged)

    :param x: The minibatch of scores per documents
    :type x: chainer.Variable
    :return: The matrix r_ij
    :rtype: chainer.Variable
    """
    tiled_x = cf.reshape(cf.tile(x, x.shape[1]),
                         (x.shape[0], x.shape[1], x.shape[1]))
    x_j = tiled_x
    x_i = cf.transpose(tiled_x, axes=(0, 2, 1))
    return x_i - x_j
