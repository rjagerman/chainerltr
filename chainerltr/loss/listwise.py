from chainer import functions as cf, as_variable
from chainerltr.functions import loginvcumsumexp, permutate2d, argsort,\
    sample_without_replacement


def listnet(x, t, nr_docs):
    """
    The Top-1 approximated ListNet loss as in Cao et al (2006), Learning to
    Rank: From Pairwise Approach to Listwise Approach

    :param x: The activation of the previous layer
    :type x: chainer.Variable

    :param t: The target labels
    :type t: chainer.Variable

    :return: The Top-1 listnet loss
    :rtype: chainer.Variable
    """
    t, nr_docs = as_variable(t), as_variable(nr_docs)
    t = t.data.astype(x.dtype)
    st = cf.softmax(t, axis=1)
    sx = cf.softmax(x, axis=1)
    sce = -cf.mean(st * cf.log(sx), axis=1)
    return cf.mean(sce)


def listmle(x, t, nr_docs):
    """
    The ListMLE loss as in Xia et al (2008), Listwise Approach to Learning to
    Rank - Theory and Algorithm.

    :param x: The activation of the previous layer
    :type x: chainer.Variable

    :param t: The target labels
    :type t: chainer.Variable

    :return: The loss
    :rtype: chainer.Variable
    """
    t, nr_docs = as_variable(t), as_variable(nr_docs)

    # Get the ground truth by sorting activations by the relevance labels
    indices = argsort(t, axis=1)
    x_hat = permutate2d(x, cf.flip(indices, axis=1))

    # Compute MLE loss
    final = loginvcumsumexp(x_hat)
    per_sample_loss = cf.sum(final - x_hat, axis=1)
    return cf.mean(per_sample_loss)


def listpl(x, t, nr_docs, α=10.0):
    """
    The ListPL loss, a stochastic variant of ListMLE that in expectation
    approximates the true ListNet loss.

    :param x: The activation of the previous layer
    :type x: chainer.Variable

    :param t: The target labels
    :type t: chainer.Variable

    :param α: The temperature parameter of the plackett-luce
    :type α: float

    :return: The loss
    :rtype: chainer.Variable
    """
    t, nr_docs = as_variable(t), as_variable(nr_docs)
    t = as_variable(t.data.astype(x.dtype))
    t = cf.softmax(t * α, axis=1)
    indices = sample_without_replacement(t)

    x_hat = permutate2d(x, indices)

    # Compute MLE loss
    final = loginvcumsumexp(x_hat)
    per_sample_loss = cf.sum(final - x_hat, axis=1)
    return cf.mean(per_sample_loss)
