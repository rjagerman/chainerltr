from chainer import as_variable, cuda
from chainerltr.functions import select_items_per_row


def unpad(permutation, nr_docs):
    """
    Unpads a permutation. The variable nr_docs indicates, for each row in the
    permutation mini-batch, how many documents were in the original entry.
    Unpadding happens by shifting documents that were padded to the end of the
    permutation while retaining the order of the other documents in the
    original.

    :param permutation: The permutation to re-arrange
    :type permutation: chainer.Variable

    :param nr_docs: The number of documents
    :type nr_docs: chainer.Variable

    :return: An unpad version of the permutation
    :rtype: chainer.Variable
    """
    xp = cuda.get_array_module(permutation, nr_docs)

    permutation_d, nr_docs_d = permutation.data, nr_docs.data

    arange = xp.broadcast_to(xp.arange(permutation.shape[1]), permutation.shape)
    arange_1 = xp.copy(arange)
    arange_2 = xp.copy(arange)

    arange_1[permutation_d >= nr_docs_d[:, None]] = permutation.shape[1] + 1
    arange_2[permutation_d < nr_docs_d[:, None]] = -1

    arange_1_s = xp.sort(arange_1)
    arange_2_s = xp.sort(arange_2)

    arange_1_s[arange_1_s == permutation.shape[1] + 1] = 0
    arange_2_s[arange_2_s == -1] = 0

    indices = arange_1_s + arange_2_s

    return select_items_per_row(permutation, as_variable(indices))
