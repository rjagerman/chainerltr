from chainer import as_variable
from chainer.backends import cuda
from chainerltr.evaluation.dcg import dcg


def ndcg(ranking, relevance_scores, nr_docs=None, k=0, exp=True):
    """
    Computes the nDCG@k for given list of true relevance labels
    (relevance_labels) and given permutation of documents (permutation)

    :param ranking: The ranking of the documents
    :type ranking: chainer.Variable

    :param relevance_scores: The ground truth relevance labels
    :type relevance_scores: chainer.Variable

    :param nr_docs: A vector of the nr_docs per row
    :type nr_docs: chainer.Variable

    :param k: The cut-off point (if set to 0, it does not cut-off, if set to
              smaller than 0, it computes all possible cut-offs and returns an
              array)
    :type k: int

    :param exp: Set to true to use the exponential variant of nDCG which has a
                stronger emphasis on retrieving relevant documents
    :type exp: bool

    :return: The nDCG@k value
    :rtype: chainer.Variable
    """
    xp = cuda.get_array_module(relevance_scores)
    optimal_ranking = as_variable(xp.fliplr(xp.argsort(relevance_scores.data,
                                                       axis=1)))
    _dcg = dcg(ranking, relevance_scores, nr_docs, k, exp).data
    _idcg = dcg(optimal_ranking, relevance_scores, nr_docs, k, exp).data

    _idcg[_idcg == 0.0] = 1.0

    return as_variable(_dcg / _idcg)
