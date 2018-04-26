from chainer import cuda, as_variable
from chainerltr.evaluation.dcg import dcg


def ndcg(ranking, relevance_scores, nr_docs=None, k=0, exp=True):
    """
    Computes the nDCG@k for given list of true relevance labels
    (relevance_labels) and given permutation of documents (permutation)

    :param ranking: The ranking of documents
    :param relevance_scores: The ground truth relevance labels
    :param k: The cut-off point (if set to smaller or equal to 0, it does not
              cut-off)
    :param exp: Set to true to use the exponential variant of nDCG which
                has a stronger emphasis on retrieving relevant documents
    :param nr_docs: When using 2d-arrays you need to specify a vector of the
                    nr_docs per row (assumed zero-padding)
    :return: The nDCG@k value
    """
    xp = cuda.get_array_module(relevance_scores)
    optimal_ranking = as_variable(xp.fliplr(xp.argsort(relevance_scores.data,
                                                       axis=1)))
    _dcg = dcg(ranking, relevance_scores, nr_docs, k, exp).data
    _idcg = dcg(optimal_ranking, relevance_scores, nr_docs, k, exp).data

    _idcg[_idcg == 0.0] = 1.0

    return as_variable(_dcg / _idcg)
