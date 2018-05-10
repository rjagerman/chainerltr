from chainer import cuda, FunctionNode, as_variable
from chainerltr.functions import select_items_per_row, unpad


class DCG(FunctionNode):
    def __init__(self, k=0, exp=True):
        self.k = k
        self.exp = exp

    def forward(self, inputs):
        xp = cuda.get_array_module(*inputs)
        ranking, relevance_labels = inputs

        # Computing nDCG on empty array should just return 0.0
        if ranking.shape[1] == 0:
            return xp.zeros(ranking.shape[0]),

        # Top-k cutoff
        last = ranking.shape[1]
        if self.k > 0:
            last = min(self.k, last)

        # For the rankings, compute the relevance labels in order
        relevance = select_items_per_row(as_variable(relevance_labels),
                                         as_variable(ranking))
        relevance = relevance[:, :last].data.astype(dtype=xp.float32)

        # Compute numerator of DCG formula
        if self.exp:
            numerator = (2.0 ** relevance) - 1.0
        else:
            numerator = relevance

        # Compute denominator of DCG formula
        arange = xp.broadcast_to(2.0 + xp.arange(relevance.shape[1]),
                                 relevance.shape)
        denominator = xp.log2(arange)

        if self.k >= 0:
            return xp.asarray(xp.sum(numerator / denominator, axis=1)),
        else:
            return xp.asarray(xp.cumsum(numerator / denominator, axis=1)),


def dcg(ranking, relevance_scores, nr_docs=None, k=0, exp=True):
    """
    Computes the DCG@k for given list of true relevance labels
    (relevance_labels) and given permutation of documents (permutation)

    :param predicted_scores: The predicted scores for the document
    :type predicted_scores: chainer.Variable

    :param relevance_scores: The ground truth relevance labels
    :type relevance_scores: chainer.Variable

    :param k: The cut-off point (if set to smaller or equal to 0, it does not
              cut-off)
    :type k: int

    :param exp: Set to true to use the exponential variant of nDCG which
                has a stronger emphasis on retrieving relevant documents
    :type exp: bool

    :param nr_docs: A vector of the nr_docs per row
    :type nr_docs: chainer.Variable

    :return: The DCG@k value
    :rtype: chainer.Variable
    """
    # Assert arrays have the same shape
    if ranking.shape != relevance_scores.shape:
        raise ValueError("Input arrays have different shapes")

    if nr_docs is not None:
        if nr_docs.shape[0] != ranking.shape[0]:
            raise ValueError(f"Nr docs has an incorrect length (expected "
                             f"{ranking.shape[0]}, but got {nr_docs.shape[0]}")
        ranking = unpad(ranking, nr_docs)

    return DCG(k=k, exp=exp).apply((ranking, relevance_scores))[0]
