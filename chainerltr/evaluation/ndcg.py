from chainer import cuda, FunctionNode, functions as F, as_variable


class NDCG(FunctionNode):
    def __init__(self, k=0, exp=True):
        self.k = k
        self.exp = exp

    def forward(self, inputs):
        xp = cuda.get_array_module(*inputs)
        y, t = inputs

        # Assert arrays have the same shape
        if t.shape != y.shape:
            raise ValueError("Input arrays have different shapes")

        # Computing nDCG on empty array should just return 0.0
        if t.shape[0] == 0:
            return xp.asarray(0.0),

        # Compute best_indices by sorting the relevance labels and then flipping
        predicted_indices = xp.flip(xp.argsort(y), axis=0)
        best_indices = xp.flip(xp.argsort(t), axis=0)

        # Select items based on permutations to get relevance grades sorted
        predicted_relevance = t[predicted_indices]
        best_relevance = t[best_indices]

        # Compute needed statistics
        length = predicted_relevance.shape[0]
        last = min(self.k, length)
        if last < 1:
            last = length

        # Compute regular DCG
        dcg = self._dcg(predicted_relevance, xp, last)

        # Compute iDCG for normalization
        idcg = self._dcg(best_relevance, xp, last)
        if idcg == 0.0:
            idcg = 1.0

        return xp.asarray(dcg / idcg),

    def _dcg(self, relevance, xp, last):
        """
        Computes the DCG for given set of sorted relevance scores

        :param relevance: The relevance scores
        :param xp: :mod:`cupy` or :mod:`numpy`
        :param arange: A numeric range from 1...length
        :param last: The last index to use
        :return: The dcg score
        """
        arange = 1.0 + xp.arange(relevance.shape[0])
        if self.exp:
            dcg_numerator = (2.0 ** relevance[:last]) - 1.0
        else:
            dcg_numerator = relevance[:last]
        dcg_denominator = xp.log2(arange[:last] + 1)
        return xp.sum(dcg_numerator / dcg_denominator)


def ndcg(predicted_scores, relevance_scores, nr_docs=None, k=0, exp=True):
    """
    Computes the nDCG@k for given list of true relevance labels
    (relevance_labels) and given permutation of documents (permutation)

    :param predicted_scores: The predicted scores for the document
    :param relevance_scores: The ground truth relevance labels
    :param k: The cut-off point (if set to smaller or equal to 0, it does not
              cut-off)
    :param exp: Set to true to use the exponential variant of nDCG which
                has a stronger emphasis on retrieving relevant documents
    :param nr_docs: When using 2d-arrays you need to specify a vector of the
                    nr_docs per row (assumed zero-padding)
    :return: The nDCG@k value
    """
    predicted_scores = as_variable(predicted_scores)
    relevance_scores = as_variable(relevance_scores)
    if predicted_scores.ndim == 1 and relevance_scores.ndim == 1:
        return NDCG(k=k, exp=exp).apply((predicted_scores, relevance_scores))[0]
    elif predicted_scores.ndim == 2 and relevance_scores.ndim == 2:
        if nr_docs is None:
            xp = cuda.get_array_module(predicted_scores)
            nr_docs = xp.ones(predicted_scores.shape[0], 'i') * predicted_scores.shape[1]
        else:
            nr_docs = as_variable(nr_docs).data
        ndcg_func = NDCG(k=k, exp=exp)
        p = predicted_scores.data
        r = relevance_scores.data
        res = []
        for i in range(predicted_scores.shape[0]):
            p_i = p[i, :nr_docs[i]]
            r_i = r[i, :nr_docs[i]]
            res.append(ndcg_func.apply((p_i, r_i))[0])
        return F.flatten(F.vstack(res))
    else:
        raise TypeError("ndcg can only be applied to 1 or 2-dimensional "
                        "tensors")
