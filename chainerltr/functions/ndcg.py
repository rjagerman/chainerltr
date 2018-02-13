from chainer import cuda, FunctionNode


class NDCG(FunctionNode):
    def __init__(self, k=0):
        self.k = k

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
        predicted_indices = y
        best_indices = xp.flip(xp.argsort(t), axis=0)

        # Select items based on permutations to get relevance grades sorted
        predicted_relevance = t[predicted_indices]
        best_relevance = t[best_indices]

        # Compute needed statistics
        length = predicted_relevance.shape[0]
        arange = xp.arange(length) * xp.ones(predicted_relevance.shape)
        last = min(self.k, length)
        if last < 1:
            last = length

        # Compute regular DCG
        dcg_numerator = 2. ** predicted_relevance[:last] - 1
        dcg_denominator = xp.log2(arange[:last] + 2)
        dcg = xp.sum(dcg_numerator / dcg_denominator)

        # Compute iDCG for normalization
        idcg_numerator = (2. ** best_relevance[:last] - 1)
        idcg_denominator = (xp.log2(arange[:last] + 2))
        idcg = xp.sum(idcg_numerator / idcg_denominator, axis=0)
        if idcg == 0.0:
            idcg = 1.0

        return xp.asarray(dcg / idcg),


def ndcg(permutation, relevance_labels, k=0):
    """
    Computes the nDCG@k for given list of true relevance labels
    (relevance_labels) and given permutation of documents (permutation)

    :param permutation: The predicted permutation or ranking
    :param relevance_labels: The ground truth relevance labels
    :param k: The cut-off point (if set to smaller or equal to 0, it does not
              cut-off)
    :return: The nDCG@k value
    """
    if permutation.ndim == 1 and relevance_labels.ndim == 1:
        return NDCG(k=k).apply((permutation, relevance_labels))[0]
    else:
        raise TypeError("ndcg can only be applied to 1-dimensional tensors")
