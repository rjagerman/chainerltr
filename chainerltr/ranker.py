from chainer import Chain, functions as F


class Ranker(Chain):
    def __init__(self, predictor):
        super().__init__(predictor=predictor)

    def __call__(self, x):
        if x.ndim == 3:
            out = F.reshape(x, (x.shape[0] * x.shape[1], x.shape[2]))
            result = self.predictor(out)
            return F.reshape(result, (x.shape[0], x.shape[1]))
        else:
            raise TypeError("Ranker can only be applied to 3-dimensional "
                            "tensors (e.g. samples coming from a "
                            "chainerltr.dataset.RankingDataset)")
