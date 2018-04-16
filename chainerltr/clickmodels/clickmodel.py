from chainer import Chain, cuda

from chainerltr.functions import unpad, permutate2d


class ClickModel(Chain):
    def __init__(self, behavior):
        """
        Initializes the click model

        :param behavior: The user behavior
        :type behavior: chainerltr.clickmodels.behavior.UserBehavior
        """
        super().__init__()
        self.behavior = behavior

    def __call__(self, ranking, relevance, nr_docs, seed=None):
        """
        Generates a click vector based on given action (ranking), truth labels
        (relevance scores) and number of documents

        :param ranking: The mini-batch of rankings
        :type ranking: chainer.Variable

        :param relevance: The mini-batch of relevance labels
        :type truth: chainer.Variable

        :param nr_docs: The nr docs per element in the minibatch
        :type nr_docs: chainer.Variable

        :param seed: The seed for random number generation
        :type seed: int|None

        :return: The click vector
        :rtype: chainer.Variable
        """

        # Reshuffle the rankings based on nr_docs
        t_action = unpad(ranking, nr_docs)
        t_relevance = permutate2d(relevance, t_action)

        xp = cuda.get_array_module(relevance)
        rng = xp.random
        if seed is not None:
            rng = xp.random.RandomState(seed)
        return self._click_vector(t_relevance, nr_docs, rng)

    def _click_vector(self, relevance, nr_docs, rng):
        """
        Generates a vector of clicks (1.0 or 0.0) based on given ranking of
        relevance probabilities.

        :param relevance: The relevance scores of the sorted array
        :type relevance: chainer.Variable

        :param nr_docs: The number of documents for each row
        :type nr_docs: chainer.Variable

        :param rng: Random number generator module

        :return: The click vector
        :rtype: chainer.Variable
        """
        raise NotImplementedError
