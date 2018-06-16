from chainer import as_variable, cuda


class UserBehavior:
    def __init__(self, maximum_relevance=1, minimum_relevance=0):
        """
        Initializes a user behavior, which provides relevance probabilities and
        stopping probabilities, to be used by a click model.

        :param maximum_relevance: The maximum possible relevance (default: 1)
        :type maximum_relevance: int

        :param minimum_relevance: The minimum possible relevance (default: 0)
        :type minimum_relevance: int
        """
        self.maximum_relevance = maximum_relevance
        self.minimum_relevance = minimum_relevance

    def _scale_relevance(self, relevance_labels):
        """
        Scales the relevance labels to [0, 1]

        :param relevance_labels: The given relevance labels
        :type relevance_labels: chainer.Variable

        :return: The scaled relevance labels
        :rtype: chainer.Variable
        """
        xp = cuda.get_array_module(relevance_labels)
        r = as_variable(relevance_labels.data.astype(xp.float))
        translated = r - self.minimum_relevance
        return translated / (self.maximum_relevance - self.minimum_relevance)

    def relevance_probability(self, relevance_labels):
        """
        Returns an array of probabilities of relevance for given relevance
        labels

        :param relevance_labels: The relevance labels
        :type relevance_labels: chainer.Variable

        :return: Probability of relevance
        :rtype: chainer.Variable
        """
        raise NotImplementedError

    def stop_probability(self, relevance_labels):
        """
        Returns an array of probabilities of stopping for given relevance labels

        :param relevance_labels: The relevance labels
        :type relevance_labels: chainer.Variable

        :return: Probability of stopping
        :rtype: chainer.Variable
        """
        raise NotImplementedError


class PerfectBehavior(UserBehavior):
    def relevance_probability(self, relevance_labels):
        return self._scale_relevance(relevance_labels)

    def stop_probability(self, relevance_labels):
        return relevance_labels * 0.0


class NavigationalBehavior(UserBehavior):
    def relevance_probability(self, relevance_labels):
        return self._scale_relevance(relevance_labels) * 0.9 + 0.05

    def stop_probability(self, relevance_labels):
        return self._scale_relevance(relevance_labels) * 0.7 + 0.2


class InformationalBehavior(UserBehavior):
    def relevance_probability(self, relevance_labels):
        return self._scale_relevance(relevance_labels) * 0.5 + 0.4

    def stop_probability(self, relevance_labels):
        return self._scale_relevance(relevance_labels) * 0.4 + 0.1

