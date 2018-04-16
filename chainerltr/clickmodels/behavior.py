class UserBehavior:
    def __init__(self, maximum_relevance=1, minimum_relevance=0):
        self.maximum_relevance = maximum_relevance
        self.minimum_relevance = minimum_relevance

    def _scale_relevance(self, relevance_labels):
        translated = relevance_labels - self.minimum_relevance
        return translated / (self.maximum_relevance - self.minimum_relevance)

    def relevance_probability(self, relevance_labels):
        raise NotImplementedError

    def stop_probability(self, relevance_labels):
        raise NotImplementedError


class PerfectBehavior(UserBehavior):
    def __init__(self, maximum_relevance=1, minimum_relevance=0):
        super().__init__(maximum_relevance, minimum_relevance)

    def relevance_probability(self, relevance_labels):
        return self._scale_relevance(relevance_labels)

    def stop_probability(self, relevance_labels):
        return relevance_labels * 0.0


class NavigationalBehavior(UserBehavior):
    def __init__(self, maximum_relevance=1, minimum_relevance=0):
        super().__init__(maximum_relevance, minimum_relevance)

    def relevance_probability(self, relevance_labels):
        return self._scale_relevance(relevance_labels) * 0.9 + 0.05

    def stop_probability(self, relevance_labels):
        return self._scale_relevance(relevance_labels) * 0.7 + 0.2


class InformationalBehavior(UserBehavior):
    def __init__(self, maximum_relevance=1, minimum_relevance=0):
        super().__init__(maximum_relevance, minimum_relevance)

    def relevance_probability(self, relevance_labels):
        return self._scale_relevance(relevance_labels) * 0.5 + 0.4

    def stop_probability(self, relevance_labels):
        return self._scale_relevance(relevance_labels) * 0.4 + 0.1

