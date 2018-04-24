from chainer import cuda, as_variable

from chainerltr.clickmodels.clickmodel import ClickModel


class DependentClickModel(ClickModel):
    def _click_vector(self, relevance, nr_docs, rng):
        xp = cuda.get_array_module(relevance)

        # Get relevance and stop probabilities
        p_relevance = self.behavior.relevance_probability(relevance).data
        s_prob = 1.0 - self.behavior.stop_probability(relevance).data

        uniform_sample = rng.uniform(size=p_relevance.shape)
        e_prob = xp.zeros(p_relevance.shape)
        e_prob[:, 0] = 1.0

        c_stack = xp.zeros(p_relevance.shape)
        for i in range(p_relevance.shape[1]):
            c = (p_relevance[:, i] >= uniform_sample[:, i]) * e_prob[:, i]
            if i < p_relevance.shape[1] - 1:
                e_prob[:, i+1] = ((1 - c + s_prob[:, i] * c) >= uniform_sample[:, i]) * e_prob[:, i]
            c_stack[:, i] = c

        return as_variable(c_stack)
