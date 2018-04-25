from chainer import cuda, as_variable

from chainerltr.clickmodels.clickmodel import ClickModel


class DependentClickModel(ClickModel):
    """
    A dependent click model. Note that this is a variation where p(stop) < 1.0,
    because clicks are generated independently through GPU accelerated
    operations. P(stop) = 1.0 would violate the model as clicks may be generated
    (although very unlikely) at later positions.
    """
    def _click_vector(self, relevance, nr_docs, rng):
        xp = cuda.get_array_module(relevance)

        # Get relevance and stop probabilities
        r_prob = self.behavior.relevance_probability(relevance).data
        s_prob = 1.0 - self.behavior.stop_probability(relevance).data

        uniform = rng.uniform(size=r_prob.shape)
        e_prob = xp.zeros(r_prob.shape)
        e_prob[:, 0] = 1.0

        c_stack = xp.zeros(r_prob.shape)
        for i in range(r_prob.shape[1]):
            c = (r_prob[:, i] >= uniform[:, i]) * e_prob[:, i]
            if i < r_prob.shape[1] - 1:
                e_prob[:, i+1] = ((1 - c + s_prob[:, i] * c) >= uniform[:, i]) * e_prob[:, i]
            c_stack[:, i] = c

        return as_variable(c_stack)
