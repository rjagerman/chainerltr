from chainer import cuda, as_variable

from chainerltr.clickmodels.clickmodel import ClickModel


class DependentClickModel(ClickModel):
    """
    A dependent click model with an efficient GPU-accelerated implementation.
    """
    def _click_vector(self, relevance, nr_docs, rng):
        xp = cuda.get_array_module(relevance)

        # Get relevance and stop probabilities
        r_prob = self.behavior.relevance_probability(relevance).data
        s_prob = 1.0 - self.behavior.stop_probability(relevance).data

        # Generate random samples
        uniform = rng.uniform(size=r_prob.shape)
        uniform2 = rng.uniform(size=r_prob.shape)

        # Generate independent clicks
        clicks = 1.0 * (r_prob >= uniform)

        # Compute evaluation probabilities based on clicks and stopping
        # probabilities
        e_prob = xp.roll(clicks * s_prob + (1.0 - clicks), 1, axis=1)
        e_prob[:, 0] = 1.0

        # Realize evaluations and then do a cumulative product to stop at the
        # first non-evaluation
        evaluations = xp.cumprod(1.0 * (e_prob >= uniform2), axis=1)

        # Final clicks are only the clicks that were also evaluated
        return as_variable(evaluations * clicks)
