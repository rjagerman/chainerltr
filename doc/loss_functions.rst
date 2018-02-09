.. _loss_functions-ref:

==============
Loss Functions
==============

ChainerLTR currently provides four ranking loss functions:

* ListNet (Top-1 approximation)
* ListMLE
* ListPL
* RankNet

Usage
-----
The loss functions can easily be plugged into a chainer architecture. We can
define a Chain class that uses the ListNet loss, assuming we have a network
architecture called :code:`predictor`, as follows:

.. code-block:: python

    from chainerltr import Ranker
    from chainerltr.loss.listwise import listnet
    from chainer import Chain

    class MyModel(Chain)
        def __init__(self, ranker):
            super().__init__(ranker=ranker)

        def __call__(xs, ys, nr_docs):
            return listnet(self.ranker(xs), ys, nr_docs)

    loss = MyModel(ranker=Ranker(predictor))


ListNet
-------
The ListNet :cite:`cao2007learning` loss function is the cross-entropy between
the probability of the labels given a permutation and the probability of the
network scores given a permutation:

.. math::

   \mathcal{L}(f(x), y) = - \sum_{\pi \in \Omega} P(\pi \mid y) \log P(\pi \mid f(x))

Where :math:`P(\pi \mid z)` is a Plackett-Luce probability model of permutation
:math:`\pi` given item-specific set of scores :math:`z`. Since the set
:math:`\Omega` has size :math:`\mathcal{O}(n!)`, it is too large to compute in
practice. Instead, we use the top-1 approximation as in the original paper,
which reduces to a softmax of the scores followed by a cross entropy.

This loss function is implemented in :code:`chainerltr.loss.listwise.listnet`.

ListMLE
-------
The ListMLE :cite:`xia2008listwise` loss function is the negative probability of
a single permutation :math:`\pi \in \{\pi \mid y_{\pi_i} \geq y_{\pi_j}; i < j\}`
of the ground truth labeling. It is defined as follows:

.. math::

   \mathcal{L}(f(x), y) = - \log P(\pi \mid f(x))

This loss function is implemented in :code:`chainerltr.loss.listwise.listmle`.

ListPL
------

The ListPL :cite:`jagerman2017modeling` loss function is an approximation to the
cross-entropy loss of ListNet. It can be seen as a stochastic variant of ListMLE
where during every update a new permutation :math:`\pi` is drawn:

.. math::

   \mathcal{L}(f(x), y) = - \log P(\pi \mid f(x)) \\
   \pi \sim P(\pi \mid y)

This loss function is implemented in :code:`chainerltr.loss.listwise.listpl`.

RankNet
-------

The RankNet :cite:`burges2005learning` loss function is a pairwise loss function
that minimizes the number of inversions between a produced ranking and the
optimal ranking:

.. math::

   \mathcal{L}(f(x), y) = \sum_{i,j} \text{sigmoid}(\bar{o}_{ij}) o_{ij} + \log(1 + e^{o_{ij}}) \\
    o_{ij} = f(x_i) - f(x_j)\\
    \bar{o}_{ij} = y_i - y_j

This loss function is implemented in :code:`chainerltr.loss.pairwise.ranknet`.

.. rubric:: References

.. bibliography:: references.bib
   :enumtype: arabic
