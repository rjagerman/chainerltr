.. _getting_started-ref:

=====================
Getting Started Guide
=====================

In this guide we will go through a simple example from start to finish. You
will learn how to load data, create an iterator, setup a neural network and
learn its parameters using a ranking loss function.

Loading a data set
==================
We will load a very simple example data set in RankSVM format.

.. code-block:: python

    from chainerltr.dataset import load

    with open('./dataset.txt', 'rb') as file:
        dataset = load(file)

Chainer works with a concept of iterators that feed a neural network with
batches of data. We need to load our data into a Learning to Rank iterator to
start using it, here we use a minibatch size of 4:

.. code-block:: python

    from chainer.iterators import SerialIterator

    iterator = SerialIterator(dataset, 4, repeat=True, shuffle=True)

Setting up a network
====================
For our simple example we will set up a single-layer linear neural network. This
is equivalent to a linear function of the input features.

.. code-block:: python

    from chainer import links
    predictor = links.Linear(None, 1)

We encourage the reader to experiment with a wide variety of neural
architectures. For more information about designing different architectures we
refer you to the `documentation of Chainer <https://docs.chainer.org>`_.

Choosing a loss function
========================
ChainerLTR currently provides 3 different list-wise loss functions and 1
pair-wise loss function. For this guide we will use the ListNet loss (top-1
approximation). We first need to wrap our predictor function in a Ranker object
that correctly accounts for the shape of learning-to-rank data:

.. code-block:: python

    from chainerltr import Ranker
    ranker = Ranker(predictor=predictor)


Next, we will construct our model as a Chain which returns the computed loss:

.. code-block:: python

    from chainer import Chain
    from chainerltr.loss.listwise import listnet

    class MyModel(Chain):
        def __init__(self, ranker):
            super().__init__(ranker=ranker)

        def __call__(self, xs, ys, nr_docs):
            loss = listnet(self.ranker(xs), ys, nr_docs)
            return loss

    loss = MyModel(ranker)


Training
========
We now have all the pieces set up to start training our network. What follows is
standard Chainer code for setting up an optimizer, updater and trainer. There
are many options and choices to be made here, but they fall outside the scope of
this guide. You will be able to find much more information about optimizing the
network on the `documentation of Chainer <https://docs.chainer.org>`_.

.. code-block:: python

    from chainer import training, optimizers
    from chainer.training import extensions
    from chainerltr.dataset import zeropad_concat

    # Build optimizer, updater and trainer
    optimizer = optimizers.Adam()
    optimizer.setup(loss)
    updater = training.StandardUpdater(iterator, optimizer, converter=zeropad_concat)
    trainer = training.Trainer(updater, (40, 'epoch'))
    trainer.extend(extensions.ProgressBar())

    # Train neural network
    trainer.run()

