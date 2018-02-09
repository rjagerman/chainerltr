.. _datasets-ref:

========
Datasets
========

Datasets in ChainerLTR are loaded using sklearn's :code:`load_svmlight_file`
function and then processed further to enable easy integration with Chainer.

Dataset Loading
===============
The most typical format to load a Learning to Rank data set is using RankSVM
format. Many publicly available anotated data sets have used this format. Given
a Learning to Rank data set called `dataset.txt`, you can load it as follows:

.. code-block:: python

    from chainerltr.dataset import load

    with open('./dataset.txt', 'rb') as file:
        dataset = load(file)

Normalization and filtering
===========================

Some data sets in SVMlight format do not come with query-level normalization.
For most loss functions that depend on exponentials it is very recommended to do
query-level normalization to prevent overflow errors. Fortunately, ChainerLTR
has normalization built-in as part of the data set loading facilities. You can
load a data set with query-level normalization by setting the `normalize`
parameter to `True`:

.. code-block:: python

    with open('./dataset.txt', 'rb') as file:
        dataset = load(file, normalize=True)

Furthermore, you can filter out queries where all documents have the same
relevance grade, which prevents learning by using the `filter` parameter:

.. code-block:: python

    with open('./dataset.txt', 'rb') as file:
        dataset = load(file, filter=True)

