import numpy as _np
from chainer.dataset import DatasetMixin as _DatasetMixin, concat_examples
from sklearn.externals.joblib import Memory as _Memory
from sklearn.datasets import load_svmlight_file as _load_svmlight_file


class RankingDataset(_DatasetMixin):
    """
    Chainer version of a ranking dataset
    """
    def __init__(self, feature_vectors, relevance_labels, qids, nr_samples=None,
                 filter=False, normalize=False):
        """
        :param feature_vectors: The numpy 3d array of samples (query, doc,
                                feature)
        :type feature_vectors: scipy.sparse.lil.lil_matrix

        :param relevance_labels: The numpy array relevance labels
        :type relevance_labels: numpy.ndarray

        :param qids: The query identifiers
        :type qids: numpy.ndarray

        :param nr_samples: The number of samples (if not provided this is
                           inferred from x)
        :type nr_samples: int

        :param filter: Whether to filter out queries with no relevant documents
        :type filter: bool

        :param normalize: Whether to perform query-level normalization of
                          features
        :type normalize: bool
        """
        self.feature_vectors = feature_vectors.astype(_np.float32)
        self.relevance_labels = relevance_labels.astype(_np.int32)
        self.maximum_relevance = _np.max(self.relevance_labels)
        self.minimum_relevance = _np.min(self.relevance_labels)
        self.qids = qids.astype(_np.int32)
        self.unique_qids = _np.unique(qids)
        self.nr_dimensions = self.feature_vectors.shape[1]

        # Perform filtering if necessary
        if filter is True:
            new_unique_qids = []
            for i in range(len(self.unique_qids)):
                ys = self.relevance_labels[self.qids == self.unique_qids[i]]
                if _np.sum(ys) > 0.0:
                    new_unique_qids.append(self.unique_qids[i])
            self.unique_qids = _np.array(new_unique_qids)

        self.nr_samples = self.unique_qids.shape[0] if nr_samples is None else nr_samples

        # Perform normalization if necessary
        if normalize is True:
            for i in range(self.nr_samples):
                mask = self.qids == self.unique_qids[i]
                minimum = self.feature_vectors[mask, :].min(axis=0)
                self.feature_vectors[mask, :] -= minimum
                maximum = _np.max(self.feature_vectors[mask, :], axis=0)
                maximum[maximum == 0.0] = 1.0
                self.feature_vectors[mask, :] /= maximum
            self.feature_vectors = _np.nan_to_num(self.feature_vectors)

        # Pre-compute start and end indices for each query
        self.starts = _np.zeros(len(self.unique_qids), dtype=_np.int32)
        self.ends = _np.zeros(len(self.unique_qids), dtype=_np.int32)
        self.nr_docs = _np.zeros(len(self.unique_qids), dtype=_np.int32)
        for i in range(len(self.unique_qids)):
            where = _np.argwhere(self.qids == self.unique_qids[i])
            self.starts[i] = _np.min(where)
            self.ends[i] = _np.max(where) + 1
            self.nr_docs[i] = where.shape[0]

    def __len__(self):
        return self.nr_samples

    def get_example(self, i):
        s = self.starts[i]
        e = self.ends[i]
        xs = self.feature_vectors[s:e]
        ys = self.relevance_labels[s:e]
        nr_docs = self.nr_docs[i]
        return xs, ys, nr_docs


def zeropad_concat(batch, device=None):
    """
    Performs a padded concatentation, which is required for the variable sized
    learning-to-rank vectors

    :param batch: The current batch
    :type batch: list

    :param device: The device to use (e.g. GPU)
    :type device: int|None

    :return: The concatenated examples
    """
    return concat_examples(batch, device, 0.0)


def load(file_path, cache_path=None, normalize=False, filter=False, verbose=0,
         cache_compress=6):
    """
    Loads from an SVMrank format into a dense data set

    :param file_path: The file path
    :type file_path: str|int|file-like

    :param cache_path: The cache path, by default it does not cache
    :type cache_path: str|None

    :param normalize: Whether to perform query-level normalization of features
    :type normalize: bool

    :param filter: Whether to filter results
    :type filter: bool

    :param verbose: Whether to print extra information during the call
    :type verbose: int

    :param cache_compress: Integer between 1-9 indicating level of compression
                           or None if no compression should be used (default: 6)
    :type cache_compress: int|None

    :return: The loaded data set
    :rtype: chainerltr.dataset.RankingDataset
    """
    def _load(file_path, filter, normalize):
        x, y, qids = _load_svmlight_file(file_path, query_id=True)
        x = x.todense().A
        return RankingDataset(x, y, qids, filter=filter, normalize=normalize)

    mem = _Memory(cache_path, compress=cache_compress)
    @mem.cache(verbose=verbose)
    def _cached_load(file_path, filter, normalize):
        return _load(file_path, filter, normalize)
    return _cached_load(file_path, filter, normalize)
