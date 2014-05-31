"""Datasets store the data used for experiments."""
from itertools import accumulate
import hashlib

import numpy as np

BUFFER_SIZE = 1000


class Dataset():
    """Interface to interact with physical dataset

    A `Dataset` presents a unified access to data, independent of the
     implementation details such as laziness.

    Parameters
    ----------
    meta_data : Metadata
    data : array_like
    target : array_like

    Attributes
    ----------
    meta_data : Metadata
        Information about the data. See `MetaData` documentation for more info.
    data : array_like
        The array of data to train on.
    target : array_like, optional
        The array of target to use for supervised learning. `target` should
        be `None` when the dataset doesn't support supervised learning.

    """
    def __init__(self, meta_data, data, target=None):
        assert len(data) == meta_data.nb_examples,\
            "The metadata ``nb_examples`` is inconsistent with the length of "\
            "the dataset."
        assert len(data) == meta_data.splits[-1] or\
            len(data) == sum(meta_data.splits),\
            "The metadata ``splits`` is inconsistent with the length of "\
            "the dataset."
        self.data = data
        self.target = target
        self.meta_data = meta_data

    def __len__(self):
        return self.meta_data.nb_examples

    def __hash__(self):
        """ Hash function used for versioning."""
        hasher = hashlib.md5()
        for l in self.data:
            hasher.update(np.array(l))
        if self.target is not None:
            for l in self.target:
                hasher.update(np.array(l))
        return hasher.hexdigest()[:8]

    def __iter__(self):
        """Provide an iterator handling if the Dataset has a target."""
        #todo: retest efficiency of this buffering in python3. With zip being now lazy, it might not be better than the vanilla iter.
        buffer = min(BUFFER_SIZE, len(self))

        # Cycle infinitely
        while True:
            if self.target is not None:
                for idx in range(0, len(self.data), buffer):
                    stop = min(idx + buffer, len(self))
                    for ex, tg in zip(self.data[idx:stop],
                                      self.target[idx:stop]):
                        yield (ex, tg)
            else:
                for idx in range(0, len(self.data), buffer):
                    stop = min(idx + buffer, len(self))
                    for ex in self.data[idx:stop]:
                        yield (ex,)

    def __getitem__(self, key):
        """Get the entry specified by the key.

        Parameters
        ----------
        key : numpy-like key
            The `key` can be a single integer, a slice or a tuple defining
            coordinates. Can be treated as a NumPy key.

        Returns
        -------
        (array_like, array_like) or (array_like,)
            Return the element specified by the key. It can be an array or
            simply a scalar of the type defined by the data [and target
            arrays].
            The returned values are put in a tuple (data, target) or (data,).

        """
        if self.target is not None:
            return (self.data[key], self.target[key])
        else:
            return (self.data[key],)

    def _split_iterators(self, start, end, minibatch_size=1):
        """ Iterate on a split.

        Parameters
        ----------
        start : int
            Id of the first element of the split.
        end : int
            Id of the next element after the last.

        """
        buffer = min(BUFFER_SIZE, end - start)

        # Cycle infinitely
        while True:
            if self.target is not None:
                for idx in range(start, end, buffer):
                    stop = min(idx+buffer, end)
                    for i in range(idx, stop, minibatch_size):
                        j = min(stop, i+minibatch_size)
                        yield (self.data[i:j], self.target[i:j].reshape((1, -1)))
            else:
                for idx in range(start, end, buffer):
                    stop = min(idx+buffer, end)
                    for i in range(idx, stop, minibatch_size):
                        j = min(stop, i+minibatch_size)
                        yield (self.data[i:j],)

    def get_splits_iterators(self, minibatch_size=1):
        """ Creates a tuple of iterator, each iterating on a split.

        Each iterators returned is used to iterate over the corresponding
        split. For example, if the ``Metadata`` specifies a ``splits`` of
        (10, 20, 30), ``get_splits_iterators`` returns a 3-tuple with an
        iterator for the ten first examples, another for the ten next and a
        third for the ten lasts.

        Parameters
        ----------
        minibatch_size : int
            The size of minibatches received each iteration.

        Returns
        -------
        tuple of iterable
            A tuple of iterator, one for each split.

        """
        sp = list(self.meta_data.splits)

        # normalize the splits
        if sum(sp) == len(self):
            sp = list(accumulate(sp))
        assert sp[-1] == len(self), "The splits couldn't be normalized"

        itors = []
        for start, end in zip([0] + sp, sp):
            itors.append(self._split_iterators(start, end, minibatch_size))
        return itors

    def apply(self):
        """Apply the preprocess specified in the associated metadata.

        This methods simply apply the function given in the metadata (the
        identity by default) to the dataset. This function is supposed to do
        work on the data and the targets, leaving the rest intact. Still,
        as long as the result is still a `Dataset`, `apply` will work.

        Returns
        -------
        Dataset
            The preprocessed dataset.

        """
        ds = self.meta_data.preprocess(self)
        assert isinstance(ds, Dataset)
        return ds


class Metadata():
    """Keep track of information about a dataset.

    An instance of this class is required to build a `Dataset`. It gives
    information on how the dataset is called, the split, etc.

    A single `Dataset` can have multiple metadata files specifying different
    split or a special pre-processing that needs to be applied. The
    philosophy is to have a single physical copy of the dataset with
    different views that can be created on the fly as needed.

    Attributes
    ----------
    name : str
        The name of the `Dataset`. Default: "Default".
    nb_examples : int
        The number of example in the dataset (including all splits). Default: 0.
    dictionary : Dictionary
        _Not yet implemented_
        Gives a mapping of words (str) to id (int). Used only when the
        dataset has been saved as an array of numbers instead of text.
        Default: None
    splits : tuple of int
        Specifies the split used by this view of the dataset. Default: ().
        The numbers can be either the number of the last examples in each
        subsets or the number of examples in each categories.
    preprocess : function or None
        A function that is callable on a `Dataset` to preprocess the data.
        The function cannot be a lambda function since those can't be pickled.
        Default: identity function.
    hash : str
        The hash of the linked ``Dataset``. Default: "".

    """
    def __init__(self):
        self.name = "Default"
        self.nb_examples = 0
        self.dictionary = None
        self.splits = ()
        self.preprocess = default_preprocess
        self.hash = ""

def default_preprocess(dset):
    return dset

class Dictionary:
    """Word / integer association list

    This dictionary is used in `Metadata` for NLP problems. This class
    ensures O(1) conversion from id to word and O(log n) conversion from word to
    id.

    Notes
    -----
    The class is *not yet implemented*.

    Plans are for the dictionary to be implemented as a list of words
    alphabetically ordered with the index of the word being its id. A method
    implements a binary search over the words in order to retrieve its id.

    """

    def __init__(self):
        raise NotImplementedError("The class Dictionary is not yet "
                                  "implemented.")
