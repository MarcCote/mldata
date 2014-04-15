"""Datasets store the data used for experiments."""
import hashlib

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
        self.data = data
        self.target = target
        assert isinstance(meta_data, Metadata)
        self.meta_data = meta_data

    def __len__(self):
        return self.meta_data.nb_examples

    def __hash__(self):
        """ Hash function used for versioning."""
        hasher = hashlib.md5()
        for l in self.data:
            hasher.update(l.copy())
        if self.target is not None:
            for l in self.target:
                hasher.update(l.copy())
        return hasher.hexdigest()[:8]

    def __iter__(self):
        """Provide an iterator when the Dataset has a target.

        ..todo: retest efficiency of this buffering in python3. With zip
                being now lazy, it might not be better than the vanilla iter.

        """
        buffer = min(BUFFER_SIZE, len(self.data))
        if self.target is not None:
            for idx in range(0, len(self.data), buffer):
                for ex, tg in zip(self.data[idx:idx+buffer],
                                  self.target[idx:idx+buffer]):
                    yield (ex,tg)
        else:
            for idx in range(0, len(self.data), buffer):
                for ex in self.data[idx:idx+buffer]:
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

    def get_splits(self):
        """Return the splits defined by the associated metadata.

        The split is given via a tuple of integer with each integers
        representing the integer after the last id used by this split. For
        example::

            (5000, 6000, 7000)

        would give a test set of all examples from 0 to 4999, a validation
        set of examples 5000 to 5999 and a test set of examples 6000 up to
        6999. This means that 7000 is also the number of examples in the
        dataset.

        Returns
        -------
        tuple of int
            Where each integer gives the id of the example coming after the
            last one in a split.

        Notes
        -----
        For now, only a tuple is accepted. Eventually, predicates over the
        examples id could be supported.

        """
        if isinstance(self.meta_data.splits, tuple):
            return self.meta_data.splits
        else:
            raise NotImplementedError("Only splits with tuple are supported.")

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
        Gives a mapping of words (str) to id (int). Used only when the
        dataset has been saved as an array of numbers instead of text.
        Default: None
    splits : tuple of int
        Specifies the split used by this view of the dataset. Default: ().
    preprocess : function or None
        A function that is callable on a `Dataset` to preprocess the data.
        Default: ``lambda x: x``.
    hash : str
        The hash of the linked ``Dataset``. Default: "".

    """
    def __init__(self):
        self.name = "Default"
        self.nb_examples = 0
        self.dictionary = None
        self.splits = ()
        self.preprocess = lambda x: x
        self.hash = ""


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
