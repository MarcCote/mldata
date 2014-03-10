"""Datasets store the data used for experiments."""


class Dataset():
    """The abstract superclass of every types of datasets used in MLData

    A `Dataset` presents a unified access to data, independent of the
     implementation details such as laziness.

    Parameters
    ----------
    data : array_like
    meta_data : MetaData

    Attributes
    ----------
    data : array_like
        The array of data to train on.
    meta_data : Metadata
        Information about the data. See `MetaData` documentation for more info.
    """
    def __init__(self, data, meta_data):
        self.data = data
        assert isinstance(meta_data, Metadata)
        self.meta_data = meta_data

    def __iter__(self):
        raise NotImplementedError

    def __getitem__(self, item):
        raise NotImplementedError

    def get_splits(self):
        pass

    def build(self): # Replace with constructor ?
        pass

    def apply(self):
        pass


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
        The name of the `Dataset`
    nb_examples : int
        The number of example in the dataset (including all splits).
    dictionary : dict
        Gives a mapping of words (str) to id (int). Used only when the
        dataset has been saved as an array of numbers instead of text.
    splits : tuple of int
        Specifies the split used by this view of the dataset.
    preprocess : function or None
        A function that is callable on a `Dataset` to preprocess the data.
    """
    def __init__(self):
        self.name = "Default"
        self.nb_examples = 0
        self.dictionary = None
        self.splits = ()
        self.preprocess = None


class InMemoryDataset(Dataset):
    """Build a dataset entirely contained in memory.

    Load the data (an array-like object) in memory. Random access is then
    insured to be fast.

    Parameters
    ----------
    examples : array_like
        The dataset.
    meta_data : Metadata
        The metadata of this dataset.
    targets : ?

    See Also
    --------
    Dataset : The parent class defining the interface of a dataset.

    """
    def __init__(self, examples, meta_data, targets=None):
        super(InMemoryDataset, self).__init__(meta_data)

        self.data = examples

        if targets is None:
            self.__iter__ = self._iter_without_target
        else:
            self.targets = targets
            self.__iter__ = self._iter_with_target


    def __getitem__(self, item):
        pass

    def _iter_with_target(self):
        pass

    def _iter_without_target(self):
        pass