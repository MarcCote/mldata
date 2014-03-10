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

