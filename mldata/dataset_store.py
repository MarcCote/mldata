""" Manages dataset read/write operations."""
#todo: Remove precise versions of datasets and manage dependencies.
import os
import pickle as pk

import h5py
import numpy as np

from SMARTdata.mldata.utils import config as cfg
from SMARTdata.mldata.dataset import Dataset, Metadata


def load(dset_name, version_name="baseDataset", lazy=False):
    """ Load a dataset given its name.

    The load function will load the ``Dataset`` ``name`` provided it exists in
    one of the datasets folders. This function allows reading of files which
    are bigger than available memory using ``h5py``.

    Parameters
    ----------
    name : str
        The name of the dataset to load. The first match from the list of
        dataset folder will be used, thus allowing private copy of a dataset.
    version_name : str
        If this is a special version of a dataset, use this name to indicate
        it. Default: "baseDataset".
    lazy : bool
        If set to ``True``, the dataset will be read with ``h5py`` without
        loading the whole dataset in memory. If set to ``False``, the file is
        mapped in memory. Default: False.

    Returns
    -------
    Dataset
        Return the loaded dataset, if it exists. Else, return ``None``.

    Raises
    ------
    LookupError
        If the dataset ``dset_name`` does not exist, a ``LookupError`` is
        raised.

    """
    path = None
    if cfg.dataset_exists(dset_name):
        path = cfg.get_dataset_path(dset_name)
    else:
        raise LookupError("This dataset does not exist.")
    return _load_from_file(dset_name + '_' + version_name, path, lazy)


def save(dataset, version_name="baseDataset"):
    """ Save the dataset, manages versions.

    A ``Dataset`` is saved according to its name and the ``version_name``
    provided. The ``version_name`` is used to denote different view of the
    data, either using the ``preprocess`` field of a ``Metadata`` class or by
    saving a new version of the dataset (with a different hash). The first
    method is the most compact while the second method is more efficient when
    loading a dataset.

    To save a dataset using the preprocessing method, the dataset *must not*
    contain the preprocessed data, but the original dataset on which the
    preprocess is applied.

    The dataset is split between two files :

    - the data file ``[hash].data``
    - the metadata file [dataset Name]_[dataset version].meta

    This function will replace a metadata file with the same version name
    without prompting.

    Parameters
    ----------
    dataset : Dataset
        The dataset to be saved.
    version_name : str
        If this is a special version of a dataset, use this name to indicate
        it. Default: "baseDataset".

    """
    dset_name = dataset.meta_data.name

    if not cfg.dataset_exists(dset_name):
        cfg.add_dataset(dset_name)

    dset_path = cfg.get_dataset_path(dset_name)

    dset_hash = dataset.__hash__()
    dataset.meta_data.hash = dset_hash # insures metadata hash is up to date

    dset_file = dataset.meta_data.hash + ".data"
    _save_dataset(dataset, dset_path, dset_file)

    meta_file = dset_name + '_' + version_name + ".meta"
    _save_metadata(dataset.meta_data, dset_path, meta_file)


def _load_from_file(name, path, lazy):
    """ Call to ``h5py`` to load the file.

    """
    metadata = None
    try:
        with open(os.path.join(path, name) + '.meta', 'rb') as f:
            metadata = pk.load(f)
    except FileNotFoundError:
        raise LookupError("This dataset/version pair does not exist : " + name)

    datasetFile = None
    file_to_load = os.path.join(path, metadata.hash + ".data")
    if lazy:
        datasetFile = h5py.File(file_to_load, mode='r', driver=None)
    else:
        datasetFile = h5py.File(file_to_load, mode='r', driver='core')

    data = datasetFile['/']["data"]
    target = None
    try:
        target = datasetFile['/']["targets"]
    except:
        pass
    dset = Dataset(metadata, data, target)
    dset._fileHandle = h5pyFileWrapper(datasetFile)
    return dset


def _save_dataset(dataset, path, filename):
    """Call to ``h5py`` to write the dataset

    Save the dataset and the associated metadata into their respective folder in
    the dataset folder.

    Parameters
    ----------
    dataset : Dataset
    path : str
    filename : str

    """
    if filename not in os.listdir(path):
        fullname = os.path.join(path, filename)
        with h5py.File(fullname, mode='w') as f:
            f.create_dataset('data', data=dataset.data)
            if dataset.target is not None:
                f.create_dataset('targets', data=dataset.target)


def _save_metadata(metadata, path, filename):
    """ Pickle the metadata.

    Parameters
    ----------
    metadata : Metadata
    path : str
    filename : str

    """
    #todo: A dataset could be orphaned if overwritten by another metadata file. This needs to be checked in a future version.
    if filename not in os.listdir(path):
        with open(os.path.join(path, filename), 'wb') as f:
            pk.dump(metadata, f, pk.HIGHEST_PROTOCOL)


def CSV_importer(filepath,
                 name,
                 splits,
                 target_column=None,
                 dtype=np.float64,
                 comments='#',
                 delimiter=' ',
                 converters=None,
                 skiprows=0,
                 usecols=None):
    """ Import a CSV file into a ``Dataset``.

    From the ``filepath`` of a CSV file (using commas), create a ``Dataset``
    which can then be saved on disk. This importer supports only numbered
    inputs (int, float, boolean values).

    Parameters
    ----------
    filepath : str
        The path of the CSV file to be imported.
    name : str
        The name of this dataset used to store the ``Dataset`` on disk.
    splits : tuple of int
        Gives the split of the dataset, like (train, valid, test). The
        integers required is the id of the last example of a sub-dataset plus 1.
        For example, if there is 8000 examples with 5000 in the training set,
        2000 in the validation set and 1000 in the test set, the splits would be
        ``(5000, 7000, 8000)``.
        An alternative form where each numbers represent the count of each
        subsets is also supported.
    target_column : int, optional
        The column number of the target. If no target is provided, set to
        ``None``. Default: None.
    dtype : data-type, optional
        Data-type of the resulting array; default: float. If this is a record
        data-type, the resulting array will be 1-dimensional, and each row will
        be interpreted as an element of the array. In this case, the number of
        columns used must match the number of fields in the data-type.
    comments : str, optional
        The character used to indicate the start of a comment; default: ‘#’.
    delimiter : str, optional
        The string used to separate values. By default, this is any whitespace.
    converters : dict, optional
        A dictionary mapping column number to a function that will convert that
        column to a float. E.g., if column 0 is a date string:
        ``converters = {0: datestr2num}``. Converters can also be used to
        provide a default value for missing data :
        ``converters = {3: lambda s: float(s.strip() or 0)}``. Default: None.
    skiprows : int, optional
        Skip the first skiprows lines; default: 0.
    usecols : sequence, optional
        Which columns to read, with 0 being the first. For example,
        ``usecols = (1,4,5)`` will extract the 2nd, 5th and 6th columns. The
        default, None, results in all columns being read.

    Returns
    -------
    Dataset
        A ``Dataset`` with default values for ``Metadata``.

    """
    data = np.loadtxt(filepath, dtype, comments, delimiter,
                      converters, skiprows, usecols)

    meta = Metadata()
    meta.name = name
    meta.splits = splits
    assert len(data) == splits[-1] or \
           len(data) == sum(splits),\
           "The dataset read is not consistent with the split given."
    meta.nb_examples = len(data)

    dset = None
    if target_column is not None:
        targets = data[:, target_column].reshape((-1, 1))
        examples = data[:, list(range(0, target_column)) +
                           list(range(target_column+1, data.shape[1]))]
        dset = Dataset(meta, examples, targets)
    else:
        dset = Dataset(meta, data)

    dset.meta_data.hash = dset.__hash__()

    return dset


def remove(name):
    """ Remove a dataset from the datasets folder.

    Parameters
    ----------
    name : str
        Name of the dataset to delete.

    """
    cfg.remove_dataset(name)


class h5pyFileWrapper:
    """ Used to close handle when a ``Dataset`` is destroyed."""
    def __init__(self, file):
        self.file = file

    def __del__(self):
        self.file.close()
