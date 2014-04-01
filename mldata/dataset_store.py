""" Manages dataset read/write operations."""
import os
import itertools
import pickle as pk
import hashlib

import h5py
import numpy as np

from mldata.dataset import Dataset, Metadata
import mldata.utils.config as cfg
from mldata.utils.utils import buffered_iter


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
    lazy : bool
        If set to ``True``, the dataset will be read with ``h5py`` without
        loading the whole dataset in memory. If set to ``False``, the file is
        mapped in memory.

    Returns
    -------
    Dataset
        Return the loaded dataset, if it exists. Else, return ``None``.

    """
    path = None
    if cfg.dataset_exists(dset_name):
        path = cfg.get_dataset_path(dset_name)
    return _load_from_file(dset_name, path, lazy)

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
        If this is a special version of a dataset,

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
    with open(os.path.join(path, name) + '.meta', 'rb') as f:
        metadata = pk.load(f)

    dataset = None
    file_to_load = metadata.hash + ".data"
    if lazy:
        dataset = h5py.File(file_to_load, mode='r', driver=None)
    else:
        dataset = h5py.File(file_to_load, mode='r', driver=core)

    data   = dataset['/']["data"]
    target = dataset['/']["target"]

    return Dataset(metadata, data, target)

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
            f.create_dataset('data', dataset.data)
            if dataset.target is not Null:
                f.create_dataset('targets', dataset.target)

def _save_metadata(metadata, path, filename):
    """ Pickle the metadata.

    Parameters
    ----------
    metadata : Metadata
    path : str
    filename : str

    .. todo:: A dataset could be orphaned if overwritten by another metadata
              file. This needs to be checked in a future version.

    """

    if filename not in os.listdir(path):
        with open(os.path.join(path, filename), 'wb') as f:
            pk.dump(metadata, f, pk.HIGHEST_PROTOCOL)