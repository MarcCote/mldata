import configparser
import os
from os.path import expanduser, join

from shutil import rmtree

CONFIGFILE = join(expanduser("~"), '.mldataConfig')

def add_dataset(dataset_name):
    """ Add a dataset to the index."""

    path = os.path.join(_load_path(), dataset_name)

    if not os.path.isdir(path):
        os.mkdir(path)

    cp = _load_config()
    cp['datasets'][dataset_name] = path
    _save_config(cp)

def remove_dataset(dataset_name):
    """ Remove a dataset from the index."""
    path = os.path.join(_load_path(), dataset_name)

    if os.path.isdir(path): # Does path exist ?
        rmtree(path, ignore_errors=True)

    cp = _load_config()
    cp.remove_option('datasets', dataset_name)
    _save_config(cp)

def get_dataset_path(dataset_name):
    """ Retreive the dataset path.

    Parameters
    ----------
    dataset_name : str
        Name of a dataset

    Returns
    -------
    str
        The string of the path where ``dataset_name`` is saved.

    Raises
    ------
    KeyError
        If the path specified in the config file does not exist in the system.
    """
    cp = _load_config()
    path = cp['datasets'][dataset_name]
    if not os.path.isdir(path):
        raise KeyError("Wrong path in .mldataConfig.")
    else:
        return path

def dataset_exists(dataset_name):
    """ Check if the dataset exists."""
    return _load_config().has_option('datasets', dataset_name)

def _save_config(config):
    """ Save a config file in the default config file emplacement."""
    with open(CONFIGFILE, 'w') as f:
        config.write(f)


def _load_config():
    """ Loads the configuration file for MLData."""
    if not os.path.exists(CONFIGFILE):
        _create_default_config()
    cfg = configparser.ConfigParser()
    cfg.read(CONFIGFILE)
    return cfg

def _create_default_config():
    """ Build and save a default config file for MLData.

    The default config is saved as ``.MLDataConfig`` in the ``$HOME`` folder
    or its equivalent. It stores the emplacement of dataset files and make an
    index of accessible datasets.

    """
    cp = configparser.ConfigParser()
    path = join(expanduser("~"), '.datasets')
    if not os.path.isdir(path):
        os.mkdir(path)
    cp['config'] = {'path': path}
    cp['datasets'] = {}
    _save_config(cp)
    with open(CONFIGFILE, 'a') as f:
        f.write("# Datasets path shouldn't be changed manually.\n")

def _load_path():
    """ Load the config file at the default emplacement.

    Returns
    -------
    str
        A list of strings giving the paths to dataset folders.

    """
    cp = _load_config()
    path = cp['config']['path']
    assert os.path.isdir(path), "Configured path is not a valid directory."
    return path
