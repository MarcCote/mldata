""" Manages the configuration file for SMART-Data."""
import configparser
import os
from os.path import expanduser, join

from shutil import rmtree

SMARTCONFIG_ENV = "SMART_CONFIG_PATH"
HOME_PATH = expanduser("~") if SMARTCONFIG_ENV not in os.environ else os.environ[SMARTCONFIG_ENV]
CONFIG_FILE = join(HOME_PATH, ".smartrc")
DATASETS_DIR = join(HOME_PATH, ".datasets")


def add_dataset(dataset_name):
    """ Add a dataset to the index."""
    path = os.path.join(_load_path(), dataset_name)

    if not os.path.isdir(path):
        os.mkdir(path)

    config = _load_config()
    config['datasets'][dataset_name] = path
    _save_config(config)

def remove_dataset(dataset_name):
    """ Remove a dataset from the index."""
    path = os.path.join(_load_path(), dataset_name)

    if os.path.isdir(path):
        rmtree(path, ignore_errors=True)

    config = _load_config()
    config.remove_option('datasets', dataset_name)
    _save_config(config)

def dataset_exists(dataset_name):
    """ Check if the dataset exists."""
    return _load_config().has_option('datasets', dataset_name)

def get_dataset_path(dataset_name):
    """ Retrieves the dataset path.

    Parameters
    ----------
    dataset_name : str
        Name of a dataset

    Returns
    -------
    path : str
        Path where ``dataset_name`` is saved.

    Raises
    ------
    KeyError
        If the path specified in the config file does not exist in the system.
    """
    cp = _load_config()
    path = cp['datasets'][dataset_name]
    if not os.path.isdir(path):
        raise KeyError("Wrong path in '{0}'!".format(CONFIG_FILE))

    return path

def _save_config(config):
    """ Save a config file in the default config file emplacement."""
    with open(CONFIG_FILE, 'w') as f:
        config.write(f)

def _load_config():
    """ Loads the configuration file for SMART-Data.

    Loads settings from ``$HOME/.smartrc`` used by the SMART-Data library.
    If ``$HOME/.smartrc`` does not exist, a default config file will be created.

    Returns
    -------
    config : configparser
        Settings load from ``$HOME/.smartrc``
    """
    if not os.path.exists(CONFIG_FILE):
        config = _create_default_config()
        if not os.path.exists(DATASETS_DIR):
            os.mkdir(DATASETS_DIR)
        _save_config(config)
        return config

    config = configparser.ConfigParser()
    config.read(CONFIG_FILE)
    return config

def _create_default_config():
    """ Build a default config file for SMART-Data.

    It stores the emplacement of dataset files (by default ``$HOME/.datasets``)
    and make an index of accessible datasets.

    Returns
    -------
    config : configparser
        Default config
    """

    config = configparser.ConfigParser(allow_no_value=True)
    config.add_section('settings')
    config.set('settings', "# Datasets path shouldn't be changed manually.")
    config.set('settings', 'path', DATASETS_DIR)
    config.add_section('datasets')
    return config

def _load_path():
    """ Loads the config file and return the path to datasets folder.

    Returns
    -------
    path : str
        Path to datasets folder.

    """
    config = _load_config()
    path = config['settings']['path']
    assert os.path.isdir(path), "Configured path is not a valid directory."
    return path
