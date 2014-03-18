import configparser
from os.path import expanduser


def create_default_config():
    """ Build and save a default config file for MLData.

    The default config is saved as ``.MLDataConfig`` in the ``$HOME`` folder
    or its equivalent. The only thing present in the config file is a list of
    path of folders where datasets are stored.

    """
    cp = configparser.ConfigParser()
    cp['datasets'] = {'paths': '[' + expanduser("~")+'/.datasets' + ']'}
    save(cp)


def save(config):
    """ Save a config file in the default config file emplacement."""
    config.write(expanduser("~")+'.mldataConfig')


def load_paths():
    """ Load the config file at the default emplacement.

    Returns
    -------
    [str]
        A list of strings giving the paths to dataset folders.

    """
    cp = configparser.ConfigParser()
    cp.read(expanduser("~")+'.mldataConfig')
    l = cp['datasets']['paths']
    assert isinstance(eval(l), list), "The paths " + l + " is not a list."
    assert all(isinstance(e, str) for e in eval(l)), "Elements of the list" +\
                                                     l + " are not strings."
    return eval(l)