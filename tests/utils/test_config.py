import os

import configparser as cp
import nose.tools as nt

import mldata.utils.config as cfg

def setup_module():
    # save current config file
    os.rename(cfg.CONFIGFILE, cfg.CONFIGFILE  +".bak")

def teardown_module():
    # restore config file
    os.rename(cfg.CONFIGFILE  +".bak", cfg.CONFIGFILE)

def test_load_config():
    cf = cfg._load_config()
    path = os.path.expanduser("~") + '/.datasets'

    nt.assert_equal(path, cf['config']['path'])
    nt.assert_equal(path, cfg._load_path())
    nt.assert_true(cp.has_section('dataset'))

def test_add_remove():
    cfg.add_dataset("test_dataset")
    nt.assert_true(cfg.dataset_exists("test_dataset"))

    nt.assert_equal(cfg.get_dataset_path("test_dataset"),
                    os.join(cfg._load_path(), "test_dataset"))
    path = cfg.get_dataset_path("test_dataset")
    nt.assert_true(os.path.isdir(path))

    cfg.remove_dataset("test_dataset")
    nt.assert_false(cfg.dataset_exists("test_dataset"))
    nt.assert_false(os.path.isdir(path))





