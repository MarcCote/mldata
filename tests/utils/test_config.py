import os

from nose.tools import assert_true, assert_false, assert_equal

from mldata.utils import config as cfg


def setup_module():
    # save current config file if needed
    if os.path.isfile(cfg.CONFIG_FILE):
        os.rename(cfg.CONFIG_FILE, cfg.CONFIG_FILE + ".bak")


def teardown_module():
    os.remove(cfg.CONFIG_FILE)
    # restore config file if needed
    if os.path.isfile(cfg.CONFIG_FILE + ".bak"):
        os.rename(cfg.CONFIG_FILE + ".bak", cfg.CONFIG_FILE)


def test_load_config():
    config = cfg._load_config()
    path = os.path.join(os.path.expanduser("~"), '.datasets')

    assert_equal(path, config['settings']['path'])
    assert_equal(path, cfg._load_path())
    assert_true(config.has_section('datasets'))


def test_add_remove():
    cfg.add_dataset("test_dataset")
    assert_true(cfg.dataset_exists("test_dataset"))

    assert_equal(cfg.get_dataset_path("test_dataset"),
                    os.path.join(cfg._load_path(), "test_dataset"))
    path = cfg.get_dataset_path("test_dataset")
    assert_true(os.path.isdir(path))

    cfg.remove_dataset("test_dataset")
    assert_false(cfg.dataset_exists("test_dataset"))
    assert_false(os.path.isdir(path))
