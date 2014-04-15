import os

import numpy as np
import nose.tools as nt

import mldata.dataset_store as ds

RND_MATRIX = np.random.rand(100,10)


def setup_module():
    np.savetxt("test.csv", RND_MATRIX)

def teardown_module():
    os.remove("test.csv")
    ds.remove("test_dset")

def test_CSV_importer():
    dset = ds.CSV_importer("test.csv",
                           "test_dset",
                           (70, 90, 100),
                           0)

    nt.assert_true(np.array_equal(RND_MATRIX[:,1:], dset.data))

def test_save_load():
    dset = ds.CSV_importer("test.csv",
                           "test_dset",
                           (70, 90, 100),
                           0)
    ds.save(dset, "v1")
    dset2 = ds.load("test_dset", "v1")

    nt.assert_equal(dset.__hash__(), dset2.__hash__())
    nt.assert_equal(dset.meta_data.name, dset2.meta_data.name)
    nt.assert_equal(dset.meta_data.dictionary, dset2.meta_data.dictionary)
    nt.assert_equal(dset.meta_data.nb_examples, dset2.meta_data.nb_examples)
    nt.assert_equal(dset.meta_data.splits, dset2.meta_data.splits)
    nt.assert_equal(dset2.meta_data.hash, dset2.__hash__())

    ndata = np.array(dset2.data)
    dset2.data = ndata * 2

    ds.save(dset2, version_name="v2")
    dset3 = ds.load("test_dset", "v2")

    nt.assert_not_equal(dset3.__hash__(), dset.__hash__())
    nt.assert_equal(dset3.meta_data.hash, dset3.__hash__())
    nt.assert_equal(dset.meta_data.name, dset3.meta_data.name)
    nt.assert_equal(dset.meta_data.dictionary, dset3.meta_data.dictionary)
    nt.assert_equal(dset.meta_data.nb_examples, dset3.meta_data.nb_examples)
    nt.assert_equal(dset.meta_data.splits, dset3.meta_data.splits)
