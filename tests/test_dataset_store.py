from ipdb import set_trace as dbg

import os
import tempfile
import hashlib
import numpy as np
import itertools
import time
from functools import partial

from numpy.testing import (assert_equal,
                           assert_almost_equal,
                           assert_array_equal,
                           assert_array_almost_equal,
                           assert_raises)


import mldata
import mldata.dataset_store as dataset_store

DATA_DIR = os.path.join(os.path.realpath(mldata.__path__[0]), "..", "tests", "data")

def load_mnist(lazy):
    """
    Load mnist dataset from a hdf5 file and test if it matches mlpython's one.
    """
    dataset_name = 'mnist'

    start = time.time()
    import mlpython.datasets.store as mlstore
    mldatasets = mlstore.get_classification_problem(dataset_name, load_to_memory= (not lazy))
    print "mlpython version loaded ({0:.2f}sec).".format(time.time() - start)

    start = time.time()
    dataset_name = os.path.join(os.environ['MLPYTHON_DATASET_REPO'], dataset_name + ".h5")
    dataset = mldata.dataset_store.load(dataset_name, lazy=lazy)
    print "mldata version loaded ({0:.2f}sec).".format(time.time() - start)

    print "Comparing first 1000..."
    count = 0
    for (e1, t1), (e2,  t2) in itertools.izip(dataset, itertools.chain(*mldatasets)):
        #print t1, t2
        assert_array_almost_equal(e1, e2)
        assert_equal(t1, t2)
        
        count += 1
        if count >= 1000:
            break


def test_load_mnist():
    """
    Load mnist dataset from a hdf5 file and test if it matches mlpython's one.
    """
    load_mnist(lazy=False)

def test_load_mnist_lazy():
    """
    Lazy load mnist dataset from a hdf5 file and test if it matches mlpython's one.
    """
    load_mnist(lazy=True)
