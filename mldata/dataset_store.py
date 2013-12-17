
import os
import h5py
import numpy as np
import itertools
import types
import mldata
import mldata.utils
from mldata.dataset import Dataset, LazyDataset


from mldata.utils.constants import DATASETS_FOLDER
from mldata.utils.utils import buffered_iter

def supervised_factory(examples, targets):
    def lazy_iter():
        for e, t in itertools.izip(buffered_iter(examples), buffered_iter(targets)):
            yield e, t

    lazy_functions = {
                        '__iter__': lazy_iter,
                     }

    return lazy_functions

def load(path_or_name, lazy=False):
    path = path_or_name
    if not os.path.isfile(path):
        if (path_or_name + ".h5") not in os.listdir(DATASETS_FOLDER):
            print "Unknown dataset: '{0}'".format(path_or_name)
            return

        path = os.path.join(DATASETS_FOLDER, path_or_name + ".h5")

    return _load_from_file(path, lazy)

def _load_from_file(path, lazy=False):
    dataset = Dataset()
    
    if not lazy:
        with h5py.File(path, mode='r') as f:
            dataset = Dataset(itertools.izip(f['input'][()], f['output'][()]))
    else:
        f = h5py.File(path, mode='r')
        lazy_functions = supervised_factory(f['input'], f['output'])
        dataset = LazyDataset(lazy_functions)

    return dataset
