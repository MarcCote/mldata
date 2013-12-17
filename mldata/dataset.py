# -*- coding: utf-8 -*-


class Dataset(list):
    info = {}

    def __init__(self, data=[]):
        super(Dataset, self).__init__(data)


class LazyDataset(Dataset):
    def __init__(self, lazy_functions):
        super(LazyDataset, self).__init__()
        self.lazy_functions = lazy_functions

    def __iter__(self):
        return self.lazy_functions['__iter__']()
