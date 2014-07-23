import copy
from itertools import chain

import numpy as np
import nose.tools as nt

from mldata.dataset import Dataset, Metadata


class Dataset_test:
    @classmethod
    def setup_class(self):
        self.dataSmall = np.random.random((30, 5))
        self.dataLarge = np.random.random((3000, 5))
        self.targetSmall = np.random.random((30, 1))
        self.targetLarge = np.random.random((3000, 1))

        self.metadataS = Metadata()
        self.metadataS.splits = (10, 20, 30)
        self.metadataS.nb_examples = 30
        self.metadataL = Metadata()
        self.metadataL.splits = (1000, 1000, 1000)
        self.metadataL.nb_examples = 3000
        self.dsetS = Dataset(self.metadataS, self.dataSmall, self.targetSmall)
        self.dsetL = Dataset(self.metadataL, self.dataLarge, self.targetLarge)

    def test_Dataset(self):
        dset = Dataset(self.metadataS, self.dataSmall)
        nt.assert_equal(dset.meta_data, self.metadataS)
        nt.assert_true(np.array_equal(dset.data, self.dataSmall))
        nt.assert_is_none(dset.target)

        dsetS = Dataset(self.metadataS, self.dataSmall, self.targetSmall)
        nt.assert_is_not_none(dsetS.target)

    def test_hash(self):
        nt.assert_equal(self.dsetS.__hash__(), self.dsetS.__hash__())

        dset = Dataset(self.metadataS, self.dataSmall, self.targetSmall)
        nt.assert_equal(dset.__hash__(), self.dsetS.__hash__())

        dset2 = Dataset(self.metadataS, self.dataSmall)
        nt.assert_not_equal(dset2.__hash__(), dset.__hash__())

        dset3 = Dataset(self.metadataL, self.dataLarge)
        nt.assert_not_equal(dset2.__hash__(), dset3.__hash__())
        nt.assert_not_equal(dset3.__hash__(), dset.__hash__())

        meta = Metadata()
        meta.name = "AnotherName"
        meta.splits = (10, 10, 10)  # alternative split form
        meta.nb_examples = 30
        dset4 = Dataset(meta, self.dataSmall)
        nt.assert_equal(dset4.__hash__(), dset2.__hash__())
        nt.assert_not_equal(dset4.__hash__(), dset3.__hash__())

    def test_len(self):
        nt.assert_equal(len(self.dsetS), len(self.dsetS.data))
        nt.assert_equal(len(self.dsetS), self.dsetS.meta_data.nb_examples)

    def test_preprocess(self):
        data2 = self.dataSmall * 2
        meta = copy.deepcopy(self.metadataS)
        meta.preprocess = double_dset
        dset2 = Dataset(meta, self.dataSmall, self.targetSmall)
        dset2 = dset2.apply()
        nt.assert_true(np.array_equal(data2, dset2.data))

    def test_iter(self):
        # With targets
        dt, tg = [[z[i] for z in self.dsetS] for i in [0, 1]]
        nt.assert_true(np.array_equal(np.array(dt), self.dataSmall))
        # Without targets
        dset = Dataset(self.metadataS, self.dataSmall)
        nt.assert_true(np.array_equal(np.array([z[0] for z in dset]),
                                      self.dataSmall))

    def test_get(self):
        for i in range(len(self.dataSmall)):
            nt.assert_true(np.array_equal(self.dataSmall[i], self.dsetS[i][0]))

    def test_get_splits_iterators(self):
        citer = chain.from_iterable(self.dsetS.get_splits_iterators())
        for a, b in zip(citer, self.dsetS):
            d1 = a[0]
            d2 = [b[0]]
            nt.assert_true(np.array_equal(d1, d2))

        sp = self.dsetL.meta_data.splits
        for splitn, it in zip(sp, self.dsetL.get_splits_iterators()):
            nt.assert_equal(sum(1 for _ in it), splitn)


def double_dset(dset):
    """ Basic preprocessing function. """
    return Dataset(dset.meta_data, dset.data * 2, dset.target * 2)
