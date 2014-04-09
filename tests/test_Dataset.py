import numpy as np
import nose.tools as nt
import copy

from mldata.dataset import Dataset, Metadata

class Dataset_test:
    def setup_class(self):
        self.dataSmall = np.random.rand(30,5)
        self.dataLarge = np.random.rand(3000, 5)
        self.targetSmall = np.random.rand(30,1)
        self.targetLarge = np.random.rand(3000,1)

        self.metadataS = Metadata()
        self.metadataS.splits = (10,20,30)
        self.metadataS.nb_examples = 30
        self.dsetS = Dataset(self.metadataS,self.dataSmall, self.targetSmall)

    def test_Dataset(self):
        dset = Dataset(self.metadataS, self.dataSmall)
        nt.assert_equal(dset.meta_data, self.metadataS)
        nt.assert_equal(dset.data, self.dataSmall)
        nt.assert_is_none(dset.target)

        dsetS = Dataset(self.metadataS, self.dataSmall, self.targetSmall)
        nt.assert_is_not_none(dsetL.target)

    def test_hash(self):
        nt.assert_equal(self.dsetS.__hash__(), self.dsetS.__hash__())

        dset = Dataset(self.metadataS, self.dataSmall, self.targetSmall)
        nt.assert_equal(dset.__hash__(), self.dsetS.__hash__())

        dset2 = Dataset(self.metadataS, self.dataSmall)
        nt.assert_not_equal(dset2.__hash__(), dset.__hash__())

        dset3 = Dataset(self.metadataS, self.dataLarge)
        nt.assert_not_equal(dset2.__hash__(), dset3.__hash__())
        nt.assert_not_equal(dset3.__hash__(), dset.__hash__())

        meta = Metadata()
        meta.name = "AnotherName"
        meta.splits = (10,20,30)
        dset4 = Dataset(meta, self.dataSmall)
        nt.assert_equal(dset4.__hash__(), dset2.__hash__())
        nt.assert_not_equal(dset4.__hash__(), dset3.__hash__())

    def test_get_splits(self):
        nt.assert_equal(self.dsetS.get_splits(), (10,20,30))

    def test_len(self):
        nt.assert_equal(len(self.dsetS), len(self.dsetS.data))
        nt.assert_equal(len(self.dsetS), self.dsetS.meta_data.nb_examples)
        nt.assert_equal(len(self.dsetS), self.dsetS.get_splits()[-1])

    def test_preprocess(self):
        data2 = self.dsetS.data * 2
        meta = copy.deepcopy(self.metadataS)
        meta.preprocess = lambda x: Dataset(self.metadataS, x.data * 2,
                                            x.data*2)
        dset2 = Dataset(meta, self.dataSmall, self.targetSmall)

        nt.assert_equal(data2, dset2.data)

    def test_iter(self):
        # Without targets
        dt, tg = [[z[i] for z in self.dsetS] for i in [0,1]]
        nt.assert_equal(np.array(dt), self.dataSmall)

        dset = Dataset(self.metadataS, self.dataSmall)
        nt.assert_equal(np.array([z[0] for z in dset]), self.dataSmall)

        for a,b in zip(self.dataSmall, self.dsetS):
            nt.assert_equal(a,b)

    def test_get(self):
        for i in range(0, len(self.dataSmall)):
            nt.assert_equal(self.dataSmall[i],self.dsetS[i][0])


