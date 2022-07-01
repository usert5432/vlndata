import unittest
import numpy as np

from vlndata.dataset.vldataset import VLDataset
from .test_dataset_base import TestDatasetBase, DATA_VLARR

class TestVLDataset(TestDatasetBase, unittest.TestCase):

    def _construct_dataset(
        self, scalar_groups, vlarr_groups, vlarr_limits = None
    ):
        return VLDataset(self.df, scalar_groups, vlarr_groups, vlarr_limits)

    def test_vlarr_limits(self):
        limit = 2
        scalar_groups = None
        vlarr_groups  = {
            'v-test1' : [ 'vc1', 'vc3' ],
            'v-test2' : [ 'vc2' ],
        }

        vlarr_limits = { 'v-test1' : limit, }

        dset = self._construct_dataset(
            scalar_groups, vlarr_groups, vlarr_limits
        )

        data_null = {
            'v-test1' : [
                np.stack((x[:limit], y[:limit]), axis = 1)
                    for (x, y) in zip(DATA_VLARR['vc1'], DATA_VLARR['vc3'])
            ],
            'v-test2' : [
                np.expand_dims(np.array(x), axis = 1)
                    for x in DATA_VLARR['vc2']
            ],
        }

        self._compare_data(dset, data_null)

if __name__ == '__main__':
    unittest.main()

