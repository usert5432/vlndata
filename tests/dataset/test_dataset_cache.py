"""A template for correctness of DataFrame parsing tests"""

import unittest

from vlndata.dataset.vldataset     import VLDataset
from vlndata.dataset.dataset_cache import DatasetCache
from .test_dataset_base import TestDatasetBase

class TestVLDataset(TestDatasetBase, unittest.TestCase):

    def _construct_dataset(
        self, scalar_groups, vlarr_groups, vlarr_limits = None
    ):
        dset = VLDataset(self.df, scalar_groups, vlarr_groups, vlarr_limits)
        dset = DatasetCache(dset)

        return dset

if __name__ == '__main__':
    unittest.main()
