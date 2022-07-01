import unittest
import numpy as np

from vlndata.data_frame.dict_frame import DictFrame
from vlndata.dataset.dataset_transform import DatasetTransform
from vlndata.dataset.transform.vlarr_sorter import VLArrSortTransform
from vlndata.dataset.vldataset import VLDataset

from ..dataset.funcs import TestDatasetFuncs

DATA_VLARR = {
    'vc1' : [ [1, 2], [], [3], [7,3,6,1],  [-1] ],
    'vc2' : [ [0, 8], [], [1], [1,-1,3,4], [-2] ],
    'vc3' : list(list(range(x)) for x in [ 2, 0, 1, 4, 1 ]),
}

class TestVLArrSorters(TestDatasetFuncs, unittest.TestCase):

    def _construct_dataset(self, vlarr_data, vlarr_groups, transform):
        df     = DictFrame(None, vlarr_data)
        result = VLDataset(df, None, vlarr_groups)

        if not isinstance(transform, list):
            transform = [ transform, ]

        return DatasetTransform(result, transform)

    def test_vlarr_sort_ascending_simple(self):
        vlarr_data   = { k : DATA_VLARR[k] for k in [ 'vc1' ] }
        vlarr_groups = { 'group1' : [ 'vc1', ] }

        transform = VLArrSortTransform(
            vlarr_group = 'group1', column = 'vc1', ascending = True
        )

        dset = self._construct_dataset(vlarr_data, vlarr_groups, transform)
        data_null = {
            'group1' : [
                np.expand_dims(x, axis = 1)
                    for x in [ [1, 2], [], [3], [1,3,6,7],  [-1] ]
            ]
        }

        self._compare_data(dset, data_null)

    def test_vlarr_sort_descending_simple(self):
        vlarr_data   = { k : DATA_VLARR[k] for k in [ 'vc1' ] }
        vlarr_groups = { 'group1' : [ 'vc1', ] }

        transform = VLArrSortTransform(
            vlarr_group = 'group1', column = 'vc1', ascending = False
        )

        dset = self._construct_dataset(vlarr_data, vlarr_groups, transform)
        data_null = {
            'group1' : [
                np.expand_dims(x, axis = 1)
                    for x in [ [2, 1], [], [3], [7,6,3,1],  [-1] ]
            ]
        }

        self._compare_data(dset, data_null)

    def test_vlarr_sort_ascending_merged(self):
        vlarr_data   = DATA_VLARR
        vlarr_groups = { 'group1' : [ 'vc1', 'vc2', 'vc1' ] }

        transform = VLArrSortTransform(
            vlarr_group = 'group1', column = 'vc2', ascending = True
        )

        dset = self._construct_dataset(vlarr_data, vlarr_groups, transform)
        data_null = {
            'group1' : [
                np.stack((x, y, z), axis = 1)
                for (x, y, z) in zip(
                # originals:
                # vc1: [ [1, 2], [], [3], [7,3,6,1],  [-1] ],
                # vc2: [ [0, 8], [], [1], [1,-1,3,4], [-2] ],
                    [ [1, 2], [], [3], [3,7,6,1],  [-1] ],
                    [ [0, 8], [], [1], [-1,1,3,4], [-2] ],
                    [ [1, 2], [], [3], [3,7,6,1],  [-1] ],
                )
            ]
        }

        self._compare_data(dset, data_null)

if __name__ == '__main__':
    unittest.main()


