from typing import Any, Dict, List, Union
import numpy as np

from vlndata.data_frame.dict_frame import DictFrame
from .funcs import TestDatasetFuncs

DATA_SCALAR = {
    'c1' : [ 1, 2, 3, 4, -1 ],
    'c2' : [ 9, 8, 1, -2, 5 ],
    'c3' : list(range(5)),
}

DATA_VLARR = {
    'vc1' : [ [1, 2], [], [3], [4,5,6,7], [-1] ],
    'vc2' : [ [0, 8], [], [1], [1,2,3,4], [-2] ],
    'vc3' : list(list(range(x)) for x in [ 2, 0, 1, 4, 1 ]),
}

class TestDatasetBase(TestDatasetFuncs):

    df = DictFrame(DATA_SCALAR, DATA_VLARR)

    def _construct_dataset(
        self, scalar_groups, vlarr_groups, vlarr_limits = None
    ):
        raise RuntimeError

    def test_single_scalar(self):
        scalar_groups = { 'test1' : [ 'c1', ] }
        vlarr_groups  = None

        dset = self._construct_dataset(scalar_groups, vlarr_groups)
        data_null = {
            'test1' : np.expand_dims(np.array(DATA_SCALAR['c1']), 1)
        }

        self._compare_data(dset, data_null)

    def test_merged_scalars(self):
        scalar_groups = { 'test1' : [ 'c1', 'c3' ] }
        vlarr_groups  = None

        dset = self._construct_dataset(scalar_groups, vlarr_groups)
        data_null = {
            'test1' : np.stack(
                (DATA_SCALAR['c1'], DATA_SCALAR['c3']), axis = 1
            )
        }

        self._compare_data(dset, data_null)

    def test_two_scalar_groups(self):
        scalar_groups = {
            'test1' : [ 'c1', 'c3' ],
            'test2' : [ 'c2' ],
        }
        vlarr_groups  = None

        dset = self._construct_dataset(scalar_groups, vlarr_groups)
        data_null = {
            'test1' : np.stack(
                (DATA_SCALAR['c1'], DATA_SCALAR['c3']), axis = 1
            ),
            'test2' : np.expand_dims(np.array(DATA_SCALAR['c2']), axis = 1)
        }

        self._compare_data(dset, data_null)

    def test_single_vlarr(self):
        scalar_groups = None
        vlarr_groups  = { 'test1' : [ 'vc1', ] }

        dset = self._construct_dataset(scalar_groups, vlarr_groups)
        data_null = {
            'test1' : [
                np.expand_dims(np.array(x), axis = 1)
                    for x in DATA_VLARR['vc1']
            ]
        }

        self._compare_data(dset, data_null)

    def test_merged_vlarr(self):
        scalar_groups = None
        vlarr_groups  = { 'test1' : [ 'vc1', 'vc3' ] }

        dset = self._construct_dataset(scalar_groups, vlarr_groups)
        data_null = {
            'test1' : [
                np.stack((x, y), axis = 1)
                    for (x, y) in zip(DATA_VLARR['vc1'], DATA_VLARR['vc3'])
            ]
        }

        self._compare_data(dset, data_null)

    def test_two_vlarr_group(self):
        scalar_groups = None
        vlarr_groups  = {
            'test1' : [ 'vc1', 'vc3' ],
            'test2' : [ 'vc2' ],
        }

        dset = self._construct_dataset(scalar_groups, vlarr_groups)
        data_null = {
            'test1' : [
                np.stack((x, y), axis = 1)
                    for (x, y) in zip(DATA_VLARR['vc1'], DATA_VLARR['vc3'])
            ],
            'test2' : [
                np.expand_dims(np.array(x), axis = 1)
                    for x in DATA_VLARR['vc2']
            ],
        }

        self._compare_data(dset, data_null)

    def test_complex_loader(self):
        scalar_groups = {
            's-test1' : [ 'c1', 'c3' ],
            's-test2' : [ 'c2' ],
        }
        vlarr_groups  = {
            'v-test1' : [ 'vc1', 'vc3' ],
            'v-test2' : [ 'vc2' ],
        }

        dset = self._construct_dataset(scalar_groups, vlarr_groups)
        data_null = {
            's-test1' : np.stack(
                (DATA_SCALAR['c1'], DATA_SCALAR['c3']), axis = 1
            ),
            's-test2' : np.expand_dims(np.array(DATA_SCALAR['c2']), axis = 1),
            'v-test1' : [
                np.stack((x, y), axis = 1)
                    for (x, y) in zip(DATA_VLARR['vc1'], DATA_VLARR['vc3'])
            ],
            'v-test2' : [
                np.expand_dims(np.array(x), axis = 1)
                    for x in DATA_VLARR['vc2']
            ],
        }

        self._compare_data(dset, data_null)

