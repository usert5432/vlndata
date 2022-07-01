import unittest
import numpy as np

from vlndata.data_frame.dict_frame import DictFrame
from vlndata.dataset.dataset_transform import DatasetTransform
from vlndata.dataset.transform.noise_transform import NoiseTransform
from vlndata.dataset.vldataset import VLDataset

from ..dataset.funcs import TestDatasetFuncs

DATA_SCALAR = {
    'c1' : [ 1, 2, 3, 4, -1 ],
    'c2' : [ 9, 8, 1, -2, 5 ],
    'c3' : list(range(5)),
}

DATA_VLARR = {
    'vc1' : [ [1, 2], [], [3], [7,3,6,1],  [-1] ],
    'vc2' : [ [0, 8], [], [1], [1,-1,3,4], [-2] ],
    'vc3' : list(list(range(x)) for x in [ 2, 0, 1, 4, 1 ]),
}

def pack(*arrays):
    if len(arrays) == 1:
        return np.expand_dims(arrays[0], 1)

    return np.stack(arrays, axis = 1)

def pack_vlarrs(*arrays):
    return [ pack(*t) for t in zip(*arrays) ]

def apply_noise(array, noise, relative = False):
    if relative:
        return np.fromiter(
            ((1 + noise) + x for x in array),
            dtype = np.float32, count = len(array)
        )

    return np.fromiter(
        (noise + x for x in array), dtype = np.float32, count = len(array)
    )

def apply_vlarr_noise(array, noise, relative = False):
    if relative:
        return [
            (1 + noise) * np.array(x) for x in array
        ]

    return [ noise + np.array(x) for x in array ]

class TestNoiseTransform(TestDatasetFuncs, unittest.TestCase):

    df = DictFrame(DATA_SCALAR, DATA_VLARR)

    def _construct_dataset(self, scalar_groups, vlarr_groups, transform):
        result = VLDataset(self.df, scalar_groups, vlarr_groups)

        if not isinstance(transform, list):
            transform = [ transform, ]

        return DatasetTransform(result, transform)

    def test_simple_scalar_noise(self):
        noise         = 2
        scalar_groups = { 'group1' : [ 'c1', ] }
        vlarr_groups  = None

        transform = NoiseTransform(
            { 'name' : 'debug', 'value' : noise },
            correlated    = True,
            relative      = False,
            scalar_groups = { 'group1' : [ 'c1', ] },
            vlarr_groups  = None
        )

        dset = self._construct_dataset(scalar_groups, vlarr_groups, transform)
        data_null = {
            'group1' : pack(apply_noise(DATA_SCALAR['c1'], noise))
        }

        self._compare_data(dset, data_null)

    def test_partial_scalar_noise(self):
        noise         = 2
        scalar_groups = {
            'group1' : [ 'c1', 'c2', 'c3' ],
            'group2' : [ 'c2' ],
        }
        vlarr_groups  = None

        transform = NoiseTransform(
            { 'name' : 'debug', 'value' : noise },
            correlated    = True,
            relative      = False,
            scalar_groups = { 'group1' : [ 'c2', ] },
            vlarr_groups  = None
        )

        dset = self._construct_dataset(scalar_groups, vlarr_groups, transform)
        data_null = {
            'group1' : pack(
                DATA_SCALAR['c1'],
                apply_noise(DATA_SCALAR['c2'], noise),
                DATA_SCALAR['c3']
            ),
            'group2' : pack(DATA_SCALAR['c2']),
        }

        self._compare_data(dset, data_null)

    def test_partial_scalar_noise_uncorrelated(self):
        noise         = 2
        scalar_groups = {
            'group1' : [ 'c1', 'c2', 'c3' ],
            'group2' : [ 'c2' ],
        }
        vlarr_groups  = None

        transform = NoiseTransform(
            { 'name' : 'debug', 'value' : noise },
            correlated    = False,
            relative      = False,
            scalar_groups = { 'group1' : [ 'c2', ] },
            vlarr_groups  = None
        )

        dset = self._construct_dataset(scalar_groups, vlarr_groups, transform)
        data_null = {
            'group1' : pack(
                DATA_SCALAR['c1'],
                apply_noise(DATA_SCALAR['c2'], noise),
                DATA_SCALAR['c3']
            ),
            'group2' : pack(DATA_SCALAR['c2']),
        }

        self._compare_data(dset, data_null)

    def test_weighted_partial_scalar_noise_uncorrelated(self):
        noise         = 2
        scalar_groups = {
            'group1' : [ 'c1', 'c2', 'c3' ],
            'group2' : [ 'c2' ],
        }
        vlarr_groups  = None

        transform = NoiseTransform(
            { 'name' : 'debug', 'value' : noise },
            correlated    = False,
            relative      = False,
            scalar_groups = {
                'group1' : { 'c2' : 1, 'c3' : 5 }
            },
            vlarr_groups  = None
        )

        dset = self._construct_dataset(scalar_groups, vlarr_groups, transform)
        data_null = {
            'group1' : pack(
                DATA_SCALAR['c1'],
                apply_noise(DATA_SCALAR['c2'], 1 * noise),
                apply_noise(DATA_SCALAR['c3'], 5 * noise),
            ),
            'group2' : pack(DATA_SCALAR['c2']),
        }

        self._compare_data(dset, data_null)

    def test_simple_vlarr_noise(self):
        noise         = 2
        scalar_groups = None
        vlarr_groups  = { 'group1' : [ 'vc1', ] }

        transform = NoiseTransform(
            { 'name' : 'debug', 'value' : noise },
            correlated    = True,
            relative      = False,
            scalar_groups = None,
            vlarr_groups  = { 'group1' : [ 'vc1', ] },
        )

        dset = self._construct_dataset(scalar_groups, vlarr_groups, transform)
        data_null = {
            'group1' : pack_vlarrs(apply_vlarr_noise(DATA_VLARR['vc1'], noise))
        }

        self._compare_data(dset, data_null)

    def test_partial_vlarr_noise(self):
        noise         = 2
        scalar_groups = None
        vlarr_groups = {
            'group1' : [ 'vc1', 'vc2', 'vc3' ],
            'group2' : [ 'vc2' ],
        }

        transform = NoiseTransform(
            { 'name' : 'debug', 'value' : noise },
            correlated    = True,
            relative      = False,
            scalar_groups = None,
            vlarr_groups  = { 'group1' : [ 'vc2', ] },
        )

        dset = self._construct_dataset(scalar_groups, vlarr_groups, transform)
        data_null = {
            'group1' : pack_vlarrs(
                DATA_VLARR['vc1'],
                apply_vlarr_noise(DATA_VLARR['vc2'], noise),
                DATA_VLARR['vc3']
            ),
            'group2' : pack_vlarrs(DATA_VLARR['vc2']),
        }

        self._compare_data(dset, data_null)

    def test_partial_vlarr_noise_uncorrelated(self):
        noise         = 2
        scalar_groups = None
        vlarr_groups = {
            'group1' : [ 'vc1', 'vc2', 'vc3' ],
            'group2' : [ 'vc2' ],
        }

        transform = NoiseTransform(
            { 'name' : 'debug', 'value' : noise },
            correlated    = False,
            relative      = False,
            scalar_groups = None,
            vlarr_groups  = { 'group1' : [ 'vc2', ] },
        )

        dset = self._construct_dataset(scalar_groups, vlarr_groups, transform)
        data_null = {
            'group1' : pack_vlarrs(
                DATA_VLARR['vc1'],
                apply_vlarr_noise(DATA_VLARR['vc2'], noise),
                DATA_VLARR['vc3']
            ),
            'group2' : pack_vlarrs(DATA_VLARR['vc2']),
        }

        self._compare_data(dset, data_null)

    def test_weighted_partial_vlarr_noise_uncorrelated(self):
        noise         = 2
        scalar_groups = None
        vlarr_groups = {
            'group1' : [ 'vc1', 'vc2', 'vc3' ],
            'group2' : [ 'vc2' ],
        }

        transform = NoiseTransform(
            { 'name' : 'debug', 'value' : noise },
            correlated    = False,
            relative      = False,
            scalar_groups = None,
            vlarr_groups  = { 'group1' : { 'vc2' : 1, 'vc3' : 5 } },
        )

        dset = self._construct_dataset(scalar_groups, vlarr_groups, transform)
        data_null = {
            'group1' : pack_vlarrs(
                DATA_VLARR['vc1'],
                apply_vlarr_noise(DATA_VLARR['vc2'], 1 * noise),
                apply_vlarr_noise(DATA_VLARR['vc3'], 5 * noise),
            ),
            'group2' : pack_vlarrs(DATA_VLARR['vc2']),
        }

        self._compare_data(dset, data_null)

if __name__ == '__main__':
    unittest.main()


