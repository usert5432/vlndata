import unittest
import numpy as np

from vlndata.data_loader.funcs import vldata_dict_collate
from .funcs import TestDataLoaderFuncs

class TestCollateFunc(TestDataLoaderFuncs, unittest.TestCase):

    def test_simple_scalar_collate(self):
        batch_test = [ { 'test' : np.array([ 1, 2, 3 ]) }, ]
        data_null  = { 'test' : np.array([[ 1, 2, 3 ],]) }

        data_test = vldata_dict_collate(batch_test)
        self._compare_data(data_test, data_null)

    def test_scalar_collate(self):
        batch_test = [
            { 'test' : np.array([ 1, 2, 3 ]) },
            { 'test' : np.array([ 5, 6, 7 ]) },
        ]
        data_null  = {
            'test' : np.array([[ 1, 2, 3 ], [ 5, 6, 7 ] ])
        }

        data_test = vldata_dict_collate(batch_test)
        self._compare_data(data_test, data_null)

    def test_simple_vlarr_collate(self):
        batch_test = [
            { 'test' : np.array([ [1, 2], [3, 4], [5, 6] ]) },
            { 'test' : np.array([ [5, 6], [7, 8], [9, 0] ]) },
        ]
        data_null  = {
            'test' : np.array([
                [ [1, 2], [3, 4], [5, 6] ],
                [ [5, 6], [7, 8], [9, 0] ],
            ])
        }

        data_test = vldata_dict_collate(batch_test)
        self._compare_data(data_test, data_null)

    def test_padded_vlarr_collate(self):
        p = -1

        batch_test = [
            { 'test' : np.array([ [1, 2], [3, 4],        ]) },
            { 'test' : np.array([ [5, 6], [7, 8], [9, 0] ]) },
        ]
        data_null  = {
            'test' : np.array([
                [ [1, 2], [3, 4], [p, p] ],
                [ [5, 6], [7, 8], [9, 0] ],
            ])
        }

        data_test = vldata_dict_collate(batch_test, pad = p)
        self._compare_data(data_test, data_null)

    def test_complex_collate(self):
        p = -1

        batch_test = [
            {
                'test-s1'  : np.array([ 1, 2, 3 ]),
                'test-vl1' : np.array([ [1, 2], [3, 4],        ])
            },
            {
                'test-s1'  : np.array([ 5, 6, 7 ]),
                'test-vl1' : np.array([ [5, 6], [7, 8], [9, 0] ])
            },
            {
                'test-s1'  : np.array([ 8, 9, 0 ]),
                'test-vl1' : np.array([ [9, 8], ])
            },
        ]

        data_null  = {
            'test-s1' : np.array([
                [ 1, 2, 3 ],
                [ 5, 6, 7 ],
                [ 8, 9, 0 ],
            ]),
            'test-vl1' : np.array([
                [ [1, 2], [3, 4], [p, p] ],
                [ [5, 6], [7, 8], [9, 0] ],
                [ [9, 8], [p, p], [p, p] ],
            ])
        }

        data_test = vldata_dict_collate(batch_test, pad = p)
        self._compare_data(data_test, data_null)

if __name__ == '__main__':
    unittest.main()

