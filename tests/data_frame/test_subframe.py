"""Test data slicing by a subframe decorator `DataFrame`"""

import unittest

from vlndata.data_frame.dict_frame import DictFrame
from vlndata.data_frame.subframe   import SubFrame
from .tests_data_frame_base        import TestDataFrameFuncs

class TestsDataSlice(unittest.TestCase, TestDataFrameFuncs):

    def test_scalar_subframe(self):
        data        = { 'var' : [ 1, 2, 3, 4, -1 ] }

        indices     = [ 0, 2, 3 ]
        data_scalar = { 'var' : [ 1, 3, 4 ] }

        df = SubFrame(DictFrame(data, None), indices)
        self._compare_scalar_columns(data_scalar, df, 'var')

    def test_vlarr_subframe(self):
        data        = { 'var' : [ [1, 2], [], [3], [4,5,6,7], [-1] ] }

        indices     = [ 0, 4 ]
        data_vlarr  = { 'var' : [ [1, 2], [-1] ] }

        df = SubFrame(DictFrame(None, data), indices)
        self._compare_vlarr_columns(data_vlarr, df, 'var')

if __name__ == '__main__':
    unittest.main()

