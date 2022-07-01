"""Test data slicing by a subframe decorator `DataFrame`"""

import unittest
import numpy as np

from vlndata.data_frame.dict_frame    import DictFrame
from vlndata.data_frame.shuffle_frame import ShuffleFrame
from .tests_data_frame_base           import TestDataFrameFuncs

class TestShuffleFrame(unittest.TestCase, TestDataFrameFuncs):

    def test_scalar_subframe(self):
        seed = 1
        data = { 'var' : [ 1, 2, 3, 4, -1 ] }

        prg = np.random.default_rng(seed)

        indices = np.arange(0, len(data['var']))
        prg.shuffle(indices)

        data_scalar = { 'var' : np.array(data['var'])[indices] }

        df = ShuffleFrame(DictFrame(data, None), seed)
        self._compare_scalar_columns(data_scalar, df, 'var')

if __name__ == '__main__':
    unittest.main()

