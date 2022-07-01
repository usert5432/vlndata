"""VarFrame tests"""

import unittest
import numpy as np

from vlndata.data_frame.dict_frame import DictFrame
from vlndata.data_frame.var_frame  import VarFrame
from .tests_data_frame_base import TestDataFrameFuncs

class TestVarFrame(TestDataFrameFuncs, unittest.TestCase):

    _data_scalar = {
        'c1' : [ 1, 2, 3, 4, -1 ],
        'c2' : [ 9, 8, 1, -2, 5 ],
        'c3' : list(range(5)),
    }

    _data_vlarr = {
        'vc1' : [ [1, 2], [], [3], [4,5,6,7], [-1] ],
        'vc2' : [ [0, 8], [], [1], [1,2,3,4], [-2] ],
        'vc3' : list(list(range(x)) for x in [ 2, 0, 1, 4, 1 ]),
    }

    def _create_data_frame(self, variables, lazy = False):
        df = DictFrame(self._data_scalar, self._data_vlarr)
        df = VarFrame(df, variables, lazy)

        return df

    def test_df_passthrough(self):
        df = self._create_data_frame(variables = None)
        self._compare_scalar_columns(self._data_scalar, df, 'c1')

    def test_df_scalar_var(self):
        variables = {
            'var1' : lambda df : 2 * df['c1']
        }
        data         = { **self._data_scalar }
        data['var1'] = 2 * np.array(self._data_scalar['c1'])

        for lazy in [ True, False ]:
            df = self._create_data_frame(variables, lazy)
            self._compare_scalar_columns(data, df, 'c1')
            self._compare_scalar_columns(data, df, 'var1')

if __name__ == '__main__':
    unittest.main()

