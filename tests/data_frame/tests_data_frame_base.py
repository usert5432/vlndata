from typing import Dict, List, Union
import numpy as np

from vlndata.data_frame.data_frame_base import DataFrameBase

NullDataType      = Union[np.ndarray, float, List[float]]
NullDataContainer = Union[np.ndarray, List[NullDataType]]
NullData          = Dict[str, NullDataContainer]

class TestDataFrameFuncs:

    def _retrieve_null_data(self, data : NullData, column : str) -> np.ndarray:
        dtype = None

        if isinstance(data[column][0], (list, tuple)):
            dtype = object

        return np.array(data[column], dtype = dtype)

    def _compare_full_columns(
        self, data : NullData, df : DataFrameBase, column : str
    ) -> None:
        data_null = data[column]
        data_test = df[column]

        self.assertTrue(np.all(np.isclose(data_test, data_null)))

    def _compare_scalar_columns_by_index(
        self, data : NullData, df : DataFrameBase, column : str
    ) -> None:
        data_null_list = self._retrieve_null_data(data, column)
        self.assertEqual(len(df), len(data_null_list))

        for i in range(len(df)):
            data_null = data_null_list[i]
            data_test = df.get_scalar(column, i)

            self.assertTrue(np.isclose(data_test, data_null))

    def _compare_scalar_columns(
        self, data : NullData, df : DataFrameBase, column : str
    ) -> None:
        self._compare_full_columns(data, df, column)
        self._compare_scalar_columns_by_index(data, df, column)

    def _compare_vlarr_columns(
        self, data : NullData, df : DataFrameBase, column : str
    ) -> None:
        data_null_list = self._retrieve_null_data(data, column)
        self.assertEqual(len(df), len(data_null_list))

        for i in range(len(df)):
            data_null = data_null_list[i]
            data_test = df.get_vlarr(column, i)

            self.assertTrue(np.all(np.isclose(data_test, data_null)))

class TestsDataFrameBase(TestDataFrameFuncs):

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

    def _create_data_frame(self, data_scalar = None, data_vlarr = None):
        raise NotImplementedError

    def test_scalar_column_1(self):
        cols = [ 'c1', ]
        data = { k : self._data_scalar[k] for k in cols }
        df   = self._create_data_frame(data_scalar = data)

        self.assertEqual(df.columns(), cols)
        self._compare_scalar_columns(data, df, cols[0])

    def test_scalar_column_2(self):
        cols = [ 'c1', 'c2', 'c3' ]
        data = { k : self._data_scalar[k] for k in cols }
        df   = self._create_data_frame(data_scalar = data)

        self.assertEqual(df.columns(), cols)

        self._compare_scalar_columns(data, df, cols[0])
        self._compare_scalar_columns(data, df, cols[1])
        self._compare_scalar_columns(data, df, cols[2])

    def test_vlarr_column_1(self):
        cols = [ 'vc1', ]
        data = { k : self._data_vlarr[k] for k in cols }
        df   = self._create_data_frame(data_vlarr = data)

        self.assertEqual(df.columns(), cols)
        self._compare_vlarr_columns(data, df, cols[0])

    def test_vlarr_column_2(self):
        cols = [ 'vc1', 'vc2', 'vc3' ]
        data = { k : self._data_vlarr[k] for k in cols }
        df   = self._create_data_frame(data_vlarr = data)

        self.assertEqual(df.columns(), cols)
        self._compare_vlarr_columns(data, df, cols[0])
        self._compare_vlarr_columns(data, df, cols[1])
        self._compare_vlarr_columns(data, df, cols[2])

    def test_mixed_df(self):
        df = self._create_data_frame(
            data_scalar = self._data_scalar, data_vlarr = self._data_vlarr
        )

        self._compare_vlarr_columns(self._data_vlarr, df, 'vc1')
        self._compare_vlarr_columns(self._data_vlarr, df, 'vc2')
        self._compare_vlarr_columns(self._data_vlarr, df, 'vc3')

        self._compare_scalar_columns(self._data_scalar, df, 'c1')
        self._compare_scalar_columns(self._data_scalar, df, 'c2')
        self._compare_scalar_columns(self._data_scalar, df, 'c3')

