from typing import Any, List, Union

import pandas as pd
import numpy  as np

from .data_frame_base import DataFrameBase

class CSVFrame(DataFrameBase):
    """Data Frame to parse csv files that uses pandas DataFrame as a backed

    The CSV file is expected to have the standard format with the first
    line being the header describing column names.

    The variable length arrays are expected to be serialized as strings as
    either
        "a0,a2,a2,a3,...,aN"
    or
        "[a0,a2,a2,a3,...,aN]"
    where ak -- scalar values.

    Parameters
    ----------
    path : str
        Input CSV file path.

    Warnings
    --------
    Pandas backend is very inefficient memory-wise, especially when str columns
    are present. If you run out of RAM using this CSVFrame, try using
    CSVMemFrame instead, which is more memory efficient, but less performant.
    """

    def __init__(self, path : str, dtype : Any = 'float32'):
        super().__init__(dtype)

        self._df = pd.read_csv(path)

        self._columns = list(self._df.columns)
        self._len     = len(self._df)
        self._path    = path

    def __getstate__(self) -> dict:
        return {
            'cols'  : self._columns,
            'dtype' : self._dtype,
            'len'   : self._len,
            'path'  : self._path,
        }

    def __setstate__(self, state : dict):
        self._columns = state['cols']
        self._dtype   = state['dtype']
        self._len     = state['len']
        self._path    = state['path']
        self._df      = pd.read_csv(self._path)

    def columns(self) -> List[str]:
        return self._columns

    @staticmethod
    def deserialize_vlarr(
        vlarr_str : Union[str, float], dtype : Any = None
    ) -> np.ndarray:
        """Parse vlarray serialized as a string

        The vlarray is expected to be serialized either as:
            "a0,a2,a2,a3,...,aN"
        or
            "[a0,a2,a2,a3,...,aN]"
        """

        if isinstance(vlarr_str, str):
            if vlarr_str.startswith('['):
                assert vlarr_str.endswith(']')
                vlarr_str = vlarr_str[1:-1]

            return np.fromstring(vlarr_str, dtype = dtype, sep = ',')

        if isinstance(vlarr_str, float):
            return np.empty((0,), dtype = dtype)

        raise ValueError(
            f"Unknown how to parse variable length array: '{vlarr_str}'"
        )

    def get_vlarr(
        self,
        column : str,
        index  : int
    ) -> np.ndarray:
        vlarr_str = self._df.loc[index, column]
        return CSVFrame.deserialize_vlarr(vlarr_str, self._dtype)

    def get_scalar(self, column : str, index : int) -> float:
        return self._df.loc[index, column].astype(self._dtype)

    def __getitem__(self, column : str) -> np.ndarray:
        return self._df[column].values

    def __len__(self):
        return self._len

