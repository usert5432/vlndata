from typing import Any, Dict, List

import h5py
import numpy as np

from .data_frame_base import DataFrameBase

class HDF5Frame(DataFrameBase):
    """Data Frame that reads data from an HDF5 file

    This data frame expects HDF5 to have a specific structure.
    All columns should be stored as a separate dataset in the root of the file,
    i.e. the HDF5 contents are expected to be
    ```
    /column_1
    /column_2
    /column_3
    ...
    /column_k
    ```

    where each column is either a scalar array of the shape (N, 1), where N is
    a number of rows, or a variable length array of the shape (N, None).

    Parameters
    ----------
    path : str
        Input HDF5 file path.

    Warnings
    --------
    This frame relies on the `h5py` library as a backend. The `h5py` library
    has abysmal performance when it comes to reading single values from an HDF5
    file. Please use `HDF5ReadAheadFrame` if the read performance is an issue.
    The `HDF5ReadAheadFrame` reads the input file in chunks reducing the
    numbers of calls to the `h5py` library, and significantly increasing the
    performance.
    """

    def __init__(self, path : str, dtype : Any = 'float32'):
        super().__init__(dtype)

        self._path    = path
        self._len     = 0
        self._file    = h5py.File(path, 'r')
        self._columns = list(self._file.keys())

        if len(self._columns) > 0:
            self._len = len(self._file[self._columns[0]])

    def __getstate__(self) -> Dict[str, Any]:
        return {
            'path' : self._path,
            'cols' : self._columns,
            'len'  : self._len,
        }

    def __setstate__(self, state : Dict[str, Any]):
        self._columns = state['cols']
        self._len     = state['len']
        self._path    = state['path']
        self._file    = h5py.File(self._path, 'r')

    def columns(self) -> List[str]:
        return self._columns

    def __len__(self):
        return self._len

    def get_scalar(self, column : str, index : int) -> Any:
        return self._file[column][index].astype(self._dtype)

    def get_vlarr(self, column : str, index : int) -> np.ndarray:
        return self._file[column][index].astype(self._dtype)

    def __getitem__(self, column):
        return self._file[column]

