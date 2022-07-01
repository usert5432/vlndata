# pylint: disable=no-member
# mistaken lint for shmem

import csv
from collections import namedtuple
from typing import Any, Dict, List

import pandas as pd
import numpy  as np

from .data_frame_base import DataFrameBase
from .csv_frame import CSVFrame
from .funcs import load_file_into_shmem

CachedLine = namedtuple('CachedLine', [ 'index', 'tokens' ])

class CSVMemFrame(DataFrameBase):
    """Data Frame to read a CSV file that caches all data into shared memory

    This Data Frame is a more memory efficient version of the `CSVFrame`, but
    uses more CPU to parse data on demand. Please refer to the `CSVFrame`
    documentation about the CSV data format.

    Parameters
    ----------
    path : str
        Input CSV file path.
    """

    def __init__(self, path : str, dtype : Any = 'float32'):
        super().__init__(dtype)

        self._shmem = load_file_into_shmem(path)

        self._offsets : List[int] = []
        self._columns : List[str] = []
        self._colmap  : Dict[str, int] = {}

        self._infer_csv_columns()
        self._infer_line_offsets()

        self._cached_line = CachedLine(-1, [])

    def __del__(self):
        self._shmem.unlink()

    def _infer_line_offsets(self):
        """Find byte offsets of lines in the csv file"""
        self._offsets = []
        idx = -1

        while True:
            idx = self._shmem.buf.obj.find(b'\n', idx + 1)
            if idx == -1:
                break

            self._offsets.append(idx)

    def _infer_csv_columns(self):
        """Parse header of a csv file and infer columns"""
        self._shmem.buf.obj.seek(0, 0)
        df = pd.read_csv(self._shmem.buf.obj, nrows = 1)

        self._columns = list(df.columns)
        self._colmap  = {
            col : idx for (idx, col) in enumerate(self._columns)
        }

    def columns(self) -> List[str]:
        return self._columns

    def get_value(self, column : str, index : int) -> str:
        """Get raw unparsed str value for column `column` and row `index`"""
        if self._cached_line.index != index:
            idx_start = self._offsets[index] + 1
            idx_end   = self._offsets[index + 1]

            line = self._shmem.buf[idx_start:idx_end].tobytes()
            line = line.decode('utf-8')

            reader = csv.reader([ line, ])
            tokens = next(reader)

            self._cached_line = CachedLine(index, tokens)

        return self._cached_line.tokens[self._colmap[column]]

    def get_vlarr(self, column : str, index  : int) -> np.ndarray:
        vlarr_str = self.get_value(column, index)
        return CSVFrame.deserialize_vlarr(vlarr_str, self._dtype)

    def get_scalar(self, column : str, index : int) -> float:
        return float(self.get_value(column, index))

    def __getitem__(self, column : str) -> np.ndarray:
        self._shmem.buf.obj.seek(0, 0)

        return pd.read_csv(
            self._shmem.buf.obj, usecols = [ column, ], squeeze = True
        )

    def __len__(self):
        return len(self._offsets) - 1

