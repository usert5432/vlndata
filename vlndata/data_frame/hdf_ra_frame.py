from collections import namedtuple
from typing import Any, Dict, List

import h5py
import numpy as np

from .data_frame_base import DataFrameBase

Chunk = namedtuple('Chunk', [ 'start_idx', 'end_idx', 'data' ])

class HDF5ReadAheadFrame(DataFrameBase):
    """Data Frame that reads data from an HDF5 file in chunks

    This data frame is analogous to `HDF5Frame`, except that it implements a
    runtime optimization and reads an HDF5 file in contiguous chunks of fixed
    size. The read chunks are cached per each data column. If the subsequent
    calls access data that is cached in a chunk, then no new file reads are
    performed and the data is retrieved from the cache.

    Caching data chunks conveys a significant speedup, provided that the access
    pattern to data is **sequential** (i.e. value are read from row 1 to N). If
    the access pattern is random, then this frame is no better than
    `HDF5Frame`.

    Please refer to the `HDF5Frame` doc strings for the file format details.

    Parameters
    ----------
    path : str
        Input HDF5 file path.
    chunk_size : int, optional
        Number of contiguous rows to read for each column. Default 1024.
    """

    def __init__(
        self, path : str, dtype : Any = 'float32', chunk_size : int = 1024
    ):

        super().__init__(dtype)

        self._path    = path
        self._len     = 0
        self._file    = h5py.File(path, 'r')
        self._columns = list(self._file.keys())

        self._chunk_size = chunk_size
        self._chunks : Dict[str, Chunk] = { }

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

    def read_chunk(self, column : str, index : int) -> Chunk:
        if column in self._chunks:
            chunk = self._chunks[column]

            # pylint: disable=chained-comparison
            if (index >= chunk.start_idx) and (index < chunk.end_idx):
                return chunk

        start_idx = (index // self._chunk_size) * self._chunk_size
        end_idx   = min(start_idx + self._chunk_size, len(self))

        chunk = Chunk(
            data      = self._file[column][start_idx:end_idx],
            start_idx = start_idx,
            end_idx   = end_idx
        )

        self._chunks[column] = chunk
        return chunk

    def get_scalar(self, column : str, index : int) -> Any:
        chunk = self.read_chunk(column, index)
        return chunk.data[index - chunk.start_idx].astype(self._dtype)

    def get_vlarr(self, column : str, index : int) -> np.ndarray:
        chunk = self.read_chunk(column, index)
        return chunk.data[index - chunk.start_idx].astype(self._dtype)

    def __getitem__(self, column):
        return self._file[column]

