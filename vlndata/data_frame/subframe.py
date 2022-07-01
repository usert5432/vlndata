from typing import Any, List
import numpy as np

from .data_frame_base import DataFrameBase

class SubFrame(DataFrameBase):
    """Data Frame decorator that select a subset of rows"""

    def __init__(self, df : DataFrameBase, indices : np.ndarray):
        super().__init__(df.dtype)
        self._df      = df
        self._indices = indices

    def columns(self) -> List[str]:
        return self._df.columns()

    def get_scalar(self, column : str, index : int) -> Any:
        return self._df.get_scalar(column, self._indices[index])

    def get_vlarr(self, column : str, index : int) -> List[Any]:
        return self._df.get_vlarr(column, self._indices[index])

    def __len__(self):
        return len(self._indices)

    def __getitem__(self, column : str) -> np.ndarray:
        return self._df[column][self._indices]

