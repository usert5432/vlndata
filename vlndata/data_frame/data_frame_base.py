from abc import ABC, abstractmethod
from typing import Any, List

import numpy as np

class DataFrameBase(ABC):
    """Base Class for vlndata Data Frames

    Data Frames allows users to load data that is arranged in to a
    two-dimentional tabular format. Each value in the table can be either
    a simple scalar value or a variable length array.

    To address each value the user need to provide a column name and a row
    index.

    Data Frames are immutable.

    Parameters
    ----------
    dtype
        Numpy compatible data type of the returned data.
    """

    def __init__(self, dtype : Any = 'float32'):
        self._dtype = np.dtype(dtype)

    @abstractmethod
    def columns(self) -> List[str]:
        """Get a list of columns of the data frame"""
        raise NotImplementedError

    @abstractmethod
    def get_scalar(self, column : str, index : int) -> Any:
        """Get a scalar value at column `column` and row `index`"""
        raise NotImplementedError

    @abstractmethod
    def get_vlarr(self, column : str, index : int) -> List[Any]:
        """Get a vlarray value at column `column` and row `index`"""
        raise NotImplementedError

    @abstractmethod
    def __len__(self):
        raise NotImplementedError

    @abstractmethod
    def __getitem__(self, column : str) -> np.ndarray:
        """Get raw values from column `column`"""
        raise NotImplementedError

    @property
    def dtype(self):
        return self._dtype

