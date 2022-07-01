from abc import ABC, abstractmethod

from typing import Dict, List
import numpy as np

from vlndata.data_frame import DataFrameBase

ColumnGroups = Dict[str, List[str]]
VLDataDict   = Dict[str, np.ndarray]

class DatasetBase(ABC):
    """Interface for a vlndata dataset

    A generic vlndata dataset follows pytorch dataset semantics.

    Usually, a subclass of `DatasetBase` it is a wrapper around a Data Frame.
    This wrapper dataset takes a row index as an input and returns a dictionary
    of arrays `VLDataDict` extracted from the Data Frame.

    The `VLDataset` class provides the default implementation of the vlndata
    dataset.
    """

    @property
    @abstractmethod
    def dtype(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def df(self) -> DataFrameBase:
        raise NotImplementedError

    @property
    @abstractmethod
    def scalar_groups(self) -> ColumnGroups:
        raise NotImplementedError

    @property
    @abstractmethod
    def vlarr_groups(self) -> ColumnGroups:
        raise NotImplementedError

    @abstractmethod
    def __len__(self):
        """Number of samples in the dataset"""
        raise NotImplementedError

    @abstractmethod
    def __getitem__(self, index : int) -> VLDataDict:
        raise NotImplementedError

