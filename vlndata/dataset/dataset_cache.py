from copy import deepcopy
from typing import List, Optional

from vlndata.data_frame import DataFrameBase
from .dataset_base      import DatasetBase, ColumnGroups, VLDataDict

class DatasetCache(DatasetBase):
    """A wrapper around any `DatasetBase` that caches __getitem__ values"""

    def __init__(self, dset : DatasetBase):
        self._dset  = dset
        self._cache : List[Optional[VLDataDict]] \
            = [ None for _ in range(len(dset)) ]

    @property
    def dtype(self):
        return self._dset.dtype

    @property
    def df(self) -> DataFrameBase:
        return self._dset.df

    @property
    def scalar_groups(self) -> ColumnGroups:
        return self._dset.scalar_groups

    @property
    def vlarr_groups(self) -> ColumnGroups:
        return self._dset.vlarr_groups

    def __len__(self):
        return len(self._dset)

    def __getitem__(self, index : int) -> VLDataDict:
        if self._cache[index] is None:
            self._cache[index] = self._dset[index]

        return deepcopy(self._cache[index]) # type: ignore

