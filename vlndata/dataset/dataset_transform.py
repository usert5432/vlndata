from typing import List

from vlndata.data_frame   import DataFrameBase
from .dataset_base        import DatasetBase, ColumnGroups, VLDataDict
from .transform.transform import Transform

class DatasetTransform(DatasetBase):
    """A wrapper around DatasetBase that adds transformations to return values

    This wrapper around any `DatasetBase` transforms values returned by the
    __getitem__ according to the list of transformations provided during
    its construction.

    Parameters
    ----------
    dset : DatasetBase
        A base dataset.
    transforms : List[Transform]
        A list of transformations to perform over the __getitem__ values.
        Each transformation is an functor of type `Transform` that implements
        a __call__ function, which is responsible for the data transformation.

    Notes
    -----
    The transformations can be performed in-place. Please keep this in mind
    if you want to mix the `DatasetTransform` with caches.
    """

    def __init__(
        self, dset : DatasetBase, transforms : List[Transform]
    ):
        self._dset       = dset
        self._transforms = transforms

        for transform in self._transforms:
            transform.set_parent(self)

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
        result = self._dset[index]

        for transform in self._transforms:
            result = transform(result, index)

        return result

