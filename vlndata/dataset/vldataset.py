from typing import Dict, Optional
import numpy as np

from vlndata.data_frame import DataFrameBase
from .dataset_base import DatasetBase, ColumnGroups, VLDataDict

class VLDataset(DatasetBase):
    """Default implementation of the vlndata dataset.

    This dataset follows the semantics of the pytorch dataset. It takes
    a Data Frame during construction and extracts values from it (
    packed in `VLDataDict`) according to the `scalar_groups` and `vlarr_groups`
    parameters.

    The `scalar_groups` parameter is a dictionary of the form
        { 'group_name' : List['column_name'] }
    where `group_name` specifies the key name in the `VLDataDict` output.
    The list of `column_name` are the scalar columns to be extracted from the
    Data Frame and packed into a single array. C.f. Examples section below.

    Similar to `scalar_groups`, `vlarr_groups` specifies the vl arrays
    to be extracted from the Data Frame.

    Parameters
    ----------
    df : DataFrameBase
        Data Frame to extract values from.
    scalar_groups : ColumnGroups, optional
        A dictionary specifying groups of scalar variables to extract from
        the data frame. Default: None.
    vlarr_groups  : ColumnGroups, optional
        A dictionary specifying groups of vlarr variables to extract from
        the data frame. Default: None.
    vlarr_limits  : Dict[str, int], optional
        A dictionary of vlarr group length limits. If a key from `vlarr_groups`
        if present in `vlarr_limits`, then the lengths of all vlarrays in that
        vlarr group will be limited by the corresponding value from the
        `vlarr_limits`. Default: None.

    Examples
    --------
    Let say one has a data frame, e.g.
    >>> from vlndata.data_frame import DictFrame
    >>> df = DictFrame({ 'col1' : [ 1, 2, 3 ], 'col2' : [ 4, 5, 6 ] })

    A dataset can be constructed from this frame, like
    >>> scalar_groups = { 'grp1' : [ 'col2' ], 'grp2' : [ 'col2', 'col1' ] }
    >>> dset = VLDataset(df, scalar_groups)

    Now, one can call `dset[idx]` to get a dictionary of values, corresponding
    to the `idx` row in the Data Frame:
    >>> dset[0]
    {'grp1': array([4.]), 'grp2': array([4., 1.])}
    >>> dset[1]
    {'grp1': array([5.]), 'grp2': array([5., 2.])}

    """

    def __init__(
        self,
        df            : DataFrameBase,
        scalar_groups : Optional[ColumnGroups] = None,
        vlarr_groups  : Optional[ColumnGroups] = None,
        vlarr_limits  : Optional[Dict[str, int]] = None
    ):
        self._df = df
        self._scalar_groups = scalar_groups or {}
        self._vlarr_groups  = vlarr_groups or {}
        self._vlarr_limits  = vlarr_limits or {}

    @property
    def dtype(self):
        return self._df.dtype

    @property
    def df(self) -> DataFrameBase:
        return self._df

    @property
    def scalar_groups(self) -> ColumnGroups:
        return self._scalar_groups

    @property
    def vlarr_groups(self) -> ColumnGroups:
        return self._vlarr_groups

    def __len__(self):
        return len(self._df)

    def extract_scalar_group(self, name : str, index : int) -> np.ndarray:
        columns = self._scalar_groups[name]

        return np.fromiter(
            ( self._df.get_scalar(column, index) for column in columns ),
            dtype = self._df.dtype,
            count = len(columns)
        )

    def extract_vlarr_group(self, name : str, index : int) -> np.ndarray:
        columns = self._vlarr_groups[name]

        if len(columns) == 0:
            return np.empty((0, 0), dtype = self._df.dtype)

        first_vlarr   = self._df.get_vlarr(columns[0], index)
        ref_vl_length = len(first_vlarr)

        vl_length = self._vlarr_limits.get(name, ref_vl_length)
        vl_length = min(vl_length, ref_vl_length)

        result = np.empty((vl_length, len(columns)), dtype = self._df.dtype)
        result[:, 0] = first_vlarr[:vl_length]

        for column_idx, column in enumerate(columns[1:], start = 1):
            vlarr = self._df.get_vlarr(column, index)
            assert len(vlarr) == ref_vl_length

            result[:, column_idx] = vlarr[:vl_length]

        return result

    def __getitem__(self, index : int) -> VLDataDict:
        result = { }

        for name in self._scalar_groups:
            result[name] = self.extract_scalar_group(name, index)

        for name in self._vlarr_groups:
            result[name] = self.extract_vlarr_group(name, index)

        return result

