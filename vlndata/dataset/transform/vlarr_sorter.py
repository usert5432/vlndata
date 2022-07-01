from typing import Optional
import numpy as np

from .transform import Transform, VLDataDict
from .funcs     import shuffle_vlarr, sort_vlarr

class VLArrShuffleTransform(Transform):
    """Transform that shuffles order of vlarr items"""

    def __init__(self, vlarr_group : str, seed : int = 0):
        super().__init__()

        self._group = vlarr_group
        self._prg   = np.random.default_rng(seed)

    def _reset_parent(self):
        pass

    def __call__(self, data : VLDataDict, _index : int) -> VLDataDict:
        shuffle_vlarr(data[self._group], self._prg)
        return data

class VLArrSortTransform(Transform):
    """Transform that sorts the order of vlarr items"""

    def __init__(
        self, vlarr_group : str, column : str, ascending : bool = True
    ):
        super().__init__()

        self._asc     = ascending
        self._group   = vlarr_group
        self._column  = column
        self._col_idx : Optional[int] = None

    def _reset_parent(self):
        vlarr_group   = self._parent.vlarr_groups[self._group]
        self._col_idx = vlarr_group.index(self._column)

        assert self._col_idx >= 0

    def __call__(self, data : VLDataDict, _index : int) -> VLDataDict:
        if self._col_idx is None:
            raise RuntimeError(
                "VLArr sort transform cannot be used before its parent has"
                " been set"
            )

        data[self._group] = sort_vlarr(
            data[self._group], self._col_idx, self._asc
        )
        return data

