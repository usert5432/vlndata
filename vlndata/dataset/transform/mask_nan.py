from typing import Any
import numpy as np

from .transform import Transform, VLDataDict

class MaskNaNTransform(Transform):
    """A tranformation that masks all NaN values"""

    def __init__(self, mask : Any = 0):
        super().__init__()
        self._mask = mask

    def _reset_parent(self):
        pass

    def __call__(self, data : VLDataDict, _index : int) -> VLDataDict:
        for values in data.values():
            values[~np.isfinite(values)] = self._mask

        return data

