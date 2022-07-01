import itertools
from typing import Any, Dict, List, Optional

import numpy as np
from .data_frame_base import DataFrameBase

class DictFrame(DataFrameBase):
    """Data Frame that extracts data from a python dictionary"""

    def __init__(
        self,
        scalar_data_dict : Optional[Dict[str, Any]] = None,
        vlarr_data_dict  : Optional[Dict[str, Any]] = None,
        dtype : Any = None
    ):
        super().__init__(dtype)

        scalar_data_dict = scalar_data_dict or {}
        vlarr_data_dict  = vlarr_data_dict or {}

        self._data_scalar : Dict[str, Any] = {}
        self._data_vlarr  : Dict[str, Any] = {}
        self._len : Optional[int] = None
        self._columns = (
            list(scalar_data_dict.keys()) + list(vlarr_data_dict.keys())
        )

        self._infer_length(scalar_data_dict, vlarr_data_dict)
        self._prepare_data(scalar_data_dict, vlarr_data_dict)

    def _infer_length(
        self, scalar_data : Dict[str, Any], vlarr_data : Dict[str, Any]
    ):
        for v in itertools.chain(scalar_data.values(), vlarr_data.values()):
            if self._len is None:
                self._len = len(v)

            assert self._len == len(v)

    def _prepare_data(
        self, scalar_data : Dict[str, Any], vlarr_data : Dict[str, Any]
    ):
        for (k, v) in scalar_data.items():
            self._data_scalar[k] = v

        for (k, v) in vlarr_data.items():
            self._data_vlarr[k] = np.array(
                [ np.array(x) for x in v ], dtype = object
            )

    def get_scalar(self, column : str, index : int) -> Any:
        return self._dtype.type(self._data_scalar[column][index])

    def get_vlarr(self, column : str, index : int) -> np.ndarray:
        return self._data_vlarr[column][index].astype(self._dtype)

    def columns(self) -> List[str]:
        return self._columns

    def __len__(self):
        return self._len

    def __getitem__(self, column : str) -> np.ndarray:
        if column in self._data_scalar:
            return np.array(self._data_scalar[column])

        return np.array(self._data_vlarr[column])

