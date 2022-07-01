from typing import Any, Iterable, List, Tuple, Union
import numpy as np

from vlndata.dataset import VLDataDict

ShapeBatchScalar = Tuple[int, int]
ShapeBatchVLArr  = Tuple[int, int, int]
ShapeUnion       = Union[ShapeBatchScalar, ShapeBatchVLArr]

def scalar_collate(
    it : Iterable[np.ndarray], shape : ShapeBatchScalar, dtype : Any
) -> np.ndarray:
    # result : (N, C)
    result = np.empty(shape, dtype)

    for (idx, array) in enumerate(it):
        # array : (C, )
        result[idx, :] = array

    return result

def vlarr_collate(
    it : Iterable[np.ndarray], shape : ShapeBatchVLArr, dtype : Any,
    pad : Any = 0
) -> np.ndarray:
    # result : (N, L, C)
    result = np.empty(shape, dtype)

    for (idx, array) in enumerate(it):
        # array : (l, C)
        result[idx, :array.shape[0], :] = array
        result[idx, array.shape[0]:, :] = pad

    return result

def infer_shape_dtype(it : Iterable[np.ndarray]) -> Tuple[ShapeUnion, Any]:
    result = None
    dtype  = None
    n = 0

    for array in it:
        shape = array.shape
        dtype = array.dtype
        n += 1

        if result is None:
            result = list(shape)
        else:
            assert len(result) == len(shape)
            assert result[-1]  == shape[-1]

            if len(shape) == 2:
                # vlarr case (L, C)
                # pylint: disable=unsubscriptable-object
                # pylint: disable=unsupported-assignment-operation
                result[0] = max(result[0], shape[0])

    assert result is not None
    assert dtype  is not None

    result = (n, ) + tuple(result)      # type: ignore
    return result, dtype                # type: ignore

def vldata_dict_collate(batch : List[VLDataDict], pad : Any = 0) -> VLDataDict:
    """Collate a list of vl data objects into a single vl data batch"""
    if len(batch) == 0:
        return {}

    keys   = list(batch[0].keys())
    result = {}

    for key in keys:
        shape, dtype = infer_shape_dtype(
            (data_dict[key] for data_dict in batch)
        )

        if len(shape) == 2:
            result[key] = scalar_collate(
                (data_dict[key] for data_dict in batch), shape, dtype
            )
        else:
            result[key] = vlarr_collate(
                (data_dict[key] for data_dict in batch), shape, dtype, pad
            )

    return result

