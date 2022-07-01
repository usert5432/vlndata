from typing import Dict, Optional, List, Tuple, Union

import numpy as np
from vlndata.funcs import Spec, unpack_name_args

from .csv_frame       import CSVFrame
from .csv_mem_frame   import CSVMemFrame
from .dict_frame      import DictFrame
from .data_frame_base import DataFrameBase
from .hdf_frame       import HDF5Frame
from .hdf_ra_frame    import HDF5ReadAheadFrame
from .shuffle_frame   import ShuffleFrame
from .subframe        import SubFrame
from .var_frame       import VarFrame, VarFunc

FRAMES_DICT = {
    'csv-frame'     : CSVFrame,
    'csv-mem-frame' : CSVMemFrame,
    'dict-frame'    : DictFrame,
    'hdf-frame'     : HDF5Frame,
    'hdf-ra-frame'  : HDF5ReadAheadFrame,
}

def select_frame(data_frame : Spec) -> DataFrameBase:
    """Select data frame based on its specification

    Please refer to the `FRAMES_DICT` for the names of the supported
    data frames.

    Parameters
    ----------
    data_frame : Spec
        A specification of the data frame. It should be either a str specifying
        a name of the data frame, or a dict of the form
        { 'name' : NAME, **kwargs }, where NAME is a data frame name and
        **kwargs are the keyword arguments to be passed to the data frame
        constructor.

    Returns
    -------
    DataFrameBase
        The constructed data frame
    """
    name, args = unpack_name_args(data_frame)
    return FRAMES_DICT[name](**args)

def train_test_split(
    frame     : DataFrameBase,
    val_size  : Optional[Union[int, float]],
    test_size : Optional[Union[int, float]],
) -> Tuple[DataFrameBase, DataFrameBase, DataFrameBase]:
    """Split Data Frame into train/val/test parts

    The split performed contiguously, where the first part of the dataset
    goes to the `train` set, the second part to the `val` set and the last part
    to the `test` set.

    Parameters
    ----------
    val_size : Union[int, float], optional
        Size of the validation set. If None, then it is assumed to be 0.
        If `val_size` is of floating type, then it is treated as a fraction
        of the original dataset. Otherwise, it is treated as an absolute
        number of samples to be assigned to the validation dataset.
    test_size : Union[int, float], optional
        Size of the test set. Follows the same rules as `val_size`.

    Returns
    -------
    (train_frame, val_frame, test_frame)
        Train, validation, and test datasets.
    """

    indices = np.arange(len(frame))

    if isinstance(val_size, float):
        val_size = int(len(frame) * val_size)

    if isinstance(test_size, float):
        test_size = int(len(frame) * test_size)

    val_size   = val_size or 0
    test_size  = test_size or 0
    train_size = max(0, len(frame) - test_size - val_size)

    train_indices = indices[:train_size]
    val_indices   = indices[train_size:train_size+val_size]
    test_indices  = indices[train_size+val_size:]

    return tuple(
        SubFrame(frame, indices) \
            for indices in [ train_indices, val_indices, test_indices ]
    ) # type: ignore

def construct_data_frame(
    data_frame : Spec,
    shuffle    : bool = False,
    val_size   : Optional[Union[int, float]] = None,
    test_size  : Optional[Union[int, float]] = None,
    extra_vars : Optional[Dict[str, VarFunc]] = None,
    seed       : int = 0,
) -> Union[DataFrameBase, Tuple[DataFrameBase, DataFrameBase, DataFrameBase]]:
    """Convenience function to construct a standard DataFrame

    This function constructs a data frame based on a specification
    `data_frame`, optionally shuffles it, splits into train/val/test parts,
    and augments the data frame by additional variables `extra_vars`.

    If at least one of `val_size` and `test_size` is not None, then the
    data frame will be split into train/val/test parts. Otherwise, the
    full data frame will be returned.

    Parameters
    ----------
    data_frame : Spec
        A specification of the data frame.
        Please refer to the documentation of `select_frame` for the details.
    shuffle : bool, optional
        Whether to shuffle rows of the data frame. Default: False.
    val_size : Union[int, float], optional
        Size of the validation dataset.
        Please refer to the `train_test_split` for the details. Default: None.
    test_size : Union[int, float], optional
        Size of the test dataset.
        Please refer to the `train_test_split` for the details. Default: None.
    extra_vars : Dict[str, VarFunc], optional
        Additional columns to be added to the data frame. The additional
        columns are specified as a map { 'column_name' : fn } where fn is
        a function that accepts the original data frame as an input and
        returns an array of values for the 'column_name'.
        C.f. `VarFrame` documentation for the details.
        Default: None
    seed : int, optional
        A seed for shuffle rng. Default: 0.

    Returns
    -------
    full_frame or (train_frame, val_frame, test_frame)
        If both `val_size` and `test_size` are None, then this function
        will return the full data frame.
        Otherwise, the full frame will be split into train/val/test parts
        and the parts will be returned in a tuple.
    """
    result = select_frame(data_frame)

    if extra_vars is not None:
        result = VarFrame(result, extra_vars)

    if shuffle:
        result = ShuffleFrame(result, seed = seed)

    if (test_size is not None) or (val_size is not None):
        return train_test_split(result, val_size, test_size)
    else:
        return result

__all__ = [
    'CSVFrame', 'CSVMemFrame', 'HDF5Frame', 'DictFrame', 'DataFrameBase',
    'SubFrame', 'ShuffleFrame', 'VarFrame', 'construct_data_frame',
    'select_frame'
]

