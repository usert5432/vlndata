from typing import Dict, List, Optional, Tuple, Union

from vlndata.consts     import SPLIT_TRAIN, SPLIT_VAL, SPLIT_TEST
from vlndata.data_frame import DataFrameBase, construct_data_frame
from vlndata.funcs      import Spec, unpack_name_args

from .dataset_base      import DatasetBase, ColumnGroups, VLDataDict
from .dataset_cache     import DatasetCache
from .dataset_transform import DatasetTransform
from .vldataset         import VLDataset
from .transform         import construct_transforms, Transform

SPLIT_INDEX = {
    SPLIT_TRAIN : 0,
    SPLIT_VAL   : 1,
    SPLIT_TEST  : 2,
}

SplitDataFrame = Tuple[DataFrameBase, DataFrameBase, DataFrameBase]

def construct_dataset_from_data_frame(
    df              : Union[DataFrameBase, SplitDataFrame],
    cache           : bool = False,
    split           : str  = SPLIT_TRAIN,
    scalar_groups   : Optional[ColumnGroups]   = None,
    vlarr_groups    : Optional[ColumnGroups]   = None,
    vlarr_limits    : Optional[Dict[str, int]] = None,
    transform_train : Optional[List[Union[Spec, Transform]]] = None,
    transform_test  : Optional[List[Union[Spec, Transform]]] = None,
) -> DatasetBase:

    if isinstance(df, (tuple, list)):
        df = df[SPLIT_INDEX[split]]

    result : DatasetBase \
        = VLDataset(df, scalar_groups, vlarr_groups, vlarr_limits)

    if cache:
        result = DatasetCache(result)

    if split == SPLIT_TRAIN:
        transforms = construct_transforms(transform_train)
    else:
        transforms = construct_transforms(transform_test)

    if transforms is not None:
        result = DatasetTransform(result, transforms)

    return result

def construct_dataset(
    frame           : Spec,
    cache           : bool = False,
    shuffle         : bool = False,
    split           : str  = SPLIT_TRAIN,
    seed            : int  = 0,
    scalar_groups   : Optional[ColumnGroups]      = None,
    vlarr_groups    : Optional[ColumnGroups]      = None,
    vlarr_limits    : Optional[Dict[str, int]]    = None,
    val_size        : Optional[Union[int, float]] = None,
    test_size       : Optional[Union[int, float]] = None,
    extra_vars      : Optional[List[Spec]]        = None,
    transform_train : Optional[List[Union[Spec, Transform]]] = None,
    transform_test  : Optional[List[Union[Spec, Transform]]] = None,
) -> DatasetBase:
    df = construct_data_frame(
        frame, shuffle, val_size, test_size, extra_vars, seed
    )

    return construct_dataset_from_data_frame(
        df, cache, split, scalar_groups, vlarr_groups, vlarr_limits,
        transform_train, transform_test
    )

