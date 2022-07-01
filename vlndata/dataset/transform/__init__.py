from typing import Optional, List, Union

from vlndata.funcs import Spec, unpack_name_args

from .transform       import Transform
from .mask_nan        import MaskNaNTransform
from .noise_transform import NoiseTransform
from .vlarr_sorter    import VLArrShuffleTransform, VLArrSortTransform

TRANSFORM_DICT = {
    'mask-nan'      : MaskNaNTransform,
    'noise'         : NoiseTransform,
    'vlarr-shuffle' : VLArrShuffleTransform,
    'vlarr-sort'    : VLArrSortTransform,
}

def select_transform(transform : Union[Spec, Transform]) -> Transform:
    if isinstance(transform, Transform):
        return transform

    name, args = unpack_name_args(transform)
    return TRANSFORM_DICT[name](**args)

def construct_transforms(
    transforms : Optional[List[Union[Spec, Transform]]]
) -> Optional[List[Transform]]:

    if transforms is None:
        return None

    return [ select_transform(t) for t in transforms ]

__all__ = [
    'Transform', 'MaskNaNTransform', 'NoiseTransform',
    'VLArrShuffleTransform', 'VLArrSortTransform',
    'select_transform'
]

