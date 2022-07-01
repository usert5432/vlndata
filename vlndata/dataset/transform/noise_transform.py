from typing import Dict, List, Tuple, Optional, Union

import numpy as np

from vlndata.funcs import Spec
from .transform import Transform, VLDataDict
from .noise import select_noise

SimpleNoiseColumns   = List[str]
WeightedNoiseColumns = Dict[str, float]

NoiseColumns   = Union[SimpleNoiseColumns, WeightedNoiseColumns]
NoiseGroupSpec = Dict[str, NoiseColumns]

class NoiseTransform(Transform):
    """A transformation that adds noise to the data.

    Parameters
    ----------
    noise : Spec
        A noise specification. Please refer to the `select_noise` function for
        the details.
    correlated : bool, optional
        Whether to apply a correlated noise to the data.

        If this parameter is True, then for each __getitem__ call a single
        random value will be sampled and all data values (specified by
        `scalar_groups` and `vlarr_groups`) will be modified by this single
        random value.

        If `correlated` is False, then for each data value (as specified by
        `scalar_groups` and `vlarr_groups`) a separate noise value will be
        sampled.

        Default: False.
    relative : bool, optional
        If `relative` is True, then the noise will be applied multiplicatively.
        Otherwise, the noise is additive.

        That is, say noise is eta, input value is x, then the output y will be
        y = x + eta when `relative` is False.  If `relative` is True, then
        y = (1 + eta) * y.

        Default: False.
    scalar_groups : NoiseGroupSpec
        A specification of scalar columns where noise should be applied.
        A noise specification is dictionary where keys are the names of
        `VLDataset` groups where the noise should be applied. The dictionary
        values are either of type `SimpleNoiseColumns` or
        `WeightedNoiseColumns`.

        If the dict values are of the type `SimpleNoiseColumns`, then these
        values specify names of columns where the noise should be applied.

        If the dict values are of the type `WeightedNoiseColumns`, then these
        values are the dictionaries themselves. Their keys specify column
        names where the noise should be applied and the corresponding values
        are the scales by which the noise should be multiplied.

        Default None.
    vlarr_groups  : NoiseGroupSpec
        A specification of vlarr columns where noise should be applied.
        C.f. `scalar_groups` for the details.  Default None.
    """

    def __init__(
        self,
        noise         : Spec,
        correlated    : bool = False,
        relative      : bool = False,
        scalar_groups : Optional[NoiseGroupSpec] = None,
        vlarr_groups  : Optional[NoiseGroupSpec] = None,
    ):
        super().__init__()

        self._noise = select_noise(noise)
        self._corr     = correlated
        self._relative = relative

        self._index_map  : Dict[str, np.ndarray] = {}
        self._weight_map : Dict[str, np.ndarray] = {}

        self._noise_scalar_groups = scalar_groups or {}
        self._noise_vlarr_groups  = vlarr_groups  or {}

    @staticmethod
    def construct_index_weight_maps(
        noise_columns : NoiseColumns, dataset_columns : List[str]
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Construct index map and weight map for the noise.

        An index map is an np.array that specifies index of positions in a
        single group of VLDataDict __getitem__ output, where the noise should
        be applied.

        A weight map is similar to an index map and it is an array of weights
        by which each noise value should be multiplied before application.
        """
        length = len(noise_columns)

        if isinstance(noise_columns, dict):
            # noise_columns : WeightedNoiseColumns
            columns    = list(noise_columns.keys())
            weight_map = np.fromiter(
                (noise_columns[c] for c in columns),
                count = length, dtype = np.float32
            )
        else:
            # noise_columns : SimpleNoiseColumns
            columns    = noise_columns
            weight_map = None

        index_map = np.fromiter(
            (dataset_columns.index(c) for c in columns),
            count = length, dtype = np.int32
        )

        return (index_map, weight_map)

    def apply_noise(
        self, group : str, data : np.array, noise : np.array
    ) -> None:
        # data  : (L, C) or (C, )
        # noise : (L, C) or (C, ) or (1, )

        index_map  = self._index_map[group]
        weight_map = self._weight_map[group]

        if weight_map is None:
            weight_map = 1

        if self._relative:
            data[..., index_map] *= (1 + weight_map * noise)
        else:
            data[..., index_map] += weight_map * noise

    def apply_correlated_noise(self, data : VLDataDict) -> None:
        noise = self._noise.generate(shape = (1,))

        for (name, nparray) in data.items():
            if name not in self._index_map:
                continue

            self.apply_noise(name, nparray, noise)

    def apply_uncorrelated_noise(self, data : VLDataDict) -> None:
        for (name, nparray) in data.items():
            if name not in self._index_map:
                continue

            shape = nparray.shape[:-1] + self._index_map[name].shape
            noise = self._noise.generate(shape = shape)

            self.apply_noise(name, nparray, noise)

    def _reset_parent(self):
        for (name, noise_columns) in self._noise_scalar_groups.items():
            index_map, weight_map = \
                NoiseTransform.construct_index_weight_maps(
                    noise_columns, self._parent.scalar_groups[name]
                )
            self._index_map[name]  = index_map
            self._weight_map[name] = weight_map

        for (name, noise_columns) in self._noise_vlarr_groups.items():
            index_map, weight_map = \
                NoiseTransform.construct_index_weight_maps(
                    noise_columns, self._parent.vlarr_groups[name]
                )

            self._index_map[name]  = index_map
            self._weight_map[name] = weight_map

    def __call__(self, data : VLDataDict, _index : int) -> VLDataDict:
        if self._corr:
            self.apply_correlated_noise(data)
        else:
            self.apply_uncorrelated_noise(data)

        return data

