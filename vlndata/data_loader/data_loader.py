import math
from typing import Any, Dict

import numpy as np

from vlndata.dataset import DatasetBase
from .funcs import vldata_dict_collate

class DataLoader:
    """A default vlarr data loader that implements pytorch-like interface

    This class extracts samples from a dataset and packs them into
    batches of fixed size numpy tensors.

    Parameters
    ----------
    dataset : DatasetBase
        Dataset to extract samples from.
    batch_size : int
        Batch size.
    shuffle : bool, optional
        Whether to shuffle dataset before the data extraction.
        Default: True.
    pad : Any, optional
        Value to pad lengths of vl arrays.
        Default: 0.
    seed : int, optional
        Value to seed shuffle prg.
        Default: 0.
    """

    def __init__(
        self,
        dataset    : DatasetBase,
        batch_size : int,
        shuffle    : bool = True,
        pad        : Any  = 0,
        seed       : int  = 0,
    ):
        self._batch_size = batch_size
        self._dataset    = dataset
        self._rng        = np.random.default_rng(seed)
        self._pad        = pad
        self._shuffle    = shuffle
        self._indices    = np.arange(len(dataset))
        self._index      = 0

    @property
    def batch_size(self) -> int:
        return self._batch_size

    @property
    def dataset(self) -> DatasetBase:
        return self._dataset

    def __len__(self):
        return math.ceil(len(self._dataset) / self._batch_size)

    def __iter__(self):
        if self._shuffle:
            self._rng.shuffle(self._indices)

        self._index = 0
        return self

    def __next__(self):
        if self._index >= len(self):
            raise StopIteration

        result = self[self._index]
        self._index += 1

        return result

    def __getitem__(self, index) -> Dict[str, np.ndarray]:
        batch_start_idx = index * self._batch_size
        batch_end_idx   = (index + 1) * self._batch_size
        batch_indices   = self._indices[batch_start_idx:batch_end_idx]

        batch = [ self._dataset[i] for i in batch_indices ]

        return vldata_dict_collate(batch, self._pad)

