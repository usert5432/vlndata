from abc import ABC, abstractmethod
from vlndata.dataset.dataset_base import DatasetBase, VLDataDict

class Transform(ABC):
    """Base class for a dataset transformation"""

    def __init__(self):
        self._parent = None

    def set_parent(self, parent : DatasetBase) -> None:
        self._parent = parent
        self._reset_parent()

    @abstractmethod
    def _reset_parent(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def __call__(self, data : VLDataDict, index : int) -> VLDataDict:
        raise NotImplementedError

