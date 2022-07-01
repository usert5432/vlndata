from typing import Any, Dict, Callable, List
import numpy as np

from .data_frame_base import DataFrameBase

VarFunc = Callable[[DataFrameBase,], np.ndarray]

class VarFrame(DataFrameBase):
    """Decorator to augment original Data Frame with new columns

    This Data Frame is a decorator around any `DataFrameBase` and it allows
    users to inject additional columns to the frame. The additional columns
    to inject are specified by the `variables` parameter, which is a map
    from a new column name to a function of signature `VarFunc` that will
    construct the corresponding values for each row.

    Parameters
    ----------
    df : DataFrameBase
        The original data frame to inject new columns to.
    variables : Dict[str, VarFunc]
        A map between a new column name and a function that will evaluate
        the corresponding values. This function receives the original
        data frame `df` as input and should return an numpy array of shape
        (N,) where N = len(df).
    lazy : bool, optional
        If lazy is False, then the values of new columns will be evaluated
        during the construction of `VarFrame`. Otherwise, the new values
        will be evaluated during the first use. Default: False.
    """

    def __init__(
        self,
        df        : DataFrameBase,
        variables : Dict[str, VarFunc],
        lazy      : bool = False
    ):
        super().__init__(df.dtype)
        self._df   = df
        self._lazy = lazy

        variables       = variables or {}
        self._var_specs = variables

        self._vars : Dict[str, np.ndarray] = {}
        self._columns = self._df.columns() + list(sorted(variables.keys()))

        if not lazy:
            for vname in self._var_specs:
                self.eval_var(vname)

    def eval_var(self, name):
        if name in self._vars:
            return self._vars[name]

        result = self._var_specs[name](self._df)
        self._vars[name] = result

        return result

    def columns(self) -> List[str]:
        return self._columns

    def get_scalar(self, column : str, index : int) -> Any:
        if column in self._var_specs:
            return self.eval_var(column)[index]

        return self._df.get_scalar(column, index)

    def get_vlarr(self, column : str, index : int) -> List[Any]:
        if column in self._var_specs:
            return self.eval_var(column)[index]

        return self._df.get_vlarr(column, index)

    def __len__(self):
        return len(self._df)

    def __getitem__(self, column : str) -> np.ndarray:
        if column in self._var_specs:
            return self.eval_var(column)

        return self._df[column]

