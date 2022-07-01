import numpy as np
from .subframe import SubFrame, DataFrameBase

class ShuffleFrame(SubFrame):
    """Data Frame decorator that shuffles rows"""

    def __init__(self, df : DataFrameBase, seed : int = 0):
        prg     = np.random.default_rng(seed)
        indices = np.arange(len(df))

        prg.shuffle(indices)

        super().__init__(df, indices)

