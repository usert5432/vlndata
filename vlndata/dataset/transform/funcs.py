import numpy as np

def sort_vlarr(
    data : np.ndarray, column_index : int, ascending = False
) -> np.ndarray:
    # data   : (L, C)
    # result : (L, C)

    # indices : (L, )
    if ascending:
        indices = data[:, column_index].argsort()
    else:
        indices = (-data[:, column_index]).argsort()

    return data[indices, :]

def shuffle_vlarr(data : np.ndarray, prg : np.random.Generator) -> None:
    # data   : (L, C)
    # result : (L, C)
    prg.shuffle(data)


