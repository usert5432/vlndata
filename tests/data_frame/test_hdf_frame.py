"""Test correctness of hdf files parsing with `HDFLoader`"""

import io
import os
import unittest

import h5py
import numpy as np

from vlndata.data_frame.hdf_frame import HDF5Frame
from .tests_data_frame_base       import TestsDataFrameBase

def create_hdf_data_bytes(data_scalar, data_vlarr):
    result = io.BytesIO()

    with h5py.File(result, 'w') as f:
        if data_scalar is not None:
            for (k, v) in data_scalar.items():
                f.create_dataset(k, data = v)

        if data_vlarr is not None:
            for (k, v) in data_vlarr.items():
                vtype  = h5py.special_dtype(vlen = np.float32)
                values = np.array([ np.array(x) for x in v ], dtype = object)
                f.create_dataset(k, data = values, dtype = vtype)

    result.seek(0)
    return result

class TestsHDF5Frame(TestsDataFrameBase, unittest.TestCase):

    def _create_data_frame(self, data_scalar = None, data_vlarr = None):
        hdf_data = create_hdf_data_bytes(data_scalar, data_vlarr)
        return HDF5Frame(hdf_data)

if __name__ == '__main__':
    unittest.main()

