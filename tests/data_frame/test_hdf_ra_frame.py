"""Test correctness of hdf files parsing with `HDFReadAheadFrame`"""

import io
import os
import unittest

import h5py
import numpy as np

from vlndata.data_frame.hdf_ra_frame import HDF5ReadAheadFrame
from .tests_data_frame_base          import TestsDataFrameBase
from .test_hdf_frame                 import create_hdf_data_bytes

class TestsHDF5ReadAheadFrame1(TestsDataFrameBase, unittest.TestCase):

    def _create_data_frame(self, data_scalar = None, data_vlarr = None):
        hdf_data = create_hdf_data_bytes(data_scalar, data_vlarr)
        return HDF5ReadAheadFrame(hdf_data, chunk_size = 1)

class TestsHDF5ReadAheadFrame2(TestsDataFrameBase, unittest.TestCase):

    def _create_data_frame(self, data_scalar = None, data_vlarr = None):
        hdf_data = create_hdf_data_bytes(data_scalar, data_vlarr)
        return HDF5ReadAheadFrame(hdf_data, chunk_size = 2)

class TestsHDF5ReadAheadFrame3(TestsDataFrameBase, unittest.TestCase):

    def _create_data_frame(self, data_scalar = None, data_vlarr = None):
        hdf_data = create_hdf_data_bytes(data_scalar, data_vlarr)
        return HDF5ReadAheadFrame(hdf_data, chunk_size = 3)

if __name__ == '__main__':
    unittest.main()

