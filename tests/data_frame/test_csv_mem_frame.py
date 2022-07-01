"""Test correctness of custom csv files parsing with `CSVMemFrame`"""

import io
import unittest

from vlndata.data_frame.csv_mem_frame import CSVMemFrame
from .tests_data_frame_base import TestsDataFrameBase
from .test_csv_frame        import create_csv_data_str

class TestsCSVMemFrame(TestsDataFrameBase, unittest.TestCase):

    def _create_data_frame(self, data_scalar = None, data_vlarr = None):
        csv_data = create_csv_data_str(data_scalar, data_vlarr)
        csv_data = io.BytesIO(csv_data.read().encode('utf8'))

        return CSVMemFrame(csv_data)

if __name__ == '__main__':
    unittest.main()

