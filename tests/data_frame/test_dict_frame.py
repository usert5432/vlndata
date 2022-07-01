"""Test DictFrame"""

import unittest

from vlndata.data_frame.dict_frame import DictFrame
from .tests_data_frame_base        import TestsDataFrameBase

class TestsDictFrame(TestsDataFrameBase, unittest.TestCase):

    def _create_data_frame(self, data_scalar = None, data_vlarr = None):
        return DictFrame(data_scalar, data_vlarr)

if __name__ == '__main__':
    unittest.main()

