"""Test correctness of custom csv files parsing with `CSVFrame`"""

import io
import unittest

from vlndata.data_frame.csv_frame import CSVFrame
from .tests_data_frame_base       import TestsDataFrameBase

def create_csv_data_str(data_scalar, data_vlarr):
    def export_vlarr(value):
        return '"[%s]"' % (",".join([str(x) for x in value]))

    columns_scalar = []
    columns_vlarr  = []
    length  = 0

    if (data_scalar is not None) and (len(data_scalar) > 0):
        columns_scalar = list(sorted(data_scalar.keys()))
        length        = max(length, len(data_scalar[columns_scalar[0]]))

    if (data_vlarr is not None) and (len(data_vlarr) > 0):
        columns_vlarr = list(sorted(data_vlarr.keys()))
        length         = max(length, len(data_vlarr[columns_vlarr[0]]))

    result = io.StringIO()
    result.write(",".join(columns_scalar + columns_vlarr) + '\n')

    for i in range(length):
        values = []
        values += [ str(data_scalar[c][i])         for c in columns_scalar ]
        values += [ export_vlarr(data_vlarr[c][i]) for c in columns_vlarr  ]

        result.write(",".join(values) + '\n')

    result.seek(0)
    return result

class TestsCSVFrame(TestsDataFrameBase, unittest.TestCase):

    def _create_data_frame(self, data_scalar = None, data_vlarr = None):
        csv_data = create_csv_data_str(data_scalar, data_vlarr)
        return CSVFrame(csv_data)

if __name__ == '__main__':
    unittest.main()

