from typing import Dict
import numpy as np

from vlndata.dataset.dataset_base import DatasetBase

class TestDatasetFuncs:

    def _compare_data(
        self, dset : DatasetBase, dataset_null : Dict[str, np.ndarray]
    ) -> None:

        null_keys = set(dataset_null.keys())

        for idx in range(len(dset)):
            test_data_dict = dset[idx]
            self.assertEqual(set(test_data_dict.keys()), null_keys)

            for key in null_keys:
                test_data = test_data_dict[key]
                null_data = dataset_null[key][idx]

                self.assertEqual(test_data.shape, null_data.shape)
                self.assertTrue(
                    np.all(np.isclose(test_data, null_data, equal_nan = True))
                )

