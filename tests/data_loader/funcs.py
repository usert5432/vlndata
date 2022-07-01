from typing import Dict
import numpy as np

class TestDataLoaderFuncs:

    def _compare_data(
        self,
        data_test : Dict[str, np.ndarray],
        data_null : Dict[str, np.ndarray]
    ) -> None:

        keys_test = set(data_test.keys())
        keys_null = set(data_null.keys())

        self.assertTrue(keys_test, keys_null)

        for key in keys_null:
            array_test = data_test[key]
            array_null = data_null[key]

            self.assertEqual(array_test.shape, array_null.shape)

            self.assertTrue(
                np.all(np.isclose(array_test, array_null, equal_nan = True))
            )

