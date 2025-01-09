import unittest

import pandas as pd
from python_package.training import prepare_data, train_linear_regression


class TestTraining(unittest.TestCase):
    def setUp(self):
        self.data = pd.DataFrame(
            {
                "median_house_value": [100000, 200000],
                "total_rooms": [5, 6],
                "households": [1, 1],
                "total_bedrooms": [1, 2],
                "population": [3, 4],
                "ocean_proximity": ["NEAR BAY", "INLAND"],
            }
        )

    def test_prepare_data(self):
        housing_prepared, housing_labels, imputer = prepare_data(self.data)
        self.assertIsNotNone(housing_prepared)
        self.assertIsNotNone(housing_labels)

    def test_train_linear_regression(self):
        housing_prepared, housing_labels, _ = prepare_data(self.data)
        model, rmse, mae = train_linear_regression(
            housing_prepared, housing_labels
        )
        self.assertIsNotNone(model)
        self.assertGreater(rmse, 0)
        self.assertGreater(mae, 0)


# if __name__ == "__main__":
#     unittest.main()
