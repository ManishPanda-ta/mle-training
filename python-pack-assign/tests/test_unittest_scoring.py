import unittest

import pandas as pd
from python_package.scoring import evaluate_model, prepare_test_data
from python_package.training import prepare_data, train_linear_regression


class TestScoring(unittest.TestCase):
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
        self.housing_prepared, self.housing_labels, self.imputer = (
            prepare_data(self.data)
        )
        self.model, _, _ = train_linear_regression(
            self.housing_prepared, self.housing_labels
        )

    def test_prepare_test_data(self):
        X_test_prepared, y_test = prepare_test_data(self.data, self.imputer)
        self.assertIsNotNone(X_test_prepared)
        self.assertIsNotNone(y_test)

    def test_evaluate_model(self):
        X_test_prepared, y_test = prepare_test_data(self.data, self.imputer)
        rmse, mae = evaluate_model(self.model, X_test_prepared, y_test)
        self.assertGreater(rmse, 0)
        self.assertGreater(mae, 0)


# if __name__ == "__main__":
#     unittest.main()
