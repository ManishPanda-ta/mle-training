import unittest

from python_package.data_ingestion import (
    fetch_housing_data,
    load_housing_data,
)


class TestDataIngestion(unittest.TestCase):
    def setUp(self):
        fetch_housing_data()

    def test_load_housing_data(self):
        data = load_housing_data()
        self.assertIsNotNone(data)
        self.assertGreater(len(data), 0)


# if __name__ == "__main__":
#     unittest.main()
