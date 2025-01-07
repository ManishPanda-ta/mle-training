# test/test_functional.py

import pandas as pd
from python_package.data_ingestion import (
    fetch_housing_data,
    load_housing_data,
)
from python_package.scoring import evaluate_model, prepare_test_data
from python_package.training import prepare_data, train_linear_regression
from sklearn.model_selection import train_test_split


def test_data_ingestion():
    """Test that data is successfully ingested and returned as DataFrame."""
    fetch_housing_data()
    housing_data = load_housing_data()
    assert isinstance(
        housing_data, pd.DataFrame
    ), "Data is not in DataFrame format"
    assert not housing_data.empty, "Loaded data is empty"


def test_data_preparation():
    """Test that data is prepared correctly for training."""
    # Simulating stratified splitting
    housing = (
        load_housing_data()
    )  # Assuming load_housing_data loads the housing dataset
    train_set, test_set = train_test_split(
        housing, test_size=0.2, random_state=42
    )
    prepared_data, labels, _ = prepare_data(train_set)
    assert prepared_data.shape[0] == len(
        labels
    ), "Prepared data does not match labels in length"


def test_model_training():
    """Test that the model training works without errors."""
    housing = load_housing_data()  # Load dataset
    train_set, _ = train_test_split(housing, test_size=0.2, random_state=42)
    prepared_data, labels, _ = prepare_data(train_set)

    model_rmse, model_mae, model = train_linear_regression(
        prepared_data, labels
    )
    assert model is not None, "Model training failed"


def test_model_evaluation():
    """Test model evaluation works correctly."""
    housing = load_housing_data()  # Load dataset
    housing.drop("ocean_proximity", axis=1)  # Drop non-numeric column

    # Split into train and test sets
    train_set, test_set = train_test_split(
        housing, test_size=0.2, random_state=42
    )

    # Prepare the data (Assuming prepare_data and other \
    # necessary functions are inside training.py)
    prepared_data, labels, _ = prepare_data(train_set)

    # Train the model (Assuming train_linear_regression is inside training.py)
    model, rmse, mae = train_linear_regression(prepared_data, labels)

    # Evaluate the model using test data
    test_data, test_labels = prepare_test_data(test_set, _)
    rmse, mae = evaluate_model(model, test_data, test_labels)

    # Assert the RMSE and MAE are positive
    assert rmse > 0, "RMSE should be positive"
    assert mae > 0, "MAE should be positive"
