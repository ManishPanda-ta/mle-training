# test/test_integration.py

import pandas as pd
from python_package.data_ingestion import (
    fetch_housing_data,
    load_housing_data,
)
from python_package.scoring import evaluate_model, prepare_test_data
from python_package.training import prepare_data, train_random_forest
from sklearn.model_selection import train_test_split


def test_end_to_end_integration():
    """Test the entire pipeline from data ingestion to model evaluation."""
    # Step 1: Data ingestion
    fetch_housing_data()
    housing_data = load_housing_data()
    assert isinstance(
        housing_data, pd.DataFrame
    ), "Data is not in DataFrame format"

    # Step 2: Split data into train/test
    train_set, test_set = train_test_split(
        housing_data, test_size=0.2, random_state=42
    )

    # Step 3: Prepare data for training
    prepared_data, labels, imputer = prepare_data(train_set)

    # Step 4: Train model
    model = train_random_forest(prepared_data, labels)
    assert model is not None, "Model training failed"

    # Step 5: Prepare test data
    test_data, test_labels = prepare_test_data(test_set, imputer)

    # Step 6: Evaluate model
    rmse, mae = evaluate_model(model, test_data, test_labels)
    assert rmse > 0, "Model evaluation resulted in negative RMSE"
    assert mae > 0, "Model evaluation resulted in negative RMSE"
