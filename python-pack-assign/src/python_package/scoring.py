import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error


def prepare_test_data(strat_test_set, imputer):
    """
    Prepares the test data using the fitted imputer.
    """
    X_test = strat_test_set.drop("median_house_value", axis=1)
    y_test = strat_test_set["median_house_value"].copy()

    # Drop the 'ocean_proximity' column, which is categorical
    X_test_num = X_test.drop("ocean_proximity", axis=1)

    # Use the pre-fitted imputer to transform the test data
    X_test_prepared = imputer.transform(X_test_num)

    # Convert the result into a DataFrame
    X_test_prepared = pd.DataFrame(
        X_test_prepared, columns=X_test_num.columns, index=X_test.index
    )
    X_test_prepared["rooms_per_household"] = (
        X_test_prepared["total_rooms"] / X_test_prepared["households"]
    )
    X_test_prepared["bedrooms_per_room"] = (
        X_test_prepared["total_bedrooms"] / X_test_prepared["total_rooms"]
    )
    X_test_prepared["population_per_household"] = (
        X_test_prepared["population"] / X_test_prepared["households"]
    )

    # One-hot encode the categorical features
    X_test_cat = X_test[["ocean_proximity"]]
    X_test_prepared = X_test_prepared.join(
        pd.get_dummies(X_test_cat, drop_first=True)
    )

    return X_test_prepared, y_test


def evaluate_model(model, X_test_prepared, y_test):
    """
    Evaluate the model by calculating the RMSE.
    """
    final_predictions = model.predict(X_test_prepared)
    final_mse = mean_squared_error(y_test, final_predictions)
    final_rmse = np.sqrt(final_mse)
    final_mae = mean_absolute_error(y_test, final_predictions)
    print(f"Final model RMSE: {final_rmse}, MAE: {final_mae}")
    return final_rmse, final_mae
