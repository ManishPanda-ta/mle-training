import numpy as np
import pandas as pd
from python_package.data_ingestion import (
    fetch_housing_data,
    load_housing_data,
)
from python_package.scoring import evaluate_model, prepare_test_data
from python_package.training import (
    perform_grid_search,
    prepare_data,
    train_decision_tree,
    train_linear_regression,
    train_random_forest,
)
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split

# Fetch and load data
fetch_housing_data()
housing = load_housing_data()

# Split the data for training and testing
train_set, test_set = train_test_split(
    housing, test_size=0.2, random_state=42
)

# Apply stratified split based on income category
housing["income_cat"] = pd.cut(
    housing["median_income"],
    bins=[0.0, 1.5, 3.0, 4.5, 6.0, np.inf],
    labels=[1, 2, 3, 4, 5],
)
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing["income_cat"]):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]

# Print the correlation matrix for the features
housing_num = strat_train_set.select_dtypes(include=[np.number])
corr_matrix = housing_num.corr()
print(corr_matrix["median_house_value"].sort_values(ascending=False))

# Prepare data
housing_prepared, housing_labels, imputer = prepare_data(strat_train_set)

# Train the models
lin_reg, lin_rmse, lin_mae = train_linear_regression(
    housing_prepared, housing_labels
)
print(f"Linear Regression RMSE: {lin_rmse}, MAE: {lin_mae}")


tree_reg, tree_rmse, tree_mae = train_decision_tree(
    housing_prepared, housing_labels
)
print(f"Decision Tree RMSE: {tree_rmse}, MAE: {tree_mae}")

rnd_search = train_random_forest(housing_prepared, housing_labels)
print("Random Forest RMSE (using Randomized Search):")
cvres = rnd_search.cv_results_
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(np.sqrt(-mean_score), params)

grid_search = perform_grid_search(housing_prepared, housing_labels)
print("Random Forest RMSE (using Grid Search):")
cvres = grid_search.cv_results_
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(np.sqrt(-mean_score), params)

# Prepare test data
X_test_prepared, y_test = prepare_test_data(strat_test_set, imputer)

# Evaluate the final model
final_model = (
    grid_search.best_estimator_
)  # Get the best model from grid search
final_rmse, final_mae = evaluate_model(final_model, X_test_prepared, y_test)
