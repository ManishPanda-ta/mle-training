import argparse
from utils.logging_utils import logging, setup_logging

import joblib  # Add this import to save the imputer
import numpy as np
import pandas as pd
from python_package.training import (
    perform_grid_search,
    prepare_data,
    train_decision_tree,
    train_linear_regression,
    train_random_forest,
)


def train_model(input_path, output_path):
    logging.info("Starting model training...")
    # Load the training data
    strat_train_set = pd.read_csv(f"{input_path}/strat_train_set.csv")

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

    # Save the best model and the imputer
    best_model = grid_search.best_estimator_
    joblib.dump(best_model, f"{output_path}/best_model.pkl")
    joblib.dump(imputer, f"{output_path}/imputer.pkl")  # Save the imputer
    logging.info("Model training completed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the model")
    parser.add_argument("input_path", type=str, help="Input dataset path")
    parser.add_argument(
        "output_path", type=str, help="Output folder for model pickles"
    )
    parser.add_argument(
        "--log-level", type=str, default="INFO", help="Set the logging level"
    )
    parser.add_argument("--log-path", type=str, help="Path to log file")
    parser.add_argument(
        "--no-console-log",
        action="store_true",
        help="Disable console logging",
    )
    args = parser.parse_args()
    # log_level = getattr(logging, args.log_level.upper(), logging.INFO)
    # logging.basicConfig(
    #     level=log_level,
    #     format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    # )

    # if args.log_path:
    #     file_handler = logging.FileHandler(args.log_path)
    #     file_handler.setLevel(log_level)
    #     file_handler.setFormatter(
    #         logging.Formatter(
    #             "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    #         )
    #     )
    #     logging.getLogger().addHandler(file_handler)

    # if args.no_console_log:
    #     logging.getLogger().handlers = [
    #         h
    #         for h in logging.getLogger().handlers
    #         if not isinstance(h, logging.StreamHandler)
    #     ]
    setup_logging(args.log_level, args.log_path, args.no_console_log)
    train_model(args.input_path, args.output_path)
    print("Train Script Ran Successfully")
