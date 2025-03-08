import argparse
import logging

import joblib  # Add this import to load the imputer
import pandas as pd
from python_package.scoring import evaluate_model, prepare_test_data


def score_model(model_path, dataset_path, output_path):
    logging.info("Starting model scoring...")
    # Load the test data
    strat_test_set = pd.read_csv(f"{dataset_path}/strat_test_set.csv")

    # Load the best model and the imputer
    best_model = joblib.load(f"{model_path}/best_model.pkl")
    imputer = joblib.load(f"{model_path}/imputer.pkl")  # Load the imputer

    # Prepare test data
    X_test_prepared, y_test = prepare_test_data(strat_test_set, imputer)

    # Evaluate the final model
    final_rmse, final_mae = evaluate_model(
        best_model, X_test_prepared, y_test
    )
    print(f"Final model RMSE: {final_rmse}, MAE: {final_mae}")
    logging.info("Model scoring completed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Score the model")
    parser.add_argument("model_path", type=str, help="Model folder path")
    parser.add_argument("dataset_path", type=str, help="Dataset folder path")
    parser.add_argument("output_path", type=str, help="Output path")
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
    log_level = getattr(logging, args.log_level.upper(), logging.INFO)
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    if args.log_path:
        file_handler = logging.FileHandler(args.log_path)
        file_handler.setLevel(log_level)
        file_handler.setFormatter(
            logging.Formatter(
                "%(asctime)s - %(name)s - %(levellevel)s - %(message)s"
            )
        )
        logging.getLogger().addHandler(file_handler)

    if args.no_console_log:
        logging.getLogger().handlers = [
            h
            for h in logging.getLogger().handlers
            if not isinstance(h, logging.StreamHandler)
        ]
    score_model(args.model_path, args.dataset_path, args.output_path)
    print("Score Script Ran Successfully")
