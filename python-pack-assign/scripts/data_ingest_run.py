import argparse

import numpy as np
import pandas as pd
from python_package.data_ingestion import (
    fetch_housing_data,
    load_housing_data,
)
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from utils.logging_utils import logging, setup_logging


def ingest_data(output_path):
    logging.info("Starting data ingestion...")
    # Fetch and load data
    fetch_housing_data(housing_path=output_path)
    housing = load_housing_data(housing_path=output_path)

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
    for train_index, test_index in split.split(
        housing, housing["income_cat"]
    ):
        strat_train_set = housing.loc[train_index]
        strat_test_set = housing.loc[test_index]

    # Save the stratified datasets
    strat_train_set.to_csv(f"{output_path}/strat_train_set.csv", index=False)
    strat_test_set.to_csv(f"{output_path}/strat_test_set.csv", index=False)

    logging.info("Data ingestion completed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Ingest data and create datasets"
    )
    parser.add_argument(
        "output_path", type=str, help="Output folder/file path"
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
    log_level = getattr(logging, args.log_level.upper(), logging.INFO)
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
    ingest_data(args.output_path)
    print("Data Ingest Script Ran Successfully")
