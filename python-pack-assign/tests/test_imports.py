# test/test_imports.py

import pytest


def test_import_data_ingestion():
    try:
        from python_package.data_ingestion import (  # noqa: F401
            fetch_housing_data,
            load_housing_data,
        )
    except ImportError:
        pytest.fail("Import failed for data_ingestion module")


def test_import_training():
    try:
        from python_package.training import (  # noqa: F401
            perform_grid_search,
            prepare_data,
            train_decision_tree,
            train_linear_regression,
            train_random_forest,
        )
    except ImportError:
        pytest.fail("Import failed for training module")


def test_import_scoring():
    try:
        from python_package.scoring import (  # noqa: F401
            evaluate_model,
            prepare_test_data,
        )
    except ImportError:
        pytest.fail("Import failed for scoring module")
