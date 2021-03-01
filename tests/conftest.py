import pandas as pd
import numpy as np
import pytest


@pytest.fixture()
def df():
    """Test data frame for unit tests.

    Returns:
        pd.DataFrame: test data
    """
    data = {
        "f1": [100.0, 1.0, 32.24, 30.34, 21.67, 18.13, 39.8, 1.85, 27.53, 11.23],
        "f2": [1.0, 15.98, 17.78, 18.86, 36.75, 12.79, 100.0, 76.07, 53.54, 47.09],
        "f3": [82.1, 1.0, 72.77, 40.7, 14.61, 78.51, 97.11, 100.0, 61.48, 73.41],
        "y": [0, 0, 1, 0, 0, 1, 0, 0, 1, 1],
    }

    return pd.DataFrame(data)


@pytest.fixture()
def X_y(df):
    """Test data as features and targets."""
    X = df[["f1", "f2", "f3"]]
    y = df["y"]

    return X, y


@pytest.fixture()
def leaves_indexes():
    """Mocked indexes of the leaf trees in an xgboost model."""
    return np.array(
        [
            [2, 2, 2, 2],
            [1, 1, 1, 1],
            [2, 2, 2, 2],
            [2, 2, 2, 2],
            [1, 2, 1, 1],
            [1, 1, 1, 1],
            [2, 2, 2, 2],
            [1, 1, 1, 1],
            [1, 2, 2, 2],
            [1, 1, 1, 1],
        ]
    )


@pytest.fixture()
def leaves_values():
    """Mocked leaf values in an xgboost model."""
    return np.array(
        [
            [0.0, -0.3, -0.1426, -0.0072, -0.0051],
            [0.0, 0.0, 0.0, -0.1116, -0.0785],
            [0.0, -0.3, -0.1426, -0.0072, -0.0051],
            [0.0, -0.3, -0.1426, -0.0072, -0.0051],
            [0.0, 0.0, -0.1426, -0.1116, -0.0785],
            [0.0, 0.0, 0.0, -0.1116, -0.0785],
            [0.0, -0.3, -0.1426, -0.0072, -0.0051],
            [0.0, 0.0, 0.0, -0.1116, -0.0785],
            [0.0, 0.0, -0.1426, -0.0072, -0.0051],
            [0.0, 0.0, 0.0, -0.1116, -0.0785],
        ]
    )


@pytest.fixture
def model_params():
    """Xgboost params."""
    return {"learning_rate": 0.3, "reg_lambda": 0}


@pytest.fixture
def pred_proba():
    """Mocked probabilities for every tree in xgboost model."""
    return np.array(
        [
            [0.4255575, 0.39113286, 0.38941234, 0.3882076],
            [0.5, 0.5, 0.47213534, 0.4526166],
            [0.4255575, 0.39113286, 0.38941234, 0.3882076],
            [0.4255575, 0.39113286, 0.38941234, 0.3882076],
            [0.5, 0.46442208, 0.43680802, 0.4175943],
            [0.5, 0.5, 0.47213534, 0.4526166],
            [0.4255575, 0.39113286, 0.38941234, 0.3882076],
            [0.5, 0.5, 0.47213534, 0.4526166],
            [0.5, 0.46442208, 0.46262413, 0.461364],
            [0.5, 0.5, 0.47213534, 0.4526166],
        ]
    )
