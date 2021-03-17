import pytest
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from tboost.models import XGBTransferLearner, LGBTransferLearner
from sklearn.exceptions import NotFittedError


def test_tboost_vs_xgb(X_y) -> None:
    """Test that using bins=1 puts everything into 1 bucket."""
    X, y = X_y

    model = xgb.XGBClassifier(max_depth=2, reg_lambda=0, num_leaves=4, n_estimators=4)

    with pytest.raises(NotFittedError):
        XGBTransferLearner(model)

    model.fit(X, y)
    probas = model.predict_proba(X)

    tbooster = XGBTransferLearner(model)
    tbooster.fit(X, y)

    tboost_probas = tbooster.predict_proba(X)

    np.testing.assert_array_almost_equal(probas, tboost_probas)


def test_tboost_vs_lgb(X_y) -> None:
    """Test that using bins=1 puts everything into 1 bucket."""
    X, y = X_y

    model = lgb.LGBMClassifier(
        max_depth=2,
        reg_lambda=0,
        num_leaves=4,
        n_estimators=4,
        min_child_samples=0,  # default value here does not allow to fit multiple trees.
    )

    with pytest.raises(NotFittedError):
        LGBTransferLearner(model)

    model.fit(X, y)
    probas = model.predict_proba(X)

    tbooster = LGBTransferLearner(model)
    tbooster.fit(X, y)

    tboost_probas = tbooster.predict_proba(X)

    np.testing.assert_array_almost_equal(probas, tboost_probas)
