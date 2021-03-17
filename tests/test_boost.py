import numpy as np
import pytest

from tboost.utils import TBoost


@pytest.fixture
def rec_leaf_values(X_y, leaves_indexes, model_params):
    """Fixture to recompute the leaf values."""
    X, y = X_y
    tb = TBoost(model_params=model_params, base_score=0.5)._fit(leaves_indexes, y)

    remapped_leaves = tb._apply_leaf_map(leaves_indexes)
    return remapped_leaves


def test_recompute_leaves(rec_leaf_values, leaves_values):
    """Test the recalculation of the leaves."""
    np.testing.assert_array_almost_equal(rec_leaf_values, leaves_values, decimal=3)


def test_probability_calc(rec_leaf_values, pred_proba):
    """Test the recalculation of the probabilities."""
    # test that the recomputed probability with no tree index corresponds to the probability
    # of the xgboost model
    new_pred_proba = TBoost.compute_probability(rec_leaf_values, tree_index=None)
    np.testing.assert_array_almost_equal(new_pred_proba, pred_proba[:, 3], decimal=3)

    # Test the probability for every index.
    for tree_index in range(4):
        # compute proba when tree_index=0 corresponds to the bias term
        new_pred_proba = TBoost.compute_probability(rec_leaf_values, tree_index=tree_index + 1)
        np.testing.assert_array_almost_equal(new_pred_proba, pred_proba[:, tree_index], decimal=3)

    bias_term = np.unique(TBoost.compute_probability(rec_leaf_values, tree_index=0))
    np.testing.assert_array_equal(bias_term, 0.5)
