import numpy as np
import pytest

from tboost.utils import recompute_leaves, compute_probability


@pytest.fixture
def rec_leaf_values(X_y, leaves_indexes, model_params):
    """Fixture to recompute the leaf values."""
    X, y = X_y
    remapped_leaves, leaf_mapping = recompute_leaves(leaves_indexes, X, y, model_params=model_params)
    return remapped_leaves


def test_recompute_leaves(rec_leaf_values, leaves_values):
    """Test the recalculation of the leaves."""
    np.testing.assert_array_almost_equal(rec_leaf_values, leaves_values, decimal=3)


def test_probability_calc(rec_leaf_values, pred_proba):
    """Test the recalculation of the probabilities."""
    # test that the recomputed probability with no tree index corresponds to the probability
    # of the xgboost model
    new_pred_proba = compute_probability(rec_leaf_values, tree_index=None)
    np.testing.assert_array_almost_equal(new_pred_proba, pred_proba[:, 3], decimal=3)

    # Test the probability for every index.
    for tree_index in range(4):
        # compute proba when tree_index=0 corresponds to the bias term
        new_pred_proba = compute_probability(rec_leaf_values, tree_index=tree_index + 1)
        np.testing.assert_array_almost_equal(new_pred_proba, pred_proba[:, tree_index], decimal=3)

    bias_term = np.unique(compute_probability(rec_leaf_values, tree_index=0))
    np.testing.assert_array_equal(bias_term, 0.5)
