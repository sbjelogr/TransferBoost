import numpy as np

# from .loss_functions import logloss, loss_from_leaves


def _calculate_leaf_values(leaves_ixs, grad, hess, tree_index, model_params):
    """Calculate the output of every leaf of the tree in position tree_index.

    Args:
        leaves_ixs (np.array): np.array, of the shape (n_rows, n_trees).
            Every entry corresponds to the index of the leaf of the model.
            This is returned by model.predict(X, return_leaf=True) for lgb, or model.apply(X)
            for xgboost models.
        grad (np.array): gradient array (for every row) computed from the previous predictions and target y.
        hess (np.array): hessian array (for every row) computed from the previous predictions and target y.
        tree_index (int): index of the decision tree
        model_params (dict): [description]
    """
    reg_lambda = model_params["reg_lambda"]
    learning_rate = model_params["learning_rate"]

    leaf_vals = {}
    # Find the leaves indices for the tree ix
    leaves = leaves_ixs[:, tree_index]

    # Calculate the leaf values as per xgboost paper
    # eq.(5) in https://arxiv.org/pdf/1603.02754.pdf

    for leave_ix in np.unique(leaves):
        leaf_vals[leave_ix] = (
            -learning_rate * grad[leaves == leave_ix].sum() / (hess[leaves == leave_ix].sum() + reg_lambda)
        )

    # Map every single leaf index to the new value
    leaf_values = np.array([leaf_vals[ix] for ix in leaves])
    return leaf_values
