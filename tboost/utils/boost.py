import numpy as np

from .utils import assure_numpy_array
from .loss_functions import logloss, _logistic


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
    # learning_rate = 1

    leaf_vals = {}
    # Find the leaves indices for the tree ix
    leaves = leaves_ixs[:, tree_index]

    # Calculate the leaf values as per xgboost paper
    # eq.(5) in https://arxiv.org/pdf/1603.02754.pdf

    for leave_ix in np.unique(leaves):
        leaf_vals[leave_ix] = (
            learning_rate * grad[leaves == leave_ix].sum() / (hess[leaves == leave_ix].sum() + reg_lambda)
        )

    # Map every single leaf index to the new value
    leaf_values = np.array([leaf_vals[ix] for ix in leaves])
    return leaf_values


def recompute_leaves(leaves_ixs, X, y, start_proba=0.5, loss_func="logloss", model_params=None):
    """Recompute the output value of the leaves.

    Args:
        leaves_ixs ([type]): [description]
        X ([type]): [description]
        y ([type]): [description]
        start_proba (float, optional): [description]. Defaults to 0.5.
        loss_func (str, optional): [description]. Defaults to 'logloss'.
        model_params ([type], optional): [description]. Defaults to None.

    Raises:
        ValueError: [description]
        NotImplementedError: [description]

    Returns:
        [type]: [description]
    """
    y = assure_numpy_array(y, assure_1d=True)
    X = assure_numpy_array(X)
    leaves_ixs = assure_numpy_array(leaves_ixs)

    assert X.shape[0] == leaves_ixs.shape[0]

    learning_rate = model_params["learning_rate"]

    # Define the starting vector of leaves for the t
    if not 0 < start_proba < 1:
        raise ValueError(f"Starting proba must be between 0 and 1. Passed {start_proba}")
    start_odds = np.log(start_proba / (1 - start_proba))
    print(f"start leaf = {start_odds}")

    # define the first array (starting point for the leaf outputs calculations)
    leaves_val_array = start_odds * np.ones(shape=(X.shape[0], 1))

    n_trees = leaves_ixs.shape[1]

    if loss_func == "logloss":
        f_loss_func = logloss
    else:
        raise NotImplementedError(f"Loss function {loss_func} not supported currently")

    # Loop over all the trees.
    for tree_index in range(n_trees):

        # Define the prediction of the trees up to tree_index-1 (sum along axis 1)
        prev_proba = compute_probability(leaves_val_array, learning_rate=learning_rate, tree_index=tree_index)

        # comput the gradient and hessian by using the predictions from the previous tree.
        g, h = f_loss_func(prev_proba, y)

        leaf_vals_ix = _calculate_leaf_values(
            leaves_ixs=leaves_ixs, grad=g, hess=h, tree_index=tree_index, model_params=model_params
        )
        leaves_val_array = np.hstack([leaves_val_array, leaf_vals_ix.reshape(-1, 1)])

    return leaves_val_array


def compute_probability(leaves_val_array, tree_index=None):
    """Compute the probability given the outputs of the leaves.

    Args:
        leaves_val_array (np.array): two dimensional numpy array. expected shape( n_rows x n_trees+1).
            Contains the output of the leaves of the model. The first column must correspond to the
            "bias" term.
            Works with the output of recompute_leaves().
        tree_index (integer): index of the tree where to truncate the calculation of the probabilities.

    Returns:
        np.array: array of predicted probabilities for class 1 at the tree_index^th tree
    """
    if tree_index is None:
        tree_index = leaves_val_array.shape[0]

    # start_val = leaves_val_array[:,0]
    # normal_leaves = leaves_val_array[:,1:(tree_index+1)]
    return _logistic(leaves_val_array[:, : (tree_index + 1)].sum(axis=1))

    # return _logistic(start_val + normal_leaves.sum(axis=1))


# def _recompute_leaves(model, X, y):
#     start_pred = np.ones(shape=X.shape[0])
#     print(start_pred.shape)

#     try:
#         leaves = model.predict(X, pred_leaf=True)
#     except:
#         leaves = model.apply(X)

#     n_trees = leaves.shape[1]
#     print(n_trees)

#     pred = start_pred
#     #     g,h = logloss_from_leaves(pred, y)

#     preds = list()
#     preds.append(pred)

#     pred_array = pred.reshape(-1, 1)

#     for tree_index in range(n_trees):

#         prev_pred = pred_array.sum(axis=1)
#         g, h = logloss_from_leaves(prev_pred, y)
#         pred = return_leaf(leaves, g, h, tree_index=tree_index, model_params=model.get_params())
#         preds.append(pred)

#         pred_array = np.hstack([pred_array, pred.reshape(-1, 1)])
#         print(pred_array.shape)

#     model_leaves = np.transpose(reduce(lambda x, y: np.vstack([x, y]), preds))  # [:,1:]

#     return leaves, model_leaves, pred_array
