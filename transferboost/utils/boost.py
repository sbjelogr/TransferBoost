import numpy as np
import warnings

from .utils import assure_numpy_array
from .loss_functions import logloss, _logistic


class TBoost:
    """Base class for Transfer Boosting.

    To be inherited in transferboost.models.xgb or transferboost.models.lgb
    """

    def __init__(self, model_params=None, loss_func="logloss", base_score=None, verbosity=0):
        """Constructor for TBoost.

        Args:
            model_params (dict): contains the model parameters of the boosted model.
            loss_func (str or function): loss function, returns gradient and hessians
                of the loss function. Currently supports only loss_func=='logloss'
            base_score (float): starting probability, None by default.
            verbosity (int): verbosity flag.
        """
        self.model_params = model_params
        if loss_func == "logloss":
            self.f_loss_func = logloss
        else:
            raise NotImplementedError(f"Loss function {loss_func} not supported currently")

        if base_score is not None:
            self._set_base_score(base_score)

        self.verbosity = verbosity

    def _set_base_score(self, base_score):
        """Set the base score.

        Args:
            base_score (float): base score.
        """
        self.base_score = base_score

        # Define the starting vector of leaves for the transferboost
        if not 0 < self.base_score < 1:
            raise ValueError(f"Starting proba must be between 0 and 1. Passed {self.base_score}")
        self.start_odds = np.log(self.base_score / (1 - self.base_score))

    def _fit(self, X_leaves_ixs, y):
        """Recompute the output value of the leaves.

        Args:
            X_leaves_ixs ([type]): [description]
            y ([type]): [description]

        Raises:
            ValueError: [description]
            NotImplementedError: [description]

        Returns:
            tuple (np.array, dict):
                leaves_val_array: numpy array of shape (n_rows, n_trees+1), containing the values for every
                    output of the tree. The first column on the array corresponds to the bias term,
                    and is equalfor every row. The second column corresponds to the first tree...
                leaf_vals_map: dictionary, contains the mapping for every tree leaf with the recomputed
                    leaf value.
        """
        X_leaves_ixs = assure_numpy_array(X_leaves_ixs)
        y = assure_numpy_array(y, assure_1d=True)
        # X = assure_numpy_array(X)

        n_rows = X_leaves_ixs.shape[0]

        # TODO: Do i need this? Probably yes
        # assert X.shape[0] == X_leaves_ixs.shape[0]

        # define the first array (starting point for the leaf outputs calculations)
        self.leaves_val_array_ = self.start_odds * np.ones(shape=(n_rows, 1))

        self.n_trees_ = X_leaves_ixs.shape[1]

        self.leaf_vals_map_ = {}
        # Loop over all the trees.
        for tree_index in range(self.n_trees_):
            # Define the prediction of the trees up to tree_index-1 (sum along axis 1)
            prev_proba = self.compute_probability(self.leaves_val_array_, tree_index=tree_index)

            # compute the gradient and hessian by using the predictions from the previous tree.
            g, h = self.f_loss_func(prev_proba, y)

            leaf_vals_ix, leaf_vals_map_ix = self._calculate_leaf_values(
                leaves_ixs=X_leaves_ixs, grad=g, hess=h, tree_index=tree_index, model_params=self.model_params
            )
            self.leaves_val_array_ = np.hstack([self.leaves_val_array_, leaf_vals_ix.reshape(-1, 1)])

            self.leaf_vals_map_[tree_index] = leaf_vals_map_ix

        return self

    def _predict_proba(self, X_leaves_ixs, tree_index=-1):

        X_leaves_ixs = assure_numpy_array(X_leaves_ixs)

        assert self.n_trees_ == X_leaves_ixs.shape[1]

        leaves_val_array = self._apply_leaf_map(X_leaves_ixs)

        if tree_index < 1:
            tree_index = self.n_trees_

        probs = self.compute_probability(leaves_val_array, tree_index=tree_index).reshape(-1, 1)

        return np.hstack([1 - probs, probs])

    def _apply_leaf_map(self, X_leaves_ixs):

        leaves_val_array = self.start_odds * np.ones(shape=(X_leaves_ixs.shape[0], 1))

        for tree_index in range(self.n_trees_):
            leaves = X_leaves_ixs[:, tree_index]

            # leaf_vals_ix = np.array([self.leaf_vals_map_[tree_index][ix] for ix in leaves])
            leaf_vals_ix = np.array([self._get_leaf_value(tree_index, ix) for ix in leaves])
            leaves_val_array = np.hstack([leaves_val_array, leaf_vals_ix.reshape(-1, 1)])

        return leaves_val_array

    def _get_leaf_value(self, tree_ix, leaf_ix):
        """It can happen that when fitting tboost, a specific leaf_ix is never found in the sample.

        For those cases, we should keep the leaf output to be 0
        Args:
            tree_ix: tree index
            leaf_ix: leaf index

        Returns: output of the leaf if leaf_index exists, otherwise 0
        """
        try:
            return self.leaf_vals_map_[tree_ix][leaf_ix]
        except KeyError:
            if self.verbosity > 0:
                warnings.warn(f"Never seen a leaf {leaf_ix} in {tree_ix}, assigning default value of 0")
            return 0

    @staticmethod
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

        leaf_vals_map = {}
        # Find the leaves indices for the tree ix
        leaves = leaves_ixs[:, tree_index]

        # Calculate the leaf values as per xgboost paper
        # eq.(5) in https://arxiv.org/pdf/1603.02754.pdf

        for leave_ix in np.unique(leaves):
            leaf_vals_map[leave_ix] = (
                learning_rate * grad[leaves == leave_ix].sum() / (hess[leaves == leave_ix].sum() + reg_lambda)
            )

        # Map every single leaf index to the new value
        leaf_values = np.array([leaf_vals_map[ix] for ix in leaves])
        return leaf_values, leaf_vals_map

    @staticmethod
    def compute_probability(leaves_val_array, tree_index=None):
        """Compute the probability given the outputs of the leaves.

        Args:
            leaves_val_array (np.array): two dimensional numpy array. expected shape( n_rows x n_trees+1).
                Contains the output of the leaves of the model. The first column must correspond to the
                "bias" term.
                Works with the output of recompute_leaves().
            tree_index (integer): index of the tree where to truncate the calculation of the probabilities.
                tree_index = 0 will return the probability of the base score (defaul 0.5 in Xgboost).
                tree_index = 1 returns the predicted probability of the 1st tree.

        Returns:
            np.array: array of predicted probabilities for class 1 at the tree_index^th tree
        """
        if tree_index is None:
            tree_index = leaves_val_array.shape[0]

        return _logistic(leaves_val_array[:, : (tree_index + 1)].sum(axis=1))
