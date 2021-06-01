from transferboost.utils.boost import TBoost
from transferboost.utils import UnsupportedModelError
from sklearn.utils.validation import check_is_fitted


class XGBTransferLearner(TBoost):
    """Main class to transfer boost a trained xgb model.

    Example of use
    ```python
        import transferboost
        from transferboost.dataset import load_data
        from transferboost.models import XGBTransferLearner
        import xgboost as xgb

        # Get X and two targets
        X, y1, y2 = load_data(return_X_y=True)

        xgb_model = xgb.XGBClassifier(
            max_depth = 2,
            n_estimators=100,
        )
        # Fit the xgb model on target 1
        xgb_model.fit(X,y1)

        #Define the transfer learning model
        t_xgb_model = XGBTransferLearner(xgb_model)

        # transfer learn to the target y2
        t_xgb_model.fit(X,y2)

        # predict transfered probabilities
        t_xgb_model.predict_proba(X)
    ```
    """

    def __init__(self, model, base_score=None, f_obj=None, verbosity=0):
        """Constructor for XGBTransferLearner.

        Args:
            model: XGBClassifier model. Must be fitted.
            base_score (float): base score for xgboost model. The default value of None will set the base_score
                to the one of the fitted model.
            f_obj (str of func): objective function. Must return the gradient and hessian of the desired loss
                function. To be passed to the constructor of TBoost.
                Default is None (it will use the standard binary logloss function).
            verbosity (int): verbosity level. If 0 no outputs printed, if >0 returns warnings
        """
        try:
            import xgboost as xgb
        except ImportError as e:
            raise ImportError("No xgboost installed. Please install via pip install xgboost") from e

        if not isinstance(model, xgb.XGBClassifier):
            msg = f"{self.__class__.__name__} does not support the model {model.__class__.__name__}"
            raise UnsupportedModelError(msg)
        check_is_fitted(model)
        self.model = model

        model_params = model.get_params()

        if base_score is None:
            base_score = model_params["base_score"]

        if f_obj is None:
            f_obj = "logloss"

        super().__init__(model_params=model_params, loss_func=f_obj, base_score=base_score, verbosity=verbosity)

    def __repr__(self):
        """Object representation for class XGBTransferLearner."""
        repr_ = f"XGBTransferLearner with base model\n\t{self.model}\n"

        if self.base_score is not None:
            repr_ += f"base_score = {self.base_score}"

        return repr_

    def fit(self, X, y):
        """Fit the XGBTransferLearner.

        Args:
            X: feature set
            y: target (to transfer learn it too)
        """
        X_leaves_ix = self.model.apply(X)
        return self._fit(X_leaves_ix, y)

    def predict_proba(self, X, tree_index=-1):
        """Predict the probabilities after transfer learning.

        Args:
            X: features
            tree_index: index of the tree up to which to predict the

        Returns (np.array): array of shape (X.shape[0],2). Transfer-learned predictions
            for class 0 and class 1.
        """
        X_leaves_ixs = self.model.apply(X)

        probas = self._predict_proba(X_leaves_ixs=X_leaves_ixs, tree_index=tree_index)

        return probas
