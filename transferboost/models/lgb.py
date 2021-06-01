from transferboost.utils.boost import TBoost
from transferboost.utils import UnsupportedModelError, assure_numpy_array
from sklearn.utils.validation import check_is_fitted


class LGBMTransferLearner(TBoost):
    """Main class to transfer boost a trained xgb model.

    Example of use
    ```python
        import transferboost
        from transferboost.dataset import load_data
        from transferboost.models import LGBTransferLearner

        import lightgbm as lgb

        # Get the data and the targets
        X, y1, y2 = load_data(return_X_y=True)

        lgb_model = lgb.LGBMClassifier(
            max_depth = 3,
            reg_lambda = 0,
            num_leaves=5,
            n_estimators=100,
        )

        lgb_model.fit(X,y1)

        t_lgb_model = LGBTransferLearner(lgb_model)
        t_lgb_model.fit(X,y2)
        t_lgb_model.predict_proba(X)
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
            import lightgbm as lgb
        except ImportError as e:
            raise ImportError("No lightgbm installed. Please install via pip install lightgbm") from e

        if not isinstance(model, lgb.LGBMClassifier):
            msg = f"{self.__class__.__name__} does not support the model {model.__class__.__name__}"
            raise UnsupportedModelError(msg)
        check_is_fitted(model)
        self.model = model

        self.base_score = base_score

        model_params = model.get_params()

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
        y = assure_numpy_array(y, assure_1d=True)
        if self.base_score is None:
            self.base_score = y.mean()
            self._set_base_score(base_score=self.base_score)
        X_leaves_ix = self.model.predict(X, pred_leaf=True)
        return self._fit(X_leaves_ix, y)

    def predict_proba(self, X, tree_index=-1):
        """Predict the probabilities after transfer learning.

        Args:
            X: features
            tree_index: index of the tree up to which to predict the

        Returns (np.array): array of shape (X.shape[0],2). Transfer-learned predictions
            for class 0 and class 1.
        """
        X_leaves_ixs = self.model.predict(X, pred_leaf=True)

        probas = self._predict_proba(X_leaves_ixs=X_leaves_ixs, tree_index=tree_index)

        return probas
