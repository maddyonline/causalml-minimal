from abc import ABCMeta, abstractclassmethod
import logging
import numpy as np
import pandas as pd



from sklearn.metrics import roc_auc_score as auc
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import StratifiedKFold


from causalml.inference.meta.utils import check_p_conditions, convert_pd_to_np

from abc import ABCMeta, abstractmethod

logger = logging.getLogger("causalml")


class PropensityModel(metaclass=ABCMeta):
    def __init__(self, clip_bounds=(1e-3, 1 - 1e-3), **model_kwargs):
        """
        Args:
            clip_bounds (tuple): lower and upper bounds for clipping propensity scores. Bounds should be implemented
                    such that: 0 < lower < upper < 1, to avoid division by zero in BaseRLearner.fit_predict() step.
            model_kwargs: Keyword arguments to be passed to the underlying classification model.
        """
        self.clip_bounds = clip_bounds
        self.model_kwargs = model_kwargs
        self.model = self._model

    @property
    @abstractmethod
    def _model(self):
        pass

    def __repr__(self):
        return self.model.__repr__()

    def fit(self, X, y):
        """
        Fit a propensity model.

        Args:
            X (numpy.ndarray): a feature matrix
            y (numpy.ndarray): a binary target vector
        """
        self.model.fit(X, y)

    def predict(self, X):
        """
        Predict propensity scores.

        Args:
            X (numpy.ndarray): a feature matrix

        Returns:
            (numpy.ndarray): Propensity scores between 0 and 1.
        """
        return np.clip(self.model.predict_proba(X)[:, 1], *self.clip_bounds)

    def fit_predict(self, X, y):
        """
        Fit a propensity model and predict propensity scores.

        Args:
            X (numpy.ndarray): a feature matrix
            y (numpy.ndarray): a binary target vector

        Returns:
            (numpy.ndarray): Propensity scores between 0 and 1.
        """
        self.fit(X, y)
        propensity_scores = self.predict(X)
        logger.info("AUC score: {:.6f}".format(auc(y, propensity_scores)))
        return propensity_scores


class LogisticRegressionPropensityModel(PropensityModel):
    """
    Propensity regression model based on the LogisticRegression algorithm.
    """

    @property
    def _model(self):
        kwargs = {
            "penalty": "elasticnet",
            "solver": "saga",
            "Cs": np.logspace(1e-3, 1 - 1e-3, 4),
            "l1_ratios": np.linspace(1e-3, 1 - 1e-3, 4),
            "cv": StratifiedKFold(
                n_splits=self.model_kwargs.pop("n_fold")
                if "n_fold" in self.model_kwargs
                else 4,
                shuffle=True,
                random_state=self.model_kwargs.get("random_state", 42),
            ),
            "random_state": 42,
        }
        kwargs.update(self.model_kwargs)

        return LogisticRegressionCV(**kwargs)
    

class ElasticNetPropensityModel(LogisticRegressionPropensityModel):
    pass




    

def compute_propensity_score(
    X, treatment, p_model=None, X_pred=None, treatment_pred=None, calibrate_p=True
):
    """Generate propensity score if user didn't provide

    Args:
        X (np.matrix): features for training
        treatment (np.array or pd.Series): a treatment vector for training
        p_model (propensity model object, optional):
            ElasticNetPropensityModel (default) / GradientBoostedPropensityModel
        X_pred (np.matrix, optional): features for prediction
        treatment_pred (np.array or pd.Series, optional): a treatment vector for prediciton
        calibrate_p (bool, optional): whether calibrate the propensity score

    Returns:
        (tuple)
            - p (numpy.ndarray): propensity score
            - p_model (PropensityModel): a trained PropensityModel object
    """
    if treatment_pred is None:
        treatment_pred = treatment.copy()
    if p_model is None:
        p_model = ElasticNetPropensityModel()

    p_model.fit(X, treatment)

    if X_pred is None:
        p = p_model.predict(X)
    else:
        p = p_model.predict(X_pred)

    if calibrate_p:
        logger.info("Ignoring Calibrating propensity scores.")
        # p = calibrate(p, treatment_pred)

    # force the p values within the range
    eps = np.finfo(float).eps
    p = np.where(p < 0 + eps, 0 + eps * 1.001, p)
    p = np.where(p > 1 - eps, 1 - eps * 1.001, p)

    return p, p_model


class BaseLearner(metaclass=ABCMeta):
    @abstractclassmethod
    def fit(self, X, treatment, y, p=None):
        pass

    @abstractclassmethod
    def predict(
        self, X, treatment=None, y=None, p=None, return_components=False, verbose=True
    ):
        pass

    def fit_predict(
        self,
        X,
        treatment,
        y,
        p=None,
        return_ci=False,
        n_bootstraps=1000,
        bootstrap_size=10000,
        return_components=False,
        verbose=True,
    ):
        self.fit(X, treatment, y, p)
        return self.predict(X, treatment, y, p, return_components, verbose)

    @abstractclassmethod
    def estimate_ate(
        self,
        X,
        treatment,
        y,
        p=None,
        bootstrap_ci=False,
        n_bootstraps=1000,
        bootstrap_size=10000,
    ):
        pass

    def bootstrap(self, X, treatment, y, p=None, size=10000):
        """Runs a single bootstrap. Fits on bootstrapped sample, then predicts on whole population."""
        idxs = np.random.choice(np.arange(0, X.shape[0]), size=size)
        X_b = X[idxs]

        if p is not None:
            p_b = {group: _p[idxs] for group, _p in p.items()}
        else:
            p_b = None

        treatment_b = treatment[idxs]
        y_b = y[idxs]
        self.fit(X=X_b, treatment=treatment_b, y=y_b, p=p_b)
        return self.predict(X=X, p=p)

    @staticmethod
    def _format_p(p, t_groups):
        """Format propensity scores into a dictionary of {treatment group: propensity scores}.

        Args:
            p (np.ndarray, pd.Series, or dict): propensity scores
            t_groups (list): treatment group names.

        Returns:
            dict of {treatment group: propensity scores}
        """
        check_p_conditions(p, t_groups)

        if isinstance(p, (np.ndarray, pd.Series)):
            treatment_name = t_groups[0]
            p = {treatment_name: convert_pd_to_np(p)}
        elif isinstance(p, dict):
            p = {
                treatment_name: convert_pd_to_np(_p) for treatment_name, _p in p.items()
            }

        return p

    def _set_propensity_models(self, X, treatment, y):
        """Set self.propensity and self.propensity_models.

        It trains propensity models for all treatment groups, save them in self.propensity_models, and
        save propensity scores in self.propensity in dictionaries with treatment groups as keys.

        It will use self.model_p if available to train propensity models. Otherwise, it will use a default
        PropensityModel (i.e. ElasticNetPropensityModel).

        Args:
            X (np.matrix or np.array or pd.Dataframe): a feature matrix
            treatment (np.array or pd.Series): a treatment vector
            y (np.array or pd.Series): an outcome vector
        """
        logger.info("Generating propensity score")
        p = dict()
        p_model = dict()
        for group in self.t_groups:
            mask = (treatment == group) | (treatment == self.control_name)
            treatment_filt = treatment[mask]
            X_filt = X[mask]
            w_filt = (treatment_filt == group).astype(int)
            w = (treatment == group).astype(int)
            propensity_model = self.model_p if hasattr(self, "model_p") else None
            p[group], p_model[group] = compute_propensity_score(
                X=X_filt,
                treatment=w_filt,
                p_model=propensity_model,
                X_pred=X,
                treatment_pred=w,
            )
        self.propensity_model = p_model
        self.propensity = p


