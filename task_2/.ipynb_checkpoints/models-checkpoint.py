"""
Main models to use for hit prediction
"""

import numpy as np
import pandas as pd

from catboost import CatBoostRegressor, Pool
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix
from mappings import MODEL_CONFIG, ALLOWED_MODELS

MODEL_FACTORY = {
    "catboost": CatBoostRegressor,
    "simple_regression": LinearRegression,
    "xgboost": XGBRegressor,
}


class HitPrediction:
    """
    Class to predict hits, allows using different models
    """

    def __init__(
        self,
        features: list[str],
        seed: int = 42,
        model_type: str = "catboost",
        model_config: dict = None,
        hit_border: float = 70,
        cat_cardinality_threshold: int = 10,
    ) -> None:

        if model_type not in ALLOWED_MODELS:
            raise ValueError(f"""Model is not in allowed models: {ALLOWED_MODELS}""")

        self.model_type = model_type
        self.features = features
        self.seed = seed
        self.model_config = model_config
        self.cat_cardinality_threshold = cat_cardinality_threshold

        if not self.model_config:
            self.model_config = MODEL_CONFIG[self.model_type].copy()
        if self.model_type == "catboost":
            self.model_config["random_seed"] = self.seed
        if self.model_type == "xgboost":
            self.model_config["random_state"] = self.seed

        self.model = MODEL_FACTORY[self.model_type](**self.model_config)

        self.hit_border = hit_border

        self.cat_features: list[str] = []
        self.num_features: list[str] = []
        self._label_encoders: dict = {}
        self._is_fitted: bool = False

    def _get_cat_num_features(self, X: pd.DataFrame) -> tuple[list[str], list[str]]:
        """
        Separates categorical and numerical features
        """
        cat_by_dtype = X.select_dtypes(include=["object", "category"]).columns.tolist()

        cat_by_cardinality = [
            col
            for col in X.select_dtypes(include=["number"]).columns
            if X[col].nunique() <= self.cat_cardinality_threshold
            and col in self.features
        ]

        cat_cols = set(cat_by_dtype + cat_by_cardinality)
        cat_cols = list(cat_cols)
        num_cols = [col for col in self.features if col not in cat_cols]

        return cat_cols, num_cols

    def _encode_cat_for_sklearn(
        self, X: pd.DataFrame, fit_encoder: bool = False
    ) -> pd.DataFrame:
        """
        Encoding categorical features for LinearRegression and XGBoost
        """
        X = X.copy()
        for col in self.cat_features:
            if fit_encoder:
                encoder = LabelEncoder()
                X[col] = encoder.fit_transform(X[col].astype(str))
                self._label_encoders[col] = encoder
            else:
                encoder = self._label_encoders[col]
                X[col] = X[col].astype(str)

                X[col] = X[col].map(
                    lambda val, le=encoder: (
                        le.transform([val])[0]
                        # for out of train categories
                        if val in le.classes_
                        else -1
                    )
                )
        return X

    def _prepare_features(
        self, X: pd.DataFrame, fit_encoder: bool = False
    ) -> pd.DataFrame:
        """
        Prepares categorical and numerical features
        """
        X = X[self.features].copy()

        if self.model_type != "catboost":
            X = self._encode_cat_for_sklearn(X, fit_encoder=fit_encoder)

        return X

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Fit choosen model"""
        self._is_fitted = True
        self.cat_features, self.num_features = self._get_cat_num_features(X)
        X_prepared = self._prepare_features(X, fit_encoder=True)

        if self.model_type == "catboost":
            pool = Pool(X_prepared, y, cat_features=self.cat_features)
            return self.model.fit(pool)

        return self.model.fit(X_prepared, y)

    def predict_popularity(self, X: pd.DataFrame) -> np.ndarray:
        """Predict treck popularity number"""
        if not self._is_fitted:
            raise RuntimeError("Call fit() before predict()")

        X_prepared = self._prepare_features(X, fit_encoder=False)

        if self.model_type == "catboost":
            pool = Pool(X_prepared, cat_features=self.cat_features)
            return self.model.predict(pool)

        return self.model.predict(X_prepared)

    def predict_hit(self, X: pd.DataFrame) -> np.ndarray:
        """Predict binary popularity (hit or not)"""
        pridicted_scores = self.predict_popularity(X)
        return (pridicted_scores >= self.hit_border).astype(int)

    def compute_f1_score(
        self, X: pd.DataFrame, y_true: pd.Series, average: str = "binary"
    ) -> float:
        """
        F1 считается по бинарным меткам.
        y_true — реальная популярность (0–100), бинаризуем внутри.
        """
        y_pred_binary = self.predict_hit(X)
        y_true_binary = (y_true.values >= self.hit_border).astype(int)
        return f1_score(y_true_binary, y_pred_binary, average=average)

    def compute_roc_auc(self, X: pd.DataFrame, y_true: pd.Series) -> float:
        """
        Computes ROC-AUC using non-binary scores
        for more informative results
        """
        y_scores = self.predict_popularity(X)
        y_true_binary = (y_true.values >= self.hit_border).astype(int)
        return roc_auc_score(y_true_binary, y_scores)

    def compute_confusion_matrix(
        self, X: pd.DataFrame, y_true: pd.Series
    ) -> np.ndarray:
        """
        Computes confusion matrix: [[TN, FP], [FN, TP]]
        """
        y_pred_binary = self.predict_hit(X)
        y_true_binary = (y_true.values >= self.hit_border).astype(int)
        return confusion_matrix(y_true_binary, y_pred_binary)

    def evaluate(self, X: pd.DataFrame, y_true: pd.Series) -> dict:
        """All metrics"""
        return {
            "f1": self.compute_f1_score(X, y_true),
            "roc_auc": self.compute_roc_auc(X, y_true),
            "confusion_matrix": self.compute_confusion_matrix(X, y_true),
        }
