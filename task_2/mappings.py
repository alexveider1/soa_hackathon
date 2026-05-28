"""
Mappings for classification models
"""

ALLOWED_MODELS = ("catboost", "simple_regression", "xgboost")

CATBOOST_CONFIG = {
    "iterations": 1000,
    "depth": 8,
    "learning_rate": 0.03,
    "loss_function": "RMSE",
    "eval_metric": "RMSE",
    "random_seed": 42,
    "verbose": 100,
    "allow_writing_files": False,
    "early_stopping_rounds": 50,
}
XGBOOST_CONFIG = {
    "n_estimators": 300,
    "max_depth": 6,
    "learning_rate": 0.05,
    "objective": "reg:squarederror",
    "n_jobs": -1,
}
REGRESSION_CONFIG = {}

MODEL_CONFIG = {
    "catboost": CATBOOST_CONFIG,
    "simple_regression": REGRESSION_CONFIG,
    "xgboost": XGBOOST_CONFIG,
}
