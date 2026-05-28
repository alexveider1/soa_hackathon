""" 
Mappings for classification models 
"""

ALLOWED_MODELS = ("catboost", "simple_regression", "xgboost")

CATBOOST_CONFIG = {
    "iterations": 500,
    "depth": 6,
    "learning_rate": 0.05,
    "loss_function": "RMSE",
    "verbose": False,
}
XGBOOST_CONFIG = {
    "n_estimators": 500,
    "max_depth": 6,
    "learning_rate": 0.05,
    "objective": "reg:squarederror",
}
REGRESSION_CONFIG = {}

MODEL_CONFIG = {
    "catboost" : CATBOOST_CONFIG,
    "simple_regression" : REGRESSION_CONFIG, 
    "xgboost" : XGBOOST_CONFIG
}