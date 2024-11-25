from predictor.test_predictor import (
    LinearRegressionPredictor,
    RidgeRegressionPredictor,
    LassoPredictor,
    RandomForestPredictor,
    XGBoostPredictor,
)

def train_linear_regression(full=True):
    """
    Train Linear Regression models.

    Args:
        full (bool): If True, train models on the full dataset. Otherwise, train on the partial dataset.
    """
    print(f"Training Linear Regression models (full={full})...")
    try:
        predictor = LinearRegressionPredictor("predictor/models/LinearRegression/")
        predictor.train_and_save_model(full=full)
        print("Linear Regression models trained and saved successfully.")
    except Exception as e:
        print(f"Error training Linear Regression models: {e}")


def train_ridge_regression(full=True):
    print(f"Training Ridge Regression models (full={full})...")
    try:
        predictor = RidgeRegressionPredictor("predictor/models/RidgeRegression/")
        predictor.train_and_save_model(full=full)
        print("Ridge Regression models trained and saved successfully.")
    except Exception as e:
        print(f"Error training Ridge Regression models: {e}")


def train_lasso(full= True):
    print(f"Training Lasso models (full={full})...")
    try:
        predictor = LassoPredictor("predictor/models/LassoCV/")
        predictor.train_and_save_model(full=full)
        print("Lasso models trained and saved successfully.")
    except Exception as e:
        print(f"Error training Lasso models: {e}")


def train_random_forest(full=True):
    print(f"Training Random Forest models (full={full})...")
    try:
        predictor = RandomForestPredictor("predictor/models/RandomForest/")
        predictor.train_and_save_model(full=full)
        print("Random Forest models trained and saved successfully.")
    except Exception as e:
        print(f"Error training Random Forest models: {e}")


def train_xgboost(full=True):
    print(f"Training XGBoost models (full = {full})...")
    try:
        predictor = XGBoostPredictor("predictor/models/XGBoost/")
        predictor.train_and_save_model(full= full)
        print("XGBoost models trained and saved successfully.")
    except Exception as e:
        print(f"Error training XGBoost models: {e}")


def train_all_full():
    """
    Train all models on the full dataset.
    """
    print("Starting training for all models on the full dataset...")

    try:
        train_linear_regression(full=True)
    except Exception as e:
        print(f"Error training Linear Regression models: {e}")

    try:
        train_ridge_regression(full=True)
    except Exception as e:
        print(f"Error training Ridge Regression models: {e}")

    try:
        train_lasso(full=True)
    except Exception as e:
        print(f"Error training Lasso models: {e}")

    try:
        train_random_forest(full=True)
    except Exception as e:
        print(f"Error training Random Forest models: {e}")

    try:
        train_xgboost(full=True)
    except Exception as e:
        print(f"Error training XGBoost models: {e}")

    print("Training completed for all models on the full dataset.")

def train_all():
    """
    Train all models with the specified dataset type (full or partial).

    Args:
        full (bool): If True, train on the full dataset. Otherwise, train on the partial dataset.
    """

    try:
        train_linear_regression(full=False)
    except Exception as e:
        print(f"Error training Linear Regression models: {e}")

    try:
        train_ridge_regression(full=False)
    except Exception as e:
        print(f"Error training Ridge Regression models: {e}")

    try:
        train_lasso(full=False)
    except Exception as e:
        print(f"Error training Lasso models: {e}")

    try:
        train_random_forest(full=False)
    except Exception as e:
        print(f"Error training Random Forest models: {e}")

    try:
        train_xgboost(full=False)
    except Exception as e:
        print(f"Error training XGBoost models: {e}")

    print("Training completed for all models on the full dataset.")

