from predictor.test_predictor import (
    LinearRegressionPredictor,
    RidgeRegressionPredictor,
    LassoPredictor,
    RandomForestPredictor,
    XGBoostPredictor,
)

def train_linear_regression():
    print("Training Linear Regression models...")
    try:
        predictor = LinearRegressionPredictor("predictor/models/LinearRegression/")
        predictor.train_and_save_model()
        print("Linear Regression models trained and saved successfully.")
    except Exception as e:
        print(f"Error training Linear Regression models: {e}")


def train_ridge_regression():
    print("Training Ridge Regression models...")
    try:
        predictor = RidgeRegressionPredictor("predictor/models/RidgeRegression/")
        predictor.train_and_save_model()
        print("Ridge Regression models trained and saved successfully.")
    except Exception as e:
        print(f"Error training Ridge Regression models: {e}")


def train_lasso():
    print("Training Lasso models...")
    try:
        predictor = LassoPredictor("predictor/models/LassoCV/")
        predictor.train_and_save_model()
        print("Lasso models trained and saved successfully.")
    except Exception as e:
        print(f"Error training Lasso models: {e}")


def train_random_forest():
    print("Training Random Forest models...")
    try:
        predictor = RandomForestPredictor("predictor/models/RandomForest/")
        predictor.train_and_save_model()
        print("Random Forest models trained and saved successfully.")
    except Exception as e:
        print(f"Error training Random Forest models: {e}")


def train_xgboost():
    print("Training XGBoost models...")
    try:
        predictor = XGBoostPredictor("predictor/models/XGBoost/")
        predictor.train_and_save_model()
        print("XGBoost models trained and saved successfully.")
    except Exception as e:
        print(f"Error training XGBoost models: {e}")


if __name__ == "__main__":
    # Train all models
    train_linear_regression()
    train_ridge_regression()
    train_lasso()
    train_random_forest()
    train_xgboost()
