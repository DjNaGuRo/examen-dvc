from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import pickle
import time

DATA_FOLDER = "../../data"
PROCESSED_DATA_FOLDER = f"{DATA_FOLDER}/processed"
MODEL_FOLDER = "../../models"

def search_best_hyperparams(estimator, param_grid, saving_path):
    search = GridSearchCV(estimator=estimator, param_grid=param_grid)
    X_train_scaled = pd.read_csv(f"{PROCESSED_DATA_FOLDER}/X_train_scaled.csv")
    y_train = pd.read_csv(f"{PROCESSED_DATA_FOLDER}/y_train.csv")
    search.fit(X_train_scaled, y_train)
    best_params = search.best_params_
    pickle.dump(best_params, saving_path)


if __name__ == "__main__":
    saving_path = f"{MODEL_FOLDER}/rf_params.pkl"
    estimator = RandomForestRegressor()
    param_grid = {
        "n_estimator": [10, 25, 50, 100, 200],
        "criterion": ["squared_error", "absolute_error", "friedman_mse"],
        "max_depth": [3, 6, None]
    }
    search_best_hyperparams(estimator, param_grid, saving_path)
