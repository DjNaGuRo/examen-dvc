from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from dotenv import load_dotenv
import pandas as pd
import pickle
import time
import os
import numpy as np


from utils.read_params import read_params

load_dotenv()
PROCESSED_DATA_FOLDER = os.getenv("PROCESSED_DATA_FOLDER")
MODEL_FOLDER = os.getenv("MODEL_FOLDER")

def search_best_hyperparams(estimator, param_grid, saving_path):
    search = GridSearchCV(estimator=estimator, param_grid=param_grid, n_jobs=-1)
    X_train_scaled = pd.read_csv(f"{PROCESSED_DATA_FOLDER}/X_train_scaled.csv")
    y_train = pd.read_csv(f"{PROCESSED_DATA_FOLDER}/y_train.csv")
    y_train = np.ravel(y_train)
    # print("Y_train shape: ", y_train.shape)
    search.fit(X_train_scaled, y_train)
    best_params = search.best_params_
    with open(saving_path, "wb") as file:
        pickle.dump(best_params, file)


if __name__ == "__main__":
    saving_path = f"{MODEL_FOLDER}/rf_params.pkl"
    estimator = RandomForestRegressor()
    param_grid = read_params()["rf_params"]
    start = time.time()
    print("Starting model hyperparameters tuning with GridSearchCV ...")
    search_best_hyperparams(estimator, param_grid, saving_path)
    end = time.time()
    print("Model hyperparameters tuning with GridSearchCV ended (After {}s)".format(end - start))
