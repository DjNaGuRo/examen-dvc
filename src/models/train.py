from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import pickle
import time
import os
from dotenv import load_dotenv
import numpy as np

load_dotenv()
PROCESSED_DATA_FOLDER = os.getenv("PROCESSED_DATA_FOLDER")
MODEL_FOLDER = os.getenv("MODEL_FOLDER")

def train_model():
    X_train_scaled = pd.read_csv(f"{PROCESSED_DATA_FOLDER}/X_train_scaled.csv")
    y_train = pd.read_csv(f"{PROCESSED_DATA_FOLDER}/y_train.csv")
    y_train = np.ravel(y_train)

    # Load the model hyperparameters
    with open(f"{MODEL_FOLDER}/rf_params.pkl", "rb") as f:
        params = pickle.load(f)
    rf = RandomForestRegressor(**params)
    rf.fit(X_train_scaled, y_train)
    model_filepath = f"{MODEL_FOLDER}/rf_regressor.pkl"

    # Save the model trained
    with open(model_filepath, "wb") as file:
        pickle.dump(rf, file)


if __name__ == "__main__":
    start = time.time()
    print("Starting model training ...")
    train_model()
    end = time.time()
    print("Model training ended (After {}s)".format(end - start))

