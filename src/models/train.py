from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import pickle
import time
import os
from dotenv import load_dotenv

load_dotenv()
PROCESSED_DATA_FOLDER = os.getenv("PROCESSED_DATA_FOLDER")
MODEL_FOLDER = os.getenv("MODEL_FOLDER")

def train_model():
    X_train_scaled = pd.read_csv(f"{PROCESSED_DATA_FOLDER}/X_train_scaled.csv")
    y_train = pd.read_csv(f"{PROCESSED_DATA_FOLDER}/y_train.csv")

    # Load the model hyperparameters
    params = pickle.load(f"{MODEL_FOLDER}/rf_params.pkl")
    rf = RandomForestRegressor(**params)
    rf.fit(X_train_scaled, y_train)
    params_filepath = f"{MODEL_FOLDER}/rf_regressor.pkl"
    with open(params_filepath, "w") as file:
        pickle.dump(rf, file)


if __name__ == "__main__":
    start = time.time()
    print("Starting model training ...")
    train_model()
    end = time.time()
    print("Model training ended (After {}s)".format(end - start))
