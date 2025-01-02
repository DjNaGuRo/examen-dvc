from sklearn.metrics import root_mean_squared_error, roc_auc_score, r2_score, mean_absolute_error
import pandas as pd
import pickle
import json
import time
import os
from dotenv import load_dotenv

load_dotenv()
PROCESSED_DATA_FOLDER = os.getenv("PROCESSED_DATA_FOLDER")
MODEL_FOLDER = os.getenv("MODEL_FOLDER")
METRICS_FOLDER = os.getenv("METRICS_FOLDER")
PREDICTIONS_FOLDER = os.getenv("PREDICTIONS_FOLDER")

def evaluate_model():
    X_test_scaled = pd.read_csv(f"{PROCESSED_DATA_FOLDER}/X_test_scaled.csv")
    y_test = pd.read_csv(f"{PROCESSED_DATA_FOLDER}/y_test.csv")
    print(f"Y_test:\n{y_test.head()}")
    print(f"Y_test shape:\n{y_test.shape}")

    # Load the model 
    with open(f"{MODEL_FOLDER}/rf_regressor.pkl", "rb") as f:
        rf_regressor = pickle.load(f)
    
    # Predictions
    y_pred = rf_regressor.predict(X_test_scaled)
    print(f"Predictions:\n{y_pred[:5]}")
    print(f"Y_pred shape: {y_pred.shape}")
    pred_filepath = f"{PREDICTIONS_FOLDER}/ped.csv"
    pd.DataFrame({"predictions": y_pred}).to_csv(pred_filepath, index=False)

    # Scores evaluation
    metrics = {
        "r2_score" : r2_score(y_true=y_test, y_pred=y_pred),
        "rmse" : root_mean_squared_error(y_test, y_pred),
        "mae" : mean_absolute_error(y_test, y_pred),
        "roc_auc_score" : roc_auc_score(y_test, y_pred)
    }
    metrics_filepath = f"{METRICS_FOLDER}/scores.json"
    with open(metrics_filepath, "w") as file:
        json.dump(metrics, file)


if __name__ == "__main__":
    start = time.time()
    print("Starting model evaluation ...")
    evaluate_model()
    end = time.time()
    print("Model evaluation ended (After {}s)".format(end - start))