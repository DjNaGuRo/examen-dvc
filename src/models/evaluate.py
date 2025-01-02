from sklearn.metrics import root_mean_squared_error, roc_auc_score, r2_score, mean_absolute_error
import pandas as pd
import pickle
import json
import time

DATA_FOLDER = "../../data"
PROCESSED_DATA_FOLDER = f"{DATA_FOLDER}/processed"
MODEL_FOLDER = "../../models"
METRICS_FOLDER = "../../metrics"

def evaluate_model():
    X_test_scaled = pd.read_csv(f"{PROCESSED_DATA_FOLDER}/X_test_scaled.csv")
    y_test = pd.read_csv(f"{PROCESSED_DATA_FOLDER}/y_test.csv")

    # Load the model 
    rf_regressor = pickle.load(f"{MODEL_FOLDER}/rf_regressor.pkl")
    
    # Predictions
    y_pred = rf_regressor.predict(X_test_scaled)
    pred_filepath = f"{DATA_FOLDER}/predictions/ped.csv"
    pd.DataFrame({"predictions": y_pred}).to_csv(pred_filepath)

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