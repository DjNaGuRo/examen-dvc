from sklearn.preprocessing import StandardScaler
import pandas as pd
from time import time

DATA_FOLDER = "../../data"
PROCESSED_DATA_FOLDER = f"{DATA_FOLDER}/processed"

def scale_data(processed_data_path):
    X_train = pd.read_csv(f"{processed_data_path}/X_train.csv")
    X_test = pd.read_csv(f"{processed_data_path}/X_test.csv")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    X_train_scaled.to_csv(f"{processed_data_path}/X_train_scaled.csv")
    X_test_scaled.to_csv(f"{processed_data_path}/X_test_scaled.csv")


if __name__ == "__main__":
    print("Start data normalization ...")
    start = time()
    scale_data(PROCESSED_DATA_FOLDER)
    end = time()
    print("Data normalization finished and last {}s".format(end - start))
