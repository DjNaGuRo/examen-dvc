from sklearn.preprocessing import StandardScaler
import pandas as pd
from time import time
import os
from dotenv import load_dotenv

load_dotenv()
PROCESSED_DATA_FOLDER = os.getenv("PROCESSED_DATA_FOLDER")

def scale_data(processed_data_path):
    print(f"Processed data folder: {processed_data_path}")
    X_train = pd.read_csv(f"{processed_data_path}/X_train.csv")
    X_test = pd.read_csv(f"{processed_data_path}/X_test.csv")
    column_names = X_test.columns
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    pd.DataFrame(data=X_train_scaled, columns=column_names).to_csv(f"{processed_data_path}/X_train_scaled.csv")
    pd.DataFrame(data=X_test_scaled, columns=column_names).to_csv(f"{processed_data_path}/X_test_scaled.csv")


if __name__ == "__main__":
    print("Start data normalization ...")
    start = time()
    scale_data(PROCESSED_DATA_FOLDER)
    end = time()
    print("Data normalization finished and last {}s".format(end - start))
