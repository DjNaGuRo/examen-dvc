from sklearn.model_selection import train_test_split
import pandas as pd
from time import time
import os
from dotenv import load_dotenv

load_dotenv()
RAW_DATA_FOLDER = os.getenv("RAW_DATA_FOLDER")
PROCESSED_DATA_FOLDER = os.getenv("PROCESSED_DATA_FOLDER")

def split_dataset(input_dataset_filepath, output_data_path):
    df = pd.read_csv(input_dataset_filepath)
    y = df["silica_concentrate"]
    X = df.drop(columns=["silica_concentrate"])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train.to_csv(f"{output_data_path}/X_tran.csv")
    X_test.to_csv(f"{output_data_path}/X_test.csv")
    y_train.to_csv(f"{output_data_path}/y_train.csv")
    y_test.to_csv(f"{output_data_path}/y_test.csv")


if __name__ == "__main__":
    input_dataset_filepath = f"{RAW_DATA_FOLDER}/raw.csv"
    print("Starting data split into training and testing sets ...")
    start = time()
    split_dataset(input_dataset_filepath, PROCESSED_DATA_FOLDER)
    end = time()
    print("Data splitting ends (After {}s)".format(end - start))