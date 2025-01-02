from sklearn.model_selection import train_test_split
from dotenv import load_dotenv
import pandas as pd
from time import time
import os

from utils.read_params import read_params


load_dotenv()
RAW_DATA_FOLDER = os.getenv("RAW_DATA_FOLDER")
PROCESSED_DATA_FOLDER = os.getenv("PROCESSED_DATA_FOLDER")

def split_dataset(input_dataset_filepath, output_data_path):
    print(f"Input data filepath: {input_dataset_filepath}")
    print(f"Output data path: {output_data_path}")
    column_names = ["ave_flot_air_flow", "ave_flot_level", "iron_feed", "starch_flow", "amina_flow", "ore_pulp_flow", "ore_pulp_pH", "ore_pulp_density"]
    df = pd.read_csv(input_dataset_filepath)
    y = df["silica_concentrate"]
    X = df[column_names]
    split_params = read_params()["split"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, **split_params)
    X_train.to_csv(f"{output_data_path}/X_train.csv", index=False)
    X_test.to_csv(f"{output_data_path}/X_test.csv", index=False)
    y_train.to_csv(f"{output_data_path}/y_train.csv", index=False)
    y_test.to_csv(f"{output_data_path}/y_test.csv", index=False)


if __name__ == "__main__":
    input_dataset_filepath = f"{RAW_DATA_FOLDER}/raw.csv"
    print("Starting data split into training and testing sets ...")
    start = time()
    split_dataset(input_dataset_filepath, PROCESSED_DATA_FOLDER)
    end = time()
    print("Data splitting ends (After {}s)".format(end - start))
