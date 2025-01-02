import yaml

def read_params():
    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)
    return params

if __name__ == "__main__":
    params = read_params()
    print(f"Splitting params:\n{params['split']}")
    print(f"RandomForestRegressor hyperparameters: {params['rf_params']}")