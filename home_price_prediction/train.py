import pathlib

import mlflow
import numpy as np
from hydra import compose, initialize
from mlflow.models import infer_signature
from skl2onnx import to_onnx
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler

from utils import get_dataframe, preprocess_data


def train():
    initialize(version_base=None, config_path="../configs", job_name="test_app")
    train_cfg = compose(config_name="train")

    hosts_cfg = compose(config_name="hosts")
    mlflow.set_tracking_uri(uri=hosts_cfg["mlflow"])

    train_df = get_dataframe(pathlib.Path("../data") / "train.csv")
    train_data = preprocess_data(train_df)

    train_labels = train_data["SalePrice"]
    train_features = train_data.drop("SalePrice", axis=1)

    train_labels = np.log(train_labels)
    scaler = StandardScaler()
    train_features = scaler.fit_transform(train_features)

    onnx_scaler = to_onnx(scaler, train_features)
    with open(pathlib.Path("../data") / "scaler.onnx", "wb") as f:
        f.write(onnx_scaler.SerializeToString())

    rf_model = RandomForestRegressor(**train_cfg)

    rf_model.fit(train_features, train_labels)

    pred = rf_model.predict(train_features)
    mae = mean_absolute_error(pred, train_labels)

    with open(pathlib.Path("../data") / "metrics.txt", "w") as f:
        f.write(f"MAE: {mae}")

    onx = to_onnx(rf_model, train_features.astype(np.float32))
    with open(pathlib.Path("../data") / "model.onnx", "wb") as f:
        f.write(onx.SerializeToString())

    mlflow.set_experiment("home-price-prediction")

    with mlflow.start_run():
        mlflow.log_params(train_cfg)
        mlflow.log_metric("mae", mae)

        signature = infer_signature(train_labels, pred)
        mlflow.sklearn.log_model(
            sk_model=rf_model,
            artifact_path="random_forest",
            signature=signature,
            input_example=train_df,
            registered_model_name="home-price-prediction-rf",
        )


if __name__ == "__main__":
    train()
