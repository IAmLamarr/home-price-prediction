import pathlib

import numpy as np
import pandas as pd
from skl2onnx import to_onnx
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import StandardScaler

from utils import preprocess_data


def main():
    train_df = pd.read_csv(pathlib.Path("data") / "train.csv")
    train_data = preprocess_data(train_df)

    train_labels = train_data["SalePrice"]
    train_features = train_data.drop("SalePrice", axis=1)

    train_labels = np.log(train_labels)
    scaler = StandardScaler()
    train_features = scaler.fit_transform(train_features)

    onnx_scaler = to_onnx(scaler, train_features)
    with open(pathlib.Path("data") / "scaler.onnx", "wb") as f:
        f.write(onnx_scaler.SerializeToString())

    param_distributions_rf = {
        "n_estimators": [100, 200, 300, 500],
        "max_depth": [3, 4, 5, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
    }

    rf_model = RandomForestRegressor()

    random_search_rf = RandomizedSearchCV(
        estimator=rf_model,
        param_distributions=param_distributions_rf,
        n_iter=75,
        scoring="neg_mean_squared_error",
        cv=5,
        verbose=1,
        random_state=42,
        n_jobs=-1,
    )

    random_search_rf.fit(train_features, train_labels)

    print("Best Parameters:", random_search_rf.best_params_)
    print("Best Score:", -random_search_rf.best_score_)

    best_model = random_search_rf.best_estimator_

    onx = to_onnx(best_model, train_features.astype(np.float32))
    with open(pathlib.Path("data") / "model.onnx", "wb") as f:
        f.write(onx.SerializeToString())


if __name__ == "__main__":
    main()
