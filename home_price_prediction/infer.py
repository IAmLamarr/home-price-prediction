import pathlib

import numpy as np
import pandas as pd

from utils import get_dataframe, get_onnx_session, preprocess_data


def infer():
    test_df = get_dataframe(pathlib.Path("../data") / "test.csv")

    test_features = preprocess_data(test_df)

    scaler_sess = get_onnx_session(pathlib.Path("../data") / "scaler.onnx")
    input_name = scaler_sess.get_inputs()[0].name
    label_name = scaler_sess.get_outputs()[0].name
    test_features = scaler_sess.run([label_name], {input_name: test_features.values})[0]

    model_sess = get_onnx_session(pathlib.Path("../data") / "model.onnx")
    input_name = model_sess.get_inputs()[0].name
    label_name = model_sess.get_outputs()[0].name
    pred = model_sess.run([label_name], {input_name: test_features.astype(np.float32)})[
        0
    ]

    true_predictions = np.exp(pred)

    answer_df = pd.DataFrame(
        {
            "Id": test_df["Id"],
            "SalePrice": true_predictions.reshape(1, -1)[0],
        }
    )

    answer_df.to_csv("../data/predictions.csv", index=False)


if __name__ == "__main__":
    infer()
