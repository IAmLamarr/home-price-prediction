import pathlib

import dvc.api as dvc
import onnxruntime as rt
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder


def preprocess_data(df):
    le = LabelEncoder()
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = le.fit_transform(df[col].astype(str))

    imputer = SimpleImputer(strategy="mean")
    df = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
    return df


def get_dataframe(path: pathlib.Path) -> pd.DataFrame:
    csv_content = dvc.read(path.as_posix(), remote="origin", mode="r")
    if not path.exists():
        file = open(path, "w")
        file.write(csv_content)
        file.close()
    df = pd.read_csv(path)
    return df


def get_onnx_session(path: pathlib.Path):
    onnx_data = dvc.read(path.as_posix(), remote="origin", mode="rb")
    if not path.exists():
        file = open(path, "wb")
        file.write(onnx_data)
        file.close()
    return rt.InferenceSession(path, providers=["CPUExecutionProvider"])
