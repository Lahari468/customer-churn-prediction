import os

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
import joblib


def load_and_preprocess(path="data/WA_Fn-UseC_-Telco-Customer-Churn.csv"):
    os.makedirs("models", exist_ok=True)

    df = pd.read_csv(path)

    # clean
    df.replace(" ", np.nan, inplace=True)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df.dropna(inplace=True)

    # target
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    # keep ONLY features that app collects
    df = df[[
        "gender",
        "SeniorCitizen",
        "Partner",
        "Dependents",
        "tenure",
        "MonthlyCharges",
        "TotalCharges",
        "Contract",
        "InternetService",
        "Churn"
    ]]

    # one-hot encode
    df = pd.get_dummies(df, drop_first=True)

    # split X / y
    X = df.drop("Churn", axis=1)
    y = df["Churn"]

    # scale numeric
    numeric_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
    scaler = StandardScaler()
    X[numeric_cols] = scaler.fit_transform(X[numeric_cols])

    # save scaler
    joblib.dump(scaler, "models/scaler.pkl")

    # train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=0.2, random_state=42
    )

    # oversample imbalance
    sm = SMOTE(random_state=42)
    X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

    # save feature list
    joblib.dump(list(X.columns), "models/feature_columns.pkl")

    return X_train_res, X_test, y_train_res, y_test
