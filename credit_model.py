from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATA_PATH = "german_credit_data.csv"
RANDOM_STATE = 42

FEATURE_COLUMNS = [
    "Age",
    "Sex",
    "Job",
    "Housing",
    "Saving accounts",
    "Checking account",
    "Credit amount",
    "Duration",
    "Purpose",
]

CATEGORICAL_COLUMNS = [
    "Sex",
    "Housing",
    "Saving accounts",
    "Checking account",
    "Purpose",
]

NUMERICAL_COLUMNS = [
    "Age",
    "Job",
    "Credit amount",
    "Duration",
]


def load_credit_data(path: str = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])
    df["Risk"] = (df["Credit amount"] <= 5000).astype(int)
    return df


def build_pipeline() -> Pipeline:
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERICAL_COLUMNS),
            ("cat", categorical_transformer, CATEGORICAL_COLUMNS),
        ]
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
        ]
    )


def get_feature_options(df: pd.DataFrame) -> dict[str, list[str]]:
    options: dict[str, list[str]] = {}
    for column in CATEGORICAL_COLUMNS:
        options[column] = (
            df[column]
            .fillna("Unknown")
            .astype(str)
            .sort_values()
            .unique()
            .tolist()
        )
    return options


def train_credit_model(path: str = DATA_PATH):
    df = load_credit_data(path)
    X = df[FEATURE_COLUMNS]
    y = df["Risk"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    metrics = {
        "train_shape": X_train.shape,
        "test_shape": X_test.shape,
        "classification_report": classification_report(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
    }

    return pipeline, df, metrics


def main() -> None:
    _, _, metrics = train_credit_model()
    print("Training data shape:", metrics["train_shape"])
    print("Testing data shape:", metrics["test_shape"])
    print("\nClassification Report:\n", metrics["classification_report"])
    print("ROC-AUC Score:", metrics["roc_auc"])
    print("\nConfusion Matrix:\n", metrics["confusion_matrix"])


if __name__ == "__main__":
    main()
