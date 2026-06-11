

import os
import glob
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from pipelines.ml_pipeline import FEATURE_COLUMNS, TARGET_COLUMNS

# Paths
RAW_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
PROCESSED_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")


def load_raw_data() -> pd.DataFrame:
   
    csv_files = glob.glob(os.path.join(RAW_DATA_DIR, "gt_*.csv"))
    if not csv_files:
        raise FileNotFoundError(
            f"No CSV files found in {RAW_DATA_DIR}. "
            "Please place CSV files inside data/raw/"
        )

    dfs = []
    for f in sorted(csv_files):
        year = int(os.path.basename(f).split("_")[1].split(".")[0])
        temp = pd.read_csv(f)
        temp["year"] = year
        dfs.append(temp)
        print(f"  Loaded {os.path.basename(f)} — {len(temp):,} rows")

    df = pd.concat(dfs, ignore_index=True)
    print(f"\nTotal rows: {len(df):,} | Columns: {list(df.columns)}")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    
    initial = len(df)
    df = df.drop_duplicates()
    print(f"Removed {initial - len(df)} duplicate rows")

    # Sanity check: filtering out physically impossible values
    df = df[df["TIT"].between(900, 1200)]   # Turbine inlet temp range
    df = df[df["TEY"] > 0]                  # Yield must be positive
    df = df[df["CO"] >= 0]                  # CO cannot be negative
    df = df[df["NOX"] >= 0]                 # NOX cannot be negative

    print(f"Rows after cleaning: {len(df):,}")
    return df.reset_index(drop=True)


def prepare_features(df: pd.DataFrame):
   
    X = df[FEATURE_COLUMNS].copy()
    y = df[TARGET_COLUMNS].copy()
    return X, y


def run_preprocessing(test_size: float = 0.2, random_state: int = 42):
    
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

    df = load_raw_data()
    df = clean_data(df)
    X, y = prepare_features(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    
    X_train.to_csv(os.path.join(PROCESSED_DATA_DIR, "X_train.csv"), index=False)
    X_test.to_csv(os.path.join(PROCESSED_DATA_DIR, "X_test.csv"), index=False)
    y_train.to_csv(os.path.join(PROCESSED_DATA_DIR, "y_train.csv"), index=False)
    y_test.to_csv(os.path.join(PROCESSED_DATA_DIR, "y_test.csv"), index=False)

    print(f"\nTrain: {len(X_train):,} rows | Test: {len(X_test):,} rows")
    print(f"Saved to {PROCESSED_DATA_DIR}")
    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    run_preprocessing()
