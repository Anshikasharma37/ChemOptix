"""
ChemOptix — Model Training Script
Trains the XGBoost MultiOutputRegressor pipeline on Gas Turbine data.
Saves the trained pipeline to models/chemoptix_model.pkl
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score

# Make src importable when run directly
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pipelines.ml_pipeline import build_pipeline, TARGET_COLUMNS
from src.preprocessing import run_preprocessing

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
MODEL_PATH = os.path.join(MODELS_DIR, "chemoptix_model.pkl")


def train():
    """Train the pipeline and save to disk."""
    print("=" * 60)
    print("ChemOptix — Training Pipeline")
    print("=" * 60)

    # Preprocess (loads CSVs, cleans, splits, saves)
    X_train, X_test, y_train, y_test = run_preprocessing()

    print("\nBuilding pipeline...")
    pipeline = build_pipeline()

    print("Training model (this may take ~30–60 seconds)...")
    pipeline.fit(X_train, y_train)
    print("Training complete!")

    # Evaluate
    y_pred = pipeline.predict(X_test)
    y_pred_df = pd.DataFrame(y_pred, columns=TARGET_COLUMNS)
    y_test_reset = y_test.reset_index(drop=True)

    print("\n" + "=" * 60)
    print("TEST SET METRICS")
    print("=" * 60)
    for col in TARGET_COLUMNS:
        mae = mean_absolute_error(y_test_reset[col], y_pred_df[col])
        r2 = r2_score(y_test_reset[col], y_pred_df[col])
        print(f"  {col:5s}  →  MAE: {mae:.4f}  |  R²: {r2:.4f}")

    # Save model
    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    print(f"\nModel saved → {MODEL_PATH}")
    return pipeline


if __name__ == "__main__":
    train()
