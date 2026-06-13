
import os
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any

from pipelines.ml_pipeline import FEATURE_COLUMNS, TARGET_COLUMNS

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "chemoptix_model.pkl")


_pipeline = None


def _load_pipeline():
    global _pipeline
    if _pipeline is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"Model not found at {MODEL_PATH}. "
                
            )
        _pipeline = joblib.load(MODEL_PATH)
    return _pipeline


def predict(input_data: Dict[str, Any]) -> Dict[str, float]:
   
    pipeline = _load_pipeline()

    row = pd.DataFrame([[input_data[col] for col in FEATURE_COLUMNS]], columns=FEATURE_COLUMNS)

    predictions = pipeline.predict(row)[0]  

    return {
        "TEY": round(float(predictions[0]), 4),
        "CO":  round(float(max(predictions[1], 0.0)), 4),  # CO cannot be negative
        "NOX": round(float(max(predictions[2], 0.0)), 4),  # NOX cannot be negative
    }
