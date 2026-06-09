

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.multioutput import MultiOutputRegressor
from xgboost import XGBRegressor

# Feature columns 
FEATURE_COLUMNS = ["AT", "AP", "AH", "AFDP", "GTEP", "TIT", "TAT", "CDP", "year"]

# Target columns
TARGET_COLUMNS = ["TEY", "CO", "NOX"]


def build_pipeline() -> Pipeline:
   
    base_model = XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        tree_method="hist",   
        verbosity=0,
    )

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", MultiOutputRegressor(base_model, n_jobs=-1)),
    ])

    return pipeline
