# configs/config.py
from pathlib import Path
from typing import Optional, Dict, Any

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_PATH = DATA_DIR / "raw" / "ship_data.csv" # Place your real data here
PROCESSED_DATA_PATH = DATA_DIR / "processed" / "ship_data_processed.csv"
MODEL_DIR = BASE_DIR / "models" / "trained"

# Column Mapping (Change these to match your real CSV headers)
COL_SPEED = "avg_speed"           # Your CSV column for speed
COL_DRAFT = "mean_draft"           # Your CSV column for draft
COL_POWER = "power"       # Your CSV column for observed power (target)
COL_LADEN = "is_laden"   # Your CSV column for condition (1=Laden, 0=Ballast)

FILTER_LIMITS: Dict[str, Any] = {
    "steam_time": [16, 26],
    "static_outlier_detection": True,
}

# Physical Constants
MIN_POWER_THRESHOLD = 0.0

# Physics Baseline Equation Coefficients
# Equation: X1*x^3 + X2*x^2 + X3*x + X4*y + X5 + X6
# where x = speed, y = draft
PHYSICS_COEFFICIENTS = {
    'X1': 4.2371,
    'X2': -40.4614,
    'X3': 100.0,
    'X4': 11.8543,
    'X5': 0.1542,
    'X6': -4.0772
}

# Model Hyperparameters
RANDOM_SEED = 42
N_ESTIMATORS = 300
MAX_DEPTH = 3
LEARNING_RATE = 0.05
REGRESSOR_TYPE = "xgboost"  # Options: "xgboost", "sklearn_gbr"

# Physics Constraints Weights
BOUNDARY_LOSS_WEIGHT = 10.0
MONOTONICITY_LOSS_WEIGHT = 5.0
DIVERGENCE_LOSS_WEIGHT = 2.0
SEPARATION_LOSS_WEIGHT = 8.0  # Penalize when laden/ballast curves are too close
DRAFT_DEPENDENCY_WEIGHT = 3.0  # Penalize samples not respecting draft coefficient
MINIMUM_LADEN_BALLAST_GAP = 500.0  # Minimum power difference between laden and ballast (kW)

# XGBoost Monotonicity Constraints
ENABLE_MONOTONIC_CONSTRAINTS = True  # Enable monotonic XGBoost
MONOTONIC_FEATURE_INDICES = [0, 1, 2]  # Speed and speed-squared should be monotonic increasing

ITERATIONS = 5