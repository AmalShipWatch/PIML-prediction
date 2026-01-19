# src/features/build_features.py
import pandas as pd
from src.utils.physics import calculate_physics_baseline
from src import config

def feature_engineering(df: pd.DataFrame, MCR) -> pd.DataFrame:
    load_pct = df["load"].values
    df["power"] = (load_pct / 100) * MCR  # Convert load % to power
    return df

def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create feature matrix X and target vector y."""
    X = pd.DataFrame()
    
    # Raw features mapped from config
    speed = df[config.COL_SPEED]
    
    X["speed"] = speed
    X["speed_sq"] = speed ** 2
    X["speed_cu"] = speed ** 3
    X["draft"] = df[config.COL_DRAFT]
    X["laden_ballast"] = df[config.COL_LADEN]
    
    # Physics Baseline Feature
    X["phys_base"] = calculate_physics_baseline(speed.values, draft=df[config.COL_DRAFT].values)
    
    return X