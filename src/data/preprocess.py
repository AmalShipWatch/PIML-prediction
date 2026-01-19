"""
Data preprocessing and cleaning functions
"""
# from loguru import logger
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List
from src.config import FILTER_LIMITS

def filter_outliers_iqr(
    df: pd.DataFrame,
    columns: List[str],
    epochs: int = 50,
    lr: float = 0.001,
    threshold_percentile: float = 95,
    iqr_multiplier: float = 3,
) -> pd.DataFrame:
    """
    Filters out outliers from the DataFrame using statistical methods and an autoencoder.
    """
    if df.empty:
        return df

    initial_len = len(df)

    # Statistical outlier removal using IQR method
    df_clean = df.copy()
    for col in columns:
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - iqr_multiplier * IQR
        upper_bound = Q3 + iqr_multiplier * IQR
        df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]
    
    return df_clean

def apply_basic_filters(df: pd.DataFrame, vessel_specs) -> pd.DataFrame:
    """
    Applies basic validity filters to the noon data.
    """
    initial_len = len(df)

    # Normalize consumption to 24-hour periods
    if "consumption" in df.columns and "steam_time" in df.columns:
        df["consumption"] = df["consumption"] * (24 / df["steam_time"])

    # Check for missing values
    # missing_pct = df.isnull().mean() * 100
    # features_over_threshold = missing_pct[missing_pct > 20].index.tolist()
    # if features_over_threshold:
    #     raise ValueError(f"Features {features_over_threshold} have more than 20% missing values")

    # Basic range checks
    conditions = [
        df["avg_speed"] > 0,
        df["mean_draft"] > 0,
        df["consumption"] > 0,
        df["power"].between(0, vessel_specs),
    ]
    if "steam_time" in df.columns:
        steam_min, steam_max = FILTER_LIMITS["steam_time"]
        conditions.append(df["steam_time"].between(steam_min, steam_max))

    # Combine all conditions
    combined_condition = conditions[0]
    for condition in conditions[1:]:
        combined_condition = combined_condition & condition

    df = df[combined_condition]

    # Apply IQR filtering to each feature
    df = filter_outliers_iqr(
        df,
        [
            "avg_speed",
            "mean_draft",
            "power",
            "consumption",
            "wind_speed",
            "wind_direction",
            "wave_height",
            "wave_direction",
        ],
    )

    print(f"Basic filtering removed {initial_len - len(df)} rows; Remaining rows: {len(df)}")

    return df

def preprocess_data(df: pd.DataFrame, MCR) -> pd.DataFrame:
    """Main preprocessing pipeline."""
    # Set index to time column
    df.index = pd.to_datetime(df["time"])
    df = df.drop(columns=["time"])
    # Clean data
    df = apply_basic_filters(df, vessel_specs=MCR)  # Replace None with actual vessel specs if available
    
    return df

# if __name__ == '__main__':
#     preprocess_data('data/raw/input.csv', 'data/processed/output.csv')
