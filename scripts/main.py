# main.py
import sys
from pathlib import Path

# Add project root to path so src module can be imported
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Import modules
from src import config
from src.data.fetch_data import collect_noon_data
from src.data.preprocess import preprocess_data
from src.features.build_features import prepare_features, feature_engineering
from src.models.piml import PIMLRegressor
from src.visualization.plot import plot_piml_curves

def main(imo_value=None, start_date=None, end_date=None, MCR=None):
    print("=== PIML Prediction Pipeline ===")
    
    # 1. Load Data
    df = collect_noon_data(imo_value=imo_value, start_date=start_date, end_date=end_date)
    print(f"Data loaded: {df.shape}")

    df = feature_engineering(df, MCR)
    df = preprocess_data(df, MCR)

    print(f"Data after preprocessing: {df.shape}")
    
    # 2. Split Data
    # Important: Stratify by Laden/Ballast to ensure both exist in train/test
    train_df, test_df = train_test_split(
        df, 
        test_size=0.2, 
        random_state=config.RANDOM_SEED,
        stratify=df[config.COL_LADEN] 
    )
    
    # 3. Prepare Features
    X_train = prepare_features(train_df)
    X_test = prepare_features(test_df)
    
    y_train = train_df[config.COL_POWER].values
    y_test = test_df[config.COL_POWER].values
    
    train_speeds = train_df[config.COL_SPEED].values
    train_drafts = train_df[config.COL_DRAFT].values
    
    # 4. Train Model
    print("Training PIML Model...")
    regressor = PIMLRegressor()
    regressor.fit(
        X_train=X_train, 
        y_train=y_train, 
        phys_base=X_train["phys_base"].values,
        speeds=train_speeds,
        drafts=train_drafts,
        laden_flags=train_df[config.COL_LADEN].values
    )
    
    # 5. Predict
    y_pred = regressor.predict(X_test)
    
    # 6. Metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = mse ** 0.5
    r2 = r2_score(y_test, y_pred)
    mape = (abs((y_test - y_pred) / y_test)).mean() * 100
    accuracy = 100 - mape
    
    print(f"\nResults:")
    print(f"RMSE: {rmse:.4f}")
    print(f"R2 Score: {r2:.4f}")
    print(f"MAPE: {mape:.4f}%")
    print(f"Accuracy: {accuracy:.4f}%")
    
    # 7. Visualization
    print("Generating Curves...")
    # We pass the full trained regressor object + dataframes
    fig = plot_piml_curves(regressor, train_df, test_df)
    fig.show()


if __name__ == "__main__":
    # Fetch noon data
    imo_value = 9833670
    MCR = 7321.00
    start_date = '2024-01-01'
    end_date = pd.Timestamp.today().strftime('%Y-%m-%d')
    main(imo_value=imo_value, start_date=start_date, end_date=end_date, MCR=MCR)