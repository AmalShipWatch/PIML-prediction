# src/visualization/plots.py
import plotly.graph_objs as go
import pandas as pd
import numpy as np
from src import config
from src.features.build_features import prepare_features

def create_prediction_grid(base_df: pd.DataFrame, laden_status: int, n_points: int = 1000) -> pd.DataFrame:
    """
    Create a synthetic dataframe covering the full speed range.
    Used to ask the model: "What does the curve look like for this condition?"
    """
    # 1. Create smooth range of speeds
    speed_min = base_df[config.COL_SPEED].min()
    speed_max = base_df[config.COL_SPEED].max()
    speed_grid = np.linspace(speed_min, speed_max, n_points)
    
    # 2. Use median draft for a representative curve
    median_draft = base_df[config.COL_DRAFT].median()
    
    # 3. Create the dataframe
    grid_df = pd.DataFrame({
        config.COL_SPEED: speed_grid,
        config.COL_DRAFT: median_draft,
        config.COL_LADEN: laden_status # 1 or 0
    })
    
    return grid_df

def get_smooth_curve(regressor, grid_df: pd.DataFrame):
    """
    1. Prepare features for the grid.
    2. Predict using the trained PIML model.
    3. Smooth the output using a polynomial fit (to remove tree-based steps).
    """
    # Prepare X features (adds speed_sq, phys_base, etc.)
    X_grid = prepare_features(grid_df)
    
    # Predict
    y_pred = regressor.predict(X_grid)
    
    # Smooth the curve (Polyfit 3rd degree)
    # This keeps the physics shape but removes the jagged "stairs" of XGBoost
    speeds = grid_df[config.COL_SPEED].values
    z = np.polyfit(speeds, y_pred, 3)
    p = np.poly1d(z)
    y_smooth = p(speeds)
    
    return speeds, y_smooth

def plot_piml_curves(regressor, train_df, test_df):
    """
    Main plotting function. Generates the interactive Plotly figure 
    with Scatter points (Data) and Smooth Lines (Model).
    """
    fig = go.Figure()
    
    # --- 1. PLOT DATA POINTS (Scatter) ---
    
    # Train - Laden
    mask_tr_l = train_df[config.COL_LADEN] == 1
    fig.add_trace(go.Scatter(
        x=train_df.loc[mask_tr_l, config.COL_SPEED],
        y=train_df.loc[mask_tr_l, config.COL_POWER],
        mode='markers', name='Train (Laden)',
        marker=dict(color='red', opacity=0.3, size=5)
    ))
    
    # Train - Ballast
    mask_tr_b = train_df[config.COL_LADEN] == 0
    fig.add_trace(go.Scatter(
        x=train_df.loc[mask_tr_b, config.COL_SPEED],
        y=train_df.loc[mask_tr_b, config.COL_POWER],
        mode='markers', name='Train (Ballast)',
        marker=dict(color='blue', opacity=0.3, size=5)
    ))
    
    # Test - Laden
    mask_te_l = test_df[config.COL_LADEN] == 1
    fig.add_trace(go.Scatter(
        x=test_df.loc[mask_te_l, config.COL_SPEED],
        y=test_df.loc[mask_te_l, config.COL_POWER],
        mode='markers', name='Test (Laden)',
        marker=dict(symbol='diamond', color='darkred', size=6)
    ))

    # Test - Ballast
    mask_te_b = test_df[config.COL_LADEN] == 0
    fig.add_trace(go.Scatter(
        x=test_df.loc[mask_te_b, config.COL_SPEED],
        y=test_df.loc[mask_te_b, config.COL_POWER],
        mode='markers', name='Test (Ballast)',
        marker=dict(symbol='diamond', color='darkblue', size=6)
    ))

    # --- 2. PLOT PREDICTION CURVES (Lines) ---
    
    # Generate Grids
    grid_laden = create_prediction_grid(train_df, laden_status=1)
    grid_ballast = create_prediction_grid(train_df, laden_status=0)
    
    # Get Smooth Model Predictions
    x_l, y_l = get_smooth_curve(regressor, grid_laden)
    x_b, y_b = get_smooth_curve(regressor, grid_ballast)
    
    # Plot Curves
    fig.add_trace(go.Scatter(
        x=x_l, y=y_l, mode='lines', name='PIML Curve (Laden)',
        line=dict(color='red', width=4)
    ))
    
    fig.add_trace(go.Scatter(
        x=x_b, y=y_b, mode='lines', name='PIML Curve (Ballast)',
        line=dict(color='blue', width=4)
    ))

    # --- 3. PLOT PHYSICS BASELINE (Reference) ---
    # Calculate physics baseline using the new equation with coefficients
    from src.utils.physics import calculate_physics_baseline
    median_draft = train_df[config.COL_DRAFT].median()
    phys_baseline = calculate_physics_baseline(x_l, draft=np.full_like(x_l, median_draft))
    fig.add_trace(go.Scatter(
        x=x_l, y=phys_baseline, mode='lines', name='Physics Baseline',
        line=dict(color='orange', width=2, dash='dash')
    ))

    # Layout Styling
    fig.update_layout(
        title="<b>PIML Ship Performance Model</b><br>Laden vs Ballast Curves",
        xaxis_title="Speed (knots)",
        yaxis_title="Power (kW)",
        template="plotly_white",
        hovermode="closest",
        height=700
    )
    
    return fig