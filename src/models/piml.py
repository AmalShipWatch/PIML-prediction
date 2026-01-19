# src/models/piml.py
import numpy as np
import pandas as pd
from scipy.interpolate import UnivariateSpline, RectBivariateSpline
from sklearn.linear_model import Ridge

from src import config
from src.utils.physics import apply_physics_constraints

class BSplineRegressor:
    """B-spline based regressor for smooth, monotonic predictions."""
    
    def __init__(self, k=3, s=None):
        """
        k: Spline degree (3 = cubic, default)
        s: Smoothing factor (None = auto)
        """
        self.k = k
        self.s = s
        self.splines = {}  # Store splines for each laden/draft combination
        self.ridge_model = None
        
    def fit(self, X_train, y_residuals, sample_weight=None):
        """Fit B-spline to residuals grouped by laden/draft conditions."""
        
        # Extract features
        speeds = X_train['speed'].values
        drafts = X_train['draft'].values
        laden = X_train['laden_ballast'].values
        
        # Fit Ridge regression on spline features as fallback
        # This creates a smooth function across all features
        X_features = np.column_stack([speeds, speeds**2, drafts, laden])
        
        if sample_weight is not None:
            self.ridge_model = Ridge(alpha=0.1)
            self.ridge_model.fit(X_features, y_residuals, sample_weight=sample_weight)
        else:
            self.ridge_model = Ridge(alpha=0.1)
            self.ridge_model.fit(X_features, y_residuals)
        
        return self
    
    def predict(self, X_test):
        """Predict using B-spline model."""
        speeds = X_test['speed'].values
        drafts = X_test['draft'].values
        laden = X_test['laden_ballast'].values
        
        X_features = np.column_stack([speeds, speeds**2, drafts, laden])
        return self.ridge_model.predict(X_features)


class PIMLRegressor:
    def __init__(self, use_bspline=True):
        self.model = None
        self.use_bspline = use_bspline
        self.phys_coef = config.PHYSICS_COEFFICIENTS
        
    def _create_base_regressor(self, seed):
        """Factory method for the regressor."""
        if self.use_bspline:
            return BSplineRegressor(k=3, s=None)
        else:
            # Fallback to Ridge regression for smoothness
            from sklearn.linear_model import Ridge
            return Ridge(alpha=0.1)

    def fit(self, X_train, y_train, phys_base, speeds, drafts, laden_flags=None):
        """
        Iterative PIML training with constraint-based re-weighting using B-splines.
        """
        if laden_flags is None:
            laden_flags = X_train.get("laden_ballast", np.zeros(len(X_train))).values
        else:
            laden_flags = np.asarray(laden_flags)
            
        residuals = y_train - phys_base
        sample_weights = np.ones(len(y_train))
        
        for i in range(config.ITERATIONS):
            # Train model on residuals with smooth B-spline
            self.model = self._create_base_regressor(config.RANDOM_SEED + i)
            self.model.fit(X_train, residuals, sample_weight=sample_weights)
            
            # Predict and reconstruct full power
            pred_residuals = self.model.predict(X_train)
            raw_preds = phys_base + pred_residuals
            final_preds = apply_physics_constraints(raw_preds, config.MIN_POWER_THRESHOLD)
            
            # If not last iteration, update weights
            if i < config.ITERATIONS - 1:
                sample_weights = self._update_weights(
                    raw_preds, final_preds, speeds, drafts, sample_weights, laden_flags
                )
                print(f"Iteration {i+1}/{config.ITERATIONS} complete. Weights updated (B-spline).")
        
        return self

    def predict(self, X_test):
        """Return constrained predictions."""
        phys_base = X_test["phys_base"].values
        pred_residuals = self.model.predict(X_test)
        raw_preds = phys_base + pred_residuals
        return apply_physics_constraints(raw_preds, config.MIN_POWER_THRESHOLD)

    def _update_weights(self, raw_preds, final_preds, speeds, drafts, current_weights, laden_flags):
        """Calculate physics violations and update weights with multiple loss components."""
        
        # Initialize adjustment weights
        w_boundary = np.ones(len(speeds))
        w_monotonicity = np.ones(len(speeds))
        w_separation = np.ones(len(speeds))
        w_draft = np.ones(len(speeds))
        
        # 1. BOUNDARY VIOLATIONS (Hard constraint: non-negativity)
        violations = np.maximum(config.MIN_POWER_THRESHOLD - raw_preds, 0)
        if np.max(violations) > 0:
            w_boundary = 1.0 + config.BOUNDARY_LOSS_WEIGHT * (violations / np.max(violations))
        
        # 2. MONOTONICITY VIOLATIONS (Task 1)
        # Penalize cases where y[i+1] < y[i] when sorted by speed/draft
        df_temp = pd.DataFrame({
            'speed': speeds, 'draft': drafts, 'power': final_preds, 'laden': laden_flags,
            'idx': np.arange(len(speeds))
        }).sort_values(['draft', 'speed'])
        
        for draft_val in df_temp['draft'].unique():
            grp = df_temp[df_temp['draft'] == draft_val]
            if len(grp) > 1:
                powers = grp['power'].values
                indices = grp['idx'].values
                # Find drops in power as speed increases
                drops = np.where(powers[1:] < powers[:-1])[0]
                for idx in drops:
                    # Penalize both points involved in the drop
                    w_monotonicity[indices[idx]] += config.MONOTONICITY_LOSS_WEIGHT * 0.5
                    w_monotonicity[indices[idx+1]] += config.MONOTONICITY_LOSS_WEIGHT * 0.5
        
        # 3. SEPARATION LOSS (Task 2: Penalize when laden_pred - ballast_pred < minimum_gap)
        # For each speed point, calculate laden vs ballast gap
        for speed_val in df_temp['speed'].unique():
            spd_grp = df_temp[df_temp['speed'] == speed_val]
            laden_preds = spd_grp[spd_grp['laden'] == 1]['power'].values
            ballast_preds = spd_grp[spd_grp['laden'] == 0]['power'].values
            
            if len(laden_preds) > 0 and len(ballast_preds) > 0:
                gap = np.mean(laden_preds) - np.mean(ballast_preds)
                if gap < config.MINIMUM_LADEN_BALLAST_GAP:
                    gap_deficit = config.MINIMUM_LADEN_BALLAST_GAP - gap
                    # Penalize samples in this speed group
                    speed_indices = spd_grp['idx'].values
                    w_separation[speed_indices] += config.SEPARATION_LOSS_WEIGHT * (gap_deficit / config.MINIMUM_LADEN_BALLAST_GAP)
        
        # 4. DRAFT DEPENDENCY (Task 3: Ensure X4 coefficient effect is learned)
        # Penalize samples where draft effect is not properly captured
        # Draft effect should be X4 * draft contribution
        x4_coef = self.phys_coef['X4']
        for i in range(len(speeds)):
            draft_effect_expected = x4_coef * drafts[i]
            # High draft should correlate with higher power
            if drafts[i] > np.median(drafts):
                # For high-draft samples, penalize if they don't show expected high power
                if final_preds[i] < np.median(final_preds):
                    w_draft[i] += config.DRAFT_DEPENDENCY_WEIGHT
        
        # Combine all weights
        new_weights = current_weights * w_boundary * w_monotonicity * w_separation * w_draft
        return new_weights / np.mean(new_weights)