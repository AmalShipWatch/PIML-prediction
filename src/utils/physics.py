# src/utils/physics.py
import numpy as np
from src import config

def calculate_physics_baseline(speed: np.ndarray, draft: np.ndarray = None) -> np.ndarray:
    """Calculate physics-based baseline power using equation:
    X1*x^3 + X2*x^2 + X3*x + X4*y + X5 + X6
    where x = speed, y = draft
    """
    coef = config.PHYSICS_COEFFICIENTS
    
    # If draft is not provided, assume zero draft contribution
    draft_term = coef['X4'] * draft if draft is not None else 0
    
    baseline = (coef['X1'] * speed**3 + 
                coef['X2'] * speed**2 + 
                coef['X3'] * speed + 
                draft_term +
                coef['X5'] + 
                coef['X6'])
    
    return baseline

def apply_physics_constraints(predictions: np.ndarray, min_threshold: float = 0.0) -> np.ndarray:
    """Apply hard constraints (non-negativity) to predictions."""
    return np.maximum(predictions, min_threshold)