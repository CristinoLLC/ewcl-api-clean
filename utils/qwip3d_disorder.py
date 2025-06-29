import joblib
import numpy as np
import pandas as pd
import os

# Path to the model file
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "qwip3d_disorder_clf.pkl")

# Load the model once
try:
    model = joblib.load(MODEL_PATH)
    print(f"✅ Loaded QWIP3D disorder classifier from {MODEL_PATH}")
except FileNotFoundError:
    print(f"❌ Model file not found: {MODEL_PATH}")
    model = None

def predict_disorder(df: pd.DataFrame, proba: bool = True) -> np.ndarray:
    """
    Predict disorder probability from QWIP3D features
    
    Args:
        df: DataFrame with required features
        proba: If True, return probabilities; if False, return binary predictions
    
    Returns:
        Array of predictions
    """
    if model is None:
        raise RuntimeError("QWIP3D disorder model not loaded")
    
    # Required features for the model
    features = ["disorder_score", "qwip_3d", "slope", "curvature", "entropy_slope", "local_peak"]
    
    # Check if all features are present
    missing_features = [f for f in features if f not in df.columns]
    if missing_features:
        raise ValueError(f"Missing required features: {missing_features}")
    
    # Prepare feature matrix
    X = df[features].fillna(0).values
    
    # Make predictions
    if proba:
        return model.predict_proba(X)[:, 1]  # Return probability of disorder class
    else:
        return model.predict(X)  # Return binary predictions
