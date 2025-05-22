import joblib
import numpy as np
import os

# Get the absolute path to the model
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "hallucination_safe_model_v5000.pkl")

model = joblib.load(MODEL_PATH)

def predict_hallucination(features: dict):
    order = [
        "mean_ewcl", "std_ewcl", "collapse_likelihood",
        "mean_plddt", "std_plddt", "mean_bfactor", "std_bfactor"
    ]
    X = np.array([[features.get(k, 0.0) for k in order]])
    return float(model.predict(X)[0])