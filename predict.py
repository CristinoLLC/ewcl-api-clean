import joblib
import numpy as np
import os

# ✅ Use correct filenames and default to local 'models/' path
HALLUC_MODEL_PATH = os.getenv("HALLUC_MODEL_PATH", "models/hallucination_safe_model_v5000.pkl")
REFOLD_MODEL_PATH = os.getenv("REFOLD_MODEL_PATH", "models/refolding_classifier_model.pkl")

if not os.path.isfile(HALLUC_MODEL_PATH):
    print(f"❌ Hallucination model file not found at: {HALLUC_MODEL_PATH}")
    model = None
else:
    try:
        model = joblib.load(HALLUC_MODEL_PATH)
        print("✅ Hallucination model loaded.")
    except Exception as e:
        print(f"❌ Error loading hallucination model: {e}")
        model = None

if not os.path.isfile(REFOLD_MODEL_PATH):
    print(f"❌ Refolding model file not found at: {REFOLD_MODEL_PATH}")
    refold_model = None
else:
    try:
        refold_model = joblib.load(REFOLD_MODEL_PATH)
        print("✅ Refolding model loaded.")
    except Exception as e:
        print(f"❌ Error loading refolding model: {e}")
        refold_model = None

def predict_hallucination(features: dict):
    order = [
        "mean_ewcl", "std_ewcl", "collapse_likelihood",
        "mean_plddt", "std_plddt", "mean_bfactor", "std_bfactor"
    ]
    X = np.array([[features.get(k, 0.0) for k in order]])
    return float(model.predict(X)[0]) if model else None