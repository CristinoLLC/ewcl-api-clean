import joblib
import os
from fastapi import UploadFile, File
from tempfile import NamedTemporaryFile
import pandas as pd
import shutil
import numpy as np
import logging

# More robust path resolution
def find_regressor_model_path():
    possible_paths = [
        os.path.abspath(os.path.join(os.path.dirname(__file__), "../../models/ewcl_regressor_v1.pkl")),
        os.path.abspath("models/ewcl_regressor_v1.pkl"),
        os.path.abspath(os.path.join(os.getcwd(), "models/ewcl_regressor_v1.pkl")),
        "/opt/render/project/src/models/ewcl_regressor_v1.pkl"
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            logging.info(f"✅ Found regressor model at: {path}")
            return path
        else:
            logging.debug(f"❌ Regressor model not found at: {path}")
    
    logging.error("❌ Regressor model file not found in any expected location")
    return None

# Load the regressor model at startup with better error handling
model = None
MODEL_PATH = find_regressor_model_path()

if MODEL_PATH:
    try:
        model = joblib.load(MODEL_PATH)
        logging.info(f"✅ Loaded EWCL Regressor model from {MODEL_PATH}")
    except Exception as e:
        logging.error(f"❌ Error loading EWCL Regressor model: {e}")
        logging.info("ℹ️ Regressor model unavailable due to compatibility issues")
        model = None
else:
    logging.error("❌ No valid regressor model path found")

async def analyze_regression(file: UploadFile = File(...)):
    if model is None:
        logging.error("❌ Regressor model not loaded. Cannot process request.")
        return {
            "status": "error", 
            "message": "Regressor model temporarily unavailable due to compatibility issues. Please use /analyze or /analyze-final endpoints instead.", 
            "results": []
        }

    tmp_path = ""
    try:
        suffix = os.path.splitext(file.filename)[-1]
        if suffix.lower() != ".csv":
            return {"status": "error", "message": "Invalid file type. Please upload a CSV file.", "results": []}

        with NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            shutil.copyfileobj(file.file, tmp)
            tmp_path = tmp.name

        logging.info(f"📂 Processing CSV for regression: {file.filename}")
        df = pd.read_csv(tmp_path)
        features = df.drop(columns=["resi", "plddt", "b_factor"], errors="ignore").values
        predictions = model.predict(features)
        
        return {"ewcl_score": np.round(predictions, 4).tolist()}

    except Exception as e:
        logging.exception("❌ Error during regression analysis")
        return {"status": "error", "message": str(e), "results": []}
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)
