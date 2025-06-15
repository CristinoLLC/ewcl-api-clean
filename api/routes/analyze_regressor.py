import joblib
import os
from fastapi import UploadFile, File
from tempfile import NamedTemporaryFile
import pandas as pd
import shutil
import numpy as np
import logging

# Load the regressor model at startup
try:
    model = joblib.load("models/ewcl_regressor_v1.pkl")
    logging.info("✅ Loaded EWCL Regressor model (ewcl_regressor_v1.pkl)")
except Exception as e:
    logging.error(f"❌ Error loading EWCL Regressor model: {e}")
    model = None

async def analyze_regression(file: UploadFile = File(...)):
    if model is None:
        logging.error("❌ Regressor model not loaded. Cannot process request.")
        return {"status": "error", "message": "Regressor model not loaded", "results": []}

    tmp_path = ""
    try:
        suffix = os.path.splitext(file.filename)[-1]
        # Ensure the uploaded file is a CSV for this endpoint
        if suffix.lower() != ".csv":
            return {"status": "error", "message": "Invalid file type. Please upload a CSV file.", "results": []}

        with NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            shutil.copyfileobj(file.file, tmp)
            tmp_path = tmp.name

        logging.info(f"📂 Processing CSV for regression: {file.filename}")
        df = pd.read_csv(tmp_path)
        # Ensure required columns for feature extraction are present, or handle missing robustly
        features = df.drop(columns=["resi", "plddt", "b_factor"], errors="ignore").values
        predictions = model.predict(features)
        
        return {"ewcl_score": np.round(predictions, 4).tolist()}

    except Exception as e:
        logging.exception("❌ Error during regression analysis")
        return {"status": "error", "message": str(e), "results": []}
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)
