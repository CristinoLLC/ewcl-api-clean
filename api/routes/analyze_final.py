import joblib
import os
from fastapi import UploadFile, File
from tempfile import NamedTemporaryFile
import pandas as pd
import shutil
import numpy as np
import logging

# Load the final (old) model
try:
    model = joblib.load("models/ewcl_final_model.pkl")
    logging.info("✅ Loaded EWCL Final model (ewcl_final_model.pkl)")
except Exception as e:
    logging.error(f"❌ Error loading EWCL Final model: {e}")
    model = None

async def analyze_final(file: UploadFile = File(...)):
    if model is None:
        logging.error("❌ Final model not loaded. Cannot process request.")
        return {"status": "error", "message": "Final model not loaded", "results": []}

    tmp_path = ""
    try:
        suffix = os.path.splitext(file.filename)[-1]
        # Ensure the uploaded file is a CSV for this endpoint
        if suffix.lower() != ".csv":
            return {"status": "error", "message": "Invalid file type. Please upload a CSV file.", "results": []}
            
        with NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            shutil.copyfileobj(file.file, tmp)
            tmp_path = tmp.name
        
        logging.info(f"📂 Processing CSV for final model analysis: {file.filename}")
        df = pd.read_csv(tmp_path)
        features = df.drop(columns=["resi", "plddt", "b_factor"], errors="ignore").values
        predictions = model.predict(features)

        return {"ewcl_score": np.round(predictions, 4).tolist()}

    except Exception as e:
        logging.exception("❌ Error during final model analysis")
        return {"status": "error", "message": str(e), "results": []}
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)
