import pickle
import os
from fastapi import UploadFile, File
import pandas as pd
import io
import logging

# Dynamically construct the absolute path to the model file
MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../models/ewcl_final_model.pkl"))

# Check if the model file exists
if os.path.exists(MODEL_PATH):
    logging.info(f"✅ Model file detected at {MODEL_PATH}")
else:
    logging.error(f"❌ Model file not found at {MODEL_PATH}")

# Load the final working model
try:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    logging.info(f"✅ Loaded final model from {MODEL_PATH}")
except Exception as e:
    logging.error(f"❌ Error loading final model: {e}")
    model = None

async def analyze_file(file: UploadFile = File(...)):
    contents = await file.read()
    df = pd.read_csv(io.BytesIO(contents))
    
    if model is None:
        logging.error("❌ Final model not loaded. Cannot process request.")
        return {"status": "error", "message": "Model not loaded", "results": []}
    
    try:
        logging.info(f"📂 Processing file in api/routes/analyze.py: {file.filename}")
        
        # Flex logic for prediction
        preds = model.predict(df) if hasattr(model, 'predict') else model(df)
        results = preds.round(4).tolist() if hasattr(preds, 'round') else preds.tolist()

        return {"status": "ok", "message": "Analysis successful", "results": results}

    except Exception as e:
        logging.exception("❌ Error during analysis in api/routes/analyze.py")
        return {"status": "error", "message": str(e), "results": []}