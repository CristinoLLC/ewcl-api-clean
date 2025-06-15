import pickle
from fastapi import UploadFile, File
import pandas as pd
import io
import logging

# Load the final working model
try:
    with open("models/ewcl_final_model.pkl", "rb") as f:
        model = pickle.load(f)
    logging.info("✅ Loaded final model (ewcl_final_model.pkl)")
except Exception as e:
    logging.error("❌ Error loading final model: %s", e)
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