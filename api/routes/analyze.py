import joblib
import os
from fastapi import UploadFile, File
from tempfile import NamedTemporaryFile
import shutil
import logging

try:
    model = joblib.load("models/ewcl_Main.pkl")
    logging.info("✅ Loaded EWCL Main model from api/routes/analyze.py")
except Exception as e:
    logging.error(f"❌ Error loading EWCL Main model: {e}")
    model = None

async def analyze_file(file: UploadFile = File(...)):
    if model is None:
        logging.error("❌ Main model not loaded. Cannot process request.")
        return {"status": "error", "message": "Model not loaded", "results": []}
    
    tmp_path = ""
    try:
        suffix = os.path.splitext(file.filename)[-1]
        with NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            shutil.copyfileobj(file.file, tmp)
            tmp_path = tmp.name
        
        logging.info(f"📂 Processing file in api/routes/analyze.py: {file.filename}")

        predictions = model.predict(tmp_path)
        
        if hasattr(predictions, 'tolist'):
            results = predictions.round(4).tolist() if hasattr(predictions, 'round') else predictions.tolist()
        else:
            results = predictions

        return {"status": "ok", "message": "Analysis successful", "results": results}

    except Exception as e:
        logging.exception("❌ Error during analysis in api/routes/analyze.py")
        return {"status": "error", "message": str(e), "results": []}
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)