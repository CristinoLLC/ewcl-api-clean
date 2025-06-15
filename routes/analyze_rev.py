import joblib
import os
from fastapi import UploadFile, File 
from tempfile import NamedTemporaryFile
import shutil
import logging 

try:
    model_rev = joblib.load("models/ewcl_Rev.pkl")
    logging.info("✅ Loaded EWCL Reversed model from routes/analyze_rev.py") # Updated path in log
except Exception as e:
    logging.error(f"❌ Error loading EWCL Reversed model: {e}")

async def analyze_reverse(file: UploadFile = File(...)):
    if model_rev is None:
        logging.error("❌ Reversed model not loaded. Cannot process request.")
        return {"status": "error", "message": "Reversed model not loaded", "results": []}

    tmp_path = ""
    try:
        with NamedTemporaryFile(delete=False) as tmp:
            shutil.copyfileobj(file.file, tmp)
            tmp_path = tmp.name
        logging.info(f"📂 Processing file in routes/analyze_rev.py: {file.filename}") # Updated path in log
        
        # Perform analysis using the reversed model
        results = model_rev.predict([tmp_path])  # Example usage
        logging.info("✅ Reverse analysis completed successfully.")
        return {"status": "success", "message": "Reverse analysis completed", "results": results}
    except Exception as e:
        logging.exception("❌ Error during reverse analysis in routes/analyze_rev.py") # Updated path in log
        return {"status": "error", "message": str(e), "results": []}
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)
            logging.info("🗑️ Temporary file deleted.")