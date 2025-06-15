import joblib
import os
from fastapi import UploadFile, File 
from tempfile import NamedTemporaryFile
import shutil
import logging 

try:
    model = joblib.load("models/ewcl_Main.pkl")
    logging.info("✅ Loaded EWCL Main model from routes/analyze.py") # Updated path in log
except Exception as e:
    logging.error(f"❌ Error loading EWCL Main model: {e}")

async def analyze_file(file: UploadFile = File(...)):
    if model is None:
        logging.error("❌ Main model not loaded. Cannot process request.")
        return {"status": "error", "message": "Model not loaded", "results": []}
    
    tmp_path = ""
    try:
        with NamedTemporaryFile(delete=False) as tmp:
            shutil.copyfileobj(file.file, tmp)
            tmp_path = tmp.name
        logging.info(f"📂 Processing file in routes/analyze.py: {file.filename}") # Updated path in log

        # Perform analysis using the model
        results = model.predict([tmp_path])  # Example usage
        logging.info("✅ Analysis complete")
        return {"status": "success", "message": "Analysis complete", "results": results}
    except Exception as e:
        logging.exception("❌ Error during analysis in routes/analyze.py") # Updated path in log
        return {"status": "error", "message": str(e), "results": []}
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)
            logging.info("🗑️ Temporary file deleted")