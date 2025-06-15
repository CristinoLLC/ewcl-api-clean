import joblib
import os
from fastapi import UploadFile, File # File is used as a type hint
from tempfile import NamedTemporaryFile
import shutil
import logging # Added for consistency

# Load the reversed model once when the module is imported.
try:
    model_rev = joblib.load("models/ewcl_Rev.pkl")
    logging.info("✅ Loaded EWCL Reversed model from api/routes/analyze_rev.py")
except Exception as e:
    logging.error(f"❌ Error loading EWCL Reversed model: {e}")
    model_rev = None

async def analyze_reverse(file: UploadFile = File(...)):
    if model_rev is None:
        logging.error("❌ Reversed model not loaded. Cannot process request.")
        return {"status": "error", "message": "Reversed model not loaded", "results": []}

    tmp_path = ""
    try:
        suffix = os.path.splitext(file.filename)[-1]
        with NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            shutil.copyfileobj(file.file, tmp)
            tmp_path = tmp.name

        logging.info(f"📂 Processing file in api/routes/analyze_rev.py: {file.filename}")
        
        # 🧠 Process the PDB file using your reversed model
        predictions = model_rev.predict(tmp_path) # Customize based on your model's input requirements

        if hasattr(predictions, 'tolist'):
            results = predictions.round(4).tolist() if hasattr(predictions, 'round') else predictions.tolist()
        else:
            results = predictions

        return {"status": "ok", "message": "Reverse analysis successful", "results": results}

    except Exception as e:
        logging.exception("❌ Error during reverse analysis in api/routes/analyze_rev.py")
        return {"status": "error", "message": str(e), "results": []}
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)
