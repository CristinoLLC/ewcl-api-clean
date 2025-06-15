import pickle
import os
from fastapi import UploadFile, File
import pandas as pd
import io
import logging

# Try to import joblib as backup
try:
    import joblib
    HAS_JOBLIB = True
except ImportError:
    HAS_JOBLIB = False

# More robust path resolution - try multiple possible locations
def find_model_path():
    possible_paths = [
        os.path.abspath(os.path.join(os.path.dirname(__file__), "../../models/ewcl_final_model.pkl")),
        os.path.abspath("models/ewcl_final_model.pkl"),
        os.path.abspath(os.path.join(os.getcwd(), "models/ewcl_final_model.pkl")),
        "/opt/render/project/src/models/ewcl_final_model.pkl"
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            logging.info(f"✅ Found model at: {path}")
            return path
        else:
            logging.debug(f"❌ Model not found at: {path}")
    
    logging.error("❌ Model file not found in any expected location")
    return None

def load_model_safely(model_path):
    """Try to load model with different methods"""
    # Try pickle first
    try:
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        logging.info(f"✅ Loaded model using pickle from {model_path}")
        return model
    except Exception as pickle_error:
        logging.warning(f"⚠️ Pickle loading failed: {pickle_error}")
    
    # Try joblib if available
    if HAS_JOBLIB:
        try:
            model = joblib.load(model_path)
            logging.info(f"✅ Loaded model using joblib from {model_path}")
            return model
        except Exception as joblib_error:
            logging.warning(f"⚠️ Joblib loading failed: {joblib_error}")
    
    # Log file details for debugging
    try:
        file_size = os.path.getsize(model_path)
        logging.error(f"❌ Failed to load model. File size: {file_size} bytes")
        
        # Check first few bytes
        with open(model_path, "rb") as f:
            first_bytes = f.read(10)
        logging.error(f"❌ First 10 bytes: {first_bytes}")
    except Exception as e:
        logging.error(f"❌ Cannot examine file: {e}")
    
    return None

MODEL_PATH = find_model_path()

# Load the final working model
model = None
if MODEL_PATH:
    model = load_model_safely(MODEL_PATH)
else:
    logging.error("❌ No valid model path found")

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