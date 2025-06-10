import joblib
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

# Create FastAPI app
app = FastAPI()

# Add health check route
@app.get("/health")
def health():
    return {"status": "ok"}

# ✅ Add Error Handling for Model Loading
try:
    model = joblib.load("models/unified_entropy_model.pkl")
    print("✅ Model loaded successfully.")
except Exception as e:
    print(f"❌ Failed to load model: {e}")
    model = None

# Define schema for prediction input
class InputData(BaseModel):
    length: float
    mean: float
    std: float

# Define schema for batch prediction input
class BatchInput(BaseModel):
    inputs: List[InputData]

# 🔒 Validate Model Availability Before Prediction
@app.post("/predict")
def predict(input: InputData):
    if model is None:
        return {"error": "Model not loaded."}
    
    try:
        features = [input.length, input.mean, input.std]
        prediction = model.predict([features])
        return {"score": float(prediction[0])}
    except Exception as e:
        return {"error": f"Prediction failed: {str(e)}"}

# 🚀 Add /predict-batch for Bulk Use
@app.post("/predict-batch")
def predict_batch(batch: BatchInput):
    if model is None:
        return {"error": "Model not loaded."}
    
    try:
        features_list = [[i.length, i.mean, i.std] for i in batch.inputs]
        predictions = model.predict(features_list)
        return {
            "scores": [float(p) for p in predictions],
            "count": len(predictions)
        }
    except Exception as e:
        return {"error": f"Batch prediction failed: {str(e)}"}

# Add model status endpoint for debugging
@app.get("/model-status")
def model_status():
    return {
        "model_loaded": model is not None,
        "model_type": str(type(model)) if model is not None else None
    }
