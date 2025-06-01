import joblib
from fastapi import FastAPI
from pydantic import BaseModel

# Create FastAPI app
app = FastAPI()

# Add health check route
@app.get("/health")
def health():
    return {"status": "ok"}

# Load your local model
model = joblib.load("models/ewcl_model_final.pkl")

# Define schema for prediction input
class InputData(BaseModel):
    length: float
    mean: float
    std: float

# Define prediction route
@app.post("/predict")
def predict(input: InputData):
    features = [input.length, input.mean, input.std]
    prediction = model.predict([features])
    return {"score": float(prediction[0])}
