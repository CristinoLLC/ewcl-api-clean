from fastapi import FastAPI
from backend.api.router import router as ewcl_router


app = FastAPI(title="EWCL Inference API", version="1.0.0")
app.include_router(ewcl_router)


@app.get("/")
def root():
    return {"status": "ok", "message": "EWCL Inference API", "routes": ["/ewcl/health", "/ewcl/predict/ewclv1m", "/ewcl/predict/ewclv1"]}


