from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from tempfile import NamedTemporaryFile
import shutil
import os
import logging
from fastapi.responses import JSONResponse

from entropy_collapse_model import infer_entropy_from_pdb

app = FastAPI()
logging.basicConfig(level=logging.INFO)

# ✅ Allow CORS from specific origins
origins = [
    "http://localhost:3000",
    "https://ewclx.com",
    "https://www.ewcl.com",
    "https://www.ewclx.com",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"status": "EWCL API running"}

@app.get("/health")
def health_check():
    return {"status": "ok", "message": "Protein Collapse Analysis API is running"}

@app.post("/analyze")
async def analyze_file(file: UploadFile = File(...)):
    try:
        # ⏳ Save uploaded file
        suffix = os.path.splitext(file.filename)[-1]
        with NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            shutil.copyfileobj(file.file, tmp)
            tmp_path = tmp.name

        logging.info(f"📂 Processing file in normal mode: {file.filename}")

        # 🔬 Run entropy-based EWCL - model returns complete JSON response
        result = infer_entropy_from_pdb(tmp_path)

        os.remove(tmp_path)

        # Return the result directly since model already provides complete JSON
        return result

    except Exception as e:
        logging.exception("❌ Error during analysis")
        return {
            "status": "error",
            "message": str(e),
            "results": [],
        }

@app.post("/analyze-rev")
async def analyze_reverse(file: UploadFile = File(...)):
    try:
        # Prevent accidental call
        if not file.filename.endswith("-reverse-true"):
            return JSONResponse(status_code=400, content={"error": "Reverse mode disabled"})

        # 🧪 Save file
        suffix = os.path.splitext(file.filename)[-1]
        with NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            shutil.copyfileobj(file.file, tmp)
            tmp_path = tmp.name

        logging.info(f"📂 Reverse mode analysis for: {file.filename}")

        # 🔄 Call same entropy model - model returns complete JSON response
        result = infer_entropy_from_pdb(tmp_path, reverse=True)

        os.remove(tmp_path)

        # Return the result directly since model already provides complete JSON
        return result

    except Exception as e:
        logging.exception("❌ Error in reverse analysis")
        return {
            "status": "error",
            "message": str(e),
            "results": [],
        }
