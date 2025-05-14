from fastapi import FastAPI, UploadFile, File

app = FastAPI()

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.get("/sample-pdb")
def get_sample_pdb(id: str):
    # Return mock EWCL results or load real PDB logic
    return {
        "protein_id": id,
        "ewcl_scores": [0.2, 0.4, 0.6, 0.85]
    }

@app.post("/analyze")
async def analyze_pdb(file: UploadFile = File(...)):
    # Process uploaded file and return EWCL scores
    return {
        "filename": file.filename,
        "ewcl_scores": [0.1, 0.3, 0.75, 0.9]
    }
