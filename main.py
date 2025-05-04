from fastapi import FastAPI, Request, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import pickle
import numpy as np
import os

from ewcl_toolkit.ewcl_static_tool import ewcl_score_protein

app = FastAPI()

# CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Input model
class SequenceRequest(BaseModel):
    sequence: str

# Load AI model
model_path = os.path.join("models", "ewcl_model.pkl")
model = None
try:
    with open(model_path, "rb") as f:
        model = pickle.load(f)
except Exception as e:
    print(f"❌ Failed to load model: {e}")

# Root
@app.get("/")
def root():
    return {"status": "online", "message": "EWCL API running"}

# Health check for GET and HEAD
@app.get("/health")
def health():
    return JSONResponse(status_code=200, content={"ok": True})

@app.head("/health")
def health_head():
    return JSONResponse(status_code=200, content=None)

# Fallback POST to root
@app.post("/")
def fallback_root():
    return JSONResponse(status_code=405, content={"error": "Use /analyze or /analyze/her2"})

# AI inference endpoint
@app.post("/runaiinference")
async def run_inference(request: Request):
    data = await request.json()
    try:
        X = np.array([[data["score"], data["avgEntropy"], data["minEntropy"], data["maxEntropy"]]])
        prediction = model.predict(X)[0]
        return {"collapseRisk": float(prediction)}
    except Exception as e:
        return {"error": str(e)}

# EWCL core scoring
@app.post("/runeucl")
def run_ewcl(req: SequenceRequest):
    try:
        result = ewcl_score_protein(req.sequence)
        return {"ewcl_map": result}
    except Exception as e:
        return {"error": str(e)}

# Analyze from uploaded file
@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        try:
            sequence = contents.decode("utf-8").strip()
            if not sequence:
                return {"error": "Empty sequence", "status": "error"}
        except UnicodeDecodeError:
            return {"error": "Invalid file format", "status": "error"}
            
        result = ewcl_score_protein(sequence)
        return {
            "filename": file.filename,
            "ewcl_map": result,
            "status": "success"
        }
    except Exception as e:
        return {"error": str(e), "status": "error"}

# HER2 built-in analysis
@app.post("/analyze/her2")
async def analyze_her2(file: UploadFile = None):
    try:
        her2_sequence = "MKLRLPASPETHLDMLRHLYQGCQVVQGNLELTYLPTNASLSFLQDIQEVQGYVLIAHNQVRQVPLQRLRIVRGTQLFEDNYALAVLDNGDPLNNTTPVTGASPGGLRELQLRSLTEILKGGVLIQRNPQLCYQDTILWKDIFHKNNQLALTLIDTNRSRACHPCSPMCKGSRCWGESSEDCQSLTRTVCAGGCARCKGPLPTDCCHEQCAAGCTGPKHSDCLACLHFNHSGICELHCPALVTYNTDTFESMPNPEGRYTFGASCVTACPYNYLSTDVGSCTLVCPLHNQEVTAEDGTQRCEKCSKPCARVCYGLGMEHLREVRAVTSANIQEFAGCKKIFGSLAFLPESFDGDPASNTAPLQPEQLQVFETLEEITGYLYISAWPDSLPDLSVFQNLQVIRGRILHNGAYSLTLQGLGISWLGLRSLRELGSGLALIHHNTHLCFVHTVPWDQLFRNPHQALLHTANRPEDECVGEGLACHQLCARGHCWGPGPTQCVNCSQFLRGQECVEECRVLQGLPREYVNARHCLPCHPECQPQNGSVTCFGPEADQCVACAHYKDPPFCVARCPSGVKPDLSYMPIWKFPDEEGACQPCPINCTHSCVDLDDKGCPAEQRASPLTSIISAVVGILLVVVLGVVFGILIKRRQQKIRKYTMRRLLQETELVEPLTPSGAMPNQAQMRILKETELRKVKVLGSGAFGTVYKGIWIPDGENVKIPVAIKVLRENTSPKANKEILDEAYVMAGVGSPYVSRLLGICLTSTVQLVTQLMPYGCLLDHVRENRGRLGSQDLLNWCMQIAKGMSYLEDVRLVHRDLAARNVLVKSPNHVKITDFGLARLLDIDETEYHADGGKVPIKWMALESILRRRFTHQSDVWSYGVTVWELMTFGAKPYDGIPAREIPDLLEKGERGERPTEMPTPKANKECVQREAKSEKFGMGSSPKDS"
        file_info = ""
        if file:
            file_info = f" (Note: Uploaded file '{file.filename}' was ignored)"
            
        result = ewcl_score_protein(her2_sequence)
        return {
            "protein_id": "HER2",
            "ewcl_map": result,
            "message": f"Using pre-defined HER2 sequence{file_info}",
            "status": "success"
        }
    except Exception as e:
        return {"error": str(e), "status": "error"}

# Analyze other proteins by ID
@app.post("/analyze/{protein_id}")
async def analyze_by_id(protein_id: str):
    try:
        protein_sequences = {
            "HER2": "...",  # Use same HER2 sequence as above
            "BRCA2": "..."  # Add BRCA2 here
        }
        if protein_id not in protein_sequences:
            return {"error": f"Protein ID '{protein_id}' not found", "status": "error"}
            
        sequence = protein_sequences[protein_id]
        result = ewcl_score_protein(sequence)
        return {
            "protein_id": protein_id,
            "ewcl_map": result,
            "status": "success"
        }
    except Exception as e:
        return {"error": str(e), "status": "error"}

# Endpoint to list supported proteins
@app.get("/files")
async def list_available_proteins():
    return {
        "proteins": [
            {"id": "HER2", "file": "AF-P04626-F1-model_v4.pdb"},
            {"id": "BRCA2", "file": "AF-P51587-F1-model_v4.pdb"}
        ]
    }