from fastapi import FastAPI, Request, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import pickle
import numpy as np
import os

from ewcl_toolkit.ewcl_static_tool import ewcl_score_protein

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Define the input model
class SequenceRequest(BaseModel):
    sequence: str

model_path = os.path.join("models", "ewcl_model.pkl")
model = None

try:
    with open(model_path, "rb") as f:
        model = pickle.load(f)
except Exception as e:
    print(f"❌ Failed to load model: {e}")

@app.get("/")
def root():
    return {"status": "online", "message": "EWCL API running"}

@app.get("/health")
def health_check():
    return JSONResponse(content={"status": "healthy"})

@app.post("/runaiinference")
async def run_inference(request: Request):
    data = await request.json()
    try:
        X = np.array([[data["score"], data["avgEntropy"], data["minEntropy"], data["maxEntropy"]]])
        prediction = model.predict(X)[0]
        return {"collapseRisk": float(prediction)}
    except Exception as e:
        return {"error": str(e)}

@app.post("/runeucl")
def run_ewcl(req: SequenceRequest):
    try:
        result = ewcl_score_protein(req.sequence)
        return {"ewcl_map": result}
    except Exception as e:
        return {"error": str(e)}

# Updated: Combined analyze endpoint for both JSON and file uploads
@app.post("/analyze")
async def analyze(file: UploadFile = File(None), req: SequenceRequest = None):
    try:
        # Case 1: File upload
        if file:
            contents = await file.read()
            # Try to decode as text
            try:
                sequence = contents.decode("utf-8")
                # Here you would add PDB file parsing if needed
            except UnicodeDecodeError:
                return JSONResponse(
                    status_code=400,
                    content={"error": "Uploaded file is not valid text", "status": "error"}
                )
                
            result = ewcl_score_protein(sequence)
            return {
                "filename": file.filename,
                "ewcl_map": result,
                "status": "success"
            }
        
        # Case 2: JSON body with sequence
        elif req and req.sequence:
            result = ewcl_score_protein(req.sequence)
            return {"ewcl_map": result, "status": "success"}
        
        # Case 3: No valid input
        else:
            return JSONResponse(
                status_code=400,
                content={"error": "Please provide either a file or a sequence", "status": "error"}
            )
            
    except Exception as e:
        return JSONResponse(
            status_code=500, 
            content={"error": str(e), "status": "error"}
        )

# Keep the protein ID endpoint
@app.post("/analyze/{protein_id}")
async def analyze_by_id(protein_id: str):
    try:
        protein_sequences = {
            "HER2": "MKLRLPASPETHLDMLRHLYQGCQVVQGNLELTYLPTNASLSFLQDIQEVQGYVLIAHNQVRQVPLQRLRIVRGTQLFEDNYALAVLDNGDPLNNTTPVTGASPGGLRELQLRSLTEILKGGVLIQRNPQLCYQDTILWKDIFHKNNQLALTLIDTNRSRACHPCSPMCKGSRCWGESSEDCQSLTRTVCAGGCARCKGPLPTDCCHEQCAAGCTGPKHSDCLACLHFNHSGICELHCPALVTYNTDTFESMPNPEGRYTFGASCVTACPYNYLSTDVGSCTLVCPLHNQEVTAEDGTQRCEKCSKPCARVCYGLGMEHLREVRAVTSANIQEFAGCKKIFGSLAFLPESFDGDPASNTAPLQPEQLQVFETLEEITGYLYISAWPDSLPDLSVFQNLQVIRGRILHNGAYSLTLQGLGISWLGLRSLRELGSGLALIHHNTHLCFVHTVPWDQLFRNPHQALLHTANRPEDECVGEGLACHQLCARGHCWGPGPTQCVNCSQFLRGQECVEECRVLQGLPREYVNARHCLPCHPECQPQNGSVTCFGPEADQCVACAHYKDPPFCVARCPSGVKPDLSYMPIWKFPDEEGACQPCPINCTHSCVDLDDKGCPAEQRASPLTSIISAVVGILLVVVLGVVFGILIKRRQQKIRKYTMRRLLQETELVEPLTPSGAMPNQAQMRILKETELRKVKVLGSGAFGTVYKGIWIPDGENVKIPVAIKVLRENTSPKANKEILDEAYVMAGVGSPYVSRLLGICLTSTVQLVTQLMPYGCLLDHVRENRGRLGSQDLLNWCMQIAKGMSYLEDVRLVHRDLAARNVLVKSPNHVKITDFGLARLLDIDETEYHADGGKVPIKWMALESILRRRFTHQSDVWSYGVTVWELMTFGAKPYDGIPAREIPDLLEKGERGERPTEMPTPKANKECVQREAKSEKFGMGSSPKDS", 
            "BRCA2": "MPIGSKERPTFFEIFKTRCNKADLGPISLNWFEELSSEAPPYNSEPAEESEHKNNNYEPNLFKTPQRKPSYNQLASTPIIFKEQGLTLPLYQSPVKELDKFKLDLGRNVPNSRHKSLRTVKTKMDQADDVSCPLLNSCLSESPVVLQCTHVTPQRDKSVVCGSLFHTPKFVKGRQTPKHISESLGAEVDPDMSWSSSLATPPTLSSTVLIVRNEEASETVFPHDTTANVKSYFSNHDESLKKNDRFIASVTDSENTNQREAASHGFGKTSGNSFKVNSCKDHIGKSMPNVLEDEVYETVVDTSEEDSFSLCFSKCRTKNLQKVRTSKTRKKIFHEANADECEKSKNQVKEKYSFVSEVEPNDTDPLDSNVANQKPFESGSDKISKEVVPSLACEWSQLTLSGLNGAQMEKIPLLHISSCDQNISEKDLLDTENKRKKDFLTSENSLPRISSLPKSEKPLNEETVVNKRDEEQHLESHTDCILAVKQAISGTSPVASSFQGIKKSIFRIRESPKETFNASFSGHMTDPNFKKETEASESGLEIHTVCSQKEDSLCPNLIDNGSWPATTTQNSVALKNAGLISTLKKKTNKFIYAIHDETSYKGKKIPKDQKSELINCSAQFEANAFEAPLTFANADSGLLHSSVKRSCSQNDSEEPTLSLTSSFGTILRKCSRNETYYIKPNCLAPLPENQRAPSAPACLSLERPVLSLPVNPSSMGEPPVLCSFGERCQLPQSQTETPSSMGEIERNVTRVLTSSPHFAQAPQSRVDSNLKSPKPSQKHMSGSKQNSRVENESPKVKMETEATQSSPGGKNGVLRRKSCCESSNPNTTQLPHRHIFQSAVPGTPSPAYSRPLSTVSVAASTRNSGSRLQPHRSIFWEISENNCSTTTASSSNSSSLKNSKFIKPCNSIESLIYCNASSVKEKCSDNYSYAGTKKRASPIKKTVVSRRASQGAFSPSSGSSSQSASVDSSKGTMKKQKLSRESCSLSTQDSGSSTTSHGFVKESTSSTSFSEQDTDKCEDIQSSNQGSRRKRSYSLL"
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

# Helper endpoint to list available proteins
@app.get("/files")
async def list_available_proteins():
    return {
        "proteins": [
            {"id": "HER2", "file": "AF-P04626-F1-model_v4.pdb"},
            {"id": "BRCA2", "file": "AF-P51587-F1-model_v4.pdb"}
        ]
    }