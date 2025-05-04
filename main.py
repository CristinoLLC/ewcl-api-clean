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

# ✅ Fix: Attach the SequenceRequest as input body
@app.post("/runeucl")
def run_ewcl(req: SequenceRequest):
    try:
        result = ewcl_score_protein(req.sequence)
        return {"ewcl_map": result}
    except Exception as e:
        return {"error": str(e)}

# Existing analyze endpoint using sequence string
@app.post("/analyze")
def analyze(req: SequenceRequest):
    try:
        result = ewcl_score_protein(req.sequence)
        return {"ewcl_map": result, "status": "success"}
    except Exception as e:
        return {"error": str(e), "status": "error"}

# New endpoint for file upload analysis
@app.post("/analyze-file")
async def analyze_file(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        sequence = contents.decode("utf-8")
        # Extract protein sequence from PDB or other format if needed
        # This is simplified - you might need more parsing logic for PDB files
        result = ewcl_score_protein(sequence)
        return {
            "filename": file.filename,
            "ewcl_map": result,
            "status": "success"
        }
    except Exception as e:
        return {"error": str(e), "status": "error"}

# New endpoint for protein ID-based analysis
@app.post("/analyze/{protein_id}")
async def analyze_by_id(protein_id: str):
    try:
        # Here you would typically look up the protein by ID
        # For example, fetch it from a database or local file system
        protein_sequences = {
            "HER2": "MKLRLPASPETHLDMLRHLYQGCQVVQGNLELTYLPTNASLSFLQDIQEVQGYVLIAHNQVRQVPLQRLRIVRGTQLFEDNYALAVLDNGDPLNNTTPVTGASPGGLRELQLRSLTEILKGGVLIQRNPQLCYQDTILWKDIFHKNNQLALTLIDTNRSRACHPCSPMCKGSRCWGESSEDCQSLTRTVCAGGCARCKGPLPTDCCHEQCAAGCTGPKHSDCLACLHFNHSGICELHCPALVTYNTDTFESMPNPEGRYTFGASCVTACPYNYLSTDVGSCTLVCPLHNQEVTAEDGTQRCEKCSKPCARVCYGLGMEHLREVRAVTSANIQEFAGCKKIFGSLAFLPESFDGDPASNTAPLQPEQLQVFETLEEITGYLYISAWPDSLPDLSVFQNLQVIRGRILHNGAYSLTLQGLGISWLGLRSLRELGSGLALIHHNTHLCFVHTVPWDQLFRNPHQALLHTANRPEDECVGEGLACHQLCARGHCWGPGPTQCVNCSQFLRGQECVEECRVLQGLPREYVNARHCLPCHPECQPQNGSVTCFGPEADQCVACAHYKDPPFCVARCPSGVKPDLSYMPIWKFPDEEGACQPCPINCTHSCVDLDDKGCPAEQRASPLTSIISAVVGILLVVVLGVVFGILIKRRQQKIRKYTMRRLLQETELVEPLTPSGAMPNQAQMRILKETELRKVKVLGSGAFGTVYKGIWIPDGENVKIPVAIKVLRENTSPKANKEILDEAYVMAGVGSPYVSRLLGICLTSTVQLVTQLMPYGCLLDHVRENRGRLGSQDLLNWCMQIAKGMSYLEDVRLVHRDLAARNVLVKSPNHVKITDFGLARLLDIDETEYHADGGKVPIKWMALESILRRRFTHQSDVWSYGVTVWELMTFGAKPYDGIPAREIPDLLEKGERGERPTEMPTPKANKECVQREAKSEKFGMGSSPKDS", 
            "BRCA2": "MPIGSKERPTFFEIFKTRCNKADLGPISLNWFEELSSEAPPYNSEPAEESEHKNNNYEPNLFKTPQRKPSYNQLASTPIIFKEQGLTLPLYQSPVKELDKFKLDLGRNVPNSRHKSLRTVKTKMDQADDVSCPLLNSCLSESPVVLQCTHVTPQRDKSVVCGSLFHTPKFVKGRQTPKHISESLGAEVDPDMSWSSSLATPPTLSSTVLIVRNEEASETVFPHDTTANVKSYFSNHDESLKKNDRFIASVTDSENTNQREAASHGFGKTSGNSFKVNSCKDHIGKSMPNVLEDEVYETVVDTSEEDSFSLCFSKCRTKNLQKVRTSKTRKKIFHEANADECEKSKNQVKEKYSFVSEVEPNDTDPLDSNVANQKPFESGSDKISKEVVPSLACEWSQLTLSGLNGAQMEKIPLLHISSCDQNISEKDLLDTENKRKKDFLTSENSLPRISSLPKSEKPLNEETVVNKRDEEQHLESHTDCILAVKQAISGTSPVASSFQGIKKSIFRIRESPKETFNASFSGHMTDPNFKKETEASESGLEIHTVCSQKEDSLCPNLIDNGSWPATTTQNSVALKNAGLISTLKKKTNKFIYAIHDETSYKGKKIPKDQKSELINCSAQFEANAFEAPLTFANADSGLLHSSVKRSCSQNDSEEPTLSLTSSFGTILRKCSRNETYYIKPNCLAPLPENQRAPSAPACLSLERPVLSLPVNPSSMGEPPVLCSFGERCQLPQSQTETPSSMGEIERNVTRVLTSSPHFAQAPQSRVDSNLKSPKPSQKHMSGSKQNSRVENESPKVKMETEATQSSPGGKNGVLRRKSCCESSNPNTTQLPHRHIFQSAVPGTPSPAYSRPLSTVSVAASTRNSGSRLQPHRSIFWEISENNCSTTTASSSNSSSLKNSKFIKPCNSIESLIYCNASSVKEKCSDNYSYAGTKKRASPIKKTVVSRRASQGAFSPSSGSSSQSASVDSSKGTMKKQKLSRESCSLSTQDSGSSTTSHGFVKESTSSTSFSEQDTDKCEDIQSSNQGSRRKRSYSLL