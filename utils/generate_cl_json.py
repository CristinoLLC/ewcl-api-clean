import json
from pathlib import Path
from datetime import datetime
import sys
import os
from Bio.PDB import PDBParser

# Add parent directory to path to import models
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.collapse_likelihood import CollapseLikelihood

# -------- config --------
pdb_path  = Path("api/data/labels/AF-P10636-F1-model_v4.pdb")
json_out  = pdb_path.with_suffix(".json")          # overwrite the old JSON
lambda_   = 3.0                                    # same λ as backend
# ------------------------

cl_model  = CollapseLikelihood(lambda_)

# Extract pLDDT scores and calculate CL scores
plddt_scores = []
parser = PDBParser(QUIET=True)
structure = parser.get_structure("model", str(pdb_path))

for model in structure:
    for chain in model:
        for residue in chain:
            if "CA" in residue:
                bfactor = residue["CA"].get_bfactor()
                plddt_scores.append(bfactor)

cl_scores = cl_model.score(plddt_scores)

# Enhanced output with metadata and complete data
output = {
    "model": "CollapseLikelihood",
    "lambda": cl_model.lambda_,
    "generated": datetime.utcnow().isoformat() + "Z",
    "scores": [
        {
            "residue_id": i + 1,
            "cl": round(cl_score, 4),
            "plddt": plddt_scores[i],
            "b_factor": plddt_scores[i]
        }
        for i, cl_score in enumerate(cl_scores)
    ]
}

with open(json_out, "w") as fh:
    json.dump(output, fh, indent=2)

print(f"✅ Saved CL JSON to {json_out}")
print(f"📊 Generated {len(cl_scores)} CL scores with λ = {lambda_}")
print(f"📋 Format: residue_id, cl, plddt, b_factor")
