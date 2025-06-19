import json
from pathlib import Path
from datetime import datetime
import sys
import os

# Add parent directory to path to import models
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.collapse_likelihood import CollapseLikelihood

# -------- config --------
pdb_path  = Path("api/data/labels/AF-P10636-F1-model_v4.pdb")
json_out  = pdb_path.with_suffix(".json")          # overwrite the old JSON
lambda_   = 3.0                                    # same λ as backend
# ------------------------

cl_model  = CollapseLikelihood(lambda_)
scores    = cl_model.score_from_pdb(pdb_path)

# Enhanced output with metadata
output = {
    "model": "CollapseLikelihood",
    "lambda": cl_model.lambda_,
    "generated": datetime.utcnow().isoformat() + "Z",
    "scores": [
        {"residue_id": i + 1, "cl": round(score, 4)}
        for i, score in enumerate(scores)
    ]
}

with open(json_out, "w") as fh:
    json.dump(output, fh, indent=2)

print(f"✅ Saved CL JSON to {json_out}")
print(f"📊 Generated {len(scores)} CL scores with λ = {lambda_}")
