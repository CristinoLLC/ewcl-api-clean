import os
import json
import numpy as np
from Bio.PDB import PDBParser

PDB_FOLDER = "api/data/labels"

def compute_ewcl_from_bfactors(b_factors):
    plddt_like = np.clip(np.array(b_factors) / 100.0, 1e-6, 1 - 1e-6)
    entropy = -(plddt_like * np.log(plddt_like) + (1 - plddt_like) * np.log(1 - plddt_like))
    ewcl = entropy / np.max(entropy)
    return ewcl.round(4).tolist()

def extract_bfactors_from_pdb(pdb_path):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("model", pdb_path)
    b_factors = []
    for model in structure:
        for chain in model:
            for residue in chain:
                if "CA" in residue:
                    b_factors.append(residue["CA"].get_bfactor())
    return b_factors

def process_all_pdbs():
    for file in os.listdir(PDB_FOLDER):
        if file.endswith(".pdb"):
            pdb_path = os.path.join(PDB_FOLDER, file)
            json_path = pdb_path.replace(".pdb", ".json")

            print(f"⚙️ Processing: {file}")
            b_factors = extract_bfactors_from_pdb(pdb_path)
            if not b_factors:
                print(f"⚠️  No B-factors found in: {file}")
                continue

            ewcl_scores = compute_ewcl_from_bfactors(b_factors)
            output = {"ewcl_score": ewcl_scores}

            with open(json_path, "w") as out_file:
                json.dump(output, out_file)
            print(f"✅ Saved: {os.path.basename(json_path)}")

if __name__ == "__main__":
    process_all_pdbs()