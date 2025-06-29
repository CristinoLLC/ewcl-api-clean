import numpy as np
from Bio.PDB import PDBParser

def compute_qwip3d(coords: np.ndarray, radius: float = 10.0) -> np.ndarray:
    """
    Compute QWIP 3D score: localized angular structure interference.
    coords: (N, 3) array of Cα coordinates.
    Returns: normalized QWIP 3D array, shape (N,)
    """
    N = coords.shape[0]
    q3 = np.zeros(N)

    for i in range(N):
        center = coords[i]
        dists = np.linalg.norm(coords - center, axis=1)
        neighbors = coords[(dists > 0) & (dists < radius)]

        if len(neighbors) < 2:
            q3[i] = 0
            continue

        u = neighbors - center
        u = u / np.linalg.norm(u, axis=1, keepdims=True)
        dots = u @ u.T
        q3[i] = np.mean(np.abs(np.triu(dots, 1)))

    return (q3 - np.min(q3)) / (np.ptp(q3) + 1e-6)

def extract_ca_coords(pdb_path):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("m", pdb_path)
    coords, res_ids = [], []
    for res in next(structure[0].get_chains()):
        if "CA" in res:
            coords.append(res["CA"].coord)
            res_ids.append(res.id[1])
    return np.array(coords), res_ids

def run_qwip_on_pdb(pdb_path):
    coords, res_ids = extract_ca_coords(pdb_path)
    qwip = compute_qwip3d(coords)
    return [{"residue_id": r, "qwip_3d": float(q)} for r, q in zip(res_ids, qwip)]
