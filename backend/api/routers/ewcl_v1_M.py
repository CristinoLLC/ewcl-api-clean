from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from typing import Dict, List
import os, io, joblib, numpy as np
from Bio import SeqIO
# Removed the problematic meta import since we're implementing the feature extraction directly

# Version safety check for EWCLv1-M model compatibility
try:
    from sklearn import __version__ as sklv
    if not sklv.startswith("1.7."):
        print(f"[ewcl-v1-m] ⚠️  Warning: EWCLv1-M expects scikit-learn 1.7.x, got {sklv}")
        print("[ewcl-v1-m] Model outputs may be inconsistent due to version mismatch")
except ImportError:
    print("[ewcl-v1-m] ⚠️  scikit-learn not available for version check")

# --- Load model once (env: EWCLV1_M_MODEL_PATH) ---
_MODEL = None
_FEATURE_ORDER: List[str] = []

def _load_model():
    global _MODEL, _FEATURE_ORDER
    if _MODEL is not None:
        return
    model_path = os.environ.get("EWCLV1_M_MODEL_PATH")
    if not model_path or not os.path.exists(model_path):
        raise RuntimeError(f"EWCLv1-M model not found. Set EWCLV1_M_MODEL_PATH (got: {model_path})")
    _MODEL = joblib.load(model_path)
    # try to discover feature order from the artifact
    # accept common names: feature_names_in_, feature_names_, required_features
    for attr in ("feature_names_in_", "feature_names_", "required_features", "FEATURES"):
        if hasattr(_MODEL, attr):
            val = getattr(_MODEL, attr)
            if isinstance(val, (list, tuple)):
                _FEATURE_ORDER = list(val)
                break
    # if still empty, fall back to a well-known sequence-only set we compute
    if not _FEATURE_ORDER:
        _FEATURE_ORDER = [
            "is_unknown_aa","hydropathy","polarity","vdw_volume","flexibility","bulkiness",
            "helix_prop","sheet_prop","charge_pH7","scd_local",
            "hydro_w5_mean","hydro_w5_std","hydro_w5_min","hydro_w5_max",
            "polar_w5_mean","polar_w5_std","polar_w5_min","polar_w5_max",
            "charge_w5_mean","charge_w5_std","charge_w5_min","charge_w5_max",
            "hydro_w11_mean","hydro_w11_std","polar_w11_mean","polar_w11_std",
            "charge_w11_mean","charge_w11_std"
        ]

# --- Basic amino-acid scales (Kyte-Doolittle etc.) ---
_AA = set("ARNDCQEGHILKMFPSTWYV")
HYDRO = { # Kyte-Doolittle
    'I':4.5,'V':4.2,'L':3.8,'F':2.8,'C':2.5,'M':1.9,'A':1.8,'G':-0.4,'T':-0.7,'S':-0.8,
    'W':-0.9,'Y':-1.3,'P':-1.6,'H':-3.2,'E':-3.5,'Q':-3.5,'D':-3.5,'N':-3.5,'K':-3.9,'R':-4.5
}
POLAR = { # arbitrary polarity scale (normalized-ish)
    'R':52,'K':49,'D':49,'E':49,'Q':41,'N':40,'H':51,'Y':41,'W':42,'S':32,'T':32,
    'G':0,'P':27,'C':15,'A':8,'M':5,'F':5,'L':4,'V':4,'I':5
}
VDW  = { # rough volumes
    'A':88.6,'R':173.4,'N':114.1,'D':111.1,'C':108.5,'Q':143.8,'E':138.4,
    'G':60.1,'H':153.2,'I':166.7,'L':166.7,'K':168.6,'M':162.9,'F':189.9,
    'P':112.7,'S':89.0,'T':116.1,'W':227.8,'Y':193.6,'V':140.0
}
FLEX = { # Bhaskaran normalized
    'G':1.00,'S':0.82,'D':0.80,'P':0.73,'N':0.73,'E':0.67,'Q':0.67,'K':0.62,
    'T':0.60,'R':0.60,'A':0.55,'W':0.54,'M':0.52,'H':0.52,'F':0.52,
    'Y':0.51,'I':0.47,'L':0.47,'V':0.46,'C':0.35
}
BULK = { # Zimmerman bulkiness
    'G':3.4,'A':11.5,'S':9.2,'P':17.4,'V':21.6,'T':15.9,'C':13.5,'I':21.4,'L':21.4,'D':13.0,
    'Q':17.2,'K':15.7,'E':12.3,'N':12.8,'H':21.0,'F':19.8,'Y':18.0,'M':16.3,'R':14.3,'W':21.6
}
HELIX = { # Chou-Fasman helix propensity (approx)
    'A':1.45,'C':0.77,'D':1.01,'E':1.51,'F':1.13,'G':0.53,'H':1.24,'I':1.08,'K':1.16,
    'L':1.34,'M':1.20,'N':0.67,'P':0.59,'Q':1.11,'R':0.79,'S':0.79,'T':0.82,'V':1.06,'W':1.14,'Y':0.61
}
SHEET = { # Chou-Fasman sheet propensity (approx)
    'A':0.97,'C':1.30,'D':0.54,'E':0.37,'F':1.38,'G':0.81,'H':0.71,'I':1.60,'K':0.74,
    'L':1.22,'M':1.67,'N':0.89,'P':0.62,'Q':1.10,'R':0.90,'S':0.72,'T':1.20,'V':1.70,'W':1.19,'Y':1.29
}
CHARGE = { # at pH 7
    'D':-1,'E':-1,'K':+1,'R':+1,'H':0,  # H ~ 0 at pH 7 (simplify)
    # others neutral:
    'A':0,'C':0,'Q':0,'N':0,'G':0,'I':0,'L':0,'M':0,'F':0,'P':0,'S':0,'T':0,'W':0,'Y':0,'V':0
}

def _sliding_stats(arr: np.ndarray, w: int):
    n = len(arr); hw = w//2
    mean = np.zeros(n); std = np.zeros(n)
    vmin = np.zeros(n); vmax = np.zeros(n)
    for i in range(n):
        s = max(0, i-hw); e = min(n, i+hw+1)
        win = arr[s:e]
        mean[i] = float(np.mean(win)) if len(win) else 0.0
        std[i]  = float(np.std(win)) if len(win) else 0.0
        vmin[i] = float(np.min(win)) if len(win) else 0.0
        vmax[i] = float(np.max(win)) if len(win) else 0.0
    return mean, std, vmin, vmax

def _entropy01(arr: np.ndarray, w: int, bins: int = 10):
    n = len(arr); hw = w//2
    out = np.zeros(n)
    for i in range(n):
        s = max(0, i-hw); e = min(n, i+hw+1)
        x = arr[s:e]
        if not len(x): 
            out[i] = 0.0
            continue
        # normalize 0..1 before histogram to stabilize
        xx = x
        if np.max(xx) > np.min(xx):
            xx = (xx - np.min(xx)) / (np.ptp(xx))
        hist, _ = np.histogram(xx, bins=bins, range=(0,1), density=True)
        p = hist + 1e-12
        p /= p.sum()
        out[i] = float(-(p * np.log(p)).sum())  # Shannon
    # scale 0..1 for readability
    if out.max() > out.min():
        out = (out - out.min()) / (out.max() - out.min())
    return out

def _sequencer_features(seq: str) -> Dict[int, Dict[str, float]]:
    """Compute per-residue sequence-only feature dicts keyed by 1-based index."""
    L = len(seq)
    aa = [c if c in _AA else 'X' for c in seq]

    hyd = np.array([HYDRO.get(a, 0.0) for a in aa], float)
    pol = np.array([POLAR.get(a, 0.0) for a in aa], float)
    vol = np.array([VDW.get(a, 0.0) for a in aa], float)
    flex= np.array([FLEX.get(a, 0.0) for a in aa], float)
    bulk= np.array([BULK.get(a, 0.0) for a in aa], float)
    hel = np.array([HELIX.get(a, 0.0) for a in aa], float)
    sht = np.array([SHEET.get(a, 0.0) for a in aa], float)
    chg = np.array([CHARGE.get(a, 0.0) for a in aa], float)

    # windows
    h5 = _sliding_stats(hyd, 5)
    p5 = _sliding_stats(pol, 5)
    c5 = _sliding_stats(chg, 5)
    h11m, h11s, _, _ = _sliding_stats(hyd, 11)
    p11m, p11s, _, _ = _sliding_stats(pol, 11)
    c11m, c11s, _, _ = _sliding_stats(chg, 11)

    # local charge decoration (very simple local |charge| density)
    scd = np.abs(c5[0])  # reuse mean(|charge|) at w=5 as a simple proxy

    # entropy overlays (for JSON only)
    hydro_ent = _entropy01(hyd, 11)
    charge_ent= _entropy01(np.abs(chg), 11)

    feats_by_pos = {}
    for i, a in enumerate(aa, start=1):
        feats = {
            "is_unknown_aa": 0.0 if a in _AA else 1.0,
            "hydropathy": float(hyd[i-1]),
            "polarity": float(pol[i-1]),
            "vdw_volume": float(vol[i-1]),
            "flexibility": float(flex[i-1]),
            "bulkiness": float(bulk[i-1]),
            "helix_prop": float(hel[i-1]),
            "sheet_prop": float(sht[i-1]),
            "charge_pH7": float(chg[i-1]),
            "scd_local": float(scd[i-1]),
            "hydro_w5_mean": float(h5[0][i-1]), "hydro_w5_std": float(h5[1][i-1]),
            "hydro_w5_min": float(h5[2][i-1]),  "hydro_w5_max": float(h5[3][i-1]),
            "polar_w5_mean": float(p5[0][i-1]), "polar_w5_std": float(p5[1][i-1]),
            "polar_w5_min": float(p5[2][i-1]),  "polar_w5_max": float(p5[3][i-1]),
            "charge_w5_mean": float(c5[0][i-1]),"charge_w5_std": float(c5[1][i-1]),
            "charge_w5_min": float(c5[2][i-1]), "charge_w5_max": float(c5[3][i-1]),
            "hydro_w11_mean": float(h11m[i-1]), "hydro_w11_std": float(h11s[i-1]),
            "polar_w11_mean": float(p11m[i-1]), "polar_w11_std": float(p11s[i-1]),
            "charge_w11_mean": float(c11m[i-1]),"charge_w11_std": float(c11s[i-1]),
            # following used only for response overlay:
            "_hydro_entropy": float(hydro_ent[i-1]),
            "_charge_entropy": float(charge_ent[i-1]),
        }
        feats_by_pos[i] = feats
    return feats_by_pos

def _build_X_for_model(feats_by_pos: Dict[int, Dict[str, float]]):
    # Ensure consistent order & drop unknowns. Fill missing with 0.0
    X = []
    for i in sorted(feats_by_pos.keys()):
        f = feats_by_pos[i]
        row = [float(f.get(k, 0.0)) for k in _FEATURE_ORDER]
        X.append(row)
    return np.asarray(X, dtype=float)

router = APIRouter(prefix="/ewcl", tags=["ewcl"])

@router.post("/analyze-fasta/ewclv1-M")
async def analyze_fasta_ewclv1_M(file: UploadFile = File(...)):
    try:
        _load_model()
        raw = await file.read()
        try:
            record = next(SeqIO.parse(io.StringIO(raw.decode("utf-8")), "fasta"))
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid FASTA")
        seq_id = record.id
        seq = str(record.seq).upper()
        if not seq:
            raise HTTPException(status_code=400, detail="Empty sequence")

        feats_by_pos = _sequencer_features(seq)
        X = _build_X_for_model(feats_by_pos)

        # predict (binary or proba)
        if hasattr(_MODEL, "predict_proba"):
            y = _MODEL.predict_proba(X)
            # assume class 1 = disorder/collapse
            if y.ndim == 2 and y.shape[1] > 1:
                cl = y[:, 1]
            else:
                cl = y.ravel()
        else:
            # regressors or decision_function
            if hasattr(_MODEL, "decision_function"):
                z = _MODEL.decision_function(X)
                cl = 1 / (1 + np.exp(-z))
            else:
                z = _MODEL.predict(X)
                # clip to [0,1] if necessary
                cl = np.clip(z, 0.0, 1.0)

        # naive confidence: 1 - |0.5 - cl|*2  (0 near 0.5; 1 near 0 or 1)
        conf = 1.0 - np.abs(cl - 0.5) * 2.0
        conf = np.clip(conf, 0.0, 1.0)

        residues = []
        for i, a in enumerate(seq, start=1):
            f = feats_by_pos[i]
            residues.append({
                "residue_index": i,
                "aa": a,
                "cl": float(cl[i-1]),
                "confidence": float(conf[i-1]),
                "hydro_entropy": float(f["_hydro_entropy"]),
                "charge_entropy": float(f["_charge_entropy"]),
                "curvature": None,   # not available for sequence-only
                "flips": 0.0
            })

        out = {
            "id": seq_id,
            "model": "ewclv1-M",
            "length": len(seq),
            "residues": residues
        }
        return JSONResponse(content=out)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"ewclv1-M FASTA analysis failed: {e}")