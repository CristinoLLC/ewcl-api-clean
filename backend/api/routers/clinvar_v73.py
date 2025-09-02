from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
import os, json, joblib, numpy as np, pandas as pd
from pathlib import Path
import lightgbm as lgb


router = APIRouter(prefix="/clinvar/v7_3", tags=["clinvar-v7.3"])

# ---------- Bundle loading ----------
BUNDLE = Path(os.environ.get("EWCLV1C_BUNDLE_DIR", str(Path(__file__).resolve().parents[3] / "backend_bundle")))

def _resolve_bundle_file() -> tuple[Optional[Path], Optional[Path], Optional[Path]]:
    models_dir = BUNDLE / "models"
    meta_dir = BUNDLE / "meta"
    # Try common names in priority order
    model_candidates = [
        models_dir / "clinvar_v7_3_model.pkl",
        models_dir / "EWCLv1C_Gate.pkl",
        models_dir / "EWCLv1-C_Gate.pkl",
        models_dir / "EWCLv1-C.pkl",
    ]
    feats_candidates = [
        meta_dir / "clinvar_v7_3_features.json",
        meta_dir / "EWCLv1-C_features.json",
        meta_dir / "EWCLv1C_features.json",
        meta_dir / "EWCLv1C_feature_info.json",
    ]
    calib_candidates = [
        meta_dir / "clinvar_v7_3_calibration.pkl",
        meta_dir / "EWCLv1-C_calibration.pkl",
        meta_dir / "EWCLv1C_calibration.pkl",
    ]
    m = next((p for p in model_candidates if p.exists()), None)
    f = next((p for p in feats_candidates if p.exists()), None)
    c = next((p for p in calib_candidates if p.exists()), None)
    # As a last resort, glob
    if m is None:
        picks = list(models_dir.glob("*v1*C*.*pkl"))
        if picks:
            m = picks[0]
    if f is None:
        picks = list(meta_dir.glob("*feature*json"))
        if picks:
            f = picks[0]
    if c is None:
        picks = list(meta_dir.glob("*calib*.*pkl"))
        if picks:
            c = picks[0]
    return m, f, c

_MODEL_PATH, _FEATS_PATH, _CALIB_PATH = _resolve_bundle_file()
SCHEMA_PATH = BUNDLE / "schema" / "clinvar_v7_3_smart_gate_features.json"

def _load_feature_list(path: Path) -> List[str]:
    obj = json.loads(path.read_text())
    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict):
        for key in ("feature_columns", "features", "all_features", "columns"):
            if key in obj and isinstance(obj[key], list):
                return obj[key]
        if obj and all(isinstance(v, int) for v in obj.values()):
            return [k for k, _ in sorted(obj.items(), key=lambda kv: kv[1])]
    raise ValueError(f"Unrecognized feature file format: {path}")

if not _MODEL_PATH or not _MODEL_PATH.exists() or not _FEATS_PATH or not _FEATS_PATH.exists():
    MODEL = None
    PREDICTOR = None
    SCALER = None
    THRESHOLDS: Dict[str, float] = {}
    FEATURES: List[str] = []
    CALIB = None
else:
    FEATURES = _load_feature_list(_FEATS_PATH)
    MODEL = joblib.load(_MODEL_PATH)
    # Optional external calibration file (fallback)
    CALIB = joblib.load(_CALIB_PATH) if (_CALIB_PATH and _CALIB_PATH.exists()) else None
    # If the model is a dict bundle, extract predictor, scaler, thresholds, and inline calibration
    PREDICTOR = None
    SCALER = None
    THRESHOLDS: Dict[str, float] = {}
    if isinstance(MODEL, dict):
        # Use the main ClinVar model (flip_detector) instead of gate logic
        PREDICTOR = MODEL.get("flip_detector") or MODEL.get("estimator") or MODEL.get("model")
        SCALER = MODEL.get("feature_scaler")
        if isinstance(MODEL.get("thresholds"), dict):
            THRESHOLDS = MODEL.get("thresholds") or {}
        # Inline calibration bundle can override CALIB if present
        if isinstance(MODEL.get("calibration_model"), dict):
            CALIB = MODEL.get("calibration_model")
        # Prefer feature names from predictor booster if available
        try:
            if isinstance(PREDICTOR, lgb.Booster):
                fn = PREDICTOR.feature_name()
                if fn and (not FEATURES or len(FEATURES) != len(fn)):
                    FEATURES = list(fn)
        except Exception:
            pass
    # If a dedicated schema file exists, override with it (authoritative order)
    try:
        if SCHEMA_PATH.exists():
            FEATURES = _load_feature_list(SCHEMA_PATH)
    except Exception:
        pass
    # If feature list missing or length mismatch, try to infer from model
    try:
        inferred = None
        if hasattr(MODEL, "feature_name_"):
            inferred = list(getattr(MODEL, "feature_name_"))
        elif hasattr(MODEL, "booster_") and hasattr(MODEL.booster_, "feature_name"):
            inferred = list(MODEL.booster_.feature_name())
        if inferred and (not FEATURES or len(FEATURES) != len(inferred)):
            FEATURES = inferred
    except Exception:
        pass
    # FINAL FALLBACK: parse feature names from an extractor module if present
    try:
        if not FEATURES:
            extractor = BUNDLE / "meta" / "ewcl_feature_extractor_v2.py"
            if extractor.exists():
                import re
                text = extractor.read_text()
                # Look for a Python list named FEATURE_COLUMNS or similar
                m = re.search(r"FEATURE[_A-Z]*\s*=\s*\[(.*?)\]", text, re.S)
                if m:
                    inner = m.group(1)
                    names = re.findall(r"'([^']+)'|\"([^\"]+)\"", inner)
                    flat = [a or b for a, b in names]
                    if flat:
                        FEATURES = flat
    except Exception:
        pass


# Optional defaults mirroring a conservative gate
BIO_COVERAGE_THR = 0.60
DEFAULT_CONF_THR = 0.80


# ---------- Schemas ----------
class VariantSample(BaseModel):
    variant_id: str
    uniprot_id: Optional[str] = None
    position: Optional[int] = None
    features: Optional[Dict[str, float]] = None
    feature_array: Optional[List[float]] = None
    biological_coverage_score: Optional[float] = None


class PredictRequest(BaseModel):
    samples: List[VariantSample] = Field(..., min_items=1)


class PredictItem(BaseModel):
    variant_id: str
    score: float
    gated_score: Optional[float] = None
    coverage: Optional[float] = None
    confidence: Optional[float] = None
    flag_low_coverage: Optional[bool] = None
    flag_trusted: Optional[bool] = None


class PredictResponse(BaseModel):
    model: str
    n: int
    feature_count: int
    items: List[PredictItem]


# ---------- Helpers ----------
def _df_from_request(req: PredictRequest) -> pd.DataFrame:
    if MODEL is None or not FEATURES:
        raise HTTPException(503, f"ClinVar model not available in bundle: {BUNDLE}")
    rows = []
    for s in req.samples:
        if s.features is not None:
            row = {f: float(s.features.get(f, 0.0)) for f in FEATURES}
        elif s.feature_array is not None:
            if len(s.feature_array) != len(FEATURES):
                raise HTTPException(400, f"feature_array length {len(s.feature_array)} != expected {len(FEATURES)}")
            row = {f: float(v) for f, v in zip(FEATURES, s.feature_array)}
        else:
            raise HTTPException(400, "Each sample must provide either 'features' or 'feature_array'.")
        row["_variant_id"] = s.variant_id
        row["_uniprot_id"] = s.uniprot_id
        row["_position"] = s.position
        row["_coverage"] = None if s.biological_coverage_score is None else float(s.biological_coverage_score)
        rows.append(row)
    return pd.DataFrame(rows)


def _predict_scores(df: pd.DataFrame) -> np.ndarray:
    X = df[FEATURES].copy()
    for c in X.columns:
        if not np.issubdtype(X[c].dtype, np.number):
            X[c] = pd.to_numeric(X[c], errors="coerce").fillna(0.0)
    # Apply optional scaler from bundle
    try:
        if 'SCALER' in globals() and SCALER is not None:
            X = pd.DataFrame(SCALER.transform(X), index=X.index, columns=X.columns)
    except Exception:
        pass
    predictor = PREDICTOR if PREDICTOR is not None else MODEL
    if isinstance(predictor, lgb.Booster):
        return predictor.predict(X)
    elif hasattr(predictor, "predict_proba"):
        return predictor.predict_proba(X)[:, 1]
    elif hasattr(predictor, "predict"):
        return predictor.predict(X)
    else:
        raise ValueError(f"Model object {type(predictor)} has no prediction method")


def _calibrated_confidence(scores: np.ndarray, coverage: np.ndarray) -> np.ndarray:
    extremeness = np.abs(scores - 0.5) * 2.0
    combined = 0.7 * extremeness + 0.3 * coverage
    if CALIB and isinstance(CALIB, dict) and "calibrator" in CALIB:
        calibrator = CALIB["calibrator"]
        try:
            return calibrator.transform(combined)
        except Exception:
            return combined
    return combined


def _gate(scores: np.ndarray, coverage: np.ndarray) -> Dict[str, np.ndarray]:
    # Prefer explicit thresholds in bundle; else calibration dict; else default
    if 'THRESHOLDS' in globals() and isinstance(THRESHOLDS, dict) and THRESHOLDS.get("confidence_threshold_80") is not None:
        conf_thr = float(THRESHOLDS.get("confidence_threshold_80", DEFAULT_CONF_THR))
    elif isinstance(CALIB, dict):
        conf_thr = float(CALIB.get("confidence_threshold_80", DEFAULT_CONF_THR))
    else:
        conf_thr = DEFAULT_CONF_THR
    low_cov = coverage < BIO_COVERAGE_THR
    conf = _calibrated_confidence(scores, coverage)
    high_conf = conf > conf_thr
    trusted = (~low_cov) | high_conf
    flagged = low_cov & (~high_conf)
    return {"confidence": conf, "low_cov": low_cov, "trusted": trusted, "flagged": flagged}


# ---------- Endpoints ----------
@router.get("/health")
def health():
    return {
        "bundle": str(BUNDLE),
        "model": str(_MODEL_PATH) if _MODEL_PATH else None,
        "features": len(FEATURES),
        "calibration": bool(CALIB is not None),
        "ready": bool(MODEL is not None and FEATURES),
    }


@router.get("/feature_schema")
def feature_schema():
    return {"count": len(FEATURES), "first_20": FEATURES[:20], "has_calibration": bool(CALIB is not None)}


# ---------- Minimal featurizer (delta features) ----------
AA_HYDRO = {
    'A': 1.8,'R': -4.5,'N': -3.5,'D': -3.5,'C': 2.5,'Q': -3.5,'E': -3.5,'G': -0.4,'H': -3.2,
    'I': 4.5,'L': 3.8,'K': -3.9,'M': 1.9,'F': 2.8,'P': -1.6,'S': -0.8,'T': -0.7,'W': -0.9,'Y': -1.3,'V': 4.2,
    'X': 0.0
}
AA_CHARGE = {'R': 1,'K': 1,'H': 0.5,'D': -1,'E': -1,'X': 0.0}

def _local_entropy_str(seq: str, pos0: int, win: int) -> float:
    n = len(seq)
    a, b = max(0, pos0 - win // 2), min(n, pos0 + win // 2 + 1)
    s = seq[a:b]
    if not s:
        return 0.0
    counts: Dict[str, int] = {}
    for ch in s:
        counts[ch] = counts.get(ch, 0) + 1
    tot = len(s)
    p = np.array(list(counts.values()), dtype=float) / float(tot)
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())

def _delta_feature_minimal(seq: str, pos1: int, ref: str, alt: str) -> Dict[str, float]:
    i = pos1 - 1
    if i < 0 or i >= len(seq):
        raise HTTPException(400, f"Position {pos1} out of range for sequence length {len(seq)}")
    dh = float(AA_HYDRO.get(alt, 0.0) - AA_HYDRO.get(ref, 0.0))
    dq = float(AA_CHARGE.get(alt, 0.0) - AA_CHARGE.get(ref, 0.0))
    e5_ref = _local_entropy_str(seq, i, 5)
    e11_ref = _local_entropy_str(seq, i, 11)
    mut = seq[:i] + alt + seq[i + 1:]
    e5_mut = _local_entropy_str(mut, i, 5)
    e11_mut = _local_entropy_str(mut, i, 11)
    return {
        "delta_hydropathy": dh,
        "delta_charge": dq,
        "delta_entropy_w5": float(e5_mut - e5_ref),
        "delta_entropy_w11": float(e11_mut - e11_ref),
    }

def _assemble_to_schema(schema: List[str], base: Dict[str, float]) -> Dict[str, float]:
    out = {name: 0.0 for name in schema}
    for k, v in base.items():
        if k in out:
            out[k] = float(v)
    return out

from pydantic import BaseModel, Field
from typing import Union

class FeaturizeRequest(BaseModel):
    sequence: str = Field(..., description="Protein sequence containing the variant site")
    position: int = Field(..., ge=1, description="1-based residue index")
    ref: str = Field(..., min_length=1, max_length=1)
    alt: str = Field(..., min_length=1, max_length=1)
    return_array: bool = Field(False)

class FeaturizeResponse(BaseModel):
    features: Union[Dict[str, float], List[float]]
    n_features: int

@router.get("/featurize_schema")
def featurize_schema():
    fills = [k for k in ["delta_hydropathy","delta_charge","delta_entropy_w5","delta_entropy_w11"] if k in FEATURES]
    return {"fills": fills, "n_filled": len(fills), "total_in_schema": len(FEATURES)}

@router.post("/featurize", response_model=FeaturizeResponse)
def featurize(req: FeaturizeRequest):
    seq = req.sequence.strip().upper()
    ref = req.ref.upper()
    alt = req.alt.upper()
    base = _delta_feature_minimal(seq, req.position, ref, alt)
    aligned = _assemble_to_schema(FEATURES, base)
    if req.return_array:
        arr = [aligned[name] for name in FEATURES]
        return FeaturizeResponse(features=arr, n_features=len(FEATURES))
    return FeaturizeResponse(features=aligned, n_features=len(FEATURES))


@router.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    df = _df_from_request(req)
    scores = _predict_scores(df)
    items = [
        PredictItem(variant_id=df["_variant_id"].iat[i], score=float(scores[i]))
        for i in range(len(df))
    ]
    return PredictResponse(model="EWCLv1-C_v7.3", n=len(items), feature_count=len(FEATURES), items=items)


@router.post("/predict_gated", response_model=PredictResponse)
def predict_gated(req: PredictRequest):
    df = _df_from_request(req)
    scores = _predict_scores(df)
    if df["_coverage"].notna().any():
        coverage = df["_coverage"].fillna(BIO_COVERAGE_THR).to_numpy(float)
    elif "biological_coverage_score" in FEATURES:
        coverage = df["biological_coverage_score"].fillna(BIO_COVERAGE_THR).to_numpy(float)
    else:
        coverage = np.full(len(df), BIO_COVERAGE_THR, dtype=float)
    gate = _gate(scores, coverage)
    items: List[PredictItem] = []
    for i in range(len(df)):
        items.append(PredictItem(
            variant_id=df["_variant_id"].iat[i],
            score=float(scores[i]),
            gated_score=float(scores[i]),
            coverage=float(coverage[i]),
            confidence=float(gate["confidence"][i]),
            flag_low_coverage=bool(gate["low_cov"][i]),
            flag_trusted=bool(gate["trusted"][i]),
        ))
    return PredictResponse(model="EWCLv1-C_v7.3+Gate", n=len(items), feature_count=len(FEATURES), items=items)


