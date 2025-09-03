from fastapi import APIRouter, HTTPException, UploadFile, File, Body
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Union
import os, json, joblib, numpy as np, pandas as pd
from pathlib import Path
import io

router = APIRouter(prefix="/clinvar/ewclv1-C", tags=["clinvar-ewclv1-C"])

# Model configuration (standardize env var names + correct default filenames)
MODEL_PATH = os.environ.get("EWCLV1_C_MODEL_PATH", "/app/models/clinvar/ewclv1-c.pkl")
FEATURES_PATH = os.environ.get("EWCLV1_C_FEATURES_PATH", "/app/models/clinvar/ewclv1-c_features.json")

# Load model and features
MODEL = None
FEATURES = []
_MODEL_NAME = "ewclv1-C"

def _load_features():
    global FEATURES
    try:
        with open(FEATURES_PATH, 'r') as f:
            data = json.load(f)
        FEATURES = data if isinstance(data, list) else data.get("features", [])
        print(f"[info] Loaded {len(FEATURES)} features for EWCLv1-C")
    except Exception as e:
        print(f"[warn] Failed to load features from {FEATURES_PATH}: {e}")
        # Fallback to hardcoded feature list
        FEATURES = [
            "position", "sequence_length", "position_ratio", "delta_hydropathy", "delta_charge",
            "delta_entropy_w5", "delta_entropy_w11", "has_embeddings", "delta_helix_prop", 
            "delta_sheet_prop", "delta_entropy_w25", "ewcl_hydropathy", "ewcl_charge_pH7",
            "ewcl_entropy_w5", "ewcl_entropy_w11"
        ] + [f"emb_{i}" for i in range(32)]

def _load_model():
    global MODEL
    try:
        if Path(MODEL_PATH).exists():
            MODEL = joblib.load(MODEL_PATH)
            print(f"[info] Loaded EWCLv1-C model from {MODEL_PATH}")
        else:
            print(f"[warn] Model not found at {MODEL_PATH}")
    except Exception as e:
        print(f"[warn] Failed to load model from {MODEL_PATH}: {e}")

# Initialize
_load_features()
_load_model()

# Schemas
class VariantSample(BaseModel):
    variant_id: str
    position: Optional[int] = None
    ref: Optional[str] = None
    alt: Optional[str] = None
    sequence: Optional[str] = None
    features: Optional[Dict[str, float]] = None
    feature_array: Optional[List[float]] = None

class PredictRequest(BaseModel):
    samples: List[VariantSample] = Field(..., min_items=1)

class PredictItem(BaseModel):
    variant_id: str
    pathogenic_prob: float
    class_prediction: str
    confidence: float
    position: Optional[int] = None
    ref: Optional[str] = None
    alt: Optional[str] = None

class PredictResponse(BaseModel):
    model: str
    count: int
    variants: List[PredictItem]

# Feature engineering helpers - EWCLv1-C specific
AA_HYDRO = {  # Kyte-Doolittle hydropathy
    'A': 1.8, 'R': -4.5, 'N': -3.5, 'D': -3.5, 'C': 2.5, 'Q': -3.5, 'E': -3.5, 'G': -0.4, 'H': -3.2,
    'I': 4.5, 'L': 3.8, 'K': -3.9, 'M': 1.9, 'F': 2.8, 'P': -1.6, 'S': -0.8, 'T': -0.7, 'W': -0.9, 'Y': -1.3, 'V': 4.2,
    'X': 0.0
}

AA_CHARGE = {  # Charge at pH 7
    'R': 1, 'K': 1, 'H': 0.1, 'D': -1, 'E': -1,  # H is slightly positive at pH 7
    'A': 0, 'N': 0, 'C': 0, 'Q': 0, 'G': 0, 'I': 0, 'L': 0, 'M': 0, 'F': 0, 'P': 0, 'S': 0, 'T': 0, 'W': 0, 'Y': 0, 'V': 0,
    'X': 0.0
}

AA_HELIX = {  # Chou-Fasman helix propensity
    'A': 1.45, 'R': 0.79, 'N': 0.67, 'D': 1.01, 'C': 0.77, 'Q': 1.11, 'E': 1.51, 'G': 0.53, 'H': 1.24,
    'I': 1.08, 'L': 1.34, 'K': 1.16, 'M': 1.20, 'F': 1.13, 'P': 0.59, 'S': 0.79, 'T': 0.82, 'W': 1.14, 'Y': 0.61, 'V': 1.06,
    'X': 1.0
}

AA_SHEET = {  # Chou-Fasman sheet propensity
    'A': 0.97, 'R': 0.90, 'N': 0.89, 'D': 0.54, 'C': 1.30, 'Q': 1.10, 'E': 0.37, 'G': 0.81, 'H': 0.71,
    'I': 1.60, 'L': 1.22, 'K': 0.74, 'M': 1.67, 'F': 1.38, 'P': 0.62, 'S': 0.72, 'T': 1.20, 'W': 1.19, 'Y': 1.29, 'V': 1.70,
    'X': 1.0
}

def _local_entropy(seq: str, pos: int, window: int) -> float:
    """Calculate local sequence entropy around a position."""
    n = len(seq)
    start = max(0, pos - window // 2)
    end = min(n, pos + window // 2 + 1)
    subseq = seq[start:end]
    
    if not subseq:
        return 0.0
    
    counts = {}
    for aa in subseq:
        counts[aa] = counts.get(aa, 0) + 1
    
    total = len(subseq)
    probs = np.array(list(counts.values())) / total
    return -np.sum(probs * np.log2(probs + 1e-10))

def _build_features(sample: VariantSample) -> Dict[str, float]:
    """Build EWCLv1-C specific features for a variant sample."""
    features = {}
    
    # Start with provided features if any
    if sample.features:
        features.update(sample.features)
    
    # Calculate sequence-based features if we have the necessary data
    if sample.sequence and sample.position and sample.ref and sample.alt:
        seq = sample.sequence.upper()
        pos = sample.position - 1  # Convert to 0-based
        ref = sample.ref.upper()
        alt = sample.alt.upper()
        
        if 0 <= pos < len(seq) and seq[pos] == ref:
            # Basic position features
            features["position"] = sample.position
            features["sequence_length"] = len(seq)
            features["position_ratio"] = sample.position / len(seq)
            
            # Delta features (change caused by mutation)
            features["delta_hydropathy"] = AA_HYDRO.get(alt, 0) - AA_HYDRO.get(ref, 0)
            features["delta_charge"] = AA_CHARGE.get(alt, 0) - AA_CHARGE.get(ref, 0)
            features["delta_helix_prop"] = AA_HELIX.get(alt, 1) - AA_HELIX.get(ref, 1)
            features["delta_sheet_prop"] = AA_SHEET.get(alt, 1) - AA_SHEET.get(ref, 1)
            
            # Calculate entropy changes
            wt_seq = seq
            mut_seq = seq[:pos] + alt + seq[pos+1:]
            
            features["delta_entropy_w5"] = _local_entropy(mut_seq, pos, 5) - _local_entropy(wt_seq, pos, 5)
            features["delta_entropy_w11"] = _local_entropy(mut_seq, pos, 11) - _local_entropy(wt_seq, pos, 11)
            features["delta_entropy_w25"] = _local_entropy(mut_seq, pos, 25) - _local_entropy(wt_seq, pos, 25)
            
            # EWCL features (wild-type context)
            features["ewcl_hydropathy"] = AA_HYDRO.get(ref, 0)
            features["ewcl_charge_pH7"] = AA_CHARGE.get(ref, 0)
            features["ewcl_entropy_w5"] = _local_entropy(wt_seq, pos, 5)
            features["ewcl_entropy_w11"] = _local_entropy(wt_seq, pos, 11)
            
            # Embeddings placeholder (set to 0 if not provided)
            features["has_embeddings"] = 0.0  # No embeddings by default
            for i in range(32):
                if f"emb_{i}" not in features:
                    features[f"emb_{i}"] = 0.0
    
    # Ensure all required features are present
    result = {feat_name: features.get(feat_name, 0.0) for feat_name in FEATURES}
    return result

@router.get("/health")
def health():
    return {
        "model": _MODEL_NAME,
        "model_path": MODEL_PATH,
        "features_path": FEATURES_PATH,
        "model_loaded": MODEL is not None,
        "features_count": len(FEATURES),
        "ready": MODEL is not None and len(FEATURES) > 0
    }

@router.post("/analyze-variants")
def predict_variants(request: PredictRequest) -> PredictResponse:
    """Predict pathogenicity of variants using EWCLv1-C model."""
    if MODEL is None:
        raise HTTPException(503, "EWCLv1-C model not loaded")
    
    if not FEATURES:
        raise HTTPException(503, "Feature schema not loaded")
    
    results = []
    
    for sample in request.samples:
        try:
            # Use pre-computed feature array if provided, otherwise compute features
            if sample.feature_array and len(sample.feature_array) == len(FEATURES):
                feature_dict = {name: val for name, val in zip(FEATURES, sample.feature_array)}
            else:
                feature_dict = _build_features(sample)
            
            # Create DataFrame for prediction
            df = pd.DataFrame([feature_dict])
            
            # Make prediction
            if hasattr(MODEL, 'predict_proba'):
                prob = MODEL.predict_proba(df)[0, 1]  # Probability of pathogenic class
            elif hasattr(MODEL, 'predict'):
                prob = MODEL.predict(df)[0]
            else:
                prob = 0.5
            
            prob = float(np.clip(prob, 0.0, 1.0))
            
            # Calculate confidence (distance from 0.5)
            confidence = abs(prob - 0.5) * 2.0
            
            # Determine class
            class_pred = "Pathogenic" if prob > 0.5 else "Benign"
            
            results.append(PredictItem(
                variant_id=sample.variant_id,
                pathogenic_prob=prob,
                class_prediction=class_pred,
                confidence=confidence,
                position=sample.position,
                ref=sample.ref,
                alt=sample.alt
            ))
            
        except Exception as e:
            print(f"[error] Prediction failed for {sample.variant_id}: {e}")
            results.append(PredictItem(
                variant_id=sample.variant_id,
                pathogenic_prob=0.5,
                class_prediction="Unknown",
                confidence=0.0,
                position=sample.position,
                ref=sample.ref,
                alt=sample.alt
            ))
    
    return PredictResponse(
        model=_MODEL_NAME,
        count=len(results),
        variants=results
    )

@router.post("/analyze-variants/batch")
async def predict_variants_batch(file: UploadFile = File(...)) -> PredictResponse:
    """Batch predict variants from uploaded TSV/CSV file."""
    if MODEL is None:
        raise HTTPException(503, "EWCLv1-C model not loaded")
    
    try:
        content = await file.read()
        df = pd.read_csv(io.StringIO(content.decode('utf-8')), sep=None, engine='python')
        
        # Expected columns: variant_id, position, ref, alt, sequence (optional: features...)
        required_cols = ['variant_id', 'position', 'ref', 'alt']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise HTTPException(400, f"Missing required columns: {missing_cols}")
        
        samples = []
        for _, row in df.iterrows():
            sample = VariantSample(
                variant_id=str(row['variant_id']),
                position=int(row['position']) if pd.notna(row['position']) else None,
                ref=str(row['ref']) if pd.notna(row['ref']) else None,
                alt=str(row['alt']) if pd.notna(row['alt']) else None,
                sequence=str(row['sequence']) if 'sequence' in row and pd.notna(row['sequence']) else None,
                features={col: float(row[col]) for col in row.index if col.startswith('emb_') or col in FEATURES}
            )
            samples.append(sample)
        
        request = PredictRequest(samples=samples)
        return predict_variants(request)
        
    except Exception as e:
        raise HTTPException(400, f"Failed to process batch file: {e}")