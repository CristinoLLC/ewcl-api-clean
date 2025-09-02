import os, joblib
from pathlib import Path

def _first_existing(paths):
    """Return the first existing path from a list of candidates."""
    for p in paths:
        if p and Path(p).exists():
            return str(Path(p))
    return None

def load_clinvar_bundle():
    """Load ClinVar model using env vars with fallbacks. Features are hardcoded in parser."""
    model_path = _first_existing([
        os.getenv("EWCLV1_C_MODEL_PATH"),
        "/Users/lucascristino/ewcl-api-clean/models/clinvar/ewclv1-C.pkl",
        "./models/clinvar/ewclv1-C.pkl",
        "./models/clinvar/ewclv1c.pkl",
    ])
    
    if not model_path:
        raise RuntimeError(f"ClinVar model not found. Tried: EWCLV1_C_MODEL_PATH env var and fallback paths")

    model = joblib.load(model_path)
    print(f"[init] ClinVar loaded: {Path(model_path).name}")
    return model

# Global variable for ClinVar
CLINVAR_MODEL = None

def get_clinvar_model():
    """Get the loaded ClinVar model, loading if necessary."""
    global CLINVAR_MODEL
    if CLINVAR_MODEL is None:
        CLINVAR_MODEL = load_clinvar_bundle()
    return CLINVAR_MODEL