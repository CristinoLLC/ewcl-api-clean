from __future__ import annotations
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
import io, os, json, joblib
import pandas as pd
from pathlib import Path
from backend.api.utils import clinvar_parser as cp
import lightgbm as lgb

router = APIRouter(prefix="/clinvar", tags=["clinvar-analyze"])

# Configure bundle directory via env; expect model + feature list inside
BUNDLE = Path(os.environ.get("EWCLV1C_BUNDLE_DIR", str(Path(__file__).resolve().parents[3] / "backend_bundle")))
MODEL_PATH = BUNDLE / "models" / "ewclv1c_model.pkl"
FEATS_PATH = BUNDLE / "meta" / "EWCLv1C_feature_list.json"

MODEL = None
FEATURE_ORDER: list[str] = []
FEATURE_SOURCE = ""
try:
    if MODEL_PATH.exists():
        MODEL = joblib.load(MODEL_PATH)
    if FEATS_PATH.exists():
        meta = json.loads(FEATS_PATH.read_text())
        FEATURE_ORDER = meta.get("features") or meta.get("feature_columns") or meta.get("all_features") or []
        FEATURE_SOURCE = "feature_list_json"
    # Fallback: infer from model if possible
    def _infer_model_features(m) -> list[str]:
        try:
            if hasattr(m, "feature_names_in_"):
                return list(getattr(m, "feature_names_in_"))
            if hasattr(m, "feature_name_"):
                return list(getattr(m, "feature_name_"))
            if isinstance(m, lgb.Booster):
                fn = m.feature_name()
                return list(fn) if fn else []
            if hasattr(m, "booster_") and isinstance(m.booster_, lgb.Booster):
                fn = m.booster_.feature_name()
                return list(fn) if fn else []
        except Exception:
            return []
        return []
    if MODEL is not None:
        model_feats = _infer_model_features(MODEL)
        if model_feats:
            if not FEATURE_ORDER:
                FEATURE_ORDER = model_feats
                FEATURE_SOURCE = "model_introspection"
            else:
                set_json, set_model = set(FEATURE_ORDER), set(model_feats)
                if set_json != set_model or len(FEATURE_ORDER) != len(model_feats):
                    print(f"[warn] ClinVar feature mismatch JSON({len(FEATURE_ORDER)}) vs MODEL({len(model_feats)})")
                    common = [f for f in FEATURE_ORDER if f in set_model]
                    if len(common) >= 1:
                        FEATURE_ORDER = common
                        FEATURE_SOURCE = "json∩model"
                    else:
                        FEATURE_ORDER = model_feats
                        FEATURE_SOURCE = "model_introspection_forced"
except Exception as e:
    print(f"[warn] ClinVar analyze loader: {e}")


@router.post("/analyze")
async def analyze(file: UploadFile = File(...), file_type: str = Form("json")):
    if MODEL is None or not FEATURE_ORDER:
        raise HTTPException(503, f"ClinVar model or feature list not available in {BUNDLE}")
    try:
        content = (await file.read()).decode("utf-8")
        if file_type == "json":
            seq, variants = cp.parse_json_variants(content)
        elif file_type == "fasta":
            seq = cp.parse_fasta(content)
            variants = []
            return {"sequence": seq, "variants": variants, "note": "Provide variants via VCF/JSON to score."}
        elif file_type == "vcf":
            variants = cp.parse_vcf(content)
            return {"variants": variants, "note": "Provide sequence via FASTA to score."}
        else:
            raise HTTPException(400, "Unsupported file_type; use json, fasta, or vcf")

        X = cp.build_features(seq, variants, FEATURE_ORDER)
        cols = list(X.columns)
        missing = [f for f in FEATURE_ORDER if f not in cols]
        extra = [c for c in cols if c not in FEATURE_ORDER]
        if missing or extra:
            raise HTTPException(400, f"Feature columns mismatch; missing={missing[:5]}, extra={extra[:5]}")
        probs = MODEL.predict_proba(X)[:, 1] if hasattr(MODEL, "predict_proba") else MODEL.predict(X)
        items = []
        for v, p in zip(variants, probs):
            items.append({
                "position": int(v["pos"]),
                "ref": str(v["ref"]),
                "alt": str(v["alt"]),
                "pathogenic_prob": float(p),
                "class": "Pathogenic" if float(p) > 0.5 else "Benign",
            })
        return JSONResponse(content={
            "model": "EWCLv1-C",
            "n_variants": len(items),
            "feature_count": len(FEATURE_ORDER),
            "predictions": items,
        })
    except Exception as e:
        raise HTTPException(500, f"ClinVar analyze failed: {e}")


@router.post("/analyze-variants")
async def analyze_variants(fasta_file: UploadFile = File(...), vcf_file: UploadFile = File(...)):
    if MODEL is None or not FEATURE_ORDER:
        raise HTTPException(503, f"ClinVar model or feature list not available in {BUNDLE}")
    try:
        fasta_content = (await fasta_file.read()).decode("utf-8")
        vcf_content = (await vcf_file.read()).decode("utf-8")
        seq = cp.parse_fasta(fasta_content)
        variants = cp.parse_vcf(vcf_content)
        if not seq:
            raise HTTPException(400, "No sequence parsed from FASTA")
        if not variants:
            raise HTTPException(400, "No variants parsed from VCF")

        X = cp.build_features(seq, variants, FEATURE_ORDER)
        cols = list(X.columns)
        missing = [f for f in FEATURE_ORDER if f not in cols]
        extra = [c for c in cols if c not in FEATURE_ORDER]
        if missing or extra:
            raise HTTPException(400, f"Feature columns mismatch; missing={missing[:5]}, extra={extra[:5]}")
        probs = MODEL.predict_proba(X)[:, 1] if hasattr(MODEL, "predict_proba") else MODEL.predict(X)
        items = []
        for v, p in zip(variants, probs):
            items.append({
                "position": int(v["pos"]),
                "ref": str(v["ref"]),
                "alt": str(v["alt"]),
                "pathogenic_prob": float(p),
                "class": "Pathogenic" if float(p) > 0.5 else "Benign",
            })
        return JSONResponse(content={
            "model": "EWCLv1-C",
            "sequence_length": len(seq),
            "n_variants": len(items),
            "feature_count": len(FEATURE_ORDER),
            "predictions": items,
        })
    except Exception as e:
        raise HTTPException(500, f"ClinVar analyze-variants failed: {e}")


@router.get("/feature_order")
def feature_order():
    return {"count": len(FEATURE_ORDER), "source": FEATURE_SOURCE, "first_20": FEATURE_ORDER[:20]}


