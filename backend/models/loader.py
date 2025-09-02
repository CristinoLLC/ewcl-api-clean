from __future__ import annotations
import json, hashlib, pickle, os  # added os
from pathlib import Path
from typing import Optional, Dict, Any
import pandas as pd
import joblib


class ModelBundle:
    def __init__(self, name: str, model_path: Path, feature_info_path: Optional[Path] = None, sha256: Optional[str] = None):
        self.name = name
        self.model_path = model_path
        self.feature_info_path = feature_info_path
        self.sha256_expected = sha256
        self.model = None
        self.feature_info: Dict[str, Any] = {}
        self._load()

    def _sha256(self, p: Path) -> str:
        h = hashlib.sha256()
        with open(p, 'rb') as f:
            for chunk in iter(lambda: f.read(1 << 20), b''):
                h.update(chunk)
        return h.hexdigest()

    def _load(self):
        if self.sha256_expected:
            digest = self._sha256(self.model_path)
            if digest != self.sha256_expected:
                raise ValueError(f"[{self.name}] SHA256 mismatch: {digest} != {self.sha256_expected}")

        try:
            self.model = joblib.load(self.model_path)
        except Exception:
            with open(self.model_path, 'rb') as f:
                self.model = pickle.load(f)

        if self.feature_info_path and self.feature_info_path.exists():
            self.feature_info = json.loads(self.feature_info_path.read_text())
            # Normalize various shapes of feature manifests
            try:
                if isinstance(self.feature_info, list):
                    self.feature_info = {"all_features": list(self.feature_info)}
                elif isinstance(self.feature_info, dict):
                    if "all_features" in self.feature_info and isinstance(self.feature_info["all_features"], list):
                        pass
                    elif "features" in self.feature_info and isinstance(self.feature_info["features"], list):
                        self.feature_info = {"all_features": list(self.feature_info["features"])}
                    elif "names" in self.feature_info and isinstance(self.feature_info["names"], list):
                        self.feature_info = {"all_features": list(self.feature_info["names"])}
                    else:
                        # name->index mapping
                        if all(isinstance(k, str) for k in self.feature_info.keys()) and \
                           all(isinstance(v, int) for v in self.feature_info.values()):
                            ordered = [k for k, _ in sorted(self.feature_info.items(), key=lambda kv: kv[1])]
                            self.feature_info = {"all_features": ordered}
            except Exception:
                pass
        # Try to infer feature list from model if not provided
        try:
            if not self.feature_info.get("all_features"):
                inferred: Optional[list[str]] = None
                if hasattr(self.model, "feature_name_"):
                    inferred = list(getattr(self.model, "feature_name_"))
                elif hasattr(self.model, "booster_") and hasattr(self.model.booster_, "feature_name"):
                    inferred = list(self.model.booster_.feature_name())
                if inferred:
                    self.feature_info["all_features"] = inferred
        except Exception:
            # Non-fatal; keep running without inferred features
            pass

    def ensure_feature_order(self, X: pd.DataFrame) -> pd.DataFrame:
        feats = self.feature_info.get("all_features")
        if feats:
            missing = [c for c in feats if c not in X.columns]
            if missing:
                raise ValueError(f"[{self.name}] Missing features: {missing[:10]}{'...' if len(missing)>10 else ''}")
            X = X.loc[:, feats]
        return X

    def predict_proba(self, X: pd.DataFrame) -> pd.Series:
        X = self.ensure_feature_order(X)
        proba = self.model.predict_proba(X)[:, 1]
        return pd.Series(proba, index=X.index, name=f"{self.name}_p")


def _env_model_paths() -> Dict[str, tuple[Path, Optional[Path]]]:
    """Collect explicit model paths from environment variables.
    Expected env vars:
      EWCLV1_MODEL_PATH, EWCLV1M_MODEL_PATH, EWCLV1P3_MODEL_PATH, EWCLV1C_MODEL_PATH
    For each model we try to auto-detect an adjacent feature info json if present.
    """
    mapping: Dict[str, tuple[Path, Optional[Path]]] = {}
    spec = {
        "ewclv1m": ("EWCLV1M_MODEL_PATH", ["EWCLv1-M_feature_info.json", "EWCLv1M_feature_info.json"]),
        "ewclv1": ("EWCLV1_MODEL_PATH", ["EWCLv1_feature_info.json", "EWCLv1_feature_info.json"]),
        "ewclv1p3": ("EWCLV1P3_MODEL_PATH", ["EWCLv1-P3_features.json", "EWCLv1-P3_feature_info.json"]),
        "ewclv1c": ("EWCLV1C_MODEL_PATH", [
            "EWCLv1C_feature_list.json",
            "EWCLv1C_feature_info.json",
            "v7_3_feature_list.json",
            "clinvar_v7_3_features.json"
        ]),
    }
    for name, (env_var, feature_candidates) in spec.items():
        p = os.environ.get(env_var)
        if not p:
            continue
        model_path = Path(p)
        if not model_path.exists():
            print(f"[warn] {env_var}={model_path} does not exist; skipping")
            continue
        feat_path: Optional[Path] = None
        for cand in feature_candidates:
            cpath = model_path.parent / cand
            if cpath.exists():
                feat_path = cpath
                break
        mapping[name] = (model_path, feat_path)
    return mapping


def load_all(bundle_dir: Path) -> Dict[str, ModelBundle]:
    # First, consider explicit env model paths (override bundle models)
    bundles: Dict[str, ModelBundle] = {}
    env_models = _env_model_paths()
    for name, (model_path, feat_path) in env_models.items():
        try:
            bundles[name] = ModelBundle(name, model_path, feat_path, None)
            print(f"[info] Loaded model '{name}' from env path {model_path}")
        except Exception as e:
            print(f"[warn] Failed loading env model {name}: {e}")

    # If all four provided via env, skip bundle scan
    if len(bundles) == 4:
        return bundles

    models_dir = Path(bundle_dir) / "models"
    meta_dir = Path(bundle_dir) / "meta"
    manifest_path = Path(bundle_dir) / "models_manifest.json"
    hashes: Dict[str, str] = {}
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text())
            models_section = manifest.get("models")
            if isinstance(models_section, dict):
                for key, entry in models_section.items():
                    file_field = entry.get("file") or entry.get("path")
                    if file_field:
                        hashes[Path(file_field).name] = entry.get("sha256")
            elif isinstance(models_section, list):
                for m in models_section:
                    file_field = m.get("file") or m.get("path")
                    if file_field:
                        hashes[Path(file_field).name] = m.get("sha256")
        except Exception:
            pass

    # Only add bundle models not already loaded via env override
    def _add(name: str, model_filename: str, feature_files: list[str]):
        if name in bundles:
            return
        m = models_dir / model_filename
        if m.exists():
            fi: Optional[Path] = None
            for f in feature_files:
                cand = meta_dir / f
                if cand.exists():
                    fi = cand
                    break
            try:
                bundles[name] = ModelBundle(name, m, fi, hashes.get(model_filename))
            except Exception as e:
                print(f"[warn] Failed loading bundle model {name}: {e}")

    _add("ewclv1m", "EWCLv1-M.pkl", ["EWCLv1-M_feature_info.json"])    
    _add("ewclv1", "EWCLv1.pkl", ["EWCLv1_feature_info.json"])        
    _add("ewclv1p3", "EWCLv1-P3.pkl", ["EWCLv1-P3_features.json", "EWCLv1-P3_feature_info.json"])  
    _add("ewclv1c", "EWCLv1C_Gate.pkl", ["EWCLv1C_feature_list.json", "EWCLv1C_feature_info.json"])  

    # Special handling for P3 CSV features if still missing
    if "ewclv1p3" in bundles and not bundles["ewclv1p3"].feature_info.get("all_features"):
        try:
            csv_path = meta_dir / "EWCLv1-P3_features_for_backend.csv"
            if csv_path.exists():
                df = pd.read_csv(csv_path)
                col = df.columns[0]
                names = df[col].astype(str).tolist()
                bundles["ewclv1p3"].feature_info = {"all_features": names}
        except Exception:
            pass

    return bundles


