from __future__ import annotations
import json, hashlib, pickle
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


def load_all(bundle_dir: Path) -> Dict[str, ModelBundle]:
    models_dir = Path(bundle_dir) / "models"
    meta_dir = Path(bundle_dir) / "meta"
    manifest_path = Path(bundle_dir) / "models_manifest.json"
    hashes: Dict[str, str] = {}
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text())
            for m in manifest.get("models", []):
                hashes[m["path"].split("/")[-1]] = m.get("sha256")
        except Exception:
            pass

    bundles: Dict[str, ModelBundle] = {}
    m = models_dir / "EWCLv1-M.pkl"
    if m.exists():
        fi = meta_dir / "EWCLv1-M_feature_info.json"
        bundles["ewclv1m"] = ModelBundle("ewclv1m", m, fi, hashes.get("EWCLv1-M.pkl"))
    m = models_dir / "EWCLv1.pkl"
    if m.exists():
        fi = meta_dir / "EWCLv1_feature_info.json"
        bundles["ewclv1"] = ModelBundle("ewclv1", m, fi, hashes.get("EWCLv1.pkl"))
    return bundles


