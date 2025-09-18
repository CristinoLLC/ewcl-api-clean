"""
EWCLp3 Prediction Service
========================

Interface to the EWCLp3 model for structure-based disorder prediction.
"""

import numpy as np
from typing import Dict, List
from backend.models.loader import load_model_forgiving
from backend.api.routers.ewclv1p3_fresh import FeatureExtractor, load_structure_unified
import os


class EWCLp3Service:
    """Service for EWCLp3 predictions."""
    
    def __init__(self):
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load the EWCLp3 model."""
        path = os.environ.get("EWCLV1_P3_MODEL_PATH", "/app/models/pdb/ewclv1p3.pkl")
        if not os.path.exists(path):
            raise FileNotFoundError(f"EWCLp3 model not found at {path}")
        
        self.model = load_model_forgiving(path)
    
    def predict(self, structure_path: str, chain_id: str) -> Dict[int, float]:
        """
        Predict EWCL scores for a structure chain.
        
        Args:
            structure_path: Path to PDB/CIF file
            chain_id: Chain identifier
        
        Returns:
            Dict mapping position to EWCL score [0,1]
        """
        if self.model is None:
            raise RuntimeError("EWCLp3 model not loaded")
        
        # Read and parse structure
        with open(structure_path, 'rb') as f:
            raw_bytes = f.read()
        
        pdb_data = load_structure_unified(raw_bytes)
        
        # Filter to requested chain
        chain_residues = [r for r in pdb_data["residues"] if chain_id in pdb_data.get("chain", "A")]
        if not chain_residues:
            raise ValueError(f"Chain {chain_id} not found in structure")
        
        # Extract sequence and confidence
        sequence = [r["aa"] for r in chain_residues]
        confidence = [r["bfactor"] for r in chain_residues]
        
        # Generate features
        extractor = FeatureExtractor(sequence, confidence, pdb_data["source"])
        feature_matrix = extractor.extract_all_features()
        
        # Make predictions
        from backend.api.routers.ewclv1p3_fresh import FEATURE_NAMES
        X = feature_matrix[FEATURE_NAMES].values
        
        if hasattr(self.model, "predict_proba"):
            probabilities = self.model.predict_proba(X)
            predictions = probabilities[:, 1] if probabilities.ndim == 2 else probabilities
        else:
            predictions = self.model.predict(X)
        
        # Map to positions
        pos2score = {}
        for i, residue in enumerate(chain_residues):
            pos = residue["resseq"]
            pos2score[pos] = float(predictions[i])
        
        return pos2score


# Global service instance
_ewclp3_service = None

def get_ewclp3_service() -> EWCLp3Service:
    """Get the global EWCLp3 service instance."""
    global _ewclp3_service
    if _ewclp3_service is None:
        _ewclp3_service = EWCLp3Service()
    return _ewclp3_service


def ewclp3_predict(structure_path: str, chain_id: str) -> Dict[int, float]:
    """
    Predict EWCL scores using EWCLp3 model.
    
    Args:
        structure_path: Path to structure file
        chain_id: Chain identifier
    
    Returns:
        Dict mapping position to EWCL score [0,1]
    """
    service = get_ewclp3_service()
    return service.predict(structure_path, chain_id)