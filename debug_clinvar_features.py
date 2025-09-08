# Test script to debug ClinVar feature vectors
import numpy as np
import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent / "backend"))

# Import the router functions
from backend.api.routers.ewclv1_C import _vectorize, _enhanced_features, FEATURE_NAMES

def test_feature_vectors():
    print("Testing ENHANCED ClinVar feature vector generation...")
    print(f"Total features: {len(FEATURE_NAMES)}")
    print(f"Feature names: {FEATURE_NAMES[:10]}... (showing first 10)")
    
    # Test two different variants
    variants = [
        {"pos": 175, "ref": "R", "alt": "H", "length": 500},
        {"pos": 260, "ref": "A", "alt": "V", "length": 600},
        {"pos": 42, "ref": "G", "alt": "E", "length": 350},
    ]
    
    feature_vectors = []
    
    for i, v in enumerate(variants):
        print(f"\n--- Variant {i+1}: {v['ref']}{v['pos']}{v['alt']} (length={v['length']}) ---")
        
        # Build feature vector using enhanced features
        base = _enhanced_features(v["ref"], v["alt"], v["pos"], v["length"])
        x = _vectorize(base, FEATURE_NAMES)
        feature_vectors.append(x)
        
        print(f"Feature vector shape: {x.shape}")
        print(f"Non-zero features: {np.count_nonzero(x)}")
        print(f"Feature vector stats: min={x.min():.3f}, max={x.max():.3f}, std={x.std():.3f}")
        
        # Show some key features
        key_features = ["position", "sequence_length", "position_ratio", "delta_hydropathy", 
                       "delta_charge", "has_embeddings", "ewcl_hydropathy", "ewcl_charge_pH7",
                       "emb_0", "emb_1", "emb_2"]
        print(f"Key feature values:")
        for feat in key_features:
            if feat in FEATURE_NAMES:
                idx = FEATURE_NAMES.index(feat)
                print(f"  {feat}: {x[idx]:.3f}")
    
    # Compare all vectors
    print(f"\n--- Vector Comparisons ---")
    for i in range(len(feature_vectors)):
        for j in range(i+1, len(feature_vectors)):
            diff = np.abs(feature_vectors[i] - feature_vectors[j])
            print(f"Variants {i+1} vs {j+1}:")
            print(f"  Total difference: {diff.sum():.3f}")
            print(f"  Max difference: {diff.max():.3f}")
            print(f"  Different features: {np.count_nonzero(diff)}")
            print(f"  Mean difference: {diff.mean():.3f}")

if __name__ == "__main__":
    test_feature_vectors()