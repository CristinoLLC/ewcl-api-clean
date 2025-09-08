# test_ewclv1_features.py
"""Quick test to verify EWCL v1 feature extraction works correctly"""
import sys
from pathlib import Path
import json

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from backend.models.feature_extractors.ewclv1_features import build_ewclv1_features

def test_feature_extraction():
    """Test feature extraction with a sample sequence"""
    # Test sequence (small protein)
    test_seq = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKREQTGQGWVPSNYITPVN"
    
    print(f"Testing EWCL v1 feature extraction with sequence length: {len(test_seq)}")
    
    # Build features without PSSM
    block = build_ewclv1_features(test_seq, pssm=None)
    
    print(f"Generated {len(block.all_df.columns)} features for {len(block.all_df)} residues")
    print(f"Has PSSM data: {block.has_pssm}")
    
    # Load expected features from the JSON
    try:
        with open("backend_bundle/meta/EWCLv1_feature_info.json", "r") as f:
            expected = json.load(f)
        
        expected_features = set(expected["all_features"])
        generated_features = set(block.all_df.columns)
        
        missing = expected_features - generated_features
        extra = generated_features - expected_features
        
        print(f"\nFeature alignment check:")
        print(f"Expected: {len(expected_features)} features")
        print(f"Generated: {len(generated_features)} features")
        print(f"Missing: {len(missing)} features")
        print(f"Extra: {len(extra)} features")
        
        if missing:
            print(f"Missing features: {sorted(list(missing))[:10]}...")
        if extra:
            print(f"Extra features: {sorted(list(extra))[:10]}...")
            
        if len(missing) == 0 and len(extra) == 0:
            print("✅ Perfect feature alignment!")
        else:
            print("❌ Feature mismatch detected")
            
        # Show sample values for first residue
        print(f"\nSample values for first residue ({test_seq[0]}):")
        first_residue = block.all_df.iloc[0]
        sample_features = ["hydropathy", "charge_pH7", "helix_prop", "sheet_prop", "has_pssm_data"]
        for feat in sample_features:
            if feat in first_residue:
                print(f"  {feat}: {first_residue[feat]:.3f}")
                
    except FileNotFoundError:
        print("Warning: Could not load expected features JSON for comparison")
    
    return block

if __name__ == "__main__":
    test_feature_extraction()