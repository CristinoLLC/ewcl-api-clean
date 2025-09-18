#!/usr/bin/env python3
"""
Test with a more diverse amino acid sequence to maximize feature coverage
"""
import sys
sys.path.append('.')
from backend.api.routers.ewclv1p3_fresh import FeatureExtractor, FEATURE_NAMES
import numpy as np

def test_diverse_sequence():
    """Test with a sequence containing all 20 amino acids"""
    
    # Create a diverse sequence with all 20 amino acids
    diverse_sequence = [
        'A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',  # First 10
        'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y'   # Last 10
    ]
    
    # Create realistic confidence values
    diverse_confidence = [50.0, 45.0, 40.0, 35.0, 30.0, 25.0, 20.0, 15.0, 10.0, 5.0,
                         50.0, 45.0, 40.0, 35.0, 30.0, 25.0, 20.0, 15.0, 10.0, 5.0]
    
    print(f"🧬 Testing with diverse sequence: {''.join(diverse_sequence)}")
    print(f"📊 Confidence values: {diverse_confidence[:10]}...")
    
    # Create feature extractor
    extractor = FeatureExtractor(diverse_sequence, diverse_confidence, "xray")
    feature_matrix = extractor.extract_all_features()
    
    print(f"\n📈 Feature Matrix Analysis:")
    print(f"  Shape: {feature_matrix.shape}")
    
    # Check for zero values
    zero_features = []
    low_variance_features = []
    
    for col in feature_matrix.columns:
        values = feature_matrix[col].values
        zero_count = np.sum(values == 0.0)
        variance = np.var(values)
        
        if zero_count == len(values):
            zero_features.append(col)
        elif variance < 1e-10:
            low_variance_features.append(col)
    
    print(f"\n🔍 Feature Quality Analysis:")
    print(f"  Total features: {len(feature_matrix.columns)}")
    print(f"  Features with all zeros: {len(zero_features)}")
    print(f"  Features with low variance: {len(low_variance_features)}")
    
    # Check amino acid one-hot encoding
    print(f"\n🧬 Amino Acid One-Hot Encoding Check:")
    aa_features = [aa for aa in 'ACDEFGHIKLMNPQRSTVWY' if aa in feature_matrix.columns]
    print(f"  Amino acid features found: {len(aa_features)}/20")
    
    for i, aa in enumerate(diverse_sequence):
        if aa in feature_matrix.columns:
            value = feature_matrix.iloc[i][aa]
            print(f"  Position {i+1:2d} ({aa}): {value}")
    
    # Check charged amino acids
    print(f"\n⚡ Charged Amino Acids Check:")
    charged_aas = ['D', 'E', 'K', 'R', 'H']
    for aa in charged_aas:
        if aa in feature_matrix.columns:
            positions = [i for i, seq_aa in enumerate(diverse_sequence) if seq_aa == aa]
            for pos in positions:
                charge_val = feature_matrix.iloc[pos]['charge_pH7']
                print(f"  {aa} at position {pos+1}: charge={charge_val}")
    
    # Check hydropathy values
    print(f"\n💧 Hydropathy Values Check:")
    for i, aa in enumerate(diverse_sequence[:10]):
        hydro_val = feature_matrix.iloc[i]['hydropathy_x']
        print(f"  {aa} at position {i+1}: hydropathy={hydro_val:.2f}")
    
    # Check windowed features
    print(f"\n🪟 Windowed Features Sample:")
    window_features = [col for col in feature_matrix.columns if '_w11_mean' in col]
    for feat in window_features[:5]:
        values = feature_matrix[feat].values
        non_zero = np.sum(values != 0.0)
        mean_val = np.mean(values)
        print(f"  {feat}: {non_zero}/{len(values)} non-zero, mean={mean_val:.3f}")
    
    # Overall assessment
    total_features = len(feature_matrix.columns)
    non_zero_features = total_features - len(zero_features)
    
    print(f"\n📊 Overall Assessment:")
    print(f"  Total features: {total_features}")
    print(f"  Non-zero features: {non_zero_features}")
    print(f"  Success rate: {non_zero_features/total_features*100:.1f}%")
    
    if non_zero_features / total_features > 0.9:
        print(f"  ✅ EXCELLENT: Almost all features have real values")
    elif non_zero_features / total_features > 0.8:
        print(f"  ✅ VERY GOOD: Most features have real values")
    elif non_zero_features / total_features > 0.7:
        print(f"  ⚠️  GOOD: Many features have real values")
    else:
        print(f"  ❌ POOR: Many features are zero")
    
    return non_zero_features / total_features

if __name__ == "__main__":
    print("🚀 Testing with diverse amino acid sequence...")
    success_rate = test_diverse_sequence()
    print(f"\n🎯 Final success rate: {success_rate*100:.1f}%")
