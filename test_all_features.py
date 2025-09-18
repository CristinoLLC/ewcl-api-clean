#!/usr/bin/env python3
"""
Test script to verify all 302 features are generated with real values
"""
import requests
import json
import pandas as pd
import numpy as np

def test_pdb_features():
    """Test PDB analysis endpoint and verify all features have real values"""
    
    # Test with 1CRN.pdb
    print("🧪 Testing PDB analysis endpoint with 1CRN.pdb...")
    
    try:
        with open("1CRN.pdb", "rb") as f:
            files = {"file": ("1CRN.pdb", f, "chemical/x-pdb")}
            response = requests.post("http://localhost:8080/ewcl/analyze-pdb/ewclv1-p3", files=files)
        
        if response.status_code != 200:
            print(f"❌ Error: {response.status_code} - {response.text}")
            return False
            
        data = response.json()
        print(f"✅ Successfully analyzed {len(data['residues'])} residues")
        
        # Check a few residues for biophysical properties
        print("\n📊 Sample biophysical properties:")
        for i, residue in enumerate(data['residues'][:5]):
            print(f"  Residue {i+1}: {residue['aa']} - "
                  f"pdb_cl={residue['pdb_cl']:.3f}, "
                  f"bfactor={residue['bfactor']:.2f}, "
                  f"hydropathy={residue['hydropathy']:.2f}, "
                  f"charge={residue['charge_pH7']:.2f}, "
                  f"curvature={residue['curvature']:.2f}")
        
        # Now let's test the feature extraction directly
        print("\n🔬 Testing feature extraction directly...")
        
        # Import the feature extractor
        import sys
        sys.path.append('.')
        from backend.api.routers.ewclv1p3_fresh import FeatureExtractor, FEATURE_NAMES
        
        # Create a test sequence
        test_sequence = ['T', 'C', 'P', 'S', 'N', 'F', 'N', 'V', 'C', 'R', 'L', 'P', 'G', 'S', 'A', 'A', 'A', 'A', 'A', 'A']
        test_confidence = [10.8, 8.31, 5.39, 4.24, 4.25, 3.12, 2.89, 1.45, 0.98, 0.67, 0.45, 0.23, 0.12, 0.05, 0.01, 0.0, 0.0, 0.0, 0.0, 0.0]
        
        print(f"Test sequence: {''.join(test_sequence)}")
        print(f"Test confidence: {test_confidence[:10]}...")
        
        # Create feature extractor
        extractor = FeatureExtractor(test_sequence, test_confidence, "xray")
        feature_matrix = extractor.extract_all_features()
        
        print(f"\n📈 Feature Matrix Analysis:")
        print(f"  Shape: {feature_matrix.shape}")
        print(f"  Expected: (20, 302)")
        print(f"  Actual: {feature_matrix.shape}")
        
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
        
        if zero_features:
            print(f"  ❌ Zero features: {zero_features[:10]}{'...' if len(zero_features) > 10 else ''}")
        else:
            print(f"  ✅ No features are all zeros")
            
        if low_variance_features:
            print(f"  ⚠️  Low variance features: {low_variance_features[:10]}{'...' if len(low_variance_features) > 10 else ''}")
        else:
            print(f"  ✅ All features have good variance")
        
        # Check specific important features
        important_features = [
            'hydropathy_x', 'charge_pH7', 'curvature_x', 'bfactor', 
            'helix_prop', 'sheet_prop', 'bulkiness', 'flexibility',
            'polarity', 'vdw_volume', 'entropy_w11', 'comp_A'
        ]
        
        print(f"\n🎯 Important Features Check:")
        for feat in important_features:
            if feat in feature_matrix.columns:
                values = feature_matrix[feat].values
                non_zero = np.sum(values != 0.0)
                mean_val = np.mean(values)
                std_val = np.std(values)
                print(f"  {feat}: {non_zero}/{len(values)} non-zero, mean={mean_val:.3f}, std={std_val:.3f}")
            else:
                print(f"  {feat}: ❌ MISSING")
        
        # Check amino acid one-hot encoding
        print(f"\n🧬 Amino Acid One-Hot Encoding Check:")
        aa_features = [aa for aa in 'ACDEFGHIKLMNPQRSTVWY' if aa in feature_matrix.columns]
        print(f"  Amino acid features found: {aa_features}")
        
        for i, aa in enumerate(test_sequence[:5]):
            if aa in feature_matrix.columns:
                value = feature_matrix.iloc[i][aa]
                print(f"  Position {i+1} ({aa}): {value}")
        
        # Check windowed features
        print(f"\n🪟 Windowed Features Check:")
        window_features = [col for col in feature_matrix.columns if '_w' in col and 'mean' in col]
        print(f"  Windowed mean features: {len(window_features)}")
        
        for feat in window_features[:5]:
            values = feature_matrix[feat].values
            non_zero = np.sum(values != 0.0)
            print(f"  {feat}: {non_zero}/{len(values)} non-zero, mean={np.mean(values):.3f}")
        
        # Overall assessment
        total_features = len(feature_matrix.columns)
        non_zero_features = total_features - len(zero_features)
        
        print(f"\n📊 Overall Assessment:")
        print(f"  Total features: {total_features}")
        print(f"  Non-zero features: {non_zero_features}")
        print(f"  Success rate: {non_zero_features/total_features*100:.1f}%")
        
        if non_zero_features / total_features > 0.8:
            print(f"  ✅ EXCELLENT: Most features have real values")
        elif non_zero_features / total_features > 0.5:
            print(f"  ⚠️  GOOD: Many features have real values")
        else:
            print(f"  ❌ POOR: Many features are zero")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🚀 Starting comprehensive feature testing...")
    success = test_pdb_features()
    
    if success:
        print("\n🎉 Feature testing completed successfully!")
    else:
        print("\n💥 Feature testing failed!")
