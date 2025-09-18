#!/usr/bin/env python3
"""
Test the fixed EWCL-H feature extraction
"""

import sys
import os
import pandas as pd

# Add backend to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

def test_pandas_access():
    """Test the pandas DataFrame access methods"""
    print("=== TESTING PANDAS DATAFRAME ACCESS ===")
    
    # Create a sample DataFrame like our feature matrix
    sample_data = {
        'hydropathy_x': [-2.5, -1.0, 0.5, 1.5],
        'charge_pH7': [1.0, 0.0, -1.0, 0.5],
        'other_col': [1, 2, 3, 4]
    }
    df = pd.DataFrame(sample_data)
    print(f"Sample DataFrame:\n{df}")
    
    print("\nTesting access methods:")
    
    # Test the old broken method (should fail)
    try:
        result = df.iloc[0].get("hydropathy_x", 0.0)
        print(f"OLD (broken): df.iloc[0].get('hydropathy_x', 0.0) = {result}")
    except AttributeError as e:
        print(f"OLD (broken): df.iloc[0].get() failed with: {e}")
    
    # Test the correct methods
    try:
        result1 = df.iloc[0]["hydropathy_x"]
        print(f"FIXED: df.iloc[0]['hydropathy_x'] = {result1}")
        
        result2 = df.at[0, "hydropathy_x"]  
        print(f"ALTERNATIVE: df.at[0, 'hydropathy_x'] = {result2}")
        
        # Test with column existence check
        hydro_val = df.iloc[0]["hydropathy_x"] if "hydropathy_x" in df.columns else 0.0
        charge_val = df.iloc[0]["charge_pH7"] if "charge_pH7" in df.columns else 0.0
        missing_val = df.iloc[0]["missing_col"] if "missing_col" in df.columns else 0.0
        
        print(f"SAFE: hydro_val = {hydro_val}, charge_val = {charge_val}, missing_val = {missing_val}")
        
    except Exception as e:
        print(f"Error in correct methods: {e}")

def test_feature_extraction():
    """Test the actual feature extraction with our fix"""
    print("\n=== TESTING FIXED FEATURE EXTRACTION ===")
    
    cif_file = "AF-P10636-F1-model_v4.cif"
    
    try:
        # Load the feature matrix as the router does
        from backend.api.routers.ewclv1p3_fresh import FeatureExtractor, load_structure_unified
        
        with open(cif_file, 'rb') as f:
            raw_bytes = f.read()
        
        pdb_data = load_structure_unified(raw_bytes)
        chain_residues = pdb_data["residues"]
        sequence = [r["aa"] for r in chain_residues]
        confidence_values = [r.get("bfactor", 0.0) for r in chain_residues]
        extractor = FeatureExtractor(sequence, confidence_values, pdb_data["source"])
        feature_matrix = extractor.extract_all_features()
        
        print(f"✓ Feature matrix loaded: {feature_matrix.shape}")
        
        # Test the fixed extraction logic
        hydro, charge = [], []
        for i in range(min(10, len(feature_matrix))):  # First 10 residues
            hydro_val = feature_matrix.iloc[i]["hydropathy_x"] if "hydropathy_x" in feature_matrix.columns else 0.0
            charge_val = feature_matrix.iloc[i]["charge_pH7"] if "charge_pH7" in feature_matrix.columns else 0.0
            hydro.append(float(hydro_val))
            charge.append(float(charge_val))
        
        print(f"✓ Extracted {len(hydro)} hydropathy values: {hydro}")
        print(f"✓ Extracted {len(charge)} charge values: {charge}")
        
        # Verify we're getting real values, not zeros
        if all(h == 0.0 for h in hydro):
            print("⚠️  All hydropathy values are zero!")
        else:
            print(f"✓ Hydropathy range: {min(hydro):.3f} to {max(hydro):.3f}")
        
        if all(c == 0.0 for c in charge):
            print("⚠️  All charge values are zero!")
        else:
            print(f"✓ Charge range: {min(charge):.3f} to {max(charge):.3f}")
            
    except Exception as e:
        print(f"✗ Error in feature extraction test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_pandas_access()
    test_feature_extraction()