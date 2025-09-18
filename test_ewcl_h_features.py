#!/usr/bin/env python3
"""
Test EWCL-H feature extraction pipeline to find where pLDDT gets lost
"""

import sys
import os
import json

# Add backend to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

def test_ewcl_h_features():
    """Test the full EWCL-H feature extraction pipeline"""
    print("=== TESTING EWCL-H FEATURE EXTRACTION ===")
    
    cif_file = "AF-P10636-F1-model_v4.cif"
    
    try:
        # Test the EWCL-H service components
        print("\n1. Testing structure parsing...")
        from backend.api.parsers.structures import extract_residue_table, load_structure_residue_plddt
        
        pos_list, empty_list, plddt, af_like, warnings = extract_residue_table(cif_file, "A")
        print(f"   ✓ Positions: {len(pos_list)}, pLDDT: {len(plddt) if plddt else 0}")
        if plddt:
            print(f"   ✓ pLDDT range: {min(plddt):.2f} - {max(plddt):.2f}")
        
        plddt_dict, status = load_structure_residue_plddt(cif_file, "A")
        print(f"   ✓ pLDDT dict: {len(plddt_dict)} entries, status: {status}")
        
        print("\n2. Testing EWCL-H specific components...")
        
        # Test the EWCLp3 feature extractor (used in EWCL-H)
        from backend.api.routers.ewclv1p3_fresh import FeatureExtractor, load_structure_unified
        
        try:
            # Load structure using the same method as EWCL-H
            with open(cif_file, 'rb') as f:
                raw_bytes = f.read()
            
            pdb_data = load_structure_unified(raw_bytes)
            print(f"   ✓ PDB data loaded: {list(pdb_data.keys())}")
            
            chain_residues = pdb_data["residues"]
            sequence = [r["aa"] for r in chain_residues]
            confidence_values = [r.get("bfactor", 0.0) for r in chain_residues]
            
            print(f"   ✓ Sequence length: {len(sequence)}")
            print(f"   ✓ Confidence values: {len(confidence_values)}")
            print(f"     Sample confidence: {confidence_values[:5]}")
            
            # Create feature extractor
            extractor = FeatureExtractor(sequence, confidence_values, pdb_data["source"])
            feature_matrix = extractor.extract_all_features()
            
            print(f"   ✓ Feature matrix shape: {feature_matrix.shape}")
            print(f"   ✓ Feature columns: {list(feature_matrix.columns)}")
            
            # Check for hydropathy and charge columns
            hydro_cols = [col for col in feature_matrix.columns if 'hydro' in col.lower()]
            charge_cols = [col for col in feature_matrix.columns if 'charge' in col.lower()]
            
            print(f"   ✓ Hydropathy columns: {hydro_cols}")
            print(f"   ✓ Charge columns: {charge_cols}")
            
            # Check specific columns the router is looking for
            if 'hydropathy_x' in feature_matrix.columns:
                hydro_vals = feature_matrix['hydropathy_x'].values
                print(f"     hydropathy_x range: {hydro_vals.min():.3f} - {hydro_vals.max():.3f}")
            else:
                print(f"     ⚠️  'hydropathy_x' column not found!")
                
            if 'charge_pH7' in feature_matrix.columns:
                charge_vals = feature_matrix['charge_pH7'].values
                print(f"     charge_pH7 range: {charge_vals.min():.3f} - {charge_vals.max():.3f}")
            else:
                print(f"     ⚠️  'charge_pH7' column not found!")
                
        except Exception as e:
            print(f"   ✗ Error in EWCLp3 feature extraction: {e}")
            import traceback
            traceback.print_exc()
        
        print("\n3. Testing full feature extraction...")
        
        # Skip the complex feature extraction test for now
        print("   (Skipping complex integration test - focusing on column names)")
        
        print("\n4. Testing hallucination evaluation...")
        
        # Skip the hallucination evaluation for now
        print("   (Skipping hallucination evaluation - need to fix column names first)")
                            
    except Exception as e:
        print(f"✗ Error in EWCL-H testing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_ewcl_h_features()