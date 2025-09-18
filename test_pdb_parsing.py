#!/usr/bin/env python3
"""
Test PDB file parsing to verify feature extraction works correctly
"""

import sys
import os

# Add backend to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

def test_pdb_files():
    """Test PDB parsing with available files"""
    print("=== TESTING PDB FILE PARSING ===")
    
    # Find PDB files in workspace
    pdb_files = [f for f in os.listdir('.') if f.endswith('.pdb')]
    print(f"Found PDB files: {pdb_files}")
    
    if not pdb_files:
        print("No PDB files found, skipping PDB test")
        return
    
    test_file = pdb_files[0]  # Use first PDB file
    print(f"\nTesting with: {test_file}")
    
    try:
        # Test structure parsing
        from backend.api.parsers.structures import extract_residue_table, load_structure_residue_plddt
        
        print("\n1. Testing basic structure parsing...")
        pos_list, empty_list, plddt, af_like, warnings = extract_residue_table(test_file, "A")
        
        print(f"   ✓ Positions: {len(pos_list)}")
        print(f"   ✓ pLDDT: {len(plddt) if plddt else 'None'}")
        print(f"   ✓ AF-like: {af_like}")
        print(f"   ✓ Warnings: {warnings}")
        
        if plddt:
            print(f"   ✓ pLDDT range: {min(plddt):.2f} - {max(plddt):.2f}")
        
        # Test pLDDT extraction function
        plddt_dict, status = load_structure_residue_plddt(test_file, "A")
        print(f"   ✓ pLDDT status: {status}")
        print(f"   ✓ pLDDT dict: {len(plddt_dict)} entries")
        
        print("\n2. Testing EWCLp3 feature extraction...")
        from backend.api.routers.ewclv1p3_fresh import FeatureExtractor, load_structure_unified
        
        # Load structure using the same method as EWCL-H
        with open(test_file, 'rb') as f:
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
        
        # Test the fixed pandas access
        print("\n3. Testing fixed pandas feature access...")
        for i in range(min(5, len(feature_matrix))):
            row = feature_matrix.iloc[i]
            
            # Test both old and new methods
            try:
                old_hydro = row.get("hydropathy_x", 0.0)
                old_charge = row.get("charge_pH7", 0.0)
                print(f"   Row {i} (old .get method): hydro={old_hydro:.3f}, charge={old_charge:.3f}")
            except Exception as e:
                print(f"   Row {i} (old .get method): FAILED - {e}")
            
            try:
                new_hydro = row["hydropathy_x"] if "hydropathy_x" in feature_matrix.columns else 0.0
                new_charge = row["charge_pH7"] if "charge_pH7" in feature_matrix.columns else 0.0
                print(f"   Row {i} (new method): hydro={new_hydro:.3f}, charge={new_charge:.3f}")
            except Exception as e:
                print(f"   Row {i} (new method): FAILED - {e}")
        
        # Check specific columns
        hydro_cols = [col for col in feature_matrix.columns if 'hydro' in col.lower()]
        charge_cols = [col for col in feature_matrix.columns if 'charge' in col.lower()]
        
        print(f"\n   Available hydropathy columns: {hydro_cols}")
        print(f"   Available charge columns: {charge_cols}")
        
        if 'hydropathy_x' in feature_matrix.columns:
            hydro_vals = feature_matrix['hydropathy_x'].values
            print(f"   ✓ hydropathy_x range: {hydro_vals.min():.3f} - {hydro_vals.max():.3f}")
        else:
            print(f"   ⚠️  'hydropathy_x' column not found!")
            
        if 'charge_pH7' in feature_matrix.columns:
            charge_vals = feature_matrix['charge_pH7'].values
            print(f"   ✓ charge_pH7 range: {charge_vals.min():.3f} - {charge_vals.max():.3f}")
        else:
            print(f"   ⚠️  'charge_pH7' column not found!")
            
    except Exception as e:
        print(f"✗ Error in PDB testing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_pdb_files()