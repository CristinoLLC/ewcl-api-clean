#!/usr/bin/env python3
"""
Test both AlphaFold and experimental PDB files to compare behavior
"""

import sys
import os

# Add backend to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

def test_experimental_pdb():
    """Test experimental PDB (1CRN.pdb) vs AlphaFold PDB"""
    print("=== COMPARING EXPERIMENTAL vs ALPHAFOLD PDB ===")
    
    files_to_test = ['1CRN.pdb', 'AF-O15552-F1-model_v4.pdb']
    
    for test_file in files_to_test:
        if not os.path.exists(test_file):
            print(f"File {test_file} not found, skipping")
            continue
            
        print(f"\n--- Testing {test_file} ---")
        
        try:
            # Test structure parsing first
            from backend.api.parsers.structures import extract_residue_table
            pos_list, empty_list, plddt, af_like, warnings = extract_residue_table(test_file, "A")
            
            print(f"Structure type: {'AlphaFold-like' if af_like else 'Experimental'}")
            print(f"Positions: {len(pos_list)}")
            print(f"pLDDT available: {plddt is not None}")
            if plddt:
                print(f"pLDDT range: {min(plddt):.2f} - {max(plddt):.2f}")
            
            # Test feature extraction
            from backend.api.routers.ewclv1p3_fresh import FeatureExtractor, load_structure_unified
            
            with open(test_file, 'rb') as f:
                raw_bytes = f.read()
            
            try:
                pdb_data = load_structure_unified(raw_bytes)
                chain_residues = pdb_data["residues"]
                sequence = [r["aa"] for r in chain_residues]
                confidence_values = [r.get("bfactor", 0.0) for r in chain_residues]
                
                print(f"Sequence extracted: {len(sequence)} residues")
                print(f"Confidence values: min={min(confidence_values):.2f}, max={max(confidence_values):.2f}")
                
                # Try feature extraction
                extractor = FeatureExtractor(sequence, confidence_values, pdb_data["source"])
                feature_matrix = extractor.extract_all_features()
                
                print(f"Feature extraction: SUCCESS - {feature_matrix.shape}")
                
                # Test feature access
                if len(feature_matrix) > 0:
                    row = feature_matrix.iloc[0]
                    hydro = row.get("hydropathy_x", "NOT_FOUND")
                    charge = row.get("charge_pH7", "NOT_FOUND") 
                    print(f"Sample features: hydro={hydro}, charge={charge}")
                    
            except Exception as e:
                print(f"Feature extraction FAILED: {e}")
                import traceback
                traceback.print_exc()
                
        except Exception as e:
            print(f"Structure parsing FAILED: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    test_experimental_pdb()