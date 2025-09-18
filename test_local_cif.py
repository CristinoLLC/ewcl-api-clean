#!/usr/bin/env python3
"""
Local test script to debug CIF parsing issues
Testing AF-P10636-F1-model_v4.cif to see why pLDDT values are null
"""

import sys
import os

# Add backend to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

try:
    import gemmi
    print("✓ gemmi imported successfully")
except ImportError:
    print("✗ gemmi not available - install with: pip install gemmi")
    sys.exit(1)

def test_cif_parsing():
    """Test parsing of the AlphaFold CIF file"""
    cif_file = "AF-P10636-F1-model_v4.cif"
    
    if not os.path.exists(cif_file):
        print(f"✗ File not found: {cif_file}")
        return
    
    print(f"Testing CIF file: {cif_file}")
    print(f"File size: {os.path.getsize(cif_file):,} bytes")
    
    try:
        # Load the structure
        structure = gemmi.read_structure(cif_file)
        print(f"✓ Structure loaded successfully")
        print(f"  Name: {structure.name}")
        print(f"  Models: {len(structure)}")
        
        if len(structure) == 0:
            print("✗ No models found")
            return
        
        model = structure[0]
        print(f"  Chains: {len(model)}")
        
        # Examine first chain
        if len(model) == 0:
            print("✗ No chains found")
            return
        
        chain = model[0]
        print(f"  Chain ID: {chain.name}")
        print(f"  Residues: {len(chain)}")
        
        if len(chain) == 0:
            print("✗ No residues found")
            return
        
        # Examine first few residues
        print("\n=== RESIDUE ANALYSIS ===")
        for i, residue in enumerate(chain[:5]):  # First 5 residues
            print(f"Residue {i+1}: {residue.name} {residue.seqid}")
            
            # Check atoms
            atoms = list(residue)
            print(f"  Atoms: {len(atoms)}")
            
            if atoms:
                # Look at first atom's B-factor (should contain pLDDT)
                first_atom = atoms[0]
                print(f"  First atom: {first_atom.name}")
                print(f"    B-factor (pLDDT?): {first_atom.b_iso}")
                print(f"    Occupancy: {first_atom.occ}")
                print(f"    Element: {first_atom.element.name}")
                
                # Check if we have CA atom specifically
                ca_atom = None
                for atom in atoms:
                    if atom.name == "CA":
                        ca_atom = atom
                        break
                
                if ca_atom:
                    print(f"  CA atom B-factor: {ca_atom.b_iso}")
                else:
                    print("  No CA atom found")
        
        print("\n=== CONFIDENCE SCORE ANALYSIS ===")
        # Extract confidence scores from all CA atoms
        confidence_scores = []
        for residue in chain:
            for atom in residue:
                if atom.name == "CA":
                    confidence_scores.append(atom.b_iso)
                    break
        
        if confidence_scores:
            print(f"Total CA atoms: {len(confidence_scores)}")
            print(f"Confidence range: {min(confidence_scores):.2f} - {max(confidence_scores):.2f}")
            print(f"Average confidence: {sum(confidence_scores)/len(confidence_scores):.2f}")
            print(f"First 10 values: {confidence_scores[:10]}")
        else:
            print("No CA atoms with confidence scores found")
            
    except Exception as e:
        print(f"✗ Error parsing CIF: {e}")
        import traceback
        traceback.print_exc()

def test_feature_extraction():
    """Test our actual feature extraction code"""
    print("\n=== TESTING BACKEND FEATURE EXTRACTION ===")
    
    try:
        from backend.api.parsers.structures import extract_residue_table
        
        cif_file = "AF-P10636-F1-model_v4.cif"
        
        print("Testing extract_residue_table function...")
        pos_list, empty_list, plddt, af_like, warnings = extract_residue_table(cif_file, "A")
        
        print(f"Result structure:")
        print(f"  pos_list: {len(pos_list)} positions")
        print(f"  empty_list: {len(empty_list)} items")
        print(f"  plddt: {len(plddt) if plddt else 'None'} values")
        print(f"  af_like: {af_like}")
        print(f"  warnings: {warnings}")
        
        if plddt:
            print(f"  pLDDT range: {min(plddt):.2f} - {max(plddt):.2f}")
            print(f"  First 10 pLDDT: {plddt[:10]}")
            
            # Test load_structure_residue_plddt too
            print("\nTesting load_structure_residue_plddt function...")
            from backend.api.parsers.structures import load_structure_residue_plddt
            plddt_dict, status = load_structure_residue_plddt(cif_file, "A")
            print(f"  Status: {status}")
            print(f"  pLDDT dict: {len(plddt_dict)} residues")
            if plddt_dict:
                sample_keys = list(plddt_dict.keys())[:5]
                print(f"  Sample entries: {[(k, plddt_dict[k]) for k in sample_keys]}")
        else:
            print("  No pLDDT values found!")
        
    except Exception as e:
        print(f"✗ Error in feature extraction: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("=== LOCAL CIF PARSING TEST ===")
    test_cif_parsing()
    test_feature_extraction()