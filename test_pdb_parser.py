#!/usr/bin/env python3
"""
Test script to debug PDB parser issues
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from backend.api.routers.ewclv1p3_fresh import _parse_with_fallback_parser

def test_pdb_parser():
    """Test the PDB parser with a sample file"""
    try:
        with open("1CRN.pdb", "rb") as f:
            raw_bytes = f.read()
        
        print(f"File size: {len(raw_bytes)} bytes")
        print(f"First 200 chars: {raw_bytes[:200].decode('utf-8', errors='ignore')}")
        
        result = _parse_with_fallback_parser(raw_bytes)
        print(f"Result type: {type(result)}")
        print(f"Result keys: {list(result.keys()) if isinstance(result, dict) else 'Not a dict'}")
        
        if isinstance(result, dict) and "residues" in result:
            print(f"Residues count: {len(result['residues'])}")
            print(f"First residue: {result['residues'][0] if result['residues'] else 'No residues'}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_pdb_parser()
