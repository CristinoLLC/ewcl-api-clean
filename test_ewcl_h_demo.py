#!/usr/bin/env python3
"""
EWCL-H Hallucination Detection Demo
===================================

Demonstrates hallucination detection on real AlphaFold CIF files using 
actual EWCL predictions from the deployed API.
"""

import numpy as np
import json
from typing import List, Dict, Any
import requests
import time

# EWCL-H Parameters (from our specification)
LAMBDA_H = 0.871
TAU = 0.50
PLDDT_STRICT = 70.0

def sigmoid(x):
    """Sigmoid activation function."""
    return 1.0 / (1.0 + np.exp(-x))

def compute_hallucination_scores(ewcl_scores: List[float], 
                               plddt_scores: List[float]) -> Dict[str, Any]:
    """
    Compute EWCL-H hallucination scores from EWCL predictions and pLDDT values.
    
    Returns dict with per-residue scores and summary statistics.
    """
    ewcl = np.array(ewcl_scores)
    plddt = np.array(plddt_scores)
    
    # Normalize pLDDT to [0,1]
    plddt_norm = np.clip(plddt / 100.0, 0.0, 1.0)
    
    # Compute hallucination score: H = sigmoid(λ * (ewcl - (1 - plddt_norm)))
    # High H when EWCL is high (disorder) but pLDDT is also high (confident)
    H = sigmoid(LAMBDA_H * (ewcl - (1.0 - plddt_norm)))
    
    # Compute flags
    is_high_H = H >= TAU
    is_disagree = (plddt >= PLDDT_STRICT) & is_high_H
    
    # Summary statistics
    results = {
        "per_residue": [
            {
                "pos": i + 1,
                "ewcl": float(ewcl[i]),
                "plddt": float(plddt[i]),
                "H": float(H[i]),
                "is_high_H": bool(is_high_H[i]),
                "is_disagree": bool(is_disagree[i])
            }
            for i in range(len(ewcl))
        ],
        "summary": {
            "n_res": len(ewcl),
            "mean_EWCL": float(np.mean(ewcl)),
            "mean_pLDDT": float(np.mean(plddt)),
            "mean_H": float(np.mean(H)),
            "p95_H": float(np.percentile(H, 95)),
            "frac_high_H": float(np.mean(is_high_H)),
            "frac_disagree": float(np.mean(is_disagree)),
            "flagged": bool(np.mean(is_disagree) >= 0.20)
        }
    }
    
    return results

def test_alphafold_cif(filename: str, api_url: str = "https://ewcl-api-production.up.railway.app"):
    """Test AlphaFold CIF file and compute hallucination scores."""
    print(f"\n🧬 **Testing {filename}**")
    print("=" * 50)
    
    # Upload file to API
    print("📤 Uploading to EWCL API...")
    start_time = time.time()
    
    try:
        with open(filename, 'rb') as f:
            files = {'file': (filename, f, 'application/octet-stream')}
            response = requests.post(
                f"{api_url}/ewcl/analyze-pdb/ewclv1-p3",
                files=files,
                timeout=30
            )
        
        upload_time = time.time() - start_time
        print(f"⏱️  Upload completed in {upload_time:.2f}s")
        
        if response.status_code != 200:
            print(f"❌ API Error: {response.status_code}")
            return None
        
        data = response.json()
        residues = data.get('residues', [])
        
        if not residues:
            print("❌ No residues found in response")
            return None
        
        print(f"✅ Successfully processed {len(residues)} residues")
        
        # Extract EWCL and pLDDT scores
        ewcl_scores = [r['pdb_cl'] for r in residues]
        plddt_scores = [r['plddt'] for r in residues if r['plddt'] is not None]
        
        if len(plddt_scores) != len(ewcl_scores):
            print(f"⚠️  pLDDT missing for some residues: {len(plddt_scores)}/{len(ewcl_scores)}")
            return None
        
        # Compute hallucination scores
        print("🔬 Computing hallucination scores...")
        h_results = compute_hallucination_scores(ewcl_scores, plddt_scores)
        
        # Display results
        summary = h_results['summary']
        print(f"""
📊 **Summary Statistics:**
   • Protein Length: {summary['n_res']} residues
   • Mean EWCL: {summary['mean_EWCL']:.3f}
   • Mean pLDDT: {summary['mean_pLDDT']:.1f}
   • Mean H-score: {summary['mean_H']:.3f}
   • 95th percentile H: {summary['p95_H']:.3f}
   • Fraction high H (≥{TAU}): {summary['frac_high_H']:.1%}
   • Fraction disagree: {summary['frac_disagree']:.1%}
   • **FLAGGED**: {summary['flagged']} (≥20% disagreement)
""")
        
        # Show most problematic residues
        per_res = h_results['per_residue']
        high_h_residues = [r for r in per_res if r['is_high_H']]
        disagree_residues = [r for r in per_res if r['is_disagree']]
        
        if high_h_residues:
            print(f"🚨 **High Hallucination Residues** ({len(high_h_residues)} total):")
            for r in sorted(high_h_residues, key=lambda x: x['H'], reverse=True)[:5]:
                print(f"   • Pos {r['pos']:3d}: H={r['H']:.3f}, EWCL={r['ewcl']:.3f}, pLDDT={r['plddt']:.1f}")
        
        if disagree_residues:
            print(f"⚠️  **Disagreement Cases** ({len(disagree_residues)} total):")
            for r in sorted(disagree_residues, key=lambda x: x['H'], reverse=True)[:5]:
                print(f"   • Pos {r['pos']:3d}: H={r['H']:.3f}, EWCL={r['ewcl']:.3f}, pLDDT={r['plddt']:.1f}")
        
        return h_results
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

def main():
    """Run hallucination detection on all AlphaFold CIF files."""
    print("🔬 **EWCL-H Hallucination Detection Demo**")
    print("Using real AlphaFold CIF files and EWCL API predictions")
    print(f"Parameters: λ={LAMBDA_H}, τ={TAU}, pLDDT_strict={PLDDT_STRICT}")
    
    # Test files (in order of complexity)
    test_files = [
        "AF-P37840-F1-model_v4.cif",  # Smallest, fastest
        "AF-P41208-F1-model_v4.cif",  # Medium size
        # "AF-O60828-F1-model_v4.cif",  # Largest (skip if timeout issues)
    ]
    
    results = {}
    for filename in test_files:
        try:
            result = test_alphafold_cif(filename)
            if result:
                results[filename] = result
        except KeyboardInterrupt:
            print("\n🛑 Interrupted by user")
            break
        except Exception as e:
            print(f"❌ Failed to process {filename}: {e}")
    
    # Compare results
    if len(results) > 1:
        print("\n📊 **Comparison Summary:**")
        print("-" * 60)
        for filename, result in results.items():
            s = result['summary']
            protein_id = filename.split('-')[1]  # Extract UniProt ID
            print(f"{protein_id}: {s['frac_disagree']:.1%} disagree, "
                  f"H̄={s['mean_H']:.3f}, flagged={s['flagged']}")

if __name__ == "__main__":
    main()