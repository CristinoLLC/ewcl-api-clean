#!/usr/bin/env python3
"""
Enhanced EWCL-H Hallucination Detection Demo
============================================

Demonstrates the new features:
- EWCL source tracking (pdb_model vs af_proxy)
- Overlap counters (n_res_total, n_ewcl_finite, n_plddt_finite, n_overlap_used)
- Comprehensive JSON response for frontend correlation
"""

import numpy as np
import json
from typing import List, Dict, Any
import requests
import time

# EWCL-H Parameters
LAMBDA_H = 0.871
TAU = 0.50
PLDDT_STRICT = 70.0

def test_enhanced_hallucination_api(filename: str, api_url: str = "https://ewcl-api-production.up.railway.app"):
    """Test the enhanced EWCL-H API with overlap counters and source tracking."""
    print(f"\n🧬 **Testing Enhanced EWCL-H API: {filename}**")
    print("=" * 60)
    
    # Test both EWCL sources
    for ewcl_source in ["pdb_model"]:  # Start with pdb_model only
        print(f"\n📊 **EWCL Source: {ewcl_source}**")
        print("-" * 40)
        
        start_time = time.time()
        
        try:
            with open(filename, 'rb') as f:
                files = {'file': (filename, f, 'application/octet-stream')}
                data = {
                    'ewcl_source': ewcl_source,
                    'lambda_h': LAMBDA_H,
                    'tau': TAU,
                    'plddt_strict': PLDDT_STRICT
                }
                
                response = requests.post(
                    f"{api_url}/api/hallucination/evaluate",
                    files=files,
                    data=data,
                    timeout=30
                )
            
            upload_time = time.time() - start_time
            print(f"⏱️  Upload completed in {upload_time:.2f}s")
            
            if response.status_code != 200:
                print(f"❌ API Error: {response.status_code}")
                print(f"Response: {response.text[:500]}")
                continue
            
            result = response.json()
            
            if not result.get('results'):
                print("❌ No results in response")
                continue
            
            chain_result = result['results'][0]  # First chain
            
            # Display enhanced metadata
            print(f"""
📊 **Enhanced Metadata:**
   • EWCL Source: {chain_result.get('ewcl_source', 'N/A')}
   • Confidence Type: {chain_result.get('confidence_type', 'N/A')}
   • Total Residues: {chain_result.get('n_res_total', 'N/A')}
   • EWCL Available: {chain_result.get('n_ewcl_finite', 'N/A')} residues
   • Confidence Available: {chain_result.get('n_plddt_finite', 'N/A')} residues
   • Overlap Used for H: {chain_result.get('n_overlap_used', 'N/A')} residues
""")
            
            # Display hallucination analysis
            print(f"""
🔬 **Hallucination Analysis:**
   • Mean EWCL: {chain_result.get('mean_EWCL', 'N/A'):.3f}
   • Mean pLDDT: {chain_result.get('mean_pLDDT', 'N/A'):.1f}
   • 95th percentile H: {chain_result.get('p95_H', 'N/A'):.3f}
   • Fraction high H (≥{TAU}): {chain_result.get('frac_high_H', 0)*100:.1f}%
   • Fraction disagree: {chain_result.get('frac_disagree', 0)*100:.1f}%
   • **FLAGGED**: {chain_result.get('flagged', False)} (≥20% disagreement)
""")
            
            # Show sample residues with comprehensive data
            residues = chain_result.get('residues', [])[:5]  # First 5 residues
            if residues:
                print("🧪 **Sample Residue Data (first 5):**")
                for r in residues:
                    print(f"   • Pos {r['pos']:3d} ({r.get('aa', 'X')}): "
                          f"EWCL={r.get('ewcl', 0):.3f}, "
                          f"pLDDT={r.get('plddt', 'N/A')}, "
                          f"H={r.get('H', 'N/A'):.3f if r.get('H') else 'N/A'}, "
                          f"disagree={r.get('is_disagree', False)}")
            
            # Coverage analysis
            n_total = chain_result.get('n_res_total', 0)
            n_ewcl = chain_result.get('n_ewcl_finite', 0)
            n_conf = chain_result.get('n_plddt_finite', 0)
            n_overlap = chain_result.get('n_overlap_used', 0)
            
            print(f"""
📈 **Coverage Analysis:**
   • EWCL Coverage: {n_ewcl}/{n_total} ({n_ewcl/n_total*100:.1f}% if n_total else 0)%
   • Confidence Coverage: {n_conf}/{n_total} ({n_conf/n_total*100:.1f}% if n_total else 0)%
   • Overlap for H-scores: {n_overlap}/{n_total} ({n_overlap/n_total*100:.1f}% if n_total else 0)%
""")
            
            return chain_result
            
        except Exception as e:
            print(f"❌ Error: {e}")
            return None

def test_api_health():
    """Test the enhanced health endpoint."""
    print("\n🔧 **Testing Enhanced Health Endpoint**")
    print("=" * 50)
    
    try:
        response = requests.get("https://ewcl-api-production.up.railway.app/api/hallucination/health", timeout=10)
        if response.status_code == 200:
            health = response.json()
            print(f"""
✅ **Health Check Results:**
   • Service OK: {health.get('ok', False)}
   • EWCLp3 Loaded: {health.get('ewclp3_loaded', False)}
   • Supported EWCL Sources: {health.get('ewcl_sources_supported', [])}
   • Default Parameters: {health.get('defaults', {})}
""")
        else:
            print(f"❌ Health check failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Health check error: {e}")

def main():
    """Run enhanced hallucination detection tests."""
    print("🔬 **Enhanced EWCL-H Hallucination Detection Demo**")
    print("Testing new features: source tracking, overlap counters, comprehensive JSON")
    
    # Test health first
    test_api_health()
    
    # Test files
    test_files = [
        "AF-P37840-F1-model_v4.cif",  # Known case: 62.1% disagree
        "AF-P41208-F1-model_v4.cif",  # Known case: 84.9% disagree  
    ]
    
    results = {}
    for filename in test_files:
        try:
            result = test_enhanced_hallucination_api(filename)
            if result:
                results[filename] = result
        except KeyboardInterrupt:
            print("\n🛑 Interrupted by user")
            break
        except Exception as e:
            print(f"❌ Failed to process {filename}: {e}")
    
    # Enhanced comparison
    if len(results) > 1:
        print("\n📊 **Enhanced Comparison Summary:**")
        print("-" * 70)
        print(f"{'Protein':<10} {'Source':<10} {'Coverage':<10} {'Disagree':<10} {'Flagged':<8}")
        print("-" * 70)
        
        for filename, result in results.items():
            protein_id = filename.split('-')[1]  # Extract UniProt ID
            source = result.get('ewcl_source', 'N/A')
            n_total = result.get('n_res_total', 0)
            n_overlap = result.get('n_overlap_used', 0)
            coverage = f"{n_overlap}/{n_total}" if n_total > 0 else "N/A"
            disagree = f"{result.get('frac_disagree', 0)*100:.1f}%"
            flagged = "YES" if result.get('flagged', False) else "NO"
            
            print(f"{protein_id:<10} {source:<10} {coverage:<10} {disagree:<10} {flagged:<8}")

if __name__ == "__main__":
    main()