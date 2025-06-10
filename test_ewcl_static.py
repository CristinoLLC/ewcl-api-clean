#!/usr/bin/env python3
"""
Test script for enhanced EWCL static tool
"""

from ewcl_toolkit.ewcl_static_tool import ewcl_score_protein, parse_fasta, ewcl_from_fasta

def test_ewcl_tool():
    """Test the enhanced EWCL static tool with various inputs"""
    
    # Test 1: Direct sequence input
    print("🧪 Test 1: Direct sequence input")
    sequence = "MTEYKLVVVGAGGVGKSALTIQLIQNHFVDEYDPTIEDSYRKQV"
    scores = ewcl_score_protein(sequence)
    print(f"Sequence length: {len(sequence)}")
    print(f"EWCL scores computed: {len(scores)}")
    print(f"Score range: {min(scores.values()):.4f} - {max(scores.values()):.4f}")
    print(f"Sample scores: {dict(list(scores.items())[:5])}")
    print()
    
    # Test 2: FASTA format input
    print("🧪 Test 2: FASTA format input")
    fasta = """>Test Protein
MTEYKLVVVGAGGVGKSALTIQLIQNHFVDEYDPTIEDSYRKQV
DEYDPTIEDSYRKQVVIDGETCLLDILDTAGQEEYSAMRDQYMRTGEGFLCVFAINNTKSFEDIHQYREQIKRVKDSDDVPMVLVGNKCDLAAR"""
    
    scores_fasta = ewcl_from_fasta(fasta)
    print(f"FASTA sequence length: {len(parse_fasta(fasta))}")
    print(f"EWCL scores computed: {len(scores_fasta)}")
    print(f"Score range: {min(scores_fasta.values()):.4f} - {max(scores_fasta.values()):.4f}")
    print(f"Sample scores: {dict(list(scores_fasta.items())[:5])}")
    print()
    
    # Test 3: Sequence with non-standard amino acids
    print("🧪 Test 3: Non-standard amino acids handling")
    sequence_with_x = "MTEYKLVVVGAGGVGKSALTIQLIQNHFVDEYDPTIEDSYRKQVX"
    scores_x = ewcl_score_protein(sequence_with_x)
    print(f"Sequence with X: {sequence_with_x}")
    print(f"Score for position {len(sequence_with_x)} (X): {scores_x[len(sequence_with_x)]}")
    print()
    
    print("✅ All tests completed successfully!")

if __name__ == "__main__":
    test_ewcl_tool()