# smart_structure_loader.py
"""
High-Performance Smart Structure Loader for PDB/CIF files

Uses gemmi (C++ library) for 10-20x faster parsing compared to pure Python.
Automatically detects file format and structure type (AlphaFold vs X-ray).
Maintains identical output format to ensure model logic remains unchanged.
"""

from __future__ import annotations
import io
import math
import statistics as stats
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict
import base64

try:
    import gemmi
except ImportError:
    raise ImportError("gemmi is required: pip install gemmi")

# ----------------------------
# Data model (unified records)
# ----------------------------
@dataclass
class ResidueRec:
    chain: str
    resno: int
    icode: str
    aa: str
    x: float
    y: float
    z: float
    b_iso: Optional[float]     # B-factor if experimental; may carry pLDDT in AF2
    plddt: Optional[float]     # pLDDT if AF2; otherwise None

@dataclass
class StructureBundle:
    source_format: str          # "PDB" or "mmCIF"
    experimental_method: str    # "X-RAY DIFFRACTION", "MODEL:ALPHAFOLD", etc.
    residues: List[ResidueRec]
    # Chosen reference series for correlation/overlap
    ref_name: str               # "plddt" or "b_factor"
    ref_values: List[Optional[float]]  # same order as residues
    # Normalized 0..1 "disorder-likeness" (higher = more disordered/uncertain)
    ref_disorder01: List[Optional[float]]

# ----------------------------
# Format detection and sniffers
# ----------------------------
def sniff_format(blob: bytes) -> str:
    """Fast format detection from file header."""
    head = blob[:2048].decode("utf-8", "ignore").lower()
    if any(pattern in head for pattern in ["data_", "_atom_site.", "loop_"]):
        return "mmCIF"
    return "PDB"

def is_alphafold_block(doc: gemmi.cif.Document) -> bool:
    """
    Detect AlphaFold structures from CIF metadata.
    Checks for MA quality assessment tables or pLDDT-like B-factors.
    """
    try:
        blk = doc.sole_block()
        # Check for AlphaFold QA tables
        if blk.find_mmcif_category("_ma_qa_metric_local") is not None:
            return True
        
        # Fallback: check B-factor range (pLDDT is typically 0-100)
        atom_site = blk.find_mmcif_category("_atom_site")
        if atom_site is not None:
            b_col = (atom_site.get("_atom_site.B_iso_or_equiv") or 
                    atom_site.get("_atom_site.b_iso_or_equiv"))
            if b_col:
                vals = []
                for v in b_col[:100]:  # Sample first 100 atoms
                    try:
                        vals.append(float(v))
                    except:
                        continue
                
                if vals and all(0.0 <= v <= 100.0 for v in vals):
                    # Check if no experimental method is specified
                    exptl = blk.find_mmcif_category("_exptl")
                    if exptl is None or not exptl.get("_exptl.method"):
                        return True
    except Exception:
        pass
    return False

def experimental_method_from_cif(doc: gemmi.cif.Document) -> str:
    """Extract experimental method from CIF metadata."""
    try:
        blk = doc.sole_block()
        exptl = blk.find_mmcif_category("_exptl")
        if exptl and "_exptl.method" in exptl:
            method = exptl["_exptl.method"][0]
            return method.upper()
    except Exception:
        pass
    
    # Fallback to AlphaFold detection
    return "MODEL:ALPHAFOLD" if is_alphafold_block(doc) else "UNKNOWN"

def experimental_method_from_pdb(st: gemmi.Structure) -> str:
    """Extract experimental method from PDB structure."""
    # Check gemmi's metadata
    method = (st.info.get("experiment") or "").upper()
    
    if not method:
        # Heuristic: AlphaFold structures often have AF- in the name
        title = (st.name or "").upper()
        if "AF-" in title or "ALPHAFOLD" in title:
            return "MODEL:ALPHAFOLD"
    
    return method or "UNKNOWN"

# ----------------------------
# High-performance parsers using gemmi
# ----------------------------
def parse_mmcif_fast(blob: bytes) -> Tuple[List[ResidueRec], str]:
    """Fast mmCIF parsing using gemmi C++ library."""
    try:
        doc = gemmi.cif.read_string(blob.decode("utf-8", "ignore"))
        method = experimental_method_from_cif(doc)
        st = gemmi.make_structure_from_block(doc.sole_block())
    except Exception as e:
        raise ValueError(f"Failed to parse mmCIF: {e}")
    
    residues: List[ResidueRec] = []
    
    # Extract CA atoms only for maximum speed
    for model in st:
        for chain in model:
            ch_id = chain.name
            for res in chain:
                # Find CA atom (or first atom if no CA)
                ca_atom = None
                for atom in res:
                    if atom.name == "CA":
                        ca_atom = atom
                        break
                
                if ca_atom is None and len(res) > 0:
                    ca_atom = res[0]  # Fallback to first atom
                
                if ca_atom is None:
                    continue
                
                # Create residue record
                rec = ResidueRec(
                    chain=ch_id,
                    resno=res.seqid.num,
                    icode=res.seqid.icode or "",
                    aa=res.name,  # 3-letter code
                    x=ca_atom.pos.x,
                    y=ca_atom.pos.y,
                    z=ca_atom.pos.z,
                    b_iso=float(ca_atom.b_iso) if not math.isnan(ca_atom.b_iso) else None,
                    plddt=None,  # Will be set based on method
                )
                residues.append(rec)
        break  # Only process first model for speed
    
    # Handle AlphaFold: move B-factor to pLDDT field
    if method.startswith("MODEL:ALPHAFOLD") or is_alphafold_block(doc):
        for r in residues:
            if r.b_iso is not None and 0 <= r.b_iso <= 100:
                r.plddt = r.b_iso
                r.b_iso = None
    
    return residues, method

def parse_pdb_fast(blob: bytes) -> Tuple[List[ResidueRec], str]:
    """Fast PDB parsing using gemmi C++ library."""
    try:
        # Write to temporary file since gemmi.read_structure expects a file path
        import tempfile
        with tempfile.NamedTemporaryFile(mode='wb', suffix='.pdb', delete=False) as tmp_file:
            tmp_file.write(blob)
            tmp_path = tmp_file.name
        
        try:
            st = gemmi.read_structure(tmp_path, format=gemmi.CoorFormat.Pdb)
            method = experimental_method_from_pdb(st)
        finally:
            # Clean up temporary file
            import os
            try:
                os.unlink(tmp_path)
            except:
                pass
    except Exception as e:
        raise ValueError(f"Failed to parse PDB: {e}")
    
    residues: List[ResidueRec] = []
    
    # Extract CA atoms only
    for model in st:
        for chain in model:
            ch_id = chain.name
            for res in chain:
                # Find CA atom
                ca_atom = None
                for atom in res:
                    if atom.name == "CA":
                        ca_atom = atom
                        break
                
                if ca_atom is None and len(res) > 0:
                    ca_atom = res[0]
                
                if ca_atom is None:
                    continue
                
                rec = ResidueRec(
                    chain=ch_id,
                    resno=res.seqid.num,
                    icode=res.seqid.icode or "",
                    aa=res.name,
                    x=ca_atom.pos.x,
                    y=ca_atom.pos.y,
                    z=ca_atom.pos.z,
                    b_iso=float(ca_atom.b_iso) if not math.isnan(ca_atom.b_iso) else None,
                    plddt=None
                )
                residues.append(rec)
        break  # Only first model
    
    # Handle AlphaFold heuristic for PDB format
    if method.startswith("MODEL:ALPHAFOLD"):
        for r in residues:
            if r.b_iso is not None and 0 <= r.b_iso <= 100:
                r.plddt = r.b_iso
                r.b_iso = None
    
    return residues, method

# ----------------------------
# Disorder normalization for consistent interpretation
# ----------------------------
def normalize_disorder(ref_name: str, values: List[Optional[float]]) -> List[Optional[float]]:
    """
    Normalize reference values to 0-1 "disorder-likeness" scale:
    - For pLDDT: disorder = (100 - plddt)/100 (higher = more disordered)
    - For B-factor: robust z-score -> sigmoid (higher = more disordered)
    """
    if ref_name == "plddt":
        # AlphaFold: convert pLDDT to disorder scale
        out = []
        for v in values:
            if v is None:
                out.append(None)
            else:
                # pLDDT 100 = very confident = low disorder (0)
                # pLDDT 0 = very uncertain = high disorder (1)
                disorder = max(0.0, min(1.0, (100.0 - float(v)) / 100.0))
                out.append(disorder)
        return out
    
    # B-factor normalization using robust statistics
    valid_vals = [float(v) for v in values if v is not None]
    if not valid_vals:
        return [None] * len(values)
    
    # Use median and MAD for robustness
    median = stats.median(valid_vals)
    mad = stats.median([abs(v - median) for v in valid_vals]) or 1.0
    
    out = []
    for v in values:
        if v is None:
            out.append(None)
        else:
            # Robust z-score
            z = (float(v) - median) / (1.4826 * mad)
            # Map to 0-1 via sigmoid (z=0 -> 0.5, high B -> 1)
            sigmoid = 1.0 / (1.0 + math.exp(-z))
            out.append(sigmoid)
    
    return out

# ----------------------------
# Convert to legacy format for model compatibility
# ----------------------------
def convert_to_legacy_format(bundle: StructureBundle) -> Dict:
    """
    Convert StructureBundle to the exact format expected by existing model.
    This ensures zero changes to model logic.
    """
    # Map 3-letter to 1-letter amino acids
    AA3_TO_1 = {
        "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
        "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
        "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
        "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
        "MSE": "M", "SEC": "C", "PYL": "K", "HYP": "P",
        "UNK": "X", "ASX": "B", "GLX": "Z"
    }
    
    # Choose longest chain (same logic as original)
    chain_counts = {}
    for r in bundle.residues:
        chain_counts[r.chain] = chain_counts.get(r.chain, 0) + 1
    
    if not chain_counts:
        raise ValueError("No residues found")
    
    chosen_chain = max(chain_counts.keys(), key=lambda c: chain_counts[c])
    
    # Filter to chosen chain and sort by residue number
    chain_residues = [r for r in bundle.residues if r.chain == chosen_chain]
    chain_residues.sort(key=lambda r: (r.resno, r.icode))
    
    # Convert to legacy format
    legacy_residues = []
    for r in chain_residues:
        aa_single = AA3_TO_1.get(r.aa, "X")
        
        # Use appropriate confidence value based on structure type
        if bundle.ref_name == "plddt":
            bfactor_val = r.plddt or 0.0
        else:
            bfactor_val = r.b_iso or 0.0
        
        legacy_residues.append({
            "aa": aa_single,
            "resseq": r.resno,
            "icode": r.icode,
            "bfactor": bfactor_val
        })
    
    # Determine source type for legacy compatibility
    if bundle.experimental_method.startswith("MODEL:ALPHAFOLD"):
        source = "alphafold"
        metric_name = "plddt"
    elif "NMR" in bundle.experimental_method:
        source = "nmr" 
        metric_name = "none"
    else:
        source = "xray"
        metric_name = "bfactor"
    
    return {
        "source": source,
        "metric_name": metric_name,
        "chain": chosen_chain,
        "residues": legacy_residues
    }

# ----------------------------
# Main entry point
# ----------------------------
def load_structure_unified(blob: bytes, *, emit_pdb_diag: bool = False) -> Tuple[StructureBundle, Optional[bytes]]:
    """
    High-performance unified structure loader.
    
    Args:
        blob: Raw file bytes (PDB or CIF)
        emit_pdb_diag: Whether to generate diagnostic PDB output
    
    Returns:
        Tuple of (StructureBundle, optional_diagnostic_pdb_bytes)
    """
    # Fast format detection
    fmt = sniff_format(blob)
    
    # Parse using appropriate fast parser
    if fmt == "mmCIF":
        residues, method = parse_mmcif_fast(blob)
        source_format = "mmCIF"
    else:
        residues, method = parse_pdb_fast(blob)
        source_format = "PDB"
    
    if not residues:
        raise ValueError("No CA atoms found in structure")
    
    # Decide reference series based on experimental method
    if method.startswith("MODEL:ALPHAFOLD"):
        ref_name = "plddt"
        ref_vals = [r.plddt for r in residues]
    elif "X-RAY" in method:
        ref_name = "b_factor"
        ref_vals = [r.b_iso for r in residues]
    else:
        # Fallback: prefer pLDDT if present, else B-factor
        if any(r.plddt is not None for r in residues):
            ref_name = "plddt"
            ref_vals = [r.plddt for r in residues]
        else:
            ref_name = "b_factor" 
            ref_vals = [r.b_iso for r in residues]
    
    # Normalize to disorder scale
    ref_disorder01 = normalize_disorder(ref_name, ref_vals)
    
    # Create bundle
    bundle = StructureBundle(
        source_format=source_format,
        experimental_method=method,
        residues=residues,
        ref_name=ref_name,
        ref_values=ref_vals,
        ref_disorder01=ref_disorder01,
    )
    
    # Optional: generate diagnostic PDB (CA-only minimal format)
    pdb_diag: Optional[bytes] = None
    if emit_pdb_diag:
        lines = []
        serial = 1
        for r in residues:
            # Use appropriate confidence value in temp factor column
            temp_factor = (r.plddt if ref_name == "plddt" else r.b_iso) or 0.0
            
            line = (f"ATOM  {serial:5d}  CA  {r.aa:>3s} {r.chain:1s}{r.resno:4d}{r.icode:1s}"
                   f"   {r.x:8.3f}{r.y:8.3f}{r.z:8.3f}  1.00{temp_factor:6.2f}\n")
            lines.append(line)
            serial += 1
        
        lines.append("END\n")
        pdb_diag = "".join(lines).encode("utf-8")
    
    return bundle, pdb_diag

def load_for_legacy_model(blob: bytes) -> Dict:
    """
    Convenience function that loads structure and converts to legacy format.
    Drop-in replacement for existing PDB parser with identical output format.
    """
    bundle, _ = load_structure_unified(blob)
    return convert_to_legacy_format(bundle)