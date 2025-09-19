"""
EWCL PDB Rewrite Service
========================

Generates PDB files with EWCL scores as B-factors (temperature factors).
This allows standard PDB viewers to display EWCL scores using built-in coloring schemes.

The service:
1. Takes the original PDB/mmCIF file
2. Applies EWCL scores as B-factors (scaled 0-100)
3. Returns a modified PDB file ready for visualization
"""

from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel
from typing import List, Dict, Optional
import re
import tempfile
import os

router = APIRouter(prefix="/ewcl", tags=["ewcl-pdb-rewrite"])

class OverlayResidue(BaseModel):
    """Residue overlay data for PDB rewriting"""
    chain: str
    resi: int
    ewcl: float  # 0-1 scale
    icode: Optional[str] = ""  # insertion code

class PdbRewriteRequest(BaseModel):
    """Request for PDB rewrite with EWCL overlay"""
    overlay: List[OverlayResidue]

def apply_ewcl_to_pdb(pdb_text: str, overlay: List[OverlayResidue]) -> str:
    """
    Rewrites B-factor field (cols 61-66) with scaled EWCL from overlay JSON.
    Keeps formatting; falls back to original B if residue not found.
    
    Args:
        pdb_text: Original PDB file content as string
        overlay: List of residue overlays with EWCL scores
        
    Returns:
        Modified PDB content with EWCL scores as B-factors
    """
    # Index overlay by chain|resi for O(1) lookups
    idx = {}
    for r in overlay:
        key = f"{r.chain}|{r.resi}"
        if r.icode:
            key += f"|{r.icode}"
        idx[key] = r
    
    lines = pdb_text.split('\n')
    
    for i, line in enumerate(lines):
        # Only process ATOM and HETATM lines
        if not re.match(r'^ATOM|^HETATM', line):
            continue
            
        # PDB fixed columns (1-based):
        # chainID: 22, resSeq: 23-26, iCode: 27, B-factor: 61-66
        if len(line) < 66:
            continue
            
        chain = line[21:22].strip()  # col 22
        resi_str = line[22:26].strip()  # cols 23-26
        icode = line[26:27].strip() if len(line) > 26 else ""  # col 27
        
        try:
            resi = int(resi_str)
        except ValueError:
            continue
            
        # Look up overlay data
        key = f"{chain}|{resi}"
        if icode:
            key += f"|{icode}"
            
        overlay_res = idx.get(key)
        if not overlay_res:
            continue
            
        # Scale EWCL to 0-100 range and format as B-factor
        ewcl_100 = max(0, min(100, overlay_res.ewcl * 100))
        b_str = f"{ewcl_100:6.2f}"  # width 6, right-aligned
        
        # Replace B-factor in cols 61-66 (0-based slice 60:66)
        if len(line) >= 66:
            lines[i] = line[:60] + b_str + line[66:]
        else:
            # Pad line if too short
            padded_line = line.ljust(66)
            lines[i] = padded_line[:60] + b_str
    
    return '\n'.join(lines)

def apply_ewcl_to_cif(cif_text: str, overlay: List[OverlayResidue]) -> str:
    """
    Rewrites B_iso_or_equiv field in mmCIF with scaled EWCL from overlay JSON.
    
    Args:
        cif_text: Original mmCIF file content as string
        overlay: List of residue overlays with EWCL scores
        
    Returns:
        Modified mmCIF content with EWCL scores as B-factors
    """
    # Index overlay by chain|resi for O(1) lookups
    idx = {}
    for r in overlay:
        key = f"{r.chain}|{r.resi}"
        if r.icode:
            key += f"|{r.icode}"
        idx[key] = r
    
    lines = cif_text.split('\n')
    in_atom_site = False
    atom_site_start = -1
    
    for i, line in enumerate(lines):
        # Check if we're in atom_site loop
        if line.strip().startswith('_atom_site.'):
            in_atom_site = True
            if atom_site_start == -1:
                atom_site_start = i
            continue
        elif line.strip().startswith('loop_') or line.strip().startswith('data_'):
            in_atom_site = False
            atom_site_start = -1
            continue
        elif in_atom_site and line.strip() and not line.startswith('#'):
            # This is a data line in atom_site loop
            parts = line.split()
            if len(parts) >= 11:  # Ensure we have enough columns
                try:
                    # mmCIF atom_site columns (typical order):
                    # group_PDB, id, type_symbol, label_atom_id, label_alt_id, 
                    # label_comp_id, label_asym_id, label_entity_id, label_seq_id, 
                    # pdbx_PDB_ins_code, Cartn_x, Cartn_y, Cartn_z, occupancy, B_iso_or_equiv
                    chain = parts[6]  # label_asym_id
                    resi = int(parts[8])  # label_seq_id
                    icode = parts[9] if len(parts) > 9 else ""  # pdbx_PDB_ins_code
                    
                    # Look up overlay data
                    key = f"{chain}|{resi}"
                    if icode and icode != ".":
                        key += f"|{icode}"
                        
                    overlay_res = idx.get(key)
                    if overlay_res:
                        # Scale EWCL to 0-100 range
                        ewcl_100 = max(0, min(100, overlay_res.ewcl * 100))
                        # Replace B_iso_or_equiv (typically column 14, 0-based index 14)
                        if len(parts) > 14:
                            parts[14] = f"{ewcl_100:.2f}"
                        elif len(parts) == 14:
                            parts.append(f"{ewcl_100:.2f}")
                        else:
                            # Pad with empty values if needed
                            while len(parts) < 14:
                                parts.append(".")
                            parts.append(f"{ewcl_100:.2f}")
                        
                        lines[i] = " ".join(parts)
                except (ValueError, IndexError):
                    continue
    
    return '\n'.join(lines)

@router.post("/rewrite-pdb", response_class=PlainTextResponse)
async def rewrite_pdb_with_ewcl(
    file: UploadFile = File(..., description="PDB or mmCIF structure file"),
    overlay: str = File(..., description="JSON overlay data with EWCL scores")
):
    """
    Rewrite PDB/mmCIF file with EWCL scores as B-factors.
    
    This endpoint takes a structure file and overlay data, then returns
    a modified PDB file where the B-factor column contains EWCL scores
    scaled to 0-100 range. This allows standard PDB viewers to display
    EWCL scores using built-in temperature factor coloring.
    
    Args:
        file: Original PDB or mmCIF structure file
        overlay: JSON string containing overlay data with EWCL scores
        
    Returns:
        Modified PDB file with EWCL scores as B-factors
    """
    try:
        # Read and validate input
        data = await file.read()
        if not data:
            raise HTTPException(status_code=400, detail="Empty upload.")
        
        # Parse overlay data
        import json
        try:
            overlay_data = json.loads(overlay)
            # Handle both flat array and nested structure
            if "residues" in overlay_data:
                residues_data = overlay_data["residues"]
            else:
                residues_data = overlay_data
            
            overlay_residues = []
            for item in residues_data:
                # Map auth_asym_id to chain and auth_seq_id to resi for compatibility
                if "auth_asym_id" in item and "auth_seq_id" in item:
                    overlay_residues.append(OverlayResidue(
                        chain=item["auth_asym_id"],
                        resi=item["auth_seq_id"],
                        ewcl=item["ewcl"],
                        icode=item.get("icode", "")
                    ))
                else:
                    overlay_residues.append(OverlayResidue(**item))
        except (json.JSONDecodeError, ValueError) as e:
            raise HTTPException(status_code=400, detail=f"Invalid overlay JSON: {e}")
        
        # Detect file format
        text_data = data.decode('utf-8')
        is_cif = text_data.strip().startswith('data_')
        
        # Apply EWCL overlay
        if is_cif:
            modified_content = apply_ewcl_to_cif(text_data, overlay_residues)
            content_type = "chemical/x-mmcif"
        else:
            modified_content = apply_ewcl_to_pdb(text_data, overlay_residues)
            content_type = "chemical/x-pdb"
        
        # Return modified file
        return PlainTextResponse(
            content=modified_content,
            media_type=content_type,
            headers={
                "Content-Disposition": f"attachment; filename=ewcl_{file.filename}",
                "X-EWCL-Applied": "true",
                "X-Scale": "0-100"
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")

@router.post("/rewrite-pdb-from-analysis", response_class=PlainTextResponse)
async def rewrite_pdb_from_analysis(
    file: UploadFile = File(..., description="PDB or mmCIF structure file")
):
    """
    Convenience endpoint that analyzes the structure and immediately returns
    a PDB file with EWCL scores as B-factors.
    
    This combines structure analysis and PDB rewriting in a single call.
    """
    try:
        # First, analyze the structure to get EWCL scores
        from backend.api.routers.ewclv1p3_fresh import analyze_pdb_ewclv1_p3_fresh
        
        # Read file data once
        data = await file.read()
        
        # Create a new UploadFile for analysis
        from fastapi import UploadFile as FastAPIUploadFile
        from io import BytesIO
        
        analysis_file = FastAPIUploadFile(
            file=BytesIO(data),
            filename=file.filename
        )
        
        # Analyze structure
        analysis_result = await analyze_pdb_ewclv1_p3_fresh(analysis_file)
        
        # Convert analysis result to overlay format
        overlay_residues = []
        for residue in analysis_result.residues:
            overlay_residues.append(OverlayResidue(
                chain=residue.auth_asym_id,
                resi=residue.auth_seq_id,
                ewcl=residue.ewcl,
                icode=residue.icode
            ))
        
        # Use the already read data
        text_data = data.decode('utf-8')
        
        # Apply EWCL overlay - always return PDB format
        is_cif = text_data.strip().startswith('data_')
        if is_cif:
            # For mmCIF files, we need to convert to PDB format first
            # Use the same parsing logic as the analysis endpoint
            from backend.api.utils.structure_io import parse_structure
            df, backend = parse_structure(data, file.filename)
            if df is None or df.empty:
                raise HTTPException(status_code=400, detail="Failed to parse mmCIF structure")
            
            # Convert DataFrame to PDB format
            pdb_lines = []
            for _, row in df.iterrows():
                if row.get('atom') in ['CA', 'CB', 'C', 'N', 'O']:  # Only backbone atoms for simplicity
                    pdb_line = f"ATOM  {row.get('serial', 1):5d} {row.get('atom', 'CA'):4s} {row.get('resname', 'UNK'):3s} {row.get('chain', 'A'):1s}{row.get('auth_seq_id', 1):4d}    {row.get('x', 0):8.3f}{row.get('y', 0):8.3f}{row.get('z', 0):8.3f}  1.00{row.get('bfactor', 0):6.2f}           {row.get('element', 'C'):2s}"
                    pdb_lines.append(pdb_line)
            
            pdb_content = '\n'.join(pdb_lines)
            modified_content = apply_ewcl_to_pdb(pdb_content, overlay_residues)
        else:
            modified_content = apply_ewcl_to_pdb(text_data, overlay_residues)
        
        content_type = "chemical/x-pdb"
        
        # Return modified file
        return PlainTextResponse(
            content=modified_content,
            media_type=content_type,
            headers={
                "Content-Disposition": f"attachment; filename=ewcl_{file.filename}",
                "X-EWCL-Applied": "true",
                "X-Scale": "0-100",
                "X-Model": analysis_result.model
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")
