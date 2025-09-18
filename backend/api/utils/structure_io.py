from __future__ import annotations
import io, gzip
from typing import Optional, Tuple, List, Dict, Any
import pandas as pd

import gemmi
from Bio.PDB import PDBParser, MMCIFParser, is_aa

PDB_MIME = {"chemical/x-pdb", "text/plain"}
CIF_MIME = {"application/mmcif+text", "chemical/x-mmcif", "application/octet-stream"}

def maybe_gunzip(data: bytes) -> bytes:
    if len(data) >= 2 and data[:2] == b"\x1f\x8b":
        return gzip.decompress(data)
    return data

def sniff_format(data: bytes, filename: str | None) -> str:
    name = (filename or "").lower()
    head = data[:4096].decode("latin-1", errors="ignore")

    if name.endswith(".cif") or "_atom_site." in head or "data_" in head:
        return "cif"
    if name.endswith(".pdb") or "ATOM  " in head or "HETATM" in head:
        return "pdb"
    # gemmi can still read either; default to cif if it smells like CIF
    return "cif" if "_atom_site." in head else "pdb"

def parse_with_gemmi(data: bytes, fmt: str) -> gemmi.Structure:
    if fmt == "cif":
        doc = gemmi.cif.read_string(data.decode("utf-8", errors="ignore"))
        return gemmi.make_structure_from_block(doc.sole_block())
    else:
        # gemmi reads PDB from text
        text = data.decode("utf-8", errors="ignore")
        return gemmi.read_pdb_string(text)

def parse_with_biopython(data: bytes, fmt: str):
    handle = io.StringIO(data.decode("utf-8", errors="ignore"))
    if fmt == "cif":
        parser = MMCIFParser(QUIET=True)
    else:
        parser = PDBParser(QUIET=True)
    return parser.get_structure("input", handle)

def to_rows_gemmi(st: gemmi.Structure) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    # first model only by default; change if you want ensemble
    model = st[0]
    # polymer chains only
    for chain in model:
        if not chain.is_polymer():
            continue
        for res in chain:
            if not res.is_amino_acid():
                continue
            # prefer CA; fallback to N, C, CB
            atom = None
            for name in ("CA", "N", "C", "CB"):
                atom = res.find_atom(name, altloc="") or res.find_atom(name, altloc="A")
                if atom:
                    break
            if not atom:
                continue
            # altloc: pick highest occupancy
            if atom.altloc:
                best = atom
                occ_best = atom.occ
                for a in res:
                    if a.name == atom.name and a.altloc and a.occ > occ_best:
                        best, occ_best = a, a.occ
                atom = best
            # numbering
            auth_seq_id = res.seqid.num
            # label_seq_id may be stored in CIF
            label_seq = res.get_canonical_num()  # continuous index or -1
            rows.append(dict(
                model=0,
                chain=str(chain.name),
                auth_seq_id=int(auth_seq_id) if auth_seq_id is not None else None,
                seq_id=int(label_seq) if label_seq is not None else None,
                resname=res.name,
                x=float(atom.pos.x),
                y=float(atom.pos.y),
                z=float(atom.pos.z),
                atom=atom.name,
                bfactor=float(atom.b_iso) if hasattr(atom, 'b_iso') else 0.0,
            ))
    return pd.DataFrame(rows)

def to_rows_biopython(st) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    model = next(st.get_models())
    for chain in model:
        # polymer only: keep amino-acids
        for res in chain:
            if not is_aa(res, standard=True):
                continue
            # CA preferred
            atom = res["CA"] if "CA" in res else None
            if atom is None:
                for alt in ("N", "C", "CB"):
                    if alt in res:
                        atom = res[alt]
                        break
            if atom is None:
                continue
            # altloc: Bio.PDB Atom has get_occupancy()
            occ = getattr(atom, "get_occupancy", lambda: 1.0)() or 1.0
            rows.append(dict(
                model=0,
                chain=chain.id,
                auth_seq_id=res.get_id()[1],  # author numbering
                seq_id=None,                  # Bio.PDB doesn't track label_seq_id
                resname=res.get_resname(),
                x=float(atom.coord[0]),
                y=float(atom.coord[1]),
                z=float(atom.coord[2]),
                atom=atom.get_name(),
                occ=float(occ),
                bfactor=float(atom.get_bfactor()) if hasattr(atom, 'get_bfactor') else 0.0,
            ))
    return pd.DataFrame(rows)

def parse_structure(data: bytes, filename: Optional[str]) -> Tuple[pd.DataFrame, str]:
    data = maybe_gunzip(data)
    fmt = sniff_format(data, filename)

    # Try Gemmi first
    try:
        st = parse_with_gemmi(data, fmt)
        df = to_rows_gemmi(st)
        if df.empty:
            raise ValueError("No polymer residues with CA/N/C/CB atoms found (gemmi).")
        return df, "gemmi"
    except Exception as e_gemmi:
        # Fallback to Bio.PDB
        try:
            st = parse_with_biopython(data, fmt)
            df = to_rows_biopython(st)
            if df.empty:
                raise ValueError("No polymer residues with CA/N/C/CB atoms found (Bio.PDB).")
            return df, "biopython"
        except Exception as e_bio:
            raise ValueError(f"Failed to parse structure: gemmi={e_gemmi} ; biopython={e_bio}")
