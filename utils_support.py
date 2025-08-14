from Bio.PDB import PDBParser

def iter_CA_records(pdb_path):
    """
    Yields tuples: (chain_id, seq_id, icode, aa3, ca_b)
    Rules:
      • Only CA atom
      • altloc must be blank ' ' (ignore A/B/etc.)
      • Keep insertion code (icode) as part of key
    """
    parser = PDBParser(QUIET=True, PERMISSIVE=True)
    structure = parser.get_structure("X", pdb_path)
    for model in structure:
        for chain in model:
            for res in chain:
                if "CA" not in res: 
                    continue
                ca_atom = res["CA"]
                # skip alternate locations
                if getattr(ca_atom, "get_altloc", lambda: ' ')() not in (' ', ''):
                    continue
                chain_id = chain.id
                (het, seq_id, icode) = res.id  # icode may be ' '
                aa3 = res.get_resname()
                ca_b = float(ca_atom.get_bfactor())
                yield (chain_id, seq_id, icode, aa3, ca_b)
