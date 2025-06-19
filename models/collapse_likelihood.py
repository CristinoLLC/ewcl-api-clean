import numpy as np
from typing import Union, Iterable, List

try:
    from Bio.PDB import PDBParser
except ImportError:
    PDBParser = None

class CollapseLikelihood:
    """
    Physics-based Collapse Likelihood (CL) model.
    Computes: CL_i = exp(-lambda * S_i), where S_i = 1 - plddt_i / 100
    """

    def __init__(self, lambda_: float = 3.0):
        self.lambda_ = lambda_

    def score(self, plddt: Union[Iterable[float], np.ndarray]) -> np.ndarray:
        plddt = np.asarray(plddt)
        Si = 1.0 - (plddt / 100.0)
        return np.exp(-self.lambda_ * Si)

    def score_df(self, df, plddt_col: str = "plddt") -> np.ndarray:
        return self.score(df[plddt_col].values)

    def score_from_pdb(self, pdb_path: str) -> List[float]:
        if PDBParser is None:
            raise ImportError("BioPython is required for score_from_pdb()")
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure("p", pdb_path)
        scores = []
        for model in structure:
            for chain in model:
                for residue in chain:
                    if "CA" in residue:
                        plddt = residue["CA"].get_bfactor()
                        Si = 1.0 - (plddt / 100.0)
                        CL = np.exp(-self.lambda_ * Si)
                        scores.append(CL)
        return scores

    def save(self, filepath: str):
        import pickle
        with open(filepath, "wb") as fh:
            pickle.dump({"lambda": self.lambda_}, fh)

    @classmethod
    def load(cls, filepath: str):
        import pickle
        with open(filepath, "rb") as fh:
            obj = pickle.load(fh)
        return cls(lambda_=obj["lambda"])

    def __repr__(self):
        return f"CollapseLikelihood(lambda_={self.lambda_})"
