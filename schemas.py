from pydantic import BaseModel
from typing import List, Optional

class ResidueScore(BaseModel):
    residue_id: int
    cl: Optional[float] = None          # normalized score
    raw_cl: Optional[float] = None      # raw score
    plddt: Optional[float] = None
    b_factor: Optional[float] = None

class AnalysisMetrics(BaseModel):
    pearson: Optional[float] = None
    spearman: Optional[float] = None
    spearman_local_avg: Optional[float] = None
    kendall_tau: Optional[float] = None
    auc_pseudo_plddt: Optional[float] = None
    n_mismatches: Optional[int] = None
    total_residues: Optional[int] = None

class EWCLRequest(BaseModel):
    pdb_string: str
    threshold: Optional[float] = 0.609
    mode: Optional[str] = "collapse"  # "collapse" or "reverse"
    normalize: Optional[bool] = True
    use_raw_ewcl: Optional[bool] = False

class AnalysisResponse(BaseModel):
    model: str
    lambda_: Optional[float] = None
    normalized: bool
    use_raw_ewcl: bool
    mode: str
    interpretation: str
    has_valid_bfactors: bool
    generated: str
    n_residues: int
    results: List[ResidueScore]
    metrics: AnalysisMetrics

class PDFRequest(BaseModel):
    structure_name: str
    mode: str = "collapse"
    raw_enabled: bool = False
    scores: List[ResidueScore]
    metrics: AnalysisMetrics

class ResidueFeatures(BaseModel):
    # Match your feature pipeline
    b_factor: float
    plddt: float
    hydropathy: float
    charge: float

class PolyPredict
