import numpy as np
from scipy.stats import pearsonr, spearmanr, kendalltau
from sklearn.metrics import roc_auc_score
import logging
import pandas as pd

def compute_correlation_metrics_cleaned(residues, mode="collapse"):
    """
    Compute correlation metrics with proper data cleaning to avoid misleading results
    """
    # Create DataFrame for easier data handling
    data_rows = []
    for res in residues:
        try:
            if res.get('cl') is not None and res.get('plddt') is not None:
                data_rows.append({
                    'cl': float(res['cl']),
                    'plddt': float(res['plddt']),
                    'b_factor': float(res['b_factor']) if res.get('b_factor') is not None else None
                })
        except (ValueError, TypeError):
            continue
    
    if not data_rows:
        return {"error": "No valid data for correlation analysis"}
    
    # Create DataFrame and drop missing values
    df = pd.DataFrame(data_rows)
    valid_df = df.dropna()
    
    metrics = {
        "mode": mode,
        "n_total_residues": len(residues),
        "n_valid_for_correlation": len(valid_df),
        "vs_pLDDT": {
            "pearson": None,
            "spearman": None,
            "kendall": None
        },
        "vs_b_factor": {
            "pearson": None,
            "spearman": None,
            "kendall": None
        }
    }
    
    # Compute correlations vs pLDDT if sufficient data
    if len(valid_df) >= 3 and len(valid_df['cl'].unique()) > 1:
        try:
            metrics["vs_pLDDT"]["pearson"] = round(float(valid_df["cl"].corr(valid_df["plddt"], method="pearson")), 4)
            metrics["vs_pLDDT"]["spearman"] = round(float(valid_df["cl"].corr(valid_df["plddt"], method="spearman")), 4)
            metrics["vs_pLDDT"]["kendall"] = round(float(valid_df["cl"].corr(valid_df["plddt"], method="kendall")), 4)
        except Exception as e:
            logging.warning(f"pLDDT correlation failed: {e}")
    
    # Compute correlations vs B-factor if available
    valid_bf_df = df.dropna(subset=['cl', 'b_factor'])
    if len(valid_bf_df) >= 3 and len(valid_bf_df['cl'].unique()) > 1:
        try:
            metrics["vs_b_factor"]["pearson"] = round(float(valid_bf_df["cl"].corr(valid_bf_df["b_factor"], method="pearson")), 4)
            metrics["vs_b_factor"]["spearman"] = round(float(valid_bf_df["cl"].corr(valid_bf_df["b_factor"], method="spearman")), 4)
            metrics["vs_b_factor"]["kendall"] = round(float(valid_bf_df["cl"].corr(valid_bf_df["b_factor"], method="kendall")), 4)
        except Exception as e:
            logging.warning(f"B-factor correlation failed: {e}")
    
    return metrics

def compute_correlation_metrics(residues, cl_key='cl', plddt_key='plddt', bfactor_key='b_factor'):
    """
    Compute focused correlation metrics between CL scores and structural data
    
    Args:
        residues: List of residue dictionaries
        cl_key: Key for CL scores in residue dict
        plddt_key: Key for pLDDT scores in residue dict  
        bfactor_key: Key for B-factor scores in residue dict
    
    Returns:
        Dict with correlation metrics
    """
    cl = []
    plddt = []
    bfactor = []

    for res in residues:
        try:
            cl_score = float(res[cl_key])
            
            if plddt_key in res and res[plddt_key] is not None:
                plddt_score = float(res[plddt_key])
                if not np.isnan(plddt_score):
                    cl.append(cl_score)
                    plddt.append(plddt_score)
                    
            if bfactor_key in res and res[bfactor_key] is not None:
                b_score = float(res[bfactor_key])
                if not np.isnan(b_score):
                    bfactor.append(b_score)
        except Exception as e:
            logging.debug(f"Skipping residue due to data issue: {e}")
            continue

    metrics = {
        "n_cl_vs_plddt": len(cl),
        "n_cl_vs_bfactor": len(bfactor),
        "pearson_r": None,
        "spearman_r": None,
        "kendall_tau": None,
        "auc": None,
    }

    # Compute correlations if sufficient data
    if len(cl) >= 3:
        try:
            metrics["pearson_r"] = round(pearsonr(cl, plddt)[0], 4)
        except Exception as e:
            logging.warning(f"Pearson correlation failed: {e}")
            
        try:
            metrics["spearman_r"] = round(spearmanr(cl, plddt)[0], 4)
        except Exception as e:
            logging.warning(f"Spearman correlation failed: {e}")
            
        try:
            metrics["kendall_tau"] = round(kendalltau(cl, plddt)[0], 4)
        except Exception as e:
            logging.warning(f"Kendall tau failed: {e}")
            
        try:
            # Binary classification for AUC: create meaningful labels
            # High CL (>0.7) = predicted disorder/collapse, Low CL (<0.3) = stable
            labels = []
            preds = []
            
            for i, score in enumerate(cl):
                if score > 0.7:
                    labels.append(1)  # High risk
                    preds.append(plddt[i])
                elif score < 0.3:
                    labels.append(0)  # Low risk
                    preds.append(plddt[i])
                    
            if len(set(labels)) > 1 and len(labels) >= 5:
                metrics["auc"] = round(roc_auc_score(labels, preds), 4)
                metrics["auc_n_samples"] = len(labels)
        except Exception as e:
            logging.warning(f"AUC computation failed: {e}")

    return metrics
