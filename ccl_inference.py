import os
import io
import joblib
import numpy as np
import pandas as pd

# ---------- paths (lazy load to avoid boot crashes) ----------
CCL_MODEL_PATH = os.getenv("CCL_MODEL_PATH", "models/ccl_v3_seqflip.pkl")
_BUNDLE = None
_MODELS = None
_META = None


def _load_ccl_once():
    global _BUNDLE, _MODELS, _META
    if _BUNDLE is not None:
        return _BUNDLE, _MODELS, _META
    if not os.path.exists(CCL_MODEL_PATH):
        raise FileNotFoundError(
            f"CCL model not found at '{CCL_MODEL_PATH}'. Set CCL_MODEL_PATH or deploy models/ccl_v3_seqflip.pkl"
        )
    bundle = joblib.load(CCL_MODEL_PATH)
    _BUNDLE = bundle
    _MODELS = bundle["models"]
    _META = bundle["meta"]
    return _BUNDLE, _MODELS, _META


# ---------- tiny utils ----------
def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def model_info() -> dict:
    try:
        _, models, meta = _load_ccl_once()
        return {
            "name": "CCL-V3 (Seq-Flip)",
            "folds": len(models),
            "n_features_expected": len(meta.get("X_cols", [])),
            "wins": meta.get("wins", [7, 15, 31]),
            "loaded": True,
        }
    except FileNotFoundError:
        return {"name": "CCL-V3 (Seq-Flip)", "loaded": False}


# ---------- build_features (EXACT as training) ----------
def build_features(df0: pd.DataFrame):
    df = df0.copy()

    for c in ["uniprot", "residue_index", "aa", "label"]:
        if c not in df.columns:
            raise RuntimeError(f"Missing column: {c}")

    AA20X = list("ACDEFGHIKLMNPQRSTVWY") + ["X"]
    aa2idx = {a: i for i, a in enumerate(AA20X)}

    hydropathy = {'A':1.8,'C':2.5,'D':-3.5,'E':-3.5,'F':2.8,'G':-0.4,'H':-3.2,'I':4.5,
                  'K':-3.9,'L':3.8,'M':1.9,'N':-3.5,'P':-1.6,'Q':-3.5,'R':-4.5,'S':-0.8,
                  'T':-0.7,'V':4.2,'W':-0.9,'Y':-1.3,'X':0.0}
    charge = {'R':1,'K':1,'H':0.5,'D':-1,'E':-1,'X':0.0}
    cf_helix = {'A':1.42,'C':0.70,'D':1.01,'E':1.51,'F':1.13,'G':0.57,'H':1.00,'I':1.08,
                'K':1.16,'L':1.21,'M':1.45,'N':0.67,'P':0.57,'Q':1.11,'R':0.98,'S':0.77,
                'T':0.83,'V':1.06,'W':1.08,'Y':0.69,'X':1.0}
    cf_sheet = {'A':0.83,'C':1.19,'D':0.54,'E':0.37,'F':1.38,'G':0.75,'H':0.87,'I':1.60,
                'K':0.74,'L':1.30,'M':1.05,'N':0.89,'P':0.55,'Q':1.10,'R':0.93,'S':0.75,
                'T':1.19,'V':1.70,'W':1.37,'Y':1.47,'X':1.0}
    cf_coil =  {'A':0.66,'C':1.19,'D':1.46,'E':1.01,'F':0.60,'G':1.64,'H':0.95,'I':0.47,
                'K':1.01,'L':0.59,'M':0.60,'N':1.56,'P':1.52,'Q':0.98,'R':0.95,'S':1.43,
                'T':0.96,'V':0.50,'W':0.88,'Y':1.14,'X':1.0}
    IDR_SET = set("EDKPRSQGN")

    df["aa"] = df["aa"].astype(str).str.upper()
    df.loc[~df["aa"].isin(AA20X), "aa"] = "X"

    df["hydropathy"] = df["aa"].map(hydropathy).astype(float)
    df["charge"]     = df["aa"].map(charge).astype(float)
    df["helix_p"]    = df["aa"].map(cf_helix).astype(float)
    df["sheet_p"]    = df["aa"].map(cf_sheet).astype(float)
    df["coil_p"]     = df["aa"].map(cf_coil).astype(float)
    df["aa_code"]    = df["aa"].map(aa2idx).astype(int)

    df = df.sort_values(["uniprot", "residue_index"], kind="stable")
    df["L"]        = df.groupby("uniprot")["residue_index"].transform("max").clip(lower=1)
    df["pos_norm"] = df["residue_index"]/df["L"]
    df["is_N10"]   = (df["residue_index"] <= (0.10*df["L"])).astype(int)
    df["is_C10"]   = ((df["L"]-df["residue_index"]) <= (0.10*df["L"])).astype(int)

    def rmean(arr, w):
        return pd.Series(arr, dtype=float).rolling(w, min_periods=1, center=True).mean().to_numpy()

    def rstd(arr, w):
        return pd.Series(arr, dtype=float).rolling(w, min_periods=1, center=True).std().fillna(0.0).to_numpy()

    def entropy_window(ws):
        vals, cnts = np.unique(ws, return_counts=True)
        p = cnts / cnts.sum()
        return float((-(p*np.log(p)).sum()) / np.log(20))

    def max_frac(ws):
        _, cnts = np.unique(ws, return_counts=True)
        return float(cnts.max()/len(ws))

    def idr_fraction(ws):
        return float(sum(a in IDR_SET for a in ws)/len(ws))

    def morf_score(ws):
        h = np.array([hydropathy.get(a,0.0) for a in ws], dtype=float)
        c = np.array([cf_coil.get(a,1.0)     for a in ws], dtype=float)
        return float(-h.mean() + (c.mean()-1.0))

    def charge_feats(ws):
        pos, neg = set("KRH"), set("DE")
        fpos = sum(a in pos for a in ws)/len(ws)
        fneg = sum(a in neg for a in ws)/len(ws)
        half = len(ws)//2
        left, right = ws[:half], ws[half:]
        lpos = sum(a in pos for a in left)/max(1,len(left))
        rpos = sum(a in pos for a in right)/max(1,len(right))
        lneg = sum(a in neg for a in left)/max(1,len(left))
        rneg = sum(a in neg for a in right)/max(1,len(right))
        kappa = abs(lpos-rpos) + abs(lneg-rneg)
        runp = runn = curp = curn = 0
        for a in ws:
            if a in pos:
                curp += 1; runp = max(runp, curp); curn = 0
            elif a in neg:
                curn += 1; runn = max(runn, curn); curp = 0
            else:
                curp = curn = 0
        return fpos, fneg, kappa, float(runp), float(runn)

    WINS = [7, 15, 31]
    for pid in list(df["uniprot"].unique()):
        sub = df[df["uniprot"] == pid]; idx = sub.index
        aa_seq = sub["aa"].tolist()
        h = sub["hydropathy"].to_numpy()
        q = sub["charge"].to_numpy()
        he = sub["helix_p"].to_numpy()
        sh = sub["sheet_p"].to_numpy()
        co = sub["coil_p"].to_numpy()

        for w in WINS:
            df.loc[idx, f"hydropathy_m{w}"] = rmean(h,w); df.loc[idx, f"hydropathy_s{w}"] = rstd(h,w)
            df.loc[idx, f"charge_m{w}"]     = rmean(q,w); df.loc[idx, f"charge_s{w}"]     = rstd(q,w)
            df.loc[idx, f"helix_p_m{w}"]    = rmean(he,w); df.loc[idx, f"helix_p_s{w}"]  = rstd(he,w)
            df.loc[idx, f"sheet_p_m{w}"]    = rmean(sh,w); df.loc[idx, f"sheet_p_s{w}"]  = rstd(sh,w)
            df.loc[idx, f"coil_p_m{w}"]     = rmean(co,w); df.loc[idx, f"coil_p_s{w}"]   = rstd(co,w)

            half = w//2; n = len(aa_seq)
            ent,cb,idrf,morf,fpos,fneg,kap,rpos,rneg = ([] for _ in range(9))
            for i in range(n):
                a = max(0, i-half); b = min(n, i+half+1)
                ws = aa_seq[a:b]
                ent.append(entropy_window(ws)); cb.append(max_frac(ws)); idrf.append(idr_fraction(ws))
                morf.append(morf_score(ws))
                c0,c1,c2,c3,c4 = charge_feats(ws)
                fpos.append(c0); fneg.append(c1); kap.append(c2); rpos.append(c3); rneg.append(c4)

            df.loc[idx, f"entropy_m{w}"]   = np.array(ent,  dtype=float)
            df.loc[idx, f"comp_bias_m{w}"] = np.array(cb,   dtype=float)
            df.loc[idx, f"idrfrac_m{w}"]   = np.array(idrf, dtype=float)
            df.loc[idx, f"morf_m{w}"]      = np.array(morf, dtype=float)
            df.loc[idx, f"fpos_m{w}"]      = np.array(fpos, dtype=float)
            df.loc[idx, f"fneg_m{w}"]      = np.array(fneg, dtype=float)
            df.loc[idx, f"kappa_m{w}"]     = np.array(kap,  dtype=float)
            df.loc[idx, f"runpos_m{w}"]    = np.array(rpos, dtype=float)
            df.loc[idx, f"runneg_m{w}"]    = np.array(rneg, dtype=float)
            df.loc[idx, f"fcr_m{w}"]       = df.loc[idx, f"fpos_m{w}"] + df.loc[idx, f"fneg_m{w}"]
            df.loc[idx, f"ncpr_m{w}"]      = df.loc[idx, f"fpos_m{w}"] - df.loc[idx, f"fneg_m{w}"]

            fpr_pos = ( np.abs(df.loc[idx, f"charge_m{w}"])
                        + np.maximum(-df.loc[idx, f"hydropathy_m{w}"] , 0.0)
                        + (df.loc[idx, f"coil_p_m{w}"] - 1.0)
                        + df.loc[idx, f"entropy_m{w}"] )
            fpr_neg = ( (df.loc[idx, f"helix_p_m{w}"] + df.loc[idx, f"sheet_p_m{w}"] - 2.0)
                        + np.maximum(df.loc[idx, f"hydropathy_m{w}"] , 0.0) )
            df.loc[idx, f"fpr_pos_m{w}"] = fpr_pos.astype(float)
            df.loc[idx, f"fpr_neg_m{w}"] = fpr_neg.astype(float)

            order_proxy    = sigmoid((df.loc[idx, f"helix_p_m{w}"] + df.loc[idx, f"sheet_p_m{w}"]) / 2.0
                                     + np.maximum(df.loc[idx, f"hydropathy_m{w}"] , 0.0))
            disorder_proxy = sigmoid(np.abs(df.loc[idx, f"charge_m{w}"])
                                     + np.maximum(-df.loc[idx, f"hydropathy_m{w}"] , 0.0)
                                     + df.loc[idx, f"coil_p_m{w}"]
                                     + df.loc[idx, f"entropy_m{w}"])
            conf = (disorder_proxy - order_proxy).astype(float)
            df.loc[idx, f"conflict_m{w}"]    = conf
            thr = np.nanquantile(conf, 0.90) if np.isfinite(conf).any() else 1e9
            df.loc[idx, f"is_conflict_m{w}"] = (conf > thr).astype(int)
            df.loc[idx, f"is_lcr_m{w}"] = ((df.loc[idx, f"entropy_m{w}"] < 0.55) &
                                           (df.loc[idx, f"comp_bias_m{w}"] > 0.35)).astype(int)

    df = pd.concat([df, pd.get_dummies(df["aa"], prefix="aa", dtype=float)], axis=1)

    label_col = "label"
    id_cols = ["uniprot","residue_index","aa",label_col]
    X_cols = [c for c in df.columns if c not in id_cols and pd.api.types.is_numeric_dtype(df[c])]
    df[X_cols] = df[X_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    return df, dict(X_cols=X_cols, label_col=label_col, wins=[7,15,31], aa_cats=list("ACDEFGHIKLMNPQRSTVWY")+["X"])


# ---------- feature alignment to the expected training columns ----------
def make_inference_frame(sequence: str, uniprot_id: str):
    _, _, meta = _load_ccl_once()
    L = len(sequence)
    df0 = pd.DataFrame({
        "uniprot": [uniprot_id]*L,
        "residue_index": np.arange(1, L+1),
        "aa": list(sequence),
        "label": [0]*L,
    })
    built, _ = build_features(df0)

    want = meta["X_cols"]
    X = pd.DataFrame(index=built.index)
    for c in want:
        X[c] = built[c].astype(float) if c in built.columns else 0.0
    return X, built


# ---------- public prediction ----------
def predict_protein(sequence: str, uniprot_id: str) -> dict:
    bundle, models, meta = _load_ccl_once()
    X, built = make_inference_frame(sequence, uniprot_id)
    preds = np.zeros(len(X), dtype=float)
    for m in models:
        preds += m["cal"].predict_proba(X)[:, 1]
    preds /= max(1, len(models))

    residues = []
    for ri, aa, p in zip(built["residue_index"].tolist(), built["aa"].tolist(), preds.tolist()):
        residues.append({
            "residue_index": int(ri),
            "amino_acid": aa,
            "prediction": {
                "disorder_score": float(p),
                "confidence_percentage": float(p * 100.0),
            },
        })

    return {
        "protein_id": uniprot_id,
        "sequence": sequence,
        "model_info": {
            "name": "CCL-V3 (Seq-Flip)",
            "folds": len(models),
            "n_features_expected": len(meta.get("X_cols", [])),
            "wins": meta.get("wins", [7, 15, 31]),
        },
        "residues": residues,
    }


