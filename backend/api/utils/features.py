import numpy as np

# if your model exposes an explicit feature order, use it; else fall back to intersect
def align_features_for_model(model, df):
    # keep the amino acids for the response
    aas = df["aa"].tolist() if "aa" in df.columns else ["?"] * len(df)

    # numeric columns only
    num_df = df.select_dtypes(include=["number"]).copy()
    num_df = num_df.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    if hasattr(model, "feature_order_"):
        cols = [c for c in model.feature_order_ if c in num_df.columns]
        X = num_df.reindex(columns=cols, fill_value=0.0).to_numpy(dtype=np.float32)
    elif hasattr(model, "n_features_in_"):
        # keep as many columns as needed; pad/trim to n_features_in_
        arr = num_df.to_numpy(dtype=np.float32)
        need = int(model.n_features_in_)
        have = arr.shape[1]
        if have == need:
            X = arr
        elif have > need:
            X = arr[:, :need]
        else:
            pad = np.zeros((arr.shape[0], need - have), dtype=np.float32)
            X = np.concatenate([arr, pad], axis=1)
    else:
        # last resort: use all numeric cols
        X = num_df.to_numpy(dtype=np.float32)

    return X, aas