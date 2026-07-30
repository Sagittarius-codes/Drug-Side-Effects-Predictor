"""predict_utils.py — Drug Side Effect Predictor inference module.

Used by app.py (Streamlit) and any downstream consumer.
Loads all artifacts once at import time.

DISCLAIMER: Predictions are for educational/portfolio purposes only.
This tool must not be used to inform real medical decisions.
"""

import ast
import numpy as np
import pandas as pd
from scipy import sparse
import joblib
from pathlib import Path

# ── Resolve paths relative to this file ───────────────────────────────────────
_HERE      = Path(__file__).resolve().parent
PROCESSED  = _HERE / "data" / "processed"
SPLITS     = _HERE / "data" / "splits"
MODELS     = _HERE / "models"
ENCODERS   = _HERE / "models" / "encoders"

THRESHOLD  = 0.50   # CV-tuned in Week 6

# ── Load all artifacts once ────────────────────────────────────────────────────
train_index = pd.read_csv(PROCESSED / "train_index.csv")
test_index  = pd.read_csv(PROCESSED / "test_index.csv")
label_names = pd.read_csv(PROCESSED / "label_names.csv")["side_effect"].tolist()

train_df    = pd.read_csv(SPLITS / "train.csv")
test_df     = pd.read_csv(SPLITS / "test.csv")
all_df      = pd.concat([train_df, test_df], ignore_index=True)

mlb         = joblib.load(ENCODERS / "mlb_atc.pkl")
tfidf       = joblib.load(ENCODERS / "tfidf_indications.pkl")
model       = joblib.load(MODELS   / "lr_tuned.pkl")

_lookup = all_df.copy()
_lookup["_name_lower"] = _lookup["drug_name"].str.lower().str.strip()
_lookup = _lookup.set_index("_name_lower")


def predict(drug_name: str, threshold: float = THRESHOLD) -> dict:
    """Predict side effects for a SIDER drug. See notebook Step 7.1 for full docs."""
    key = drug_name.lower().strip()

    if key not in _lookup.index:
        return {
            "found"       : False,
            "drug_name"   : drug_name,
            "stitch_id"   : None,
            "predictions" : pd.DataFrame(columns=["side_effect", "score"]),
            "n_predicted" : 0,
            "split"       : "unknown",
        }

    row = _lookup.loc[key]
    if isinstance(row, pd.DataFrame):
        row = row.iloc[0]

    canonical_name = row["drug_name"]
    stitch_id      = row["stitch_id_flat"]

    if stitch_id in train_index["stitch_id_flat"].values:
        split = "train"
    elif stitch_id in test_index["stitch_id_flat"].values:
        split = "test"
    else:
        split = "unknown"

    raw_atc = row["atc_codes"]
    try:
        atc_list = ast.literal_eval(raw_atc) if isinstance(raw_atc, str) else []
    except (ValueError, SyntaxError):
        atc_list = []

    atc_3char = list({c[:3] for c in atc_list if len(c) >= 3})
    X_atc   = mlb.transform([atc_3char])
    indication = row["indications_text"] if pd.notna(row.get("indications_text")) else ""
    X_tfidf = tfidf.transform([str(indication)])
    X       = sparse.hstack([X_atc, X_tfidf], format="csr")

    scores = model.predict_proba(X)[0]
    mask   = scores >= threshold
    if not mask.any():
        mask[np.argmax(scores)] = True

    results = pd.DataFrame({
        "side_effect": [label_names[i] for i in np.where(mask)[0]],
        "score"      : scores[mask],
    }).sort_values("score", ascending=False).reset_index(drop=True)

    return {
        "found"       : True,
        "drug_name"   : canonical_name,
        "stitch_id"   : stitch_id,
        "predictions" : results,
        "n_predicted" : len(results),
        "split"       : split,
    }


def list_all_drugs() -> list:
    """Return sorted list of all drug names in the local dataset."""
    return sorted(all_df["drug_name"].str.lower().str.strip().unique().tolist())