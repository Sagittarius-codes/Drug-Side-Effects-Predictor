# Drug Side Effect Predictor

> **Medical disclaimer:** This project is for educational and portfolio purposes only.  
> Predictions are statistical associations — not clinical diagnoses.  
> **Do not use this tool to make real medical decisions.**

---

## Overview

A multi-label classification model that predicts which side effects a drug is historically associated with, based on its therapeutic class (ATC codes) and indication description. Built on the [SIDER](http://sideeffects.embl.de/) public drug–side-effect dataset.

This is an end-to-end data science learning project — from raw data exploration through feature engineering, model tuning, error analysis, and a Streamlit demo app.

---

## Results

| Model | Micro-F1 | Note |
|---|---|---|
| Frequency baseline (K=68) | 0.3987 | Always predicts the 68 most common side effects |
| Week 5 LR (threshold=0.40) | 0.4165 | Test-set leakage |
| **Week 6 LR — CV-tuned** | **0.4455** | C=5.0, threshold=0.50, clean CV |

The tuned model beats the frequency baseline by **+0.0468 micro-F1** using honest cross-validation with no test-set leakage.

---

## Setup

```bash
pip install scikit-learn pandas numpy scipy joblib streamlit
```

---

## Run the app

```bash
streamlit run app.py
```

Enter any drug name from the SIDER dataset (1,117 drugs). Predictions are returned as a ranked list of side effects with model scores.

---

## Project structure

```
Drug-Side-Effect-Predictor/
├── data/
│   ├── splits/          — train.csv, test.csv (raw attributes + labels)
│   └── processed/       — X_train.npz, X_test.npz, y_train.npy, y_test.npy, label_names.csv
├── models/
│   ├── encoders/        — mlb_atc.pkl, tfidf_indications.pkl
│   └── lr_tuned.pkl     — final tuned model
├── figures/             — all EDA and evaluation plots
├── predict_utils.py     — inference module (used by app.py)
├── app.py               — Streamlit app
├── week1_feasibility.ipynb
├── week2_data_preparation.ipynb
├── week3_feature_engineering.ipynb
├── week4_baseline_and_models.ipynb
├── week5_diagnostics.ipynb
├── week6_hyperparameter_tuning.ipynb
└── week7_error_analysis_and_app.ipynb
```

---

## Features used

- **82 ATC subgroup flags** (3-character therapeutic subgroup level, MultiLabelBinarizer)
- **200 TF-IDF indication terms** (unigrams + bigrams, fit on training set only)
- Total: 282 sparse features, ~4% density

Chemical/structural features (SMILES, fingerprints) and patient-level personalization are explicitly out of scope for v1.

---

## Limitations

- **Dataset scope:** only drugs present in SIDER (1,117 drugs) can be predicted. Unknown drugs return a "not found" message.
- **What the model learns:** statistical associations between drug attributes and historically recorded side effects — not biological causality.
- **Label volume:** the model predicts ~187 side effects per drug on average at threshold=0.50. High recall, moderate precision.
- **Generalization varies by class:** test drugs from well-represented therapeutic categories in training are predicted more reliably than drugs from sparse categories (see Step 7.2 error analysis).
- **No patient-level personalization:** SIDER is drug-level, not patient-level.

---

## What I learned

- Multi-label classification at scale (845 labels, OneVsRest strategy) behaves very differently from single-label problems — the frequency baseline is surprisingly competitive because label imbalance gives it a structural advantage at fixed K.
- Threshold-based prediction outperformed fixed-K top-K for this dataset: jointly tuning C and threshold via CV was the key step that unlocked meaningful improvement.
- Test-set leakage is easy to introduce accidentally (picking a threshold by observing test performance). The Week 6 CV harness fixed this and actually *improved* the result — a sign the Week 5 leaky threshold was suboptimal.
- Sparse features (4% density) are not a bottleneck if the signal is there; feature differentiation across drugs was healthy.
- Error analysis by ATC class turned the model's known limitation (drug-level generalization is graded, not binary) into a documented, interpretable finding rather than a blind spot.

---

*Prepared by Muhammad Abdullah — Zynvex Solutions internship project, ID: ZYNVEX-CERT-0610*