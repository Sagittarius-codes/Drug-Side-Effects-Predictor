# Drug Side Effect Predictor

[![Live App](https://img.shields.io/badge/Streamlit-Live%20App-34d399?logo=streamlit&logoColor=white)](https://drug-side-effects-predictor.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-orange?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)

> **Medical Disclaimer:** This project is for educational and portfolio purposes only.
> Predictions are statistical associations derived from historical drug records — they are not clinical diagnoses.
> **Do not use this tool to make real medical decisions.**

---

## Abstract

This project builds a multi-label classification system that predicts which side effects a drug is historically associated with, given only its therapeutic class codes and indication description. Using the publicly available SIDER 4.1 database — which catalogues 1,117 approved drugs and 845 documented side effects — a Logistic Regression model with One-vs-Rest strategy was trained on 893 drugs and evaluated on 224 held out. After a structured seven-week development pipeline involving data cleaning, feature engineering, baseline comparison, diagnosis, and hyperparameter tuning via cross-validation, the final model achieved a micro-F1 of **0.4455** — a **+0.0468 improvement** over a frequency-weighted baseline, with no test-set leakage. The project is delivered as a Streamlit web application with a `predict(drug_name)` inference function and documented error analysis by therapeutic class. It was built end-to-end using AI-assisted development (Claude, Anthropic) as the primary co-developer, making it an example of how AI tooling can accelerate structured data science work for developers at any experience level.

---

## Table of Contents

1. [What the Model Does](#1-what-the-model-does)
2. [Dataset](#2-dataset)
3. [Feature Engineering](#3-feature-engineering)
4. [Model Architecture and Training](#4-model-architecture-and-training)
5. [How ATC Codes Drive Predictions](#5-how-atc-codes-drive-predictions)
6. [Evaluation and Results](#6-evaluation-and-results)
7. [Reliability and Limitations](#7-reliability-and-limitations)
8. [Error Analysis by Therapeutic Class](#8-error-analysis-by-therapeutic-class)
9. [Project Structure](#9-project-structure)
10. [Setup and Usage](#10-setup-and-usage)
11. [AI-Assisted Development](#11-ai-assisted-development)
12. [What I Learned](#12-what-i-learned)
13. [Future Work](#13-future-work)

---

## 1. What the Model Does

Given a drug name, the model returns a ranked list of side effects it predicts that drug is historically associated with, along with a confidence score for each.

**Input:** A drug name (e.g. `ibuprofen`, `lorazepam`, `metformin`) — must be present in the SIDER dataset.

**Output:** A ranked list of side effects, each with a model score between 0 and 1. Side effects scoring ≥ 0.50 are returned as active predictions.

**What it is not:** This is not a clinical tool. The model learns statistical patterns — which side effects tend to co-occur with drugs that share a therapeutic class or indication — not the underlying pharmacology or biology. A high score means the model has seen similar drugs associated with that side effect frequently in training data. It does not mean a given patient will experience it.

---

## 2. Dataset

**SIDER 4.1** (Side Effect Resource) is a public database maintained by the European Molecular Biology Laboratory (EMBL). It links approved drugs to their documented adverse effects, sourced from drug package inserts and FDA adverse event reports.

| Property | Value |
|---|---|
| Total drugs | 1,117 |
| Side effect labels | 845 |
| Drug identifier | STITCH compound ID |
| Additional attributes | ATC codes, indication text |
| Source | [sideeffects.embl.de](http://sideeffects.embl.de/) |

Each drug has a binary label vector of length 845 — a `1` in position `i` means that drug has a documented association with side effect `i` in the SIDER records. This is a **multi-label** classification problem: a single drug can and typically does have many positive labels simultaneously (the average is around 120 true side effects per drug in this dataset).

The data was split 80/20 into 893 training drugs and 224 test drugs. The split was done at the drug level (no overlap), and the test set was held out completely until the final evaluation in Week 6.

---

## 3. Feature Engineering

The model has no access to molecular structure (SMILES strings, fingerprints, or 3D geometry). Instead, two sources of pharmacological context are used:

### 3.1 ATC Subgroup Flags (82 features)

The World Health Organisation's **Anatomical Therapeutic Chemical (ATC) classification system** assigns each drug a hierarchical code describing what it treats and how it works. For example:

```
Metformin:   A10BA02
             │││└── Chemical substance (02 = metformin)
             ││└─── Pharmacological subgroup (B = biguanides)
             │└──── Therapeutic subgroup (10 = drugs used in diabetes)
             └───── Anatomical main group (A = alimentary tract & metabolism)
```

The model uses the **3-character level** (e.g. `A10`, `C09`, `N05`) — the therapeutic subgroup. There are 82 unique subgroups across all 1,117 drugs. Each drug is encoded as an 82-dimensional binary vector: `1` if the drug belongs to that subgroup, `0` otherwise. A drug can belong to multiple subgroups (e.g. a combination drug).

This is the model's primary signal. Drugs within the same therapeutic subgroup tend to share side effect profiles: statins (C10) consistently cause muscle-related side effects; NSAIDs (M01) consistently cause GI irritation. The model learns these associations.

### 3.2 TF-IDF Indication Terms (200 features)

Each drug has an `indications_text` field describing what conditions it treats. This text is converted to a 200-dimensional TF-IDF (Term Frequency–Inverse Document Frequency) vector. TF-IDF weights words by how informative they are across all drugs — common words like "treatment" get low weight; specific terms like "rheumatoid" or "hypertension" get higher weight.

The TF-IDF vectoriser was fitted only on training drugs, then applied to test drugs, to prevent any information from the test set influencing the feature representation.

### 3.3 Combined Feature Matrix

The ATC flags and TF-IDF weights are concatenated horizontally into a single feature vector of length **282** per drug. The matrix is sparse (~4% of entries are non-zero), which is expected — most drugs only belong to a few subgroups and use a small fraction of the vocabulary.

---

## 4. Model Architecture and Training

### 4.1 Algorithm: Logistic Regression with One-vs-Rest

With 845 side effect labels, the task cannot be treated as single-label classification. The model uses the **One-vs-Rest (OvR)** strategy: one independent binary Logistic Regression classifier is trained per label. Each classifier answers the question: "given this drug's features, is it associated with side effect *i*?"

At inference time, all 845 classifiers run in parallel. Each returns a probability. The final prediction applies a threshold: labels with probability ≥ 0.50 are marked as predicted positive.

### 4.2 Why Logistic Regression?

Logistic Regression was chosen deliberately over more complex models (Random Forests, Neural Networks) for three reasons:

- **Interpretability.** Each label has a set of weights — one per feature. It is possible to inspect which ATC subgroup or indication term contributed most to a given prediction.
- **Speed.** With 845 classifiers and only 282 features, LR trains in under 2 minutes on a standard CPU. This made the cross-validation grid search tractable.
- **Data size.** With 893 training samples, a model with too many parameters would overfit. LR's regularisation (controlled by `C`) handles this cleanly.

### 4.3 Hyperparameters

Two hyperparameters were jointly tuned using 5-fold cross-validation on the training set:

**`C` — regularisation strength**
Controls how much the model is penalised for fitting the training data too closely. Small C = strong regularisation (simpler, more conservative model). Large C = weak regularisation (more aggressive fitting). Grid searched over: `[0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0]`.

**`threshold` — probability cutoff**
The decision boundary for predicting a label positive. Lower threshold = more predictions, higher recall but lower precision. Higher threshold = fewer predictions, higher precision but lower recall. Grid searched over: `[0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65]`.

**Total grid:** 7 × 8 = 56 combinations, each evaluated with 5-fold CV → 280 model fits.

**Best combination found by CV:** `C = 5.0`, `threshold = 0.50`

### 4.4 Cross-Validation Discipline

A critical discipline maintained throughout: the test set (224 drugs) was never used to select hyperparameters. All tuning decisions were made on CV scores from the training set alone. The test set was used exactly once — to report the final honest performance estimate in Week 6.

An earlier version (Week 5) accidentally introduced test-set leakage by choosing the threshold after observing test performance. Week 6 corrected this, and interestingly, the leakage-corrected result was actually *better* (0.4455 vs 0.4165) — evidence that the earlier threshold choice was suboptimal even when cheating.

---

## 5. How ATC Codes Drive Predictions

ATC codes are the model's strongest signal. Here is a concrete walkthrough of how a prediction is made for **ibuprofen**:

**Step 1 — Retrieve ATC codes**
Ibuprofen has ATC code `M01AE01`. At the 3-character level: `M01` (anti-inflammatory and antirheumatic products).

**Step 2 — Build the ATC feature vector**
The 82 binary flags are set: position corresponding to `M01` is `1`; all others are `0` (unless ibuprofen has additional ATC assignments).

**Step 3 — Build the TF-IDF vector**
Ibuprofen's indication text ("pain, fever, inflammation") is tokenised and weighted.

**Step 4 — Run 845 binary classifiers**
Each classifier was trained to recognise the pattern: "drugs in group `M01` with indication terms like 'inflammation' tend to have side effect X." For ibuprofen, the model assigns high probability to GI-related side effects (nausea, dyspepsia, vomiting) because NSAIDs in `M01` consistently show these in training data.

**Step 5 — Apply threshold**
Any label scoring ≥ 0.50 is returned as a prediction, ranked by score.

This is why the model cannot predict for unknown drugs: it has no mechanism to generalise beyond the ATC subgroups and indication vocabulary it was trained on.

---

## 6. Evaluation and Results

### 6.1 Primary Metric: Micro-F1

Micro-F1 is the harmonic mean of micro-precision and micro-recall, computed by aggregating true positives, false positives, and false negatives across all 845 labels and all 224 test drugs together. It gives more weight to common (frequently positive) labels, which is appropriate here — rare side effects are inherently harder to predict and contribute less signal.

### 6.2 Results Table

| Model | Micro-F1 | Macro-F1 | Precision@K | Recall@K | Avg predicted/drug |
|---|---|---|---|---|---|
| Frequency baseline (K=68) | 0.3987 | 0.0545 | 0.5221 | 0.4020 | 68.0 |
| Week 5 LR (C=1.0, threshold=0.40) — *leaky* | 0.4165 | 0.3162 | — | — | ~278 |
| **Week 6 LR (C=5.0, threshold=0.50) — CV-tuned** | **0.4455** | **0.3285** | 0.3197 | 0.5802 | 186.9 |

### 6.3 Baseline

The frequency baseline always predicts the 68 most common side effects in the training set (K=68 is the median label count per drug). It requires no model — just counting. It achieves micro-F1 = 0.3987 and is surprisingly competitive because the label distribution is highly skewed: a handful of very common side effects (nausea, vomiting, headache) dominate the positive label space.

The tuned model beats this baseline by **+0.0468 micro-F1** on 224 unseen test drugs.

### 6.4 Reading the Metrics

- **Micro-F1 = 0.4455** — the model correctly identifies about 44.6% of the true positive (drug, side effect) pairs on the test set, balancing precision and recall.
- **Recall@K = 0.5802** — of all true side effects a drug actually has, the model recovers about 58%.
- **Precision@K = 0.3197** — of all side effects the model predicts, about 32% are correct.
- **Avg predicted/drug = 186.9** — the model is recall-oriented at threshold 0.50; it predicts more labels than a drug actually has, erring on the side of over-predicting rather than missing associations.

This recall–precision trade-off is intentional for a safety-relevant context: missing a true side effect (false negative) is worse than flagging a spurious one (false positive).

---

## 7. Reliability and Limitations

### What the model is reliable for

- Drugs from **well-represented therapeutic classes** in training (Nervous system N=155, Cardiovascular C=119, Alimentary A=108) — these have the most training signal.
- **High-confidence predictions (score ≥ 0.90)** — these reflect patterns the model encountered consistently across many similar drugs.
- **Broad association screening** — identifying which general categories of side effects a drug class tends to produce.

### Known limitations

| Limitation | Detail |
|---|---|
| **Closed-world assumption** | Only works for the 1,117 drugs in SIDER. Novel drugs, generics not in the database, or misspelled names return "not found." |
| **No molecular information** | Chemical structure, binding affinity, pharmacokinetics — none of this is used. Two drugs with the same ATC code but different mechanisms will look identical to the model. |
| **Statistical, not causal** | The model learns correlations in recorded drug data, not biological causality. A predicted association means "drugs like this have been reported with this side effect" — not "this drug causes this effect." |
| **Label inflation** | At threshold 0.50, the model predicts ~187 side effects per drug on average. Real drugs have ~120 true associations in SIDER. The model over-predicts. |
| **No patient-level personalisation** | SIDER is drug-level data. Age, weight, comorbidities, and drug interactions are not modelled. |
| **Training data quality** | SIDER itself is derived from package inserts and regulatory filings — these are comprehensive but not identical to clinical trial outcomes. |
| **Sparse therapeutic classes** | Drugs from underrepresented classes (P — Antiparasitic, with only 15 training drugs) are predicted less reliably. |

---

## 8. Error Analysis by Therapeutic Class

Performance varies meaningfully by ATC main group (1-character level). This was measured on the 224 test drugs:

| ATC Group | Micro-F1 | Test drugs | Training drugs |
|---|---|---|---|
| N — Nervous system | 0.508 | 41 | 155 |
| G — Genito-urinary & sex hormones | 0.479 | 11 | 46 |
| C — Cardiovascular system | 0.477 | 28 | 119 |
| L — Antineoplastic & immunomodulating | 0.453 | 32 | 82 |
| J — Anti-infectives (systemic) | 0.440 | 22 | 98 |
| V — Various | 0.420 | 11 | 36 |
| M — Musculo-skeletal | ~0.41 | — | 43 |
| A — Alimentary tract | ~0.40 | — | 108 |

The Nervous system class (N) achieves the highest test performance and has the most training drugs (155), which is consistent with the hypothesis that representation drives generalisation. However, the relationship is not strictly monotonic — some well-represented classes (A — Alimentary, 108 training drugs) do not outperform sparser ones, suggesting label structure within the class also matters.

**Failure pattern for worst-predicted drugs:** The 5 lowest-scoring test drugs (didanosine, azathioprine, rufinamide, rasagiline, tolcapone) all had the same pattern — true label count of 69–240 but the model predicted 420–507. This is a threshold-level over-prediction issue specific to those drugs, not a feature representation failure (all had high cosine similarity to training drugs and strong ATC overlap).

---

## 9. Project Structure

```
Drug-Side-Effect-Predictor/
│
├── data/
│   ├── raw/                        ← Original SIDER files (not committed)
│   ├── splits/
│   │   ├── train.csv               ← 893 training drugs (name, ATC, indications, labels)
│   │   └── test.csv                ← 224 test drugs
│   └── processed/
│       ├── X_train.npz             ← Sparse feature matrix (893 × 282)
│       ├── X_test.npz              ← Sparse feature matrix (224 × 282)
│       ├── y_train.npy             ← Label matrix (893 × 845)
│       ├── y_test.npy              ← Label matrix (224 × 845)
│       ├── label_names.csv         ← 845 side effect names
│       ├── train_index.csv         ← Drug names + STITCH IDs for train set
│       └── test_index.csv          ← Drug names + STITCH IDs for test set
│
├── models/
│   ├── encoders/
│   │   ├── mlb_atc.pkl             ← Fitted MultiLabelBinarizer (ATC subgroups)
│   │   └── tfidf_indications.pkl   ← Fitted TF-IDF vectoriser (indication text)
│   ├── lr_tuned.pkl                ← Final trained model (C=5.0, OvR LR)
│   ├── week6_best_params.csv       ← Best CV hyperparameters
│   ├── week6_grid_search.csv       ← Full 56-combination CV results
│   └── week7_failure_diagnosis.csv ← Worst-5 drug failure analysis
│
├── figures/                        ← All plots generated across weeks
│
├── notebooks/
│   ├── week1_feasibility.ipynb
│   ├── week2_data_preparation.ipynb
│   ├── week3_feature_engineering.ipynb
│   ├── week4_baseline_and_first_model.ipynb
│   ├── week5_diagnosis_and_fixes.ipynb
│   ├── week6_hyperparameter_tuning.ipynb
│   └── week7_error_analysis_and_app.ipynb
│
├── predict_utils.py                ← Self-contained inference module
├── app.py                          ← Streamlit web application
├── requirements.txt
└── README.md
```

---

## 10. Setup and Usage

### Live demo

**[https://drug-side-effects-predictor.streamlit.app/](https://drug-side-effects-predictor.streamlit.app/)**

No setup required — open the link and start searching drug names directly.

### Install dependencies (local)

```bash
pip install scikit-learn pandas numpy scipy joblib streamlit plotly
```

### Run the Streamlit app

```bash
cd Drug-Side-Effect-Predictor
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser. Enter any drug name from the SIDER dataset to see its predicted side effects ranked by model confidence.

### Use the inference function directly

```python
from predict_utils import predict

result = predict("ibuprofen")

if result["found"]:
    print(f"{result['n_predicted']} side effects predicted")
    print(result["predictions"].head(10))
else:
    print("Drug not found in dataset")
```

The `predict()` function returns a dict with keys: `found`, `drug_name`, `stitch_id`, `predictions` (DataFrame), `n_predicted`, `split`.

### Reproduce the full pipeline

Run the notebooks in order from `week1` to `week7`. Each notebook is self-contained but expects the outputs of prior weeks to exist in the `data/` and `models/` directories.

---

## 11. AI-Assisted Development

This project was built using **Claude (Anthropic)** as the primary development partner throughout the entire seven-week pipeline.

### Context: Where I was starting from

At the start of this project, I was in the early stages of learning data science — actively working through data cleaning fundamentals and with limited hands-on experience beyond Week 2-level pandas operations. Concepts like sparse matrices, cross-validation harnesses, TF-IDF vectorisation, multi-label classification, and threshold tuning were genuinely new territory.

This means the project is an honest example of what AI-assisted development looks like from a learner's perspective — not a senior engineer using AI to move faster, but a beginner using AI to build things that would otherwise be out of reach.

### How AI was used

**Architecture and decision-making.** Each week began with a structured conversation: the goal for the week was described, Claude proposed the approach (which algorithm, which metric, which CV strategy, why), and I reviewed and approved or pushed back on those decisions. I did not write the architecture — I learned it through the conversation.

**Code generation.** All notebook code was written by Claude following locked-in decisions. My role was to run the code, interpret the outputs, and flag when something looked wrong or unexpected. For instance: when Week 5 showed the threshold still rising at 0.40, I reported that, and Claude identified the search range problem and proposed the Week 6 fix.

**Explaining outputs.** A recurring pattern was: code runs → something in the output is surprising → I ask Claude to explain it. The K=68 baseline problem (why did a dumb baseline beat the trained model?), what test-set leakage means and why it matters, why micro-F1 and macro-F1 diverge so sharply — all of these were explained in response to real outputs from the notebooks.

**Error diagnosis.** The Week 7 failure analysis was entirely AI-generated: the cosine similarity diagnostic, the ATC familiarity score, and the interpretation that the worst-predicted drugs suffered from over-prediction at the threshold level (not feature representation failure) were all Claude's analysis.

**What I provided.** Domain curiosity, judgment calls on scope (what to include vs exclude), decision approval, running the code, reporting back what happened, and asking the right follow-up questions.

### Why this matters for the portfolio

AI-assisted development is a real and growing skill. Knowing how to frame a problem for an AI collaborator, evaluate the outputs critically, catch leakage or errors in generated code, and build a coherent multi-week project from AI-generated components — these are practical skills distinct from knowing how to write the code yourself.

This project demonstrates that pipeline: a structured, documented, reproducible ML project built through seven weeks of human-AI collaboration, where the human brought the problem framing and the AI brought the technical implementation. The code is real, the results are reproducible, and the limitations are documented honestly.

---

## 12. What I Learned

**Multi-label classification is fundamentally different from single-label.** With 845 binary targets, the label distribution dominates everything. A baseline that always predicts the most common 68 labels is hard to beat because those labels are genuinely common. The model only wins by being smarter about *which* drugs get *which* subset — not just predicting the same labels for everyone.

**Fixed-K prediction vs. threshold prediction.** The early model used fixed-K top-K prediction (always predict exactly 68 labels per drug). Switching to threshold-based prediction — letting each drug have a variable number of predictions based on model confidence — was the single biggest performance unlock.

**Hyperparameter interactions matter.** C and threshold are not independent. A high C (weak regularisation) produces sharper, more confident probability scores, which means the same threshold produces fewer predictions. A low C produces softer probabilities, so the threshold matters more. Tuning them jointly is necessary; tuning one at a time gives suboptimal results.

**Test-set leakage is subtle.** In Week 5, the threshold was chosen by observing test-set performance — a common mistake. The fix in Week 6 (choosing threshold only by CV) actually improved results, demonstrating that the leaked Week 5 choice was suboptimal. Proper CV is not just good hygiene — it finds better hyperparameters.

**Sparse features are not a problem if the signal is there.** 282 features at 4% density might look impoverished, but cosine similarity between most drug pairs was near zero — meaning the features genuinely differentiated drugs. Feature sparsity was not the limiting factor.

---

## 13. Future Work

The current system is a v1 with deliberate scope constraints. Several directions could meaningfully improve it:

**Richer features**

- **Molecular fingerprints** (Morgan/ECFP): encode the 2D chemical structure of each drug as a binary vector. This would allow the model to generalise across structurally similar drugs regardless of ATC classification.
- **SMILES-based embeddings**: use a pretrained molecular encoder (e.g. ChemBERTa) to get a dense representation of each drug's chemistry.
- **Drug-drug interaction counts**: drugs with many known interactions may have systematically broader side effect profiles.
- **Finer ATC granularity**: using 4- or 5-character ATC codes (pharmacological or chemical substance level) instead of 3-character therapeutic subgroups.

**Better models**

- **Gradient Boosting (XGBoost, LightGBM)**: may capture non-linear interactions between ATC subgroups and indication terms better than LR.
- **Label-aware methods**: Label Powerset or Classifier Chain strategies that model correlations between side effects (e.g. nausea and vomiting tend to co-occur).
- **Neural multi-label models**: a small MLP or attention-based model on the 282 features could be tried with the existing feature set.

**Better evaluation**

- **Label-stratified CV**: ensure rare side effects are distributed across folds more evenly.
- **Per-label calibration**: the model's probability scores are not well-calibrated for rare labels — Platt scaling or isotonic regression could improve this.
- **Precision@K analysis by ATC class**: breakdown of precision and recall per therapeutic class rather than just micro-F1.

**Deployment**

- **Streamlit Cloud deployment**: the current app runs locally; it could be hosted publicly as a portfolio demo.
- **Drug name fuzzy matching**: currently the lookup is exact (case-insensitive). A fuzzy match (e.g. using `rapidfuzz`) would handle common misspellings gracefully.
- **Autocomplete dropdown**: replacing the free text input with a searchable dropdown from `list_all_drugs()` would eliminate "drug not found" errors entirely.

---

*Built by Muhammad Abdullah — Zynvex Solutions internship project, ID: ZYNVEX-CERT-0610*
*AI-assisted development pipeline: Claude (Anthropic), Weeks 1–7*
