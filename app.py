"""
app.py — Drug Side Effect Predictor
Streamlit interface wrapping predict_utils.predict()

Run with:
    streamlit run app.py

DISCLAIMER: This tool is for educational and portfolio purposes only.
Predictions are statistical associations, not clinical diagnoses.
Do not use this tool to inform real medical decisions.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from predict_utils import predict, list_all_drugs

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Drug Side Effect Predictor",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Import fonts */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

/* Global */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Hide Streamlit branding */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* Main container */
.main .block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1100px;
}

/* Hero header */
.hero-block {
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 60%, #0f2a1e 100%);
    border-radius: 16px;
    padding: 2.5rem 2.8rem 2rem;
    margin-bottom: 1.8rem;
    border: 1px solid #1e3a2e;
}
.hero-title {
    font-size: 2.1rem;
    font-weight: 700;
    color: #f1f5f9;
    letter-spacing: -0.03em;
    margin: 0 0 0.3rem 0;
    line-height: 1.15;
}
.hero-title span {
    color: #34d399;
}
.hero-sub {
    font-size: 0.95rem;
    color: #94a3b8;
    font-weight: 400;
    margin: 0;
    line-height: 1.6;
}
.hero-badge {
    display: inline-block;
    background: #0f2a1e;
    border: 1px solid #34d399;
    color: #34d399;
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    padding: 0.18rem 0.65rem;
    border-radius: 99px;
    margin-bottom: 0.9rem;
}

/* Disclaimer */
.disclaimer-box {
    background: #1c1008;
    border-left: 3px solid #f59e0b;
    border-radius: 0 8px 8px 0;
    padding: 0.75rem 1rem;
    margin-bottom: 1.6rem;
    font-size: 0.82rem;
    color: #fbbf24;
    line-height: 1.55;
}
.disclaimer-box strong {
    color: #f59e0b;
}

/* Search section */
.search-label {
    font-size: 0.82rem;
    font-weight: 600;
    color: #94a3b8;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    margin-bottom: 0.4rem;
}

/* Result header */
.result-header {
    background: #0a1a10;
    border: 1px solid #166534;
    border-radius: 12px;
    padding: 1.1rem 1.4rem;
    margin-bottom: 1.4rem;
}
.result-drug-name {
    font-size: 1.5rem;
    font-weight: 700;
    color: #f1f5f9;
    letter-spacing: -0.02em;
    margin: 0 0 0.25rem;
}
.result-meta {
    font-size: 0.82rem;
    color: #6b7280;
    font-family: 'JetBrains Mono', monospace;
}
.result-meta span {
    color: #34d399;
    font-weight: 500;
}

/* Train badge */
.badge-train {
    display: inline-block;
    background: #1e3a5f;
    border: 1px solid #3b82f6;
    color: #93c5fd;
    font-size: 0.68rem;
    font-weight: 600;
    letter-spacing: 0.07em;
    text-transform: uppercase;
    padding: 0.15rem 0.55rem;
    border-radius: 99px;
    margin-left: 0.6rem;
    vertical-align: middle;
}
.badge-test {
    display: inline-block;
    background: #1a2e1a;
    border: 1px solid #34d399;
    color: #34d399;
    font-size: 0.68rem;
    font-weight: 600;
    letter-spacing: 0.07em;
    text-transform: uppercase;
    padding: 0.15rem 0.55rem;
    border-radius: 99px;
    margin-left: 0.6rem;
    vertical-align: middle;
}

/* Metric cards */
.metric-row {
    display: flex;
    gap: 0.9rem;
    margin-bottom: 1.5rem;
}
.metric-card {
    flex: 1;
    background: #111827;
    border: 1px solid #1f2937;
    border-radius: 10px;
    padding: 0.9rem 1rem;
    text-align: center;
}
.metric-value {
    font-size: 1.6rem;
    font-weight: 700;
    color: #34d399;
    font-family: 'JetBrains Mono', monospace;
    line-height: 1.1;
}
.metric-label {
    font-size: 0.72rem;
    color: #6b7280;
    font-weight: 500;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    margin-top: 0.25rem;
}

/* Section headers */
.section-eyebrow {
    font-size: 0.72rem;
    font-weight: 600;
    color: #34d399;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-bottom: 0.5rem;
}
.section-title {
    font-size: 1.05rem;
    font-weight: 600;
    color: #e2e8f0;
    margin: 0 0 1rem;
}

/* Score bar row */
.score-row {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    margin-bottom: 0.45rem;
}
.score-name {
    font-size: 0.82rem;
    color: #cbd5e1;
    min-width: 200px;
    max-width: 200px;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}
.score-bar-bg {
    flex: 1;
    background: #1f2937;
    border-radius: 99px;
    height: 6px;
    overflow: hidden;
}
.score-bar-fill {
    height: 100%;
    border-radius: 99px;
    background: linear-gradient(90deg, #059669, #34d399);
}
.score-num {
    font-size: 0.75rem;
    color: #6b7280;
    font-family: 'JetBrains Mono', monospace;
    min-width: 42px;
    text-align: right;
}

/* Not found */
.not-found-box {
    background: #1a0f0f;
    border: 1px solid #7f1d1d;
    border-radius: 12px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 1.2rem;
}
.not-found-title {
    font-size: 1rem;
    font-weight: 600;
    color: #fca5a5;
    margin-bottom: 0.4rem;
}
.not-found-body {
    font-size: 0.84rem;
    color: #9ca3af;
    line-height: 1.6;
}

/* Sidebar styles */
.sidebar-section-title {
    font-size: 0.72rem;
    font-weight: 600;
    color: #34d399;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-bottom: 0.6rem;
}
.stat-item {
    display: flex;
    justify-content: space-between;
    padding: 0.35rem 0;
    border-bottom: 1px solid #1f2937;
    font-size: 0.8rem;
}
.stat-key { color: #6b7280; }
.stat-val { color: #e2e8f0; font-weight: 500; font-family: 'JetBrains Mono', monospace; }
.stat-val-green { color: #34d399; font-weight: 600; font-family: 'JetBrains Mono', monospace; }

.how-it-works-step {
    display: flex;
    gap: 0.65rem;
    margin-bottom: 0.7rem;
    align-items: flex-start;
}
.step-num {
    background: #0f2a1e;
    border: 1px solid #34d399;
    color: #34d399;
    font-size: 0.65rem;
    font-weight: 700;
    width: 18px;
    height: 18px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
    margin-top: 1px;
}
.step-text {
    font-size: 0.78rem;
    color: #94a3b8;
    line-height: 1.5;
}
.step-text strong { color: #cbd5e1; }
</style>
""", unsafe_allow_html=True)


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sidebar-section-title">Model Specs</div>', unsafe_allow_html=True)
    stats = [
        ("Algorithm",       "Logistic Regression",  False),
        ("Strategy",        "One-vs-Rest (845)",     False),
        ("Regularisation C","5.0",                   False),
        ("Threshold",       "0.50",                  False),
        ("Test micro-F1",   "0.4455",                True),
        ("Baseline micro-F1","0.3987",               False),
        ("Improvement",     "+0.0468",               True),
        ("Training drugs",  "893",                   False),
        ("Test drugs",      "224",                   False),
        ("Side effect labels","845",                 False),
        ("Features",        "282 sparse",            False),
    ]
    for key, val, green in stats:
        cls = "stat-val-green" if green else "stat-val"
        st.markdown(
            f'<div class="stat-item"><span class="stat-key">{key}</span>'
            f'<span class="{cls}">{val}</span></div>',
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="sidebar-section-title">How It Works</div>', unsafe_allow_html=True)

    steps = [
        ("Name lookup", "Drug name matched case-insensitively against 1,117 SIDER drugs."),
        ("ATC encoding", "ATC codes truncated to 3-char subgroups → 82 binary flags via MultiLabelBinarizer."),
        ("TF-IDF encoding", "Indication text → 200 TF-IDF term weights (unigrams + bigrams)."),
        ("Inference", "LR predict_proba() over 845 binary classifiers; labels scoring ≥ 0.50 are returned."),
        ("Ranking", "Predictions sorted by score descending. Top-20 shown by default."),
    ]
    for i, (title, body) in enumerate(steps, 1):
        st.markdown(
            f'<div class="how-it-works-step">'
            f'<div class="step-num">{i}</div>'
            f'<div class="step-text"><strong>{title}</strong> — {body}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="sidebar-section-title">Feature Breakdown</div>', unsafe_allow_html=True)
    st.markdown(
        '<div style="font-size:0.78rem;color:#6b7280;line-height:1.7;">'
        '82 ATC subgroup flags<br>'
        '+ 200 TF-IDF indication terms<br>'
        '= <strong style="color:#94a3b8;">282 total features</strong><br>'
        '~4% density (sparse matrix)'
        '</div>',
        unsafe_allow_html=True,
    )


# ── Hero ───────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-block">
    <div class="hero-badge">SIDER Dataset · 1,117 Drugs · 845 Side Effects</div>
    <div class="hero-title">Drug Side Effect <span>Predictor</span></div>
    <div class="hero-sub">
        Enter any drug name from the SIDER dataset to see its statistically predicted side effects,
        ranked by model confidence. Built with Logistic Regression trained on ATC therapeutic class
        codes and indication text — micro-F1 of 0.4455 on 224 held-out test drugs.
    </div>
</div>
""", unsafe_allow_html=True)

# Disclaimer
st.markdown("""
<div class="disclaimer-box">
    <strong>⚠ Medical disclaimer</strong> — Predictions are statistical associations learned from
    historical drug records in the SIDER database. They are <strong>not clinical diagnoses</strong>
    and must not be used to inform real medical decisions. For health concerns, consult a qualified
    healthcare professional.
</div>
""", unsafe_allow_html=True)


# ── Search ─────────────────────────────────────────────────────────────────────
st.markdown('<div class="search-label">Drug name</div>', unsafe_allow_html=True)
col_input, col_btn = st.columns([5, 1])
with col_input:
    drug_input = st.text_input(
        label="Drug name",
        placeholder="e.g. ibuprofen, lorazepam, metformin, atorvastatin …",
        label_visibility="collapsed",
        key="drug_input",
    )
with col_btn:
    search = st.button("Search", use_container_width=True, type="primary")


# ── Empty state ────────────────────────────────────────────────────────────────
if not drug_input.strip():
    st.markdown("<br>", unsafe_allow_html=True)
    col_a, col_b, col_c = st.columns(3)
    examples = [
        ("💊", "Ibuprofen", "NSAID · pain & inflammation"),
        ("💊", "Lorazepam", "Benzodiazepine · anxiety"),
        ("💊", "Metformin", "Biguanide · type 2 diabetes"),
    ]
    for col, (icon, name, desc) in zip([col_a, col_b, col_c], examples):
        with col:
            st.markdown(
                f'<div style="background:#111827;border:1px solid #1f2937;border-radius:10px;'
                f'padding:0.9rem 1rem;text-align:center;">'
                f'<div style="font-size:1.4rem;margin-bottom:0.3rem;">{icon}</div>'
                f'<div style="font-size:0.88rem;font-weight:600;color:#e2e8f0;">{name}</div>'
                f'<div style="font-size:0.75rem;color:#6b7280;margin-top:0.15rem;">{desc}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    st.markdown("<br>", unsafe_allow_html=True)
    with st.expander("📋 Browse all 1,117 drugs in the dataset"):
        drugs = list_all_drugs()
        search_filter = st.text_input("Filter list", placeholder="Type to filter…", key="filter")
        filtered = [d for d in drugs if search_filter.lower() in d] if search_filter else drugs
        st.markdown(f'<div style="font-size:0.78rem;color:#6b7280;margin-bottom:0.5rem;">'
                    f'Showing {len(filtered)} of {len(drugs)} drugs</div>', unsafe_allow_html=True)
        st.dataframe(
            pd.DataFrame(filtered, columns=["Drug name"]),
            use_container_width=True,
            hide_index=True,
            height=300,
        )
    st.stop()


# ── Run prediction ─────────────────────────────────────────────────────────────
with st.spinner("Running prediction …"):
    result = predict(drug_input.strip())

st.markdown("<br>", unsafe_allow_html=True)

# ── Not found ──────────────────────────────────────────────────────────────────
if not result["found"]:
    st.markdown(f"""
    <div class="not-found-box">
        <div class="not-found-title">"{drug_input}" not found in dataset</div>
        <div class="not-found-body">
            The SIDER dataset contains 1,117 drugs. This model can only predict side effects
            for drugs it has on record — it cannot generalise to arbitrary drug names.<br><br>
            Check spelling, or browse the full list below.
        </div>
    </div>
    """, unsafe_allow_html=True)
    with st.expander("📋 Browse all 1,117 drugs"):
        drugs = list_all_drugs()
        sf = st.text_input("Filter", placeholder="Type to filter…", key="filter2")
        filtered = [d for d in drugs if sf.lower() in d] if sf else drugs
        st.dataframe(pd.DataFrame(filtered, columns=["Drug name"]),
                     use_container_width=True, hide_index=True, height=300)
    st.stop()


# ── Result header ──────────────────────────────────────────────────────────────
predictions = result["predictions"]
n           = result["n_predicted"]
split       = result["split"]
drug_name   = result["drug_name"].title()
stitch_id   = result["stitch_id"]

split_badge = ""
split_note  = ""
if split == "train":
    split_badge = '<span class="badge-train">Training drug</span>'
    split_note  = "This drug was in the training set — the model has seen its labels before."
elif split == "test":
    split_badge = '<span class="badge-test">Test drug</span>'
    split_note  = "This drug was held out during training — a fair test of generalisation."

st.markdown(f"""
<div class="result-header">
    <div class="result-drug-name">{drug_name}{split_badge}</div>
    <div class="result-meta">
        STITCH: <span>{stitch_id}</span> &nbsp;·&nbsp;
        Predicted: <span>{n} side effects</span> &nbsp;·&nbsp;
        Threshold: <span>≥ 0.50</span>
    </div>
    {f'<div style="font-size:0.78rem;color:#6b7280;margin-top:0.5rem;">{split_note}</div>' if split_note else ''}
</div>
""", unsafe_allow_html=True)


# ── Metric cards ───────────────────────────────────────────────────────────────
top5_avg   = predictions["score"].head(5).mean() if n > 0 else 0
pct_above  = round(n / 845 * 100, 1)
confidence = "High" if top5_avg >= 0.90 else "Medium" if top5_avg >= 0.75 else "Lower"

st.markdown(f"""
<div class="metric-row">
    <div class="metric-card">
        <div class="metric-value">{n}</div>
        <div class="metric-label">Side effects predicted</div>
    </div>
    <div class="metric-card">
        <div class="metric-value">{pct_above}%</div>
        <div class="metric-label">of 845 labels triggered</div>
    </div>
    <div class="metric-card">
        <div class="metric-value">{top5_avg:.3f}</div>
        <div class="metric-label">Top-5 avg. score</div>
    </div>
    <div class="metric-card">
        <div class="metric-value" style="font-size:1.2rem;">{confidence}</div>
        <div class="metric-label">Confidence tier</div>
    </div>
</div>
""", unsafe_allow_html=True)


# ── Top 20 visual bars ─────────────────────────────────────────────────────────
TOP_N = 20
top   = predictions.head(TOP_N)

st.markdown('<div class="section-eyebrow">Ranked predictions</div>', unsafe_allow_html=True)
st.markdown(f'<div class="section-title">Top {min(TOP_N, n)} predicted side effects</div>',
            unsafe_allow_html=True)

for _, row in top.iterrows():
    score = row["score"]
    pct   = score * 100
    st.markdown(f"""
    <div class="score-row">
        <div class="score-name" title="{row['side_effect']}">{row['side_effect']}</div>
        <div class="score-bar-bg">
            <div class="score-bar-fill" style="width:{pct:.1f}%"></div>
        </div>
        <div class="score-num">{score:.3f}</div>
    </div>
    """, unsafe_allow_html=True)


# ── Chart: top 20 horizontal bar ───────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
with st.expander("📊 View as chart"):
    chart_df = top.copy().iloc[::-1]  # reverse for horizontal bar (highest at top)
    fig = go.Figure(go.Bar(
        x=chart_df["score"],
        y=chart_df["side_effect"],
        orientation="h",
        marker=dict(
            color=chart_df["score"],
            colorscale=[[0, "#059669"], [1, "#34d399"]],
            showscale=False,
            line=dict(width=0),
        ),
        hovertemplate="%{y}<br>Score: %{x:.4f}<extra></extra>",
    ))
    fig.update_layout(
        paper_bgcolor="#111827",
        plot_bgcolor="#111827",
        font=dict(family="Inter", color="#94a3b8", size=11),
        margin=dict(l=10, r=20, t=20, b=10),
        height=520,
        xaxis=dict(
            gridcolor="#1f2937", tickfont=dict(size=10),
            title=dict(text="Model score", font=dict(size=11)),
            range=[predictions["score"].min() * 0.95, 1.0],
        ),
        yaxis=dict(tickfont=dict(size=10), title=None),
    )
    st.plotly_chart(fig, use_container_width=True)


# ── Full list expander ─────────────────────────────────────────────────────────
if n > TOP_N:
    with st.expander(f"📋 Show all {n} predicted side effects"):
        st.markdown(
            f'<div style="font-size:0.78rem;color:#6b7280;margin-bottom:0.6rem;">'
            f'Showing predictions ≥ 0.50 confidence. '
            f'Score represents the model\'s estimated probability of this side effect being '
            f'associated with {drug_name}.</div>',
            unsafe_allow_html=True,
        )
        display_df = predictions.copy()
        display_df["score"] = display_df["score"].round(4)
        display_df.index   = display_df.index + 1
        st.dataframe(
            display_df.rename(columns={"side_effect": "Side effect", "score": "Score"}),
            use_container_width=True,
            height=400,
        )


# ── Reliability note ───────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("""
<div style="background:#0c111d;border:1px solid #1e293b;border-radius:10px;
            padding:1rem 1.2rem;font-size:0.8rem;color:#6b7280;line-height:1.7;">
    <strong style="color:#94a3b8;">Reading these results</strong><br>
    The model predicts ~187 side effects per drug on average at this threshold.
    <strong style="color:#94a3b8;">High scores (≥ 0.90)</strong> reflect patterns the model found consistently
    across similar drugs in training.
    <strong style="color:#94a3b8;">Scores near 0.50</strong> are borderline associations — treat them with more caution.
    Drugs from the Nervous system (N) and Cardiovascular (C) therapeutic classes were best-predicted during testing.
    Overall test micro-F1 = 0.4455 versus a frequency baseline of 0.3987.
</div>
""", unsafe_allow_html=True)
