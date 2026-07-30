"""app.py -- Drug Side Effect Predictor (Streamlit)

Run with:  streamlit run app.py

DISCLAIMER: This tool is for educational and portfolio purposes only.
Predictions are statistical associations, not clinical diagnoses.
Do not use this tool to inform real medical decisions.
"""

import streamlit as st
import pandas as pd
from predict_utils import predict, list_all_drugs

# -- Page config
st.set_page_config(
    page_title="Drug Side Effect Predictor",
    page_icon="pill",
    layout="centered",
)

# -- Sidebar
with st.sidebar:
    st.header("Model info")
    st.markdown(
        "| Parameter | Value |\n"
        "|---|---|\n"
        "| Model | Logistic Regression (OvR) |\n"
        "| Regularisation C | 5.0 |\n"
        "| Threshold | 0.50 |\n"
        "| Test micro-F1 | **0.4455** |\n"
        "| Baseline micro-F1 | 0.3987 |\n"
        "| Training drugs | 893 |\n"
        "| Side effect labels | 845 |\n"
    )
    st.markdown("---")
    st.caption(
        "Features: 82 ATC therapeutic subgroup flags + "
        "200 TF-IDF indication terms."
    )

# -- Title and disclaimer
st.title("Drug Side Effect Predictor")
st.warning(
    "**Medical disclaimer:** Predictions are statistical associations "
    "learned from the SIDER dataset and are for educational purposes only. "
    "They are not clinical diagnoses. Do not use this tool to make medical decisions."
)

# -- Drug input
st.markdown("### Enter a drug name")
drug_input = st.text_input(
    label="Drug name",
    placeholder="e.g. ibuprofen, lorazepam, metformin",
    label_visibility="collapsed",
)

if not drug_input.strip():
    st.info("Type a drug name above to see predicted side effects.")
    with st.expander("Browse all drugs in the dataset"):
        drugs = list_all_drugs()
        st.write(f"{len(drugs)} drugs available")
        st.dataframe(pd.DataFrame(drugs, columns=["drug_name"]), use_container_width=True)
    st.stop()

# -- Run prediction
with st.spinner("Running prediction ..."):
    result = predict(drug_input.strip())

if not result["found"]:
    st.error(
        f"**'{drug_input}'** was not found in the local SIDER dataset.  \n"
        f"The dataset contains 1,117 drugs. Try checking spelling, or browse the full list below."
    )
    with st.expander("Browse all drugs in the dataset"):
        drugs = list_all_drugs()
        st.dataframe(pd.DataFrame(drugs, columns=["drug_name"]), use_container_width=True)
    st.stop()

# -- Display results
predictions = result["predictions"]
n           = result["n_predicted"]
split_note  = "(training drug -- model has seen this drug's labels)" if result["split"] == "train" else ""

st.success(f"**{result['drug_name'].title()}** -- {n} side effects predicted above threshold. {split_note}")

TOP_N = 20
top   = predictions.head(TOP_N).copy()
top["score"] = top["score"].round(4)

st.markdown(f"#### Top {min(TOP_N, n)} predicted side effects")
st.dataframe(
    top.rename(columns={"side_effect": "Side effect", "score": "Score"}),
    use_container_width=True,
    hide_index=True,
)

if n > TOP_N:
    with st.expander(f"Show all {n} predicted side effects"):
        st.dataframe(
            predictions.rename(columns={"side_effect": "Side effect", "score": "Score"}),
            use_container_width=True,
            hide_index=True,
        )