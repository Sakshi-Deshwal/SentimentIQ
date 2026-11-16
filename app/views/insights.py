"""Insights and analytics: charts that answer a question, all computed from real data and reports."""
import pandas as pd
import streamlit as st

from app import common, theme
from src import config as C
from src import viz


def render():
    theme.page_header("Insights & Analytics", "Where the data is skewed, which models win, and where they fail.")
    task = common.task("ins")
    ev = common.require_evaluation(task.key)
    labels = list(task.labels)
    models = ev["models"]

    # ------------------------------------------------------------------ data
    st.subheader("Sentiment distribution")
    theme.weak_label_notice()
    ov = common.domain_overview()
    table = pd.DataFrame({r["title"]: r.get("sentiment", {}) for r in ov.values() if r["available"]}).T.fillna(0)
    if len(table):
        a, b = st.columns(2)
        a.plotly_chart(viz.sentiment_distribution(table.sum().to_dict(), labels, "All available domains"), width="stretch")
        b.plotly_chart(viz.domain_sentiment(table, labels, title="Domain-wise sentiment (share of each domain)"), width="stretch")
        st.caption("Positive dominates every domain; Neutral (3-star) is the rarest class, so macro-F1 matters more than accuracy.")
    audit = common.dataset_audit()
    if audit:
        st.caption(f"Training used: {', '.join(audit['domains_used'])}; sample sizes: "
                   f"{ {d: v['sample_size_requested'] for d, v in audit['domains'].items()} }.")

    # ------------------------------------------------------------------ models
    st.subheader("Model performance")
    tbl = pd.DataFrame([{"Model": m["name"], "Family": m["family"], "Accuracy": m["test"]["accuracy"],
                         "Macro F1": m["test"]["f1_macro"], "Weighted F1": m["test"]["f1_weighted"]} for m in models.values()])
    st.plotly_chart(viz.metric_comparison(tbl, ["Accuracy", "Macro F1", "Weighted F1"], "Test performance"), width="stretch")
    f1 = pd.DataFrame({m["name"]: {l: m["test"]["classification_report"][l]["f1-score"] for l in labels} for m in models.values()}).T
    st.plotly_chart(viz.class_f1_heatmap(f1), width="stretch")
    worst = f1.stack().idxmin()
    st.caption(f"Weakest cell: {worst[0]} on {worst[1]} (F1 {f1.stack().min():.3f}).")

    key = st.selectbox("Confusion matrix for", list(models), index=list(models).index(ev["best_model_key_by_validation"]),
                       key=f"insmodel_{task.key}", format_func=lambda k: models[k]["name"])
    st.plotly_chart(viz.confusion_matrix_fig(models[key]["test"]["confusion_matrix"], labels,
                                             normalize=True, title=f"{models[key]['name']} (row-normalised)"), width="stretch")

    # ------------------------------------------------------------------ errors
    st.subheader("Error distribution")
    preds = common.predictions(task.key)
    if preds is not None:
        rows = []
        for k in models:
            col = f"pred_{k}"
            if col in preds:
                for t in labels:
                    sub = preds[preds["true_label"] == t]
                    rows.append({"Model": models[k]["name"], "True class": t, "errors": int((sub[col] != sub["true_label"]).sum())})
        err = pd.DataFrame(rows)
        st.plotly_chart(viz.error_bars(err, "Model", "errors", color="True class", title="Errors by true class"), width="stretch")
        if "domain" in preds and key and f"pred_{key}" in preds:
            d = preds.assign(wrong=preds[f"pred_{key}"] != preds["true_label"]).groupby("domain")["wrong"].agg(["sum", "size"])
            d["error_rate"] = d["sum"] / d["size"]
            st.dataframe(d.rename(columns={"sum": "errors", "size": "reviews"}).reset_index(), hide_index=True, width="stretch",
                         column_config={"error_rate": st.column_config.NumberColumn(format="percent")})
