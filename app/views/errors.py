
import pandas as pd
import streamlit as st

from app import common, theme
from src import config as C
from src import error_analysis as ea
from src import viz


def render():
    theme.page_header("Error Analysis", "Inspect what the models get wrong, and where.")
    tab_test, tab_rob = st.tabs(["Test-set errors", "Robustness challenge set"])
    with tab_test:
        _test_errors()
    with tab_rob:
        _robustness()


def _test_errors():
    task = common.task("err")
    common.require_evaluation(task.key)
    preds = common.predictions(task.key)
    if preds is None:
        st.error("test_predictions.csv is missing. Run the evaluation script.")
        return
    theme.weak_label_notice()
    keys = [k[5:] for k in preds.columns if k.startswith("pred_")]
    labels = list(task.labels)
    key = st.selectbox("Model", keys, key=f"errmodel_{task.key}", format_func=lambda k: f"{C.MODEL_NAMES[k]} · {C.MODEL_FAMILY[k]}")
    pred_col, conf_col = f"pred_{key}", f"top_prob_{key}"
    flagged = ea.add_flags(preds, pred_col)
    n_err = int((~flagged["correct"]).sum())
    c1, c2, c3 = st.columns(3)
    c1.metric("Test reviews", f"{len(flagged):,}")
    c2.metric("Misclassified", f"{n_err:,}")
    c3.metric("Error rate", f"{n_err / len(flagged):.1%}")
    if n_err == 0:
        st.success("No errors on the test split. That is unusual for a real task; check the data for artefacts.")
        return

    st.subheader("Where the errors are")
    pairs = ea.confusion_pairs(flagged, pred_col)
    st.dataframe(pairs.assign(share_of_errors=(pairs["share_of_errors"] * 100).round(1)).rename(columns={
        "true_label": "True", "predicted_label": "Predicted", "count": "Errors", "share_of_errors": "% of all errors"}),
        hide_index=True, width="stretch")

    if "domain" in flagged:
        by = flagged.groupby("domain").agg(reviews=("correct", "size"), errors=("correct", lambda s: int((~s).sum())))
        by["error_rate"] = by["errors"] / by["reviews"]
        a, b = st.columns([2, 3])
        a.dataframe(by.reset_index(), hide_index=True, width="stretch",
                    column_config={"error_rate": st.column_config.NumberColumn(format="percent")})
        b.plotly_chart(viz.error_bars(by.reset_index(), "domain", "errors", title="Errors by domain"), width="stretch")

    st.caption("Neutral here means a 3-star rating. Many 'Neutral' errors are reviews whose text is clearly positive or negative.")

    st.subheader("Failure patterns")
    st.dataframe(ea.pattern_table(flagged, pred_col), hide_index=True, width="stretch")
    st.caption("Correct vs misclassified reviews of the same true class. Contrast = but/however/though/yet...; negation = not/never/n't/without... "
               "Rating (when present) is shown for analysis only; it is never a model input.")
    if conf_col in flagged:
        wrong = flagged[~flagged["correct"]]
        hi = int((wrong[conf_col] >= 0.7).sum())
        st.markdown(f"- **{hi}** of {len(wrong)} errors ({hi / len(wrong):.0%}) had a top probability ≥ 70%: confidence is not correctness.")
    top = pairs.iloc[0]
    terms = ea.distinctive_terms(flagged, pred_col, top["true_label"], top["predicted_label"])
    if len(terms):
        with st.expander(f"Words over-represented in the most common error ({top['true_label']} → {top['predicted_label']})"):
            st.dataframe(terms, hide_index=True, width="stretch")

    st.subheader("Browse misclassified reviews")
    f1, f2, f3 = st.columns(3)
    t_sel = f1.multiselect("True label", labels, key=f"ft_{task.key}")
    p_sel = f2.multiselect("Predicted label", labels, key=f"fp_{task.key}")
    d_sel = f3.multiselect("Domain", sorted(flagged["domain"].unique()), key=f"fd_{task.key}") if "domain" in flagged else []
    errs = flagged[~flagged["correct"]]
    if t_sel:
        errs = errs[errs["true_label"].isin(t_sel)]
    if p_sel:
        errs = errs[errs[pred_col].isin(p_sel)]
    if d_sel:
        errs = errs[errs["domain"].isin(d_sel)]
    only_conf = st.checkbox("Sort by model confidence (highest first)", value=conf_col in errs, key=f"fs_{task.key}") if conf_col in errs else False
    if only_conf:
        errs = errs.sort_values(conf_col, ascending=False)
    cols = ["review_id", C.TEXT_COL, "true_label", pred_col] + ([conf_col] if conf_col in errs else []) + \
           [c for c in ("domain", "rating", "category") if c in errs]
    view = errs[cols].rename(columns={pred_col: "predicted", conf_col: "confidence"}).head(300)
    view.insert(3, "model", C.MODEL_NAMES[key])
    st.write(f"{len(errs):,} misclassified reviews match the filters (showing up to 300).")
    st.dataframe(view, hide_index=True, width="stretch",
                 column_config={"confidence": st.column_config.NumberColumn(format="percent")})


def _robustness():
    st.write("Hand-written reviews (sarcasm, negation, typos, ambiguity...) that are **not** in the training data. "
             "Expected labels are the author's judgement; 'debatable' rows are genuinely ambiguous.")
    res = common.read_csv(C.ROBUSTNESS_RESULTS_PATH)
    summ = common.read_json(C.ROBUSTNESS_SUMMARY_PATH)
    if res is None or summ is None:
        st.info("Run the challenge first:")
        st.code("python scripts/robustness_test.py", language="bash")
        return
    st.warning(f"With only {summ['n']} examples this is a diagnostic probe, not evidence of real-world generalisation.")
    rows = [{"Model": C.MODEL_NAMES[k], "Family": C.MODEL_FAMILY[k], "Accuracy": v["accuracy"], "Macro F1": v["f1_macro"],
             f"Accuracy (clear labels only)": v["accuracy_clear_labels_only"]} for k, v in summ["models"].items()]
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch",
                 column_config={c: st.column_config.NumberColumn(format="%.3f") for c in ("Accuracy", "Macro F1", "Accuracy (clear labels only)")})
    keys = list(summ["models"])
    by = pd.DataFrame({C.MODEL_NAMES[k]: {t: f"{r['correct']}/{r['n']}" for t, r in summ["models"][k]["accuracy_by_challenge"].items()}
                       for k in keys})
    st.dataframe(by, width="stretch")
    key = st.selectbox("Show failures for", keys, format_func=lambda k: C.MODEL_NAMES[k], key="rob_model")
    fails = res[~res[f"correct_{key}"]]
    cols = ["test_id", "challenge_type", "label_confidence", C.TEXT_COL, "expected_sentiment", f"pred_{key}"]
    if f"top_prob_{key}" in res:
        cols.append(f"top_prob_{key}")
    st.write(f"{len(fails)} of {len(res)} examples misclassified")
    st.dataframe(fails[cols], hide_index=True, width="stretch")
