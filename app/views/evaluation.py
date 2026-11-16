"""Model evaluation: traditional ML vs neural networks, per task. Every number comes from reports/."""
import pandas as pd
import streamlit as st

from app import common, theme
from src import config as C
from src import viz


def render():
    theme.page_header("Model Evaluation", "Traditional ML vs neural networks. Train, validation and test results are shown separately.")
    task = common.task("eval")
    ev = common.require_evaluation(task.key)
    labels = list(task.labels)
    models = ev["models"]
    best = ev["best_model_key_by_validation"]
    theme.weak_label_notice()
    theme.english_notice()
    common.torch_notice(task.key)
    st.caption(f"Domains used for training: {', '.join(ev.get('domains_trained_on') or []) or 'n/a'}.")
    st.caption(f"Train {ev['n']['train']:,} · validation {ev['n']['validation']:,} · test {ev['n']['test']:,} reviews. "
               f"The best model ({C.MODEL_NAMES[best]}) was chosen by **validation** macro-F1; the test split was evaluated once. "
               "Single split, single seed: small differences between models are within noise.")

    # ------------------------------------------------------------------ comparison table
    rows = []
    for k, m in models.items():
        t, v = m["test"], m["validation"]
        rows.append({"Model": m["name"], "Family": m["family"],
                     "Train acc": m["splits"]["train"]["accuracy"], "Val acc": v["accuracy"], "Val macro-F1": v["f1_macro"],
                     "Test acc": t["accuracy"], "Test macro-F1": t["f1_macro"], "Test weighted-F1": t["f1_weighted"],
                     "Train time (s)": m.get("train_seconds"), "Inference (ms/review)": m["inference"]["ms_per_review"],
                     "Parameters": m.get("parameters")})
    tbl = pd.DataFrame(rows)
    st.subheader("Comparison")
    st.dataframe(tbl, hide_index=True, width="stretch", column_config={
        c: st.column_config.NumberColumn(format="%.4f") for c in
        ("Train acc", "Val acc", "Val macro-F1", "Test acc", "Test macro-F1", "Test weighted-F1")})
    fam = tbl.groupby("Family")[["Test acc", "Test macro-F1"]].max().round(4)
    st.caption("Best score per family: " + "; ".join(f"{f}: accuracy {r['Test acc']:.4f}, macro-F1 {r['Test macro-F1']:.4f}"
                                                      for f, r in fam.iterrows()))

    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(viz.metric_comparison(tbl, ["Test acc", "Test macro-F1"], "Test accuracy and macro-F1"), width="stretch")
    with c2:
        gap = tbl.rename(columns={"Train acc": "Train", "Val acc": "Validation", "Test acc": "Test"})
        st.plotly_chart(viz.split_gap(gap), width="stretch")
    st.caption("Training accuracy is measured in eval mode on the training split. A large gap to validation/test means overfitting.")

    # ------------------------------------------------------------------ per model detail
    st.subheader("Model detail")
    keys = list(models)
    key = st.selectbox("Model", keys, index=keys.index(best), key=f"evalmodel_{task.key}",
                       format_func=lambda k: f"{models[k]['name']} · {models[k]['family']}" + (" ★" if k == best else ""))
    m = models[key]
    t = m["test"]
    a, b, c, d = st.columns(4)
    a.metric("Test accuracy", f"{t['accuracy']:.4f}")
    b.metric("Test macro-F1", f"{t['f1_macro']:.4f}")
    c.metric("Train time", f"{m['train_seconds']:.1f} s" if m.get("train_seconds") else "n/a")
    d.metric("Inference", f"{m['inference']['ms_per_review']:.3f} ms/review")

    left, right = st.columns(2)
    with left:
        norm = st.toggle("Row-normalised (recall view)", value=False, key=f"norm_{task.key}")
        st.plotly_chart(viz.confusion_matrix_fig(t["confusion_matrix"], labels, normalize=norm,
                                                 title=f"{m['name']} · test confusion matrix"), width="stretch")
    with right:
        rep = t["classification_report"]
        cls = pd.DataFrame([{"Class": l, "Precision": rep[l]["precision"], "Recall": rep[l]["recall"],
                             "F1": rep[l]["f1-score"], "Support": int(rep[l]["support"])} for l in labels])
        st.dataframe(cls, hide_index=True, width="stretch",
                     column_config={c_: st.column_config.NumberColumn(format="%.4f") for c_ in ("Precision", "Recall", "F1")})
        st.plotly_chart(viz.per_class_bars(rep, labels), width="stretch")
    with st.expander("Full classification report (test)"):
        st.code(t["classification_report_text"])
    with st.expander("Validation classification report"):
        st.code(m["validation"]["classification_report_text"])

    if m.get("test_by_domain"):
        st.markdown("##### Test performance by domain")
        dom = pd.DataFrame([{"Domain": d_, "Reviews": v["n"], "Accuracy": v["accuracy"], "Macro-F1": v["f1_macro"],
                             **{f"{l} F1": v["classification_report"][l]["f1-score"] for l in labels}}
                            for d_, v in m["test_by_domain"].items()])
        st.dataframe(dom, hide_index=True, width="stretch")

    # ------------------------------------------------------------------ learning curves
    if m.get("history"):
        st.subheader("Learning curves")
        st.plotly_chart(viz.learning_curves(m["history"], m.get("best_epoch"), f"{m['name']}: per-epoch history"), width="stretch")
        st.caption(f"{m['epochs_run']} epochs run, best epoch {m['best_epoch']} (dotted line); "
                   + ("early stopping triggered." if m.get("stopped_early") else "ran to the epoch limit.")
                   + " Training curves use dropout, so training loss is not directly comparable to evaluation-mode numbers.")
    elif m["family"] == "Neural network":
        st.info("Learning-curve history was not recorded for this model. Retrain it to regenerate it.")
    else:
        st.caption("Traditional models have no epochs, so there are no learning curves. "
                   + (f"Best parameters: {m.get('best_params')}" if m.get("best_params") else ""))
