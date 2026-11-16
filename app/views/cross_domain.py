
import pandas as pd
import streamlit as st

from app import common, theme
from src import config as C
from src import datasets as D
from src import viz
from src.cross_domain import MODELS, POOLED_DOMAINS, experiment_id, run_experiment, run_pooled_experiment

SHORT = {"product_food": "Amazon", "product_clothing": "Clothing"}


def _name(domains) -> str:
    return " + ".join(SHORT.get(d, d) for d in domains)


def _results_frame(res: dict) -> pd.DataFrame:
    base = {r["id"]: r for r in res["results"] if r["kind"] == "in_domain_baseline"}
    rows = []
    for r in res["results"]:
        if r["kind"] != "held_out_domain":
            continue
        b = base.get(f"{r['test_domain']}->{r['test_domain']}::{r['model']}")
        rows.append({"Experiment": f"Train {_name(r['train_domains'])} → test {_name([r['test_domain']])}",
                     "Model": C.MODEL_NAMES[r["model"]], "Train n": r["n_train"], "Test n": r["n_test"],
                     "Accuracy": r["metrics"]["accuracy"], "Cross-domain macro F1": r["metrics"]["f1_macro"],
                     "In-domain baseline macro F1": b["metrics"]["f1_macro"] if b else None,
                     "Test OOV rate": r["test_oov_rate"]})
    return pd.DataFrame(rows)


def _pooled_frame(res: dict) -> pd.DataFrame:
    rows = []
    for r in res["results"]:
        if r["kind"] != "pooled_test":
            continue
        rows.append({"Model": C.MODEL_NAMES[r["model"]], "Train n": r["n_train"], "Test n": r["n_test"],
                     "Accuracy": r["metrics"]["accuracy"], "Macro F1": r["metrics"]["f1_macro"],
                     **{f"{SHORT.get(d, d)} macro F1": v["f1_macro"] for d, v in r["by_domain"].items()}})
    return pd.DataFrame(rows)


def _detail(r: dict) -> None:
    labels = C.THREE_LABELS
    m = r["metrics"]
    c = st.columns(5)
    c[0].metric("Training data", _name(r["train_domains"]))
    c[1].metric("Test data", r["test_domain"] if r["kind"] == "pooled_test" else _name([r["test_domain"]]))
    c[2].metric("Train / test samples", f"{r['n_train']:,} / {r['n_test']:,}")
    c[3].metric("Accuracy", f"{m['accuracy']:.4f}")
    c[4].metric("Macro F1", f"{m['f1_macro']:.4f}")
    if r["kind"] == "held_out_domain":
        st.caption(f"Out-of-vocabulary rate of the test text under the training vocabulary: {r['test_oov_rate']:.1%}. "
                   f"Hyper-parameters were fixed in advance, not tuned on the held-out domain. Test class counts: {r['test_class_counts']}.")
    a, b = st.columns(2)
    with a:
        st.plotly_chart(viz.confusion_matrix_fig(m["confusion_matrix"], labels, normalize=st.toggle(
            "Row-normalised", key=f"xdnorm_{r['id']}"), title="Confusion matrix"), width="stretch")
    with b:
        rep = m["classification_report"]
        st.dataframe(pd.DataFrame([{"Class": l, "Precision": rep[l]["precision"], "Recall": rep[l]["recall"],
                                    "F1": rep[l]["f1-score"], "Support": int(rep[l]["support"])} for l in labels]),
                     hide_index=True, width="stretch",
                     column_config={k: st.column_config.NumberColumn(format="%.4f") for k in ("Precision", "Recall", "F1")})


def render():
    theme.page_header("Cross-Domain Evaluation", "Train on one dataset and test on the other, or train on both and test on held-out reviews.")
    theme.weak_label_notice()
    theme.english_notice()
    st.info("**Cross-domain is not random-split performance.** Each transfer result is shown next to an in-domain baseline "
            "(trained and tested inside the test dataset). A model that scores well in-domain and poorly here is "
            "learning dataset-specific vocabulary. Amazon is food reviews; Clothing is women's clothing reviews.")

    res = common.cross_domain_results()
    tab_saved, tab_run = st.tabs(["Saved experiments", "Run a new experiment"])

    with tab_saved:
        if not res or not res.get("results"):
            st.warning("No cross-domain results saved yet. Run:")
            st.code("python scripts/cross_domain.py", language="bash")
        else:
            tbl = _results_frame(res)
            if len(tbl):
                st.subheader("Transfer between datasets")
                st.dataframe(tbl, hide_index=True, width="stretch", column_config={
                    c: st.column_config.NumberColumn(format="%.4f") for c in
                    ("Accuracy", "Cross-domain macro F1", "In-domain baseline macro F1")} | {
                    "Test OOV rate": st.column_config.NumberColumn(format="percent")})
                model = st.selectbox("Chart model", sorted(tbl["Model"].unique()), key="xd_chart_model")
                sub = tbl[(tbl["Model"] == model) & tbl["In-domain baseline macro F1"].notna()]
                if len(sub):
                    st.plotly_chart(viz.cross_domain_bars(sub, f"{model}: macro F1, transfer vs in-domain baseline"),
                                    width="stretch")
            pooled = _pooled_frame(res)
            if len(pooled):
                st.subheader("Both datasets pooled, evaluated on held-out test reviews")
                st.caption("Trained on the pooled training splits of Amazon and Clothing and scored on their held-out test splits. "
                           "This is ordinary multi-domain performance, not a transfer test.")
                st.dataframe(pooled, hide_index=True, width="stretch",
                             column_config={c: st.column_config.NumberColumn(format="%.4f") for c in pooled.columns
                                            if c not in ("Model", "Train n", "Test n")})
            allr = [r for r in res["results"] if r["kind"] in ("held_out_domain", "pooled_test")]
            if allr:
                pick = st.selectbox("Inspect one experiment", allr, key="xd_pick", format_func=lambda r: (
                    f"Pooled → held-out test · {C.MODEL_NAMES[r['model']]}" if r["kind"] == "pooled_test"
                    else f"{_name(r['train_domains'])} → {_name([r['test_domain']])} · {C.MODEL_NAMES[r['model']]}"))
                _detail(pick)
            for s in res.get("skipped", []):
                st.caption(f"Skipped: {' + '.join(s['train_domains'])} → {s['test_domain']} ({s['reason']})")

    with tab_run:
        st.write("Experiments train a fresh model (fixed hyper-parameters; the validation split of the training data drives early "
                 "stopping) and score it on the chosen test data. Nothing is tuned on the test data.")
        try:
            data = common.get_three_class_data()
        except Exception as exc:
            st.error(f"Could not prepare the datasets: {exc}")
            return
        for d, why in data.domains_skipped.items():
            st.warning(f"Unavailable dataset {d}: {why}")
        avail = [d for d in POOLED_DOMAINS if d in data.domains_used]
        if len(avail) < 2:
            st.error("Both datasets must be available to run a cross-domain experiment.")
            return
        mode = st.radio("Experiment", ["Train on Amazon, test on Clothing", "Train on Clothing, test on Amazon",
                                       "Train on both, evaluate on held-out test reviews"], key="xd_mode")
        model = st.selectbox("Model", list(MODELS), key="xd_model", format_func=lambda k: f"{C.MODEL_NAMES[k]} · {C.MODEL_FAMILY[k]}")
        c1, c2, c3 = st.columns(3)
        train_cap = c1.number_input("Max training rows", 1000, 100000, 10000, 1000, key="xd_trcap")
        test_cap = c2.number_input("Max test rows", 500, 50000, 5000, 500, key="xd_tecap")
        epochs = c3.number_input("Epochs (neural nets)", 1, 30, 6, 1, key="xd_epochs")
        from src.compat import TORCH_HELP, torch_available
        nn_blocked = model in C.NN_KEYS and not torch_available()
        if nn_blocked:
            st.warning(TORCH_HELP)
        if st.button("Run experiment", type="primary", key="xd_run", disabled=nn_blocked):
            try:
                with st.spinner("Training and evaluating..."):
                    over = {"epochs": int(epochs)}
                    if mode.startswith("Train on both"):
                        r = run_pooled_experiment(data.per_domain, model, train_cap=int(train_cap), nn_overrides=over)
                    else:
                        tr, te = ("product_food", "product_clothing") if "Amazon, test" in mode else ("product_clothing", "product_food")
                        r = run_experiment(data.per_domain, [tr], te, model, train_cap=int(train_cap),
                                           test_cap=int(test_cap), nn_overrides=over)
            except Exception as exc:
                st.error(f"Experiment failed: {exc}")
                return
            st.session_state["xd_last"] = r
        r = st.session_state.get("xd_last")
        if r:
            st.success(f"Finished: {r['id']} (not saved to disk; use scripts/cross_domain.py to save experiments).")
            _detail(r)
