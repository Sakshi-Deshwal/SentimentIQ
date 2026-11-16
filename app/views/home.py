"""Home: overview, datasets, model status and navigation."""
from pathlib import Path

import pandas as pd
import streamlit as st

from app import common, theme
from src import config as C
from src import datasets as D
from src import inference as I
from src import settings, viz
from src.tasks import TASKS


def render():
    theme.page_header("SentimentIQ", "Understand what customers say. Every review is classed Positive, Neutral or Negative, "
                      "with honest evaluation of the models behind it.")
    task = common.task("home")
    ev = common.evaluation(task.key)
    trained = I.trained_models(task.key)
    badges = theme.badge(f"{len(trained)}/{len(task.model_keys)} models ready", "green" if trained else "amber")
    badges += theme.badge("rating-derived labels", "amber")
    body = f"<p>{task.blurb}</p>"
    if ev:
        b = ev["best_model_key_by_validation"]
        m = ev["models"][b]["test"]
        body += (f"<p><b>Best on validation:</b> {C.MODEL_NAMES[b]}<br>"
                 f"<b>Test macro-F1:</b> {m['f1_macro']:.3f} · <b>accuracy:</b> {m['accuracy']:.3f}</p>")
    else:
        body += "<p><i>Not evaluated yet.</i></p>"
    st.markdown(theme.card("Three-class sentiment analysis", body, badges), unsafe_allow_html=True)
    common.torch_notice(task.key)

    # ------------------------------------------------------------------ navigation
    st.subheader("Go to")
    nav = [("Live Sentiment Analysis", "Live analysis", "Analyze one review"),
           ("Bulk Sentiment Analysis", "Bulk analysis", "Upload a CSV and get insights"),
           ("Model Evaluation", "Evaluation", "Compare all models"),
           ("Cross-Domain Evaluation", "Cross-domain", "Held-out-domain experiments"),
           ("Error Analysis", "Error analysis", "Inspect mistakes"),
           ("Insights & Analytics", "Insights", "Charts and breakdowns")]
    for col, (page, label, hint) in zip(st.columns(len(nav)), nav):
        col.button(label, key=f"nav_{page}", on_click=common.go_to, args=(page,), help=hint, width="stretch")

    # ------------------------------------------------------------------ dataset statistics
    st.subheader("Dataset statistics")
    theme.weak_label_notice()
    ov = common.domain_overview()
    rows, totals, dom_counts = [], {l: 0 for l in C.THREE_LABELS}, {}
    for dom, r in ov.items():
        sent = r.get("sentiment", {})
        n = r.get("rows", 0)
        rows.append({"Domain": r["title"], "Status": "available" if r["available"] else "NOT FOUND",
                     "Reviews": n if r["available"] else None, "File (MB)": r["file_mb"] or None,
                     **{l: (sent.get(l, 0) / n if n else None) for l in C.THREE_LABELS}})
        if r["available"]:
            dom_counts[r["title"]] = n
            for l in totals:
                totals[l] += int(sent.get(l, 0))
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch",
                 column_config={l: st.column_config.NumberColumn(f"{l} %", format="percent") for l in C.THREE_LABELS})
    if sum(totals.values()):
        a, b = st.columns(2)
        with a:
            st.plotly_chart(viz.sentiment_distribution(totals, C.THREE_LABELS, "Sentiment distribution (all available domains)"),
                            width="stretch")
        with b:
            st.plotly_chart(viz.bar_counts(list(dom_counts), list(dom_counts.values()), "Domain distribution (reviews)"),
                            width="stretch")
    theme.english_notice()
    _local_dataset_panel(ov)

    # ------------------------------------------------------------------ model status
    st.subheader("Model training status")
    rows = []
    for task in TASKS.values():
        ev = common.evaluation(task.key)
        for k in task.model_keys:
            ok = I.is_trained(task.key, k)
            meta = I.read_model_meta(task.key, k) if ok else {}
            rows.append({"Model": C.MODEL_NAMES[k], "Family": C.MODEL_FAMILY[k],
                         "Status": "trained" if ok else "not trained",
                         "Evaluated": bool(ev and k in ev["models"]),
                         "Train time (s)": meta.get("train_seconds") or meta.get("fit_seconds"),
                         "Parameters": meta.get("parameters"),
                         "Trained on": ", ".join(meta.get("domains") or []) or None,
                         "Command to train": None if ok else I.train_command(task.key, k)})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.caption("Models are trained offline with the scripts in `scripts/` and only loaded here. "
               "Opening or switching pages never retrains anything.")


def _local_dataset_panel(ov: dict) -> None:
    food = ov["product_food"]
    with st.expander("Amazon dataset location (local file, read directly with pandas)", expanded=not food["available"]):
        if food["available"]:
            st.success(f"Found: `{food['path']}` ({food['file_mb']:,} MB, {food.get('rows', 0):,} rows)")
        else:
            st.warning(f"Amazon dataset not found at `{food['path']}`. The rest of the app still works; Amazon-dependent "
                       "features (food-domain training and experiments) are unavailable until the path is valid.")
        st.caption("The file is read from disk in chunks and never uploaded through Streamlit, so the "
                   f"{C.UPLOAD_MAX_MB} MB upload limit of the Bulk page does not apply to it. The original file is never "
                   "modified. Resolution order: environment variable `SENTIMENTIQ_AMAZON_PATH`, then `local_paths.json`, "
                   "then the default Windows Downloads path.")
        path_text = st.text_input("Path to amazon_fine_food_reviews_cleaned.csv", value=settings.get_amazon_path_string(),
                                  key="amazon_path_input")
        c1, c2, c3 = st.columns([1, 1, 2])
        if c1.button("Check path", key="amazon_check"):
            stt = D.file_status(Path(settings.clean_path_string(path_text)))
            (st.success if stt.exists else st.error)(f"{stt.message}" + (f" ({stt.size_mb:,.0f} MB)" if stt.exists else ""))
        if c2.button("Save path", key="amazon_save"):
            try:
                settings.save_amazon_path(path_text)
                st.cache_data.clear()
                st.cache_resource.clear()
                st.success("Saved to local_paths.json. Reloading...")
                st.rerun()
            except OSError as exc:
                st.error(f"Could not save (read-only environment?): {exc}. Use the SENTIMENTIQ_AMAZON_PATH environment variable instead.")
        sizes = settings.get_sample_sizes()
        cur = sizes.get("product_food") or 0
        n = c3.number_input("Training sample size for Amazon (stratified by sentiment; 0 = all rows)", min_value=0,
                            value=int(cur), step=5000, key="amazon_sample")
        if n != cur and st.button("Save sample size", key="amazon_sample_save"):
            try:
                settings.save_sample_size("product_food", n)
                st.cache_resource.clear()
                st.success("Saved. Retrain with `python scripts/train_models.py` to use it.")
            except OSError as exc:
                st.error(f"Could not save: {exc}")
        st.caption("Sampling is applied before splitting, preserving natural class proportions. "
                   "Models are never retrained when the app starts.")
