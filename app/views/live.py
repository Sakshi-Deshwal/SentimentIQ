"""Live sentiment analysis for a single review."""
import pandas as pd
import streamlit as st

from app import common, theme
from src import config as C
from src import datasets as D
from src import inference as I
from src import viz
from src.predict import explain_terms

EXAMPLES = {
    "Positive": "Love this! The flavour is fantastic, it arrived quickly and it was packed well. I will definitely order again.",
    "Negative": "Terrible. It arrived late and stale, the taste was awful and nobody replied to my complaint. Do not buy.",
    "Neutral": "It is okay. The quality is average, the price is fair, nothing special but nothing wrong either.",
}


def _set_example(text: str) -> None:
    st.session_state["live_text"] = text


def render():
    theme.page_header("Live Sentiment Analysis", "Analyze a single review.")
    task = common.task("live")
    if task.weak_labels:
        theme.weak_label_notice()

    key = common.model_picker(task.key, "live")
    if key is None:
        return

    st.caption("Try an example")
    ex = EXAMPLES
    for col, (name, text) in zip(st.columns(len(ex)), ex.items()):
        col.button(name, key=f"ex_{task.key}_{name}", on_click=_set_example, args=(text,), width="stretch")

    text = st.text_area("Enter a customer review", key="live_text", height=140, placeholder="Type or paste a review...")
    compare = st.checkbox("Also show every other trained model for this task", value=False, key="live_compare")

    if st.button("Analyze", type="primary", key="live_go"):
        if not text or not text.strip():
            st.warning("Please enter some review text first.")
        else:
            try:
                res = common.get_predictor(task.key, key).predict_df([text]).iloc[0]
                others = {}
                if compare:
                    for k in I.trained_models(task.key):
                        others[k] = common.get_predictor(task.key, k).predict_df([text]).iloc[0]
            except (I.ModelNotFound, RuntimeError) as exc:
                st.error(str(exc))
                return
            except Exception as exc:
                st.error(f"Prediction failed: {exc}")
                return
            st.session_state["live_result"] = {"task": task.key, "key": key, "text": text, "res": res, "others": others}

    r = st.session_state.get("live_result")
    if r and r["task"] == task.key:
        _show(r, task)

    st.divider()
    st.info("**Confidence is not correctness.** A probability only says how strongly the text resembles one class "
            "according to patterns in the training data. Neural-network softmax scores are not calibrated, and any model "
            "can be confidently wrong, especially on sarcasm, negation, misspellings, very short text, reviews that praise and criticise at once, and "
            "language or domains it was not trained on. See the Error Analysis page for confident mistakes.")


def _show(r: dict, task) -> None:
    key, text, res = r["key"], r["text"], r["res"]
    labels = list(task.labels)
    st.markdown(f"##### Result from {C.MODEL_NAMES[key]}")
    label = res["prediction"]
    st.markdown(f"<div class='siq-pred' style='color:{viz.SENT[label]}'>{label}</div>", unsafe_allow_html=True)

    if len(text.split()) < 4:
        st.warning("Very short text: the models were trained on reviews of roughly 20-60 words, so short inputs are unreliable.")

    if f"prob_{labels[0]}" in res:
        probs = {l: res[f"prob_{l}"] for l in labels}
        c1, c2 = st.columns([1, 2])
        c1.metric("Top class probability", f"{max(probs.values()):.1%}")
        c1.caption("Model confidence, not a guarantee of correctness.")
        c2.plotly_chart(viz.probability_bars(probs, labels), width="stretch")
    elif f"score_{labels[0]}" in res:
        scores = {l: res[f"score_{l}"] for l in labels}
        st.info("Linear SVM does not output probabilities. Showing raw decision scores (higher = stronger lean towards "
                "that class). They are not probabilities and are not calibrated here.")
        st.plotly_chart(viz.probability_bars(scores, labels, "Decision scores", pct=False), width="stretch")

    pred = common.get_predictor(task.key, key)
    if getattr(pred, "pipeline", None) is not None:
        terms = explain_terms(pred.pipeline, text, label)
        if len(terms):
            with st.expander(f"Terms pushing the model towards '{label}'"):
                st.dataframe(terms.assign(contribution=terms["contribution"].round(3)), hide_index=True, width="stretch")
                st.caption("tf-idf weight × learned coefficient: a description of the model's arithmetic, "
                           "not proof of how it 'understands' the review. Sarcasm and context are not captured.")

    if r["others"]:
        rows = []
        for k, o in r["others"].items():
            tp = o.get("top_probability") if "top_probability" in o else None
            rows.append({"Model": C.MODEL_NAMES[k], "Family": C.MODEL_FAMILY[k], "Prediction": o["prediction"],
                         "Top probability": tp if tp is not None and pd.notna(tp) else None})
        st.markdown("##### All trained models")
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch",
                     column_config={"Top probability": st.column_config.NumberColumn(format="percent")})
        if len({row["Prediction"] for row in rows}) > 1:
            st.caption("The models disagree on this review, a sign that it is genuinely ambiguous or unlike the training data.")
