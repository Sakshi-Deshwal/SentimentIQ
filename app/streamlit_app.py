"""SentimentIQ - Streamlit entry point.

Run from the project root:  streamlit run app/streamlit_app.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st  # noqa: E402

st.set_page_config(page_title="SentimentIQ", page_icon="💬", layout="wide")

from app import common, theme  # noqa: E402
from app.views import batch, cross_domain, errors, evaluation, home, insights, live  # noqa: E402

PAGES = {
    "Home": home.render,
    "Live Sentiment Analysis": live.render,
    "Bulk Sentiment Analysis": batch.render,
    "Model Evaluation": evaluation.render,
    "Cross-Domain Evaluation": cross_domain.render,
    "Error Analysis": errors.render,
    "Insights & Analytics": insights.render,
}

theme.inject_css()
if st.session_state.get("nav") not in PAGES:
    st.session_state["nav"] = "Home"

with st.sidebar:
    st.markdown("<p class='siq-brand'>SentimentIQ</p><p class='siq-brand-sub'>Multi-model sentiment analysis</p>",
                unsafe_allow_html=True)
    page = st.radio("Navigate", list(PAGES), key="nav", label_visibility="collapsed")
    st.divider()
    card = common.model_card("three_class")
    if card:
        st.caption(f"{len(card.get('models', {}))} models · updated {str(card.get('updated_utc') or card.get('created_utc', ''))[:10]}")
    else:
        st.caption("Models not trained yet")
    st.caption("Positive · Neutral · Negative. Input: review text only. Labels are rating-derived.")

try:
    PAGES[page]()
except Exception as exc:  # last-resort guard so users never see a raw traceback
    st.error(f"Something went wrong on this page: {exc}")
    with st.expander("Technical details"):
        st.exception(exc)
