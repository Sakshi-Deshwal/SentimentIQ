"""Visual theme: one stylesheet and a few small layout helpers (no business logic)."""
import streamlit as st

CSS = """
<style>
html, body, [class*="css"], .stMarkdown, button, input, textarea {
  font-family: Inter, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
}
.block-container { padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1240px; }
h1, h2, h3 { letter-spacing: -0.015em; color: #0F172A; }
h1 { font-weight: 750; font-size: 2rem; margin-bottom: .1rem; }
[data-testid="stMetric"] {
  background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 12px; padding: 14px 18px;
  box-shadow: 0 1px 2px rgba(15,23,42,.04);
}
[data-testid="stMetricLabel"] p { color: #64748B; font-size: .82rem; font-weight: 600; }
[data-testid="stMetricValue"] { font-size: 1.55rem; font-weight: 700; }
[data-testid="stSidebar"] { background: #F8FAFC; border-right: 1px solid #E2E8F0; }
.siq-sub { color: #475569; font-size: 1.02rem; margin: 0 0 1.2rem 0; }
.siq-card { border: 1px solid #E2E8F0; border-radius: 14px; padding: 18px 20px; background: #FFFFFF;
            box-shadow: 0 1px 2px rgba(15,23,42,.04); height: 100%; }
.siq-card h4 { margin: 0 0 .35rem 0; font-size: 1.05rem; }
.siq-card p { margin: .2rem 0; color: #475569; font-size: .92rem; }
.siq-badge { display: inline-block; padding: 2px 10px; border-radius: 999px; font-size: .75rem; font-weight: 650;
             margin-right: 6px; border: 1px solid transparent; }
.siq-pred { font-size: 2.1rem; font-weight: 800; letter-spacing: -.02em; margin: .2rem 0 .6rem 0; }
.siq-brand { font-size: 1.35rem; font-weight: 800; letter-spacing: -.02em; color: #4F46E5; margin-bottom: 0; }
.siq-brand-sub { color: #64748B; font-size: .8rem; margin-top: 0; }
div[data-testid="stTabs"] button[role="tab"] { font-weight: 600; }
</style>
"""

BADGES = {
    "indigo": ("#EEF2FF", "#4338CA", "#C7D2FE"),
    "green": ("#F0FDF4", "#166534", "#BBF7D0"),
    "amber": ("#FFFBEB", "#92400E", "#FDE68A"),
    "red": ("#FEF2F2", "#991B1B", "#FECACA"),
    "slate": ("#F1F5F9", "#334155", "#CBD5E1"),
}


def inject_css() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def page_header(title: str, subtitle: str = "") -> None:
    st.markdown(f"# {title}")
    if subtitle:
        st.markdown(f"<p class='siq-sub'>{subtitle}</p>", unsafe_allow_html=True)


def badge(text: str, tone: str = "indigo") -> str:
    bg, fg, border = BADGES[tone]
    return f"<span class='siq-badge' style='background:{bg};color:{fg};border-color:{border}'>{text}</span>"


def card(title: str, body_html: str, badges_html: str = "") -> str:
    return f"<div class='siq-card'><h4>{title}</h4>{badges_html}{body_html}</div>"


def weak_label_notice() -> None:
    st.warning("**Weak labels.** In the three-class task, sentiment was derived from star ratings "
               "(1-2 = Negative, 3 = Neutral, 4-5 = Positive); it is not manually annotated sentiment. "
               "A 3-star review can be clearly positive or negative in its text, so these scores measure how well "
               "a model recovers the *rating-based* label, not true human sentiment.", icon="⚠️")


def english_notice() -> None:
    st.info("**English only.** Both datasets (Amazon fine food and women's clothing) are English-language reviews and the "
            "models are trained on English text only. Behaviour on other languages is untested and not expected to be reliable.",
            icon="🌐")
