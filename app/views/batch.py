
import io

import pandas as pd
import streamlit as st

from app import common, theme
from src import config as C
from src import inference as I
from src import viz
from src.metrics import compute_metrics
from src import insights as ins
from src.tasks import get_task

MAX_ROWS = 100_000            # analysed per upload; larger files are truncated with a warning
BATCH = 5_000                 # rows per prediction batch (keeps memory bounded, drives the progress bar)


def _read_upload(uploaded) -> pd.DataFrame:
    raw = uploaded.getvalue()
    for enc in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            return pd.read_csv(io.BytesIO(raw), encoding=enc)
        except UnicodeDecodeError:
            continue
        except pd.errors.EmptyDataError:
            raise ValueError("The file is empty.")
        except pd.errors.ParserError as exc:
            raise ValueError(f"The file is not a valid CSV: {exc}")
    raise ValueError("Could not decode the file. Please save it as UTF-8 CSV.")


def render():
    theme.page_header("Bulk Sentiment Analysis", "Upload a CSV of reviews: get a prediction for every row, plus sentiment trends and plain-language insights.")
    task = common.task("bulk")
    if task.weak_labels:
        theme.weak_label_notice()
    key = common.model_picker(task.key, "bulk")
    if key is None:
        return

    template = pd.DataFrame({C.TEXT_COL: ["Great food and friendly staff.", "Waited an hour and the soup was cold."],
                             "branch": ["Store A", "Store A"], "date": ["2026-01-05", "2026-01-06"]})
    st.caption("Required: a review-text column. Optional: a **date** column (for trends) and a **branch / store / location / domain** "
               "column (for per-location insights). Extra columns are kept in the download and never used by the model.")
    st.download_button("Download CSV template", template.to_csv(index=False).encode("utf-8"), "bulk_template.csv", "text/csv")
    st.caption(f"Uploads are limited to {C.UPLOAD_MAX_MB} MB (a Streamlit setting). Large local datasets such as the Amazon file "
               "are loaded from disk by the training scripts and are not subject to this limit.")

    uploaded = st.file_uploader("Reviews CSV", type=["csv"], key="bulk_upload")
    if uploaded is None:
        return
    if uploaded.size > C.UPLOAD_MAX_MB * 1024 * 1024:
        st.error(f"File is larger than {C.UPLOAD_MAX_MB} MB. Split it into smaller files.")
        return
    try:
        df = _read_upload(uploaded)
    except ValueError as exc:
        st.error(str(exc))
        return
    if df.empty or df.shape[1] == 0:
        st.error("The file has no rows or columns.")
        return
    if len(df) > MAX_ROWS:
        st.warning(f"The file has {len(df):,} rows; only the first {MAX_ROWS:,} are analysed.")
        df = df.head(MAX_ROWS)
    st.caption(f"Loaded {len(df):,} rows × {df.shape[1]} columns")
    st.dataframe(df.head(5), width="stretch")

    cols = list(df.columns)
    text_col = st.selectbox("Column containing the review text", cols,
                            index=cols.index(C.TEXT_COL) if C.TEXT_COL in cols else 0, key="bulk_textcol")
    rest = [c for c in cols if c != text_col]
    auto_date, auto_group = ins.detect_date_column(df[rest]) if rest else None, ins.detect_group_column(df, exclude=[text_col])
    date_col = st.selectbox("Optional: date column (enables the trend chart)", ["(none)"] + rest,
                            index=(rest.index(auto_date) + 1) if auto_date in rest else 0, key="bulk_datecol")
    group_col = st.selectbox("Optional: branch / store / location column (enables per-location insights)", ["(none)"] + rest,
                             index=(rest.index(auto_group) + 1) if auto_group in rest else 0, key="bulk_groupcol")
    truth_col = st.selectbox("Optional: column with true labels (to measure agreement on your file)",
                             ["(none)"] + [c for c in cols if c != text_col], key="bulk_truth")

    if st.button("Run analysis", type="primary", key="bulk_run"):
        texts = df[text_col].fillna("").astype(str)
        valid = texts.str.strip() != ""
        if not valid.any():
            st.error("The selected column has no non-empty text.")
            return
        try:
            predictor = common.get_predictor(task.key, key)
            idx = texts[valid].index
            parts, bar = [], st.progress(0.0, text="Predicting...")
            for i in range(0, len(idx), BATCH):
                chunk = idx[i:i + BATCH]
                p = predictor.predict_df(texts.loc[chunk].tolist())
                p.index = chunk
                parts.append(p)
                bar.progress(min((i + BATCH) / len(idx), 1.0), text=f"Predicted {min(i + BATCH, len(idx)):,} of {len(idx):,}")
            bar.empty()
        except (I.ModelNotFound, RuntimeError) as exc:
            st.error(str(exc))
            return
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")
            return
        pred = pd.concat(parts)
        out = df.copy()
        out["predicted_sentiment"] = pd.NA
        out["status"] = "empty_text"
        out.loc[pred.index, "predicted_sentiment"] = pred["prediction"]
        out.loc[pred.index, "status"] = "ok"
        for c in pred.columns:
            if c != "prediction":
                out[c] = pd.NA
                out.loc[pred.index, c] = pred[c].round(4)
        out["model"] = C.MODEL_NAMES[key]
        st.session_state["bulk_result"] = (out, int((~valid).sum()), key, task.key, truth_col, date_col, group_col)

    res = st.session_state.get("bulk_result")
    if not res:
        return
    out, n_empty, used_key, used_task, used_truth, used_date, used_group = res
    labels = list(get_task(used_task).labels)
    if n_empty:
        st.warning(f"{n_empty:,} row(s) had empty text; they were kept in the output with status 'empty_text' and no prediction.")

    st.subheader("Results")
    st.caption(f"Model: {C.MODEL_NAMES[used_key]}")
    counts = out["predicted_sentiment"].value_counts().reindex(labels).fillna(0).astype(int)
    a, b = st.columns([3, 2])
    a.plotly_chart(viz.sentiment_distribution(counts.to_dict(), labels, "Predicted sentiment distribution"), width="stretch")
    b.dataframe(pd.DataFrame({"Reviews": counts, "Share": counts / max(counts.sum(), 1)}), width="stretch",
                column_config={"Share": st.column_config.NumberColumn(format="percent")})
    _business_view(out, used_date, used_group, labels)
    st.subheader("Predictions")
    st.dataframe(out.head(200), width="stretch")
    if len(out) > 200:
        st.caption("Showing the first 200 rows. The download contains all rows.")
    st.download_button("Download predictions (CSV)", out.to_csv(index=False).encode("utf-8"), "sentiment_predictions.csv",
                       "text/csv", type="primary")

    if used_truth != "(none)" and used_truth in out.columns:
        lab = out[out["status"] == "ok"]
        lab = lab[lab[used_truth].isin(labels)]
        if len(lab):
            m = compute_metrics(lab[used_truth], lab["predicted_sentiment"], labels)
            st.subheader("Agreement with your labels")
            c1, c2, c3 = st.columns(3)
            c1.metric("Rows compared", f"{len(lab):,}")
            c2.metric("Accuracy", f"{m['accuracy']:.1%}")
            c3.metric("Macro F1", f"{m['f1_macro']:.3f}")
            st.plotly_chart(viz.confusion_matrix_fig(m["confusion_matrix"], labels, normalize=True,
                                                     title="Row-normalised confusion matrix"), width="stretch")
        else:
            st.info(f"No rows in '{used_truth}' match this task's labels ({', '.join(labels)}).")
    st.caption("Predictions are statistical estimates and can be wrong; review important cases manually.")


def _business_view(out: pd.DataFrame, date_col: str, group_col: str, labels: list) -> None:
    ok = out[out["status"] == "ok"]
    dcol = date_col if date_col != "(none)" else None
    gcol = group_col if group_col != "(none)" else None
    st.subheader("Business insights")
    for line in ins.business_insights(ok, dcol, gcol):
        st.markdown(f"- {line}")
    if dcol:
        t = ins.trend(ok, dcol)
        if len(t):
            st.plotly_chart(viz.sentiment_trend(t, labels, f"Sentiment over time (per {t.attrs.get('period', 'period')})"),
                            width="stretch")
            st.caption("Periods with few reviews are noisy; read the trend together with the review counts below.")
            st.dataframe(t.assign(**{l: t[l] for l in labels}), width="stretch",
                         column_config={l: st.column_config.NumberColumn(format="percent") for l in labels})
        else:
            st.info(f"The '{dcol}' column could not be read as dates, so no trend is shown.")
    if gcol:
        g = ins.group_table(ok, gcol)
        if len(g):
            st.plotly_chart(viz.domain_sentiment(g[labels], labels, title=f"Sentiment by {gcol}"), width="stretch")
            show = g[["Reviews", "Positive %", "Neutral %", "Negative %", "Net sentiment"]].round(1)
            st.dataframe(show, width="stretch")
            st.caption("Net sentiment = % Positive minus % Negative. Groups with fewer than "
                       f"{ins.MIN_GROUP_REVIEWS} reviews are shown but not ranked.")
