
import re
from typing import Optional

import pandas as pd

from src.config import THREE_LABELS

DATE_SUBSTRINGS = ("date", "timestamp", "created", "posted", "reviewed")   # match anywhere in the column name
DATE_TOKENS = {"time", "day", "month", "when"}                              # match only as a whole word (so "sentiment" never matches "time")
GROUP_HINTS = ("restaurant", "outlet", "branch", "location", "store", "venue", "business", "place",
               "domain", "category", "product", "source")
MIN_GROUP_REVIEWS = 20          # smaller groups are listed but never named best/worst
MIN_PERIOD_REVIEWS = 10


def _to_datetime(series: pd.Series) -> pd.Series:
    """Parse dates leniently (mixed formats allowed); unparseable values become NaT."""
    if pd.api.types.is_datetime64_any_dtype(series):
        return series
    return pd.to_datetime(series.astype("string"), errors="coerce", format="mixed")


def detect_date_column(df: pd.DataFrame) -> Optional[str]:
    """First column whose name hints at a date and whose values mostly parse as dates (else None)."""
    for col in df.columns:
        name = str(col).lower()
        if col == "predicted_sentiment" or not (any(h in name for h in DATE_SUBSTRINGS)
                                                or DATE_TOKENS & set(re.split(r"[^a-z0-9]+", name))):
            continue
        parsed = _to_datetime(df[col])
        if parsed.notna().mean() >= 0.8:
            return col
    return None


def detect_group_column(df: pd.DataFrame, exclude=()) -> Optional[str]:
    """A categorical column that splits reviews by branch / store / location / domain (2-50 distinct values)."""
    for col in df.columns:
        if col in exclude or not any(h in str(col).lower() for h in GROUP_HINTS):
            continue
        n = df[col].nunique(dropna=True)
        if 2 <= n <= 50:
            return col
    return None


def sentiment_counts(df: pd.DataFrame) -> dict:
    s = df["predicted_sentiment"].value_counts()
    return {l: int(s.get(l, 0)) for l in THREE_LABELS}


def net_sentiment(counts: dict) -> float:
    """Share Positive minus share Negative, in percentage points (range -100..+100)."""
    n = sum(counts.values())
    return 100.0 * (counts.get("Positive", 0) - counts.get("Negative", 0)) / n if n else float("nan")


def trend(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    """Share of each sentiment per period; the period (day / week / month) is chosen from the date span."""
    d = df.assign(_date=_to_datetime(df[date_col]))
    d = d.dropna(subset=["_date", "predicted_sentiment"])
    if d.empty:
        return pd.DataFrame()
    span = (d["_date"].max() - d["_date"].min()).days
    freq, label = ("D", "day") if span <= 31 else (("W", "week") if span <= 180 else ("M", "month"))
    period = d["_date"].dt.to_period(freq).dt.start_time
    counts = pd.crosstab(period, d["predicted_sentiment"]).reindex(columns=THREE_LABELS, fill_value=0)
    counts["Reviews"] = counts[THREE_LABELS].sum(axis=1)
    out = counts[THREE_LABELS].div(counts["Reviews"], axis=0)
    out["Reviews"] = counts["Reviews"]
    out.attrs["period"] = label
    return out


def group_table(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    d = df.dropna(subset=["predicted_sentiment", group_col])
    counts = pd.crosstab(d[group_col], d["predicted_sentiment"]).reindex(columns=THREE_LABELS, fill_value=0)
    out = counts.copy()
    out["Reviews"] = counts.sum(axis=1)
    for l in THREE_LABELS:
        out[f"{l} %"] = 100 * counts[l] / out["Reviews"]
    out["Net sentiment"] = 100 * (counts["Positive"] - counts["Negative"]) / out["Reviews"]
    return out.sort_values("Net sentiment", ascending=False)


def business_insights(df: pd.DataFrame, date_col: Optional[str] = None, group_col: Optional[str] = None) -> list:
    """A short list of plain-English findings. Every statement is computed from the data shown."""
    out = []
    d = df.dropna(subset=["predicted_sentiment"])
    if d.empty:
        return ["No reviews were analysed."]
    c = sentiment_counts(d)
    n = sum(c.values())
    out.append(f"Of {n:,} reviews, {c['Positive'] / n:.0%} are Positive, {c['Neutral'] / n:.0%} Neutral and "
               f"{c['Negative'] / n:.0%} Negative (net sentiment {net_sentiment(c):+.0f}).")
    if c["Negative"] / n >= 0.25:
        out.append("One in four or more reviews is Negative, which is worth reading through manually.")
    if group_col and group_col in d:
        g = group_table(d, group_col)
        big = g[g["Reviews"] >= MIN_GROUP_REVIEWS]
        if len(big) >= 2:
            best, worst = big.index[0], big.index[-1]
            out.append(f"Highest net sentiment: {best} ({big.iloc[0]['Net sentiment']:+.0f}, {int(big.iloc[0]['Reviews'])} reviews). "
                       f"Lowest: {worst} ({big.iloc[-1]['Net sentiment']:+.0f}, {int(big.iloc[-1]['Reviews'])} reviews).")
        small = int((g["Reviews"] < MIN_GROUP_REVIEWS).sum())
        if small:
            out.append(f"{small} group(s) have fewer than {MIN_GROUP_REVIEWS} reviews and are not ranked.")
    if date_col and date_col in d:
        t = trend(d, date_col)
        t = t[t["Reviews"] >= MIN_PERIOD_REVIEWS] if len(t) else t
        if len(t) >= 2:
            last, prev = t.iloc[-1], t.iloc[-2]
            diff = 100 * (last["Positive"] - prev["Positive"])
            unit = t.attrs.get("period", "period")
            if abs(diff) >= 2:
                out.append(f"Positive share {'rose' if diff > 0 else 'fell'} by {abs(diff):.0f} points in the latest {unit} "
                           f"({t.index[-1]:%Y-%m-%d}) compared with the one before.")
            else:
                out.append(f"Positive share was stable (within 2 points) between the last two {unit}s.")
        elif date_col:
            out.append("Not enough reviews per period to describe a trend.")
    out.append("These counts come from model predictions, which can be wrong on sarcasm, negation and unfamiliar language.")
    return out
