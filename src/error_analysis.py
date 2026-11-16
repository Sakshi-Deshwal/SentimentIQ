
import re

import numpy as np
import pandas as pd

from src.config import TEXT_COL, THREE_LABELS
from src.preprocessing import clean_text

CONTRAST_RE = re.compile(
    r"\b(but|however|although|though|yet|whereas|while|that said|on the other hand|except|unfortunately|despite|still)\b")
NEGATION_RE = re.compile(r"\b\w+n't\b|\b(not|no|never|nothing|hardly|barely|without|cannot)\b")


def add_flags(df: pd.DataFrame, pred_col: str) -> pd.DataFrame:
    out = df.copy()
    low = out[TEXT_COL].map(clean_text)
    out["correct"] = out["true_label"] == out[pred_col]
    out["n_words"] = low.str.split().str.len()
    out["has_contrast"] = low.map(lambda t: bool(CONTRAST_RE.search(t)))
    out["has_negation"] = low.map(lambda t: bool(NEGATION_RE.search(t)))
    return out


def confusion_pairs(df: pd.DataFrame, pred_col: str) -> pd.DataFrame:
    errs = df[df["true_label"] != df[pred_col]]
    pairs = (errs.groupby(["true_label", pred_col]).size().reset_index(name="count")
             .rename(columns={pred_col: "predicted_label"}).sort_values("count", ascending=False))
    total = max(len(errs), 1)
    pairs["share_of_errors"] = pairs["count"] / total
    return pairs.reset_index(drop=True)


def pattern_table(flagged: pd.DataFrame, pred_col: str) -> pd.DataFrame:
    """Correct vs misclassified reviews per true class: length, contrast words, negation, rating."""
    rows = []
    for label in THREE_LABELS:
        sub = flagged[flagged["true_label"] == label]
        for status, grp in (("correct", sub[sub["correct"]]), ("misclassified", sub[~sub["correct"]])):
            if len(grp) == 0:
                continue
            rows.append({
                "true_label": label, "outcome": status, "n": len(grp),
                "avg_words": round(grp["n_words"].mean(), 1),
                "% with contrast word": round(100 * grp["has_contrast"].mean(), 1),
                "% with negation": round(100 * grp["has_negation"].mean(), 1),
                "avg_rating": round(grp["rating"].mean(), 2) if "rating" in grp else None,
            })
    return pd.DataFrame(rows)


def distinctive_terms(flagged: pd.DataFrame, pred_col: str, true_label: str, pred_label: str,
                      k: int = 12, min_df: int = 4) -> pd.DataFrame:
    """Words over-represented in (true_label -> pred_label) errors vs correctly-classified true_label reviews."""
    err = flagged[(flagged["true_label"] == true_label) & (flagged[pred_col] == pred_label)]
    ok = flagged[(flagged["true_label"] == true_label) & flagged["correct"]]
    if len(err) < 5 or len(ok) < 5:
        return pd.DataFrame(columns=["term", "in_errors", "in_correct", "score"])

    def doc_freq(series):
        c = {}
        for t in series.map(clean_text):
            for w in set(re.findall(r"\b\w[\w']*\b", t)):
                c[w] = c.get(w, 0) + 1
        return c

    de, dc = doc_freq(err[TEXT_COL]), doc_freq(ok[TEXT_COL])
    rows = []
    for w, ce in de.items():
        if ce < min_df:
            continue
        cc = dc.get(w, 0)
        score = np.log((ce + 1) / (len(err) + 2)) - np.log((cc + 1) / (len(ok) + 2))
        rows.append({"term": w, "in_errors": ce, "in_correct": cc, "score": round(float(score), 2)})
    return pd.DataFrame(rows).sort_values("score", ascending=False).head(k).reset_index(drop=True)
