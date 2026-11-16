
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from src.config import MODEL_KEYS, RANDOM_STATE
from src.preprocessing import clean_text

# Keeps apostrophes so "can't" / "wasn't" stay single tokens (negation!).
TOKEN_PATTERN = r"(?u)\b\w[\w']*\b"

_TFIDF_GRID = {
    "tfidf__ngram_range": [(1, 1), (1, 2), (1, 3)],
    "tfidf__min_df": [1, 2, 5],
}

PARAM_GRIDS = {
    "naive_bayes": {**_TFIDF_GRID, "clf__alpha": [0.01, 0.03, 0.1, 0.3, 1.0], "clf__fit_prior": [True, False]},
    "logistic_regression": {**_TFIDF_GRID, "clf__C": [0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0]},
    "linear_svm": {**_TFIDF_GRID, "clf__C": [0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0]},
}


def build_pipeline(key: str) -> Pipeline:
    if key not in MODEL_KEYS:
        raise ValueError(f"Unknown model key '{key}'. Expected one of {MODEL_KEYS}.")
    tfidf = TfidfVectorizer(
        preprocessor=clean_text,
        token_pattern=TOKEN_PATTERN,
        sublinear_tf=True,
        dtype=np.float64,
    )
    if key == "naive_bayes":
        clf = MultinomialNB()
    elif key == "logistic_regression":
        clf = LogisticRegression(max_iter=3000, random_state=RANDOM_STATE)
    else:
        clf = LinearSVC(max_iter=20000, random_state=RANDOM_STATE)
    return Pipeline([("tfidf", tfidf), ("clf", clf)])
