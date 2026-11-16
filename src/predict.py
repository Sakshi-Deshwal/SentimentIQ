
import numpy as np
import pandas as pd


def predict_texts(pipeline, texts) -> pd.DataFrame:
    """Predict labels for texts.

    Returns a DataFrame with ``prediction`` plus either ``prob_<label>`` columns
    (models with ``predict_proba``) or ``score_<label>`` columns (raw decision
    scores for Linear SVM; these are NOT probabilities).
    """
    texts = pd.Series(list(texts), dtype="object").fillna("").astype(str)
    clf = pipeline.named_steps["clf"]
    classes = list(clf.classes_)
    out = pd.DataFrame({"prediction": pipeline.predict(texts)})
    if hasattr(pipeline, "predict_proba"):
        proba = pipeline.predict_proba(texts)
        for i, c in enumerate(classes):
            out[f"prob_{c}"] = proba[:, i]
        out["top_probability"] = proba.max(axis=1)
    elif hasattr(pipeline, "decision_function"):
        scores = pipeline.decision_function(texts)
        for i, c in enumerate(classes):
            out[f"score_{c}"] = scores[:, i]
    return out


def supports_probabilities(pipeline) -> bool:
    return hasattr(pipeline, "predict_proba")


def explain_terms(pipeline, text: str, label: str = None, top_k: int = 8) -> pd.DataFrame:
    """Terms in ``text`` that push the model towards ``label`` (default: predicted).

    Linear models and Naive Bayes: contribution = tf-idf weight x class coefficient (Naive Bayes: class
    log-probability minus the mean over classes). This is a
    description of the model's arithmetic, not proof of causal reasoning.
    """
    tfidf = pipeline.named_steps["tfidf"]
    clf = pipeline.named_steps["clf"]
    x = tfidf.transform([text])
    if x.nnz == 0:
        return pd.DataFrame(columns=["term", "contribution"])
    classes = list(clf.classes_)
    if label is None:
        label = pipeline.predict([text])[0]
    ci = classes.index(label)
    if hasattr(clf, "coef_"):
        w = clf.coef_[ci]
    else:
        lp = clf.feature_log_prob_
        w = lp[ci] - lp.mean(axis=0)
    idx = x.indices
    names = np.asarray(tfidf.get_feature_names_out())[idx]
    df = pd.DataFrame({"term": names, "contribution": x.data * w[idx]})
    return df.sort_values("contribution", ascending=False).head(top_k).reset_index(drop=True)
