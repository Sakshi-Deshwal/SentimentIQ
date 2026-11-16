
from datetime import datetime, timezone
from typing import Callable, Optional

import pandas as pd

from src import config as C
from src import datasets as D
from src.inference import time_inference
from src.metrics import compute_metrics
from src.text_tokens import Vocab
from src.training import train_one

MODELS = ("naive_bayes", "logistic_regression", "linear_svm", "mlp", "textcnn", "bilstm")

STANDARD_EXPERIMENTS = [
    (["product_food"], "product_clothing"),      # train on Amazon, test on Clothing
    (["product_clothing"], "product_food"),      # train on Clothing, test on Amazon
]
POOLED_DOMAINS = ["product_food", "product_clothing"]


def experiment_id(train_domains, test_domain, model) -> str:
    return f"{'+'.join(sorted(train_domains))}->{test_domain}::{model}"


def pooled_id(model) -> str:
    return f"pooled::{model}"


def _stratified_cap(df: pd.DataFrame, cap: Optional[int], seed: int) -> pd.DataFrame:
    return D.stratified_sample(df, cap, seed) if cap and cap < len(df) else df.reset_index(drop=True)


def assert_no_overlap(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    """Leakage safeguard: no identical normalised text across train/validation/test, and no shared domain."""
    k = lambda df: set(df["review_text"].map(D.normalize_text_key))
    a, b, c = k(train_df), k(val_df), k(test_df)
    for name, inter in (("train/test", a & c), ("validation/test", b & c), ("train/validation", a & b)):
        if inter:
            raise AssertionError(f"{len(inter)} identical texts shared by {name}")
    shared = (set(train_df["domain"]) | set(val_df["domain"])) & set(test_df["domain"])
    if shared:
        raise AssertionError(f"held-out domain(s) {sorted(shared)} also present in training data")


def run_experiment(domain_data: dict, train_domains: list, test_domain: str, model_key: str, *,
                   seed: int = C.RANDOM_STATE, train_cap: Optional[int] = 20000, test_cap: Optional[int] = 10000,
                   nn_overrides: Optional[dict] = None, log: Callable = lambda *_: None) -> dict:
    """``domain_data``: {domain: {"train"|"validation"|"test": DataFrame}} from ThreeClassData.per_domain."""
    if test_domain in train_domains:
        raise ValueError("The test domain must not be one of the training domains.")
    missing = [d for d in [*train_domains, test_domain] if d not in domain_data]
    if missing:
        raise ValueError(f"Domains not available: {missing}")
    if model_key not in MODELS:
        raise ValueError(f"Unknown model '{model_key}'.")

    train_df = _stratified_cap(pd.concat([domain_data[d]["train"] for d in train_domains], ignore_index=True), train_cap, seed)
    val_df = pd.concat([domain_data[d]["validation"] for d in train_domains], ignore_index=True)
    held_out = pd.concat(domain_data[test_domain].values(), ignore_index=True)       # every record of the held-out domain
    test_df = _stratified_cap(held_out, test_cap, seed)
    assert_no_overlap(train_df, val_df, test_df)

    overrides = {"seed": seed, **(nn_overrides or {})}
    tm = train_one("three_class", model_key, train_df, val_df, tune=False, overrides=overrides, log=log)
    pred = tm.predictor.predict_df(test_df["review_text"])
    metrics = compute_metrics(test_df["sentiment"], pred["prediction"], C.THREE_LABELS)
    timing = time_inference(tm.predictor, test_df["review_text"].tolist())
    oov = Vocab.build(train_df["review_text"], min_freq=2, max_size=30000).oov_rate(test_df["review_text"])
    return {
        "id": experiment_id(train_domains, test_domain, model_key), "kind": "held_out_domain",
        "train_domains": list(train_domains), "test_domain": test_domain, "model": model_key,
        "n_train": int(len(train_df)), "n_validation": int(len(val_df)), "n_test": int(len(test_df)),
        "test_class_counts": {l: int((test_df["sentiment"] == l).sum()) for l in C.THREE_LABELS},
        "metrics": metrics, "train_seconds": tm.meta.get("train_seconds"), "inference": timing,
        "test_oov_rate": round(float(oov), 4),
        "hyperparameters": "fixed in advance, not tuned on the held-out domain",
        "train_cap": train_cap, "test_cap": test_cap, "seed": seed,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def run_in_domain_baseline(domain_data: dict, domain: str, model_key: str, *, seed: int = C.RANDOM_STATE,
                           train_cap: Optional[int] = 20000, nn_overrides: Optional[dict] = None,
                           log: Callable = lambda *_: None) -> dict:
    """Ordinary random-split reference: train and test inside ONE domain (its own splits)."""
    sp = domain_data[domain]
    train_df = _stratified_cap(sp["train"], train_cap, seed)
    train_keys = set(train_df["review_text"].map(D.normalize_text_key))
    if train_keys & set(sp["test"]["review_text"].map(D.normalize_text_key)):
        raise AssertionError("in-domain baseline: train and test share identical texts")
    tm = train_one("three_class", model_key, train_df, sp["validation"], tune=False,
                   overrides={"seed": seed, **(nn_overrides or {})}, log=log)
    pred = tm.predictor.predict_df(sp["test"]["review_text"])
    metrics = compute_metrics(sp["test"]["sentiment"], pred["prediction"], C.THREE_LABELS)
    return {"id": f"{domain}->{domain}::{model_key}", "kind": "in_domain_baseline", "train_domains": [domain],
            "test_domain": domain, "model": model_key, "n_train": int(len(train_df)),
            "n_validation": int(len(sp["validation"])), "n_test": int(len(sp["test"])),
            "metrics": metrics, "train_seconds": tm.meta.get("train_seconds"),
            "note": "random split inside one domain; NOT a cross-domain result", "seed": seed,
            "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}


def run_pooled_experiment(domain_data: dict, model_key: str, *, seed: int = C.RANDOM_STATE,
                          train_cap: Optional[int] = 20000, nn_overrides: Optional[dict] = None,
                          log: Callable = lambda *_: None) -> dict:
    """Train on the pooled TRAIN splits of every available domain, early-stop on the pooled VALIDATION
    splits and score on each domain's held-out TEST split (and on the pooled test set)."""
    domains = [d for d in POOLED_DOMAINS if d in domain_data]
    if len(domains) < 2:
        raise ValueError("The pooled experiment needs both datasets.")
    train_df = _stratified_cap(pd.concat([domain_data[d]["train"] for d in domains], ignore_index=True), train_cap, seed)
    val_df = pd.concat([domain_data[d]["validation"] for d in domains], ignore_index=True)
    test_df = pd.concat([domain_data[d]["test"] for d in domains], ignore_index=True)
    assert_no_overlap(train_df, val_df, test_df.assign(domain="__test__"))
    tm = train_one("three_class", model_key, train_df, val_df, tune=False,
                   overrides={"seed": seed, **(nn_overrides or {})}, log=log)
    pred = tm.predictor.predict_df(test_df["review_text"])
    out = {"id": pooled_id(model_key), "kind": "pooled_test", "train_domains": domains, "test_domain": "pooled test split",
           "model": model_key, "n_train": int(len(train_df)), "n_validation": int(len(val_df)), "n_test": int(len(test_df)),
           "metrics": compute_metrics(test_df["sentiment"], pred["prediction"], C.THREE_LABELS),
           "by_domain": {}, "train_seconds": tm.meta.get("train_seconds"),
           "note": "Trained on both datasets; scored on their held-out test splits. Not a transfer test.",
           "hyperparameters": "fixed in advance", "train_cap": train_cap, "seed": seed,
           "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    for d in domains:
        m = (test_df["domain"] == d).to_numpy()
        out["by_domain"][d] = {"n": int(m.sum()), **compute_metrics(test_df.loc[m, "sentiment"], pred.loc[m, "prediction"], C.THREE_LABELS)}
    return out
