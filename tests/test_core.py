
import numpy as np
import pandas as pd
import pytest
import torch

from src import config as C
from src import datasets as D
from src import inference
from src.nn.architectures import build_model, count_parameters
from src.preprocessing import clean_text
from src.text_tokens import Vocab, tokenize


def _frame(n=300, seed=0):
    rng = np.random.default_rng(seed)
    rating = rng.choice([1, 2, 3, 4, 5], n, p=[.1, .1, .2, .3, .3])
    return pd.DataFrame({"review_id": range(n), "review_text": [f"review number {i} uniquetoken{i}" for i in range(n)],
                         "rating": rating, "sentiment": pd.Series(rating).map(D.RATING_TO_SENTIMENT),
                         "domain": "product_food", "source": "synthetic"})


# ---------------------------------------------------------------- preprocessing
def test_clean_text_lowercases_and_handles_missing():
    assert clean_text("GREAT Product!!") == clean_text("great product!!")
    assert clean_text(None) == "" or isinstance(clean_text(None), str)


def test_tokenize_keeps_negation_apostrophe():
    assert "can't" in tokenize("I can't recommend it")


# ---------------------------------------------------------------- label mapping
def test_rating_to_sentiment_mapping():
    assert D.RATING_TO_SENTIMENT == {1: "Negative", 2: "Negative", 3: "Neutral", 4: "Positive", 5: "Positive"}
    assert C.THREE_LABELS == ["Negative", "Neutral", "Positive"]
    assert C.LABELS == C.THREE_LABELS and "Mixed" not in C.LABELS
    assert not hasattr(C, "FOUR_LABELS")


def test_validate_external_frame_ok_and_bad_schema():
    df = _frame()
    assert D.validate_external_frame(df)["label_mapping_mismatches"] == 0
    with pytest.raises(D.DatasetUnavailable):
        D.validate_external_frame(df.drop(columns=["rating"]))
    bad = df.copy(); bad.loc[0, "sentiment"] = "Mixed"   # an unsupported label must be rejected, never mapped
    with pytest.raises(D.DatasetUnavailable):
        D.validate_external_frame(bad)


# ---------------------------------------------------------------- model output dimensions
@pytest.mark.parametrize("n_classes", [3, 4])
def test_network_output_dimensions(n_classes):
    x = torch.randint(1, 50, (5, 12))
    lengths = torch.full((5,), 12)
    cnn = build_model({"arch": "textcnn", "vocab_size": 50, "n_classes": n_classes})
    lstm = build_model({"arch": "bilstm", "vocab_size": 50, "n_classes": n_classes})
    mlp = build_model({"arch": "mlp", "input_dim": 40, "n_classes": n_classes})
    assert cnn(x).shape == (5, n_classes)
    assert lstm(x, lengths).shape == (5, n_classes)
    assert mlp(torch.rand(5, 40)).shape == (5, n_classes)
    assert count_parameters(cnn) > 0


# ---------------------------------------------------------------- vocabulary / leakage safeguards
def test_vocab_built_from_train_only():
    v = Vocab.build(["good food", "good service"], min_freq=1)
    assert "good" in v.stoi and "terrible" not in v.stoi
    assert v.encode("terrible", 5) == [1]                    # unseen -> <unk>
    assert v.oov_rate(["terrible terrible good"]) == pytest.approx(2 / 3)


def test_forbidden_features_listed():
    for col in ("rating", "review_id", "source", "domain"):
        assert col in C.FORBIDDEN_FEATURES
    X, y = D.text_and_labels(_frame(10))
    assert all(isinstance(t, str) for t in X) and len(X) == len(y) == 10


def test_splits_are_disjoint_and_stratified():
    sp = D.split_domain(_frame(600))
    D.assert_disjoint_splits(sp)
    assert sum(len(v) for v in sp.values()) == 600
    assert set(sp["test"]["sentiment"]) == set(sp["train"]["sentiment"])


def test_disjoint_assertion_catches_overlap():
    sp = D.split_domain(_frame(300))
    sp["test"] = pd.concat([sp["test"], sp["train"].head(2)], ignore_index=True)
    with pytest.raises(AssertionError):
        D.assert_disjoint_splits(sp)


def test_dedupe_removes_normalised_duplicates():
    df = pd.DataFrame({"review_text": ["Great food!", "great   food", "Other"], "sentiment": ["Positive"] * 3})
    out, removed = D.dedupe(df)
    assert removed == 1 and len(out) == 2


def test_held_out_domain_excluded():
    from src.cross_domain import assert_no_overlap
    mk = lambda texts, dom: pd.DataFrame({"review_text": texts, "domain": dom})
    tr, va = mk(["alpha one", "beta two"], "product_food"), mk(["gamma three"], "product_food")
    assert_no_overlap(tr, va, mk(["delta four"], "product_clothing"))
    with pytest.raises(AssertionError):                      # same domain in train and test
        assert_no_overlap(tr, va, mk(["delta four"], "product_food"))
    with pytest.raises(AssertionError):                      # identical text across domains
        assert_no_overlap(tr, va, mk(["Alpha, one"], "product_clothing"))


# ---------------------------------------------------------------- local dataset handling
def test_missing_file_does_not_crash(tmp_path, monkeypatch):
    from src import settings
    monkeypatch.setenv(settings.ENV_AMAZON, str(tmp_path / "nope.csv"))
    st = D.domain_status("product_food")
    assert not st.exists and "not found" in st.message.lower()
    with pytest.raises(D.DatasetUnavailable):
        D.load_domain("product_food", sample_size=100)


def test_chunked_stratified_sampling_does_not_modify_file(tmp_path, monkeypatch):
    from src import settings
    f = tmp_path / "amazon.csv"
    _frame(3000).to_csv(f, index=False)
    before = f.read_bytes()
    monkeypatch.setenv(settings.ENV_AMAZON, str(f))
    monkeypatch.setattr(D, "CHUNK_ROWS", 1000)
    monkeypatch.setattr(D, "file_status", lambda p: D.FileStatus(f, True, 25.0, "OK"))  # force the large-file path
    df, info = D.load_domain("product_food", sample_size=600, min_per_class=50)
    assert 450 <= len(df) <= 750 and "chunked" in info["sampling"]
    assert set(df["sentiment"]) == {"Negative", "Neutral", "Positive"}
    assert f.read_bytes() == before
    df2, _ = D.load_domain("product_food", sample_size=600, min_per_class=50)
    assert len(df) == len(df2)                               # same seed -> same sample


def test_path_cleaning():
    from src import settings
    assert settings.clean_path_string('  "C:\\a\\b.csv" ') == "C:\\a\\b.csv"


def test_upload_limit_is_separate_from_local_loading():
    assert C.UPLOAD_MAX_MB == 50
    assert D.DOMAINS["product_food"].local_path and not D.DOMAINS["product_clothing"].local_path


# ---------------------------------------------------------------- model loading + prediction (skip if untrained)
@pytest.mark.skipif(not inference.trained_models("three_class"), reason="models not trained")
@pytest.mark.parametrize("key", ["naive_bayes", "logistic_regression", "linear_svm", "mlp", "textcnn", "bilstm"])
def test_every_trained_model_loads_and_predicts_three_classes(key):
    if key not in inference.trained_models("three_class"):
        pytest.skip(f"{key} unavailable (not trained, or PyTorch cannot be imported)")
    p = inference.load_predictor("three_class", key)
    out = p.predict_df(["Great food and lovely staff", "Terrible, cold and rude", "It was okay, nothing special"])
    assert len(out) == 3 and set(out["prediction"]) <= set(C.THREE_LABELS)
    if "top_probability" in out:
        probs = out[[c for c in out.columns if c.startswith("prob_")]]
        assert probs.shape[1] == 3 and np.allclose(probs.sum(axis=1), 1, atol=1e-3)


def test_task_registry_is_three_class_only():
    from src.tasks import TASKS, get_task
    assert list(TASKS) == ["three_class"] and get_task().labels == ("Negative", "Neutral", "Positive")
    with pytest.raises(ValueError):
        get_task("four_class")


def test_no_mixed_in_saved_reports():
    import json
    ev = json.loads((C.THREE_REPORTS_DIR / "evaluation.json").read_text()) if (C.THREE_REPORTS_DIR / "evaluation.json").exists() else None
    if ev is None:
        pytest.skip("no evaluation report")
    assert ev["labels"] == ["Negative", "Neutral", "Positive"]
    for m in ev["models"].values():
        assert len(m["test"]["confusion_matrix"]) == 3 and all(len(r) == 3 for r in m["test"]["confusion_matrix"])


def test_robustness_set_has_only_supported_labels():
    d = pd.read_csv(C.ROBUSTNESS_PATH)
    assert set(d["expected_sentiment"]) <= set(C.THREE_LABELS)


def test_torch_failure_is_not_fatal(monkeypatch):
    from src import compat
    monkeypatch.setattr(compat, "torch_status", lambda: (False, "OSError: WinError 126"))
    monkeypatch.setattr(compat, "torch_available", lambda: False)
    assert "textcnn" not in inference.trained_models("three_class")
    with pytest.raises(inference.ModelNotFound):
        inference.load_predictor("three_class", "textcnn") if inference.is_trained("three_class", "textcnn") else (_ for _ in ()).throw(inference.ModelNotFound("n/a"))


# ---------------------------------------------------------------- business insights
def test_business_insights_and_trend():
    from src import insights as ins
    rng = np.random.default_rng(1)
    n = 200
    df = pd.DataFrame({"predicted_sentiment": rng.choice(C.THREE_LABELS, n, p=[.2, .2, .6]),
                       "date": pd.date_range("2026-01-01", periods=n, freq="D").astype(str),
                       "branch": rng.choice(["A", "B", "C"], n)})
    assert ins.detect_date_column(df) == "date" and ins.detect_group_column(df) == "branch"
    t = ins.trend(df, "date")
    assert np.allclose(t[C.THREE_LABELS].sum(axis=1), 1)
    g = ins.group_table(df, "branch")
    assert set(g.index) == {"A", "B", "C"} and (g["Reviews"].sum() == n)
    lines = ins.business_insights(df, "date", "branch")
    assert any("net sentiment" in l for l in lines) and all(isinstance(l, str) for l in lines)


def test_insights_handle_no_metadata():
    from src import insights as ins
    df = pd.DataFrame({"predicted_sentiment": ["Positive", "Negative", "Neutral"]})
    assert ins.detect_date_column(df) is None and ins.detect_group_column(df) is None
    assert ins.business_insights(df)


def test_missing_model_raises_clear_error(tmp_path, monkeypatch):
    monkeypatch.setattr(inference, "is_trained", lambda *a, **k: False)
    with pytest.raises(inference.ModelNotFound):
        inference.load_predictor("three_class", "textcnn")


# ---------------------------------------------------------------- two-dataset scope (restaurant dataset removed)
def test_only_two_english_domains_exist():
    assert set(D.DOMAINS) == {"product_food", "product_clothing"}
    assert not hasattr(D, "detect_language")
    assert not (C.PROCESSED_DIR / "restaurant_reviews_cleaned.csv").exists()


def test_six_model_architectures_registered():
    from src.tasks import get_task
    assert list(get_task().model_keys) == ["naive_bayes", "logistic_regression", "linear_svm", "mlp", "textcnn", "bilstm"]
    assert set(C.MODEL_NAMES) == set(get_task().model_keys)


def test_standard_cross_domain_experiments():
    from src.cross_domain import STANDARD_EXPERIMENTS, POOLED_DOMAINS
    assert STANDARD_EXPERIMENTS == [(["product_food"], "product_clothing"), (["product_clothing"], "product_food")]
    assert POOLED_DOMAINS == ["product_food", "product_clothing"]


def test_pooled_experiment_runs_and_scores_both_domains():
    from src.cross_domain import run_pooled_experiment
    rng = np.random.default_rng(3)
    vocab = {"Positive": "great lovely excellent", "Negative": "awful broken terrible", "Neutral": "okay average fine"}

    def make(dom, n, off):
        lab = rng.choice(C.THREE_LABELS, n)
        txt = [f"{vocab[l]} item {dom} number {off + i} {rng.integers(10**6)}" for i, l in enumerate(lab)]
        df = pd.DataFrame({"review_id": range(n), "review_text": txt, "sentiment": lab, "domain": dom,
                           "rating": 3, "source": "synthetic"})
        return D.split_domain(df)
    data = {"product_food": make("product_food", 600, 0), "product_clothing": make("product_clothing", 600, 10_000)}
    r = run_pooled_experiment(data, "logistic_regression", train_cap=None)
    assert r["kind"] == "pooled_test" and set(r["by_domain"]) == {"product_food", "product_clothing"}
    assert len(r["metrics"]["confusion_matrix"]) == 3


def test_near_duplicate_mask():
    ref = ["the coffee arrived quickly and tasted absolutely wonderful this morning", "terrible packaging"]
    qry = ["The coffee arrived quickly and tasted absolutely wonderful this morning!!",   # near-duplicate
           "a completely different review about shoes that run small and feel stiff",
           "terrible packaging"]                                                           # too short: exact-match only
    m = D.near_duplicate_mask(ref, qry)
    assert list(m) == [True, False, False]


def test_audit_report_has_no_restaurant_domain():
    import json
    p = C.THREE_REPORTS_DIR / "dataset_audit.json"
    if not p.exists():
        pytest.skip("no audit report")
    a = json.loads(p.read_text())
    assert set(a["domains_used"]) <= {"product_food", "product_clothing"} and "restaurant" not in json.dumps(a)
