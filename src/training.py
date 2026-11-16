
import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, Optional

import numpy as np
import pandas as pd
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV, PredefinedSplit

from src import config as C
from src.inference import SklearnPredictor, TorchPredictor, nn_dir, sklearn_path
from src.models import build_pipeline
from src.preprocessing import clean_text
from src.tasks import get_task
from src.text_tokens import Vocab

# ----------------------------------------------------------------------------- defaults
# Fixed, documented starting points. They are NOT tuned on any held-out domain.
SKLEARN_FIXED = {
    "naive_bayes": {"clf__alpha": 0.1, "clf__fit_prior": False, "tfidf__ngram_range": (1, 2), "tfidf__min_df": 2},
    "logistic_regression": {"clf__C": 2.0, "tfidf__ngram_range": (1, 2), "tfidf__min_df": 2},
    "linear_svm": {"clf__C": 0.5, "tfidf__ngram_range": (1, 2), "tfidf__min_df": 2},
}
# Small grids used when tuning on the (pooled) validation split of the three-class task.
SKLEARN_GRID_3C = {
    "naive_bayes": {"clf__alpha": [0.03, 0.1, 0.3, 1.0], "clf__fit_prior": [True, False],
                    "tfidf__ngram_range": [(1, 1), (1, 2)], "tfidf__min_df": [2]},
    "logistic_regression": {"clf__C": [0.5, 2.0, 8.0], "tfidf__ngram_range": [(1, 1), (1, 2)], "tfidf__min_df": [2]},
    "linear_svm": {"clf__C": [0.1, 0.5, 2.0], "tfidf__ngram_range": [(1, 1), (1, 2)], "tfidf__min_df": [2]},
}


def nn_defaults(task_key: str, key: str) -> dict:
    """Default configuration for a neural model on a task (overridable from the command line)."""
    from src.nn.trainer import TrainConfig
    three = task_key == "three_class"
    max_len = 160 if three else 100
    train = TrainConfig(epochs=20, batch_size=128 if three else 64, patience=3, use_class_weights=three,
                        lr=2e-3 if key == "bilstm" else 1e-3)
    if key == "mlp":
        return {"train": train, "tfidf": {"max_features": 15000, "ngram_range": (1, 2), "min_df": 2},
                "arch": {"hidden1": 256, "hidden2": 128, "dropout1": 0.3, "dropout2": 0.2}}
    if key == "textcnn":
        return {"train": train, "max_len": max_len, "vocab": {"min_freq": 2, "max_size": 30000},
                "arch": {"embed_dim": 128, "num_filters": 100, "kernel_sizes": [3, 4, 5], "dropout": 0.5}}
    if key == "bilstm":
        return {"train": train, "max_len": max_len, "vocab": {"min_freq": 2, "max_size": 30000},
                "arch": {"embed_dim": 128, "hidden_dim": 96, "num_layers": 1, "dropout": 0.4}}
    raise ValueError(f"No neural defaults for '{key}'")


@dataclass
class TrainedModel:
    key: str
    task: str
    predictor: object
    meta: dict = field(default_factory=dict)


def text_and_labels(df: pd.DataFrame) -> tuple:
    """The ONLY model inputs/targets taken from a frame: review text and sentiment."""
    return df[C.TEXT_COL].astype(str).tolist(), df[C.TARGET_COL].tolist()


# ----------------------------------------------------------------------------- traditional ML
def train_sklearn(task_key: str, key: str, train_df: pd.DataFrame, val_df: pd.DataFrame,
                  tune: bool = False, grid: Optional[dict] = None, params: Optional[dict] = None,
                  log: Callable = print) -> TrainedModel:
    """Fit TF-IDF + classifier on the TRAIN frame only.

    tune=True : grid search, fit on train and score macro-F1 on validation (PredefinedSplit).
    tune=False: fixed ``params`` (default: SKLEARN_FIXED) - used for held-out-domain experiments.
    """
    task = get_task(task_key)
    X_tr, y_tr = text_and_labels(train_df)
    X_va, y_va = text_and_labels(val_df)
    base = build_pipeline(key)
    if task.weak_labels and key in ("logistic_regression", "linear_svm"):
        base.set_params(clf__class_weight="balanced")        # external data is heavily imbalanced
    t_total = time.perf_counter()
    tuning = None
    if tune:
        grid = grid or SKLEARN_GRID_3C[key]
        fold = np.r_[np.full(len(X_tr), -1), np.zeros(len(X_va), dtype=int)]
        search = GridSearchCV(base, grid, scoring="f1_macro", cv=PredefinedSplit(fold), refit=False, n_jobs=1,
                              error_score="raise")
        t0 = time.perf_counter()
        search.fit(pd.Series(X_tr + X_va), pd.Series(y_tr + y_va))
        tuning = {"seconds": round(time.perf_counter() - t0, 1), "best_val_macro_f1": float(search.best_score_),
                  "n_configs": len(search.cv_results_["params"])}
        params = search.best_params_
        log(f"  tuned {len(search.cv_results_['params'])} configs: best val macro-F1 {search.best_score_:.4f} {params}")
    else:
        params = params or SKLEARN_FIXED.get(key, {})
    pipe = base.set_params(**params)
    t0 = time.perf_counter()
    pipe.fit(X_tr, y_tr)                                            # train only: no refit on validation
    fit_seconds = round(time.perf_counter() - t0, 2)
    meta = {"key": key, "family": "Traditional ML", "best_params": _jsonable(params), "fit_seconds": fit_seconds,
            "train_seconds": round(time.perf_counter() - t_total, 2), "tuning": tuning,
            "n_train": len(X_tr), "n_val": len(X_va), "fit_mode": "train_only",
            "class_weight": "balanced" if task.weak_labels else None}
    return TrainedModel(key, task_key, SklearnPredictor(key, pipe, task.labels, meta), meta)


# ----------------------------------------------------------------------------- neural networks
def train_nn(task_key: str, key: str, train_df: pd.DataFrame, val_df: pd.DataFrame,
             overrides: Optional[dict] = None, checkpoint_dir=None, log: Callable = print) -> TrainedModel:
    """Train MLP / TextCNN / BiLSTM. Vocabulary or TF-IDF is fitted on ``train_df`` only."""
    import torch
    from src.nn.architectures import build_model, count_parameters
    from src.nn.trainer import TfidfData, TrainConfig, encode_sequences, fit, get_device
    task = get_task(task_key)
    d = nn_defaults(task_key, key)
    cfg: TrainConfig = d["train"]
    ov = dict(overrides or {})
    for k in ("epochs", "batch_size", "lr", "patience", "seed", "weight_decay"):
        if ov.get(k) is not None:
            setattr(cfg, k, ov[k])
    max_len = ov.get("max_len") or d.get("max_len")

    X_tr, y_tr_lab = text_and_labels(train_df)
    X_va, y_va_lab = text_and_labels(val_df)
    l2i = task.label_to_id
    y_tr = np.array([l2i[l] for l in y_tr_lab])
    y_va = np.array([l2i[l] for l in y_va_lab])

    if key == "mlp":
        tf = d["tfidf"]
        vec = TfidfVectorizer(preprocessor=clean_text, token_pattern=r"(?u)\b\w[\w']*\b", sublinear_tf=True,
                              max_features=ov.get("max_features") or tf["max_features"],
                              ngram_range=tf["ngram_range"], min_df=tf["min_df"], dtype=np.float32)
        Xtr = vec.fit_transform(X_tr)                               # fit on TRAIN only
        train_data = TfidfData(Xtr, torch.as_tensor(y_tr))
        val_data = TfidfData(vec.transform(X_va), torch.as_tensor(y_va))
        arch = {"arch": "mlp", "input_dim": int(Xtr.shape[1]), "n_classes": task.n_classes, **d["arch"]}
        feat = {"type": "tfidf", "vectorizer": vec}
        feat_info = {"tfidf_features": int(Xtr.shape[1])}
    else:
        vocab = Vocab.build(X_tr, **d["vocab"])                     # TRAIN only
        train_data = encode_sequences(X_tr, vocab, max_len, y_tr)
        val_data = encode_sequences(X_va, vocab, max_len, y_va)
        arch = {"arch": key, "vocab_size": len(vocab), "n_classes": task.n_classes, **d["arch"]}
        feat = {"type": "vocab", "vocab": vocab, "max_len": max_len}
        feat_info = {"vocab_size": len(vocab), "max_len": max_len}

    torch.manual_seed(cfg.seed)                                     # identical init for a given seed
    model = build_model(arch)
    log(f"  {key}: {count_parameters(model):,} parameters, device={get_device()}, "
        f"train={len(train_data):,} val={len(val_data):,}")
    ckpt = None if checkpoint_dir is None else nn_dir(task, key) / "checkpoint.pt"
    result = fit(model, train_data, val_data, cfg, task.n_classes, checkpoint_path=ckpt, log=log)
    meta = {"family": "Neural network", **result, "train_config": cfg.to_dict(), "parameters": count_parameters(model),
            **feat_info, "n_train": len(X_tr), "n_val": len(X_va), "fit_mode": "train_only",
            "class_weighted_loss": cfg.use_class_weights}
    pred = TorchPredictor(key, model, arch, task.labels, feat, meta)
    return TrainedModel(key, task_key, pred, meta)


def train_one(task_key: str, key: str, train_df: pd.DataFrame, val_df: pd.DataFrame, *, tune: bool = False,
              overrides: Optional[dict] = None, params: Optional[dict] = None, checkpoint_dir=None,
              log: Callable = print) -> TrainedModel:
    if key in C.NN_KEYS:
        return train_nn(task_key, key, train_df, val_df, overrides, checkpoint_dir, log)
    return train_sklearn(task_key, key, train_df, val_df, tune=tune, params=params, log=log)


# ----------------------------------------------------------------------------- saving
def _jsonable(obj):
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    return obj


def save_trained(tm: TrainedModel, extra_meta: Optional[dict] = None, header: Optional[dict] = None) -> None:
    """Persist a trained model under models/<task>/ and register it in that task's model_card.json."""
    task = get_task(tm.task)
    task.models_dir.mkdir(parents=True, exist_ok=True)
    if tm.key in C.NN_KEYS:
        tm.predictor.save(nn_dir(task, tm.key), {**(extra_meta or {}), "task": tm.task})
        entry = {k: v for k, v in {**tm.meta, **(extra_meta or {})}.items() if k not in ("history",)}
    else:
        tm.predictor.save(sklearn_path(task, tm.key))
        entry = {**tm.meta, **(extra_meta or {})}
    update_model_card(tm.task, tm.key, entry, header)


def update_model_card(task_key: str, key: str, entry: dict, header: Optional[dict] = None) -> None:
    task = get_task(task_key)
    path = task.models_dir / "model_card.json"
    try:
        card = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        card = {}
    card.setdefault("models", {})
    from src.compat import torch_status
    ok, msg = torch_status()
    if ok:
        card["torch"] = msg.replace("PyTorch ", "")
    card.update({"task": task_key, "labels": list(task.labels), "scikit_learn": sklearn.__version__,
                 "random_state": C.RANDOM_STATE,
                 "model_input": "review_text only", **(header or {})})
    card["updated_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    card["models"][key] = {"name": C.MODEL_NAMES[key], **_jsonable(entry)}
    path.write_text(json.dumps(card, indent=2, default=str), encoding="utf-8")
