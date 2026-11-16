"""Shared Streamlit helpers: cached loaders, model picker, safe model loading.

Nothing in the app trains a model. Models are loaded from disk (cached per file modification
time, so retraining on disk is picked up automatically) and reports are read from ``reports/``.
"""
import json
from pathlib import Path
from typing import Optional

import pandas as pd
import streamlit as st

from src import config as C
from src import datasets as D
from src import inference as I
from src import settings
from src.tasks import TASK, TASKS, TaskSpec, get_task


# ----------------------------------------------------------------------------- file readers
def _mtime(path: Path) -> float:
    try:
        return Path(path).stat().st_mtime
    except OSError:
        return 0.0


@st.cache_data(show_spinner=False)
def _read_json(path_str: str, mtime: float):
    return json.loads(Path(path_str).read_text(encoding="utf-8"))


def read_json(path: Path):
    return _read_json(str(path), _mtime(path)) if Path(path).exists() else None


@st.cache_data(show_spinner=False)
def _read_csv(path_str: str, mtime: float) -> pd.DataFrame:
    return pd.read_csv(path_str)


def read_csv(path: Path):
    return _read_csv(str(path), _mtime(path)) if Path(path).exists() else None


def evaluation(task_key: str):
    return read_json(get_task(task_key).evaluation_path)


def predictions(task_key: str):
    return read_csv(get_task(task_key).predictions_path)


def model_card(task_key: str):
    return read_json(get_task(task_key).models_dir / "model_card.json")


def dataset_audit():
    return read_json(TASKS["three_class"].reports_dir / "dataset_audit.json")


def cross_domain_results():
    return read_json(TASKS["three_class"].reports_dir / "cross_domain.json")


# ----------------------------------------------------------------------------- models
def _signature(task_key: str, key: str) -> float:
    task = get_task(task_key)
    p = I.nn_dir(task, key) / "model.pt" if key in C.NN_KEYS else I.sklearn_path(task, key)
    return _mtime(p)


@st.cache_resource(show_spinner="Loading model...")
def _load_predictor(task_key: str, key: str, signature: float):
    return I.load_predictor(task_key, key)


def get_predictor(task_key: str, key: str):
    """Cached predictor. Raises I.ModelNotFound / RuntimeError with an actionable message."""
    return _load_predictor(task_key, key, _signature(task_key, key))


def best_model_key(task_key: str) -> Optional[str]:
    ev = evaluation(task_key)
    return ev.get("best_model_key_by_validation") if ev else None


def task(page: str = "") -> TaskSpec:
    """The single supported task (three-class sentiment). ``page`` is accepted for call-site symmetry."""
    return TASK


def torch_notice(task_key: str = "three_class") -> None:
    """Explain (once, on the page) why neural models are missing when PyTorch cannot be imported."""
    from src.compat import TORCH_HELP, torch_status
    hidden = I.neural_models_unavailable(task_key)
    if hidden:
        st.warning(f"{TORCH_HELP}  ({torch_status()[1]}). Hidden models: "
                   + ", ".join(C.MODEL_NAMES[k] for k in hidden) + ".", icon="⚠️")


def model_picker(task_key: str, page: str, label: str = "Model") -> Optional[str]:
    """Selectbox of TRAINED models; shows exactly what to run when none exist."""
    trained = I.trained_models(task_key)
    torch_notice(task_key)
    if not trained:
        st.error("No usable trained models yet.")
        st.code("python scripts/train_models.py", language="bash")
        return None
    best = best_model_key(task_key)
    idx = trained.index(best) if best in trained else 0
    return st.selectbox(label, trained, index=idx, key=f"model_{page}_{task_key}",
                        format_func=lambda k: f"{C.MODEL_NAMES[k]}  ·  {C.MODEL_FAMILY[k]}"
                                              + ("  ★ best on validation" if k == best else ""))


def require_evaluation(task_key: str):
    ev = evaluation(task_key)
    if ev is None:
        st.error("No evaluation report yet. Train, then evaluate:")
        st.code("python scripts/train_models.py\npython scripts/evaluate.py", language="bash")
        st.stop()
    return ev


def go_to(page: str) -> None:
    st.session_state["nav"] = page


# ----------------------------------------------------------------------------- datasets
@st.cache_data(show_spinner="Counting reviews (large files are read in chunks)...")
def _domain_overview(amazon_path: str, amazon_mtime: float) -> dict:
    out = {}
    for dom, spec in D.DOMAINS.items():
        status = D.domain_status(dom)
        row = {"title": spec.title, "path": str(status.path),
               "available": status.exists, "message": status.message, "file_mb": round(status.size_mb, 1)}
        if status.exists:
            try:
                if spec.local_path:
                    s = D.amazon_stats(status.path)
                    row.update(rows=s["rows"], sentiment=s["sentiment"], rating=s["rating"])
                else:
                    df = pd.read_csv(status.path, usecols=["sentiment", "rating"])
                    row.update(rows=len(df), sentiment=df["sentiment"].value_counts().to_dict(),
                               rating={int(k): int(v) for k, v in df["rating"].value_counts().items()})
            except Exception as exc:               # unreadable file: report, never crash
                row.update(available=False, message=f"Could not read {status.path.name}: {exc}")
        out[dom] = row
    return out


def domain_overview() -> dict:
    p = settings.get_amazon_path()
    return _domain_overview(str(p), _mtime(p))


@st.cache_resource(show_spinner="Preparing three-class data (loading, de-duplicating, splitting)...")
def _three_class_data(sizes_key: tuple, amazon_path: str, amazon_mtime: float):
    from src.three_class import build_three_class_data
    return build_three_class_data(sample_sizes=dict(sizes_key), log=lambda *_: None)


def get_three_class_data():
    """Cached domain splits (Amazon read from its local path with the configured stratified sample)."""
    p = settings.get_amazon_path()
    sizes = tuple(sorted(settings.get_sample_sizes().items()))
    return _three_class_data(sizes, str(p), _mtime(p))
