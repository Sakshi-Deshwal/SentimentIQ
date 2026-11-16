
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd

from src import config as C
from src.tasks import TaskSpec, get_task


class ModelNotFound(FileNotFoundError):
    """Raised with an actionable message when a model has not been trained yet."""


def train_command(task: str, key: str) -> str:
    return f"python scripts/train_models.py --models {key}"


def sklearn_path(task: TaskSpec, key: str) -> Path:
    return task.models_dir / f"{key}.joblib"


def nn_dir(task: TaskSpec, key: str) -> Path:
    return task.models_dir / key


def is_trained(task_key: str, key: str) -> bool:
    task = get_task(task_key)
    if key in C.NN_KEYS:
        return (nn_dir(task, key) / "model.pt").exists() and (nn_dir(task, key) / "meta.json").exists()
    return sklearn_path(task, key).exists()


def trained_models(task_key: str) -> list:
    """Models that are trained AND usable now (neural models are hidden when PyTorch cannot be imported)."""
    from src.compat import torch_available
    return [k for k in get_task(task_key).model_keys
            if is_trained(task_key, k) and (k not in C.NN_KEYS or torch_available())]


def neural_models_unavailable(task_key: str) -> list:
    """Trained neural models that cannot be used because PyTorch failed to import."""
    from src.compat import torch_available
    if torch_available():
        return []
    return [k for k in get_task(task_key).model_keys if k in C.NN_KEYS and is_trained(task_key, k)]


# ============================================================================ traditional ML
class SklearnPredictor:
    family = "Traditional ML"

    def __init__(self, key: str, pipeline, labels, meta: Optional[dict] = None):
        self.key, self.pipeline, self.labels, self.meta = key, pipeline, list(labels), meta or {}

    @property
    def supports_proba(self) -> bool:
        return hasattr(self.pipeline, "predict_proba")

    def predict_df(self, texts, batch_size: int = 5000) -> pd.DataFrame:
        from src.predict import predict_texts
        texts = pd.Series(list(texts), dtype="object").fillna("").astype(str)
        parts = [predict_texts(self.pipeline, texts.iloc[i:i + batch_size]) for i in range(0, len(texts), batch_size)]
        return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["prediction"])

    def save(self, path: Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.pipeline, path)


# ============================================================================ PyTorch
class TorchPredictor:
    family = "Neural network"
    supports_proba = True        # softmax over logits (an uncalibrated score, not a guarantee of correctness)

    def __init__(self, key: str, model, arch_config: dict, labels, featurizer: dict, meta: Optional[dict] = None):
        """featurizer: {"type": "tfidf", "vectorizer": TfidfVectorizer}
                     or {"type": "vocab", "vocab": Vocab, "max_len": int}"""
        self.key, self.model, self.arch_config = key, model, arch_config
        self.labels, self.featurizer, self.meta = list(labels), featurizer, meta or {}
        self.model.eval()

    # ---------------------------------------------------------------- features
    def _data(self, texts):
        from src.nn.trainer import TfidfData, encode_sequences
        texts = [("" if (t is None or (isinstance(t, float) and t != t)) else str(t)) for t in texts]
        if self.featurizer["type"] == "tfidf":
            return TfidfData(self.featurizer["vectorizer"].transform(texts))
        return encode_sequences(texts, self.featurizer["vocab"], self.featurizer["max_len"])

    def predict_proba(self, texts, batch_size: int = 512) -> np.ndarray:
        import torch
        from src.nn.trainer import get_device, predict_logits
        device = get_device()
        self.model.to(device)
        texts = list(texts)
        out = []
        for i in range(0, len(texts), 20000):                       # bound memory on huge files
            data = self._data(texts[i:i + 20000])
            logits = predict_logits(self.model, data, device, batch_size)
            out.append(torch.softmax(logits, dim=1).numpy())
        return np.vstack(out) if out else np.empty((0, len(self.labels)))

    def predict_df(self, texts, batch_size: int = 512) -> pd.DataFrame:
        proba = self.predict_proba(texts, batch_size)
        out = pd.DataFrame({"prediction": np.array(self.labels, dtype=object)[proba.argmax(1)] if len(proba) else []})
        for i, c in enumerate(self.labels):
            out[f"prob_{c}"] = proba[:, i]
        out["top_probability"] = proba.max(axis=1) if len(proba) else []
        return out

    # ---------------------------------------------------------------- persistence
    def save(self, directory: Path, extra_meta: Optional[dict] = None) -> None:
        import torch
        import sklearn
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), directory / "model.pt")
        if self.featurizer["type"] == "tfidf":
            joblib.dump(self.featurizer["vectorizer"], directory / "tfidf.joblib")
            feat_meta = {"type": "tfidf"}
        else:
            (directory / "vocab.json").write_text(json.dumps(self.featurizer["vocab"].to_dict()), encoding="utf-8")
            feat_meta = {"type": "vocab", "max_len": int(self.featurizer["max_len"]),
                         "vocab_size": len(self.featurizer["vocab"])}
        meta = {
            **self.meta, **(extra_meta or {}),
            "key": self.key, "labels": self.labels,
            "class_to_id": {l: i for i, l in enumerate(self.labels)},
            "arch_config": self.arch_config, "featurizer": feat_meta,
            "saved_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "torch_version": torch.__version__, "scikit_learn": sklearn.__version__,
        }
        (directory / "meta.json").write_text(json.dumps(meta, indent=2, default=float), encoding="utf-8")

    @classmethod
    def load(cls, directory: Path) -> "TorchPredictor":
        import torch
        from src.nn.architectures import build_model
        from src.text_tokens import Vocab
        directory = Path(directory)
        try:
            meta = json.loads((directory / "meta.json").read_text(encoding="utf-8"))
            model = build_model(meta["arch_config"])
            model.load_state_dict(torch.load(directory / "model.pt", map_location="cpu", weights_only=True))
            if meta["featurizer"]["type"] == "tfidf":
                feat = {"type": "tfidf", "vectorizer": joblib.load(directory / "tfidf.joblib")}
            else:
                vocab = Vocab.from_dict(json.loads((directory / "vocab.json").read_text(encoding="utf-8")))
                feat = {"type": "vocab", "vocab": vocab, "max_len": meta["featurizer"]["max_len"]}
        except FileNotFoundError as exc:
            raise ModelNotFound(f"Incomplete model folder {directory}: {exc}") from exc
        return cls(meta["key"], model, meta["arch_config"], meta["labels"], feat, meta)


# ============================================================================ loading
def read_model_meta(task_key: str, key: str) -> dict:
    """Training metadata for any model (never loads weights). Empty dict if unavailable."""
    task = get_task(task_key)
    try:
        if key in C.NN_KEYS:
            return json.loads((nn_dir(task, key) / "meta.json").read_text(encoding="utf-8"))
        card = json.loads((task.models_dir / "model_card.json").read_text(encoding="utf-8"))
        return {**card.get("models", {}).get(key, {}), "domains": card.get("domains"),
                "created_utc": card.get("created_utc")}
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def load_predictor(task_key: str, key: str):
    """Load a trained model of either family. Raises ModelNotFound with the command to train it."""
    task = get_task(task_key)
    if key not in task.model_keys:
        raise ValueError(f"Model '{key}' is not part of the {task.key} task.")
    if not is_trained(task_key, key):
        raise ModelNotFound(f"{C.MODEL_NAMES[key]} has not been trained for the {task.key} task. "
                            f"Run:  {train_command(task_key, key)}")
    if key in C.NN_KEYS:
        from src.compat import TORCH_HELP, torch_available
        if not torch_available():
            raise ModelNotFound(TORCH_HELP)
        return TorchPredictor.load(nn_dir(task, key))
    try:
        pipe = joblib.load(sklearn_path(task, key))
    except Exception as exc:
        raise RuntimeError(f"Could not load {sklearn_path(task, key).name}: {exc}. The scikit-learn version must match "
                           f"the one used for training (see requirements.txt); retrain with "
                           f"`{train_command(task_key, key)}`.") from exc
    return SklearnPredictor(key, pipe, task.labels, read_model_meta(task_key, key))


def time_inference(predictor, texts, repeats: int = 1) -> dict:
    """Wall-clock prediction time over ``texts`` (best of ``repeats``)."""
    texts = list(texts)
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        predictor.predict_df(texts)
        best = min(best, time.perf_counter() - t0)
    return {"n": len(texts), "seconds": round(best, 4), "ms_per_review": round(1000 * best / max(len(texts), 1), 4)}
