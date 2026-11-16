
import json
import os
from pathlib import Path

from src.config import ROOT

DEFAULT_AMAZON_PATH = r"C:\Users\HP\Downloads\amazon_fine_food_reviews_cleaned.csv"
LOCAL_PATHS_FILE = ROOT / "local_paths.json"
ENV_AMAZON = "SENTIMENTIQ_AMAZON_PATH"


def clean_path_string(raw: str) -> str:
    """Strip whitespace and the quotes Windows 'Copy as path' adds."""
    return (raw or "").strip().strip('"').strip("'").strip()


def _read_local_file() -> dict:
    try:
        return json.loads(LOCAL_PATHS_FILE.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}


def get_amazon_path_string() -> str:
    env = clean_path_string(os.environ.get(ENV_AMAZON, ""))
    if env:
        return env
    saved = clean_path_string(_read_local_file().get("amazon_path", ""))
    return saved or DEFAULT_AMAZON_PATH


def get_amazon_path() -> Path:
    return Path(os.path.expandvars(os.path.expanduser(get_amazon_path_string())))


def save_amazon_path(path_string: str) -> Path:
    """Persist the path (only the path string, never the data) to local_paths.json."""
    data = _read_local_file()
    data["amazon_path"] = clean_path_string(path_string)
    LOCAL_PATHS_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return LOCAL_PATHS_FILE


# --------------------------------------------------------------------------- training sample sizes
# Rows kept per domain BEFORE splitting (stratified by sentiment). None = keep every row.
# The Amazon file is very large; a 30k sample keeps CPU training practical. Raise it for a
# stronger model: edit ``sample_sizes`` in local_paths.json, set SENTIMENTIQ_SAMPLE_PRODUCT_FOOD,
# or pass --sample product_food=100000 to the training scripts.
DEFAULT_SAMPLE_SIZES = {"product_food": 30000, "product_clothing": None}


def get_sample_sizes() -> dict:
    sizes = dict(DEFAULT_SAMPLE_SIZES)
    for k, v in (_read_local_file().get("sample_sizes") or {}).items():
        if k in sizes:
            sizes[k] = int(v) if v else None
    for k in sizes:
        env = os.environ.get(f"SENTIMENTIQ_SAMPLE_{k.upper()}", "").strip()
        if env:
            sizes[k] = int(env) if int(env) > 0 else None
    return sizes


def save_sample_size(domain: str, n) -> Path:
    data = _read_local_file()
    data.setdefault("sample_sizes", {})[domain] = int(n) if n else None
    LOCAL_PATHS_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return LOCAL_PATHS_FILE
