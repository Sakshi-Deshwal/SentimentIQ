
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src import settings
from src.config import PROCESSED_DIR, RANDOM_STATE, THREE_LABELS

EXTERNAL_COLUMNS = ["review_id", "review_text", "rating", "sentiment", "domain", "source"]
RATING_TO_SENTIMENT = {1: "Negative", 2: "Negative", 3: "Neutral", 4: "Positive", 5: "Positive"}
CHUNK_ROWS = 100_000


class DatasetUnavailable(Exception):
    """The dataset file is missing/unreadable/invalid. Callers show the message; nothing crashes."""


@dataclass(frozen=True)
class DomainSpec:
    key: str
    title: str
    filename: str
    local_path: bool = False  # True -> path comes from settings (Amazon), False -> data/processed/


DOMAINS = {
    "product_food": DomainSpec("product_food", "Amazon Fine Food reviews",
                               "amazon_fine_food_reviews_cleaned.csv", local_path=True),
    "product_clothing": DomainSpec("product_clothing", "Women's clothing reviews",
                                   "womens_clothing_reviews_cleaned.csv"),
}


def domain_path(domain: str) -> Path:
    spec = DOMAINS[domain]
    return settings.get_amazon_path() if spec.local_path else PROCESSED_DIR / spec.filename


# ------------------------------------------------------------------ file checks
@dataclass
class FileStatus:
    path: Path
    exists: bool
    size_mb: float = 0.0
    message: str = ""


def file_status(path) -> FileStatus:
    """Check a dataset file WITHOUT reading it. Never raises."""
    try:
        p = Path(path)
        if not p.exists():
            return FileStatus(p, False, 0.0, f"File not found: {p}")
        if not p.is_file():
            return FileStatus(p, False, 0.0, f"Not a file: {p}")
        return FileStatus(p, True, p.stat().st_size / 1_048_576, "OK")
    except OSError as exc:
        return FileStatus(Path(str(path)), False, 0.0, f"Cannot access {path}: {exc}")


def domain_status(domain: str) -> FileStatus:
    return file_status(domain_path(domain))


def available_domains() -> list:
    return [d for d in DOMAINS if domain_status(d).exists]


# ------------------------------------------------------------------ validation
def validate_external_frame(df: pd.DataFrame, domain: Optional[str] = None, name: str = "dataset") -> dict:
    """Raise DatasetUnavailable if the schema is wrong; return mapping-consistency info otherwise."""
    missing = [c for c in EXTERNAL_COLUMNS if c not in df.columns]
    if missing:
        raise DatasetUnavailable(f"{name} is missing required columns: {missing}")
    bad = set(df["sentiment"].dropna().unique()) - set(THREE_LABELS)
    if bad:
        raise DatasetUnavailable(f"{name} has unexpected sentiment labels: {sorted(bad)}")
    expected = df["rating"].map(RATING_TO_SENTIMENT)
    mismatch = int((expected != df["sentiment"]).sum())
    return {"label_mapping_mismatches": mismatch, "rows_checked": int(len(df))}


def _compact(df: pd.DataFrame) -> pd.DataFrame:
    out = df[EXTERNAL_COLUMNS].copy()
    out["review_text"] = out["review_text"].fillna("").astype(str)
    out = out[out["review_text"].str.strip() != ""]
    return out.reset_index(drop=True)


# ------------------------------------------------------------------ stratified sampling
def class_targets(counts: dict, n_total: int, strategy: str = "proportional", min_per_class: int = 500) -> dict:
    """How many rows to keep per class.

    proportional: keep natural class proportions (floor of ``min_per_class`` for rare classes).
    balanced:     equal share per class (capped by availability).
    """
    counts = {k: int(v) for k, v in counts.items() if v > 0}
    if n_total <= 0 or n_total >= sum(counts.values()):
        return dict(counts)
    k = len(counts)
    if strategy == "balanced":
        base = {c: min(n, n_total // k) for c, n in counts.items()}
        spare = n_total - sum(base.values())
        for c in sorted(counts, key=lambda x: counts[x] - base[x], reverse=True):
            add = min(spare, counts[c] - base[c])
            base[c] += add
            spare -= add
        return base
    if strategy != "proportional":
        raise ValueError("strategy must be 'proportional' or 'balanced'")
    total = sum(counts.values())
    t = {c: min(n, max(min_per_class, round(n_total * n / total))) for c, n in counts.items()}
    # shrink the largest class if the floors pushed us over budget
    over = sum(t.values()) - n_total
    if over > 0:
        big = max(t, key=t.get)
        t[big] = max(min_per_class if counts[big] >= min_per_class else counts[big], t[big] - over)
    return t


def stratified_sample(df: pd.DataFrame, n_total: int, seed: int = RANDOM_STATE,
                      strategy: str = "proportional", min_per_class: int = 500) -> pd.DataFrame:
    """Exact stratified sample of an in-memory frame (small datasets)."""
    if n_total is None or n_total <= 0 or n_total >= len(df):
        return df.reset_index(drop=True)
    targets = class_targets(df["sentiment"].value_counts().to_dict(), n_total, strategy, min_per_class)
    parts = [g.sample(n=min(targets.get(c, 0), len(g)), random_state=seed) for c, g in df.groupby("sentiment")]
    return pd.concat(parts).sample(frac=1.0, random_state=seed).reset_index(drop=True)


# ------------------------------------------------------------------ loading
def _read_chunks(path: Path, usecols):
    return pd.read_csv(path, usecols=usecols, chunksize=CHUNK_ROWS, dtype={"review_text": "string"},
                       encoding="utf-8", encoding_errors="replace")


def load_domain(domain: str, sample_size: Optional[int] = None, seed: int = RANDOM_STATE,
                strategy: str = "proportional", min_per_class: int = 500) -> tuple:
    """Load one domain read-only. Returns (DataFrame, info). Raises DatasetUnavailable on any problem.

    ``sample_size=None`` keeps every row. For large files a bounded-memory, two-pass
    stratified sample is drawn: pass 1 counts classes, pass 2 keeps each row of class c with
    probability target_c / count_c (seeded per chunk, so the same file + seed gives the same sample).
    The sample size is therefore approximate (within sampling noise of the target).
    """
    if domain not in DOMAINS:
        raise DatasetUnavailable(f"Unknown domain '{domain}'.")
    spec = DOMAINS[domain]
    status = file_status(domain_path(domain))
    if not status.exists:
        hint = ""
        if spec.local_path:
            hint = (f" Set the path on the Home page, in local_paths.json, or with the "
                    f"{settings.ENV_AMAZON} environment variable.")
        raise DatasetUnavailable(status.message + hint)

    info = {"domain": domain, "path": str(status.path), "file_mb": round(status.size_mb, 1),
            "sample_size_requested": sample_size, "sampling": "none", "seed": seed}
    try:
        if sample_size and status.size_mb > 20:               # large file: chunked stratified sampling
            counts = {}
            for ch in _read_chunks(status.path, ["sentiment"]):
                for k, v in ch["sentiment"].value_counts().items():
                    counts[k] = counts.get(k, 0) + int(v)
            n_rows = sum(counts.values())
            targets = class_targets(counts, sample_size, strategy, min_per_class)
            prob = {c: targets.get(c, 0) / counts[c] for c in counts}
            kept = []
            for i, ch in enumerate(_read_chunks(status.path, EXTERNAL_COLUMNS)):
                rng = np.random.default_rng([seed, i])
                p = ch["sentiment"].map(prob).fillna(0.0).to_numpy(dtype=float)
                kept.append(ch[rng.random(len(ch)) < p])
            df = pd.concat(kept, ignore_index=True) if kept else pd.DataFrame(columns=EXTERNAL_COLUMNS)
            info.update(sampling=f"chunked stratified ({strategy})", rows_in_file=n_rows,
                        class_counts_in_file=counts, class_targets=targets)
        else:
            df = pd.read_csv(status.path, usecols=EXTERNAL_COLUMNS, encoding="utf-8", encoding_errors="replace")
            info["rows_in_file"] = int(len(df))
            if sample_size and sample_size < len(df):
                df = stratified_sample(df, sample_size, seed, strategy, min_per_class)
                info["sampling"] = f"stratified ({strategy})"
    except DatasetUnavailable:
        raise
    except ValueError as exc:    # e.g. usecols not found
        raise DatasetUnavailable(f"{status.path.name} does not have the expected columns "
                                 f"{EXTERNAL_COLUMNS}: {exc}") from exc
    except Exception as exc:
        raise DatasetUnavailable(f"Could not read {status.path}: {exc}") from exc

    info.update(validate_external_frame(df, domain, status.path.name))
    df = _compact(df)
    df["domain"] = domain if df["domain"].isna().all() else df["domain"]
    info["rows_loaded"] = int(len(df))
    return df, info


def amazon_stats(path=None) -> dict:
    """Lightweight chunked statistics for a (large) external file: counts only, no text held in memory."""
    status = file_status(path or domain_path("product_food"))
    if not status.exists:
        raise DatasetUnavailable(status.message)
    sent, rating, n = {}, {}, 0
    try:
        for ch in pd.read_csv(status.path, usecols=["sentiment", "rating"], chunksize=CHUNK_ROWS):
            n += len(ch)
            for k, v in ch["sentiment"].value_counts().items():
                sent[k] = sent.get(k, 0) + int(v)
            for k, v in ch["rating"].value_counts().items():
                rating[int(k)] = rating.get(int(k), 0) + int(v)
    except Exception as exc:
        raise DatasetUnavailable(f"Could not read {status.path}: {exc}") from exc
    return {"rows": n, "sentiment": sent, "rating": rating, "file_mb": round(status.size_mb, 1)}


# ------------------------------------------------------------------ dedupe + splits
def normalize_text_key(text: str) -> str:
    return re.sub(r"[^\w]+", " ", str(text).lower()).strip()


def dedupe(df: pd.DataFrame) -> tuple:
    """Drop rows whose normalised text already appeared (keeps first). Returns (frame, n_removed)."""
    key = df["review_text"].map(normalize_text_key)
    keep = ~key.duplicated(keep="first")
    return df[keep].reset_index(drop=True), int((~keep).sum())


def split_domain(df: pd.DataFrame, seed: int = RANDOM_STATE, fractions=(0.70, 0.15, 0.15)) -> dict:
    """Reproducible stratified train/validation/test split of ONE domain (after de-duplication)."""
    assert abs(sum(fractions) - 1) < 1e-9
    train, rest = train_test_split(df, test_size=1 - fractions[0], stratify=df["sentiment"], random_state=seed)
    val, test = train_test_split(rest, test_size=fractions[2] / (fractions[1] + fractions[2]),
                                 stratify=rest["sentiment"], random_state=seed)
    return {"train": train.reset_index(drop=True), "validation": val.reset_index(drop=True),
            "test": test.reset_index(drop=True)}


def assert_disjoint_splits(splits: dict) -> None:
    """Leakage safeguard: no review (by id and by normalised text) may appear in two splits."""
    keys = {name: set(df["review_text"].map(normalize_text_key)) for name, df in splits.items()}
    names = list(keys)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            both = keys[names[i]] & keys[names[j]]
            if both:
                raise AssertionError(f"{len(both)} texts appear in both '{names[i]}' and '{names[j]}'")


def text_and_labels(df: pd.DataFrame) -> tuple:
    """The ONLY model inputs: review text (X) and sentiment (y). Nothing else is exposed."""
    return df["review_text"].astype(str).tolist(), df["sentiment"].tolist()


# ------------------------------------------------------------------ near-duplicate detection
NEAR_DUP_THRESHOLD = 0.90       # cosine similarity of TF-IDF vectors
NEAR_DUP_MIN_TOKENS = 6         # very short reviews ("great product") are only compared by exact match


def near_duplicate_mask(reference_texts, query_texts, threshold: float = NEAR_DUP_THRESHOLD,
                        min_tokens: int = NEAR_DUP_MIN_TOKENS, chunk: int = 2000):
    """Boolean array: True where a query text is a near-duplicate (cosine >= threshold) of ANY reference text.

    An audit/cleaning tool, not a model feature: the TF-IDF space is fitted on the texts being compared only.
    Queries with fewer than ``min_tokens`` tokens are never flagged here (exact duplicates are handled by
    ``dedupe`` / ``normalize_text_key``).
    """
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer
    ref = [normalize_text_key(t) for t in reference_texts]
    qry = [normalize_text_key(t) for t in query_texts]
    mask = np.zeros(len(qry), dtype=bool)
    if not ref or not qry:
        return mask
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True, dtype=np.float32)
    vec.fit(ref + qry)
    R = vec.transform(ref).T.tocsc()
    ntok = np.array([len(q.split()) for q in qry])
    for i in range(0, len(qry), chunk):
        Q = vec.transform(qry[i:i + chunk])
        best = (Q @ R).max(axis=1).toarray().ravel()
        mask[i:i + chunk] = best >= threshold
    return mask & (ntok >= min_tokens)
