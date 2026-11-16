
import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import pandas as pd

from src import datasets as D
from src import settings
from src.config import RANDOM_STATE, THREE_LABELS


@dataclass
class ThreeClassData:
    pooled: dict                                   # split -> DataFrame (all domains)
    per_domain: dict                               # domain -> {split -> DataFrame}
    domains_used: list
    domains_skipped: dict                          # domain -> reason
    audit: dict = field(default_factory=dict)

    def domain_all(self, domain: str) -> pd.DataFrame:
        """Every (de-duplicated, sampled) record of one domain, all splits together."""
        return pd.concat(self.per_domain[domain].values(), ignore_index=True)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def _class_counts(df: pd.DataFrame) -> dict:
    vc = df["sentiment"].value_counts()
    return {l: int(vc.get(l, 0)) for l in THREE_LABELS}


def build_three_class_data(domains: Optional[list] = None, sample_sizes: Optional[dict] = None,
                           seed: int = RANDOM_STATE, strategy: str = "proportional",
                           log: Callable = print) -> ThreeClassData:
    sizes = {**settings.get_sample_sizes(), **(sample_sizes or {})}
    wanted = domains or list(D.DOMAINS)
    loaded, skipped, audit_domains = {}, {}, {}

    for dom in wanted:
        try:
            df, info = D.load_domain(dom, sample_size=sizes.get(dom), seed=seed, strategy=strategy)
        except D.DatasetUnavailable as exc:
            skipped[dom] = str(exc)
            log(f"  [skip] {dom}: {exc}")
            continue
        n_loaded = len(df)
        df, n_dup = D.dedupe(df)                                    # within-domain duplicate texts
        audit_domains[dom] = {
            "path": info["path"], "file_mb": info["file_mb"], "rows_in_file": info.get("rows_in_file", n_loaded),
            "sampling": info["sampling"], "sample_size_requested": sizes.get(dom),
            "rows_loaded": n_loaded, "duplicate_texts_removed": n_dup, "rows_after_dedupe": len(df),
            "label_mapping_mismatches": info["label_mapping_mismatches"],
            "class_counts_after_dedupe": _class_counts(df),
            "sha256": _sha256(Path(info["path"])),
            "avg_words": round(float(df["review_text"].str.split().str.len().mean()), 1),
        }
        loaded[dom] = df
        log(f"  [ok]   {dom}: {n_loaded:,} loaded, {n_dup:,} duplicate texts removed -> {len(df):,}")

    # ---- cross-domain overlap: identical normalised texts in two domains -> keep the earlier domain's copy
    keys = {d: set(df["review_text"].map(D.normalize_text_key)) for d, df in loaded.items()}
    overlap, order = {}, list(loaded)
    for i, a in enumerate(order):
        for b in order[i + 1:]:
            common = keys[a] & keys[b]
            overlap[f"{a}|{b}"] = len(common)
            if common:
                loaded[b] = loaded[b][~loaded[b]["review_text"].map(D.normalize_text_key).isin(common)].reset_index(drop=True)
    # ---- cross-domain NEAR-duplicates (TF-IDF cosine >= threshold): drop from the later domain
    near = {}
    for i, a in enumerate(order):
        for b in order[i + 1:]:
            mask = D.near_duplicate_mask(loaded[a]["review_text"], loaded[b]["review_text"])
            near[f"{a}|{b}"] = int(mask.sum())
            if mask.any():
                loaded[b] = loaded[b][~mask].reset_index(drop=True)
                log(f"  removed {int(mask.sum())} near-duplicate review(s) of {a} from {b}")
    for d in loaded:
        audit_domains[d]["rows_after_cross_domain_dedupe"] = len(loaded[d])

    # ---- splits
    per_domain, pooled_parts = {}, {"train": [], "validation": [], "test": []}
    for dom, df in loaded.items():
        sp = D.split_domain(df, seed=seed)
        # near-duplicates of a TRAINING review may not sit in validation/test (leakage): drop them there
        removed = {}
        for name in ("validation", "test"):
            m = D.near_duplicate_mask(sp["train"]["review_text"], sp[name]["review_text"])
            removed[name] = int(m.sum())
            sp[name] = sp[name][~m].reset_index(drop=True)
        audit_domains[dom]["near_duplicates_removed_from_eval_splits"] = removed
        D.assert_disjoint_splits(sp)
        per_domain[dom] = sp
        for name in pooled_parts:
            pooled_parts[name].append(sp[name])
    pooled = {n: pd.concat(p, ignore_index=True) if p else pd.DataFrame(columns=D.EXTERNAL_COLUMNS)
              for n, p in pooled_parts.items()}
    if loaded:
        D.assert_disjoint_splits(pooled)
        for name, df in pooled.items():
            if df.duplicated(["domain", "review_id"]).any():
                raise AssertionError(f"duplicate (domain, review_id) pairs inside the {name} split")

    audit = {
        "task": "three_class", "label_source": "rating-derived weak labels (1-2 Negative, 3 Neutral, 4-5 Positive)",
        "seed": seed, "split_fractions": {"train": 0.70, "validation": 0.15, "test": 0.15},
        "sampling_strategy": strategy, "sampling_applied": "before splitting, stratified by sentiment",
        "domains_used": list(loaded), "domains_skipped": skipped, "domains": audit_domains,
        "cross_domain_identical_text_overlap": overlap,
        "cross_domain_near_duplicates_removed": near,
        "near_duplicate_rule": (f"TF-IDF cosine >= {D.NEAR_DUP_THRESHOLD} on reviews with >= {D.NEAR_DUP_MIN_TOKENS} tokens; "
                                "shorter reviews are compared by exact match only"),
        "split_sizes": {n: int(len(df)) for n, df in pooled.items()},
        "split_class_counts": {n: _class_counts(df) for n, df in pooled.items() if len(df)},
        "split_domain_counts": {n: {k: int(v) for k, v in df["domain"].value_counts().items()}
                                for n, df in pooled.items() if len(df)},
        "model_input": "review_text only (rating, source, domain, review_id are never features)",
        "language_note": "Both datasets are English-language reviews; no translation is applied.",
    }
    return ThreeClassData(pooled, per_domain, list(loaded), skipped, audit)


def split_manifest(data: ThreeClassData) -> pd.DataFrame:
    """(domain, review_id, split) for every record - lets anyone verify the splits without the text."""
    rows = []
    for dom, sp in data.per_domain.items():
        for name, df in sp.items():
            rows.append(pd.DataFrame({"domain": dom, "review_id": df["review_id"], "split": name,
                                      "sentiment": df["sentiment"]}))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
