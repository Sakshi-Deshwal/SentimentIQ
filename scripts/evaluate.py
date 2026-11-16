
import _bootstrap  # noqa: F401

import argparse
import json
import sys
from datetime import datetime, timezone

from src import config as C
from src.inference import ModelNotFound, is_trained, load_predictor, read_model_meta, time_inference
from src.metrics import compute_metrics
from src.tasks import get_task


def load_frames(task_key: str, card: dict):
    """Rebuild the exact splits used in training."""
    from src.three_class import build_three_class_data
    data = build_three_class_data(domains=card.get("domains"), sample_sizes=card.get("sample_sizes"),
                                  seed=card.get("seed", C.RANDOM_STATE), log=lambda *_: None)
    expect = card.get("split_sizes")
    got = data.audit["split_sizes"]
    if expect and expect != got:
        raise RuntimeError(f"The data no longer matches what the models were trained on "
                           f"(trained {expect}, now {got}). Retrain: python scripts/train_models.py")
    return data.pooled, data.per_domain, data


def slim(m: dict) -> dict:
    return {k: m[k] for k in ("n", "accuracy", "f1_macro", "f1_weighted")}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="*", default=None)
    args = ap.parse_args()
    task = get_task()
    args.task = task.key

    try:
        card = json.loads((task.models_dir / "model_card.json").read_text(encoding="utf-8"))
    except FileNotFoundError:
        card = {}
    keys = [k for k in (args.models or task.model_keys) if is_trained(args.task, k)]
    if not keys:
        print(f"ERROR: no trained models for {args.task}. Train first (see README).")
        return 1
    try:
        parts, per_domain, data = load_frames(args.task, card)
    except Exception as exc:
        print(f"ERROR: {exc}")
        return 1
    train, val, test = parts["train"], parts["validation"], parts["test"]
    labels = list(task.labels)
    task.reports_dir.mkdir(parents=True, exist_ok=True)
    domains = sorted(test["domain"].unique()) if "domain" in test else []

    meta_cols = ["review_id", "review_text", "sentiment"] + [c for c in ("domain", "rating", "category") if c in test]
    preds = test[meta_cols].rename(columns={"sentiment": "true_label"}).reset_index(drop=True)
    results = {}
    for key in keys:
        try:
            predictor = load_predictor(args.task, key)
        except (ModelNotFound, RuntimeError) as exc:
            print(f"ERROR: {exc}")
            return 1
        meta = read_model_meta(args.task, key)
        out = {}
        for name, df in (("train", train), ("validation", val), ("test", test)):
            pr = predictor.predict_df(df["review_text"])
            out[name] = (pr, compute_metrics(df["sentiment"], pr["prediction"], labels))
        timing = time_inference(predictor, test["review_text"].tolist())
        pr_test = out["test"][0]
        preds[f"pred_{key}"] = pr_test["prediction"].values
        if "top_probability" in pr_test:
            preds[f"top_prob_{key}"] = pr_test["top_probability"].values

        entry = {
            "name": C.MODEL_NAMES[key], "family": C.MODEL_FAMILY[key], "supports_probabilities": predictor.supports_proba,
            "splits": {"train": slim(out["train"][1])},
            "validation": out["validation"][1], "test": out["test"][1],
            "train_seconds": meta.get("train_seconds") or meta.get("fit_seconds"),
            "fit_seconds": meta.get("fit_seconds"), "tuning_seconds": (meta.get("tuning") or {}).get("seconds"),
            "inference": timing, "parameters": meta.get("parameters"),
            "best_params": meta.get("best_params"), "best_epoch": meta.get("best_epoch"),
            "epochs_run": meta.get("epochs_run"), "stopped_early": meta.get("stopped_early"),
            "history": meta.get("history"), "train_config": meta.get("train_config"),
            "device": meta.get("device"), "domains_trained_on": meta.get("domains"),
        }
        if domains:
            by = {}
            for d in domains:
                mask = (test["domain"] == d).values
                by[d] = compute_metrics(test.loc[mask, "sentiment"], pr_test["prediction"][mask], labels)
            entry["test_by_domain"] = by
        results[key] = entry
        print(f"{C.MODEL_NAMES[key]:24s} train acc {entry['splits']['train']['accuracy']:.4f} | "
              f"val macro-F1 {entry['validation']['f1_macro']:.4f} | test acc {entry['test']['accuracy']:.4f} "
              f"macro-F1 {entry['test']['f1_macro']:.4f} | {timing['ms_per_review']:.3f} ms/review", flush=True)

    best = max(results, key=lambda k: results[k]["validation"]["f1_macro"])
    payload = {
        "task": args.task, "title": task.title, "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "labels": labels, "weak_labels": task.weak_labels,
        "n": {"train": len(train), "validation": len(val), "test": len(test)},
        "domains": domains, "domains_trained_on": card.get("domains"),
        "best_model_key_by_validation": best, "n_test": int(len(test)),
        "fit_mode": card.get("fit_mode"),
        "notes": ["Test numbers come from the untouched test split, evaluated once after training.",
                  "Best model is chosen by validation macro-F1, never by test score."]
                 + (["Labels are rating-derived weak labels; see the dataset audit."] if task.weak_labels else []),
        "models": results,
    }
    task.evaluation_path.write_text(json.dumps(payload, indent=2, default=float), encoding="utf-8")
    preds.to_csv(task.predictions_path, index=False)
    print(f"\nBest model by validation macro-F1: {C.MODEL_NAMES[best]}")
    print(f"Saved {task.evaluation_path} and {task.predictions_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
