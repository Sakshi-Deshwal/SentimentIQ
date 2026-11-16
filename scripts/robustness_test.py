
import _bootstrap  # noqa: F401

import argparse
import json
import sys

import pandas as pd

from src import config as C

from src.config import (ALL_MODEL_KEYS, THREE_LABELS as LABELS, MODEL_NAMES, ROBUSTNESS_PATH,
                        ROBUSTNESS_RESULTS_PATH, ROBUSTNESS_SUMMARY_PATH, TEXT_COL)
from src.metrics import compute_metrics
from src.preprocessing import clean_text
from src.inference import load_predictor, trained_models


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", choices=ALL_MODEL_KEYS + ["all"], default="all")
    args = parser.parse_args()
    available = trained_models("three_class")
    keys = available if args.model == "all" else [args.model]
    if not keys:
        print("ERROR: no usable trained models. Run python scripts/train_models.py first.")
        return 1

    try:
        tests = pd.read_csv(ROBUSTNESS_PATH)
        bad = set(tests["expected_sentiment"]) - set(LABELS)
        if bad:
            print(f"ERROR: unsupported expected labels in the robustness set: {sorted(bad)}")
            return 1
        card = json.loads((C.THREE_MODELS_DIR / "model_card.json").read_text(encoding="utf-8"))
        from src.three_class import build_three_class_data
        data = build_three_class_data(domains=card.get("domains"), sample_sizes=card.get("sample_sizes"),
                                      seed=card.get("seed", C.RANDOM_STATE), log=lambda *_: None)
    except Exception as exc:
        print(f"ERROR: {exc}")
        return 1

    # Guard: the challenge set must be disjoint from every training/validation/test review.
    seen = set()
    for part in data.pooled.values():
        seen |= set(part[TEXT_COL].map(clean_text))
    overlap = tests[tests[TEXT_COL].map(clean_text).isin(seen)]
    if len(overlap):
        print(f"ERROR: {len(overlap)} robustness reviews also appear in the review data: {overlap['test_id'].tolist()}")
        return 1
    print(f"Loaded {len(tests)} robustness reviews (verified: none appear in the training dataset).\n")

    out = tests.copy()
    summary = {"n": int(len(tests)), "models": {}}
    try:
        for key in keys:
            res = load_predictor("three_class", key).predict_df(tests[TEXT_COL])
            pred = res["prediction"]
            out[f"pred_{key}"] = pred
            out[f"correct_{key}"] = pred == tests["expected_sentiment"]
            if "top_probability" in res:
                out[f"top_prob_{key}"] = res["top_probability"].values
            m = compute_metrics(tests["expected_sentiment"], pred, LABELS)
            by_type = out.groupby("challenge_type")[f"correct_{key}"].agg(["mean", "sum", "count"])
            clear = out[out["label_confidence"] == "clear"][f"correct_{key}"]
            summary["models"][key] = {
                "accuracy": m["accuracy"], "f1_macro": m["f1_macro"],
                "accuracy_clear_labels_only": float(clear.mean()), "n_clear": int(len(clear)),
                "accuracy_by_challenge": {t: {"accuracy": float(r["mean"]), "correct": int(r["sum"]), "n": int(r["count"])}
                                          for t, r in by_type.iterrows()},
                "confusion_matrix": m["confusion_matrix"], "labels": LABELS,
            }
    except Exception as exc:
        print(f"ERROR: {exc}")
        return 1

    C.THREE_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(ROBUSTNESS_RESULTS_PATH, index=False)
    ROBUSTNESS_SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    for key in keys:
        s = summary["models"][key]
        print(f"=== {MODEL_NAMES[key]} ===")
        print(f"accuracy={s['accuracy']:.3f}  macro-F1={s['f1_macro']:.3f}  "
              f"accuracy (clear labels only, n={s['n_clear']})={s['accuracy_clear_labels_only']:.3f}")
        for t, r in s["accuracy_by_challenge"].items():
            print(f"   {t:18s} {r['correct']}/{r['n']}")
        print()

    key0 = keys[0]
    print(f"--- Failures ({MODEL_NAMES[key0]}) ---")
    fails = out[~out[f"correct_{key0}"]]
    for _, r in fails.iterrows():
        extra = f" (p={r[f'top_prob_{key0}']:.2f})" if f"top_prob_{key0}" in out else ""
        print(f"[{r['test_id']}|{r['challenge_type']}|{r['label_confidence']}] expected={r['expected_sentiment']} "
              f"got={r[f'pred_{key0}']}{extra}\n     {r[TEXT_COL][:110]}")
    print("\nCaution: ~50 hand-written examples cannot establish real-world generalisation.")
    print(f"Saved {ROBUSTNESS_RESULTS_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
