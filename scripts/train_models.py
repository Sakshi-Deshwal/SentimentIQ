
import _bootstrap  # noqa: F401

import argparse
import json
import sys
import time

from src import config as C
from src.tasks import get_task
from src.training import save_trained, train_one


def parse_samples(items):
    out = {}
    for item in items or []:
        dom, _, n = item.partition("=")
        if not n.isdigit():
            raise SystemExit(f"--sample expects DOMAIN=N (e.g. product_food=60000), got '{item}'")
        out[dom] = int(n) if int(n) > 0 else None
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", default=None, help="model keys (default: all valid for the task)")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--patience", type=int, default=None)
    ap.add_argument("--max-len", type=int, default=None, help="token truncation length (TextCNN/BiLSTM)")
    ap.add_argument("--seed", type=int, default=C.RANDOM_STATE)
    ap.add_argument("--sample", nargs="*", help="rows kept per domain, e.g. product_food=60000 (0 = all)")
    ap.add_argument("--domains", nargs="*", help="restrict to these domains")
    ap.add_argument("--no-tune", action="store_true", help="traditional models: skip grid search, use fixed params")
    args = ap.parse_args()

    task = get_task()
    args.task = task.key
    keys = args.models or list(task.model_keys)
    bad = [k for k in keys if k not in task.model_keys]
    if bad:
        print(f"ERROR: {bad} are not available models ({list(task.model_keys)}).")
        return 1
    from src.compat import TORCH_HELP, torch_available
    if any(k in C.NN_KEYS for k in keys) and not torch_available():
        print("ERROR: " + TORCH_HELP)
        print("Classical models only:  python scripts/train_models.py --models logistic_regression linear_svm")
        return 1

    overrides = {"epochs": args.epochs, "batch_size": args.batch_size, "lr": args.lr, "patience": args.patience,
                 "max_len": args.max_len, "seed": args.seed}

    # ------------------------------------------------------------------ data
    extra = {"task": args.task}
    from src.three_class import build_three_class_data, split_manifest
    print("Preparing three-class data ...")
    data = build_three_class_data(domains=args.domains, sample_sizes=parse_samples(args.sample), seed=args.seed)
    if not data.domains_used:
        print("ERROR: no dataset is available.\n" + "\n".join(data.domains_skipped.values()))
        return 1
    train, val = data.pooled["train"], data.pooled["validation"]
    task.reports_dir.mkdir(parents=True, exist_ok=True)
    (task.reports_dir / "dataset_audit.json").write_text(json.dumps(data.audit, indent=2), encoding="utf-8")
    split_manifest(data).to_csv(task.reports_dir / "splits_manifest.csv", index=False)
    sizes = {d: v["sample_size_requested"] for d, v in data.audit["domains"].items()}
    extra.update(domains=data.domains_used, domains_skipped=list(data.domains_skipped),
                 split_sizes=data.audit["split_sizes"], sample_sizes=sizes,
                 label_source="rating-derived weak labels")
    header = {"domains": data.domains_used, "domains_skipped": list(data.domains_skipped), "seed": args.seed,
              "sample_sizes": sizes, "split_sizes": data.audit["split_sizes"],
              "label_source": "rating-derived weak labels", "fit_mode": "train_only"}
    if data.domains_skipped:
        print(f"NOTE: trained WITHOUT {list(data.domains_skipped)} (dataset unavailable).")

    print(f"\nTask {args.task}: train={len(train):,} validation={len(val):,}  (test split not loaded)")

    # ------------------------------------------------------------------ train
    summary = []
    for key in keys:
        print(f"\n=== {C.MODEL_NAMES[key]} ===", flush=True)
        t0 = time.perf_counter()
        try:
            tm = train_one(args.task, key, train, val, tune=not args.no_tune,
                           overrides=overrides, checkpoint_dir=True)
        except Exception as exc:
            print(f"ERROR training {key}: {exc}")
            return 1
        save_trained(tm, extra, header)
        secs = time.perf_counter() - t0
        summary.append((key, secs, tm.meta.get("best_epoch"), tm.meta.get("epochs_run")))
        print(f"saved {key} in {secs:.1f}s", flush=True)

    print("\nTraining summary:")
    for key, secs, best, ran in summary:
        extra_txt = f" (best epoch {best} of {ran} run)" if best else ""
        print(f"  {C.MODEL_NAMES[key]:24s} {secs:8.1f}s{extra_txt}")
    print("\nNext: python scripts/evaluate.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
