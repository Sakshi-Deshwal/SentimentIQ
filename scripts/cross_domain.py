
import _bootstrap  # noqa: F401

import argparse
import json
import sys
import time
from datetime import datetime, timezone

from src.cross_domain import (MODELS, STANDARD_EXPERIMENTS, experiment_id, pooled_id, run_experiment,
                              run_in_domain_baseline, run_pooled_experiment)
from src.tasks import THREE_CLASS
from src.three_class import build_three_class_data

OUT = THREE_CLASS.reports_dir / "cross_domain.json"


def load_existing() -> dict:
    try:
        return json.loads(OUT.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {"results": [], "skipped": []}


def save(state: dict, used_domains) -> None:
    state["generated_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    state["domains_available"] = list(used_domains)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(state, indent=2, default=float), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", default=list(MODELS), choices=MODELS)
    ap.add_argument("--train", nargs="+", help="custom experiment: training domain(s)")
    ap.add_argument("--test", help="custom experiment: held-out test domain")
    ap.add_argument("--train-cap", type=int, default=20000, help="max training rows (stratified); 0 = no cap")
    ap.add_argument("--test-cap", type=int, default=10000, help="max held-out test rows (stratified); 0 = no cap")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--sample", nargs="*", help="rows kept per domain, e.g. product_food=60000")
    ap.add_argument("--no-baseline", action="store_true")
    ap.add_argument("--fresh", action="store_true", help="discard previously saved results")
    args = ap.parse_args()

    sizes = {}
    for item in args.sample or []:
        d, _, n = item.partition("=")
        sizes[d] = int(n) or None
    print("Preparing three-class data ...")
    data = build_three_class_data(sample_sizes=sizes)
    if not data.domains_used:
        print("ERROR: no dataset available.")
        return 1
    state = {"results": [], "skipped": []} if args.fresh else load_existing()
    state["protocol"] = ("Held-out-domain: train on the training domain(s) only (fixed hyper-parameters, "
                         "validation split of the training domain(s) for early stopping); test on the held-out domain.")
    state["skipped"] = []
    done = {r["id"]: r for r in state["results"]}

    experiments = [(args.train, args.test)] if args.train and args.test else STANDARD_EXPERIMENTS
    plan = []
    for tr, te in experiments:
        need = [d for d in [*tr, te] if d not in data.domains_used]
        if need:
            reason = "; ".join(f"{d}: {data.domains_skipped.get(d, 'not available')}" for d in need)
            state["skipped"].append({"train_domains": tr, "test_domain": te, "reason": reason})
            print(f"[skip] {'+'.join(tr)} -> {te}: {reason}")
        else:
            plan.append((tr, te))

    caps = dict(train_cap=args.train_cap or None)
    nn_over = {"epochs": args.epochs} if args.epochs else None
    log = lambda m: print(m, flush=True)

    # ---- in-domain baselines (one per test domain x model)
    if not args.no_baseline:
        for dom in sorted({te for _, te in plan}):
            for m in args.models:
                rid = f"{dom}->{dom}::{m}"
                if rid in done:
                    continue
                print(f"\n[baseline] {dom} (random split) :: {m}", flush=True)
                t0 = time.perf_counter()
                r = run_in_domain_baseline(data.per_domain, dom, m, nn_overrides=nn_over, log=log, **caps)
                state["results"].append(r); done[rid] = r
                save(state, data.domains_used)
                print(f"   macro-F1 {r['metrics']['f1_macro']:.4f}  acc {r['metrics']['accuracy']:.4f}  ({time.perf_counter()-t0:.0f}s)")

    # ---- held-out-domain experiments
    for tr, te in plan:
        for m in args.models:
            rid = experiment_id(tr, te, m)
            print(f"\n[cross-domain] {'+'.join(tr)} -> {te} :: {m}", flush=True)
            t0 = time.perf_counter()
            r = run_experiment(data.per_domain, tr, te, m, test_cap=args.test_cap or None, nn_overrides=nn_over,
                               log=log, **caps)
            state["results"] = [x for x in state["results"] if x["id"] != rid] + [r]
            save(state, data.domains_used)
            print(f"   macro-F1 {r['metrics']['f1_macro']:.4f}  acc {r['metrics']['accuracy']:.4f}  "
                  f"OOV {r['test_oov_rate']:.1%}  n_train={r['n_train']:,} n_test={r['n_test']:,} "
                  f"({time.perf_counter()-t0:.0f}s)")

    # ---- both datasets pooled -> held-out test splits (only for the standard run)
    if not (args.train and args.test) and len(data.domains_used) >= 2:
        for m in args.models:
            rid = pooled_id(m)
            print(f"\n[pooled] both datasets -> held-out test splits :: {m}", flush=True)
            t0 = time.perf_counter()
            r = run_pooled_experiment(data.per_domain, m, nn_overrides=nn_over, log=log, **caps)
            state["results"] = [x for x in state["results"] if x["id"] != rid] + [r]
            save(state, data.domains_used)
            print(f"   macro-F1 {r['metrics']['f1_macro']:.4f}  acc {r['metrics']['accuracy']:.4f}  "
                  f"n_train={r['n_train']:,} n_test={r['n_test']:,} ({time.perf_counter()-t0:.0f}s)")
    save(state, data.domains_used)
    print(f"\nSaved {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
