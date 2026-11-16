
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"


def _j(p):
    p = ROOT / p
    return json.loads(p.read_text()) if p.exists() else None


def _eval_table(ev) -> str:
    if not ev:
        return "_Not generated yet. Run the training and evaluation commands in section 13._"
    best = ev["best_model_key_by_validation"]
    rows = ["| Model | Family | Train acc | Val macro-F1 | Test acc | Test macro-F1 | Train time (s) | Inference (ms/review) | Params |",
            "|---|---|---|---|---|---|---|---|---|"]
    for k, m in ev["models"].items():
        rows.append(f"| {m['name']}{' ★' if k == best else ''} | {m['family']} | {m['splits']['train']['accuracy']:.4f} | "
                    f"{m['validation']['f1_macro']:.4f} | {m['test']['accuracy']:.4f} | {m['test']['f1_macro']:.4f} | "
                    f"{m['train_seconds']:.1f} | {m['inference']['ms_per_review']:.3f} | {m['parameters'] if m.get('parameters') else '-'} |")
    n = ev["n"]
    rows.append(f"\n★ = best by **validation** macro-F1. Split sizes: train {n['train']:,}, validation {n['validation']:,}, test {n['test']:,}. "
                f"Fit mode: `{ev.get('fit_mode', 'n/a')}`. Single split, single seed.")
    return "\n".join(rows)


def _domain_table(ev) -> str:
    if not ev:
        return ""
    best = ev["best_model_key_by_validation"]
    m = ev["models"][best]
    if not m.get("test_by_domain"):
        return ""
    rows = [f"Per-domain test performance of the best model ({m['name']}):\n", "| Domain | Reviews | Accuracy | Macro-F1 |", "|---|---|---|---|"]
    for d, v in m["test_by_domain"].items():
        rows.append(f"| {d} | {v['n']:,} | {v['accuracy']:.4f} | {v['f1_macro']:.4f} |")
    return "\n".join(rows)


def _xd_table(res) -> str:
    if not res or not res.get("results"):
        return "_Not generated yet. Run `python scripts/cross_domain.py`._"
    names = {"product_food": "Amazon", "product_clothing": "Clothing"}
    nm = lambda ds: " + ".join(names.get(d, d) for d in ds)
    base = {r["id"]: r for r in res["results"] if r["kind"] == "in_domain_baseline"}
    rows = ["**Transfer between datasets** (trained on one, tested on the other):\n",
            "| Train -> test | Model | Accuracy | Macro-F1 | In-domain baseline macro-F1 | Test OOV rate |", "|---|---|---|---|---|---|"]
    for r in res["results"]:
        if r["kind"] != "held_out_domain":
            continue
        b = base.get(f"{r['test_domain']}->{r['test_domain']}::{r['model']}")
        bf = f"{b['metrics']['f1_macro']:.4f}" if b else "n/a"
        rows.append(f"| {nm(r['train_domains'])} -> {nm([r['test_domain']])} | {r['model']} | {r['metrics']['accuracy']:.4f} | "
                    f"{r['metrics']['f1_macro']:.4f} | {bf} | {r['test_oov_rate']:.1%} |")
    pooled = [r for r in res["results"] if r["kind"] == "pooled_test"]
    if pooled:
        rows += ["\n**Both datasets pooled**, scored on the held-out test splits of both (not a transfer test):\n",
                 "| Model | Accuracy | Macro-F1 | Amazon macro-F1 | Clothing macro-F1 |", "|---|---|---|---|---|"]
        for r in pooled:
            bd = r["by_domain"]
            rows.append(f"| {r['model']} | {r['metrics']['accuracy']:.4f} | {r['metrics']['f1_macro']:.4f} | "
                        f"{bd['product_food']['f1_macro']:.4f} | {bd['product_clothing']['f1_macro']:.4f} |")
    return "\n".join(rows)


def _rob_table(s) -> str:
    if not s:
        return "_Not generated yet. Run `python scripts/robustness_test.py`._"
    rows = [f"{s['n']} hand-written examples (diagnostic only):\n", "| Model | Accuracy | Macro-F1 | Accuracy (clear labels only) |", "|---|---|---|---|"]
    for k, v in s["models"].items():
        rows.append(f"| {k} | {v['accuracy']:.3f} | {v['f1_macro']:.3f} | {v['accuracy_clear_labels_only']:.3f} |")
    return "\n".join(rows)


def _audit(a) -> str:
    if not a:
        return "_Not generated yet. Run `python scripts/prepare_three_class.py`._"
    rows = ["| Dataset | Rows loaded | Exact duplicates removed | Rows after de-dup | Negative | Neutral | Positive |",
            "|---|---|---|---|---|---|---|"]
    for d, v in a["domains"].items():
        c = v["class_counts_after_dedupe"]
        rows.append(f"| {d} | {v['rows_loaded']:,} | {v['duplicate_texts_removed']} | {v['rows_after_dedupe']:,} | "
                    f"{c.get('Negative', 0):,} | {c.get('Neutral', 0):,} | {c.get('Positive', 0):,} |")
    s = a["split_sizes"]
    near = a.get("cross_domain_near_duplicates_removed", {})
    ev = {d: v.get("near_duplicates_removed_from_eval_splits") for d, v in a["domains"].items()}
    rows.append(f"\nSplits: train {s['train']:,} / validation {s['validation']:,} / test {s['test']:,}. "
                f"Identical texts shared by the two datasets: {sum(a['cross_domain_identical_text_overlap'].values())}. "
                f"Near-duplicates between the datasets removed: {sum(near.values())}. "
                f"Near-duplicates of training reviews removed from validation/test: {ev}. "
                f"Rule: {a.get('near_duplicate_rule', 'n/a')}.")
    return "\n".join(rows)


BLOCKS = {
    "THREE": lambda: _eval_table(_j("reports/three_class/evaluation.json")) + "\n\n" + _domain_table(_j("reports/three_class/evaluation.json")),
    "XD": lambda: _xd_table(_j("reports/three_class/cross_domain.json")),
    "ROB": lambda: _rob_table(_j("reports/three_class/robustness_summary.json")),
    "AUDIT": lambda: _audit(_j("reports/three_class/dataset_audit.json")),
}


def main() -> None:
    text = README.read_text(encoding="utf-8")
    for name, fn in BLOCKS.items():
        pat = re.compile(rf"(<!-- RESULTS:{name}:START -->).*?(<!-- RESULTS:{name}:END -->)", re.S)
        if not pat.search(text):
            print(f"marker {name} not found")
            continue
        text = pat.sub(lambda m: f"{m.group(1)}\n{fn()}\n{m.group(2)}", text)
    README.write_text(text, encoding="utf-8")
    print("README results updated from reports/")


if __name__ == "__main__":
    main()
