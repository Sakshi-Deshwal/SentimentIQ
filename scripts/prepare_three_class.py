
import _bootstrap  # noqa: F401

import argparse
import json
import os
import sys

from src import settings
from src.tasks import THREE_CLASS


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--amazon-path", help="override the Amazon CSV location for this run")
    ap.add_argument("--sample", nargs="*", help="rows kept per domain, e.g. product_food=100000 (0 = all)")
    args = ap.parse_args()
    if args.amazon_path:
        os.environ[settings.ENV_AMAZON] = args.amazon_path

    from src.three_class import build_three_class_data, split_manifest
    sizes = {}
    for item in args.sample or []:
        d, _, n = item.partition("=")
        sizes[d] = int(n) or None

    print(f"Amazon path: {settings.get_amazon_path()}")
    data = build_three_class_data(sample_sizes=sizes)
    if not data.domains_used:
        print("ERROR: no dataset could be loaded.")
        return 1
    THREE_CLASS.reports_dir.mkdir(parents=True, exist_ok=True)
    (THREE_CLASS.reports_dir / "dataset_audit.json").write_text(json.dumps(data.audit, indent=2), encoding="utf-8")
    split_manifest(data).to_csv(THREE_CLASS.reports_dir / "splits_manifest.csv", index=False)

    a = data.audit
    print("\nSplit sizes:", a["split_sizes"])
    print("Class counts (train):", a["split_class_counts"]["train"])
    print("Cross-domain identical-text overlap:", a["cross_domain_identical_text_overlap"])
    print("Cross-domain near-duplicates removed:", a["cross_domain_near_duplicates_removed"])
    for d, v in a["domains"].items():
        print(f"  class counts {d}: {v['class_counts_after_dedupe']}  eval near-dups removed: {v['near_duplicates_removed_from_eval_splits']}")
    for d, v in a["domains"].items():
        print(f"  {d:18s} rows={v['rows_after_cross_domain_dedupe']:,}  dup removed={v['duplicate_texts_removed']}  "
              f"mapping mismatches={v['label_mapping_mismatches']}")
    if data.domains_skipped:
        print("\nSkipped (unavailable):")
        for d, why in data.domains_skipped.items():
            print(f"  {d}: {why}")
    print(f"\nSaved audit and split manifest to {THREE_CLASS.reports_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
