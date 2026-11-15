"""Build a human-review gold-set template from image QC outputs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import pyarrow.parquet as pq

from app.gold_set import build_gold_set, write_gold_set


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-metrics", type=Path, required=True)
    parser.add_argument("--image-hits", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--max-hits-per-rule", type=int, default=50)
    parser.add_argument("--max-controls-per-rule", type=int, default=50)
    parser.add_argument(
        "--census-threshold",
        type=int,
        default=30,
        help="Use every auto hit for rules with fewer hits than this threshold.",
    )
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    metric_rows = pq.read_table(args.image_metrics).to_pylist()
    with args.image_hits.open(encoding="utf-8-sig", newline="") as file_handle:
        hit_rows = list(csv.DictReader(file_handle))
    rows, metadata = build_gold_set(
        metric_rows,
        hit_rows,
        max_hits_per_rule=args.max_hits_per_rule,
        max_controls_per_rule=args.max_controls_per_rule,
        census_threshold=args.census_threshold,
        seed=args.seed,
    )
    write_gold_set(rows, metadata, args.output, args.metadata)
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
