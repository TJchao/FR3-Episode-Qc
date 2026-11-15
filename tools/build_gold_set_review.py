"""Generate a reusable local HTML reviewer for a gold-set CSV."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.gold_set_review import (
    build_review_items,
    read_csv_rows,
    read_metric_rows,
    write_review_html,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold-set", type=Path, required=True)
    parser.add_argument("--image-metrics", type=Path, required=True)
    parser.add_argument("--image-hits", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--no-copy-evidence",
        action="store_true",
        help="Reference the original evidence paths instead of copying JPGs beside the HTML.",
    )
    args = parser.parse_args()

    gold_rows = read_csv_rows(args.gold_set)
    metric_rows = read_metric_rows(args.image_metrics)
    hit_rows = read_csv_rows(args.image_hits) if args.image_hits else []
    output_dir = args.output_dir.resolve()
    items, summary = build_review_items(
        gold_rows,
        metric_rows,
        hit_rows,
        output_dir=output_dir,
        copy_evidence=not args.no_copy_evidence,
    )
    html_path = output_dir / "gold_set_review.html"
    write_review_html(items, output_path=html_path, summary=summary)
    print(json.dumps({
        **summary,
        "html_output": str(html_path),
        "instructions": (
            "Open gold_set_review.html in a local browser. Decisions are stored in "
            "localStorage and can be exported as gold_set_reviewed.csv."
        ),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
