"""Evaluate a reviewed image-rule gold set."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.gold_set import evaluate_gold_set, read_gold_set


def _markdown(metrics: dict) -> str:
    lines = [
        "# Image Rule Gold Set Metrics",
        "",
        f"Rows: `{metrics['rows']}`",
        f"Evaluated rows: `{metrics['evaluated_rows']}`",
        f"Review completion: `{metrics['review_completion']}`",
        f"Precision: `{metrics['precision']}`",
        f"Recall: `{metrics['recall']}`",
        f"FPR estimate: `{metrics['fpr_estimate']}`",
        f"Recall: `{metrics['recall']}`",
        f"F1: `{metrics['f1']}`",
        f"Recall/F1 null reason: `{metrics['recall_f1_null_reason']}`",
        "",
        "## Per Rule",
        "",
        "| Rule | Evaluated | TP | FP | FN | TN | Precision | FPR estimate | Automation coverage |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for rule, row in metrics.get("per_rule", {}).items():
        matrix = row.get("confusion_matrix", {})
        lines.append(
            f"| `{rule}` | {row.get('evaluated_rows', 0)} | "
            f"{matrix.get('tp', 0)} | {matrix.get('fp', 0)} | "
            f"{matrix.get('fn', 0)} | {matrix.get('tn', 0)} | "
            f"{row.get('precision')} | {row.get('fpr_estimate')} | "
            f"{row.get('automation_coverage')} |"
        )
    lines.extend([
        "",
        "Automation coverage is `1 - review candidates / sampled frames` under the current policy that every auto hit needs human review. It is a review-work metric, not an accuracy claim. FPR is a case-control weighted estimate using source hit/non-hit counts. Recall and F1 remain null because this sampling design does not cover all false negatives.",
        "",
    ])
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold-set", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    args = parser.parse_args()

    rows, metadata = read_gold_set(args.gold_set, args.metadata)
    metrics = evaluate_gold_set(rows, metadata.get("source_counts", {}))
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    args.output_md.write_text(_markdown(metrics), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
