"""Calibrate image-rule thresholds using reviewed gold-set labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.image_rule_calibration import (
    calibrate,
    markdown,
    read_csv_rows,
    read_metric_rows,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold-set", type=Path, required=True)
    parser.add_argument("--image-metrics", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    args = parser.parse_args()

    summary = calibrate(
        read_csv_rows(args.gold_set),
        read_metric_rows(args.image_metrics),
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    args.output_md.write_text(markdown(summary), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
