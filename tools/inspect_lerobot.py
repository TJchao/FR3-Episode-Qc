"""Inspect a LeRobot dataset schema before writing an embodiment profile."""

from __future__ import annotations

import argparse
import json

from app.readers.lerobot import discover_layout, inspect_layout


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_dir", type=str)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    layout = discover_layout(args.dataset_dir)
    report = inspect_layout(layout)
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as file_handle:
            file_handle.write(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
