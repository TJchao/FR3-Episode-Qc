"""Generate a batch-level LeRobot dataset profile."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from app.dataset_profile import build_dataset_profile


FIELDS = (
    "episode_id",
    "status",
    "severity",
    "error_types",
    "error_count",
    "camera_age_p95_ms",
    "frame_count",
    "episode_hash",
    "dataset_version",
    "git_commit",
    "created_at",
    "duration_s",
)


def _markdown(profile: dict) -> str:
    quality = profile["quality"]
    dataset = profile["dataset"]
    lines = [
        "# LeRobot Dataset Profile",
        "",
        f"Robot type: `{dataset.get('robot_type')}`",
        f"Codebase version: `{dataset.get('codebase_version')}`",
        f"Selected episodes: `{dataset.get('selected_episodes')}`",
        f"FPS: `{dataset.get('fps')}`",
        "",
        "## Quality",
        "",
        f"Status counts: `{quality.get('status_counts')}`",
        f"Severity counts: `{quality.get('severity_counts')}`",
        f"Error types: `{quality.get('error_type_counts')}`",
        f"Frame count: `{quality.get('frame_count')}`",
        f"Duration seconds: `{quality.get('duration_s')}`",
    ]
    rules_skipped = quality.get("rules_skipped", [])
    lines.extend([
        "",
        "## Rule Coverage Boundary",
        "",
        f"Rules evaluated: `{quality.get('rules_evaluated')}`",
        f"Rules skipped: `{len(rules_skipped)}`",
    ])
    for item in rules_skipped:
        lines.append(f"- `{item.get('rule')}`: `{item.get('reason')}`")
    lines.append(quality.get("rule_coverage_interpretation", ""))
    lines.extend([
        "",
        "## Analysis Coverage",
        "",
        f"Low-dimensional episodes: `{profile.get('coverage', {}).get('low_dimensional_episodes')}`",
        f"Image episodes: `{profile.get('coverage', {}).get('image_episodes')}`",
        f"Embedding episodes: `{profile.get('coverage', {}).get('embedding_episodes')}`",
        f"Gold-set review rows: `{profile.get('coverage', {}).get('gold_set_review_rows')}`",
    ])
    lines.extend([
        "",
        "## Motion",
        "",
        f"State unit: `{profile['motion'].get('state_unit')}`",
    ])
    for group, metrics in profile["motion"].get("groups", {}).items():
        lines.append(f"- `{group}` velocity_p95: `{metrics.get('velocity_p95')}`")
    comparison = profile["motion"].get("group_comparison", {})
    if comparison:
        lines.extend([
            "",
            "### Group-Scale Finding",
            "",
            f"Right-hand velocity_p95 exceeds left-arm in `{comparison.get('right_hand_greater_count')}` / `{comparison.get('episode_count')}` episodes.",
            f"Paired right-hand/left-arm ratio p50: `{comparison.get('paired_ratio_p50')}`",
            comparison.get("boundary", ""),
        ])
    jerk = profile["motion"].get("finite_difference_boundary", {})
    if jerk:
        lines.extend([
            "",
            "### Jerk Boundary",
            "",
            f"Finite-difference noise gain at `{jerk.get('fps')}` Hz: `{jerk.get('theoretical_noise_gain')}`",
            f"Jerk noise floor: `{jerk.get('jerk_noise_floor')}`",
            jerk.get("jerk_noise_floor_reason", ""),
            jerk.get("interpretation", ""),
        ])
    image = profile.get("image", {})
    if image.get("available"):
        lines.extend([
            "",
            "## Image",
            "",
            f"Sample count: `{image.get('sample_count')}`",
            f"Rule hit counts: `{image.get('rule_hit_counts', {})}`",
        ])
    embedding = profile.get("embedding", {})
    if embedding.get("available", False) or embedding.get("method"):
        lines.extend([
            "",
            "## Embedding",
            "",
            f"Method: `{embedding.get('method')}`",
            f"Episode count: `{embedding.get('episode_count')}`",
            f"Relative low-density tail: `{embedding.get('relative_low_density_tail_count')}`",
            f"Near-duplicate candidate pairs: `{embedding.get('near_duplicate_candidate_pairs')}`",
            f"Pairwise similarity range: `{embedding.get('pairwise_similarity', {}).get('min')}` to `{embedding.get('pairwise_similarity', {}).get('max')}`",
            f"Closest pair: `{embedding.get('closest_pair')}`",
        ])
    gold = profile.get("gold_set", {})
    lines.extend([
        "",
        "## Gold Set",
        "",
        f"Available: `{gold.get('available', False)}`",
    ])
    if gold.get("precision") is not None:
        lines.append(f"Precision: `{gold.get('precision')}`")
        lines.append(f"Recall: `{gold.get('recall')}`")
        lines.append(f"FPR estimate: `{gold.get('fpr_estimate')}`")
        lines.append(f"Recall/F1 null reason: `{gold.get('recall_f1_null_reason')}`")
    else:
        lines.append("Human labels are not yet available; metrics are intentionally not claimed.")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_dir", type=Path)
    parser.add_argument("--profile", default="unitree_g1")
    parser.add_argument("--episodes", default="0")
    parser.add_argument("--image-metrics", type=Path)
    parser.add_argument("--image-hits", type=Path)
    parser.add_argument("--embedding-summary", type=Path)
    parser.add_argument("--gold-set-metrics", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset-version", default="unversioned")
    parser.add_argument("--git-commit", default="unknown")
    args = parser.parse_args()

    profile, records, motions = build_dataset_profile(
        args.dataset_dir,
        profile_name=args.profile,
        episode_selection=args.episodes,
        image_metrics_path=args.image_metrics,
        image_hits_path=args.image_hits,
        embedding_summary_path=args.embedding_summary,
        gold_set_metrics_path=args.gold_set_metrics,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for record in records:
        record["dataset_version"] = args.dataset_version
        record["git_commit"] = args.git_commit

    schema = pa.schema([
        pa.field("episode_id", pa.string(), nullable=False),
        pa.field("status", pa.string(), nullable=False),
        pa.field("severity", pa.string(), nullable=False),
        pa.field("error_types", pa.string(), nullable=False),
        pa.field("error_count", pa.int32(), nullable=False),
        pa.field("camera_age_p95_ms", pa.float64()),
        pa.field("frame_count", pa.int64(), nullable=False),
        pa.field("episode_hash", pa.string(), nullable=False),
        pa.field("dataset_version", pa.string(), nullable=False),
        pa.field("git_commit", pa.string(), nullable=False),
        pa.field("created_at", pa.string(), nullable=False),
        pa.field("duration_s", pa.float64(), nullable=False),
    ])
    pq.write_table(
        pa.Table.from_pylist(records, schema=schema),
        args.output_dir / "qc_results.parquet",
        compression="snappy",
    )
    with (args.output_dir / "qc_results.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as file_handle:
        writer = csv.DictWriter(file_handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(records)
    (args.output_dir / "motion_profile.json").write_text(
        json.dumps(motions, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "dataset_profile.json").write_text(
        json.dumps(profile, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "dataset_profile.md").write_text(
        _markdown(profile), encoding="utf-8"
    )
    print(json.dumps(profile, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
