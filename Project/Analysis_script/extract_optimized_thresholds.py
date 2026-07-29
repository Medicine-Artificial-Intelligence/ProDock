#!/usr/bin/env python3
"""
Extract optimized threshold values from 9 ProDock result folders.

Expected layout:
root/
  aff_log/TARGET/TARGET_all_structure_results.json
  aff_pr/TARGET/TARGET_all_structure_results.json
  aff_roc/TARGET/TARGET_all_structure_results.json
  cnn_log/TARGET/TARGET_all_structure_results.json
  cnn_pr/TARGET/TARGET_all_structure_results.json
  cnn_roc/TARGET/TARGET_all_structure_results.json
  combined_log/TARGET/TARGET_all_structure_results.json
  combined_pr/TARGET/TARGET_all_structure_results.json
  combined_roc/TARGET/TARGET_all_structure_results.json

Output:
  threshold_csv/aff_log_thresholds.csv
  ... one CSV per config folder ...

CSV column order:
  protein -> metric -> threshold metric columns -> optimization-metric columns -> ROC-AUC columns

Notes:
  - config_folder and json_file are intentionally NOT written.
  - metric is the optimization_metric from the JSON, e.g. roc-auc, pr-auc, logauc.
"""

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

DEFAULT_CONFIGS = [
    "aff_log",
    "aff_pr",
    "aff_roc",
    "cnn_log",
    "cnn_pr",
    "cnn_roc",
    "combined_log",
    "combined_pr",
    "combined_roc",
]

ROC_AUC_COLS = [
    "optimized_roc_auc",
    "baseline_roc_auc",
    "roc_auc_improvement",
    "test_optimized_roc_auc",
    "test_baseline_roc_auc",
    "test_roc_auc_improvement",
]


def find_result_json(target_dir: Path) -> Optional[Path]:
    """Find the result JSON inside one target/protein subfolder."""
    target = target_dir.name

    preferred = target_dir / f"{target}_all_structure_results.json"
    if preferred.exists():
        return preferred

    matches = sorted(target_dir.glob("*_all_structure_results.json"))
    if matches:
        return matches[0]

    return None


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def metric_specific_cols(metric: str) -> List[str]:
    """Return original JSON keys for the optimized metric, e.g. pr-auc or logauc."""
    if not metric:
        return []

    return [
        f"optimized_{metric}",
        f"baseline_{metric}",
        f"{metric}_improvement",
        f"test_optimized_{metric}",
        f"test_baseline_{metric}",
        f"test_{metric}_improvement",
    ]


def extract_rows_from_config(config_dir: Path) -> Tuple[List[Dict[str, Any]], List[str]]:  # noqa: C901
    """Extract rows and return threshold columns in first-seen JSON order."""
    rows: List[Dict[str, Any]] = []
    threshold_cols: List[str] = []

    if not config_dir.exists():
        print(f"[WARN] Missing config folder: {config_dir}")
        return rows, threshold_cols

    for target_dir in sorted(config_dir.iterdir()):
        if not target_dir.is_dir():
            continue

        json_path = find_result_json(target_dir)
        if json_path is None:
            # Usually harmless: non-target folders inside a config folder.
            print(f"[WARN] No *_all_structure_results.json found in {target_dir}")
            continue

        try:
            data = load_json(json_path)
        except Exception as e:
            print(f"[WARN] Could not read {json_path}: {e}")
            continue

        thresholds = data.get("thresholds", {})
        if not isinstance(thresholds, dict) or not thresholds:
            print(f"[WARN] No thresholds found in {json_path}")
            continue

        metric = data.get("optimization_metric", "")

        row: Dict[str, Any] = {
            "protein": data.get("protein", target_dir.name),
            "metric": metric,
        }

        # Threshold columns, preserving the JSON order as much as possible.
        for threshold_name, threshold_value in thresholds.items():
            row[threshold_name] = threshold_value
            if threshold_name not in threshold_cols:
                threshold_cols.append(threshold_name)

        # Optimization metric columns, e.g. PR-AUC/logAUC/ROC-AUC with hyphenated names.
        # These are written after thresholds.
        for key in metric_specific_cols(metric):
            if key in data:
                row[key] = data.get(key, "")

        # ROC-AUC comparison columns are written last.
        for key in ROC_AUC_COLS:
            if key in data:
                row[key] = data.get(key, "")

        rows.append(row)

    return rows, threshold_cols


def write_csv(rows: List[Dict[str, Any]], threshold_cols: List[str], out_path: Path) -> None:
    if not rows:
        print(f"[WARN] No rows to write for {out_path.name}")
        return

    all_keys = set().union(*(row.keys() for row in rows))

    first_cols = ["protein", "metric"]
    threshold_cols = [c for c in threshold_cols if c in all_keys]

    # Metric-specific columns depend on the folder. Usually all rows in a CSV have one metric.
    # Preserve this preferred order and include only columns that exist.
    metrics_in_file: List[str] = []
    for row in rows:
        metric = str(row.get("metric", ""))
        if metric and metric not in metrics_in_file:
            metrics_in_file.append(metric)

    optimize_cols: List[str] = []
    for metric in metrics_in_file:
        for col in metric_specific_cols(metric):
            if col in all_keys and col not in optimize_cols:
                optimize_cols.append(col)

    roc_cols = [c for c in ROC_AUC_COLS if c in all_keys]

    used = set(first_cols + threshold_cols + optimize_cols + roc_cols)
    # ROC-AUC columns are kept last, so any unexpected leftover columns are
    # placed between the optimization-metric columns and the ROC-AUC columns.
    leftover_cols = sorted(all_keys - used)

    fieldnames = first_cols + threshold_cols + optimize_cols + leftover_cols + roc_cols

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"[OK] Wrote {len(rows)} rows -> {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract optimized thresholds from the 9 ProDock config folders into 9 CSV files."
    )
    parser.add_argument(
        "--root",
        default=".",
        help="Root directory containing aff_log, aff_pr, aff_roc, cnn_log, etc.",
    )
    parser.add_argument(
        "--output-dir",
        default="threshold_csv",
        help="Directory to save output CSV files.",
    )
    parser.add_argument(
        "--configs",
        nargs="*",
        default=DEFAULT_CONFIGS,
        help="Config folders to process. Default: the 9 expected folders only.",
    )
    args = parser.parse_args()

    root = Path(args.root).resolve()
    out_dir = Path(args.output_dir).resolve()

    print("Processing config folders:")
    for config in args.configs:
        print(f"  - {config}")

    for config in args.configs:
        config_dir = root / config
        rows, threshold_cols = extract_rows_from_config(config_dir)
        out_path = out_dir / f"{config}_thresholds.csv"
        write_csv(rows, threshold_cols, out_path)


if __name__ == "__main__":
    main()
