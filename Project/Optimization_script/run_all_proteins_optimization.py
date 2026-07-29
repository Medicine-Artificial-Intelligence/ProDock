#!/usr/bin/env python3
"""
Batch script to run Optuna optimization for all available proteins.

This script discovers repository-style flat benchmark CSVs or nested campaign
exports and runs the combined GNINA+DiffDock optimization for each target.

Usage:
    python run_all_proteins_optimization.py --dude-labels
    python run_all_proteins_optimization.py --dude-labels --n-trials 100 --parallel 4
    python run_all_proteins_optimization.py --dude-labels --metric pr-auc --parallel 2
"""

import subprocess
import argparse
from pathlib import Path
import logging
from typing import List
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
import sys

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
OPTIMIZER_SCRIPT = Path(__file__).with_name("optuna_combine_all_structure.py")


def generate_temp_skip_compounds() -> List[str]:
    """
    Temporary function to generate compound names to skip (L000 to L014).

    Returns:
        List of compound names to skip
    """
    skip_compounds = []
    for i in range(15):  # 0 to 14 inclusive
        compound_name = f"L{i:03d}"  # Format as L000, L001, etc.
        skip_compounds.append(compound_name)

    logger.info(f"Generated {len(skip_compounds)} compound names to skip: {skip_compounds[:5]}...{skip_compounds[-5:]}")
    return skip_compounds


def discover_proteins(base_dir: str, split: str = "train") -> List[str]:
    """
    Discover all available proteins in the base directory.

    Args:
        base_dir: Base directory containing gnina/ and diffdock/ folders
        split: Input split to discover. Legacy unsplit files are accepted for train.

    Returns:
        List of protein names that have both gnina and diffdock data
    """
    root = Path(base_dir)

    def has_nested_input(protein_dir: Path) -> bool:
        score_dir = protein_dir / "Confidence_score"
        split_csv = score_dir / f"{protein_dir.name}_{split}_final.csv"
        legacy_csv = score_dir / f"{protein_dir.name}_final.csv"
        return split_csv.exists() or (split == "train" and legacy_csv.exists())

    def discover_engine(engine: str) -> set:
        proteins = set()
        nested_dir = root / engine
        if nested_dir.is_dir():
            for protein_dir in nested_dir.iterdir():
                if protein_dir.is_dir() and has_nested_input(protein_dir):
                    proteins.add(protein_dir.name)

        flat_dir = root / f"result_{engine}"
        if flat_dir.is_dir():
            suffix = f"_{split}_final.csv"
            for csv_path in flat_dir.glob(f"*{suffix}"):
                proteins.add(csv_path.name.removesuffix(suffix))
            if split == "train":
                for csv_path in flat_dir.glob("*_final.csv"):
                    name = csv_path.name.removesuffix("_final.csv")
                    if not name.endswith("_train") and not name.endswith("_test"):
                        proteins.add(name)
        return proteins

    gnina_proteins = discover_engine("gnina")
    diffdock_proteins = discover_engine("diffdock")

    if not gnina_proteins:
        logger.error("No GNINA inputs found below %s", root)
    if not diffdock_proteins:
        logger.error("No DiffDock inputs found below %s", root)

    # Return proteins that have both gnina and diffdock data
    common_proteins = sorted(list(gnina_proteins & diffdock_proteins))

    logger.info(f"Found {len(gnina_proteins)} proteins with gnina data")
    logger.info(f"Found {len(diffdock_proteins)} proteins with diffdock data")
    logger.info(f"Found {len(common_proteins)} proteins with both gnina and diffdock data")

    return common_proteins


def discover_proteins_from_split_merged(split_merged_dir: str) -> List[str]:
    """
    Discover proteins from a split_merged directory containing {protein}_train.csv and {protein}_test.csv.

    Args:
        split_merged_dir: Directory with merged train/test CSVs (e.g. split_merged)

    Returns:
        List of protein names that have both _train.csv and _test.csv
    """
    root = Path(split_merged_dir)
    if not root.exists() or not root.is_dir():
        logger.error(f"Split-merged directory not found: {root}")
        return []
    train_files = set(f.stem.replace("_train", "") for f in root.glob("*_train.csv"))
    test_files = set(f.stem.replace("_test", "") for f in root.glob("*_test.csv"))
    common = sorted(list(train_files & test_files))
    logger.info(f"Found {len(common)} proteins in {split_merged_dir} (with both _train.csv and _test.csv)")
    return common


def run_single_optimization(protein: str, args) -> dict:  # noqa: C901
    """
    Run optimization for a single protein.

    Args:
        protein: Protein name
        args: Command line arguments

    Returns:
        Dictionary with results
    """
    start_time = time.time()

    # When using split_merged, pass --data-dir so optuna loads from merged CSVs; else pass --base-dir
    data_dir = getattr(args, "split_merged_dir", None)
    cmd = [
        sys.executable,
        str(OPTIMIZER_SCRIPT),
        "--protein",
        protein,
        "--base-dir",
        args.base_dir,
        "--scoring-metric",
        args.scoring_metric,
        "--metric",
        args.metric,
        "--n-trials",
        str(args.n_trials),
        "--n-jobs",
        str(args.n_jobs),
        "--top-k",
        str(args.top_k),
        "--output-dir",
        args.output_dir,
        "--log-level",
        args.log_level,
        "--split",
        args.split,
    ]
    if data_dir:
        cmd.extend(["--data-dir", data_dir])

    if args.save_study:
        cmd.append("--save-study")

    if getattr(args, "evaluate_test", False) and args.split == "train":
        cmd.append("--eval-on-test")

    if args.activity_column:
        cmd.extend(["--activity-column", args.activity_column])
    if args.dude_labels:
        cmd.append("--dude-labels")

    # Add temporary skip compounds if enabled
    if args.skip_temp_compounds:
        skip_compounds = generate_temp_skip_compounds()
        cmd.extend(["--skip-compounds"] + skip_compounds)

    # Add custom skip compounds if provided
    if args.skip_compounds:
        cmd.extend(["--skip-compounds"] + args.skip_compounds)

    logger.info(f"Starting optimization for {protein} with metric {args.metric}")
    logger.debug(f"Command: {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=args.timeout)

        end_time = time.time()
        duration = end_time - start_time

        if result.returncode == 0:
            logger.info(f"{protein} completed successfully in {duration:.1f}s")

            # Try to parse results from JSON file
            results_file = Path(args.output_dir) / protein / f"{protein}_all_structure_results.json"
            if results_file.exists():
                with open(results_file, "r") as f:
                    optimization_results = json.load(f)
                out = {
                    "protein": protein,
                    "status": "success",
                    "duration": duration,
                    "metric": args.metric,
                    f"{args.metric}": optimization_results.get(f"optimized_{args.metric}", 0.0),
                    "roc_auc": optimization_results.get("optimized_roc_auc", 0.0),
                    f"{args.metric}_improvement": optimization_results.get(f"{args.metric}_improvement", 0.0),
                    "roc_auc_improvement": optimization_results.get("roc_auc_improvement", 0.0),
                    "n_threshold_metrics": optimization_results.get("total_threshold_metrics", 0),
                    "n_scoring_metrics": optimization_results.get("n_scoring_metrics", 0),
                    "thresholds": optimization_results.get("thresholds", {}),
                    "results_file": str(results_file),
                }
                if "test_optimized_roc_auc" in optimization_results:
                    out["test_optimized_roc_auc"] = optimization_results["test_optimized_roc_auc"]
                    out["test_baseline_roc_auc"] = optimization_results.get("test_baseline_roc_auc")
                    out["test_roc_auc_improvement"] = optimization_results.get("test_roc_auc_improvement")
                # Also store test values for the optimization metric (e.g. test_optimized_pr-auc)
                test_opt_key = f"test_optimized_{args.metric}"
                if test_opt_key in optimization_results:
                    out[test_opt_key] = optimization_results[test_opt_key]
                    out[f"test_baseline_{args.metric}"] = optimization_results.get(f"test_baseline_{args.metric}")
                    out[f"test_{args.metric}_improvement"] = optimization_results.get(f"test_{args.metric}_improvement")
                return out
            else:
                return {
                    "protein": protein,
                    "status": "success",
                    "duration": duration,
                    "metric": args.metric,
                    f"{args.metric}": 0.0,
                    "roc_auc": 0.0,
                    f"{args.metric}_improvement": 0.0,
                    "roc_auc_improvement": 0.0,
                    "n_threshold_metrics": 0,
                    "n_scoring_metrics": 0,
                    "results_file": None,
                }
        else:
            logger.error(f"{protein} failed with return code {result.returncode}")
            logger.error(f"Error output: {result.stderr}")
            return {
                "protein": protein,
                "status": "failed",
                "duration": duration,
                "error": result.stderr[:500],  # First 500 chars of error
                "return_code": result.returncode,
            }

    except subprocess.TimeoutExpired:
        logger.error(f"{protein} timed out after {args.timeout}s")
        return {
            "protein": protein,
            "status": "timeout",
            "duration": args.timeout,
            "error": f"Timed out after {args.timeout}s",
        }
    except Exception as e:
        logger.error(f"{protein} failed with exception: {str(e)}")
        return {"protein": protein, "status": "exception", "duration": time.time() - start_time, "error": str(e)}


def save_batch_results(results: List[dict], output_file: str):
    """Save batch optimization results to JSON file."""

    # Calculate summary statistics
    successful = [r for r in results if r["status"] == "success"]
    failed = [r for r in results if r["status"] != "success"]

    # Get the metric used (should be the same for all runs)
    metric = successful[0]["metric"] if successful else "roc-auc"

    summary = {
        "total_proteins": len(results),
        "successful": len(successful),
        "failed": len(failed),
        "success_rate": len(successful) / len(results) if results else 0,
        "avg_duration": sum(r["duration"] for r in successful) / len(successful) if successful else 0,
        "metric": metric,
        f"avg_{metric}": sum(r.get(metric, 0) for r in successful) / len(successful) if successful else 0,
        "avg_roc_auc": sum(r.get("roc_auc", 0) for r in successful) / len(successful) if successful else 0,
        f"avg_{metric}_improvement": (
            sum(r.get(f"{metric}_improvement", 0) for r in successful) / len(successful) if successful else 0
        ),
        "avg_roc_auc_improvement": (
            sum(r.get("roc_auc_improvement", 0) for r in successful) / len(successful) if successful else 0
        ),
    }
    with_test = [r for r in successful if "test_optimized_roc_auc" in r]
    if with_test:
        summary["avg_test_optimized_roc_auc"] = sum(r["test_optimized_roc_auc"] for r in with_test) / len(with_test)
        summary["avg_test_baseline_roc_auc"] = sum(r.get("test_baseline_roc_auc", 0) for r in with_test) / len(
            with_test
        )
        summary["avg_test_roc_auc_improvement"] = sum(r.get("test_roc_auc_improvement", 0) for r in with_test) / len(
            with_test
        )
        # Test optimization metric (e.g. avg test PR-AUC when metric is pr-auc)
        test_opt_key = f"test_optimized_{metric}"
        if test_opt_key in with_test[0]:
            summary[f"avg_test_optimized_{metric}"] = sum(r[test_opt_key] for r in with_test) / len(with_test)
            summary[f"avg_test_baseline_{metric}"] = sum(r.get(f"test_baseline_{metric}", 0) for r in with_test) / len(
                with_test
            )
            summary[f"avg_test_{metric}_improvement"] = sum(
                r.get(f"test_{metric}_improvement", 0) for r in with_test
            ) / len(with_test)

    batch_results = {"summary": summary, "individual_results": results, "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}

    with open(output_file, "w") as f:
        json.dump(batch_results, f, indent=2)

    logger.info(f"Batch results saved to: {output_file}")


def print_summary(results: List[dict]):
    """Print summary of batch optimization results."""

    successful = [r for r in results if r["status"] == "success"]
    failed = [r for r in results if r["status"] != "success"]

    # Get the metric used (should be the same for all runs)
    metric = successful[0]["metric"] if successful else "roc-auc"
    metric_label = {"roc-auc": "ROC-AUC", "pr-auc": "PR-AUC", "logauc": "logAUC"}.get(metric, metric.upper())

    print(f"\n{'='*80}")
    print("BATCH OPTIMIZATION SUMMARY")
    print(f"{'='*80}")
    print(f"Optimization metric: {metric_label}")
    print(f"Total proteins: {len(results)}")
    print(f"Successful: {len(successful)} ({len(successful)/len(results)*100:.1f}%)")
    print(f"Failed: {len(failed)} ({len(failed)/len(results)*100:.1f}%)")

    if successful:
        avg_duration = sum(r["duration"] for r in successful) / len(successful)
        avg_metric = sum(r.get(metric, 0) for r in successful) / len(successful)
        avg_metric_improvement = sum(r.get(f"{metric}_improvement", 0) for r in successful) / len(successful)
        avg_roc_auc = sum(r.get("roc_auc", 0) for r in successful) / len(successful)
        avg_roc_auc_improvement = sum(r.get("roc_auc_improvement", 0) for r in successful) / len(successful)

        print("\nSuccessful optimizations:")
        print(f"  Average duration: {avg_duration:.1f}s")
        print(f"  Average {metric_label}: {avg_metric:.4f}")
        print(f"  Average {metric_label} improvement: {avg_metric_improvement:+.4f}")
        # ROC-AUC under the same thresholds (we did not optimize ROC-AUC; only report when metric is PR-AUC or logAUC)
        if metric != "roc-auc":
            print(f"  Average ROC-AUC (with same thresholds): {avg_roc_auc:.4f}")
            print(f"  Average ROC-AUC improvement (with same thresholds): {avg_roc_auc_improvement:+.4f}")
        if avg_metric <= 0.5001 and avg_roc_auc <= 0.5001:
            print(
                "  Optimized metrics are ~0.5 for all proteins: often means constant "
                "scores (e.g. no conformation passed thresholds -> all get global worst). "
                "Check threshold bounds or data."
            )
        with_test = [r for r in successful if "test_optimized_roc_auc" in r]
        if with_test:
            test_opt_key = f"test_optimized_{metric}"
            has_test_opt_metric = test_opt_key in with_test[0]
            print("\nTest set (thresholds from train):")
            print(f"  Proteins with test eval: {len(with_test)}")
            # Primary: show the optimization metric on test (PR-AUC, logAUC, or ROC-AUC)
            if has_test_opt_metric:
                avg_test_opt = sum(r[test_opt_key] for r in with_test) / len(with_test)
                avg_test_base = sum(r.get(f"test_baseline_{metric}", 0) for r in with_test) / len(with_test)
                avg_test_imp = sum(r.get(f"test_{metric}_improvement", 0) for r in with_test) / len(with_test)
                print(f"  Average test {metric_label} (optimized): {avg_test_opt:.4f}")
                print(f"  Average test {metric_label} (baseline):  {avg_test_base:.4f}")
                print(f"  Average test {metric_label} improvement: {avg_test_imp:+.4f}")
            # ROC-AUC under same thresholds (not optimized for ROC-AUC when metric is PR-AUC or logAUC)
            if not has_test_opt_metric or metric != "roc-auc":
                avg_test_roc_opt = sum(r["test_optimized_roc_auc"] for r in with_test) / len(with_test)
                avg_test_roc_base = sum(r.get("test_baseline_roc_auc", 0) for r in with_test) / len(with_test)
                avg_test_roc_imp = sum(r.get("test_roc_auc_improvement", 0) for r in with_test) / len(with_test)
                print(f"  Average test ROC-AUC (with same thresholds): {avg_test_roc_opt:.4f}")
                print(f"  Average test ROC-AUC (baseline):  {avg_test_roc_base:.4f}")
                print(f"  Average test ROC-AUC improvement (with same thresholds): {avg_test_roc_imp:+.4f}")

        # Top 5 performers by the selected metric (print the optimized metric value)
        top_performers = sorted(successful, key=lambda x: x.get(metric, 0), reverse=True)[:5]
        print(f"\nTop 5 performers by {metric_label}:")
        for i, result in enumerate(top_performers, 1):
            val = result.get(metric, 0)
            line = f"  {i}. {result['protein']}: {metric_label}: {val:.4f}"
            if metric != "roc-auc":
                line += f" (ROC-AUC: {result.get('roc_auc', 0):.4f})"
            print(line)

    if failed:
        print("\nFailed optimizations:")
        for result in failed:
            print(f"  {result['protein']}: {result['status']} - {result.get('error', 'Unknown error')[:100]}")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Batch GNINA+DiffDock Optuna reranking", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Data arguments
    parser.add_argument(
        "--base-dir",
        type=str,
        default="Project/benchmark",
        help=(
            "Base directory containing repository-style result_gnina/ and "
            "result_diffdock/ folders, or nested gnina/ and diffdock/ "
            "exports (ignored if --split-merged-dir is set)"
        ),
    )

    parser.add_argument(
        "--split-merged-dir",
        type=str,
        default=None,
        metavar="DIR",
        help="Use merged train/test from DIR (e.g. split_merged): expects "
        "{protein}_train.csv and {protein}_test.csv; optimize on train, "
        "optional --evaluate-test on test",
    )

    # Protein selection
    parser.add_argument(
        "--proteins", nargs="*", help="Specific proteins to process (if not specified, process all found)"
    )

    parser.add_argument("--exclude-proteins", nargs="*", default=[], help="Proteins to exclude from processing")

    parser.add_argument(
        "--scoring-metric",
        type=str,
        choices=["affinity", "cnnaffinity", "cnn-combined"],
        default="affinity",
        help="Metric for final scoring: 'affinity' (force affinity), "
        "'cnnaffinity' (force CNNaffinity), 'cnn-combined' (CNNpose * CNNaffinity)",
    )

    parser.add_argument(
        "--metric",
        type=str,
        choices=["roc-auc", "pr-auc", "logauc"],
        default="roc-auc",
        help="Performance metric to optimize: ROC-AUC, PR-AUC (precision-recall), or logAUC (early enrichment)",
    )

    # Optimization arguments
    parser.add_argument("--n-trials", type=int, default=100, help="Number of Optuna trials per protein")

    parser.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help="Number of parallel jobs for Optuna trials per protein (1 = sequential, -1 = use all CPUs)",
    )

    parser.add_argument("--top-k", type=int, default=10, help="Number of top conformations to consider per molecule")

    parser.add_argument("--timeout", type=int, default=3600, help="Timeout per protein in seconds (default: 1 hour)")

    # Parallelization
    parser.add_argument(
        "--parallel", type=int, default=1, help="Number of proteins to process in parallel (default: 1, sequential)"
    )

    # Temporary filtering
    parser.add_argument(
        "--skip-temp-compounds", action="store_true", help="Temporary option to skip compounds L000-L014"
    )

    parser.add_argument("--skip-compounds", nargs="*", default=[], help="Custom list of compound names to skip/exclude")

    # Output arguments
    parser.add_argument("--output-dir", type=str, default="results_all_structure", help="Directory to save results")

    parser.add_argument(
        "--batch-results-file",
        type=str,
        default="batch_optimization_results.json",
        help="File to save batch results summary",
    )

    parser.add_argument("--save-study", action="store_true", help="Save Optuna study objects for each protein")

    # Logging
    parser.add_argument(
        "--log-level", type=str, choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="INFO", help="Logging level"
    )

    parser.add_argument(
        "--dry-run", action="store_true", help="Show what would be processed without running optimization"
    )

    parser.add_argument(
        "--activity-column",
        type=str,
        default="Active",
        help="Embedded CSV column containing binary activity labels",
    )

    parser.add_argument(
        "--dude-labels",
        action="store_true",
        help=(
            "Use the DUD-E identifier convention when no activity column is "
            "present: ZINC*=decoy and every other identifier=active"
        ),
    )

    parser.add_argument(
        "--split",
        type=str,
        choices=["train", "test"],
        default="train",
        help="Which split to use: 'train' or 'test' (default: train)",
    )

    parser.add_argument(
        "--evaluate-test",
        action="store_true",
        help="After optimizing on train, apply best thresholds to test set "
        "and add test_* metrics to each protein's results JSON",
    )

    return parser.parse_args()


def main():
    """Main function to run batch optimization."""
    args = parse_args()

    # Set logging level
    logging.getLogger().setLevel(getattr(logging, args.log_level))

    # Discover available proteins (from split_merged or from base_dir)
    if getattr(args, "split_merged_dir", None):
        logger.info(f"Discovering proteins from split_merged: {args.split_merged_dir}")
        all_proteins = discover_proteins_from_split_merged(args.split_merged_dir)
        if not all_proteins:
            logger.error("No proteins found in split_merged (need *_train.csv and *_test.csv per protein)!")
            return 1
    else:
        logger.info(f"Discovering proteins in: {args.base_dir}")
        all_proteins = discover_proteins(args.base_dir, split=args.split)
        if not all_proteins:
            logger.error("No proteins found with both gnina and diffdock data!")
            return 1

    # Filter proteins if specific ones were requested
    if args.proteins:
        requested_proteins = set(args.proteins)
        available_proteins = set(all_proteins)
        missing_proteins = requested_proteins - available_proteins

        if missing_proteins:
            logger.warning(f"Requested proteins not found: {', '.join(missing_proteins)}")

        proteins_to_process = sorted(list(requested_proteins & available_proteins))
    else:
        proteins_to_process = all_proteins

    # Exclude proteins if specified
    if args.exclude_proteins:
        exclude_set = set(args.exclude_proteins)
        proteins_to_process = [p for p in proteins_to_process if p not in exclude_set]
        logger.info(f"Excluded {len(exclude_set)} proteins")

    if not proteins_to_process:
        logger.error("No proteins to process after filtering!")
        return 1

    logger.info(f"Will process {len(proteins_to_process)} proteins: {', '.join(proteins_to_process)}")
    logger.info(f"Using optimization metric: {args.metric.upper()}")

    if args.dry_run:
        logger.info("Dry run mode - not running actual optimizations")
        for protein in proteins_to_process:
            logger.info(f"Would process: {protein}")
        return 0

    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Run optimizations
    start_time = time.time()

    if args.parallel <= 1:
        # Sequential processing
        logger.info("Running optimizations sequentially")
        results = []
        for protein in proteins_to_process:
            result = run_single_optimization(protein, args)
            results.append(result)
    else:
        # Parallel processing
        logger.info(f"Running optimizations with {args.parallel} parallel processes")
        results = []

        with ProcessPoolExecutor(max_workers=args.parallel) as executor:
            # Submit all jobs
            future_to_protein = {
                executor.submit(run_single_optimization, protein, args): protein for protein in proteins_to_process
            }

            # Collect results as they complete
            for future in as_completed(future_to_protein):
                result = future.result()
                results.append(result)

                # Print progress
                completed = len(results)
                total = len(proteins_to_process)
                logger.info(f"Progress: {completed}/{total} proteins completed")

    total_duration = time.time() - start_time

    # Sort results by protein name for consistent output
    results.sort(key=lambda x: x["protein"])

    # Save batch results
    save_batch_results(results, args.batch_results_file)

    # Print summary
    print_summary(results)

    print(f"\nTotal batch duration: {total_duration:.1f}s")
    print(f"Batch results saved to: {args.batch_results_file}")

    # Return appropriate exit code
    failed_count = len([r for r in results if r["status"] != "success"])
    return 0 if failed_count == 0 else 1


if __name__ == "__main__":
    exit(main())
