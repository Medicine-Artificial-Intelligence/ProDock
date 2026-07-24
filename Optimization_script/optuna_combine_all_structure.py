#!/usr/bin/env python3
"""
Optuna-based reranking of combined GNINA + DiffDock docking results.

AllStructureReranker loads and merges the per-target gnina and diffdock CSVs,
assigns active/inactive labels, and computes ROC-AUC, PR-AUC and logAUC.
AllStructureOptunaOptimizer tunes per-metric thresholds with Optuna to maximise
the chosen metric.

Expected directory structure:
    all/
    ├── gnina/{target}/Confidence_score/{target}_final.csv
    └── diffdock/{target}/Confidence_score/{target}_final.csv

Usage:
    python optuna_combine_all_structure.py --protein ABL1 --base-dir all
    python optuna_combine_all_structure.py --protein EGFR --base-dir all --n-trials 200
    python optuna_combine_all_structure.py --protein ABL1 --base-dir all --metric pr-auc
"""

import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve, average_precision_score
from typing import Dict, List, Optional, Any
import logging
from pathlib import Path
import warnings
import argparse
warnings.filterwarnings('ignore')


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_temp_active_labels(csv_file: str, activity_column: str, skip_compounds: List[str] = None) -> Dict[str, int]:
    """
    Temporary function to load active labels from CSV file based on chosen column.

    Args:
        csv_file: Path to CSV file containing compound activity data
        activity_column: Column name to use for activity labels (e.g., 'MSSA', 'MRSA')
        skip_compounds: List of compound names to skip/exclude

    Returns:
        Dictionary mapping compound names to activity labels (0 or 1)
    """
    if not Path(csv_file).exists():
        logger.error(f"Activity CSV file not found: {csv_file}")
        return {}

    try:
        # Load the CSV file
        activity_df = pd.read_csv(csv_file)
        logger.info(f"Loaded activity data from {csv_file}")
        logger.info(f"Available columns: {list(activity_df.columns)}")
        logger.info(f"Using activity column: {activity_column}")

        # Check if the specified column exists
        if activity_column not in activity_df.columns:
            logger.error(f"Column '{activity_column}' not found in {csv_file}")
            logger.error(f"Available columns: {list(activity_df.columns)}")
            return {}

        # Check if Compounds column exists
        if 'Compounds' not in activity_df.columns:
            logger.error(f"'Compounds' column not found in {csv_file}")
            return {}

        # Create mapping dictionary
        activity_mapping = {}
        for _, row in activity_df.iterrows():
            compound = row['Compounds']
            activity = int(row[activity_column])  # Ensure integer (0 or 1)

            # Skip compounds if specified
            if skip_compounds and compound in skip_compounds:
                logger.debug(f"Skipping compound {compound} as requested")
                continue

            activity_mapping[compound] = activity

        # Log statistics
        total_compounds = len(activity_mapping)
        active_compounds = sum(1 for v in activity_mapping.values() if v == 1)
        inactive_compounds = total_compounds - active_compounds

        logger.info(f"Loaded activity labels for {total_compounds} compounds from {activity_column}:")
        logger.info(f"  Active compounds: {active_compounds}")
        logger.info(f"  Inactive compounds: {inactive_compounds}")

        if skip_compounds:
            skipped_count = len([c for c in skip_compounds if c in activity_df['Compounds'].values])
            logger.info(f"  Skipped compounds: {skipped_count}")

        return activity_mapping

    except Exception as e:
        logger.error(f"Error loading activity labels from {csv_file}: {str(e)}")
        return {}


class AllStructureReranker:
    """
    A class to rerank molecular conformations using combined metrics from
    the organized /all directory structure (gnina + diffdock) optimized
    via Optuna to maximize ROC-AUC.
    """

    # Column prefixes in merged (split_merged) CSVs: gnina vs diffdock
    GNINA_RANK_PREFIXES = ('Affinity', 'CNNpose', 'CNNaffinity', 'Solvation', 'Similarity-type1', 'Similarity-type2')
    DIFFDOCK_RANK_PREFIXES = ('Confidence_score', 'Occupation', '%Occupation', '%atoms')

    def __init__(self,
                 protein: str,
                 scoring_metric: str,
                 base_dir: str = "all",
                 metric: str = "roc-auc",
                 activity_column: str = 'Active',
                 skip_compounds: List[str] = None,
                 split: str = "train",
                 data_dir: Optional[str] = None):
        """
        Initialize the reranker with data from the /all directory structure or from split_merged.

        Args:
            protein: Target protein name (e.g., ABL1, EGFR)
            base_dir: Base directory containing gnina/ and diffdock/ folders (used when data_dir is None)
            scoring_metric: Metric for ROC-AUC calculation ("auto", "affinity", "cnnaffinity", "cnn-combined")
            metric: Performance metric to optimize ("roc-auc", "pr-auc", "logauc")
            activity_column: Column name in the activity CSV to use for activity labels.
            skip_compounds: List of compound names to skip/exclude.
            split: Which split to use ("train" or "test")
            data_dir: If set (e.g. "split_merged"), load from
                {data_dir}/{protein}_{split}.csv (merged train/test) instead of base_dir gnina/diffdock.
        """
        self.protein = protein
        self.base_dir = base_dir
        self.data_dir = Path(data_dir) if data_dir else None
        self.scoring_metric = scoring_metric.lower()
        self.metric = metric.lower()
        self.data = None
        self.molecules = None
        self.gnina_metrics = []
        self.diffdock_metrics = []
        self.all_metrics = []
        self.activity_column = activity_column
        self.skip_compounds = skip_compounds or []
        self.split = split.lower()
        if self.split not in ("train", "test"):
            raise ValueError(f"Invalid split: {self.split}. Must be 'train' or 'test'.")
        if self.data_dir is not None:
            self.gnina_csv_path = self.data_dir / f"{protein}_{self.split}.csv"
            self.diffdock_csv_path = None
            self.load_and_combine_from_merged_dir()
        else:
            gnina_file = f"{protein}_train_final.csv" if self.split == "train" else f"{protein}_test_final.csv"
            self.gnina_csv_path = Path(base_dir) / "gnina" / protein / "Confidence_score" / gnina_file
            self.diffdock_csv_path = Path(base_dir) / "diffdock" / protein / "Confidence_score" / gnina_file
            self.load_and_combine_data()

    def load_and_combine_data(self):
        """Load and combine data from gnina and diffdock CSV files."""
        logger.info(f"Loading data for protein: {self.protein}")
        # logger.info(f"Gnina CSV: {self.gnina_csv_path}")
        # logger.info(f"DiffDock CSV: {self.diffdock_csv_path}")

        # Check if files exist
        if not self.gnina_csv_path.exists():
            raise FileNotFoundError(f"Gnina CSV file not found: {self.gnina_csv_path}")
        if self.diffdock_csv_path is not None and not self.diffdock_csv_path.exists():
            raise FileNotFoundError(f"DiffDock CSV file not found: {self.diffdock_csv_path}")

        # Load data
        gnina_data = pd.read_csv(self.gnina_csv_path)
        diffdock_data = pd.read_csv(self.diffdock_csv_path)

        # Debug: Print available columns
        # logger.info(f"Gnina file columns: {list(gnina_data.columns)}")
        # logger.info(f"DiffDock file columns: {list(diffdock_data.columns)}")

        # Process gnina data
        gnina_reshaped = self._reshape_rank_data(gnina_data, "gnina")
        diffdock_reshaped = self._reshape_rank_data(diffdock_data, "diffdock")

        # Merge the datasets on compound and rank
        self.data = pd.merge(
            gnina_reshaped,
            diffdock_reshaped,
            on=['Compounds', 'rank', 'molecule'],
            how='inner',
            suffixes=('_gnina', '_diffdock')
        )
        # Debug: Print label distribution after merge
        # if self.activity_column in self.data.columns:
        #     print(f"[DEBUG] After merge: '{self.activity_column}' value counts:")
        #     print(self.data[self.activity_column].value_counts())
        # elif f"{self.activity_column}_gnina" in self.data.columns:
        #     print(f"[DEBUG] After merge: '{self.activity_column}_gnina' value counts:")
        #     print(self.data[f"{self.activity_column}_gnina"].value_counts())
        # elif f"{self.activity_column}_diffdock" in self.data.columns:
        #     print(f"[DEBUG] After merge: '{self.activity_column}_diffdock' value counts:")
        #     print(self.data[f"{self.activity_column}_diffdock"].value_counts())

        # Filter out skipped compounds
        if self.skip_compounds:
            before_count = len(self.data)
            self.data = self.data[~self.data['molecule'].isin(self.skip_compounds)]
            after_count = len(self.data)
            logger.info(
                f"Filtered out {before_count - after_count} conformations "
                f"from {len(self.skip_compounds)} skipped compounds")

        # Check for activity column (may have been renamed during merge)
        activity_column_found = False

        # Try direct match first
        if self.activity_column in self.data.columns:
            activity_column_found = True
            logger.info(f"Found '{self.activity_column}' column in merged data")

        # Try with suffixes (_gnina or _diffdock)
        elif f"{self.activity_column}_gnina" in self.data.columns:
            # Prefer gnina version if both exist
            self.data[self.activity_column] = self.data[f"{self.activity_column}_gnina"]
            activity_column_found = True
            logger.info(f"Using '{self.activity_column}_gnina' as activity column")
        elif f"{self.activity_column}_diffdock" in self.data.columns:
            self.data[self.activity_column] = self.data[f"{self.activity_column}_diffdock"]
            activity_column_found = True
            logger.info(f"Using '{self.activity_column}_diffdock' as activity column")

        # If still not found, raise error
        if not activity_column_found:
            raise KeyError(
                f"Activity column '{self.activity_column}' not found in data. "
                f"Available columns: {list(self.data.columns)}")

        # Use activity column directly for is_active
        self.data['is_active'] = self.data[self.activity_column]

        # Extract molecule names
        self.molecules = self.data['molecule'].unique()

        # Identify available metrics
        self._identify_metrics()

        logger.info(f"Combined {len(self.data)} conformations for {len(self.molecules)} molecules")

        # Count unique active and decoy molecules
        active_molecules = self.data[self.data['is_active'] == 1]['molecule'].nunique()
        decoy_molecules = self.data[self.data['is_active'] == 0]['molecule'].nunique()

        logger.info(f"Active molecules: {active_molecules}")
        logger.info(f"Decoy molecules: {decoy_molecules}")
        # logger.info(f"Total metrics available: {len(self.all_metrics)}")
        logger.info(f"Gnina metrics: {len(self.gnina_metrics)} - {self.gnina_metrics}")
        logger.info(f"DiffDock metrics: {len(self.diffdock_metrics)} - {self.diffdock_metrics}")

    def load_and_combine_from_merged_dir(self):
        """Load a single merged CSV from data_dir
        (e.g. split_merged/{protein}_{split}.csv) and reshape to long format."""
        merged_path = self.gnina_csv_path
        if not merged_path.exists():
            raise FileNotFoundError(f"Merged CSV not found: {merged_path}")
        logger.info(f"Loading merged data from {merged_path}")
        merged_df = pd.read_csv(merged_path)
        self.data = self._reshape_merged_wide_to_long(merged_df)
        # Same activity/skip/molecules/identify flow as after load_and_combine_data merge
        if self.skip_compounds:
            before_count = len(self.data)
            self.data = self.data[~self.data['molecule'].isin(self.skip_compounds)]
            logger.info(
                f"Filtered out {before_count - len(self.data)} conformations "
                f"from {len(self.skip_compounds)} skipped compounds")
        activity_column_found = False
        if self.activity_column in self.data.columns:
            activity_column_found = True
        elif f"{self.activity_column}_gnina" in self.data.columns:
            self.data[self.activity_column] = self.data[f"{self.activity_column}_gnina"]
            activity_column_found = True
        elif f"{self.activity_column}_diffdock" in self.data.columns:
            self.data[self.activity_column] = self.data[f"{self.activity_column}_diffdock"]
            activity_column_found = True
        if not activity_column_found:
            raise KeyError(
                f"Activity column '{self.activity_column}' not found. "
                f"Available: {[c for c in self.data.columns if 'ctive' in c.lower()]}")
        self.data['is_active'] = self.data[self.activity_column]
        self.molecules = self.data['molecule'].unique()
        self._identify_metrics()
        logger.info(f"Combined {len(self.data)} conformations for {len(self.molecules)} molecules (from merged)")
        active_molecules = self.data[self.data['is_active'] == 1]['molecule'].nunique()
        decoy_molecules = self.data[self.data['is_active'] == 0]['molecule'].nunique()
        logger.info(f"Active molecules: {active_molecules}, Decoy molecules: {decoy_molecules}")

    def _reshape_merged_wide_to_long(self, merged_df: pd.DataFrame) -> pd.DataFrame:
        """Reshape merged wide CSV (one row per compound, columns like
        Affinity_rank1, Confidence_score_rank1) to long format with
        _gnina/_diffdock suffixes.
        Final_* and Clashing* columns are excluded: Final is rank-only;
        Clashing is typically binary/count, not used as a continuous threshold.
        """
        rank_cols = [
            c for c in merged_df.columns
            if '_rank' in c and not c.startswith('Final_') and not c.startswith('Clashing')]
        ranks = sorted(set(int(c.rsplit('_rank', 1)[-1]) for c in rank_cols if c.rsplit('_rank', 1)[-1].isdigit()))
        if not ranks:
            raise ValueError("No rank columns found in merged CSV")
        rows = []
        for _, row in merged_df.iterrows():
            compounds = row['Compounds']
            molecule = compounds  # or strip suffix if needed
            for r in ranks:
                rank_suffix = f"_rank{r}"
                out = {'Compounds': compounds, 'rank': r, 'molecule': molecule}
                for col in merged_df.columns:
                    if not col.endswith(rank_suffix) or col.startswith('Final_') or col.startswith('Clashing'):
                        continue
                    base = col.replace(rank_suffix, '')
                    if base in self.GNINA_RANK_PREFIXES:
                        out[f"{base}_gnina"] = row[col]
                    elif base in self.DIFFDOCK_RANK_PREFIXES:
                        out[f"{base}_diffdock"] = row[col]
                # Activity: same for all ranks (copy all activity-related columns so resolution works)
                for ac in (self.activity_column,
                           f"{self.activity_column}_gnina",
                           f"{self.activity_column}_diffdock", 'Active'):
                    if ac in merged_df.columns:
                        out[ac] = row[ac]
                rows.append(out)
        out_df = pd.DataFrame(rows)
        return out_df

    def _reshape_rank_data(self, data: pd.DataFrame, source: str) -> pd.DataFrame:
        """Reshape rank-based data to long format.

        Example output (no is_active yet):
        Compounds	Affinity_gnina	CNNaffinity_gnina	rank	molecule
        CHEMBL123	-8.5	        0.72	            1	    CHEMBL123
        ZINC000001	-7.2	        0.55	            1	    ZINC000001
        CHEMBL123	-8.0	        0.68	            2	    CHEMBL123
        ZINC000001	-7.0	        0.50	            2	    ZINC000001

        """
        reshaped_data = []

        # Get all rank columns
        rank_columns = [col for col in data.columns if '_rank' in col]
        ranks = sorted(list(set([int(col.split('_rank')[1]) for col in rank_columns])))

        logger.info(f"Found ranks {ranks} in {source} data")

        # Check if activity column exists in original data
        has_activity_column = self.activity_column in data.columns
        if has_activity_column:
            logger.info(f"Found {self.activity_column} column in original {source} data")

        for rank in ranks:
            rank_suffix = f"_rank{rank}"

            # Find all metrics for this rank; exclude Final_* (rank-only) and
            # Clashing* (binary/count, not used as continuous threshold)
            rank_metrics = {}
            for col in data.columns:
                if col.endswith(rank_suffix) and not col.startswith('Final_') and not col.startswith('Clashing'):
                    metric_name = col.replace(rank_suffix, '')
                    rank_metrics[f"{metric_name}_{source}"] = col

            # Always include Compounds column
            rank_metrics['Compounds'] = 'Compounds'

            # Include activity column if it exists
            if has_activity_column:
                rank_metrics[self.activity_column] = self.activity_column

            # Create subset for this rank
            available_cols = [col for col in rank_metrics.values() if col in data.columns]
            rank_data = data[available_cols].copy()

            # Rename columns
            new_names = {}
            for new_name, old_name in rank_metrics.items():
                if old_name in data.columns:
                    new_names[old_name] = new_name

            rank_data = rank_data.rename(columns=new_names)
            rank_data['rank'] = rank

            # Extract molecule names
            rank_data['molecule'] = rank_data['Compounds'].str.replace(r'_rank_\d+$', '', regex=True)

            reshaped_data.append(rank_data)

        return pd.concat(reshaped_data, ignore_index=True)

    def _identify_metrics(self):  # noqa: C901
        """Identify available metrics from the combined dataset.
        All-NaN (empty) columns are skipped, so e.g. empty Solvation is not used as a threshold metric.
        """
        # Find all columns that are metrics (exclude system columns)
        exclude_cols = ['Compounds', 'rank', 'molecule', 'is_active', self.activity_column]

        # Initialize metrics lists
        self.gnina_metrics = []
        self.diffdock_metrics = []
        self.all_metrics = []

        # Identify scoring metrics (Affinity and CNNaffinity columns)
        self.scoring_metrics = []

        # Only include columns with at least one non-NaN (skip empty metrics like Solvation when not computed)
        for col in self.data.columns:
            if col not in exclude_cols and not col.endswith('_rank') and not pd.isna(self.data[col]).all():
                if '_gnina' in col:
                    self.gnina_metrics.append(col)
                    if 'affinity' in col.lower():
                        self.scoring_metrics.append(col)
                elif '_diffdock' in col:
                    self.diffdock_metrics.append(col)
                    # Note: Affinity and CNNaffinity metrics only exist in gnina data

                self.all_metrics.append(col)

        # Extract threshold metrics (all metrics except scoring metrics and activity columns)
        threshold_metrics = []
        for metric in self.all_metrics:
            # Skip any "Active" columns (both gnina and diffdock)
            if 'active' in metric.lower():
                continue

            # Exclude Affinity and CNNaffinity metrics from thresholds
            if metric not in self.scoring_metrics:
                # If using cnn-combined scoring, also exclude CNNpose metrics from gnina
                if self.scoring_metric == "cnn-combined" and 'cnnpose' in metric.lower() and '_gnina' in metric:
                    continue
                threshold_metrics.append(metric)

        # Remove any metrics with all NaN values (e.g. empty Solvation_gnina)
        self.threshold_metrics = [m for m in threshold_metrics if not pd.isna(self.data[m]).all()]

        # Log excluded metrics
        logger.info(f"Excluding affinity metrics from thresholds: {self.scoring_metrics}")
        logger.info(
            f"Excluding activity columns from thresholds: "
            f"{[col for col in self.data.columns if 'active' in col.lower()]}")

        # If using cnn-combined, ensure we have CNNpose metrics for calculation
        cnnpose_metrics = [m for m in self.gnina_metrics if 'cnnpose' in m.lower()]
        if self.scoring_metric == "cnn-combined":
            logger.info(f"Also excluding CNNpose metrics for cnn-combined scoring: {cnnpose_metrics}")
            # CNNpose only exists in gnina data
            if not cnnpose_metrics:
                raise ValueError("CNNpose metrics not found in gnina data, cannot use cnn-combined scoring")

        # Log available metrics based on user's scoring choice
        if self.scoring_metric == "cnn-combined":
            logger.info("Using cnn-combined as scoring metric for final ranking")
        elif self.scoring_metric == "affinity":
            # Check if we have affinity metrics specifically from gnina
            affinity_metrics = [
                m for m in self.scoring_metrics
                if 'affinity' in m.lower() and 'cnn' not in m.lower() and '_gnina' in m]
            if not affinity_metrics:
                # Try diffdock as fallback
                affinity_metrics = [
                    m for m in self.diffdock_metrics
                    if 'affinity' in m.lower() and 'cnn' not in m.lower()]
                if not affinity_metrics:
                    raise ValueError("Affinity metrics not found in data")
            logger.info("Using affinity as scoring metric for final ranking")
        elif self.scoring_metric == "cnnaffinity":
            # Check if we have CNNaffinity metrics specifically from gnina
            cnnaffinity_metrics = [m for m in self.scoring_metrics if 'cnnaffinity' in m.lower() and '_gnina' in m]
            if not cnnaffinity_metrics:
                # Try diffdock as fallback
                cnnaffinity_metrics = [m for m in self.diffdock_metrics if 'cnnaffinity' in m.lower()]
                if not cnnaffinity_metrics:
                    raise ValueError("CNNaffinity metrics not found in data")
            logger.info("Using CNNaffinity as scoring metric for final ranking")

        logger.info(f"Available scoring metrics: {self.scoring_metrics}")
        logger.info(f"Threshold metrics (for optimization): {self.threshold_metrics}")

    def _calculate_cnn_combined_score(self, data: pd.DataFrame, return_scaled: bool = False):  # noqa: C901
        """Calculate combined CNN score (CNNpose * CNNaffinity) for given data.
        If return_scaled is True, also return min-max scaled version.
        """
        # Find CNNpose and CNNaffinity columns (prioritize gnina)
        cnnpose_col = None
        cnnaffinity_col = None

        # First look for CNNpose and CNNaffinity in gnina metrics
        for col in data.columns:
            if 'cnnpose' in col.lower() and '_gnina' in col:
                cnnpose_col = col
                break

        for col in data.columns:
            if 'cnnaffinity' in col.lower() and '_gnina' in col:
                cnnaffinity_col = col
                break

        # If not found in gnina, try diffdock as fallback
        if cnnpose_col is None:
            for col in data.columns:
                if 'cnnpose' in col.lower() and '_diffdock' in col:
                    cnnpose_col = col
                    break

        if cnnaffinity_col is None:
            for col in data.columns:
                if 'cnnaffinity' in col.lower() and '_diffdock' in col:
                    cnnaffinity_col = col
                    break

        if cnnpose_col is None or cnnaffinity_col is None:
            logger.warning(
                f"Missing columns for CNN combined score: "
                f"CNNpose={cnnpose_col}, CNNaffinity={cnnaffinity_col}")
            raw = pd.Series(0.0, index=data.index)
            if return_scaled:
                return raw, raw
            return raw

        # logger.info(f"Calculating CNN combined score using: {cnnpose_col} * {cnnaffinity_col}")
        combined_scores = data[cnnpose_col] * data[cnnaffinity_col]
        combined_scores = combined_scores.fillna(combined_scores.min())

        if return_scaled:
            min_val = combined_scores.min()
            max_val = combined_scores.max()
            scaled = (combined_scores - min_val) / (max_val - min_val)
            return combined_scores, scaled

        return combined_scores

    def get_top_k_conformations(self, k: int = 10) -> pd.DataFrame:
        """Get top-k conformations for each molecule based on original pre-computed ranking."""
        top_k_data = []

        for molecule in self.molecules:
            mol_data = self.data[self.data['molecule'] == molecule].copy()

            # Use pre-existing ranks from docking software (rank1, rank2, etc.)
            # Trust the original ranking which used sophisticated multi-metric algorithms
            mol_data = mol_data[mol_data['rank'] <= k]

            top_k_data.append(mol_data)

        logger.info(f"Using pre-ranked conformations from docking software (ranks 1-{k})")
        return pd.concat(top_k_data, ignore_index=True)

    def get_baseline_conformations(self) -> pd.DataFrame:
        """Get baseline conformations (rank1 only) for each molecule."""
        baseline_data = []

        for molecule in self.molecules:
            mol_data = self.data[self.data['molecule'] == molecule].copy()
            # Get only rank 1 conformations
            rank1_data = mol_data[mol_data['rank'] == 1]
            if len(rank1_data) > 0:
                baseline_data.append(rank1_data.iloc[0])  # Should be only one rank1 per molecule

        logger.info(f"Baseline model using {len(baseline_data)} rank1 conformations")
        return pd.DataFrame(baseline_data)

    def apply_thresholds(self, data: pd.DataFrame, thresholds: Dict[str, float]) -> pd.DataFrame:  # noqa: C901
        """Apply threshold-based reranking to select best conformation per molecule."""
        best_conformations = []
        inactive_compounds = 0

        # Find scoring column for final ranking based on user preference
        scoring_col = None

        if self.scoring_metric == "affinity":
            # Force use of Affinity only
            for col in self.scoring_metrics:
                if 'affinity' in col.lower() and 'cnn' not in col.lower():
                    scoring_col = col
                    break
        elif self.scoring_metric == "cnnaffinity":
            # Force use of CNNaffinity only
            for col in self.scoring_metrics:
                if 'cnnaffinity' in col.lower():
                    scoring_col = col
                    break
        elif self.scoring_metric == "cnn-combined":
            # Use combined CNN score for ranking
            scoring_col = "cnn-combined"

        if scoring_col is None:
            raise ValueError("No scoring column found for final ranking")

        if scoring_col == "cnn-combined":
            # logger.info("Using CNN combined score (CNNposeScore * CNNaffinity) for final conformation scoring")
            pass
        else:
            logger.info(f"Using {scoring_col} for final conformation scoring")

        # Global worst scoring value (across entire dataset) so that "no threshold satisfied"
        # is penalized in the global ranking: failed molecules rank below all others.
        if scoring_col == "cnn-combined":
            raw_combined, _ = self._calculate_cnn_combined_score(data, return_scaled=True)
            global_worst_score = float(raw_combined.min()) if len(raw_combined) else 0.0
        elif 'cnnaffinity' in scoring_col.lower():
            global_worst_score = float(data[scoring_col].min())
        else:
            # Affinity: higher (less negative) = worse
            global_worst_score = float(data[scoring_col].max())

        for molecule in data['molecule'].unique():
            mol_data = data[data['molecule'] == molecule].copy()

            # Check which conformations satisfy ALL thresholds
            satisfies_all_thresholds = pd.Series(True, index=mol_data.index)

            for metric, threshold in thresholds.items():
                if metric in mol_data.columns and not mol_data[metric].isna().all():
                    # Apply threshold based on metric type
                    # Occupation, Solvation: lower is better (≤ threshold). %Occupation: higher is better (≥ threshold).
                    base_metric_name = metric.split('_')[0]
                    if 'solvation' in metric.lower() or base_metric_name.lower() == 'occupation':
                        # Solvation and Occupation (no %): lower values are better (≤ threshold)
                        satisfies_all_thresholds &= (mol_data[metric] <= threshold)
                    else:
                        # All other metrics (including %Occupation): higher values are better (≥ threshold)
                        satisfies_all_thresholds &= (mol_data[metric] >= threshold)

            # Filter to only conformations that satisfy ALL thresholds
            valid_conformations = mol_data[satisfies_all_thresholds]

            if len(valid_conformations) == 0:
                # No conformations satisfy all thresholds -> penalize in global ranking
                # Use global worst score (across entire dataset) so this molecule ranks
                # below any molecule that passed thresholds; ROC-AUC reflects the penalty.
                inactive_entry = mol_data.iloc[0].copy()  # Use first conformation as template

                if scoring_col == "cnn-combined":
                    inactive_entry['cnn_combined_score'] = global_worst_score
                else:
                    inactive_entry[scoring_col] = global_worst_score

                inactive_entry['composite_score'] = 0.0
                inactive_entry['satisfies_thresholds'] = False

                best_conformations.append(inactive_entry)
                inactive_compounds += 1

            else:
                # Select best conformation based directly on the scoring metric
                valid_conformations = valid_conformations.copy()

                # Select best conformation based on metric type
                if scoring_col == "cnn-combined":
                    # Store raw CNNpose×CNNaffinity in cnn_combined_score so that
                    # _extract_scores can later apply a single global min-max across
                    # both passing and failed molecules on the same raw scale.
                    # (scaled_combined is local to this subset; do not use it for ranking.)
                    raw_combined, _ = self._calculate_cnn_combined_score(valid_conformations, return_scaled=True)
                    valid_conformations['cnn_combined_score'] = raw_combined
                    valid_conformations['composite_score'] = raw_combined  # raw; global scaling done in metric
                    # Select conformation with highest combined score
                    best_conf = valid_conformations.loc[valid_conformations['cnn_combined_score'].idxmax()].copy()
                elif 'cnnaffinity' in scoring_col.lower():
                    # CNNaffinity: higher is better
                    valid_conformations['composite_score'] = valid_conformations[scoring_col]
                    best_conf = valid_conformations.loc[valid_conformations[scoring_col].idxmax()].copy()
                else:
                    # Regular Affinity: more negative is better (lower is better)
                    valid_conformations['composite_score'] = valid_conformations[scoring_col]
                    best_conf = valid_conformations.loc[valid_conformations[scoring_col].idxmin()].copy()

                best_conf['satisfies_thresholds'] = True
                best_conformations.append(best_conf)

        logger.info(f"Compounds marked as inactive (no conformations satisfy all thresholds): {inactive_compounds}")

        return pd.DataFrame(best_conformations)

    def _no_info_score(self, labels: np.ndarray, metric: str = None) -> float:
        """
        Return the no-information (random-classifier) baseline for the given metric.
        - ROC-AUC and logAUC: 0.5 for any class balance.
        - PR-AUC: class prevalence (fraction of positives), NOT 0.5.
          Returning 0.5 as a PR-AUC fallback creates a false Optuna attractor because
          the optimizer discovers that forcing all molecules to fail thresholds yields a
          reliable 0.5, which beats the genuine prevalence-level baseline when prevalence << 0.5.

        The metric argument must be passed explicitly by each caller so that
        calculate_roc_auc() always returns 0.5 even during a --metric pr-auc run
        (evaluate_thresholds always computes ROC-AUC alongside the chosen metric).
        Defaulting to self.metric when no argument is given keeps objective() working.
        """
        metric = (metric or self.metric).lower()
        if metric == "pr-auc":
            return float(labels.mean()) if len(labels) > 0 else 0.0
        return 0.5

    def _extract_scores(self, predictions: pd.DataFrame) -> np.ndarray:
        """
        Extract and min-max scale the per-molecule scores from predictions.

        For cnn-combined: reads the pre-computed 'cnn_combined_score' column that
        apply_thresholds writes for every molecule (global-worst penalty for failures,
        best raw CNNpose×CNNaffinity for passes).  Recomputing via
        _calculate_cnn_combined_score would silently discard the penalty because it
        reads the raw CNNpose/CNNaffinity columns, which are untouched for failed molecules.

        Raises ValueError with an explicit message on any unrecoverable edge case so
        callers can log it and return the appropriate no-info baseline — no silent returns.
        """
        if self.scoring_metric == "cnn-combined":
            if 'cnn_combined_score' in predictions.columns:
                # Thresholded predictions: cnn_combined_score holds the raw CNNpose×CNNaffinity
                # for passing molecules and the global-worst penalty for failed molecules.
                # Reading it directly preserves that penalty signal.
                raw = predictions['cnn_combined_score'].values.astype(float)
            else:
                # Baseline / unthresholded rank-1 data (get_baseline_conformations) does not
                # pass through apply_thresholds, so cnn_combined_score is never set.
                # No molecules have failed thresholds here, so recomputing CNNpose×CNNaffinity
                # from the raw columns is correct and equivalent.
                raw = self._calculate_cnn_combined_score(
                    predictions, return_scaled=False
                ).values.astype(float)
        else:
            scoring_col = None
            if self.scoring_metric == "affinity":
                for col in self.scoring_metrics:
                    if 'affinity' in col.lower() and 'cnn' not in col.lower():
                        scoring_col = col
                        break
            elif self.scoring_metric == "cnnaffinity":
                for col in self.scoring_metrics:
                    if 'cnnaffinity' in col.lower():
                        scoring_col = col
                        break
            if scoring_col is None:
                raise ValueError(
                    f"No scoring column found for scoring_metric='{self.scoring_metric}'. "
                    f"Available scoring metrics: {list(self.scoring_metrics)}"
                )
            raw = predictions[scoring_col].values.astype(float)
            if self.scoring_metric == "affinity":
                raw = -raw  # Invert so higher is better

        finite = raw[np.isfinite(raw)]
        if len(finite) == 0:
            raise ValueError(
                f"All score values are NaN/Inf for scoring_metric='{self.scoring_metric}'."
            )
        min_val, max_val = float(finite.min()), float(finite.max())
        if max_val <= min_val:
            raise ValueError(
                f"Constant scores for scoring_metric='{self.scoring_metric}' "
                f"(all values == {min_val:.6f}). "
                "All molecules likely failed all thresholds and received the global-worst penalty."
            )
        return (raw - min_val) / (max_val - min_val)

    def calculate_roc_auc(self, predictions: pd.DataFrame) -> float:
        """Calculate ROC-AUC score for the predictions using scoring metrics."""
        labels = predictions['is_active'].values
        no_info = self._no_info_score(labels, "roc-auc")

        if len(np.unique(labels)) < 2:
            logger.warning("ROC-AUC: only one class in predictions — returning %.4f", no_info)
            return no_info

        try:
            scores = self._extract_scores(predictions)
        except ValueError as e:
            logger.warning("ROC-AUC: %s — returning %.4f", e, no_info)
            return no_info

        try:
            return float(roc_auc_score(labels, scores))
        except Exception as e:
            logger.warning("ROC-AUC: roc_auc_score raised %s — returning %.4f", e, no_info)
            return no_info

    def calculate_pr_auc(self, predictions: pd.DataFrame) -> float:
        """Calculate PR-AUC (average precision) score for the predictions."""
        labels = predictions['is_active'].values
        no_info = self._no_info_score(labels, "pr-auc")

        if len(np.unique(labels)) < 2:
            logger.warning("PR-AUC: only one class in predictions — returning prevalence %.4f", no_info)
            return no_info

        try:
            scores = self._extract_scores(predictions)
        except ValueError as e:
            logger.warning("PR-AUC: %s — returning prevalence %.4f", e, no_info)
            return no_info

        try:
            return float(average_precision_score(labels, scores))
        except Exception as e:
            logger.warning("PR-AUC: average_precision_score raised %s — returning prevalence %.4f", e, no_info)
            return no_info

    def calculate_logauc(self, predictions: pd.DataFrame) -> float:
        """Calculate logAUC score for the predictions using scoring metrics."""
        labels = predictions['is_active'].values
        no_info = self._no_info_score(labels, "logauc")

        if len(np.unique(labels)) < 2:
            logger.warning("logAUC: only one class in predictions — returning %.4f", no_info)
            return no_info

        try:
            scores = self._extract_scores(predictions)
        except ValueError as e:
            logger.warning("logAUC: %s — returning %.4f", e, no_info)
            return no_info

        try:
            fpr, tpr, _ = roc_curve(labels, scores)
            unique_indices = np.unique(fpr, return_index=True)[1]
            fpr = fpr[unique_indices]
            tpr = tpr[unique_indices]
            if fpr[0] != 0:
                fpr = np.concatenate([[0], fpr])
                tpr = np.concatenate([[0], tpr])
            if fpr[-1] != 1:
                fpr = np.concatenate([fpr, [1]])
                tpr = np.concatenate([tpr, [tpr[-1]]])
            min_fpr = max(0.001, min(fpr[fpr > 0]))
            log_fpr = np.logspace(np.log10(min_fpr), 0, 100)
            interp_tpr = np.interp(log_fpr, fpr, tpr)
            return float(np.trapz(interp_tpr, log_fpr) / (1 - min_fpr))
        except Exception as e:
            logger.warning("logAUC: roc_curve raised %s — returning %.4f", e, no_info)
            return no_info

    def calculate_metric(self, predictions: pd.DataFrame) -> float:
        """Calculate the selected performance metric for the predictions."""
        if self.metric == "roc-auc":
            return self.calculate_roc_auc(predictions)
        elif self.metric == "pr-auc":
            return self.calculate_pr_auc(predictions)
        elif self.metric == "logauc":
            return self.calculate_logauc(predictions)
        else:
            logger.warning(f"Unknown metric '{self.metric}', falling back to ROC-AUC")
            return self.calculate_roc_auc(predictions)


class AllStructureOptunaOptimizer:
    """Optuna-based optimizer for the /all directory structure."""

    def __init__(self, reranker: AllStructureReranker, n_trials: int = 100,
                 n_jobs: int = 1, optuna_config: Dict = None):
        """Initialize the optimizer."""
        self.reranker = reranker
        self.n_trials = n_trials
        self.n_jobs = n_jobs
        self.study = None
        self.best_thresholds = None
        self.optuna_config = optuna_config or {}

    def objective(self, trial: Any) -> float:  # noqa: C901
        """Objective function for Optuna optimization."""
        thresholds = {}

        # Get data statistics for reasonable threshold ranges
        top_k_data = self.reranker.get_top_k_conformations(10)

        for metric in self.reranker.threshold_metrics:
            if metric in top_k_data.columns and not top_k_data[metric].isna().all():
                values = top_k_data[metric].dropna()

                if len(values) == 0:
                    continue

                # Filter out extreme outliers that are likely data errors
                if len(values) > 10:  # Only apply outlier filtering if we have enough data
                    Q1 = values.quantile(0.25)
                    Q3 = values.quantile(0.75)
                    IQR = Q3 - Q1

                    # Use a more aggressive outlier threshold for suspected error values
                    lower_bound = Q1 - 3 * IQR
                    upper_bound = Q3 + 3 * IQR

                    original_count = len(values)
                    values = values[(values >= lower_bound) & (values <= upper_bound)]

                    if len(values) != original_count:
                        logger.warning(f"Filtered {original_count - len(values)} outliers from {metric}")
                        logger.warning(
                            f"  Original range: [{top_k_data[metric].min():.6f}, "
                            f"{top_k_data[metric].max():.6f}]")
                        logger.warning(f"  Filtered range: [{values.min():.6f}, {values.max():.6f}]")

                if len(values) == 0:
                    logger.warning(f"All values filtered out for {metric}")
                    continue

                min_val = float(values.min())
                max_val = float(values.max())
                mean_val = float(values.mean())
                std_val = float(values.std())

                # Handle edge cases
                if pd.isna(min_val) or pd.isna(max_val) or pd.isna(std_val):
                    logger.warning(f"Invalid statistics for metric {metric}, skipping")
                    continue

                    # Use actual data bounds - simple and reliable
                low = min_val
                high = max_val

                logger.info(f"Using data bounds for {metric}: [{low:.6f}, {high:.6f}]")

                logger.debug(f"Metric {metric}: range=[{low:.4f}, {high:.4f}], mean={mean_val:.4f}, std={std_val:.4f}")

                # Suggest threshold in the safe range
                thresholds[metric] = trial.suggest_float(
                    f'threshold_{metric}',
                    low=low,
                    high=high
                )

        # No-information baseline for the objective: used whenever a trial is uncomputable.
        # For PR-AUC this is class prevalence (not 0.5) to avoid a false Optuna attractor.
        labels_all = top_k_data.drop_duplicates(subset='molecule')['is_active'].values
        no_info = self.reranker._no_info_score(labels_all, self.reranker.metric)

        if not thresholds:
            logger.warning("No valid thresholds could be generated — returning no-info baseline %.4f", no_info)
            return no_info

        # Apply thresholds and get predictions
        try:
            predictions = self.reranker.apply_thresholds(top_k_data, thresholds)

            if len(predictions) == 0:
                logger.warning("No predictions generated — returning no-info baseline %.4f", no_info)
                return no_info

            # Check for sufficient data diversity
            if len(predictions['is_active'].unique()) < 2:
                logger.warning("Insufficient class diversity in predictions — returning no-info baseline %.4f", no_info)
                return no_info

            # Calculate the selected metric
            metric_value = self.reranker.calculate_metric(predictions)

            # Sanity check the result
            if pd.isna(metric_value) or not (0 <= metric_value <= 1):
                logger.warning("Invalid metric value %s — returning no-info baseline %.4f", metric_value, no_info)
                return no_info

            return metric_value

        except Exception as e:
            logger.warning("Error in objective function: %s — returning no-info baseline %.4f", e, no_info)
            return no_info

    def optimize(self, direction: str = 'maximize') -> Dict[str, float]:
        """Run the optimization process."""
        logger.info(f"Starting Optuna optimization with {self.n_trials} trials")
        logger.info(f"Using {self.n_jobs} parallel jobs for trial execution")
        logger.info(f"Optimizing {len(self.reranker.threshold_metrics)} threshold metrics")
        logger.info(f"Scoring metrics (not optimized): {len(self.reranker.scoring_metrics)}")
        logger.info(f"  - Gnina threshold metrics: {[m for m in self.reranker.threshold_metrics if '_gnina' in m]}")
        logger.info(
            f"  - DiffDock threshold metrics: "
            f"{[m for m in self.reranker.threshold_metrics if '_diffdock' in m]}")
        logger.info(f"  - Scoring metrics: {self.reranker.scoring_metrics}")

        # Create study
        import optuna
        self.study = optuna.create_study(direction=direction)

        # Optimize with parallel jobs
        self.study.optimize(self.objective, n_trials=self.n_trials, n_jobs=self.n_jobs)

        # Extract best thresholds
        self.best_thresholds = {}
        for param_name, value in self.study.best_params.items():
            metric_name = param_name.replace('threshold_', '')
            self.best_thresholds[metric_name] = value

        logger.info(f"Optimization completed. Best {self.reranker.metric.upper()}: {self.study.best_value:.4f}")
        logger.info(f"Best thresholds found for {len(self.best_thresholds)} metrics")

        return self.best_thresholds

    def evaluate_thresholds(self, thresholds: Dict[str, float] = None) -> Dict[str, float]:
        """Evaluate the performance of given thresholds."""
        if thresholds is None:
            thresholds = self.best_thresholds

        if thresholds is None:
            logger.warning("No thresholds provided and no optimization run yet.")
            return {}

        # Get top-k conformations
        top_k_data = self.reranker.get_top_k_conformations(10)

        # Apply thresholds
        predictions = self.reranker.apply_thresholds(top_k_data, thresholds)

        # Calculate metrics
        metric_value = self.reranker.calculate_metric(predictions)

        # Calculate ROC-AUC for comparison regardless of optimization metric
        roc_auc = self.reranker.calculate_roc_auc(predictions)

        # Baseline (rank1 conformations only)
        baseline_df = self.reranker.get_baseline_conformations()
        baseline_metric = self.reranker.calculate_metric(baseline_df)
        baseline_roc_auc = self.reranker.calculate_roc_auc(baseline_df)

        results = {
            f'optimized_{self.reranker.metric}': metric_value,
            'optimized_roc_auc': roc_auc,
            f'baseline_{self.reranker.metric}': baseline_metric,
            'baseline_roc_auc': baseline_roc_auc,
            f'{self.reranker.metric}_improvement': metric_value - baseline_metric,
            'roc_auc_improvement': roc_auc - baseline_roc_auc,
            'thresholds': thresholds,
            'n_gnina_threshold_metrics': len([m for m in self.reranker.threshold_metrics if '_gnina' in m]),
            'n_diffdock_threshold_metrics': len([m for m in self.reranker.threshold_metrics if '_diffdock' in m]),
            'n_scoring_metrics': len(self.reranker.scoring_metrics),
            'total_threshold_metrics': len(self.reranker.threshold_metrics),
            'protein': self.reranker.protein,
            'optimization_metric': self.reranker.metric
        }

        logger.info(f"Evaluation Results for {self.reranker.protein}:")
        logger.info(f"  Optimized {self.reranker.metric.upper()}: {metric_value:.4f}")
        logger.info(f"  Baseline {self.reranker.metric.upper()}: {baseline_metric:.4f}")
        logger.info(f"  {self.reranker.metric.upper()} Improvement: {metric_value - baseline_metric:.4f}")
        logger.info(f"  Optimized ROC-AUC: {roc_auc:.4f}")
        logger.info(f"  Baseline ROC-AUC: {baseline_roc_auc:.4f}")
        logger.info(f"  ROC-AUC Improvement: {roc_auc - baseline_roc_auc:.4f}")
        logger.info(f"  Gnina threshold metrics: {len([m for m in self.reranker.threshold_metrics if '_gnina' in m])}")
        logger.info(
            f"  DiffDock threshold metrics: "
            f"{len([m for m in self.reranker.threshold_metrics if '_diffdock' in m])}")
        logger.info(f"  Scoring metrics: {len(self.reranker.scoring_metrics)}")

        return results


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Combined Gnina+DiffDock Optuna Optimization for /all structure",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Data arguments
    parser.add_argument(
        "--protein",
        type=str,
        required=True,
        help="Protein target (e.g., ABL1, EGFR, AA2AR)"
    )

    parser.add_argument(
        "--base-dir",
        type=str,
        default="all",
        help="Base directory containing gnina/ and diffdock/ folders (ignored if --data-dir is set)"
    )

    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Use merged train/test CSVs from this dir (e.g. split_merged): "
             "loads {data_dir}/{protein}_{split}.csv; overrides base_dir"
    )

    # Active/Decoy configuration
    # parser.add_argument( # This line is removed as per the edit hint
    #     "--active-prefix", # This line is removed as per the edit hint
    #     type=str, # This line is removed as per the edit hint
    #     default="active_", # This line is removed as per the edit hint
    #     help="Prefix for active compounds (fallback if CSV not provided)" # This line is removed as per the edit hint
    # ) # This line is removed as per the edit hint

    # parser.add_argument( # This line is removed as per the edit hint
    #     "--decoy-prefix", # This line is removed as per the edit hint
    #     type=str, # This line is removed as per the edit hint
    #     default="decoy_", # This line is removed as per the edit hint
    #     help="Prefix for decoy compounds (fallback if CSV not provided)" # This line is removed as per the edit hint
    # ) # This line is removed as per the edit hint

    parser.add_argument(
        "--scoring-metric",
        type=str,
        choices=["affinity", "cnnaffinity", "cnn-combined"],
        default="affinity",
        help="Metric for ROC-AUC calculation:'affinity' (force affinity), "
             "'cnnaffinity' (force CNNaffinity), 'cnn-combined' (CNNpose * CNNaffinity)"
    )
    parser.add_argument(
        "--combine-only",
        action="store_true",
        help="Only load and combine gnina + diffdock CSVs, resolve is_active, "
             "save to Combined/{protein}_{split}_final.csv and exit (no Optuna)"
    )

    parser.add_argument(
        "--metric",
        type=str,
        choices=["roc-auc", "pr-auc", "logauc"],
        default="roc-auc",
        help="Performance metric to optimize: ROC-AUC, PR-AUC (precision-recall), or logAUC (early enrichment)"
    )

    # Temporary activity labeling
    # parser.add_argument( # This line is removed as per the edit hint
    #     "--activity-csv", # This line is removed as per the edit hint
    #     type=str, # This line is removed as per the edit hint
    #     help="Temporary: CSV file with compound activity labels" # This line is removed as per the edit hint
    # ) # This line is removed as per the edit hint

    parser.add_argument(
        "--activity-column",
        type=str,
        default="Active",
        help="Column in the activity CSV to use for activity labels (e.g., 'MSSA', 'MRSA')"
    )

    parser.add_argument(
        "--skip-compounds",
        nargs="*",
        default=[],
        help="List of compound names to skip/exclude"
    )

    parser.add_argument(
        "--split",
        type=str,
        choices=["train", "test"],
        default="train",
        help="Which split to use: 'train' or 'test' (default: train)"
    )

    # Optimization arguments
    parser.add_argument(
        "--n-trials",
        type=int,
        default=200,
        help="Number of Optuna trials to run"
    )

    parser.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help="Number of parallel jobs for Optuna trials (1 = sequential, -1 = use all CPUs)"
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of top conformations to consider per molecule"
    )

    # Output arguments
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results_all_structure",
        help="Directory to save results"
    )

    parser.add_argument(
        "--save-study",
        action="store_true",
        help="Save the Optuna study object for later analysis"
    )

    parser.add_argument(
        "--eval-on-test",
        action="store_true",
        help="When optimizing on train (--split train), also apply best "
             "thresholds to test set and add test_* metrics to results JSON"
    )

    # Logging
    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level"
    )

    return parser.parse_args()


def main():
    """Main function to run the optimization."""
    args = parse_args()

    # Set logging level
    logging.getLogger().setLevel(getattr(logging, args.log_level))

    # Create output directory
    output_dir = Path(args.output_dir) / args.protein
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Initialize reranker
        logger.info(f"Starting optimization for {args.protein} using /all structure")
        if args.combine_only:
            logger.info("Combine-only mode: will save combined CSV and exit")
        else:
            logger.info(f"Optimizing for {args.metric.upper()}")
        data_dir = getattr(args, 'data_dir', None)
        reranker = AllStructureReranker(
            protein=args.protein,
            base_dir=args.base_dir,
            scoring_metric=args.scoring_metric,
            metric=args.metric,
            activity_column=args.activity_column,
            skip_compounds=args.skip_compounds,
            split=args.split,
            data_dir=data_dir
        )

        # Combine-only: save combined CSV and exit
        if args.combine_only:
            combined_dir = Path("Combined")
            combined_dir.mkdir(parents=True, exist_ok=True)
            out_path = combined_dir / f"{args.protein}_{args.split}_final.csv"
            reranker.data.to_csv(out_path, index=False)
            logger.info(
                f"Saved combined CSV to {out_path} (rows={len(reranker.data)}, "
                f"columns={list(reranker.data.columns)[:8]}...)")
            print(f"\n{'='*70}")
            print(f"COMBINE-ONLY: {args.protein.upper()} ({args.split})")
            print(f"{'='*70}")
            print(f"Output: {out_path}")
            print(f"Rows: {len(reranker.data)}, Molecules: {reranker.data['molecule'].nunique()}")
            print(
                f"Active: {(reranker.data['is_active'] == 1).sum()} conformations, "
                f"Decoy: {(reranker.data['is_active'] == 0).sum()}")
            return None

        # Initialize optimizer
        optimizer = AllStructureOptunaOptimizer(reranker, n_trials=args.n_trials, n_jobs=args.n_jobs)

        # Run optimization
        optimizer.optimize()

        # Evaluate results
        results = optimizer.evaluate_thresholds()

        # Save results
        import json
        results_file = output_dir / f"{args.protein}_all_structure_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)

        # Optionally evaluate on test set (apply train-found thresholds to test data)
        if getattr(args, 'eval_on_test', False) and args.split == 'train':
            logger.info(f"Evaluating best thresholds on test set for {args.protein}")
            data_dir = getattr(args, 'data_dir', None)
            reranker_test = AllStructureReranker(
                protein=args.protein,
                base_dir=args.base_dir,
                scoring_metric=args.scoring_metric,
                metric=args.metric,
                activity_column=args.activity_column,
                skip_compounds=args.skip_compounds,
                split='test',
                data_dir=data_dir
            )
            optimizer_test = AllStructureOptunaOptimizer(reranker_test, n_trials=1, n_jobs=1)
            optimizer_test.best_thresholds = results['thresholds']
            test_results = optimizer_test.evaluate_thresholds(thresholds=results['thresholds'])
            for key in ['optimized_roc_auc', 'baseline_roc_auc',
                        f'optimized_{args.metric}', f'baseline_{args.metric}',
                        f'{args.metric}_improvement', 'roc_auc_improvement']:
                if key in test_results:
                    results[f'test_{key}'] = test_results[key]
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            # Print test results: optimization metric first, then ROC-AUC when not the optimization metric
            opt_key = f'test_optimized_{args.metric}'
            metric_display = {"roc-auc": "ROC-AUC", "pr-auc": "PR-AUC",
                              "logauc": "logAUC"}.get(args.metric, args.metric.upper())
            if opt_key in results:
                logger.info(f"Test {metric_display} (optimized): {results.get(opt_key, 'N/A')}")
                logger.info(f"Test {metric_display} (baseline):  {results.get(f'test_baseline_{args.metric}', 'N/A')}")
                print(f"Test {metric_display} (optimized): {results.get(opt_key, 'N/A')}")
                print(f"Test {metric_display} (baseline):  {results.get(f'test_baseline_{args.metric}', 'N/A')}")
            if args.metric != "roc-auc":
                logger.info(f"Test ROC-AUC (with same thresholds): {results.get('test_optimized_roc_auc', 'N/A')}")
                logger.info(f"Test ROC-AUC (baseline):  {results.get('test_baseline_roc_auc', 'N/A')}")
                print(f"Test ROC-AUC (with same thresholds): {results.get('test_optimized_roc_auc', 'N/A')}")
                print(f"Test ROC-AUC (baseline):  {results.get('test_baseline_roc_auc', 'N/A')}")

        # Save study if requested
        if args.save_study:
            import joblib
            study_file = output_dir / f"{args.protein}_all_structure_study.pkl"
            joblib.dump(optimizer.study, study_file)

        logger.info(f"Results saved to {output_dir}")

        # Print summary
        print(f"\n{'='*70}")
        print(f"OPTIMIZATION RESULTS FOR {args.protein.upper()} (/all structure)")
        print(f"{'='*70}")
        print(f"Optimization metric: {args.metric.upper()}")
        print(f"Gnina CSV: {reranker.gnina_csv_path}")
        print(f"DiffDock CSV: {reranker.diffdock_csv_path}")
        print(f"Threshold metrics optimized: {results['total_threshold_metrics']}")
        print(f"  - Gnina threshold metrics: {results['n_gnina_threshold_metrics']}")
        print(f"  - DiffDock threshold metrics: {results['n_diffdock_threshold_metrics']}")
        print(f"Scoring metrics (for evaluation): {results['n_scoring_metrics']}")
        print(f"Best {args.metric.upper()} (optimized): {results[f'optimized_{args.metric}']:.4f}")
        print(f"Baseline {args.metric.upper()}:         {results[f'baseline_{args.metric}']:.4f}")
        print(f"{args.metric.upper()} Improvement:      +{results[f'{args.metric}_improvement']:.4f}")
        print(f"ROC-AUC (optimized):           {results['optimized_roc_auc']:.4f}")
        print(f"ROC-AUC (baseline):            {results['baseline_roc_auc']:.4f}")
        print(f"ROC-AUC Improvement:           +{results['roc_auc_improvement']:.4f}")
        print(f"Results saved to: {results_file}")

        return results

    except Exception as e:
        logger.error(f"Error in optimization: {e}")
        raise


if __name__ == "__main__":
    results = main()
