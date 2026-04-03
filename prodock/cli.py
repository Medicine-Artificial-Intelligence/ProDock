from __future__ import annotations

"""Command-line interface for running end-to-end ProDock workflows.

This CLI is intentionally JSON-first. The most convenient workflow is to put
receptors, ligands, and run options into one JSON file and execute::

    python -m prodock --config run.json

A recommended config layout is::

    {
      "project_dir": "Data/testcase/Multi",
      "receptors": [
        {
          "pdb_id": "4WKQ",
          "receptor_name": "EGFR_4WKQ",
          "ligand_code": "IRE",
          "chains": ["A"],
          "cofactors": []
        }
      ],
      "ligands": [
        {
          "id": "erlotinib",
          "smiles": "COCCOc1cc2c(ncnc2cc1OCCOC)Nc1cccc(c1)C#C"
        },
        {
          "id": "gefitinib",
          "smiles": "COc1cc2ncnc(c2cc1OCCCN1CCOCC1)Nc1ccc(c(c1)Cl)F"
        }
      ],
      "config": {
        "engines": ["qvina", "qvina-w"],
        "extract_interaction": true,
        "db_name": "test.db",
        "cpu": 8,
        "n_jobs": 8,
        "exhaustiveness": 16,
        "n_poses": 20
      }
    }

The same file can also use ``prepared_receptors`` and ``ligand_dir`` instead of
``receptors`` and ``ligands``.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, MutableMapping, Sequence

from .core import ProDockResult, prodock

JSONDict = dict[str, Any]

BANNER = r"""
======================================================================
   ____            ____             _
  |  _ \ _ __ ___ |  _ \  ___   ___| | __
  | |_) | '__/ _ \| | | |/ _ \ / __| |/ /
  |  __/| | | (_) | |_| | (_) | (__|   <
  |_|   |_|  \___/|____/ \___/ \___|_|\_\

   Multi-receptor • Multi-ligand • Multi-engine • Database-ready
======================================================================
"""

_STRUCTURAL_KEYS = {
    "project_dir",
    "receptors",
    "prepared_receptors",
    "ligands",
    "ligand_dir",
    "config",
    "options",
    "run",
}


class CLIError(ValueError):
    """Raised when CLI input or config structure is invalid."""


def _positive_int(value: str) -> int:
    """Parse a strictly positive integer for argparse."""
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be > 0")
    return parsed


def _nonnegative_int(value: str) -> int:
    """Parse a non-negative integer for argparse."""
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be >= 0")
    return parsed


def _load_json(path: Path) -> JSONDict:
    """Load and validate a JSON config file."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise CLIError(f"Invalid JSON in config file {path}: {exc}") from exc

    if not isinstance(payload, dict):
        raise CLIError("Config JSON must contain a top-level object/dictionary.")
    return payload


def _resolve_pathlike(value: Any, *, base_dir: Path) -> Any:
    """Resolve a path-like string relative to the config file directory."""
    if value is None or not isinstance(value, (str, Path)):
        return value

    p = Path(value)
    if p.is_absolute():
        return str(p)
    return str((base_dir / p).resolve())


def _resolve_record_paths(
    records: Sequence[Mapping[str, Any]],
    *,
    keys: Iterable[str],
    base_dir: Path,
) -> list[dict[str, Any]]:
    """Resolve selected path fields inside a sequence of record dictionaries."""
    keyset = set(keys)
    resolved: list[dict[str, Any]] = []

    for item in records:
        if not isinstance(item, Mapping):
            raise CLIError("Each record in the JSON config must be a dictionary.")

        row = dict(item)
        for key in keyset:
            if key in row and row[key] is not None:
                row[key] = _resolve_pathlike(row[key], base_dir=base_dir)
        resolved.append(row)

    return resolved


def _normalize_payload(payload: Mapping[str, Any], *, config_path: Path) -> JSONDict:
    """Normalize a JSON payload into kwargs accepted by ``prodock``."""
    base_dir = config_path.parent.resolve()

    config_obj = payload.get("config")
    options_obj = payload.get("options")
    run_obj = payload.get("run")
    merged_options: JSONDict = {}

    for obj_name, obj in (
        ("config", config_obj),
        ("options", options_obj),
        ("run", run_obj),
    ):
        if obj is None:
            continue
        if not isinstance(obj, Mapping):
            raise CLIError(f"'{obj_name}' must be a dictionary if provided.")
        merged_options.update(dict(obj))

    flat_options = {k: v for k, v in payload.items() if k not in _STRUCTURAL_KEYS}
    merged_options.update(flat_options)

    normalized: JSONDict = dict(merged_options)

    if normalized.get("project_dir") is not None:
        normalized["project_dir"] = _resolve_pathlike(
            normalized["project_dir"], base_dir=base_dir
        )

    if normalized.get("ligand_dir") is not None:
        normalized["ligand_dir"] = _resolve_pathlike(
            normalized["ligand_dir"], base_dir=base_dir
        )

    if "project_dir" in payload and payload["project_dir"] is not None:
        normalized["project_dir"] = _resolve_pathlike(
            payload["project_dir"], base_dir=base_dir
        )

    if "ligand_dir" in payload and payload["ligand_dir"] is not None:
        normalized["ligand_dir"] = _resolve_pathlike(
            payload["ligand_dir"], base_dir=base_dir
        )

    if "receptors" in payload and payload["receptors"] is not None:
        receptors = payload["receptors"]
        if not isinstance(receptors, Sequence) or isinstance(
            receptors, (str, bytes, bytearray)
        ):
            raise CLIError("'receptors' must be a list of dictionaries.")
        normalized["receptors"] = _resolve_record_paths(
            receptors,
            keys=("reference_ligand",),
            base_dir=base_dir,
        )

    if "prepared_receptors" in payload and payload["prepared_receptors"] is not None:
        prepared = payload["prepared_receptors"]
        if not isinstance(prepared, Sequence) or isinstance(
            prepared, (str, bytes, bytearray)
        ):
            raise CLIError("'prepared_receptors' must be a list of dictionaries.")
        normalized["prepared_receptors"] = _resolve_record_paths(
            prepared,
            keys=("receptor_pdbqt", "receptor", "reference_ligand"),
            base_dir=base_dir,
        )

    if "ligands" in payload and payload["ligands"] is not None:
        ligands = payload["ligands"]
        if not isinstance(ligands, Sequence) or isinstance(
            ligands, (str, bytes, bytearray)
        ):
            raise CLIError("'ligands' must be a list of dictionaries.")
        normalized["ligands"] = _resolve_record_paths(
            ligands,
            keys=(),
            base_dir=base_dir,
        )

    return normalized


def _build_parser() -> argparse.ArgumentParser:
    """Build the top-level argument parser."""
    parser = argparse.ArgumentParser(
        prog="prodock",
        description="Run the end-to-end ProDock docking pipeline from a JSON config.",
    )

    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to a JSON config file containing receptors/ligands and optional run config.",
    )

    parser.add_argument(
        "--project-dir", type=str, help="Override project_dir from the JSON config."
    )
    parser.add_argument(
        "--engines",
        nargs="+",
        help="Override docking engines, e.g. --engines qvina qvina-w smina",
    )
    parser.add_argument(
        "--cpu", type=_positive_int, help="Override per-engine CPU value."
    )
    parser.add_argument("--seed", type=int, help="Override random seed.")
    parser.add_argument(
        "--exhaustiveness",
        type=_positive_int,
        help="Override docking exhaustiveness.",
    )
    parser.add_argument(
        "--n-poses", type=_positive_int, help="Override number of poses to keep."
    )
    parser.add_argument(
        "--n-jobs",
        type=_positive_int,
        help="Override number of parallel jobs for batch docking.",
    )
    parser.add_argument(
        "--progress",
        dest="progress",
        action="store_true",
        default=None,
        help="Enable docking progress output.",
    )
    parser.add_argument(
        "--no-progress",
        dest="progress",
        action="store_false",
        help="Disable docking progress output.",
    )
    parser.add_argument(
        "--receptor-use-meeko",
        action="store_true",
        default=None,
        help="Use Meeko during receptor preparation.",
    )
    parser.add_argument(
        "--ligand-output-format",
        type=str,
        help="Override ligand output format, e.g. pdbqt or sdf.",
    )
    parser.add_argument(
        "--ligand-backend",
        type=str,
        help="Override ligand conversion backend, e.g. meeko.",
    )
    parser.add_argument(
        "--box-scale",
        type=float,
        help="Override ligand-derived box scale factor.",
    )
    parser.add_argument(
        "--box-isotropic",
        dest="box_isotropic",
        action="store_true",
        default=None,
        help="Use isotropic ligand-derived grid boxes.",
    )
    parser.add_argument(
        "--box-anisotropic",
        dest="box_isotropic",
        action="store_false",
        help="Use anisotropic ligand-derived grid boxes.",
    )
    parser.add_argument(
        "--campaign-name",
        type=str,
        help="Override output campaign JSON filename.",
    )
    parser.add_argument(
        "--crawl-backend",
        type=str,
        help="Override pose crawler backend, e.g. obabel.",
    )

    parser.add_argument(
        "--extract-interaction",
        dest="extract_interaction",
        action="store_true",
        default=None,
        help="Enable protein-ligand interaction extraction.",
    )
    parser.add_argument(
        "--no-extract-interaction",
        dest="extract_interaction",
        action="store_false",
        help="Disable protein-ligand interaction extraction.",
    )
    parser.add_argument(
        "--interaction-batch-size",
        type=_positive_int,
        help="Override interaction extraction batch size.",
    )
    parser.add_argument(
        "--interaction-progress",
        dest="interaction_progress",
        action="store_true",
        default=None,
        help="Enable progress output for interaction extraction.",
    )
    parser.add_argument(
        "--no-interaction-progress",
        dest="interaction_progress",
        action="store_false",
        help="Disable progress output for interaction extraction.",
    )
    parser.add_argument(
        "--interaction-n-jobs",
        type=_positive_int,
        help="Override number of interaction extraction jobs.",
    )
    parser.add_argument(
        "--include-fingerprint-columns",
        dest="include_fingerprint_columns",
        action="store_true",
        default=None,
        help="Include fingerprint columns in interaction output.",
    )
    parser.add_argument(
        "--no-include-fingerprint-columns",
        dest="include_fingerprint_columns",
        action="store_false",
        help="Do not include fingerprint columns in interaction output.",
    )
    parser.add_argument(
        "--include-interaction-events",
        dest="include_interaction_events",
        action="store_true",
        default=None,
        help="Include long-form interaction events.",
    )
    parser.add_argument(
        "--no-include-interaction-events",
        dest="include_interaction_events",
        action="store_false",
        help="Do not include long-form interaction events.",
    )
    parser.add_argument(
        "--include-bitvectors",
        action="store_true",
        default=None,
        help="Include bitvectors in interaction output.",
    )
    parser.add_argument(
        "--include-countvectors",
        action="store_true",
        default=None,
        help="Include countvectors in interaction output.",
    )
    parser.add_argument(
        "--fail-fast",
        dest="fail_fast",
        action="store_true",
        default=None,
        help="Fail immediately if interaction extraction encounters an error.",
    )
    parser.add_argument(
        "--no-fail-fast",
        dest="fail_fast",
        action="store_false",
        help="Continue collecting interaction extraction errors when possible.",
    )
    parser.add_argument(
        "--use-interaction-profiler",
        action="store_true",
        default=None,
        help="Use InteractionProfiler instead of extract_pose_table_interactions.",
    )

    parser.add_argument(
        "--save-to-database",
        dest="save_to_database",
        action="store_true",
        default=None,
        help="Write merged results to the SQLite database.",
    )
    parser.add_argument(
        "--no-save-to-database",
        dest="save_to_database",
        action="store_false",
        help="Skip writing results to the SQLite database.",
    )
    parser.add_argument(
        "--db-name",
        type=str,
        help="Override database file name or relative path under project_dir.",
    )
    parser.add_argument(
        "--replace",
        dest="replace",
        action="store_true",
        default=None,
        help="Replace existing pose rows during database insertion.",
    )
    parser.add_argument(
        "--no-replace",
        dest="replace",
        action="store_false",
        help="Do not replace existing pose rows during database insertion.",
    )
    parser.add_argument(
        "--replace-interactions",
        dest="replace_interactions",
        action="store_true",
        default=None,
        help="Replace existing interaction rows during database insertion.",
    )
    parser.add_argument(
        "--no-replace-interactions",
        dest="replace_interactions",
        action="store_false",
        help="Do not replace existing interaction rows during database insertion.",
    )

    parser.add_argument(
        "--summary-json",
        type=str,
        default=None,
        help="Optional path to write a compact run summary as JSON.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress human-readable console summary and only rely on exit status / --summary-json.",
    )
    parser.add_argument(
        "--traceback",
        action="store_true",
        help="Show full Python traceback on errors.",
    )

    return parser


def _apply_cli_overrides(
    args: argparse.Namespace, kwargs: MutableMapping[str, Any]
) -> None:
    """Apply explicit CLI overrides on top of config-derived kwargs."""
    override_keys = {
        "project_dir",
        "engines",
        "cpu",
        "seed",
        "exhaustiveness",
        "n_poses",
        "n_jobs",
        "progress",
        "receptor_use_meeko",
        "ligand_output_format",
        "ligand_backend",
        "box_scale",
        "box_isotropic",
        "campaign_name",
        "crawl_backend",
        "extract_interaction",
        "interaction_batch_size",
        "interaction_progress",
        "interaction_n_jobs",
        "include_fingerprint_columns",
        "include_interaction_events",
        "include_bitvectors",
        "include_countvectors",
        "fail_fast",
        "use_interaction_profiler",
        "save_to_database",
        "db_name",
        "replace",
        "replace_interactions",
    }

    for key in override_keys:
        value = getattr(args, key, None)
        if value is not None:
            kwargs[key] = value


def _validate_run_inputs(kwargs: Mapping[str, Any]) -> None:
    """Validate the minimum high-level CLI requirements before execution."""
    if "project_dir" not in kwargs or not kwargs["project_dir"]:
        raise CLIError(
            "Missing 'project_dir'. Provide it in the JSON config or via --project-dir."
        )

    has_raw_receptors = kwargs.get("receptors") is not None
    has_prepared_receptors = kwargs.get("prepared_receptors") is not None
    if has_raw_receptors == has_prepared_receptors:
        raise CLIError(
            "Provide exactly one receptor mode: 'receptors' or 'prepared_receptors'."
        )

    has_ligands = kwargs.get("ligands") is not None
    has_ligand_dir = kwargs.get("ligand_dir") is not None
    if has_ligands == has_ligand_dir:
        raise CLIError("Provide exactly one ligand mode: 'ligands' or 'ligand_dir'.")


def _result_summary(result: ProDockResult) -> JSONDict:
    """Convert the pipeline result to a compact JSON-serializable summary."""
    return {
        "project_dir": str(result.project_dir),
        "ligand_dir": str(result.ligand_dir),
        "campaign_json": str(result.campaign_json),
        "db_path": str(result.db_path) if result.db_path is not None else None,
        "receptor_ids": [spec.receptor_id for spec in result.receptors],
        "n_receptors": len(result.receptors),
        "n_pose_rows": int(len(result.pose_df)),
        "n_merged_rows": int(len(result.merged_df)),
        "interaction_extracted": result.interaction_result is not None,
        "has_interaction_df": result.interaction_df is not None,
        "has_summary_df": result.summary_df is not None,
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point for the ProDock CLI."""
    print(BANNER, file=sys.stderr)
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        config_path = args.config.resolve()
        if not config_path.exists():
            raise CLIError(f"Config file not found: {config_path}")
        if not config_path.is_file():
            raise CLIError(f"Config path is not a file: {config_path}")

        payload = _load_json(config_path)
        run_kwargs = _normalize_payload(payload, config_path=config_path)
        _apply_cli_overrides(args, run_kwargs)
        _validate_run_inputs(run_kwargs)

        result = prodock(**run_kwargs)
        summary = _result_summary(result)

        if args.summary_json:
            summary_path = Path(args.summary_json).expanduser().resolve()
            summary_path.parent.mkdir(parents=True, exist_ok=True)
            summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

        if not args.quiet:
            print(json.dumps(summary, indent=2))

        return 0

    except Exception as exc:
        if args.traceback:
            raise
        print(f"[prodock] ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
