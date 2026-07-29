from __future__ import annotations

"""Command-line interface for running end-to-end ProDock workflows.

ProDock provides a JSON-first command-line interface for running multi-receptor,
multi-ligand, and multi-engine docking campaigns from one command.

The CLI supports two main input patterns:

1. **All-in-one JSON**
   A single config file contains the project directory, receptor definitions,
   ligand definitions, and run options.

2. **Split JSON files**
   The main config file contains project-level and run-level options, while
   receptor and ligand definitions are stored in separate JSON files and passed
   through ``--receptor-json`` and ``--ligand-json``.

CLI override precedence
-----------------------
Configuration values are resolved in the following order:

1. ``--config`` provides the base payload.
2. ``--receptor-json`` overrides embedded ``receptors`` from ``--config``.
3. ``--ligand-json`` overrides embedded ``ligands`` from ``--config``.
4. Explicit CLI flags override all JSON-derived values.

Supported structural input modes
--------------------------------
Receptors:
- ``receptors`` for raw receptor specifications
- ``prepared_receptors`` for already prepared receptor inputs

Ligands:
- ``ligands`` for inline ligand dictionaries
- ``ligand_dir`` for a directory of prepared ligand files

Exactly one receptor mode and exactly one ligand mode must be supplied after all
config merging and CLI overrides are applied.

Quick start
-----------
Run from one JSON file::

    python -m prodock --config run.json

Run from split JSON files::

    python -m prodock \
        --config config.json \
        --receptor-json receptor.json \
        --ligand-json ligand.json

Validate merged configuration only::

    python -m prodock \
        --config config.json \
        --receptor-json receptor.json \
        --ligand-json ligand.json \
        --validate-only

Write a compact run summary::

    python -m prodock \
        --config run.json \
        --summary-json summary.json

Write the final merged effective config for reproducibility::

    python -m prodock \
        --config config.json \
        --receptor-json receptor.json \
        --ligand-json ligand.json \
        --effective-config-json effective.json

Examples
--------
All-in-one JSON
^^^^^^^^^^^^^^^
A single file may contain receptors, ligands, and run settings together::

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
        "n_poses": 20,
        "save_to_database": true
      }
    }

Run it with::

    python -m prodock --config run.json

Split JSON input
^^^^^^^^^^^^^^^^
You may separate receptor and ligand definitions from the main config.

Example ``receptor.json``::

    {
      "receptors": [
        {
          "pdb_id": "4WKQ",
          "receptor_name": "EGFR_4WKQ",
          "ligand_code": "IRE",
          "chains": ["A"],
          "cofactors": []
        }
      ]
    }

Example ``ligand.json``::

    {
      "ligands": [
        {
          "id": "erlotinib",
          "smiles": "COCCOc1cc2c(ncnc2cc1OCCOC)Nc1cccc(c1)C#C"
        },
        {
          "id": "gefitinib",
          "smiles": "COc1cc2ncnc(c2cc1OCCCN1CCOCC1)Nc1ccc(c(c1)Cl)F"
        }
      ]
    }

Example ``config.json``::

    {
      "project_dir": "Demo",
      "config": {
        "engines": ["qvina", "qvina-w"],
        "extract_interaction": true,
        "db_name": "demo.db",
        "cpu": 8,
        "n_jobs": 8,
        "exhaustiveness": 16,
        "n_poses": 20,
        "save_to_database": true
      }
    }

Run with split inputs::

    python -m prodock \
        --config config.json \
        --receptor-json receptor.json \
        --ligand-json ligand.json

Embedded fallback behavior
^^^^^^^^^^^^^^^^^^^^^^^^^^
If ``--receptor-json`` is not provided, the CLI will look for ``receptors`` in
``--config``.

If ``--ligand-json`` is not provided, the CLI will look for ``ligands`` in
``--config``.

This means the following is valid as long as ``config.json`` already contains
both ``receptors`` and ``ligands``::

    python -m prodock --config config.json

Prepared receptor mode
^^^^^^^^^^^^^^^^^^^^^^
Instead of raw ``receptors``, a config may provide ``prepared_receptors``::

    {
      "project_dir": "DemoPrepared",
      "prepared_receptors": [
        {
          "receptor_id": "4WKQ",
          "receptor_pdbqt": "prepared/4WKQ/4WKQ.pdbqt",
          "center": [5.0, 10.0, 12.0],
          "size": [20.0, 20.0, 20.0]
        }
      ],
      "ligands": [
        {
          "id": "erlotinib",
          "smiles": "COCCOc1cc2c(ncnc2cc1OCCOC)Nc1cccc(c1)C#C"
        }
      ],
      "config": {
        "engines": ["qvina"],
        "save_to_database": true
      }
    }

Ligand directory mode
^^^^^^^^^^^^^^^^^^^^^
Instead of inline ``ligands``, a config may provide ``ligand_dir``::

    {
      "project_dir": "DemoLigandDir",
      "receptors": [
        {
          "pdb_id": "4WKQ",
          "receptor_name": "EGFR_4WKQ",
          "ligand_code": "IRE",
          "chains": ["A"],
          "cofactors": []
        }
      ],
      "ligand_dir": "prepared_ligands",
      "config": {
        "engines": ["qvina", "vina"],
        "extract_interaction": false
      }
    }

CLI override examples
^^^^^^^^^^^^^^^^^^^^^
Override engines and compute settings from the command line::

    python -m prodock \
        --config config.json \
        --receptor-json receptor.json \
        --ligand-json ligand.json \
        --engines qvina smina vina \
        --cpu 8 \
        --n-jobs 8 \
        --exhaustiveness 16 \
        --n-poses 20

Enable or disable boolean options using one flag family::

    python -m prodock --config run.json --progress
    python -m prodock --config run.json --no-progress

    python -m prodock --config run.json --extract-interaction
    python -m prodock --config run.json --no-extract-interaction

    python -m prodock --config run.json --save-to-database
    python -m prodock --config run.json --no-save-to-database

    python -m prodock --config run.json --replace
    python -m prodock --config run.json --no-replace

Interaction extraction examples
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Run interaction extraction with more detailed controls::

    python -m prodock \
        --config run.json \
        --extract-interaction \
        --interaction-batch-size 8 \
        --interaction-n-jobs 4 \
        --interaction-progress \
        --include-fingerprint-columns \
        --include-interaction-events

Use the InteractionProfiler backend::

    python -m prodock \
        --config run.json \
        --extract-interaction \
        --use-interaction-profiler

Database examples
^^^^^^^^^^^^^^^^^
Write merged outputs to SQLite and replace existing records if desired::

    python -m prodock \
        --config run.json \
        --save-to-database \
        --db-name demo.db \
        --replace \
        --replace-interactions

Reproducibility and debugging
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Print the merged effective configuration::

    python -m prodock \
        --config config.json \
        --receptor-json receptor.json \
        --ligand-json ligand.json \
        --print-effective-config

Write the merged effective configuration to disk::

    python -m prodock \
        --config config.json \
        --receptor-json receptor.json \
        --ligand-json ligand.json \
        --effective-config-json effective.json

Validate inputs without running docking::

    python -m prodock \
        --config config.json \
        --receptor-json receptor.json \
        --ligand-json ligand.json \
        --validate-only

Show a traceback on errors::

    python -m prodock --config run.json --traceback

Notes
-----
- Relative paths inside each JSON file are resolved relative to the directory of
  that JSON file.
- ``--summary-json`` and ``--effective-config-json`` are resolved relative to
  the main ``--config`` directory when given as relative paths.
- Boolean CLI options are implemented with
  :class:`argparse.BooleanOptionalAction`, so each option naturally supports both
  positive and negative forms.
- The CLI validates the final merged configuration before calling
  :func:`prodock`.
"""

import argparse
import json
import sys
from collections.abc import Iterable, Mapping, MutableMapping, Sequence
from pathlib import Path
from textwrap import dedent
from typing import Any

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


class _HelpFormatter(
    argparse.ArgumentDefaultsHelpFormatter,
    argparse.RawDescriptionHelpFormatter,
):
    """Formatter with defaults and preserved line breaks."""


def _positive_int(value: str) -> int:
    """Parse a strictly positive integer."""
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("value must be an integer") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be > 0")
    return parsed


def _nonnegative_int(value: str) -> int:
    """Parse a non-negative integer."""
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("value must be an integer") from exc
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be >= 0")
    return parsed


def _load_json(path: Path) -> JSONDict:
    """Load and validate a JSON file."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise CLIError(f"Could not read JSON file {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise CLIError(f"Invalid JSON in file {path}: {exc}") from exc

    if not isinstance(payload, dict):
        raise CLIError(f"JSON file must contain a top-level object: {path}")

    return payload


def _resolve_pathlike(value: Any, *, base_dir: Path) -> Any:
    """Resolve a path-like string relative to a base directory."""
    if value is None or not isinstance(value, (str, Path)):
        return value

    if isinstance(value, str) and not value.strip():
        return value

    path = Path(value).expanduser()
    if path.is_absolute():
        return str(path.resolve())
    return str((base_dir / path).resolve())


def _resolve_output_path(value: str | None, *, base_dir: Path) -> Path | None:
    """Resolve an output path relative to the main config directory."""
    if value is None:
        return None

    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    return path.resolve()


def _ensure_record_sequence(
    value: Any,
    *,
    key_name: str,
) -> Sequence[Mapping[str, Any]]:
    """Validate that a JSON field is a list of dictionaries."""
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise CLIError(f"'{key_name}' must be a list of dictionaries.")

    normalized: list[Mapping[str, Any]] = []
    for item in value:
        if not isinstance(item, Mapping):
            raise CLIError(f"Each item in '{key_name}' must be a dictionary.")
        normalized.append(item)
    return normalized


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
        row = dict(item)
        for key in keyset:
            if key in row and row[key] is not None:
                row[key] = _resolve_pathlike(row[key], base_dir=base_dir)
        resolved.append(row)

    return resolved


def _normalize_payload(payload: Mapping[str, Any], *, source_path: Path) -> JSONDict:
    """Normalize a JSON payload into kwargs accepted by ``prodock``."""
    base_dir = source_path.parent.resolve()

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
        receptors = _ensure_record_sequence(payload["receptors"], key_name="receptors")
        normalized["receptors"] = _resolve_record_paths(
            receptors,
            keys=("reference_ligand",),
            base_dir=base_dir,
        )

    if "prepared_receptors" in payload and payload["prepared_receptors"] is not None:
        prepared = _ensure_record_sequence(
            payload["prepared_receptors"],
            key_name="prepared_receptors",
        )
        normalized["prepared_receptors"] = _resolve_record_paths(
            prepared,
            keys=("receptor_pdbqt", "receptor", "reference_ligand"),
            base_dir=base_dir,
        )

    if "ligands" in payload and payload["ligands"] is not None:
        ligands = _ensure_record_sequence(payload["ligands"], key_name="ligands")
        normalized["ligands"] = _resolve_record_paths(
            ligands,
            keys=(),
            base_dir=base_dir,
        )

    return normalized


def _overlay_json_section(
    run_kwargs: MutableMapping[str, Any],
    *,
    json_path: Path | None,
    expected_key: str,
    incompatible_key: str | None = None,
) -> None:
    """Overlay one structural section from an external JSON file."""
    if json_path is None:
        return

    source_path = json_path.expanduser().resolve()

    if not source_path.exists():
        raise CLIError(f"JSON file not found: {source_path}")
    if not source_path.is_file():
        raise CLIError(f"JSON path is not a file: {source_path}")

    payload = _load_json(source_path)

    if expected_key not in payload or payload[expected_key] is None:
        raise CLIError(
            f"{source_path} must contain a top-level '{expected_key}' field."
        )

    if incompatible_key and payload.get(incompatible_key) is not None:
        raise CLIError(
            f"{source_path} cannot define both '{expected_key}' and "
            f"'{incompatible_key}'."
        )

    normalized = _normalize_payload(payload, source_path=source_path)

    if expected_key not in normalized or normalized[expected_key] is None:
        raise CLIError(
            f"Failed to normalize '{expected_key}' from JSON file: {source_path}"
        )

    run_kwargs[expected_key] = normalized[expected_key]

    if incompatible_key:
        run_kwargs.pop(incompatible_key, None)


def _add_bool_arg(
    parser: argparse._ActionsContainer,
    name: str,
    *,
    dest: str | None = None,
    help: str,
    default: bool | None = None,
) -> None:
    """Add a boolean flag with automatic --foo / --no-foo variants."""
    parser.add_argument(
        name,
        dest=dest,
        action=argparse.BooleanOptionalAction,
        default=default,
        help=help,
    )


def _build_examples_advanced() -> str:
    """Return advanced help text with JSON examples."""
    return dedent("""
        JSON examples
        -------------
        1) All-in-one config JSON

          {
            "project_dir": "Demo",
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
              "db_name": "demo.db",
              "cpu": 8,
              "n_jobs": 8,
              "exhaustiveness": 16,
              "n_poses": 20,
              "save_to_database": true
            }
          }

        2) Split JSON files

          receptor.json
          {
            "receptors": [
              {
                "pdb_id": "4WKQ",
                "receptor_name": "EGFR_4WKQ",
                "ligand_code": "IRE",
                "chains": ["A"],
                "cofactors": []
              }
            ]
          }

          ligand.json
          {
            "ligands": [
              {
                "id": "erlotinib",
                "smiles": "COCCOc1cc2c(ncnc2cc1OCCOC)Nc1cccc(c1)C#C"
              },
              {
                "id": "gefitinib",
                "smiles": "COc1cc2ncnc(c2cc1OCCCN1CCOCC1)Nc1ccc(c(c1)Cl)F"
              }
            ]
          }

          config.json
          {
            "project_dir": "Demo",
            "config": {
              "engines": ["qvina", "qvina-w"],
              "extract_interaction": true,
              "db_name": "demo.db",
              "cpu": 8,
              "n_jobs": 8,
              "exhaustiveness": 16,
              "n_poses": 20,
              "save_to_database": true
            }
          }

        Example commands
        ----------------
          prodock --config run.json

          prodock --config config.json --receptor-json receptor.json --ligand-json ligand.json

          prodock --config config.json --receptor-json receptor.json --ligand-json ligand.json \\
              --engines qvina smina --cpu 8 --n-jobs 8 --progress --save-to-database

          prodock --config config.json --receptor-json receptor.json --ligand-json ligand.json \\
              --validate-only --print-effective-config
        """).strip()


def _build_parser(*, show_advanced: bool) -> argparse.ArgumentParser:
    """Build the CLI argument parser.

    Parameters
    ----------
    show_advanced
        Whether to include the extended help view with advanced option groups and
        JSON examples.

    Returns
    -------
    argparse.ArgumentParser
        Configured parser instance for the ProDock CLI.
    """
    if show_advanced:
        description = (
            "Run the end-to-end ProDock docking pipeline from JSON inputs. "
            "This advanced help shows the full option set and JSON examples."
        )
        epilog = _build_examples_advanced()
    else:
        description = (
            "Run the end-to-end ProDock docking pipeline from JSON inputs.\n\n"
            "Use '-h advanced' or '--help-advanced' for full options and JSON examples."
        )
        epilog = dedent("""
            Common examples
            ---------------
              prodock --config run.json

              prodock --config config.json --receptor-json receptor.json --ligand-json ligand.json

            For full option list and JSON templates:
              prodock -h advanced
            """).strip()

    parser = argparse.ArgumentParser(
        prog="prodock",
        add_help=False,
        allow_abbrev=False,
        formatter_class=_HelpFormatter,
        description=description,
        epilog=epilog,
    )

    help_group = parser.add_argument_group("Help")
    help_group.add_argument(
        "-h",
        "--help",
        action="store_true",
        help="Show normal help.",
    )
    help_group.add_argument(
        "--help-advanced",
        action="store_true",
        help="Show advanced help with full option list and JSON examples.",
    )

    input_group = parser.add_argument_group("Input JSON files")
    input_group.add_argument(
        "--config",
        type=Path,
        required=True,
        help=(
            "Main JSON file. This may be all-in-one, or may contain only "
            "project-level/config-level settings when used with "
            "--receptor-json and --ligand-json."
        ),
    )
    input_group.add_argument(
        "--receptor-json",
        type=Path,
        default=None,
        help=(
            "Optional JSON file containing a top-level 'receptors' list. "
            "If provided, it overrides embedded 'receptors' in --config."
        ),
    )
    input_group.add_argument(
        "--ligand-json",
        type=Path,
        default=None,
        help=(
            "Optional JSON file containing a top-level 'ligands' list. "
            "If provided, it overrides embedded 'ligands' in --config."
        ),
    )

    common_group = parser.add_argument_group("Common run options")
    common_group.add_argument(
        "--project-dir",
        type=str,
        help="Override project_dir from JSON.",
    )
    common_group.add_argument(
        "--engines",
        nargs="+",
        help="Override docking engines, for example: --engines qvina qvina-w smina",
    )
    common_group.add_argument(
        "--cpu",
        type=_positive_int,
        help="Override per-engine CPU value.",
    )
    common_group.add_argument(
        "--seed",
        type=int,
        help="Override random seed.",
    )
    common_group.add_argument(
        "--exhaustiveness",
        type=_positive_int,
        help="Override docking exhaustiveness.",
    )
    common_group.add_argument(
        "--n-poses",
        dest="n_poses",
        type=_positive_int,
        help="Override number of poses to keep per job.",
    )
    common_group.add_argument(
        "--n-jobs",
        dest="n_jobs",
        type=_positive_int,
        help="Override number of parallel jobs for batch docking.",
    )
    _add_bool_arg(
        common_group,
        "--progress",
        help="Enable or disable docking progress output.",
    )
    _add_bool_arg(
        common_group,
        "--extract-interaction",
        dest="extract_interaction",
        help="Enable or disable protein-ligand interaction extraction.",
    )
    _add_bool_arg(
        common_group,
        "--save-to-database",
        dest="save_to_database",
        help="Enable or disable writing merged results to SQLite.",
    )
    common_group.add_argument(
        "--db-name",
        type=str,
        help="Override database file name or relative path under project_dir.",
    )
    common_group.add_argument(
        "--summary-json",
        type=str,
        default=None,
        help="Optional path to write a compact run summary as JSON.",
    )
    common_group.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate merged inputs and exit without running the pipeline.",
    )
    common_group.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress banner and final summary output.",
    )
    common_group.add_argument(
        "--traceback",
        action="store_true",
        help="Show full Python traceback on errors.",
    )

    if show_advanced:
        prep_group = parser.add_argument_group("Preparation and campaign options")
        _add_bool_arg(
            prep_group,
            "--receptor-use-meeko",
            dest="receptor_use_meeko",
            help="Enable or disable Meeko during receptor preparation.",
        )
        prep_group.add_argument(
            "--ligand-output-format",
            type=str,
            help="Override ligand output format, for example pdbqt or sdf.",
        )
        prep_group.add_argument(
            "--ligand-backend",
            type=str,
            help="Override ligand conversion backend, for example meeko.",
        )
        prep_group.add_argument(
            "--box-algorithm",
            choices=("pad", "scale"),
            help=(
                "Override the ligand-derived box algorithm. The default is "
                "isotropic 4-Angstrom padding; scale remains available for "
                "legacy campaigns."
            ),
        )
        prep_group.add_argument(
            "--box-pad",
            type=float,
            help="Override symmetric ligand-derived box padding in Angstrom.",
        )
        prep_group.add_argument(
            "--box-scale",
            type=float,
            help=(
                "Override ligand-derived box scale factor. When supplied "
                "without --box-algorithm, this selects legacy scale behavior."
            ),
        )
        _add_bool_arg(
            prep_group,
            "--box-isotropic",
            dest="box_isotropic",
            help="Use isotropic or anisotropic ligand-derived grid boxes.",
        )
        prep_group.add_argument(
            "--campaign-name",
            type=str,
            help="Override output campaign JSON filename.",
        )
        prep_group.add_argument(
            "--crawl-backend",
            type=str,
            help="Override pose crawler backend, for example obabel.",
        )

        interaction_group = parser.add_argument_group("Interaction extraction options")
        interaction_group.add_argument(
            "--interaction-batch-size",
            type=_positive_int,
            help="Override interaction extraction batch size.",
        )
        _add_bool_arg(
            interaction_group,
            "--interaction-progress",
            dest="interaction_progress",
            help="Enable or disable progress output for interaction extraction.",
        )
        interaction_group.add_argument(
            "--interaction-n-jobs",
            dest="interaction_n_jobs",
            type=_positive_int,
            help="Override number of interaction extraction jobs.",
        )
        _add_bool_arg(
            interaction_group,
            "--include-fingerprint-columns",
            dest="include_fingerprint_columns",
            help="Include or exclude fingerprint columns in interaction output.",
        )
        _add_bool_arg(
            interaction_group,
            "--include-interaction-events",
            dest="include_interaction_events",
            help="Include or exclude long-form interaction events in output.",
        )
        _add_bool_arg(
            interaction_group,
            "--include-bitvectors",
            dest="include_bitvectors",
            help="Include or exclude bitvectors in interaction output.",
        )
        _add_bool_arg(
            interaction_group,
            "--include-countvectors",
            dest="include_countvectors",
            help="Include or exclude countvectors in interaction output.",
        )
        _add_bool_arg(
            interaction_group,
            "--fail-fast",
            dest="fail_fast",
            help="Fail immediately or continue collecting interaction errors when possible.",
        )
        _add_bool_arg(
            interaction_group,
            "--use-interaction-profiler",
            dest="use_interaction_profiler",
            help="Use InteractionProfiler instead of extract_pose_table_interactions.",
        )
        _add_bool_arg(
            interaction_group,
            "--receptor-guess-bonds",
            dest="receptor_guess_bonds",
            help=(
                "Guess receptor bonds with ProLIF during interaction extraction. "
                "Default is disabled; enabling it can segfault on some receptors."
            ),
        )

        db_group = parser.add_argument_group("Database insertion options")
        _add_bool_arg(
            db_group,
            "--replace",
            dest="replace",
            help="Replace or keep existing pose rows during database insertion.",
        )
        _add_bool_arg(
            db_group,
            "--replace-interactions",
            dest="replace_interactions",
            help="Replace or keep existing interaction rows during database insertion.",
        )

        debug_group = parser.add_argument_group("Debugging and reproducibility")
        debug_group.add_argument(
            "--effective-config-json",
            type=str,
            default=None,
            help="Optional path to write the merged effective run configuration as JSON.",
        )
        debug_group.add_argument(
            "--print-effective-config",
            action="store_true",
            help="Print the merged effective configuration before execution.",
        )

    return parser


def _apply_cli_overrides(
    args: argparse.Namespace,
    kwargs: MutableMapping[str, Any],
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
        "box_algorithm",
        "box_pad",
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
        "receptor_guess_bonds",
        "save_to_database",
        "db_name",
        "replace",
        "replace_interactions",
    }

    for key in override_keys:
        if hasattr(args, key):
            value = getattr(args, key)
            if value is not None:
                kwargs[key] = value


def _validate_run_inputs(kwargs: Mapping[str, Any]) -> None:
    """Validate the minimum high-level CLI requirements before execution."""
    project_dir = kwargs.get("project_dir")
    if not project_dir:
        raise CLIError(
            "Missing 'project_dir'. Provide it in the JSON config or via --project-dir."
        )

    has_raw_receptors = kwargs.get("receptors") is not None
    has_prepared_receptors = kwargs.get("prepared_receptors") is not None
    if has_raw_receptors == has_prepared_receptors:
        raise CLIError(
            "Provide exactly one receptor mode: 'receptors' or 'prepared_receptors'."
        )

    if has_raw_receptors and not kwargs["receptors"]:
        raise CLIError("'receptors' is present but empty.")

    if has_prepared_receptors and not kwargs["prepared_receptors"]:
        raise CLIError("'prepared_receptors' is present but empty.")

    has_ligands = kwargs.get("ligands") is not None
    has_ligand_dir = kwargs.get("ligand_dir") is not None
    if has_ligands == has_ligand_dir:
        raise CLIError("Provide exactly one ligand mode: 'ligands' or 'ligand_dir'.")

    if has_ligands and not kwargs["ligands"]:
        raise CLIError("'ligands' is present but empty.")

    if has_ligand_dir and not kwargs["ligand_dir"]:
        raise CLIError("'ligand_dir' is present but empty.")


def _result_summary(result: ProDockResult) -> JSONDict:
    """Convert the pipeline result to a compact JSON-serializable summary."""
    pose_df = result.pose_df
    merged_df = result.merged_df

    engines: list[str] = []
    ligand_ids: list[str] = []

    if pose_df is not None and not pose_df.empty:
        if "engine" in pose_df.columns:
            engines = sorted({str(x) for x in pose_df["engine"].dropna().unique()})
        if "ligand_id" in pose_df.columns:
            ligand_ids = sorted(
                {str(x) for x in pose_df["ligand_id"].dropna().unique()}
            )

    return {
        "project_dir": str(result.project_dir),
        "ligand_dir": str(result.ligand_dir),
        "campaign_json": str(result.campaign_json),
        "db_path": str(result.db_path) if result.db_path is not None else None,
        "receptor_ids": [spec.receptor_id for spec in result.receptors],
        "ligand_ids": ligand_ids,
        "engines": engines,
        "n_receptors": len(result.receptors),
        "n_unique_ligands": len(ligand_ids),
        "n_engines": len(engines),
        "n_pose_rows": int(len(pose_df)),
        "n_merged_rows": int(len(merged_df)),
        "interaction_extracted": result.interaction_result is not None,
        "has_interaction_df": result.interaction_df is not None,
        "has_summary_df": result.summary_df is not None,
    }


def _write_json_file(path: Path, payload: Mapping[str, Any]) -> None:
    """Write a JSON payload to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _resolve_help_mode(argv: Sequence[str]) -> str | None:
    """Return help mode: 'normal', 'advanced', or None."""
    args = list(argv)

    if args[:2] == ["-h", "advanced"] or args[:2] == ["--help", "advanced"]:
        return "advanced"

    if "--help-advanced" in args:
        return "advanced"

    if "-h" in args or "--help" in args:
        return "normal"

    return None


def main(argv: Sequence[str] | None = None) -> int:
    """Run the ProDock command-line interface.

    :param argv:
        Optional command-line argument sequence. If ``None``, arguments are read
        from ``sys.argv``.
    :return:
        Exit status code. Returns ``0`` on success and ``1`` on handled failure.
    """
    argv_list = list(sys.argv[1:] if argv is None else argv)

    help_mode = _resolve_help_mode(argv_list)
    if help_mode == "advanced":
        _build_parser(show_advanced=True).print_help()
        return 0
    if help_mode == "normal":
        _build_parser(show_advanced=False).print_help()
        return 0

    parser = _build_parser(show_advanced=True)
    args = parser.parse_args(argv_list)

    try:
        if not args.quiet:
            print(BANNER, file=sys.stderr)

        config_path = args.config.expanduser().resolve()
        if not config_path.exists():
            raise CLIError(f"Config file not found: {config_path}")
        if not config_path.is_file():
            raise CLIError(f"Config path is not a file: {config_path}")

        payload = _load_json(config_path)
        run_kwargs = _normalize_payload(payload, source_path=config_path)

        _overlay_json_section(
            run_kwargs,
            json_path=args.receptor_json,
            expected_key="receptors",
            incompatible_key="prepared_receptors",
        )
        _overlay_json_section(
            run_kwargs,
            json_path=args.ligand_json,
            expected_key="ligands",
            incompatible_key="ligand_dir",
        )

        _apply_cli_overrides(args, run_kwargs)
        _validate_run_inputs(run_kwargs)

        if getattr(args, "print_effective_config", False):
            print("[prodock] Effective run configuration:", file=sys.stderr)
            print(json.dumps(run_kwargs, indent=2), file=sys.stderr)

        effective_config_path = _resolve_output_path(
            getattr(args, "effective_config_json", None),
            base_dir=config_path.parent.resolve(),
        )
        if effective_config_path is not None:
            _write_json_file(effective_config_path, run_kwargs)

        if args.validate_only:
            if not args.quiet:
                print("[prodock] Configuration is valid.", file=sys.stderr)
            return 0

        result = prodock(**run_kwargs)
        summary = _result_summary(result)

        summary_path = _resolve_output_path(
            args.summary_json,
            base_dir=config_path.parent.resolve(),
        )
        if summary_path is not None:
            _write_json_file(summary_path, summary)

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
