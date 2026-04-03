from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from prodock.core import ProDockResult, prodock


def _load_json_file(path: str) -> Any:
    """
    Load a JSON file.

    :param path:
        Path to the JSON file.
    :type path: str

    :returns:
        Parsed JSON content.
    :rtype: Any

    :raises FileNotFoundError:
        If the file does not exist.
    :raises json.JSONDecodeError:
        If the file is not valid JSON.
    """
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"JSON file not found: {file_path}")
    with file_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _parse_json_list_file(
    path: Optional[str],
    *,
    label: str,
) -> Optional[List[Dict[str, Any]]]:
    """
    Load a JSON file expected to contain a list of dictionaries.

    :param path:
        Path to the JSON file or ``None``.
    :type path: Optional[str]
    :param label:
        Human-readable label used in validation errors.
    :type label: str

    :returns:
        Parsed list of dictionaries, or ``None`` if no path is provided.
    :rtype: Optional[List[Dict[str, Any]]]

    :raises TypeError:
        If the JSON content is not a list.
    """
    if path is None:
        return None

    obj = _load_json_file(path)
    if not isinstance(obj, list):
        raise TypeError(f"{label} JSON must contain a list of objects.")
    return obj


def _build_parser() -> argparse.ArgumentParser:
    """
    Build the command-line argument parser.

    :returns:
        Configured CLI parser.
    :rtype: argparse.ArgumentParser
    """
    parser = argparse.ArgumentParser(
        prog="prodock",
        description=(
            "Run ProDock receptor/ligand preparation, campaign generation, "
            "and batch docking from the command line."
        ),
    )

    parser.add_argument(
        "project_dir",
        help="Root project directory used for generated files and campaign output.",
    )

    receptor_group = parser.add_mutually_exclusive_group(required=True)
    receptor_group.add_argument(
        "--receptors-json",
        help=(
            "Path to a JSON file containing raw receptor records compatible "
            "with PDBQuery.process_batch."
        ),
    )
    receptor_group.add_argument(
        "--prepared-receptors-json",
        help=(
            "Path to a JSON file containing prepared receptor records with "
            "'receptor_pdbqt'/'receptor', 'center', and 'size'."
        ),
    )

    ligand_group = parser.add_mutually_exclusive_group(required=True)
    ligand_group.add_argument(
        "--ligands-json",
        help="Path to a JSON file containing ligand records with 'id' and 'smiles'.",
    )
    ligand_group.add_argument(
        "--ligand-dir",
        help="Path to an existing directory containing prepared ligand files.",
    )

    parser.add_argument(
        "--engines",
        nargs="+",
        default=None,
        help="Docking engines to include, e.g. --engines smina qvina vina.",
    )
    parser.add_argument(
        "--cpu",
        type=int,
        default=4,
        help="Per-engine CPU value stored in the campaign. Default: 4.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed stored in the campaign. Default: 42.",
    )
    parser.add_argument(
        "--exhaustiveness",
        type=int,
        default=8,
        help="Docking exhaustiveness stored in the campaign. Default: 8.",
    )
    parser.add_argument(
        "--n-poses",
        dest="n_poses",
        type=int,
        default=10,
        help="Number of poses stored in the campaign. Default: 10.",
    )
    parser.add_argument(
        "--n-jobs",
        dest="n_jobs",
        type=int,
        default=None,
        help="Parallel job count used by BatchDock. Default: same as cpu.",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable progress reporting in BatchDock.",
    )
    parser.add_argument(
        "--receptor-use-meeko",
        action="store_true",
        help="Use Meeko during receptor preparation.",
    )
    parser.add_argument(
        "--ligand-output-format",
        default="pdbqt",
        help="Final ligand output format. Default: pdbqt.",
    )
    parser.add_argument(
        "--ligand-backend",
        default="meeko",
        help="Ligand conversion backend. Default: meeko.",
    )
    parser.add_argument(
        "--box-scale",
        type=float,
        default=2.0,
        help="Scale factor for ligand-derived grid boxes. Default: 2.0.",
    )
    parser.add_argument(
        "--box-anisotropic",
        action="store_true",
        help="Use anisotropic ligand-derived grid boxes instead of isotropic.",
    )
    parser.add_argument(
        "--campaign-name",
        default="campaign.json",
        help="Output campaign JSON filename. Default: campaign.json.",
    )
    parser.add_argument(
        "--print-json",
        action="store_true",
        help="Print a JSON summary of the pipeline result.",
    )

    return parser


def _result_to_dict(result: ProDockResult) -> Dict[str, Any]:
    """
    Convert a :class:`ProDockResult` to a JSON-serializable dictionary.

    :param result:
        Pipeline result object.
    :type result: ProDockResult

    :returns:
        JSON-serializable summary dictionary.
    :rtype: Dict[str, Any]
    """
    return {
        "project_dir": str(result.project_dir),
        "ligand_dir": str(result.ligand_dir),
        "campaign_json": str(result.campaign_json),
        "receptors": [
            {
                "receptor_id": spec.receptor_id,
                "receptor_pdbqt": str(spec.receptor_pdbqt),
                "center": list(spec.center),
                "size": list(spec.size),
            }
            for spec in result.receptors
        ],
        "results": result.results,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    """
    Run the ProDock command-line interface.

    :param argv:
        Optional CLI argument sequence. If ``None``, arguments are taken from
        ``sys.argv``.
    :type argv: Optional[Sequence[str]]

    :returns:
        Process exit code.
    :rtype: int

    Example
    -------
    .. code-block:: bash

        prodock Data/testcase/Multi \\
            --receptors-json receptors.json \\
            --ligands-json ligands.json

    Example
    -------
    .. code-block:: bash

        prodock Data/testcase/Multi \\
            --prepared-receptors-json prepared_receptors.json \\
            --ligand-dir Data/testcase/Multi/ligands \\
            --engines smina qvina \\
            --cpu 4 \\
            --n-jobs 4
    """
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        receptors = _parse_json_list_file(
            args.receptors_json,
            label="Raw receptors",
        )
        prepared_receptors = _parse_json_list_file(
            args.prepared_receptors_json,
            label="Prepared receptors",
        )
        ligands = _parse_json_list_file(
            args.ligands_json,
            label="Ligands",
        )

        result = prodock(
            args.project_dir,
            receptors=receptors,
            prepared_receptors=prepared_receptors,
            ligands=ligands,
            ligand_dir=args.ligand_dir,
            engines=args.engines,
            cpu=args.cpu,
            seed=args.seed,
            exhaustiveness=args.exhaustiveness,
            n_poses=args.n_poses,
            n_jobs=args.n_jobs,
            progress=not args.no_progress,
            receptor_use_meeko=args.receptor_use_meeko,
            ligand_output_format=args.ligand_output_format,
            ligand_backend=args.ligand_backend,
            box_scale=args.box_scale,
            box_isotropic=not args.box_anisotropic,
            campaign_name=args.campaign_name,
        )

        if args.print_json:
            print(json.dumps(_result_to_dict(result), indent=2))
        else:
            print(f"Campaign JSON: {result.campaign_json}")
            print(f"Ligand directory: {result.ligand_dir}")
            print(f"Prepared receptors: {len(result.receptors)}")

        return 0

    except Exception as exc:
        parser.exit(status=1, message=f"Error: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
