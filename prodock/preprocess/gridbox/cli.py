from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional
from prodock.preprocess.gridbox.gridbox import GridBox


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        "gridbox",
        description="Compute docking grid boxes from ligands with selectable algorithms.",
    )
    p.add_argument(
        "-l",
        "--ligand",
        required=True,
        help="Ligand file (pdb, pdbqt, sdf, mol2, xyz) or raw text (use --fmt).",
    )
    p.add_argument("--fmt", default=None, help="Format hint when passing raw text.")

    # Algorithm choice
    p.add_argument(
        "--algo",
        default="scale",
        choices=[
            "scale",
            "pad",
            "advanced",
            "percentile",
            "pca-aabb",
            "centroid-fixed",
            "union",
        ],
        help="Which algorithm to use for the grid box.",
    )

    # Common params
    p.add_argument(
        "--scale",
        type=float,
        default=2.0,
        help="Scaling factor for --algo=scale (size = span * scale).",
    )
    p.add_argument(
        "--pad",
        type=float,
        default=4.0,
        help="Padding (Å) per side for --algo=pad/advanced.",
    )
    p.add_argument(
        "--isotropic", action="store_true", help="Use cubic box with the max axis."
    )
    p.add_argument(
        "--min-size",
        type=float,
        default=0.0,
        help="Minimum edge length (Å) after expansion.",
    )
    p.add_argument(
        "--heavy-only",
        action="store_true",
        help="Exclude hydrogens for span (advanced only).",
    )
    p.add_argument(
        "--snap",
        type=float,
        default=None,
        help="Snap step (Å) for center/size (advanced only).",
    )
    p.add_argument(
        "--round",
        dest="round_ndigits",
        type=int,
        default=3,
        help="Decimal places to round.",
    )

    # Percentile algorithm
    p.add_argument(
        "--low",
        type=float,
        default=5.0,
        help="Low percentile (%%) for --algo=percentile.",
    )
    p.add_argument(
        "--high",
        type=float,
        default=95.0,
        help="High percentile (%%) for --algo=percentile.",
    )

    # PCA-AABB algorithm
    p.add_argument(
        "--pca-scale",
        type=float,
        default=1.0,
        help="Scale in PCA frame for --algo=pca-aabb.",
    )
    p.add_argument(
        "--pca-pad",
        type=float,
        default=0.0,
        help="Pad in PCA frame for --algo=pca-aabb.",
    )

    # Centroid-fixed
    p.add_argument(
        "--size",
        type=float,
        nargs=3,
        metavar=("SX", "SY", "SZ"),
        help="Explicit size (Å) for --algo=centroid-fixed (center = ligand centroid).",
    )

    # Union (multi-ligand): accept multiple paths
    p.add_argument("--ligand2", help="Second ligand file for --algo=union.")
    p.add_argument("--ligand3", help="Third ligand file for --algo=union.")

    p.add_argument(
        "--print-center", action="store_true", help="Also print center lines."
    )
    return p


def _validate_primary_file_if_path(ligand: str) -> None:
    """
    If ligand looks like a path and exists, perform a quick extension sanity check.
    Do not fail if ligand is raw text.
    """
    p = Path(ligand)
    if p.exists():
        ext = p.suffix.lower()
        if ext not in (".pdb", ".pdbqt", ".sdf", ".mol2", ".xyz"):
            print(f"Unsupported file format: {ext}")
            sys.exit(2)


def _print_result(gb: GridBox, print_center: bool = False) -> None:
    cx, cy, cz = gb.center
    sx, sy, sz = gb.size
    print("GridBox")
    if print_center:
        print(f"╰─○ Grid Box Center:  X {cx:>8.3f}  Y {cy:>8.3f}  Z {cz:>8.3f}")
    print(f"╰─○ Grid Box Size  :  W {sx:>8.3f}  H {sy:>8.3f}  D {sz:>8.3f}")
    # Also print a Vina-style snippet for convenience
    print("\n# Vina-style snippet")
    print(gb.to_vina_lines())


def main(argv: Optional[List[str]] = None) -> None:
    argv = argv if argv is not None else sys.argv[1:]
    args = build_parser().parse_args(argv)

    # quick file sanity check (only if path exists)
    _validate_primary_file_if_path(args.ligand)

    # build the GridBox
    gb = GridBox().load_ligand(args.ligand, fmt=args.fmt)

    try:
        if args.algo == "scale":
            gb.from_ligand_scale(
                scale=args.scale,
                isotropic=args.isotropic,
                round_ndigits=args.round_ndigits,
            )

        elif args.algo == "pad":
            gb.from_ligand_pad(
                pad=args.pad,
                isotropic=args.isotropic,
                min_size=args.min_size,
                round_ndigits=args.round_ndigits,
            )

        elif args.algo == "advanced":
            gb.from_ligand_pad_adv(
                pad=args.pad,
                isotropic=args.isotropic,
                min_size=args.min_size,
                heavy_only=args.heavy_only,
                snap_step=args.snap,
                round_ndigits=args.round_ndigits,
            )

        elif args.algo == "percentile":
            gb.from_ligand_percentile(
                low=args.low,
                high=args.high,
                pad=args.pad,
                isotropic=args.isotropic,
                round_ndigits=args.round_ndigits,
            )

        elif args.algo == "pca-aabb":
            gb.from_ligand_pca_aabb(
                scale=args.pca_scale,
                pad=args.pca_pad,
                isotropic=args.isotropic,
                round_ndigits=args.round_ndigits,
            )

        elif args.algo == "centroid-fixed":
            if not args.size or len(args.size) != 3:
                print("For --algo=centroid-fixed you must provide --size SX SY SZ")
                sys.exit(2)
            gb.from_centroid_fixed(tuple(float(x) for x in args.size))

        elif args.algo == "union":
            paths = [args.ligand] + [p for p in [args.ligand2, args.ligand3] if p]
            gb.from_union(
                paths, fmt=args.fmt, pad=args.pad, round_ndigits=args.round_ndigits
            )

        else:
            print(f"Unknown --algo: {args.algo}")
            sys.exit(2)

    except Exception as exc:
        print(f"Error while computing grid box: {exc}")
        sys.exit(3)

    _print_result(gb, print_center=args.print_center)


if __name__ == "__main__":
    main()
