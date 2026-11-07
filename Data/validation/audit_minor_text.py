"""Read-only checks for the twelve explicit Reviewer 1 minor wording requests."""

from pathlib import Path
import argparse
import json
import re

ROOT = Path(__file__).resolve().parents[2]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manuscript", type=Path, default=ROOT / "paper/main.tex")
    p.add_argument("--output", type=Path)
    args = p.parse_args()
    source = args.manuscript.read_text()
    plain = re.sub(r"\\LoLO\{([^{}]*)\}", r"\1", source)
    checks = [
        (
            "1.m01",
            "Detected interactions, rather than calculated interactions",
            "detect interactions" in plain,
        ),
        (
            "1.m02",
            "EGFR expanded at first abstract occurrence",
            "epidermal growth factor receptor (EGFR)" in plain.split("\\maketitle")[0],
        ),
        (
            "1.m03",
            "Repeated abstract integration list removed",
            (
                "Scientific Contribution.} \\LoLO{The contribution is a linked campaign "
                "representation and a relational query interface."
            )
            in source,
        ),
        (
            "1.m04",
            "Molecular docking terminology",
            "Structure-based docking" not in plain
            and "structure-based docking" not in plain,
        ),
        (
            "1.m05",
            "Docking engine explicitly defined",
            "a docking engine means the program" in plain,
        ),
        ("1.m06", "PDB entry terminology", "PDB identifier" not in plain),
        ("1.m07", "PDBQT format wording", "PDBQT} format" in source),
        (
            "1.m08",
            "Represent known recognition patterns",
            "represent known recognition patterns" in plain,
        ),
        (
            "1.m09",
            "Bound ligands rather than inferred experimental procedure",
            "from cocrystal ligands" not in plain
            and "co-crystallized binders" not in plain,
        ),
        (
            "1.m10",
            "Comma after structure in redocking sentence",
            "For each EGFR structure, we redocked" in plain,
        ),
        (
            "1.m11",
            "CCD files/ideal-geometry definition",
            "PDB Chemical Component Dictionary (CCD)" in plain,
        ),
        (
            "1.m12",
            "Best-scored rank-1 terminology",
            "engine-ranked first pose" not in plain and "best-scored pose" in plain,
        ),
    ]
    report = dict(
        manuscript=str(args.manuscript),
        checks=[dict(id=i, request=r, passed=bool(ok)) for i, r, ok in checks],
        passed=sum(ok for _, _, ok in checks),
        total=len(checks),
        caveat="Literal wording checks plus manual editorial review; no docking/scientific validation is implied.",
    )
    text = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    print(text)
    if not all(x[2] for x in checks):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
