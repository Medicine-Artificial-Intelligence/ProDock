"""Metadata-only cluster preflight: no docking, preparation, profiling or DB writes."""

from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strict-environment", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    checks = []

    def check(name, passed, detail):
        checks.append(dict(check=name, passed=bool(passed), detail=detail))

    base = ROOT / "Data/validation/inputs"
    path = base / "chembl_egfr/library.json"
    library = json.loads(path.read_text())
    expected = (path.parent / "library.sha256").read_text().split()[0]
    check("frozen_library_sha256", sha(path) == expected, sha(path))
    check(
        "full_library_count", len(library["ligands"]) == 1000, len(library["ligands"])
    )
    check(
        "unique_compound_identifiers",
        len({r["id"] for r in library["ligands"]}) == 1000,
        "IDs are distinct",
    )
    manifest = base / "egfr_structures/manifest.json"
    data = json.loads(manifest.read_text())
    check(
        "two_declared_EGFR_structures",
        {r["pdb"] for r in data["structures"]} == {"1M17", "4ZAU"},
        "1M17 and 4ZAU",
    )
    for row in data["structures"]:
        for key, expected_hash in row["sha256"].items():
            p = manifest.parent / row[key]
            check(
                f"{row['pdb']}:{key}",
                p.is_file() and sha(p) == expected_hash,
                str(p.relative_to(ROOT)),
            )
    cached = base / "comparison_native"
    sources = json.loads((cached / "source_manifest.json").read_text())
    for name, record in sources.items():
        for entry in record["files"]:
            p = cached / "sources" / name / entry["path"]
            check(
                f"pinned_native_source:{name}/{entry['path']}",
                p.is_file() and sha(p) == entry["sha256"],
                entry["sha256"],
            )
    for name in ["smina", "qvina"]:
        p = ROOT / "prodock/dock/binary" / name
        check(
            "executable:" + name,
            p.is_file() and os.access(p, os.X_OK),
            str(p.relative_to(ROOT)),
        )
    required = [
        "rdkit",
        "numpy",
        "pandas",
        "scipy",
        "matplotlib",
        "psutil",
        "meeko",
        "gemmi",
        "prolif",
        "MDAnalysis",
        "pdbfixer",
        "openmm",
        "pymol",
    ]
    environment = {m: importlib.util.find_spec(m) is not None for m in required}
    report = dict(
        mode="metadata_only",
        scientific_execution=False,
        python=sys.executable,
        checks=checks,
        passed=sum(r["passed"] for r in checks),
        total=len(checks),
        environment_modules=environment,
        environment_ready=all(environment.values()),
        caveat="File/dependency availability does not qualify receptor chemistry or validate the native controls.",
    )
    text = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    print(text)
    if not all(r["passed"] for r in checks) or (
        args.strict_environment and not report["environment_ready"]
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
