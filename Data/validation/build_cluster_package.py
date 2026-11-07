"""Build a source/input transfer archive. Does not run scientific experiments."""

from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    files = {}
    for name in [
        "pyproject.toml",
        "README.md",
        "REPRODUCIBILITY.md",
        "LICENSE",
        "prodock-env.yml",
        "environment.yml",
        "requirements.txt",
        "requirements-reranking.txt",
        "requirements-dev.txt",
    ]:
        p = ROOT / name
        if p.is_file():
            files[name] = p
    for directory in [
        "prodock",
        "Test",
        "Data/validation",
        "Data/benchmark",
        "Data/case",
    ]:
        for p in (ROOT / directory).rglob("*"):
            if not p.is_file():
                continue
            rel = p.relative_to(ROOT)
            if "__pycache__" in rel.parts or p.suffix == ".pyc":
                continue
            if directory == "Data/validation" and (
                "runs" in rel.parts or "raw" in rel.parts
            ):
                continue
            if directory == "Data/benchmark" and "runs" in rel.parts:
                continue
            files[str(rel)] = p
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        commit = None
    manifest = dict(
        package="ProDock revision cluster source and frozen inputs",
        scientific_results_included=False,
        default_execution="dry-run",
        core_packages=4,
        planned_docking_calls=dict(quality=8, primary=2000, secondary=64, total=2072),
        source_base_commit=commit,
        note=("Exact local source modifications are captured by individual file checksums; "
              "no Git checkout or network is required for cached core tasks."),
        files={
            name: hashlib.sha256(path.read_bytes()).hexdigest()
            for name, path in sorted(files.items())
        },
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.output, "w", zipfile.ZIP_DEFLATED) as z:
        z.comment = (
            b"No experiments autorun. Read prodock_cluster/Data/validation/README.md."
        )
        for name, path in sorted(files.items()):
            z.write(path, "prodock_cluster/" + name)
        z.writestr(
            "prodock_cluster/CLUSTER_PACKAGE_MANIFEST.json",
            json.dumps(manifest, indent=2) + "\n",
        )
    with zipfile.ZipFile(args.output) as z:
        assert z.testzip() is None
        for name, expected in manifest["files"].items():
            assert (
                hashlib.sha256(z.read("prodock_cluster/" + name)).hexdigest()
                == expected
            )
    print(
        f"Verified {len(files)} source/input files; no runtime/results directories. Archive: {args.output}"
    )


if __name__ == "__main__":
    main()
