#!/usr/bin/env python3
"""Run the EGFR redocking and screening campaigns (single entry point).

For each receptor in ``egfr_config.RECEPTORS`` the co-crystallized native
ligand is extracted with :class:`PDBQuery`, its bond orders repaired from the
RCSB *ideal* ligand definition (PDB HETATM records carry no bond orders), and
saved as the RMSD ground truth / autobox reference. Two phases then run,
each checkpointed per receptor so an interrupted run can simply be
re-launched:

1. Redocking: the native ligand is redocked into its own receptor, in an
   isolated per-receptor project directory (no cross-docking), across every
   engine. Written to ``runs/redocking/redocking.db``.
2. Screening: the full ligand library (actives + decoys) is docked against
   every receptor and engine, with interaction extraction enabled. Written to
   ``runs/screening/screening.db``.

Run:
    python Data/benchmark/run_benchmark.py
"""

from __future__ import annotations

import json
import urllib.request
from pathlib import Path

import egfr_config as cfg  # noqa: E402  (must precede prodock import)

from rdkit import Chem  # noqa: E402
from rdkit.Chem import AllChem  # noqa: E402

from prodock import prodock  # noqa: E402
from prodock.database import PoseDatabase, PoseQuery  # noqa: E402
from prodock.structure import PDBQuery  # noqa: E402

log_redock = cfg.get_logger("redock")
log_screen = cfg.get_logger("screen")

RCSB_IDEAL = "https://files.rcsb.org/ligands/download/{code}_ideal.sdf"


def _ideal_ligand(code: str, cache_dir: Path):
    """Fetch the RCSB ideal ligand (correct bond orders) as an RDKit mol."""
    cache = cache_dir / f"{code}_ideal.sdf"
    if not cache.exists():
        url = RCSB_IDEAL.format(code=code)
        log_redock.info("[%s] fetching ideal ligand definition from RCSB", code)
        urllib.request.urlretrieve(url, cache)  # noqa: S310 (trusted host)
    mol = Chem.SDMolSupplier(str(cache), removeHs=True)[0]
    return Chem.RemoveHs(mol) if mol is not None else None


def prepare_native(pdb: str, code: str, chains, workdir: Path):
    """Return (smiles, reference_sdf_path) with correct chemistry, or (None, None)."""
    # 1. crystal pose (3D coords, but lossy bond orders)
    q = PDBQuery(
        pdb_id=pdb,
        output_dir=str(workdir / pdb),
        chains=chains,
        ligand_code=code,
    ).run_all()
    ref_src = q.reference_ligand_path or q.cocrystal_ligand_path
    if not ref_src:
        log_redock.warning("[%s] no reference ligand extracted", pdb)
        return None, None
    crystal = Chem.SDMolSupplier(str(ref_src), sanitize=False, removeHs=True)[0]
    if crystal is None:
        log_redock.warning("[%s] could not read crystal ligand", pdb)
        return None, None
    # Some structures deposit multiple copies of the ligand in one record; keep
    # the single largest fragment as the reference pose.
    frags = Chem.GetMolFrags(crystal, asMols=True, sanitizeFrags=False)
    if len(frags) > 1:
        crystal = max(frags, key=lambda m: m.GetNumAtoms())
        log_redock.info(
            "[%s] crystal had %d fragments; kept largest (%d atoms)", pdb, len(frags), crystal.GetNumAtoms()
        )

    # 2. template with correct bond orders from RCSB
    template = _ideal_ligand(code, cfg.REF_DIR)
    if template is None:
        log_redock.warning("[%s] could not load ideal ligand %s", pdb, code)
        return None, None
    smiles = Chem.MolToSmiles(template)

    # 3. transfer correct bond orders onto the crystal pose (RMSD ground truth).
    # If the crystal ligand has incomplete density (fewer atoms than the ideal
    # definition), template matching fails and the structure cannot be used as
    # a redocking ground truth -- skip it (it is still used for screening).
    if crystal.GetNumAtoms() != template.GetNumAtoms():
        log_redock.warning(
            "[%s] crystal ligand incomplete (%d/%d heavy atoms); " "excluding from redocking benchmark",
            pdb,
            crystal.GetNumAtoms(),
            template.GetNumAtoms(),
        )
        return None, None
    try:
        fixed = AllChem.AssignBondOrdersFromTemplate(template, crystal)
    except Exception as exc:
        log_redock.warning("[%s] bond-order assignment failed (%s); excluding", pdb, exc)
        return None, None
    ref_dst = cfg.REF_DIR / f"{pdb}_native.sdf"
    w = Chem.SDWriter(str(ref_dst))
    w.write(fixed)
    w.close()
    return smiles, ref_dst


def _redocking_complete(pdb_id: str) -> bool:
    if not cfg.REDOCK_DB.exists():
        return False
    try:
        query = PoseQuery(str(cfg.REDOCK_DB))
        for engine in cfg.ENGINES:
            pose = query.pose(
                receptor_id=pdb_id,
                ligand_id=f"native_{pdb_id}",
                engine=engine,
                pose_rank=1,
            )
            if pose is None or pose.mol is None:
                return False
        return True
    except Exception:
        return False


def run_redocking_phase() -> None:
    cfg.ensure_dirs()
    proj_root = cfg.REDOCK_DIR / "projects"
    proj_root.mkdir(parents=True, exist_ok=True)

    log_redock.info("START redocking campaign (%d receptors x %d engines)", len(cfg.RECEPTORS), len(cfg.ENGINES))
    for i, rec in enumerate(cfg.RECEPTORS, 1):
        pdb, code = rec["pdb_id"], rec["ligand_code"]
        if _redocking_complete(pdb):
            log_redock.info("[%d/%d] %s : checkpoint, already complete", i, len(cfg.RECEPTORS), pdb)
            continue

        log_redock.info("[%d/%d] %s : preparing native ligand %s", i, len(cfg.RECEPTORS), pdb, code)
        smi, ref = prepare_native(pdb, code, rec["chains"], cfg.REF_DIR)
        if not smi:
            log_redock.warning("[%s] skipping (no usable native ligand)", pdb)
            continue
        log_redock.info("[%s] native SMILES (RCSB-corrected): %s", pdb, smi)

        # Redock into an isolated project dir -> only this native is present.
        # Use the same single-fragment, bond-corrected crystal ligand for both
        # box construction and RMSD.
        log_redock.info("[%s] redocking native_%s across %s", pdb, pdb, ",".join(cfg.ENGINES))
        dock_rec = dict(rec)
        dock_rec["reference_ligand"] = str(ref)
        prodock(
            project_dir=str(proj_root / pdb),
            receptors=[dock_rec],
            ligands=[{"id": f"native_{pdb}", "smiles": smi}],
            engines=cfg.ENGINES,
            extract_interaction=False,
            save_to_database=True,
            db_name=str(cfg.REDOCK_DB),
            replace=True,
            **cfg.DOCK_PARAMS,
        )
        log_redock.info("[%s] done", pdb)

    log_redock.info("references in %s", cfg.REF_DIR)
    log_redock.info("database: %s", cfg.REDOCK_DB)
    log_redock.info("DONE redocking campaign")


def _screening_complete(pdb_id: str, n_ligands: int) -> bool:
    if not cfg.SCREEN_DB.exists():
        return False
    try:
        db = PoseDatabase(str(cfg.SCREEN_DB))
        count = db.count_poses(receptor_id=pdb_id, pose_rank=1)
        return count >= n_ligands * len(cfg.ENGINES)
    except Exception:
        return False


def run_screening_phase() -> None:
    cfg.ensure_dirs()
    ligands = json.loads(cfg.LIGAND_JSON.read_text())["ligands"]
    log_screen.info("START screening campaign")
    log_screen.info(
        "inputs: %d ligands (%d actives) x %d receptors x %d engines",
        len(ligands),
        len(cfg.ACTIVES),
        len(cfg.RECEPTORS),
        len(cfg.ENGINES),
    )
    log_screen.info("receptors: %s", ", ".join(r["pdb_id"] for r in cfg.RECEPTORS))
    log_screen.info("engines:   %s", ", ".join(cfg.ENGINES))
    log_screen.info("output db: %s", cfg.SCREEN_DB)

    for i, rec in enumerate(cfg.RECEPTORS, 1):
        pdb = rec["pdb_id"]
        if _screening_complete(pdb, len(ligands)):
            log_screen.info("[%d/%d] %s : checkpoint, already complete", i, len(cfg.RECEPTORS), pdb)
            continue

        log_screen.info(
            "[%d/%d] %s : docking %d ligands x %d engines", i, len(cfg.RECEPTORS), pdb, len(ligands), len(cfg.ENGINES)
        )
        result = prodock(
            project_dir=str(cfg.SCREEN_DIR / pdb),
            receptors=[rec],
            ligands=ligands,
            engines=cfg.ENGINES,
            extract_interaction=True,
            use_interaction_profiler=True,
            include_bitvectors=True,  # needed for the fingerprint heatmap
            save_to_database=True,
            db_name=str(cfg.SCREEN_DB),
            replace=True,
            **cfg.DOCK_PARAMS,
        )
        log_screen.info("[%s] done | campaign json: %s", pdb, result.campaign_json)

    log_screen.info("database: %s", cfg.SCREEN_DB)
    log_screen.info("DONE screening campaign")


def main() -> None:
    run_redocking_phase()
    run_screening_phase()


if __name__ == "__main__":
    main()
