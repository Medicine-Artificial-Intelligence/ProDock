# prodock/structure/batch.py
from __future__ import annotations
from typing import Any, Dict, List, Optional, Union
from pathlib import Path

from prodock.io.logging import get_logger
from .core import PDBQuery

logger = get_logger(__name__)


def process_batch(
    items: Union[List[Dict[str, Any]], Any],
    output_dir: Union[str, Path],
    keys: Optional[Dict[str, str]] = None,
    default_kwargs: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    try:
        import pandas as _pd  # type: ignore
    except Exception:
        _pd = None  # type: ignore

    keys = keys or {}
    default_kwargs = default_kwargs or {}

    pdb_id_key = keys.get("pdb_id_key", "pdb_id")
    ligand_key = keys.get("ligand_key", "ligand_code")
    chains_key = keys.get("chains_key", "chains")
    cofactors_key = keys.get("cofactors_key", "cofactors")
    protein_name_key = keys.get("protein_name_key", "protein_name")

    if isinstance(items, list):
        rows = list(items)
    else:
        if _pd is not None:
            try:
                df = _pd.DataFrame(items)
                rows = df.to_dict(orient="records")
            except Exception:
                rows = list(items)
        else:
            rows = list(items)

    results = []
    for idx, row in enumerate(rows):
        pdb_id = row.get(pdb_id_key) or row.get("pdb") or row.get("id")
        protein_name = row.get(protein_name_key) or pdb_id
        ligand_code = row.get(ligand_key) or row.get("ligand") or ""
        chains = row.get(chains_key) or row.get("chain") or []
        cofactors = row.get(cofactors_key) or []

        try:
            per_out = Path(output_dir) / str(pdb_id)
            proc = PDBQuery(
                pdb_id=str(pdb_id),
                output_dir=str(per_out),
                chains=chains,
                ligand_code=str(ligand_code),
                cofactors=cofactors,
                protein_name=str(protein_name),
                **(default_kwargs or {}),
            )
            proc.run_all()
            results.append(
                {
                    "pdb_id": pdb_id,
                    "protein_name": protein_name,
                    "reference": proc.reference_ligand_path,
                    "cocrystal": proc.cocrystal_ligand_path,
                    "filtered_protein": proc.filtered_protein_path,
                    "success": True,
                    "error": None,
                }
            )
        except Exception as exc:
            logger.exception("Failed to process row %s (pdb=%s): %s", idx, pdb_id, exc)
            results.append(
                {
                    "pdb_id": pdb_id,
                    "protein_name": protein_name,
                    "reference": None,
                    "cocrystal": None,
                    "filtered_protein": None,
                    "success": False,
                    "error": str(exc),
                }
            )
    return results
