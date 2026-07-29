from pathlib import Path
from unittest import mock

from prodock.structure.batch import process_batch


def test_process_batch_accepts_documented_receptor_name(tmp_path: Path) -> None:
    query = mock.Mock()
    query.reference_ligand_path = "reference.sdf"
    query.cocrystal_ligand_path = "cocrystal.sdf"
    query.filtered_protein_path = "protein.pdb"

    with mock.patch("prodock.structure.batch.PDBQuery", return_value=query) as cls:
        result = process_batch(
            [
                {
                    "pdb_id": "4WKQ",
                    "receptor_name": "EGFR_4WKQ",
                    "ligand_code": "IRE",
                    "chains": ["A"],
                }
            ],
            output_dir=tmp_path,
        )

    cls.assert_called_once_with(
        pdb_id="4WKQ",
        output_dir=str(tmp_path / "4WKQ"),
        chains=["A"],
        ligand_code="IRE",
        cofactors=[],
        protein_name="EGFR_4WKQ",
    )
    query.run_all.assert_called_once_with()
    assert result[0]["success"] is True
    assert result[0]["protein_name"] == "EGFR_4WKQ"


def test_process_batch_keeps_existing_protein_name_precedence(
    tmp_path: Path,
) -> None:
    query = mock.Mock()
    query.reference_ligand_path = None
    query.cocrystal_ligand_path = None
    query.filtered_protein_path = None

    with mock.patch("prodock.structure.batch.PDBQuery", return_value=query) as cls:
        process_batch(
            [
                {
                    "pdb_id": "4WKQ",
                    "protein_name": "legacy-name",
                    "receptor_name": "documented-name",
                }
            ],
            output_dir=tmp_path,
        )

    assert cls.call_args.kwargs["protein_name"] == "legacy-name"
