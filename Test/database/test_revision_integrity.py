"""LoLO Blue: chemical/event round-trips and failed-write regression checks."""

from pathlib import Path
import importlib.util

import pytest

SOURCE = (
    Path(__file__).resolve().parents[2] / "Data/validation/integrity_experiments.py"
)
SPEC = importlib.util.spec_from_file_location("revision_integrity_experiments", SOURCE)
EXPERIMENTS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EXPERIMENTS)


@pytest.mark.parametrize("compressed", [False, True])
def test_full_chemical_coordinate_score_and_interaction_roundtrip(tmp_path, compressed):
    report = EXPERIMENTS.roundtrip_case(tmp_path, compressed)
    assert report["status"] == "passed"
    assert report["pose_records"] == 96
    assert report["conformers_checked"] == 192
    assert report["coordinate_axes_checked"] == ["x", "y", "z"]
    assert report["interaction_events"] == 192


@pytest.mark.parametrize(
    "operation,replace",
    [
        ("insert_many", False),
        ("insert_many", True),
        ("insert_dataframe", False),
        ("insert_dataframe", True),
        ("insert_interactions", True),
        ("dataframe_and_interactions", True),
        ("interaction_payload", True),
        ("caller_transaction", True),
    ],
)
def test_failed_public_write_rolls_back_complete_scope(tmp_path, operation, replace):
    report = EXPERIMENTS.atomicity_case(tmp_path, operation, replace=replace)
    assert report["status"] == "passed", report


def test_controlled_keyboard_interrupt_rolls_back_batch(tmp_path):
    report = EXPERIMENTS.atomicity_case(tmp_path, "insert_many", interrupt=True)
    assert report["status"] == "passed", report
    assert report["verified_workflow_checkpoint_resume"] is False


def test_separate_databases_preserve_parameter_variants(tmp_path):
    report = EXPERIMENTS.parameter_history_case(tmp_path)
    assert report["isolated_databases_preserve_variants"] is True
    assert report["independent_pose_histories_retained"] is False


def test_execution_campaign_snapshot_survives_source_file_mutation(tmp_path):
    report = EXPERIMENTS.provenance_case(tmp_path)
    assert report["status"] == "passed", report
