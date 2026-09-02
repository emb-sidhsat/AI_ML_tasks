"""Tests for the standalone pipeline dispatcher and CLI."""
from unittest.mock import patch

import pytest

from src.pipeline import cli
from src.pipeline.runner import run_all, run_phase

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("phase,stage_name", [
    ("1", "run_ingestion"),
    ("2", "run_preprocessing"),
    ("4", "run_aggregation"),
    ("5", "run_validation"),
])
def test_run_phase_dispatches_to_requested_stage(phase, stage_name):
    with patch(f"src.pipeline.runner.stages.{stage_name}") as stage:
        run_phase(phase)
    stage.assert_called_once_with()


def test_run_phase_passes_semantic_flag_to_scoring():
    with patch("src.pipeline.runner.stages.run_scoring") as scoring:
        run_phase(3, skip_semantic=True)
    scoring.assert_called_once_with(skip_semantic=True)


def test_run_phase_rejects_unknown_phase():
    with pytest.raises(ValueError, match="phase must be one of"):
        run_phase("9")


def test_run_all_runs_phases_in_dependency_order():
    with patch("src.pipeline.runner.run_phase") as dispatch:
        run_all(skip_semantic=True)
    assert [call.args[0] for call in dispatch.call_args_list] == ["1", "2", "3", "4", "5"]
    assert dispatch.call_args_list[2].kwargs == {"skip_semantic": True}


def test_cli_routes_score_and_semantic_flag():
    with patch("src.pipeline.cli.run_phase") as dispatch:
        cli.main(["score", "--skip-semantic"])
    dispatch.assert_called_once_with("3", skip_semantic=True)


def test_cli_routes_run_all():
    with patch("src.pipeline.cli.run_all") as dispatch:
        cli.main(["run-all"])
    dispatch.assert_called_once_with(skip_semantic=False)