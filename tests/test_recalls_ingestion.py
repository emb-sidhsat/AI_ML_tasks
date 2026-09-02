"""Unit tests for NHTSA recall ingestion."""
from unittest.mock import Mock

import pandas as pd
import pytest

import config
from src.ingestion import recalls

pytestmark = pytest.mark.unit


def test_fetch_uses_recall_api_endpoint(monkeypatch):
    response = Mock(status_code=200)
    response.json.return_value = {"results": [{"NHTSACampaignNumber": "20V000000"}]}
    request = Mock(return_value=response)
    monkeypatch.setattr(recalls.requests, "get", request)
    monkeypatch.setattr(config, "YEARS_TO_TARGET", [2020])
    monkeypatch.setattr(config, "API_DELAY", 0)

    result = recalls.fetch_recalls_for_vehicle("HONDA", "CIVIC")

    assert len(result) == 1
    assert request.call_args.args[0].endswith("/recalls/recallsByVehicle")
    assert request.call_args.kwargs["params"] == {
        "make": "HONDA",
        "model": "CIVIC",
        "modelYear": 2020,
    }


def test_all_failed_recall_requests_raise(monkeypatch):
    response = Mock(status_code=403, text='{"message":"Missing Authentication Token"}')
    monkeypatch.setattr(recalls.requests, "get", Mock(return_value=response))
    monkeypatch.setattr(config, "YEARS_TO_TARGET", [2020])
    monkeypatch.setattr(config, "API_DELAY", 0)

    with pytest.raises(RuntimeError, match="every target year"):
        recalls.fetch_recalls_for_vehicle("HONDA", "CIVIC")


def test_empty_malformed_cache_is_refetched(tmp_path, monkeypatch):
    (tmp_path / "recalls_raw.csv").write_text(" \n", encoding="utf-8")
    monkeypatch.setattr(config, "DATA_RAW", tmp_path)
    monkeypatch.setattr(config, "FORCE_REFETCH", False)
    monkeypatch.setattr(config, "API_DELAY", 0)
    monkeypatch.setattr(
        recalls,
        "fetch_recalls_for_vehicle",
        lambda make, model: [{
            "ReportReceivedDate": "01/01/2020",
            "make_pulled": make,
            "model_pulled": model,
            "year_pulled": 2020,
        }],
    )

    result = recalls.load_or_fetch_recalls([("HONDA", "CIVIC")])

    assert len(result) == 1
    assert pd.read_csv(tmp_path / "recalls_raw.csv").shape[0] == 1


def test_vehicle_with_all_failed_recall_requests_is_skipped(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DATA_RAW", tmp_path)
    monkeypatch.setattr(config, "FORCE_REFETCH", True)
    monkeypatch.setattr(
        recalls,
        "fetch_recalls_for_vehicle",
        Mock(side_effect=[
            RuntimeError("Recall API failed for every target year for FORD F-150 REGULAR CAB"),
            [{"NHTSACampaignNumber": "20V000000"}],
        ]),
    )

    result = recalls.load_or_fetch_recalls([
        ("FORD", "F-150 REGULAR CAB"),
        ("HONDA", "CIVIC"),
    ])

    assert len(result) == 1
    assert result.iloc[0]["NHTSACampaignNumber"] == "20V000000"