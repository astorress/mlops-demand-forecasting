from unittest.mock import MagicMock

import pytest

from src.services import athena
from src.utilities.errors import AthenaQueryError


def _client(state):
    client = MagicMock()
    client.start_query_execution.return_value = {
        "QueryExecutionId": "q-1"
    }
    client.get_query_execution.return_value = {
        "QueryExecution": {
            "Status": {
                "State": state,
                "StateChangeReason": "boom",
            },
        },
    }
    return client


def test_run_query_succeeds(monkeypatch):
    monkeypatch.setattr(athena, "CLIENT", _client("SUCCEEDED"))
    assert athena.run_query("SELECT 1", "wg") == "q-1"


def test_run_query_fails(monkeypatch):
    monkeypatch.setattr(athena, "CLIENT", _client("FAILED"))
    with pytest.raises(AthenaQueryError, match="boom"):
        athena.run_query("SELECT 1", "wg")
