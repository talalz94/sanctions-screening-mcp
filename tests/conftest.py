"""Shared fixtures. Tests use a small fictional watchlist, never OpenSanctions data."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from sanctions_mcp.data import load_watchlist, snapshot_path
from sanctions_mcp.index import WatchlistIndex
from sanctions_mcp.matcher import Matcher
from sanctions_mcp.scoring import RuleScorer

FIXTURE_DATA = Path(__file__).parent / "fixtures" / "data"


@pytest.fixture(scope="session")
def entities():
    return load_watchlist(snapshot_path(FIXTURE_DATA))


@pytest.fixture(scope="session")
def matcher(entities):
    return Matcher(WatchlistIndex(entities), RuleScorer())


@pytest.fixture
def data_dir(tmp_path):
    """A writable copy of the fixture data directory."""
    target = tmp_path / "data"
    shutil.copytree(FIXTURE_DATA, target)
    return target
