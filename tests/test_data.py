"""The snapshot file: downloads replace it only on purpose, and results say what it is.
No network: every client here is a mock."""

import hashlib
from pathlib import Path

import httpx
import pytest

from sanctions_mcp.data import (
    COLLECTION,
    LICENSE,
    SNAPSHOT,
    ChecksumError,
    NotPinnedSnapshotError,
    default_data_dir,
    describe_list,
    download,
    load_watchlist,
    sha256_of,
    snapshot_path,
)

FIXTURE_LIST = snapshot_path(Path(__file__).parent / "fixtures" / "data")
BODY = b"id,schema,name\n"


def serving(body: bytes = BODY, status: int = 200) -> httpx.Client:
    return httpx.Client(
        transport=httpx.MockTransport(lambda request: httpx.Response(status, content=body))
    )


def no_network() -> httpx.Client:
    def refuse(request):
        raise AssertionError(f"nothing should be downloaded, but {request.url} was requested")

    return httpx.Client(transport=httpx.MockTransport(refuse))


def place(data_dir: Path, body: bytes) -> Path:
    dest = snapshot_path(data_dir)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(body)
    return dest


def test_a_file_that_is_not_the_snapshot_is_replaced_only_with_force(tmp_path):
    dest = place(tmp_path, b"a file placed by hand\n")
    with pytest.raises(NotPinnedSnapshotError, match="--force"):
        download(tmp_path, client=no_network())
    assert dest.read_bytes() == b"a file placed by hand\n"

    good = hashlib.sha256(BODY).hexdigest()
    path = download(tmp_path, client=serving(), url="https://x.test/f.csv", sha256=good, force=True)
    assert path == dest and dest.read_bytes() == BODY


def test_the_pinned_file_is_kept_without_a_download(tmp_path):
    dest = place(tmp_path, BODY)
    assert download(tmp_path, client=no_network(), sha256=hashlib.sha256(BODY).hexdigest()) == dest


def test_the_fictional_list_is_never_replaced(tmp_path):
    dest = place(tmp_path, FIXTURE_LIST.read_bytes())
    with pytest.raises(NotPinnedSnapshotError):
        download(tmp_path, client=no_network())
    assert dest.read_bytes() == FIXTURE_LIST.read_bytes()


@pytest.mark.parametrize(("body", "status"), [(b"", 404), (b"<html>moved</html>", 200)])
def test_a_failed_download_leaves_nothing_behind(tmp_path, body, status):
    expected = ChecksumError if status == 200 else httpx.HTTPStatusError
    with pytest.raises(expected):
        download(tmp_path, client=serving(body, status), url="https://x.test/f.csv")
    assert list(snapshot_path(tmp_path).parent.iterdir()) == []


def test_a_download_cut_off_midway_leaves_nothing_behind(tmp_path):
    def cut_off(done: int) -> None:  # called once the first bytes are on disk
        assert snapshot_path(tmp_path).with_name("targets.simple.csv.part").exists()
        raise httpx.ReadError("connection lost")

    with pytest.raises(httpx.ReadError):
        download(
            tmp_path, client=serving(b"x" * 10), url="https://x.test/f.csv", on_progress=cut_off
        )
    assert list(snapshot_path(tmp_path).parent.iterdir()) == []


@pytest.mark.parametrize(("value", "expected"), [("", "data"), ("   ", "data"), ("x/y", "x/y")])
def test_an_empty_data_dir_variable_means_the_default(monkeypatch, value, expected):
    monkeypatch.setenv("SANCTIONS_DATA_DIR", value)
    assert default_data_dir() == Path(expected)
    monkeypatch.delenv("SANCTIONS_DATA_DIR")
    assert default_data_dir() == Path("data")


def test_the_fictional_list_stays_fictional():
    rows = load_watchlist(FIXTURE_LIST, schemas=())
    assert rows and all(e.datasets == ("Fictional Test List",) for e in rows)
    assert FIXTURE_LIST.stat().st_size < 10_000  # the real list is about 79 MB


def test_only_the_pinned_file_is_named_as_the_snapshot(monkeypatch):
    plain = describe_list(FIXTURE_LIST, 12)
    assert plain == {
        "source": "unverified list file",
        "collection": None,
        "snapshot": None,
        "license": None,
        "sha256": sha256_of(FIXTURE_LIST),
        "people": 12,
    }
    monkeypatch.setattr("sanctions_mcp.data.SHA256", sha256_of(FIXTURE_LIST))
    pinned = describe_list(FIXTURE_LIST, 12)
    assert (pinned["source"], pinned["collection"], pinned["snapshot"], pinned["license"]) == (
        "OpenSanctions",
        COLLECTION,
        SNAPSHOT,
        LICENSE,
    )
