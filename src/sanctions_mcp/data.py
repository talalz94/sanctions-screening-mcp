"""Download and load the OpenSanctions snapshot this repo is pinned to.

The data is CC BY-NC 4.0 and is never committed here. `sanctions download` fetches one
immutable snapshot, checks its SHA-256 and keeps it in the local data directory.
"""

from __future__ import annotations

import csv
import hashlib
import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import httpx

SNAPSHOT = "20261109194706-bbu"
COLLECTION = "sanctions"
FILENAME = "targets.simple.csv"
SNAPSHOT_URL = f"https://data.opensanctions.org/artifacts/{COLLECTION}/{SNAPSHOT}/{FILENAME}"
SHA256 = "eef1d4021b079ee8e2a2149180d41681efe01c14e0a81a422c4cf57bb8bf6697"
SHA1_PUBLISHED = "98d9b4f67e4ee0d250e3334a03d20304e0524fdc"  # from the snapshot's index.json
LICENSE = "CC BY-NC 4.0, noncommercial use only"

NOTICE = f"""\
OpenSanctions data notice
=========================
Source:  https://www.opensanctions.org, collection "{COLLECTION}", snapshot {SNAPSHOT}
License: Creative Commons Attribution NonCommercial 4.0 (CC BY-NC 4.0)

The data is free for noncommercial use. OpenSanctions treats any use inside a business,
including compliance screening, as commercial use that needs a data license from them.
This project is a personal, noncommercial research project. The file is downloaded to
your machine only and is never published here. Dated snapshots stay public for a limited
time only; when this one is gone, pin a current one in src/sanctions_mcp/data.py.
"""


class ChecksumError(RuntimeError):
    """The downloaded file does not match the pinned checksum."""


class NotPinnedSnapshotError(RuntimeError):
    """A file already sits where the snapshot goes, but it is not the pinned snapshot."""


@dataclass(frozen=True, slots=True)
class Entity:
    id: str
    schema: str
    name: str
    aliases: tuple[str, ...]
    birth_dates: tuple[str, ...]
    countries: tuple[str, ...]
    datasets: tuple[str, ...]
    sanctions: tuple[str, ...] = ()  # free text from the list: data, not instructions

    @property
    def names(self) -> tuple[str, ...]:
        """The primary name first, then every alias, without exact duplicates."""
        seen: dict[str, None] = {self.name: None}
        for alias in self.aliases:
            seen.setdefault(alias, None)
        return tuple(seen)


def default_data_dir() -> Path:
    """$SANCTIONS_DATA_DIR, or ./data when it is unset or empty. An empty value (the line
    `SANCTIONS_DATA_DIR=` copied from .env.example) would otherwise mean the current folder,
    and the list, eval sets and models would land in the repo root."""
    value = os.environ.get("SANCTIONS_DATA_DIR", "")
    return Path(value) if value.strip() else Path("data")


def snapshot_path(data_dir: Path | None = None) -> Path:
    return (data_dir or default_data_dir()) / "opensanctions" / SNAPSHOT / FILENAME


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def download(
    data_dir: Path | None = None,
    *,
    force: bool = False,
    client: httpx.Client | None = None,
    url: str = SNAPSHOT_URL,
    sha256: str | None = None,
    on_progress: Callable[[int], None] | None = None,
) -> Path:
    """Fetch the pinned snapshot, verify it and return its local path.

    A file already at the snapshot path is kept when it matches the pinned SHA-256. When
    it does not match, nothing is downloaded and NotPinnedSnapshotError is raised, unless
    `force` is set: such a file was not written by this function (it moves only verified
    downloads into place), so it may be a test fixture or a file placed by hand.
    """
    expected = sha256 or SHA256
    dest = snapshot_path(data_dir)
    if dest.exists() and not force:
        if sha256_of(dest) == expected:
            return dest
        raise NotPinnedSnapshotError(
            f"{dest.as_posix()} exists but is not the pinned snapshot {SNAPSHOT}, so it was "
            "left as it is and nothing was downloaded. Run `sanctions download --force` to "
            "replace it."
        )
    dest.parent.mkdir(parents=True, exist_ok=True)
    partial = dest.with_name(dest.name + ".part")
    own_client = client is None
    http = client or httpx.Client(timeout=120.0, follow_redirects=True)
    try:
        with http.stream("GET", url) as response:
            response.raise_for_status()
            done = 0
            with partial.open("wb") as out:
                for chunk in response.iter_bytes(1 << 20):
                    out.write(chunk)
                    done += len(chunk)
                    if on_progress:
                        on_progress(done)
    except BaseException:
        partial.unlink(missing_ok=True)  # a broken download leaves nothing behind
        raise
    finally:
        if own_client:
            http.close()
    actual = sha256_of(partial)
    if actual != expected:
        partial.unlink(missing_ok=True)
        raise ChecksumError(f"{FILENAME}: expected sha256 {expected}, got {actual}")
    partial.replace(dest)
    return dest


def describe_list(path: Path, people: int) -> dict:
    """What results files, `sanctions stats` and `list_sources` say about the list file
    that was loaded. It is named as the OpenSanctions snapshot only when its SHA-256 is the
    pinned one; any other file (the fictional test list, a file placed by hand) is not."""
    sha = sha256_of(path)
    pinned = sha == SHA256
    return {
        "source": "OpenSanctions" if pinned else "unverified list file",
        "collection": COLLECTION if pinned else None,
        "snapshot": SNAPSHOT if pinned else None,
        "license": LICENSE if pinned else None,
        "sha256": sha,
        "people": people,
    }


def _split(value: str) -> tuple[str, ...]:
    return tuple(v.strip() for v in value.split(";") if v.strip())


def load_watchlist(path: Path, *, schemas: tuple[str, ...] = ("Person",)) -> list[Entity]:
    """Read targets.simple.csv. By default only people, the focus of this repo."""
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run `sanctions download` first.")
    csv.field_size_limit(1 << 30)
    entities = []
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if schemas and row["schema"] not in schemas:
                continue
            if not row["name"].strip():
                continue
            entities.append(
                Entity(
                    id=row["id"],
                    schema=row["schema"],
                    name=row["name"].strip(),
                    aliases=_split(row["aliases"]),
                    birth_dates=_split(row["birth_date"]),
                    countries=_split(row["countries"]),
                    datasets=_split(row["dataset"]),
                    sanctions=_split(row.get("sanctions", "")),
                )
            )
    return entities
