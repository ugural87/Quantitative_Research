"""Checksummed archive acquisition and reproducibility manifests."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import tempfile
from typing import Any, Iterable
from urllib.request import urlopen

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ArchiveFile:
    url: str
    path: str
    sha256: str
    size_bytes: int
    reused_cache: bool

    def to_dict(self) -> dict[str, str | int | bool]:
        return asdict(self)


def sha256_file(path: str | Path, *, chunk_size: int = 1 << 20) -> str:
    """Return the SHA-256 digest of a file without loading it all into memory."""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def fetch_archive(
    url: str,
    destination: str | Path,
    *,
    timeout: float = 120.0,
) -> ArchiveFile:
    """Download to an atomic temporary file, or checksum an existing archive."""

    if timeout <= 0:
        raise ValueError("timeout must be positive")
    target = Path(destination)
    target.parent.mkdir(parents=True, exist_ok=True)

    if target.exists():
        return ArchiveFile(
            url=url,
            path=str(target),
            sha256=sha256_file(target),
            size_bytes=target.stat().st_size,
            reused_cache=True,
        )

    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=target.parent,
            prefix=f".{target.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temp_path = Path(temporary.name)
            with urlopen(url, timeout=timeout) as response:
                while chunk := response.read(1 << 20):
                    temporary.write(chunk)
            temporary.flush()
        temp_path.replace(target)
    except Exception:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
        raise

    return ArchiveFile(
        url=url,
        path=str(target),
        sha256=sha256_file(target),
        size_bytes=target.stat().st_size,
        reused_cache=False,
    )


def installed_versions(packages: Iterable[str]) -> dict[str, str]:
    """Resolve exact installed distribution versions for a run manifest."""

    versions: dict[str, str] = {}
    for package in packages:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not-installed"
    return versions


def write_research_manifest(
    path: str | Path,
    *,
    study: dict[str, Any],
    archives: Iterable[dict[str, Any] | ArchiveFile],
    configuration: dict[str, Any],
    outputs: dict[str, Any] | None = None,
    package_versions: dict[str, str] | None = None,
) -> Path:
    """Write the data and run contract as stable, human-readable JSON."""

    archive_rows = [
        item.to_dict() if isinstance(item, ArchiveFile) else dict(item)
        for item in archives
    ]
    payload = {
        "manifest_schema": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "study": study,
        "archives": archive_rows,
        "configuration": configuration,
        "package_versions": package_versions or {},
        "outputs": outputs or {},
    }

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=target.parent,
            prefix=f".{target.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temp_path = Path(temporary.name)
            json.dump(payload, temporary, indent=2, sort_keys=True, default=str)
            temporary.write("\n")
            temporary.flush()
        temp_path.replace(target)
    except Exception:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
        raise
    return target



def infer_timestamp_unit(values: pd.Series) -> str:
    """Infer Binance millisecond versus microsecond epoch timestamps."""

    finite = pd.to_numeric(values, errors="coerce").dropna()
    if finite.empty:
        raise ValueError("no valid timestamps")
    return "us" if float(finite.median()) > 1e14 else "ms"


def to_availability_bars(
    trades: pd.DataFrame,
    date: str,
    bar_seconds: int,
) -> pd.DataFrame:
    """Aggregate trades into bars labelled when their contents are available.

    A bar labelled ``t`` contains trades in ``[t-bar, t)``.  The label is the
    decision-time boundary, not the beginning of the interval.  Event age is
    measured from the actual timestamp of the last trade, preserving sub-second
    staleness instead of setting every active bar's age to zero.
    """

    if not isinstance(bar_seconds, int) or bar_seconds < 1:
        raise ValueError("bar_seconds must be a positive integer")
    required = {"timestamp", "price", "quantity"}
    missing = required - set(trades.columns)
    if missing:
        raise KeyError(f"missing required trade columns: {sorted(missing)}")

    day_start = pd.Timestamp(date, tz="UTC")
    day_end = day_start + pd.Timedelta(days=1)
    step = pd.Timedelta(seconds=bar_seconds)
    if pd.Timedelta(days=1) % step != pd.Timedelta(0):
        raise ValueError("bar_seconds must divide one UTC day exactly")

    frame = trades.loc[:, ["timestamp", "price", "quantity"]].copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    frame["price"] = pd.to_numeric(frame["price"], errors="coerce")
    frame["quantity"] = pd.to_numeric(frame["quantity"], errors="coerce")
    frame = frame.loc[
        frame["timestamp"].between(day_start, day_end, inclusive="left")
        & np.isfinite(frame["price"])
        & np.isfinite(frame["quantity"])
        & (frame["price"] > 0.0)
        & (frame["quantity"] > 0.0)
    ].sort_values("timestamp")

    frame["event_time"] = frame["timestamp"]
    indexed = frame.set_index("timestamp")
    rule = f"{bar_seconds}s"
    bars = indexed.resample(
        rule,
        origin=day_start,
        closed="left",
        label="right",
    ).agg(
        last=("price", "last"),
        volume=("quantity", "sum"),
        trades=("price", "size"),
        last_event_time=("event_time", "last"),
    )

    full_index = pd.date_range(day_start + step, day_end, freq=rule)
    bars = bars.reindex(full_index)
    bars.index.name = "timestamp"

    bars["had_trade"] = bars["last"].notna()
    bars["last"] = bars["last"].ffill()
    bars["last_event_time"] = bars["last_event_time"].ffill()
    bars["age_s"] = (
        bars.index.to_series(index=bars.index) - bars["last_event_time"]
    ).dt.total_seconds()
    bars["volume"] = bars["volume"].fillna(0.0)
    bars["trades"] = bars["trades"].fillna(0).astype(int)
    return bars


def git_revision(path: str | Path = ".") -> dict[str, str | bool | None]:
    """Return commit and dirty-state metadata without requiring Git."""

    cwd = Path(path)
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=cwd,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
        return {"git_commit": commit, "git_dirty": dirty}
    except (FileNotFoundError, subprocess.CalledProcessError):
        return {"git_commit": None, "git_dirty": None}


__all__ = [
    "ArchiveFile",
    "fetch_archive",
    "git_revision",
    "infer_timestamp_unit",
    "installed_versions",
    "sha256_file",
    "to_availability_bars",
    "write_research_manifest",
]
