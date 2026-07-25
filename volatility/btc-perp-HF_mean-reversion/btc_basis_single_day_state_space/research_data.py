"""Checksummed archive acquisition and reproducibility manifests."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import tempfile
from typing import Any, Iterable
from urllib.request import urlopen


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



__all__ = [
    "ArchiveFile",
    "fetch_archive",
    "installed_versions",
    "sha256_file",
    "write_research_manifest",
]
