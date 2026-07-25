from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path

import pytest

import research_data
from research_data import ArchiveFile, fetch_archive, sha256_file, write_research_manifest


def test_sha256_file_matches_hashlib(tmp_path: Path):
    payload = b"basis-research\n" * 100
    path = tmp_path / "archive.zip"
    path.write_bytes(payload)
    assert sha256_file(path) == hashlib.sha256(payload).hexdigest()


def test_fetch_archive_is_atomic_and_reuses_cache(tmp_path: Path, monkeypatch):
    payload = b"fake zip bytes"

    def fake_urlopen(url, timeout):
        assert url == "https://example.test/archive.zip"
        assert timeout == 30.0
        return io.BytesIO(payload)

    monkeypatch.setattr(research_data, "urlopen", fake_urlopen)
    target = tmp_path / "raw" / "archive.zip"
    first = fetch_archive(
        "https://example.test/archive.zip",
        target,
        timeout=30.0,
    )
    second = fetch_archive(
        "https://example.test/archive.zip",
        target,
        timeout=30.0,
    )

    assert target.read_bytes() == payload
    assert not first.reused_cache
    assert second.reused_cache
    assert first.sha256 == second.sha256


def test_write_manifest_serialises_archive_and_configuration(tmp_path: Path):
    archive = ArchiveFile(
        url="https://example.test/a.zip",
        path="data/a.zip",
        sha256="a" * 64,
        size_bytes=12,
        reused_cache=True,
    )
    target = write_research_manifest(
        tmp_path / "manifest.json",
        study={"scope": "one-day"},
        archives=[archive],
        configuration={"entry_z": 2.0},
        outputs={"trades": 3},
        package_versions={"numpy": "2.0.0"},
    )
    payload = json.loads(target.read_text())

    assert payload["manifest_schema"] == 1
    assert payload["study"]["scope"] == "one-day"
    assert payload["archives"][0]["sha256"] == "a" * 64
    assert payload["configuration"]["entry_z"] == pytest.approx(2.0)


def test_failed_download_leaves_no_partial_file(tmp_path: Path, monkeypatch):
    def failing_urlopen(url, timeout):
        raise OSError("network failure")

    monkeypatch.setattr(research_data, "urlopen", failing_urlopen)
    target = tmp_path / "raw" / "archive.zip"
    with pytest.raises(OSError, match="network failure"):
        fetch_archive("https://example.test/archive.zip", target)

    assert not target.exists()
    assert not list(target.parent.glob("*.tmp"))
