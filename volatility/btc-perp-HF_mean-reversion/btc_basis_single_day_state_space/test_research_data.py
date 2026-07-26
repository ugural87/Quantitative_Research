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


def test_timestamp_unit_infers_binance_ms_and_us():
    import pandas as pd

    assert research_data.infer_timestamp_unit(pd.Series([1_700_000_000_000])) == "ms"
    assert research_data.infer_timestamp_unit(pd.Series([1_700_000_000_000_000])) == "us"


def test_availability_bars_use_right_edge_and_actual_event_age():
    import pandas as pd

    trades = pd.DataFrame({
        "timestamp": pd.to_datetime([
            "2026-01-01T00:00:00.800Z",
            "2026-01-01T00:00:02.200Z",
        ]),
        "price": [100.0, 101.0],
        "quantity": [1.0, 2.0],
    })
    bars = research_data.to_availability_bars(trades, "2026-01-01", 1)

    assert bars.index[0] == pd.Timestamp("2026-01-01T00:00:01Z")
    assert bars.index[-1] == pd.Timestamp("2026-01-02T00:00:00Z")
    assert len(bars) == 86_400
    assert bars.iloc[0]["last"] == pytest.approx(100.0)
    assert bars.iloc[0]["age_s"] == pytest.approx(0.2)
    assert bars.iloc[1]["last"] == pytest.approx(100.0)
    assert bars.iloc[1]["age_s"] == pytest.approx(1.2)
    assert bars.iloc[2]["last"] == pytest.approx(101.0)
    assert bars.iloc[2]["age_s"] == pytest.approx(0.8)


def test_git_revision_returns_metadata_or_none(tmp_path: Path):
    metadata = research_data.git_revision(tmp_path)
    assert set(metadata) == {"git_commit", "git_dirty"}
    assert metadata["git_commit"] is None
    assert metadata["git_dirty"] is None
