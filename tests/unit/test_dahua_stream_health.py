"""Node 5.10.1 validation tests."""

from dataclasses import asdict

import pytest

from company_vision.behaviors.dahua_stream_health import (
    StreamHealthError,
    StreamHealthReport,
    validate_dimensions,
    validate_recording,
)


def test_valid_recording(tmp_path):
    path = tmp_path / "sample.dav"
    path.write_bytes(b"x" * 50000)
    assert validate_recording(path) == 50000


def test_missing_recording(tmp_path):
    with pytest.raises(StreamHealthError):
        validate_recording(tmp_path / "missing.dav")


def test_empty_recording(tmp_path):
    path = tmp_path / "sample.dav"
    path.touch()
    with pytest.raises(StreamHealthError):
        validate_recording(path)


def test_short_recording(tmp_path):
    path = tmp_path / "sample.dav"
    path.write_bytes(b"x" * 64)
    with pytest.raises(StreamHealthError):
        validate_recording(path)


def test_invalid_minimum(tmp_path):
    with pytest.raises(ValueError):
        validate_recording(
            tmp_path / "sample.dav",
            minimum_bytes=0,
        )


def test_valid_dimensions():
    validate_dimensions(1920, 1080)


def test_invalid_dimensions():
    with pytest.raises(StreamHealthError):
        validate_dimensions(32, 32)


def test_report_excludes_credentials():
    report = StreamHealthReport(
        status="pass",
        transport="Dahua NetSDK",
        channel=0,
        stream_type=0,
        duration_seconds=8,
        recorded_bytes=100000,
        decoded_frame=True,
        width=1920,
        height=1080,
    )
    data = asdict(report)

    assert data["decoded_frame"]
    assert "password" not in data
    assert "username" not in data
    assert "host" not in data