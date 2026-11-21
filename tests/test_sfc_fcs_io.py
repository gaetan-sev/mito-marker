"""
test_sfc_fcs_io.py

Tests for read_fcs_events() and write_fcs_events(): a file written after a
preprocessing step must hold the same channels, values and keywords as its source,
minus the removed events, so that FlowJo reopens it like one of its own exports.
"""

import os

import numpy as np
import pytest

from mito_marker.sfc.fcs_io import fcs_file_stem, list_fcs_files, read_fcs_events, write_fcs_events


def build_source(write_synthetic_fcs, n_events: int = 50) -> str:
    rng = np.random.default_rng(0)
    return write_synthetic_fcs(
        "WORM_001_D02_AL_DR.fcs",
        {"FSC-A": rng.uniform(10, 1e5, n_events), "SSC-A": rng.uniform(10, 1e5, n_events),
         "R2-A": rng.normal(0, 100, n_events), "Time": np.arange(n_events, dtype=float)},
        keywords={"threshold": "(FSC,1000)And(SSC,1000)"},
        ranges={"Time": 350517},
    )


def test_read_keeps_every_channel_in_file_order_and_keywords(write_synthetic_fcs):
    events = read_fcs_events(build_source(write_synthetic_fcs))
    assert events.channel_names == ["FSC-A", "SSC-A", "R2-A", "Time"]
    assert events.event_matrix.shape == (50, 4)
    assert events.text_keywords["date"] == "01-Jan-2026"
    assert events.channel_ranges().tolist() == [4194304, 4194304, 4194304, 350517]


def test_write_round_trip_with_mask_and_extra_channel(write_synthetic_fcs, tmp_path):
    source = read_fcs_events(build_source(write_synthetic_fcs))
    mask = np.arange(source.n_events) % 2 == 0
    labels = np.where(mask, 2000.0, 6000.0)
    output_path = write_fcs_events(source, str(tmp_path / "out.fcs"), event_mask=mask, extra_channels={"FlowAI": labels})
    written = read_fcs_events(output_path)
    assert written.channel_names == ["FSC-A", "SSC-A", "R2-A", "Time", "FlowAI"]
    assert written.n_events == int(mask.sum())
    np.testing.assert_array_equal(written.event_matrix[:, :4].astype(np.float32), source.event_matrix[mask].astype(np.float32))
    assert set(np.unique(written.channel_values("FlowAI"))) == {2000.0}
    # Keywords FlowJo needs to show the same axes and the acquisition details are kept.
    for keyword in ("date", "btim", "cyt", "timestep", "threshold", "p1r", "p4r"):
        assert written.text_keywords[keyword].strip() == source.text_keywords[keyword].strip()
    assert written.text_keywords["fil"] == "out.fcs"
    assert written.text_keywords["p5r"] == "20000"


def test_write_rejects_wrong_mask_shape_and_duplicate_channel(write_synthetic_fcs, tmp_path):
    source = read_fcs_events(build_source(write_synthetic_fcs))
    with pytest.raises(ValueError, match="event_mask"):
        write_fcs_events(source, str(tmp_path / "a.fcs"), event_mask=np.ones(3, dtype=bool))
    with pytest.raises(ValueError, match="already exists"):
        write_fcs_events(source, str(tmp_path / "b.fcs"), extra_channels={"R2-A": np.zeros(source.n_events)})


def test_read_reports_missing_empty_and_unknown_channel(write_synthetic_fcs, tmp_path):
    with pytest.raises(FileNotFoundError):
        read_fcs_events(str(tmp_path / "missing.fcs"))
    empty = tmp_path / "empty.fcs"
    empty.write_bytes(b"")
    with pytest.raises(ValueError, match="empty"):
        read_fcs_events(str(empty))
    events = read_fcs_events(build_source(write_synthetic_fcs))
    with pytest.raises(KeyError, match="R7-A"):
        events.channel_values("R7-A")


def test_list_fcs_files_and_stem(write_synthetic_fcs, tmp_path):
    build_source(write_synthetic_fcs)
    (tmp_path / "notes.txt").write_text("not an fcs")
    files = list_fcs_files(str(tmp_path))
    assert [os.path.basename(path) for path in files] == ["WORM_001_D02_AL_DR.fcs"]
    assert fcs_file_stem(files[0]) == "WORM_001_D02_AL_DR"
    with pytest.raises(FileNotFoundError):
        list_fcs_files(str(tmp_path / "nowhere"))
