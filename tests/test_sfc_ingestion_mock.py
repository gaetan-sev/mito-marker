"""
test_sfc_ingestion_mock.py

Mock-based ingestion tests that run unconditionally in CI — no real FCS binary files
are required.

load_fcs_file is replaced with a deterministic stub that returns a known data matrix,
so these tests run even when the real fixture .fcs files are absent from the repository.

Tests verify:
  - ingest_sfc_folder() returns a valid AnnData.
  - .obs fields are populated from the filename (subject_ID, specie, marker, FlowAI_Pass).
  - .X has the correct shape, dtype, and no NaN or Inf values.
  - Error handling (missing directory, empty directory) raises the expected exceptions.
"""

import os

import anndata
import numpy as np
import pytest

from mito_marker.controlled_vocabulary import SFC_CHANNELS_TO_KEEP
from mito_marker.sfc.ingestion import ingest_sfc_folder


# ---------------------------------------------------------------------------
# Mock stub
# ---------------------------------------------------------------------------

# Use a small, fixed subset of real channel names — all are non-FJComp so none
# will be removed by the effective_channels_to_exclude list inside ingest_sfc_folder.
_MOCK_CHANNELS: list[str] = list(SFC_CHANNELS_TO_KEEP[:10])
_MOCK_N_EVENTS: int = 100


def _mock_load_fcs_file(
    fcs_file_path: str,
    channels_to_exclude: list[str] | None = None,
) -> tuple:
    """
    Deterministic stub for load_fcs_file.

    Returns 100 events × 10 channels of float32 data with empty instrument and
    FCS metadata dicts. The channels are drawn from SFC_CHANNELS_TO_KEEP so that
    none are excluded by the FJComp exclusion logic in ingest_sfc_folder.
    """
    channel_names = _MOCK_CHANNELS
    channel_descriptions = [f"Description of {ch}" for ch in channel_names]
    rng = np.random.default_rng(seed=42)
    event_matrix = rng.random((_MOCK_N_EVENTS, len(channel_names))).astype(np.float32)
    instrument_metadata: dict[str, str] = {}
    fcs_obs_metadata: dict[str, str | None] = {
        "fcs_acquisition_date": None,
        "fcs_tube_name": None,
        "fcs_volume_uL": None,
        "fcs_total_events": None,
    }
    return channel_names, channel_descriptions, event_matrix, instrument_metadata, fcs_obs_metadata


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_fcs_directory(tmp_path, monkeypatch):
    """
    Temporary directory containing one FCS file with a valid parseable filename.
    load_fcs_file is replaced with the deterministic stub so no real FCS binary is read.

    Returns a dict with:
        fcs_dir:     path to the directory containing the mock .fcs file
        output_path: path where ingest_sfc_folder should write its .h5ad output
    """
    fcs_dir = tmp_path / "fcs"
    fcs_dir.mkdir()
    # The file is empty — the mock intercepts before FlowIO reads any bytes.
    fcs_file = fcs_dir / "good_events_MNMS_020_FlowAIGoodEvents_DeepRed.fcs"
    fcs_file.write_bytes(b"")
    output_path = str(tmp_path / "output.h5ad")
    monkeypatch.setattr("mito_marker.sfc.ingestion.load_fcs_file", _mock_load_fcs_file)
    return {"fcs_dir": str(fcs_dir), "output_path": output_path}


@pytest.fixture
def mock_fcs_directory_two_files(tmp_path, monkeypatch):
    """
    Temporary directory with two FCS files from different subjects.
    Used to test partial-duplicate append scenarios.

    Returns a dict with:
        fcs_dir:       path containing two mock .fcs files (subjects 020 and 021)
        output_path:   path for the .h5ad output
        file_subject1: basename of the first file (subject 020)
        file_subject2: basename of the second file (subject 021)
    """
    fcs_dir = tmp_path / "fcs"
    fcs_dir.mkdir()
    file1 = fcs_dir / "good_events_MNMS_020_FlowAIGoodEvents_DeepRed.fcs"
    file2 = fcs_dir / "good_events_MNMS_021_FlowAIGoodEvents_DeepRed.fcs"
    file1.write_bytes(b"")
    file2.write_bytes(b"")
    output_path = str(tmp_path / "output.h5ad")
    monkeypatch.setattr("mito_marker.sfc.ingestion.load_fcs_file", _mock_load_fcs_file)
    return {
        "fcs_dir": str(fcs_dir),
        "output_path": output_path,
        "file_subject1": file1.name,
        "file_subject2": file2.name,
    }


# ---------------------------------------------------------------------------
# Tests: single-file ingestion with mock
# ---------------------------------------------------------------------------


class TestSingleFileIngestionMock:
    """Verifies AnnData construction from a mocked FCS file without real binary data."""

    def test_returns_anndata_object(self, mock_fcs_directory):
        result = ingest_sfc_folder(
            data_directory_path=mock_fcs_directory["fcs_dir"],
            output_file_path=mock_fcs_directory["output_path"],
        )
        assert isinstance(result, anndata.AnnData)

    def test_event_count(self, mock_fcs_directory):
        result = ingest_sfc_folder(
            data_directory_path=mock_fcs_directory["fcs_dir"],
            output_file_path=mock_fcs_directory["output_path"],
        )
        assert result.n_obs == _MOCK_N_EVENTS

    def test_channel_count(self, mock_fcs_directory):
        """Mock returns 10 non-FJComp channels; none should be removed by the exclusion logic."""
        result = ingest_sfc_folder(
            data_directory_path=mock_fcs_directory["fcs_dir"],
            output_file_path=mock_fcs_directory["output_path"],
        )
        assert result.n_vars == len(_MOCK_CHANNELS)

    def test_obs_subject_id(self, mock_fcs_directory):
        result = ingest_sfc_folder(
            data_directory_path=mock_fcs_directory["fcs_dir"],
            output_file_path=mock_fcs_directory["output_path"],
        )
        assert (result.obs["subject_ID"] == "020").all()

    def test_obs_specie(self, mock_fcs_directory):
        result = ingest_sfc_folder(
            data_directory_path=mock_fcs_directory["fcs_dir"],
            output_file_path=mock_fcs_directory["output_path"],
        )
        assert (result.obs["specie"] == "MNMS").all()

    def test_obs_marker(self, mock_fcs_directory):
        result = ingest_sfc_folder(
            data_directory_path=mock_fcs_directory["fcs_dir"],
            output_file_path=mock_fcs_directory["output_path"],
        )
        assert (result.obs["marker"] == "MtDeepRed").all()

    def test_obs_flowai_pass_true(self, mock_fcs_directory):
        result = ingest_sfc_folder(
            data_directory_path=mock_fcs_directory["fcs_dir"],
            output_file_path=mock_fcs_directory["output_path"],
        )
        assert (result.obs["FlowAI_Pass"] == True).all()  # noqa: E712

    def test_x_dtype_float32(self, mock_fcs_directory):
        result = ingest_sfc_folder(
            data_directory_path=mock_fcs_directory["fcs_dir"],
            output_file_path=mock_fcs_directory["output_path"],
        )
        assert result.X.dtype == np.float32

    def test_x_no_nan(self, mock_fcs_directory):
        result = ingest_sfc_folder(
            data_directory_path=mock_fcs_directory["fcs_dir"],
            output_file_path=mock_fcs_directory["output_path"],
        )
        assert not np.isnan(result.X).any()

    def test_x_no_inf(self, mock_fcs_directory):
        result = ingest_sfc_folder(
            data_directory_path=mock_fcs_directory["fcs_dir"],
            output_file_path=mock_fcs_directory["output_path"],
        )
        assert not np.isinf(result.X).any()


# ---------------------------------------------------------------------------
# Tests: error handling (no mock required — pure path validation)
# ---------------------------------------------------------------------------


class TestAppendDeduplicationMock:
    """
    Verifies that append_to_existing=True does not duplicate events when the
    same .fcs files are ingested a second time into an already existing .h5ad.
    """

    def test_second_run_does_not_duplicate_events(self, mock_fcs_directory):
        """
        Running ingest_sfc_folder twice on the same directory with
        append_to_existing=True must keep event count identical to the first run.
        """
        first_result = ingest_sfc_folder(
            data_directory_path=mock_fcs_directory["fcs_dir"],
            output_file_path=mock_fcs_directory["output_path"],
        )
        first_n_obs = first_result.n_obs

        second_result = ingest_sfc_folder(
            data_directory_path=mock_fcs_directory["fcs_dir"],
            output_file_path=mock_fcs_directory["output_path"],
            append_to_existing=True,
        )
        assert second_result.n_obs == first_n_obs, (
            f"Expected {first_n_obs} events after second run but got "
            f"{second_result.n_obs} — duplication occurred."
        )

    def test_new_file_is_appended(self, mock_fcs_directory_two_files):
        """
        If the existing .h5ad contains only subject 020 and we re-run with both
        020 and 021 present in the directory, only subject 021 events are appended.
        """
        fcs_dir = mock_fcs_directory_two_files["fcs_dir"]
        output_path = mock_fcs_directory_two_files["output_path"]
        file2 = mock_fcs_directory_two_files["file_subject2"]

        # First run: ingest only subject 020 by temporarily hiding file2
        file2_path = os.path.join(fcs_dir, file2)
        hidden_path = file2_path + ".hidden"
        os.rename(file2_path, hidden_path)

        first_result = ingest_sfc_folder(
            data_directory_path=fcs_dir,
            output_file_path=output_path,
        )
        first_n_obs = first_result.n_obs  # only subject 020 events

        # Restore file2 so both subjects are now in the directory
        os.rename(hidden_path, file2_path)

        second_result = ingest_sfc_folder(
            data_directory_path=fcs_dir,
            output_file_path=output_path,
            append_to_existing=True,
        )
        # Subject 021 events must have been added exactly once
        assert second_result.n_obs == first_n_obs + _MOCK_N_EVENTS, (
            f"Expected {first_n_obs + _MOCK_N_EVENTS} events after appending new "
            f"subject but got {second_result.n_obs}."
        )

    def test_partial_duplicate_only_new_subject_appended(self, mock_fcs_directory_two_files):
        """
        After ingesting both subjects, a third run must skip both files
        and leave the event count unchanged.
        """
        fcs_dir = mock_fcs_directory_two_files["fcs_dir"]
        output_path = mock_fcs_directory_two_files["output_path"]

        first_result = ingest_sfc_folder(
            data_directory_path=fcs_dir,
            output_file_path=output_path,
        )
        first_n_obs = first_result.n_obs  # both subjects ingested

        third_result = ingest_sfc_folder(
            data_directory_path=fcs_dir,
            output_file_path=output_path,
            append_to_existing=True,
        )
        assert third_result.n_obs == first_n_obs, (
            f"Expected event count to remain {first_n_obs} when all files are "
            f"already ingested, but got {third_result.n_obs}."
        )


class TestErrorHandlingMock:
    """
    Error-handling tests that do not require real FCS files.
    These verify that ingest_sfc_folder raises the documented exceptions for bad inputs.
    """

    def test_missing_directory_raises_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            ingest_sfc_folder(
                data_directory_path="/nonexistent/path/that/does/not/exist",
                output_file_path=str(tmp_path / "output.h5ad"),
            )

    def test_empty_directory_raises_value_error(self, tmp_path):
        """An existing directory with no .fcs files must raise ValueError."""
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        with pytest.raises(ValueError):
            ingest_sfc_folder(
                data_directory_path=str(empty_dir),
                output_file_path=str(tmp_path / "output.h5ad"),
            )
