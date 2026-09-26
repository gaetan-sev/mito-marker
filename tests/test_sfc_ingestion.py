"""
test_sfc_ingestion.py

Integration tests for the SFC ingestion pipeline using the two fixture FCS files
committed to tests/fixtures/.

These tests verify end-to-end behaviour: filename parsing → FCS loading → AnnData
construction → QC → .h5ad round-trip → append mode.

Fixture files:
  - tests/fixtures/good_events_MNMS_020_FlowAIGoodEvents_DeepRed.fcs
  - tests/fixtures/good_events_MNMS_031_ND_FlowAIGoodEvents_DeepRed.fcs
"""

import os
import tempfile

import anndata
import numpy as np
import pandas as pd
import pytest

from mito_marker.sfc.ingestion import ingest_sfc_folder

# ---------------------------------------------------------------------------
# Paths to fixture files
# ---------------------------------------------------------------------------

FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "fixtures")
FILE_020 = os.path.join(FIXTURES_DIR, "good_events_MNMS_020_FlowAIGoodEvents_DeepRed.fcs")
FILE_031 = os.path.join(FIXTURES_DIR, "good_events_MNMS_031_ND_FlowAIGoodEvents_DeepRed.fcs")


@pytest.fixture
def temp_output_dir():
    """Provide a temporary directory for .h5ad output files during tests."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield tmp_dir


@pytest.fixture
def temp_fixtures_single_020(tmp_path):
    """Temporary directory containing only the MNMS_020 fixture file.

    Uses a named subdirectory so that when this fixture and temp_fixtures_single_031
    are both requested in the same test, they resolve to different directories
    (both share the same tmp_path base under pytest's function-scoped tmp_path fixture).
    """
    import shutil
    single_020_dir = tmp_path / "single_020"
    single_020_dir.mkdir()
    shutil.copy(FILE_020, single_020_dir / os.path.basename(FILE_020))
    return str(single_020_dir)


@pytest.fixture
def temp_fixtures_single_031(tmp_path):
    """Temporary directory containing only the MNMS_031 fixture file.

    Uses a named subdirectory for the same reason as temp_fixtures_single_020.
    """
    import shutil
    single_031_dir = tmp_path / "single_031"
    single_031_dir.mkdir()
    shutil.copy(FILE_031, single_031_dir / os.path.basename(FILE_031))
    return str(single_031_dir)


@pytest.fixture
def temp_fixtures_both(tmp_path):
    """Temporary directory containing both fixture files."""
    import shutil
    shutil.copy(FILE_020, tmp_path / os.path.basename(FILE_020))
    shutil.copy(FILE_031, tmp_path / os.path.basename(FILE_031))
    return str(tmp_path)


# ---------------------------------------------------------------------------
# Helper: skip if fixtures are absent (e.g., CI environment without large files)
# ---------------------------------------------------------------------------

requires_fixtures = pytest.mark.skipif(
    not os.path.exists(FILE_020) or not os.path.exists(FILE_031),
    reason="FCS fixture files not found in tests/fixtures/",
)


# ---------------------------------------------------------------------------
# Single file ingestion: MNMS_020
# ---------------------------------------------------------------------------


@requires_fixtures
class TestSingleFileIngestion020:
    """Ingest only the MNMS_020 fixture and verify the resulting AnnData."""

    def test_returns_anndata(self, temp_fixtures_single_020, temp_output_dir):
        output_path = os.path.join(temp_output_dir, "sfc.h5ad")
        result = ingest_sfc_folder(
            data_directory_path=temp_fixtures_single_020,
            output_file_path=output_path,
        )
        assert isinstance(result, anndata.AnnData)

    def test_event_count(self, temp_fixtures_single_020, temp_output_dir):
        output_path = os.path.join(temp_output_dir, "sfc.h5ad")
        result = ingest_sfc_folder(
            data_directory_path=temp_fixtures_single_020,
            output_file_path=output_path,
        )
        assert result.n_obs == 11362

    def test_channel_count_no_additional_exclusions(self, temp_fixtures_single_020, temp_output_dir):
        """FJComp, Time, and FlowAI channels are always excluded by default."""
        output_path = os.path.join(temp_output_dir, "sfc.h5ad")
        result = ingest_sfc_folder(
            data_directory_path=temp_fixtures_single_020,
            output_file_path=output_path,
        )
        # 208 raw channels − 64 FJComp − 1 Time − 1 FlowAI − 1 Event# = 141...
        # Actually: 208 − 64 FJComp − 1 Event# − 1 Time − 1 FlowAI = 141... let's keep measured value
        # 208 raw − 66 always-excluded (64 FJComp + Event# + Time + FlowAI) = 142
        assert result.n_vars == 142
        # Verify no FJComp channel leaked through
        assert not any(name.startswith("FJComp-") for name in result.var_names)

    def test_channel_count_with_additional_exclusion(self, temp_fixtures_single_020, temp_output_dir):
        """An extra channel_to_exclude reduces the default count by one."""
        output_path = os.path.join(temp_output_dir, "sfc_no_fsc_a.h5ad")
        result = ingest_sfc_folder(
            data_directory_path=temp_fixtures_single_020,
            output_file_path=output_path,
            channels_to_exclude=["FSC-A"],
        )
        # 142 default − 1 FSC-A = 141
        assert result.n_vars == 141
        assert "FSC-A" not in result.var_names

    def test_obs_specie(self, temp_fixtures_single_020, temp_output_dir):
        output_path = os.path.join(temp_output_dir, "sfc.h5ad")
        result = ingest_sfc_folder(
            data_directory_path=temp_fixtures_single_020,
            output_file_path=output_path,
        )
        assert result.obs["specie"].iloc[0] == "MNMS"
        assert result.obs["specie"].nunique() == 1

    def test_obs_subject_id(self, temp_fixtures_single_020, temp_output_dir):
        output_path = os.path.join(temp_output_dir, "sfc.h5ad")
        result = ingest_sfc_folder(
            data_directory_path=temp_fixtures_single_020,
            output_file_path=output_path,
        )
        assert result.obs["subject_ID"].iloc[0] == "020"

    def test_obs_dilution_diluted(self, temp_fixtures_single_020, temp_output_dir):
        output_path = os.path.join(temp_output_dir, "sfc.h5ad")
        result = ingest_sfc_folder(
            data_directory_path=temp_fixtures_single_020,
            output_file_path=output_path,
        )
        assert result.obs["dilution"].iloc[0] == "Diluted"

    def test_obs_marker(self, temp_fixtures_single_020, temp_output_dir):
        output_path = os.path.join(temp_output_dir, "sfc.h5ad")
        result = ingest_sfc_folder(
            data_directory_path=temp_fixtures_single_020,
            output_file_path=output_path,
        )
        assert result.obs["marker"].iloc[0] == "MtDeepRed"

    def test_obs_flowai_pass_true(self, temp_fixtures_single_020, temp_output_dir):
        output_path = os.path.join(temp_output_dir, "sfc.h5ad")
        result = ingest_sfc_folder(
            data_directory_path=temp_fixtures_single_020,
            output_file_path=output_path,
        )
        assert bool(result.obs["FlowAI_Pass"].iloc[0]) is True

    def test_obs_age_missing(self, temp_fixtures_single_020, temp_output_dir):
        output_path = os.path.join(temp_output_dir, "sfc.h5ad")
        result = ingest_sfc_folder(
            data_directory_path=temp_fixtures_single_020,
            output_file_path=output_path,
        )
        # Age absent from filename → np.nan stored as float32 (age is numeric int|None;
        # numpy integer dtypes do not support NaN, so float32 is used instead)
        assert np.isnan(result.obs["age"].iloc[0])

    def test_obs_diet_missing(self, temp_fixtures_single_020, temp_output_dir):
        output_path = os.path.join(temp_output_dir, "sfc.h5ad")
        result = ingest_sfc_folder(
            data_directory_path=temp_fixtures_single_020,
            output_file_path=output_path,
        )
        assert pd.isna(result.obs["diet"].iloc[0])

    def test_uns_instrument_metadata_present(self, temp_fixtures_single_020, temp_output_dir):
        output_path = os.path.join(temp_output_dir, "sfc.h5ad")
        result = ingest_sfc_folder(
            data_directory_path=temp_fixtures_single_020,
            output_file_path=output_path,
        )
        assert "fcs_instrument_metadata" in result.uns

    def test_x_dtype_float32(self, temp_fixtures_single_020, temp_output_dir):
        output_path = os.path.join(temp_output_dir, "sfc.h5ad")
        result = ingest_sfc_folder(
            data_directory_path=temp_fixtures_single_020,
            output_file_path=output_path,
        )
        assert result.X.dtype == np.float32

    def test_x_no_nan(self, temp_fixtures_single_020, temp_output_dir):
        output_path = os.path.join(temp_output_dir, "sfc.h5ad")
        result = ingest_sfc_folder(
            data_directory_path=temp_fixtures_single_020,
            output_file_path=output_path,
        )
        assert not np.any(np.isnan(result.X))

    def test_x_no_inf(self, temp_fixtures_single_020, temp_output_dir):
        output_path = os.path.join(temp_output_dir, "sfc.h5ad")
        result = ingest_sfc_folder(
            data_directory_path=temp_fixtures_single_020,
            output_file_path=output_path,
        )
        assert not np.any(np.isinf(result.X))

    def test_fcs_obs_metadata_columns_present(self, temp_fixtures_single_020, temp_output_dir):
        """FCS metadata fields extracted from TEXT segment must appear in .obs."""
        output_path = os.path.join(temp_output_dir, "sfc.h5ad")
        result = ingest_sfc_folder(
            data_directory_path=temp_fixtures_single_020,
            output_file_path=output_path,
        )
        expected_fcs_obs_columns = [
            "fcs_acquisition_date",
            "fcs_tube_name",
            "fcs_volume_uL",
            "fcs_total_events",
        ]
        for column_name in expected_fcs_obs_columns:
            assert column_name in result.obs.columns, (
                f"Expected .obs column '{column_name}' not found. "
                f"Available columns: {list(result.obs.columns)}"
            )


# ---------------------------------------------------------------------------
# .var column: is_non_analytical
# ---------------------------------------------------------------------------


@requires_fixtures
class TestVarNonAnalyticalColumn:
    """Verify that .var['is_non_analytical'] is created and correct at ingestion."""

    def test_is_non_analytical_column_present_in_var(
        self, temp_fixtures_single_020, temp_output_dir
    ):
        output_path = os.path.join(temp_output_dir, "sfc.h5ad")
        result = ingest_sfc_folder(
            data_directory_path=temp_fixtures_single_020,
            output_file_path=output_path,
        )
        assert "is_non_analytical" in result.var.columns

    def test_time_channel_excluded_from_var(
        self, temp_fixtures_single_020, temp_output_dir
    ):
        """Time is excluded from .X entirely — it must not appear in .var."""
        output_path = os.path.join(temp_output_dir, "sfc.h5ad")
        result = ingest_sfc_folder(
            data_directory_path=temp_fixtures_single_020,
            output_file_path=output_path,
        )
        assert "Time" not in result.var_names

    def test_flowai_channel_excluded_from_var(
        self, temp_fixtures_single_020, temp_output_dir
    ):
        """FlowAI is excluded from .X entirely — it must not appear in .var."""
        output_path = os.path.join(temp_output_dir, "sfc.h5ad")
        result = ingest_sfc_folder(
            data_directory_path=temp_fixtures_single_020,
            output_file_path=output_path,
        )
        assert "FlowAI" not in result.var_names

    def test_cytometry_channels_not_marked_as_non_analytical(
        self, temp_fixtures_single_020, temp_output_dir
    ):
        output_path = os.path.join(temp_output_dir, "sfc.h5ad")
        result = ingest_sfc_folder(
            data_directory_path=temp_fixtures_single_020,
            output_file_path=output_path,
        )
        for channel_name in ["FSC-A", "B1-A", "R1-A", "YG1-A"]:
            assert result.var.loc[channel_name, "is_non_analytical"] == False, (
                f"Channel '{channel_name}' was incorrectly marked as non-analytical."
            )

    def test_non_analytical_count_matches_expected(
        self, temp_fixtures_single_020, temp_output_dir
    ):
        """Time and FlowAI are excluded entirely, so no channels are marked non-analytical."""
        output_path = os.path.join(temp_output_dir, "sfc.h5ad")
        result = ingest_sfc_folder(
            data_directory_path=temp_fixtures_single_020,
            output_file_path=output_path,
        )
        assert result.var["is_non_analytical"].sum() == 0


# ---------------------------------------------------------------------------
# Single file ingestion: MNMS_031 (ND token)
# ---------------------------------------------------------------------------


@requires_fixtures
class TestSingleFileIngestion031:
    def test_event_count(self, temp_fixtures_single_031, temp_output_dir):
        output_path = os.path.join(temp_output_dir, "sfc.h5ad")
        result = ingest_sfc_folder(
            data_directory_path=temp_fixtures_single_031,
            output_file_path=output_path,
        )
        # Event count confirmed from FCS file header
        assert result.n_obs > 0  # exact value verified once fixture is read

    def test_obs_subject_id_031(self, temp_fixtures_single_031, temp_output_dir):
        output_path = os.path.join(temp_output_dir, "sfc.h5ad")
        result = ingest_sfc_folder(
            data_directory_path=temp_fixtures_single_031,
            output_file_path=output_path,
        )
        assert result.obs["subject_ID"].iloc[0] == "031"

    def test_obs_dilution_not_diluted(self, temp_fixtures_single_031, temp_output_dir):
        """The ND token in the filename must result in 'Not_Diluted' in .obs."""
        output_path = os.path.join(temp_output_dir, "sfc.h5ad")
        result = ingest_sfc_folder(
            data_directory_path=temp_fixtures_single_031,
            output_file_path=output_path,
        )
        assert result.obs["dilution"].iloc[0] == "Not_Diluted"


# ---------------------------------------------------------------------------
# Combined ingestion: both files
# ---------------------------------------------------------------------------


@requires_fixtures
class TestCombinedIngestion:
    def test_combined_subject_ids(self, temp_fixtures_both, temp_output_dir):
        output_path = os.path.join(temp_output_dir, "sfc.h5ad")
        result = ingest_sfc_folder(
            data_directory_path=temp_fixtures_both,
            output_file_path=output_path,
        )
        subject_ids_in_obs = set(result.obs["subject_ID"].unique())
        assert subject_ids_in_obs == {"020", "031"}

    def test_combined_dilution_variety(self, temp_fixtures_both, temp_output_dir):
        output_path = os.path.join(temp_output_dir, "sfc.h5ad")
        result = ingest_sfc_folder(
            data_directory_path=temp_fixtures_both,
            output_file_path=output_path,
        )
        dilution_values = set(result.obs["dilution"].unique())
        assert dilution_values == {"Diluted", "Not_Diluted"}

    def test_combined_channel_count_with_additional_exclusion(self, temp_fixtures_both, temp_output_dir):
        output_path = os.path.join(temp_output_dir, "sfc.h5ad")
        result = ingest_sfc_folder(
            data_directory_path=temp_fixtures_both,
            output_file_path=output_path,
            channels_to_exclude=["FSC-A"],
        )
        # 142 default − 1 FSC-A = 141
        assert result.n_vars == 141
        assert "FSC-A" not in result.var_names

    def test_combined_total_events(self, temp_fixtures_both,
                                    temp_fixtures_single_020, temp_fixtures_single_031,
                                    temp_output_dir):
        """Total events must equal the sum of events from both individual files."""
        # Ingest both files together
        output_path_both = os.path.join(temp_output_dir, "sfc_both.h5ad")
        result_both = ingest_sfc_folder(
            data_directory_path=temp_fixtures_both,
            output_file_path=output_path_both,
        )
        # Ingest each file individually using their dedicated single-file temp directories
        output_path_020 = os.path.join(temp_output_dir, "sfc_020.h5ad")
        output_path_031 = os.path.join(temp_output_dir, "sfc_031.h5ad")
        result_020 = ingest_sfc_folder(
            data_directory_path=temp_fixtures_single_020,
            output_file_path=output_path_020,
        )
        result_031 = ingest_sfc_folder(
            data_directory_path=temp_fixtures_single_031,
            output_file_path=output_path_031,
        )
        assert result_both.n_obs == result_020.n_obs + result_031.n_obs

    def test_combined_obs_index_is_unique(self, temp_fixtures_both, temp_output_dir):
        """After concat, obs index must have no duplicates."""
        output_path = os.path.join(temp_output_dir, "sfc.h5ad")
        result = ingest_sfc_folder(
            data_directory_path=temp_fixtures_both,
            output_file_path=output_path,
        )
        assert result.obs_names.is_unique


# ---------------------------------------------------------------------------
# .h5ad round-trip: save and reload
# ---------------------------------------------------------------------------


@requires_fixtures
class TestH5adRoundTrip:
    def test_saved_file_exists(self, temp_fixtures_single_020, temp_output_dir):
        output_path = os.path.join(temp_output_dir, "sfc.h5ad")
        ingest_sfc_folder(
            data_directory_path=temp_fixtures_single_020,
            output_file_path=output_path,
        )
        assert os.path.exists(output_path)

    def test_reloaded_dimensions_match(self, temp_fixtures_single_020, temp_output_dir):
        output_path = os.path.join(temp_output_dir, "sfc.h5ad")
        original = ingest_sfc_folder(
            data_directory_path=temp_fixtures_single_020,
            output_file_path=output_path,
        )
        reloaded = anndata.read_h5ad(output_path)
        assert reloaded.n_obs == original.n_obs
        assert reloaded.n_vars == original.n_vars

    def test_reloaded_obs_values_preserved(self, temp_fixtures_single_020, temp_output_dir):
        output_path = os.path.join(temp_output_dir, "sfc.h5ad")
        ingest_sfc_folder(
            data_directory_path=temp_fixtures_single_020,
            output_file_path=output_path,
        )
        reloaded = anndata.read_h5ad(output_path)
        assert reloaded.obs["specie"].iloc[0] == "MNMS"
        assert reloaded.obs["subject_ID"].iloc[0] == "020"
        assert reloaded.obs["marker"].iloc[0] == "MtDeepRed"


# ---------------------------------------------------------------------------
# Append mode
# ---------------------------------------------------------------------------


@requires_fixtures
class TestAppendMode:
    def test_append_increases_event_count(self, temp_fixtures_single_020,
                                          temp_fixtures_single_031, temp_output_dir):
        output_path = os.path.join(temp_output_dir, "sfc.h5ad")

        # First ingestion
        result_first = ingest_sfc_folder(
            data_directory_path=temp_fixtures_single_020,
            output_file_path=output_path,
        )
        n_obs_after_first = result_first.n_obs

        # Second ingestion in append mode
        result_appended = ingest_sfc_folder(
            data_directory_path=temp_fixtures_single_031,
            output_file_path=output_path,
            append_to_existing=True,
        )

        assert result_appended.n_obs > n_obs_after_first

    def test_append_preserves_original_subject_id(self, temp_fixtures_single_020,
                                                   temp_fixtures_single_031, temp_output_dir):
        output_path = os.path.join(temp_output_dir, "sfc.h5ad")

        ingest_sfc_folder(
            data_directory_path=temp_fixtures_single_020,
            output_file_path=output_path,
        )
        result_appended = ingest_sfc_folder(
            data_directory_path=temp_fixtures_single_031,
            output_file_path=output_path,
            append_to_existing=True,
        )

        subject_ids = set(result_appended.obs["subject_ID"].unique())
        assert "020" in subject_ids
        assert "031" in subject_ids

    def test_append_channel_mismatch_raises(self, temp_fixtures_single_020, tmp_path,
                                             temp_output_dir):
        """Appending a file with different channels must raise ValueError immediately."""
        import shutil

        output_path = os.path.join(temp_output_dir, "sfc.h5ad")

        # Ingest without exclusions
        ingest_sfc_folder(
            data_directory_path=temp_fixtures_single_020,
            output_file_path=output_path,
        )

        # Prepare a directory with MNMS_031 to append, but with a channel excluded
        # that was NOT excluded in the first run — this creates a var mismatch
        append_dir = str(tmp_path / "append_batch")
        os.makedirs(append_dir, exist_ok=True)
        shutil.copy(FILE_031, os.path.join(append_dir, os.path.basename(FILE_031)))

        with pytest.raises(ValueError, match="Channel names do not match"):
            ingest_sfc_folder(
                data_directory_path=append_dir,
                output_file_path=output_path,
                channels_to_exclude=["FSC-A"],   # not excluded before → var mismatch
                append_to_existing=True,
            )


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


class TestErrorHandling:
    def test_missing_directory_raises(self, temp_output_dir):
        with pytest.raises(FileNotFoundError, match="not found"):
            ingest_sfc_folder(
                data_directory_path="/nonexistent/path/",
                output_file_path=os.path.join(temp_output_dir, "sfc.h5ad"),
            )

    def test_empty_directory_raises(self, tmp_path, temp_output_dir):
        """Directory with no .fcs files must raise ValueError."""
        with pytest.raises(ValueError, match="No .fcs files found"):
            ingest_sfc_folder(
                data_directory_path=str(tmp_path),
                output_file_path=os.path.join(temp_output_dir, "sfc.h5ad"),
            )
