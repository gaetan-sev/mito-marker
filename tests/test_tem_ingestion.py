"""
Tests for tem_ingestion.py

Uses the three committed fixture files. Tests single-file ingestion,
multi-file concatenation, feature mismatch detection, .h5ad round-trip,
and append mode.
"""

import os
import tempfile

import anndata
import numpy as np
import pytest

from mito_marker.controlled_vocabulary import TEM_FEATURE_COLUMNS
from mito_marker.tem.tem_ingestion import ingest_tem_folder

FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "fixtures")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ingest_fixtures(
    tmp_path: str,
    filenames: list[str],
    append_to_existing: bool = False,
) -> anndata.AnnData:
    """
    Copy a subset of fixture files to a temporary directory and run ingestion.
    """
    import shutil

    for filename in filenames:
        src = os.path.join(FIXTURES_DIR, filename)
        shutil.copy(src, os.path.join(tmp_path, filename))

    output_path = os.path.join(tmp_path, "output.h5ad")
    return ingest_tem_folder(
        data_directory_path=tmp_path,
        output_file_path=output_path,
        append_to_existing=append_to_existing,
    )


# ---------------------------------------------------------------------------
# Single file
# ---------------------------------------------------------------------------


class TestSingleFileIngestion:
    def test_returns_anndata(self, tmp_path):
        result = _ingest_fixtures(
            str(tmp_path),
            ["SourisJ3_Prot1533_Ech1755_Grille13E_x1500_field_19_MITO_measurements.txt"],
        )
        assert isinstance(result, anndata.AnnData)

    def test_x_shape_single_file(self, tmp_path):
        result = _ingest_fixtures(
            str(tmp_path),
            ["SourisJ3_Prot1533_Ech1755_Grille13E_x1500_field_19_MITO_measurements.txt"],
        )
        assert result.n_vars == len(TEM_FEATURE_COLUMNS)
        assert result.n_obs > 0

    def test_x_dtype_float32(self, tmp_path):
        result = _ingest_fixtures(
            str(tmp_path),
            ["SourisJ3_Prot1533_Ech1755_Grille13E_x1500_field_19_MITO_measurements.txt"],
        )
        assert result.X.dtype == np.float32

    def test_obs_columns_present(self, tmp_path):
        result = _ingest_fixtures(
            str(tmp_path),
            ["SourisJ3_Prot1533_Ech1755_Grille13E_x1500_field_19_MITO_measurements.txt"],
        )
        for col in ["specie", "condition", "subject_ID", "unique_subject_ID", "Image_Name", "source_filename"]:
            assert col in result.obs.columns

    def test_obs_has_no_age_group_column(self, tmp_path):
        result = _ingest_fixtures(
            str(tmp_path),
            ["SourisJ3_Prot1533_Ech1755_Grille13E_x1500_field_19_MITO_measurements.txt"],
        )
        assert "age_group" not in result.obs.columns

    def test_obs_index_is_clean_integers(self, tmp_path):
        result = _ingest_fixtures(
            str(tmp_path),
            ["SourisJ3_Prot1533_Ech1755_Grille13E_x1500_field_19_MITO_measurements.txt"],
        )
        expected_index = [str(i) for i in range(result.n_obs)]
        assert list(result.obs_names) == expected_index

    def test_source_filename_stored(self, tmp_path):
        filename = "SourisJ3_Prot1533_Ech1755_Grille13E_x1500_field_19_MITO_measurements.txt"
        result = _ingest_fixtures(str(tmp_path), [filename])
        assert (result.obs["source_filename"] == filename).all()


# ---------------------------------------------------------------------------
# Multi-file concatenation
# ---------------------------------------------------------------------------


class TestMultiFileIngestion:
    def test_two_files_concatenated(self, tmp_path):
        result = _ingest_fixtures(
            str(tmp_path),
            [
                "SourisJ3_Prot1533_Ech1755_Grille13E_x1500_field_19_MITO_measurements.txt",
                "SourisV1_Prot1533_Ech1757_Grille14B_x1500_field_31_MITO_measurements.txt",
            ],
        )
        assert result.n_obs > 0
        assert result.n_vars == len(TEM_FEATURE_COLUMNS)

    def test_three_files_concatenated(self, tmp_path):
        result = _ingest_fixtures(
            str(tmp_path),
            [
                "SourisJ3_Prot1533_Ech1755_Grille13E_x1500_field_19_MITO_measurements.txt",
                "SourisV1_Prot1533_Ech1757_Grille14B_x1500_field_31_MITO_measurements.txt",
                "M2124_J1_G5_0114_MITO_measurements.txt",
            ],
        )
        assert result.n_vars == len(TEM_FEATURE_COLUMNS)

    def test_concatenated_obs_index_is_clean_integers(self, tmp_path):
        result = _ingest_fixtures(
            str(tmp_path),
            [
                "SourisJ3_Prot1533_Ech1755_Grille13E_x1500_field_19_MITO_measurements.txt",
                "SourisV1_Prot1533_Ech1757_Grille14B_x1500_field_31_MITO_measurements.txt",
            ],
        )
        expected_index = [str(i) for i in range(result.n_obs)]
        assert list(result.obs_names) == expected_index

    def test_concatenated_n_obs_equals_sum_of_files(self, tmp_path):
        """The combined mitochondrion count must equal the sum of the two files."""
        import shutil

        file_a = "SourisJ3_Prot1533_Ech1755_Grille13E_x1500_field_19_MITO_measurements.txt"
        file_b = "SourisV1_Prot1533_Ech1757_Grille14B_x1500_field_31_MITO_measurements.txt"

        # Count rows in each fixture separately
        tmp_a = tempfile.mkdtemp()
        shutil.copy(os.path.join(FIXTURES_DIR, file_a), os.path.join(tmp_a, file_a))
        result_a = ingest_tem_folder(
            tmp_a, os.path.join(tmp_a, "out.h5ad")
        )
        n_a = result_a.n_obs

        tmp_b = tempfile.mkdtemp()
        shutil.copy(os.path.join(FIXTURES_DIR, file_b), os.path.join(tmp_b, file_b))
        result_b = ingest_tem_folder(
            tmp_b, os.path.join(tmp_b, "out.h5ad")
        )
        n_b = result_b.n_obs

        combined = _ingest_fixtures(str(tmp_path), [file_a, file_b])
        assert combined.n_obs == n_a + n_b


# ---------------------------------------------------------------------------
# .h5ad round-trip
# ---------------------------------------------------------------------------


class TestH5adRoundTrip:
    def test_saves_h5ad(self, tmp_path):
        output_path = str(tmp_path / "output.h5ad")
        import shutil

        file_a = "SourisJ3_Prot1533_Ech1755_Grille13E_x1500_field_19_MITO_measurements.txt"
        shutil.copy(os.path.join(FIXTURES_DIR, file_a), str(tmp_path / file_a))
        ingest_tem_folder(
            data_directory_path=str(tmp_path),
            output_file_path=output_path,
        )
        assert os.path.exists(output_path)

    def test_reloaded_anndata_matches(self, tmp_path):
        output_path = str(tmp_path / "output.h5ad")
        import shutil

        file_a = "SourisJ3_Prot1533_Ech1755_Grille13E_x1500_field_19_MITO_measurements.txt"
        shutil.copy(os.path.join(FIXTURES_DIR, file_a), str(tmp_path / file_a))
        original = ingest_tem_folder(
            data_directory_path=str(tmp_path),
            output_file_path=output_path,
        )
        reloaded = anndata.read_h5ad(output_path)
        assert reloaded.n_obs == original.n_obs
        assert reloaded.n_vars == original.n_vars
        np.testing.assert_array_almost_equal(reloaded.X, original.X, decimal=5)


# ---------------------------------------------------------------------------
# Append mode
# ---------------------------------------------------------------------------


class TestAppendMode:
    def test_append_increases_n_obs(self, tmp_path):
        import shutil

        file_a = "SourisJ3_Prot1533_Ech1755_Grille13E_x1500_field_19_MITO_measurements.txt"
        file_b = "SourisV1_Prot1533_Ech1757_Grille14B_x1500_field_31_MITO_measurements.txt"
        output_path = str(tmp_path / "output.h5ad")

        # First ingestion
        dir_a = tempfile.mkdtemp()
        shutil.copy(os.path.join(FIXTURES_DIR, file_a), os.path.join(dir_a, file_a))
        first_result = ingest_tem_folder(
            data_directory_path=dir_a,
            output_file_path=output_path,
        )
        n_first = first_result.n_obs

        # Second ingestion in append mode — file_b goes to a separate directory
        dir_b = tempfile.mkdtemp()
        shutil.copy(os.path.join(FIXTURES_DIR, file_b), os.path.join(dir_b, file_b))
        appended_result = ingest_tem_folder(
            data_directory_path=dir_b,
            output_file_path=output_path,
            append_to_existing=True,
        )
        assert appended_result.n_obs > n_first


# ---------------------------------------------------------------------------
# Error cases
# ---------------------------------------------------------------------------


class TestErrorCases:
    def test_empty_directory_raises_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            ingest_tem_folder(
                data_directory_path=str(tmp_path),
                output_file_path=str(tmp_path / "out.h5ad"),
            )


# ---------------------------------------------------------------------------
# use_filename_for_metadata=False mode
# ---------------------------------------------------------------------------


def _ingest_fixtures_no_filename(
    tmp_path: str,
    filenames: list[str],
    specie: str,
) -> anndata.AnnData:
    """
    Copy fixture files to a temporary directory and run ingestion in override mode.
    """
    import shutil

    for filename in filenames:
        src = os.path.join(FIXTURES_DIR, filename)
        shutil.copy(src, os.path.join(tmp_path, filename))

    output_path = os.path.join(tmp_path, "output.h5ad")
    return ingest_tem_folder(
        data_directory_path=tmp_path,
        output_file_path=output_path,
        specie=specie,
        use_filename_for_metadata=False,
    )


class TestNoFilenameMetadataMode:
    def test_specie_overridden_correctly(self, tmp_path):
        result = _ingest_fixtures_no_filename(
            str(tmp_path),
            ["SourisJ3_Prot1533_Ech1755_Grille13E_x1500_field_19_MITO_measurements.txt"],
            specie="Mouse",
        )
        assert (result.obs["specie"] == "Mouse").all()

    def test_subject_id_is_001_for_single_file(self, tmp_path):
        result = _ingest_fixtures_no_filename(
            str(tmp_path),
            ["SourisJ3_Prot1533_Ech1755_Grille13E_x1500_field_19_MITO_measurements.txt"],
            specie="Mouse",
        )
        assert (result.obs["subject_ID"] == "001").all()

    def test_subject_ids_increment_across_files(self, tmp_path):
        result = _ingest_fixtures_no_filename(
            str(tmp_path),
            [
                "SourisJ3_Prot1533_Ech1755_Grille13E_x1500_field_19_MITO_measurements.txt",
                "SourisV1_Prot1533_Ech1757_Grille14B_x1500_field_31_MITO_measurements.txt",
            ],
            specie="Mouse",
        )
        subject_ids = result.obs["subject_ID"].unique().tolist()
        assert "001" in subject_ids
        assert "002" in subject_ids
        assert len(subject_ids) == 2

    def test_condition_read_from_file_content(self, tmp_path):
        # The fixture file has Condition_Name = "Young" — must be stored in .obs
        result = _ingest_fixtures_no_filename(
            str(tmp_path),
            ["SourisJ3_Prot1533_Ech1755_Grille13E_x1500_field_19_MITO_measurements.txt"],
            specie="Mouse",
        )
        assert (result.obs["condition"] == "Young").all()

    def test_unique_subject_id_uses_override_specie(self, tmp_path):
        result = _ingest_fixtures_no_filename(
            str(tmp_path),
            ["SourisJ3_Prot1533_Ech1755_Grille13E_x1500_field_19_MITO_measurements.txt"],
            specie="Mouse",
        )
        assert (result.obs["unique_subject_ID"] == "Mouse_001").all()

    def test_obs_columns_present(self, tmp_path):
        result = _ingest_fixtures_no_filename(
            str(tmp_path),
            ["SourisJ3_Prot1533_Ech1755_Grille13E_x1500_field_19_MITO_measurements.txt"],
            specie="Mouse",
        )
        for col in ["specie", "condition", "subject_ID", "unique_subject_ID", "Image_Name", "source_filename"]:
            assert col in result.obs.columns

    def test_missing_specie_raises_value_error(self, tmp_path):
        import shutil
        filename = "SourisJ3_Prot1533_Ech1755_Grille13E_x1500_field_19_MITO_measurements.txt"
        shutil.copy(os.path.join(FIXTURES_DIR, filename), os.path.join(str(tmp_path), filename))
        with pytest.raises(ValueError, match="specie"):
            ingest_tem_folder(
                data_directory_path=str(tmp_path),
                output_file_path=str(tmp_path / "out.h5ad"),
                specie=None,
                use_filename_for_metadata=False,
            )

    def test_unknown_specie_raises_value_error(self, tmp_path):
        import shutil
        filename = "SourisJ3_Prot1533_Ech1755_Grille13E_x1500_field_19_MITO_measurements.txt"
        shutil.copy(os.path.join(FIXTURES_DIR, filename), os.path.join(str(tmp_path), filename))
        with pytest.raises(ValueError, match="Unknown specie"):
            ingest_tem_folder(
                data_directory_path=str(tmp_path),
                output_file_path=str(tmp_path / "out.h5ad"),
                specie="NotASpecies",
                use_filename_for_metadata=False,
            )

    def test_returns_anndata(self, tmp_path):
        result = _ingest_fixtures_no_filename(
            str(tmp_path),
            ["SourisJ3_Prot1533_Ech1755_Grille13E_x1500_field_19_MITO_measurements.txt"],
            specie="Mouse",
        )
        assert isinstance(result, anndata.AnnData)
