"""
Tests for tem_file_loader.py

Uses the committed fixture files in tests/fixtures/. Does not require
Google Drive access or any external data.
"""

import os
import tempfile

import numpy as np
import pandas as pd
import pytest

from mito_marker.controlled_vocabulary import TEM_FEATURE_COLUMNS
from mito_marker.tem.tem_file_loader import load_tem_file

# Absolute path to the fixtures directory
FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "fixtures")

FIXTURE_SOURIS_J3 = os.path.join(
    FIXTURES_DIR,
    "SourisJ3_Prot1533_Ech1755_Grille13E_x1500_field_19_MITO_measurements.txt",
)
FIXTURE_SOURIS_V1 = os.path.join(
    FIXTURES_DIR,
    "SourisV1_Prot1533_Ech1757_Grille14B_x1500_field_31_MITO_measurements.txt",
)
FIXTURE_M2124 = os.path.join(
    FIXTURES_DIR,
    "M2124_J1_G5_0114_MITO_measurements.txt",
)


# ---------------------------------------------------------------------------
# Successful loading
# ---------------------------------------------------------------------------


class TestSuccessfulLoading:
    def test_load_souris_j3_returns_three_tuple(self):
        result = load_tem_file(FIXTURE_SOURIS_J3)
        assert isinstance(result, tuple)
        assert len(result) == 3

    def test_souris_j3_feature_matrix_shape(self):
        feature_matrix, _, _ = load_tem_file(FIXTURE_SOURIS_J3)
        assert feature_matrix.ndim == 2
        assert feature_matrix.shape[1] == len(TEM_FEATURE_COLUMNS)

    def test_souris_j3_feature_matrix_dtype(self):
        feature_matrix, _, _ = load_tem_file(FIXTURE_SOURIS_J3)
        assert feature_matrix.dtype == np.float32

    def test_souris_j3_image_name(self):
        _, image_name, _ = load_tem_file(FIXTURE_SOURIS_J3)
        assert image_name == "SourisJ3_Prot1533_Ech1755_Grille13E_x1500_field_19"

    def test_souris_j3_condition_name(self):
        _, _, condition_name = load_tem_file(FIXTURE_SOURIS_J3)
        assert condition_name == "Young"

    def test_load_souris_v1(self):
        feature_matrix, image_name, condition_name = load_tem_file(FIXTURE_SOURIS_V1)
        assert feature_matrix.shape[1] == len(TEM_FEATURE_COLUMNS)
        assert image_name == "SourisV1_Prot1533_Ech1757_Grille14B_x1500_field_31"
        assert condition_name == "Old"

    def test_load_m2124(self):
        feature_matrix, image_name, condition_name = load_tem_file(FIXTURE_M2124)
        assert feature_matrix.shape[1] == len(TEM_FEATURE_COLUMNS)
        assert image_name == "M2124_J1_G5_0114"
        assert condition_name == "Young"

    def test_feature_matrix_has_no_nan(self):
        feature_matrix, _, _ = load_tem_file(FIXTURE_SOURIS_J3)
        assert not np.isnan(feature_matrix).any()

    def test_feature_matrix_row_count_positive(self):
        feature_matrix, _, _ = load_tem_file(FIXTURE_SOURIS_J3)
        assert feature_matrix.shape[0] > 0


# ---------------------------------------------------------------------------
# Column validation
# ---------------------------------------------------------------------------


class TestColumnValidation:
    def _write_temp_file(self, dataframe: pd.DataFrame) -> str:
        """Write a DataFrame to a temp .txt file and return the path."""
        tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix="_MITO_measurements.txt", delete=False
        )
        dataframe.to_csv(tmp.name, sep="\t", index=False)
        return tmp.name

    def _make_valid_dataframe(self, n_rows: int = 3) -> pd.DataFrame:
        """Build a minimal valid TEM DataFrame."""
        from mito_marker.controlled_vocabulary import TEM_EXPECTED_FILE_COLUMNS

        data = {}
        for col in TEM_EXPECTED_FILE_COLUMNS:
            if col == "Experiment_Name":
                data[col] = ["Test_EXP"] * n_rows
            elif col == "Condition_Name":
                data[col] = ["Young"] * n_rows
            elif col == "Image_Name":
                data[col] = ["TestImage_field1"] * n_rows
            elif col == "Mito_ID":
                data[col] = list(range(1, n_rows + 1))
            else:
                data[col] = [1.0] * n_rows  # valid positive floats
        return pd.DataFrame(data)

    def test_missing_column_raises_value_error(self):
        valid_df = self._make_valid_dataframe()
        # Remove one required column
        incomplete_df = valid_df.drop(columns=["Mito_Circularity"])
        tmp_path = self._write_temp_file(incomplete_df)
        try:
            with pytest.raises(ValueError, match="missing required columns"):
                load_tem_file(tmp_path)
        finally:
            os.unlink(tmp_path)

    def test_missing_metadata_column_raises_value_error(self):
        valid_df = self._make_valid_dataframe()
        incomplete_df = valid_df.drop(columns=["Image_Name"])
        tmp_path = self._write_temp_file(incomplete_df)
        try:
            with pytest.raises(ValueError, match="missing required columns"):
                load_tem_file(tmp_path)
        finally:
            os.unlink(tmp_path)


# ---------------------------------------------------------------------------
# Mito_Area validation
# ---------------------------------------------------------------------------


class TestMitoAreaValidation:
    def _write_temp_file(self, dataframe: pd.DataFrame) -> str:
        tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix="_MITO_measurements.txt", delete=False
        )
        dataframe.to_csv(tmp.name, sep="\t", index=False)
        return tmp.name

    def _make_valid_dataframe(self) -> pd.DataFrame:
        from mito_marker.controlled_vocabulary import TEM_EXPECTED_FILE_COLUMNS

        data = {}
        for col in TEM_EXPECTED_FILE_COLUMNS:
            if col == "Experiment_Name":
                data[col] = ["Test_EXP", "Test_EXP"]
            elif col == "Condition_Name":
                data[col] = ["Young", "Young"]
            elif col == "Image_Name":
                data[col] = ["TestImage"] * 2
            elif col == "Mito_ID":
                data[col] = [1, 2]
            else:
                data[col] = [1.0, 2.0]
        return pd.DataFrame(data)

    def test_zero_area_raises_assertion_error(self):
        df = self._make_valid_dataframe()
        df.at[0, "Mito_Area"] = 0.0
        tmp_path = self._write_temp_file(df)
        try:
            with pytest.raises(AssertionError, match="Mito_Area"):
                load_tem_file(tmp_path)
        finally:
            os.unlink(tmp_path)

    def test_negative_area_raises_assertion_error(self):
        df = self._make_valid_dataframe()
        df.at[1, "Mito_Area"] = -5.0
        tmp_path = self._write_temp_file(df)
        try:
            with pytest.raises(AssertionError, match="Mito_Area"):
                load_tem_file(tmp_path)
        finally:
            os.unlink(tmp_path)


# ---------------------------------------------------------------------------
# NaN validation
# ---------------------------------------------------------------------------


class TestNanValidation:
    def _write_temp_file(self, dataframe: pd.DataFrame) -> str:
        tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix="_MITO_measurements.txt", delete=False
        )
        dataframe.to_csv(tmp.name, sep="\t", index=False)
        return tmp.name

    def _make_valid_dataframe(self) -> pd.DataFrame:
        from mito_marker.controlled_vocabulary import TEM_EXPECTED_FILE_COLUMNS

        data = {}
        for col in TEM_EXPECTED_FILE_COLUMNS:
            if col == "Experiment_Name":
                data[col] = ["Test_EXP"]
            elif col == "Condition_Name":
                data[col] = ["Young"]
            elif col == "Image_Name":
                data[col] = ["TestImage"]
            elif col == "Mito_ID":
                data[col] = [1]
            else:
                data[col] = [1.0]
        return pd.DataFrame(data)

    def test_nan_in_feature_raises_assertion_error(self):
        import math

        df = self._make_valid_dataframe()
        df.at[0, "Mito_Circularity"] = float("nan")
        tmp_path = self._write_temp_file(df)
        try:
            with pytest.raises(AssertionError, match="NaN"):
                load_tem_file(tmp_path)
        finally:
            os.unlink(tmp_path)
