"""
Tests for tem_anndata_builder.py

Validates the AnnData structure (shape, dtypes, categories, warnings) produced
by build_tem_anndata.
"""

import warnings
from typing import Optional

import anndata
import numpy as np
import pandas as pd
import pytest

from mito_marker.controlled_vocabulary import (
    ALLOWED_CONDITIONS_TEM,
    ALLOWED_DIET_TEM,
    ALLOWED_SPECIES_TEM,
    TEM_FEATURE_COLUMNS,
)
from mito_marker.tem.tem_anndata_builder import build_tem_anndata


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_feature_matrix(n_rows: int = 5) -> np.ndarray:
    """Return a float32 feature matrix of shape (n_rows, 28) with positive values."""
    rng = np.random.default_rng(42)
    matrix = rng.uniform(low=0.1, high=10.0, size=(n_rows, len(TEM_FEATURE_COLUMNS)))
    return matrix.astype(np.float32)


def _make_filename_metadata(
    specie: str = "ZFish",
    condition: str = "Young",
    diet: Optional[str] = None,
    subject_id: str = "22511",
) -> dict:
    return {"specie": specie, "condition": condition, "diet": diet, "subject_ID": subject_id}


def _build(
    n_rows: int = 5,
    specie: str = "ZFish",
    condition: str = "Young",
    diet: Optional[str] = None,
    subject_id: str = "22511",
    image_name: str = "TestImage_field1",
    condition_name_from_file: str = "Young",
    source_filename: str = "YoungZebrafish22511grille1Dx1200field17_MITO_measurements.txt",
    source_file_path: str = "/fake/path/file.txt",
) -> anndata.AnnData:
    """Build a test AnnData object with controlled parameters."""
    feature_matrix = _make_feature_matrix(n_rows)
    filename_metadata = _make_filename_metadata(specie, condition, diet, subject_id)
    return build_tem_anndata(
        feature_matrix=feature_matrix,
        image_name=image_name,
        condition_name_from_file=condition_name_from_file,
        filename_metadata=filename_metadata,
        source_filename=source_filename,
        source_file_path=source_file_path,
    )


# ---------------------------------------------------------------------------
# AnnData shape and dtype
# ---------------------------------------------------------------------------


class TestAnnDataShape:
    def test_x_shape(self):
        result = _build(n_rows=7)
        assert result.X.shape == (7, len(TEM_FEATURE_COLUMNS))

    def test_x_dtype(self):
        result = _build()
        assert result.X.dtype == np.float32

    def test_n_obs(self):
        result = _build(n_rows=10)
        assert result.n_obs == 10

    def test_n_vars(self):
        result = _build()
        assert result.n_vars == len(TEM_FEATURE_COLUMNS)

    def test_x_values_preserved(self):
        feature_matrix = _make_feature_matrix(5)
        filename_metadata = _make_filename_metadata()
        result = build_tem_anndata(
            feature_matrix=feature_matrix,
            image_name="TestImage",
            condition_name_from_file="Young",
            filename_metadata=filename_metadata,
            source_filename="test.txt",
            source_file_path="/fake/test.txt",
        )
        np.testing.assert_array_equal(result.X, feature_matrix)


# ---------------------------------------------------------------------------
# .obs columns and values
# ---------------------------------------------------------------------------


class TestObsColumns:
    def test_obs_has_required_columns(self):
        result = _build()
        for column in ["specie", "condition", "diet", "subject_ID", "unique_subject_ID", "Image_Name", "source_filename"]:
            assert column in result.obs.columns, f"Missing column: {column}"

    def test_obs_has_no_age_group_column(self):
        result = _build()
        assert "age_group" not in result.obs.columns

    def test_obs_specie_value(self):
        result = _build(specie="ZFish")
        assert (result.obs["specie"] == "ZFish").all()

    def test_obs_condition_value(self):
        result = _build(condition="Old")
        assert (result.obs["condition"] == "Old").all()

    def test_obs_subject_id_value(self):
        result = _build(subject_id="42518")
        assert (result.obs["subject_ID"] == "42518").all()

    def test_unique_subject_id_combines_specie_and_subject_id(self):
        result = _build(specie="Mouse", subject_id="1")
        assert (result.obs["unique_subject_ID"] == "Mouse_1").all()

    def test_unique_subject_id_is_species_specific(self):
        """Two subjects with the same short ID but different species must differ."""
        result_mouse = _build(specie="Mouse", subject_id="1")
        result_zfish = _build(specie="ZFish", subject_id="1")
        assert result_mouse.obs["unique_subject_ID"].iloc[0] != result_zfish.obs["unique_subject_ID"].iloc[0]

    def test_unique_subject_id_none_specie_uses_unknown(self):
        result = _build(specie=None, subject_id="42")
        assert (result.obs["unique_subject_ID"] == "Unknown_42").all()

    def test_unique_subject_id_empty_subject_id_uses_unknown(self):
        result = _build(subject_id="")
        assert result.obs["unique_subject_ID"].iloc[0].endswith("_Unknown")

    def test_obs_image_name_value(self):
        result = _build(image_name="OldZebrafish42518grille4Bx1200field4")
        assert (result.obs["Image_Name"] == "OldZebrafish42518grille4Bx1200field4").all()

    def test_obs_source_filename_value(self):
        fname = "OldZebrafish42518grille4Bx1200field4_MITO_measurements.txt"
        result = _build(source_filename=fname)
        assert (result.obs["source_filename"] == fname).all()


# ---------------------------------------------------------------------------
# Categorical columns
# ---------------------------------------------------------------------------


class TestCategoricalColumns:
    def test_specie_is_categorical(self):
        result = _build(specie="ZFish")
        assert hasattr(result.obs["specie"], "cat"), "specie should be Categorical"

    def test_condition_is_categorical(self):
        result = _build(condition="Young")
        assert hasattr(result.obs["condition"], "cat")

    def test_specie_categories_include_all_allowed(self):
        result = _build(specie="KFish")
        for canonical_species in ALLOWED_SPECIES_TEM.keys():
            assert canonical_species in result.obs["specie"].cat.categories

    def test_condition_categories_include_all_allowed(self):
        result = _build(condition="Old")
        for canonical_condition in ALLOWED_CONDITIONS_TEM.keys():
            assert canonical_condition in result.obs["condition"].cat.categories

    def test_none_specie_stored_correctly(self):
        """When specie is None (parsing failed), categorical must handle it."""
        result = _build(specie=None)
        assert result.obs["specie"].isna().all()

    def test_none_condition_falls_back_to_file_condition(self):
        """When filename has no condition token, condition_name_from_file is used."""
        result = _build(condition=None, condition_name_from_file="Old")
        assert (result.obs["condition"] == "Old").all()

    def test_none_condition_and_invalid_file_condition_stores_nan(self):
        """When both filename and file condition are unusable, condition is NaN."""
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            result = _build(condition=None, condition_name_from_file="Unknown")
        assert result.obs["condition"].isna().all()

    def test_control_condition_accepted(self):
        """'Control' is a valid condition and must be stored correctly."""
        result = _build(condition=None, condition_name_from_file="Control")
        assert (result.obs["condition"] == "Control").all()

    def test_double_ko_condition_accepted(self):
        """'Double_KO_Mfn1_Mfn2' is a valid condition and must be stored correctly."""
        result = _build(condition=None, condition_name_from_file="Double_KO_Mfn1_Mfn2")
        assert (result.obs["condition"] == "Double_KO_Mfn1_Mfn2").all()

    def test_control_condition_in_category_axis(self):
        """'Control' must appear in the declared category axis."""
        result = _build(condition="Young")
        assert "Control" in result.obs["condition"].cat.categories

    def test_double_ko_condition_in_category_axis(self):
        """'Double_KO_Mfn1_Mfn2' must appear in the declared category axis."""
        result = _build(condition="Young")
        assert "Double_KO_Mfn1_Mfn2" in result.obs["condition"].cat.categories


# ---------------------------------------------------------------------------
# Diet column
# ---------------------------------------------------------------------------


class TestDietColumn:
    def test_diet_column_present(self):
        result = _build()
        assert "diet" in result.obs.columns

    def test_diet_is_categorical(self):
        result = _build(diet="AL")
        assert hasattr(result.obs["diet"], "cat"), "diet should be Categorical"

    def test_diet_al_value(self):
        result = _build(diet="AL")
        assert (result.obs["diet"] == "AL").all()

    def test_diet_if_value(self):
        result = _build(diet="IF")
        assert (result.obs["diet"] == "IF").all()

    def test_diet_none_stored_correctly(self):
        """When no diet token is in the filename, diet must be NaN (not a string)."""
        result = _build(diet=None)
        assert result.obs["diet"].isna().all()

    def test_diet_categories_include_all_allowed(self):
        """The full category axis must be declared even when the value is None."""
        result = _build(diet=None)
        for canonical_diet in ALLOWED_DIET_TEM.keys():
            assert canonical_diet in result.obs["diet"].cat.categories


# ---------------------------------------------------------------------------
# .var index
# ---------------------------------------------------------------------------


class TestVarIndex:
    def test_var_index_matches_feature_columns(self):
        result = _build()
        assert list(result.var_names) == TEM_FEATURE_COLUMNS

    def test_var_has_feature_description_column(self):
        result = _build()
        assert "feature_description" in result.var.columns

    def test_var_feature_description_empty_strings(self):
        result = _build()
        assert (result.var["feature_description"] == "").all()


# ---------------------------------------------------------------------------
# .uns
# ---------------------------------------------------------------------------


class TestUns:
    def test_uns_source_file_key_present(self):
        result = _build(source_file_path="/data/raw/test.txt")
        assert "source_file" in result.uns

    def test_uns_source_file_is_absolute_path(self):
        result = _build(source_file_path="/data/raw/test.txt")
        import os
        assert os.path.isabs(result.uns["source_file"])


# ---------------------------------------------------------------------------
# Condition double-check warning
# ---------------------------------------------------------------------------


class TestConditionDoubleCheck:
    def test_mismatch_emits_warning(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _build(condition="Young", condition_name_from_file="Old")
        warning_messages = [str(w.message) for w in caught]
        assert any("mismatch" in msg.lower() for msg in warning_messages)

    def test_match_emits_no_mismatch_warning(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _build(condition="Young", condition_name_from_file="Young")
        mismatch_warnings = [w for w in caught if "mismatch" in str(w.message).lower()]
        assert len(mismatch_warnings) == 0

    def test_none_condition_does_not_emit_mismatch_warning(self):
        """When filename parsing returned no condition, no mismatch warning should fire."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _build(condition=None, condition_name_from_file="Young")
        mismatch_warnings = [w for w in caught if "mismatch" in str(w.message).lower()]
        assert len(mismatch_warnings) == 0
