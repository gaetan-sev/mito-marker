"""
test_analysis_colors.py

Unit tests for mito_marker.analysis.colors.

Tests verify:
  - All condition values present in the AnnData get a color assignment.
  - subject_ID values are excluded from the condition palette.
  - Float columns (age) are excluded from the condition palette.
  - Bool columns (FlowAI_Pass) — their string representations are included.
  - PREFERRED_CONDITION_COLORS overrides take precedence over auto-assigned colors.
  - assign_color_palette() is idempotent: calling it twice does not change existing assignments.
  - get_color_for_value() returns the correct color from the stored palette.
  - get_color_for_value() returns the fallback color for unknown values.
  - get_subject_colors() returns one color per subject ID.
  - Colors are valid hex strings (e.g. "#rrggbb").
  - Empty AnnData (no filterable columns) produces an empty palette.
"""

import re

import anndata
import numpy as np
import pandas as pd
import pytest

from mito_marker.analysis.colors import (
    COLOR_PALETTE_KEY,
    assign_color_palette,
    get_color_for_value,
    get_subject_colors,
)
from mito_marker.controlled_vocabulary import PREFERRED_CONDITION_COLORS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_HEX_COLOR_PATTERN = re.compile(r"^#[0-9a-f]{6}$", re.IGNORECASE)


def _is_valid_hex_color(color: str) -> bool:
    return bool(_HEX_COLOR_PATTERN.match(color))


def _make_test_anndata(
    n_obs: int = 60,
    include_diet: bool = True,
    include_dilution: bool = True,
    include_flowai: bool = True,
    include_age: bool = True,
    include_subject: bool = True,
) -> anndata.AnnData:
    """
    Build a minimal AnnData with a subset of FILTERABLE_OBS_COLUMNS populated.

    diet: "AL" / "IF"
    dilution: "Diluted" / "Not_Diluted"
    FlowAI_Pass: bool True/False
    age: float (10.0 or 20.0)
    subject_ID: "S001"/"S002"/"S003"
    """
    np.random.seed(42)
    x_matrix = np.random.randn(n_obs, 5).astype(np.float32)
    obs_dict = {}

    if include_subject:
        subjects = ["S001", "S002", "S003"]
        obs_dict["subject_ID"] = np.tile(subjects, n_obs // len(subjects) + 1)[:n_obs]

    if include_diet:
        obs_dict["diet"] = np.where(np.arange(n_obs) % 2 == 0, "AL", "IF")

    if include_dilution:
        dilution_values = ["Diluted", "Not_Diluted"]
        obs_dict["dilution"] = np.tile(dilution_values, n_obs // 2 + 1)[:n_obs]

    if include_flowai:
        obs_dict["FlowAI_Pass"] = np.tile([True, False], n_obs // 2 + 1)[:n_obs].astype(bool)

    if include_age:
        obs_dict["age"] = np.tile([10.0, 20.0], n_obs // 2 + 1)[:n_obs].astype(np.float32)

    obs = pd.DataFrame(obs_dict)
    var = pd.DataFrame(index=[f"CH{i}" for i in range(5)])
    return anndata.AnnData(X=x_matrix, obs=obs, var=var)


# ---------------------------------------------------------------------------
# Tests: assign_color_palette — basic palette coverage
# ---------------------------------------------------------------------------


class TestAssignColorPaletteBasic:
    """Test that condition values receive colors and non-condition values are excluded."""

    def test_returns_anndata_object(self):
        adata = _make_test_anndata()
        result = assign_color_palette(adata)
        assert isinstance(result, anndata.AnnData)

    def test_returns_same_object(self):
        """assign_color_palette() modifies in-place and returns the same object."""
        adata = _make_test_anndata()
        result = assign_color_palette(adata)
        assert result is adata

    def test_uns_key_created(self):
        adata = _make_test_anndata()
        assign_color_palette(adata)
        assert COLOR_PALETTE_KEY in adata.uns

    def test_uns_palette_is_dict(self):
        adata = _make_test_anndata()
        assign_color_palette(adata)
        assert isinstance(adata.uns[COLOR_PALETTE_KEY], dict)

    def test_diet_values_get_colors(self):
        adata = _make_test_anndata()
        assign_color_palette(adata)
        palette = adata.uns[COLOR_PALETTE_KEY]
        assert "AL" in palette
        assert "IF" in palette

    def test_dilution_values_get_colors(self):
        adata = _make_test_anndata()
        assign_color_palette(adata)
        palette = adata.uns[COLOR_PALETTE_KEY]
        assert "Diluted" in palette
        assert "Not_Diluted" in palette

    def test_flowai_pass_values_get_colors(self):
        """FlowAI_Pass is bool — its string representations 'True'/'False' should be in palette."""
        adata = _make_test_anndata()
        assign_color_palette(adata)
        palette = adata.uns[COLOR_PALETTE_KEY]
        assert "True" in palette
        assert "False" in palette

    def test_subject_id_excluded_from_palette(self):
        """subject_ID values must NOT appear in the condition color palette."""
        adata = _make_test_anndata()
        assign_color_palette(adata)
        palette = adata.uns[COLOR_PALETTE_KEY]
        assert "S001" not in palette
        assert "S002" not in palette
        assert "S003" not in palette

    def test_float_age_excluded_from_palette(self):
        """Numerical float columns (age) must NOT be in the condition palette."""
        adata = _make_test_anndata()
        assign_color_palette(adata)
        palette = adata.uns[COLOR_PALETTE_KEY]
        # Age values are 10.0 and 20.0 → their string representations should not be present
        assert "10.0" not in palette
        assert "20.0" not in palette
        assert "10" not in palette
        assert "20" not in palette

    def test_all_colors_are_valid_hex(self):
        adata = _make_test_anndata()
        assign_color_palette(adata)
        palette = adata.uns[COLOR_PALETTE_KEY]
        for value, color in palette.items():
            assert _is_valid_hex_color(color), (
                f"Color '{color}' for value '{value}' is not a valid #rrggbb hex string"
            )

    def test_palette_size_matches_condition_values(self):
        """Palette should contain exactly the condition values: AL, IF, Diluted, Not_Diluted, True, False."""
        adata = _make_test_anndata()
        assign_color_palette(adata)
        palette = adata.uns[COLOR_PALETTE_KEY]
        # These are the expected condition values from the test AnnData
        expected_condition_values = {"AL", "IF", "Diluted", "Not_Diluted", "True", "False"}
        assert set(palette.keys()) == expected_condition_values


# ---------------------------------------------------------------------------
# Tests: idempotency
# ---------------------------------------------------------------------------


class TestAssignColorPaletteIdempotency:
    """Calling assign_color_palette() twice must not change existing assignments."""

    def test_second_call_does_not_change_colors(self):
        adata = _make_test_anndata()
        assign_color_palette(adata)
        first_palette = dict(adata.uns[COLOR_PALETTE_KEY])

        assign_color_palette(adata)
        second_palette = adata.uns[COLOR_PALETTE_KEY]

        assert first_palette == second_palette

    def test_second_call_preserves_all_keys(self):
        adata = _make_test_anndata()
        assign_color_palette(adata)
        keys_after_first = set(adata.uns[COLOR_PALETTE_KEY].keys())

        assign_color_palette(adata)
        keys_after_second = set(adata.uns[COLOR_PALETTE_KEY].keys())

        assert keys_after_first == keys_after_second

    def test_pre_existing_palette_entry_is_preserved(self):
        """If a color is already stored in .uns['color_palette'], it must not be overwritten."""
        adata = _make_test_anndata()
        # Pre-seed a custom color before calling assign_color_palette
        adata.uns[COLOR_PALETTE_KEY] = {"AL": "#aabbcc"}

        assign_color_palette(adata)
        assert adata.uns[COLOR_PALETTE_KEY]["AL"] == "#aabbcc"


# ---------------------------------------------------------------------------
# Tests: PREFERRED_CONDITION_COLORS override
# ---------------------------------------------------------------------------


class TestPreferredConditionColorsOverride:
    """PREFERRED_CONDITION_COLORS from controlled_vocabulary must take precedence."""

    def test_preferred_color_overrides_auto(self, monkeypatch):
        """When PREFERRED_CONDITION_COLORS contains a value, that hex must be used."""
        import mito_marker.analysis.colors as colors_module

        # Temporarily inject an override for "AL"
        monkeypatch.setattr(
            colors_module,
            "PREFERRED_CONDITION_COLORS",
            {"AL": "#123456"},
        )
        adata = _make_test_anndata()
        assign_color_palette(adata)
        assert adata.uns[COLOR_PALETTE_KEY]["AL"] == "#123456"

    def test_non_overridden_values_get_auto_colors(self, monkeypatch):
        """Values NOT in PREFERRED_CONDITION_COLORS still receive a valid auto color."""
        import mito_marker.analysis.colors as colors_module

        monkeypatch.setattr(
            colors_module,
            "PREFERRED_CONDITION_COLORS",
            {"AL": "#123456"},
        )
        adata = _make_test_anndata()
        assign_color_palette(adata)
        palette = adata.uns[COLOR_PALETTE_KEY]
        # "IF" is not overridden — should still have a valid color
        assert "IF" in palette
        assert _is_valid_hex_color(palette["IF"])
        # And the override must not bleed into "IF"
        assert palette["IF"] != "#123456"

    def test_preferred_color_survives_second_call(self, monkeypatch):
        """After two calls, the preferred color must remain unchanged."""
        import mito_marker.analysis.colors as colors_module

        monkeypatch.setattr(
            colors_module,
            "PREFERRED_CONDITION_COLORS",
            {"IF": "#fedcba"},
        )
        adata = _make_test_anndata()
        assign_color_palette(adata)
        assign_color_palette(adata)
        assert adata.uns[COLOR_PALETTE_KEY]["IF"] == "#fedcba"


# ---------------------------------------------------------------------------
# Tests: edge cases
# ---------------------------------------------------------------------------


class TestAssignColorPaletteEdgeCases:
    """Edge cases: empty AnnData, only subject column, only float column."""

    def test_no_condition_columns_produces_empty_palette(self):
        """If the only column is subject_ID (excluded), the palette should be empty."""
        adata = _make_test_anndata(
            include_diet=False,
            include_dilution=False,
            include_flowai=False,
            include_age=False,
            include_subject=True,
        )
        assign_color_palette(adata)
        assert adata.uns[COLOR_PALETTE_KEY] == {}

    def test_only_float_column_produces_empty_palette(self):
        """A float age column alone yields no condition values."""
        adata = _make_test_anndata(
            include_diet=False,
            include_dilution=False,
            include_flowai=False,
            include_age=True,
            include_subject=False,
        )
        assign_color_palette(adata)
        assert adata.uns[COLOR_PALETTE_KEY] == {}

    def test_diet_only_palette(self):
        """Only diet column → palette contains exactly 'AL' and 'IF'."""
        adata = _make_test_anndata(
            include_diet=True,
            include_dilution=False,
            include_flowai=False,
            include_age=False,
            include_subject=False,
        )
        assign_color_palette(adata)
        palette = adata.uns[COLOR_PALETTE_KEY]
        assert set(palette.keys()) == {"AL", "IF"}

    def test_anndata_without_any_filterable_columns(self):
        """AnnData with no FILTERABLE_OBS_COLUMNS present → empty palette."""
        x_matrix = np.random.randn(20, 3).astype(np.float32)
        obs = pd.DataFrame({"custom_column": ["A"] * 20})
        var = pd.DataFrame(index=["CH0", "CH1", "CH2"])
        adata = anndata.AnnData(X=x_matrix, obs=obs, var=var)
        assign_color_palette(adata)
        assert adata.uns[COLOR_PALETTE_KEY] == {}


# ---------------------------------------------------------------------------
# Tests: get_color_for_value
# ---------------------------------------------------------------------------


class TestGetColorForValue:
    """Unit tests for get_color_for_value() lookup helper."""

    def test_returns_stored_color(self):
        adata = _make_test_anndata()
        assign_color_palette(adata)
        color = get_color_for_value(adata, "AL")
        assert _is_valid_hex_color(color)
        assert color == adata.uns[COLOR_PALETTE_KEY]["AL"]

    def test_returns_fallback_for_unknown_value(self):
        adata = _make_test_anndata()
        assign_color_palette(adata)
        color = get_color_for_value(adata, "NonExistentValue")
        assert color == "#999999"

    def test_custom_fallback_color(self):
        adata = _make_test_anndata()
        assign_color_palette(adata)
        color = get_color_for_value(adata, "UnknownCondition", fallback_color="#ff0000")
        assert color == "#ff0000"

    def test_returns_fallback_when_no_palette_in_uns(self):
        """If .uns has no color_palette at all, must return the fallback."""
        x_matrix = np.random.randn(10, 3).astype(np.float32)
        obs = pd.DataFrame({"subject_ID": ["S001"] * 10})
        var = pd.DataFrame(index=["CH0", "CH1", "CH2"])
        adata = anndata.AnnData(X=x_matrix, obs=obs, var=var)
        color = get_color_for_value(adata, "AL")
        assert color == "#999999"

    def test_condition_value_cast_to_string(self):
        """Condition values are cast to str before lookup, so passing True looks up 'True'."""
        adata = _make_test_anndata()
        assign_color_palette(adata)
        # FlowAI_Pass stores 'True'/'False' in the palette
        color_via_string = get_color_for_value(adata, "True")
        color_via_bool = get_color_for_value(adata, True)
        assert color_via_string == color_via_bool


# ---------------------------------------------------------------------------
# Tests: get_subject_colors
# ---------------------------------------------------------------------------


class TestGetSubjectColors:
    """Unit tests for get_subject_colors() — colors generated at plot time."""

    def test_returns_one_color_per_subject(self):
        subject_ids = ["S001", "S002", "S003", "S004"]
        result = get_subject_colors(subject_ids)
        assert len(result) == 4

    def test_all_subject_colors_are_valid_hex(self):
        subject_ids = ["S001", "S002", "S003"]
        result = get_subject_colors(subject_ids)
        for sid, color in result.items():
            assert _is_valid_hex_color(color), (
                f"Color '{color}' for subject '{sid}' is not valid hex"
            )

    def test_subject_ids_are_keys(self):
        subject_ids = ["SubjectA", "SubjectB"]
        result = get_subject_colors(subject_ids)
        assert set(result.keys()) == {"SubjectA", "SubjectB"}

    def test_empty_list_returns_empty_dict(self):
        result = get_subject_colors([])
        assert result == {}

    def test_single_subject_returns_one_color(self):
        result = get_subject_colors(["OnlyOne"])
        assert len(result) == 1
        assert _is_valid_hex_color(result["OnlyOne"])

    def test_different_palettes_produce_different_colors(self):
        subject_ids = ["S001", "S002"]
        result_tab10 = get_subject_colors(subject_ids, palette="tab10")
        result_set2 = get_subject_colors(subject_ids, palette="Set2")
        # Different palettes should typically yield different colors
        assert result_tab10 != result_set2
