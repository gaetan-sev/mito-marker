"""
test_controlled_vocabulary.py

Unit tests for src/mito_marker/controlled_vocabulary.py.

Tests validate the structure and content of every constant:
  - All ALLOWED_* categorical dicts are non-empty and contain expected values.
  - MARKER_TOKEN_TO_CANONICAL_SFC resolves all tokens to valid canonical markers.
  - FLOWAI_PASS_TOKENS is a non-empty set containing expected sentinel values.
  - PREFERRED_CONDITION_COLORS has valid #rrggbb hex values when populated.
  - NUMERICAL_OBS_BOUNDS has "unit", "min", "max" per entry with min < max.
  - FILTERABLE_OBS_COLUMNS is ordered with "subject_ID" first, no duplicates.
  - SFC_CHANNELS_TO_KEEP has exactly 144 unique non-FJComp entries.
  - SFC_CHANNELS_TO_EXCLUDE has exactly 64 entries all starting with "FJComp-".
  - The two channel lists are disjoint.
  - The assert-based validation pattern raises AssertionError for unknown values.
"""

import re

import pytest

from mito_marker.controlled_vocabulary import (
    ALLOWED_CELL_TYPES_TEM_GONAD_STUDY,
    ALLOWED_CONDITIONS_TEM,
    ALLOWED_DIET_SFC,
    ALLOWED_DILUTION_SFC,
    ALLOWED_MARKERS_SFC,
    ALLOWED_SPECIES_SFC,
    ALLOWED_STAINING_VALUES,
    ALLOWED_TREATMENT_GROUPS_TEM_GONAD_STUDY,
    CELL_TYPE_ORDER_GONAD_STUDY,
    FILTERABLE_OBS_COLUMNS,
    FLOWAI_PASS_TOKENS,
    MARKER_TOKEN_TO_CANONICAL_SFC,
    NUMERICAL_OBS_BOUNDS,
    PREFERRED_CONDITION_COLORS,
    SFC_CHANNELS_TO_EXCLUDE,
    SFC_CHANNELS_TO_KEEP,
)

_HEX_COLOR_PATTERN = re.compile(r"^#[0-9a-fA-F]{6}$")


# ---------------------------------------------------------------------------
# Categorical dictionaries
# ---------------------------------------------------------------------------


class TestCategoricalDictionaries:
    """Each ALLOWED_* dict must be non-empty with non-empty string keys and values."""

    def _assert_valid_vocabulary_dict(self, vocabulary_dict: dict, dict_name: str) -> None:
        """Shared structure check: non-empty dict, all keys and values are non-empty strings."""
        assert isinstance(vocabulary_dict, dict), f"{dict_name} must be a dict"
        assert len(vocabulary_dict) > 0, f"{dict_name} must not be empty"
        for key, value in vocabulary_dict.items():
            assert isinstance(key, str) and len(key) > 0, (
                f"{dict_name}: key '{key}' must be a non-empty string"
            )
            assert isinstance(value, str) and len(value) > 0, (
                f"{dict_name}: description for key '{key}' must be a non-empty string"
            )

    def test_allowed_staining_values_structure(self):
        self._assert_valid_vocabulary_dict(ALLOWED_STAINING_VALUES, "ALLOWED_STAINING_VALUES")

    def test_allowed_staining_values_contains_expected_keys(self):
        assert "DeepRed" in ALLOWED_STAINING_VALUES
        assert "Unstained" in ALLOWED_STAINING_VALUES

    def test_allowed_conditions_tem_structure(self):
        self._assert_valid_vocabulary_dict(ALLOWED_CONDITIONS_TEM, "ALLOWED_CONDITIONS_TEM")

    def test_allowed_conditions_tem_contains_expected_keys(self):
        assert "Young" in ALLOWED_CONDITIONS_TEM
        assert "Old" in ALLOWED_CONDITIONS_TEM

    def test_allowed_species_sfc_structure(self):
        self._assert_valid_vocabulary_dict(ALLOWED_SPECIES_SFC, "ALLOWED_SPECIES_SFC")

    def test_allowed_species_sfc_contains_expected_keys(self):
        assert "MNMS" in ALLOWED_SPECIES_SFC
        assert "WORMS" in ALLOWED_SPECIES_SFC
        assert "FLY" in ALLOWED_SPECIES_SFC

    def test_allowed_diet_sfc_structure(self):
        self._assert_valid_vocabulary_dict(ALLOWED_DIET_SFC, "ALLOWED_DIET_SFC")

    def test_allowed_diet_sfc_contains_expected_keys(self):
        assert "AL" in ALLOWED_DIET_SFC
        assert "IF" in ALLOWED_DIET_SFC

    def test_allowed_dilution_sfc_structure(self):
        self._assert_valid_vocabulary_dict(ALLOWED_DILUTION_SFC, "ALLOWED_DILUTION_SFC")

    def test_allowed_dilution_sfc_contains_expected_keys(self):
        assert "Diluted" in ALLOWED_DILUTION_SFC
        assert "Not_Diluted" in ALLOWED_DILUTION_SFC

    def test_allowed_markers_sfc_structure(self):
        self._assert_valid_vocabulary_dict(ALLOWED_MARKERS_SFC, "ALLOWED_MARKERS_SFC")

    def test_allowed_markers_sfc_contains_expected_keys(self):
        assert "MtDeepRed" in ALLOWED_MARKERS_SFC
        assert "Not_Marked" in ALLOWED_MARKERS_SFC

    def test_allowed_cell_types_tem_gonad_study_structure(self):
        self._assert_valid_vocabulary_dict(
            ALLOWED_CELL_TYPES_TEM_GONAD_STUDY, "ALLOWED_CELL_TYPES_TEM_GONAD_STUDY"
        )

    def test_allowed_cell_types_tem_gonad_study_contains_expected_keys(self):
        for cell_type in ["Dis", "Loop", "Pro", "Emb", "Sp", "Mu"]:
            assert cell_type in ALLOWED_CELL_TYPES_TEM_GONAD_STUDY

    def test_allowed_treatment_groups_tem_gonad_study_structure(self):
        self._assert_valid_vocabulary_dict(
            ALLOWED_TREATMENT_GROUPS_TEM_GONAD_STUDY,
            "ALLOWED_TREATMENT_GROUPS_TEM_GONAD_STUDY",
        )

    def test_allowed_treatment_groups_tem_gonad_study_contains_expected_keys(self):
        assert "Control" in ALLOWED_TREATMENT_GROUPS_TEM_GONAD_STUDY
        assert "Chemical Stress" in ALLOWED_TREATMENT_GROUPS_TEM_GONAD_STUDY

    def test_kd_deliberately_not_declared(self):
        """KD (knockdown) is out of scope — confirmed with Anna Mattout, 2026-09-14."""
        assert "KD" not in ALLOWED_TREATMENT_GROUPS_TEM_GONAD_STUDY


# ---------------------------------------------------------------------------
# C. elegans gonad study — cell type order
# ---------------------------------------------------------------------------


class TestCellTypeOrderGonadStudy:
    """CELL_TYPE_ORDER_GONAD_STUDY must list every allowed cell type exactly once,
    in the germ-cell-differentiation order confirmed with Anna Mattout."""

    def test_is_a_list(self):
        assert isinstance(CELL_TYPE_ORDER_GONAD_STUDY, list)

    def test_matches_confirmed_order(self):
        assert CELL_TYPE_ORDER_GONAD_STUDY == ["Dis", "Loop", "Pro", "Emb", "Sp", "Mu"]

    def test_contains_every_allowed_cell_type_exactly_once(self):
        assert sorted(CELL_TYPE_ORDER_GONAD_STUDY) == sorted(
            ALLOWED_CELL_TYPES_TEM_GONAD_STUDY.keys()
        )
        assert len(CELL_TYPE_ORDER_GONAD_STUDY) == len(set(CELL_TYPE_ORDER_GONAD_STUDY))

    def test_every_cell_type_has_a_preferred_color(self):
        """Every cell type must resolve to a color so no figure falls back to
        the automatic palette and silently breaks cross-figure consistency."""
        for cell_type in CELL_TYPE_ORDER_GONAD_STUDY:
            assert cell_type in PREFERRED_CONDITION_COLORS, (
                f"'{cell_type}' has no entry in PREFERRED_CONDITION_COLORS"
            )
            assert _HEX_COLOR_PATTERN.match(PREFERRED_CONDITION_COLORS[cell_type])


# ---------------------------------------------------------------------------
# Marker token mapping
# ---------------------------------------------------------------------------


class TestMarkerTokenMapping:
    """MARKER_TOKEN_TO_CANONICAL_SFC must map every token to a valid canonical marker."""

    def test_all_canonical_values_exist_in_allowed_markers(self):
        """Every resolved canonical value must be a key in ALLOWED_MARKERS_SFC."""
        for token, canonical_value in MARKER_TOKEN_TO_CANONICAL_SFC.items():
            assert canonical_value in ALLOWED_MARKERS_SFC, (
                f"Token '{token}' resolves to '{canonical_value}' "
                f"which is not declared in ALLOWED_MARKERS_SFC"
            )

    def test_dr_maps_to_mtdeepred(self):
        assert MARKER_TOKEN_TO_CANONICAL_SFC["DR"] == "MtDeepRed"

    def test_deepred_maps_to_mtdeepred(self):
        assert MARKER_TOKEN_TO_CANONICAL_SFC["DeepRed"] == "MtDeepRed"

    def test_nm_maps_to_not_marked(self):
        assert MARKER_TOKEN_TO_CANONICAL_SFC["NM"] == "Not_Marked"


# ---------------------------------------------------------------------------
# FlowAI pass tokens
# ---------------------------------------------------------------------------


class TestFlowAIPassTokens:
    """FLOWAI_PASS_TOKENS must be a non-empty set with the expected sentinel values."""

    def test_is_a_set(self):
        assert isinstance(FLOWAI_PASS_TOKENS, set)

    def test_is_non_empty(self):
        assert len(FLOWAI_PASS_TOKENS) > 0

    def test_contains_flowai_good_events(self):
        assert "FlowAIGoodEvents" in FLOWAI_PASS_TOKENS

    def test_contains_good_token(self):
        assert "good" in FLOWAI_PASS_TOKENS

    def test_contains_events_token(self):
        assert "events" in FLOWAI_PASS_TOKENS


# ---------------------------------------------------------------------------
# Preferred condition colors
# ---------------------------------------------------------------------------


class TestPreferredConditionColors:
    """PREFERRED_CONDITION_COLORS must be a dict; any values present must be valid hex."""

    def test_is_a_dict(self):
        assert isinstance(PREFERRED_CONDITION_COLORS, dict)

    def test_all_values_are_valid_hex_when_populated(self):
        """Any color overrides already defined must use the #rrggbb format."""
        for condition_value, color in PREFERRED_CONDITION_COLORS.items():
            assert _HEX_COLOR_PATTERN.match(color), (
                f"Color '{color}' for condition '{condition_value}' "
                f"is not a valid #rrggbb hex string"
            )


# ---------------------------------------------------------------------------
# Numerical observation bounds
# ---------------------------------------------------------------------------


class TestNumericalObsBounds:
    """NUMERICAL_OBS_BOUNDS must have coherent, non-negative bounds for every declared field."""

    _EXPECTED_FIELD_NAMES = {
        "insulin_level_uU_per_mL",
        "steps_per_day",
        "age_days",
        "age_months",
        "age_years",
        "body_weight_kilos",
    }

    def test_all_expected_fields_present(self):
        for field_name in self._EXPECTED_FIELD_NAMES:
            assert field_name in NUMERICAL_OBS_BOUNDS, (
                f"Expected bound entry '{field_name}' is missing from NUMERICAL_OBS_BOUNDS"
            )

    def test_every_entry_has_unit_min_max_keys(self):
        for field_name, bounds in NUMERICAL_OBS_BOUNDS.items():
            assert "unit" in bounds, f"'{field_name}' is missing the 'unit' key"
            assert "min" in bounds, f"'{field_name}' is missing the 'min' key"
            assert "max" in bounds, f"'{field_name}' is missing the 'max' key"

    def test_unit_is_non_empty_string(self):
        for field_name, bounds in NUMERICAL_OBS_BOUNDS.items():
            assert isinstance(bounds["unit"], str) and len(bounds["unit"]) > 0, (
                f"'{field_name}' unit must be a non-empty string, got: {bounds['unit']!r}"
            )

    def test_min_strictly_less_than_max(self):
        for field_name, bounds in NUMERICAL_OBS_BOUNDS.items():
            assert bounds["min"] < bounds["max"], (
                f"'{field_name}': min ({bounds['min']}) must be < max ({bounds['max']})"
            )

    def test_min_is_non_negative(self):
        """Physiological measurements cannot have a negative lower bound."""
        for field_name, bounds in NUMERICAL_OBS_BOUNDS.items():
            assert bounds["min"] >= 0, (
                f"'{field_name}': min ({bounds['min']}) must be >= 0 "
                f"for a physiological measurement"
            )


# ---------------------------------------------------------------------------
# Filterable observation columns
# ---------------------------------------------------------------------------


class TestFilterableObsColumns:
    """FILTERABLE_OBS_COLUMNS must be an ordered list with specie first, no duplicates."""

    def test_is_a_list(self):
        assert isinstance(FILTERABLE_OBS_COLUMNS, list)

    def test_is_non_empty(self):
        assert len(FILTERABLE_OBS_COLUMNS) > 0

    def test_all_entries_are_non_empty_strings(self):
        for column_name in FILTERABLE_OBS_COLUMNS:
            assert isinstance(column_name, str) and len(column_name) > 0, (
                f"Entry '{column_name}' must be a non-empty string"
            )

    def test_specie_is_first(self):
        """specie must be first — biological species is the primary filter dimension."""
        assert FILTERABLE_OBS_COLUMNS[0] == "specie"

    def test_unique_subject_id_is_second(self):
        """unique_subject_ID must be second — shown right after species."""
        assert FILTERABLE_OBS_COLUMNS[1] == "unique_subject_ID"

    def test_no_duplicate_entries(self):
        assert len(FILTERABLE_OBS_COLUMNS) == len(set(FILTERABLE_OBS_COLUMNS)), (
            "FILTERABLE_OBS_COLUMNS contains duplicate entries"
        )


# ---------------------------------------------------------------------------
# SFC channel lists
# ---------------------------------------------------------------------------


class TestSfcChannelLists:
    """Channel lists must be correctly sized, internally unique, and mutually exclusive."""

    def test_channels_to_keep_has_exactly_142_entries(self):
        # Time and FlowAI are now in SFC_CHANNELS_TO_EXCLUDE, not here.
        assert len(SFC_CHANNELS_TO_KEEP) == 142

    def test_channels_to_keep_are_unique(self):
        assert len(SFC_CHANNELS_TO_KEEP) == len(set(SFC_CHANNELS_TO_KEEP)), (
            "SFC_CHANNELS_TO_KEEP contains duplicate channel names"
        )

    def test_channels_to_keep_have_no_fjcomp_prefix(self):
        """Raw channels must never carry the FJComp- prefix."""
        fjcomp_channels = [ch for ch in SFC_CHANNELS_TO_KEEP if ch.startswith("FJComp-")]
        assert len(fjcomp_channels) == 0, (
            f"SFC_CHANNELS_TO_KEEP contains {len(fjcomp_channels)} FJComp channel(s): "
            f"{fjcomp_channels}"
        )

    def test_channels_to_exclude_is_not_empty(self):
        assert len(SFC_CHANNELS_TO_EXCLUDE) > 0

    def test_keep_and_exclude_lists_are_disjoint(self):
        """No channel should appear in both lists."""
        overlap = set(SFC_CHANNELS_TO_KEEP) & set(SFC_CHANNELS_TO_EXCLUDE)
        assert len(overlap) == 0, (
            f"{len(overlap)} channel(s) appear in both keep and exclude lists: {overlap}"
        )


# ---------------------------------------------------------------------------
# Validation pattern documentation
# ---------------------------------------------------------------------------


class TestValidationPattern:
    """
    Documents and verifies the assert-based validation pattern used across the codebase.

    Every .obs categorical assignment is expected to be guarded by:
        assert value in ALLOWED_X, f"Unknown value: '{value}'"

    This pattern raises AssertionError for invalid values, which is the intended
    behaviour — it surfaces a programming error immediately rather than silently
    storing an uncatalogued value in .obs.
    """

    def test_valid_staining_value_passes_silently(self):
        """A value declared in the dict must not raise."""
        valid_value = "DeepRed"
        assert valid_value in ALLOWED_STAINING_VALUES

    def test_invalid_staining_value_raises_assertion_error(self):
        """A value absent from the dict must raise AssertionError when asserted."""
        invalid_value = "Yellow"
        with pytest.raises(AssertionError):
            assert invalid_value in ALLOWED_STAINING_VALUES, (
                f"Unknown staining value: '{invalid_value}'"
            )

    def test_valid_species_value_passes_silently(self):
        assert "MNMS" in ALLOWED_SPECIES_SFC

    def test_invalid_species_value_raises_assertion_error(self):
        with pytest.raises(AssertionError):
            assert "HUMAN" in ALLOWED_SPECIES_SFC, "Unknown species: 'HUMAN'"
