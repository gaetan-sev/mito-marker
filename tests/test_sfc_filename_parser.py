"""
test_sfc_filename_parser.py

Unit tests for src/mito_marker/sfc/filename_parser.py.

Tests cover:
- Both fixture filenames produce correct obs metadata
- Required field detection (specie, subject_ID, marker)
- Optional field defaults (age=None, diet=None, condition=None, dilution='Diluted')
- Token mapping: 'DR' and 'DeepRed' both resolve to 'MtDeepRed'
- FlowAI_Pass detection from multiple token variants
- ValueError raised when required fields are missing
- Age and diet parsed correctly when present
- Condition: single token, combined tokens, canonical ordering, case variants
- Mouse species token recognised
"""

import pytest

from mito_marker.sfc.filename_parser import parse_fcs_filename


# ---------------------------------------------------------------------------
# Fixture filenames — expected full output
# ---------------------------------------------------------------------------


class TestFixtureFilename020:
    """Tests against the first SFC fixture file."""

    FILENAME = "good_events_MNMS_020_FlowAIGoodEvents_DeepRed.fcs"

    def test_specie(self):
        result = parse_fcs_filename(self.FILENAME)
        assert result["specie"] == "MNMS"

    def test_subject_id(self):
        result = parse_fcs_filename(self.FILENAME)
        assert result["subject_ID"] == "020"

    def test_age_is_none(self):
        result = parse_fcs_filename(self.FILENAME)
        assert result["age"] is None

    def test_diet_is_none(self):
        result = parse_fcs_filename(self.FILENAME)
        assert result["diet"] is None

    def test_dilution_default(self):
        result = parse_fcs_filename(self.FILENAME)
        assert result["dilution"] == "Diluted"

    def test_marker(self):
        result = parse_fcs_filename(self.FILENAME)
        assert result["marker"] == "MtDeepRed"

    def test_flowai_pass_true(self):
        result = parse_fcs_filename(self.FILENAME)
        assert result["FlowAI_Pass"] is True

    def test_source_filename(self):
        result = parse_fcs_filename(self.FILENAME)
        assert result["source_filename"] == self.FILENAME


class TestFixtureFilename031:
    """Tests against the second SFC fixture file (includes ND token)."""

    FILENAME = "good_events_MNMS_031_ND_FlowAIGoodEvents_DeepRed.fcs"

    def test_subject_id(self):
        result = parse_fcs_filename(self.FILENAME)
        assert result["subject_ID"] == "031"

    def test_dilution_not_diluted(self):
        """ND token must produce 'Not_Diluted'."""
        result = parse_fcs_filename(self.FILENAME)
        assert result["dilution"] == "Not_Diluted"


# ---------------------------------------------------------------------------
# Marker token mapping: both 'DR' and 'DeepRed' → 'MtDeepRed'
# ---------------------------------------------------------------------------


class TestMarkerTokenMapping:
    def test_deepred_token(self):
        result = parse_fcs_filename("MNMS_020_FlowAIGoodEvents_DeepRed.fcs")
        assert result["marker"] == "MtDeepRed"

    def test_dr_token(self):
        """'DR' abbreviation must resolve to the same canonical value as 'DeepRed'."""
        result = parse_fcs_filename("MNMS_020_FlowAIGoodEvents_DR.fcs")
        assert result["marker"] == "MtDeepRed"

    def test_tmrm_single(self):
        result = parse_fcs_filename("MOUSE_001_Young_TMRM.fcs")
        assert result["marker"] == "TMRM"

    def test_tmrm_and_deepred_combined(self):
        """Both TMRM and DeepRed tokens must produce the joined canonical value."""
        result = parse_fcs_filename("MOUSE_001_Young_Pilot_TMRM_DeepRed.fcs")
        assert result["marker"] == "TMRM+MtDeepRed"

    def test_combined_marker_order_is_canonical(self):
        """Token order in filename must not affect join order."""
        result_normal = parse_fcs_filename("MOUSE_001_Young_TMRM_DeepRed.fcs")
        result_reversed = parse_fcs_filename("MOUSE_001_Young_DeepRed_TMRM.fcs")
        assert result_normal["marker"] == result_reversed["marker"] == "TMRM+MtDeepRed"

    def test_duplicate_marker_token_not_doubled(self):
        """Same marker token appearing twice must produce a single canonical value."""
        result = parse_fcs_filename("MOUSE_001_Young_DeepRed_DeepRed.fcs")
        assert result["marker"] == "MtDeepRed"


# ---------------------------------------------------------------------------
# FlowAI_Pass detection
# ---------------------------------------------------------------------------


class TestFlowAIPassDetection:
    def test_flowai_good_events_token(self):
        result = parse_fcs_filename("MNMS_020_FlowAIGoodEvents_DeepRed.fcs")
        assert result["FlowAI_Pass"] is True

    def test_good_and_events_tokens(self):
        result = parse_fcs_filename("good_events_MNMS_020_DeepRed.fcs")
        assert result["FlowAI_Pass"] is True

    def test_no_flowai_tokens(self):
        """File with no FlowAI tokens → FlowAI_Pass must be False."""
        result = parse_fcs_filename("MNMS_020_DeepRed.fcs")
        assert result["FlowAI_Pass"] is False


# ---------------------------------------------------------------------------
# Optional fields: age and diet
# ---------------------------------------------------------------------------


class TestOptionalFields:
    def test_age_day(self):
        # _extract_age returns the numeric part as int: 'D03' → 3
        result = parse_fcs_filename("MNMS_020_D03_FlowAIGoodEvents_DeepRed.fcs")
        assert result["age"] == 3

    def test_age_month(self):
        # 'M06' → 6
        result = parse_fcs_filename("MNMS_020_M06_FlowAIGoodEvents_DeepRed.fcs")
        assert result["age"] == 6

    def test_age_year(self):
        # 'Y02' → 2
        result = parse_fcs_filename("MNMS_020_Y02_FlowAIGoodEvents_DeepRed.fcs")
        assert result["age"] == 2

    def test_diet_al(self):
        result = parse_fcs_filename("MNMS_020_AL_FlowAIGoodEvents_DeepRed.fcs")
        assert result["diet"] == "AL"

    def test_diet_if(self):
        result = parse_fcs_filename("MNMS_020_IF_FlowAIGoodEvents_DeepRed.fcs")
        assert result["diet"] == "IF"

    def test_all_optional_fields_together(self):
        result = parse_fcs_filename("MNMS_020_D03_AL_FlowAIGoodEvents_DeepRed.fcs")
        assert result["age"] == 3
        assert result["diet"] == "AL"

    def test_age_absent_default(self):
        result = parse_fcs_filename("MNMS_020_FlowAIGoodEvents_DeepRed.fcs")
        assert result["age"] is None

    def test_diet_absent_default(self):
        result = parse_fcs_filename("MNMS_020_FlowAIGoodEvents_DeepRed.fcs")
        assert result["diet"] is None

    def test_age_group_absent_default(self):
        result = parse_fcs_filename("MNMS_020_FlowAIGoodEvents_DeepRed.fcs")
        assert result["age_group"] is None

    def test_treatment_absent_default(self):
        result = parse_fcs_filename("MNMS_020_FlowAIGoodEvents_DeepRed.fcs")
        assert result["treatment"] is None


# ---------------------------------------------------------------------------
# Species variants
# ---------------------------------------------------------------------------


class TestAgeGroupExtraction:
    """Tests for the age_group field."""

    def test_young_token(self):
        result = parse_fcs_filename("MOUSE_001_Young_DeepRed.fcs")
        assert result["age_group"] == "Young"

    def test_old_token(self):
        result = parse_fcs_filename("MOUSE_001_Old_DeepRed.fcs")
        assert result["age_group"] == "Old"

    def test_lowercase_young(self):
        result = parse_fcs_filename("MOUSE_001_young_DeepRed.fcs")
        assert result["age_group"] == "Young"

    def test_uppercase_old(self):
        result = parse_fcs_filename("MOUSE_001_OLD_DeepRed.fcs")
        assert result["age_group"] == "Old"

    def test_age_group_absent_returns_none(self):
        result = parse_fcs_filename("MOUSE_001_Vehicule_DeepRed.fcs")
        assert result["age_group"] is None

    def test_age_group_absent_for_human_cohort(self):
        result = parse_fcs_filename("MNMS_020_FlowAIGoodEvents_DeepRed.fcs")
        assert result["age_group"] is None


class TestTreatmentExtraction:
    """Tests for the treatment field."""

    def test_vehicule_token(self):
        result = parse_fcs_filename("MOUSE_001_Vehicule_DeepRed.fcs")
        assert result["treatment"] == "Vehicule"

    def test_arac15_token(self):
        result = parse_fcs_filename("MOUSE_001_Arac15_DeepRed.fcs")
        assert result["treatment"] == "Arac15"

    def test_arac30_token(self):
        result = parse_fcs_filename("MOUSE_001_Arac30_DeepRed.fcs")
        assert result["treatment"] == "Arac30"

    def test_lowercase_arac30(self):
        result = parse_fcs_filename("MOUSE_001_arac30_DeepRed.fcs")
        assert result["treatment"] == "Arac30"

    def test_uppercase_vehicule(self):
        result = parse_fcs_filename("MOUSE_001_VEHICULE_DeepRed.fcs")
        assert result["treatment"] == "Vehicule"

    def test_treatment_absent_returns_none(self):
        result = parse_fcs_filename("MOUSE_001_Young_DeepRed.fcs")
        assert result["treatment"] is None

    def test_treatment_absent_for_human_cohort(self):
        result = parse_fcs_filename("MNMS_020_FlowAIGoodEvents_DeepRed.fcs")
        assert result["treatment"] is None


class TestAgeGroupAndTreatmentCombined:
    """Tests for the combination of age_group + treatment — the two fields are independent."""

    def test_young_vehicule(self):
        result = parse_fcs_filename("MOUSE_001_Young_Vehicule_DeepRed.fcs")
        assert result["age_group"] == "Young"
        assert result["treatment"] == "Vehicule"

    def test_old_arac30(self):
        result = parse_fcs_filename("MOUSE_002_Old_Arac30_DeepRed.fcs")
        assert result["age_group"] == "Old"
        assert result["treatment"] == "Arac30"

    def test_young_arac15(self):
        result = parse_fcs_filename("MOUSE_003_Young_Arac15_DeepRed.fcs")
        assert result["age_group"] == "Young"
        assert result["treatment"] == "Arac15"

    def test_token_order_does_not_matter(self):
        """age_group and treatment are parsed independently — order is irrelevant."""
        result_normal = parse_fcs_filename("MOUSE_001_Young_Vehicule_DeepRed.fcs")
        result_reversed = parse_fcs_filename("MOUSE_001_Vehicule_Young_DeepRed.fcs")
        assert result_normal["age_group"] == result_reversed["age_group"] == "Young"
        assert result_normal["treatment"] == result_reversed["treatment"] == "Vehicule"


class TestMouseSpecies:
    """Tests for the Mouse species token in SFC filenames."""

    def test_mouse_specie_recognised(self):
        result = parse_fcs_filename("MOUSE_001_Young_DeepRed.fcs")
        assert result["specie"] == "MOUSE"

    def test_mouse_full_filename(self):
        result = parse_fcs_filename("MOUSE_042_Old_Arac30_FlowAIGoodEvents_DeepRed.fcs")
        assert result["specie"] == "MOUSE"
        assert result["subject_ID"] == "042"
        assert result["age_group"] == "Old"
        assert result["treatment"] == "Arac30"
        assert result["marker"] == "MtDeepRed"
        assert result["FlowAI_Pass"] is True


class TestSpeciesVariants:
    def test_worms(self):
        result = parse_fcs_filename("WORMS_020_FlowAIGoodEvents_DeepRed.fcs")
        assert result["specie"] == "WORMS"

    def test_fly(self):
        result = parse_fcs_filename("FLY_020_FlowAIGoodEvents_DeepRed.fcs")
        assert result["specie"] == "FLY"


# ---------------------------------------------------------------------------
# Missing required fields → ValueError
# ---------------------------------------------------------------------------


class TestMissingRequiredFields:
    def test_missing_specie_raises(self):
        with pytest.raises(ValueError, match="specie"):
            parse_fcs_filename("unknown_020_FlowAIGoodEvents_DeepRed.fcs")

    def test_missing_subject_id_raises(self):
        with pytest.raises(ValueError, match="subject ID"):
            parse_fcs_filename("MNMS_FlowAIGoodEvents_DeepRed.fcs")

    def test_missing_marker_raises(self):
        with pytest.raises(ValueError, match="marker"):
            parse_fcs_filename("MNMS_020_FlowAIGoodEvents.fcs")


# ---------------------------------------------------------------------------
# Filename input format robustness
# ---------------------------------------------------------------------------


class TestFilenameInputFormats:
    def test_with_fcs_extension(self):
        """Explicit .fcs extension must not break parsing."""
        result = parse_fcs_filename("MNMS_020_FlowAIGoodEvents_DeepRed.fcs")
        assert result["specie"] == "MNMS"
        assert result["source_filename"] == "MNMS_020_FlowAIGoodEvents_DeepRed.fcs"

    def test_without_fcs_extension(self):
        """Filename without extension must also parse correctly."""
        result = parse_fcs_filename("MNMS_020_FlowAIGoodEvents_DeepRed")
        assert result["specie"] == "MNMS"

    def test_with_directory_prefix_stripped(self):
        """Directory path prefix must be stripped; only basename matters."""
        result = parse_fcs_filename("/some/path/to/MNMS_020_FlowAIGoodEvents_DeepRed.fcs")
        assert result["specie"] == "MNMS"
        assert result["source_filename"] == "MNMS_020_FlowAIGoodEvents_DeepRed.fcs"
