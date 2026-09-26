"""
Tests for tem_filename_parser.py

Covers all naming conventions encountered in real experiments:
  - KFish (with {n}fish{ID} separator)
  - killi abbreviation
  - Zebrafish and ZFish (no separator)
  - Souris (French for mouse, no condition token)
  - M2124-style (no recognisable species or condition token)
  - Warning emission for unknown tokens
"""

import warnings

import pytest

from mito_marker.tem.tem_filename_parser import parse_tem_filename


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse(filename: str) -> dict:
    """Call parse_tem_filename, capturing any warnings."""
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        return parse_tem_filename(filename)


# ---------------------------------------------------------------------------
# KFish filenames — {n}fish{ID} separator convention
# ---------------------------------------------------------------------------


class TestKFishFilenames:
    def test_kfish_young_1fish565(self):
        result = _parse("KFishYoung1fish565x1200grille2Dfield2_MITO_measurements.txt")
        assert result["specie"] == "KFish"
        assert result["condition"] == "Young"
        assert result["subject_ID"] == "565"

    def test_kfish_young_3fish567(self):
        result = _parse("KFishYoung3fish567x1200grille3Cfield6_MITO_measurements.txt")
        assert result["specie"] == "KFish"
        assert result["condition"] == "Young"
        assert result["subject_ID"] == "567"

    def test_killi_old_1569(self):
        """'killi' is a known alias for KFish."""
        result = _parse("Oldkilli1569grille4bx1200field7_MITO_measurements.txt")
        assert result["specie"] == "KFish"
        assert result["condition"] == "Old"
        assert result["subject_ID"] == "1569"

    def test_killi_old_4572(self):
        result = _parse("Oldkilli4572grille5Cx1200field24_MITO_measurements.txt")
        assert result["specie"] == "KFish"
        assert result["condition"] == "Old"
        assert result["subject_ID"] == "4572"

    def test_killi_young_4568(self):
        result = _parse("Youngkilli4568grille3Ex1200field16_MITO_measurements.txt")
        assert result["specie"] == "KFish"
        assert result["condition"] == "Young"
        assert result["subject_ID"] == "4568"


# ---------------------------------------------------------------------------
# ZFish / Zebrafish filenames — no fish separator
# ---------------------------------------------------------------------------


class TestZFishFilenames:
    def test_zebrafish_old_42518(self):
        result = _parse("OldZebrafish42518grille4Bx1200field4_MITO_measurements.txt")
        assert result["specie"] == "ZFish"
        assert result["condition"] == "Old"
        assert result["subject_ID"] == "42518"

    def test_zebrafish_young_22511(self):
        result = _parse(
            "YoungZebrafish22511grille1Dx1200field17_MITO_measurements.txt"
        )
        assert result["specie"] == "ZFish"
        assert result["condition"] == "Young"
        assert result["subject_ID"] == "22511"

    def test_zfish_young_12510(self):
        """ZFish abbreviation with no fish separator: '12510' is the full subject ID."""
        result = _parse("YoungZFish12510grille1Bx1200field1_MITO_measurements.txt")
        assert result["specie"] == "ZFish"
        assert result["condition"] == "Young"
        assert result["subject_ID"] == "12510"


# ---------------------------------------------------------------------------
# Souris / Mouse fixtures — no condition token in filename
# ---------------------------------------------------------------------------


class TestSourisFixtures:
    def test_souris_j3_specie(self):
        result = _parse(
            "SourisJ3_Prot1533_Ech1755_Grille13E_x1500_field_19_MITO_measurements.txt"
        )
        assert result["specie"] == "Mouse"

    def test_souris_j3_no_condition_warns(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = parse_tem_filename(
                "SourisJ3_Prot1533_Ech1755_Grille13E_x1500_field_19_MITO_measurements.txt"
            )
        assert result["condition"] is None
        warning_messages = [str(w.message) for w in caught]
        assert any("condition" in msg.lower() for msg in warning_messages)

    def test_souris_j3_subject_id(self):
        """First 3+ digit sequence after removing 'Souris' from prefix is '1533'."""
        result = _parse(
            "SourisJ3_Prot1533_Ech1755_Grille13E_x1500_field_19_MITO_measurements.txt"
        )
        assert result["subject_ID"] == "1533"

    def test_souris_v1_specie_and_subject(self):
        result = _parse(
            "SourisV1_Prot1533_Ech1757_Grille14B_x1500_field_31_MITO_measurements.txt"
        )
        assert result["specie"] == "Mouse"
        assert result["subject_ID"] == "1533"


# ---------------------------------------------------------------------------
# Human / MNMS filenames — species from MNMS token, condition from file content
# ---------------------------------------------------------------------------


class TestHumanMNMSFilenames:
    def test_mnms_specie(self):
        result = _parse("x1200MNMS051field26_MITO_measurements.txt")
        assert result["specie"] == "Human"

    def test_mnms_subject_id_3digits(self):
        """Three-digit MNMS subject IDs are preserved with leading zeros."""
        assert _parse("x1200MNMS051field26_MITO_measurements.txt")["subject_ID"] == "051"
        assert _parse("x1200MNMS016field2_MITO_measurements.txt")["subject_ID"] == "016"
        assert _parse("x1200MNMS039field6_MITO_measurements.txt")["subject_ID"] == "039"

    def test_mnms_three_digit_variants(self):
        """Edge cases: leading zeros and boundary values."""
        assert _parse("x1200MNMS008field26_MITO_measurements.txt")["subject_ID"] == "008"
        assert _parse("x1200MNMS162field26_MITO_measurements.txt")["subject_ID"] == "162"

    def test_mnms_condition_is_none_without_warning(self):
        """Human files have no condition token — condition is None and NO warning fires."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = parse_tem_filename("x1200MNMS051field26_MITO_measurements.txt")
        assert result["condition"] is None
        # No "condition" warning should be emitted for Human files.
        warning_messages = [str(w.message) for w in caught]
        assert not any("condition token" in msg.lower() for msg in warning_messages)


# ---------------------------------------------------------------------------
# Mouse new-style filenames — lowercase 'mouse', single-digit subject_ID before grille
# ---------------------------------------------------------------------------


class TestMouseNewStyleFilenames:
    def test_mouseold_specie(self):
        assert _parse("mouseold1grilleC5x1200field2_MITO_measurements.txt")["specie"] == "Mouse"

    def test_mouseyoung_specie(self):
        assert _parse("mouseyoung1grilleB4x1200field2_MITO_measurements.txt")["specie"] == "Mouse"

    def test_mouseold_age_group(self):
        assert _parse("mouseold1grilleC5x1200field2_MITO_measurements.txt")["condition"] == "Old"

    def test_mouseyoung_age_group(self):
        assert _parse("mouseyoung1grilleB4x1200field2_MITO_measurements.txt")["condition"] == "Young"

    def test_mouseOld_capital_O(self):
        """'Old' and 'old' tokens both map to 'Old'."""
        assert _parse("mouseOld3grilleD5Field1_MITO_measurements.txt")["condition"] == "Old"

    def test_subject_id_before_grille_single_digit(self):
        """Subject ID is the digit immediately before 'grille' even when it is 1 digit."""
        assert _parse("mouseold1grilleC5x1200field2_MITO_measurements.txt")["subject_ID"] == "1"
        assert _parse("mouseold3grilleD4x1200field27_MITO_measurements.txt")["subject_ID"] == "3"
        assert _parse("mouseyoung1grilleB4x1200field2_MITO_measurements.txt")["subject_ID"] == "1"
        assert _parse("mouseyoung3grilleC3x1200field33_MITO_measurements.txt")["subject_ID"] == "3"

    def test_mouseOld3_subject_id(self):
        assert _parse("mouseOld3grilleD5Field1_MITO_measurements.txt")["subject_ID"] == "3"

    def test_mouseyoung1_field20(self):
        result = _parse("mouseyoung1grilleB5Field20_MITO_measurements.txt")
        assert result["specie"] == "Mouse"
        assert result["condition"] == "Young"
        assert result["subject_ID"] == "1"


# ---------------------------------------------------------------------------
# M2124 fixture — no recognisable species or condition token
# ---------------------------------------------------------------------------


class TestM2124Fixture:
    def test_m2124_no_specie_warns(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = parse_tem_filename("M2124_J1_G5_0114_MITO_measurements.txt")
        assert result["specie"] is None
        warning_messages = [str(w.message) for w in caught]
        assert any("species" in msg.lower() or "specie" in msg.lower() for msg in warning_messages)

    def test_m2124_no_condition_warns(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = parse_tem_filename("M2124_J1_G5_0114_MITO_measurements.txt")
        assert result["condition"] is None
        warning_messages = [str(w.message) for w in caught]
        assert any("condition" in msg.lower() for msg in warning_messages)

    def test_m2124_subject_id_is_first_long_digit_sequence(self):
        """Digit sequences in stem: '2124' (4 digits), '1' (skip), '5' (skip), '0114' (4 digits).
        First sequence ≥ 3 digits = '2124'."""
        result = _parse("M2124_J1_G5_0114_MITO_measurements.txt")
        assert result["subject_ID"] == "2124"


# ---------------------------------------------------------------------------
# Extension handling
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Drosophila filenames
# ---------------------------------------------------------------------------


class TestDrosoFilenames:
    """Drosophila filenames: Droso_{SubjectID}_{Condition}_longitudinalfield{N}_MITO_measurements.txt"""

    def test_droso_young_specie(self):
        result = _parse("Droso_03_Young_longitudinalfield9_MITO_measurements.txt")
        assert result["specie"] == "Droso"

    def test_droso_young_condition(self):
        result = _parse("Droso_03_Young_longitudinalfield9_MITO_measurements.txt")
        assert result["condition"] == "Young"

    def test_droso_young_subject_id(self):
        result = _parse("Droso_03_Young_longitudinalfield9_MITO_measurements.txt")
        assert result["subject_ID"] == "03"

    def test_droso_old_condition(self):
        result = _parse("Droso_12_Old_longitudinalfield10_MITO_measurements.txt")
        assert result["condition"] == "Old"

    def test_droso_old_subject_id(self):
        result = _parse("Droso_12_Old_longitudinalfield10_MITO_measurements.txt")
        assert result["subject_ID"] == "12"

    def test_droso_without_extension(self):
        result = _parse("Droso_03_Young_longitudinalfield9_MITO_measurements")
        assert result["specie"] == "Droso"
        assert result["condition"] == "Young"
        assert result["subject_ID"] == "03"


# ---------------------------------------------------------------------------
# ZFish diet filenames — snake_case with IF / AL token
# ---------------------------------------------------------------------------


class TestZFishDietFilenames:
    """ZFish files with diet token (IF/AL) and no condition (Young/Old) token."""

    def test_zfish_if_specie(self):
        result = _parse("ZFish_2551_IF_grille11C_x1200_field3_MITO_measurements.txt")
        assert result["specie"] == "ZFish"

    def test_zfish_if_diet(self):
        result = _parse("ZFish_2551_IF_grille11C_x1200_field3_MITO_measurements.txt")
        assert result["diet"] == "IF"

    def test_zfish_if_subject_id(self):
        result = _parse("ZFish_2551_IF_grille11C_x1200_field3_MITO_measurements.txt")
        assert result["subject_ID"] == "2551"

    def test_zfish_if_condition_is_none(self):
        """No Young/Old token → condition must be None (warning expected)."""
        result = _parse("ZFish_2551_IF_grille11C_x1200_field3_MITO_measurements.txt")
        assert result["condition"] is None

    def test_zfish_al_diet(self):
        result = _parse("ZFish_2540_AL_grille7A_x1200_field12_MITO_measurements.txt")
        assert result["diet"] == "AL"

    def test_zfish_al_subject_id(self):
        result = _parse("ZFish_2540_AL_grille7A_x1200_field12_MITO_measurements.txt")
        assert result["subject_ID"] == "2540"

    def test_zfish_al_specie(self):
        result = _parse("ZFish_2540_AL_grille7A_x1200_field12_MITO_measurements.txt")
        assert result["specie"] == "ZFish"

    def test_diet_absent_returns_none(self):
        """Standard KFish file with no diet token → diet must be None."""
        result = _parse("KFishYoung1fish565x1200grille2Dfield2_MITO_measurements.txt")
        assert result["diet"] is None

    def test_diet_absent_emits_no_warning(self):
        """Absence of diet is normal — no warning should fire."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            parse_tem_filename("KFishYoung1fish565x1200grille2Dfield2_MITO_measurements.txt")
        diet_warnings = [w for w in caught if "diet" in str(w.message).lower()]
        assert len(diet_warnings) == 0


class TestExtensionHandling:
    def test_with_txt_extension(self):
        result = _parse("OldZebrafish42518grille4Bx1200field4_MITO_measurements.txt")
        assert result["specie"] == "ZFish"

    def test_without_txt_extension(self):
        """Parser should work even if .txt is already stripped."""
        result = _parse("OldZebrafish42518grille4Bx1200field4_MITO_measurements")
        assert result["specie"] == "ZFish"
        assert result["subject_ID"] == "42518"


# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------


class TestReturnType:
    def test_returns_dict_with_required_keys(self):
        result = _parse("KFishYoung1fish565x1200grille2Dfield2_MITO_measurements.txt")
        assert isinstance(result, dict)
        assert "specie" in result
        assert "condition" in result
        assert "diet" in result
        assert "subject_ID" in result

    def test_all_values_are_strings_or_none(self):
        result = _parse("KFishYoung1fish565x1200grille2Dfield2_MITO_measurements.txt")
        for value in result.values():
            assert value is None or isinstance(value, str)
