"""
tests/test_sfc_clinical_data.py

Unit tests for src/mito_marker/sfc/clinical_data.py.

Test strategy:
- All tests use in-memory AnnData fixtures and in-memory CSV strings.
  No files are read from disk; no .fcs files are needed.
- Fixtures cover: 3 subjects in CSV, 10 events in AnnData (4 from subject 001,
  4 from subject 002, 2 from subject 999 which is NOT in the CSV).
"""

import io
import math

import anndata
import numpy as np
import pandas as pd
import pytest

from mito_marker.sfc.clinical_data import (
    _parse_subject_id_from_csv_index,
    _sanitize_column_name,
    enrich_with_clinical_data,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

# The last column header contains an embedded newline (as Excel wraps column names).
# It must be quoted in the CSV so pandas reads it as a single column name.
# Using explicit string concatenation to make the embedded \n visible in code.
CLINICAL_CSV_CONTENT = (
    'MNMS,Age,"BMI (kg/m2)",Cholestérol total,"Glucose\n*aléatoire*"\n'
    "MNMS 001,25,22.5,4.8,5.1\n"
    "MNMS 002,40,28.1,-1.0,6.2\n"
    "MNMS 003,55,31.0,5.5,-1.0\n"
    "invalid_row,99,99.0,99.0,99.0\n"
)


def _make_anndata_fixture() -> anndata.AnnData:
    """Build a minimal AnnData with 10 events: 4 from 001, 4 from 002, 2 from 999."""
    n_events = 10
    n_channels = 3
    event_matrix = np.ones((n_events, n_channels), dtype=np.float32)
    channel_names = ["R1-A", "R2-A", "R3-A"]

    subject_ids = ["001"] * 4 + ["002"] * 4 + ["999"] * 2
    obs_dataframe = pd.DataFrame(
        {"subject_ID": subject_ids},
        index=[f"event_{i:03d}" for i in range(n_events)],
    )
    var_dataframe = pd.DataFrame(index=channel_names)

    return anndata.AnnData(X=event_matrix, obs=obs_dataframe, var=var_dataframe)


def _load_clinical_csv_from_string(csv_content: str) -> str:
    """Write the CSV content to a temporary file path for testing.

    We use io.StringIO internally in the helper but return a real temp file
    path so enrich_with_clinical_data() can use pd.read_csv(path, ...).
    """
    # Tests use tmp_path for disk-backed CSV files.
    raise NotImplementedError("Use the tmp_path fixture directly in each test.")


# ---------------------------------------------------------------------------
# Tests for _sanitize_column_name
# ---------------------------------------------------------------------------

class TestSanitizeColumnName:
    def test_spaces_replaced_with_underscores(self):
        assert _sanitize_column_name("BMI (kg/m2)") == "bmi_kg_m2"

    def test_french_accent_removed(self):
        assert _sanitize_column_name("Cholestérol total") == "cholesterol_total"

    def test_newline_removed(self):
        result = _sanitize_column_name("Glucose\n*aléatoire*")
        assert "\n" not in result
        assert " " not in result
        assert result == "glucose_aleatoire"

    def test_parentheses_and_slash_removed(self):
        assert _sanitize_column_name("Physical activity_Total (min/week)") == "physical_activity_total_min_week"

    def test_result_is_lowercase(self):
        result = _sanitize_column_name("AGE")
        assert result == result.lower()

    def test_no_leading_trailing_underscores(self):
        result = _sanitize_column_name("  Age  ")
        assert not result.startswith("_")
        assert not result.endswith("_")

    def test_consecutive_underscores_collapsed(self):
        result = _sanitize_column_name("A  B")
        assert "__" not in result

    def test_plain_ascii_unchanged_except_case(self):
        assert _sanitize_column_name("Age") == "age"

    def test_percent_sign_replaced(self):
        result = _sanitize_column_name("Fat mass (%)")
        assert "%" not in result

    def test_plus_sign_replaced(self):
        result = _sanitize_column_name("GLU+MAL")
        assert "+" not in result
        assert result == "glu_mal"


# ---------------------------------------------------------------------------
# Tests for _parse_subject_id_from_csv_index
# ---------------------------------------------------------------------------

class TestParseSubjectIdFromCsvIndex:
    def test_mnms_space_3digits(self):
        assert _parse_subject_id_from_csv_index("MNMS 005") == "005"

    def test_mnms_space_2digits_zero_padded(self):
        assert _parse_subject_id_from_csv_index("MNMS 67") == "067"

    def test_mnms_no_space(self):
        assert _parse_subject_id_from_csv_index("MNMS005") == "005"

    def test_leading_whitespace_stripped(self):
        assert _parse_subject_id_from_csv_index("  MNMS 010  ") == "010"

    def test_no_digits_returns_none(self):
        assert _parse_subject_id_from_csv_index("invalid_row") is None

    def test_empty_string_returns_none(self):
        assert _parse_subject_id_from_csv_index("") is None

    def test_three_digit_preserved(self):
        assert _parse_subject_id_from_csv_index("MNMS 123") == "123"


# ---------------------------------------------------------------------------
# Tests for enrich_with_clinical_data
# ---------------------------------------------------------------------------

@pytest.fixture()
def clinical_csv_file(tmp_path):
    """Write the test CSV content to a temporary file and return its path."""
    csv_path = tmp_path / "clinical.csv"
    csv_path.write_text(CLINICAL_CSV_CONTENT, encoding="utf-8")
    return str(csv_path)


@pytest.fixture()
def base_anndata():
    """Return a fresh AnnData fixture for each test."""
    return _make_anndata_fixture()


class TestEnrichWithClinicalData:

    def test_returns_same_anndata_object(self, base_anndata, clinical_csv_file):
        """enrich_with_clinical_data must return the same object it received."""
        returned = enrich_with_clinical_data(base_anndata, clinical_csv_file)
        assert returned is base_anndata

    def test_all_clinical_columns_added_to_obs(self, base_anndata, clinical_csv_file):
        """All 4 clinical columns from the CSV must appear in .obs after enrichment."""
        enrich_with_clinical_data(base_anndata, clinical_csv_file)
        # Original .obs has 1 column (subject_ID) + 4 clinical columns = 5 total
        assert len(base_anndata.obs.columns) == 5

    def test_matched_subject_gets_correct_age(self, base_anndata, clinical_csv_file):
        """Events from subject 001 must carry age = 25."""
        enrich_with_clinical_data(base_anndata, clinical_csv_file)
        events_from_001 = base_anndata.obs[base_anndata.obs["subject_ID"] == "001"]
        assert (events_from_001["age"] == 25.0).all()

    def test_matched_subject_gets_correct_bmi(self, base_anndata, clinical_csv_file):
        """Events from subject 002 must carry BMI = 28.1."""
        enrich_with_clinical_data(base_anndata, clinical_csv_file)
        events_from_002 = base_anndata.obs[base_anndata.obs["subject_ID"] == "002"]
        assert all(abs(events_from_002["bmi_kg_m2"] - 28.1) < 1e-4)

    def test_unmatched_subject_gets_nan(self, base_anndata, clinical_csv_file):
        """Events from subject 999 (not in CSV) must have NaN for all clinical columns."""
        enrich_with_clinical_data(base_anndata, clinical_csv_file)
        events_from_999 = base_anndata.obs[base_anndata.obs["subject_ID"] == "999"]
        assert events_from_999["age"].isna().all()
        assert events_from_999["bmi_kg_m2"].isna().all()

    def test_missing_value_code_converted_to_nan(self, base_anndata, clinical_csv_file):
        """The -1.0 sentinel in the CSV (subject 002's Cholesterol) must become NaN."""
        enrich_with_clinical_data(base_anndata, clinical_csv_file)
        events_from_002 = base_anndata.obs[base_anndata.obs["subject_ID"] == "002"]
        # MNMS 002 has -1.0 for Cholestérol → must be NaN
        assert events_from_002["cholesterol_total"].isna().all()

    def test_float32_dtype_for_numeric_clinical_columns(self, base_anndata, clinical_csv_file):
        """Numeric clinical columns must be stored as float32; string columns as object."""
        enrich_with_clinical_data(base_anndata, clinical_csv_file)
        # Age, BMI and the Glucose column are numeric → float32
        assert base_anndata.obs["age"].dtype == np.float32
        assert base_anndata.obs["bmi_kg_m2"].dtype == np.float32
        assert base_anndata.obs["glucose_aleatoire"].dtype == np.float32

    def test_string_column_stored_as_categorical(self, base_anndata, tmp_path):
        """A CSV column containing strings must be stored as pd.Categorical in .obs."""
        csv_content = (
            "MNMS,Category\n"
            "MNMS 001,active\n"
            "MNMS 002,inactive\n"
        )
        csv_path = tmp_path / "with_string.csv"
        csv_path.write_text(csv_content, encoding="utf-8")

        enrich_with_clinical_data(base_anndata, str(csv_path))

        assert hasattr(base_anndata.obs["category"], "cat"), (
            "Expected pd.Categorical dtype for string column"
        )
        # The categories must include all unique values from the CSV plus empty string
        assert set(base_anndata.obs["category"].cat.categories) == {"", "active", "inactive"}

    def test_unmatched_subject_string_column_gets_empty_string(self, base_anndata, tmp_path):
        """Events from an unmatched subject must get '' (not 'nan') in string columns."""
        csv_content = (
            "MNMS,Category\n"
            "MNMS 001,active\n"
            "MNMS 002,inactive\n"
        )
        csv_path = tmp_path / "with_string.csv"
        csv_path.write_text(csv_content, encoding="utf-8")

        enrich_with_clinical_data(base_anndata, str(csv_path))

        events_from_999 = base_anndata.obs[base_anndata.obs["subject_ID"] == "999"]
        assert (events_from_999["category"] == "").all()

    def test_column_names_sanitized_no_spaces(self, base_anndata, clinical_csv_file):
        """No .obs column name should contain spaces after enrichment."""
        enrich_with_clinical_data(base_anndata, clinical_csv_file)
        for column_name in base_anndata.obs.columns:
            assert " " not in column_name, f"Column '{column_name}' contains a space"

    def test_column_names_sanitized_no_newlines(self, base_anndata, clinical_csv_file):
        """No .obs column name should contain newline characters."""
        enrich_with_clinical_data(base_anndata, clinical_csv_file)
        for column_name in base_anndata.obs.columns:
            assert "\n" not in column_name, f"Column '{column_name}' contains a newline"

    def test_column_names_sanitized_no_special_chars(self, base_anndata, clinical_csv_file):
        """No .obs column name should contain parentheses, slashes, or percent signs."""
        enrich_with_clinical_data(base_anndata, clinical_csv_file)
        for column_name in base_anndata.obs.columns:
            for forbidden_char in ["(", ")", "/", "%", "+", "*"]:
                assert forbidden_char not in column_name, (
                    f"Column '{column_name}' contains '{forbidden_char}'"
                )

    def test_original_names_stored_in_uns(self, base_anndata, clinical_csv_file):
        """The mapping from sanitized → original column names must be in .uns."""
        enrich_with_clinical_data(base_anndata, clinical_csv_file)
        assert "clinical_column_name_mapping" in base_anndata.uns
        mapping = base_anndata.uns["clinical_column_name_mapping"]
        assert isinstance(mapping, dict)
        # The 4 original CSV columns must appear as values in the mapping
        assert len(mapping) == 4

    def test_invalid_csv_index_rows_skipped(self, base_anndata, clinical_csv_file):
        """Rows with an unparseable index (like 'invalid_row') must be silently skipped."""
        enrich_with_clinical_data(base_anndata, clinical_csv_file)
        # There are 4 rows in the CSV: 3 valid (MNMS 001/002/003) + 1 invalid.
        # Only subjects 001 and 002 appear in AnnData — but no error should be raised.
        events_from_001 = base_anndata.obs[base_anndata.obs["subject_ID"] == "001"]
        assert not events_from_001["age"].isna().any()

    def test_custom_missing_value_code(self, base_anndata, tmp_path):
        """A custom missing_value_code (0.0) must also be converted to NaN."""
        csv_content = "MNMS,Age\nMNMS 001,25\nMNMS 002,0.0\n"
        csv_path = tmp_path / "custom_missing.csv"
        csv_path.write_text(csv_content, encoding="utf-8")

        enrich_with_clinical_data(base_anndata, str(csv_path), missing_value_code=0.0)

        events_from_002 = base_anndata.obs[base_anndata.obs["subject_ID"] == "002"]
        assert events_from_002["age"].isna().all()

    def test_duplicate_sanitized_column_names_deduplicated(self, base_anndata, tmp_path):
        """Two CSV columns that sanitize to the same name must get distinct .obs keys."""
        # "Glucose total" and "Glucose-total" both sanitize to "glucose_total".
        # The second occurrence must be renamed to "glucose_total_2".
        csv_content = (
            "MNMS,Glucose total,Glucose-total\n"
            "MNMS 001,5.1,5.3\n"
            "MNMS 002,6.2,6.4\n"
        )
        csv_path = tmp_path / "dupes.csv"
        csv_path.write_text(csv_content, encoding="utf-8")

        enrich_with_clinical_data(base_anndata, str(csv_path))

        # Both columns must be present — neither overwrites the other
        assert "glucose_total" in base_anndata.obs.columns
        assert "glucose_total_2" in base_anndata.obs.columns

    def test_qc_block_prints_to_stdout(self, base_anndata, clinical_csv_file, capsys):
        """The function must produce console output (QC block)."""
        enrich_with_clinical_data(base_anndata, clinical_csv_file)
        captured = capsys.readouterr()
        assert len(captured.out) > 0
        assert "Clinical data enrichment" in captured.out

    def test_french_accent_column_name_in_obs(self, base_anndata, clinical_csv_file):
        """'Cholestérol total' must appear as 'cholesterol_total' in .obs."""
        enrich_with_clinical_data(base_anndata, clinical_csv_file)
        assert "cholesterol_total" in base_anndata.obs.columns

    def test_newline_in_column_name_sanitized(self, base_anndata, clinical_csv_file):
        """'Glucose\\n*aléatoire*' must be sanitized to a clean key in .obs."""
        enrich_with_clinical_data(base_anndata, clinical_csv_file)
        # Check that no column name contains a literal newline
        for column_name in base_anndata.obs.columns:
            assert "\n" not in column_name
        # The sanitized name should be present
        assert "glucose_aleatoire" in base_anndata.obs.columns

    def test_existing_obs_column_replaced_with_warning(self, base_anndata, tmp_path, capsys):
        """When a CSV column sanitizes to a name already in .obs, the old column is
        replaced by the CSV value and a WARNING is printed — no duplicate columns."""
        # Pre-populate .obs with an 'age' column (as if parsed from the filename)
        base_anndata.obs["age"] = 99.0

        csv_content = "MNMS,Age\nMNMS 001,25\nMNMS 002,40\n"
        csv_path = tmp_path / "age_collision.csv"
        csv_path.write_text(csv_content, encoding="utf-8")

        enrich_with_clinical_data(base_anndata, str(csv_path))

        captured = capsys.readouterr()

        # Exactly one 'age' column must exist — no duplicates
        assert list(base_anndata.obs.columns).count("age") == 1

        # The value must come from the CSV (25.0 for subject 001), not the old sentinel 99.0
        events_from_001 = base_anndata.obs[base_anndata.obs["subject_ID"] == "001"]
        assert (events_from_001["age"] == 25.0).all()

        # A WARNING must have been printed to stdout
        assert "WARNING" in captured.out
        assert "age" in captured.out
