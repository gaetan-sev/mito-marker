"""
test_analysis_selection.py

Unit tests for mito_marker.analysis.selection.select_sfc_subset().

All tests use a small synthetic AnnData (120 events, 5 channels) with known
.obs metadata, and monkeypatch builtins.input to simulate user console input
without any real interaction.

Synthetic AnnData structure:
    - 3 subjects:  "S001", "S002", "S003"  (40 events each)
    - unique_subject_ID: "MNMS_S001", "MNMS_S002", "MNMS_S003" (40 each)
    - 2 dilutions: "Diluted", "Not_Diluted" (20 each per subject)
    - 2 diets:     "AL", "IF"               (10 each per dilution per subject)
    - 2 FlowAI:    True, False              (20 each per subject)
    - 2 ages:      3.0, 6.0                (20 each per subject)
    - 1 specie:    "MNMS" only             → auto-skipped (1 unique value)
    - 1 date:      "" (missing) only        → auto-skipped (1 unique value)

Columns that trigger prompts (in FILTERABLE_OBS_COLUMNS order):
    unique_subject_ID, dilution, diet, FlowAI_Pass, age
    (specie and fcs_acquisition_date are auto-skipped)
Total prompts for the full fixture: 5
"""

import numpy as np
import pandas as pd
import pytest

import anndata

from mito_marker.analysis.selection import (
    _filter_obs_by_values,
    _get_display_values,
    _get_subject_column,
    _prompt_subsampling_size,
    _subsample_obs_per_subject,
    select_sfc_subset,
)


# ---------------------------------------------------------------------------
# Fixture: synthetic AnnData
# ---------------------------------------------------------------------------


def _make_test_anndata() -> anndata.AnnData:
    """
    Build a small synthetic AnnData with known metadata for testing selection.

    Layout per subject (40 events each, 3 subjects = 120 total):
        Events  0-19: Diluted, AL (0-9) + IF (10-19), True,  age 3.0
        Events 20-39: Not_Diluted, AL (20-29) + IF (30-39), False, age 6.0
    """
    np.random.seed(42)
    n_channels = 5
    n_per_subject = 40

    subjects = ["S001"] * n_per_subject + ["S002"] * n_per_subject + ["S003"] * n_per_subject
    unique_subject_ids = (
        ["MNMS_S001"] * n_per_subject
        + ["MNMS_S002"] * n_per_subject
        + ["MNMS_S003"] * n_per_subject
    )
    dilutions = (["Diluted"] * 20 + ["Not_Diluted"] * 20) * 3
    diets = (["AL"] * 10 + ["IF"] * 10 + ["AL"] * 10 + ["IF"] * 10) * 3
    flowai_pass = ([True] * 20 + [False] * 20) * 3
    ages = ([3.0] * 20 + [6.0] * 20) * 3
    species = ["MNMS"] * (n_per_subject * 3)          # 1 unique value → auto-skipped
    acq_dates = [""] * (n_per_subject * 3)             # 1 unique value → auto-skipped

    obs = pd.DataFrame({
        "subject_ID": subjects,
        "unique_subject_ID": unique_subject_ids,
        "specie": species,
        "dilution": dilutions,
        "diet": diets,
        "FlowAI_Pass": np.array(flowai_pass, dtype=np.bool_),
        "age": np.array(ages, dtype=np.float32),
        "fcs_acquisition_date": acq_dates,
    })

    x_matrix = np.random.randn(len(subjects), n_channels).astype(np.float32)
    var = pd.DataFrame(
        {"channel_description": [f"desc_{i}" for i in range(n_channels)]},
        index=[f"CH{i}" for i in range(n_channels)],
    )

    return anndata.AnnData(X=x_matrix, obs=obs, var=var)


@pytest.fixture
def test_anndata() -> anndata.AnnData:
    """Synthetic AnnData fixture for all selection tests."""
    return _make_test_anndata()


def _make_input_sequence(responses: list) -> callable:
    """
    Return a mock for builtins.input that pops responses one at a time.

    If more input() calls happen than responses provided, raises StopIteration,
    which makes the test fail clearly rather than hanging.
    """
    iterator = iter(responses)

    def mock_input(prompt: str = "") -> str:
        return next(iterator)

    return mock_input


# ---------------------------------------------------------------------------
# Tests: _get_subject_column (unit tests — no console interaction)
# ---------------------------------------------------------------------------


class TestGetSubjectColumn:
    """Unit tests for the _get_subject_column helper."""

    def test_returns_unique_subject_id_when_present(self):
        obs = pd.DataFrame({
            "unique_subject_ID": ["A", "B"],
            "subject_ID": ["1", "2"],
        })
        assert _get_subject_column(obs) == "unique_subject_ID"

    def test_falls_back_to_subject_id_when_unique_absent(self):
        obs = pd.DataFrame({"subject_ID": ["1", "2"]})
        assert _get_subject_column(obs) == "subject_ID"

    def test_returns_none_when_neither_present(self):
        obs = pd.DataFrame({"dilution": ["Diluted", "Not_Diluted"]})
        assert _get_subject_column(obs) is None


# ---------------------------------------------------------------------------
# Tests: _get_display_values (unit tests — no console interaction)
# ---------------------------------------------------------------------------


class TestGetDisplayValues:
    """Unit tests for the _get_display_values helper."""

    def test_boolean_column_returns_true_before_false(self):
        series = pd.Series([True, True, False, True, False], dtype=np.bool_)
        result = _get_display_values(series)
        assert result[0] == (True, "True", 3)
        assert result[1] == (False, "False", 2)

    def test_boolean_column_excludes_absent_values(self):
        # All True → only one entry → caller auto-skips the column.
        series = pd.Series([True, True, True], dtype=np.bool_)
        result = _get_display_values(series)
        assert len(result) == 1
        assert result[0][0] is True

    def test_float_column_whole_numbers_displayed_as_int(self):
        series = pd.Series([3.0, 6.0, 3.0, 6.0], dtype=np.float32)
        result = _get_display_values(series)
        assert result[0] == (3.0, "3", 2)
        assert result[1] == (6.0, "6", 2)

    def test_float_column_nan_shown_as_unknown(self):
        series = pd.Series([3.0, np.nan, np.nan], dtype=np.float32)
        result = _get_display_values(series)
        raw_vals = [entry[0] for entry in result]
        display_vals = [entry[1] for entry in result]
        assert 3.0 in raw_vals
        assert "Unknown (not recorded)" in display_vals

    def test_float_column_all_nan_returns_one_entry(self):
        series = pd.Series([np.nan, np.nan], dtype=np.float32)
        result = _get_display_values(series)
        assert len(result) == 1
        assert result[0][1] == "Unknown (not recorded)"

    def test_string_column_sorted_alphabetically(self):
        series = pd.Series(["Not_Diluted", "Diluted", "Diluted", "Not_Diluted"])
        result = _get_display_values(series)
        assert result[0][0] == "Diluted"
        assert result[1][0] == "Not_Diluted"

    def test_string_column_empty_string_grouped_as_unknown(self):
        series = pd.Series(["AL", "", "AL", ""])
        result = _get_display_values(series)
        display_strings = [entry[1] for entry in result]
        assert "AL" in display_strings
        assert "Unknown (not recorded)" in display_strings

    def test_string_column_all_empty_returns_one_unknown(self):
        series = pd.Series(["", "", ""])
        result = _get_display_values(series)
        assert len(result) == 1
        assert result[0][1] == "Unknown (not recorded)"

    def test_counts_are_correct(self):
        series = pd.Series(["A", "A", "B", "A", "B"])
        result = _get_display_values(series)
        count_map = {entry[0]: entry[2] for entry in result}
        assert count_map["A"] == 3
        assert count_map["B"] == 2


# ---------------------------------------------------------------------------
# Tests: _filter_obs_by_values (unit tests — no console interaction)
# ---------------------------------------------------------------------------


class TestFilterObsByValues:
    """Unit tests for the _filter_obs_by_values helper."""

    def test_filters_string_column(self):
        obs = pd.DataFrame({
            "dilution": ["Diluted", "Not_Diluted", "Diluted", "Not_Diluted"],
        })
        result = _filter_obs_by_values(obs, "dilution", ["Diluted"])
        assert list(result["dilution"]) == ["Diluted", "Diluted"]

    def test_filters_multiple_values(self):
        obs = pd.DataFrame({"diet": ["AL", "IF", "AL", "IF", "AL"]})
        result = _filter_obs_by_values(obs, "diet", ["AL", "IF"])
        assert len(result) == 5  # all rows kept

    def test_filters_boolean_column(self):
        obs = pd.DataFrame({
            "FlowAI_Pass": np.array([True, False, True, False], dtype=np.bool_),
        })
        result = _filter_obs_by_values(obs, "FlowAI_Pass", [True])
        assert len(result) == 2
        assert result["FlowAI_Pass"].all()

    def test_filters_float_column(self):
        obs = pd.DataFrame({
            "age": np.array([3.0, 6.0, 3.0, 6.0], dtype=np.float32),
        })
        result = _filter_obs_by_values(obs, "age", [3.0])
        assert len(result) == 2
        assert (result["age"] == 3.0).all()

    def test_missing_sentinel_matches_both_empty_string_and_nan(self):
        obs = pd.DataFrame({
            "fcs_acquisition_date": ["2024-01-01", "", np.nan, "2024-01-01"],
        })
        # "" is the missing sentinel in _get_display_values
        result = _filter_obs_by_values(obs, "fcs_acquisition_date", [""])
        assert len(result) == 2  # "" and np.nan rows

    def test_preserves_original_index(self):
        obs = pd.DataFrame(
            {"dilution": ["Diluted", "Not_Diluted", "Diluted"]},
            index=["obs_0", "obs_1", "obs_2"],
        )
        result = _filter_obs_by_values(obs, "dilution", ["Diluted"])
        assert list(result.index) == ["obs_0", "obs_2"]


# ---------------------------------------------------------------------------
# Tests: select_sfc_subset (integration tests with mocked input)
# ---------------------------------------------------------------------------


class TestSelectSfcSubset:
    """Integration tests for the interactive select_sfc_subset() function."""

    def test_all_selection_returns_full_anndata(self, test_anndata, monkeypatch):
        """Selecting 'A' for every prompt returns an AnnData identical in size."""
        # 5 prompted columns: unique_subject_ID, dilution, diet, FlowAI_Pass, age
        # + 1 sub-sampling prompt answered with "" (skip — new default=0)
        monkeypatch.setattr("builtins.input", _make_input_sequence(["A"] * 5 + [""]))
        result = select_sfc_subset(test_anndata)
        assert result.n_obs == test_anndata.n_obs
        assert result.n_vars == test_anndata.n_vars

    def test_select_single_subject(self, test_anndata, monkeypatch):
        """Selecting unique_subject_ID [1] (MNMS_S001) returns exactly 40 events."""
        # Subjects displayed alphabetically:
        #   [1] MNMS_S001, [2] MNMS_S002, [3] MNMS_S003
        monkeypatch.setattr(
            "builtins.input",
            _make_input_sequence(["1", "A", "A", "A", "A", ""]),
        )
        result = select_sfc_subset(test_anndata)
        assert result.n_obs == 40
        assert (result.obs["subject_ID"] == "S001").all()

    def test_select_multiple_subjects(self, test_anndata, monkeypatch):
        """Selecting unique_subject_IDs [1,2] (MNMS_S001, MNMS_S002) returns 80 events."""
        monkeypatch.setattr(
            "builtins.input",
            _make_input_sequence(["1,2", "A", "A", "A", "A", ""]),
        )
        result = select_sfc_subset(test_anndata)
        assert result.n_obs == 80
        assert set(result.obs["subject_ID"].unique()) == {"S001", "S002"}

    def test_select_one_dilution(self, test_anndata, monkeypatch):
        """Selecting dilution [1] (Diluted) from all subjects returns 60 events."""
        # dilutions displayed alphabetically: [1] Diluted, [2] Not_Diluted
        monkeypatch.setattr(
            "builtins.input",
            _make_input_sequence(["A", "1", "A", "A", "A", ""]),
        )
        result = select_sfc_subset(test_anndata)
        assert result.n_obs == 60
        assert (result.obs["dilution"] == "Diluted").all()

    def test_filter_flowai_pass_true_only(self, test_anndata, monkeypatch):
        """Selecting FlowAI_Pass [1] (True) from all events returns 60 events."""
        # FlowAI_Pass displayed: [1] True, [2] False
        monkeypatch.setattr(
            "builtins.input",
            _make_input_sequence(["A", "A", "A", "1", "A", ""]),
        )
        result = select_sfc_subset(test_anndata)
        assert result.n_obs == 60
        assert result.obs["FlowAI_Pass"].all()

    def test_filter_single_age(self, test_anndata, monkeypatch):
        """Selecting age [1] (3) from all subjects returns 60 events."""
        # ages displayed ascending: [1] 3, [2] 6
        monkeypatch.setattr(
            "builtins.input",
            _make_input_sequence(["A", "A", "A", "A", "1", ""]),
        )
        result = select_sfc_subset(test_anndata)
        assert result.n_obs == 60
        assert (result.obs["age"] == 3.0).all()

    def test_multi_column_filter(self, test_anndata, monkeypatch):
        """Combined filter: MNMS_S001 + Diluted → 20 events
        (FlowAI and age auto-skipped after this filter reduces to 1 unique value each).

        After selecting MNMS_S001 + Diluted, all remaining events have FlowAI_Pass=True
        and age=3.0 (single values) so those two columns are auto-skipped.
        Only 4 filter prompts fire: unique_subject_ID, dilution, diet, then skip.
        """
        monkeypatch.setattr(
            "builtins.input",
            _make_input_sequence(["1", "1", "A", ""]),
        )
        result = select_sfc_subset(test_anndata)
        assert result.n_obs == 20
        assert (result.obs["subject_ID"] == "S001").all()
        assert (result.obs["dilution"] == "Diluted").all()
        assert result.obs["FlowAI_Pass"].all()

    def test_n_vars_unchanged_after_selection(self, test_anndata, monkeypatch):
        """Feature (channel) count must not change during obs filtering."""
        monkeypatch.setattr("builtins.input", _make_input_sequence(["1"] + ["A"] * 4 + [""]))
        result = select_sfc_subset(test_anndata)
        assert result.n_vars == test_anndata.n_vars

    def test_analysis_config_written_to_uns(self, test_anndata, monkeypatch):
        """selection choices must be persisted in .uns['analysis_config']."""
        monkeypatch.setattr("builtins.input", _make_input_sequence(["A"] * 5 + [""]))
        result = select_sfc_subset(test_anndata)
        assert "analysis_config" in result.uns
        config = result.uns["analysis_config"]
        assert "selection" in config
        assert "active_layer" in config
        assert "active_selection" in config

    def test_analysis_config_active_layer_is_none_initially(
        self, test_anndata, monkeypatch
    ):
        """active_layer must be None until normalization runs."""
        monkeypatch.setattr("builtins.input", _make_input_sequence(["A"] * 5 + [""]))
        result = select_sfc_subset(test_anndata)
        assert result.uns["analysis_config"]["active_layer"] is None

    def test_auto_skip_single_value_columns(self, test_anndata, monkeypatch):
        """Columns with only 1 unique value (specie, fcs_acquisition_date)
        must not generate a prompt — exactly 6 inputs are consumed (5 filters +
        1 sub-sampling), not 8."""
        responses = ["A"] * 5 + [""]
        input_mock = _make_input_sequence(responses)
        monkeypatch.setattr("builtins.input", input_mock)
        # If more than 6 inputs were consumed, StopIteration would propagate
        # and the test would fail. Completing without error proves auto-skip works.
        result = select_sfc_subset(test_anndata)
        assert result.n_obs == test_anndata.n_obs

    def test_invalid_then_valid_input_retries(self, test_anndata, monkeypatch):
        """An invalid response (bad number) must be rejected; next response used."""
        # unique_subject_ID prompt: first "99" is invalid, then "A" is accepted.
        # Remaining 4 prompts: all "A". Then "" to skip sub-sampling.
        monkeypatch.setattr(
            "builtins.input",
            _make_input_sequence(["99", "A", "A", "A", "A", "A", ""]),
        )
        result = select_sfc_subset(test_anndata)
        assert result.n_obs == test_anndata.n_obs

    def test_x_matrix_values_preserved(self, test_anndata, monkeypatch):
        """The .X values for selected rows must be identical to the original."""
        monkeypatch.setattr("builtins.input", _make_input_sequence(["1"] + ["A"] * 4 + [""]))
        result = select_sfc_subset(test_anndata)
        # MNMS_S001 is the first 40 rows of the original AnnData.
        original_x_s001 = test_anndata.X[:40]
        np.testing.assert_array_equal(result.X, original_x_s001)

    def test_anndata_with_missing_filterable_column_is_silently_skipped(
        self, monkeypatch
    ):
        """Columns listed in FILTERABLE_OBS_COLUMNS but absent from .obs are ignored.

        This obs only has subject_ID and dilution. unique_subject_ID (in FILTERABLE)
        is absent → silently skipped. subject_ID is not in FILTERABLE → never shown.
        Only dilution triggers a prompt.
        """
        # Build an AnnData with only subject_ID and dilution.
        obs = pd.DataFrame({
            "subject_ID": ["S001", "S001", "S002", "S002"],
            "dilution": ["Diluted", "Not_Diluted", "Diluted", "Not_Diluted"],
        })
        x_matrix = np.ones((4, 2), dtype=np.float32)
        var = pd.DataFrame(index=["CH0", "CH1"])
        small_anndata = anndata.AnnData(X=x_matrix, obs=obs, var=var)

        # Only dilution has 2 unique values → 1 prompt + 1 sub-sampling.
        monkeypatch.setattr("builtins.input", _make_input_sequence(["A", ""]))
        result = select_sfc_subset(small_anndata)
        assert result.n_obs == 4

    def test_non_unique_obs_names_do_not_inflate_result(self, monkeypatch):
        """Filtering must return the exact events that survived the filter even
        when obs_names are not unique (duplicate index labels).

        Bug scenario: if the final slice uses .isin() on obs index values,
        non-unique obs_names cause over-selection — rows from different subjects
        that share an index label are all retained.  The positional approach must
        return exactly the 40 events belonging to MNMS_S001.
        """
        # Build an AnnData where all three subjects share the SAME obs_names
        # (index values "0".."39" repeat for every subject — worst-case duplicates).
        obs = pd.DataFrame({
            "unique_subject_ID": ["MNMS_S001"] * 40 + ["MNMS_S002"] * 40 + ["MNMS_S003"] * 40,
            "subject_ID": ["S001"] * 40 + ["S002"] * 40 + ["S003"] * 40,
            "dilution": (["Diluted"] * 20 + ["Not_Diluted"] * 20) * 3,
        }, index=list(range(40)) * 3)

        x_matrix = np.random.default_rng(0).random((120, 2)).astype(np.float32)
        var = pd.DataFrame(index=["CH0", "CH1"])
        dup_anndata = anndata.AnnData(X=x_matrix, obs=obs, var=var)

        # unique_subject_ID → "1" (MNMS_S001), dilution → "A" (all),
        # sub-sampling → "" (skip)
        monkeypatch.setattr("builtins.input", _make_input_sequence(["1", "A", ""]))
        result = select_sfc_subset(dup_anndata)

        assert result.n_obs == 40, (
            f"Expected 40 events for MNMS_S001, got {result.n_obs}. "
            "Non-unique obs_names likely caused over-selection."
        )
        # All returned events must belong to S001.
        assert (result.obs["subject_ID"] == "S001").all()
        # The helper column must not leak into the output.
        assert "_row_position" not in result.obs.columns


# ---------------------------------------------------------------------------
# Tests: _prompt_subsampling_size (unit tests — no AnnData needed)
# ---------------------------------------------------------------------------


class TestPromptSubsamplingSize:
    """Unit tests for the _prompt_subsampling_size helper."""

    def _make_obs(self, subjects: list, counts: list) -> pd.DataFrame:
        """Build a minimal obs DataFrame with a subject_ID column."""
        rows = []
        for subject, count in zip(subjects, counts):
            rows.extend([subject] * count)
        return pd.DataFrame({"subject_ID": rows})

    def test_returns_integer_on_valid_input(self, monkeypatch):
        """A valid integer string is returned directly."""
        obs = self._make_obs(["S001", "S002"], [40, 40])
        monkeypatch.setattr("builtins.input", _make_input_sequence(["10"]))
        result = _prompt_subsampling_size(obs)
        assert result == 10

    def test_empty_string_returns_default_zero(self, monkeypatch):
        """Pressing Enter (empty string) returns the default value (0 = skip)."""
        obs = self._make_obs(["S001"], [40])
        monkeypatch.setattr("builtins.input", _make_input_sequence([""]))
        result = _prompt_subsampling_size(obs)
        assert result == 0

    def test_empty_string_with_explicit_default_returns_it(self, monkeypatch):
        """Pressing Enter returns an explicitly passed default_n."""
        obs = self._make_obs(["S001"], [40])
        monkeypatch.setattr("builtins.input", _make_input_sequence([""]))
        result = _prompt_subsampling_size(obs, default_n=200)
        assert result == 200

    def test_zero_returns_zero(self, monkeypatch):
        """Entering 0 is valid and signals 'skip sub-sampling'."""
        obs = self._make_obs(["S001"], [40])
        monkeypatch.setattr("builtins.input", _make_input_sequence(["0"]))
        result = _prompt_subsampling_size(obs)
        assert result == 0

    def test_invalid_string_then_valid_retries(self, monkeypatch):
        """A non-numeric string is rejected; the following valid input is used."""
        obs = self._make_obs(["S001"], [40])
        monkeypatch.setattr("builtins.input", _make_input_sequence(["abc", "25"]))
        result = _prompt_subsampling_size(obs)
        assert result == 25

    def test_negative_number_then_valid_retries(self, monkeypatch):
        """A negative integer is rejected; the following valid input is used."""
        obs = self._make_obs(["S001"], [40])
        monkeypatch.setattr("builtins.input", _make_input_sequence(["-5", "10"]))
        result = _prompt_subsampling_size(obs)
        assert result == 10

    def test_no_subject_id_column_does_not_crash(self, monkeypatch):
        """obs without subject_ID must still produce a prompt without error."""
        obs = pd.DataFrame({"dilution": ["Diluted"] * 20})
        monkeypatch.setattr("builtins.input", _make_input_sequence(["0"]))
        result = _prompt_subsampling_size(obs)
        assert result == 0

    def test_uses_unique_subject_id_when_present(self, monkeypatch, capsys):
        """When unique_subject_ID is present, subjects are listed by that column."""
        obs = pd.DataFrame({
            "unique_subject_ID": ["MNMS_S001"] * 10 + ["MNMS_S002"] * 10,
            "subject_ID": ["S001"] * 10 + ["S002"] * 10,
        })
        monkeypatch.setattr("builtins.input", _make_input_sequence(["0"]))
        _prompt_subsampling_size(obs)
        captured = capsys.readouterr()
        # The unique IDs should appear in the printed output.
        assert "MNMS_S001" in captured.out
        assert "MNMS_S002" in captured.out


# ---------------------------------------------------------------------------
# Tests: _subsample_obs_per_subject (unit tests — no AnnData needed)
# ---------------------------------------------------------------------------


class TestSubsampleObsPerSubject:
    """Unit tests for the _subsample_obs_per_subject helper."""

    def _make_obs_with_positions(
        self, subjects: list, counts: list, use_unique: bool = False
    ) -> pd.DataFrame:
        """Build an obs DataFrame that mimics the real one inside select_sfc_subset."""
        rows = []
        position = 0
        for subject, count in zip(subjects, counts):
            for _ in range(count):
                row = {"_row_position": position}
                if use_unique:
                    row["unique_subject_ID"] = subject
                else:
                    row["subject_ID"] = subject
                rows.append(row)
                position += 1
        return pd.DataFrame(rows)

    def test_without_replacement_returns_exact_n(self):
        """When enough events exist, each subject contributes exactly n rows."""
        obs = self._make_obs_with_positions(["S001", "S002"], [50, 50])
        result = _subsample_obs_per_subject(obs, n_events_per_subject=10)
        assert len(result) == 20
        assert (result["subject_ID"] == "S001").sum() == 10
        assert (result["subject_ID"] == "S002").sum() == 10

    def test_without_replacement_no_duplicate_positions(self):
        """Without-replacement sampling must not repeat the same row position."""
        obs = self._make_obs_with_positions(["S001"], [100])
        result = _subsample_obs_per_subject(obs, n_events_per_subject=30)
        assert result["_row_position"].nunique() == 30

    def test_with_replacement_when_insufficient_returns_requested_n(self, capsys):
        """When a subject has fewer events than requested, the result still has n rows."""
        obs = self._make_obs_with_positions(["S001"], [10])
        result = _subsample_obs_per_subject(obs, n_events_per_subject=25)
        assert len(result) == 25

    def test_with_replacement_prints_warning(self, capsys):
        """A WARNING must be printed to stdout when with-replacement is used."""
        obs = self._make_obs_with_positions(["S001"], [10])
        _subsample_obs_per_subject(obs, n_events_per_subject=25)
        captured = capsys.readouterr()
        assert "WARNING" in captured.out
        assert "S001" in captured.out

    def test_with_replacement_has_duplicate_positions(self):
        """With-replacement sampling may repeat row positions."""
        obs = self._make_obs_with_positions(["S001"], [5])
        result = _subsample_obs_per_subject(obs, n_events_per_subject=20)
        # With only 5 unique positions and 20 draws, duplicates are guaranteed.
        assert result["_row_position"].nunique() <= 5

    def test_reproducible_with_same_seed(self):
        """Same random_seed must produce identical results."""
        obs = self._make_obs_with_positions(["S001"], [100])
        result_a = _subsample_obs_per_subject(obs, n_events_per_subject=30, random_seed=7)
        result_b = _subsample_obs_per_subject(obs, n_events_per_subject=30, random_seed=7)
        np.testing.assert_array_equal(
            result_a["_row_position"].values, result_b["_row_position"].values
        )

    def test_no_subject_id_column_samples_globally(self):
        """Without subject_ID, the whole dataset is treated as one group."""
        obs = pd.DataFrame({"_row_position": np.arange(100)})
        result = _subsample_obs_per_subject(obs, n_events_per_subject=20)
        assert len(result) == 20

    def test_no_subject_id_warning_when_insufficient(self, capsys):
        """Without subject_ID, a WARNING is still printed when not enough events."""
        obs = pd.DataFrame({"_row_position": np.arange(5)})
        _subsample_obs_per_subject(obs, n_events_per_subject=15)
        captured = capsys.readouterr()
        assert "WARNING" in captured.out

    def test_uses_unique_subject_id_when_present(self):
        """Subsampling groups by unique_subject_ID when that column is available."""
        obs = self._make_obs_with_positions(
            ["MNMS_S001", "MNMS_S002"], [50, 50], use_unique=True
        )
        result = _subsample_obs_per_subject(obs, n_events_per_subject=10)
        assert len(result) == 20
        assert (result["unique_subject_ID"] == "MNMS_S001").sum() == 10
        assert (result["unique_subject_ID"] == "MNMS_S002").sum() == 10


# ---------------------------------------------------------------------------
# Tests: subsampling integration inside select_sfc_subset
# ---------------------------------------------------------------------------


class TestSubsamplingIntegration:
    """Integration tests for the subsampling step inside select_sfc_subset."""

    def test_subsampling_reduces_total_events(self, test_anndata, monkeypatch):
        """Requesting 10 events per subject from 3 subjects × 40 events = 30 total."""
        # 5 filter prompts (all A) + sub-sampling → 10
        monkeypatch.setattr("builtins.input", _make_input_sequence(["A"] * 5 + ["10"]))
        result = select_sfc_subset(test_anndata)
        assert result.n_obs == 30

    def test_subsampling_equal_per_subject(self, test_anndata, monkeypatch):
        """Each subject must contribute exactly the requested number of events."""
        monkeypatch.setattr("builtins.input", _make_input_sequence(["A"] * 5 + ["15"]))
        result = select_sfc_subset(test_anndata)
        for subject_id in ["S001", "S002", "S003"]:
            count = (result.obs["subject_ID"] == subject_id).sum()
            assert count == 15, f"Expected 15 events for {subject_id}, got {count}"

    def test_subsampling_skip_enter_keeps_all_events(self, test_anndata, monkeypatch):
        """Pressing Enter (new default=0) must skip sub-sampling and keep all events."""
        monkeypatch.setattr("builtins.input", _make_input_sequence(["A"] * 5 + [""]))
        result = select_sfc_subset(test_anndata)
        assert result.n_obs == test_anndata.n_obs

    def test_subsampling_zero_keeps_all_events(self, test_anndata, monkeypatch):
        """Entering 0 must skip sub-sampling and keep all filtered events."""
        monkeypatch.setattr("builtins.input", _make_input_sequence(["A"] * 5 + ["0"]))
        result = select_sfc_subset(test_anndata)
        assert result.n_obs == test_anndata.n_obs

    def test_subsampling_logged_in_analysis_config(self, test_anndata, monkeypatch):
        """n_events_per_subject and applied flag must be stored in analysis_config."""
        monkeypatch.setattr("builtins.input", _make_input_sequence(["A"] * 5 + ["20"]))
        result = select_sfc_subset(test_anndata)
        subsampling_config = result.uns["analysis_config"]["subsampling"]
        assert subsampling_config["n_events_per_subject"] == 20
        assert subsampling_config["applied"] is True

    def test_subsampling_skip_logged_as_not_applied(self, test_anndata, monkeypatch):
        """When sub-sampling is skipped, applied must be False in analysis_config."""
        monkeypatch.setattr("builtins.input", _make_input_sequence(["A"] * 5 + [""]))
        result = select_sfc_subset(test_anndata)
        subsampling_config = result.uns["analysis_config"]["subsampling"]
        assert subsampling_config["n_events_per_subject"] == 0
        assert subsampling_config["applied"] is False

    def test_subsampling_with_replacement_prints_warning(
        self, test_anndata, monkeypatch, capsys
    ):
        """Requesting more events than available per subject must print a WARNING."""
        # Fixture has 40 events per subject; request 100 → triggers replacement.
        monkeypatch.setattr("builtins.input", _make_input_sequence(["A"] * 5 + ["100"]))
        result = select_sfc_subset(test_anndata)
        captured = capsys.readouterr()
        assert "WARNING" in captured.out
        # Each subject contributes 100 events despite only having 40.
        assert result.n_obs == 300

    def test_n_vars_unchanged_after_subsampling(self, test_anndata, monkeypatch):
        """Channel count must be unaffected by sub-sampling."""
        monkeypatch.setattr("builtins.input", _make_input_sequence(["A"] * 5 + ["10"]))
        result = select_sfc_subset(test_anndata)
        assert result.n_vars == test_anndata.n_vars
