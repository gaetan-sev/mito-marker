"""
test_sfc_filename_columns.py

Unit tests for the .obs columns declared in the notebook and read from FCS
filename words (ADR-021):
  - parse_filename_columns() and _validate_filename_columns() in
    src/mito_marker/sfc/filename_parser.py
  - annotate_sfc_filename_columns() in src/mito_marker/sfc/filename_columns.py
  - their interplay with annotate_sfc_conditions() in src/mito_marker/sfc/conditions.py

Tests cover:
- One value per column read from whole filename words; NaN when absent;
  case-sensitive; aliases; two values for one column raise
- Extendable built-in columns (diet, age_group, treatment): built-in words keep
  their value, declared words add values
- Validation: form of column names and words, reserved columns, words that
  already have a meaning, a word in two columns, a word both product and value
- .obs columns: categorical, categories, NaN, replaced column, .uns record,
  merge with saved declarations, .h5ad round trip
- annotate_sfc_conditions() recognises the declared words, rejects a product
  that is a column word, and its warning shows how to declare a column
"""

import warnings

import anndata
import numpy as np
import pandas as pd
import pytest

from mito_marker import annotate_sfc_filename_columns as top_level_annotate
from mito_marker.controlled_vocabulary import (
    SFC_CONDITION_TOKENS_UNS_KEY,
    SFC_FILENAME_COLUMNS_UNS_KEY,
    SFC_OBS_COLUMNS_EXTENDABLE_FROM_FILENAME,
)
from mito_marker.sfc.conditions import annotate_sfc_conditions
from mito_marker.sfc.filename_columns import annotate_sfc_filename_columns
from mito_marker.sfc.filename_parser import (
    _BUILT_IN_VALUE_BY_TOKEN_FOR_EXTENDABLE_COLUMN,
    _validate_filename_columns,
    parse_filename_columns,
)

_SMURF_COLUMNS: dict[str, dict[str, str]] = {
    "smurf_status": {"S": "Smurf", "NonS": "Non_Smurf"},
}

_SMURF_FILENAMES: list[str] = [
    "export_FLY_001_D02_S_DR_FlowAIGoodEvents_DR+.fcs",
    "export_FLY_002_D02_NonS_DR_FlowAIGoodEvents_DR+.fcs",
    "export_FLY_003_D30_S_DR_FlowAIGoodEvents_DR+.fcs",
    "export_FLY_004_D30_DR_FlowAIGoodEvents_DR+.fcs",
]


def _build_anndata(filenames: list[str], events_per_file: int = 3) -> anndata.AnnData:
    """Small SFC-like AnnData: events_per_file events per filename, 2 channels."""
    observation_dataframe = pd.DataFrame(
        {"source_filename": np.repeat(filenames, events_per_file)}
    )
    observation_dataframe.index = observation_dataframe.index.astype(str)
    return anndata.AnnData(
        X=np.ones((len(observation_dataframe), 2), dtype=np.float32),
        obs=observation_dataframe,
    )


def _one_value_per_file(sfc_anndata: anndata.AnnData, column_name: str, events_per_file: int = 3) -> list:
    """Value of column_name for the first event of each file, NaN kept as None."""
    values = sfc_anndata.obs[column_name].iloc[::events_per_file]
    return [None if pd.isna(value) else value for value in values]


# ---------------------------------------------------------------------------
# parse_filename_columns
# ---------------------------------------------------------------------------


class TestParseFilenameColumns:
    """One value per declared column, read from whole filename words."""

    @pytest.mark.parametrize(
        "filename, expected_value",
        [
            ("export_FLY_001_D02_S_DR_FlowAIGoodEvents_DR+.fcs", "Smurf"),
            ("export_FLY_002_D02_NonS_DR_FlowAIGoodEvents_DR+.fcs", "Non_Smurf"),
            ("export_FLY_004_D30_DR_FlowAIGoodEvents_DR+.fcs", None),
            ("FLY_001_D02_S_DR", "Smurf"),
        ],
    )
    def test_value_read_from_filename(self, filename, expected_value):
        assert parse_filename_columns(filename, _SMURF_COLUMNS) == {"smurf_status": expected_value}

    def test_whole_words_only(self):
        """'S' must not match inside another word such as 'SSC' or 'NonS'."""
        result = parse_filename_columns("FLY_001_SSC_NonS_DR", _SMURF_COLUMNS)
        assert result == {"smurf_status": "Non_Smurf"}

    def test_words_are_case_sensitive(self):
        assert parse_filename_columns("FLY_001_s_DR", _SMURF_COLUMNS) == {"smurf_status": None}

    def test_two_values_for_one_column_raise(self):
        with pytest.raises(ValueError, match="two values to the .obs column 'smurf_status'"):
            parse_filename_columns("FLY_001_S_NonS_DR", _SMURF_COLUMNS)

    def test_aliases_giving_one_value_are_accepted(self):
        columns = {"smurf_status": {"S": "Smurf", "Smurf": "Smurf"}}
        assert parse_filename_columns("FLY_001_S_Smurf_DR", columns) == {"smurf_status": "Smurf"}

    def test_several_columns(self):
        columns = {**_SMURF_COLUMNS, "sex": {"Male": "Male", "Female": "Female"}}
        result = parse_filename_columns("FLY_001_D02_Female_S_DR", columns)
        assert result == {"smurf_status": "Smurf", "sex": "Female"}

    def test_extendable_column_keeps_built_in_words(self):
        columns = {"diet": {"CR": "Caloric_Restriction"}}
        assert parse_filename_columns("WORM_001_AL_DR", columns) == {"diet": "AL"}
        assert parse_filename_columns("WORM_001_CR_DR", columns) == {"diet": "Caloric_Restriction"}
        assert parse_filename_columns("WORM_001_DR", columns) == {"diet": None}

    def test_extendable_column_built_in_and_declared_conflict(self):
        with pytest.raises(ValueError, match="two values"):
            parse_filename_columns("WORM_001_AL_CR_DR", {"diet": {"CR": "Caloric_Restriction"}})

    def test_declared_alias_of_a_built_in_value(self):
        columns = {"age_group": {"Aged": "Old"}}
        assert parse_filename_columns("MOUSE_001_Aged_DR", columns) == {"age_group": "Old"}
        assert parse_filename_columns("MOUSE_001_old_DR", columns) == {"age_group": "Old"}

    def test_built_in_words_cover_every_extendable_column(self):
        assert set(_BUILT_IN_VALUE_BY_TOKEN_FOR_EXTENDABLE_COLUMN) == set(
            SFC_OBS_COLUMNS_EXTENDABLE_FROM_FILENAME
        )


class TestFilenameColumnsValidation:
    """The declaration is checked before any filename is read."""

    def test_not_a_dict_raises_type_error(self):
        with pytest.raises(TypeError, match="filename_columns must be a dict"):
            _validate_filename_columns(["smurf_status"])

    def test_words_not_a_dict_raise_type_error(self):
        with pytest.raises(TypeError, match="words of column 'smurf_status'"):
            _validate_filename_columns({"smurf_status": ["S", "NonS"]})

    def test_column_without_word_raises(self):
        with pytest.raises(ValueError, match="declares no word"):
            _validate_filename_columns({"smurf_status": {}})

    @pytest.mark.parametrize("bad_column_name", ["", "1st_group", "smurf status", "smurf-status", 3])
    def test_malformed_column_name_raises(self, bad_column_name):
        with pytest.raises(ValueError, match="Invalid column name"):
            _validate_filename_columns({bad_column_name: {"S": "Smurf"}})

    @pytest.mark.parametrize(
        "reserved_column",
        ["specie", "subject_ID", "age", "marker", "dilution", "FlowAI_Pass",
         "source_filename", "condition", "number_of_additions",
         "minutes_after_last_addition"],
    )
    def test_reserved_column_raises(self, reserved_column):
        with pytest.raises(ValueError, match=f"'{reserved_column}' cannot be declared"):
            _validate_filename_columns({reserved_column: {"S": "Smurf"}})

    @pytest.mark.parametrize("reserved_column", ["with_ADP", "fcs_tube_name"])
    def test_product_presence_and_fcs_columns_raise(self, reserved_column):
        with pytest.raises(ValueError, match="cannot be declared"):
            _validate_filename_columns({reserved_column: {"S": "Smurf"}})

    @pytest.mark.parametrize("extendable_column", ["diet", "age_group", "treatment"])
    def test_extendable_columns_accepted(self, extendable_column):
        _validate_filename_columns({extendable_column: {"Xnew": "New_value"}})

    @pytest.mark.parametrize("bad_token", ["S_1", "1S", "", "S S", 7])
    def test_malformed_word_raises(self, bad_token):
        with pytest.raises(ValueError, match="Invalid word"):
            _validate_filename_columns({"smurf_status": {bad_token: "Smurf"}})

    @pytest.mark.parametrize("token_with_a_meaning", ["AL", "DR", "FLY", "D02", "old", "export", "ND"])
    def test_word_with_a_meaning_raises(self, token_with_a_meaning):
        with pytest.raises(ValueError, match="already has a meaning"):
            _validate_filename_columns({"smurf_status": {token_with_a_meaning: "Smurf"}})

    def test_word_in_two_columns_raises(self):
        with pytest.raises(ValueError, match="declared in two columns"):
            _validate_filename_columns({"smurf_status": {"S": "Smurf"}, "sex": {"S": "Male"}})

    def test_word_both_product_and_value_raises(self):
        with pytest.raises(ValueError, match="both as a product"):
            _validate_filename_columns(_SMURF_COLUMNS, condition_tokens={"S": "Sucrose"})

    @pytest.mark.parametrize("bad_value", ["", "   ", None, 2])
    def test_missing_value_raises(self, bad_value):
        with pytest.raises(ValueError, match="needs a value"):
            _validate_filename_columns({"smurf_status": {"S": bad_value}})


# ---------------------------------------------------------------------------
# annotate_sfc_filename_columns
# ---------------------------------------------------------------------------


class TestAnnotateFilenameColumns:
    """Declared columns written to .obs and recorded in .uns."""

    @pytest.fixture
    def annotated(self) -> anndata.AnnData:
        return annotate_sfc_filename_columns(_build_anndata(_SMURF_FILENAMES), _SMURF_COLUMNS)

    def test_exported_at_top_level(self):
        assert top_level_annotate is annotate_sfc_filename_columns

    def test_returns_the_same_object(self):
        sfc_anndata = _build_anndata(_SMURF_FILENAMES)
        assert annotate_sfc_filename_columns(sfc_anndata, _SMURF_COLUMNS) is sfc_anndata

    def test_values_per_file(self, annotated):
        assert _one_value_per_file(annotated, "smurf_status") == [
            "Smurf", "Non_Smurf", "Smurf", None,
        ]

    def test_categorical_with_sorted_declared_values(self, annotated):
        column_values = annotated.obs["smurf_status"]
        assert isinstance(column_values.dtype, pd.CategoricalDtype)
        assert list(column_values.cat.categories) == ["Non_Smurf", "Smurf"]

    def test_declaration_saved_in_uns(self, annotated):
        assert annotated.uns[SFC_FILENAME_COLUMNS_UNS_KEY] == _SMURF_COLUMNS

    def test_x_unchanged(self, annotated):
        assert np.all(annotated.X == 1)

    def test_categorical_source_filename(self):
        sfc_anndata = _build_anndata(_SMURF_FILENAMES)
        sfc_anndata.obs["source_filename"] = sfc_anndata.obs["source_filename"].astype("category")
        annotate_sfc_filename_columns(sfc_anndata, _SMURF_COLUMNS)
        assert _one_value_per_file(sfc_anndata, "smurf_status")[:2] == ["Smurf", "Non_Smurf"]

    def test_existing_column_is_replaced(self):
        sfc_anndata = _build_anndata(_SMURF_FILENAMES)
        sfc_anndata.obs["smurf_status"] = "typed by hand"
        annotate_sfc_filename_columns(sfc_anndata, _SMURF_COLUMNS)
        assert "typed by hand" not in set(sfc_anndata.obs["smurf_status"].dropna())

    def test_extendable_column(self):
        filenames = ["WORM_001_D08_AL_DR.fcs", "WORM_002_D08_CR_DR.fcs", "WORM_003_D08_DR.fcs"]
        sfc_anndata = annotate_sfc_filename_columns(
            _build_anndata(filenames), {"diet": {"CR": "Caloric_Restriction"}}
        )
        assert _one_value_per_file(sfc_anndata, "diet") == ["AL", "Caloric_Restriction", None]
        assert list(sfc_anndata.obs["diet"].cat.categories) == ["AL", "IF", "Caloric_Restriction"]

    def test_saved_columns_used_when_none_given(self, annotated):
        annotated.obs = annotated.obs.drop(columns="smurf_status")
        annotate_sfc_filename_columns(annotated)
        assert _one_value_per_file(annotated, "smurf_status")[0] == "Smurf"

    def test_new_columns_merged_with_saved_ones(self, annotated):
        annotate_sfc_filename_columns(annotated, {"smurf_status": {"Smurf": "Smurf"}})
        assert annotated.uns[SFC_FILENAME_COLUMNS_UNS_KEY] == {
            "smurf_status": {"S": "Smurf", "NonS": "Non_Smurf", "Smurf": "Smurf"},
        }

    def test_new_value_replaces_saved_one(self, annotated):
        annotate_sfc_filename_columns(annotated, {"smurf_status": {"S": "Smurf_fly"}})
        assert _one_value_per_file(annotated, "smurf_status")[0] == "Smurf_fly"

    def test_nothing_declared_writes_nothing(self):
        sfc_anndata = annotate_sfc_filename_columns(_build_anndata(_SMURF_FILENAMES))
        assert SFC_FILENAME_COLUMNS_UNS_KEY not in sfc_anndata.uns
        assert list(sfc_anndata.obs.columns) == ["source_filename"]

    def test_missing_source_filename_raises(self):
        sfc_anndata = _build_anndata(_SMURF_FILENAMES)
        sfc_anndata.obs = sfc_anndata.obs.drop(columns="source_filename")
        with pytest.raises(ValueError, match="source_filename"):
            annotate_sfc_filename_columns(sfc_anndata, _SMURF_COLUMNS)

    def test_conflicts_listed_and_nothing_written(self):
        filenames = _SMURF_FILENAMES + ["FLY_005_S_NonS_DR.fcs", "FLY_006_NonS_S_DR.fcs"]
        sfc_anndata = _build_anndata(filenames)
        with pytest.raises(ValueError, match="2 file") as raised:
            annotate_sfc_filename_columns(sfc_anndata, _SMURF_COLUMNS)
        assert "FLY_005_S_NonS_DR.fcs" in str(raised.value)
        assert "FLY_006_NonS_S_DR.fcs" in str(raised.value)
        assert "smurf_status" not in sfc_anndata.obs.columns
        assert SFC_FILENAME_COLUMNS_UNS_KEY not in sfc_anndata.uns

    def test_word_already_a_saved_product_raises(self):
        sfc_anndata = _build_anndata(_SMURF_FILENAMES)
        sfc_anndata.uns[SFC_CONDITION_TOKENS_UNS_KEY] = {"S": "Sucrose"}
        with pytest.raises(ValueError, match="both as a product"):
            annotate_sfc_filename_columns(sfc_anndata, _SMURF_COLUMNS)

    def test_qc_flags_a_word_found_in_no_filename(self, capsys):
        annotate_sfc_filename_columns(
            _build_anndata(_SMURF_FILENAMES), {"smurf_status": {"S": "Smurf", "Nons": "Non_Smurf"}}
        )
        printed = capsys.readouterr().out
        assert "'Nons' (smurf_status)" in printed
        assert "is NaN for 2 file(s)" in printed


class TestAnnotateFilenameColumnsRoundTrip:
    """Columns and declaration survive write_h5ad / read_h5ad."""

    def test_round_trip(self, tmp_path):
        sfc_anndata = annotate_sfc_filename_columns(
            _build_anndata(_SMURF_FILENAMES), {**_SMURF_COLUMNS, "diet": {"CR": "Caloric_Restriction"}}
        )
        output_path = tmp_path / "filename_columns.h5ad"
        sfc_anndata.write_h5ad(output_path)
        reloaded = anndata.read_h5ad(output_path)
        assert _one_value_per_file(reloaded, "smurf_status") == ["Smurf", "Non_Smurf", "Smurf", None]
        assert reloaded.uns[SFC_FILENAME_COLUMNS_UNS_KEY] == {
            **_SMURF_COLUMNS, "diet": {"CR": "Caloric_Restriction"},
        }

    def test_categories_stable_after_round_trip(self, tmp_path):
        """Annotating again a reloaded file gives the same categories."""
        columns = {"severity": {"Mild": "Mild", "Severe": "Severe", "Absent": "Absent"}}
        filenames = ["FLY_001_Mild_DR.fcs", "FLY_002_Severe_DR.fcs", "FLY_003_Absent_DR.fcs"]
        sfc_anndata = annotate_sfc_filename_columns(_build_anndata(filenames), columns)
        categories_before = list(sfc_anndata.obs["severity"].cat.categories)
        output_path = tmp_path / "severity.h5ad"
        sfc_anndata.write_h5ad(output_path)
        reloaded = annotate_sfc_filename_columns(anndata.read_h5ad(output_path))
        assert list(reloaded.obs["severity"].cat.categories) == categories_before


# ---------------------------------------------------------------------------
# Interplay with annotate_sfc_conditions
# ---------------------------------------------------------------------------


class TestConditionsRecogniseDeclaredColumns:
    """The conditions step knows the declared words and explains how to declare one."""

    def test_declared_words_not_reported(self):
        sfc_anndata = annotate_sfc_filename_columns(_build_anndata(_SMURF_FILENAMES), _SMURF_COLUMNS)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            annotate_sfc_conditions(sfc_anndata)
        assert (sfc_anndata.obs["condition"] == "No_addition").all()
        assert (sfc_anndata.obs["number_of_additions"] == 0).all()

    def test_undeclared_words_reported(self):
        with pytest.warns(UserWarning, match="'NonS' in 1 file"):
            annotate_sfc_conditions(_build_anndata(_SMURF_FILENAMES))

    def test_warning_shows_how_to_declare_a_column(self):
        with pytest.warns(UserWarning) as caught:
            annotate_sfc_conditions(_build_anndata(_SMURF_FILENAMES))
        message = str(caught[0].message)
        assert "filename_columns = {" in message
        assert '"<column_name>": {' in message
        assert '"S": "<value stored in .obs>"' in message
        assert '"NonS": "<value stored in .obs>"' in message
        assert "ingest_sfc_folder(..., filename_columns=filename_columns)" in message
        assert "annotate_sfc_filename_columns(" in message
        assert "diet, age_group, treatment" in message

    def test_warning_lists_declared_columns(self):
        sfc_anndata = _build_anndata(_SMURF_FILENAMES + ["FLY_005_D02_Female_DR.fcs"])
        annotate_sfc_filename_columns(sfc_anndata, _SMURF_COLUMNS)
        with pytest.warns(UserWarning) as caught:
            annotate_sfc_conditions(sfc_anndata)
        message = str(caught[0].message)
        assert "'Female' in 1 file" in message
        assert '"smurf_status": {' in message
        assert '"S": "Smurf",' in message
        assert "'S' in" not in message

    def test_warning_suggests_a_declared_column_word(self):
        sfc_anndata = _build_anndata(["FLY_001_D02_Nons_DR.fcs", "FLY_002_D02_NonS_DR.fcs"])
        annotate_sfc_filename_columns(sfc_anndata, _SMURF_COLUMNS)
        with pytest.warns(UserWarning, match="Did you mean 'NonS'"):
            annotate_sfc_conditions(sfc_anndata)

    def test_product_that_is_a_column_word_raises(self):
        sfc_anndata = annotate_sfc_filename_columns(_build_anndata(_SMURF_FILENAMES), _SMURF_COLUMNS)
        with pytest.raises(ValueError, match="both as a product"):
            annotate_sfc_conditions(sfc_anndata, {"S": "Sucrose"})

    def test_column_words_and_products_together(self):
        filenames = ["FLY_001_D02_S_DR.fcs", "FLY_001_D02_S_PGM_ADP_30min_DR.fcs"]
        sfc_anndata = annotate_sfc_filename_columns(_build_anndata(filenames), _SMURF_COLUMNS)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            annotate_sfc_conditions(sfc_anndata, {"PGM": "Pyruvate", "ADP": "Adenosine diphosphate"})
        assert _one_value_per_file(sfc_anndata, "condition") == ["No_addition", "PGM_ADP_30min"]
        assert _one_value_per_file(sfc_anndata, "number_of_additions") == [0, 2]
        assert _one_value_per_file(sfc_anndata, "smurf_status") == ["Smurf", "Smurf"]
