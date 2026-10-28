"""
test_sfc_conditions.py

Unit tests for the SFC experimental conditions (ADR-019):
  - parse_condition_from_filename() in src/mito_marker/sfc/filename_parser.py
  - annotate_sfc_conditions() in src/mito_marker/sfc/conditions.py

Tests cover:
- The four tubes of an incremental protocol (none, PGM, PGM+ADP 1 min, 30 min)
- FlowJo-exported names and historical names give no unrecognised token
- Waiting times: units, case, conversion to minutes, reset by a new addition,
  bounds, a time written before any product
- Validation of the products declared in the notebook
- .obs columns, dtypes, protocol order of the categories, .uns record
- Products saved in .uns are reused and merged with new ones
- One UnrecognisedFilenameTokenWarning listing unrecognised words, with a typo
  suggestion, shown again when the same call is repeated, and silenceable
- .h5ad round trip
"""

import math
import subprocess
import sys
import textwrap
import warnings

import anndata
import numpy as np
import pandas as pd
import pytest

from mito_marker.controlled_vocabulary import SFC_CONDITION_TOKENS_UNS_KEY
from mito_marker.sfc.conditions import (
    UnrecognisedFilenameTokenWarning,
    annotate_sfc_conditions,
)
from mito_marker.sfc.filename_parser import parse_condition_from_filename

_PRODUCT_TOKENS: dict[str, str] = {
    "PGM": "Pyruvate + glutamate + malate (complex I substrates)",
    "ADP": "Adenosine diphosphate",
}

# The four tubes of one sample, as exported by FlowJo.
_PROTOCOL_FILENAMES: list[str] = [
    "export_WORM_001_D08_AL_DR_FlowAIGoodEvents_DR+.fcs",
    "export_WORM_001_D08_AL_PGM_DR_FlowAIGoodEvents_DR+.fcs",
    "export_WORM_001_D08_AL_PGM_ADP_1min_DR_FlowAIGoodEvents_DR+.fcs",
    "export_WORM_001_D08_AL_PGM_ADP_30min_DR_FlowAIGoodEvents_DR+.fcs",
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


# ---------------------------------------------------------------------------
# parse_condition_from_filename
# ---------------------------------------------------------------------------


class TestParseIncrementalProtocol:
    """The four tubes of the PGM → ADP protocol."""

    @pytest.mark.parametrize(
        "filename, condition, additions",
        [
            (_PROTOCOL_FILENAMES[0], "No_addition", 0),
            (_PROTOCOL_FILENAMES[1], "PGM", 1),
            (_PROTOCOL_FILENAMES[2], "PGM_ADP_1min", 2),
            (_PROTOCOL_FILENAMES[3], "PGM_ADP_30min", 2),
        ],
    )
    def test_condition_and_number_of_additions(self, filename, condition, additions):
        result = parse_condition_from_filename(filename, _PRODUCT_TOKENS)
        assert result["condition"] == condition
        assert result["number_of_additions"] == additions

    def test_products_in_filename_order(self):
        result = parse_condition_from_filename(_PROTOCOL_FILENAMES[3], _PRODUCT_TOKENS)
        assert result["products"] == ["PGM", "ADP"]

    def test_minutes_after_last_addition(self):
        assert parse_condition_from_filename(
            _PROTOCOL_FILENAMES[2], _PRODUCT_TOKENS
        )["minutes_after_last_addition"] == 1.0
        assert parse_condition_from_filename(
            _PROTOCOL_FILENAMES[3], _PRODUCT_TOKENS
        )["minutes_after_last_addition"] == 30.0

    def test_no_time_gives_nan(self):
        result = parse_condition_from_filename(_PROTOCOL_FILENAMES[1], _PRODUCT_TOKENS)
        assert math.isnan(result["minutes_after_last_addition"])

    def test_exported_names_have_no_unrecognised_token(self):
        for filename in _PROTOCOL_FILENAMES:
            result = parse_condition_from_filename(filename, _PRODUCT_TOKENS)
            assert result["unrecognised_tokens"] == [], filename

    def test_name_without_extension(self):
        result = parse_condition_from_filename("WORM_001_D08_AL_PGM_DR", _PRODUCT_TOKENS)
        assert result["condition"] == "PGM"


class TestParseHistoricalNames:
    """Names ingested before ADR-019 hold no product and no unknown word."""

    @pytest.mark.parametrize(
        "filename",
        [
            "good_events_MNMS_020_FlowAIGoodEvents_DeepRed.fcs",
            "good_events_MNMS_031_ND_FlowAIGoodEvents_DeepRed.fcs",
            "expor_good_events_MNMS_075_FlowAIGoodEvents_DeepRed.fcs",
            "export_WORM_001_D00_AL_NM_FlowAIGoodEvents_P1.fcs",
            "export_FLY_D02_AL_011_DR_bis_FlowAIGoodEvents_DR+.fcs",
            "export_export_MNMS_074_BIS_DR_FlowAIGoodEvents_DR+.fcs",
            "MOUSE_001_Young_Vehicule_DR.fcs",
        ],
    )
    def test_no_addition_and_nothing_unrecognised(self, filename):
        result = parse_condition_from_filename(filename, {})
        assert result["condition"] == "No_addition"
        assert result["number_of_additions"] == 0
        assert result["unrecognised_tokens"] == []


class TestParseWaitingTimes:
    """Waiting-time tokens: units, conversion and position."""

    @pytest.mark.parametrize(
        "time_token, expected_minutes, expected_label",
        [
            ("90s", 1.5, "PGM_90s"),
            ("30min", 30.0, "PGM_30min"),
            ("30MIN", 30.0, "PGM_30min"),
            ("2h", 120.0, "PGM_2h"),
        ],
    )
    def test_units_converted_to_minutes(self, time_token, expected_minutes, expected_label):
        result = parse_condition_from_filename(f"WORM_001_PGM_{time_token}_DR", _PRODUCT_TOKENS)
        assert result["minutes_after_last_addition"] == pytest.approx(expected_minutes)
        assert result["condition"] == expected_label

    def test_new_addition_resets_the_waiting_time(self):
        result = parse_condition_from_filename("WORM_001_PGM_5min_ADP_DR", _PRODUCT_TOKENS)
        assert result["condition"] == "PGM_5min_ADP"
        assert math.isnan(result["minutes_after_last_addition"])

    def test_time_before_any_product_is_unrecognised(self):
        result = parse_condition_from_filename("WORM_001_30min_DR", _PRODUCT_TOKENS)
        assert result["unrecognised_tokens"] == ["30min"]
        assert result["condition"] == "No_addition"

    def test_time_out_of_bounds_raises(self):
        with pytest.raises(AssertionError, match="outside the expected bounds"):
            parse_condition_from_filename("WORM_001_PGM_2000min_DR", _PRODUCT_TOKENS)

    def test_product_added_twice_counts_twice(self):
        result = parse_condition_from_filename(
            "WORM_001_ADP_5min_ADP_30min_DR", _PRODUCT_TOKENS
        )
        assert result["number_of_additions"] == 2
        assert result["condition"] == "ADP_5min_ADP_30min"
        assert result["minutes_after_last_addition"] == 30.0


class TestParseUnrecognisedTokens:
    """Words that no rule recognises are returned, not stored."""

    def test_typo_is_unrecognised(self):
        result = parse_condition_from_filename("WORM_001_PMG_ADP_DR", _PRODUCT_TOKENS)
        assert result["unrecognised_tokens"] == ["PMG"]
        assert result["condition"] == "ADP"

    def test_undeclared_product_is_unrecognised(self):
        result = parse_condition_from_filename("WORM_001_PGM_DR", {})
        assert result["unrecognised_tokens"] == ["PGM"]
        assert result["condition"] == "No_addition"

    def test_product_tokens_are_case_sensitive(self):
        result = parse_condition_from_filename("WORM_001_pgm_DR", _PRODUCT_TOKENS)
        assert result["unrecognised_tokens"] == ["pgm"]


class TestConditionTokenValidation:
    """The products declared in the notebook are checked before use."""

    def test_not_a_dict_raises_type_error(self):
        with pytest.raises(TypeError, match="must be a dict"):
            parse_condition_from_filename("WORM_001_DR", ["PGM"])

    @pytest.mark.parametrize("bad_token", ["PG_M", "1PGM", "", "P GM"])
    def test_malformed_token_raises(self, bad_token):
        with pytest.raises(ValueError, match="Invalid product token"):
            parse_condition_from_filename("WORM_001_DR", {bad_token: "description"})

    @pytest.mark.parametrize("reserved_token", ["AL", "WORM", "DR", "ND", "D08", "export"])
    def test_token_with_a_meaning_raises(self, reserved_token):
        with pytest.raises(ValueError, match="already has a meaning"):
            parse_condition_from_filename("WORM_001_DR", {reserved_token: "description"})

    @pytest.mark.parametrize("bad_description", ["", "   ", None])
    def test_missing_description_raises(self, bad_description):
        with pytest.raises(ValueError, match="needs a description"):
            parse_condition_from_filename("WORM_001_DR", {"PGM": bad_description})

    def test_plus_and_minus_allowed(self):
        result = parse_condition_from_filename("WORM_001_Ca2+_DR", {"Ca2+": "Calcium"})
        assert result["condition"] == "Ca2+"


# ---------------------------------------------------------------------------
# annotate_sfc_conditions
# ---------------------------------------------------------------------------


class TestAnnotateColumns:
    """Columns written to .obs and their content."""

    @pytest.fixture
    def annotated(self):
        return annotate_sfc_conditions(_build_anndata(_PROTOCOL_FILENAMES), _PRODUCT_TOKENS)

    def test_returns_the_same_object(self):
        sfc_anndata = _build_anndata(_PROTOCOL_FILENAMES)
        assert annotate_sfc_conditions(sfc_anndata, _PRODUCT_TOKENS) is sfc_anndata

    def test_condition_categories_in_protocol_order(self, annotated):
        assert list(annotated.obs["condition"].cat.categories) == [
            "No_addition", "PGM", "PGM_ADP_1min", "PGM_ADP_30min",
        ]

    def test_condition_per_event(self, annotated):
        assert list(annotated.obs["condition"][::3]) == [
            "No_addition", "PGM", "PGM_ADP_1min", "PGM_ADP_30min",
        ]

    def test_number_of_additions(self, annotated):
        assert annotated.obs["number_of_additions"].dtype == np.int64
        assert list(annotated.obs["number_of_additions"][::3]) == [0, 1, 2, 2]

    def test_minutes_after_last_addition(self, annotated):
        minutes = annotated.obs["minutes_after_last_addition"]
        assert minutes.dtype == np.float32
        assert np.isnan(minutes.iloc[0]) and np.isnan(minutes.iloc[3])
        assert list(minutes[6::3]) == [1.0, 30.0]

    def test_one_presence_column_per_product(self, annotated):
        assert list(annotated.obs["with_PGM"][::3]) == [False, True, True, True]
        assert list(annotated.obs["with_ADP"][::3]) == [False, False, True, True]
        assert annotated.obs["with_ADP"].dtype == bool

    def test_tokens_saved_in_uns(self, annotated):
        assert annotated.uns[SFC_CONDITION_TOKENS_UNS_KEY] == _PRODUCT_TOKENS

    def test_x_unchanged(self, annotated):
        assert np.array_equal(annotated.X, np.ones((12, 2), dtype=np.float32))

    def test_categorical_source_filename(self):
        sfc_anndata = _build_anndata(_PROTOCOL_FILENAMES)
        sfc_anndata.obs["source_filename"] = sfc_anndata.obs["source_filename"].astype("category")
        annotate_sfc_conditions(sfc_anndata, _PRODUCT_TOKENS)
        assert list(sfc_anndata.obs["condition"][::3]) == [
            "No_addition", "PGM", "PGM_ADP_1min", "PGM_ADP_30min",
        ]

    def test_without_tokens_every_tube_is_no_addition(self):
        sfc_anndata = _build_anndata(_PROTOCOL_FILENAMES[:1])
        annotate_sfc_conditions(sfc_anndata)
        assert (sfc_anndata.obs["condition"] == "No_addition").all()
        assert sfc_anndata.uns[SFC_CONDITION_TOKENS_UNS_KEY] == {}

    def test_missing_source_filename_raises(self):
        sfc_anndata = _build_anndata(_PROTOCOL_FILENAMES)
        del sfc_anndata.obs["source_filename"]
        with pytest.raises(ValueError, match="source_filename"):
            annotate_sfc_conditions(sfc_anndata, _PRODUCT_TOKENS)

    def test_invalid_tokens_raise_before_writing(self):
        sfc_anndata = _build_anndata(_PROTOCOL_FILENAMES)
        with pytest.raises(ValueError):
            annotate_sfc_conditions(sfc_anndata, {"AL": "not a product"})
        assert "condition" not in sfc_anndata.obs.columns


class TestAnnotateSavedTokens:
    """Products saved in .uns are reused and completed by later calls."""

    def test_saved_tokens_used_when_none_given(self):
        sfc_anndata = _build_anndata(_PROTOCOL_FILENAMES)
        sfc_anndata.uns[SFC_CONDITION_TOKENS_UNS_KEY] = dict(_PRODUCT_TOKENS)
        annotate_sfc_conditions(sfc_anndata)
        assert list(sfc_anndata.obs["condition"].cat.categories)[-1] == "PGM_ADP_30min"

    def test_new_tokens_merged_with_saved_ones(self):
        sfc_anndata = _build_anndata(_PROTOCOL_FILENAMES)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # ADP is not declared yet
            annotate_sfc_conditions(sfc_anndata, {"PGM": _PRODUCT_TOKENS["PGM"]})
        annotate_sfc_conditions(sfc_anndata, {"ADP": _PRODUCT_TOKENS["ADP"]})
        assert sfc_anndata.uns[SFC_CONDITION_TOKENS_UNS_KEY] == _PRODUCT_TOKENS
        assert "with_PGM" in sfc_anndata.obs.columns
        assert "with_ADP" in sfc_anndata.obs.columns

    def test_new_description_replaces_saved_one(self):
        sfc_anndata = _build_anndata(_PROTOCOL_FILENAMES)
        sfc_anndata.uns[SFC_CONDITION_TOKENS_UNS_KEY] = dict(_PRODUCT_TOKENS)
        annotate_sfc_conditions(sfc_anndata, {"ADP": "ADP 1 mM"})
        assert sfc_anndata.uns[SFC_CONDITION_TOKENS_UNS_KEY]["ADP"] == "ADP 1 mM"


class TestAnnotateWarning:
    """One UserWarning explains how to declare an unrecognised word."""

    _FILENAMES_WITH_TYPO = _PROTOCOL_FILENAMES + [
        "export_WORM_002_D08_AL_PMG_DR_FlowAIGoodEvents_DR+.fcs",
    ]

    def test_warning_names_word_and_file(self):
        sfc_anndata = _build_anndata(self._FILENAMES_WITH_TYPO)
        with pytest.warns(UserWarning, match="'PMG' in 1 file") as caught:
            annotate_sfc_conditions(sfc_anndata, _PRODUCT_TOKENS)
        message = str(caught[0].message)
        assert "export_WORM_002_D08_AL_PMG_DR_FlowAIGoodEvents_DR+.fcs" in message

    def test_warning_suggests_declared_token(self):
        sfc_anndata = _build_anndata(self._FILENAMES_WITH_TYPO)
        with pytest.warns(UserWarning, match="Did you mean 'PGM'"):
            annotate_sfc_conditions(sfc_anndata, _PRODUCT_TOKENS)

    def test_warning_shows_how_to_declare(self):
        sfc_anndata = _build_anndata(self._FILENAMES_WITH_TYPO)
        with pytest.warns(UserWarning) as caught:
            annotate_sfc_conditions(sfc_anndata, _PRODUCT_TOKENS)
        message = str(caught[0].message)
        assert '"PMG": "<what this product is>"' in message
        assert '"PGM": "Pyruvate + glutamate + malate (complex I substrates)"' in message
        assert "ingest_sfc_folder(..., condition_tokens=condition_tokens)" in message
        assert "annotate_sfc_conditions(" in message

    def test_single_warning_for_several_files(self):
        """PGM, ADP, and the two times (which then follow no declared product)."""
        sfc_anndata = _build_anndata(_PROTOCOL_FILENAMES)
        with pytest.warns(UserWarning, match="4 word") as caught:
            annotate_sfc_conditions(sfc_anndata, {})
        assert len([w for w in caught if issubclass(w.category, UserWarning)]) == 1

    def test_no_warning_when_everything_is_recognised(self):
        sfc_anndata = _build_anndata(_PROTOCOL_FILENAMES)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            annotate_sfc_conditions(sfc_anndata, _PRODUCT_TOKENS)

    def test_warning_is_a_user_warning_subclass(self):
        assert issubclass(UnrecognisedFilenameTokenWarning, UserWarning)

    def test_warning_repeated_on_second_call(self):
        """
        Python shows a warning once per location by default; this one must repeat,
        as when a notebook cell is run twice. Run in a fresh interpreter so that
        pytest's own warning filters do not interfere.
        """
        script = textwrap.dedent(
            f"""
            import anndata, numpy as np, pandas as pd
            from mito_marker.sfc.conditions import annotate_sfc_conditions
            filenames = {self._FILENAMES_WITH_TYPO!r}
            observation_dataframe = pd.DataFrame({{"source_filename": filenames}})
            observation_dataframe.index = observation_dataframe.index.astype(str)
            sfc_anndata = anndata.AnnData(
                X=np.ones((len(filenames), 1), dtype=np.float32), obs=observation_dataframe
            )
            for _ in range(2):
                annotate_sfc_conditions(sfc_anndata, {_PRODUCT_TOKENS!r})
            """
        )
        completed = subprocess.run(
            [sys.executable, "-c", script], capture_output=True, text=True, check=True
        )
        assert completed.stderr.count("UnrecognisedFilenameTokenWarning") == 2

    def test_warning_can_be_silenced(self):
        sfc_anndata = _build_anndata(self._FILENAMES_WITH_TYPO)
        with warnings.catch_warnings(record=True) as caught:
            warnings.filterwarnings("ignore", category=UnrecognisedFilenameTokenWarning)
            annotate_sfc_conditions(sfc_anndata, _PRODUCT_TOKENS)
        assert not [w for w in caught if "PMG" in str(w.message)]


class TestAnnotateH5adRoundTrip:
    """Columns and declared products survive write_h5ad / read_h5ad."""

    def test_round_trip(self, tmp_path):
        sfc_anndata = annotate_sfc_conditions(
            _build_anndata(_PROTOCOL_FILENAMES), _PRODUCT_TOKENS
        )
        output_path = tmp_path / "conditions.h5ad"
        sfc_anndata.write_h5ad(output_path)
        reloaded = anndata.read_h5ad(output_path)
        assert list(reloaded.obs["condition"].cat.categories) == list(
            sfc_anndata.obs["condition"].cat.categories
        )
        assert dict(reloaded.uns[SFC_CONDITION_TOKENS_UNS_KEY]) == _PRODUCT_TOKENS
        assert list(reloaded.obs["with_ADP"][::3]) == [False, False, True, True]

    def test_round_trip_without_tokens(self, tmp_path):
        sfc_anndata = annotate_sfc_conditions(_build_anndata(_PROTOCOL_FILENAMES[:1]))
        output_path = tmp_path / "no_tokens.h5ad"
        sfc_anndata.write_h5ad(output_path)
        reloaded = anndata.read_h5ad(output_path)
        assert (reloaded.obs["condition"] == "No_addition").all()
