"""
Tests for src/mito_marker/analysis/_plot_context.py

Covers get_run_context_console_text() and get_run_context_footer_text() under:
  - No analysis_config in .uns (graceful fallback)
  - Empty analysis_config (graceful fallback)
  - Typical SFC subset: specie list, diet list, layer, feature selection
  - Species shown as "all" string
  - Condition key instead of diet key
  - Missing condition key (no diet, no condition)
  - Layer = None (raw .X)
  - Feature selection = None
"""

import anndata
import numpy as np
import pytest

from mito_marker.analysis._plot_context import (
    get_run_context_console_text,
    get_run_context_footer_text,
)


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------


def _make_empty_anndata() -> anndata.AnnData:
    """Return a minimal AnnData with no analysis_config."""
    return anndata.AnnData(X=np.zeros((3, 2)))


def _make_anndata_with_config(config: dict) -> anndata.AnnData:
    """Return an AnnData with .uns['analysis_config'] set to config."""
    adata = anndata.AnnData(X=np.zeros((3, 2)))
    adata.uns["analysis_config"] = config
    return adata


# ---------------------------------------------------------------------------
# get_run_context_console_text
# ---------------------------------------------------------------------------


class TestGetRunContextConsoleText:
    def test_no_analysis_config_shows_na(self) -> None:
        adata = _make_empty_anndata()
        result = get_run_context_console_text(adata)
        assert "N/A" in result
        # All four fields default to N/A or their own fallbacks when config is absent
        assert "Species" in result
        assert "Subset selected" in result
        assert "Layer (norm)" in result
        assert "Feature sel." in result

    def test_output_is_multiline(self) -> None:
        adata = _make_empty_anndata()
        result = get_run_context_console_text(adata)
        assert "\n" in result

    def test_header_present(self) -> None:
        adata = _make_empty_anndata()
        result = get_run_context_console_text(adata)
        assert "PLOT CONTEXT" in result

    def test_separator_lines_present(self) -> None:
        adata = _make_empty_anndata()
        result = get_run_context_console_text(adata)
        assert "=" * 20 in result  # At least some of the separator chars are there

    def test_species_list_joined_with_comma(self) -> None:
        adata = _make_anndata_with_config({
            "selection": {"specie": ["MNMS", "WORM"]},
            "active_layer": None,
            "active_selection": None,
        })
        result = get_run_context_console_text(adata)
        assert "MNMS,WORM" in result

    def test_species_all_string_shown_as_all(self) -> None:
        adata = _make_anndata_with_config({
            "selection": {"specie": "all"},
            "active_layer": None,
            "active_selection": None,
        })
        result = get_run_context_console_text(adata)
        assert "all" in result

    def test_species_single_value_list(self) -> None:
        adata = _make_anndata_with_config({
            "selection": {"specie": ["MNMS"]},
            "active_layer": None,
            "active_selection": None,
        })
        result = get_run_context_console_text(adata)
        assert "MNMS" in result

    def test_diet_key_shown_in_subset(self) -> None:
        adata = _make_anndata_with_config({
            "selection": {"specie": "all", "diet": ["IF"]},
            "active_layer": None,
            "active_selection": None,
        })
        result = get_run_context_console_text(adata)
        assert "diet=IF" in result

    def test_diet_all_not_shown_in_subset(self) -> None:
        adata = _make_anndata_with_config({
            "selection": {"specie": "all", "diet": "all"},
            "active_layer": None,
            "active_selection": None,
        })
        result = get_run_context_console_text(adata)
        # "all" diet should be omitted; subset should fall back to "all"
        assert "diet=all" not in result

    def test_diet_auto_resolves_to_real_obs_value(self) -> None:
        # When diet was auto-skipped, read the actual value from .obs
        import pandas as pd

        adata = _make_anndata_with_config({
            "selection": {"specie": "all", "diet": "auto (only one value present)"},
            "active_layer": None,
            "active_selection": None,
        })
        adata.obs["diet"] = pd.Categorical(["AL", "AL", "AL"])
        result = get_run_context_console_text(adata)
        # "auto" text must not appear; the real value must
        assert "auto" not in result
        assert "diet=AL" in result

    def test_species_auto_resolves_to_real_obs_value(self) -> None:
        import pandas as pd

        adata = _make_anndata_with_config({
            "selection": {"specie": "auto (only one value present)"},
            "active_layer": None,
            "active_selection": None,
        })
        adata.obs["specie"] = pd.Categorical(["WORM", "WORM", "WORM"])
        result = get_run_context_console_text(adata)
        assert "auto" not in result
        assert "WORM" in result

    def test_condition_key_shown_when_no_diet(self) -> None:
        adata = _make_anndata_with_config({
            "selection": {"specie": "all", "condition": ["Young"]},
            "active_layer": None,
            "active_selection": None,
        })
        result = get_run_context_console_text(adata)
        assert "condition=Young" in result

    def test_no_diet_no_condition_shows_all_as_subset(self) -> None:
        adata = _make_anndata_with_config({
            "selection": {"specie": "all"},
            "active_layer": None,
            "active_selection": None,
        })
        result = get_run_context_console_text(adata)
        # Subset should say "all" when no condition key is present
        assert "all" in result

    def test_layer_shown_when_set(self) -> None:
        adata = _make_anndata_with_config({
            "selection": {},
            "active_layer": "arcsinh__zscore_col",
            "active_selection": None,
        })
        result = get_run_context_console_text(adata)
        assert "arcsinh__zscore_col" in result

    def test_layer_raw_when_none(self) -> None:
        adata = _make_anndata_with_config({
            "selection": {},
            "active_layer": None,
            "active_selection": None,
        })
        result = get_run_context_console_text(adata)
        assert "raw .X" in result

    def test_feature_selection_shown_when_set(self) -> None:
        adata = _make_anndata_with_config({
            "selection": {},
            "active_layer": None,
            "active_selection": "PCALoadings",
        })
        result = get_run_context_console_text(adata)
        assert "PCALoadings" in result

    def test_feature_selection_none_when_not_set(self) -> None:
        adata = _make_anndata_with_config({
            "selection": {},
            "active_layer": None,
            "active_selection": None,
        })
        result = get_run_context_console_text(adata)
        assert "None" in result

    def test_returns_string(self) -> None:
        adata = _make_empty_anndata()
        result = get_run_context_console_text(adata)
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# get_run_context_footer_text
# ---------------------------------------------------------------------------


class TestGetRunContextFooterText:
    def test_no_analysis_config_returns_string(self) -> None:
        adata = _make_empty_anndata()
        result = get_run_context_footer_text(adata)
        assert isinstance(result, str)

    def test_all_four_fields_present(self) -> None:
        adata = _make_empty_anndata()
        result = get_run_context_footer_text(adata)
        assert "Species:" in result
        assert "Subset:" in result
        assert "Layer:" in result
        assert "Feat.sel.:" in result

    def test_single_line(self) -> None:
        adata = _make_empty_anndata()
        result = get_run_context_footer_text(adata)
        assert "\n" not in result

    def test_fields_separated_by_pipe(self) -> None:
        adata = _make_empty_anndata()
        result = get_run_context_footer_text(adata)
        assert result.count("|") >= 3

    def test_species_value_in_footer(self) -> None:
        adata = _make_anndata_with_config({
            "selection": {"specie": ["MNMS"]},
            "active_layer": "arcsinh__zscore_col",
            "active_selection": "PCALoadings",
        })
        result = get_run_context_footer_text(adata)
        assert "MNMS" in result

    def test_layer_value_in_footer(self) -> None:
        adata = _make_anndata_with_config({
            "selection": {},
            "active_layer": "logicle__minmax_col",
            "active_selection": None,
        })
        result = get_run_context_footer_text(adata)
        assert "logicle__minmax_col" in result

    def test_feature_sel_value_in_footer(self) -> None:
        adata = _make_anndata_with_config({
            "selection": {},
            "active_layer": None,
            "active_selection": "MIM",
        })
        result = get_run_context_footer_text(adata)
        assert "MIM" in result

    def test_diet_condition_in_footer(self) -> None:
        adata = _make_anndata_with_config({
            "selection": {"specie": "all", "diet": ["AL", "IF"]},
            "active_layer": None,
            "active_selection": None,
        })
        result = get_run_context_footer_text(adata)
        assert "diet=AL,IF" in result

    def test_full_typical_run_footer(self) -> None:
        """Integration-style: typical SFC run with all fields populated."""
        adata = _make_anndata_with_config({
            "selection": {
                "specie": ["MNMS"],
                "diet": ["IF"],
                "FlowAI_Pass": ["True"],
                "age": "auto (only one value present)",
            },
            "active_layer": "arcsinh__zscore_col",
            "active_selection": "PCALoadings",
        })
        result = get_run_context_footer_text(adata)
        assert "MNMS" in result
        assert "diet=IF" in result
        assert "arcsinh__zscore_col" in result
        assert "PCALoadings" in result
        # age is not a condition key — should not appear regardless
        assert "age" not in result

    def test_diet_auto_in_footer_resolves_to_real_value(self) -> None:
        import pandas as pd

        adata = _make_anndata_with_config({
            "selection": {"specie": "all", "diet": "auto (only one value present)"},
            "active_layer": None,
            "active_selection": None,
        })
        adata.obs["diet"] = pd.Categorical(["IF", "IF", "IF"])
        result = get_run_context_footer_text(adata)
        assert "auto" not in result
        assert "diet=IF" in result
