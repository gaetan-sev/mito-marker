"""
test_analysis_channel_aggregation.py

Unit tests for mito_marker.analysis.channel_aggregation.aggregate_sfc_channels_by_color().

Tests verify:
  - Output .n_vars equals the number of (laser, pulse) groups.
  - .obs is identical to the original.
  - Aggregated .X values match manual np.mean of source columns.
  - active_layer is respected: aggregation uses the layer, not raw .X.
  - active_layer is reset to None in the result.
  - active_selection is reset to None in the result.
  - Variable names follow the "{LASER}-{PULSE}" format.
  - Variable order follows canonical laser order (FSC→SSC→UV→V→B→YG→R).
  - Channels with no pattern match are excluded with a warning.
  - ValueError raised when no channels match the pattern.
  - Non-analytical channels are excluded before aggregation.
  - FSC/SSC (single detector) are preserved as-is (mean of 1 = itself).
  - n_source_channels column in .var is correct.
  - .uns aggregation_source metadata is stored.
"""

import warnings

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest
import anndata

from mito_marker.analysis.channel_aggregation import (
    _LASER_GROUP_ORDER,
    aggregate_sfc_channels_by_color,
)
from mito_marker.analysis.selection import _ANALYSIS_CONFIG_KEY

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

N_OBS = 30


def _make_sfc_anndata_full(n_obs: int = N_OBS) -> anndata.AnnData:
    """
    Build a minimal SFC AnnData with a realistic subset of Cytek Aurora channels:
      FSC-A, FSC-H, FSC-W
      SSC-A, SSC-H, SSC-W
      UV1-A, UV2-A, UV1-H, UV2-H
      YG1-A, YG2-A, YG1-H, YG2-H
      R1-A, R2-A, R1-H, R2-H

    Total: 18 channels → expected 9 aggregated groups
    (FSC-A, FSC-H, FSC-W, SSC-A, SSC-H, SSC-W, UV-A, UV-H, YG-A, YG-H, R-A, R-H)
    = 12 aggregated channels
    """
    np.random.seed(0)
    channels = [
        "FSC-A", "FSC-H", "FSC-W",
        "SSC-A", "SSC-H", "SSC-W",
        "UV1-A", "UV2-A",
        "UV1-H", "UV2-H",
        "YG1-A", "YG2-A",
        "YG1-H", "YG2-H",
        "R1-A", "R2-A",
        "R1-H", "R2-H",
    ]
    n_vars = len(channels)
    x_matrix = np.random.randn(n_obs, n_vars).astype(np.float32) * 100 + 500

    obs = pd.DataFrame({
        "subject_ID": np.tile(["S001", "S002"], n_obs // 2 + 1)[:n_obs],
        "condition": np.tile(["Young", "Old"], n_obs // 2 + 1)[:n_obs],
    })
    var = pd.DataFrame(index=channels)
    adata = anndata.AnnData(X=x_matrix, obs=obs, var=var)
    adata.uns[_ANALYSIS_CONFIG_KEY] = {"active_layer": None, "active_selection": None}
    adata.uns["color_palette"] = {"Young": "#4daf4a", "Old": "#e41a1c"}
    return adata


def _make_sfc_anndata_with_layer(n_obs: int = N_OBS) -> anndata.AnnData:
    adata = _make_sfc_anndata_full(n_obs)
    layer_data = np.arcsinh(adata.X / 150.0).astype(np.float32)
    adata.layers["arcsinh__zscore_col"] = layer_data
    adata.uns[_ANALYSIS_CONFIG_KEY]["active_layer"] = "arcsinh__zscore_col"
    return adata


def _make_sfc_anndata_with_non_analytical(n_obs: int = N_OBS) -> anndata.AnnData:
    adata = _make_sfc_anndata_full(n_obs)
    adata.var["is_non_analytical"] = False
    # Mark FSC-W and SSC-W as non-analytical (unusual but tests the exclusion).
    adata.var.loc["FSC-W", "is_non_analytical"] = True
    adata.var.loc["SSC-W", "is_non_analytical"] = True
    return adata


# ---------------------------------------------------------------------------
# Output shape
# ---------------------------------------------------------------------------


class TestOutputShape:
    def test_n_vars_equals_expected_groups(self) -> None:
        adata = _make_sfc_anndata_full()
        result = aggregate_sfc_channels_by_color(adata)
        # FSC: A/H/W (3), SSC: A/H/W (3), UV: A/H (2), YG: A/H (2), R: A/H (2) = 12
        assert result.n_vars == 12

    def test_n_obs_unchanged(self) -> None:
        adata = _make_sfc_anndata_full()
        result = aggregate_sfc_channels_by_color(adata)
        assert result.n_obs == N_OBS

    def test_x_dtype_is_float32(self) -> None:
        adata = _make_sfc_anndata_full()
        result = aggregate_sfc_channels_by_color(adata)
        assert result.X.dtype == np.float32


# ---------------------------------------------------------------------------
# .obs integrity
# ---------------------------------------------------------------------------


class TestObsIntegrity:
    def test_obs_identical_to_original(self) -> None:
        adata = _make_sfc_anndata_full()
        result = aggregate_sfc_channels_by_color(adata)
        pd.testing.assert_frame_equal(result.obs, adata.obs)

    def test_obs_index_identical(self) -> None:
        adata = _make_sfc_anndata_full()
        result = aggregate_sfc_channels_by_color(adata)
        assert list(result.obs.index) == list(adata.obs.index)


# ---------------------------------------------------------------------------
# Variable names and order
# ---------------------------------------------------------------------------


class TestVariableNames:
    def test_var_names_format(self) -> None:
        adata = _make_sfc_anndata_full()
        result = aggregate_sfc_channels_by_color(adata)
        for var_name in result.var_names:
            parts = var_name.split("-")
            assert len(parts) == 2, f"Expected 'LASER-PULSE', got '{var_name}'"
            laser, pulse = parts
            assert laser in _LASER_GROUP_ORDER or laser in ("FSC", "SSC"), laser
            assert pulse in ("A", "H", "W"), pulse

    def test_laser_order_respected(self) -> None:
        adata = _make_sfc_anndata_full()
        result = aggregate_sfc_channels_by_color(adata)
        var_names = list(result.var_names)
        # FSC must come before SSC, UV, YG, R in the output.
        fsc_indices = [i for i, n in enumerate(var_names) if n.startswith("FSC")]
        uv_indices = [i for i, n in enumerate(var_names) if n.startswith("UV")]
        yg_indices = [i for i, n in enumerate(var_names) if n.startswith("YG")]
        assert max(fsc_indices) < min(uv_indices)
        assert max(uv_indices) < min(yg_indices)

    def test_pulse_order_within_laser(self) -> None:
        adata = _make_sfc_anndata_full()
        result = aggregate_sfc_channels_by_color(adata)
        var_names = list(result.var_names)
        uv_names = [n for n in var_names if n.startswith("UV")]
        # A must come before H.
        if "UV-A" in uv_names and "UV-H" in uv_names:
            assert var_names.index("UV-A") < var_names.index("UV-H")


# ---------------------------------------------------------------------------
# Aggregation math
# ---------------------------------------------------------------------------


class TestAggregationMath:
    def test_uv_a_equals_mean_of_uv1_a_and_uv2_a(self) -> None:
        adata = _make_sfc_anndata_full()
        result = aggregate_sfc_channels_by_color(adata)
        uv1_a = adata.X[:, list(adata.var_names).index("UV1-A")]
        uv2_a = adata.X[:, list(adata.var_names).index("UV2-A")]
        expected = np.mean(np.column_stack([uv1_a, uv2_a]), axis=1).astype(np.float32)
        actual = result.X[:, list(result.var_names).index("UV-A")]
        np.testing.assert_allclose(actual, expected, rtol=1e-5)

    def test_fsc_a_equals_original_fsc_a(self) -> None:
        """FSC-A has exactly one source channel → mean(1 value) = itself."""
        adata = _make_sfc_anndata_full()
        result = aggregate_sfc_channels_by_color(adata)
        original_fsc_a = adata.X[:, list(adata.var_names).index("FSC-A")]
        result_fsc_a = result.X[:, list(result.var_names).index("FSC-A")]
        np.testing.assert_allclose(result_fsc_a, original_fsc_a, rtol=1e-5)

    def test_r_a_equals_mean_of_r1_a_and_r2_a(self) -> None:
        adata = _make_sfc_anndata_full()
        result = aggregate_sfc_channels_by_color(adata)
        r1_a = adata.X[:, list(adata.var_names).index("R1-A")]
        r2_a = adata.X[:, list(adata.var_names).index("R2-A")]
        expected = np.mean(np.column_stack([r1_a, r2_a]), axis=1).astype(np.float32)
        actual = result.X[:, list(result.var_names).index("R-A")]
        np.testing.assert_allclose(actual, expected, rtol=1e-5)

    def test_n_source_channels_correct(self) -> None:
        adata = _make_sfc_anndata_full()
        result = aggregate_sfc_channels_by_color(adata)
        # UV-A should have n_source_channels = 2 (UV1-A, UV2-A)
        uv_a_row = result.var.loc["UV-A"]
        assert uv_a_row["n_source_channels"] == 2
        # FSC-A should have n_source_channels = 1
        fsc_a_row = result.var.loc["FSC-A"]
        assert fsc_a_row["n_source_channels"] == 1


# ---------------------------------------------------------------------------
# Layer handling
# ---------------------------------------------------------------------------


class TestLayerHandling:
    def test_active_layer_used_for_aggregation(self) -> None:
        adata = _make_sfc_anndata_with_layer()
        result = aggregate_sfc_channels_by_color(adata)
        # UV-A in result should equal mean of UV1-A and UV2-A from the arcsinh layer.
        layer = adata.layers["arcsinh__zscore_col"]
        uv1_a = layer[:, list(adata.var_names).index("UV1-A")]
        uv2_a = layer[:, list(adata.var_names).index("UV2-A")]
        expected = np.mean(np.column_stack([uv1_a, uv2_a]), axis=1).astype(np.float32)
        actual = result.X[:, list(result.var_names).index("UV-A")]
        np.testing.assert_allclose(actual, expected, rtol=1e-5)

    def test_active_layer_not_raw_x(self) -> None:
        """When a layer differs from .X, result should NOT equal raw .X averages."""
        adata = _make_sfc_anndata_with_layer()
        result = aggregate_sfc_channels_by_color(adata)
        # Build reference from raw .X
        uv1_a_raw = adata.X[:, list(adata.var_names).index("UV1-A")]
        uv2_a_raw = adata.X[:, list(adata.var_names).index("UV2-A")]
        raw_average = np.mean(np.column_stack([uv1_a_raw, uv2_a_raw]), axis=1)
        actual = result.X[:, list(result.var_names).index("UV-A")]
        # arcsinh-transformed values must differ from raw values.
        assert not np.allclose(actual, raw_average, rtol=1e-3)

    def test_active_layer_reset_to_none_in_result(self) -> None:
        adata = _make_sfc_anndata_with_layer()
        result = aggregate_sfc_channels_by_color(adata)
        assert result.uns[_ANALYSIS_CONFIG_KEY]["active_layer"] is None

    def test_active_selection_reset_to_none_in_result(self) -> None:
        adata = _make_sfc_anndata_full()
        adata.var["is_selected_MIM"] = True
        adata.uns[_ANALYSIS_CONFIG_KEY]["active_selection"] = "MIM"
        result = aggregate_sfc_channels_by_color(adata)
        assert result.uns[_ANALYSIS_CONFIG_KEY]["active_selection"] is None

    def test_result_has_empty_layers(self) -> None:
        adata = _make_sfc_anndata_with_layer()
        result = aggregate_sfc_channels_by_color(adata)
        assert len(result.layers) == 0


# ---------------------------------------------------------------------------
# Non-analytical exclusion
# ---------------------------------------------------------------------------


class TestNonAnalyticalExclusion:
    def test_non_analytical_channels_excluded(self) -> None:
        adata = _make_sfc_anndata_with_non_analytical()
        result = aggregate_sfc_channels_by_color(adata)
        # FSC-W and SSC-W are non-analytical → those pulse-type groups disappear
        # if they had no other W channels.
        var_names = list(result.var_names)
        assert "FSC-W" not in var_names
        assert "SSC-W" not in var_names

    def test_analytical_channels_still_present(self) -> None:
        adata = _make_sfc_anndata_with_non_analytical()
        result = aggregate_sfc_channels_by_color(adata)
        assert "FSC-A" in result.var_names
        assert "SSC-A" in result.var_names


# ---------------------------------------------------------------------------
# Edge cases and error handling
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_unmatched_channels_produce_warning(self) -> None:
        adata = _make_sfc_anndata_full()
        # Add a non-matching channel name.
        extra_matrix = np.zeros((N_OBS, 1), dtype=np.float32)
        extra_var = pd.DataFrame(index=["Time"])
        adata_extra = anndata.AnnData(
            X=np.hstack([adata.X, extra_matrix]),
            obs=adata.obs.copy(),
            var=pd.concat([adata.var, extra_var]),
            uns=adata.uns,
        )
        with warnings.catch_warnings(record=True) as warning_list:
            warnings.simplefilter("always")
            result = aggregate_sfc_channels_by_color(adata_extra)
        assert any("Time" in str(w.message) for w in warning_list)

    def test_no_matching_channels_raises(self) -> None:
        obs = pd.DataFrame({"condition": ["A"] * 5})
        var = pd.DataFrame(index=["FeatureX", "FeatureY", "FeatureZ"])
        x_matrix = np.ones((5, 3), dtype=np.float32)
        adata = anndata.AnnData(X=x_matrix, obs=obs, var=var)
        adata.uns[_ANALYSIS_CONFIG_KEY] = {"active_layer": None, "active_selection": None}
        with pytest.raises(ValueError, match="No channels matched"):
            aggregate_sfc_channels_by_color(adata)

    def test_uns_aggregation_source_stored(self) -> None:
        adata = _make_sfc_anndata_full()
        result = aggregate_sfc_channels_by_color(adata)
        assert "aggregation_source" in result.uns
        meta = result.uns["aggregation_source"]
        assert meta["method"] == "aggregate_sfc_channels_by_color"
        assert meta["n_source_channels"] == adata.n_vars
        assert meta["n_aggregated_channels"] == result.n_vars

    def test_original_anndata_not_mutated(self) -> None:
        adata = _make_sfc_anndata_full()
        original_x = adata.X.copy()
        original_n_vars = adata.n_vars
        _ = aggregate_sfc_channels_by_color(adata)
        assert adata.n_vars == original_n_vars
        np.testing.assert_array_equal(adata.X, original_x)
