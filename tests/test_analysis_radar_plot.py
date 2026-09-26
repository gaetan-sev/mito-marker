"""
test_analysis_radar_plot.py

Unit tests for mito_marker.analysis.radar_plot.

Tests verify:
  - plot_radar() runs without error on a synthetic AnnData (Agg backend).
  - Both group_by="subject_ID" and group_by="dilution" produce a figure.
  - plot_radar() raises ValueError for a missing group_by column.
  - The active layer is read correctly (layer vs. raw .X).
  - Feature selection filtering works: only selected channels are plotted.
  - _get_data_and_channels() returns the full matrix when no selection is active.
  - _build_color_map() uses tab10 for subject_ID, palette lookup for conditions.
  - _draw_scatter_cloud() does not crash on empty groups or when max_points
    is smaller than the group size.
  - _nested_group_aggregate() gives each individual equal weight regardless
    of its raw event count (ADR-004 / ADR-011).
  - _nested_group_aggregate() with a 2-D hierarchy also gives each COARSER
    group (e.g. species) equal weight regardless of how many finer groups
    (e.g. subjects) it has (ADR-011's multi-level case).
  - plot_radar(nest_aggregate_by=...) wires the nested aggregate into the actual
    plotted profile line and the legend title, for both a single column and
    a hierarchical list of columns.
"""

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend — no window required.

import numpy as np
import pandas as pd
import pytest
import anndata

from mito_marker.analysis.radar_plot import (
    _build_color_map,
    _draw_scatter_cloud,
    _get_data_and_channels,
    _get_group_labels,
    _nested_group_aggregate,
    _sort_channels_by_laser,
    _split_channels_by_suffix,
    plot_radar,
)
from mito_marker.analysis.colors import assign_color_palette, COLOR_PALETTE_KEY
from mito_marker.analysis.selection import _ANALYSIS_CONFIG_KEY


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

N_OBS = 80
N_VARS = 12


def _make_test_anndata(
    include_layer: bool = False,
    include_selection: bool = False,
    n_obs: int = N_OBS,
    n_vars: int = N_VARS,
) -> anndata.AnnData:
    """
    Build a minimal SFC-like AnnData for radar plot testing.

    .obs contains: subject_ID (3 subjects), dilution (Diluted/Not_Diluted).
    .X contains random float32 values in a cytometry-like range.
    """
    np.random.seed(99)
    x_matrix = np.abs(np.random.randn(n_obs, n_vars) * 5).astype(np.float32)
    obs = pd.DataFrame({
        "subject_ID": np.tile(["S001", "S002", "S003"], n_obs // 3 + 1)[:n_obs],
        "dilution": np.tile(["Diluted", "Not_Diluted"], n_obs // 2 + 1)[:n_obs],
    })
    # Use realistic cytometry channel names with -A/-H/-W suffixes (4 of each type).
    # n_vars must be a multiple of 3 for this split to be even.
    n_per_suffix = n_vars // 3
    suffixes = ["-A"] * n_per_suffix + ["-H"] * n_per_suffix + ["-W"] * n_per_suffix
    var = pd.DataFrame(index=[f"CH0{i % n_per_suffix}{suffix}" for i, suffix in enumerate(suffixes)])
    adata = anndata.AnnData(X=x_matrix, obs=obs, var=var)

    adata.uns[_ANALYSIS_CONFIG_KEY] = {
        "active_layer": None,
        "active_selection": None,
    }

    if include_layer:
        # Simulate a normalized layer (arcsinh + zscore).
        from sklearn.preprocessing import StandardScaler
        transformed = np.arcsinh(x_matrix / 150.0)
        normalized = StandardScaler().fit_transform(transformed).astype(np.float32)
        adata.layers["arcsinh__zscore_col"] = normalized
        adata.uns[_ANALYSIS_CONFIG_KEY]["active_layer"] = "arcsinh__zscore_col"

    if include_selection:
        # Mark the first 6 channels as selected (MIM method).
        adata.var["MIM_score"] = np.arange(n_vars, dtype=float)[::-1]
        adata.var["is_selected_MIM"] = adata.var["MIM_score"] >= (n_vars - 6)
        adata.uns[_ANALYSIS_CONFIG_KEY]["active_selection"] = "MIM"

    return adata


@pytest.fixture
def test_anndata() -> anndata.AnnData:
    adata = _make_test_anndata()
    assign_color_palette(adata)
    return adata


@pytest.fixture
def test_anndata_with_layer() -> anndata.AnnData:
    adata = _make_test_anndata(include_layer=True)
    assign_color_palette(adata)
    return adata


@pytest.fixture
def test_anndata_with_selection() -> anndata.AnnData:
    adata = _make_test_anndata(include_selection=True)
    assign_color_palette(adata)
    return adata


# ---------------------------------------------------------------------------
# Tests: plot_radar — integration (Agg backend, no window)
# ---------------------------------------------------------------------------


class TestPlotRadarIntegration:
    """Integration tests — verify that plot_radar() runs without errors."""

    def test_group_by_subject_id_creates_figure(self, test_anndata):
        """plot_radar with group_by='subject_ID' must create one figure per suffix (A, H, W)."""
        import matplotlib.pyplot as plt
        n_figs_before = len(plt.get_fignums())
        plot_radar(test_anndata, group_by="subject_ID")
        assert len(plt.get_fignums()) == n_figs_before + 3
        plt.close("all")

    def test_group_by_dilution_creates_figure(self, test_anndata):
        """plot_radar with group_by='dilution' must create one figure per suffix (A, H, W)."""
        import matplotlib.pyplot as plt
        n_figs_before = len(plt.get_fignums())
        plot_radar(test_anndata, group_by="dilution")
        assert len(plt.get_fignums()) == n_figs_before + 3
        plt.close("all")

    def test_invalid_group_by_raises_value_error(self, test_anndata):
        with pytest.raises(ValueError, match="not found in .obs"):
            plot_radar(test_anndata, group_by="nonexistent_column")

    def test_custom_title_accepted(self, test_anndata):
        import matplotlib.pyplot as plt
        plot_radar(test_anndata, group_by="dilution", title="My Custom Title")
        plt.close("all")

    def test_max_points_per_group_respected(self, test_anndata):
        """Passing a very small max_points_per_group must not raise an error."""
        import matplotlib.pyplot as plt
        plot_radar(test_anndata, group_by="subject_ID", max_points_per_group=5)
        plt.close("all")

    def test_with_normalized_layer_runs(self, test_anndata_with_layer):
        """plot_radar on an AnnData with an active normalized layer must succeed."""
        import matplotlib.pyplot as plt
        plot_radar(test_anndata_with_layer, group_by="dilution")
        plt.close("all")

    def test_with_feature_selection_runs(self, test_anndata_with_selection):
        """plot_radar with active feature selection must succeed (only selected channels)."""
        import matplotlib.pyplot as plt
        plot_radar(test_anndata_with_selection, group_by="dilution")
        plt.close("all")


# ---------------------------------------------------------------------------
# Tests: _nested_group_aggregate — ADR-004 / ADR-011 nested mean helper
# ---------------------------------------------------------------------------


class TestNestedGroupAggregate:
    """Tests for _nested_group_aggregate() — per-individual-then-across-individuals."""

    def test_equal_weight_per_individual_regardless_of_event_count(self):
        """
        Subject S1 has 5 events at value 0.0, subject S2 has 45 events at
        value 10.0. The nested mean must be the midpoint (5.0) — each
        subject counting once — NOT the pooled mean (~9.0), which would be
        skewed toward S2 purely because it has 9x more events.
        """
        matrix = np.concatenate([
            np.zeros((5, 1)),
            np.full((45, 1), 10.0),
        ])
        individual_values = np.array(["S1"] * 5 + ["S2"] * 45)

        nested_mean = _nested_group_aggregate(matrix, individual_values, np.nanmean)
        pooled_mean = np.nanmean(matrix, axis=0)

        assert nested_mean[0] == pytest.approx(5.0)
        assert pooled_mean[0] == pytest.approx(9.0)
        assert nested_mean[0] != pytest.approx(pooled_mean[0])

    def test_matches_manual_mean_of_per_individual_means(self):
        np.random.seed(3)
        matrix = np.random.randn(30, 4)
        individual_values = np.array(["I1"] * 10 + ["I2"] * 12 + ["I3"] * 8)

        nested_mean = _nested_group_aggregate(matrix, individual_values, np.nanmean)

        manual_means = np.stack([
            matrix[individual_values == "I1"].mean(axis=0),
            matrix[individual_values == "I2"].mean(axis=0),
            matrix[individual_values == "I3"].mean(axis=0),
        ])
        expected = manual_means.mean(axis=0)
        np.testing.assert_allclose(nested_mean, expected)

    def test_works_with_median_aggregate(self):
        matrix = np.concatenate([
            np.zeros((5, 1)),
            np.full((45, 1), 10.0),
        ])
        individual_values = np.array(["S1"] * 5 + ["S2"] * 45)
        nested_median = _nested_group_aggregate(matrix, individual_values, np.nanmedian)
        assert nested_median[0] == pytest.approx(5.0)

    def test_ignores_rows_with_missing_individual_label(self):
        matrix = np.array([[0.0], [10.0], [100.0]])
        individual_values = np.array(["S1", "S2", None], dtype=object)
        nested_mean = _nested_group_aggregate(matrix, individual_values, np.nanmean)
        # The row with a missing individual label (value 100.0) must be excluded.
        assert nested_mean[0] == pytest.approx(5.0)

    def test_falls_back_to_pooled_aggregate_when_all_labels_missing(self):
        matrix = np.array([[0.0], [10.0]])
        individual_values = np.array([None, None], dtype=object)
        nested_mean = _nested_group_aggregate(matrix, individual_values, np.nanmean)
        assert nested_mean[0] == pytest.approx(5.0)  # pooled fallback == plain mean here

    def test_two_level_hierarchy_gives_each_coarser_group_equal_weight(self):
        """
        Species "SpA" has 1 subject (40 events at 0.0); species "SpB" has 3
        subjects (5 events at 10.0 each, 15 events total). Three different
        numbers should all be distinguishable:
          - pooled mean (no nesting):            (40*0 + 15*10) / 55 ≈ 2.73
          - single-level nesting by subject only:  (0 + 10 + 10 + 10) / 4 = 7.5
            (SpB dominates 3-to-1 purely because it has more subjects)
          - two-level nesting (species, then subject): mean(mean(SpA subjects),
            mean(SpB subjects)) = mean(0.0, 10.0) = 5.0 (each species counts once)
        """
        matrix = np.concatenate([
            np.zeros((40, 1)),
            np.full((15, 1), 10.0),
        ])
        species = np.array(["SpA"] * 40 + ["SpB"] * 15)
        subject = np.array(
            ["SpA-1"] * 40
            + ["SpB-1"] * 5 + ["SpB-2"] * 5 + ["SpB-3"] * 5
        )

        pooled_mean = np.nanmean(matrix, axis=0)
        single_level = _nested_group_aggregate(matrix, subject, np.nanmean)
        two_level = _nested_group_aggregate(
            matrix, np.stack([species, subject], axis=1), np.nanmean
        )

        assert pooled_mean[0] == pytest.approx(150 / 55)
        assert single_level[0] == pytest.approx(7.5)
        assert two_level[0] == pytest.approx(5.0)

    def test_two_level_hierarchy_matches_manual_nested_mean(self):
        np.random.seed(7)
        matrix = np.random.randn(60, 3)
        species = np.array(["Sp1"] * 30 + ["Sp2"] * 30)
        subject = np.array(
            ["Sp1-A"] * 10 + ["Sp1-B"] * 20
            + ["Sp2-A"] * 15 + ["Sp2-B"] * 15
        )
        hierarchy_values = np.stack([species, subject], axis=1)

        nested_mean = _nested_group_aggregate(matrix, hierarchy_values, np.nanmean)

        per_subject_means = {
            subject_id: matrix[subject == subject_id].mean(axis=0)
            for subject_id in ["Sp1-A", "Sp1-B", "Sp2-A", "Sp2-B"]
        }
        sp1_mean = np.stack([per_subject_means["Sp1-A"], per_subject_means["Sp1-B"]]).mean(axis=0)
        sp2_mean = np.stack([per_subject_means["Sp2-A"], per_subject_means["Sp2-B"]]).mean(axis=0)
        expected = np.stack([sp1_mean, sp2_mean]).mean(axis=0)

        np.testing.assert_allclose(nested_mean, expected)


# ---------------------------------------------------------------------------
# Tests: plot_radar(nest_aggregate_by=...) — integration
# ---------------------------------------------------------------------------


def _make_imbalanced_species_anndata() -> anndata.AnnData:
    """
    Single group ("all_cond"), two species with very unequal subject counts:
    species SpA has 1 subject (40 events at 0.0), species SpB has 3 subjects
    (5 events at 10.0 each). Used to check that a hierarchical
    nest_aggregate_by=["specie", "subject_ID"] gives each SPECIES equal
    weight, not just each subject — see TestNestedGroupAggregate's
    test_two_level_hierarchy_gives_each_coarser_group_equal_weight for the
    expected numbers (pooled ≈2.73, single-level=7.5, two-level=5.0).
    """
    n_spa, n_spb = 40, 15
    n_vars = 3
    feature_matrix = np.zeros((n_spa + n_spb, n_vars), dtype=np.float32)
    feature_matrix[:n_spa, 0] = 0.0
    feature_matrix[n_spa:, 0] = 10.0
    feature_matrix[:, 1:] = 1.0
    obs = pd.DataFrame({
        "cond": ["all_cond"] * (n_spa + n_spb),
        "specie": ["SpA"] * n_spa + ["SpB"] * n_spb,
        "subject_ID": (
            ["SpA-1"] * n_spa
            + ["SpB-1"] * 5 + ["SpB-2"] * 5 + ["SpB-3"] * 5
        ),
    })
    var = pd.DataFrame(index=["feat0", "feat1", "feat2"])
    adata = anndata.AnnData(X=feature_matrix, obs=obs, var=var)
    adata.uns[_ANALYSIS_CONFIG_KEY] = {"active_layer": None, "active_selection": None}
    assign_color_palette(adata)
    return adata


def _make_imbalanced_individual_anndata() -> anndata.AnnData:
    """
    Single group ("all_cond"), two subjects with very different event counts
    AND different means on the first channel — used to check that
    nest_aggregate_by changes the actually-plotted profile line, not just runs
    without crashing. Channel names deliberately avoid -A/-H/-W suffixes so
    plot_radar renders exactly one ("all"/TEM-style) figure, making the
    single resulting profile line easy to locate and inspect.
    """
    n_subject_1, n_subject_2 = 5, 45
    n_vars = 3
    feature_matrix = np.zeros((n_subject_1 + n_subject_2, n_vars), dtype=np.float32)
    feature_matrix[:n_subject_1, 0] = 0.0
    feature_matrix[n_subject_1:, 0] = 10.0
    # Other channels constant so they don't affect the y-axis margin logic.
    feature_matrix[:, 1:] = 1.0
    obs = pd.DataFrame({
        "cond": ["all_cond"] * (n_subject_1 + n_subject_2),
        "subject_ID": ["S1"] * n_subject_1 + ["S2"] * n_subject_2,
    })
    var = pd.DataFrame(index=["feat0", "feat1", "feat2"])
    adata = anndata.AnnData(X=feature_matrix, obs=obs, var=var)
    adata.uns[_ANALYSIS_CONFIG_KEY] = {"active_layer": None, "active_selection": None}
    assign_color_palette(adata)
    return adata


class TestPlotRadarNestAggregateBy:
    """Integration tests for plot_radar(nest_aggregate_by=...)."""

    def test_invalid_nest_aggregate_by_raises_value_error(self, test_anndata):
        with pytest.raises(ValueError, match="not found in .obs"):
            plot_radar(test_anndata, group_by="dilution", nest_aggregate_by="nonexistent_column")

    def test_runs_without_error(self, test_anndata):
        import matplotlib.pyplot as plt
        plot_radar(test_anndata, group_by="dilution", nest_aggregate_by="subject_ID")
        plt.close("all")

    def test_legend_title_mentions_nested_aggregation(self):
        import matplotlib.pyplot as plt
        plt.close("all")
        adata = _make_imbalanced_individual_anndata()
        plot_radar(adata, group_by="cond", nest_aggregate_by="subject_ID")
        fig = plt.figure(plt.get_fignums()[0])
        legend_title = fig.legends[0].get_title().get_text()
        assert "nested" in legend_title
        assert "subject_ID" in legend_title
        plt.close("all")

    def test_profile_line_uses_nested_mean_not_pooled_mean(self):
        """
        The core ADR-004/ADR-011 claim, verified against the actual rendered
        line: with nest_aggregate_by set, the group profile line on the first
        spoke (feat0) must sit at the nested mean (5.0) rather than the
        pooled mean (~9.0, skewed toward the 45-event subject).
        """
        import matplotlib.pyplot as plt
        plt.close("all")
        adata = _make_imbalanced_individual_anndata()
        plot_radar(adata, group_by="cond", nest_aggregate_by="subject_ID")
        fig = plt.figure(plt.get_fignums()[0])
        ax = fig.get_axes()[0]
        # Exactly one group ("all_cond") and no global mean/median lines
        # requested -> exactly one Line2D, the group's aggregate profile.
        assert len(ax.lines) == 1
        first_spoke_value = ax.lines[0].get_ydata()[0]
        assert first_spoke_value == pytest.approx(5.0, abs=0.05)
        plt.close("all")

    def test_pooled_profile_line_without_nest_aggregate_by_is_skewed(self):
        """Same data, nest_aggregate_by=None: profile line should sit near the
        pooled mean (~9.0), not the nested mean (5.0) — confirms the two
        modes actually produce different output."""
        import matplotlib.pyplot as plt
        plt.close("all")
        adata = _make_imbalanced_individual_anndata()
        plot_radar(adata, group_by="cond")
        fig = plt.figure(plt.get_fignums()[0])
        ax = fig.get_axes()[0]
        assert len(ax.lines) == 1
        first_spoke_value = ax.lines[0].get_ydata()[0]
        assert first_spoke_value == pytest.approx(9.0, abs=0.5)
        plt.close("all")

    def test_invalid_column_in_hierarchical_list_raises_value_error(self, test_anndata):
        with pytest.raises(ValueError, match="not found in .obs"):
            plot_radar(
                test_anndata,
                group_by="dilution",
                nest_aggregate_by=["subject_ID", "nonexistent_column"],
            )

    def test_hierarchical_legend_title_mentions_both_columns(self):
        import matplotlib.pyplot as plt
        plt.close("all")
        adata = _make_imbalanced_species_anndata()
        plot_radar(adata, group_by="cond", nest_aggregate_by=["specie", "subject_ID"])
        fig = plt.figure(plt.get_fignums()[0])
        legend_title = fig.legends[0].get_title().get_text()
        assert "nested" in legend_title
        assert "specie" in legend_title
        assert "subject_ID" in legend_title
        plt.close("all")

    def test_hierarchical_profile_line_gives_each_species_equal_weight(self):
        """
        The core claim from the user-reported scenario: with
        nest_aggregate_by=["specie", "subject_ID"], the group profile line
        must sit at the two-level nested mean (5.0) — each species counting
        once — NOT at the single-level nested mean (7.5, where SpB's 3
        subjects outweigh SpA's 1) and NOT at the pooled mean (~2.73).
        """
        import matplotlib.pyplot as plt
        plt.close("all")
        adata = _make_imbalanced_species_anndata()
        plot_radar(adata, group_by="cond", nest_aggregate_by=["specie", "subject_ID"])
        fig = plt.figure(plt.get_fignums()[0])
        ax = fig.get_axes()[0]
        assert len(ax.lines) == 1
        first_spoke_value = ax.lines[0].get_ydata()[0]
        assert first_spoke_value == pytest.approx(5.0, abs=0.05)
        plt.close("all")

    def test_single_level_nest_aggregate_by_still_skewed_by_species_imbalance(self):
        """Same data as the hierarchical test above, but nest_aggregate_by is
        just "subject_ID" (no species level): profile line should sit near
        7.5 (SpB's 3 subjects outweighing SpA's 1), not 5.0 — confirms the
        hierarchical and single-level modes actually differ."""
        import matplotlib.pyplot as plt
        plt.close("all")
        adata = _make_imbalanced_species_anndata()
        plot_radar(adata, group_by="cond", nest_aggregate_by="subject_ID")
        fig = plt.figure(plt.get_fignums()[0])
        ax = fig.get_axes()[0]
        assert len(ax.lines) == 1
        first_spoke_value = ax.lines[0].get_ydata()[0]
        assert first_spoke_value == pytest.approx(7.5, abs=0.05)
        plt.close("all")


# ---------------------------------------------------------------------------
# Tests: _get_data_and_channels
# ---------------------------------------------------------------------------


class TestGetDataAndChannels:
    """Unit tests for _get_data_and_channels()."""

    def test_returns_full_matrix_when_no_layer(self):
        adata = _make_test_anndata()
        matrix, channels = _get_data_and_channels(adata)
        assert matrix.shape == (N_OBS, N_VARS)
        assert len(channels) == N_VARS

    def test_returns_layer_matrix_when_active_layer_set(self):
        adata = _make_test_anndata()
        # Inject a layer manually.
        layer_data = (np.random.randn(N_OBS, N_VARS) * 2.0).astype(np.float32)
        adata.layers["test_layer"] = layer_data
        adata.uns[_ANALYSIS_CONFIG_KEY]["active_layer"] = "test_layer"
        matrix, channels = _get_data_and_channels(adata)
        np.testing.assert_array_equal(matrix, layer_data)

    def test_falls_back_to_x_when_layer_key_missing(self):
        """If active_layer is set but the layer doesn't exist, fall back to .X."""
        adata = _make_test_anndata()
        adata.uns[_ANALYSIS_CONFIG_KEY]["active_layer"] = "nonexistent_layer"
        matrix, channels = _get_data_and_channels(adata)
        np.testing.assert_array_equal(matrix, adata.X)

    def test_feature_selection_reduces_channels(self):
        adata = _make_test_anndata(include_selection=True)
        matrix, channels = _get_data_and_channels(adata)
        # 6 channels are marked as selected (the 6 highest MIM scores).
        assert matrix.shape[1] == 6
        assert len(channels) == 6

    def test_channel_names_match_var_index(self):
        adata = _make_test_anndata()
        _, channels = _get_data_and_channels(adata)
        assert channels == adata.var_names.tolist()

    def test_no_analysis_config_falls_back_to_x(self):
        """AnnData without analysis_config must still return .X."""
        adata = _make_test_anndata()
        del adata.uns[_ANALYSIS_CONFIG_KEY]
        matrix, channels = _get_data_and_channels(adata)
        np.testing.assert_array_equal(matrix, adata.X)


# ---------------------------------------------------------------------------
# Tests: _get_group_labels
# ---------------------------------------------------------------------------


class TestGetGroupLabels:
    """Unit tests for _get_group_labels()."""

    def test_returns_series_for_valid_column(self):
        adata = _make_test_anndata()
        labels = _get_group_labels(adata, "subject_ID")
        assert isinstance(labels, pd.Series)
        assert len(labels) == N_OBS

    def test_raises_for_missing_column(self):
        adata = _make_test_anndata()
        with pytest.raises(ValueError, match="not found in .obs"):
            _get_group_labels(adata, "nonexistent")

    def test_correct_values_returned(self):
        adata = _make_test_anndata()
        labels = _get_group_labels(adata, "dilution")
        assert set(labels.unique()) == {"Diluted", "Not_Diluted"}


# ---------------------------------------------------------------------------
# Tests: _build_color_map
# ---------------------------------------------------------------------------


class TestBuildColorMap:
    """Unit tests for _build_color_map()."""

    def test_subject_id_uses_tab10_colors(self):
        adata = _make_test_anndata()
        assign_color_palette(adata)
        labels = adata.obs["subject_ID"]
        color_map = _build_color_map(adata, "subject_ID", labels)
        # All three subjects should be present.
        assert set(color_map.keys()) == {"S001", "S002", "S003"}

    def test_condition_colors_come_from_palette(self):
        adata = _make_test_anndata()
        # Manually inject palette.
        adata.uns[COLOR_PALETTE_KEY] = {
            "Diluted": "#aabbcc",
            "Not_Diluted": "#ddeeff",
        }
        labels = adata.obs["dilution"]
        color_map = _build_color_map(adata, "dilution", labels)
        assert color_map["Diluted"] == "#aabbcc"
        assert color_map["Not_Diluted"] == "#ddeeff"

    def test_all_groups_have_colors(self):
        adata = _make_test_anndata()
        assign_color_palette(adata)
        labels = adata.obs["dilution"]
        color_map = _build_color_map(adata, "dilution", labels)
        assert len(color_map) == 2

    def test_unknown_condition_gets_auto_color(self):
        """If condition values are absent from the palette, distinct auto-colors are generated."""
        adata = _make_test_anndata()
        adata.uns[COLOR_PALETTE_KEY] = {}  # Empty palette — nothing assigned.
        labels = adata.obs["dilution"]
        color_map = _build_color_map(adata, "dilution", labels)
        # Both values must receive a color (not the gray fallback) and be distinct.
        assert len(color_map) == 2
        for color in color_map.values():
            assert color != "#999999", "Expected a distinct auto-color, not the gray fallback"
        assert len(set(color_map.values())) == 2, "Each group must receive a unique color"


# ---------------------------------------------------------------------------
# Tests: _draw_scatter_cloud
# ---------------------------------------------------------------------------


class TestDrawScatterCloud:
    """Unit tests for _draw_scatter_cloud() — must not raise on edge cases."""

    def _make_polar_ax(self):
        """Return a fresh polar Axes for testing."""
        import matplotlib.pyplot as plt
        _, ax = plt.subplots(subplot_kw={"projection": "polar"})
        return ax

    def test_basic_scatter_does_not_raise(self):
        import matplotlib.pyplot as plt
        ax = self._make_polar_ax()
        matrix = np.random.randn(50, 6).astype(np.float32)
        angles = [2 * np.pi * i / 6 for i in range(6)]
        _draw_scatter_cloud(ax, matrix, angles, color="#ff0000", max_points=300)
        plt.close("all")

    def test_max_points_limits_displayed_rows(self):
        """With max_points=10 and 100 events, only 10 should be shown per channel.
        We can't directly count scatter children easily, so just verify no error."""
        import matplotlib.pyplot as plt
        ax = self._make_polar_ax()
        matrix = np.random.randn(100, 4).astype(np.float32)
        angles = [2 * np.pi * i / 4 for i in range(4)]
        _draw_scatter_cloud(ax, matrix, angles, color="#0000ff", max_points=10)
        plt.close("all")

    def test_empty_group_does_not_raise(self):
        import matplotlib.pyplot as plt
        ax = self._make_polar_ax()
        matrix = np.empty((0, 6), dtype=np.float32)
        angles = [2 * np.pi * i / 6 for i in range(6)]
        _draw_scatter_cloud(ax, matrix, angles, color="#00ff00", max_points=300)
        plt.close("all")

    def test_nan_values_are_skipped(self):
        """Channels with all-NaN values must not cause an error."""
        import matplotlib.pyplot as plt
        ax = self._make_polar_ax()
        matrix = np.random.randn(20, 3).astype(np.float32)
        matrix[:, 1] = np.nan  # All NaN in channel 1.
        angles = [2 * np.pi * i / 3 for i in range(3)]
        _draw_scatter_cloud(ax, matrix, angles, color="#ff8800", max_points=300)
        plt.close("all")


# ---------------------------------------------------------------------------
# Tests: _split_channels_by_suffix
# ---------------------------------------------------------------------------


class TestSplitChannelsBySuffix:
    """Unit tests for _split_channels_by_suffix()."""

    def test_basic_fsc_channels(self):
        """Standard FSC channel names map correctly to A, H, W groups."""
        channels = ["FSC-A", "FSC-H", "FSC-W"]
        result = _split_channels_by_suffix(channels)
        assert result["A"] == [0]
        assert result["H"] == [1]
        assert result["W"] == [2]

    def test_non_suffix_channels_excluded(self):
        """Channels without a recognised suffix (Time, FlowAI) are excluded."""
        channels = ["FSC-A", "FSC-H", "FSC-W", "Time", "FlowAI"]
        result = _split_channels_by_suffix(channels)
        assert result["A"] == [0]
        assert result["H"] == [1]
        assert result["W"] == [2]

    def test_empty_input(self):
        """Empty channel list triggers the fallback 'all' group, which is also empty."""
        result = _split_channels_by_suffix([])
        assert list(result.keys()) == ["all"]
        assert result["all"] == []

    def test_only_a_channels(self):
        """Only -A channels present: H and W groups are empty."""
        channels = ["B1-A", "B2-A", "B3-A"]
        result = _split_channels_by_suffix(channels)
        assert result["A"] == [0, 1, 2]
        assert result["H"] == []
        assert result["W"] == []

    def test_no_matching_suffix(self):
        """Channels with no SFC-style suffix (e.g. TEM features) return a single 'all' group."""
        channels = ["CH00", "CH01", "CH02"]
        result = _split_channels_by_suffix(channels)
        assert list(result.keys()) == ["all"]
        assert result["all"] == [0, 1, 2]

    def test_tem_morphological_features_return_all_group(self):
        """TEM morphological feature names produce a single 'all' group containing every feature,
        reordered into contiguous Size / Shape / Intensity / Cristae Orientation / Other blocks."""
        from mito_marker.controlled_vocabulary import (
            TEM_FEATURE_CATEGORIES,
            TEM_FEATURE_CATEGORY_ORDER,
            TEM_FEATURE_COLUMNS,
        )

        result = _split_channels_by_suffix(TEM_FEATURE_COLUMNS)
        assert list(result.keys()) == ["all"]
        indices = result["all"]

        # Every feature is present exactly once, just reordered.
        assert sorted(indices) == list(range(len(TEM_FEATURE_COLUMNS)))

        # Categories form contiguous blocks in TEM_FEATURE_CATEGORY_ORDER (then
        # "Other"), and each feature's relative order within its own category
        # is preserved.
        category_order = [*TEM_FEATURE_CATEGORY_ORDER, "Other"]
        categories_seen = [
            TEM_FEATURE_CATEGORIES.get(TEM_FEATURE_COLUMNS[i], "Other")
            for i in indices
        ]
        ranks_seen = [category_order.index(category) for category in categories_seen]
        assert ranks_seen == sorted(ranks_seen)

        for category in category_order:
            original_positions = [
                i for i in range(len(TEM_FEATURE_COLUMNS))
                if TEM_FEATURE_CATEGORIES.get(TEM_FEATURE_COLUMNS[i], "Other") == category
            ]
            reordered_positions = [i for i in indices if i in original_positions]
            assert reordered_positions == original_positions

    def test_mixed_real_names(self):
        """Realistic mixed channel names are routed to the correct groups."""
        channels = ["B1-A", "SSC-B-H", "UV8-W", "Time"]
        result = _split_channels_by_suffix(channels)
        assert result["A"] == [0]
        assert result["H"] == [1]
        assert result["W"] == [2]
        # Time is at index 3 — must not appear in any group.
        for indices in result.values():
            assert 3 not in indices

    def test_index_order_preserved(self):
        """Indices are appended in iteration order (ascending) for channels not in the canonical list."""
        channels = ["CH00-A", "CH01-A", "CH00-H", "CH01-H", "CH00-W"]
        result = _split_channels_by_suffix(channels)
        assert result["A"] == [0, 1]
        assert result["H"] == [2, 3]
        assert result["W"] == [4]

    def test_channels_sorted_by_laser_prefix(self):
        """Real SFC channel names from different lasers are sorted FSC/SSC → UV → V → B → YG → R."""
        channels = ["R1-A", "UV1-A", "B1-A", "YG1-A", "V1-A"]
        result = _split_channels_by_suffix(channels)
        sorted_names = [channels[i] for i in result["A"]]
        assert sorted_names == ["UV1-A", "V1-A", "B1-A", "YG1-A", "R1-A"]


# ---------------------------------------------------------------------------
# Tests: mean profile line
# ---------------------------------------------------------------------------


class TestMeanProfileLine:
    """Verify that each radar figure carries a dashed global mean profile line."""

    def test_mean_profile_label_in_legend(self, test_anndata):
        """'Mean Profile' must appear in the figure legend when show_global_mean=True."""
        import matplotlib.pyplot as plt
        plt.close("all")
        plot_radar(test_anndata, group_by="subject_ID", show_global_mean=True)
        for fig_num in plt.get_fignums():
            fig = plt.figure(fig_num)
            # Legend is attached to the figure (fig.legend()), not to the axes.
            assert fig.legends, f"No figure-level legend found in figure {fig_num}."
            legend_texts = [text.get_text() for text in fig.legends[0].get_texts()]
            assert "Mean Profile" in legend_texts, (
                f"'Mean Profile' missing from legend in figure {fig_num}. "
                f"Found: {legend_texts}"
            )
        plt.close("all")

    def test_mean_profile_line_is_dashed(self, test_anndata):
        """At least one line per figure must have a dashed linestyle when show_global_mean=True."""
        import matplotlib.pyplot as plt
        plt.close("all")
        plot_radar(test_anndata, group_by="dilution", show_global_mean=True)
        for fig_num in plt.get_fignums():
            fig = plt.figure(fig_num)
            ax = fig.get_axes()[0]
            line_styles = [line.get_linestyle() for line in ax.lines]
            assert "--" in line_styles, (
                f"No dashed line found in figure {fig_num}. "
                f"Line styles present: {line_styles}"
            )
        plt.close("all")

    def test_median_profile_label_in_legend(self, test_anndata):
        """'Median Profile' must appear in the figure legend when show_global_median=True."""
        import matplotlib.pyplot as plt
        plt.close("all")
        plot_radar(test_anndata, group_by="subject_ID", show_global_median=True)
        for fig_num in plt.get_fignums():
            fig = plt.figure(fig_num)
            # Legend is attached to the figure (fig.legend()), not to the axes.
            assert fig.legends, f"No figure-level legend found in figure {fig_num}."
            legend_texts = [text.get_text() for text in fig.legends[0].get_texts()]
            assert "Median Profile" in legend_texts, (
                f"'Median Profile' missing from legend in figure {fig_num}. "
                f"Found: {legend_texts}"
            )
        plt.close("all")


# ---------------------------------------------------------------------------
# Tests: per-suffix figure content
# ---------------------------------------------------------------------------


class TestPerSuffixFigures:
    """Verify that each figure only contains channels of its suffix type."""

    def test_only_a_suffix_when_a_channels_present(self):
        """An AnnData with only -A channels must produce exactly one figure."""
        import matplotlib.pyplot as plt
        adata = _make_test_anndata()
        # Override var to contain only -A channels.
        a_only_var = pd.DataFrame(index=["CH00-A", "CH01-A", "CH02-A"])
        x_only = np.abs(np.random.randn(N_OBS, 3) * 5).astype(np.float32)
        a_only_adata = anndata.AnnData(X=x_only, obs=adata.obs, var=a_only_var)
        a_only_adata.uns[_ANALYSIS_CONFIG_KEY] = {
            "active_layer": None,
            "active_selection": None,
        }
        assign_color_palette(a_only_adata)
        plt.close("all")
        n_figs_before = len(plt.get_fignums())
        plot_radar(a_only_adata, group_by="dilution")
        assert len(plt.get_fignums()) == n_figs_before + 1
        plt.close("all")

    def test_a_figure_tick_labels_end_with_a(self, test_anndata):
        """All spoke labels in the -A figure must end with '-A' when strip_suffix=False."""
        import matplotlib.pyplot as plt
        plt.close("all")
        # strip_suffix=False preserves the original channel names on spokes.
        # Spoke labels are now drawn as ax.text() objects, not xticklabels.
        plot_radar(test_anndata, group_by="dilution", strip_suffix=False)
        # The first figure produced corresponds to the -A suffix.
        fig = plt.figure(plt.get_fignums()[0])
        ax = fig.get_axes()[0]
        spoke_labels = [t.get_text() for t in ax.texts]
        assert all(label.endswith("-A") for label in spoke_labels), (
            f"Expected all spoke labels to end with '-A', got: {spoke_labels}"
        )
        plt.close("all")


# ---------------------------------------------------------------------------
# Tests: _sort_channels_by_laser
# ---------------------------------------------------------------------------


class TestSortChannelsByLaser:
    """Unit tests for _sort_channels_by_laser()."""

    def test_uv_before_v_before_b_before_yg_before_r(self):
        """Channels from five lasers are sorted into canonical order."""
        channels = ["R1-A", "UV1-A", "B1-A", "YG1-A", "V1-A"]
        result = _sort_channels_by_laser(channels, list(range(5)))
        assert [channels[i] for i in result] == ["UV1-A", "V1-A", "B1-A", "YG1-A", "R1-A"]

    def test_unknown_channels_appended_last(self):
        """Channels with no recognised laser prefix sort after all known-prefix channels."""
        channels = ["B1-A", "CUSTOM-A"]
        result = _sort_channels_by_laser(channels, [0, 1])
        assert channels[result[0]] == "B1-A"
        assert channels[result[1]] == "CUSTOM-A"

    def test_empty_input_returns_empty(self):
        """Empty input list produces an empty result."""
        assert _sort_channels_by_laser([], []) == []

    def test_detector_order_preserved_within_same_laser(self):
        """Within one laser group, detector numbers are preserved in canonical order."""
        channels = ["UV8-A", "UV1-A", "UV3-A"]
        result = _sort_channels_by_laser(channels, [0, 1, 2])
        assert [channels[i] for i in result] == ["UV1-A", "UV3-A", "UV8-A"]
