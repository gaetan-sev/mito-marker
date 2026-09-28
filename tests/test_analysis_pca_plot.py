"""
test_analysis_pca_plot.py

Unit tests for mito_marker.analysis.pca_plot.

Tests verify:
  - compute_pca() populates .obsm['X_pca'] with the correct shape.
  - .uns['pca_loadings'] has shape (n_channels, n_components).
  - .uns['pca_explained_variance_ratio'] sums to <= 1.0.
  - compute_pca() is cached: a second call with the same AnnData skips refitting.
  - plot_pca_scatter() runs without error (Agg backend).
  - plot_pca_scatter() raises ValueError for a missing group_by column.
  - plot_pca_scatter() raises ValueError when PCA is not computed.
  - plot_pca_biplot() runs without error (Agg backend).
  - get_top_loading_channels() returns the correct number and ranking.
  - _get_data_and_channels() respects active layer and feature selection.
  - _compute_nested_equal_weights() gives equal total weight per species and
    per subject regardless of raw event counts (ADR-011).
  - compute_pca(weight_by=...) stores weighting metadata and corrects the
    imbalance bias a plain (unweighted) mean would have.
"""

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest
import anndata

from mito_marker.analysis.pca_plot import (
    _OBSM_PCA_KEY,
    _PCA_CHANNEL_NAMES_KEY,
    _PCA_LOADINGS_KEY,
    _PCA_VAR_RATIO_KEY,
    _PCA_WEIGHT_BY_KEY,
    _build_loadings_panel_text,
    _compute_loading_contributions,
    _compute_nested_equal_weights,
    _compute_total_contributions,
    _get_data_and_channels,
    _get_group_labels,
    _print_loadings_method,
    compute_pca,
    get_top_loading_channels,
    plot_pca_3d_biplot,
    plot_pca_3d_scatter,
    plot_pca_3d_trajectory,
    plot_pca_biplot,
    plot_pca_loadings_bar,
    plot_pca_scatter,
    plot_pca_trajectory,
)
from mito_marker.analysis.colors import assign_color_palette
from mito_marker.analysis.selection import _ANALYSIS_CONFIG_KEY


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

N_OBS = 100
N_VARS = 15


def _make_test_anndata(
    n_obs: int = N_OBS,
    n_vars: int = N_VARS,
    with_layer: bool = False,
    with_selection: bool = False,
) -> anndata.AnnData:
    """Synthetic AnnData with 3 subjects and 2 conditions."""
    np.random.seed(55)
    x_matrix = np.abs(np.random.randn(n_obs, n_vars) * 5).astype(np.float32)
    obs = pd.DataFrame({
        "subject_ID": np.tile(["S001", "S002", "S003"], n_obs // 3 + 1)[:n_obs],
        "dilution": np.tile(["Diluted", "Not_Diluted"], n_obs // 2 + 1)[:n_obs],
    })
    var = pd.DataFrame(index=[f"CH{i:02d}" for i in range(n_vars)])
    adata = anndata.AnnData(X=x_matrix, obs=obs, var=var)
    adata.uns[_ANALYSIS_CONFIG_KEY] = {
        "active_layer": None,
        "active_selection": None,
    }
    if with_layer:
        normalized = (x_matrix - x_matrix.mean(axis=0)) / (x_matrix.std(axis=0) + 1e-8)
        adata.layers["arcsinh__zscore_col"] = normalized.astype(np.float32)
        adata.uns[_ANALYSIS_CONFIG_KEY]["active_layer"] = "arcsinh__zscore_col"
    if with_selection:
        adata.var["MIM_score"] = np.arange(n_vars, dtype=float)[::-1]
        adata.var["is_selected_MIM"] = adata.var["MIM_score"] >= (n_vars - 8)
        adata.uns[_ANALYSIS_CONFIG_KEY]["active_selection"] = "MIM"
    return adata


@pytest.fixture
def test_anndata() -> anndata.AnnData:
    adata = _make_test_anndata()
    assign_color_palette(adata)
    return adata


@pytest.fixture
def test_anndata_with_pca(test_anndata) -> anndata.AnnData:
    compute_pca(test_anndata, n_components=10)
    return test_anndata


# ---------------------------------------------------------------------------
# Tests: compute_pca — structure and caching
# ---------------------------------------------------------------------------


class TestComputePca:
    """Tests for compute_pca() output structure and caching behaviour."""

    def test_obsm_key_created(self, test_anndata):
        compute_pca(test_anndata, n_components=10)
        assert _OBSM_PCA_KEY in test_anndata.obsm

    def test_obsm_shape_is_n_obs_by_n_components(self, test_anndata):
        compute_pca(test_anndata, n_components=10)
        assert test_anndata.obsm[_OBSM_PCA_KEY].shape == (N_OBS, 10)

    def test_loadings_shape_is_n_channels_by_n_components(self, test_anndata):
        compute_pca(test_anndata, n_components=10)
        loadings = test_anndata.uns[_PCA_LOADINGS_KEY]
        assert loadings.shape == (N_VARS, 10)

    def test_explained_variance_sums_to_at_most_one(self, test_anndata):
        compute_pca(test_anndata, n_components=10)
        var_ratios = test_anndata.uns[_PCA_VAR_RATIO_KEY]
        assert float(var_ratios.sum()) <= 1.0 + 1e-6

    def test_explained_variance_all_non_negative(self, test_anndata):
        compute_pca(test_anndata, n_components=10)
        var_ratios = test_anndata.uns[_PCA_VAR_RATIO_KEY]
        assert (var_ratios >= 0).all()

    def test_channel_names_stored(self, test_anndata):
        compute_pca(test_anndata, n_components=5)
        names = test_anndata.uns[_PCA_CHANNEL_NAMES_KEY]
        assert len(names) == N_VARS

    def test_returns_same_object(self, test_anndata):
        result = compute_pca(test_anndata, n_components=5)
        assert result is test_anndata

    def test_cache_skips_recomputation(self, test_anndata):
        compute_pca(test_anndata, n_components=5)
        original_coords = test_anndata.obsm[_OBSM_PCA_KEY].copy()

        # Tamper to detect if recomputed.
        test_anndata.obsm[_OBSM_PCA_KEY][0, 0] = 9999.0

        compute_pca(test_anndata, n_components=5)
        # Cache worked: the tampered value remains.
        assert test_anndata.obsm[_OBSM_PCA_KEY][0, 0] == 9999.0

    def test_force_recompute_overwrites_cached_coordinates(self, test_anndata):
        compute_pca(test_anndata, n_components=5)
        test_anndata.obsm[_OBSM_PCA_KEY][0, 0] = 9999.0
        compute_pca(test_anndata, n_components=5, force_recompute=True)
        # After forced recompute the tampered value must be gone.
        assert test_anndata.obsm[_OBSM_PCA_KEY][0, 0] != 9999.0

    def test_force_recompute_false_keeps_cache(self, test_anndata):
        compute_pca(test_anndata, n_components=5)
        test_anndata.obsm[_OBSM_PCA_KEY][0, 0] = 9999.0
        compute_pca(test_anndata, n_components=5, force_recompute=False)
        assert test_anndata.obsm[_OBSM_PCA_KEY][0, 0] == 9999.0

    def test_n_components_capped_at_min_obs_vars(self):
        """n_components is capped to min(n_obs, n_vars, n_components)."""
        adata = _make_test_anndata(n_obs=20, n_vars=8)
        assign_color_palette(adata)
        compute_pca(adata, n_components=50)
        # With 20 obs and 8 vars, max rank = 8 (limited by vars).
        assert adata.obsm[_OBSM_PCA_KEY].shape[1] == 8

    def test_with_active_layer(self):
        adata = _make_test_anndata(with_layer=True)
        assign_color_palette(adata)
        compute_pca(adata, n_components=5)
        assert _OBSM_PCA_KEY in adata.obsm

    def test_with_feature_selection(self):
        adata = _make_test_anndata(with_selection=True)
        assign_color_palette(adata)
        compute_pca(adata, n_components=5)
        # Only 8 channels were selected; loadings should have 8 rows.
        loadings = adata.uns[_PCA_LOADINGS_KEY]
        assert loadings.shape[0] == 8


# ---------------------------------------------------------------------------
# Tests: _compute_nested_equal_weights — ADR-011 weighting helper
# ---------------------------------------------------------------------------


def _make_imbalanced_species_obs() -> pd.DataFrame:
    """
    obs with 2 species and unequal subject/event counts, mirroring the
    real Aging-paper imbalance (e.g. Worm 12 subjects/~3k events vs Mouse
    6 subjects/~16k events).

    SpeciesA: 3 subjects with 10, 20, 30 events (60 total).
    SpeciesB: 2 subjects with 5, 45 events (50 total).
    """
    rows = (
        [{"specie": "SpeciesA", "subject_ID": "A1"}] * 10
        + [{"specie": "SpeciesA", "subject_ID": "A2"}] * 20
        + [{"specie": "SpeciesA", "subject_ID": "A3"}] * 30
        + [{"specie": "SpeciesB", "subject_ID": "B1"}] * 5
        + [{"specie": "SpeciesB", "subject_ID": "B2"}] * 45
    )
    return pd.DataFrame(rows)


class TestComputeNestedEqualWeights:
    """Tests for _compute_nested_equal_weights() — the ADR-011 weighting helper."""

    def test_weights_sum_to_one(self):
        obs_dataframe = _make_imbalanced_species_obs()
        weights = _compute_nested_equal_weights(obs_dataframe, ["specie", "subject_ID"])
        assert weights.sum() == pytest.approx(1.0, abs=1e-9)

    def test_each_species_gets_equal_total_weight(self):
        obs_dataframe = _make_imbalanced_species_obs()
        weights = _compute_nested_equal_weights(obs_dataframe, ["specie", "subject_ID"])
        species_totals = pd.Series(weights).groupby(obs_dataframe["specie"].values).sum()
        # 2 species -> each should carry ~50% of total weight, despite
        # SpeciesA having 60 events and SpeciesB having 50.
        assert species_totals["SpeciesA"] == pytest.approx(0.5, abs=1e-9)
        assert species_totals["SpeciesB"] == pytest.approx(0.5, abs=1e-9)

    def test_each_subject_gets_equal_weight_within_its_species(self):
        obs_dataframe = _make_imbalanced_species_obs()
        weights = _compute_nested_equal_weights(obs_dataframe, ["specie", "subject_ID"])
        subject_totals = pd.Series(weights).groupby(obs_dataframe["subject_ID"].values).sum()
        # SpeciesA has 3 subjects -> each carries 1/3 of SpeciesA's 0.5 share.
        assert subject_totals["A1"] == pytest.approx(0.5 / 3, abs=1e-9)
        assert subject_totals["A2"] == pytest.approx(0.5 / 3, abs=1e-9)
        assert subject_totals["A3"] == pytest.approx(0.5 / 3, abs=1e-9)
        # SpeciesB has 2 subjects -> each carries 1/2 of SpeciesB's 0.5 share,
        # even though B2 (45 events) has 9x more events than B1 (5 events).
        assert subject_totals["B1"] == pytest.approx(0.25, abs=1e-9)
        assert subject_totals["B2"] == pytest.approx(0.25, abs=1e-9)

    def test_single_hierarchy_column_gives_equal_weight_per_group(self):
        obs_dataframe = _make_imbalanced_species_obs()
        weights = _compute_nested_equal_weights(obs_dataframe, ["subject_ID"])
        subject_totals = pd.Series(weights).groupby(obs_dataframe["subject_ID"].values).sum()
        # 5 subjects total -> each should carry 1/5 of the total weight.
        for subject_id in ["A1", "A2", "A3", "B1", "B2"]:
            assert subject_totals[subject_id] == pytest.approx(0.2, abs=1e-9)

    def test_raises_on_missing_hierarchy_values(self):
        obs_dataframe = _make_imbalanced_species_obs()
        obs_dataframe.loc[0, "subject_ID"] = None
        with pytest.raises(ValueError, match="missing values"):
            _compute_nested_equal_weights(obs_dataframe, ["specie", "subject_ID"])


# ---------------------------------------------------------------------------
# Tests: compute_pca(weight_by=...) — ADR-011 weighted PCA
# ---------------------------------------------------------------------------


def _make_weighted_pca_test_anndata() -> anndata.AnnData:
    """
    Synthetic AnnData where SpeciesA (90 events) and SpeciesB (10 events)
    have different means on feature 0 — used to check that weight_by
    corrects the imbalance bias a plain pooled mean would have.
    """
    np.random.seed(7)
    n_species_a, n_species_b = 90, 10
    n_vars = 6
    feature_matrix = np.random.randn(n_species_a + n_species_b, n_vars).astype(np.float32)
    # SpeciesA centered at 0, SpeciesB centered at 10 on feature 0 only.
    feature_matrix[n_species_a:, 0] += 10.0
    obs = pd.DataFrame({
        "specie": ["SpeciesA"] * n_species_a + ["SpeciesB"] * n_species_b,
        "subject_ID": (
            ["A1"] * 45 + ["A2"] * 45 + ["B1"] * 5 + ["B2"] * 5
        ),
    })
    var = pd.DataFrame(index=[f"CH{i:02d}" for i in range(n_vars)])
    adata = anndata.AnnData(X=feature_matrix, obs=obs, var=var)
    adata.uns["analysis_config"] = {"active_layer": None, "active_selection": None}
    return adata


class TestComputePcaWeighted:
    """Tests for compute_pca(weight_by=...)."""

    def test_weight_by_none_is_unchanged_default(self, test_anndata):
        compute_pca(test_anndata, n_components=5)
        assert test_anndata.uns[_PCA_WEIGHT_BY_KEY] is None

    def test_weight_by_stored_in_uns(self):
        adata = _make_weighted_pca_test_anndata()
        compute_pca(adata, n_components=3, weight_by=["specie", "subject_ID"])
        assert adata.uns[_PCA_WEIGHT_BY_KEY] == ["specie", "subject_ID"]

    def test_weighted_pca_output_shapes(self):
        adata = _make_weighted_pca_test_anndata()
        compute_pca(adata, n_components=3, weight_by=["specie", "subject_ID"])
        assert adata.obsm[_OBSM_PCA_KEY].shape == (100, 3)
        assert adata.uns[_PCA_LOADINGS_KEY].shape == (6, 3)
        assert adata.uns[_PCA_VAR_RATIO_KEY].shape == (3,)

    def test_weighted_pca_explained_variance_valid(self):
        adata = _make_weighted_pca_test_anndata()
        compute_pca(adata, n_components=3, weight_by=["specie", "subject_ID"])
        var_ratios = adata.uns[_PCA_VAR_RATIO_KEY]
        assert (var_ratios >= 0).all()
        assert float(var_ratios.sum()) <= 1.0 + 1e-6

    def test_weighted_pca_every_event_still_projected(self):
        """No event is dropped — every row still gets its own PCA coordinate."""
        adata = _make_weighted_pca_test_anndata()
        compute_pca(adata, n_components=3, weight_by=["specie"])
        assert adata.obsm[_OBSM_PCA_KEY].shape[0] == adata.n_obs
        assert np.isfinite(adata.obsm[_OBSM_PCA_KEY]).all()

    def test_weighting_corrects_species_imbalance_bias(self):
        """
        Core ADR-011 claim: with weight_by=["specie"], PC1 (dominated by the
        only differentiating feature, CH00) should reflect the MIDPOINT
        between the two species means (~5.0), not the raw pooled mean, which
        is skewed toward SpeciesA because it has 9x more events (~1.0).
        """
        adata = _make_weighted_pca_test_anndata()
        weights = _compute_nested_equal_weights(adata.obs, ["specie"])
        weighted_mean_feature_0 = float(weights @ adata.X[:, 0])
        pooled_mean_feature_0 = float(adata.X[:, 0].mean())

        # Raw pooled mean is skewed toward SpeciesA (mean ~0): expect << 5.
        assert pooled_mean_feature_0 < 3.0
        # Species-weighted mean should sit near the true midpoint (0, 10) / 2 = 5.
        assert weighted_mean_feature_0 == pytest.approx(5.0, abs=1.0)

    def test_force_recompute_required_to_change_weight_by(self):
        adata = _make_weighted_pca_test_anndata()
        compute_pca(adata, n_components=3, weight_by=None)
        assert adata.uns[_PCA_WEIGHT_BY_KEY] is None
        # Without force_recompute, the cached (unweighted) result is kept.
        compute_pca(adata, n_components=3, weight_by=["specie"])
        assert adata.uns[_PCA_WEIGHT_BY_KEY] is None
        # With force_recompute, the weighted result replaces it.
        compute_pca(adata, n_components=3, weight_by=["specie"], force_recompute=True)
        assert adata.uns[_PCA_WEIGHT_BY_KEY] == ["specie"]


# ---------------------------------------------------------------------------
# Tests: plot_pca_scatter — integration
# ---------------------------------------------------------------------------


class TestPlotPcaScatter:
    """Integration tests for plot_pca_scatter()."""

    def test_group_by_subject_id_creates_figure(self, test_anndata_with_pca):
        import matplotlib.pyplot as plt
        n_figs_before = len(plt.get_fignums())
        plot_pca_scatter(test_anndata_with_pca, group_by="subject_ID")
        assert len(plt.get_fignums()) == n_figs_before + 1
        plt.close("all")

    def test_group_by_dilution_creates_figure(self, test_anndata_with_pca):
        import matplotlib.pyplot as plt
        n_figs_before = len(plt.get_fignums())
        plot_pca_scatter(test_anndata_with_pca, group_by="dilution")
        assert len(plt.get_fignums()) == n_figs_before + 1
        plt.close("all")

    def test_invalid_group_by_raises(self, test_anndata_with_pca):
        with pytest.raises(ValueError, match="not found in .obs"):
            plot_pca_scatter(test_anndata_with_pca, group_by="nonexistent")

    def test_raises_when_pca_not_computed(self, test_anndata):
        with pytest.raises(ValueError, match="PCA not yet computed"):
            plot_pca_scatter(test_anndata, group_by="dilution")

    def test_small_n_events_per_group_does_not_raise(self, test_anndata_with_pca):
        import matplotlib.pyplot as plt
        plot_pca_scatter(test_anndata_with_pca, group_by="dilution", n_events_per_group=5)
        plt.close("all")

    @staticmethod
    def _panel_texts(figure) -> list:
        """Return the texts of the loadings-panel annotations drawn on the scatter Axes."""
        return [
            child.get_text()
            for child in figure.axes[0].texts
            if "PCA loadings" in child.get_text()
        ]

    def test_loadings_panel_drawn_by_default(self, test_anndata_with_pca):
        import matplotlib.pyplot as plt
        plot_pca_scatter(test_anndata_with_pca, group_by="dilution")
        panel_texts = self._panel_texts(plt.gcf())
        plt.close("all")
        assert len(panel_texts) == 1
        for pc_name in ["PC1", "PC2", "PC3"]:
            assert pc_name in panel_texts[0]
        assert "PC4" not in panel_texts[0]

    def test_loadings_panel_hidden_when_disabled(self, test_anndata_with_pca):
        import matplotlib.pyplot as plt
        plot_pca_scatter(test_anndata_with_pca, group_by="dilution", show_loadings=False)
        panel_texts = self._panel_texts(plt.gcf())
        plt.close("all")
        assert panel_texts == []

    def test_loadings_panel_with_many_groups(self, test_anndata_with_pca):
        # More than 15 groups moves the legend below the plot; the panel must still be drawn.
        import matplotlib.pyplot as plt
        test_anndata_with_pca.obs["many_groups"] = [f"G{i % 20:02d}" for i in range(N_OBS)]
        plot_pca_scatter(test_anndata_with_pca, group_by="many_groups")
        panel_texts = self._panel_texts(plt.gcf())
        plt.close("all")
        assert len(panel_texts) == 1

    def test_invalid_loadings_top_n_raises(self, test_anndata_with_pca):
        with pytest.raises(ValueError, match="at least 1"):
            plot_pca_scatter(test_anndata_with_pca, group_by="dilution", loadings_top_n=0)


# ---------------------------------------------------------------------------
# Tests: loadings panel helpers
# ---------------------------------------------------------------------------


class TestLoadingsPanel:
    """Tests for the loadings contribution helpers and the panel text (ADR-016)."""

    def test_contributions_sum_to_100_per_component(self, test_anndata):
        # All components kept → loadings are a full orthonormal basis.
        compute_pca(test_anndata, n_components=N_VARS)
        contributions = _compute_loading_contributions(test_anndata.uns[_PCA_LOADINGS_KEY])
        np.testing.assert_allclose(contributions.sum(axis=0), 100.0, atol=1e-3)

    def test_contributions_are_squared_loadings(self):
        loadings_matrix = np.array([[0.8], [-0.6]])
        np.testing.assert_allclose(
            _compute_loading_contributions(loadings_matrix)[:, 0], [64.0, 36.0]
        )

    def test_total_contributions_weighted_by_variance(self):
        # Channel 0 carries all of PC1 (75% variance), channel 1 all of PC2 (25%).
        contributions = np.array([[100.0, 0.0], [0.0, 100.0]])
        totals = _compute_total_contributions(contributions, np.array([0.75, 0.25]))
        np.testing.assert_allclose(totals, [75.0, 25.0])

    def test_panel_lists_top_channels_in_order_with_sign(self):
        channel_names = ["Area", "Perimeter", "Circularity", "AR"]
        loadings_matrix = np.array([
            [0.8, 0.0],
            [-0.6, 0.0],
            [0.0, 1.0],
            [0.0, 0.0],
        ])
        panel_text = _build_loadings_panel_text(
            channel_names, loadings_matrix, np.array([0.6, 0.3]), top_n=2, pc_indices=[0, 1, 2],
        )
        lines = panel_text.splitlines()
        pc1_line = lines.index("PC1 — 60.0% of variance")
        assert lines[pc1_line + 1].split() == ["+", "Area", "64.0%"]
        assert lines[pc1_line + 2].split() == ["-", "Perimeter", "36.0%"]
        # PC3 is skipped: only 2 components are available.
        assert "PC2 — 30.0% of variance" in lines
        assert "PC3" not in panel_text

    def test_panel_follows_requested_pc_order(self):
        loadings_matrix = np.eye(4)
        panel_text = _build_loadings_panel_text(
            ["A", "B", "C", "D"], loadings_matrix, np.full(4, 0.25), top_n=1, pc_indices=[3, 0],
        )
        assert panel_text.index("PC4") < panel_text.index("PC1")
        assert "PC2" not in panel_text

    def test_panel_truncates_long_channel_names(self):
        long_name = "Cristae_Density_Per_Area_Unit_Long_Name"
        panel_text = _build_loadings_panel_text(
            [long_name], np.array([[1.0]]), np.array([1.0]), top_n=5, pc_indices=[0, 1, 2],
        )
        assert long_name not in panel_text
        assert long_name[:10] in panel_text

    def test_method_printed_for_standard_pca(self, test_anndata_with_pca, capsys):
        _print_loadings_method(test_anndata_with_pca, "some_plot")
        output = capsys.readouterr().out
        assert "standard PCA" in output
        assert "loading_jk² × 100" in output

    def test_method_printed_for_weighted_pca(self, capsys):
        adata = _make_weighted_pca_test_anndata()
        compute_pca(adata, n_components=3, weight_by=["specie"])
        capsys.readouterr()
        _print_loadings_method(adata, "some_plot")
        output = capsys.readouterr().out
        assert "weighted PCA, weight_by=['specie']" in output


class TestLoadingsPanelOnAllPlots:
    """The loadings panel and the method line appear on every PCA plot."""

    @staticmethod
    def _matplotlib_panel_texts() -> list:
        import matplotlib.pyplot as plt
        return [
            child.get_text()
            for child in plt.gcf().axes[0].texts
            if "PCA loadings" in child.get_text()
        ]

    @staticmethod
    def _plotly_panel_texts(figure) -> list:
        return [
            annotation.text
            for annotation in figure.layout.annotations
            if "PCA" in annotation.text and "loadings" in annotation.text
        ]

    def test_biplot_panel_and_method(self, test_anndata_with_pca, capsys):
        import matplotlib.pyplot as plt
        plot_pca_biplot(test_anndata_with_pca, group_by="dilution")
        panel_texts = self._matplotlib_panel_texts()
        plt.close("all")
        assert len(panel_texts) == 1
        assert "loading_jk² × 100" in capsys.readouterr().out

    def test_biplot_panel_hidden_when_disabled(self, test_anndata_with_pca):
        import matplotlib.pyplot as plt
        plot_pca_biplot(test_anndata_with_pca, group_by="dilution", show_loadings=False)
        panel_texts = self._matplotlib_panel_texts()
        plt.close("all")
        assert panel_texts == []

    def test_trajectory_panel_and_method(self, trajectory_anndata_with_pca, capsys):
        import matplotlib.pyplot as plt
        plot_pca_trajectory(trajectory_anndata_with_pca, condition_column="diet", time_column="age")
        panel_texts = self._matplotlib_panel_texts()
        plt.close("all")
        assert len(panel_texts) == 1
        assert "loading_jk² × 100" in capsys.readouterr().out

    def test_many_groups_legend_below_x_axis_label(self, test_anndata_with_pca):
        # With > 15 groups the legend goes below the plot and must not cover the x-axis label.
        import matplotlib.pyplot as plt
        test_anndata_with_pca.obs["many_groups"] = [f"G{i % 20:02d}" for i in range(N_OBS)]
        plot_pca_scatter(test_anndata_with_pca, group_by="many_groups")
        figure = plt.gcf()
        figure.canvas.draw()
        renderer = figure.canvas.get_renderer()
        axis = figure.axes[0]
        legend_top = axis.get_legend().get_window_extent(renderer).y1
        x_label_bottom = axis.xaxis.label.get_window_extent(renderer).y0
        plt.close("all")
        assert legend_top < x_label_bottom

    def test_3d_scatter_panel(self, test_anndata_with_pca):
        figure = plot_pca_3d_scatter(test_anndata_with_pca, group_by="dilution")
        assert len(self._plotly_panel_texts(figure)) == 1

    def test_3d_scatter_panel_hidden_when_disabled(self, test_anndata_with_pca):
        figure = plot_pca_3d_scatter(
            test_anndata_with_pca, group_by="dilution", show_loadings=False
        )
        assert self._plotly_panel_texts(figure) == []

    def test_3d_biplot_panel_lists_displayed_pcs(self, test_anndata_with_pca):
        figure = plot_pca_3d_biplot(test_anndata_with_pca, group_by="dilution", pc_z=4)
        panel_text = self._plotly_panel_texts(figure)[0]
        assert "PC4" in panel_text
        assert "PC3" not in panel_text

    def test_3d_trajectory_panel(self, trajectory_anndata_with_pca):
        figure = plot_pca_3d_trajectory(
            trajectory_anndata_with_pca, condition_column="diet", time_column="age"
        )
        assert len(self._plotly_panel_texts(figure)) == 1

    def test_3d_many_groups_moves_legend_below(self, test_anndata_with_pca):
        test_anndata_with_pca.obs["many_groups"] = [f"G{i % 30:02d}" for i in range(N_OBS)]
        figure = plot_pca_3d_scatter(test_anndata_with_pca, group_by="many_groups")
        assert figure.layout.legend.orientation == "h"
        assert len(self._plotly_panel_texts(figure)) == 1

    def test_3d_invalid_loadings_top_n_raises(self, test_anndata_with_pca):
        with pytest.raises(ValueError, match="loadings_top_n must be at least 1"):
            plot_pca_3d_scatter(test_anndata_with_pca, group_by="dilution", loadings_top_n=0)


class TestPlotPcaLoadingsBar:
    """plot_pca_loadings_bar() uses the same contribution formula as the panels (ADR-016)."""

    def test_console_shows_squared_loading_contributions(self, test_anndata_with_pca, capsys):
        import matplotlib.pyplot as plt
        plot_pca_loadings_bar(test_anndata_with_pca, top_n=3, n_components=2)
        plt.close("all")
        output = capsys.readouterr().out
        loadings_pc1 = test_anndata_with_pca.uns[_PCA_LOADINGS_KEY][:, 0].astype(np.float64)
        top_contribution = float(np.max(loadings_pc1 ** 2) * 100)
        assert f"{top_contribution:5.1f}%" in output
        assert "loading_jk² × 100" in output
        assert "variance-weighted" in output

    def test_bar_chart_x_label(self, test_anndata_with_pca):
        import matplotlib.pyplot as plt
        plot_pca_loadings_bar(test_anndata_with_pca, top_n=3, n_components=2)
        bar_figure = plt.figure(plt.get_fignums()[-2])
        x_label = bar_figure.axes[0].get_xlabel()
        plt.close("all")
        assert x_label == "% contribution (loading² × 100)"


# ---------------------------------------------------------------------------
# Tests: plot_pca_biplot — integration
# ---------------------------------------------------------------------------


class TestPlotPcaBiplot:
    """Integration tests for plot_pca_biplot()."""

    def test_group_by_subject_id_creates_figure(self, test_anndata_with_pca):
        import matplotlib.pyplot as plt
        n_figs_before = len(plt.get_fignums())
        plot_pca_biplot(test_anndata_with_pca, group_by="subject_ID")
        assert len(plt.get_fignums()) == n_figs_before + 1
        plt.close("all")

    def test_group_by_dilution_creates_figure(self, test_anndata_with_pca):
        import matplotlib.pyplot as plt
        n_figs_before = len(plt.get_fignums())
        plot_pca_biplot(test_anndata_with_pca, group_by="dilution")
        assert len(plt.get_fignums()) == n_figs_before + 1
        plt.close("all")

    def test_invalid_group_by_raises(self, test_anndata_with_pca):
        with pytest.raises(ValueError, match="not found in .obs"):
            plot_pca_biplot(test_anndata_with_pca, group_by="nonexistent")

    def test_raises_when_pca_not_computed(self, test_anndata):
        with pytest.raises(ValueError, match="PCA not yet computed"):
            plot_pca_biplot(test_anndata, group_by="dilution")

    def test_top_n_variables_parameter_accepted(self, test_anndata_with_pca):
        """Different top_n_variables values must not raise errors."""
        import matplotlib.pyplot as plt
        plot_pca_biplot(test_anndata_with_pca, group_by="dilution", top_n_variables=3)
        plt.close("all")

    def test_labels_string_accepted(self, test_anndata_with_pca):
        """labels='diet' runs without error when the column exists."""
        import matplotlib.pyplot as plt
        # Add a per-subject 'diet' column to .obs.
        diet_map = {"S001": "Low-fat", "S002": "High-fat", "S003": "Control"}
        test_anndata_with_pca.obs["diet"] = (
            test_anndata_with_pca.obs["subject_ID"].map(diet_map)
        )
        plot_pca_biplot(test_anndata_with_pca, group_by="subject_ID", labels="diet")
        plt.close("all")

    def test_labels_list_accepted(self, test_anndata_with_pca):
        """labels=['diet', 'dilution'] appends two extra columns."""
        import matplotlib.pyplot as plt
        diet_map = {"S001": "Low-fat", "S002": "High-fat", "S003": "Control"}
        test_anndata_with_pca.obs["diet"] = (
            test_anndata_with_pca.obs["subject_ID"].map(diet_map)
        )
        plot_pca_biplot(
            test_anndata_with_pca,
            group_by="subject_ID",
            labels=["diet", "dilution"],
        )
        plt.close("all")

    def test_labels_missing_column_skipped_silently(self, test_anndata_with_pca, capsys):
        """A label column absent from .obs triggers a warning but no exception."""
        import matplotlib.pyplot as plt
        plot_pca_biplot(
            test_anndata_with_pca,
            group_by="subject_ID",
            labels=["nonexistent_column"],
        )
        captured = capsys.readouterr()
        assert "nonexistent_column" in captured.out
        plt.close("all")

    def test_labels_none_behaves_as_before(self, test_anndata_with_pca):
        """labels=None (default) produces a figure without raising."""
        import matplotlib.pyplot as plt
        n_figs_before = len(plt.get_fignums())
        plot_pca_biplot(test_anndata_with_pca, group_by="subject_ID", labels=None)
        assert len(plt.get_fignums()) == n_figs_before + 1
        plt.close("all")

    def test_build_centroid_label_helper(self, test_anndata_with_pca):
        """_build_centroid_label returns correct pipe-separated string."""
        from mito_marker.analysis.pca_plot import _build_centroid_label
        diet_map = {"S001": "Low-fat", "S002": "High-fat", "S003": "Control"}
        obs = test_anndata_with_pca.obs.copy()
        obs["diet"] = obs["subject_ID"].map(diet_map)
        result = _build_centroid_label("S001", obs, "subject_ID", ["diet"])
        assert result == "S001 | Low-fat"

    def test_build_centroid_label_missing_column_skipped(self, test_anndata_with_pca):
        """_build_centroid_label skips columns not present in obs."""
        from mito_marker.analysis.pca_plot import _build_centroid_label
        obs = test_anndata_with_pca.obs.copy()
        result = _build_centroid_label("S001", obs, "subject_ID", ["ghost_column"])
        assert result == "S001"


# ---------------------------------------------------------------------------
# Tests: _get_group_labels (PCA)
# ---------------------------------------------------------------------------


class TestPcaGetGroupLabels:
    """Tests for the multi-column grouping label helper in pca_plot."""

    def test_single_string_returns_column(self):
        adata = _make_test_anndata()
        labels = _get_group_labels(adata, "dilution")
        assert list(labels) == list(adata.obs["dilution"])

    def test_two_columns_combined_with_slash(self):
        adata = _make_test_anndata()
        labels = _get_group_labels(adata, ["subject_ID", "dilution"])
        expected = adata.obs["subject_ID"].astype(str) + " / " + adata.obs["dilution"].astype(str)
        assert list(labels) == list(expected)

    def test_invalid_column_raises(self):
        adata = _make_test_anndata()
        with pytest.raises(ValueError, match="not found in .obs"):
            _get_group_labels(adata, "nonexistent")

    def test_invalid_column_in_list_raises(self):
        adata = _make_test_anndata()
        with pytest.raises(ValueError, match="not found in .obs"):
            _get_group_labels(adata, ["dilution", "nonexistent"])


# ---------------------------------------------------------------------------
# Tests: plot_pca_scatter with list group_by
# ---------------------------------------------------------------------------


class TestPlotPcaScatterGroupByList:
    """Tests for plot_pca_scatter() with group_by as a list."""

    def test_group_by_list_creates_figure(self, test_anndata_with_pca):
        import matplotlib.pyplot as plt
        n_figs_before = len(plt.get_fignums())
        plot_pca_scatter(test_anndata_with_pca, group_by=["subject_ID", "dilution"])
        assert len(plt.get_fignums()) == n_figs_before + 1
        plt.close("all")

    def test_group_by_list_title_contains_cross_separator(self, test_anndata_with_pca):
        import matplotlib.pyplot as plt
        plot_pca_scatter(test_anndata_with_pca, group_by=["subject_ID", "dilution"])
        title = plt.gcf().axes[0].get_title()
        assert "subject_ID × dilution" in title
        plt.close("all")

    def test_group_by_list_invalid_column_raises(self, test_anndata_with_pca):
        with pytest.raises(ValueError, match="not found in .obs"):
            plot_pca_scatter(test_anndata_with_pca, group_by=["dilution", "nonexistent"])


# ---------------------------------------------------------------------------
# Tests: plot_pca_biplot with list group_by
# ---------------------------------------------------------------------------


class TestPlotPcaBiplotGroupByList:
    """Tests for plot_pca_biplot() with group_by as a list."""

    def test_group_by_list_creates_figure(self, test_anndata_with_pca):
        import matplotlib.pyplot as plt
        n_figs_before = len(plt.get_fignums())
        plot_pca_biplot(test_anndata_with_pca, group_by=["subject_ID", "dilution"])
        assert len(plt.get_fignums()) == n_figs_before + 1
        plt.close("all")

    def test_group_by_list_title_contains_cross_separator(self, test_anndata_with_pca):
        import matplotlib.pyplot as plt
        plot_pca_biplot(test_anndata_with_pca, group_by=["subject_ID", "dilution"])
        title = plt.gcf().axes[0].get_title()
        assert "subject_ID × dilution" in title
        plt.close("all")

    def test_group_by_list_invalid_column_raises(self, test_anndata_with_pca):
        with pytest.raises(ValueError, match="not found in .obs"):
            plot_pca_biplot(test_anndata_with_pca, group_by=["dilution", "nonexistent"])


# ---------------------------------------------------------------------------
# Tests: get_top_loading_channels
# ---------------------------------------------------------------------------


class TestGetTopLoadingChannels:
    """Unit tests for the top-variable ranking helper."""

    def test_returns_correct_count(self, test_anndata_with_pca):
        result = get_top_loading_channels(test_anndata_with_pca, pc_index=0, top_n=5)
        assert len(result) == 5

    def test_returns_tuples_of_name_and_float(self, test_anndata_with_pca):
        result = get_top_loading_channels(test_anndata_with_pca, pc_index=0, top_n=3)
        for name, value in result:
            assert isinstance(name, str)
            assert isinstance(value, float)

    def test_sorted_by_abs_loading_descending(self, test_anndata_with_pca):
        result = get_top_loading_channels(test_anndata_with_pca, pc_index=0, top_n=5)
        abs_values = [abs(v) for _, v in result]
        assert abs_values == sorted(abs_values, reverse=True)

    def test_raises_for_invalid_pc_index(self, test_anndata_with_pca):
        with pytest.raises(ValueError, match="pc_index=99 out of range"):
            get_top_loading_channels(test_anndata_with_pca, pc_index=99, top_n=5)

    def test_raises_when_pca_not_computed(self, test_anndata):
        with pytest.raises(ValueError, match="PCA not yet computed"):
            get_top_loading_channels(test_anndata, pc_index=0, top_n=3)

    def test_works_for_pc2(self, test_anndata_with_pca):
        result = get_top_loading_channels(test_anndata_with_pca, pc_index=1, top_n=4)
        assert len(result) == 4


# ---------------------------------------------------------------------------
# Tests: _get_data_and_channels
# ---------------------------------------------------------------------------


class TestGetDataAndChannels:
    """Unit tests for the internal data extraction helper."""

    def test_returns_full_matrix_by_default(self):
        adata = _make_test_anndata()
        matrix, channels = _get_data_and_channels(adata)
        assert matrix.shape == (N_OBS, N_VARS)
        assert len(channels) == N_VARS

    def test_returns_layer_when_active(self):
        adata = _make_test_anndata(with_layer=True)
        matrix, channels = _get_data_and_channels(adata)
        # Layer values differ from raw X (it's z-scored).
        assert not np.allclose(matrix, adata.X, atol=0.01)

    def test_feature_selection_reduces_columns(self):
        adata = _make_test_anndata(with_selection=True)
        matrix, channels = _get_data_and_channels(adata)
        assert matrix.shape[1] == 8
        assert len(channels) == 8

    def test_non_analytical_channels_excluded_from_pca_input(self):
        """Time and FlowAI must be absent from the data matrix passed to PCA."""
        np.random.seed(1)
        n_obs, n_analytical = N_OBS, N_VARS
        x_matrix = np.hstack([
            np.abs(np.random.randn(n_obs, n_analytical) * 5),
            np.linspace(0, 10_000, n_obs).reshape(-1, 1),  # Time
            np.ones((n_obs, 1)) * 999.0,                   # FlowAI
        ]).astype(np.float32)
        var_names = [f"CH{i:02d}" for i in range(n_analytical)] + ["Time", "FlowAI"]
        obs = pd.DataFrame({
            "subject_ID": ["S001"] * n_obs,
            "dilution": ["Diluted"] * n_obs,
        })
        var = pd.DataFrame(
            {"is_non_analytical": [False] * n_analytical + [True, True]},
            index=var_names,
        )
        adata = anndata.AnnData(X=x_matrix, obs=obs, var=var)
        adata.uns[_ANALYSIS_CONFIG_KEY] = {"active_layer": None, "active_selection": None}

        matrix, channels = _get_data_and_channels(adata)
        assert matrix.shape[1] == n_analytical
        assert "Time" not in channels
        assert "FlowAI" not in channels


# ---------------------------------------------------------------------------
# Fixture helpers for trajectory tests
# ---------------------------------------------------------------------------


def _make_trajectory_anndata(numeric_time: bool = True) -> anndata.AnnData:
    """
    Synthetic AnnData with 2 conditions × 3 timepoints, suitable for trajectory tests.

    condition_column: "diet"  — values "AL" and "IF"
    time_column:      "age"   — values 0, 2, 8 (numeric) or "D00", "D02", "D08" (string)
    """
    np.random.seed(77)
    n_conditions = 2
    n_timepoints = 3
    n_events_per_group = 20
    n_total = n_conditions * n_timepoints * n_events_per_group
    n_vars = 12

    x_matrix = np.abs(np.random.randn(n_total, n_vars) * 4).astype(np.float32)

    conditions = ["AL", "IF"]
    timepoints_numeric = [0, 2, 8]
    timepoints_string = ["D00", "D02", "D08"]

    diet_list = []
    age_list = []
    for cond in conditions:
        for tp in (timepoints_numeric if numeric_time else timepoints_string):
            diet_list.extend([cond] * n_events_per_group)
            age_list.extend([tp] * n_events_per_group)

    obs = pd.DataFrame({"diet": diet_list, "age": age_list})
    var = pd.DataFrame(index=[f"CH{i:02d}" for i in range(n_vars)])
    adata = anndata.AnnData(X=x_matrix, obs=obs, var=var)
    adata.uns[_ANALYSIS_CONFIG_KEY] = {"active_layer": None, "active_selection": None}
    return adata


@pytest.fixture
def trajectory_anndata_with_pca() -> anndata.AnnData:
    """Trajectory AnnData with numeric timepoints and PCA already computed."""
    adata = _make_trajectory_anndata(numeric_time=True)
    compute_pca(adata, n_components=5)
    return adata


# ---------------------------------------------------------------------------
# Tests: plot_pca_trajectory
# ---------------------------------------------------------------------------


class TestPlotPcaTrajectory:
    """Integration tests for plot_pca_trajectory()."""

    def test_returns_none(self, trajectory_anndata_with_pca):
        import matplotlib.pyplot as plt
        result = plot_pca_trajectory(
            trajectory_anndata_with_pca, condition_column="diet", time_column="age"
        )
        assert result is None
        plt.close("all")

    def test_creates_figure_with_numeric_time(self, trajectory_anndata_with_pca):
        import matplotlib.pyplot as plt
        n_figs_before = len(plt.get_fignums())
        plot_pca_trajectory(
            trajectory_anndata_with_pca, condition_column="diet", time_column="age"
        )
        assert len(plt.get_fignums()) == n_figs_before + 1
        plt.close("all")

    def test_creates_figure_with_string_time(self):
        import matplotlib.pyplot as plt
        adata = _make_trajectory_anndata(numeric_time=False)
        compute_pca(adata, n_components=5)
        n_figs_before = len(plt.get_fignums())
        plot_pca_trajectory(adata, condition_column="diet", time_column="age")
        assert len(plt.get_fignums()) == n_figs_before + 1
        plt.close("all")

    def test_time_order_respected(self, trajectory_anndata_with_pca):
        """Explicit time_order should run without error (reverse order)."""
        import matplotlib.pyplot as plt
        plot_pca_trajectory(
            trajectory_anndata_with_pca,
            condition_column="diet",
            time_column="age",
            time_order=[8, 2, 0],
        )
        plt.close("all")

    def test_show_loading_arrows_false(self, trajectory_anndata_with_pca):
        """Disabling loading arrows must not raise an error."""
        import matplotlib.pyplot as plt
        plot_pca_trajectory(
            trajectory_anndata_with_pca,
            condition_column="diet",
            time_column="age",
            show_loading_arrows=False,
        )
        plt.close("all")

    def test_raises_for_missing_condition_column(self, trajectory_anndata_with_pca):
        with pytest.raises(ValueError, match="condition_column 'nonexistent'"):
            plot_pca_trajectory(
                trajectory_anndata_with_pca,
                condition_column="nonexistent",
                time_column="age",
            )

    def test_raises_for_missing_time_column(self, trajectory_anndata_with_pca):
        with pytest.raises(ValueError, match="time_column 'nonexistent'"):
            plot_pca_trajectory(
                trajectory_anndata_with_pca,
                condition_column="diet",
                time_column="nonexistent",
            )

    def test_raises_when_pca_not_computed(self):
        adata = _make_trajectory_anndata()
        with pytest.raises(ValueError, match="PCA not yet computed"):
            plot_pca_trajectory(adata, condition_column="diet", time_column="age")

    def test_partial_timepoints_do_not_crash(self):
        """If one condition is missing a timepoint, the function must not crash."""
        import matplotlib.pyplot as plt
        adata = _make_trajectory_anndata(numeric_time=True)
        # Remove all IF events at timepoint 8 to create a gap in the trajectory.
        mask = ~((adata.obs["diet"] == "IF") & (adata.obs["age"] == 8))
        adata = adata[mask].copy()
        adata.uns[_ANALYSIS_CONFIG_KEY] = {"active_layer": None, "active_selection": None}
        compute_pca(adata, n_components=5)
        plot_pca_trajectory(adata, condition_column="diet", time_column="age")
        plt.close("all")


# ---------------------------------------------------------------------------
# Tests: plot_pca_3d_scatter
# ---------------------------------------------------------------------------


class TestPlotPca3dScatter:
    """Integration tests for plot_pca_3d_scatter()."""

    def test_returns_go_figure(self, test_anndata_with_pca):
        import plotly.graph_objects as go
        result = plot_pca_3d_scatter(test_anndata_with_pca, group_by="dilution")
        assert isinstance(result, go.Figure)

    def test_one_trace_per_group(self, test_anndata_with_pca):
        figure = plot_pca_3d_scatter(test_anndata_with_pca, group_by="dilution")
        # 1 data trace (all shuffled points) + 2 invisible legend traces (one per group)
        assert len(figure.data) == 3

    def test_trace_names_match_group_values(self, test_anndata_with_pca):
        figure = plot_pca_3d_scatter(test_anndata_with_pca, group_by="dilution")
        # Legend traces (showlegend=True) carry the group names; the data trace has name=None.
        legend_names = {trace.name for trace in figure.data if trace.showlegend}
        assert legend_names == {"Diluted", "Not_Diluted"}

    def test_axis_labels_include_variance_percentage(self, test_anndata_with_pca):
        figure = plot_pca_3d_scatter(test_anndata_with_pca, group_by="dilution")
        x_title = figure.layout.scene.xaxis.title.text
        assert "PC1" in x_title
        assert "%" in x_title

    def test_raises_for_invalid_group_by(self, test_anndata_with_pca):
        with pytest.raises(ValueError, match="not found in .obs"):
            plot_pca_3d_scatter(test_anndata_with_pca, group_by="nonexistent")

    def test_raises_when_pca_not_computed(self, test_anndata):
        with pytest.raises(ValueError, match="PCA not yet computed"):
            plot_pca_3d_scatter(test_anndata, group_by="dilution")

    def test_raises_for_pc_z_out_of_range(self, test_anndata_with_pca):
        with pytest.raises(ValueError, match="pc_z=99 is out of range"):
            plot_pca_3d_scatter(test_anndata_with_pca, group_by="dilution", pc_z=99)

    def test_raises_for_duplicate_pc_indices(self, test_anndata_with_pca):
        with pytest.raises(ValueError, match="must all be distinct"):
            plot_pca_3d_scatter(test_anndata_with_pca, group_by="dilution", pc_x=1, pc_y=1, pc_z=3)

    def test_custom_pc_axes_reflected_in_axis_titles(self, test_anndata_with_pca):
        import plotly.graph_objects as go
        figure = plot_pca_3d_scatter(
            test_anndata_with_pca, group_by="dilution", pc_x=2, pc_y=3, pc_z=4
        )
        assert isinstance(figure, go.Figure)
        assert "PC2" in figure.layout.scene.xaxis.title.text

    def test_missing_hover_column_prints_warning_does_not_raise(
        self, test_anndata_with_pca, capsys
    ):
        plot_pca_3d_scatter(
            test_anndata_with_pca,
            group_by="dilution",
            hover_columns=["nonexistent_column"],
        )
        captured = capsys.readouterr()
        assert "nonexistent_column" in captured.out

    def test_subsampling_caps_events_per_group(self, test_anndata_with_pca):
        figure = plot_pca_3d_scatter(
            test_anndata_with_pca, group_by="dilution", n_events_per_group=10
        )
        # The single data trace holds all plotted events (≤ n_events_per_group × n_groups).
        data_trace = next(t for t in figure.data if not t.showlegend)
        assert len(data_trace.x) <= 10 * 2  # 2 groups × 10 events each

    def test_group_by_subject_id_creates_three_traces(self, test_anndata_with_pca):
        figure = plot_pca_3d_scatter(test_anndata_with_pca, group_by="subject_ID")
        # 1 data trace + 3 invisible legend traces (one per subject)
        assert len(figure.data) == 4


# ---------------------------------------------------------------------------
# Tests: plot_pca_3d_biplot
# ---------------------------------------------------------------------------


class TestPlotPca3dBiplot:
    """Integration tests for plot_pca_3d_biplot()."""

    def test_returns_go_figure(self, test_anndata_with_pca):
        import plotly.graph_objects as go
        result = plot_pca_3d_biplot(test_anndata_with_pca, group_by="dilution")
        assert isinstance(result, go.Figure)

    def test_centroid_traces_one_per_group(self, test_anndata_with_pca):
        figure = plot_pca_3d_biplot(test_anndata_with_pca, group_by="dilution")
        # 2 centroid traces + 2 arrow traces (lines + labels)
        centroid_traces = [t for t in figure.data if t.name in {"Diluted", "Not_Diluted"}]
        assert len(centroid_traces) == 2

    def test_loading_arrow_traces_present(self, test_anndata_with_pca):
        figure = plot_pca_3d_biplot(
            test_anndata_with_pca, group_by="dilution", top_n_variables=3
        )
        # One "Loadings" line trace is always present when top_n > 0.
        loading_traces = [t for t in figure.data if t.name == "Loadings"]
        assert len(loading_traces) == 1

    def test_axis_labels_include_variance_percentage(self, test_anndata_with_pca):
        figure = plot_pca_3d_biplot(test_anndata_with_pca, group_by="dilution")
        assert "%" in figure.layout.scene.xaxis.title.text
        assert "PC1" in figure.layout.scene.xaxis.title.text

    def test_raises_for_invalid_group_by(self, test_anndata_with_pca):
        with pytest.raises(ValueError, match="not found in .obs"):
            plot_pca_3d_biplot(test_anndata_with_pca, group_by="nonexistent")

    def test_raises_when_pca_not_computed(self, test_anndata):
        with pytest.raises(ValueError, match="PCA not yet computed"):
            plot_pca_3d_biplot(test_anndata, group_by="dilution")

    def test_raises_for_pc_z_out_of_range(self, test_anndata_with_pca):
        with pytest.raises(ValueError, match="pc_z=99 is out of range"):
            plot_pca_3d_biplot(test_anndata_with_pca, group_by="dilution", pc_z=99)

    def test_raises_for_duplicate_pc_indices(self, test_anndata_with_pca):
        with pytest.raises(ValueError, match="must all be distinct"):
            plot_pca_3d_biplot(test_anndata_with_pca, group_by="dilution", pc_x=2, pc_y=2, pc_z=3)

    def test_custom_pc_axes_reflected_in_axis_titles(self, test_anndata_with_pca):
        figure = plot_pca_3d_biplot(
            test_anndata_with_pca, group_by="dilution", pc_x=2, pc_y=3, pc_z=4
        )
        assert "PC2" in figure.layout.scene.xaxis.title.text
        assert "PC3" in figure.layout.scene.yaxis.title.text

    def test_centroids_and_arrows_share_scale(self, test_anndata_with_pca):
        """All points (centroids + arrow tips) must be within radius 0.9 of origin."""
        import plotly.graph_objects as go
        import math
        figure = plot_pca_3d_biplot(
            test_anndata_with_pca, group_by="dilution", top_n_variables=3
        )
        for trace in figure.data:
            if not isinstance(trace, go.Scatter3d):
                continue
            # Skip loading label trace (mode="text") and the None-containing line trace.
            if trace.mode == "text":
                continue
            for x, y, z in zip(trace.x, trace.y, trace.z):
                if x is None:
                    continue
                norm = math.sqrt(x ** 2 + y ** 2 + z ** 2)
                assert norm <= 0.91, f"Point ({x},{y},{z}) has norm {norm:.3f} > 0.9"


# ---------------------------------------------------------------------------
# Tests: plot_pca_3d_trajectory
# ---------------------------------------------------------------------------


class TestPlotPca3dTrajectory:
    """Integration tests for plot_pca_3d_trajectory()."""

    def test_returns_go_figure(self, trajectory_anndata_with_pca):
        import plotly.graph_objects as go
        result = plot_pca_3d_trajectory(
            trajectory_anndata_with_pca, condition_column="diet", time_column="age"
        )
        assert isinstance(result, go.Figure)

    def test_one_trajectory_trace_per_condition(self, trajectory_anndata_with_pca):
        figure = plot_pca_3d_trajectory(
            trajectory_anndata_with_pca, condition_column="diet", time_column="age"
        )
        # 2 condition traces + 2 arrow traces (lines + labels)
        condition_traces = [t for t in figure.data if t.name in {"AL", "IF"}]
        assert len(condition_traces) == 2

    def test_show_loading_arrows_false_removes_arrow_traces(
        self, trajectory_anndata_with_pca
    ):
        figure = plot_pca_3d_trajectory(
            trajectory_anndata_with_pca,
            condition_column="diet",
            time_column="age",
            show_loading_arrows=False,
        )
        loading_traces = [t for t in figure.data if t.name == "Loadings"]
        assert len(loading_traces) == 0

    def test_show_loading_arrows_true_adds_arrow_traces(
        self, trajectory_anndata_with_pca
    ):
        figure = plot_pca_3d_trajectory(
            trajectory_anndata_with_pca,
            condition_column="diet",
            time_column="age",
            show_loading_arrows=True,
        )
        loading_traces = [t for t in figure.data if t.name == "Loadings"]
        assert len(loading_traces) == 1

    def test_raises_for_missing_condition_column(self, trajectory_anndata_with_pca):
        with pytest.raises(ValueError, match="condition_column 'nonexistent'"):
            plot_pca_3d_trajectory(
                trajectory_anndata_with_pca,
                condition_column="nonexistent",
                time_column="age",
            )

    def test_raises_for_missing_time_column(self, trajectory_anndata_with_pca):
        with pytest.raises(ValueError, match="time_column 'nonexistent'"):
            plot_pca_3d_trajectory(
                trajectory_anndata_with_pca,
                condition_column="diet",
                time_column="nonexistent",
            )

    def test_raises_when_pca_not_computed(self):
        adata = _make_trajectory_anndata()
        with pytest.raises(ValueError, match="PCA not yet computed"):
            plot_pca_3d_trajectory(adata, condition_column="diet", time_column="age")

    def test_raises_for_pc_z_out_of_range(self, trajectory_anndata_with_pca):
        with pytest.raises(ValueError, match="pc_z=99 is out of range"):
            plot_pca_3d_trajectory(
                trajectory_anndata_with_pca,
                condition_column="diet",
                time_column="age",
                pc_z=99,
            )

    def test_raises_for_duplicate_pc_indices(self, trajectory_anndata_with_pca):
        with pytest.raises(ValueError, match="must all be distinct"):
            plot_pca_3d_trajectory(
                trajectory_anndata_with_pca,
                condition_column="diet",
                time_column="age",
                pc_x=1,
                pc_y=1,
                pc_z=3,
            )

    def test_custom_pc_axes_reflected_in_axis_titles(self, trajectory_anndata_with_pca):
        figure = plot_pca_3d_trajectory(
            trajectory_anndata_with_pca,
            condition_column="diet",
            time_column="age",
            pc_x=2,
            pc_y=3,
            pc_z=4,
        )
        assert "PC2" in figure.layout.scene.xaxis.title.text

    def test_partial_timepoints_do_not_crash(self):
        """A condition missing a timepoint must not crash — trajectory line breaks."""
        adata = _make_trajectory_anndata(numeric_time=True)
        mask = ~((adata.obs["diet"] == "IF") & (adata.obs["age"] == 8))
        adata = adata[mask].copy()
        adata.uns[_ANALYSIS_CONFIG_KEY] = {"active_layer": None, "active_selection": None}
        compute_pca(adata, n_components=5)
        import plotly.graph_objects as go
        result = plot_pca_3d_trajectory(adata, condition_column="diet", time_column="age")
        assert isinstance(result, go.Figure)

    def test_time_order_respected(self, trajectory_anndata_with_pca):
        """Custom time_order must run without error."""
        import plotly.graph_objects as go
        result = plot_pca_3d_trajectory(
            trajectory_anndata_with_pca,
            condition_column="diet",
            time_column="age",
            time_order=[8, 2, 0],
        )
        assert isinstance(result, go.Figure)


# ---------------------------------------------------------------------------
# Tests: projection parameters (frozen reference space, ADR-015)
# ---------------------------------------------------------------------------

from mito_marker.analysis.normalization import (  # noqa: E402
    _fit_and_apply_layer,
    _store_layer_parameters,
)
from mito_marker.analysis.pca_plot import (  # noqa: E402
    _PCA_FINGERPRINT_KEY,
    _PCA_INPUT_FINGERPRINT_KEY,
    _PCA_INPUT_LAYER_KEY,
    _PCA_MEAN_KEY,
)


def _make_anndata_with_frozen_layer() -> anndata.AnnData:
    """Synthetic AnnData normalized through the frozen-parameter path."""
    adata = _make_test_anndata()
    processed, parameters = _fit_and_apply_layer(
        adata.X.astype(np.float32, copy=True), "none", "zscore_col",
        adata.var_names.tolist(), "none__zscore_col",
    )
    adata.layers["none__zscore_col"] = processed
    _store_layer_parameters(adata, parameters)
    adata.uns[_ANALYSIS_CONFIG_KEY]["active_layer"] = "none__zscore_col"
    return adata


class TestPcaProjectionParameters:
    """compute_pca() keeps what is needed to project new data on the same axes."""

    def test_manual_projection_reproduces_coordinates(self):
        adata = _make_anndata_with_frozen_layer()
        compute_pca(adata, n_components=5)
        projected = (
            (adata.layers["none__zscore_col"] - adata.uns[_PCA_MEAN_KEY])
            @ adata.uns[_PCA_LOADINGS_KEY]
        )
        np.testing.assert_allclose(projected, adata.obsm[_OBSM_PCA_KEY], atol=1e-4)

    def test_weighted_projection_reproduces_coordinates(self):
        adata = _make_anndata_with_frozen_layer()
        adata.obs["specie"] = "Mouse"
        compute_pca(adata, n_components=5, weight_by=["specie", "subject_ID"])
        projected = (
            (adata.layers["none__zscore_col"] - adata.uns[_PCA_MEAN_KEY])
            @ adata.uns[_PCA_LOADINGS_KEY]
        )
        np.testing.assert_allclose(projected, adata.obsm[_OBSM_PCA_KEY], atol=1e-4)

    def test_input_layer_and_fingerprints_recorded(self):
        adata = _make_anndata_with_frozen_layer()
        compute_pca(adata, n_components=5)
        assert adata.uns[_PCA_INPUT_LAYER_KEY] == "none__zscore_col"
        layer_fingerprint = adata.uns["layer_parameters"]["none__zscore_col"]["fingerprint"]
        assert adata.uns[_PCA_INPUT_FINGERPRINT_KEY] == layer_fingerprint
        assert len(adata.uns[_PCA_FINGERPRINT_KEY]) == 12

    def test_raw_x_input_is_recorded(self, test_anndata):
        compute_pca(test_anndata, n_components=5)
        assert test_anndata.uns[_PCA_INPUT_LAYER_KEY] == ""
        assert test_anndata.uns[_PCA_INPUT_FINGERPRINT_KEY] == "raw_X"

    def test_layer_without_parameters_is_flagged(self, capsys):
        adata = _make_test_anndata(with_layer=True)
        compute_pca(adata, n_components=5)
        assert adata.uns[_PCA_INPUT_FINGERPRINT_KEY] == "unknown"
        assert "cannot be used as a reference space" in capsys.readouterr().out

    def test_subset_inherits_reference_axes(self):
        adata = _make_anndata_with_frozen_layer()
        compute_pca(adata, n_components=5)
        subset = adata[adata.obs["subject_ID"] == "S001"].copy()
        # A second call on the subset hits the cache: axes are NOT refitted.
        compute_pca(subset, n_components=5)
        assert subset.uns[_PCA_FINGERPRINT_KEY] == adata.uns[_PCA_FINGERPRINT_KEY]
