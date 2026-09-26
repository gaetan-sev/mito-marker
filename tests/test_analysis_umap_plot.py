"""
test_analysis_umap_plot.py

Unit tests for mito_marker.analysis.umap_plot.

Tests verify:
  - compute_umap() runs on a small synthetic AnnData and populates .obsm.
  - .obsm key is named correctly ('X_umap_n{n}_d{min_dist}').
  - Second call with same parameters does not recompute (cache hit).
  - Different parameters create a distinct .obsm key.
  - plot_umap() runs without error on a pre-populated AnnData (Agg backend).
  - plot_umap() raises ValueError for a missing group_by column.
  - plot_umap() raises ValueError when the expected .obsm key is absent.
  - _stratified_sample() returns at most max_total indices.
  - _stratified_sample() is balanced across subjects when subject_ID present.
  - _build_umap_key() produces correctly formatted key strings.
  - _get_active_umap_key() reads from .uns; falls back to default key.
"""

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest
import anndata

from mito_marker.analysis.umap_plot import (
    _build_umap_key,
    _get_active_umap_key,
    _get_data_matrix,
    _get_group_labels,
    _stratified_sample,
    compute_umap,
    plot_umap,
)
from mito_marker.analysis.colors import assign_color_palette
from mito_marker.analysis.selection import _ANALYSIS_CONFIG_KEY


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

N_OBS = 120
N_VARS = 10


def _make_test_anndata(
    n_obs: int = N_OBS,
    n_vars: int = N_VARS,
    with_layer: bool = False,
) -> anndata.AnnData:
    """Small synthetic AnnData with 3 subjects and 2 conditions."""
    np.random.seed(7)
    x_matrix = np.abs(np.random.randn(n_obs, n_vars) * 3).astype(np.float32)
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
        adata.layers["arcsinh__zscore_col"] = (x_matrix - x_matrix.mean(axis=0)) / (x_matrix.std(axis=0) + 1e-8)
        adata.uns[_ANALYSIS_CONFIG_KEY]["active_layer"] = "arcsinh__zscore_col"
    return adata


def _make_umap_populated_anndata() -> anndata.AnnData:
    """AnnData with pre-populated UMAP coordinates (no UMAP fitting required)."""
    adata = _make_test_anndata()
    assign_color_palette(adata)
    # Inject fake UMAP coordinates directly to avoid running actual UMAP.
    np.random.seed(99)
    fake_coords = np.random.randn(N_OBS, 2).astype(np.float32)
    key = _build_umap_key(15, 0.1)
    adata.obsm[key] = fake_coords
    adata.uns[_ANALYSIS_CONFIG_KEY]["umap_params"] = {
        "active_key": key,
        "n_neighbors": 15,
        "min_dist": 0.1,
        "random_state": 42,
        "n_events_sampled": N_OBS,
    }
    return adata


# ---------------------------------------------------------------------------
# Tests: _build_umap_key
# ---------------------------------------------------------------------------


class TestBuildUmapKey:
    """Unit tests for the .obsm key naming convention."""

    def test_default_parameters(self):
        assert _build_umap_key(15, 0.1) == "X_umap_n15_d0.1"

    def test_different_parameters(self):
        assert _build_umap_key(30, 0.5) == "X_umap_n30_d0.5"

    def test_starts_with_x_umap(self):
        key = _build_umap_key(10, 0.2)
        assert key.startswith("X_umap_")

    def test_contains_n_neighbors(self):
        key = _build_umap_key(25, 0.3)
        assert "n25" in key

    def test_contains_min_dist(self):
        key = _build_umap_key(15, 0.05)
        assert "d0.05" in key


# ---------------------------------------------------------------------------
# Tests: _get_active_umap_key
# ---------------------------------------------------------------------------


class TestGetActiveUmapKey:
    """Unit tests for reading the active UMAP key from .uns."""

    def test_returns_stored_key(self):
        adata = _make_test_anndata()
        adata.uns[_ANALYSIS_CONFIG_KEY]["umap_params"] = {
            "active_key": "X_umap_n30_d0.5"
        }
        assert _get_active_umap_key(adata) == "X_umap_n30_d0.5"

    def test_falls_back_to_default_when_no_config(self):
        adata = _make_test_anndata()
        del adata.uns[_ANALYSIS_CONFIG_KEY]
        key = _get_active_umap_key(adata)
        assert key == "X_umap_n15_d0.1"

    def test_falls_back_to_default_when_no_umap_params(self):
        adata = _make_test_anndata()
        # No umap_params key inside analysis_config.
        key = _get_active_umap_key(adata)
        assert key == "X_umap_n15_d0.1"


# ---------------------------------------------------------------------------
# Tests: _stratified_sample
# ---------------------------------------------------------------------------


class TestStratifiedSample:
    """Unit tests for the stratified random sampling helper."""

    def test_returns_all_when_n_obs_less_than_max(self):
        obs = pd.DataFrame({
            "subject_ID": ["S001"] * 50,
        })
        indices = _stratified_sample(obs, max_total=200, random_state=0)
        assert len(indices) == 50

    def test_returns_at_most_max_total(self):
        obs = pd.DataFrame({
            "subject_ID": np.tile(["S001", "S002", "S003"], 100),
        })
        indices = _stratified_sample(obs, max_total=50, random_state=0)
        assert len(indices) <= 50

    def test_indices_are_within_range(self):
        n = 300
        obs = pd.DataFrame({
            "subject_ID": np.tile(["S001", "S002"], n // 2),
        })
        indices = _stratified_sample(obs, max_total=100, random_state=0)
        assert indices.min() >= 0
        assert indices.max() < n

    def test_no_duplicate_indices(self):
        obs = pd.DataFrame({
            "subject_ID": np.tile(["S001", "S002", "S003"], 100),
        })
        indices = _stratified_sample(obs, max_total=200, random_state=0)
        assert len(indices) == len(np.unique(indices))

    def test_balanced_across_subjects(self):
        """Each subject should contribute roughly equal numbers of events."""
        obs = pd.DataFrame({
            "subject_ID": np.tile(["S001", "S002", "S003"], 200),
        })
        indices = _stratified_sample(obs, max_total=90, random_state=0)
        subject_labels = obs["subject_ID"].values[indices]
        counts = {s: (subject_labels == s).sum() for s in ["S001", "S002", "S003"]}
        # Each subject should have at most 2x as many events as any other.
        min_count = min(counts.values())
        max_count = max(counts.values())
        assert max_count <= min_count * 2 + 5  # Allow small rounding slack.

    def test_fallback_when_no_subject_id_column(self):
        """Simple random sample when subject_ID is not present."""
        obs = pd.DataFrame({"dilution": ["Diluted"] * 500})
        indices = _stratified_sample(obs, max_total=100, random_state=0)
        assert len(indices) == 100


# ---------------------------------------------------------------------------
# Tests: compute_umap — integration (requires umap-learn)
# ---------------------------------------------------------------------------


class TestComputeUmap:
    """Integration tests for compute_umap()."""

    def test_obsm_key_created(self):
        adata = _make_test_anndata()
        compute_umap(adata, n_neighbors=5, min_dist=0.1, max_total_events=60)
        assert "X_umap_n5_d0.1" in adata.obsm

    def test_obsm_shape_is_n_obs_by_2(self):
        adata = _make_test_anndata()
        compute_umap(adata, n_neighbors=5, min_dist=0.1, max_total_events=60)
        coords = adata.obsm["X_umap_n5_d0.1"]
        assert coords.shape == (N_OBS, 2)

    def test_sampled_events_have_valid_coords(self):
        """Events included in the UMAP sample must have finite coordinates."""
        adata = _make_test_anndata()
        compute_umap(adata, n_neighbors=5, min_dist=0.1, max_total_events=60)
        coords = adata.obsm["X_umap_n5_d0.1"]
        # At least max_total_events rows should be finite.
        valid_count = np.isfinite(coords[:, 0]).sum()
        assert valid_count >= min(60, N_OBS)

    def test_returns_anndata_object(self):
        adata = _make_test_anndata()
        result = compute_umap(adata, n_neighbors=5, min_dist=0.1, max_total_events=60)
        assert isinstance(result, anndata.AnnData)
        assert result is adata

    def test_analysis_config_active_key_updated(self):
        adata = _make_test_anndata()
        compute_umap(adata, n_neighbors=5, min_dist=0.1, max_total_events=60)
        active_key = adata.uns[_ANALYSIS_CONFIG_KEY]["umap_params"]["active_key"]
        assert active_key == "X_umap_n5_d0.1"

    def test_cache_hit_does_not_recalculate(self):
        """A second call with the same parameters must skip computation."""
        adata = _make_test_anndata()
        compute_umap(adata, n_neighbors=5, min_dist=0.1, max_total_events=60)
        original_coords = adata.obsm["X_umap_n5_d0.1"].copy()

        # Manually corrupt to detect if recomputed.
        adata.obsm["X_umap_n5_d0.1"][0, 0] = 9999.0

        compute_umap(adata, n_neighbors=5, min_dist=0.1, max_total_events=60)
        # If cache worked, the tampered value must still be there.
        assert adata.obsm["X_umap_n5_d0.1"][0, 0] == 9999.0

    def test_force_recompute_overwrites_cached_coordinates(self):
        """force_recompute=True must drop the cached key and recompute."""
        adata = _make_test_anndata()
        compute_umap(adata, n_neighbors=5, min_dist=0.1, max_total_events=60)

        # Manually corrupt to detect if recomputed.
        adata.obsm["X_umap_n5_d0.1"][0, 0] = 9999.0

        compute_umap(adata, n_neighbors=5, min_dist=0.1, max_total_events=60, force_recompute=True)
        # After forced recompute the tampered value must be gone.
        assert adata.obsm["X_umap_n5_d0.1"][0, 0] != 9999.0

    def test_force_recompute_false_keeps_cache(self):
        """force_recompute=False (default) must keep the existing cache."""
        adata = _make_test_anndata()
        compute_umap(adata, n_neighbors=5, min_dist=0.1, max_total_events=60)
        adata.obsm["X_umap_n5_d0.1"][0, 0] = 9999.0
        compute_umap(adata, n_neighbors=5, min_dist=0.1, max_total_events=60, force_recompute=False)
        assert adata.obsm["X_umap_n5_d0.1"][0, 0] == 9999.0

    def test_different_parameters_create_new_key(self):
        adata = _make_test_anndata()
        compute_umap(adata, n_neighbors=5, min_dist=0.1, max_total_events=60)
        compute_umap(adata, n_neighbors=10, min_dist=0.3, max_total_events=60)
        assert "X_umap_n5_d0.1" in adata.obsm
        assert "X_umap_n10_d0.3" in adata.obsm

    def test_max_total_events_limits_sample_size(self):
        """Even with max_total_events smaller than n_obs, coord array is still n_obs × 2."""
        adata = _make_test_anndata()
        compute_umap(adata, n_neighbors=5, min_dist=0.1, max_total_events=30)
        coords = adata.obsm["X_umap_n5_d0.1"]
        assert coords.shape[0] == N_OBS
        # Only 30 events should have valid (non-NaN) coords.
        valid_count = np.isfinite(coords[:, 0]).sum()
        assert valid_count == 30


# ---------------------------------------------------------------------------
# Tests: plot_umap — integration (Agg backend, pre-populated .obsm)
# ---------------------------------------------------------------------------


class TestPlotUmap:
    """Integration tests for plot_umap() — no actual UMAP fitting required."""

    def test_group_by_subject_id_creates_figure(self):
        import matplotlib.pyplot as plt
        adata = _make_umap_populated_anndata()
        n_figs_before = len(plt.get_fignums())
        plot_umap(adata, group_by="subject_ID")
        assert len(plt.get_fignums()) == n_figs_before + 1
        plt.close("all")

    def test_group_by_dilution_creates_figure(self):
        import matplotlib.pyplot as plt
        adata = _make_umap_populated_anndata()
        n_figs_before = len(plt.get_fignums())
        plot_umap(adata, group_by="dilution")
        assert len(plt.get_fignums()) == n_figs_before + 1
        plt.close("all")

    def test_invalid_group_by_raises_value_error(self):
        adata = _make_umap_populated_anndata()
        with pytest.raises(ValueError, match="not found in .obs"):
            plot_umap(adata, group_by="nonexistent_column")

    def test_missing_obsm_key_raises_value_error(self):
        adata = _make_test_anndata()
        assign_color_palette(adata)
        # No .obsm key exists — should raise.
        with pytest.raises(ValueError, match="not found in .obsm"):
            plot_umap(adata, group_by="dilution")

    def test_n_events_per_group_limits_displayed(self):
        """Passing a tiny n_events_per_group must not raise an error."""
        import matplotlib.pyplot as plt
        adata = _make_umap_populated_anndata()
        plot_umap(adata, group_by="dilution", n_events_per_group=3)
        plt.close("all")


# ---------------------------------------------------------------------------
# Tests: non-analytical channel exclusion from _get_data_matrix
# ---------------------------------------------------------------------------


class TestGetDataMatrixNonAnalytical:
    """Tests that Time and FlowAI are excluded from the UMAP data matrix."""

    def _make_anndata_with_time_flowai(self) -> anndata.AnnData:
        """Synthetic AnnData that includes Time and FlowAI channels in .var."""
        np.random.seed(5)
        x_analytical = np.abs(np.random.randn(N_OBS, N_VARS) * 3).astype(np.float32)
        x_time = np.linspace(0, 10_000, N_OBS).reshape(-1, 1).astype(np.float32)
        x_flowai = np.ones((N_OBS, 1), dtype=np.float32) * 999.0
        x_matrix = np.hstack([x_analytical, x_time, x_flowai])
        var_names = [f"CH{i:02d}" for i in range(N_VARS)] + ["Time", "FlowAI"]
        obs = pd.DataFrame({
            "subject_ID": np.tile(["S001", "S002"], N_OBS // 2 + 1)[:N_OBS],
            "dilution": np.tile(["Diluted", "Not_Diluted"], N_OBS // 2 + 1)[:N_OBS],
        })
        var = pd.DataFrame(
            {"is_non_analytical": [False] * N_VARS + [True, True]},
            index=var_names,
        )
        adata = anndata.AnnData(X=x_matrix, obs=obs, var=var)
        adata.uns[_ANALYSIS_CONFIG_KEY] = {"active_layer": None, "active_selection": None}
        return adata

    def test_time_and_flowai_excluded_from_data_matrix(self):
        adata = self._make_anndata_with_time_flowai()
        matrix = _get_data_matrix(adata)
        # Only analytical channels should remain.
        assert matrix.shape == (N_OBS, N_VARS)

    def test_only_analytical_channels_remain_when_feature_selection_also_active(self):
        adata = self._make_anndata_with_time_flowai()
        # Apply feature selection selecting 5 analytical channels.
        n_analytical = N_VARS + 2  # total vars including Time and FlowAI
        is_selected = [True] * 5 + [False] * (N_VARS - 5) + [False, False]
        adata.var["HighVariance_score"] = [1.0] * 5 + [0.0] * (N_VARS - 5) + [0.0, 0.0]
        adata.var["is_selected_HighVariance"] = is_selected
        adata.uns[_ANALYSIS_CONFIG_KEY]["active_selection"] = "HighVariance"
        matrix = _get_data_matrix(adata)
        # Feature selection → 5 channels; Time/FlowAI already excluded by is_selected=False.
        assert matrix.shape == (N_OBS, 5)


# ---------------------------------------------------------------------------
# Tests: _get_group_labels
# ---------------------------------------------------------------------------


class TestGetGroupLabels:
    """Tests for the multi-column grouping label helper."""

    def test_single_string_returns_column_as_is(self):
        adata = _make_test_anndata()
        labels = _get_group_labels(adata, "dilution")
        assert list(labels) == list(adata.obs["dilution"])

    def test_single_element_list_returns_column_as_is(self):
        adata = _make_test_anndata()
        labels = _get_group_labels(adata, ["dilution"])
        assert list(labels) == list(adata.obs["dilution"])

    def test_two_columns_combined_with_slash(self):
        adata = _make_test_anndata()
        labels = _get_group_labels(adata, ["subject_ID", "dilution"])
        expected = adata.obs["subject_ID"].astype(str) + " / " + adata.obs["dilution"].astype(str)
        assert list(labels) == list(expected)

    def test_invalid_column_raises(self):
        adata = _make_test_anndata()
        with pytest.raises(ValueError, match="not found in .obs"):
            _get_group_labels(adata, "nonexistent_column")

    def test_invalid_column_in_list_raises(self):
        adata = _make_test_anndata()
        with pytest.raises(ValueError, match="not found in .obs"):
            _get_group_labels(adata, ["dilution", "nonexistent_column"])


# ---------------------------------------------------------------------------
# Tests: plot_umap with list group_by
# ---------------------------------------------------------------------------


class TestPlotUmapGroupByList:
    """Tests for plot_umap() with group_by as a list of column names."""

    def test_group_by_list_creates_figure(self):
        import matplotlib.pyplot as plt
        adata = _make_umap_populated_anndata()
        n_figs_before = len(plt.get_fignums())
        plot_umap(adata, group_by=["subject_ID", "dilution"])
        assert len(plt.get_fignums()) == n_figs_before + 1
        plt.close("all")

    def test_group_by_list_combined_labels_in_title(self):
        """Title should contain the × separator when group_by is a list."""
        import matplotlib.pyplot as plt
        adata = _make_umap_populated_anndata()
        plot_umap(adata, group_by=["subject_ID", "dilution"])
        fig = plt.gcf()
        title = fig.axes[0].get_title()
        assert "subject_ID × dilution" in title
        plt.close("all")

    def test_group_by_list_invalid_column_raises(self):
        adata = _make_umap_populated_anndata()
        with pytest.raises(ValueError, match="not found in .obs"):
            plot_umap(adata, group_by=["dilution", "nonexistent"])


# ---------------------------------------------------------------------------
# Tests: _build_umap_key with n_pca_components
# ---------------------------------------------------------------------------


class TestBuildUmapKeyWithPca:
    """Unit tests for the PCA-variant key naming convention."""

    def test_pca_key_format(self):
        assert _build_umap_key(15, 0.1, n_pca_components=10) == "X_umap_pca10_n15_d0.1"

    def test_pca_key_different_components(self):
        assert _build_umap_key(30, 0.5, n_pca_components=20) == "X_umap_pca20_n30_d0.5"

    def test_none_pca_gives_raw_key(self):
        """n_pca_components=None must produce the unchanged raw-channel key format."""
        assert _build_umap_key(15, 0.1, n_pca_components=None) == "X_umap_n15_d0.1"

    def test_pca_key_distinct_from_raw_key(self):
        raw_key = _build_umap_key(15, 0.1)
        pca_key = _build_umap_key(15, 0.1, n_pca_components=15)
        assert raw_key != pca_key

    def test_pca_key_contains_pca_prefix(self):
        key = _build_umap_key(15, 0.1, n_pca_components=5)
        assert "pca5" in key


# ---------------------------------------------------------------------------
# Tests: compute_umap with n_pca_components
# ---------------------------------------------------------------------------


def _inject_fake_pca(adata: anndata.AnnData, n_components: int = 20) -> anndata.AnnData:
    """Inject fake PCA coordinates into an AnnData's .obsm['X_pca']."""
    np.random.seed(13)
    adata.obsm["X_pca"] = np.random.randn(adata.n_obs, n_components).astype(np.float32)
    return adata


class TestComputeUmapOnPcaSpace:
    """Integration tests for compute_umap() with n_pca_components set."""

    def test_pca_key_created_in_obsm(self):
        adata = _make_test_anndata()
        _inject_fake_pca(adata, n_components=20)
        compute_umap(adata, n_neighbors=5, min_dist=0.1, max_total_events=60, n_pca_components=10)
        assert "X_umap_pca10_n5_d0.1" in adata.obsm

    def test_raw_key_not_created(self):
        """When n_pca_components is set, the raw key must not be added."""
        adata = _make_test_anndata()
        _inject_fake_pca(adata, n_components=20)
        compute_umap(adata, n_neighbors=5, min_dist=0.1, max_total_events=60, n_pca_components=10)
        assert "X_umap_n5_d0.1" not in adata.obsm

    def test_obsm_shape_is_n_obs_by_2(self):
        adata = _make_test_anndata()
        _inject_fake_pca(adata, n_components=20)
        compute_umap(adata, n_neighbors=5, min_dist=0.1, max_total_events=60, n_pca_components=10)
        coords = adata.obsm["X_umap_pca10_n5_d0.1"]
        assert coords.shape == (N_OBS, 2)

    def test_active_key_in_uns_is_pca_key(self):
        adata = _make_test_anndata()
        _inject_fake_pca(adata, n_components=20)
        compute_umap(adata, n_neighbors=5, min_dist=0.1, max_total_events=60, n_pca_components=10)
        active_key = adata.uns["analysis_config"]["umap_params"]["active_key"]
        assert active_key == "X_umap_pca10_n5_d0.1"

    def test_uns_records_n_pca_components(self):
        adata = _make_test_anndata()
        _inject_fake_pca(adata, n_components=20)
        compute_umap(adata, n_neighbors=5, min_dist=0.1, max_total_events=60, n_pca_components=10)
        stored = adata.uns["analysis_config"]["umap_params"]["n_pca_components"]
        assert stored == 10

    def test_raw_and_pca_keys_coexist(self):
        """Both raw-channel UMAP and PCA-space UMAP can coexist in .obsm."""
        adata = _make_test_anndata()
        _inject_fake_pca(adata, n_components=20)
        compute_umap(adata, n_neighbors=5, min_dist=0.1, max_total_events=60)
        compute_umap(adata, n_neighbors=5, min_dist=0.1, max_total_events=60, n_pca_components=10)
        assert "X_umap_n5_d0.1" in adata.obsm
        assert "X_umap_pca10_n5_d0.1" in adata.obsm

    def test_error_when_x_pca_missing(self):
        """AssertionError must be raised when X_pca is not in .obsm."""
        adata = _make_test_anndata()
        with pytest.raises(AssertionError, match="X_pca"):
            compute_umap(adata, n_neighbors=5, min_dist=0.1, max_total_events=60, n_pca_components=10)

    def test_error_when_n_pca_components_exceeds_available(self):
        """AssertionError must be raised when n_pca_components > stored PCA components."""
        adata = _make_test_anndata()
        _inject_fake_pca(adata, n_components=5)
        with pytest.raises(AssertionError, match="n_pca_components=10 exceeds"):
            compute_umap(adata, n_neighbors=5, min_dist=0.1, max_total_events=60, n_pca_components=10)

    def test_force_recompute_overwrites_pca_key(self):
        adata = _make_test_anndata()
        _inject_fake_pca(adata, n_components=20)
        compute_umap(adata, n_neighbors=5, min_dist=0.1, max_total_events=60, n_pca_components=10)
        adata.obsm["X_umap_pca10_n5_d0.1"][0, 0] = 9999.0
        compute_umap(adata, n_neighbors=5, min_dist=0.1, max_total_events=60,
                     n_pca_components=10, force_recompute=True)
        assert adata.obsm["X_umap_pca10_n5_d0.1"][0, 0] != 9999.0

    def test_cache_hit_on_second_call_with_pca(self):
        adata = _make_test_anndata()
        _inject_fake_pca(adata, n_components=20)
        compute_umap(adata, n_neighbors=5, min_dist=0.1, max_total_events=60, n_pca_components=10)
        adata.obsm["X_umap_pca10_n5_d0.1"][0, 0] = 9999.0
        compute_umap(adata, n_neighbors=5, min_dist=0.1, max_total_events=60, n_pca_components=10)
        # Cache hit — tampered value must remain.
        assert adata.obsm["X_umap_pca10_n5_d0.1"][0, 0] == 9999.0


# ---------------------------------------------------------------------------
# Tests: embed_all_events / keep_reducer (frozen reference space, ADR-015)
# ---------------------------------------------------------------------------

from mito_marker.analysis.umap_plot import UMAP_REDUCERS_KEY, _UMAP_PARAMS_KEY  # noqa: E402


class TestUmapReferenceSpace:
    """compute_umap() can embed every event and keep its reducer for reuse."""

    def test_default_leaves_unsampled_rows_nan(self):
        adata = _make_test_anndata(n_obs=150)
        compute_umap(adata, max_total_events=60)
        key = _get_active_umap_key(adata)
        assert np.isnan(adata.obsm[key][:, 0]).sum() == 90

    def test_embed_all_events_fills_every_row(self):
        adata = _make_test_anndata(n_obs=150)
        compute_umap(adata, max_total_events=60, embed_all_events=True)
        key = _get_active_umap_key(adata)
        assert np.isfinite(adata.obsm[key]).all()
        params = adata.uns[_ANALYSIS_CONFIG_KEY][_UMAP_PARAMS_KEY]
        assert params["n_events_embedded_by_transform"] == 90

    def test_keep_reducer_stores_loadable_reducer(self):
        import pickle

        adata = _make_test_anndata(n_obs=90)
        compute_umap(adata, keep_reducer=True)
        key = _get_active_umap_key(adata)
        reducer = pickle.loads(adata.uns[UMAP_REDUCERS_KEY][key].tobytes())
        new_coordinates = reducer.transform(adata.X[:5])
        assert new_coordinates.shape == (5, 2)
        params = adata.uns[_ANALYSIS_CONFIG_KEY][_UMAP_PARAMS_KEY]
        assert len(params["reducer_fingerprint"]) == 12
        assert params["input_channel_names"] == adata.var_names.tolist()
        assert params["input_fingerprint"] == "raw_X"

    def test_reducer_not_stored_by_default(self):
        adata = _make_test_anndata(n_obs=90)
        compute_umap(adata)
        assert UMAP_REDUCERS_KEY not in adata.uns
        assert adata.uns[_ANALYSIS_CONFIG_KEY][_UMAP_PARAMS_KEY]["reducer_fingerprint"] == ""
