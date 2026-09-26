"""
test_analysis_shap_clustering.py

Unit tests for mito_marker.analysis.shap_clustering.

Covers:
  _extract_per_sample_shap():
    - Reads logo_shap_oof_values when present (preferred).
    - Falls back to shap_values when logo_shap_oof_values is absent.
    - Returns correct shape (n_samples, n_features).
    - positive_class_index selects correct class array.
    - Raises AssertionError when neither SHAP source is available.

  _run_clustering():
    - HDBScan returns int array of same length as input.
    - KMeans returns int array with exactly n_clusters unique labels.
    - Unknown method raises AssertionError.
    - KMeans without n_clusters raises AssertionError.

  cluster_shap_values():
    - Returns DataFrame with shap_cluster column.
    - Returns DataFrame with {feature}__shap columns.
    - Stores result in .uns['shap_clustering_results'].
    - Raises AssertionError when ml_results missing from .uns.
    - Works with method="kmeans" and n_clusters=2.

  plot_shap_cluster_heatmap():
    - Returns a matplotlib Figure.
    - Feature subset (feature_names parameter) filters columns.
    - Custom title used verbatim.
    - Default title applied when empty string passed.
"""

import numpy as np
import pandas as pd
import pytest
import anndata
import matplotlib

matplotlib.use("Agg")

from mito_marker.analysis.shap_clustering import (
    _extract_per_sample_shap,
    _run_clustering,
    cluster_shap_values,
    plot_shap_cluster_heatmap,
)


# ---------------------------------------------------------------------------
# Helpers to build mock ml_results
# ---------------------------------------------------------------------------


def _make_ml_results_logo(
    n_samples: int = 40,
    n_features: int = 5,
    n_classes: int = 2,
    random_state: int = 0,
) -> dict:
    """Build a minimal ml_results dict as produced by LOGO + SHAP."""
    rng = np.random.default_rng(random_state)
    logo_oof_shap = [
        rng.standard_normal((n_samples, n_features)).astype(np.float32)
        for _ in range(n_classes)
    ]
    return {
        "logo_shap_oof_values": logo_oof_shap,
        "logo_shap_oof_features": rng.random((n_samples, n_features)).astype(np.float32),
        "shap_feature_names": [f"feat_{i}" for i in range(n_features)],
        "feature_names": [f"feat_{i}" for i in range(n_features)],
        "evaluation_strategy": "LOGO",
    }


def _make_ml_results_standard_cv(
    n_samples: int = 40,
    n_features: int = 5,
    random_state: int = 0,
) -> dict:
    """Build a minimal ml_results dict as produced by StandardCV + SHAP (binary)."""
    rng = np.random.default_rng(random_state)
    # For binary StandardCV, shap_values is the array for the positive class
    shap_matrix = rng.standard_normal((n_samples, n_features)).astype(np.float32)
    return {
        "logo_shap_oof_values": None,
        "shap_values": shap_matrix,
        "shap_feature_names": [f"feat_{i}" for i in range(n_features)],
        "feature_names": [f"feat_{i}" for i in range(n_features)],
        "evaluation_strategy": "StandardCV",
    }


def _make_anndata_with_logo_shap(
    n_samples: int = 40,
    n_features: int = 5,
) -> anndata.AnnData:
    """Build an AnnData object with ml_results already in .uns."""
    rng = np.random.default_rng(0)
    obs = pd.DataFrame({"condition": ["young"] * (n_samples // 2) + ["old"] * (n_samples // 2)})
    adata = anndata.AnnData(
        X=rng.random((n_samples, n_features)).astype(np.float32),
        obs=obs,
    )
    adata.uns["ml_results"] = _make_ml_results_logo(n_samples=n_samples, n_features=n_features)
    return adata


# ---------------------------------------------------------------------------
# Tests for _extract_per_sample_shap
# ---------------------------------------------------------------------------


class TestExtractPerSampleShap:
    def test_reads_logo_oof_values_preferred(self):
        ml_results = _make_ml_results_logo(n_samples=40, n_features=5)
        shap_matrix, feature_names = _extract_per_sample_shap(
            ml_results=ml_results, positive_class_index=1
        )
        assert shap_matrix.shape == (40, 5)

    def test_fallback_to_standard_cv_shap(self):
        ml_results = _make_ml_results_standard_cv(n_samples=30, n_features=4)
        shap_matrix, feature_names = _extract_per_sample_shap(
            ml_results=ml_results, positive_class_index=1
        )
        assert shap_matrix.shape == (30, 4)

    def test_positive_class_index_selects_correct_class(self):
        rng = np.random.default_rng(7)
        class_0_shap = rng.normal(0, 1, (20, 3)).astype(np.float32)
        class_1_shap = rng.normal(10, 1, (20, 3)).astype(np.float32)
        ml_results = {
            "logo_shap_oof_values": [class_0_shap, class_1_shap],
            "shap_feature_names": ["a", "b", "c"],
        }
        shap_matrix_0, _ = _extract_per_sample_shap(ml_results, positive_class_index=0)
        shap_matrix_1, _ = _extract_per_sample_shap(ml_results, positive_class_index=1)
        assert shap_matrix_0.mean() < 1.0   # class 0 centered near 0
        assert shap_matrix_1.mean() > 5.0   # class 1 centered near 10

    def test_returns_feature_names(self):
        ml_results = _make_ml_results_logo(n_features=6)
        _, feature_names = _extract_per_sample_shap(ml_results, positive_class_index=1)
        assert len(feature_names) == 6

    def test_raises_when_no_shap_available(self):
        ml_results = {
            "logo_shap_oof_values": None,
            "shap_values": None,
            "shap_feature_names": ["a", "b"],
        }
        with pytest.raises(AssertionError):
            _extract_per_sample_shap(ml_results, positive_class_index=1)

    def test_shap_matrix_dtype_float32(self):
        ml_results = _make_ml_results_logo(n_samples=20, n_features=4)
        shap_matrix, _ = _extract_per_sample_shap(ml_results, positive_class_index=0)
        assert shap_matrix.dtype == np.float32


# ---------------------------------------------------------------------------
# Tests for _run_clustering
# ---------------------------------------------------------------------------


class TestRunClustering:
    def test_hdbscan_returns_correct_length(self):
        rng = np.random.default_rng(0)
        shap_scaled = rng.random((50, 5)).astype(np.float32)
        labels = _run_clustering(
            shap_matrix_scaled=shap_scaled,
            method="hdbscan",
            n_clusters=None,
            min_cluster_size=5,
            random_state=0,
        )
        assert len(labels) == 50

    def test_kmeans_returns_exactly_n_clusters(self):
        rng = np.random.default_rng(0)
        # Clearly separable data to guarantee k-means finds k clusters
        data = np.vstack([rng.normal([i * 10, i * 10], 0.1, (20, 2)) for i in range(3)])
        labels = _run_clustering(
            shap_matrix_scaled=data.astype(np.float32),
            method="kmeans",
            n_clusters=3,
            min_cluster_size=5,
            random_state=0,
        )
        assert len(set(labels)) == 3

    def test_unknown_method_raises(self):
        shap_scaled = np.random.rand(20, 5).astype(np.float32)
        with pytest.raises(AssertionError):
            _run_clustering(
                shap_matrix_scaled=shap_scaled,
                method="birch",
                n_clusters=None,
                min_cluster_size=5,
                random_state=0,
            )

    def test_kmeans_without_n_clusters_raises(self):
        shap_scaled = np.random.rand(20, 5).astype(np.float32)
        with pytest.raises(AssertionError):
            _run_clustering(
                shap_matrix_scaled=shap_scaled,
                method="kmeans",
                n_clusters=None,
                min_cluster_size=5,
                random_state=0,
            )


# ---------------------------------------------------------------------------
# Tests for cluster_shap_values
# ---------------------------------------------------------------------------


class TestClusterShapValues:
    def test_returns_dataframe(self):
        adata = _make_anndata_with_logo_shap()
        shap_df = cluster_shap_values(adata, method="hdbscan", min_cluster_size=3)
        assert isinstance(shap_df, pd.DataFrame)

    def test_shap_cluster_column_present(self):
        adata = _make_anndata_with_logo_shap()
        shap_df = cluster_shap_values(adata, method="hdbscan", min_cluster_size=3)
        assert "shap_cluster" in shap_df.columns

    def test_shap_feature_columns_present(self):
        adata = _make_anndata_with_logo_shap(n_features=5)
        shap_df = cluster_shap_values(adata, method="hdbscan", min_cluster_size=3)
        shap_cols = [c for c in shap_df.columns if c.endswith("__shap")]
        assert len(shap_cols) == 5

    def test_result_stored_in_uns(self):
        adata = _make_anndata_with_logo_shap()
        cluster_shap_values(adata, method="hdbscan", min_cluster_size=3)
        assert "shap_clustering_results" in adata.uns

    def test_raises_when_ml_results_missing(self):
        rng = np.random.default_rng(0)
        adata = anndata.AnnData(X=rng.random((20, 3)))
        with pytest.raises(AssertionError):
            cluster_shap_values(adata)

    def test_works_with_kmeans(self):
        adata = _make_anndata_with_logo_shap(n_samples=40, n_features=5)
        shap_df = cluster_shap_values(adata, method="kmeans", n_clusters=2)
        assert len(set(shap_df["shap_cluster"])) == 2

    def test_row_count_matches_oof_samples(self):
        n_oof = 40
        adata = _make_anndata_with_logo_shap(n_samples=n_oof, n_features=5)
        shap_df = cluster_shap_values(adata, method="hdbscan", min_cluster_size=3)
        assert len(shap_df) == n_oof


# ---------------------------------------------------------------------------
# Tests for plot_shap_cluster_heatmap
# ---------------------------------------------------------------------------


class TestPlotShapClusterHeatmap:
    def _make_shap_df(self) -> pd.DataFrame:
        rng = np.random.default_rng(0)
        n = 60
        shap_data = rng.standard_normal((n, 4)).astype(np.float32)
        df = pd.DataFrame(
            shap_data,
            columns=["area__shap", "perimeter__shap", "circularity__shap", "ar__shap"],
        )
        df.insert(0, "shap_cluster", [0] * 20 + [1] * 20 + [-1] * 20)
        return df

    def test_returns_figure(self):
        import matplotlib.figure
        shap_df = self._make_shap_df()
        fig = plot_shap_cluster_heatmap(shap_df)
        assert isinstance(fig, matplotlib.figure.Figure)
        import matplotlib.pyplot as plt
        plt.close("all")

    def test_custom_title_used(self):
        shap_df = self._make_shap_df()
        fig = plot_shap_cluster_heatmap(shap_df, title="My SHAP Heatmap")
        # seaborn clustermap places the title via fig.suptitle (stored in fig.texts).
        all_text = " ".join(t.get_text() for t in fig.texts)
        assert "My SHAP Heatmap" in all_text
        import matplotlib.pyplot as plt
        plt.close("all")

    def test_default_title_when_empty(self):
        shap_df = self._make_shap_df()
        fig = plot_shap_cluster_heatmap(shap_df, title="")
        all_text = " ".join(t.get_text() for t in fig.texts)
        assert len(all_text) > 0
        import matplotlib.pyplot as plt
        plt.close("all")

    def test_feature_subset_filters_columns(self):
        shap_df = self._make_shap_df()
        # Only show 2 of the 4 features.
        fig = plot_shap_cluster_heatmap(
            shap_df, feature_names=["area", "circularity"]
        )
        # seaborn clustermap: find the heatmap axis (the one with x-tick labels).
        heatmap_axis = next(
            ax for ax in fig.axes if len(ax.get_xticklabels()) > 0
        )
        tick_labels = [t.get_text() for t in heatmap_axis.get_xticklabels()]
        assert len(tick_labels) == 2
        import matplotlib.pyplot as plt
        plt.close("all")
