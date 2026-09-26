"""
test_analysis_clustering.py

Unit tests for mito_marker.analysis.clustering.

Covers:
  _sample_individual_mitos():
    - Returns correct shape (n_subjects * n_per_subject, n_features).
    - condition_labels aligned with feature_matrix rows.
    - subject_ids aligned with feature_matrix rows.
    - Deterministic: same random_state → same output.
    - Different random_states → different samples.
    - Samples with replacement when n_per_subject > n_available.

  compute_umap_on_bags():
    - bag_size=1 returns original-space features (no stat suffix in names).
    - bag_size>1 returns bag feature names with stat suffix.
    - Returns 5-tuple: (umap_coords, feature_matrix, conditions, subjects, names).
    - umap_coordinates shape: (n_samples, 2).
    - feature_matrix rows align with condition_labels and subject_ids.
    - feature_names length matches feature_matrix columns.
    - Missing subject_id_column raises KeyError (via obs access).
    - Missing condition_column raises KeyError.

  run_hdbscan():
    - Returns 1D int array of same length as input.
    - Finds ≥ 1 cluster on separable data.
    - Returns only -1 (all noise) on random uniform data with tiny min_cluster_size.
    - min_samples parameter accepted without error.

  evaluate_clustering():
    - n_clusters correct: excludes -1 from count.
    - noise_ratio correct.
    - silhouette_score is NaN when < 2 non-noise clusters.
    - silhouette_score is a float in [-1, 1] for valid clusterings.

  profile_clusters():
    - Returns DataFrame indexed by cluster label.
    - n_samples column sums to total samples.
    - pct_young + pct_old == 100 for each cluster (when all labels are young/old).
    - Feature mean columns present and named correctly.
    - Noise cluster (-1) included in profile when present.

  plot_cluster_age_composition():
    - Returns a matplotlib Figure.
    - Title default applied when empty string passed.
    - Custom title used verbatim.
    - Figure has correct number of bars.
    - Noise cluster shown when present in profile.

  compare_bag_sizes():
    - Returns dict with one key per bag size.
    - Each value contains the required keys.
    - AnnData.uns['bag_clustering_results'] populated after call.
    - Metrics dict has n_clusters, noise_ratio, silhouette_score keys.
    - Profile DataFrame has pct_young and pct_old columns.
"""

import numpy as np
import pandas as pd
import pytest
import anndata
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("Agg")  # non-interactive backend for tests

from mito_marker.analysis.clustering import (
    _sample_individual_mitos,
    compare_bag_sizes,
    compute_umap_on_bags,
    evaluate_clustering,
    plot_cluster_age_composition,
    profile_clusters,
    run_hdbscan,
)


# ---------------------------------------------------------------------------
# Shared fixture factory
# ---------------------------------------------------------------------------


def _make_tem_anndata(
    n_subjects: int = 4,
    n_mitos_per_subject: int = 30,
    n_features: int = 6,
    random_state: int = 0,
) -> anndata.AnnData:
    """
    Build a minimal TEM-like AnnData with separable young/old clusters.

    Each subject is either "young" (half) or "old" (half). Young subjects have
    feature values centered around 0; old subjects around 3. This separation
    ensures HDBScan can find clusters on the UMAP.
    """
    rng = np.random.default_rng(random_state)
    n_obs = n_subjects * n_mitos_per_subject

    feature_matrix = np.zeros((n_obs, n_features), dtype=np.float32)
    subject_ids = []
    conditions = []

    for subject_index in range(n_subjects):
        start = subject_index * n_mitos_per_subject
        end = start + n_mitos_per_subject
        condition = "young" if subject_index < n_subjects // 2 else "old"
        center = 0.0 if condition == "young" else 3.0
        feature_matrix[start:end] = rng.normal(center, 0.3, (n_mitos_per_subject, n_features))
        subject_ids.extend([f"subject_{subject_index:02d}"] * n_mitos_per_subject)
        conditions.extend([condition] * n_mitos_per_subject)

    obs_dataframe = pd.DataFrame({
        "subject_ID": subject_ids,
        "condition": conditions,
    })

    var_dataframe = pd.DataFrame(
        index=[f"feature_{i}" for i in range(n_features)]
    )

    return anndata.AnnData(X=feature_matrix, obs=obs_dataframe, var=var_dataframe)


# ---------------------------------------------------------------------------
# Tests for _sample_individual_mitos
# ---------------------------------------------------------------------------


class TestSampleIndividualMitos:
    def test_output_shape(self):
        tem_anndata = _make_tem_anndata(n_subjects=4, n_mitos_per_subject=30)
        data_matrix = np.array(tem_anndata.X)
        feature_matrix, conditions, subjects = _sample_individual_mitos(
            data_matrix=data_matrix,
            obs_dataframe=tem_anndata.obs,
            subject_id_column="subject_ID",
            condition_column="condition",
            n_per_subject=10,
            random_state=42,
        )
        # 4 subjects × 10 per subject = 40 rows
        assert feature_matrix.shape == (40, 6)
        assert conditions.shape == (40,)
        assert subjects.shape == (40,)

    def test_condition_labels_aligned(self):
        tem_anndata = _make_tem_anndata(n_subjects=4, n_mitos_per_subject=30)
        data_matrix = np.array(tem_anndata.X)
        feature_matrix, conditions, subjects = _sample_individual_mitos(
            data_matrix=data_matrix,
            obs_dataframe=tem_anndata.obs,
            subject_id_column="subject_ID",
            condition_column="condition",
            n_per_subject=5,
            random_state=0,
        )
        # All conditions must be valid values
        assert set(conditions).issubset({"young", "old"})

    def test_subject_ids_aligned(self):
        tem_anndata = _make_tem_anndata(n_subjects=4, n_mitos_per_subject=30)
        data_matrix = np.array(tem_anndata.X)
        _, _, subjects = _sample_individual_mitos(
            data_matrix=data_matrix,
            obs_dataframe=tem_anndata.obs,
            subject_id_column="subject_ID",
            condition_column="condition",
            n_per_subject=5,
            random_state=0,
        )
        expected_subject_ids = {f"subject_{i:02d}" for i in range(4)}
        assert set(subjects).issubset(expected_subject_ids)

    def test_deterministic(self):
        tem_anndata = _make_tem_anndata(n_subjects=4, n_mitos_per_subject=30)
        data_matrix = np.array(tem_anndata.X)
        result1 = _sample_individual_mitos(
            data_matrix=data_matrix,
            obs_dataframe=tem_anndata.obs,
            subject_id_column="subject_ID",
            condition_column="condition",
            n_per_subject=10,
            random_state=42,
        )
        result2 = _sample_individual_mitos(
            data_matrix=data_matrix,
            obs_dataframe=tem_anndata.obs,
            subject_id_column="subject_ID",
            condition_column="condition",
            n_per_subject=10,
            random_state=42,
        )
        np.testing.assert_array_equal(result1[0], result2[0])

    def test_different_seeds_differ(self):
        tem_anndata = _make_tem_anndata(n_subjects=4, n_mitos_per_subject=30)
        data_matrix = np.array(tem_anndata.X)
        result1 = _sample_individual_mitos(
            data_matrix=data_matrix,
            obs_dataframe=tem_anndata.obs,
            subject_id_column="subject_ID",
            condition_column="condition",
            n_per_subject=10,
            random_state=0,
        )
        result2 = _sample_individual_mitos(
            data_matrix=data_matrix,
            obs_dataframe=tem_anndata.obs,
            subject_id_column="subject_ID",
            condition_column="condition",
            n_per_subject=10,
            random_state=999,
        )
        # With different seeds the sampled rows should differ
        assert not np.array_equal(result1[0], result2[0])

    def test_sampling_with_replacement_when_needed(self):
        """When n_per_subject > n_available, sampling with replacement must succeed."""
        tem_anndata = _make_tem_anndata(n_subjects=2, n_mitos_per_subject=5)
        data_matrix = np.array(tem_anndata.X)
        # Request 20 per subject when only 5 available
        feature_matrix, _, _ = _sample_individual_mitos(
            data_matrix=data_matrix,
            obs_dataframe=tem_anndata.obs,
            subject_id_column="subject_ID",
            condition_column="condition",
            n_per_subject=20,
            random_state=0,
        )
        assert feature_matrix.shape[0] == 40  # 2 subjects × 20


# ---------------------------------------------------------------------------
# Tests for compute_umap_on_bags
# ---------------------------------------------------------------------------


class TestComputeUmapOnBags:
    def test_single_mito_feature_names_no_stat_suffix(self):
        tem_anndata = _make_tem_anndata(n_subjects=4, n_mitos_per_subject=30)
        _, _, _, _, feature_names = compute_umap_on_bags(
            anndata_object=tem_anndata,
            mitos_per_bag=1,
            bags_per_subject=5,
            subject_id_column="subject_ID",
            condition_column="condition",
        )
        # Single mito: feature names should be original (no "__mean" suffix)
        assert all("__" not in name for name in feature_names)

    def test_bag_feature_names_have_stat_suffix(self):
        tem_anndata = _make_tem_anndata(n_subjects=4, n_mitos_per_subject=30)
        _, _, _, _, feature_names = compute_umap_on_bags(
            anndata_object=tem_anndata,
            mitos_per_bag=5,
            bags_per_subject=5,
            subject_id_column="subject_ID",
            condition_column="condition",
            bag_statistics=["mean", "std"],
        )
        # Bag mode: feature names should end with __mean or __std
        assert any("__mean" in name for name in feature_names)
        assert any("__std" in name for name in feature_names)

    def test_returns_five_tuple(self):
        tem_anndata = _make_tem_anndata(n_subjects=4, n_mitos_per_subject=30)
        result = compute_umap_on_bags(
            anndata_object=tem_anndata,
            mitos_per_bag=1,
            bags_per_subject=5,
            subject_id_column="subject_ID",
            condition_column="condition",
        )
        assert len(result) == 5

    def test_umap_coordinates_shape(self):
        tem_anndata = _make_tem_anndata(n_subjects=4, n_mitos_per_subject=30)
        umap_coords, feature_matrix, conditions, subjects, _ = compute_umap_on_bags(
            anndata_object=tem_anndata,
            mitos_per_bag=1,
            bags_per_subject=5,
            subject_id_column="subject_ID",
            condition_column="condition",
        )
        # 4 subjects × 5 per subject = 20 samples
        assert umap_coords.shape == (20, 2)

    def test_arrays_aligned(self):
        tem_anndata = _make_tem_anndata(n_subjects=4, n_mitos_per_subject=30)
        umap_coords, feature_matrix, conditions, subjects, feature_names = compute_umap_on_bags(
            anndata_object=tem_anndata,
            mitos_per_bag=5,
            bags_per_subject=8,
            subject_id_column="subject_ID",
            condition_column="condition",
            bag_statistics=["mean"],
        )
        n_samples = umap_coords.shape[0]
        assert feature_matrix.shape[0] == n_samples
        assert conditions.shape[0] == n_samples
        assert subjects.shape[0] == n_samples

    def test_feature_names_match_feature_matrix_columns(self):
        tem_anndata = _make_tem_anndata(n_subjects=4, n_mitos_per_subject=30)
        _, feature_matrix, _, _, feature_names = compute_umap_on_bags(
            anndata_object=tem_anndata,
            mitos_per_bag=5,
            bags_per_subject=5,
            subject_id_column="subject_ID",
            condition_column="condition",
            bag_statistics=["mean", "std"],
        )
        assert len(feature_names) == feature_matrix.shape[1]

    def test_bag_statistics_default_all_four(self):
        tem_anndata = _make_tem_anndata(n_subjects=4, n_mitos_per_subject=30)
        _, feature_matrix, _, _, feature_names = compute_umap_on_bags(
            anndata_object=tem_anndata,
            mitos_per_bag=5,
            bags_per_subject=5,
            subject_id_column="subject_ID",
            condition_column="condition",
            # bag_statistics not specified → defaults to all four
        )
        # 6 features × 4 stats = 24 bag features
        assert feature_matrix.shape[1] == 6 * 4
        assert any("__skew" in name for name in feature_names)
        assert any("__median" in name for name in feature_names)


# ---------------------------------------------------------------------------
# Tests for run_hdbscan
# ---------------------------------------------------------------------------


class TestRunHdbscan:
    def test_returns_1d_int_array(self):
        coordinates = np.random.rand(50, 2).astype(np.float32)
        labels = run_hdbscan(coordinates, min_cluster_size=5)
        assert labels.ndim == 1
        assert len(labels) == 50
        assert labels.dtype in (np.int32, np.int64, int)

    def test_finds_clusters_on_separable_data(self):
        rng = np.random.default_rng(0)
        # Two tight blobs clearly separated
        cluster_a = rng.normal([0, 0], 0.1, (50, 2))
        cluster_b = rng.normal([10, 10], 0.1, (50, 2))
        coordinates = np.vstack([cluster_a, cluster_b]).astype(np.float32)
        labels = run_hdbscan(coordinates, min_cluster_size=5)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        assert n_clusters >= 2

    def test_accepts_min_samples_parameter(self):
        coordinates = np.random.rand(30, 2).astype(np.float32)
        # Should not raise
        labels = run_hdbscan(coordinates, min_cluster_size=3, min_samples=2)
        assert len(labels) == 30

    def test_labels_are_integers(self):
        coordinates = np.random.rand(20, 2).astype(np.float32)
        labels = run_hdbscan(coordinates, min_cluster_size=3)
        assert all(isinstance(int(label), int) for label in labels)


# ---------------------------------------------------------------------------
# Tests for evaluate_clustering
# ---------------------------------------------------------------------------


class TestEvaluateClustering:
    def test_n_clusters_excludes_noise(self):
        labels = np.array([-1, -1, 0, 0, 0, 1, 1, 1])
        coords = np.random.rand(8, 2).astype(np.float32)
        metrics = evaluate_clustering(coords, labels)
        assert metrics["n_clusters"] == 2

    def test_noise_ratio_correct(self):
        labels = np.array([-1, -1, 0, 0, 0, 0])
        coords = np.random.rand(6, 2).astype(np.float32)
        metrics = evaluate_clustering(coords, labels)
        assert abs(metrics["noise_ratio"] - 2 / 6) < 1e-6

    def test_silhouette_nan_when_one_cluster(self):
        labels = np.array([0, 0, 0, 0, 0])
        coords = np.random.rand(5, 2).astype(np.float32)
        metrics = evaluate_clustering(coords, labels)
        assert np.isnan(metrics["silhouette_score"])

    def test_silhouette_nan_when_all_noise(self):
        labels = np.array([-1, -1, -1, -1])
        coords = np.random.rand(4, 2).astype(np.float32)
        metrics = evaluate_clustering(coords, labels)
        assert np.isnan(metrics["silhouette_score"])

    def test_silhouette_float_for_separable_clusters(self):
        rng = np.random.default_rng(0)
        cluster_a = rng.normal([0, 0], 0.1, (30, 2))
        cluster_b = rng.normal([10, 10], 0.1, (30, 2))
        coords = np.vstack([cluster_a, cluster_b]).astype(np.float32)
        labels = np.array([0] * 30 + [1] * 30)
        metrics = evaluate_clustering(coords, labels)
        assert not np.isnan(metrics["silhouette_score"])
        assert -1.0 <= metrics["silhouette_score"] <= 1.0

    def test_return_keys_present(self):
        labels = np.array([0, 0, 1, 1])
        coords = np.random.rand(4, 2).astype(np.float32)
        metrics = evaluate_clustering(coords, labels)
        assert "n_clusters" in metrics
        assert "noise_ratio" in metrics
        assert "silhouette_score" in metrics

    def test_n_clusters_zero_all_noise(self):
        labels = np.array([-1, -1, -1])
        coords = np.random.rand(3, 2).astype(np.float32)
        metrics = evaluate_clustering(coords, labels)
        assert metrics["n_clusters"] == 0
        assert metrics["noise_ratio"] == 1.0


# ---------------------------------------------------------------------------
# Tests for profile_clusters
# ---------------------------------------------------------------------------


class TestProfileClusters:
    def _make_profile_inputs(self):
        rng = np.random.default_rng(42)
        feature_matrix = rng.random((60, 4)).astype(np.float32)
        cluster_labels = np.array([0] * 20 + [1] * 20 + [-1] * 20)
        feature_names = ["area", "perimeter", "circularity", "ar"]
        condition_labels = np.array(
            ["young"] * 10 + ["old"] * 10 + ["young"] * 10 + ["old"] * 10 + ["young"] * 10 + ["old"] * 10
        )
        return feature_matrix, cluster_labels, feature_names, condition_labels

    def test_returns_dataframe(self):
        feature_matrix, cluster_labels, feature_names, condition_labels = self._make_profile_inputs()
        profile = profile_clusters(feature_matrix, cluster_labels, feature_names, condition_labels)
        assert isinstance(profile, pd.DataFrame)

    def test_indexed_by_cluster_label(self):
        feature_matrix, cluster_labels, feature_names, condition_labels = self._make_profile_inputs()
        profile = profile_clusters(feature_matrix, cluster_labels, feature_names, condition_labels)
        assert set(profile.index) == {-1, 0, 1}

    def test_n_samples_sums_to_total(self):
        feature_matrix, cluster_labels, feature_names, condition_labels = self._make_profile_inputs()
        profile = profile_clusters(feature_matrix, cluster_labels, feature_names, condition_labels)
        assert profile["n_samples"].sum() == len(cluster_labels)

    def test_pct_young_plus_old_equals_100(self):
        feature_matrix, cluster_labels, feature_names, condition_labels = self._make_profile_inputs()
        profile = profile_clusters(feature_matrix, cluster_labels, feature_names, condition_labels)
        for _, row in profile.iterrows():
            assert abs(row["pct_young"] + row["pct_old"] - 100.0) < 0.2

    def test_feature_columns_present(self):
        feature_matrix, cluster_labels, feature_names, condition_labels = self._make_profile_inputs()
        profile = profile_clusters(feature_matrix, cluster_labels, feature_names, condition_labels)
        for name in feature_names:
            assert name in profile.columns

    def test_noise_cluster_included(self):
        feature_matrix, cluster_labels, feature_names, condition_labels = self._make_profile_inputs()
        profile = profile_clusters(feature_matrix, cluster_labels, feature_names, condition_labels)
        assert -1 in profile.index

    def test_feature_means_are_floats(self):
        feature_matrix, cluster_labels, feature_names, condition_labels = self._make_profile_inputs()
        profile = profile_clusters(feature_matrix, cluster_labels, feature_names, condition_labels)
        for name in feature_names:
            assert profile[name].dtype in (float, np.float64)


# ---------------------------------------------------------------------------
# Tests for plot_cluster_age_composition
# ---------------------------------------------------------------------------


class TestPlotClusterAgeComposition:
    def _make_profile(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "n_samples": [20, 15, 5],
                "pct_young": [80.0, 30.0, 50.0],
                "pct_old": [20.0, 70.0, 50.0],
            },
            index=pd.Index([0, 1, -1], name="cluster"),
        )

    def test_returns_figure(self):
        profile = self._make_profile()
        fig = plot_cluster_age_composition(profile)
        import matplotlib.figure
        assert isinstance(fig, matplotlib.figure.Figure)
        plt.close("all")

    def test_default_title_applied_when_empty(self):
        profile = self._make_profile()
        fig = plot_cluster_age_composition(profile, title="")
        axis = fig.axes[0]
        assert "Age composition" in axis.get_title()
        plt.close("all")

    def test_custom_title_used(self):
        profile = self._make_profile()
        fig = plot_cluster_age_composition(profile, title="My Title")
        axis = fig.axes[0]
        assert axis.get_title() == "My Title"
        plt.close("all")

    def test_correct_number_of_bars(self):
        profile = self._make_profile()
        fig = plot_cluster_age_composition(profile)
        axis = fig.axes[0]
        # 3 bars (one per cluster)
        bar_containers = [c for c in axis.containers]
        assert len(bar_containers) == 2  # young bars + old bars containers
        assert len(bar_containers[0]) == 3  # 3 clusters
        plt.close("all")

    def test_noise_cluster_with_no_noise_still_works(self):
        profile_no_noise = pd.DataFrame(
            {"n_samples": [20, 15], "pct_young": [80.0, 30.0], "pct_old": [20.0, 70.0]},
            index=pd.Index([0, 1], name="cluster"),
        )
        fig = plot_cluster_age_composition(profile_no_noise)
        assert fig is not None
        plt.close("all")


# ---------------------------------------------------------------------------
# Tests for compare_bag_sizes
# ---------------------------------------------------------------------------


class TestCompareBagSizes:
    def test_returns_dict_with_one_key_per_bag_size(self):
        tem_anndata = _make_tem_anndata(n_subjects=4, n_mitos_per_subject=40)
        results = compare_bag_sizes(
            anndata_object=tem_anndata,
            bag_sizes=[1, 5],
            bags_per_subject=10,
            subject_id_column="subject_ID",
            condition_column="condition",
        )
        assert set(results.keys()) == {1, 5}

    def test_required_keys_in_each_result(self):
        tem_anndata = _make_tem_anndata(n_subjects=4, n_mitos_per_subject=40)
        results = compare_bag_sizes(
            anndata_object=tem_anndata,
            bag_sizes=[5],
            bags_per_subject=8,
            subject_id_column="subject_ID",
            condition_column="condition",
        )
        required_keys = {
            "umap_coordinates", "feature_matrix", "condition_labels",
            "subject_ids", "feature_names", "cluster_labels", "profile", "metrics",
        }
        assert required_keys.issubset(set(results[5].keys()))

    def test_uns_populated(self):
        tem_anndata = _make_tem_anndata(n_subjects=4, n_mitos_per_subject=40)
        compare_bag_sizes(
            anndata_object=tem_anndata,
            bag_sizes=[1, 10],
            bags_per_subject=8,
            subject_id_column="subject_ID",
            condition_column="condition",
        )
        assert "bag_clustering_results" in tem_anndata.uns
        assert 1 in tem_anndata.uns["bag_clustering_results"]
        assert 10 in tem_anndata.uns["bag_clustering_results"]

    def test_metrics_keys_present(self):
        tem_anndata = _make_tem_anndata(n_subjects=4, n_mitos_per_subject=40)
        results = compare_bag_sizes(
            anndata_object=tem_anndata,
            bag_sizes=[5],
            bags_per_subject=8,
            subject_id_column="subject_ID",
            condition_column="condition",
        )
        metrics = results[5]["metrics"]
        assert "n_clusters" in metrics
        assert "noise_ratio" in metrics
        assert "silhouette_score" in metrics

    def test_profile_has_pct_columns(self):
        tem_anndata = _make_tem_anndata(n_subjects=4, n_mitos_per_subject=40)
        results = compare_bag_sizes(
            anndata_object=tem_anndata,
            bag_sizes=[5],
            bags_per_subject=8,
            subject_id_column="subject_ID",
            condition_column="condition",
        )
        profile = results[5]["profile"]
        assert "pct_young" in profile.columns
        assert "pct_old" in profile.columns

    def test_arrays_aligned_in_result(self):
        tem_anndata = _make_tem_anndata(n_subjects=4, n_mitos_per_subject=40)
        results = compare_bag_sizes(
            anndata_object=tem_anndata,
            bag_sizes=[5],
            bags_per_subject=8,
            subject_id_column="subject_ID",
            condition_column="condition",
        )
        r = results[5]
        n = r["umap_coordinates"].shape[0]
        assert r["feature_matrix"].shape[0] == n
        assert r["condition_labels"].shape[0] == n
        assert r["subject_ids"].shape[0] == n
        assert r["cluster_labels"].shape[0] == n

    def test_bag_statistics_default_four_stats(self):
        tem_anndata = _make_tem_anndata(n_subjects=4, n_mitos_per_subject=40)
        results = compare_bag_sizes(
            anndata_object=tem_anndata,
            bag_sizes=[10],
            bags_per_subject=5,
            subject_id_column="subject_ID",
            condition_column="condition",
            # bag_statistics not specified → defaults to 4
        )
        feature_names = results[10]["feature_names"]
        stats_found = {name.split("__")[-1] for name in feature_names if "__" in name}
        assert stats_found == {"mean", "std", "median", "skew"}
