"""
test_analysis_cluster_plots.py

Unit tests for mito_marker.analysis.cluster_plots.

Tests verify:
  - Every plot runs on the Agg backend and returns / displays without error.
  - plot_cluster_radar() restores the active layer and selection afterwards.
  - compute_cluster_proportions(): pct_of_subset sums to 100 per dataset,
    pct_of_total_population uses the subset history (and the explicit
    override), per-subject statistics are computed per subject.
  - compare_cluster_proportions(): n = subjects, BH adjustment, effect-size
    direction, Kruskal-Wallis for 3+ datasets, errors on bad input.
"""

import matplotlib

matplotlib.use("Agg")

import anndata  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from mito_marker.analysis.cluster_model import fit_cluster_model, predict_cluster_model  # noqa: E402
from mito_marker.analysis.cluster_plots import (  # noqa: E402
    compare_cluster_proportions,
    compute_cluster_proportions,
    plot_cluster_embedding,
    plot_cluster_model_selection,
    plot_cluster_proportions,
    plot_cluster_radar,
)
from mito_marker.analysis.feature_subset import subset_by_feature_values, subset_by_obs_values  # noqa: E402
from mito_marker.analysis.pca_plot import compute_pca  # noqa: E402
from mito_marker.analysis.preprocessing_config import get_default_preprocessing_config  # noqa: E402
from mito_marker.analysis.preprocessing_pipeline import run_preprocessing  # noqa: E402
from mito_marker.analysis.umap_plot import compute_umap  # noqa: E402

_BLOB_CENTERS = np.array([[0, 0, 0, 0], [6, 6, 0, 0], [0, 6, 6, 6]], dtype=float)
_FEATURE_NAMES = ["Mito_Area", "Mito_Perimeter", "Mito_Circularity", "Mito_AR"]


def _make_raw_anndata(n_per_subject: int = 150, seed: int = 0) -> anndata.AnnData:
    """Three subjects per condition; Young 60/30/10% and Old 20/30/50% across three blobs."""
    rng = np.random.default_rng(seed)
    blocks, obs_rows = [], []
    for condition, blob_weights in [("Young", [0.6, 0.3, 0.1]), ("Old", [0.2, 0.3, 0.5])]:
        for subject_index in range(3):
            blob_labels = rng.choice(3, size=n_per_subject, p=blob_weights)
            blocks.append(
                _BLOB_CENTERS[blob_labels] + rng.normal(scale=0.7, size=(n_per_subject, 4)) + 20.0
            )
            obs_rows += [{"condition": condition, "unique_subject_ID": f"{condition}_{subject_index}",
                          "specie": "Mouse"} for _ in blob_labels]
    obs = pd.DataFrame(obs_rows)
    obs.index = [f"mito_{index}" for index in range(len(obs))]
    return anndata.AnnData(
        X=np.vstack(blocks).astype(np.float32), obs=obs, var=pd.DataFrame(index=_FEATURE_NAMES)
    )


@pytest.fixture(scope="module")
def annotated_datasets():
    """Reference space on all data, Young/Old subsets, KMeans fit on Young."""
    reference = run_preprocessing(_make_raw_anndata(), get_default_preprocessing_config())
    reference = compute_pca(reference, n_components=4)
    young = subset_by_obs_values(reference, {"condition": ["Young"]})
    old = subset_by_obs_values(reference, {"condition": ["Old"]})
    young, model = fit_cluster_model(young, "plots_model", max_clusters=5, n_pca_components_used=4)
    old = predict_cluster_model(old, model)
    return reference, young, old, model


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


class TestPlots:
    """Figures render without error."""

    def test_model_selection(self, annotated_datasets):
        _, _, _, model = annotated_datasets
        figure = plot_cluster_model_selection(model)
        assert len(figure.axes) == 2

    def test_radar_restores_active_layer(self, annotated_datasets):
        _, young, _, model = annotated_datasets
        young.uns["analysis_config"]["active_selection"] = "something"
        before = dict(young.uns["analysis_config"])
        plot_cluster_radar(young, model, nest_aggregate_by="unique_subject_ID")
        assert young.uns["analysis_config"]["active_layer"] == before["active_layer"]
        assert young.uns["analysis_config"]["active_selection"] == "something"
        young.uns["analysis_config"]["active_selection"] = None

    def test_embedding_pca(self, annotated_datasets):
        _, young, old, model = annotated_datasets
        plot_cluster_embedding(young, model)
        plot_cluster_embedding(old, model, group_by="condition")
        assert plt.get_fignums()

    def test_embedding_umap(self):
        reference = run_preprocessing(_make_raw_anndata(), get_default_preprocessing_config())
        reference = compute_pca(reference, n_components=4)
        compute_umap(reference, n_pca_components=4, embed_all_events=True, keep_reducer=True)
        young = subset_by_obs_values(reference, {"condition": ["Young"]})
        young, model = fit_cluster_model(young, "umap_plots", clustering_space="umap", n_clusters=3)
        plot_cluster_embedding(young, model)
        assert plt.get_fignums()

    def test_plots_require_annotation(self, annotated_datasets):
        reference, _, _, model = annotated_datasets
        with pytest.raises(KeyError, match="predict_cluster_model"):
            plot_cluster_radar(reference, model)

    @pytest.mark.parametrize("relative_to", ["subset", "total_population"])
    def test_proportions_plot(self, annotated_datasets, relative_to):
        _, young, old, model = annotated_datasets
        proportions = compute_cluster_proportions({"Young": young, "Old": old}, model)
        figure = plot_cluster_proportions(proportions, model, relative_to=relative_to)
        assert len(figure.axes) == 2

    def test_proportions_plot_rejects_bad_relative_to(self, annotated_datasets):
        _, young, _, model = annotated_datasets
        proportions = compute_cluster_proportions({"Young": young}, model)
        with pytest.raises(ValueError):
            plot_cluster_proportions(proportions, model, relative_to="everything")


class TestComputeClusterProportions:
    """Three percentages, three denominators."""

    def test_subset_percentages_sum_to_100(self, annotated_datasets):
        _, young, old, model = annotated_datasets
        proportions = compute_cluster_proportions({"Young": young, "Old": old}, model)
        sums = proportions.groupby("dataset")["pct_of_subset"].sum()
        np.testing.assert_allclose(sums.to_numpy(), 100.0)
        assert len(proportions) == 2 * model.n_clusters

    def test_total_population_from_history(self, annotated_datasets):
        _, young, _, model = annotated_datasets
        young_small = subset_by_feature_values(
            young, "Mito_Area", "lowest_fraction", fraction=0.25, within_group="unique_subject_ID"
        )
        predict_cluster_model(young_small, model)
        proportions = compute_cluster_proportions({"Young small": young_small}, model)
        assert (proportions["n_total_population"] == young.n_obs).all()
        np.testing.assert_allclose(
            proportions["pct_of_total_population"].sum(), 100.0 * young_small.n_obs / young.n_obs
        )

    def test_obs_filter_after_feature_filter(self, annotated_datasets):
        reference, _, _, model = annotated_datasets
        small = subset_by_feature_values(reference, "Mito_Area", "below", upper_value=21.0)
        small_young = subset_by_obs_values(small, {"condition": ["Young"]})
        predict_cluster_model(small_young, model)
        proportions = compute_cluster_proportions({"small young": small_young}, model)
        n_young_total = int((reference.obs["condition"] == "Young").sum())
        assert (proportions["n_total_population"] == n_young_total).all()

    def test_override_total_population(self, annotated_datasets):
        _, young, _, model = annotated_datasets
        proportions = compute_cluster_proportions(
            {"Young": young}, model, total_population_sizes={"Young": 900}
        )
        np.testing.assert_allclose(proportions["pct_of_total_population"].sum(), 100.0 * young.n_obs / 900)

    def test_per_subject_statistics(self, annotated_datasets):
        _, young, _, model = annotated_datasets
        proportions = compute_cluster_proportions({"Young": young}, model)
        first_row = proportions.iloc[0]
        subject_values = young.obs.groupby("unique_subject_ID", observed=True)[model.obs_column].apply(
            lambda labels: 100.0 * (labels.astype(str) == first_row["cluster"]).mean()
        )
        assert first_row["n_subjects"] == 3
        np.testing.assert_allclose(first_row["per_subject_mean_pct"], subject_values.mean())
        np.testing.assert_allclose(
            sorted(first_row["per_subject_pct_values"]), sorted(subject_values.round(4))
        )


class TestCompareClusterProportions:
    """Per-subject statistics."""

    def test_two_groups_mann_whitney(self, annotated_datasets):
        _, young, old, model = annotated_datasets
        comparison = compare_cluster_proportions({"Young": young, "Old": old}, model)
        assert (comparison["test"] == "Mann-Whitney U").all()
        assert (comparison["Young_n_subjects"] == 3).all()
        # 3 vs 3 subjects: smallest reachable p = 2 / C(6, 3) = 0.1.
        np.testing.assert_allclose(comparison["minimum_reachable_p_value"], 0.1)
        assert (comparison["p_value_adjusted"] >= comparison["p_value"]).all()

    def test_effect_direction(self, annotated_datasets):
        _, young, old, model = annotated_datasets
        comparison = compare_cluster_proportions({"Young": young, "Old": old}, model).set_index("cluster")
        # Cluster_1 (blob 0) is 60% in Young, 20% in Old: Old (second) is lower -> r < 0.
        assert comparison.loc["Cluster_1", "effect_size"] < 0
        assert comparison.loc["Cluster_3", "effect_size"] > 0

    def test_three_groups_kruskal(self, annotated_datasets):
        reference, _, _, model = annotated_datasets
        datasets = {}
        for index, subject_group in enumerate([["Young_0", "Young_1", "Young_2"],
                                               ["Old_0", "Old_1"], ["Old_2", "Young_0"]]):
            subset = subset_by_obs_values(reference, {"unique_subject_ID": subject_group})
            datasets[f"group_{index}"] = predict_cluster_model(subset, model)
        comparison = compare_cluster_proportions(datasets, model)
        assert (comparison["test"] == "Kruskal-Wallis H").all()
        assert (comparison["effect_size_metric"] == "eta_squared").all()

    def test_too_few_subjects_not_run(self, annotated_datasets):
        reference, young, _, model = annotated_datasets
        single = predict_cluster_model(
            subset_by_obs_values(reference, {"unique_subject_ID": ["Old_0"]}), model
        )
        comparison = compare_cluster_proportions({"Young": young, "one old": single}, model)
        assert comparison["test"].str.startswith("not run").all()
        assert comparison["p_value"].isna().all()

    def test_needs_two_datasets(self, annotated_datasets):
        _, young, _, model = annotated_datasets
        with pytest.raises(ValueError):
            compare_cluster_proportions({"Young": young}, model)

    def test_needs_subject_column(self, annotated_datasets):
        _, young, old, model = annotated_datasets
        young_no_subject = young.copy()
        young_no_subject.obs = young_no_subject.obs.drop(columns=["unique_subject_ID"])
        with pytest.raises(ValueError, match="subject"):
            compare_cluster_proportions({"Young": young_no_subject, "Old": old}, model)
