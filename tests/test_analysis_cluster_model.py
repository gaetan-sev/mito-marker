"""
test_analysis_cluster_model.py

Unit tests for mito_marker.analysis.cluster_model.

Synthetic data: three well-separated blobs in 4 features. Young subjects
draw 60/30/10% of their mitochondria from the three blobs, Old subjects
20/30/50%, so every expected cluster and proportion is known.

Tests verify:
  - KMeans and GMM recover k = 3 when scanning k = 2..6; n_clusters imposes k.
  - Clusters are named by decreasing training size, with fixed colors.
  - predict on a subset of the reference reuses the stored space (no
    reprojection) and matches the fit labels on the training set.
  - A dataset loaded separately is projected from raw .X with the FROZEN
    scaling: a +2 SD shift stays at +2 SD (no refit), and labels equal those
    of the same mitochondria reached through the reference subset.
  - save / load round trip gives identical predictions.
  - "scaled", "pca" and "umap" spaces all work; missing or non-reusable
    spaces raise explicit errors.
  - weight_by changes a KMeans fit; GMM with weight_by raises.
  - Normalization recomputed after subsetting triggers a warning.
  - The AnnData stays writable to .h5ad after fit and predict.
"""

import os

import anndata
import numpy as np
import pandas as pd
import pytest

from mito_marker.analysis.cluster_model import (
    CLUSTER_MODELS_UNS_KEY,
    MitoClusterModel,
    describe_cluster_model,
    fit_cluster_model,
    load_cluster_model,
    predict_cluster_model,
    project_to_reference_space,
    save_cluster_model,
)
from mito_marker.analysis.feature_subset import subset_by_obs_values
from mito_marker.analysis.normalization import get_layer_parameters
from mito_marker.analysis.pca_plot import compute_pca
from mito_marker.analysis.preprocessing_config import get_default_preprocessing_config
from mito_marker.analysis.preprocessing_pipeline import run_preprocessing
from mito_marker.analysis.umap_plot import compute_umap

_BLOB_CENTERS = np.array([[0, 0, 0, 0], [6, 6, 0, 0], [0, 6, 6, 6]], dtype=float)
_FEATURE_NAMES = ["Mito_Area", "Mito_Perimeter", "Mito_Circularity", "Mito_AR"]


def _make_raw_anndata(n_per_subject: int = 150, seed: int = 0) -> anndata.AnnData:
    """Three subjects per condition, blob proportions differing by condition."""
    rng = np.random.default_rng(seed)
    blocks, obs_rows = [], []
    for condition, blob_weights in [("Young", [0.6, 0.3, 0.1]), ("Old", [0.2, 0.3, 0.5])]:
        for subject_index in range(3):
            blob_labels = rng.choice(3, size=n_per_subject, p=blob_weights)
            blocks.append(
                _BLOB_CENTERS[blob_labels] + rng.normal(scale=0.7, size=(n_per_subject, 4)) + 20.0
            )
            obs_rows += [{
                "condition": condition,
                "unique_subject_ID": f"{condition}_{subject_index}",
                "specie": "Mouse",
                "true_blob": int(label),
            } for label in blob_labels]
    obs = pd.DataFrame(obs_rows)
    obs.index = [f"mito_{index}" for index in range(len(obs))]
    return anndata.AnnData(
        X=np.vstack(blocks).astype(np.float32), obs=obs, var=pd.DataFrame(index=_FEATURE_NAMES)
    )


def _make_reference_anndata() -> anndata.AnnData:
    """Raw data + reference space (z-score layer + PCA) built on everything."""
    reference = run_preprocessing(_make_raw_anndata(), get_default_preprocessing_config())
    return compute_pca(reference, n_components=4)


@pytest.fixture(scope="module")
def reference_anndata() -> anndata.AnnData:
    return _make_reference_anndata()


@pytest.fixture
def young_and_old(reference_anndata):
    young = subset_by_obs_values(reference_anndata, {"condition": ["Young"]})
    old = subset_by_obs_values(reference_anndata, {"condition": ["Old"]})
    return young, old


@pytest.fixture
def fitted_kmeans(young_and_old):
    young, old = young_and_old
    young, model = fit_cluster_model(
        young, "young_kmeans", algorithm="kmeans", clustering_space="pca",
        max_clusters=6, n_pca_components_used=4,
    )
    return young, old, model


def _agreement_with_blobs(cluster_labels: pd.Series, true_blobs: pd.Series) -> float:
    """Share of rows whose cluster is the majority cluster of their true blob."""
    contingency = pd.crosstab(true_blobs, cluster_labels.astype(str))
    return float(contingency.max(axis=1).sum() / contingency.to_numpy().sum())


class TestFit:
    """fit_cluster_model()."""

    def test_kmeans_recovers_three_clusters(self, fitted_kmeans):
        young, _, model = fitted_kmeans
        assert isinstance(model, MitoClusterModel)
        assert model.n_clusters == 3
        assert _agreement_with_blobs(young.obs[model.obs_column], young.obs["true_blob"]) > 0.98

    def test_gmm_recovers_three_clusters_with_probabilities(self, young_and_old):
        young, _ = young_and_old
        young, model = fit_cluster_model(
            young, "young_gmm", algorithm="gmm", clustering_space="pca",
            max_clusters=6, n_pca_components_used=4,
        )
        assert model.n_clusters == 3
        probabilities = young.obs[model.probability_obs_column]
        assert probabilities.between(0, 1).all() and probabilities.mean() > 0.95

    def test_clusters_named_by_decreasing_size(self, fitted_kmeans):
        young, _, model = fitted_kmeans
        counts = young.obs[model.obs_column].value_counts()
        assert counts["Cluster_1"] >= counts["Cluster_2"] >= counts["Cluster_3"]
        # Young draws 60% from blob 0, so Cluster_1 is that blob.
        assert abs(model.training_proportions["Cluster_1"] - 0.6) < 0.05

    def test_n_clusters_imposes_k(self, young_and_old):
        young, _ = young_and_old
        _, model = fit_cluster_model(young, "forced", n_clusters=5, n_pca_components_used=4)
        assert model.n_clusters == 5
        assert len(model.model_selection_table) == 1

    def test_scaled_space(self, young_and_old):
        young, _ = young_and_old
        young, model = fit_cluster_model(young, "scaled_model", clustering_space="scaled", max_clusters=5)
        assert model.n_clusters == 3
        assert model.feature_names == _FEATURE_NAMES
        assert model.pca_parameters is None

    def test_profiles_in_raw_units(self, fitted_kmeans):
        _, _, model = fitted_kmeans
        # Raw data are centered around 20: every raw profile lies near 20-26.
        profile_values = model.cluster_profiles_raw[_FEATURE_NAMES].to_numpy()
        assert profile_values.min() > 18 and profile_values.max() < 28
        assert model.cluster_profiles_raw["n_mitochondria"].sum() == 450

    def test_model_selection_table(self, fitted_kmeans):
        _, _, model = fitted_kmeans
        table = model.model_selection_table
        assert table["n_clusters"].tolist() == [2, 3, 4, 5, 6]
        assert table.loc[table["chosen"], "n_clusters"].item() == 3

    def test_weight_by_changes_kmeans(self, reference_anndata):
        # Make one subject dominate, then compare weighted and unweighted fits.
        young = subset_by_obs_values(reference_anndata, {"condition": ["Young"]})
        extra = young[young.obs["unique_subject_ID"] == "Young_0"].copy()
        extra.obs_names = [f"{name}_copy" for name in extra.obs_names]
        imbalanced = anndata.concat([young, extra, extra], uns_merge="first")
        _, unweighted = fit_cluster_model(imbalanced.copy(), "unweighted", n_clusters=2, n_pca_components_used=4)
        _, weighted = fit_cluster_model(
            imbalanced.copy(), "weighted", n_clusters=2, n_pca_components_used=4,
            weight_by=["unique_subject_ID"],
        )
        assert not np.allclose(
            np.sort(unweighted.clusterer.cluster_centers_, axis=0),
            np.sort(weighted.clusterer.cluster_centers_, axis=0),
        )

    def test_uns_summary_written(self, fitted_kmeans):
        young, _, model = fitted_kmeans
        summary = young.uns[CLUSTER_MODELS_UNS_KEY][model.model_name]
        assert summary["role"] == "fit"
        assert summary["reference_space_fingerprint"] == model.reference_space_fingerprint
        assert young.uns["color_palette"]["Cluster_1"] == model.cluster_colors["Cluster_1"]


class TestFitErrors:
    """Invalid arguments and unusable spaces."""

    @pytest.mark.parametrize("kwargs", [
        {"model_name": "bad name"},
        {"model_name": "ok", "algorithm": "dbscan"},
        {"model_name": "ok", "clustering_space": "tsne"},
        {"model_name": "ok", "max_clusters": 1},
        {"model_name": "ok", "n_clusters": 1},
        {"model_name": "ok", "algorithm": "gmm", "weight_by": ["unique_subject_ID"]},
    ])
    def test_invalid_arguments(self, young_and_old, kwargs):
        young, _ = young_and_old
        with pytest.raises(ValueError):
            fit_cluster_model(young, **kwargs)

    def test_pca_missing(self):
        raw = run_preprocessing(_make_raw_anndata(), get_default_preprocessing_config())
        with pytest.raises(ValueError, match="compute_pca"):
            fit_cluster_model(raw, "no_pca", clustering_space="pca")

    def test_scaled_without_layer(self):
        raw = _make_raw_anndata()
        with pytest.raises(ValueError, match="normalized layer"):
            fit_cluster_model(raw, "no_layer", clustering_space="scaled")

    def test_layer_without_frozen_parameters(self):
        raw = _make_raw_anndata()
        raw.layers["none__zscore_col"] = raw.X.copy()
        raw.uns["analysis_config"] = {"active_layer": "none__zscore_col", "active_selection": None}
        with pytest.raises(ValueError, match="frozen parameters"):
            fit_cluster_model(raw, "old_layer", clustering_space="scaled")

    def test_umap_without_reducer(self):
        reference = _make_reference_anndata()
        compute_umap(reference, n_pca_components=4)
        with pytest.raises(ValueError, match="keep_reducer"):
            fit_cluster_model(reference, "no_reducer", clustering_space="umap")

    def test_space_recomputed_on_subset_warns(self, capsys):
        raw = _make_raw_anndata()
        young = subset_by_obs_values(raw, {"condition": ["Young"]})
        young = run_preprocessing(young, get_default_preprocessing_config())
        capsys.readouterr()
        fit_cluster_model(young, "subset_space", clustering_space="scaled", n_clusters=3)
        assert "probably recomputed AFTER subsetting" in capsys.readouterr().out


class TestPredict:
    """predict_cluster_model() and project_to_reference_space()."""

    def test_predict_on_training_set_reproduces_fit(self, fitted_kmeans):
        young, _, model = fitted_kmeans
        fit_labels = young.obs[model.obs_column].astype(str).copy()
        predict_cluster_model(young, model)
        assert (young.obs[model.obs_column].astype(str) == fit_labels).all()

    def test_subset_uses_stored_space(self, fitted_kmeans, capsys):
        _, old, model = fitted_kmeans
        predict_cluster_model(old, model)
        output = capsys.readouterr().out
        assert "already in the model's reference space" in output
        assert "Exactness check" in output and "✔" in output
        # Old draws 50% from blob 2, which is Cluster_3 in the Young model.
        proportions = old.obs[model.obs_column].value_counts(normalize=True)
        assert abs(proportions["Cluster_3"] - 0.5) < 0.07

    def test_external_dataset_projected_with_frozen_scaling(self, fitted_kmeans):
        _, old, model = fitted_kmeans
        predict_cluster_model(old, model)
        external = anndata.AnnData(
            X=old.X.copy(), obs=old.obs[["condition", "unique_subject_ID"]].copy(),
            var=pd.DataFrame(index=old.var_names),
        )
        predict_cluster_model(external, model)
        assert (external.obs[model.obs_column].astype(str).values
                == old.obs[model.obs_column].astype(str).values).all()
        np.testing.assert_allclose(external.obsm["X_pca"], old.obsm["X_pca"], atol=1e-4)
        assert get_layer_parameters(external)["fingerprint"] == model.layer_parameters["fingerprint"]

    def test_no_refit_on_shifted_dataset(self, fitted_kmeans):
        _, old, model = fitted_kmeans
        scale = np.asarray(model.layer_parameters["scaler_scale"])
        shifted = anndata.AnnData(
            X=(old.X + 2.0 * scale).astype(np.float32),
            obs=old.obs[["condition", "unique_subject_ID"]].copy(),
            var=pd.DataFrame(index=old.var_names),
        )
        project_to_reference_space(shifted, model)
        shift = shifted.layers["none__zscore_col"].mean(axis=0) - old.layers["none__zscore_col"].mean(axis=0)
        np.testing.assert_allclose(shift, 2.0, atol=1e-3)

    def test_missing_feature_raises(self, fitted_kmeans):
        _, old, model = fitted_kmeans
        incomplete = anndata.AnnData(
            X=old.X[:, :3].copy(), obs=old.obs.copy(), var=pd.DataFrame(index=old.var_names[:3])
        )
        with pytest.raises(KeyError, match="Mito_AR"):
            predict_cluster_model(incomplete, model)

    def test_atypical_share_reported(self, fitted_kmeans, capsys):
        _, old, model = fitted_kmeans
        far_away = anndata.AnnData(
            X=(old.X + 50.0).astype(np.float32), obs=old.obs[["unique_subject_ID"]].copy(),
            var=pd.DataFrame(index=old.var_names),
        )
        predict_cluster_model(far_away, model)
        output = capsys.readouterr().out
        assert "Atypical mitochondria: 100.0%" in output
        assert "describes this dataset poorly" in output

    def test_umap_space_fit_and_predict(self):
        reference = _make_reference_anndata()
        compute_umap(reference, n_pca_components=4, embed_all_events=True, keep_reducer=True,
                     max_total_events=500)
        young = subset_by_obs_values(reference, {"condition": ["Young"]})
        old = subset_by_obs_values(reference, {"condition": ["Old"]})
        young, model = fit_cluster_model(young, "umap_model", clustering_space="umap", max_clusters=5)
        assert model.umap_parameters is not None and model.n_space_dimensions == 2
        predict_cluster_model(old, model)
        assert old.obs[model.obs_column].notna().all()
        external = anndata.AnnData(X=old.X.copy(), obs=old.obs[["unique_subject_ID"]].copy(),
                                   var=pd.DataFrame(index=old.var_names))
        predict_cluster_model(external, model)
        agreement = (external.obs[model.obs_column].astype(str).values
                     == old.obs[model.obs_column].astype(str).values).mean()
        # reducer.transform() is not bit-identical to the fit embedding, but
        # well-separated blobs must land in the same clusters.
        assert agreement > 0.95


class TestPersistence:
    """save / load / describe and .h5ad compatibility."""

    def test_round_trip_identical_predictions(self, fitted_kmeans, tmp_path):
        _, old, model = fitted_kmeans
        file_path = str(tmp_path / "models" / "young_kmeans.joblib")
        save_cluster_model(model, file_path)
        reloaded = load_cluster_model(file_path)
        first = predict_cluster_model(old.copy(), model).obs[model.obs_column]
        second = predict_cluster_model(old.copy(), reloaded).obs[model.obs_column]
        assert (first.astype(str) == second.astype(str)).all()
        assert reloaded.reference_space_fingerprint == model.reference_space_fingerprint

    def test_saved_model_does_not_keep_training_labels(self, fitted_kmeans, tmp_path):
        _, _, model = fitted_kmeans
        assert not hasattr(model.clusterer, "labels_")
        file_path = str(tmp_path / "small.joblib")
        save_cluster_model(model, file_path)
        assert os.path.getsize(file_path) < 100_000

    def test_save_refuses_overwrite_and_bad_extension(self, fitted_kmeans, tmp_path):
        _, _, model = fitted_kmeans
        file_path = str(tmp_path / "model.joblib")
        save_cluster_model(model, file_path)
        with pytest.raises(FileExistsError):
            save_cluster_model(model, file_path)
        save_cluster_model(model, file_path, overwrite=True)
        with pytest.raises(ValueError):
            save_cluster_model(model, str(tmp_path / "model.pkl"))

    def test_load_rejects_other_objects(self, tmp_path):
        import joblib

        joblib.dump({"not": "a model"}, tmp_path / "other.joblib")
        with pytest.raises(TypeError):
            load_cluster_model(str(tmp_path / "other.joblib"))

    def test_describe_runs(self, fitted_kmeans, capsys):
        _, _, model = fitted_kmeans
        describe_cluster_model(model)
        output = capsys.readouterr().out
        assert "young_kmeans" in output and "obs filter" in output

    def test_h5ad_after_fit_and_predict(self, fitted_kmeans, tmp_path):
        young, old, model = fitted_kmeans
        predict_cluster_model(old, model)
        for name, dataset in (("young", young), ("old", old)):
            dataset.write_h5ad(tmp_path / f"{name}.h5ad")
            reloaded = anndata.read_h5ad(tmp_path / f"{name}.h5ad")
            assert model.obs_column in reloaded.obs.columns
            # A reloaded dataset still carries the reference space.
            predict_cluster_model(reloaded, model)
            assert (reloaded.obs[model.obs_column].astype(str).values
                    == dataset.obs[model.obs_column].astype(str).values).all()
