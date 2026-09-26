"""
cluster_model.py

Reusable clustering models: fit once on a training dataset, save to a file,
re-apply (predict) to any other dataset WITHOUT refitting.

Principle — one frozen reference space (ADR-015)
------------------------------------------------
Cluster centroids learned on young mitochondria only mean something for old
mitochondria if both are placed in the SAME space: the same z-score (means and
standard deviations of the reference dataset), the same PCA axes, the same
UMAP map. That space is built ONCE, on the full reference dataset (e.g. every
Young AND Old mitochondrion), with the existing functions:

    run_preprocessing() / transform_and_normalize()   -> normalized layer
    compute_pca()                                     -> .obsm['X_pca']
    compute_umap(embed_all_events=True, keep_reducer=True)   (optional)

Those functions record their fitted parameters and a fingerprint. Every subset
taken afterwards (subset_by_obs_values, subset_by_feature_values) inherits the
layer, the coordinates and the parameters. This module never computes a space
itself: fit_cluster_model() reads the space you built, and copies its frozen
parameters into the model so that the saved model file is self-contained.

Life cycle of a MitoClusterModel
--------------------------------
  fit_cluster_model()          creates and fills the model (never built by hand)
  save_cluster_model()         writes it to a .joblib file
  load_cluster_model()         reads it back, in any notebook
  predict_cluster_model()      assigns every mitochondrion of a dataset to the
                               learned clusters (no refit)
  project_to_reference_space() places a dataset that does NOT come from the
                               reference dataset (another experiment) in the
                               frozen space; predict_cluster_model() calls it
                               automatically when needed
  describe_cluster_model()     prints a readable summary

Algorithms: KMeans (k chosen by silhouette) and Gaussian mixture (k chosen by
BIC) — both have a native predict().

Typical usage:
    tem_anndata = run_preprocessing(tem_anndata, preprocessing_config)
    tem_anndata = compute_pca(tem_anndata, n_components=10)
    young_small = subset_by_feature_values(...)
    young_small, model = fit_cluster_model(young_small, "young_small_kmeans",
                                           algorithm="kmeans", clustering_space="pca",
                                           max_clusters=8)
    save_cluster_model(model, "models/young_small_kmeans.joblib")
    old_small = predict_cluster_model(old_small, model)
"""

import gc
import os
import pickle
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import anndata
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture

from mito_marker.analysis._fingerprint import compute_fingerprint
from mito_marker.analysis.colors import COLOR_PALETTE_KEY
from mito_marker.analysis.feature_selection import _get_analytical_mask
from mito_marker.analysis.feature_subset import get_subset_history, resolve_subject_column
from mito_marker.analysis.normalization import (
    LAYER_PARAMETERS_KEY,
    _apply_stored_layer_parameters,
    get_layer_parameters,
)
from mito_marker.analysis.pca_plot import _compute_nested_equal_weights
from mito_marker.analysis.selection import _ANALYSIS_CONFIG_KEY
from mito_marker.analysis.umap_plot import UMAP_REDUCERS_KEY
from mito_marker.controlled_vocabulary import (
    ALLOWED_CLUSTERING_ALGORITHMS,
    ALLOWED_CLUSTERING_SPACES,
    CLUSTER_LABEL_PREFIX,
    CLUSTER_MODEL_NAME_PATTERN,
    CLUSTER_OBS_COLUMN_PREFIX,
    CLUSTER_PROBABILITY_OBS_COLUMN_PREFIX,
)

# .uns key holding a small, h5ad-compatible summary of every model fitted or
# applied on this AnnData (the model objects themselves live in .joblib files).
CLUSTER_MODELS_UNS_KEY = "cluster_models"

# Silhouette is O(n²): computed on at most this many rows (random sample).
_SILHOUETTE_SAMPLE_SIZE = 10_000
# A mean silhouette below this is conventionally read as "no substantial
# cluster structure" (Kaufman & Rousseeuw 1990) — a continuum is likely.
_WEAK_SILHOUETTE_THRESHOLD = 0.25
# Drift QC thresholds, in reference units (z-scores for a z-score layer).
_DRIFT_MEAN_SHIFT_THRESHOLD = 1.0
_DRIFT_OUT_OF_RANGE_THRESHOLD = 0.05
# Share of rows reprojected from raw .X to check a dataset's stored coordinates.
_EXACTNESS_CHECK_ROWS = 2_000

_UMAP_PARAMS_KEY = "umap_params"


@dataclass
class MitoClusterModel:
    """
    A fitted clustering model and the frozen reference space it lives in.

    Created and filled ONLY by fit_cluster_model(); read by
    predict_cluster_model(), project_to_reference_space(), the plotting
    functions of cluster_plots.py and describe_cluster_model(); written and
    read back by save_cluster_model() / load_cluster_model().

    Attributes:
        model_name: Name given at fit time; used in .obs column names.
        algorithm: "kmeans" or "gmm".
        clustering_space: "scaled", "pca" or "umap".
        n_space_dimensions: Number of dimensions the clusterer was fitted on.
        feature_names: Raw features (.var_names) the space is built from.
        clusterer: The fitted sklearn KMeans or GaussianMixture object.
        n_clusters: Number of clusters.
        cluster_labels: "Cluster_1", ... ordered by decreasing training size.
        raw_label_to_name: cluster_labels name for each raw clusterer label.
        cluster_colors: {cluster label: hex color}.
        layer_name: Name of the normalized layer ("" when the chain starts from raw .X).
        layer_parameters: Frozen normalization parameters (None = raw .X).
        pca_parameters: Frozen PCA parameters (None when not used).
        umap_parameters: Frozen UMAP parameters incl. the pickled reducer (None when not used).
        space_channel_names: Columns read from the layer for "scaled" space
                             (or for a UMAP fitted directly on channels).
        reference_space_fingerprint: Fingerprint of the whole chain.
        model_selection_table: One row per tested k (silhouette, BIC, inertia).
        cluster_profiles_raw: Mean raw value of each feature, per cluster.
        cluster_profiles_scaled: Mean normalized value of each feature, per cluster.
        training_proportions: Share of training mitochondria per cluster.
        training_feature_mean / training_feature_std / training_feature_min /
        training_feature_max: Per-feature statistics of the training set in
                              the normalized layer, for the drift QC.
        atypicality_threshold: KMeans: 99th percentile of the training
                               distance to the assigned centroid. GMM: 1st
                               percentile of the training log-likelihood.
        provenance: Where the model comes from (history, counts, date, version...).
    """

    model_name: str
    algorithm: str
    clustering_space: str
    n_space_dimensions: int
    feature_names: List[str]
    clusterer: Any
    n_clusters: int
    cluster_labels: List[str]
    raw_label_to_name: List[str]
    cluster_colors: Dict[str, str]
    layer_name: str
    layer_parameters: Optional[Dict[str, Any]]
    pca_parameters: Optional[Dict[str, Any]]
    umap_parameters: Optional[Dict[str, Any]]
    space_channel_names: List[str]
    reference_space_fingerprint: str
    model_selection_table: pd.DataFrame
    cluster_profiles_raw: pd.DataFrame
    cluster_profiles_scaled: pd.DataFrame
    training_proportions: pd.Series
    training_feature_mean: pd.Series
    training_feature_std: pd.Series
    training_feature_min: pd.Series
    training_feature_max: pd.Series
    atypicality_threshold: float
    provenance: Dict[str, Any] = field(default_factory=dict)

    @property
    def obs_column(self) -> str:
        """Name of the .obs column holding this model's cluster labels."""
        return f"{CLUSTER_OBS_COLUMN_PREFIX}{self.model_name}"

    @property
    def probability_obs_column(self) -> str:
        """Name of the .obs column holding the GMM membership probability."""
        return f"{CLUSTER_PROBABILITY_OBS_COLUMN_PREFIX}{self.model_name}"


# ---------------------------------------------------------------------------
# Fit
# ---------------------------------------------------------------------------


def fit_cluster_model(
    training_anndata: anndata.AnnData,
    model_name: str,
    algorithm: str = "kmeans",
    clustering_space: str = "pca",
    max_clusters: int = 8,
    n_clusters: Optional[int] = None,
    n_pca_components_used: Optional[int] = 10,
    weight_by: Optional[List[str]] = None,
    random_state: int = 42,
) -> Tuple[anndata.AnnData, MitoClusterModel]:
    """
    Fit a KMeans or Gaussian-mixture clustering on an existing space.

    No space is computed here. The function reads, from training_anndata:
      - "scaled": the active normalized layer (analytical, selected features);
      - "pca":    the first n_pca_components_used columns of .obsm['X_pca'];
      - "umap":   the active UMAP coordinates.
    Build that space ONCE on the full reference dataset BEFORE taking the
    training subset, so every other subset shares it (ADR-015).

    Choosing k: every k from 2 to max_clusters is fitted.
      - KMeans keeps the k with the highest mean silhouette (how much closer
        each point is to its own cluster than to the next one, from -1 to 1).
      - GMM keeps the k with the lowest BIC (log-likelihood penalized by the
        number of parameters). The silhouette is also reported, for information.
    Pass n_clusters to impose k instead.

    Cluster names: raw clusterer labels are renamed "Cluster_1", "Cluster_2",
    ... by decreasing size in the training set, so names are stable and
    readable (Cluster_1 = the most common type).

    Arguments:
        training_anndata: Dataset the clusters are learned on (modified in
                          place: cluster labels are added to .obs).
        model_name: Name of the model (letters, digits, "_" and "-" only).
        algorithm: "kmeans" or "gmm" (ALLOWED_CLUSTERING_ALGORITHMS).
        clustering_space: "scaled", "pca" or "umap" (ALLOWED_CLUSTERING_SPACES).
        max_clusters: Largest k tested (k starts at 2).
        n_clusters: Impose k; when set, max_clusters is ignored.
        n_pca_components_used: Number of PCs used in "pca" space (None = all).
        weight_by: .obs columns from coarsest to finest (e.g.
                   ["unique_subject_ID"]) so every subject weighs the same in
                   the KMeans fit (ADR-011). Not available for GMM.
        random_state: Seed for every random step.

    Returns:
        Tuple (training_anndata with .obs['cluster__<model_name>'] added,
        the fitted MitoClusterModel).

    Raises:
        ValueError: Invalid parameters, or the requested space is missing,
                    incomplete (NaN) or not reusable (no frozen parameters).
    """
    _validate_fit_arguments(model_name, algorithm, clustering_space, max_clusters, n_clusters, weight_by)

    print("=" * 60)
    print(f"FIT CLUSTER MODEL '{model_name}'")
    print("=" * 60)
    print(
        f"Parameters: algorithm='{algorithm}', clustering_space='{clustering_space}', "
        f"max_clusters={max_clusters}, n_clusters={n_clusters}, "
        f"n_pca_components_used={n_pca_components_used}, weight_by={weight_by}, "
        f"random_state={random_state}"
    )
    print(f"Algorithm: {ALLOWED_CLUSTERING_ALGORITHMS[algorithm]}")
    print(f"Space:     {ALLOWED_CLUSTERING_SPACES[clustering_space]}")
    print(f"Training dataset: {training_anndata.n_obs:,} mitochondria")

    space_definition = _read_space_definition(training_anndata, clustering_space, n_pca_components_used)
    space_matrix = _read_space_matrix(training_anndata, space_definition)
    _check_finite(space_matrix, "clustering space matrix")
    _warn_if_space_fitted_on_subset(training_anndata, space_definition)

    sample_weight = None
    if weight_by is not None:
        sample_weight = _compute_nested_equal_weights(training_anndata.obs, weight_by)
        # KMeans only needs relative weights; rescaling to mean 1 keeps the
        # reported inertia on the same scale as an unweighted fit.
        sample_weight = sample_weight * len(sample_weight)
        print(
            f"Weights: every value of {weight_by} contributes equally "
            f"(weight min={sample_weight.min():.3g}, max={sample_weight.max():.3g})."
        )

    candidate_k = [n_clusters] if n_clusters is not None else list(range(2, max_clusters + 1))
    candidate_k = [k for k in candidate_k if k < training_anndata.n_obs]
    if not candidate_k:
        raise ValueError(f"Not enough mitochondria ({training_anndata.n_obs}) to fit 2 clusters.")

    model_selection_table, fitted_by_k = _scan_number_of_clusters(
        space_matrix, algorithm, candidate_k, sample_weight, random_state
    )
    chosen_k = _choose_number_of_clusters(model_selection_table, algorithm)
    clusterer = fitted_by_k[chosen_k]
    # KMeans keeps one training label per mitochondrion (labels_), which
    # predict() never uses: dropping it keeps the saved model a few kB instead
    # of megabytes for large training sets.
    if hasattr(clusterer, "labels_"):
        del clusterer.labels_
    model_selection_table["chosen"] = model_selection_table["n_clusters"] == chosen_k
    print("Model selection (one row per k):")
    print(model_selection_table.to_string(index=False))
    print(f"=> k = {chosen_k} retained ({'max silhouette' if algorithm == 'kmeans' else 'min BIC'}"
          f"{', imposed by n_clusters' if n_clusters is not None else ''}).")

    raw_labels = clusterer.predict(space_matrix)
    raw_label_to_name = _name_clusters_by_size(raw_labels, chosen_k)
    cluster_labels = [f"{CLUSTER_LABEL_PREFIX}{rank + 1}" for rank in range(chosen_k)]
    cluster_names = np.asarray(raw_label_to_name, dtype=object)[raw_labels]
    cluster_colors = _build_cluster_colors(cluster_labels)

    atypicality_threshold = _compute_atypicality_threshold(clusterer, algorithm, space_matrix, raw_labels)
    scaled_features = _read_scaled_features(training_anndata, space_definition)

    model = MitoClusterModel(
        model_name=model_name,
        algorithm=algorithm,
        clustering_space=clustering_space,
        n_space_dimensions=int(space_matrix.shape[1]),
        feature_names=list(space_definition["feature_names"]),
        clusterer=clusterer,
        n_clusters=chosen_k,
        cluster_labels=cluster_labels,
        raw_label_to_name=list(raw_label_to_name),
        cluster_colors=cluster_colors,
        layer_name=space_definition["layer_name"],
        layer_parameters=space_definition["layer_parameters"],
        pca_parameters=space_definition["pca_parameters"],
        umap_parameters=space_definition["umap_parameters"],
        space_channel_names=list(space_definition["space_channel_names"]),
        reference_space_fingerprint=space_definition["reference_space_fingerprint"],
        model_selection_table=model_selection_table,
        cluster_profiles_raw=_cluster_profiles(
            _read_raw_features(training_anndata, space_definition["feature_names"]),
            cluster_names, cluster_labels, space_definition["feature_names"],
        ),
        cluster_profiles_scaled=_cluster_profiles(
            scaled_features, cluster_names, cluster_labels, space_definition["feature_names"]
        ) if scaled_features is not None else pd.DataFrame(),
        training_proportions=pd.Series(cluster_names).value_counts(normalize=True)
        .reindex(cluster_labels, fill_value=0.0),
        training_feature_mean=_feature_statistic(scaled_features, space_definition, np.nanmean),
        training_feature_std=_feature_statistic(scaled_features, space_definition, np.nanstd),
        training_feature_min=_feature_statistic(scaled_features, space_definition, np.nanmin),
        training_feature_max=_feature_statistic(scaled_features, space_definition, np.nanmax),
        atypicality_threshold=atypicality_threshold,
        provenance=_build_provenance(
            training_anndata, space_definition, weight_by, random_state, n_pca_components_used
        ),
    )

    _write_assignments(training_anndata, model, cluster_names, space_matrix, role="fit")

    _print_fit_qc(training_anndata, model, space_matrix, cluster_names)
    del space_matrix, scaled_features
    gc.collect()
    return training_anndata, model


# ---------------------------------------------------------------------------
# Projection and predict
# ---------------------------------------------------------------------------


def project_to_reference_space(
    anndata_object: anndata.AnnData,
    cluster_model: MitoClusterModel,
) -> anndata.AnnData:
    """
    Place a dataset in the model's frozen reference space, without refitting.

    Needed only for a dataset that does NOT come from the reference dataset
    (another experiment, loaded separately). Subsets of the reference dataset
    already carry the right coordinates. predict_cluster_model() calls this
    function automatically when the dataset's fingerprints do not match.

    Steps, each a transform() with the reference parameters only:
      1. read raw .X for the model's features (error if any is missing),
      2. replay the frozen transformation + normalization -> layer
         (same name as in the reference),
      3. project on the frozen PCA axes -> .obsm['X_pca'] (if the model uses PCA),
      4. place on the frozen UMAP map -> .obsm[<umap key>] (if the model uses UMAP).
    The .uns entries (layer parameters, PCA parameters, UMAP parameters and
    fingerprints) are written exactly as in the reference, so plot_radar(),
    plot_pca_scatter() and plot_umap() work on the projected dataset.

    Arguments:
        anndata_object: Dataset to project (modified in place).
        cluster_model: Model whose reference space is used.

    Returns:
        The same AnnData, now in the reference space.

    Raises:
        KeyError: A feature required by the model is missing from .var_names.
    """
    print("=" * 60)
    print(f"PROJECT TO REFERENCE SPACE of model '{cluster_model.model_name}'")
    print("=" * 60)
    print(f"Dataset: {anndata_object.n_obs:,} mitochondria × {anndata_object.n_vars} variables")

    _ensure_analysis_config(anndata_object)
    scaled_matrix, scaled_names = _replay_layer(anndata_object, cluster_model)

    if cluster_model.pca_parameters is not None:
        pca_parameters = cluster_model.pca_parameters
        pca_input = _columns(scaled_matrix, scaled_names, list(pca_parameters["channel_names"]))
        pca_coordinates = (
            (pca_input.astype(np.float64) - np.asarray(pca_parameters["mean"]))
            @ np.asarray(pca_parameters["loadings"], dtype=np.float64)
        ).astype(np.float32)
        _warn_if_overwriting(anndata_object, "pca_fingerprint", pca_parameters["fingerprint"], ".obsm['X_pca']")
        anndata_object.obsm["X_pca"] = pca_coordinates
        anndata_object.uns["pca_loadings"] = np.asarray(pca_parameters["loadings"], dtype=np.float32)
        anndata_object.uns["pca_explained_variance_ratio"] = np.asarray(
            pca_parameters["explained_variance_ratio"], dtype=np.float32
        )
        anndata_object.uns["pca_channel_names"] = list(pca_parameters["channel_names"])
        anndata_object.uns["pca_weight_by"] = pca_parameters["weight_by"]
        anndata_object.uns["pca_mean"] = np.asarray(pca_parameters["mean"], dtype=np.float64)
        anndata_object.uns["pca_input_layer"] = cluster_model.layer_name
        anndata_object.uns["pca_input_fingerprint"] = pca_parameters["input_fingerprint"]
        anndata_object.uns["pca_n_obs_fitted"] = pca_parameters["n_obs_fitted"]
        anndata_object.uns["pca_fingerprint"] = pca_parameters["fingerprint"]
        print(f"=> Projected on the frozen PCA axes -> .obsm['X_pca'] {pca_coordinates.shape} "
              f"(PCA fingerprint {pca_parameters['fingerprint']})")

    if cluster_model.umap_parameters is not None:
        umap_parameters = cluster_model.umap_parameters
        if int(umap_parameters["n_pca_components"]) > 0:
            umap_input = np.asarray(
                anndata_object.obsm["X_pca"][:, : int(umap_parameters["n_pca_components"])],
                dtype=np.float32,
            )
        else:
            umap_input = _columns(scaled_matrix, scaled_names, list(umap_parameters["input_channel_names"]))
        reducer = pickle.loads(bytes(umap_parameters["reducer_bytes"]))
        print(f"Placing {umap_input.shape[0]:,} mitochondria on the frozen UMAP map…")
        umap_coordinates = np.asarray(reducer.transform(umap_input), dtype=np.float32)
        obsm_key = umap_parameters["obsm_key"]
        anndata_object.obsm[obsm_key] = umap_coordinates
        anndata_object.uns[_ANALYSIS_CONFIG_KEY][_UMAP_PARAMS_KEY] = dict(umap_parameters["umap_params"])
        print(f"=> Placed on the frozen UMAP map -> .obsm['{obsm_key}'] "
              f"(reducer fingerprint {umap_parameters['fingerprint']})")
        del reducer, umap_input

    print(f"=> Dataset is now in reference space {cluster_model.reference_space_fingerprint}.")
    del scaled_matrix
    gc.collect()
    return anndata_object


def predict_cluster_model(
    anndata_object: anndata.AnnData,
    cluster_model: MitoClusterModel,
    force_projection: bool = False,
) -> anndata.AnnData:
    """
    Assign every mitochondrion of a dataset to the model's clusters (no refit).

    1. If the dataset already lives in the model's reference space (same
       fingerprints — e.g. a subset of the reference dataset), its stored
       coordinates are used directly. Otherwise, or with force_projection=True,
       project_to_reference_space() is called first.
    2. The frozen clusterer assigns each mitochondrion to the nearest cluster
       (KMeans) or the most probable one (GMM, with its probability).
    3. The same .obs columns as at fit time are written, with the same names
       and colors, so fit and predict datasets can be plotted side by side.

    The QC block checks: same variables, same scaler (fingerprint), distribution
    drift relative to the training set, share of atypical mitochondria (far
    from every centroid), and cluster proportions.

    Arguments:
        anndata_object: Dataset to annotate (modified in place).
        cluster_model: A fitted (or loaded) MitoClusterModel.
        force_projection: Reproject from raw .X even when fingerprints match.

    Returns:
        The same AnnData with .obs['cluster__<model_name>'] added.
    """
    print("=" * 60)
    print(f"PREDICT WITH CLUSTER MODEL '{cluster_model.model_name}'")
    print("=" * 60)
    print(
        f"Model: {cluster_model.algorithm}, space='{cluster_model.clustering_space}', "
        f"k={cluster_model.n_clusters}, reference space {cluster_model.reference_space_fingerprint}"
    )
    print(f"Dataset: {anndata_object.n_obs:,} mitochondria")
    _ensure_analysis_config(anndata_object)
    _check_features_present(anndata_object, cluster_model.feature_names, raise_error=False)

    in_reference_space = _dataset_matches_reference_space(anndata_object, cluster_model)
    if in_reference_space and not force_projection:
        print(
            "✔ Dataset already in the model's reference space (same fingerprints) — "
            "its stored coordinates are used, nothing is reprojected."
        )
        _check_stored_coordinates_exactness(anndata_object, cluster_model)
    else:
        reason = "force_projection=True" if force_projection else "fingerprints differ or space missing"
        print(f"→ Projecting from raw .X ({reason}).")
        project_to_reference_space(anndata_object, cluster_model)

    space_definition = _space_definition_from_model(cluster_model)
    space_matrix = _read_space_matrix(anndata_object, space_definition)
    _check_finite(space_matrix, "clustering space matrix")

    raw_labels = cluster_model.clusterer.predict(space_matrix)
    cluster_names = np.asarray(cluster_model.raw_label_to_name, dtype=object)[raw_labels]
    _write_assignments(anndata_object, cluster_model, cluster_names, space_matrix, role="predict")

    _print_predict_qc(anndata_object, cluster_model, space_matrix, raw_labels, cluster_names)
    del space_matrix
    gc.collect()
    return anndata_object


# ---------------------------------------------------------------------------
# Save / load / describe
# ---------------------------------------------------------------------------


def save_cluster_model(
    cluster_model: MitoClusterModel,
    file_path: str,
    overwrite: bool = False,
) -> None:
    """
    Write a cluster model to a .joblib file (e.g. on Google Drive).

    The file holds everything needed to reuse the model: the clusterer, the
    frozen scaler / PCA / UMAP parameters, the cluster profiles and the
    provenance.

    Arguments:
        cluster_model: Model to save.
        file_path: Destination path; must end with ".joblib".
        overwrite: Replace an existing file (default False).

    Raises:
        ValueError: Wrong extension.
        FileExistsError: File exists and overwrite is False.
    """
    import joblib  # noqa: PLC0415

    if not file_path.endswith(".joblib"):
        raise ValueError(f"file_path must end with '.joblib', got '{file_path}'.")
    if os.path.exists(file_path) and not overwrite:
        raise FileExistsError(f"'{file_path}' already exists. Pass overwrite=True to replace it.")
    directory = os.path.dirname(file_path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    joblib.dump(cluster_model, file_path)
    size_megabytes = os.path.getsize(file_path) / 1e6
    print(
        f"=> Cluster model '{cluster_model.model_name}' saved to {file_path} "
        f"({size_megabytes:.2f} MB, reference space {cluster_model.reference_space_fingerprint})."
    )


def load_cluster_model(file_path: str) -> MitoClusterModel:
    """
    Read a cluster model written by save_cluster_model().

    Security: a .joblib file can run code when loaded. Only load model files
    produced by your own team.

    Arguments:
        file_path: Path of the .joblib file.

    Returns:
        The MitoClusterModel.

    Raises:
        TypeError: The file does not contain a MitoClusterModel.
    """
    import joblib  # noqa: PLC0415

    cluster_model = joblib.load(file_path)
    if not isinstance(cluster_model, MitoClusterModel):
        raise TypeError(f"'{file_path}' does not contain a MitoClusterModel (got {type(cluster_model)}).")
    saved_version = cluster_model.provenance.get("package_version", "unknown")
    current_version = _package_version()
    print(
        f"=> Cluster model '{cluster_model.model_name}' loaded from {file_path} "
        f"({cluster_model.algorithm}, space '{cluster_model.clustering_space}', "
        f"k={cluster_model.n_clusters}, reference space {cluster_model.reference_space_fingerprint})."
    )
    if saved_version != current_version:
        print(
            f"WARNING: model saved with mito-marker {saved_version}, "
            f"current version is {current_version}."
        )
    return cluster_model


def describe_cluster_model(cluster_model: MitoClusterModel) -> None:
    """
    Print a readable summary of a cluster model.

    Arguments:
        cluster_model: Model to describe.
    """
    provenance = cluster_model.provenance
    print("=" * 60)
    print(f"CLUSTER MODEL '{cluster_model.model_name}'")
    print("=" * 60)
    print(f"Algorithm:        {cluster_model.algorithm} — {ALLOWED_CLUSTERING_ALGORITHMS[cluster_model.algorithm]}")
    print(f"Space:            {cluster_model.clustering_space} ({cluster_model.n_space_dimensions} dimensions)")
    print(f"Clusters:         {cluster_model.n_clusters} — {cluster_model.cluster_labels}")
    print(f"Features ({len(cluster_model.feature_names)}): {cluster_model.feature_names}")
    print(f"Normalized layer: '{cluster_model.layer_name or 'raw .X'}'")
    print(f"Reference space:  {cluster_model.reference_space_fingerprint}")
    print(f"Created:          {provenance.get('created_at')} with mito-marker {provenance.get('package_version')}")
    print(f"Training set:     {provenance.get('n_obs_training'):,} mitochondria, "
          f"{provenance.get('n_subjects_training')} subjects")
    for column_name, counts in provenance.get("training_counts", {}).items():
        print(f"  per {column_name}: {counts}")
    print("Subset history of the training set:")
    for entry in provenance.get("subset_history", []):
        print(f"  - {_describe_history_entry(entry)}")
    print("Training proportions:")
    print((cluster_model.training_proportions * 100).round(1).to_string())
    print("Mean raw profile per cluster:")
    print(cluster_model.cluster_profiles_raw.round(4).to_string())
    print("Model selection:")
    print(cluster_model.model_selection_table.to_string(index=False))
    print("=" * 60)


# ---------------------------------------------------------------------------
# Space definition and reading
# ---------------------------------------------------------------------------


def _read_space_definition(
    anndata_object: anndata.AnnData,
    clustering_space: str,
    n_pca_components_used: Optional[int],
) -> Dict[str, Any]:
    """
    Describe the space stored in anndata_object and collect its frozen parameters.

    Arguments:
        anndata_object: Training dataset.
        clustering_space: "scaled", "pca" or "umap".
        n_pca_components_used: PCs used in "pca" space (None = all).

    Returns:
        Dict with: clustering_space, layer_name, layer_parameters,
        pca_parameters, umap_parameters, space_channel_names, feature_names,
        n_dimensions, reference_space_fingerprint.

    Raises:
        ValueError: The space is missing or cannot be reused.
    """
    analysis_config = anndata_object.uns.get(_ANALYSIS_CONFIG_KEY, {})
    definition: Dict[str, Any] = {
        "clustering_space": clustering_space,
        "layer_name": "",
        "layer_parameters": None,
        "pca_parameters": None,
        "umap_parameters": None,
        "space_channel_names": [],
    }

    if clustering_space == "scaled":
        active_layer = analysis_config.get("active_layer")
        definition.update(_layer_definition(anndata_object, active_layer, required=True))
        channel_names = _active_channel_names(anndata_object)
        definition["space_channel_names"] = channel_names
        definition["feature_names"] = channel_names
        definition["n_dimensions"] = len(channel_names)

    elif clustering_space == "pca":
        definition.update(_pca_definition(anndata_object, n_pca_components_used))

    elif clustering_space == "umap":
        umap_params = dict(analysis_config.get(_UMAP_PARAMS_KEY, {}))
        obsm_key = umap_params.get("active_key")
        if not obsm_key or obsm_key not in anndata_object.obsm:
            raise ValueError(
                "No UMAP coordinates found. Run compute_umap(..., embed_all_events=True, "
                "keep_reducer=True) on the full reference dataset first."
            )
        reducers = anndata_object.uns.get(UMAP_REDUCERS_KEY, {})
        if obsm_key not in reducers or not umap_params.get("reducer_fingerprint"):
            raise ValueError(
                f"The UMAP reducer of '{obsm_key}' was not kept, so new datasets could not be "
                "placed on the same map. Re-run compute_umap(..., keep_reducer=True, "
                "embed_all_events=True, force_recompute=True) on the reference dataset."
            )
        n_pca_components = umap_params.get("n_pca_components")
        if n_pca_components is not None and int(n_pca_components) > 0:
            definition.update(_pca_definition(anndata_object, int(n_pca_components)))
        else:
            layer_name = analysis_config.get("active_layer")
            definition.update(_layer_definition(anndata_object, layer_name, required=False))
            definition["space_channel_names"] = list(umap_params["input_channel_names"])
            definition["feature_names"] = list(umap_params["input_channel_names"])
        definition["umap_parameters"] = {
            "obsm_key": obsm_key,
            "reducer_bytes": np.asarray(reducers[obsm_key], dtype=np.uint8).tobytes(),
            "n_pca_components": int(n_pca_components) if n_pca_components is not None else 0,
            "input_channel_names": [str(name) for name in umap_params.get("input_channel_names", [])],
            "fingerprint": str(umap_params["reducer_fingerprint"]),
            "umap_params": umap_params,
        }
        definition["n_dimensions"] = 2
        print(
            "WARNING: clustering in UMAP space. UMAP distorts distances and densities "
            "(Chari & Pachter 2023); clusters found there depend on n_neighbors. "
            "Prefer 'pca' or 'scaled' for the main analysis."
        )

    definition["reference_space_fingerprint"] = compute_fingerprint(
        clustering_space,
        definition["layer_parameters"]["fingerprint"] if definition["layer_parameters"] else "raw_X",
        definition["pca_parameters"]["fingerprint"] if definition["pca_parameters"] else "",
        definition["umap_parameters"]["fingerprint"] if definition["umap_parameters"] else "",
        definition["n_dimensions"],
        definition["space_channel_names"],
    )
    return definition


def _layer_definition(
    anndata_object: anndata.AnnData,
    layer_name: Optional[str],
    required: bool,
) -> Dict[str, Any]:
    """
    Return the layer part of a space definition.

    Arguments:
        anndata_object: Dataset holding the layer.
        layer_name: Layer name, or None / "" for raw .X.
        required: Raise when no normalized layer is active.

    Returns:
        Dict with layer_name and layer_parameters.

    Raises:
        ValueError: Layer required but absent, or present without frozen parameters.
    """
    if not layer_name:
        if required:
            raise ValueError(
                "clustering_space='scaled' needs an active normalized layer. Run "
                "run_preprocessing() (or transform_and_normalize()) on the full "
                "reference dataset before subsetting."
            )
        return {"layer_name": "", "layer_parameters": None}
    layer_parameters = get_layer_parameters(anndata_object, layer_name)
    if layer_parameters is None:
        raise ValueError(
            f"Layer '{layer_name}' has no frozen parameters in .uns['{LAYER_PARAMETERS_KEY}'] "
            "(created by an older version, or computed by hand). Its scaling cannot be "
            "replayed on another dataset. Delete the layer and re-run the normalization "
            "on the full reference dataset."
        )
    return {"layer_name": str(layer_name), "layer_parameters": _plain_layer_parameters(layer_parameters)}


def _pca_definition(anndata_object: anndata.AnnData, n_components_used: Optional[int]) -> Dict[str, Any]:
    """
    Return the PCA part of a space definition (plus its input layer).

    Arguments:
        anndata_object: Dataset holding .obsm['X_pca'] and the PCA parameters.
        n_components_used: PCs to use (None = all stored).

    Returns:
        Dict with layer_name, layer_parameters, pca_parameters, feature_names,
        space_channel_names, n_dimensions.

    Raises:
        ValueError: PCA missing or not reusable.
    """
    if "X_pca" not in anndata_object.obsm:
        raise ValueError("No .obsm['X_pca'] found. Run compute_pca() on the full reference dataset first.")
    if "pca_mean" not in anndata_object.uns:
        raise ValueError(
            "The PCA has no stored centring mean (.uns['pca_mean']), so new data cannot be "
            "projected on its axes. Re-run compute_pca(..., force_recompute=True) on the "
            "full reference dataset with the current package version."
        )
    input_fingerprint = str(anndata_object.uns.get("pca_input_fingerprint", "unknown"))
    if input_fingerprint == "unknown":
        raise ValueError(
            "The PCA was fitted on a layer without frozen parameters. Re-run the "
            "normalization, then compute_pca(..., force_recompute=True), on the reference dataset."
        )
    n_stored = anndata_object.obsm["X_pca"].shape[1]
    n_used = n_stored if n_components_used is None else int(n_components_used)
    if not 1 <= n_used <= n_stored:
        raise ValueError(f"n_pca_components_used={n_used} but only {n_stored} components are stored.")

    input_layer = str(anndata_object.uns.get("pca_input_layer", ""))
    definition = _layer_definition(anndata_object, input_layer, required=False)
    if definition["layer_parameters"] is not None and definition["layer_parameters"]["fingerprint"] != input_fingerprint:
        raise ValueError(
            f"Layer '{input_layer}' was re-normalized after the PCA was computed "
            "(fingerprints differ): the chain scaling -> PCA is inconsistent. Re-run compute_pca()."
        )
    loadings = np.asarray(anndata_object.uns["pca_loadings"], dtype=np.float64)
    channel_names = [str(name) for name in anndata_object.uns["pca_channel_names"]]
    weight_by = anndata_object.uns.get("pca_weight_by")
    definition["pca_parameters"] = {
        "mean": np.asarray(anndata_object.uns["pca_mean"], dtype=np.float64),
        # Only the components actually used are needed to rebuild the space,
        # but every stored component is kept so X_pca is rebuilt in full.
        "loadings": loadings,
        "explained_variance_ratio": np.asarray(anndata_object.uns["pca_explained_variance_ratio"], dtype=np.float64),
        "channel_names": channel_names,
        "weight_by": list(weight_by) if weight_by is not None else None,
        "input_fingerprint": input_fingerprint,
        "n_obs_fitted": int(anndata_object.uns.get("pca_n_obs_fitted", -1)),
        "fingerprint": str(anndata_object.uns["pca_fingerprint"]),
        "n_components_used": n_used,
    }
    definition["feature_names"] = channel_names
    definition["space_channel_names"] = []
    definition["n_dimensions"] = n_used
    return definition


def _space_definition_from_model(cluster_model: MitoClusterModel) -> Dict[str, Any]:
    """
    Rebuild the space definition stored in a model (for predict).

    Arguments:
        cluster_model: Fitted model.

    Returns:
        Space definition dict, same structure as _read_space_definition().
    """
    return {
        "clustering_space": cluster_model.clustering_space,
        "layer_name": cluster_model.layer_name,
        "layer_parameters": cluster_model.layer_parameters,
        "pca_parameters": cluster_model.pca_parameters,
        "umap_parameters": cluster_model.umap_parameters,
        "space_channel_names": cluster_model.space_channel_names,
        "feature_names": cluster_model.feature_names,
        "n_dimensions": cluster_model.n_space_dimensions,
        "reference_space_fingerprint": cluster_model.reference_space_fingerprint,
    }


def _read_space_matrix(anndata_object: anndata.AnnData, space_definition: Dict[str, Any]) -> np.ndarray:
    """
    Read the clustering matrix of a dataset according to a space definition.

    Arguments:
        anndata_object: Dataset already in the reference space.
        space_definition: Output of _read_space_definition() / _space_definition_from_model().

    Returns:
        float64 matrix of shape (n_obs, n_dimensions).
    """
    clustering_space = space_definition["clustering_space"]
    if clustering_space == "scaled":
        layer_matrix = _dense(anndata_object.layers[space_definition["layer_name"]])
        column_indices = [anndata_object.var_names.get_loc(name) for name in space_definition["space_channel_names"]]
        return np.asarray(layer_matrix[:, column_indices], dtype=np.float64)
    if clustering_space == "pca":
        return np.asarray(
            anndata_object.obsm["X_pca"][:, : space_definition["n_dimensions"]], dtype=np.float64
        )
    return np.asarray(anndata_object.obsm[space_definition["umap_parameters"]["obsm_key"]], dtype=np.float64)


def _active_channel_names(anndata_object: anndata.AnnData) -> List[str]:
    """
    Return the analytical channels of the active feature selection, in .var order.

    Arguments:
        anndata_object: Dataset to inspect.

    Returns:
        List of channel names.
    """
    keep_mask = _get_analytical_mask(anndata_object).copy()
    active_selection = anndata_object.uns.get(_ANALYSIS_CONFIG_KEY, {}).get("active_selection")
    selection_column = f"is_selected_{active_selection}"
    if active_selection is not None and selection_column in anndata_object.var.columns:
        keep_mask &= anndata_object.var[selection_column].to_numpy(dtype=bool)
    return anndata_object.var_names[keep_mask].tolist()


def _read_raw_features(anndata_object: anndata.AnnData, feature_names: List[str]) -> np.ndarray:
    """
    Return raw .X values for the given features, as float64.

    Arguments:
        anndata_object: Dataset.
        feature_names: Variables to read.

    Returns:
        Matrix (n_obs, len(feature_names)).
    """
    column_indices = [anndata_object.var_names.get_loc(name) for name in feature_names]
    return np.asarray(_dense(anndata_object.X)[:, column_indices], dtype=np.float64)


def _read_scaled_features(
    anndata_object: anndata.AnnData,
    space_definition: Dict[str, Any],
) -> Optional[np.ndarray]:
    """
    Return the normalized-layer values of the model features, or None when the
    chain starts from raw .X.

    Arguments:
        anndata_object: Dataset in the reference space.
        space_definition: Space definition.

    Returns:
        Matrix (n_obs, n_features) or None.
    """
    layer_name = space_definition["layer_name"]
    if not layer_name or layer_name not in anndata_object.layers:
        return None
    column_indices = [anndata_object.var_names.get_loc(name) for name in space_definition["feature_names"]]
    return np.asarray(_dense(anndata_object.layers[layer_name])[:, column_indices], dtype=np.float64)


def _replay_layer(
    anndata_object: anndata.AnnData,
    cluster_model: MitoClusterModel,
) -> Tuple[np.ndarray, List[str]]:
    """
    Replay the model's frozen normalization on raw .X and store the layer.

    When the normalization works column by column (z-score, min-max), only the
    model's features are needed. A row-wise L2 step uses every column of the
    reference layer, so every one of them must then be present.

    Arguments:
        anndata_object: Dataset (modified in place: layer + parameters).
        cluster_model: Model.

    Returns:
        Tuple (scaled matrix restricted to the replayed columns, their names).
    """
    layer_parameters = cluster_model.layer_parameters
    if layer_parameters is None:
        # The chain starts from raw .X: nothing to replay.
        _check_features_present(anndata_object, cluster_model.feature_names, raise_error=True)
        print("Chain starts from raw .X — no normalization to replay.")
        return _read_raw_features(anndata_object, cluster_model.feature_names), list(cluster_model.feature_names)

    all_layer_features = [str(name) for name in layer_parameters["feature_names"]]
    row_wise = "l2norm_row" in str(layer_parameters["normalization"])
    needed_features = all_layer_features if row_wise else list(cluster_model.feature_names)
    _check_features_present(anndata_object, needed_features, raise_error=True)

    parameters_to_replay = layer_parameters if row_wise else _restrict_layer_parameters(layer_parameters, needed_features)
    raw_matrix = _read_raw_features(anndata_object, needed_features).astype(np.float32)
    scaled_matrix = _apply_stored_layer_parameters(raw_matrix, parameters_to_replay)

    # The layer must span every variable of the dataset; columns outside the
    # model's features are left NaN (they were not scaled by the reference).
    layer_name = cluster_model.layer_name
    full_layer = np.full((anndata_object.n_obs, anndata_object.n_vars), np.nan, dtype=np.float32)
    for column_position, feature_name in enumerate(needed_features):
        full_layer[:, anndata_object.var_names.get_loc(feature_name)] = scaled_matrix[:, column_position]

    existing_parameters = get_layer_parameters(anndata_object, layer_name)
    if existing_parameters is not None and existing_parameters["fingerprint"] != layer_parameters["fingerprint"]:
        print(
            f"WARNING: layer '{layer_name}' of this dataset was scaled with its OWN parameters "
            f"(fingerprint {existing_parameters['fingerprint']}); it is replaced by the reference "
            f"scaling (fingerprint {layer_parameters['fingerprint']})."
        )
    anndata_object.layers[layer_name] = full_layer
    stored_parameters = dict(anndata_object.uns.get(LAYER_PARAMETERS_KEY, {}))
    stored_parameters[layer_name] = dict(layer_parameters)
    anndata_object.uns[LAYER_PARAMETERS_KEY] = stored_parameters
    anndata_object.uns[_ANALYSIS_CONFIG_KEY]["active_layer"] = layer_name
    n_untouched = anndata_object.n_vars - len(needed_features)
    print(
        f"=> Reference scaling replayed on {len(needed_features)} features -> .layers['{layer_name}'] "
        f"(fingerprint {layer_parameters['fingerprint']})"
        + (f"; {n_untouched} other variables left NaN in this layer." if n_untouched else ".")
    )
    return scaled_matrix, needed_features


def _restrict_layer_parameters(layer_parameters: Dict[str, Any], feature_names: List[str]) -> Dict[str, Any]:
    """
    Keep only the per-column parameters of the given features.

    The fingerprint is kept unchanged: it identifies the reference scaling,
    of which this is a faithful subset.

    Arguments:
        layer_parameters: Full frozen parameters.
        feature_names: Features to keep, in the wanted order.

    Returns:
        New parameter dict.
    """
    all_names = [str(name) for name in layer_parameters["feature_names"]]
    positions = [all_names.index(name) for name in feature_names]
    restricted = dict(layer_parameters)
    restricted["feature_names"] = list(feature_names)
    for key in ("logicle_channel_scale", "scaler_mean", "scaler_scale", "minmax_scale", "minmax_min"):
        values = np.asarray(layer_parameters[key], dtype=np.float64)
        restricted[key] = values[positions] if values.size else values
    return restricted


def _plain_layer_parameters(layer_parameters: Dict[str, Any]) -> Dict[str, Any]:
    """
    Copy layer parameters into plain Python / numpy types (safe to pickle and
    independent of the AnnData they were read from).

    Arguments:
        layer_parameters: Parameters read from .uns.

    Returns:
        New dict.
    """
    plain: Dict[str, Any] = {}
    for key, value in layer_parameters.items():
        if isinstance(value, np.ndarray) and value.dtype.kind in "OUS":
            plain[key] = [str(item) for item in value]
        elif isinstance(value, np.ndarray):
            plain[key] = value.astype(np.float64).copy()
        else:
            plain[key] = value
    plain["feature_names"] = [str(name) for name in layer_parameters["feature_names"]]
    return plain


# ---------------------------------------------------------------------------
# Fingerprint checks
# ---------------------------------------------------------------------------


def _dataset_matches_reference_space(anndata_object: anndata.AnnData, cluster_model: MitoClusterModel) -> bool:
    """
    Tell whether a dataset already carries the model's space (same fingerprints).

    Arguments:
        anndata_object: Dataset.
        cluster_model: Model.

    Returns:
        True when every component of the chain used by the model matches.
    """
    if cluster_model.layer_parameters is not None:
        dataset_layer = get_layer_parameters(anndata_object, cluster_model.layer_name)
        if dataset_layer is None or dataset_layer["fingerprint"] != cluster_model.layer_parameters["fingerprint"]:
            return False
        if cluster_model.layer_name not in anndata_object.layers:
            return False
    if cluster_model.pca_parameters is not None:
        if anndata_object.uns.get("pca_fingerprint") != cluster_model.pca_parameters["fingerprint"]:
            return False
        if "X_pca" not in anndata_object.obsm:
            return False
    if cluster_model.umap_parameters is not None:
        umap_params = anndata_object.uns.get(_ANALYSIS_CONFIG_KEY, {}).get(_UMAP_PARAMS_KEY, {})
        obsm_key = cluster_model.umap_parameters["obsm_key"]
        if umap_params.get("reducer_fingerprint") != cluster_model.umap_parameters["fingerprint"]:
            return False
        if obsm_key not in anndata_object.obsm or not np.isfinite(anndata_object.obsm[obsm_key]).all():
            return False
    return True


def _check_stored_coordinates_exactness(anndata_object: anndata.AnnData, cluster_model: MitoClusterModel) -> None:
    """
    Reproject a sample of rows from raw .X and compare with the stored
    coordinates. For "scaled" and "pca" spaces the two must agree to
    floating-point precision; a large gap means the stored coordinates do not
    come from the chain the fingerprints claim.

    UMAP coordinates are not checked: reducer.transform() never reproduces the
    fit-time coordinates exactly, by construction.

    Arguments:
        anndata_object: Dataset whose fingerprints match the model.
        cluster_model: Model.
    """
    if cluster_model.clustering_space == "umap":
        return
    if not all(name in anndata_object.var_names for name in _replay_feature_names(cluster_model)):
        print("  (raw features unavailable — exactness check skipped)")
        return
    rows = np.random.default_rng(0).choice(
        anndata_object.n_obs, size=min(_EXACTNESS_CHECK_ROWS, anndata_object.n_obs), replace=False
    )
    sample_anndata = anndata_object[np.sort(rows)].copy()
    stored = _read_space_matrix(sample_anndata, _space_definition_from_model(cluster_model))
    # Drop the stored space so the sample is genuinely reprojected.
    sample_anndata.uns.pop("pca_fingerprint", None)
    project_quietly = _silence(project_to_reference_space)
    project_quietly(sample_anndata, cluster_model)
    reprojected = _read_space_matrix(sample_anndata, _space_definition_from_model(cluster_model))
    maximum_gap = float(np.nanmax(np.abs(stored - reprojected))) if stored.size else 0.0
    status = "✔" if maximum_gap < 1e-3 else "✘"
    print(
        f"{status} Exactness check on {len(rows):,} rows: reprojection from raw .X vs stored "
        f"coordinates, max |difference| = {maximum_gap:.2e}"
    )
    if maximum_gap >= 1e-3:
        print("WARNING: stored coordinates differ from the reference chain — use force_projection=True.")


def _replay_feature_names(cluster_model: MitoClusterModel) -> List[str]:
    """Raw features needed to replay the model's chain."""
    if cluster_model.layer_parameters is not None and "l2norm_row" in str(cluster_model.layer_parameters["normalization"]):
        return [str(name) for name in cluster_model.layer_parameters["feature_names"]]
    return list(cluster_model.feature_names)


def _warn_if_space_fitted_on_subset(anndata_object: anndata.AnnData, space_definition: Dict[str, Any]) -> None:
    """
    Warn when the normalization or the PCA seems to have been fitted on the
    training subset itself rather than on the full reference dataset.

    The rule of ADR-015 is: build the space on the reference dataset, THEN
    subset. If a subset step is recorded but the scaling was fitted on exactly
    as many rows as the training set holds, the order was probably reversed.

    Arguments:
        anndata_object: Training dataset.
        space_definition: Its space definition.
    """
    history = get_subset_history(anndata_object)
    fitted_counts = []
    if space_definition["layer_parameters"] is not None:
        fitted_counts.append(("normalization", int(space_definition["layer_parameters"]["n_obs_fitted"])))
    if space_definition["pca_parameters"] is not None:
        fitted_counts.append(("PCA", int(space_definition["pca_parameters"]["n_obs_fitted"])))
    for step_name, n_obs_fitted in fitted_counts:
        print(f"Reference space: {step_name} fitted on {n_obs_fitted:,} mitochondria "
              f"(training set: {anndata_object.n_obs:,}).")
        if history and n_obs_fitted == anndata_object.n_obs:
            print(
                f"WARNING: the {step_name} was fitted on exactly as many mitochondria as this "
                "subset holds — it was probably recomputed AFTER subsetting. The space is then "
                "specific to this subset (ADR-015: build the space on the full reference dataset first)."
            )


# ---------------------------------------------------------------------------
# Clustering helpers
# ---------------------------------------------------------------------------


def _scan_number_of_clusters(
    space_matrix: np.ndarray,
    algorithm: str,
    candidate_k: List[int],
    sample_weight: Optional[np.ndarray],
    random_state: int,
) -> Tuple[pd.DataFrame, Dict[int, Any]]:
    """
    Fit one model per k and score it.

    Arguments:
        space_matrix: Clustering matrix.
        algorithm: "kmeans" or "gmm".
        candidate_k: k values to try.
        sample_weight: Optional KMeans weights.
        random_state: Seed.

    Returns:
        Tuple (table with n_clusters, silhouette, bic, inertia, smallest_cluster_share;
        {k: fitted model}).
    """
    rows, fitted_by_k = [], {}
    for n_clusters in candidate_k:
        if algorithm == "kmeans":
            clusterer = KMeans(n_clusters=n_clusters, n_init=10, random_state=random_state)
            clusterer.fit(space_matrix, sample_weight=sample_weight)
            inertia, bic = float(clusterer.inertia_), float("nan")
        else:
            clusterer = GaussianMixture(
                n_components=n_clusters, covariance_type="full", n_init=3, random_state=random_state
            )
            clusterer.fit(space_matrix)
            inertia, bic = float("nan"), float(clusterer.bic(space_matrix))
        labels = clusterer.predict(space_matrix)
        n_distinct = len(np.unique(labels))
        silhouette = float("nan")
        if 1 < n_distinct < len(labels):
            silhouette = float(silhouette_score(
                space_matrix, labels,
                sample_size=min(_SILHOUETTE_SAMPLE_SIZE, len(labels)), random_state=random_state,
            ))
        smallest_share = float(np.bincount(labels, minlength=n_clusters).min() / len(labels))
        rows.append({
            "n_clusters": n_clusters, "silhouette": silhouette, "bic": bic,
            "inertia": inertia, "smallest_cluster_share": smallest_share,
        })
        fitted_by_k[n_clusters] = clusterer
    return pd.DataFrame(rows), fitted_by_k


def _choose_number_of_clusters(model_selection_table: pd.DataFrame, algorithm: str) -> int:
    """
    Pick k: highest silhouette (KMeans) or lowest BIC (GMM).

    Arguments:
        model_selection_table: Output of _scan_number_of_clusters().
        algorithm: "kmeans" or "gmm".

    Returns:
        Chosen k.
    """
    if len(model_selection_table) == 1:
        return int(model_selection_table["n_clusters"].iloc[0])
    if algorithm == "kmeans":
        scores = model_selection_table["silhouette"].fillna(-np.inf)
        return int(model_selection_table.loc[scores.idxmax(), "n_clusters"])
    return int(model_selection_table.loc[model_selection_table["bic"].idxmin(), "n_clusters"])


def _name_clusters_by_size(raw_labels: np.ndarray, n_clusters: int) -> List[str]:
    """
    Map each raw label to "Cluster_<rank>", rank 1 = largest cluster.

    Arguments:
        raw_labels: Clusterer labels on the training set.
        n_clusters: Number of clusters.

    Returns:
        List where position = raw label and value = cluster name.
    """
    counts = np.bincount(raw_labels, minlength=n_clusters)
    # Sort by decreasing count; ties broken by raw label for reproducibility.
    order = sorted(range(n_clusters), key=lambda raw_label: (-counts[raw_label], raw_label))
    names = [""] * n_clusters
    for rank, raw_label in enumerate(order):
        names[raw_label] = f"{CLUSTER_LABEL_PREFIX}{rank + 1}"
    return names


def _build_cluster_colors(cluster_labels: List[str]) -> Dict[str, str]:
    """
    Fixed colors: Cluster_1 is always the first tab10 color, and so on, so a
    cluster keeps its color across figures, models and datasets.

    Arguments:
        cluster_labels: Ordered cluster names.

    Returns:
        {cluster name: hex color}.
    """
    palette = list(plt.get_cmap("tab10").colors) + list(plt.get_cmap("tab20b").colors)
    return {label: mcolors.to_hex(palette[index % len(palette)]) for index, label in enumerate(cluster_labels)}


def _compute_atypicality_threshold(
    clusterer: Any,
    algorithm: str,
    space_matrix: np.ndarray,
    raw_labels: np.ndarray,
) -> float:
    """
    Training reference for the "atypical mitochondria" QC of predict.

    Arguments:
        clusterer: Fitted model.
        algorithm: "kmeans" or "gmm".
        space_matrix: Training clustering matrix.
        raw_labels: Training raw labels.

    Returns:
        KMeans: 99th percentile of the distance to the assigned centroid.
        GMM: 1st percentile of the per-point log-likelihood.
    """
    if algorithm == "kmeans":
        return float(np.percentile(_distance_to_assigned_centroid(clusterer, space_matrix, raw_labels), 99))
    return float(np.percentile(clusterer.score_samples(space_matrix), 1))


def _atypical_share(cluster_model: MitoClusterModel, space_matrix: np.ndarray, raw_labels: np.ndarray) -> float:
    """
    Share of mitochondria less typical than 99% of the training set.

    Arguments:
        cluster_model: Model.
        space_matrix: Clustering matrix of the predicted dataset.
        raw_labels: Predicted raw labels.

    Returns:
        Share between 0 and 1.
    """
    if cluster_model.algorithm == "kmeans":
        distances = _distance_to_assigned_centroid(cluster_model.clusterer, space_matrix, raw_labels)
        return float(np.mean(distances > cluster_model.atypicality_threshold))
    return float(np.mean(cluster_model.clusterer.score_samples(space_matrix) < cluster_model.atypicality_threshold))


def _distance_to_assigned_centroid(clusterer: Any, space_matrix: np.ndarray, raw_labels: np.ndarray) -> np.ndarray:
    """Euclidean distance of each row to the centroid of its cluster."""
    return np.linalg.norm(space_matrix - clusterer.cluster_centers_[raw_labels], axis=1)


def _cluster_profiles(
    feature_matrix: np.ndarray,
    cluster_names: np.ndarray,
    cluster_labels: List[str],
    feature_names: List[str],
) -> pd.DataFrame:
    """
    Mean value of each feature per cluster (pooled over mitochondria).

    Arguments:
        feature_matrix: (n_obs, n_features).
        cluster_names: Cluster name per row.
        cluster_labels: Ordered cluster names (index of the result).
        feature_names: Column names.

    Returns:
        DataFrame (clusters × features) with an extra n_mitochondria column.
    """
    profile_frame = pd.DataFrame(feature_matrix, columns=feature_names)
    profile_frame["cluster"] = cluster_names
    profiles = profile_frame.groupby("cluster").mean().reindex(cluster_labels)
    profiles.insert(0, "n_mitochondria", profile_frame["cluster"].value_counts().reindex(cluster_labels).fillna(0).astype(int))
    return profiles


def _feature_statistic(
    scaled_features: Optional[np.ndarray],
    space_definition: Dict[str, Any],
    statistic: Any,
) -> pd.Series:
    """Per-feature statistic of the training set in the normalized layer (empty when raw .X)."""
    if scaled_features is None:
        return pd.Series(dtype=float)
    return pd.Series(statistic(scaled_features, axis=0), index=space_definition["feature_names"])


# ---------------------------------------------------------------------------
# Writing results to the AnnData
# ---------------------------------------------------------------------------


def _write_assignments(
    anndata_object: anndata.AnnData,
    cluster_model: MitoClusterModel,
    cluster_names: np.ndarray,
    space_matrix: np.ndarray,
    role: str,
) -> None:
    """
    Write cluster labels (and GMM probabilities), colors and a summary into the AnnData.

    Arguments:
        anndata_object: Dataset (modified in place).
        cluster_model: Model.
        cluster_names: Cluster name per row.
        space_matrix: Clustering matrix (for GMM probabilities).
        role: "fit" or "predict" (recorded in the summary).
    """
    if cluster_model.obs_column in anndata_object.obs.columns:
        print(f"Note: .obs['{cluster_model.obs_column}'] already existed and is overwritten.")
    anndata_object.obs[cluster_model.obs_column] = pd.Categorical(
        cluster_names, categories=cluster_model.cluster_labels
    )
    if cluster_model.algorithm == "gmm":
        anndata_object.obs[cluster_model.probability_obs_column] = (
            cluster_model.clusterer.predict_proba(space_matrix).max(axis=1).astype(np.float32)
        )

    palette = dict(anndata_object.uns.get(COLOR_PALETTE_KEY, {}))
    palette.update(cluster_model.cluster_colors)
    anndata_object.uns[COLOR_PALETTE_KEY] = palette

    counts = pd.Series(cluster_names).value_counts().reindex(cluster_model.cluster_labels, fill_value=0)
    summaries = dict(anndata_object.uns.get(CLUSTER_MODELS_UNS_KEY, {}))
    summaries[cluster_model.model_name] = {
        "role": role,
        "algorithm": cluster_model.algorithm,
        "clustering_space": cluster_model.clustering_space,
        "n_clusters": int(cluster_model.n_clusters),
        "cluster_labels": list(cluster_model.cluster_labels),
        "cluster_counts": {label: int(count) for label, count in counts.items()},
        "feature_names": list(cluster_model.feature_names),
        "reference_space_fingerprint": cluster_model.reference_space_fingerprint,
        "created_at": str(cluster_model.provenance.get("created_at", "")),
    }
    anndata_object.uns[CLUSTER_MODELS_UNS_KEY] = summaries


def _build_provenance(
    training_anndata: anndata.AnnData,
    space_definition: Dict[str, Any],
    weight_by: Optional[List[str]],
    random_state: int,
    n_pca_components_used: Optional[int],
) -> Dict[str, Any]:
    """
    Record where the model comes from.

    Arguments:
        training_anndata: Training dataset.
        space_definition: Space definition.
        weight_by: Weighting columns.
        random_state: Seed.
        n_pca_components_used: PCs used.

    Returns:
        Provenance dict.
    """
    subject_column = resolve_subject_column(training_anndata)
    training_counts = {}
    for column_name in ("specie", "condition", "diet"):
        if column_name in training_anndata.obs.columns:
            training_counts[column_name] = {
                str(value): int(count)
                for value, count in training_anndata.obs[column_name].astype(str).value_counts().items()
            }
    return {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "package_version": _package_version(),
        "n_obs_training": int(training_anndata.n_obs),
        "subject_column": subject_column or "",
        "n_subjects_training": int(training_anndata.obs[subject_column].nunique()) if subject_column else 0,
        "training_counts": training_counts,
        "subset_history": get_subset_history(training_anndata),
        "reference_n_obs_normalization": int(space_definition["layer_parameters"]["n_obs_fitted"])
        if space_definition["layer_parameters"] else -1,
        "reference_n_obs_pca": int(space_definition["pca_parameters"]["n_obs_fitted"])
        if space_definition["pca_parameters"] else -1,
        "weight_by": list(weight_by) if weight_by else [],
        "random_state": int(random_state),
        "n_pca_components_used": n_pca_components_used if n_pca_components_used is not None else -1,
    }


# ---------------------------------------------------------------------------
# QC printing
# ---------------------------------------------------------------------------


def _print_fit_qc(
    training_anndata: anndata.AnnData,
    cluster_model: MitoClusterModel,
    space_matrix: np.ndarray,
    cluster_names: np.ndarray,
) -> None:
    """Print the QC block of fit_cluster_model()."""
    from mito_marker.analysis.feature_subset import print_anndata_qc  # noqa: PLC0415

    print("--- QC fit_cluster_model ---")
    print(
        f"Space matrix: {space_matrix.shape[0]:,} × {space_matrix.shape[1]} | "
        f"min={space_matrix.min():.4g}, max={space_matrix.max():.4g}, mean={space_matrix.mean():.4g}"
    )
    print(f"Features ({len(cluster_model.feature_names)}): {cluster_model.feature_names}")
    print(f"Reference space fingerprint: {cluster_model.reference_space_fingerprint}")
    if cluster_model.layer_parameters is not None:
        print(f"  normalization '{cluster_model.layer_name}' fingerprint {cluster_model.layer_parameters['fingerprint']}")
    if cluster_model.pca_parameters is not None:
        print(f"  PCA fingerprint {cluster_model.pca_parameters['fingerprint']} "
              f"({cluster_model.pca_parameters['n_components_used']} components used)")
    if cluster_model.umap_parameters is not None:
        print(f"  UMAP reducer fingerprint {cluster_model.umap_parameters['fingerprint']}")
    chosen_row = cluster_model.model_selection_table[cluster_model.model_selection_table["chosen"]].iloc[0]
    silhouette = chosen_row["silhouette"]
    print(f"Silhouette at k={cluster_model.n_clusters}: {silhouette:.3f}")
    if np.isfinite(silhouette) and silhouette < _WEAK_SILHOUETTE_THRESHOLD:
        print(
            f"WARNING: silhouette < {_WEAK_SILHOUETTE_THRESHOLD} — weak cluster structure. The "
            "mitochondria may form a continuum rather than discrete types; read the clusters "
            "as a partition of a continuum, not as distinct populations."
        )
    _print_proportions(cluster_names, cluster_model.cluster_labels)
    print_anndata_qc(training_anndata, "fit_cluster_model")


def _print_predict_qc(
    anndata_object: anndata.AnnData,
    cluster_model: MitoClusterModel,
    space_matrix: np.ndarray,
    raw_labels: np.ndarray,
    cluster_names: np.ndarray,
) -> None:
    """Print the QC block of predict_cluster_model()."""
    from mito_marker.analysis.feature_subset import print_anndata_qc  # noqa: PLC0415

    print("--- QC predict_cluster_model ---")
    print(f"✔ Same variables (n={len(cluster_model.feature_names)}) read in the model's order: "
          f"{cluster_model.feature_names}")
    if cluster_model.layer_parameters is not None:
        dataset_layer = get_layer_parameters(anndata_object, cluster_model.layer_name)
        same_scaler = dataset_layer is not None and dataset_layer["fingerprint"] == cluster_model.layer_parameters["fingerprint"]
        print(
            f"{'✔' if same_scaler else '✘'} Same scaler: dataset "
            f"{dataset_layer['fingerprint'] if dataset_layer else 'none'} vs model "
            f"{cluster_model.layer_parameters['fingerprint']}"
        )
    print(
        f"Space matrix: {space_matrix.shape[0]:,} × {space_matrix.shape[1]} | "
        f"min={space_matrix.min():.4g}, max={space_matrix.max():.4g}, mean={space_matrix.mean():.4g}"
    )
    _print_drift(anndata_object, cluster_model)
    atypical = _atypical_share(cluster_model, space_matrix, raw_labels)
    criterion = ("farther from their centroid than 99% of training mitochondria"
                 if cluster_model.algorithm == "kmeans"
                 else "less likely under the mixture than 99% of training mitochondria")
    print(f"Atypical mitochondria: {atypical:.1%} {criterion} (expected ≈ 1%).")
    if atypical > 0.05:
        print("WARNING: many atypical mitochondria — the model describes this dataset poorly; "
              "some of its mitochondria may belong to a type absent from the training set.")
    if cluster_model.algorithm == "gmm":
        mean_probability = float(anndata_object.obs[cluster_model.probability_obs_column].mean())
        print(f"Mean membership probability of the assigned cluster: {mean_probability:.3f}")
    _print_proportions(cluster_names, cluster_model.cluster_labels, cluster_model.training_proportions)
    print_anndata_qc(anndata_object, "predict_cluster_model")


def _print_drift(anndata_object: anndata.AnnData, cluster_model: MitoClusterModel) -> None:
    """
    Compare each feature's distribution with the training set, in the
    reference normalized units (z-scores for a z-score layer).

    A shift is expected between biological groups; this block tells how large
    it is, and whether the new dataset leaves the range the model has seen.
    """
    if cluster_model.training_feature_mean.empty:
        print("Drift: chain starts from raw .X — no normalized reference units to compare in.")
        return
    scaled_features = _read_scaled_features(anndata_object, _space_definition_from_model(cluster_model))
    if scaled_features is None:
        return
    drift_table = pd.DataFrame({
        "training_mean": cluster_model.training_feature_mean,
        "dataset_mean": np.nanmean(scaled_features, axis=0),
        "dataset_std": np.nanstd(scaled_features, axis=0),
        "share_outside_training_range": np.mean(
            (scaled_features < cluster_model.training_feature_min.to_numpy())
            | (scaled_features > cluster_model.training_feature_max.to_numpy()), axis=0
        ),
    })
    drift_table["mean_shift"] = drift_table["dataset_mean"] - drift_table["training_mean"]
    flagged = drift_table[
        (drift_table["mean_shift"].abs() > _DRIFT_MEAN_SHIFT_THRESHOLD)
        | (drift_table["share_outside_training_range"] > _DRIFT_OUT_OF_RANGE_THRESHOLD)
    ]
    print(f"Drift vs training set (reference units of layer '{cluster_model.layer_name}'):")
    print(drift_table.round(3).to_string())
    if len(flagged):
        print(
            f"NOTE: {len(flagged)} feature(s) shifted by more than {_DRIFT_MEAN_SHIFT_THRESHOLD} "
            f"reference unit or with > {_DRIFT_OUT_OF_RANGE_THRESHOLD:.0%} of values outside the "
            f"training range: {flagged.index.tolist()}. A shift between biological groups is "
            "expected — it is reported, not corrected."
        )


def _print_proportions(
    cluster_names: np.ndarray,
    cluster_labels: List[str],
    training_proportions: Optional[pd.Series] = None,
) -> None:
    """Print the count and share of each cluster (and the training share, if given)."""
    counts = pd.Series(cluster_names).value_counts().reindex(cluster_labels, fill_value=0)
    table = pd.DataFrame({"n_mitochondria": counts, "percent": (counts / counts.sum() * 100).round(1)})
    if training_proportions is not None:
        table["training_percent"] = (training_proportions * 100).round(1)
    print("Cluster proportions:")
    print(table.to_string())


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _validate_fit_arguments(
    model_name: str,
    algorithm: str,
    clustering_space: str,
    max_clusters: int,
    n_clusters: Optional[int],
    weight_by: Optional[List[str]],
) -> None:
    """Raise ValueError for invalid fit_cluster_model() arguments."""
    if not re.match(CLUSTER_MODEL_NAME_PATTERN, model_name or ""):
        raise ValueError(
            f"model_name '{model_name}' is invalid: use letters, digits, '_' and '-' only."
        )
    if algorithm not in ALLOWED_CLUSTERING_ALGORITHMS:
        raise ValueError(f"Unknown algorithm '{algorithm}'. Allowed: {list(ALLOWED_CLUSTERING_ALGORITHMS)}")
    if clustering_space not in ALLOWED_CLUSTERING_SPACES:
        raise ValueError(f"Unknown clustering_space '{clustering_space}'. Allowed: {list(ALLOWED_CLUSTERING_SPACES)}")
    if n_clusters is not None and n_clusters < 2:
        raise ValueError(f"n_clusters must be at least 2, got {n_clusters}.")
    if n_clusters is None and max_clusters < 2:
        raise ValueError(f"max_clusters must be at least 2, got {max_clusters}.")
    if weight_by is not None and algorithm == "gmm":
        raise ValueError(
            "weight_by is not available with algorithm='gmm': scikit-learn's GaussianMixture "
            "does not accept sample weights, and resampling subjects to equal size is rejected "
            "by ADR-011. Use algorithm='kmeans' with weight_by, or GMM without weights."
        )


def _check_features_present(anndata_object: anndata.AnnData, feature_names: List[str], raise_error: bool) -> None:
    """
    Check that every required feature is a variable of the dataset.

    Arguments:
        anndata_object: Dataset.
        feature_names: Required variables.
        raise_error: Raise KeyError when some are missing (else just print).

    Raises:
        KeyError: Missing features and raise_error is True.
    """
    missing = [name for name in feature_names if name not in anndata_object.var_names]
    extra = [name for name in anndata_object.var_names if name not in set(feature_names)]
    if missing:
        message = f"Features required by the model are missing from this dataset: {missing}"
        if raise_error:
            raise KeyError(message)
        print(f"WARNING: {message}")
    if extra and raise_error:
        print(f"Note: {len(extra)} variable(s) of this dataset are not used by the model and are ignored.")


def _check_finite(matrix: np.ndarray, label: str) -> None:
    """Raise ValueError when a matrix holds NaN or infinite values."""
    n_nan, n_infinite = int(np.isnan(matrix).sum()), int(np.isinf(matrix).sum())
    if n_nan or n_infinite:
        raise ValueError(
            f"The {label} holds {n_nan} NaN and {n_infinite} infinite values. For a UMAP space, "
            "run compute_umap(..., embed_all_events=True) so every mitochondrion has coordinates."
        )


def _warn_if_overwriting(anndata_object: anndata.AnnData, uns_key: str, new_fingerprint: str, what: str) -> None:
    """Print a warning when a stored space with another fingerprint is replaced."""
    existing = anndata_object.uns.get(uns_key)
    if existing is not None and str(existing) != str(new_fingerprint):
        print(f"WARNING: {what} of this dataset (fingerprint {existing}) is replaced by the reference one.")


def _columns(matrix: np.ndarray, matrix_names: List[str], wanted_names: List[str]) -> np.ndarray:
    """Return the columns of matrix named wanted_names, in that order."""
    positions = [matrix_names.index(name) for name in wanted_names]
    return matrix[:, positions]


def _dense(matrix: Any) -> np.ndarray:
    """Return a dense numpy array."""
    return matrix.toarray() if hasattr(matrix, "toarray") else np.asarray(matrix)


def _ensure_analysis_config(anndata_object: anndata.AnnData) -> None:
    """Make sure .uns['analysis_config'] exists and is a mutable dict."""
    analysis_config = dict(anndata_object.uns.get(_ANALYSIS_CONFIG_KEY, {}))
    analysis_config.setdefault("active_layer", None)
    analysis_config.setdefault("active_selection", None)
    anndata_object.uns[_ANALYSIS_CONFIG_KEY] = analysis_config


def _package_version() -> str:
    """Installed mito-marker version, or "unknown"."""
    try:
        from importlib.metadata import version  # noqa: PLC0415
        return version("mito-marker")
    except Exception:  # noqa: BLE001 — any failure just means "unknown"
        return "unknown"


def _describe_history_entry(entry: Dict[str, Any]) -> str:
    """One-line description of a subset history entry."""
    if entry.get("subset_type") == "obs_values":
        return f"obs filter {entry.get('obs_filters')}: {entry.get('n_obs_before')} -> {entry.get('n_obs_after')}"
    return (
        f"{entry.get('feature_name')} {entry.get('mode')} "
        f"(fraction={entry.get('fraction')}, kept range "
        f"[{entry.get('lower_value_applied'):.4g}, {entry.get('upper_value_applied'):.4g}], "
        f"within_group='{entry.get('within_group')}'): "
        f"{entry.get('n_obs_before')} -> {entry.get('n_obs_after')}"
    )


def _silence(function: Any) -> Any:
    """Wrap a function so its console output is discarded."""
    import contextlib  # noqa: PLC0415
    import io  # noqa: PLC0415

    def silenced(*args: Any, **kwargs: Any) -> Any:
        with contextlib.redirect_stdout(io.StringIO()):
            return function(*args, **kwargs)

    return silenced
