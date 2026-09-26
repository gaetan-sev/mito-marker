"""
shap_clustering.py

Clustering of samples in SHAP value space to find mitochondrial aging endotypes.

After running the ML pipeline, each sample has a SHAP vector: one value per
morphological feature indicating how much that feature pushed the prediction
towards "old".  Clustering these vectors reveals subpopulations of mitochondria
that are classified as old (or young) for different reasons — different
mechanistic subtypes of aging.

Example: Cluster A may be driven by high Mito_Area (large mitochondria → old),
while Cluster B is driven by low Mito_Circularity (fragmented mitochondria → old).
These represent biologically distinct aging mechanisms.

  cluster_shap_values()      — cluster samples in SHAP space (HDBScan or k-means).
  plot_shap_cluster_heatmap() — heatmap of mean SHAP per cluster × feature.

Reads from anndata_object.uns['ml_results'], which is populated by run_ml_analysis().
LOGO evaluation with compute_shap=True is required to obtain per-sample SHAP values.

Typical usage:
    from mito_marker.analysis import cluster_shap_values, plot_shap_cluster_heatmap

    # First run ML with LOGO + SHAP
    results = run_ml_analysis(tem_anndata, ml_config=config)

    # Then cluster the SHAP values
    shap_df = cluster_shap_values(tem_anndata)
    plot_shap_cluster_heatmap(shap_df)
"""

from typing import Dict, List, Optional, Tuple

import anndata
import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import HDBSCAN, KMeans
from sklearn.preprocessing import StandardScaler

# .uns key where ML pipeline results are stored.
_ML_RESULTS_KEY = "ml_results"

# .uns key where SHAP clustering results are stored.
_SHAP_CLUSTERING_KEY = "shap_clustering_results"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def cluster_shap_values(
    anndata_object: anndata.AnnData,
    method: str = "hdbscan",
    n_clusters: Optional[int] = None,
    min_cluster_size: int = 5,
    positive_class_index: int = 1,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Cluster samples in SHAP value space to find mitochondrial aging endotypes.

    Each sample has a SHAP vector (one value per feature) indicating how much
    each morphological feature contributed to predicting the sample as "old".
    Clustering these vectors reveals subpopulations where aging is driven by
    different features — different biological mechanisms of aging.

    Requires LOGO evaluation with compute_shap=True in run_ml_analysis().
    The per-sample SHAP values are read from:
        anndata_object.uns['ml_results']['logo_shap_oof_values']

    For binary classification (young vs old), the SHAP values for the positive
    class (index positive_class_index, default = 1 = "old") are used.

    Results are stored in anndata_object.uns['shap_clustering_results'].

    Arguments:
        anndata_object:       AnnData after run_ml_analysis() with LOGO + SHAP.
        method:               Clustering method: "hdbscan" or "kmeans".
                              HDBScan is recommended (no fixed k required).
        n_clusters:           Number of clusters for k-means. Required when
                              method="kmeans". Ignored for "hdbscan".
        min_cluster_size:     HDBScan min_cluster_size. Ignored for "kmeans".
        positive_class_index: Index of the positive class in the SHAP value list.
                              For binary "young" / "old" classification, index 1
                              corresponds to "old" (alphabetical order).
        random_state:         Random seed for reproducibility.

    Returns:
        DataFrame with one row per sample and columns:
          shap_cluster  — int cluster label (-1 = noise for HDBScan)
          {feature_name}__shap — SHAP value for each feature

    Side effect:
        anndata_object.uns['shap_clustering_results'] is set to the returned
        DataFrame (serialized as a dict for .h5ad compatibility).
    """
    assert _ML_RESULTS_KEY in anndata_object.uns, (
        f"'.uns['{_ML_RESULTS_KEY}']' not found. "
        "Run run_ml_analysis() with evaluation_strategy='LOGO' and compute_shap=True first."
    )

    ml_results = anndata_object.uns[_ML_RESULTS_KEY]

    shap_matrix, feature_names = _extract_per_sample_shap(
        ml_results=ml_results,
        positive_class_index=positive_class_index,
    )

    print("=" * 60)
    print("CLUSTERING SHAP VALUES")
    print("=" * 60)
    print(f"Method:         {method}")
    print(f"Samples:        {shap_matrix.shape[0]}")
    print(f"SHAP features:  {shap_matrix.shape[1]}")
    print(f"Target class:   index {positive_class_index} (0-indexed, alphabetical order)")

    # Standardize SHAP values before clustering so that features with naturally
    # large SHAP magnitudes (e.g. area) do not dominate the clustering.
    scaler = StandardScaler()
    shap_matrix_scaled = scaler.fit_transform(shap_matrix)

    cluster_labels = _run_clustering(
        shap_matrix_scaled=shap_matrix_scaled,
        method=method,
        n_clusters=n_clusters,
        min_cluster_size=min_cluster_size,
        random_state=random_state,
    )

    n_clusters_found = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    noise_count = int(np.sum(cluster_labels == -1))
    print(f"\nClusters found: {n_clusters_found}")
    print(f"Noise samples:  {noise_count}/{len(cluster_labels)} ({noise_count/len(cluster_labels):.1%})")

    # Build output DataFrame: cluster label + raw (unscaled) SHAP per feature.
    shap_column_names = [f"{name}__shap" for name in feature_names]
    shap_dataframe = pd.DataFrame(shap_matrix, columns=shap_column_names)
    shap_dataframe.insert(0, "shap_cluster", cluster_labels)

    # Attach per-row subject and condition metadata when available.
    # These are stored by the ML pipeline alongside the OOF SHAP values so
    # that endotype clusters can be linked back to individual subjects.
    oof_subject_ids = ml_results.get("logo_shap_oof_unique_subject_ids")
    oof_conditions = ml_results.get("logo_shap_oof_conditions")
    if oof_subject_ids is not None and len(oof_subject_ids) == len(shap_dataframe):
        shap_dataframe.insert(1, "unique_subject_ID", oof_subject_ids)
    if oof_conditions is not None and len(oof_conditions) == len(shap_dataframe):
        col_pos = 2 if "unique_subject_ID" in shap_dataframe.columns else 1
        shap_dataframe.insert(col_pos, "condition", oof_conditions)

    # Print per-cluster summary.
    print("\nPer-cluster mean SHAP (top 3 features by absolute value):")
    for cluster_id in sorted(set(cluster_labels)):
        cluster_mask = cluster_labels == cluster_id
        cluster_shap_means = shap_matrix[cluster_mask].mean(axis=0)
        top3_indices = np.argsort(np.abs(cluster_shap_means))[-3:][::-1]
        label = f"Cluster {cluster_id}" if cluster_id != -1 else "Noise    "
        top3_text = ", ".join(
            f"{feature_names[i]}={cluster_shap_means[i]:+.3f}"
            for i in top3_indices
        )
        print(f"  {label} (n={int(cluster_mask.sum())}): {top3_text}")

    anndata_object.uns[_SHAP_CLUSTERING_KEY] = shap_dataframe.to_dict()
    print(f"\nResults stored in .uns['{_SHAP_CLUSTERING_KEY}'].")

    return shap_dataframe


def plot_shap_cluster_heatmap(
    shap_cluster_dataframe: pd.DataFrame,
    feature_names: Optional[List[str]] = None,
    title: str = "",
    center_colormap_at_zero: bool = True,
    show_membership: bool = True,
    show_dendrograms: bool = True,
) -> matplotlib.figure.Figure:
    """
    Plot a heatmap of mean SHAP values per cluster × feature, with dendrograms.

    Each row is a cluster. Each column is a morphological feature. The cell
    color shows the mean SHAP value of that feature in that cluster:
      - Red  (+): the feature pushes predictions towards the positive class ("old").
      - Blue (-): the feature pushes predictions towards the negative class ("young").
      - White (0): the feature has no influence for samples in that cluster.

    Row and column dendrograms (hierarchical clustering) are shown by default so
    similar clusters and co-varying features are grouped together visually.

    When "unique_subject_ID" and "condition" columns are present in the DataFrame
    (populated automatically by cluster_shap_values() after running run_ml_analysis()),
    each row label is extended with the list of subjects per condition, and a
    colour bar on the left shows the dominant condition per cluster.

    Arguments:
        shap_cluster_dataframe: Output of cluster_shap_values(). Must contain
                                "shap_cluster" column and "{feature}__shap" columns.
        feature_names:          Subset of features to display. Defaults to all.
        title:                  Plot title.
        center_colormap_at_zero: Centers the colormap at 0 (default True) so
                                  red/blue symmetrically encode positive/negative SHAP.
        show_membership:        When True (default) and subject metadata is present,
                                extends row labels with "Y: subj1 | O: subj2" and
                                adds a condition colour bar on the left.
        show_dendrograms:       When True (default), adds row and column dendrograms
                                using hierarchical clustering (Ward linkage).
                                Skipped automatically when fewer than 2 rows or cols.

    Returns:
        matplotlib Figure object.
    """
    import seaborn as sns  # noqa: PLC0415

    shap_columns = [
        col for col in shap_cluster_dataframe.columns if col.endswith("__shap")
    ]
    if feature_names is not None:
        shap_columns = [
            f"{name}__shap" for name in feature_names
            if f"{name}__shap" in shap_columns
        ]

    display_names = [col.replace("__shap", "") for col in shap_columns]

    # Build per-cluster mean SHAP matrix.
    cluster_ids = sorted(shap_cluster_dataframe["shap_cluster"].unique())
    mean_shap_rows = []
    for cluster_id in cluster_ids:
        cluster_mask = shap_cluster_dataframe["shap_cluster"] == cluster_id
        cluster_means = shap_cluster_dataframe.loc[cluster_mask, shap_columns].mean()
        mean_shap_rows.append(cluster_means.values)
    heatmap_matrix = np.array(mean_shap_rows)  # shape (n_clusters, n_features)

    n_clusters_plot = len(cluster_ids)
    n_features_plot = len(display_names)

    # Hierarchical clustering requires at least 2 points.
    can_cluster_rows = show_dendrograms and n_clusters_plot >= 2
    can_cluster_cols = show_dendrograms and n_features_plot >= 2

    # Detect condition values when membership metadata is available.
    has_membership = (
        show_membership
        and "unique_subject_ID" in shap_cluster_dataframe.columns
        and "condition" in shap_cluster_dataframe.columns
    )
    young_value: Optional[str] = None
    old_value: Optional[str] = None
    if has_membership:
        from mito_marker.analysis.clustering import _detect_condition_values  # noqa: PLC0415
        young_value, old_value = _detect_condition_values(
            shap_cluster_dataframe["condition"].values
        )

    # Build base row labels ("Cluster 0 (n=50)").
    row_labels = [
        f"Noise (n={int((shap_cluster_dataframe['shap_cluster'] == c).sum())})"
        if c == -1
        else f"Cluster {c} (n={int((shap_cluster_dataframe['shap_cluster'] == c).sum())})"
        for c in cluster_ids
    ]

    # Left colour bar: dominant condition per cluster (green=young, red=old).
    row_colors_series: Optional[pd.Series] = None
    if has_membership:
        condition_colors = []
        for cluster_id in cluster_ids:
            mask = shap_cluster_dataframe["shap_cluster"] == cluster_id
            conditions = shap_cluster_dataframe.loc[mask, "condition"]
            dominant = (
                young_value
                if (conditions == young_value).sum() >= (conditions == old_value).sum()
                else old_value
            )
            condition_colors.append("#4CAF50" if dominant == young_value else "#F44336")
        row_colors_series = pd.Series(
            condition_colors, index=row_labels, name="Condition"
        )

    # Colormap boundaries.
    vmax = float(np.abs(heatmap_matrix).max()) if center_colormap_at_zero else None

    # Figure dimensions.
    figure_height = max(5, n_clusters_plot * 1.0 + 3)
    figure_width  = max(14, n_features_plot * 0.7 + 5)

    heatmap_df = pd.DataFrame(heatmap_matrix, index=row_labels, columns=display_names)

    clustermap_kwargs: dict = dict(
        cmap="RdBu_r",
        center=0,
        row_colors=row_colors_series,
        row_cluster=can_cluster_rows,
        col_cluster=can_cluster_cols,
        figsize=(figure_width, figure_height),
        annot=True,
        fmt=".2f",
        annot_kws={"size": 7},
        dendrogram_ratio=(0.12, 0.06),
        cbar_pos=(0.01, 0.78, 0.02, 0.15),
        linewidths=0.3,
        linecolor="lightgrey",
    )
    if center_colormap_at_zero and vmax is not None and vmax > 0:
        clustermap_kwargs["vmin"] = -vmax
        clustermap_kwargs["vmax"] = vmax

    grid = sns.clustermap(heatmap_df, **clustermap_kwargs)

    # Style x-axis ticks.
    grid.ax_heatmap.set_xticklabels(
        grid.ax_heatmap.get_xticklabels(), rotation=45, ha="right", fontsize=9
    )
    grid.ax_heatmap.set_xlabel("Morphological feature", fontsize=11)
    grid.ax_heatmap.set_ylabel("")

    # Determine cluster order after dendrogram reordering.
    if can_cluster_rows:
        cluster_ids_ordered = [
            cluster_ids[i] for i in grid.dendrogram_row.reordered_ind
        ]
    else:
        cluster_ids_ordered = cluster_ids

    # Build y-tick labels: extend with subject membership when available.
    if has_membership:
        new_ytick_labels = []
        for cluster_id in cluster_ids_ordered:
            mask = shap_cluster_dataframe["shap_cluster"] == cluster_id
            per_subject = (
                shap_cluster_dataframe[mask]
                .drop_duplicates("unique_subject_ID")[["unique_subject_ID", "condition"]]
            )
            young_subjects = sorted(
                per_subject.loc[per_subject["condition"] == young_value, "unique_subject_ID"]
            )
            old_subjects = sorted(
                per_subject.loc[per_subject["condition"] == old_value, "unique_subject_ID"]
            )
            parts = []
            if young_subjects:
                parts.append(f"Y: {', '.join(young_subjects)}")
            if old_subjects:
                parts.append(f"O: {', '.join(old_subjects)}")
            n = int(mask.sum())
            cluster_name = "Noise" if cluster_id == -1 else f"Cluster {cluster_id}"
            new_ytick_labels.append(
                f"{cluster_name} (n={n})  —  {' | '.join(parts)}"
            )
        grid.ax_heatmap.set_yticklabels(new_ytick_labels, fontsize=9, rotation=0)
    else:
        reordered_labels = (
            [row_labels[i] for i in grid.dendrogram_row.reordered_ind]
            if can_cluster_rows else row_labels
        )
        grid.ax_heatmap.set_yticklabels(reordered_labels, fontsize=10, rotation=0)

    grid.fig.suptitle(
        title or "Mean SHAP value per cluster (endotypes of mitochondrial aging)",
        fontsize=13, fontweight="bold", y=1.01,
    )

    return grid.fig


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _extract_per_sample_shap(
    ml_results: Dict,
    positive_class_index: int,
) -> Tuple[np.ndarray, List[str]]:
    """
    Extract the per-sample SHAP value matrix from ml_results.

    For LOGO evaluation, reads logo_shap_oof_values (list of arrays, one per
    class) and returns the array for positive_class_index.

    For StandardCV with binary classification, reads shap_values directly
    (already a 2D array for the positive class).

    Arguments:
        ml_results:            anndata_object.uns['ml_results'] dict.
        positive_class_index:  Index of the class whose SHAP values to extract.

    Returns:
        Tuple of:
          shap_matrix    — float array (n_samples, n_features)
          feature_names  — list of feature names
    """
    feature_names: List[str] = ml_results.get("shap_feature_names") or ml_results.get("feature_names", [])

    # Prefer LOGO per-sample OOF values (richer, fold-validated).
    logo_oof_shap = ml_results.get("logo_shap_oof_values")
    if logo_oof_shap is not None:
        assert isinstance(logo_oof_shap, list), (
            "logo_shap_oof_values must be a list of arrays (one per class)."
        )
        if len(logo_oof_shap) > positive_class_index:
            shap_matrix = np.array(logo_oof_shap[positive_class_index], dtype=np.float32)
        else:
            # Fall back to index 0 if positive_class_index is out of range.
            shap_matrix = np.array(logo_oof_shap[0], dtype=np.float32)
        return shap_matrix, feature_names

    # Fall back to StandardCV shap_values.
    standard_shap = ml_results.get("shap_values")
    assert standard_shap is not None, (
        "No per-sample SHAP values found in ml_results. "
        "Run run_ml_analysis() with compute_shap=True. "
        "For LOGO evaluation, logo_shap_oof_values is required."
    )

    if isinstance(standard_shap, list):
        # Multi-class or binary list format.
        if len(standard_shap) > positive_class_index:
            shap_matrix = np.array(standard_shap[positive_class_index], dtype=np.float32)
        else:
            shap_matrix = np.array(standard_shap[0], dtype=np.float32)
    elif isinstance(standard_shap, np.ndarray):
        if standard_shap.ndim == 3:
            # 3D: (n_samples, n_features, n_classes) from LinearExplainer.
            shap_matrix = standard_shap[:, :, positive_class_index].astype(np.float32)
        else:
            shap_matrix = standard_shap.astype(np.float32)
    else:
        raise ValueError(
            f"Unexpected shap_values type: {type(standard_shap)}. "
            "Expected list or numpy array."
        )

    return shap_matrix, feature_names


def _draw_membership_panel(
    axis: "matplotlib.axes.Axes",
    shap_cluster_dataframe: pd.DataFrame,
    cluster_ids: List[int],
) -> None:
    """
    Draw the right annotation panel of the SHAP endotype heatmap.

    For each cluster row, lists the unique subjects with their condition shown
    as a coloured square: green for young, orange/red for old.

    Arguments:
        axis:                   The axes to draw into.
        shap_cluster_dataframe: DataFrame with "shap_cluster", "unique_subject_ID",
                                and "condition" columns.
        cluster_ids:            Ordered list of cluster IDs (same order as heatmap rows).
    """
    from mito_marker.analysis.clustering import _detect_condition_values

    all_conditions = shap_cluster_dataframe["condition"].values
    young_value, old_value = _detect_condition_values(all_conditions)

    condition_color_map = {young_value: "#4CAF50", old_value: "#F44336"}

    axis.set_xlim(0, 1)
    axis.set_ylim(-0.5, len(cluster_ids) - 0.5)
    axis.invert_yaxis()
    axis.axis("off")
    axis.set_title("Subjects", fontsize=10, fontweight="bold")

    for row_index, cluster_id in enumerate(cluster_ids):
        cluster_mask = shap_cluster_dataframe["shap_cluster"] == cluster_id
        cluster_rows = shap_cluster_dataframe[cluster_mask]

        # Deduplicate: each subject appears once per cluster (many bags per subject).
        subject_condition_pairs = (
            cluster_rows[["unique_subject_ID", "condition"]]
            .drop_duplicates()
            .sort_values("condition")
        )

        x_offset = 0.02
        for _, row in subject_condition_pairs.iterrows():
            subject_label = str(row["unique_subject_ID"])
            condition_value = str(row["condition"])
            square_color = condition_color_map.get(condition_value, "#AAAAAA")

            # Coloured square for condition.
            axis.add_patch(
                plt.Rectangle(
                    (x_offset, row_index - 0.3),
                    0.06, 0.6,
                    color=square_color,
                    transform=axis.transData,
                )
            )
            # Subject label next to the square.
            axis.text(
                x_offset + 0.08,
                row_index,
                subject_label,
                va="center",
                fontsize=7,
                color="black",
            )
            x_offset += 0.08 + len(subject_label) * 0.045 + 0.04

            # Wrap to next line if too wide.
            if x_offset > 0.85:
                row_index += 0.45
                x_offset = 0.02


def _run_clustering(
    shap_matrix_scaled: np.ndarray,
    method: str,
    n_clusters: Optional[int],
    min_cluster_size: int,
    random_state: int,
) -> np.ndarray:
    """
    Apply the selected clustering algorithm to the scaled SHAP matrix.

    Arguments:
        shap_matrix_scaled: 2D float array (n_samples, n_features), standardized.
        method:             "hdbscan" or "kmeans".
        n_clusters:         Number of clusters for k-means; ignored for HDBScan.
        min_cluster_size:   HDBScan parameter; ignored for k-means.
        random_state:       Random seed.

    Returns:
        1D int array of cluster labels (n_samples,).
    """
    assert method in ("hdbscan", "kmeans"), (
        f"Unknown clustering method: '{method}'. Choose 'hdbscan' or 'kmeans'."
    )

    if method == "hdbscan":
        clusterer = HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_cluster_size,
        )
        return clusterer.fit_predict(shap_matrix_scaled)

    # k-means
    assert n_clusters is not None, (
        "n_clusters must be provided when method='kmeans'."
    )
    clusterer = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init="auto",
    )
    return clusterer.fit_predict(shap_matrix_scaled)
