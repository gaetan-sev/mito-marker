"""
clustering.py

Density-based clustering of mitochondria at single-mito and bag levels.

Provides tools to test whether the biological signal (young vs old) emerges
more clearly when mitochondria are aggregated into increasingly large bags.

  compute_umap_on_bags()    — aggregate mitos into bags, then run UMAP.
  run_hdbscan()             — HDBScan clustering on any 2D embedding.
  evaluate_clustering()     — silhouette score, n_clusters, noise ratio.
  profile_clusters()        — mean feature values + age composition per cluster.
  plot_cluster_age_composition() — stacked bar: % young/old per cluster.
  compare_bag_sizes()       — run the full pipeline across multiple bag sizes.

The bagging logic reuses create_bags() from ml_bagging.py — the same sampling
strategy as the ML pipeline — so the two analyses are directly comparable.

Typical usage:
    from mito_marker.analysis import compare_bag_sizes, plot_cluster_age_composition

    results = compare_bag_sizes(
        tem_anndata,
        bag_sizes=[1, 5, 30, 150],
        bags_per_subject=30,
        subject_id_column="subject_ID",
        condition_column="condition",
    )
    plot_cluster_age_composition(results[30]["profile"], title="30 mitos/bag")
"""

from typing import Dict, List, Optional, Tuple

import anndata
import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import HDBSCAN
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from mito_marker.analysis.ml_bagging import create_bags
from mito_marker.analysis.umap_plot import _build_umap_reducer

# .uns key where bag clustering results are stored.
_BAG_CLUSTERING_KEY = "bag_clustering_results"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_umap_on_bags(
    anndata_object: anndata.AnnData,
    mitos_per_bag: int,
    bags_per_subject: int,
    subject_id_column: str,
    condition_column: str,
    bag_statistics: Optional[List[str]] = None,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """
    Aggregate mitochondria into statistical bags and compute UMAP on the bag vectors.

    When mitos_per_bag == 1, the function samples individual mitochondria
    directly without aggregation. This provides the single-mito baseline for
    the compare_bag_sizes() analysis.

    For mitos_per_bag > 1, the function calls create_bags() — the same
    sampling strategy as the ML pipeline — to aggregate mitos into bags, then
    runs UMAP on the bag feature vectors (mean/std/median/skew per feature).

    Features are standardized (StandardScaler) before UMAP to avoid any single
    morphological measurement dominating the embedding.

    Arguments:
        anndata_object:    TEM AnnData with morphological features in .X.
        mitos_per_bag:     Number of mitochondria per bag.
                           Use 1 for the single-mito baseline (no aggregation).
        bags_per_subject:  Number of bags to sample per subject.
        subject_id_column: .obs column identifying individual subjects.
        condition_column:  .obs column with condition labels (e.g. "condition",
                           values "young" / "old").
        bag_statistics:    Subset of ["mean", "std", "median", "skew"] to
                           compute when mitos_per_bag > 1. Defaults to all four.
                           Ignored when mitos_per_bag == 1.
        n_neighbors:       UMAP n_neighbors parameter (neighborhood size).
        min_dist:          UMAP min_dist parameter (minimum embedded distance).
        random_state:      Random seed for reproducibility.

    Returns:
        Tuple of five elements:
          umap_coordinates — float array (n_samples, 2)
          feature_matrix   — float array (n_samples, n_features), unscaled —
                             aligned with umap_coordinates rows, used for
                             profiling in compare_bag_sizes()
          condition_labels — str array (n_samples,) — "young" / "old"
          subject_ids      — str array (n_samples,)
          feature_names    — list of feature names (original for bag_size=1,
                             e.g. "Mito_Area__mean" for bag_size > 1)
    """
    if bag_statistics is None:
        bag_statistics = ["mean", "std", "median", "skew"]

    data_matrix = np.array(anndata_object.X, dtype=np.float32)
    original_feature_names = list(anndata_object.var_names)

    print(f"\nComputing UMAP — bag size = {mitos_per_bag} mito(s)/bag")
    print(f"  Subjects: {anndata_object.obs[subject_id_column].nunique()}")
    print(f"  Bags per subject: {bags_per_subject}")

    if mitos_per_bag == 1:
        print("  Mode: single-mito sampling (no aggregation)")
        feature_matrix, condition_labels, subject_ids = _sample_individual_mitos(
            data_matrix=data_matrix,
            obs_dataframe=anndata_object.obs,
            subject_id_column=subject_id_column,
            condition_column=condition_column,
            n_per_subject=bags_per_subject,
            random_state=random_state,
        )
        feature_names = original_feature_names
    else:
        print(f"  Mode: bagging (statistics: {', '.join(bag_statistics)})")
        feature_matrix, condition_labels, subject_ids, feature_names = create_bags(
            data_matrix=data_matrix,
            obs_dataframe=anndata_object.obs,
            subject_id_column=subject_id_column,
            bags_per_subject=bags_per_subject,
            mitos_per_bag=mitos_per_bag,
            bag_statistics=bag_statistics,
            target_obs_column=condition_column,
            task_type="classification",
            random_state=random_state,
            n_jobs=-1,
            feature_names=original_feature_names,
        )

    print(f"  Samples: {feature_matrix.shape[0]} × {feature_matrix.shape[1]} features")

    # Standardize before UMAP so no single feature dominates the embedding.
    scaler = StandardScaler()
    feature_matrix_scaled = scaler.fit_transform(feature_matrix)

    reducer = _build_umap_reducer(
        n_neighbors=n_neighbors, min_dist=min_dist, random_state=random_state
    )
    umap_coordinates = reducer.fit_transform(feature_matrix_scaled).astype(np.float32)

    print(f"  UMAP complete. Shape: {umap_coordinates.shape}")

    return umap_coordinates, feature_matrix, condition_labels, subject_ids, feature_names


def run_hdbscan(
    coordinates: np.ndarray,
    min_cluster_size: int = 5,
    min_samples: Optional[int] = None,
) -> np.ndarray:
    """
    Run HDBScan density-based clustering on a 2D embedding or feature matrix.

    HDBScan (Hierarchical Density-Based Spatial Clustering) does not require
    specifying the number of clusters. Dense regions become clusters; sparse
    samples are assigned label -1 (noise). This makes it well-suited for
    biological data where the number of subpopulations is unknown.

    Uses sklearn.cluster.HDBSCAN (scikit-learn >= 1.3).

    Arguments:
        coordinates:       2D float array (n_samples, n_dims). Typically UMAP
                           coordinates, but any embedding works.
        min_cluster_size:  Minimum samples for a region to be a cluster.
                           Increase for larger, cleaner clusters; decrease to
                           detect smaller subpopulations.
        min_samples:       Minimum neighbors for a core point. Defaults to
                           min_cluster_size when None.

    Returns:
        1D int array (n_samples,) of cluster labels.
        Label -1 indicates noise (not assigned to any cluster).
    """
    effective_min_samples = (
        min_samples if min_samples is not None else min_cluster_size
    )

    clusterer = HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=effective_min_samples,
    )
    cluster_labels = clusterer.fit_predict(coordinates)

    n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    noise_count = int(np.sum(cluster_labels == -1))
    noise_ratio = noise_count / len(cluster_labels)

    print(
        f"  HDBScan: {n_clusters} cluster(s), "
        f"{noise_count}/{len(cluster_labels)} noise points ({noise_ratio:.1%})"
    )

    return cluster_labels


def evaluate_clustering(
    coordinates: np.ndarray,
    cluster_labels: np.ndarray,
) -> Dict[str, float]:
    """
    Compute quality metrics for a clustering result.

    Metrics:
      n_clusters:       number of clusters (noise label -1 excluded).
      noise_ratio:      fraction of samples assigned to noise.
      silhouette_score: mean silhouette coefficient across non-noise samples,
                        ranging from -1 (wrong cluster) to +1 (perfect cluster).
                        NaN when fewer than 2 non-noise clusters exist.

    Arguments:
        coordinates:    2D float array (n_samples, n_dims).
        cluster_labels: 1D int array (n_samples,) from run_hdbscan().

    Returns:
        Dict with keys "n_clusters", "noise_ratio", "silhouette_score".
    """
    unique_labels = set(cluster_labels)
    n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
    noise_ratio = float(np.mean(cluster_labels == -1))

    non_noise_mask = cluster_labels != -1
    non_noise_labels = cluster_labels[non_noise_mask]
    n_unique_non_noise = len(set(non_noise_labels))

    silhouette = float("nan")
    if n_unique_non_noise >= 2 and int(non_noise_mask.sum()) >= 2:
        silhouette = float(
            silhouette_score(coordinates[non_noise_mask], non_noise_labels)
        )

    return {
        "n_clusters": n_clusters,
        "noise_ratio": noise_ratio,
        "silhouette_score": silhouette,
    }


def profile_clusters(
    feature_matrix: np.ndarray,
    cluster_labels: np.ndarray,
    feature_names: List[str],
    condition_labels: np.ndarray,
    condition_young_value: str = "young",
    condition_old_value: str = "old",
) -> pd.DataFrame:
    """
    Build a summary profile for each cluster: mean feature values + age composition.

    For each cluster (including noise cluster -1), computes the number of
    samples, the percentage of young and old samples, and the mean value of
    every feature.

    When called after compare_bag_sizes() with bags (bag_size > 1), feature_names
    will include statistic suffixes such as "Mito_Area__mean". Use the
    "__mean" columns for morphological comparison radar plots.

    Arguments:
        feature_matrix:         2D float array (n_samples, n_features).
        cluster_labels:         1D int array (n_samples,) from run_hdbscan().
        feature_names:          List of feature names aligned with feature_matrix
                                columns.
        condition_labels:       1D str array (n_samples,) with condition values.
        condition_young_value:  String identifying the "young" condition.
        condition_old_value:    String identifying the "old" condition.

    Returns:
        DataFrame indexed by cluster label (-1, 0, 1, 2, ...) with columns:
          cluster, n_samples, pct_young, pct_old, {feature_names...}
    """
    unique_clusters = sorted(set(cluster_labels))
    rows = []

    for cluster_id in unique_clusters:
        mask = cluster_labels == cluster_id
        cluster_features = feature_matrix[mask]
        cluster_conditions = condition_labels[mask]

        n_samples = int(mask.sum())
        n_young = int(np.sum(cluster_conditions == condition_young_value))
        n_old = int(np.sum(cluster_conditions == condition_old_value))
        pct_young = 100.0 * n_young / n_samples if n_samples > 0 else 0.0
        pct_old = 100.0 * n_old / n_samples if n_samples > 0 else 0.0

        mean_features = cluster_features.mean(axis=0)

        row: Dict = {
            "cluster": cluster_id,
            "n_samples": n_samples,
            "pct_young": round(pct_young, 1),
            "pct_old": round(pct_old, 1),
        }
        for feature_index, feature_name in enumerate(feature_names):
            row[feature_name] = float(mean_features[feature_index])

        rows.append(row)

    profile_dataframe = pd.DataFrame(rows).set_index("cluster")
    return profile_dataframe


def plot_cluster_age_composition(
    profile_dataframe: pd.DataFrame,
    condition_young_value: str = "young",
    condition_old_value: str = "old",
    young_color: str = "#4CAF50",
    old_color: str = "#F44336",
    title: str = "",
) -> matplotlib.figure.Figure:
    """
    Plot a stacked bar chart showing the age composition of each cluster.

    Each bar represents one cluster. The green portion shows the percentage of
    young samples; the red portion shows old samples. Clusters are sorted by
    their young fraction (highest first) to reveal a gradient from young-dominant
    to old-dominant clusters.

    Noise samples (cluster -1) are shown with a dashed border.

    Arguments:
        profile_dataframe:      Output of profile_clusters(), indexed by cluster.
        condition_young_value:  Label for young samples (legend entry).
        condition_old_value:    Label for old samples (legend entry).
        young_color:            Hex color for the young fraction bars.
        old_color:              Hex color for the old fraction bars.
        title:                  Plot title. Defaults to "Age composition per cluster".

    Returns:
        matplotlib Figure object.
    """
    # Separate noise from named clusters for sorting.
    non_noise = profile_dataframe[profile_dataframe.index != -1].copy()
    noise_rows = profile_dataframe[profile_dataframe.index == -1].copy()

    sorted_profile = non_noise.sort_values("pct_young", ascending=False)
    if not noise_rows.empty:
        sorted_profile = pd.concat([sorted_profile, noise_rows])

    n_bars = len(sorted_profile)
    figure, axis = plt.subplots(figsize=(max(6, n_bars * 1.5), 6))

    cluster_ids = sorted_profile.index.tolist()
    pct_young = sorted_profile["pct_young"].values
    pct_old = sorted_profile["pct_old"].values
    x_positions = np.arange(n_bars)

    bars_young = axis.bar(
        x_positions, pct_young, color=young_color, label=condition_young_value
    )
    bars_old = axis.bar(
        x_positions, pct_old, bottom=pct_young, color=old_color, label=condition_old_value
    )

    # Dashed border on noise cluster bars.
    for bar_index, cluster_id in enumerate(cluster_ids):
        if cluster_id == -1:
            for bar in (bars_young[bar_index], bars_old[bar_index]):
                bar.set_linestyle("--")
                bar.set_edgecolor("black")
                bar.set_linewidth(1.5)

    # Annotate bars with percentage labels.
    for bar_index in range(n_bars):
        if pct_young[bar_index] >= 5:
            axis.text(
                x_positions[bar_index],
                pct_young[bar_index] / 2,
                f"{pct_young[bar_index]:.0f}%",
                ha="center", va="center", fontsize=9, color="white", fontweight="bold",
            )
        if pct_old[bar_index] >= 5:
            axis.text(
                x_positions[bar_index],
                pct_young[bar_index] + pct_old[bar_index] / 2,
                f"{pct_old[bar_index]:.0f}%",
                ha="center", va="center", fontsize=9, color="white", fontweight="bold",
            )

    # X-axis tick labels include sample count.
    tick_labels = [
        f"Noise\n(n={sorted_profile.loc[c, 'n_samples']})" if c == -1
        else f"Cluster {c}\n(n={sorted_profile.loc[c, 'n_samples']})"
        for c in cluster_ids
    ]
    axis.set_xticks(x_positions)
    axis.set_xticklabels(tick_labels, fontsize=10)
    axis.set_ylabel("Percentage of samples (%)", fontsize=11)
    axis.set_ylim(0, 100)
    axis.set_yticks(range(0, 101, 10))
    axis.legend(loc="upper right", fontsize=10)
    axis.set_title(title or "Age composition per cluster", fontsize=13, fontweight="bold")

    plt.tight_layout()
    return figure


def compare_bag_sizes(
    anndata_object: anndata.AnnData,
    bag_sizes: List[int],
    bags_per_subject: int,
    subject_id_column: str,
    condition_column: str,
    bag_statistics: Optional[List[str]] = None,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    min_cluster_size: int = 5,
    random_state: int = 42,
) -> Dict[int, Dict]:
    """
    Compare clustering quality and age composition across increasing bag sizes.

    For each bag size, this function:
      1. Aggregates mitos into bags (or samples individual mitos for bag_size=1).
      2. Runs UMAP on the resulting feature vectors.
      3. Runs HDBScan on UMAP coordinates.
      4. Builds a cluster profile with age composition.
      5. Evaluates silhouette score and noise ratio.

    The biological hypothesis: does the young/old signal become clearer
    (more distinct clusters, higher silhouette score) as bag size increases?
    If yes, the aging signal is a population-level effect rather than
    detectable in individual mitochondria.

    Results are stored in anndata_object.uns['bag_clustering_results'][bag_size].

    Arguments:
        anndata_object:    TEM AnnData with morphological features in .X.
        bag_sizes:         List of bag sizes to compare, e.g. [1, 5, 30, 150].
                           Use 1 as the single-mito baseline.
        bags_per_subject:  Number of bags (or mito samples) per subject.
        subject_id_column: .obs column identifying individual subjects.
        condition_column:  .obs column with condition labels ("young" / "old").
        bag_statistics:    Subset of ["mean", "std", "median", "skew"].
                           Defaults to all four. Ignored for bag_size=1.
        n_neighbors:       UMAP n_neighbors parameter.
        min_dist:          UMAP min_dist parameter.
        min_cluster_size:  HDBScan min_cluster_size parameter.
        random_state:      Random seed for reproducibility.

    Returns:
        Dict mapping each bag_size (int) to a results dict with keys:
          "umap_coordinates" — float array (n_samples, 2)
          "feature_matrix"   — float array (n_samples, n_features), unscaled
          "condition_labels" — str array (n_samples,)
          "subject_ids"      — str array (n_samples,)
          "feature_names"    — list of feature names
          "cluster_labels"   — int array (n_samples,)
          "profile"          — DataFrame from profile_clusters()
          "metrics"          — Dict from evaluate_clustering()

    Side effect:
        anndata_object.uns['bag_clustering_results'] is populated with the
        same results (keyed by bag_size) for downstream notebook access.
    """
    if bag_statistics is None:
        bag_statistics = ["mean", "std", "median", "skew"]

    print("=" * 60)
    print("COMPARING BAG SIZES FOR CLUSTERING ANALYSIS")
    print("=" * 60)
    print(f"Bag sizes to test: {bag_sizes}")
    print(f"Bags per subject:  {bags_per_subject}")
    print(f"Statistics:        {bag_statistics}")
    print(f"UMAP n_neighbors:  {n_neighbors}, min_dist: {min_dist}")
    print(f"HDBScan min_cluster_size: {min_cluster_size}")

    all_results: Dict[int, Dict] = {}

    if _BAG_CLUSTERING_KEY not in anndata_object.uns:
        anndata_object.uns[_BAG_CLUSTERING_KEY] = {}

    for bag_size in bag_sizes:
        print(f"\n{'─' * 50}")
        print(f"Bag size = {bag_size} mito(s)/bag")
        print(f"{'─' * 50}")

        umap_coordinates, feature_matrix, condition_labels, subject_ids, feature_names = (
            compute_umap_on_bags(
                anndata_object=anndata_object,
                mitos_per_bag=bag_size,
                bags_per_subject=bags_per_subject,
                subject_id_column=subject_id_column,
                condition_column=condition_column,
                bag_statistics=bag_statistics,
                n_neighbors=n_neighbors,
                min_dist=min_dist,
                random_state=random_state,
            )
        )

        cluster_labels = run_hdbscan(
            coordinates=umap_coordinates,
            min_cluster_size=min_cluster_size,
        )

        metrics = evaluate_clustering(umap_coordinates, cluster_labels)

        young_value, old_value = _detect_condition_values(condition_labels)

        profile = profile_clusters(
            feature_matrix=feature_matrix,
            cluster_labels=cluster_labels,
            feature_names=feature_names,
            condition_labels=condition_labels,
            condition_young_value=young_value,
            condition_old_value=old_value,
        )

        silhouette_display = (
            f"{metrics['silhouette_score']:.3f}"
            if not np.isnan(metrics["silhouette_score"])
            else "N/A (< 2 clusters)"
        )
        print("  Metrics:")
        print(f"    n_clusters      = {metrics['n_clusters']}")
        print(f"    noise_ratio     = {metrics['noise_ratio']:.1%}")
        print(f"    silhouette      = {silhouette_display}")

        bag_result: Dict = {
            "umap_coordinates": umap_coordinates,
            "feature_matrix": feature_matrix,
            "condition_labels": condition_labels,
            "subject_ids": subject_ids,
            "feature_names": feature_names,
            "cluster_labels": cluster_labels,
            "profile": profile,
            "metrics": metrics,
            "condition_young_value": young_value,
            "condition_old_value": old_value,
        }

        all_results[bag_size] = bag_result
        anndata_object.uns[_BAG_CLUSTERING_KEY][bag_size] = bag_result

    # Print comparison summary.
    print("\n" + "=" * 60)
    print("SUMMARY — CLUSTERING QUALITY BY BAG SIZE")
    print("=" * 60)
    summary_rows = []
    for bag_size, result in all_results.items():
        m = result["metrics"]
        summary_rows.append({
            "bag_size": bag_size,
            "n_samples": len(result["cluster_labels"]),
            "n_clusters": m["n_clusters"],
            "noise_ratio": f"{m['noise_ratio']:.1%}",
            "silhouette": (
                f"{m['silhouette_score']:.3f}"
                if not np.isnan(m["silhouette_score"]) else "N/A"
            ),
        })
    summary_dataframe = pd.DataFrame(summary_rows).set_index("bag_size")
    print(summary_dataframe.to_string())

    print("\nResults stored in .uns['bag_clustering_results'][bag_size].")

    return all_results


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _sample_individual_mitos(
    data_matrix: np.ndarray,
    obs_dataframe: pd.DataFrame,
    subject_id_column: str,
    condition_column: str,
    n_per_subject: int,
    random_state: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Sample individual mitochondria per subject without aggregation.

    Used as the single-mito baseline in compute_umap_on_bags() when
    mitos_per_bag == 1.  Samples without replacement when enough mitos
    are available; otherwise samples with replacement (same behaviour
    as create_bags()).

    Per-subject seeds are derived from random_state + subject_index to match
    the deterministic seeding strategy in create_bags().

    Arguments:
        data_matrix:       2D float array (n_mitos, n_features).
        obs_dataframe:     DataFrame aligned with data_matrix rows (.obs).
        subject_id_column: .obs column identifying subjects.
        condition_column:  .obs column with condition labels.
        n_per_subject:     Number of individual mitos to sample per subject.
        random_state:      Master seed; per-subject seed = (random_state + index).

    Returns:
        Tuple of three arrays:
          feature_matrix   — float array (n_samples, n_features)
          condition_labels — str array (n_samples,)
          subject_ids      — str array (n_samples,)
    """
    unique_subjects = sorted(obs_dataframe[subject_id_column].unique())
    all_features: List[np.ndarray] = []
    all_labels: List[np.ndarray] = []
    all_subjects: List[str] = []

    for subject_index, subject_id in enumerate(unique_subjects):
        seed = (random_state + subject_index) % (2**31 - 1)
        rng = np.random.default_rng(seed)

        subject_mask = obs_dataframe[subject_id_column].values == subject_id
        subject_matrix = data_matrix[subject_mask]
        subject_conditions = obs_dataframe.loc[
            obs_dataframe[subject_id_column] == subject_id, condition_column
        ].values

        n_available = subject_matrix.shape[0]
        replace = n_available < n_per_subject
        sampled_indices = rng.choice(n_available, size=n_per_subject, replace=replace)

        all_features.append(subject_matrix[sampled_indices])
        all_labels.append(subject_conditions[sampled_indices])
        all_subjects.extend([str(subject_id)] * n_per_subject)

    return (
        np.vstack(all_features).astype(np.float32),
        np.concatenate(all_labels).astype(str),
        np.array(all_subjects, dtype=str),
    )


def _detect_condition_values(condition_labels: np.ndarray) -> Tuple[str, str]:
    """
    Detect the actual string values used for "young" and "old" in condition_labels.

    Case-insensitive matching: works whether the labels are "young"/"old",
    "Young"/"Old", or any other capitalisation variant (e.g. from
    TEM_CONDITION_TOKEN_TO_CANONICAL which produces "Young" / "Old").

    Falls back to sorted order when canonical names are not found, and emits
    a warning so the caller knows auto-detection was used.

    Arguments:
        condition_labels: 1D array of condition strings from .obs[condition_column].

    Returns:
        Tuple (young_value, old_value) — the exact strings as found in the data.
    """
    import warnings

    unique_values = {
        str(v) for v in condition_labels
        if v is not None and str(v) not in ("nan", "None")
    }
    young_value = next((v for v in unique_values if v.lower() == "young"), None)
    old_value = next((v for v in unique_values if v.lower() == "old"), None)

    if young_value is None or old_value is None:
        sorted_vals = sorted(unique_values)
        young_value = sorted_vals[0] if len(sorted_vals) >= 1 else "young"
        old_value = sorted_vals[1] if len(sorted_vals) >= 2 else "old"
        warnings.warn(
            f"Could not detect 'young'/'old' labels in condition_labels. "
            f"Found: {unique_values}. "
            f"Using '{young_value}' as young and '{old_value}' as old. "
            "Pass condition_young_value and condition_old_value explicitly to "
            "profile_clusters() to override."
        )

    return young_value, old_value
