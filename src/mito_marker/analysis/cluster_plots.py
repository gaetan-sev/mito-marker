"""
cluster_plots.py

Visualizations and summaries for cluster models (analysis/cluster_model.py).

  - plot_cluster_model_selection()  silhouette / BIC / inertia against k.
  - plot_cluster_radar()            mean profile of each cluster (plot_radar()).
  - plot_cluster_embedding()        PCA or UMAP colored by cluster (or any .obs column).
  - compute_cluster_proportions()   % of mitochondria per cluster, three ways.
  - plot_cluster_proportions()      stacked bars + per-subject points.
  - compare_cluster_proportions()   per-cluster test on PER-SUBJECT proportions.

Three percentages, three denominators (compute_cluster_proportions):
  pct_of_subset            n in cluster / n mitochondria of the dataset given.
  pct_of_total_population  n in cluster / n mitochondria of the population the
                           subset was drawn from (before the first feature-value
                           filter, read from the subset history).
  per_subject_mean_pct     % computed inside each subject, then averaged over
                           subjects: each subject counts once (ADR-004).

Statistics are always run on per-subject proportions: the true sample size
is the number of subjects, not of mitochondria (ADR-008).

Typical usage:
    proportions = compute_cluster_proportions({"Young": young_small, "Old": old_small}, model)
    plot_cluster_proportions(proportions, model, relative_to="subset")
    compare_cluster_proportions({"Young": young_small, "Old": old_small}, model)
"""

from math import comb
from typing import Any, Dict, List, Optional

import anndata
import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import false_discovery_control, kruskal, mannwhitneyu

from mito_marker.analysis.cluster_model import MitoClusterModel
from mito_marker.analysis.feature_subset import get_subset_history, resolve_subject_column
from mito_marker.analysis.group_comparison import (
    _eta_squared,
    _label_effect_size,
    _rank_biserial_correlation,
)
from mito_marker.analysis.radar_plot import plot_radar
from mito_marker.analysis.selection import _ANALYSIS_CONFIG_KEY

_UMAP_PARAMS_KEY = "umap_params"
_ALLOWED_RELATIVE_TO = ("subset", "total_population")


# ---------------------------------------------------------------------------
# Model selection
# ---------------------------------------------------------------------------


def plot_cluster_model_selection(
    cluster_model: MitoClusterModel,
    title: str = "",
) -> matplotlib.figure.Figure:
    """
    Plot the scores used to choose k, with the retained k marked.

    KMeans: silhouette (higher is better) and inertia (elbow curve).
    GMM: BIC (lower is better) and silhouette (for information).
    Silhouette and BIC often disagree: show this figure, not only the k retained.

    Arguments:
        cluster_model: Fitted model.
        title: Optional title.

    Returns:
        The matplotlib Figure (also displayed).
    """
    table = cluster_model.model_selection_table
    second_metric = "inertia" if cluster_model.algorithm == "kmeans" else "bic"
    chosen_k = int(table.loc[table["chosen"], "n_clusters"].iloc[0])

    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    for axis, metric, better in (
        (axes[0], "silhouette", "higher is better"),
        (axes[1], second_metric, "lower is better" if _metric_is_bic(second_metric) else "look for the elbow"),
    ):
        axis.plot(table["n_clusters"], table[metric], marker="o", color="#4a4a4a")
        chosen_value = table.loc[table["n_clusters"] == chosen_k, metric].iloc[0]
        axis.scatter([chosen_k], [chosen_value], s=120, color="#d62728", zorder=5, label=f"k = {chosen_k} retained")
        axis.set_xlabel("number of clusters k")
        axis.set_ylabel(metric)
        axis.set_title(f"{metric} ({better})", fontsize=10)
        axis.set_xticks(table["n_clusters"])
        axis.grid(True, alpha=0.3)
        axis.legend(fontsize=8)
    figure.suptitle(title or f"Choice of k — model '{cluster_model.model_name}' ({cluster_model.algorithm}, "
                    f"{cluster_model.clustering_space} space)", fontsize=11)
    figure.tight_layout()
    plt.show()
    return figure


def _metric_is_bic(metric_name: str) -> bool:
    """True when the metric is the BIC (lower is better)."""
    return metric_name == "bic"


# ---------------------------------------------------------------------------
# Radar and embeddings
# ---------------------------------------------------------------------------


def plot_cluster_radar(
    anndata_object: anndata.AnnData,
    cluster_model: MitoClusterModel,
    nest_aggregate_by: Optional[str] = None,
    title: str = "",
    **radar_keyword_arguments: Any,
) -> None:
    """
    Radar plot of the mean profile of each cluster.

    Calls plot_radar() with group_by = the model's cluster column and
    channels = the model's features, on the model's reference normalized layer
    (switched on temporarily, restored afterwards). Profiles of a fit dataset
    and of a predict dataset are therefore drawn on the same scale.

    Arguments:
        anndata_object: Dataset annotated by fit_cluster_model() or predict_cluster_model().
        cluster_model: The model.
        nest_aggregate_by: Subject column for a nested mean (one vote per
                           subject, ADR-011), e.g. "unique_subject_ID".
        title: Optional title.
        **radar_keyword_arguments: Passed through to plot_radar().

    Returns:
        None. Figures are displayed by plot_radar().
    """
    _require_cluster_column(anndata_object, cluster_model)
    analysis_config = dict(anndata_object.uns.get(_ANALYSIS_CONFIG_KEY, {}))
    previous_layer = analysis_config.get("active_layer")
    previous_selection = analysis_config.get("active_selection")
    if cluster_model.layer_name and cluster_model.layer_name in anndata_object.layers:
        analysis_config["active_layer"] = cluster_model.layer_name
    else:
        print("[plot_cluster_radar] Model chain starts from raw .X — profiles drawn in raw units.")
        analysis_config["active_layer"] = None
    # The model's own feature list defines the spokes, not any active selection.
    analysis_config["active_selection"] = None
    anndata_object.uns[_ANALYSIS_CONFIG_KEY] = analysis_config
    try:
        plot_radar(
            anndata_object,
            group_by=cluster_model.obs_column,
            channels=list(cluster_model.feature_names),
            nest_aggregate_by=nest_aggregate_by,
            title=title or f"Cluster profiles — model '{cluster_model.model_name}'",
            **radar_keyword_arguments,
        )
    finally:
        analysis_config = dict(anndata_object.uns[_ANALYSIS_CONFIG_KEY])
        analysis_config["active_layer"] = previous_layer
        analysis_config["active_selection"] = previous_selection
        anndata_object.uns[_ANALYSIS_CONFIG_KEY] = analysis_config


def plot_cluster_embedding(
    anndata_object: anndata.AnnData,
    cluster_model: MitoClusterModel,
    group_by: Optional[str] = None,
    **plot_keyword_arguments: Any,
) -> None:
    """
    PCA or UMAP scatter of a dataset in the model's frozen space.

    "pca" and "scaled" spaces: plot_pca_scatter() on .obsm['X_pca'] (the
    reference PCA). "umap" space: plot_umap() on the model's UMAP coordinates.
    Because every dataset is in the same frozen space, a fit dataset and a
    predict dataset can be compared axis for axis.

    Arguments:
        anndata_object: Dataset annotated by fit or predict.
        cluster_model: The model.
        group_by: .obs column used for colors; defaults to the cluster column.
                  Pass e.g. "condition" to see the groups in the same space.
        **plot_keyword_arguments: Passed through to plot_pca_scatter() / plot_umap().

    Returns:
        None. The figure is displayed by the underlying function.
    """
    from mito_marker.analysis.pca_plot import plot_pca_scatter  # noqa: PLC0415
    from mito_marker.analysis.umap_plot import plot_umap  # noqa: PLC0415

    _require_cluster_column(anndata_object, cluster_model)
    color_column = group_by or cluster_model.obs_column

    if cluster_model.clustering_space == "umap":
        analysis_config = dict(anndata_object.uns.get(_ANALYSIS_CONFIG_KEY, {}))
        umap_params = dict(analysis_config.get(_UMAP_PARAMS_KEY, {}))
        previous_key = umap_params.get("active_key")
        umap_params["active_key"] = cluster_model.umap_parameters["obsm_key"]
        analysis_config[_UMAP_PARAMS_KEY] = umap_params
        anndata_object.uns[_ANALYSIS_CONFIG_KEY] = analysis_config
        try:
            plot_umap(anndata_object, group_by=color_column, **plot_keyword_arguments)
        finally:
            umap_params["active_key"] = previous_key
            analysis_config[_UMAP_PARAMS_KEY] = umap_params
            anndata_object.uns[_ANALYSIS_CONFIG_KEY] = analysis_config
        return

    if "X_pca" not in anndata_object.obsm:
        print(
            "[plot_cluster_embedding] No .obsm['X_pca'] on this dataset — run compute_pca() "
            "on the reference dataset before subsetting to get a shared PCA view."
        )
        return
    plot_pca_scatter(anndata_object, group_by=color_column, **plot_keyword_arguments)


# ---------------------------------------------------------------------------
# Proportions
# ---------------------------------------------------------------------------


def compute_cluster_proportions(
    anndata_objects: Dict[str, anndata.AnnData],
    cluster_model: MitoClusterModel,
    subject_column: Optional[str] = None,
    total_population_sizes: Optional[Dict[str, int]] = None,
) -> pd.DataFrame:
    """
    Percentage of mitochondria in each cluster, for one or several datasets.

    Three percentages are returned, each with its own denominator:
      pct_of_subset: n in the cluster / n mitochondria of that dataset.
      pct_of_total_population: n in the cluster / n mitochondria of the
          population the dataset was drawn from, i.e. before its FIRST
          feature-value filter (read from the subset history). Example: 25%
          smallest Young mitochondria -> denominator = all Young mitochondria.
          When an .obs filter was applied AFTER that feature filter, the
          denominator is restricted to the subjects still present.
      per_subject_mean_pct (± per_subject_sd_pct): the % inside each subject
          of the dataset, averaged over subjects — every subject counts once,
          however many mitochondria it contributes (ADR-004).

    Note: when a dataset was obtained with a per-subject fraction
    (within_group=subject, e.g. 25% of each subject), pct_of_total_population
    is exactly pct_of_subset × fraction and adds no information. It is
    informative for absolute-threshold subsets (e.g. every Old mitochondrion
    below the Young size cut).

    Arguments:
        anndata_objects: {dataset label: AnnData annotated by fit or predict}.
        cluster_model: The model.
        subject_column: Subject column; None picks unique_subject_ID, then subject_ID.
        total_population_sizes: {dataset label: n} to override the denominator
            of pct_of_total_population (only needed when the subset history is
            missing, e.g. a subset built by hand with anndata[mask]).

    Returns:
        Tidy DataFrame, one row per (dataset, cluster), with columns:
        dataset, cluster, n_mitochondria, n_subset, pct_of_subset,
        n_total_population, pct_of_total_population, n_subjects,
        per_subject_mean_pct, per_subject_sd_pct, per_subject_median_pct,
        per_subject_pct_values (tuple of the per-subject percentages).
    """
    rows: List[Dict[str, Any]] = []
    print("=" * 60)
    print(f"CLUSTER PROPORTIONS — model '{cluster_model.model_name}'")
    print("=" * 60)
    for dataset_label, anndata_object in anndata_objects.items():
        _require_cluster_column(anndata_object, cluster_model)
        dataset_subject_column = subject_column or resolve_subject_column(anndata_object)
        cluster_series = anndata_object.obs[cluster_model.obs_column].astype(str)
        n_subset = int(len(cluster_series))

        n_total_population, denominator_source = _total_population_size(
            anndata_object, dataset_label, dataset_subject_column, total_population_sizes
        )
        print(f"[{dataset_label}] {n_subset:,} mitochondria in the dataset; total population "
              f"= {n_total_population if n_total_population >= 0 else 'unknown'} ({denominator_source})")
        _print_fraction_note(anndata_object, dataset_label)

        per_subject_table = _per_subject_percentages(
            anndata_object, cluster_model, dataset_subject_column
        )
        for cluster_label in cluster_model.cluster_labels:
            n_in_cluster = int((cluster_series == cluster_label).sum())
            subject_values = (
                per_subject_table[cluster_label].to_numpy() if per_subject_table is not None else np.array([])
            )
            rows.append({
                "dataset": dataset_label,
                "cluster": cluster_label,
                "n_mitochondria": n_in_cluster,
                "n_subset": n_subset,
                "pct_of_subset": 100.0 * n_in_cluster / n_subset if n_subset else float("nan"),
                "n_total_population": n_total_population,
                "pct_of_total_population": (
                    100.0 * n_in_cluster / n_total_population if n_total_population > 0 else float("nan")
                ),
                "n_subjects": int(len(subject_values)),
                "per_subject_mean_pct": float(np.mean(subject_values)) if subject_values.size else float("nan"),
                "per_subject_sd_pct": float(np.std(subject_values, ddof=1)) if subject_values.size > 1 else float("nan"),
                "per_subject_median_pct": float(np.median(subject_values)) if subject_values.size else float("nan"),
                "per_subject_pct_values": tuple(np.round(subject_values, 4)),
            })

    proportions = pd.DataFrame(rows)
    display_columns = [
        "dataset", "cluster", "n_mitochondria", "pct_of_subset", "pct_of_total_population",
        "per_subject_mean_pct", "per_subject_sd_pct", "n_subjects",
    ]
    print(proportions[display_columns].round(2).to_string(index=False))
    sums = proportions.groupby("dataset")["pct_of_subset"].sum()
    print(f"QC: pct_of_subset sums per dataset (should be 100): {sums.round(6).to_dict()}")
    print("=" * 60)
    return proportions


def plot_cluster_proportions(
    proportions_dataframe: pd.DataFrame,
    cluster_model: Optional[MitoClusterModel] = None,
    relative_to: str = "subset",
    show_subject_points: bool = True,
    title: str = "",
) -> matplotlib.figure.Figure:
    """
    Two panels: stacked composition bars, and per-cluster bars with one point
    per subject.

    Left: one stacked bar per dataset. relative_to="subset" -> bars sum to
    100% of each dataset; relative_to="total_population" -> bars sum to the
    share of the whole population the subset represents (e.g. 25%).
    Right: per-subject mean % of each cluster (bar) with each subject's own
    % (points) — the variability a statistical test works with.

    Arguments:
        proportions_dataframe: Output of compute_cluster_proportions().
        cluster_model: The model (for cluster colors); None uses tab10.
        relative_to: "subset" or "total_population".
        show_subject_points: Draw one point per subject on the right panel.
        title: Optional title.

    Returns:
        The matplotlib Figure (also displayed).
    """
    if relative_to not in _ALLOWED_RELATIVE_TO:
        raise ValueError(f"relative_to must be one of {_ALLOWED_RELATIVE_TO}, got '{relative_to}'.")
    value_column = "pct_of_subset" if relative_to == "subset" else "pct_of_total_population"
    datasets = list(dict.fromkeys(proportions_dataframe["dataset"]))
    clusters = list(dict.fromkeys(proportions_dataframe["cluster"]))
    colors = _cluster_colors(clusters, cluster_model)

    figure, (stacked_axis, subject_axis) = plt.subplots(
        1, 2, figsize=(max(8, 2.2 * len(datasets) + 2 * len(clusters)), 5),
        gridspec_kw={"width_ratios": [max(1, len(datasets)), max(1.5, len(clusters))]},
    )

    bottoms = np.zeros(len(datasets))
    for cluster_label in clusters:
        heights = np.array([
            proportions_dataframe.loc[
                (proportions_dataframe["dataset"] == dataset) & (proportions_dataframe["cluster"] == cluster_label),
                value_column,
            ].sum()
            for dataset in datasets
        ])
        stacked_axis.bar(datasets, heights, bottom=bottoms, color=colors[cluster_label],
                         edgecolor="white", label=cluster_label)
        for position, (height, bottom) in enumerate(zip(heights, bottoms)):
            if height >= 4:
                stacked_axis.text(position, bottom + height / 2, f"{height:.0f}%", ha="center",
                                  va="center", fontsize=8, color="white")
        bottoms += heights
    stacked_axis.set_ylabel(
        "% of the dataset's mitochondria" if relative_to == "subset"
        else "% of the total population's mitochondria"
    )
    stacked_axis.set_title("Pooled composition", fontsize=10)
    stacked_axis.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    stacked_axis.spines[["top", "right"]].set_visible(False)

    bar_width = 0.8 / max(1, len(datasets))
    hatches = ["", "//", "..", "xx", "\\\\", "oo"]
    for dataset_index, dataset in enumerate(datasets):
        dataset_rows = proportions_dataframe[proportions_dataframe["dataset"] == dataset].set_index("cluster")
        positions = np.arange(len(clusters)) + (dataset_index - (len(datasets) - 1) / 2) * bar_width
        means = [dataset_rows.loc[cluster, "per_subject_mean_pct"] for cluster in clusters]
        subject_axis.bar(
            positions, means, width=bar_width * 0.95,
            color=[colors[cluster] for cluster in clusters], alpha=0.55,
            edgecolor="black", linewidth=0.6, hatch=hatches[dataset_index % len(hatches)],
            label=dataset,
        )
        if show_subject_points:
            rng = np.random.default_rng(dataset_index)
            for position, cluster in zip(positions, clusters):
                values = np.asarray(dataset_rows.loc[cluster, "per_subject_pct_values"], dtype=float)
                jitter = rng.uniform(-bar_width * 0.25, bar_width * 0.25, size=values.size)
                subject_axis.scatter(position + jitter, values, s=14, color="black", zorder=5)
    subject_axis.set_xticks(np.arange(len(clusters)))
    subject_axis.set_xticklabels(clusters)
    subject_axis.set_ylabel("% of mitochondria, per subject")
    subject_axis.set_title("Per subject (bar = mean, dot = one subject)", fontsize=10)
    subject_axis.legend(fontsize=8, title="dataset")
    subject_axis.spines[["top", "right"]].set_visible(False)

    model_name = f" — model '{cluster_model.model_name}'" if cluster_model is not None else ""
    figure.suptitle(title or f"Cluster proportions{model_name}", fontsize=11)
    figure.tight_layout()
    plt.show()
    return figure


def compare_cluster_proportions(
    anndata_objects: Dict[str, anndata.AnnData],
    cluster_model: MitoClusterModel,
    subject_column: Optional[str] = None,
) -> pd.DataFrame:
    """
    Test, cluster by cluster, whether the per-subject proportion differs between datasets.

    For each cluster:
      1. the % of that cluster is computed inside each subject of each dataset;
      2. the per-subject values are compared: Mann-Whitney U for two datasets
         (effect size: rank-biserial r, positive when the SECOND dataset has
         the higher proportion), Kruskal-Wallis for three or more (effect
         size: eta squared);
      3. p-values are adjusted for the number of clusters tested
         (Benjamini-Hochberg).
    n = number of subjects, never the number of mitochondria (ADR-008). With
    few subjects the smallest reachable p-value is large; it is printed.

    Arguments:
        anndata_objects: {dataset label: annotated AnnData}, in comparison order.
        cluster_model: The model.
        subject_column: Subject column; None picks unique_subject_ID, then subject_ID.

    Returns:
        DataFrame, one row per cluster: per-dataset median / Q1 / Q3 of the
        per-subject %, n_subjects per dataset, test, statistic, p_value,
        p_value_adjusted, effect_size, effect_size_metric, effect_size_label,
        minimum_reachable_p_value.
    """
    if len(anndata_objects) < 2:
        raise ValueError("compare_cluster_proportions() needs at least two datasets.")

    per_dataset_tables: Dict[str, pd.DataFrame] = {}
    for dataset_label, anndata_object in anndata_objects.items():
        _require_cluster_column(anndata_object, cluster_model)
        column_name = subject_column or resolve_subject_column(anndata_object)
        if column_name is None:
            raise ValueError(f"Dataset '{dataset_label}' has no subject column: per-subject tests are impossible.")
        per_dataset_tables[dataset_label] = _per_subject_percentages(anndata_object, cluster_model, column_name)

    dataset_labels = list(per_dataset_tables)
    n_subjects = {label: len(table) for label, table in per_dataset_tables.items()}
    minimum_p_value = _minimum_reachable_p_value(list(n_subjects.values()))

    rows = []
    for cluster_label in cluster_model.cluster_labels:
        groups = [per_dataset_tables[label][cluster_label].to_numpy() for label in dataset_labels]
        row: Dict[str, Any] = {"cluster": cluster_label}
        for label, values in zip(dataset_labels, groups):
            row[f"{label}_median_pct"] = float(np.median(values)) if values.size else float("nan")
            row[f"{label}_q1_pct"] = float(np.percentile(values, 25)) if values.size else float("nan")
            row[f"{label}_q3_pct"] = float(np.percentile(values, 75)) if values.size else float("nan")
            row[f"{label}_n_subjects"] = int(values.size)
        row.update(_test_groups(groups))
        row["minimum_reachable_p_value"] = minimum_p_value
        rows.append(row)

    comparison = pd.DataFrame(rows)
    valid = comparison["p_value"].notna()
    comparison["p_value_adjusted"] = np.nan
    if valid.any():
        comparison.loc[valid, "p_value_adjusted"] = false_discovery_control(
            comparison.loc[valid, "p_value"].to_numpy(), method="bh"
        )
    comparison["effect_size_label"] = [
        _label_effect_size(effect, metric)
        for effect, metric in zip(comparison["effect_size"], comparison["effect_size_metric"])
    ]

    print("=" * 60)
    print(f"COMPARE CLUSTER PROPORTIONS — model '{cluster_model.model_name}'")
    print("=" * 60)
    print(f"Datasets (in order): {dataset_labels}; subjects per dataset: {n_subjects}")
    print("n = number of SUBJECTS (per-subject %), not mitochondria (ADR-008).")
    print(f"Smallest p-value reachable with these group sizes: {minimum_p_value:.4g}")
    if len(dataset_labels) == 2:
        print(f"Effect size: rank-biserial r > 0 means '{dataset_labels[1]}' has the higher proportion.")
    print(comparison.round(4).to_string(index=False))
    print("=" * 60)
    return comparison


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _require_cluster_column(anndata_object: anndata.AnnData, cluster_model: MitoClusterModel) -> None:
    """Raise when the dataset was not annotated by this model."""
    if cluster_model.obs_column not in anndata_object.obs.columns:
        raise KeyError(
            f".obs['{cluster_model.obs_column}'] not found — run fit_cluster_model() or "
            "predict_cluster_model() on this dataset first."
        )


def _per_subject_percentages(
    anndata_object: anndata.AnnData,
    cluster_model: MitoClusterModel,
    subject_column: Optional[str],
) -> Optional[pd.DataFrame]:
    """
    % of each cluster inside each subject.

    Arguments:
        anndata_object: Annotated dataset.
        cluster_model: The model.
        subject_column: Subject column (None -> no per-subject table).

    Returns:
        DataFrame (subjects × clusters) of percentages, or None.
    """
    if subject_column is None or subject_column not in anndata_object.obs.columns:
        return None
    counts = pd.crosstab(
        anndata_object.obs[subject_column].astype(str),
        anndata_object.obs[cluster_model.obs_column].astype(str),
    ).reindex(columns=cluster_model.cluster_labels, fill_value=0)
    counts = counts[counts.sum(axis=1) > 0]
    return counts.div(counts.sum(axis=1), axis=0) * 100.0


def _total_population_size(
    anndata_object: anndata.AnnData,
    dataset_label: str,
    subject_column: Optional[str],
    total_population_sizes: Optional[Dict[str, int]],
) -> tuple:
    """
    Size of the population a dataset was drawn from.

    Arguments:
        anndata_object: Dataset.
        dataset_label: Its label.
        subject_column: Subject column.
        total_population_sizes: Optional overrides.

    Returns:
        Tuple (size or -1 when unknown, text explaining where it comes from).
    """
    if total_population_sizes and dataset_label in total_population_sizes:
        return int(total_population_sizes[dataset_label]), "given in total_population_sizes"

    history = get_subset_history(anndata_object)
    feature_positions = [index for index, entry in enumerate(history) if entry.get("subset_type") == "feature_values"]
    if not feature_positions:
        return int(anndata_object.n_obs), "no feature-value filter in the subset history: the dataset is its own population"
    first_entry = history[feature_positions[0]]
    obs_filter_after = any(
        entry.get("subset_type") == "obs_values" for entry in history[feature_positions[0] + 1:]
    )
    if not obs_filter_after:
        return int(first_entry["n_obs_before"]), (
            f"before the first feature filter ({first_entry['feature_name']} {first_entry['mode']})"
        )
    # An .obs filter came after the feature filter: count only the subjects
    # still present, taken from the per-subject counts recorded before it.
    counts_before = first_entry.get("n_obs_before_by_subject", {})
    if subject_column is None or not counts_before:
        return -1, "unknown (an .obs filter follows the feature filter and no per-subject counts exist)"
    present_subjects = set(anndata_object.obs[subject_column].astype(str))
    total = sum(int(count) for subject, count in counts_before.items() if subject in present_subjects)
    return total, "before the first feature filter, restricted to the subjects present"


def _print_fraction_note(anndata_object: anndata.AnnData, dataset_label: str) -> None:
    """Remind the reader when pct_of_total_population carries no new information."""
    for entry in get_subset_history(anndata_object):
        if entry.get("subset_type") == "feature_values" and entry.get("within_group"):
            print(
                f"  note [{dataset_label}]: subset taken as a fraction WITHIN each "
                f"'{entry['within_group']}' — pct_of_total_population = pct_of_subset × "
                f"{entry['fraction']} and adds no information."
            )
            return


def _test_groups(groups: List[np.ndarray]) -> Dict[str, Any]:
    """
    Mann-Whitney (2 groups) or Kruskal-Wallis (3+ groups) on per-subject values.

    Arguments:
        groups: One array of per-subject percentages per dataset.

    Returns:
        Dict with test, statistic, p_value, effect_size, effect_size_metric.
    """
    empty = {"test": "not run", "statistic": float("nan"), "p_value": float("nan"),
             "effect_size": float("nan"), "effect_size_metric": "rank_biserial_r"}
    if any(values.size < 2 for values in groups):
        return {**empty, "test": "not run (fewer than 2 subjects in a dataset)"}
    all_values = np.concatenate(groups)
    if np.allclose(all_values, all_values[0]):
        return {**empty, "test": "not run (identical values)"}
    if len(groups) == 2:
        result = mannwhitneyu(groups[0], groups[1], alternative="two-sided")
        return {
            "test": "Mann-Whitney U",
            "statistic": float(result.statistic),
            "p_value": float(result.pvalue),
            "effect_size": _rank_biserial_correlation(result.statistic, groups[0].size, groups[1].size),
            "effect_size_metric": "rank_biserial_r",
        }
    result = kruskal(*groups)
    return {
        "test": "Kruskal-Wallis H",
        "statistic": float(result.statistic),
        "p_value": float(result.pvalue),
        "effect_size": _eta_squared(result.statistic, int(all_values.size)),
        "effect_size_metric": "eta_squared",
    }


def _minimum_reachable_p_value(group_sizes: List[int]) -> float:
    """
    Smallest two-sided p-value an exact Mann-Whitney test can return for two
    groups of these sizes: 2 / C(n1 + n2, n1). With 3 subjects per group it is
    0.1 — "not significant" may then mean "not reachable".

    Arguments:
        group_sizes: Number of subjects per dataset.

    Returns:
        Minimum reachable p-value, or NaN for other than two groups.
    """
    if len(group_sizes) != 2 or min(group_sizes) < 1:
        return float("nan")
    first_size, second_size = group_sizes
    return float(min(1.0, 2.0 / comb(first_size + second_size, first_size)))


def _cluster_colors(clusters: List[str], cluster_model: Optional[MitoClusterModel]) -> Dict[str, str]:
    """Cluster colors from the model, or tab10 when no model is given."""
    if cluster_model is not None:
        return {cluster: cluster_model.cluster_colors.get(cluster, "#999999") for cluster in clusters}
    palette = plt.get_cmap("tab10").colors
    return {cluster: palette[index % len(palette)] for index, cluster in enumerate(clusters)}
