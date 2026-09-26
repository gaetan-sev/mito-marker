"""
clustermap_plot.py

Feature x group clustermap: hierarchical clustering heatmap of morphological
features (or channels), aggregated by group mean, with optional dendrogram
node rotation for either axis.

Each row is a feature. Each column is a group (any .obs column — species,
condition, treatment, timepoint, ...). Cell color shows the group's mean
value for that feature.

How that mean is computed is controlled by nest_aggregate_by. By default it
POOLS every observation, so a subject that contributed 1,612 mitochondria
outweighs one that contributed 268. Passing nest_aggregate_by switches to the
nested ("one vote per subject") mean that ADR-011 §3 makes the rule for any
comparison with unequal subject/event counts.

What is plotted is, by default, the real un-rescaled group mean, so heatmap
color intensity reflects each feature's actual effect size (how strongly it
separates groups). An optional row-wise z-score across groups
(standardize_group_means=True) is available for the opposite goal — giving
every feature's row the same visual spread regardless of effect size, useful
mainly on raw/unnormalized features of very different scales, or to reveal
co-varying direction rather than magnitude. Either way, this aggregated
z-score (applied to the group-mean matrix, after aggregation) is distinct
from — and does not repeat — any per-observation normalization already
applied upstream via transform_and_normalize().

Both axes are hierarchically clustered by default (scipy linkage + seaborn
clustermap). The two children of any dendrogram node can optionally be
rotated (swapped) to control left-right display order without changing
cluster membership — read the printed merge-order diagnostic to pick a
node's row index, then pass it via rotate_group_nodes / rotate_feature_nodes.

Typical usage:
    from mito_marker.analysis import plot_feature_clustermap
    result = plot_feature_clustermap(tem_anndata, group_by="specie")
    # Read the printed merge-order diagnostic, then rotate a node if needed:
    result = plot_feature_clustermap(tem_anndata, group_by="specie", rotate_group_nodes=[3])
    # Give every subject one vote in its species profile (ADR-011 / ADR-013):
    result = plot_feature_clustermap(
        tem_anndata, group_by="specie", nest_aggregate_by="unique_subject_ID"
    )
"""

import warnings
from typing import Dict, List, Optional, Tuple, Union

import anndata
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, optimal_leaf_ordering
from scipy.spatial.distance import pdist
from scipy.stats import zscore

from mito_marker.analysis._aggregation import (
    _compute_group_mean_matrix,
    _resolve_nest_aggregate_columns,
)
from mito_marker.analysis.colors import sort_values_for_legend

# .uns key where preprocessing/analysis configuration (active layer, active
# feature selection) is stored. Same key used by pca_plot.py, radar_plot.py,
# umap_plot.py.
_ANALYSIS_CONFIG_KEY = "analysis_config"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def plot_feature_clustermap(
    anndata_object: anndata.AnnData,
    group_by: Union[str, List[str]],
    group_order: Optional[List[str]] = None,
    feature_order: Optional[List[str]] = None,
    nest_aggregate_by: Optional[Union[str, List[str]]] = None,
    standardize_group_means: bool = False,
    cluster_method: str = "average",
    cluster_metric: str = "correlation",
    show_dendrograms: bool = True,
    cluster_features: bool = True,
    rotate_group_nodes: Optional[List[int]] = None,
    rotate_feature_nodes: Optional[List[int]] = None,
    show_merge_diagnostic: bool = True,
    title: str = "",
    figsize: Tuple[float, float] = (7.0, 14.0),
    colormap: str = "RdBu_r",
    center_colormap_at_zero: bool = True,
) -> Dict[str, object]:
    """
    Plot a hierarchically-clustered heatmap of features (rows) by group (columns).

    Aggregates feature values by group mean, optionally z-scores each feature's
    row across groups (standardize_group_means, for heatmap-color comparability
    only — this is separate from any per-observation normalization already
    applied upstream via transform_and_normalize()), then clusters both axes
    with scipy linkage + seaborn clustermap.

    Rotation (rotate_group_nodes / rotate_feature_nodes) swaps the two children
    of a chosen dendrogram node to control its left-right display order. This
    never changes which groups/features belong under that node — only how they
    are drawn. Node indices are the 0-indexed linkage row numbers printed in the
    merge-order diagnostic (see show_merge_diagnostic): run once without
    rotation, read the diagnostic, then re-run with the desired row indices.

    Arguments:
        anndata_object: AnnData with .obs[group_by] and, optionally,
            .uns['analysis_config']['active_layer'] / ['active_selection'].
        group_by: Name of a single .obs column, or a list of column names to
            combine (joined with " / "), used to group observations before
            averaging.
        group_order: Optional explicit column (group) order. Groups listed here
            and present in the data come first, in the given order; any other
            present groups are appended afterwards (sorted). Only affects the
            initial leaf order fed into clustering (and therefore the merge-order
            row numbering) — final display order is still determined by the
            dendrogram. Defaults to a sorted order when omitted. No group is
            ever silently dropped for being unlisted.
        feature_order: Same as group_order, but for the row (feature) axis.
        nest_aggregate_by: Optional name of the .obs column identifying each
            individual/subject inside a group (e.g. "unique_subject_ID"), or
            a list of column names ordered from COARSEST to FINEST (e.g.
            ["specie", "unique_subject_ID"]) — the same argument, semantics
            and ordering convention as plot_radar(nest_aggregate_by=...).
            When None (default), each group's profile is the plain mean of
            every observation in it, so a subject that contributed 1,612
            mitochondria outweighs one that contributed 268 by 6-to-1. When
            set, the mean is computed at the finest level first and then
            re-averaged at each coarser level in turn, so every unit at every
            level counts exactly once regardless of how many rows or children
            it has. This is the nested-mean rule of ADR-004 / ADR-011 §3,
            extended to multiple levels by ADR-013 — use it for any
            multi-species or multi-cohort comparison with unequal
            subject/event counts. The returned dict records the mode used
            under "nest_aggregate_by", so the Methods section can declare it.
        standardize_group_means: When False (default), the group-mean matrix
            is plotted as-is — heatmap color intensity reflects each
            feature's real effect size (how strongly it separates groups),
            which is the natural choice when the input is already
            per-observation normalized (active_layer set, or raw features on
            comparable scales). When True, z-scores each feature's row across
            groups (mean 0, std 1) after aggregation — every feature's row
            gets the same visual spread regardless of its real effect size,
            which trades away effect-size information for a "same weight per
            feature" pattern view (useful mainly on raw, unnormalized features
            of very different scales, or to visually cluster features by
            co-varying direction rather than magnitude). This only ever
            affects group (column) clustering/color — feature (row)
            clustering is mathematically unaffected either way (correlation
            distance is invariant to independent per-row rescaling).
        cluster_method: Linkage method passed to scipy (default "average").
        cluster_metric: Distance metric passed to scipy (default "correlation").
        show_dendrograms: When True (default), both axes are hierarchically
            clustered and dendrograms are drawn. Feature (row) clustering is
            automatically skipped when fewer than 2 features are present.
            Group (column) clustering is automatically skipped when fewer
            than 3 groups are present — clustering exactly 2 groups only ever
            produces one trivial merge (an arbitrary left/right order, no
            real structure), so a 2-class comparison (e.g. "Young" vs "Old")
            is shown as a plain side-by-side heatmap instead, with features
            still clustered normally for a readable multi-feature comparison.
            In all skipped cases the natural/requested order is used instead,
            with a warning.
        cluster_features: When False, feature (row) clustering is disabled
            regardless of show_dendrograms or how many features are present —
            features are shown in feature_order (or sorted) with no dendrogram
            and no warning (this is a deliberate choice, not a data
            limitation). Group (column) clustering is unaffected. Use this to
            keep a fixed, meaningful feature order (e.g. a biological grouping
            passed via feature_order) instead of a statistically-derived one.
        rotate_group_nodes: Optional list of linkage row indices (as printed by
            the merge-order diagnostic) whose two children should be swapped in
            the column (group) dendrogram before plotting.
        rotate_feature_nodes: Same as rotate_group_nodes, but for the row
            (feature) dendrogram.
        show_merge_diagnostic: When True (default), prints each linkage row's
            two merged children and their distance, for both axes — this is
            how a target row index for rotate_group_nodes / rotate_feature_nodes
            is chosen.
        title: Plot title. Defaults to an auto-generated title naming group_by,
            the number of groups/features, and the active data source.
        figsize: Figure size in inches (width, height).
        colormap: Matplotlib colormap name for the heatmap.
        center_colormap_at_zero: When True (default), centers the colormap at 0.

    Returns:
        Dict with keys:
          "figure":                   matplotlib.figure.Figure
          "group_order":              List[str], final column order
          "feature_order":            List[str], final row order
          "group_linkage":            Optional[np.ndarray], rotated linkage
                                       actually used for columns (None if < 2 groups)
          "feature_linkage":          Optional[np.ndarray], rotated linkage
                                       actually used for rows (None if < 2 features)
          "standardized_group_means": pd.DataFrame, the feature x group matrix
                                       actually plotted (post-standardization,
                                       pre-dendrogram-reordering). Built from
                                       the nested means when nest_aggregate_by
                                       is set, from the pooled means otherwise.
          "nest_aggregate_by":        Optional[List[str]], the .obs columns
                                       used to nest the group means (None when
                                       pooled) — declare this in the Methods
                                       section, per ADR-011.

    Raises:
        ValueError: If group_by or nest_aggregate_by references a column
            absent from .obs, if a
            rotate_*_nodes index is out of range, or if rotate_*_nodes is
            requested on an axis where clustering is not being performed
            (show_dendrograms=False, or fewer than 2 items on that axis).
    """
    import seaborn as sns  # noqa: PLC0415

    print("=" * 60)
    print(f"FEATURE CLUSTERMAP — grouped by '{group_by}'")
    print("=" * 60)

    nest_aggregate_by_columns = _resolve_nest_aggregate_columns(
        anndata_object, nest_aggregate_by, caller_name="plot_feature_clustermap"
    )

    data_matrix, channel_names = _get_data_and_channels(anndata_object)
    group_labels = _get_group_labels(anndata_object, group_by)

    valid_mask = group_labels.notna().values
    n_excluded = int((~valid_mask).sum())
    nest_hierarchy_values = (
        anndata_object.obs[nest_aggregate_by_columns].values[valid_mask]
        if nest_aggregate_by_columns is not None
        else None
    )
    group_mean_dataframe = _compute_group_mean_matrix(
        data_matrix=data_matrix[valid_mask],
        channel_names=channel_names,
        group_label_values=group_labels.values[valid_mask],
        nest_hierarchy_values=nest_hierarchy_values,
        nest_aggregate_by_columns=nest_aggregate_by_columns,
        caller_name="plot_feature_clustermap",
    )

    resolved_group_order = _resolve_display_order(
        requested_order=group_order,
        present_values=group_mean_dataframe.index.tolist(),
    )
    resolved_feature_order = _resolve_display_order(
        requested_order=feature_order,
        present_values=group_mean_dataframe.columns.tolist(),
    )

    # Rows = features, columns = groups (transpose after reordering).
    feature_by_group_dataframe = group_mean_dataframe.loc[resolved_group_order, resolved_feature_order].T

    n_groups = feature_by_group_dataframe.shape[1]
    n_features = feature_by_group_dataframe.shape[0]

    print(
        f"Observations used: {int(valid_mask.sum())} ({n_excluded} excluded — missing '{group_by}') "
        f"| Groups: {n_groups} | Features: {n_features}"
    )
    print(f"Group order (before clustering): {resolved_group_order}")
    _print_matrix_summary("Group-mean matrix (before standardization)", feature_by_group_dataframe.values)

    standardized_dataframe = _standardize_group_means(
        feature_by_group_dataframe, standardize_group_means
    )

    print(f"Clustering parameters: method='{cluster_method}', metric='{cluster_metric}'")

    can_cluster_groups = _can_cluster_axis(
        "group", n_groups, n_features, show_dendrograms, cluster_metric, minimum_leaf_count=3
    )
    can_cluster_features = _can_cluster_axis(
        "feature", n_features, n_groups, show_dendrograms and cluster_features, cluster_metric
    )
    _validate_rotation_request("group", rotate_group_nodes, can_cluster_groups)
    _validate_rotation_request("feature", rotate_feature_nodes, can_cluster_features)

    group_linkage = _compute_rotated_linkage(
        standardized_dataframe.T.values, cluster_method, cluster_metric, rotate_group_nodes
    ) if can_cluster_groups else None
    feature_linkage = _compute_rotated_linkage(
        standardized_dataframe.values, cluster_method, cluster_metric, rotate_feature_nodes
    ) if can_cluster_features else None

    if show_merge_diagnostic:
        _print_merge_diagnostic("GROUP", group_linkage, standardized_dataframe.columns.tolist())
        _print_merge_diagnostic("FEATURE", feature_linkage, standardized_dataframe.index.tolist())

    clustermap_kwargs: dict = dict(
        col_linkage=group_linkage,
        row_linkage=feature_linkage,
        row_cluster=can_cluster_features,
        col_cluster=can_cluster_groups,
        method=cluster_method,
        metric=cluster_metric,
        cmap=colormap,
        center=0 if center_colormap_at_zero else None,
        figsize=figsize,
        yticklabels=True,
        xticklabels=True,
        linewidths=0.3,
        linecolor="lightgray",
    )
    clustermap = sns.clustermap(standardized_dataframe, **clustermap_kwargs)

    final_group_order = (
        [standardized_dataframe.columns[i] for i in clustermap.dendrogram_col.reordered_ind]
        if can_cluster_groups
        else resolved_group_order
    )
    final_feature_order = (
        [standardized_dataframe.index[i] for i in clustermap.dendrogram_row.reordered_ind]
        if can_cluster_features
        else resolved_feature_order
    )

    clustermap.fig.suptitle(
        title or (
            f"Morphological features — mean"
            f"{' z-score' if standardize_group_means else ''} by {group_by}"
        ),
        y=1.02,
        fontsize=12,
    )

    rotations_summary = f"groups: {rotate_group_nodes or 'none'} | features: {rotate_feature_nodes or 'none'}"
    print("-" * 60)
    print(f"Rotations applied — {rotations_summary}")
    print("=" * 60)
    print("Group order after hierarchical clustering:")
    for rank, group_value in enumerate(final_group_order, 1):
        print(f"  {rank}. {group_value}")
    print("\nFeature order after hierarchical clustering:")
    for rank, feature_value in enumerate(final_feature_order, 1):
        print(f"  {rank:2d}. {feature_value}")
    print("=" * 60)

    return {
        "figure": clustermap.fig,
        "group_order": final_group_order,
        "feature_order": final_feature_order,
        "group_linkage": group_linkage,
        "feature_linkage": feature_linkage,
        "standardized_group_means": standardized_dataframe,
        "nest_aggregate_by": nest_aggregate_by_columns,
    }


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _get_analytical_mask_for_channels(
    anndata_object: anndata.AnnData,
    channel_names: List[str],
) -> np.ndarray:
    """
    Return a boolean mask (length = len(channel_names)) that is True for
    analytically usable channels.

    Reads .var['is_non_analytical'] when the column is present, selecting only
    the rows corresponding to channel_names. Falls back to
    SFC_NON_ANALYTICAL_CHANNELS / TEM_NON_ANALYTICAL_FEATURES for AnnData
    objects that pre-date the introduction of this column.

    Arguments:
        anndata_object: The AnnData whose .var is inspected.
        channel_names: Ordered list of channel names to mask.

    Returns:
        Boolean numpy array of length len(channel_names).
    """
    if "is_non_analytical" in anndata_object.var.columns:
        return ~anndata_object.var.loc[channel_names, "is_non_analytical"].values.astype(bool)

    from mito_marker.controlled_vocabulary import (
        SFC_NON_ANALYTICAL_CHANNELS,
        TEM_NON_ANALYTICAL_FEATURES,
    )
    non_analytical_set = set(SFC_NON_ANALYTICAL_CHANNELS) | set(TEM_NON_ANALYTICAL_FEATURES)
    return np.array([ch not in non_analytical_set for ch in channel_names], dtype=bool)


def _get_data_and_channels(
    anndata_object: anndata.AnnData,
    caller_name: str = "plot_feature_clustermap",
) -> Tuple[np.ndarray, List[str]]:
    """
    Return the data matrix and channel name list used to build the clustermap.

    Respects the active layer (falls back to raw .X) and always excludes
    non-analytical channels — the same contract used by compute_pca(),
    compute_umap() and plot_radar().

    Arguments:
        anndata_object: AnnData with optional .uns['analysis_config'].
        caller_name: Public function name shown in the printed QC lines, so a
            figure built by plot_phylo_tanglegram() does not report itself as
            plot_feature_clustermap().

    Returns:
        Tuple of (data_matrix, channel_names).
    """
    analysis_config = anndata_object.uns.get(_ANALYSIS_CONFIG_KEY, {})
    active_layer = analysis_config.get("active_layer")

    if active_layer is not None and active_layer in anndata_object.layers:
        full_matrix = anndata_object.layers[active_layer]
        print(f"[{caller_name}] Using layer: '{active_layer}'")
    else:
        full_matrix = anndata_object.X
        print(f"[{caller_name}] Using raw .X")

    if hasattr(full_matrix, "toarray"):
        full_matrix = full_matrix.toarray()
    full_matrix = np.asarray(full_matrix, dtype=np.float64)

    channel_names = anndata_object.var_names.tolist()

    analytical_mask = _get_analytical_mask_for_channels(anndata_object, channel_names)
    if not analytical_mask.all():
        excluded = [ch for ch, keep in zip(channel_names, analytical_mask) if not keep]
        full_matrix = full_matrix[:, analytical_mask]
        channel_names = [ch for ch, keep in zip(channel_names, analytical_mask) if keep]
        print(f"[{caller_name}] Non-analytical channels excluded: {excluded}")

    return full_matrix, channel_names


def _get_group_labels(
    anndata_object: anndata.AnnData,
    group_by: Union[str, List[str]],
) -> pd.Series:
    """
    Extract and combine grouping columns from .obs into a single label Series.

    When group_by is a string, the corresponding column is returned as-is.
    When group_by is a list, the values from each column are concatenated with
    " / " to form a combined label.

    Arguments:
        anndata_object: AnnData whose .obs will be queried.
        group_by: Name of a single .obs column, or a list of column names.

    Returns:
        A pandas Series indexed like anndata_object.obs with one label per row.

    Raises:
        ValueError: If any specified column is not present in .obs.
    """
    columns = [group_by] if isinstance(group_by, str) else group_by

    for column in columns:
        if column not in anndata_object.obs.columns:
            raise ValueError(
                f"Column '{column}' not found in .obs. "
                f"Available columns: {sorted(anndata_object.obs.columns.tolist())}"
            )

    if len(columns) == 1:
        return anndata_object.obs[columns[0]]

    combined = anndata_object.obs[columns[0]].astype(str)
    for column in columns[1:]:
        combined = combined + " / " + anndata_object.obs[column].astype(str)
    combined.name = " / ".join(columns)
    return combined


def _resolve_display_order(
    requested_order: Optional[List[str]],
    present_values: List[str],
) -> List[str]:
    """
    Resolve the initial (pre-clustering) leaf order for one axis.

    Values listed in requested_order and present in the data come first, in
    the given order. Any other present values are appended afterwards, sorted
    via sort_values_for_legend(). No present value is ever dropped for being
    unlisted. Defaults to a fully sorted order when requested_order is None.

    Arguments:
        requested_order: Optional explicit order (e.g. a caller-supplied
            phylogenetic order). May include values absent from the data.
        present_values: Values actually present in the aggregated data.

    Returns:
        Ordered list containing exactly the values in present_values.
    """
    if requested_order is None:
        return sort_values_for_legend(present_values)
    listed = [value for value in requested_order if value in present_values]
    unlisted = sort_values_for_legend(value for value in present_values if value not in requested_order)
    return listed + unlisted


def _standardize_group_means(
    feature_by_group_dataframe: pd.DataFrame,
    standardize_group_means: bool,
    caller_name: str = "plot_feature_clustermap",
) -> pd.DataFrame:
    """
    Z-score each feature's row across groups, for heatmap-color comparability.

    This is distinct from per-observation normalization applied upstream via
    transform_and_normalize() — it operates on the already-aggregated
    feature x group mean matrix, after grouping.

    Arguments:
        feature_by_group_dataframe: Feature x group mean matrix.
        standardize_group_means: When False, or when fewer than 2 groups are
            present (z-score of a single value is undefined), the matrix is
            returned unchanged.
        caller_name: Public function name shown in the warning message.

    Returns:
        Standardized (or original) DataFrame, same shape and labels as input.
    """
    if not standardize_group_means:
        return feature_by_group_dataframe

    if feature_by_group_dataframe.shape[1] < 2:
        warnings.warn(
            f"[{caller_name}] standardize_group_means requires >= 2 "
            "groups — skipped (would produce NaN)."
        )
        return feature_by_group_dataframe

    standardized_values = zscore(feature_by_group_dataframe.values, axis=1)
    standardized_dataframe = pd.DataFrame(
        standardized_values,
        index=feature_by_group_dataframe.index,
        columns=feature_by_group_dataframe.columns,
    )
    print("standardize_group_means=True — z-scoring each feature's group means (row-wise, across groups)")
    _print_matrix_summary("Standardized matrix (after z-score)", standardized_dataframe.values)
    return standardized_dataframe


def _print_matrix_summary(label: str, matrix: np.ndarray) -> None:
    """
    Print a one-line QC summary (min/max/mean, NaN/Inf counts) for a matrix.

    Arguments:
        label: Description prefixed to the summary line.
        matrix: Numeric array to summarize.
    """
    print(f"=> {label}: min={np.nanmin(matrix):.4f}, max={np.nanmax(matrix):.4f}, mean={np.nanmean(matrix):.4f}")
    print(f"=> NaN count: {int(np.isnan(matrix).sum())} | Inf count: {int(np.isinf(matrix).sum())}")


def _can_cluster_axis(
    axis_label: str,
    axis_leaf_count: int,
    other_axis_dimension_count: int,
    show_dendrograms: bool,
    cluster_metric: str,
    minimum_leaf_count: int = 2,
) -> bool:
    """
    Decide whether hierarchical clustering can safely run on one axis.

    Clustering needs at least minimum_leaf_count leaves on the axis itself.
    Additionally, the "correlation" metric is mathematically undefined for a
    single-dimensional vector (each leaf's profile has length = the other
    axis's item count), so it also requires at least 2 items on the other
    axis — other metrics (e.g. "euclidean") do not have this restriction.

    Arguments:
        axis_label: "group" or "feature", used in the printed warning.
        axis_leaf_count: Number of items being clustered on this axis.
        other_axis_dimension_count: Number of items on the other axis (the
            dimensionality of each leaf's profile vector).
        show_dendrograms: Whether clustering was requested at all.
        cluster_metric: The distance metric that will be used.
        minimum_leaf_count: Minimum number of leaves required to cluster this
            axis (default 2, the hard mathematical minimum). Clustering
            exactly 2 leaves only ever produces one trivial merge — an
            arbitrary left/right order with no real branching structure — so
            the group axis is called with minimum_leaf_count=3 to skip that
            uninformative dendrogram and fall back to a plain side-by-side
            display, which is what a 2-class comparison (e.g. "Young" vs
            "Old") should look like.

    Returns:
        True if clustering this axis is safe to run.
    """
    if not show_dendrograms:
        return False
    if axis_leaf_count < minimum_leaf_count:
        warnings.warn(
            f"[plot_feature_clustermap] Fewer than {minimum_leaf_count} {axis_label}s "
            f"({axis_leaf_count}) — {axis_label} clustering skipped, requested order used."
        )
        return False
    if cluster_metric == "correlation" and other_axis_dimension_count < 2:
        warnings.warn(
            f"[plot_feature_clustermap] cluster_metric='correlation' requires >= 2 "
            f"items on the other axis to define a profile vector — {axis_label} "
            "clustering skipped, requested order used."
        )
        return False
    return True


def _apply_node_rotations(
    linkage_matrix: np.ndarray,
    rotate_node_rows: Optional[List[int]],
) -> np.ndarray:
    """
    Swap the two children of each requested linkage row (display order only).

    Row indices use the same 0-indexed numbering printed by
    _print_merge_diagnostic. Swapping a row's two children changes only the
    left-right display order of that node's subtree — it never changes which
    leaves belong under it.

    Arguments:
        linkage_matrix: scipy linkage matrix, shape (n_leaves - 1, 4).
        rotate_node_rows: Row indices to rotate. None or empty leaves the
            matrix unchanged.

    Returns:
        A rotated copy of linkage_matrix.

    Raises:
        ValueError: If a requested row index is out of range.
    """
    rotated_matrix = linkage_matrix.copy()
    n_merges = rotated_matrix.shape[0]
    for row_index in (rotate_node_rows or []):
        if not (0 <= row_index < n_merges):
            raise ValueError(
                f"rotate node row index {row_index} out of range — valid rows "
                f"are 0..{n_merges - 1} (see the merge-order diagnostic printed above)."
            )
        rotated_matrix[row_index, [0, 1]] = rotated_matrix[row_index, [1, 0]]
    return rotated_matrix


def _compute_rotated_linkage(
    matrix_for_clustering: np.ndarray,
    cluster_method: str,
    cluster_metric: str,
    rotate_node_rows: Optional[List[int]],
    optimize_leaf_order: bool = False,
) -> np.ndarray:
    """
    Compute a linkage matrix and apply any requested node rotations.

    Arguments:
        matrix_for_clustering: Rows are the leaves to cluster.
        cluster_method: scipy linkage method.
        cluster_metric: scipy linkage distance metric.
        rotate_node_rows: Row indices whose children should be swapped.
        optimize_leaf_order: When True, runs scipy's optimal_leaf_ordering()
            on the linkage before applying the manual rotations. Like a
            manual rotation, this only swaps the two children of nodes — it
            never changes which leaves belong to which cluster — so it makes
            the drawn leaf order tidier without altering the clustering
            result. Default False keeps the historical behaviour.

    Returns:
        The (possibly reordered and rotated) linkage matrix.
    """
    linkage_matrix = linkage(matrix_for_clustering, method=cluster_method, metric=cluster_metric)
    if optimize_leaf_order:
        linkage_matrix = optimal_leaf_ordering(
            linkage_matrix, pdist(matrix_for_clustering, metric=cluster_metric)
        )
    return _apply_node_rotations(linkage_matrix, rotate_node_rows)


def _validate_rotation_request(
    axis_label: str,
    rotate_node_rows: Optional[List[int]],
    can_cluster: bool,
) -> None:
    """
    Raise a clear error when rotation is requested on an axis not being clustered.

    Arguments:
        axis_label: "group" or "feature", used in the error message.
        rotate_node_rows: The rotation request for this axis.
        can_cluster: Whether this axis will actually be clustered.

    Raises:
        ValueError: If rotation is requested but this axis has no dendrogram
            (show_dendrograms=False, or fewer than 2 items on this axis).
    """
    if rotate_node_rows and not can_cluster:
        raise ValueError(
            f"rotate_{axis_label}_nodes was set but the {axis_label} axis is not "
            "being clustered (show_dendrograms=False, or fewer than 2 items on "
            "this axis). Remove the rotation request or enable clustering."
        )


def _print_merge_diagnostic(
    axis_name: str,
    linkage_matrix: Optional[np.ndarray],
    leaf_labels: List[str],
) -> None:
    """
    Print each linkage row's two merged children and their distance.

    This is how a target row index for rotate_group_nodes / rotate_feature_nodes
    is chosen: read the printed rows, find the merge to flip, pass its row
    number in the rotate_*_nodes list.

    Arguments:
        axis_name: "GROUP" or "FEATURE", used in the printed header.
        linkage_matrix: scipy linkage matrix, or None if this axis was not
            clustered (nothing is printed in that case).
        leaf_labels: Leaf labels in the same order used to build the linkage
            matrix (i.e. the pre-clustering axis order).
    """
    if linkage_matrix is None:
        return

    print("-" * 60)
    print(
        f"{axis_name} MERGE ORDER — use these row indices in "
        f"rotate_{axis_name.lower()}_nodes"
    )
    node_labels = list(leaf_labels)
    for row_index, row in enumerate(linkage_matrix):
        left, right = int(row[0]), int(row[1])
        node_labels.append(f"({node_labels[left]} + {node_labels[right]})")
        print(f"  Row {row_index}: [{node_labels[left]}]  +  [{node_labels[right]}]  dist={row[2]:.4f}")
