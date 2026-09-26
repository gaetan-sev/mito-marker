"""
_aggregation.py

Shared private aggregation helpers for the analysis stack.

Home of the nested ("one vote per subject") aggregation rule established in
ADR-004, generalized to any group profile in ADR-011, renamed in ADR-012 and
extended to an arbitrary number of hierarchy levels in ADR-013
(see docs/DECISIONS.md).

Plain-language statement of the problem these helpers solve: if every
mitochondrion from every subject is dumped into one pile and averaged, the
subject that happened to be imaged the most heavily dominates the result —
not because it is more biologically important, but purely because it
contributed more rows of data. The fix is to average within each subject
first, then average those per-subject profiles together, so every subject
counts exactly once.

This module is private (leading underscore). It exists so that
radar_plot.py, clustermap_plot.py and phylo_tanglegram.py all share one
implementation of that rule instead of three drifting copies.
"""

from typing import Callable, List, Optional, Union

import anndata
import numpy as np
import pandas as pd


def _nested_group_aggregate(
    matrix: np.ndarray,
    hierarchy_values: np.ndarray,
    aggregate_fn: Callable[..., np.ndarray],
    caller_name: str = "nested aggregate",
) -> np.ndarray:
    """
    Multi-level ("nested") aggregate: aggregate_fn is applied once at the
    finest hierarchy level, then again at each coarser level in turn, so
    every group at every level counts equally in the final profile
    regardless of how many rows/children it contributed.

    This is the shared implementation of the rule established for MHI in
    ADR-004 ("per-individual aggregation before group-level summary") —
    generalized in ADR-011 (docs/DECISIONS.md) to any group profile line,
    and to more than one hierarchy level in ADR-013. Plain-language example
    with one level: a subject with 1,600 mitochondria and a subject with 100
    mitochondria each get exactly one vote in the group average, instead of
    the 1,600-event subject outweighing the other 16-to-1. With two levels
    (e.g. species, then subject): a species with 12 sampled subjects and one
    with 3 each get exactly one vote too, once their own subjects have
    already been averaged down to a single per-species value.

    Arguments:
        matrix: Data matrix already restricted to the rows being aggregated
               (e.g. one group), shape (n_events, n_channels).
        hierarchy_values: Array of hierarchy identifiers, shape
                          (n_events,) or (n_events, n_levels), one row per
                          row of matrix (same order). When 2-D, columns are
                          ordered COARSEST to FINEST (e.g. species, then
                          subject ID) — the same convention as
                          compute_pca(weight_by=...) in pca_plot.py. Rows
                          with a missing (NaN/None) value at ANY level are
                          excluded.
        aggregate_fn: np.nanmean or np.nanmedian — applied at EVERY level
                     (once at the finest level, then once per coarser level).
        caller_name: Public function name used in the printed fallback
                     warning (e.g. "plot_radar"), so a beginner reading the
                     console knows which figure the warning belongs to.

    Returns:
        1-D array, shape (n_channels,) — the nested profile. Falls back to
        the plain pooled aggregate (with a printed warning) if no row in
        this slice has a fully non-missing hierarchy path.
    """
    if hierarchy_values.ndim == 1:
        hierarchy_values = hierarchy_values.reshape(-1, 1)

    valid_mask = ~pd.isnull(hierarchy_values).any(axis=1)
    valid_matrix = matrix[valid_mask]
    valid_hierarchy = hierarchy_values[valid_mask]

    if valid_hierarchy.shape[0] == 0:
        print(
            f"WARNING [{caller_name}]: nest_aggregate_by has no non-missing values for "
            "this group — falling back to a pooled (non-nested) aggregate."
        )
        # Fall back on the ORIGINAL (unfiltered) matrix — valid_matrix is
        # empty here since every hierarchy path was missing.
        return aggregate_fn(matrix, axis=0)

    # Finest level: one profile per unique value in the last hierarchy column.
    finest_values = valid_hierarchy[:, -1]
    unique_units = pd.unique(finest_values)
    profiles = np.stack([
        aggregate_fn(valid_matrix[finest_values == unit], axis=0)
        for unit in unique_units
    ])

    # A representative raw-row index per unit, used to look up that unit's
    # (constant) value at each coarser level as we walk up the hierarchy.
    representative_row = np.array([
        np.flatnonzero(finest_values == unit)[0] for unit in unique_units
    ])

    n_levels = valid_hierarchy.shape[1]
    for level in range(n_levels - 2, -1, -1):
        parent_values = valid_hierarchy[representative_row, level]
        unique_parents = pd.unique(parent_values)
        profiles = np.stack([
            aggregate_fn(profiles[parent_values == parent], axis=0)
            for parent in unique_parents
        ])
        representative_row = np.array([
            representative_row[np.flatnonzero(parent_values == parent)[0]]
            for parent in unique_parents
        ])

    return aggregate_fn(profiles, axis=0)


def _resolve_nest_aggregate_columns(
    anndata_object: anndata.AnnData,
    nest_aggregate_by: Optional[Union[str, List[str]]],
    caller_name: str,
) -> Optional[List[str]]:
    """
    Normalize a nest_aggregate_by argument to a validated list of .obs columns.

    Accepts either a single column name or a list ordered COARSEST to FINEST,
    exactly as documented for plot_radar(nest_aggregate_by=...) (ADR-013).

    Arguments:
        anndata_object: AnnData whose .obs must contain every named column.
        nest_aggregate_by: A single .obs column name, a list of column names
                           ordered coarsest to finest, or None.
        caller_name: Public function name used in the error message.

    Returns:
        The list of column names, or None when nest_aggregate_by is None.

    Raises:
        ValueError: If any named column is absent from .obs.
    """
    if nest_aggregate_by is None:
        return None

    columns = [nest_aggregate_by] if isinstance(nest_aggregate_by, str) else list(nest_aggregate_by)
    for column in columns:
        if column not in anndata_object.obs.columns:
            raise ValueError(
                f"[{caller_name}] nest_aggregate_by column '{column}' not found in .obs. "
                f"Available columns: {sorted(anndata_object.obs.columns.tolist())}"
            )
    return columns


def _compute_group_mean_matrix(
    data_matrix: np.ndarray,
    channel_names: List[str],
    group_label_values: np.ndarray,
    nest_hierarchy_values: Optional[np.ndarray],
    nest_aggregate_by_columns: Optional[List[str]],
    caller_name: str,
) -> pd.DataFrame:
    """
    Build the group x channel mean matrix, pooled or nested.

    When nest_hierarchy_values is None the matrix is the plain pooled mean of
    every row of each group — the historical behaviour. When it is provided,
    each group's row is replaced by the NESTED mean computed by
    _nested_group_aggregate(), so every unit at every hierarchy level counts
    exactly once (ADR-011 §3). The set of groups and their index order are
    identical in both modes.

    Arguments:
        data_matrix: Data matrix restricted to rows with a non-missing group
                     label, shape (n_valid_rows, n_channels).
        channel_names: Column names for data_matrix.
        group_label_values: One group label per row of data_matrix.
        nest_hierarchy_values: Hierarchy identifiers per row of data_matrix,
                               shape (n_valid_rows, n_levels), ordered
                               COARSEST to FINEST — or None to pool.
        nest_aggregate_by_columns: The .obs column names behind
                                   nest_hierarchy_values, used for QC output.
        caller_name: Public function name used in printed QC lines.

    Returns:
        DataFrame indexed by group value, one column per channel.
    """
    # The pooled matrix always defines the group set and the index order, so
    # switching aggregation mode never silently adds, drops or reorders a group.
    group_mean_dataframe = (
        pd.DataFrame(
            data_matrix,
            columns=channel_names,
            index=group_label_values,
        )
        .groupby(level=0)
        .mean()
    )

    if nest_hierarchy_values is None:
        print(f"[{caller_name}] nest_aggregate_by=None — group means POOL every observation.")
        return group_mean_dataframe

    hierarchy_display = " > ".join(nest_aggregate_by_columns or [])
    print(
        f"[{caller_name}] nest_aggregate_by={nest_aggregate_by_columns!r} — group means use a "
        f"NESTED mean ({hierarchy_display}, finest level averaged first, then each coarser "
        "level in turn) instead of pooling every observation. See ADR-004 / ADR-011 / ADR-013."
    )

    finest_column = (nest_aggregate_by_columns or ["unit"])[-1]
    nested_rows: List[np.ndarray] = []
    for group_value in group_mean_dataframe.index:
        group_mask = group_label_values == group_value
        n_observations = int(group_mask.sum())
        if n_observations == 0:
            # Empty group (e.g. an unused pandas category): keep the pooled row
            # rather than calling the aggregate on an empty slice.
            nested_rows.append(group_mean_dataframe.loc[group_value].values)
            continue

        group_hierarchy = nest_hierarchy_values[group_mask]
        nested_rows.append(
            _nested_group_aggregate(
                data_matrix[group_mask], group_hierarchy, np.nanmean, caller_name=caller_name
            )
        )

        unit_counts = [
            int(pd.unique(group_hierarchy[:, level]).shape[0])
            for level in range(group_hierarchy.shape[1])
        ]
        counts_display = " / ".join(
            f"{count} {column}(s)"
            for count, column in zip(unit_counts, nest_aggregate_by_columns or [finest_column])
        )
        print(
            f"    {group_value}: {counts_display} -> 1 profile "
            f"(from {n_observations:,} observations)"
        )

    return pd.DataFrame(
        np.vstack(nested_rows),
        index=group_mean_dataframe.index,
        columns=group_mean_dataframe.columns,
    )
