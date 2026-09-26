"""
stratification.py

Population stratification utilities: create new .obs columns from existing ones.

Three functions are provided:

bin_obs_column()
    Classifies events into equal-frequency bins (tertiles, quartiles, or any N)
    based on a numerical .obs column (e.g. age, glucose, insulin_level).
    Uses pandas.qcut so each bin contains approximately the same number of events.

split_obs_by_threshold()
    Splits events into exactly two groups based on a fixed numerical threshold:
    events strictly below the threshold vs. events at or above it. Useful for
    clinically meaningful cut-offs (e.g. fasting glucose ≥ 1.26 g/L = diabetes).

map_obs_values()
    Maps categorical values of an existing .obs column to user-defined group
    labels, merging multiple original values into a single label. Useful for
    collapsing species into biological categories (e.g. "Long life" = worm + fly
    + K fish) or any other custom grouping of categorical metadata.

All three functions store the result as a new .obs column and inject consistent
colors into .uns['color_palette'] so that plot_radar(), plot_umap(), and
plot_pca_scatter() work immediately via their group_by parameter.

Typical usage in a notebook:
    from mito_marker.analysis import (
        bin_obs_column, split_obs_by_threshold, map_obs_values, plot_radar
    )

    # Equal-frequency tertiles
    anndata_object = bin_obs_column(anndata_object, obs_column="age", n_bins=3)
    plot_radar(anndata_object, group_by="age_tertiles")

    # Fixed clinical threshold
    anndata_object = split_obs_by_threshold(
        anndata_object, obs_column="glucose_level", threshold=1.26
    )
    plot_radar(anndata_object, group_by="glucose_level_split")

    # Custom categorical grouping
    lifespan_column = map_obs_values(
        anndata_object,
        source_column="specie",
        value_map={
            "Long life": ["worm", "fly", "K fish"],
            "Short life": ["Z fish", "mouse", "human"],
        },
    )
    plot_radar(anndata_object, group_by=lifespan_column)
"""

import warnings
from typing import Dict, List, Optional

import anndata
import numpy as np
import pandas as pd

from mito_marker.analysis.colors import COLOR_PALETTE_KEY, _generate_auto_colors
from mito_marker.analysis.selection import _ANALYSIS_CONFIG_KEY


def bin_obs_column(
    anndata_object: anndata.AnnData,
    obs_column: str,
    n_bins: int = 3,
    new_column_name: Optional[str] = None,
    labels: Optional[List[str]] = None,
    palette: str = "Set2",
) -> anndata.AnnData:
    """
    Bin a continuous .obs column into equal-frequency quantile groups.

    Adds a new categorical .obs column whose values are readable bin labels
    that include the actual data range (e.g. "T1 – Low (< 35.2)"). Colors
    for the new labels are injected into .uns['color_palette'] so all
    existing plot functions recognize them automatically.

    Equal-frequency binning is used (pandas.qcut): each bin contains
    approximately the same number of events. NaN values in the source column
    remain NaN in the output column.

    Arguments:
        anndata_object: AnnData produced by the ingestion or earlier analysis
                        pipeline steps.
        obs_column: Name of an existing numerical .obs column to bin.
        n_bins: Number of bins. 3 = tertiles (default), 4 = quartiles.
                Must be >= 2.
        new_column_name: Name for the new .obs column. Auto-generated when
                         None: "{obs_column}_tertiles" (n=3),
                         "{obs_column}_quartiles" (n=4),
                         "{obs_column}_{n}bins" (any other n).
        labels: Custom string labels for the bins, in ascending order.
                Length must equal n_bins (or the actual number of bins when
                duplicate quantile edges are dropped). When None, labels are
                auto-generated with actual value ranges.
        palette: Matplotlib colormap name used to assign colors to bins
                 (default "Set2").

    Returns:
        The same AnnData object with the new .obs column and updated
        .uns['color_palette'] and .uns['analysis_config']['stratification'].

    Raises:
        KeyError: If obs_column is not found in .obs.
        TypeError: If obs_column is not a numerical dtype.
        ValueError: If n_bins < 2, or if labels length does not match the
                    actual number of bins created.
    """
    # --- Input validation ---

    if obs_column not in anndata_object.obs.columns:
        raise KeyError(
            f"Column '{obs_column}' not found in .obs. "
            f"Available columns: {list(anndata_object.obs.columns)}"
        )

    if not pd.api.types.is_numeric_dtype(anndata_object.obs[obs_column]):
        raise TypeError(
            f"Column '{obs_column}' is not numerical "
            f"(dtype: {anndata_object.obs[obs_column].dtype}). "
            "Only continuous numerical columns can be binned."
        )

    if n_bins < 2:
        raise ValueError(f"n_bins must be >= 2, got {n_bins}.")

    # --- Determine the output column name ---

    effective_new_column_name = _build_column_name(obs_column, n_bins, new_column_name)

    if effective_new_column_name in anndata_object.obs.columns:
        warnings.warn(
            f"Column '{effective_new_column_name}' already exists in .obs and will be overwritten.",
            UserWarning,
            stacklevel=2,
        )

    # --- First pass: get actual bin edges without labels ---

    numeric_series = anndata_object.obs[obs_column]
    nan_mask = numeric_series.isna()
    nan_count = int(nan_mask.sum())

    if nan_count == len(numeric_series):
        raise ValueError(
            f"Column '{obs_column}' contains only NaN values ({nan_count} / {len(numeric_series)} events). "
            "Cannot create any bins. "
            "This column was likely not populated — check that enrich_with_clinical_data() was called "
            "and that the CSV contains this variable. "
            f"Tip: print(anndata_object.obs['{obs_column}'].describe()) to inspect the column."
        )

    try:
        _, bin_edges = pd.qcut(
            numeric_series,
            q=n_bins,
            labels=False,
            retbins=True,
            duplicates="drop",
        )
    except ValueError as exc:
        raise ValueError(
            f"Could not bin '{obs_column}' into {n_bins} bins: {exc}. "
            "Try reducing n_bins or using a column with more distinct values."
        ) from exc

    actual_n_bins = len(bin_edges) - 1

    if actual_n_bins < n_bins:
        warnings.warn(
            f"Some quantile cut points are identical in '{obs_column}'. "
            f"Reduced from {n_bins} to {actual_n_bins} bins (duplicate edges dropped). "
            "This can happen when many events share the same value at a percentile boundary.",
            UserWarning,
            stacklevel=2,
        )

    # --- Resolve labels ---

    if labels is not None:
        if len(labels) != actual_n_bins:
            raise ValueError(
                f"labels has {len(labels)} elements but {actual_n_bins} bin(s) were created. "
                f"Provide exactly {actual_n_bins} label(s)."
            )
        final_labels = list(labels)
    else:
        final_labels = _build_auto_labels(bin_edges, actual_n_bins)

    # --- Second pass: actual binning with string labels ---

    binned_series, _ = pd.qcut(
        numeric_series,
        q=n_bins,
        labels=final_labels,
        retbins=True,
        duplicates="drop",
    )

    # Store as plain string dtype to avoid AnnData Categorical warnings.
    anndata_object.obs[effective_new_column_name] = binned_series.astype(str).where(
        ~nan_mask, other=np.nan
    )

    # --- Inject colors into .uns['color_palette'] ---

    _inject_bin_colors(anndata_object, final_labels, palette)

    # --- Store stratification metadata in .uns ---

    if _ANALYSIS_CONFIG_KEY not in anndata_object.uns:
        anndata_object.uns[_ANALYSIS_CONFIG_KEY] = {}

    anndata_object.uns[_ANALYSIS_CONFIG_KEY]["stratification"] = {
        "source_column": obs_column,
        "new_column": effective_new_column_name,
        "n_bins_requested": n_bins,
        "n_bins_created": actual_n_bins,
        "bin_edges": bin_edges.tolist(),
        "labels": final_labels,
        "palette": palette,
    }

    # --- QC report ---

    _print_qc_report(
        anndata_object=anndata_object,
        obs_column=obs_column,
        new_column_name=effective_new_column_name,
        final_labels=final_labels,
        bin_edges=bin_edges,
        nan_count=nan_count,
        actual_n_bins=actual_n_bins,
    )

    return anndata_object


def split_obs_by_threshold(
    anndata_object: anndata.AnnData,
    obs_column: str,
    threshold: float,
    new_column_name: Optional[str] = None,
    labels: Optional[List[str]] = None,
    palette: str = "Set2",
) -> anndata.AnnData:
    """
    Split a continuous .obs column into two groups using a fixed threshold.

    Events with a value strictly below the threshold are placed in the first
    group; events with a value equal to or above the threshold are placed in
    the second group. This is appropriate for clinically meaningful cut-offs
    (e.g. fasting glucose ≥ 1.26 g/L for diabetes diagnosis).

    Adds a new categorical .obs column and injects colors for the two groups
    into .uns['color_palette'] so all existing plot functions recognize them
    automatically via their group_by parameter.

    NaN values in the source column remain NaN in the output column.

    Arguments:
        anndata_object: AnnData produced by the ingestion or earlier analysis
                        pipeline steps.
        obs_column: Name of an existing numerical .obs column to split.
        threshold: The cut-off value. Events with obs_column < threshold go
                   into the first group; events with obs_column >= threshold
                   go into the second group.
        new_column_name: Name for the new .obs column. Auto-generated when
                         None: "{obs_column}_split".
        labels: List of exactly two strings [below_label, above_label].
                When None, auto-generated as:
                  ["Low (< {threshold:.1f})", "High (≥ {threshold:.1f})"]
        palette: Matplotlib colormap name used to assign colors to the two
                 groups (default "Set2").

    Returns:
        The same AnnData object with the new .obs column and updated
        .uns['color_palette'] and .uns['analysis_config']['threshold_split'].

    Raises:
        KeyError: If obs_column is not found in .obs.
        TypeError: If obs_column is not a numerical dtype.
        ValueError: If labels is provided but does not contain exactly 2 elements.
    """
    # --- Input validation ---

    if obs_column not in anndata_object.obs.columns:
        raise KeyError(
            f"Column '{obs_column}' not found in .obs. "
            f"Available columns: {list(anndata_object.obs.columns)}"
        )

    if not pd.api.types.is_numeric_dtype(anndata_object.obs[obs_column]):
        raise TypeError(
            f"Column '{obs_column}' is not numerical "
            f"(dtype: {anndata_object.obs[obs_column].dtype}). "
            "Only continuous numerical columns can be split by threshold."
        )

    if labels is not None and len(labels) != 2:
        raise ValueError(
            f"labels must contain exactly 2 elements [below_label, above_label], "
            f"got {len(labels)}."
        )

    # --- Determine the output column name and labels ---

    effective_new_column_name = (
        new_column_name if new_column_name is not None else f"{obs_column}_split"
    )

    if effective_new_column_name in anndata_object.obs.columns:
        warnings.warn(
            f"Column '{effective_new_column_name}' already exists in .obs and will be overwritten.",
            UserWarning,
            stacklevel=2,
        )

    below_label, above_label = (
        labels
        if labels is not None
        else [f"Low (< {threshold:.1f})", f"High (≥ {threshold:.1f})"]
    )

    # --- Apply threshold split ---

    numeric_series = anndata_object.obs[obs_column]
    nan_mask = numeric_series.isna()
    nan_count = int(nan_mask.sum())

    if nan_count == len(numeric_series):
        raise ValueError(
            f"Column '{obs_column}' contains only NaN values ({nan_count} / {len(numeric_series)} events). "
            "Cannot split — this column was likely not populated. "
            "Check that enrich_with_clinical_data() was called and that the CSV contains this variable. "
            f"Tip: print(anndata_object.obs['{obs_column}'].describe()) to inspect the column."
        )

    result_series = pd.Series(index=anndata_object.obs.index, dtype=object)
    result_series[numeric_series < threshold] = below_label
    result_series[numeric_series >= threshold] = above_label
    result_series[nan_mask] = np.nan

    anndata_object.obs[effective_new_column_name] = result_series

    # --- Inject colors into .uns['color_palette'] ---

    final_labels = [below_label, above_label]
    _inject_bin_colors(anndata_object, final_labels, palette)

    # --- Store metadata in .uns ---

    if _ANALYSIS_CONFIG_KEY not in anndata_object.uns:
        anndata_object.uns[_ANALYSIS_CONFIG_KEY] = {}

    anndata_object.uns[_ANALYSIS_CONFIG_KEY]["threshold_split"] = {
        "source_column": obs_column,
        "new_column": effective_new_column_name,
        "threshold": threshold,
        "below_label": below_label,
        "above_label": above_label,
        "palette": palette,
    }

    # --- QC report ---

    _print_threshold_qc_report(
        anndata_object=anndata_object,
        obs_column=obs_column,
        new_column_name=effective_new_column_name,
        threshold=threshold,
        below_label=below_label,
        above_label=above_label,
        nan_count=nan_count,
    )

    return anndata_object


def map_obs_values(
    anndata_object: anndata.AnnData,
    source_column: str,
    value_map: Dict[str, List[str]],
    new_column_name: Optional[str] = None,
    unmapped_label: str = "Other",
    palette: str = "Set2",
) -> str:
    """
    Create a new .obs column by mapping values of an existing column to user-defined labels.

    Multiple original values can be merged into the same label, making this
    suitable for collapsing species into biological categories (e.g.
    "Long life" = worm + fly + K fish) or any other custom grouping of
    categorical metadata.

    The original source column is never modified. Colors for the new labels
    are injected into .uns['color_palette'] so all plot functions (plot_radar,
    plot_umap, plot_pca_scatter) recognize them automatically via group_by.

    If a source value appears under more than one category key a UserWarning is
    emitted and the value is assigned to the *last* matching key (dict insertion
    order). Source values not covered by value_map are assigned unmapped_label.

    Arguments:
        anndata_object: AnnData object to modify in-place.
        source_column: Name of the existing .obs column whose values are remapped.
                       Must be present in anndata_object.obs.
        value_map: Mapping from new category label → list of original values to
                   include in that category.
                   Example: {"Long life": ["worm", "fly", "K fish"],
                             "Short life": ["Z fish", "mouse", "human"]}
        new_column_name: Name for the new .obs column.  Defaults to
                         "{source_column}_group" when None.
        unmapped_label: Label assigned to source values not covered by value_map.
                        Default: "Other".
        palette: Matplotlib colormap name used to assign colors to the new
                 category labels (default "Set2").

    Returns:
        The name of the newly created .obs column (useful to pass directly to
        group_by in plot_radar, plot_umap, plot_pca_scatter).

    Raises:
        KeyError: If source_column is not found in .obs.
    """
    # --- Input validation ---

    if source_column not in anndata_object.obs.columns:
        raise KeyError(
            f"Column '{source_column}' not found in .obs. "
            f"Available columns: {list(anndata_object.obs.columns)}"
        )

    # --- Determine output column name ---

    effective_new_column_name = (
        new_column_name if new_column_name is not None else f"{source_column}_group"
    )

    if effective_new_column_name in anndata_object.obs.columns:
        warnings.warn(
            f"Column '{effective_new_column_name}' already exists in .obs and will be overwritten.",
            UserWarning,
            stacklevel=2,
        )

    # --- Build reverse lookup: original_value → new_label ---
    # Process keys in dict order; last key wins for duplicates.

    reverse_lookup: Dict[str, str] = {}
    for new_label, original_values in value_map.items():
        for original_value in original_values:
            original_key = str(original_value)
            if original_key in reverse_lookup and reverse_lookup[original_key] != new_label:
                warnings.warn(
                    f"Source value '{original_key}' appears in multiple category keys "
                    f"('{reverse_lookup[original_key]}' and '{new_label}'). "
                    f"It will be assigned to '{new_label}' (last key wins).",
                    UserWarning,
                    stacklevel=2,
                )
            reverse_lookup[original_key] = new_label

    # --- Apply mapping ---

    source_series = anndata_object.obs[source_column].astype(str)
    mapped_series = source_series.map(reverse_lookup)  # NaN for unrecognized values

    # Fill unmapped (NaN after .map) with the unmapped_label string.
    unmapped_mask = mapped_series.isna()
    mapped_series = mapped_series.where(~unmapped_mask, other=unmapped_label)

    anndata_object.obs[effective_new_column_name] = mapped_series

    # --- Determine ordered list of all group labels for color assignment ---

    # Order: declared keys first (preserving dict insertion order), then
    # unmapped_label if any rows ended up there.
    ordered_labels: List[str] = list(value_map.keys())
    if unmapped_mask.any() and unmapped_label not in ordered_labels:
        ordered_labels.append(unmapped_label)

    # --- Inject colors into .uns['color_palette'] ---

    _inject_bin_colors(anndata_object, ordered_labels, palette)

    # --- Store mapping metadata in .uns ---

    if _ANALYSIS_CONFIG_KEY not in anndata_object.uns:
        anndata_object.uns[_ANALYSIS_CONFIG_KEY] = {}

    if "obs_value_map" not in anndata_object.uns[_ANALYSIS_CONFIG_KEY]:
        anndata_object.uns[_ANALYSIS_CONFIG_KEY]["obs_value_map"] = {}

    anndata_object.uns[_ANALYSIS_CONFIG_KEY]["obs_value_map"][effective_new_column_name] = {
        "source_column": source_column,
        "value_map": {k: list(v) for k, v in value_map.items()},
        "unmapped_label": unmapped_label,
        "palette": palette,
    }

    # --- QC report ---

    _print_map_qc_report(
        anndata_object=anndata_object,
        source_column=source_column,
        new_column_name=effective_new_column_name,
        ordered_labels=ordered_labels,
        unmapped_label=unmapped_label,
        unmapped_count=int(unmapped_mask.sum()),
    )

    return effective_new_column_name


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _build_column_name(obs_column: str, n_bins: int, new_column_name: Optional[str]) -> str:
    """
    Return the output .obs column name, auto-generated when new_column_name is None.

    Arguments:
        obs_column: Source column name.
        n_bins: Number of bins requested.
        new_column_name: User-supplied name, or None for auto-generation.

    Returns:
        Column name string.
    """
    if new_column_name is not None:
        return new_column_name

    suffix_map = {3: "tertiles", 4: "quartiles"}
    suffix = suffix_map.get(n_bins, f"{n_bins}bins")
    return f"{obs_column}_{suffix}"


def _build_auto_labels(bin_edges: np.ndarray, actual_n_bins: int) -> List[str]:
    """
    Generate readable bin labels that include the actual data range.

    For tertiles (n=3): "T1 – Low (< 35.2)", "T2 – Mid (35.2 – 58.7)", "T3 – High (≥ 58.7)"
    For quartiles (n=4): "Q1 (< 25.0)", "Q2 (25.0 – 50.0)", "Q3 (50.0 – 75.0)", "Q4 (≥ 75.0)"
    For any N: "Bin 1 (< X)", "Bin i (X – Y)", "Bin N (≥ Y)"

    Arguments:
        bin_edges: Array of bin boundary values from pd.qcut (length = n_bins + 1).
        actual_n_bins: Number of bins actually created (may be less than requested
                       when duplicate edges are dropped).

    Returns:
        List of label strings, one per bin, in ascending order.
    """
    # Internal cut points (exclude the extended min and max).
    cut_points = bin_edges[1:-1]

    # Special case: all values collapsed into a single bin (all quantile cut points were identical).
    if actual_n_bins == 1:
        return [f"Bin 1 ({bin_edges[0]:.1f} – {bin_edges[-1]:.1f})"]

    if actual_n_bins == 3:
        prefixes = ["T1 – Low", "T2 – Mid", "T3 – High"]
    elif actual_n_bins == 4:
        prefixes = ["Q1", "Q2", "Q3", "Q4"]
    else:
        prefixes = [f"Bin {i}" for i in range(1, actual_n_bins + 1)]

    generated_labels = []
    for bin_index in range(actual_n_bins):
        prefix = prefixes[bin_index]
        if bin_index == 0:
            generated_labels.append(f"{prefix} (< {cut_points[0]:.1f})")
        elif bin_index == actual_n_bins - 1:
            generated_labels.append(f"{prefix} (≥ {cut_points[-1]:.1f})")
        else:
            generated_labels.append(
                f"{prefix} ({cut_points[bin_index - 1]:.1f} – {cut_points[bin_index]:.1f})"
            )

    return generated_labels


def _inject_bin_colors(
    anndata_object: anndata.AnnData,
    bin_labels: List[str],
    palette: str,
) -> None:
    """
    Add colors for the new bin labels into .uns['color_palette'].

    Only injects colors for labels not already present in the palette,
    making successive calls idempotent.

    Arguments:
        anndata_object: AnnData whose .uns['color_palette'] is updated in-place.
        bin_labels: Ordered list of bin label strings to colorize.
        palette: Matplotlib colormap name.
    """
    existing_palette: dict = anndata_object.uns.get(COLOR_PALETTE_KEY, {})

    uncolored_labels = [label for label in bin_labels if label not in existing_palette]
    new_colors = _generate_auto_colors(len(uncolored_labels), palette)

    updated_palette = dict(existing_palette)
    for label, color in zip(uncolored_labels, new_colors):
        updated_palette[label] = color

    anndata_object.uns[COLOR_PALETTE_KEY] = updated_palette


def _print_qc_report(
    anndata_object: anndata.AnnData,
    obs_column: str,
    new_column_name: str,
    final_labels: List[str],
    bin_edges: np.ndarray,
    nan_count: int,
    actual_n_bins: int,
) -> None:
    """
    Print a human-readable QC summary for the binning operation.

    Arguments:
        anndata_object: AnnData with the new column already added.
        obs_column: Name of the source column.
        new_column_name: Name of the newly created .obs column.
        final_labels: Bin labels in ascending order.
        bin_edges: Full array of bin boundaries from pd.qcut.
        nan_count: Number of NaN values in the source column.
        actual_n_bins: Number of bins actually created.
    """
    total_events = anndata_object.n_obs
    assigned_events = total_events - nan_count

    # Count events per bin from the new column (string values, excluding NaN).
    non_nan_column = anndata_object.obs[new_column_name].dropna()
    bin_counts = non_nan_column.value_counts()

    source_series = anndata_object.obs[obs_column]
    source_min = float(source_series.min())
    source_max = float(source_series.max())
    source_mean = float(source_series.mean())

    print("=" * 65)
    print("BIN OBS COLUMN — QC REPORT")
    print("=" * 65)
    print(f"  Source column   : {obs_column}")
    print(f"  New column      : {new_column_name}")
    print("  Method          : equal-frequency quantile binning (pd.qcut)")
    print(f"  Bins created    : {actual_n_bins}")
    print(f"  Source range    : [{source_min:.2f}, {source_max:.2f}]  (mean: {source_mean:.2f})")
    print(f"  Bin edges       : {[f'{e:.2f}' for e in bin_edges]}")
    print()
    print("  Bin distribution:")
    print(f"  {'Label':<42} {'Events':>8}  {'%':>6}")
    print("  " + "-" * 60)
    for label in final_labels:
        count = int(bin_counts.get(label, 0))
        pct = 100.0 * count / assigned_events if assigned_events > 0 else 0.0
        print(f"  {label:<42} {count:>8}  {pct:>5.1f}%")
    print()
    print(f"  Total events           : {total_events}")
    print(f"  Events assigned to bins: {assigned_events}")
    print(f"  Events with NaN source : {nan_count}")

    color_palette = anndata_object.uns.get(COLOR_PALETTE_KEY, {})
    print()
    print("  Colors injected into .uns['color_palette']:")
    for label in final_labels:
        color = color_palette.get(label, "(not found)")
        print(f"    {label:<42} {color}")

    print()
    print(f"  => New column '{new_column_name}' added to .obs.")
    print(f"  => Use group_by='{new_column_name}' in plot_radar(), plot_umap(), plot_pca_scatter().")
    print("=" * 65)


def _print_map_qc_report(
    anndata_object: anndata.AnnData,
    source_column: str,
    new_column_name: str,
    ordered_labels: List[str],
    unmapped_label: str,
    unmapped_count: int,
) -> None:
    """
    Print a human-readable QC summary for the map_obs_values operation.

    Arguments:
        anndata_object: AnnData with the new column already added.
        source_column: Name of the source column.
        new_column_name: Name of the newly created .obs column.
        ordered_labels: Category labels in display order (declared keys first).
        unmapped_label: The label used for source values not in value_map.
        unmapped_count: Number of rows that received the unmapped_label.
    """
    total_events = anndata_object.n_obs
    new_column_series = anndata_object.obs[new_column_name]
    group_counts = new_column_series.value_counts()
    color_palette = anndata_object.uns.get(COLOR_PALETTE_KEY, {})

    print("=" * 65)
    print("MAP OBS VALUES — QC REPORT")
    print("=" * 65)
    print(f"  Source column   : {source_column}")
    print(f"  New column      : {new_column_name}")
    print()
    print("  Group distribution:")
    print(f"  {'Label':<42} {'Events':>8}  {'%':>6}")
    print("  " + "-" * 60)
    for label in ordered_labels:
        count = int(group_counts.get(label, 0))
        pct = 100.0 * count / total_events if total_events > 0 else 0.0
        print(f"  {label:<42} {count:>8}  {pct:>5.1f}%")
    print()
    if unmapped_count > 0:
        print(f"  WARNING: {unmapped_count} event(s) were not covered by value_map")
        print(f"           and were assigned to unmapped_label='{unmapped_label}'.")
        print()
    print("  Colors injected into .uns['color_palette']:")
    for label in ordered_labels:
        color = color_palette.get(label, "(not found)")
        print(f"    {label:<42} {color}")
    print()
    print(f"  => New column '{new_column_name}' added to .obs.")
    print(f"  => Use group_by='{new_column_name}' in plot_radar(), plot_umap(), plot_pca_scatter().")
    print("=" * 65)

def _print_threshold_qc_report(
    anndata_object: anndata.AnnData,
    obs_column: str,
    new_column_name: str,
    threshold: float,
    below_label: str,
    above_label: str,
    nan_count: int,
) -> None:
    """
    Print a human-readable QC summary for the threshold split operation.

    Arguments:
        anndata_object: AnnData with the new column already added.
        obs_column: Name of the source column.
        new_column_name: Name of the newly created .obs column.
        threshold: The cut-off value used.
        below_label: Label for events strictly below the threshold.
        above_label: Label for events at or above the threshold.
        nan_count: Number of NaN values in the source column.
    """
    total_events = anndata_object.n_obs
    assigned_events = total_events - nan_count

    source_series = anndata_object.obs[obs_column]
    source_min = float(source_series.min())
    source_max = float(source_series.max())
    source_mean = float(source_series.mean())

    non_nan_column = anndata_object.obs[new_column_name].dropna()
    group_counts = non_nan_column.value_counts()

    below_count = int(group_counts.get(below_label, 0))
    above_count = int(group_counts.get(above_label, 0))
    below_pct = 100.0 * below_count / assigned_events if assigned_events > 0 else 0.0
    above_pct = 100.0 * above_count / assigned_events if assigned_events > 0 else 0.0

    color_palette = anndata_object.uns.get(COLOR_PALETTE_KEY, {})

    print("=" * 65)
    print("SPLIT OBS BY THRESHOLD — QC REPORT")
    print("=" * 65)
    print(f"  Source column   : {obs_column}")
    print(f"  New column      : {new_column_name}")
    print(f"  Threshold       : {threshold:.4g}  (< threshold → Group 1 ; ≥ threshold → Group 2)")
    print(f"  Source range    : [{source_min:.2f}, {source_max:.2f}]  (mean: {source_mean:.2f})")
    print()
    print("  Group distribution:")
    print(f"  {'Label':<42} {'Events':>8}  {'%':>6}")
    print("  " + "-" * 60)
    print(f"  {below_label:<42} {below_count:>8}  {below_pct:>5.1f}%")
    print(f"  {above_label:<42} {above_count:>8}  {above_pct:>5.1f}%")
    print()
    print(f"  Total events           : {total_events}")
    print(f"  Events assigned        : {assigned_events}")
    print(f"  Events with NaN source : {nan_count}")
    print()
    print("  Colors injected into .uns['color_palette']:")
    for label in [below_label, above_label]:
        color = color_palette.get(label, "(not found)")
        print(f"    {label:<42} {color}")
    print()
    print(f"  => New column '{new_column_name}' added to .obs.")
    print(f"  => Use group_by='{new_column_name}' in plot_radar(), plot_umap(), plot_pca_scatter().")
    print("=" * 65)
