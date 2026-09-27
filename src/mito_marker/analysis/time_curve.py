"""
time_curve.py

Longitudinal (time-series) line plots for SFC and TEM AnnData objects.

Plots one or more channel / feature values on the Y-axis against a categorical
or pseudo-numerical .obs column (e.g. timepoint, age_group, condition) on the
X-axis.  For each X-value the aggregate (mean or median) is computed per group,
with an optional shaded ±1 std error band around the line.

When both ``y_channels`` and ``group_by`` are provided, each (group, channel)
combination receives its own distinct colour.  All lines use the same solid
style; the legend identifies each curve by "group / channel".

When ``group_by`` is None, each channel receives its own colour.

Typical usage::

    from mito_marker.analysis import plot_time_curve

    # Single channel, coloured by condition across timepoints
    plot_time_curve(
        sfc_anndata,
        x_column="timepoint",
        y_channels=["FSC-A"],
        group_by="condition",
        x_order=["J0", "J7", "J14", "J20"],
    )

    # Several channels, no grouping
    plot_time_curve(
        sfc_anndata,
        x_column="timepoint",
        y_channels=["FSC-A", "SSC-A", "UV1-A"],
    )
"""

import re
from math import ceil
from typing import Dict, List, Optional, Tuple

import anndata
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from mito_marker.analysis._plot_context import (
    get_run_context_console_text,
    get_run_context_footer_text,
    get_species_label,
)
from mito_marker.analysis.colors import sort_values_for_legend
from mito_marker.controlled_vocabulary import PREFERRED_CONDITION_COLORS

_ANALYSIS_CONFIG_KEY = "analysis_config"



def plot_time_curve(
    anndata_object: anndata.AnnData,
    x_column: str,
    y_channels: Optional[List[str]] = None,
    group_by: Optional[str] = None,
    aggregate: str = "mean",
    show_error_band: bool = True,
    x_order: Optional[List[str]] = None,
    title: str = "",
) -> None:
    """
    Plot channel values over a time-like .obs column as a longitudinal curve.

    One line is drawn for each (channel, group) combination.  Each point on a
    line represents the aggregate (mean or median) of all observations in that
    group at that X-value, with an optional shaded ±1 std band.

    Colour conventions:
      - ``group_by=None``  → colour = channel.
      - ``group_by`` provided, 1 channel → colour = group.
      - ``group_by`` provided, N channels → colour = (group, channel) pair.
        All lines are solid; each curve is distinguished by colour alone.

    The active layer is read from .uns['analysis_config']['active_layer'].
    When active_layer is None, raw .X is used.

    Arguments:
        anndata_object: SFC or TEM AnnData after optional normalisation and
                        feature selection.
        x_column: Name of the .obs column to use as the X-axis (e.g.
                  ``"timepoint"``, ``"age_group"``).
        y_channels: List of channel / feature names to plot on the Y-axis.
                    When None, all analytical channels are used (subject to any
                    active feature selection).  Names not found in the resolved
                    channel list raise a ValueError.
        group_by: Optional .obs column used to split observations into coloured
                  groups.  When None, all observations are treated as one group
                  and colour encodes the channel instead.
        aggregate: Aggregation applied per (x_value, group) cell.  ``"mean"``
                   (default) or ``"median"``.
        show_error_band: When True (default), a shaded ±1 std band is drawn
                         around each aggregate line.
        x_order: Explicit ordering for the X-axis values.  When None, values
                 are sorted numerically if possible (extracting leading digits),
                 otherwise alphabetically.
        title: Optional figure title.  Defaults to an auto-generated description.

    Returns:
        None.  The figure is displayed via plt.show().

    Raises:
        ValueError: If ``x_column`` is not in .obs, if any name in
                    ``y_channels`` is not found in the resolved channel list,
                    if ``aggregate`` is not ``"mean"`` or ``"median"``.
    """
    if aggregate not in ("mean", "median"):
        raise ValueError(
            f"aggregate must be 'mean' or 'median', got '{aggregate}'."
        )
    if x_column not in anndata_object.obs.columns:
        raise ValueError(
            f"x_column '{x_column}' not found in .obs. "
            f"Available columns: {list(anndata_object.obs.columns)}"
        )

    data_matrix, channel_names = _get_data_and_channels(anndata_object)

    # Apply y_channels filter.
    if y_channels is not None:
        missing = [ch for ch in y_channels if ch not in channel_names]
        if missing:
            raise ValueError(
                f"The following channels were not found in the resolved channel "
                f"list: {missing}. Available: {channel_names}"
            )
        keep_indices = [channel_names.index(ch) for ch in y_channels]
        data_matrix = data_matrix[:, keep_indices]
        channel_names = list(y_channels)

    if group_by is not None and group_by not in anndata_object.obs.columns:
        raise ValueError(
            f"group_by '{group_by}' not found in .obs. "
            f"Available columns: {list(anndata_object.obs.columns)}"
        )

    x_values = _resolve_x_order(anndata_object.obs[x_column], x_order)

    print(get_run_context_console_text(anndata_object))
    print(
        f"[plot_time_curve] x_column='{x_column}' | "
        f"{len(channel_names)} channel(s) | "
        f"group_by={repr(group_by)} | aggregate={aggregate}"
    )

    fig, ax = plt.subplots(figsize=(10, 5))

    if group_by is None:
        _plot_no_groupby(
            ax=ax,
            anndata_object=anndata_object,
            data_matrix=data_matrix,
            channel_names=channel_names,
            obs_series=anndata_object.obs[x_column],
            x_values=x_values,
            aggregate=aggregate,
            show_error_band=show_error_band,
        )
    else:
        _plot_with_groupby(
            ax=ax,
            anndata_object=anndata_object,
            data_matrix=data_matrix,
            channel_names=channel_names,
            obs_series=anndata_object.obs[x_column],
            group_series=anndata_object.obs[group_by],
            x_values=x_values,
            aggregate=aggregate,
            show_error_band=show_error_band,
        )

    ax.set_xlabel(x_column, fontsize=12)
    active_layer = anndata_object.uns.get(_ANALYSIS_CONFIG_KEY, {}).get("active_layer")
    layer_label = active_layer if active_layer else "raw .X"
    ax.set_ylabel(f"Value ({layer_label})", fontsize=12)

    if not title:
        species = get_species_label(anndata_object)
        species_prefix = f"[{species}]  " if species else ""
        group_info = f" — grouped by '{group_by}'" if group_by else ""
        title = (
            f"{species_prefix}Time Curve — '{x_column}'{group_info}\n"
            f"(Layer: {layer_label} | aggregate: {aggregate})"
        )
    ax.set_title(title, fontsize=11, pad=12)
    ax.tick_params(axis="x", rotation=30)

    handles, labels = ax.get_legend_handles_labels()
    # handlelength=3.5 ensures dashed/dotted line styles are clearly visible
    # in the legend even when a circle marker overlaps the centre of the handle.
    legend_kwargs = {"fontsize": 9, "handlelength": 3.5, "numpoints": 2}
    if len(labels) > 12:
        ncols = ceil(len(labels) / 12)
        ax.legend(
            handles,
            labels,
            ncol=ncols,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.20),
            **legend_kwargs,
        )
        plt.tight_layout(rect=[0, 0.12, 1, 1])
    else:
        ax.legend(
            handles,
            labels,
            loc="upper left",
            bbox_to_anchor=(1.02, 1),
            **legend_kwargs,
        )
        plt.tight_layout()

    footer_text = get_run_context_footer_text(anndata_object)
    if footer_text:
        fig.text(0.5, -0.02, footer_text, ha="center", fontsize=7, color="#666666")

    plt.show()


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _get_data_and_channels(
    anndata_object: anndata.AnnData,
) -> Tuple[np.ndarray, List[str]]:
    """
    Return (data_matrix, channel_names) respecting active_layer, feature
    selection, and the non-analytical mask.

    Follows the same resolution chain used by radar_plot, pca_plot, and
    umap_plot: active_layer → feature selection → non-analytical exclusion.
    """
    analysis_config = anndata_object.uns.get(_ANALYSIS_CONFIG_KEY, {})
    active_layer = analysis_config.get("active_layer")
    active_selection = analysis_config.get("active_selection")

    if active_layer is not None and active_layer in anndata_object.layers:
        full_matrix = anndata_object.layers[active_layer]
        layer_label = active_layer
    else:
        full_matrix = anndata_object.X
        layer_label = "raw .X"

    if hasattr(full_matrix, "toarray"):
        full_matrix = full_matrix.toarray()

    print(f"[plot_time_curve] Using data from: '{layer_label}'")

    if active_selection is not None:
        selection_column = f"is_selected_{active_selection}"
        if selection_column in anndata_object.var.columns:
            selection_mask = anndata_object.var[selection_column].values
            channel_names = anndata_object.var_names[selection_mask].tolist()
            print(
                f"[plot_time_curve] Feature selection active ({active_selection}): "
                f"{len(channel_names)} of {anndata_object.n_vars} channels used."
            )
            return np.asarray(full_matrix, dtype=np.float32)[:, selection_mask], channel_names

    if "is_non_analytical" in anndata_object.var.columns:
        analytical_mask = ~anndata_object.var["is_non_analytical"].values.astype(bool)
        channel_names = anndata_object.var_names[analytical_mask].tolist()
        return np.asarray(full_matrix, dtype=np.float32)[:, analytical_mask], channel_names

    channel_names = anndata_object.var_names.tolist()
    return np.asarray(full_matrix, dtype=np.float32), channel_names


def _resolve_x_order(obs_series: pd.Series, x_order: Optional[List[str]]) -> List[str]:
    """
    Return the ordered list of unique X-axis values.

    Priority:
    1. Explicit ``x_order`` list (only values present in the data are kept).
    2. Numeric sort by extracting the first sequence of digits from each label
       (handles "D1", "D18", "J0", "J7").
    3. Alphabetical fallback.
    """
    unique_values = obs_series.dropna().unique().tolist()
    unique_str = [str(v) for v in unique_values]

    if x_order is not None:
        ordered = [v for v in x_order if v in unique_str]
        extra = sorted(set(unique_str) - set(ordered))
        return ordered + extra

    # Attempt numeric sort by extracting leading digits.
    def _numeric_key(label: str) -> Tuple[int, str]:
        match = re.search(r"\d+", label)
        return (int(match.group()), label) if match else (999_999, label)

    return sorted(unique_str, key=_numeric_key)


def _aggregate_values(
    values: np.ndarray,
    aggregate: str,
) -> Tuple[float, float]:
    """Return (aggregate_value, std) for a 1-D array of finite values."""
    finite = values[np.isfinite(values)]
    if len(finite) == 0:
        return np.nan, np.nan
    agg = float(np.mean(finite)) if aggregate == "mean" else float(np.median(finite))
    std = float(np.std(finite))
    return agg, std


def _auto_channel_colors(channel_names: List[str]) -> Dict[str, str]:
    """Generate one distinct colour per channel using the tab10 palette."""
    cmap = plt.colormaps.get_cmap("tab10").resampled(max(len(channel_names), 1))
    return {name: f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}"
            for i, (name, (r, g, b, _)) in enumerate(
                zip(channel_names, [cmap(i) for i in range(len(channel_names))])
            )}


def _plot_no_groupby(
    ax: plt.Axes,
    anndata_object: anndata.AnnData,
    data_matrix: np.ndarray,
    channel_names: List[str],
    obs_series: pd.Series,
    x_values: List[str],
    aggregate: str,
    show_error_band: bool,
) -> None:
    """Draw one line per channel with automatic colour assignment."""
    obs_str = obs_series.astype(str)
    channel_colors = _auto_channel_colors(channel_names)

    for channel_idx, channel_name in enumerate(channel_names):
        col_values = data_matrix[:, channel_idx]
        color = channel_colors[channel_name]
        agg_line = []
        std_line = []

        for x_val in x_values:
            mask = obs_str == x_val
            agg, std = _aggregate_values(col_values[mask], aggregate)
            agg_line.append(agg)
            std_line.append(std)

        x_positions = list(range(len(x_values)))
        agg_array = np.array(agg_line, dtype=float)
        std_array = np.array(std_line, dtype=float)

        ax.plot(x_positions, agg_array, color=color, linewidth=2,
                marker="o", markersize=5, label=channel_name)
        if show_error_band:
            ax.fill_between(
                x_positions,
                agg_array - std_array,
                agg_array + std_array,
                color=color,
                alpha=0.15,
            )

    ax.set_xticks(list(range(len(x_values))))
    ax.set_xticklabels(x_values)


def _build_group_color_map(
    anndata_object: anndata.AnnData,
    unique_groups: List[str],
) -> Dict[str, str]:
    """
    Map group values to hex colours.

    Priority:
    1. PREFERRED_CONDITION_COLORS (AL, IF, and any other declared conditions)
    2. .uns['color_palette'] (populated by assign_color_palette())
    3. Auto-generated Set2 colours for any remaining values

    This ensures distinct colours even when assign_color_palette() has not
    been called — avoiding the #999999 fallback-grey for all groups.
    """
    stored_palette = anndata_object.uns.get("color_palette", {})
    colors: Dict[str, str] = {}
    unassigned: List[str] = []

    for group_value in unique_groups:
        if group_value in PREFERRED_CONDITION_COLORS:
            colors[group_value] = PREFERRED_CONDITION_COLORS[group_value]
        elif group_value in stored_palette:
            colors[group_value] = stored_palette[group_value]
        else:
            unassigned.append(group_value)

    if unassigned:
        auto = _generate_distinct_colors(len(unassigned))
        for group_value, color in zip(unassigned, auto):
            colors[group_value] = color

    return colors


def _generate_distinct_colors(n: int) -> List[str]:
    """Generate n visually distinct hex colours, cycling through tab20 for large n."""
    # tab20 provides 20 perceptually distinct colours; cycle when n > 20.
    cmap = plt.colormaps.get_cmap("tab20")
    return [
        f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}"
        for r, g, b, _ in [cmap((i % 20) / 20) for i in range(n)]
    ]


def _plot_with_groupby(
    ax: plt.Axes,
    anndata_object: anndata.AnnData,
    data_matrix: np.ndarray,
    channel_names: List[str],
    obs_series: pd.Series,
    group_series: pd.Series,
    x_values: List[str],
    aggregate: str,
    show_error_band: bool,
) -> None:
    """
    Draw one line per (group, channel) combination.

    When only one channel is requested, colour = group (using the stored
    palette / PREFERRED_CONDITION_COLORS).  When multiple channels are
    requested, each (group, channel) pair receives its own distinct colour
    drawn from tab20 — all lines are solid.
    """
    obs_str = obs_series.astype(str)
    group_str = group_series.astype(str)
    unique_groups = sort_values_for_legend(group_str.unique().tolist())
    x_positions = list(range(len(x_values)))

    # Build a flat ordered list of (group, channel) pairs so colours can be
    # assigned as a single block, ensuring maximum visual separation.
    pairs: List[Tuple[str, str]] = [
        (group_value, channel_name)
        for channel_name in channel_names
        for group_value in unique_groups
    ]

    if len(channel_names) == 1:
        # Single channel: reuse the stored group palette for consistency with
        # other plots (radar, PCA…).
        group_colors = _build_group_color_map(anndata_object, unique_groups)
        pair_colors = {
            (group_value, channel_names[0]): group_colors[group_value]
            for group_value in unique_groups
        }
    else:
        # Multiple channels: one colour per (group, channel) pair.
        distinct = _generate_distinct_colors(len(pairs))
        pair_colors = {pair: color for pair, color in zip(pairs, distinct)}

    for channel_name in channel_names:
        col_idx = channel_names.index(channel_name)
        col_values = data_matrix[:, col_idx]

        for group_value in unique_groups:
            group_mask = group_str == group_value
            color = pair_colors[(group_value, channel_name)]
            agg_line = []
            std_line = []

            for x_val in x_values:
                x_mask = obs_str == x_val
                combined_mask = group_mask & x_mask
                agg, std = _aggregate_values(col_values[combined_mask.values], aggregate)
                agg_line.append(agg)
                std_line.append(std)

            agg_array = np.array(agg_line, dtype=float)
            std_array = np.array(std_line, dtype=float)

            label = f"{group_value} / {channel_name}" if len(channel_names) > 1 else group_value

            ax.plot(x_positions, agg_array, color=color, linestyle="-",
                    linewidth=2, marker="o", markersize=5, label=label)
            if show_error_band:
                ax.fill_between(
                    x_positions,
                    agg_array - std_array,
                    agg_array + std_array,
                    color=color,
                    alpha=0.15,
                )

    ax.set_xticks(x_positions)
    ax.set_xticklabels(x_values)
