"""
radar_plot.py

Radar (spider) plot with individual event distribution clouds for SFC and TEM AnnData.

Produces one polar-axes figure per pulse-type suffix (-A, -H, -W) for SFC data,
or a single figure covering all morphological features for TEM data.  Each spoke
represents one channel/feature.  For every group, two visual layers are drawn:
  1. Jittered scatter of individual events — shows the actual data density.
  2. A mean or median line (zorder=2) drawn on top of the scatter cloud —
     highlights the group's aggregate profile without occluding other group lines.

An optional dashed global mean line and an optional solid global median line
(across all events in the current selection) can be overlaid on each figure
to provide population-level references — controlled by show_global_mean and
show_global_median (both default to False).

When feature selection is active (.uns['analysis_config']['active_selection']
is set), only the selected channels are plotted. When no feature selection is
active, all channels in .X are shown.

group_by accepts either a single .obs column name or a list of column names.
When a list is given, values from each column are combined with " / " to form
one group label per unique combination — enabling cross-factor comparisons
(e.g. specie × age_group produces "worm / young", "worm / old", "fly / young", …).

nest_aggregate_by opts every aggregate profile line into a NESTED aggregate
(per individual, then across individuals) instead of pooling raw events —
so a heavily-sampled subject cannot outweigh a lightly-sampled one. Mirrors
ADR-004's per-individual-first rule (already used for MHI); see ADR-011 in
docs/DECISIONS.md. Accepts a list of columns ordered COARSEST to FINEST
(e.g. ["specie", "unique_subject_ID"], same convention as
compute_pca(weight_by=...)) to nest one level deeper — every species then
also counts equally, regardless of how many subjects it has.

Typical usage:
    from mito_marker.analysis import plot_radar
    plot_radar(sfc_subset, group_by="subject_ID")
    plot_radar(sfc_subset, group_by="dilution")
    plot_radar(tem_anndata, group_by=["specie", "age_group"])
"""

import os
import warnings
from math import ceil, degrees, pi
from typing import Dict, List, Optional, Tuple, Union

import anndata
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from mito_marker.analysis._aggregation import (
    _nested_group_aggregate,
    _resolve_nest_aggregate_columns,
)
from mito_marker.analysis._plot_context import (
    get_run_context_console_text,
    get_run_context_footer_text,
    get_species_label,
)
from mito_marker.analysis.colors import get_subject_colors, sort_values_for_legend
from mito_marker.analysis.feature_selection import _get_analytical_mask
from mito_marker.controlled_vocabulary import (
    PREFERRED_CONDITION_COLORS,
    TEM_FEATURE_CATEGORIES,
    TEM_FEATURE_CATEGORY_COLORS,
    TEM_FEATURE_CATEGORY_ORDER,
    TEM_RADAR_FEATURE_ABBREVIATIONS,
)

# Category order for TEM radar spoke grouping — the canonical
# TEM_FEATURE_CATEGORY_ORDER (Size, Shape, Intensity, Cristae Orientation)
# followed by the "Other" fallback rank for any feature missing from
# TEM_FEATURE_CATEGORIES.
_TEM_RADAR_CATEGORY_ORDER: List[str] = [*TEM_FEATURE_CATEGORY_ORDER, "Other"]

# .uns key where the analysis pipeline configuration is stored.
_ANALYSIS_CONFIG_KEY = "analysis_config"

# Laser group order for radar spoke sorting (FSC/SSC → UV → V → B → YG → R).
# Keys are the exact channel-name prefixes used by the Cytek Aurora.
# The dict preserves insertion order (Python 3.7+), so iteration order is the
# laser order — no separate ordering list is needed.
_LASER_GROUP_ORDER: Dict[str, int] = {
    "FSC": 0,
    "SSC": 1,
    "UV":  2,
    "V":   3,
    "B":   4,
    "YG":  5,
    "R":   6,
}

# subject_ID is always colored separately at plot time — not from the condition palette.
_SUBJECT_ID_COLUMN = "subject_ID"


def plot_radar(
    anndata_object: anndata.AnnData,
    group_by: Union[str, List[str]],
    channels: Optional[List[str]] = None,
    max_points_per_group: int = 300,
    title: str = "",
    show_global_mean: bool = False,
    show_global_median: bool = False,
    group_aggregate: str = "mean",
    nest_aggregate_by: Optional[Union[str, List[str]]] = None,
    legend_fontsize: int = 30,
    spoke_fontsize: int = 30,
    strip_suffix: bool = True,
    legend_fraction: float = 0.50,
    figure_size_inches: Optional[float] = None,
    legend_bbox_to_anchor: Tuple[float, float] = (0.60, 0.50),
    line_width: float = 2.0,
    show_category_boundaries: bool = True,
    interactive: bool = False,
    save_html_path: Optional[str] = None,
) -> None:
    """
    Draw radar plots of spectral channel profiles grouped by one or more .obs columns.

    Produces three figures — one per pulse-type suffix (-A, -H, -W).  Each
    figure contains one spoke per channel of that type.  Suffixes with no
    matching channels are skipped with a printed warning.

    When group_by is a list of column names, group labels are formed by
    concatenating the values from each column with " / " (e.g. "worm / young",
    "fly / old").  This produces one aggregate line per unique combination, so you
    can simultaneously compare specie × age_group on a single figure.

    Rendering per figure:
      - One spoke per channel (selected channels if feature selection is active,
        all channels otherwise), restricted to one pulse type.  When the
        ``channels`` argument is provided, only those channels are plotted.
      - Jittered scatter of up to max_points_per_group events per group
        (alpha=0.5, no edge, size=10). Scatter is per-spoke normalized: each
        spoke's values are linearly remapped from the spoke's own 1st–99th
        percentile range to the visible radial axis range, so the distribution
        shape fills the full axis on every spoke. Scatter radial position
        therefore shows relative density, not absolute measurement values.
        All groups share the same per-spoke bounds so relative ordering between
        groups is preserved. Groups are drawn in a single randomized pass so no
        group visually dominates by z-order.
      - Aggregate line per group (linewidth=line_width, zorder=2) plotted at
        actual values on top of scatter — these ARE readable from the radial
        axis. Whether this line represents the mean or the median is controlled
        by group_aggregate.
      - Dashed global mean line across all selected events (linewidth=1.5,
        zorder=3) — population-level mean reference, actual values.
        Only drawn when show_global_mean=True.
      - Solid global median line across all selected events (linewidth=1.5,
        zorder=3) — population-level median reference, actual values.
        Only drawn when show_global_median=True.
      - Feature labels rendered radially (parallel to each spoke) starting
        just outside the outer circle — uses label height rather than width
        for angular footprint, preventing overlap on 140-channel plots.
        TEM plots (no -A/-H/-W suffix) are the one exception: spokes are
        reordered into contiguous category blocks in
        TEM_FEATURE_CATEGORY_ORDER (Size, Shape, Intensity, Cristae
        Orientation — see TEM_FEATURE_CATEGORIES), labels are abbreviated
        (TEM_RADAR_FEATURE_ABBREVIATIONS), colored by category
        (TEM_FEATURE_CATEGORY_COLORS) with a small category legend top-left,
        and drawn horizontally instead of radially — readable at the ~24
        TEM feature count without SFC's overlap risk. A thick dashed radial
        line marks each category boundary (e.g. Size → Shape),
        toggled with show_category_boundaries, and labels within 15° of
        12/6 o'clock — where horizontal neighbors
        sit closest together — alternate between two label radii to avoid
        collisions. This TEM-only styling does not affect SFC (-A/-H/-W)
        rendering in any way.
      - Radial limits anchored to the group aggregate value range (+ 30% margin)
        so profile lines always dominate the visual.
      - Legend placed in a reserved right panel via figure-level coordinates,
        never overlapping the polar axes or spoke labels.

    The active layer is read from .uns['analysis_config']['active_layer'].
    When active_layer is None, raw .X is used.

    Arguments:
        anndata_object: SFC or TEM AnnData after selection, optional feature
                        selection, and optional normalization.
        group_by: Name of the .obs column to use for grouping (coloring), or
                  a list of column names whose values are combined into a single
                  group label.  Examples:
                    - group_by="subject_ID"
                    - group_by=["specie", "age_group"]
        channels: Optional list of channel/feature names to include as spokes.
                  When None (default), all analytical channels are shown
                  (subject to any active feature selection).  Names not present
                  in the resolved channel list are silently ignored.
        max_points_per_group: Maximum number of individual event points drawn
                              per group (random sample without replacement).
        title: Optional base title. Each figure appends "[A/H/W channels]".
               Defaults to an auto-generated description, which mentions
               `nest_aggregate_by`'s value ("Nested by '<nest_aggregate_by>'") when
               that argument is used. A custom title is shown as-is and does
               not get this suffix appended automatically.
        show_global_mean: Whether to draw the dashed black global mean line
                          across all events. Defaults to False.
        show_global_median: Whether to draw the solid black global median line
                            across all events. Defaults to False.
        group_aggregate: Aggregation function used for per-group profile lines.
                         "mean" (default) uses np.nanmean; "median" uses
                         np.nanmedian. Raises ValueError for any other value.
        nest_aggregate_by: Optional name of the .obs column identifying each
                       individual/subject within a group (e.g. "subject_ID"),
                       or a list of column names ordered from COARSEST to
                       FINEST (e.g. ["specie", "unique_subject_ID"]) — the
                       same convention as compute_pca(weight_by=...) in
                       pca_plot.py. When set, every profile line normally
                       computed by pooling raw events (the per-group
                       aggregate line, and the global mean/median reference
                       lines) is instead computed with a NESTED aggregate:
                       group_aggregate is applied once at the finest level
                       first, then again at each coarser level in turn, up to
                       the group itself — so every group at every hierarchy
                       level counts equally, regardless of how many
                       rows/children it has. This mirrors ADR-004's
                       "per-individual aggregation before group-level
                       summary" rule (already used for MHI) and is documented
                       in ADR-011 in docs/DECISIONS.md. With a single column,
                       it prevents a heavily-sampled subject (e.g. 1,600
                       events) from silently outweighing a lightly-sampled
                       one (e.g. 100 events). With two columns, e.g.
                       ["specie", "unique_subject_ID"], it additionally
                       prevents a species with more sampled subjects (e.g. 12)
                       from outweighing one with fewer (e.g. 3) when group_by
                       cuts across species (e.g. group_by="condition") — each
                       species gets exactly one vote in the group profile,
                       same as each subject gets exactly one vote within its
                       species. Only affects the aggregate line(s) — the
                       scatter cloud still shows every individual raw event,
                       unchanged. When None (default), behavior is unchanged
                       (pooled raw-event aggregate).
        legend_fontsize: Font size in pt for legend group labels and the legend
                         title. Default 30 — sized for publication print DPI.
                         Reduce to 12–14 for screen-only use.
        spoke_fontsize: Font size in pt for the radial channel/feature labels.
                        Labels are parallel to their spoke so this controls
                        label HEIGHT (angular footprint), not width — 7–10 pt
                        is safe even on 140-channel plots. Default 30.
        strip_suffix: When True (default), remove the trailing pulse-type suffix
                      (-H, -A, -W) from every spoke label.  On a single-suffix
                      figure the suffix is redundant; removing it saves 2 chars
                      per label and reduces visual clutter.
        legend_fraction: Fraction of figure WIDTH reserved for the legend panel.
                         The polar axes is compressed to (1 − legend_fraction)
                         of the figure width. Default 0.50. Increase if the
                         legend overflows right; decrease to enlarge the circle.
                         Unlike legend_bbox_to_anchor, this is not adjusted for
                         TEM plots — the polar axes keeps its full size and only
                         the legend's position (not the plot's) adapts.
        figure_size_inches: Figure HEIGHT in inches; width = height × 1.45
                            (wider than tall to accommodate the legend panel).
                            When None (default), auto-scales as
                            max(20, n_channels × 0.15) so 140-channel plots
                            always receive ≥ 20 inches of height.
        legend_bbox_to_anchor: (x, y) anchor for the legend in figure
                               coordinates [0, 1], passed directly to
                               fig.legend(bbox_to_anchor=...).  Default (0.60,
                               0.50) places the legend left edge at 60% of
                               figure width (10% past the polar axes edge when
                               legend_fraction=0.50). Increase x to push the
                               legend further right. On TEM plots this x is
                               increased automatically, scaled to spoke_fontsize
                               and the longest spoke label — horizontal labels
                               need more clearance than SFC's radial ones, and
                               how much more depends on how big the font is.
        line_width: Stroke width in points for the per-group aggregate profile
                    lines. Default 2.0. Increase (e.g. 3.0–5.0) for
                    publication figures where thin lines disappear after
                    down-scaling; decrease for dense multi-group plots where
                    thick lines obscure each other. Does not affect the global
                    mean/median reference lines (fixed at 1.5 pt).
        show_category_boundaries: TEM plots only. When True (default), draws
                                  the thick dashed radial line at each
                                  category boundary (Size → Shape → Intensity
                                  → Cristae Orientation). Set False to hide it
                                  — the contiguous block grouping, category
                                  label colors, and category legend are unaffected;
                                  this only toggles the boundary line itself.
                                  Ignored for SFC (-A/-H/-W) plots, which never
                                  draw it.
        interactive: When True, render each figure as an interactive Plotly
                     chart instead of a static matplotlib one. Hovering over a
                     group's line highlights it (other lines dim, the hovered
                     line thickens) and shows its name/value in a tooltip —
                     useful when many groups (e.g. a numeric column like age
                     with 15+ distinct values) make a static legend hard to
                     match to overlapping lines by color alone. Only the
                     per-group aggregate lines are drawn (no per-point jitter
                     cloud — plotting thousands of individual points as
                     interactive Plotly markers would be slow and is not
                     needed to read group identity). Requires the optional
                     ``plotly`` dependency. Default False (static matplotlib,
                     unchanged behavior).
        save_html_path: Only used when interactive=True. When given, each
                         figure is additionally saved as a self-contained
                         standalone .html file (hover-highlight JavaScript and
                         the plotly.js library both embedded — no internet
                         connection or Python needed to open it, and hover
                         highlighting still works). Share that file directly
                         with a colleague (e.g. email attachment, Slack, or a
                         path under a mounted Google Drive folder in Colab —
                         the same pattern used to save PDF reports). One file
                         per suffix is written: "path/radar.html" becomes
                         "path/radar_A.html", "path/radar_H.html",
                         "path/radar_W.html" (or "path/radar.html" unchanged
                         for TEM data, which has no suffix). Ignored (with a
                         warning) when interactive=False — a static matplotlib
                         figure isn't a file format, save it directly instead
                         (e.g. inside a `with ReportBuilder(...)` block, or
                         `plt.savefig()` right after calling plot_radar()).

    Returns:
        None. Each figure is displayed via plt.show() (or figure.show() when
        interactive=True).
    """
    if group_aggregate not in ("mean", "median"):
        raise ValueError(
            f"group_aggregate must be 'mean' or 'median', got '{group_aggregate}'."
        )
    nest_aggregate_by_columns: Optional[List[str]] = _resolve_nest_aggregate_columns(
        anndata_object, nest_aggregate_by, caller_name="plot_radar"
    )
    if save_html_path is not None and not interactive:
        warnings.warn(
            "save_html_path is ignored when interactive=False — a static "
            "matplotlib figure has no HTML export path. Set interactive=True, "
            "or save the matplotlib figure directly (e.g. plt.savefig()).",
            stacklevel=2,
        )
    data_matrix, channel_names = _get_data_and_channels(anndata_object)

    if channels is not None:
        keep_indices = [i for i, name in enumerate(channel_names) if name in channels]
        if not keep_indices:
            raise ValueError(
                f"None of the requested channels {channels} were found in the "
                f"resolved channel list. Available channels: {channel_names}"
            )
        data_matrix = data_matrix[:, keep_indices]
        channel_names = [channel_names[i] for i in keep_indices]
    group_labels = _get_group_labels(anndata_object, group_by)
    color_map = _build_color_map(anndata_object, group_by, group_labels)

    # Human-readable group_by label for title and legend.
    group_by_label = " × ".join(group_by) if isinstance(group_by, list) else group_by

    individual_labels = (
        anndata_object.obs[nest_aggregate_by_columns] if nest_aggregate_by_columns is not None else None
    )
    nest_aggregate_by_display: Optional[str] = (
        " > ".join(nest_aggregate_by_columns) if nest_aggregate_by_columns is not None else None
    )
    if individual_labels is not None:
        print(
            f"[plot_radar] nest_aggregate_by={nest_aggregate_by_columns!r} — profile lines use a "
            f"NESTED {group_aggregate} ({nest_aggregate_by_display}, finest level averaged first, "
            "then each coarser level in turn) instead of pooling raw events. See ADR-004 / ADR-011."
        )
        finest_column = nest_aggregate_by_columns[-1]
        coarsest_column = nest_aggregate_by_columns[0]
        for group_value in sort_values_for_legend(group_labels.dropna().unique()):
            group_mask = (group_labels == group_value).values
            n_events = int(group_mask.sum())
            finest_values = individual_labels[finest_column].values[group_mask]
            n_individuals = int(pd.unique(finest_values).shape[0])
            if len(nest_aggregate_by_columns) > 1:
                coarsest_values = individual_labels[coarsest_column].values[group_mask]
                n_coarsest = int(pd.unique(coarsest_values).shape[0])
                print(
                    f"    {group_value}: {n_events:,} events across {n_individuals} "
                    f"{finest_column}(s), {n_coarsest} {coarsest_column}(s)"
                )
            else:
                print(f"    {group_value}: {n_events:,} events across {n_individuals} individual(s)")

    # Build the base title once — reused (with suffix appended) for each figure.
    if not title:
        active_layer = anndata_object.uns.get(_ANALYSIS_CONFIG_KEY, {}).get("active_layer")
        active_selection = anndata_object.uns.get(_ANALYSIS_CONFIG_KEY, {}).get("active_selection")
        selection_info = f" [{active_selection}]" if active_selection else ""
        layer_info = active_layer if active_layer else "raw .X"
        species = get_species_label(anndata_object)
        species_prefix = f"[{species}]  " if species else ""
        nest_aggregate_by_info = (
            f" | Nested by '{nest_aggregate_by_display}'" if nest_aggregate_by_display is not None else ""
        )
        base_title = (
            f"{species_prefix}Radar Plot — grouped by '{group_by_label}'{selection_info}\n"
            f"(Points = per-spoke normalized distribution | Layer: {layer_info}{nest_aggregate_by_info})"
        )
    else:
        base_title = title

    suffix_to_indices = _split_channels_by_suffix(channel_names)

    print(get_run_context_console_text(anndata_object))
    footer_text = get_run_context_footer_text(anndata_object)

    for suffix, indices in suffix_to_indices.items():
        if not indices:
            # Only SFC suffix groups (A/H/W) can be legitimately empty.
            # The TEM fallback group ("all") is never empty.
            print(f"[plot_radar] No -{suffix} channels found — skipping.")
            continue
        if suffix in ("A", "H", "W"):
            print(
                f"[plot_radar] Rendering radar plot for -{suffix} channels "
                f"({len(indices)} channels)..."
            )
        else:
            print(
                f"[plot_radar] Rendering radar plot for {len(indices)} features..."
            )
        if interactive:
            _render_radar_interactive_for_channels(
                data_matrix=data_matrix,
                channel_indices=indices,
                channel_names=[channel_names[i] for i in indices],
                group_labels=group_labels,
                color_map=color_map,
                suffix_label=suffix,
                base_title=base_title,
                group_by=group_by_label,
                show_global_mean=show_global_mean,
                show_global_median=show_global_median,
                group_aggregate=group_aggregate,
                individual_labels=individual_labels,
                nest_aggregate_by_name=nest_aggregate_by_display,
                strip_suffix=strip_suffix,
                line_width=line_width,
                save_html_path=save_html_path,
            )
        else:
            _render_radar_for_channels(
                data_matrix=data_matrix,
                channel_indices=indices,
                channel_names=[channel_names[i] for i in indices],
                group_labels=group_labels,
                color_map=color_map,
                suffix_label=suffix,
                max_points_per_group=max_points_per_group,
                base_title=base_title,
                group_by=group_by_label,
                footer_text=footer_text,
                show_global_mean=show_global_mean,
                show_global_median=show_global_median,
                group_aggregate=group_aggregate,
                individual_labels=individual_labels,
                nest_aggregate_by_name=nest_aggregate_by_display,
                legend_fontsize=legend_fontsize,
                spoke_fontsize=spoke_fontsize,
                strip_suffix=strip_suffix,
                legend_fraction=legend_fraction,
                figure_size_inches=figure_size_inches,
                legend_bbox_to_anchor=legend_bbox_to_anchor,
                line_width=line_width,
                show_category_boundaries=show_category_boundaries,
            )


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _split_channels_by_suffix(
    channel_names: List[str],
) -> Dict[str, List[int]]:
    """
    Group channel indices by their pulse-type suffix (-A, -H, -W).

    For SFC AnnData (Cytek Aurora channels ending in -A / -H / -W), produces
    three groups sorted by canonical laser order.

    For TEM AnnData (morphological feature names with no recognised suffix),
    all channels are returned together in a single group under the key "all".
    This allows plot_radar to produce one radar figure covering all
    morphological features instead of silently skipping everything. Within
    that "all" group, channels are additionally reordered so that features
    sharing a TEM_FEATURE_CATEGORIES category are contiguous (see
    _sort_tem_channels_by_category) — the raw file column order interleaves
    features from different categories, which defeats category-based
    spoke coloring in the radar renderer.

    Arguments:
        channel_names: Ordered list of channel name strings from AnnData.var_names.

    Returns:
        Dict mapping suffix keys to lists of integer indices into channel_names.
        SFC: {"A": [...], "H": [...], "W": [...]}.
        TEM (fallback): {"all": [0, 1, ..., n-1]}.
    """
    suffix_to_indices: Dict[str, List[int]] = {"A": [], "H": [], "W": []}
    for index, name in enumerate(channel_names):
        for suffix in suffix_to_indices:
            if name.endswith(f"-{suffix}"):
                suffix_to_indices[suffix].append(index)
                break

    # Sort each suffix group by canonical laser order.
    sorted_groups = {
        suffix: _sort_channels_by_laser(channel_names, indices)
        for suffix, indices in suffix_to_indices.items()
    }

    # If no SFC-style channel was matched (e.g. TEM morphological features),
    # fall back to a single group, reordered so categories are contiguous.
    if not any(sorted_groups.values()):
        return {"all": _sort_tem_channels_by_category(channel_names)}

    return sorted_groups


def _sort_tem_channels_by_category(channel_names: List[str]) -> List[int]:
    """
    Sort TEM channel indices so that features sharing a radar category are contiguous.

    Uses a stable sort keyed on TEM_FEATURE_CATEGORIES (Size block first, then
    Shape, Intensity, Cristae Orientation), preserving each feature's original
    relative order within its category. Features absent from the dict are
    grouped last under "Other" and reported with a warning — this is a safety
    net for a feature added to TEM_FEATURE_COLUMNS without updating
    TEM_FEATURE_SUBSETS, not the expected path.

    Arguments:
        channel_names: Ordered list of TEM feature name strings.

    Returns:
        Indices into channel_names, reordered by category.
    """
    unmapped_names = [
        name for name in channel_names if name not in TEM_FEATURE_CATEGORIES
    ]
    if unmapped_names:
        print(
            f"[plot_radar] WARNING: {len(unmapped_names)} TEM feature(s) missing from "
            f"TEM_FEATURE_CATEGORIES, grouped as 'Other': {unmapped_names}. "
            "Add them to controlled_vocabulary.TEM_FEATURE_SUBSETS."
        )

    def _category_rank(name: str) -> int:
        category = TEM_FEATURE_CATEGORIES.get(name, "Other")
        return _TEM_RADAR_CATEGORY_ORDER.index(category)

    return sorted(range(len(channel_names)), key=lambda i: _category_rank(channel_names[i]))


def _sort_channels_by_laser(
    channel_names: List[str],
    channel_indices: List[int],
) -> List[int]:
    """
    Sort channel indices by laser group and detector number.

    Derives a (laser_group, detector_number) sort key directly from each
    channel name, so the sort is correct for every suffix (-A, -H, -W) and
    for any instrument variant — no hard-coded channel list is required.

    Sort key derivation:
      - Laser group (0–6): first matching prefix in _LASER_GROUP_ORDER
        (FSC=0, SSC=1, UV=2, V=3, B=4, YG=5, R=6).
      - Detector number: leading digits immediately after the prefix
        (e.g. "UV8-W" → 8, "UV16-A" → 16, "FSC-A" → 0).
      - Channels whose name matches no known prefix sort last (group 999).

    Arguments:
        channel_names: Full list of channel name strings from AnnData.var_names.
        channel_indices: Indices into channel_names to be sorted.

    Returns:
        Sorted copy of channel_indices.
    """
    def _sort_key(idx: int) -> Tuple[int, int]:
        name = channel_names[idx]
        for prefix, group in _LASER_GROUP_ORDER.items():
            if name.startswith(prefix):
                # Extract the detector number: leading digits after the prefix.
                # e.g. "UV16-A" → prefix "UV" → remainder "16-A" → digits "16"
                remainder = name[len(prefix):]
                digit_chars = ""
                for char in remainder:
                    if char.isdigit():
                        digit_chars += char
                    else:
                        break
                detector_num = int(digit_chars) if digit_chars else 0
                return (group, detector_num)
        return (999, 0)  # Unknown prefix → sort after all recognised lasers.

    return sorted(channel_indices, key=_sort_key)


def _render_radar_for_channels(
    data_matrix: np.ndarray,
    channel_indices: List[int],
    channel_names: List[str],
    group_labels: pd.Series,
    color_map: Dict[str, str],
    suffix_label: str,
    max_points_per_group: int,
    base_title: str,
    group_by: str,
    footer_text: str = "",
    show_global_mean: bool = False,
    show_global_median: bool = False,
    group_aggregate: str = "mean",
    individual_labels: Optional[pd.Series] = None,
    nest_aggregate_by_name: Optional[str] = None,
    legend_fontsize: int = 30,
    spoke_fontsize: int = 30,
    strip_suffix: bool = True,
    legend_fraction: float = 0.50,
    figure_size_inches: Optional[float] = None,
    legend_bbox_to_anchor: Tuple[float, float] = (0.60, 0.50),
    line_width: float = 2.0,
    show_category_boundaries: bool = True,
) -> None:
    """
    Render one radar figure for a subset of channels sharing the same suffix.

    Draws per-group scatter clouds and aggregate profile lines, then optionally
    overlays a dashed global mean line and/or a solid global median line.
    Spoke labels are rendered radially (parallel to each spoke) so label HEIGHT
    determines angular footprint — preventing overlap on 140-channel plots.
    The legend is placed in a reserved right panel using figure-level coordinates
    so it never overlaps the polar axes or the spoke labels.

    Arguments:
        data_matrix: Full data matrix, shape (n_events, n_all_channels).
        channel_indices: Indices into data_matrix columns for this suffix.
        channel_names: Names corresponding to channel_indices (pre-sliced).
        group_labels: Series of group values, one per event row.
        color_map: {group_value_str: hex_color} for per-group coloring.
        suffix_label: "A", "H", "W", or "all" — appended to the plot title for SFC.
        max_points_per_group: Maximum scatter points drawn per group.
        base_title: Base title string; suffix label is appended before display.
        group_by: .obs column name — used as legend title.
        footer_text: Optional run-context footer placed at figure bottom.
        show_global_mean: Whether to draw the dashed black global mean line.
        show_global_median: Whether to draw the solid black global median line.
        group_aggregate: "mean" or "median" — controls the per-group profile line.
        individual_labels: Optional Series of individual/subject IDs, one per
                           event row (same index/order as group_labels). When
                           given, every aggregate profile line (per-group,
                           global mean, global median) is computed with the
                           nested per-individual-then-across-individuals
                           aggregate instead of pooling raw events — see
                           plot_radar()'s nest_aggregate_by argument.
        nest_aggregate_by_name: The .obs column name individual_labels came from —
                            used only for the legend title text.
        legend_fontsize: Font size in pt for legend labels and title.
        spoke_fontsize: Font size in pt for radial spoke labels.
        strip_suffix: When True, remove trailing -H/-A/-W from spoke labels.
        legend_fraction: Fraction of figure width reserved for the legend panel;
                         polar axes compressed to (1 − legend_fraction).
        figure_size_inches: Figure height in inches (width = height × 1.45).
                            None auto-scales as max(20, n_channels × 0.15).
        legend_bbox_to_anchor: (x, y) legend anchor in figure coordinates,
                               passed directly to fig.legend().
        line_width: Stroke width in points for per-group aggregate profile lines.
                    Does not affect global mean/median reference lines (1.5 pt).
        show_category_boundaries: TEM plots only — draws the dashed category
                                  boundary line when True (default).

    Returns:
        None. The figure is displayed via plt.show().
    """
    subset_matrix = data_matrix[:, channel_indices]
    n_channels = len(channel_names)

    if n_channels == 0:
        print(f"[plot_radar] WARNING: No -{suffix_label} channels available to plot.")
        return

    # Angular positions: one spoke per channel, evenly spaced on [0, 2π).
    angles = [2 * pi * i / n_channels for i in range(n_channels)]
    angles_closed = angles + angles[:1]

    # Y limits anchored to per-group aggregate values so profile lines dominate.
    # Individual events can span ±3–4σ; anchoring to aggregates keeps profile
    # lines — the primary information — always the dominant visual element.
    aggregate_fn = np.nanmedian if group_aggregate == "median" else np.nanmean
    individual_values = individual_labels.values if individual_labels is not None else None

    def _group_profile(row_mask: np.ndarray, agg) -> np.ndarray:
        """Profile for the rows in row_mask, nested by individual when available."""
        matrix_slice = subset_matrix[row_mask]
        if individual_values is None:
            return agg(matrix_slice, axis=0)
        return _nested_group_aggregate(
            matrix_slice, individual_values[row_mask], agg, caller_name="plot_radar"
        )

    full_mask = np.ones(subset_matrix.shape[0], dtype=bool)
    unique_group_values = group_labels.dropna().unique()
    group_means_list = [
        _group_profile((group_labels == g).values, aggregate_fn)
        for g in unique_group_values
    ]
    if show_global_mean:
        group_means_list.append(_group_profile(full_mask, np.nanmean))
    if show_global_median:
        group_means_list.append(_group_profile(full_mask, np.nanmedian))
    all_mean_values = np.concatenate(group_means_list)
    mean_min = float(np.nanmin(all_mean_values))
    mean_max = float(np.nanmax(all_mean_values))
    margin = max((mean_max - mean_min) * 0.30, abs(mean_max) * 0.10, 0.05)
    y_min = mean_min - margin
    y_max = mean_max + margin

    # Wider than tall (×1.45) so the right margin accommodates spoke labels
    # and the legend panel without overlap.
    if figure_size_inches is None:
        figure_size_inches = max(20.0, n_channels * 0.15)

    fig, ax = plt.subplots(
        figsize=(figure_size_inches * 1.45, figure_size_inches),
        subplot_kw={"projection": "polar"},
    )

    # First spoke at 12 o'clock (North), proceeding clockwise:
    # FSC/SSC → UV → V → B → YG → R reads top-to-right around the plot.
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)

    unique_groups = sort_values_for_legend(group_labels.dropna().unique())

    # --- A. Scatter clouds ---
    # Shuffle all points before drawing so no group dominates by z-order.
    group_matrices_for_scatter: Dict[str, np.ndarray] = {
        str(group_value): subset_matrix[(group_labels == group_value).values]
        for group_value in unique_groups
    }
    _draw_all_scatter_clouds(
        ax=ax,
        group_matrices=group_matrices_for_scatter,
        group_colors={str(gv): color_map.get(str(gv), "#999999") for gv in unique_groups},
        angles=angles,
        max_points_per_group=max_points_per_group,
        y_min=y_min,
        y_max=y_max,
    )

    # --- B. Per-group aggregate profile lines ---
    # Colored patches are built in parallel for the legend — easier to read
    # than short line segments, especially at large publication font sizes.
    legend_handles = []
    for group_value in unique_groups:
        group_mask = (group_labels == group_value).values
        color = color_map.get(str(group_value), "#999999")

        group_profile = _group_profile(group_mask, aggregate_fn).tolist()
        group_profile_closed = group_profile + group_profile[:1]
        ax.plot(
            angles_closed,
            group_profile_closed,
            linewidth=line_width,
            linestyle="-",
            color=color,
            zorder=2,
        )
        legend_handles.append(
            mpatches.Patch(
                facecolor=color,
                label=str(group_value),
                edgecolor="#333333",
                linewidth=0.6,
            )
        )

    # --- C. Global mean profile line ---
    if show_global_mean:
        global_mean = _group_profile(full_mask, np.nanmean).tolist()
        global_mean_closed = global_mean + global_mean[:1]
        ax.plot(
            angles_closed,
            global_mean_closed,
            linewidth=1.5,
            linestyle="--",
            color="black",
            zorder=3,
        )
        legend_handles.append(
            Line2D([0], [0], color="black", linewidth=1.5,
                   linestyle="--", label="Mean Profile")
        )

    # --- D. Global median profile line ---
    if show_global_median:
        global_median = _group_profile(full_mask, np.nanmedian).tolist()
        global_median_closed = global_median + global_median[:1]
        ax.plot(
            angles_closed,
            global_median_closed,
            linewidth=1.5,
            linestyle="-",
            color="black",
            zorder=3,
        )
        legend_handles.append(
            Line2D([0], [0], color="black", linewidth=1.5,
                   linestyle="-", label="Median Profile")
        )

    # --- E. Radial spoke labels ---
    # Labels run along each spoke (parallel to the spoke direction) so label
    # HEIGHT — not width — determines angular footprint.  This prevents overlap
    # even at 140 channels where inter-spoke gap ≈ 2.6°.
    ax.set_xticks(angles)
    ax.set_xticklabels([])  # disable default horizontal labels

    # TEM plots (suffix "all") use abbreviated, category-colored, horizontal
    # labels — 24-ish spokes read more easily as short flat text than as
    # radially-rotated text on a publication figure. SFC plots (A/H/W, up to
    # 140 spokes) keep the original radial-label rendering unchanged, since
    # that scheme relies on label HEIGHT rather than WIDTH to avoid overlap.
    is_tem_plot = suffix_label not in ("A", "H", "W")

    if is_tem_plot:
        display_names = [
            TEM_RADAR_FEATURE_ABBREVIATIONS.get(name, name.removeprefix("Mito_"))
            for name in channel_names
        ]
        label_colors = [
            TEM_FEATURE_CATEGORY_COLORS.get(
                TEM_FEATURE_CATEGORIES.get(name, "Other"), TEM_FEATURE_CATEGORY_COLORS["Other"]
            )
            for name in channel_names
        ]
    else:
        if strip_suffix:
            display_names = [
                name[:-2] if name.endswith(("-H", "-A", "-W")) else name
                for name in channel_names
            ]
        else:
            display_names = channel_names
        label_colors = ["#1a1a1a"] * len(channel_names)

    # 4% gap keeps left-half labels (ha="right", body extends inward) fully
    # outside the outer circle line at typical font sizes and figure scales.
    # TEM horizontal labels get a slightly larger gap (6%) since their text
    # extends further from the spoke than radially-rotated labels do.
    label_r_base = y_max + (y_max - y_min) * (0.06 if is_tem_plot else 0.04)
    # TEM only: a second, further-out ring used to stagger every other label
    # near 12/6 o'clock (see near_vertical below) — pushed 6% further than
    # label_r_base so alternating labels land on visibly different rings.
    label_r_far = label_r_base + (y_max - y_min) * 0.06
    # Spokes within this many degrees of the 12 o'clock / 6 o'clock axis have
    # the smallest tangential (horizontal) gap between neighboring horizontal
    # labels — that's where alternating near/far radius is needed to keep
    # adjacent label rows from touching.
    vertical_zone_deg = 15.0

    for spoke_index, (angle, display_name, label_color) in enumerate(
        zip(angles, display_names, label_colors)
    ):
        angle_deg = degrees(angle) % 360
        # Convert polar angle (clockwise from North) → display angle (CCW from East).
        display_deg = (90.0 - angle_deg) % 360.0

        if is_tem_plot:
            # Horizontal text (no rotation) — only left/right alignment flips
            # depending on which half of the circle the spoke falls in, so
            # label bodies always extend away from the plot, never over it.
            text_rotation = 0.0
            # The spoke(s) sitting exactly on the vertical axis (12 o'clock —
            # always spoke 0 — and, when n_channels is even, 6 o'clock) point
            # straight up/down: left- or right-aligning that label pulls its
            # text sideways off the spoke's own axis. Center it instead, and
            # push it to the far ring so it visually lines up with its
            # left/right-anchored neighbors rather than sitting closer to
            # the outer circle than they do.
            is_polar_axis_spoke = (
                abs(display_deg - 90.0) < 1e-6 or abs(display_deg - 270.0) < 1e-6
            )
            if is_polar_axis_spoke:
                ha = "center"
                label_r = label_r_far
            else:
                ha = "right" if 90.0 < display_deg <= 270.0 else "left"
                near_vertical = (
                    abs(display_deg - 90.0) <= vertical_zone_deg
                    or abs(display_deg - 270.0) <= vertical_zone_deg
                )
                # Only alternate every other spoke — staggering both neighbors
                # the same way would not separate them.
                label_r = label_r_far if (near_vertical and spoke_index % 2 == 1) else label_r_base
        elif 90.0 < display_deg <= 270.0:
            # Avoid upside-down text: flip labels whose spokes point left.
            # Right half (display_deg ≤ 90° or > 270°): ha=left, text extends outward.
            # Left half  (display_deg in (90°, 270°]):   flip 180°, ha=right.
            text_rotation = display_deg - 180.0
            ha = "right"
            label_r = label_r_base
        else:
            text_rotation = display_deg
            ha = "left"
            label_r = label_r_base

        ax.text(
            angle, label_r, display_name,
            ha=ha, va="center",
            rotation=text_rotation,
            rotation_mode="anchor",
            fontsize=spoke_fontsize,
            fontweight="bold",
            color=label_color,
        )

    # --- E2. Category boundary separators (TEM only) ---
    # A thick dashed radial line at the angular midpoint between the last
    # spoke of one category and the first spoke of the next — marks every
    # transition between contiguous category blocks (Size → Shape → Intensity
    # → Cristae Orientation) and, since spokes form a closed circle, also
    # where the last category wraps back around to the first. Drawn behind
    # the scatter/profile lines (zorder below 1) but above the faint
    # background grid. Toggled off entirely via show_category_boundaries —
    # the block grouping, label colors, and category legend are independent
    # of this line and stay on.
    if is_tem_plot and show_category_boundaries and n_channels > 1:
        channel_categories = [
            TEM_FEATURE_CATEGORIES.get(name, "Other") for name in channel_names
        ]
        half_gap = pi / n_channels
        for spoke_index in range(n_channels):
            next_index = (spoke_index + 1) % n_channels
            if channel_categories[spoke_index] != channel_categories[next_index]:
                boundary_angle = angles[spoke_index] + half_gap
                ax.plot(
                    [boundary_angle, boundary_angle],
                    [y_min, label_r_far],
                    linestyle=(0, (4, 3)),
                    linewidth=2.2,
                    color="#333333",
                    alpha=0.8,
                    zorder=0.5,
                )

    # --- F. Axis aesthetics ---
    ax.set_ylim(y_min, y_max)
    ax.grid(True, linestyle="--", alpha=0.4)

    # --- G. Title ---
    if suffix_label in ("A", "H", "W"):
        plot_title = f"{base_title}\n[{suffix_label} channels]"
    else:
        plot_title = base_title
    fig.suptitle(plot_title, size=14, y=0.94, fontweight="bold")

    # --- H. Legend (figure-level coordinates, reserved right panel) ---
    # Compress polar axes to (1 − legend_fraction) of figure width so the
    # legend panel never overlaps the polar circle or its spoke labels.
    # fig.legend() with figure coordinates is used rather than ax.legend()
    # to decouple legend placement from the axes bounding box entirely.
    # TEM plots use horizontal labels that extend outward by their full text
    # WIDTH on both sides (SFC's radial labels extend by HEIGHT only, which is
    # why SFC keeps a slim fixed margin on both left_margin and the legend
    # push below). How far depends on spoke_fontsize and the longest label,
    # not a fixed offset — a fixed value either truncates labels at large
    # font sizes or wastes space at small ones. Estimate the longest label's
    # rendered width once (0.55 em/char is a safe upper bound for bold
    # sans-serif glyphs) and use it, symmetrically, to size the left margin
    # (clears left-half labels) and the legend push (clears right-half
    # labels) — the polar axes itself keeps its normal size; only the margin
    # and the legend position adapt.
    left_margin = 0.02
    effective_legend_bbox_to_anchor = legend_bbox_to_anchor
    if is_tem_plot:
        max_label_chars = max((len(name) for name in display_names), default=0)
        figure_width_inches = figure_size_inches * 1.45
        estimated_label_width_fraction = (
            (spoke_fontsize / 72.0) * 0.55 * max_label_chars / figure_width_inches
        )
        label_clearance = min(estimated_label_width_fraction + 0.02, 0.30)
        left_margin = label_clearance
        effective_legend_bbox_to_anchor = (
            min(legend_bbox_to_anchor[0] + label_clearance, 0.95),
            legend_bbox_to_anchor[1],
        )
    fig.subplots_adjust(left=left_margin, right=1.0 - legend_fraction, bottom=0.05, top=0.90)

    if individual_labels is not None:
        legend_title_text = (
            f"{group_by}\n(line = nested {group_aggregate} per group,\n"
            f"by '{nest_aggregate_by_name}')"
        )
    else:
        legend_title_text = f"{group_by}\n(line = {group_aggregate} per group)"
    n_unique_groups = len(unique_groups)
    legend_ncol = max(1, ceil(n_unique_groups / 12))

    legend = fig.legend(
        handles=legend_handles,
        title=legend_title_text,
        loc="center left",
        bbox_to_anchor=effective_legend_bbox_to_anchor,
        ncol=legend_ncol,
        fontsize=legend_fontsize,
        title_fontsize=legend_fontsize + 1,
        framealpha=0.97,
        edgecolor="#444444",
        borderpad=1.0,
        labelspacing=0.8,
        handlelength=1.4,
        handleheight=1.5,
        handletextpad=0.8,
        columnspacing=1.2,
    )
    legend.get_frame().set_linewidth(1.5)
    legend.get_title().set_fontweight("bold")

    # --- H2. Feature category legend (TEM only) ---
    # A colored spoke label is meaningless without a key — this small legend
    # maps each TEM_FEATURE_CATEGORY_COLORS color back to its category name.
    # Placed top-left, above the polar axes (y > 0.90), so it never competes
    # with the plot itself or the group legend reserved on the right.
    if is_tem_plot:
        used_categories = sorted(
            {TEM_FEATURE_CATEGORIES.get(name, "Other") for name in channel_names},
            key=_TEM_RADAR_CATEGORY_ORDER.index,
        )
        category_legend_handles = [
            mpatches.Patch(
                facecolor=TEM_FEATURE_CATEGORY_COLORS.get(
                    category, TEM_FEATURE_CATEGORY_COLORS["Other"]
                ),
                label=category,
                edgecolor="#333333",
                linewidth=0.6,
            )
            for category in used_categories
        ]
        category_legend_fontsize = max(10, legend_fontsize * 0.6)
        category_legend = fig.legend(
            handles=category_legend_handles,
            title="Feature category",
            loc="upper left",
            bbox_to_anchor=(0.01, 0.985),
            fontsize=category_legend_fontsize,
            title_fontsize=category_legend_fontsize + 1,
            framealpha=0.9,
            edgecolor="#444444",
        )
        category_legend.get_title().set_fontweight("bold")

    # --- I. Footer ---
    if footer_text:
        fig.text(
            0.5, 0.01, footer_text,
            ha="center", va="bottom",
            fontsize=6, color="gray", style="italic",
        )

    plt.show()


def _import_plotly():
    """
    Lazily import plotly.graph_objects and return the module.

    Using a lazy import keeps plotly an optional dependency for the rest of
    radar_plot.py — only the interactive path raises if it's missing.

    Returns:
        The plotly.graph_objects module.

    Raises:
        ImportError: If plotly is not installed.
    """
    try:
        import plotly.graph_objects as go  # type: ignore[import-untyped]
        return go
    except ImportError as error:
        raise ImportError(
            "plotly is required for interactive radar plots (interactive=True). "
            "Install it with: pip install plotly"
        ) from error


# Injected client-side after each interactive figure renders. Dims every line
# except the one under the cursor and thickens it, so a group can be told
# apart from overlapping neighbors even when many groups share similar hues
# (e.g. 15+ distinct values of a numeric column like age). `{plot_id}` is
# substituted by Plotly with this specific figure's div id — see
# plotly.io._html.to_html — so each figure only wires up its own div, safe to
# render several figures (one per suffix) in the same notebook cell.
_INTERACTIVE_HOVER_HIGHLIGHT_JS = """
var plotDiv = document.getElementById('{plot_id}');
var originalWidths = plotDiv.data.map(function(trace) {
    return (trace.line && trace.line.width) ? trace.line.width : 2;
});
plotDiv.on('plotly_hover', function(eventData) {
    var hoveredIndex = eventData.points[0].curveNumber;
    var opacities = plotDiv.data.map(function(trace, i) {
        return i === hoveredIndex ? 1 : 0.12;
    });
    var widths = plotDiv.data.map(function(trace, i) {
        return i === hoveredIndex ? originalWidths[i] * 2.5 : originalWidths[i];
    });
    Plotly.restyle(plotDiv, {opacity: opacities, 'line.width': widths});
});
plotDiv.on('plotly_unhover', function(eventData) {
    var opacities = plotDiv.data.map(function() { return 1; });
    Plotly.restyle(plotDiv, {opacity: opacities, 'line.width': originalWidths});
});
"""


def _insert_suffix_before_extension(path: str, suffix_label: str) -> str:
    """
    Insert a pulse-type suffix before a file path's extension.

    Used to derive one output filename per suffix figure (A/H/W) from a
    single base path given by the caller, e.g. "radar.html" -> "radar_A.html".
    TEM data has no pulse-type suffix ("all") — the path is returned unchanged.

    Arguments:
        path: Base file path, e.g. "results/radar.html".
        suffix_label: "A", "H", "W", or "all".

    Returns:
        The path with "_{suffix_label}" inserted before the extension, or the
        original path unchanged when suffix_label == "all".
    """
    if suffix_label not in ("A", "H", "W"):
        return path
    base, extension = os.path.splitext(path)
    if not extension:
        extension = ".html"
    return f"{base}_{suffix_label}{extension}"


def _render_radar_interactive_for_channels(
    data_matrix: np.ndarray,
    channel_indices: List[int],
    channel_names: List[str],
    group_labels: pd.Series,
    color_map: Dict[str, str],
    suffix_label: str,
    base_title: str,
    group_by: str,
    show_global_mean: bool = False,
    show_global_median: bool = False,
    group_aggregate: str = "mean",
    individual_labels: Optional[pd.Series] = None,
    nest_aggregate_by_name: Optional[str] = None,
    strip_suffix: bool = True,
    line_width: float = 2.0,
    save_html_path: Optional[str] = None,
) -> None:
    """
    Render one interactive Plotly radar figure for a subset of channels.

    Draws only the per-group aggregate profile lines (no per-point jitter
    cloud — see the `interactive` argument of plot_radar() for why). Hovering
    a line highlights it via injected JavaScript (_INTERACTIVE_HOVER_HIGHLIGHT_JS)
    and shows its group name/value through Plotly's native tooltip. The
    legend is Plotly's built-in figure legend (click to hide a group,
    double-click to isolate it — no custom code needed for that part).

    Arguments:
        data_matrix: Full data matrix, shape (n_events, n_all_channels).
        channel_indices: Indices into data_matrix columns for this suffix.
        channel_names: Names corresponding to channel_indices (pre-sliced,
                       already sorted by laser group).
        group_labels: Series of group values, one per event row.
        color_map: {group_value_str: hex_color} for per-group coloring —
                   same mapping used by the static matplotlib renderer, so
                   colors match if both are used side by side.
        suffix_label: "A", "H", "W", or "all" — appended to the plot title.
        base_title: Base title string; suffix label is appended before display.
        group_by: .obs column name — used as legend title.
        show_global_mean: Whether to draw the dashed black global mean line.
        show_global_median: Whether to draw the solid black global median line.
        group_aggregate: "mean" or "median" — controls the per-group profile line.
        individual_labels: Optional Series of individual/subject IDs, one per
                           event row. When given, every aggregate profile
                           line uses the nested per-individual-then-
                           across-individuals aggregate — see plot_radar()'s
                           nest_aggregate_by argument and _nested_group_aggregate().
        nest_aggregate_by_name: The .obs column name individual_labels came from —
                            used only for the legend title text.
        strip_suffix: When True, remove trailing -H/-A/-W from spoke labels.
        line_width: Stroke width in points for per-group aggregate profile lines.
        save_html_path: Optional base path — when given, this figure is also
                         written to a standalone .html file (suffix inserted
                         before the extension, e.g. "radar.html" -> "radar_A.html").
                         See plot_radar()'s save_html_path argument for details.

    Returns:
        None. The figure is displayed via figure.show(), and optionally saved
        to save_html_path.
    """
    go = _import_plotly()

    subset_matrix = data_matrix[:, channel_indices]
    n_channels = len(channel_names)

    if n_channels == 0:
        print(f"[plot_radar] WARNING: No -{suffix_label} channels available to plot.")
        return

    # TEM plots (suffix "all") use the same abbreviated names as the static
    # renderer, for consistency between the two. Category spoke-coloring is
    # static-only — Plotly's angular axis has no per-tick-label color API.
    if suffix_label not in ("A", "H", "W"):
        display_names = [
            TEM_RADAR_FEATURE_ABBREVIATIONS.get(name, name.removeprefix("Mito_"))
            for name in channel_names
        ]
    elif strip_suffix:
        display_names = [
            name[:-2] if name.endswith(("-H", "-A", "-W")) else name
            for name in channel_names
        ]
    else:
        display_names = channel_names
    theta_closed = display_names + display_names[:1]

    # Y limits anchored to per-group aggregate values, same rationale as the
    # static renderer: individual events can span far wider than the group
    # profiles, so anchoring to aggregates keeps the profile lines readable.
    aggregate_fn = np.nanmedian if group_aggregate == "median" else np.nanmean
    individual_values = individual_labels.values if individual_labels is not None else None

    def _group_profile(row_mask: np.ndarray, agg) -> np.ndarray:
        """Profile for the rows in row_mask, nested by individual when available."""
        matrix_slice = subset_matrix[row_mask]
        if individual_values is None:
            return agg(matrix_slice, axis=0)
        return _nested_group_aggregate(
            matrix_slice, individual_values[row_mask], agg, caller_name="plot_radar"
        )

    full_mask = np.ones(subset_matrix.shape[0], dtype=bool)
    unique_groups = sort_values_for_legend(group_labels.dropna().unique())
    group_profiles = {
        group_value: _group_profile((group_labels == group_value).values, aggregate_fn)
        for group_value in unique_groups
    }
    all_profile_values = list(group_profiles.values())
    if show_global_mean:
        all_profile_values.append(_group_profile(full_mask, np.nanmean))
    if show_global_median:
        all_profile_values.append(_group_profile(full_mask, np.nanmedian))
    all_mean_values = np.concatenate(all_profile_values)
    mean_min = float(np.nanmin(all_mean_values))
    mean_max = float(np.nanmax(all_mean_values))
    margin = max((mean_max - mean_min) * 0.30, abs(mean_max) * 0.10, 0.05)
    y_min = mean_min - margin
    y_max = mean_max + margin

    traces = []
    for group_value in unique_groups:
        color = color_map.get(str(group_value), "#999999")
        group_profile = group_profiles[group_value].tolist()
        group_profile_closed = group_profile + group_profile[:1]
        traces.append(
            go.Scatterpolar(
                r=group_profile_closed,
                theta=theta_closed,
                mode="lines",
                name=str(group_value),
                line=dict(color=color, width=line_width),
                hovertemplate=(
                    f"<b>{group_by}: {group_value}</b><br>"
                    "%{theta}: %{r:.3f}<extra></extra>"
                ),
            )
        )

    if show_global_mean:
        global_mean = _group_profile(full_mask, np.nanmean).tolist()
        global_mean_closed = global_mean + global_mean[:1]
        traces.append(
            go.Scatterpolar(
                r=global_mean_closed,
                theta=theta_closed,
                mode="lines",
                name="Mean Profile",
                line=dict(color="black", width=1.5, dash="dash"),
                hovertemplate="<b>Mean Profile</b><br>%{theta}: %{r:.3f}<extra></extra>",
            )
        )

    if show_global_median:
        global_median = _group_profile(full_mask, np.nanmedian).tolist()
        global_median_closed = global_median + global_median[:1]
        traces.append(
            go.Scatterpolar(
                r=global_median_closed,
                theta=theta_closed,
                mode="lines",
                name="Median Profile",
                line=dict(color="black", width=1.5, dash="solid"),
                hovertemplate="<b>Median Profile</b><br>%{theta}: %{r:.3f}<extra></extra>",
            )
        )

    if suffix_label in ("A", "H", "W"):
        plot_title = f"{base_title}  [{suffix_label} channels]"
    else:
        plot_title = base_title

    if individual_labels is not None:
        legend_title_text = (
            f"{group_by}<br>(line = nested {group_aggregate} per group, "
            f"by '{nest_aggregate_by_name}')"
        )
    else:
        legend_title_text = f"{group_by}<br>(line = {group_aggregate} per group)"

    figure = go.Figure(data=traces)
    figure.update_layout(
        title=dict(text=plot_title, x=0.5, xanchor="center"),
        polar=dict(
            radialaxis=dict(range=[y_min, y_max]),
            # Matches the static renderer: 0° at North, proceeding clockwise.
            angularaxis=dict(rotation=90, direction="clockwise"),
        ),
        legend=dict(title=dict(text=legend_title_text)),
        template="plotly_white",
        height=800,
        width=1100,
        hovermode="closest",
    )

    if save_html_path is not None:
        output_path = _insert_suffix_before_extension(save_html_path, suffix_label)
        figure.write_html(
            output_path,
            post_script=_INTERACTIVE_HOVER_HIGHLIGHT_JS,
            include_plotlyjs=True,
            full_html=True,
        )
        print(f"[plot_radar] Interactive figure saved to: {output_path}")

    figure.show(post_script=_INTERACTIVE_HOVER_HIGHLIGHT_JS)


def _get_data_and_channels(
    anndata_object: anndata.AnnData,
) -> Tuple[np.ndarray, List[str]]:
    """
    Return the data matrix and channel names to use for plotting.

    If feature selection is active, only the selected channels are returned.
    Otherwise, all channels from the active layer (or .X) are returned.

    Arguments:
        anndata_object: AnnData with .uns['analysis_config'] populated.

    Returns:
        Tuple of (data_matrix, channel_names) where data_matrix is
        shape (n_obs, n_channels) and channel_names is a list of strings.
    """
    analysis_config = anndata_object.uns.get(_ANALYSIS_CONFIG_KEY, {})
    active_layer = analysis_config.get("active_layer")
    active_selection = analysis_config.get("active_selection")

    # Determine which layer to read from.
    if active_layer is not None and active_layer in anndata_object.layers:
        full_matrix = anndata_object.layers[active_layer]
        layer_label = active_layer
    else:
        full_matrix = anndata_object.X
        layer_label = "raw .X"

    # Convert sparse matrices to dense if needed.
    if hasattr(full_matrix, "toarray"):
        full_matrix = full_matrix.toarray()

    print(f"[plot_radar] Using data from: '{layer_label}'")

    # Apply feature selection if active.
    if active_selection is not None:
        selection_column = f"is_selected_{active_selection}"
        if selection_column in anndata_object.var.columns:
            selection_mask = anndata_object.var[selection_column].values
            data_matrix = full_matrix[:, selection_mask]
            channel_names = anndata_object.var_names[selection_mask].tolist()
            print(
                f"[plot_radar] Feature selection active ({active_selection}): "
                f"{len(channel_names)} of {anndata_object.n_vars} channels plotted."
            )
            return data_matrix, channel_names

    # No active feature selection.
    # Still exclude non-analytical features (spatial coordinates, redundant
    # measurements) — same contract as PCA and UMAP. For SFC data this is
    # additionally enforced by the -A/-H/-W suffix filter in
    # _split_channels_by_suffix; for TEM data (the "all" fallback group)
    # this mask is the only safeguard. _get_analytical_mask() falls back to
    # a name-based match (TEM_NON_ANALYTICAL_FEATURES / SFC_NON_ANALYTICAL_CHANNELS)
    # when .var["is_non_analytical"] itself is missing — e.g. after merging
    # several per-species AnnData objects with a plain anndata.concat() call
    # that omitted merge="same" (which silently drops every .var column, not
    # just this one — see tem_ingestion._concatenate_anndata_objects for the
    # guarded version used at ingestion time).
    analytical_mask = _get_analytical_mask(anndata_object)
    channel_names = anndata_object.var_names[analytical_mask].tolist()
    # np.asarray avoids an unnecessary copy when the matrix is already a
    # contiguous numpy array (the common case after normalisation).
    return np.asarray(full_matrix)[:, analytical_mask], channel_names


def _get_group_labels(
    anndata_object: anndata.AnnData,
    group_by: Union[str, List[str]],
) -> pd.Series:
    """
    Extract and combine grouping columns from .obs into a single label Series.

    When group_by is a string, the corresponding column is returned as-is.
    When group_by is a list, the values from each column are concatenated with
    " / " to form a combined label (e.g. "worm / young", "fly / old").

    Arguments:
        anndata_object: AnnData whose .obs will be queried.
        group_by: Name of a single .obs column, or a list of column names to
                  combine.

    Returns:
        A pandas Series indexed like anndata_object.obs, containing the group
        label for each observation.

    Raises:
        ValueError: If any specified column is not present in .obs.
    """
    columns = [group_by] if isinstance(group_by, str) else group_by

    for column in columns:
        if column not in anndata_object.obs.columns:
            raise ValueError(
                f"Column '{column}' not found in .obs. "
                f"Available columns: {list(anndata_object.obs.columns)}"
            )

    if len(columns) == 1:
        return anndata_object.obs[columns[0]]

    # Combine multiple columns into one label: "value1 / value2 / ..."
    combined = anndata_object.obs[columns[0]].astype(str)
    for column in columns[1:]:
        combined = combined + " / " + anndata_object.obs[column].astype(str)
    combined.name = " / ".join(columns)
    return combined


def _build_color_map(
    anndata_object: anndata.AnnData,
    group_by: Union[str, List[str]],
    group_labels: pd.Series,
) -> Dict[str, str]:
    """
    Build a {group_value: hex_color} mapping for the given grouping column(s).

    For subject_ID (single column): generates colors from tab10 at plot time.
    For combined groups (list of columns): auto-generates distinct colors from
    tab20 — combined labels have no entry in PREFERRED_CONDITION_COLORS or the
    stored color palette.
    For single condition columns: reads from .uns['color_palette'] with
    PREFERRED_CONDITION_COLORS override, auto-generates for unknown values.

    Arguments:
        anndata_object: AnnData with .uns['color_palette'] optionally populated.
        group_by: Name of the .obs grouping column, or a list of column names.
        group_labels: Series of group values for the plotted observations.

    Returns:
        Dict mapping each unique group value (as string) to a hex color.
    """
    unique_groups = sort_values_for_legend(group_labels.dropna().unique())
    unique_group_strings = [str(g) for g in unique_groups]

    # Multi-column combination: combined labels are not in any stored palette.
    if isinstance(group_by, list):
        return get_subject_colors(unique_group_strings, palette="tab20")

    if group_by == _SUBJECT_ID_COLUMN:
        # Subject colors are generated fresh from tab10 — not stored in .uns.
        return get_subject_colors(unique_group_strings, palette="tab10")

    # Condition colors — priority order:
    #   1. PREFERRED_CONDITION_COLORS — researcher-defined overrides that always win.
    #   2. .uns['color_palette']      — stored palette built by assign_color_palette().
    #   3. Auto-generated Set2 colors — fallback for values absent from both sources.
    # Checking PREFERRED_CONDITION_COLORS directly ensures that custom colors are
    # applied even when assign_color_palette() has not been called beforehand.
    stored_palette: Dict[str, str] = anndata_object.uns.get("color_palette", {})
    color_map = {}
    missing_values = []
    for group_value in unique_groups:
        key = str(group_value)
        if key in PREFERRED_CONDITION_COLORS:
            color_map[key] = PREFERRED_CONDITION_COLORS[key]
        elif key in stored_palette:
            color_map[key] = stored_palette[key]
        else:
            missing_values.append(key)

    if missing_values:
        # Auto-generate distinct colors for values absent from the palette.
        auto_colors = get_subject_colors(missing_values, palette="Set2")
        color_map.update(auto_colors)

    return color_map


def _draw_all_scatter_clouds(
    ax: plt.Axes,
    group_matrices: Dict[str, np.ndarray],
    group_colors: Dict[str, str],
    angles: List[float],
    max_points_per_group: int,
    y_min: float,
    y_max: float,
) -> None:
    """
    Draw jittered scatter points for all groups using per-spoke normalization.

    Each spoke's scatter values are linearly remapped from the spoke's own
    1st–99th percentile range (computed across all groups combined) to the
    visible radial range [y_min, y_max].  This lets scatter fill the full
    visible axis on every spoke regardless of the normalization method, while
    mean profile lines remain plotted at their actual values on the same axis.

    Visual contract:
      - Scatter radial position → relative density within each spoke (not
        the actual measurement value).  Two groups at different raw levels
        will appear at different positions within the spoke, correctly
        preserving relative ordering.
      - Mean lines → actual values, readable from the radial axis.

    All groups are collected and shuffled before drawing so no group
    visually dominates the others by z-ordering.

    Arguments:
        ax: Matplotlib polar Axes object.
        group_matrices: {group_value_str: data_matrix (n_events, n_channels)}.
        group_colors: {group_value_str: hex_color}.
        angles: Base angle (radians) for each channel spoke.
        max_points_per_group: Maximum number of events per group before random
                              subsampling is applied.
        y_min: Lower bound of the radial axis (from mean-based Y limits).
        y_max: Upper bound of the radial axis (from mean-based Y limits).
    """
    n_channels = len(angles)
    if n_channels == 0:
        return

    # Per-spoke normalization bounds computed across ALL groups combined.
    # Using a single global bound per spoke ensures that the same raw value
    # maps to the same radial position regardless of which group it belongs
    # to — so relative differences between groups are preserved in the scatter.
    all_group_data = np.concatenate(list(group_matrices.values()), axis=0)
    spoke_p1 = np.nanpercentile(all_group_data, 1, axis=0)   # shape (n_channels,)
    spoke_p99 = np.nanpercentile(all_group_data, 99, axis=0)  # shape (n_channels,)
    spoke_range = spoke_p99 - spoke_p1
    # Avoid division by zero for constant spokes (all events identical).
    spoke_range = np.where(spoke_range == 0, 1.0, spoke_range)

    # Width of the jitter corridor: 15% of the inter-spoke gap.
    jitter_width = (2 * pi / n_channels) * 0.15
    axis_range = y_max - y_min

    all_thetas: List[float] = []
    all_radii: List[float] = []
    all_colors: List[str] = []

    for group_value, group_matrix in group_matrices.items():
        color = group_colors.get(group_value, "#999999")
        n_events = group_matrix.shape[0]
        if n_events == 0:
            continue

        # Subsample if the group exceeds the display limit.
        if n_events > max_points_per_group:
            selected_indices = np.random.choice(n_events, max_points_per_group, replace=False)
            group_matrix = group_matrix[selected_indices]

        for channel_index, angle_base in enumerate(angles):
            channel_values = group_matrix[:, channel_index]
            finite_mask = np.isfinite(channel_values)
            channel_values = channel_values[finite_mask]

            if len(channel_values) == 0:
                continue

            # Remap actual values to [y_min, y_max] using the spoke's global
            # percentile range so scatter fills the visible axis on every spoke.
            normalized = (
                (channel_values - spoke_p1[channel_index])
                / spoke_range[channel_index]
            )
            remapped = y_min + normalized * axis_range

            noise = np.random.uniform(-jitter_width, jitter_width, size=len(remapped))
            all_thetas.extend(angle_base + noise)
            all_radii.extend(remapped.tolist())
            all_colors.extend([color] * len(remapped))

    if not all_thetas:
        return

    # Shuffle all collected points so no group consistently appears on top.
    shuffle_order = np.random.permutation(len(all_thetas))
    ax.scatter(
        np.array(all_thetas)[shuffle_order],
        np.array(all_radii)[shuffle_order],
        c=[all_colors[i] for i in shuffle_order],
        s=10,
        alpha=0.5,
        edgecolors="none",
        zorder=1,
        label=None,
    )


def _draw_scatter_cloud(
    ax: plt.Axes,
    group_matrix: np.ndarray,
    angles: List[float],
    color: str,
    max_points: int,
) -> None:
    """
    Draw jittered scatter points for one group onto polar axes.

    For each spoke (channel), plots individual event values with a small
    random angular perturbation (15% of the inter-spoke gap) to spread
    overlapping points into a visible cloud.

    Note: prefer _draw_all_scatter_clouds() for multi-group rendering —
    it draws all groups in random order to avoid z-ordering bias.

    Arguments:
        ax: Matplotlib polar Axes object.
        group_matrix: Data matrix for this group, shape (n_events, n_channels).
        angles: Base angle (radians) for each channel spoke.
        color: Hex color string for this group.
        max_points: Maximum number of events to display (random sample if more).
    """
    n_events, n_channels = group_matrix.shape
    if n_events == 0 or n_channels == 0:
        return

    # Subsample if the group has more events than the display limit.
    if n_events > max_points:
        selected_indices = np.random.choice(n_events, max_points, replace=False)
        group_matrix = group_matrix[selected_indices]

    # Width of the jitter corridor: 15% of the inter-spoke gap.
    jitter_width = (2 * pi / n_channels) * 0.15

    for channel_index, angle_base in enumerate(angles):
        channel_values = group_matrix[:, channel_index]
        # Remove NaN values before plotting.
        finite_mask = np.isfinite(channel_values)
        channel_values = channel_values[finite_mask]

        if len(channel_values) == 0:
            continue

        noise = np.random.uniform(-jitter_width, jitter_width, size=len(channel_values))
        theta_jittered = angle_base + noise

        ax.scatter(
            theta_jittered,
            channel_values,
            color=color,
            s=10,
            alpha=0.5,
            label=None,
            edgecolors="none",
            zorder=1,
        )
