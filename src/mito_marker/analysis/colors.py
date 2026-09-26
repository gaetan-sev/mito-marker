"""
colors.py

Condition color palette management for SFC analysis visualizations.

Assigns consistent hex color strings to every unique condition value found
across the filterable categorical .obs columns of an SFC AnnData. Colors are
stored in .uns['color_palette'] so that all downstream plot functions (radar,
UMAP, PCA) use the same mapping automatically.

Priority order:
  1. PREFERRED_CONDITION_COLORS from controlled_vocabulary.py — researcher-defined
     overrides that persist across runs and datasets.
  2. Automatic palette (default: matplotlib's "Set2") — assigned to any condition
     value not covered by the overrides.

Subject IDs are NOT given fixed colors here — they are auto-assigned at plot
time from "tab10" since the number of subjects varies too much between projects.

Typical usage:
    from mito_marker.analysis import assign_color_palette
    sfc_subset = assign_color_palette(sfc_subset)
"""

from typing import Any, Dict, List

import anndata
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

from mito_marker.controlled_vocabulary import (
    FILTERABLE_OBS_COLUMNS,
    PREFERRED_CONDITION_COLORS,
)

# .uns key where the condition→color mapping is stored.
COLOR_PALETTE_KEY = "color_palette"

# Default matplotlib colormap used for auto-assignment.
_DEFAULT_PALETTE = "Set2"

# Columns that are not condition labels — excluded from the color palette.
# subject_ID has too many possible values and is colored at plot time.
_NON_CONDITION_COLUMNS = {"subject_ID"}


def assign_color_palette(
    anndata_object: anndata.AnnData,
    palette: str = _DEFAULT_PALETTE,
) -> anndata.AnnData:
    """
    Assign a consistent color to every unique condition value in the AnnData.

    Iterates over all filterable .obs columns that are present in the AnnData
    (excluding subject_ID). For each unique value, checks PREFERRED_CONDITION_COLORS
    first; falls back to the automatic palette for unspecified values.

    The result is written to .uns['color_palette'] as a flat dict:
        {"Diluted": "#66c2a5", "Not_Diluted": "#fc8d62", "AL": "#8da0cb", ...}

    Already-existing entries in .uns['color_palette'] are preserved: calling
    this function twice with the same AnnData is idempotent.

    Overriding colors (e.g. in Google Colab where source files are not editable):

      Option 1 — post-assign edit (simplest, one-off):
          sfc_subset = assign_color_palette(sfc_subset)
          sfc_subset.uns["color_palette"]["AL"] = "#4daf4a"
          sfc_subset.uns["color_palette"]["IF"] = "#984ea3"

      Option 2 — pre-populate before calling (idempotent-safe):
          sfc_subset.uns["color_palette"] = {"AL": "#4daf4a", "IF": "#984ea3"}
          sfc_subset = assign_color_palette(sfc_subset)  # fills in the rest

      Option 3 — monkey-patch the module dict (persists for the whole session):
          import mito_marker.controlled_vocabulary as cv
          cv.PREFERRED_CONDITION_COLORS["AL"] = "#4daf4a"
          cv.PREFERRED_CONDITION_COLORS["IF"] = "#984ea3"
          sfc_subset = assign_color_palette(sfc_subset)

    In a local development environment, edit PREFERRED_CONDITION_COLORS directly
    in src/mito_marker/controlled_vocabulary.py instead.

    Arguments:
        anndata_object: SFC AnnData produced by select_sfc_subset() or earlier
                        pipeline steps.
        palette: Matplotlib colormap name used for automatic assignment
                 (default "Set2"). Only values not in PREFERRED_CONDITION_COLORS
                 receive auto-assigned colors.

    Returns:
        The same AnnData object with .uns['color_palette'] populated.
    """
    # Collect all unique condition values across relevant .obs columns.
    condition_values = _collect_condition_values(anndata_object)

    # Load any existing partial palette (idempotency: don't overwrite).
    existing_palette: Dict[str, str] = anndata_object.uns.get(COLOR_PALETTE_KEY, {})

    # Determine which values still need a color assignment.
    unassigned_values = [
        value
        for value in condition_values
        if value not in existing_palette and value not in PREFERRED_CONDITION_COLORS
    ]

    auto_colors = _generate_auto_colors(len(unassigned_values), palette)
    auto_color_map = dict(zip(unassigned_values, auto_colors))

    # Build the final palette: existing entries take precedence, then PREFERRED,
    # then auto-assigned — so a pre-seeded color is never overwritten.
    final_palette: Dict[str, str] = dict(existing_palette)
    for value in condition_values:
        if value in existing_palette:
            pass  # already set — preserve the caller's value
        elif value in PREFERRED_CONDITION_COLORS:
            final_palette[value] = PREFERRED_CONDITION_COLORS[value]
        elif value not in final_palette:
            final_palette[value] = auto_color_map[value]

    anndata_object.uns[COLOR_PALETTE_KEY] = final_palette

    print("=" * 60)
    print("COLOR PALETTE ASSIGNED")
    print("=" * 60)
    for value, color in sorted(final_palette.items()):
        source = (
            "custom (PREFERRED_CONDITION_COLORS)"
            if value in PREFERRED_CONDITION_COLORS
            else "auto"
        )
        print(f"  {value:<30} {color}  [{source}]")
    print(f"\n=> {len(final_palette)} condition values mapped.")
    print(f"=> Stored in .uns['{COLOR_PALETTE_KEY}'].")
    print("=" * 60)

    return anndata_object


def get_color_for_value(
    anndata_object: anndata.AnnData,
    condition_value: str,
    fallback_color: str = "#999999",
) -> str:
    """
    Return the hex color for a condition value from the stored palette.

    Arguments:
        anndata_object: AnnData with .uns['color_palette'] populated.
        condition_value: The condition label to look up.
        fallback_color: Color returned when the value is not in the palette.

    Returns:
        Hex color string (e.g. "#66c2a5").
    """
    palette = anndata_object.uns.get(COLOR_PALETTE_KEY, {})
    return palette.get(str(condition_value), fallback_color)


def get_subject_colors(
    subject_ids: List[str],
    palette: str = "tab10",
) -> Dict[str, str]:
    """
    Generate a color mapping for subject IDs at plot time.

    Subject colors are not stored in .uns because the subject list changes
    with each select_sfc_subset() call. This function generates them fresh
    from the given list.

    Arguments:
        subject_ids: List of unique subject ID strings (order determines color).
        palette: Matplotlib colormap name (default "tab10").

    Returns:
        Dict mapping each subject_id to a hex color string.
    """
    colors = _generate_auto_colors(len(subject_ids), palette)
    return dict(zip(subject_ids, colors))


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _collect_condition_values(anndata_object: anndata.AnnData) -> List[str]:
    """
    Collect all unique non-missing condition values from relevant .obs columns.

    Only considers columns in FILTERABLE_OBS_COLUMNS that are:
      - Present in the AnnData's .obs
      - Not in _NON_CONDITION_COLUMNS (subject_ID is excluded)
      - Not a float column (numerical cols like age are not conditions)
      - Not a boolean column (FlowAI_Pass values become "True"/"False" strings)

    Arguments:
        anndata_object: AnnData to inspect.

    Returns:
        Sorted list of unique condition value strings.
    """
    import pandas as pd

    all_values = set()
    for column_name in FILTERABLE_OBS_COLUMNS:
        if column_name in _NON_CONDITION_COLUMNS:
            continue
        if column_name not in anndata_object.obs.columns:
            continue

        column = anndata_object.obs[column_name]

        # Float columns (age) are numerical, not categorical conditions.
        if pd.api.types.is_float_dtype(column):
            continue

        # Collect non-missing unique string representations.
        for raw_val in column.dropna().unique():
            as_string = str(raw_val)
            if as_string and as_string not in ("", "nan"):
                all_values.add(as_string)

    return sorted(all_values)


def sort_values_for_legend(values: Any) -> List:
    """
    Sort a collection of values numerically when all can be cast to float, alphabetically otherwise.

    Use this wherever unique group / label values are sorted for display in legends,
    axes, or color maps — so that numeric class labels (e.g. age bins "0", "1", "10")
    appear in natural numeric order instead of lexicographic order.

    Arguments:
        values: Any iterable of values (strings, ints, floats, or mixed).

    Returns:
        Sorted list — numeric order if all values are numeric, alphabetical otherwise.
    """
    values_list = list(values)
    try:
        return sorted(values_list, key=lambda x: float(x))
    except (ValueError, TypeError):
        return sorted(values_list, key=str)


# Chained qualitative palettes used when a group count exceeds a single
# palette's swatches (e.g. 17 age values vs. Set2's 8). Each is a discrete,
# maximally-distinct ListedColormap — chaining them (the same convention
# scanpy/AnnData tooling uses for many-category data) gives 20+20+20 = 60
# genuinely distinct colors before any two groups must share a hue family.
# Preferred over a continuous colormap (e.g. viridis) because a smooth ramp
# makes visually neighboring values (close ages) hard to tell apart by eye.
_QUALITATIVE_OVERFLOW_CHAIN = ["tab20", "tab20b", "tab20c"]

# Last-resort fallback once even the chained qualitative palettes run out.
_CONTINUOUS_OVERFLOW_PALETTE = "viridis"


def _generate_auto_colors(n_colors: int, palette: str) -> List[str]:
    """
    Generate n_colors distinct hex color strings from a matplotlib colormap.

    Arguments:
        n_colors: Number of colors to generate. If 0, returns an empty list.
        palette: Matplotlib colormap name (e.g. "Set2", "tab10").

    Returns:
        List of hex color strings.
    """
    if n_colors == 0:
        return []

    colormap = plt.get_cmap(palette)

    # ListedColormap (qualitative palettes) has a finite `.N` swatches. Sampling
    # more points than that (e.g. 17 age values from Set2's 8 colors) wraps
    # multiple groups onto the same swatch. Chain in bigger discrete palettes
    # first so every group still gets its own clearly distinct color.
    if isinstance(colormap, mcolors.ListedColormap) and n_colors > colormap.N:
        chained_colors = []
        for chain_palette_name in _QUALITATIVE_OVERFLOW_CHAIN:
            chained_colors.extend(plt.get_cmap(chain_palette_name).colors)
        if n_colors <= len(chained_colors):
            return [mcolors.to_hex(color) for color in chained_colors[:n_colors]]
        # More groups than even the chained palettes can cover distinctly —
        # only then fall back to a continuous colormap.
        colormap = plt.get_cmap(_CONTINUOUS_OVERFLOW_PALETTE)

    # Sample colors evenly across the colormap range [0, 1].
    sample_points = np.linspace(0, 1, n_colors, endpoint=False)
    hex_colors = []
    for point in sample_points:
        r, g, b, _ = colormap(point)
        hex_colors.append(f"#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}")

    return hex_colors
