"""
distribution_plots.py

Feature-level distribution visualizations for TEM morphological data.

Three public functions are provided, all sharing the same ``group_by``
parameter pattern used by ``plot_radar()``, ``plot_umap()``, and the PCA
family:

* ``plot_histogram()`` — overlapping histograms (density-normalised) per group,
  one subplot per feature, arranged in a configurable grid.

* ``plot_density()`` — overlapping Gaussian KDE curves per group, with an
  optional shaded fill under each curve.

* ``plot_violin()`` — violin plots in two styles:
    - ``"grouped"`` — one violin per group value on the x-axis.
    - ``"split"``   — each violin is split between exactly two group conditions;
      requires an ``x_by`` column that defines the x-axis categories
      (e.g. species on x-axis, condition as left/right half of each violin).

Color assignment follows the existing priority order: PREFERRED_CONDITION_COLORS
from controlled_vocabulary → .uns['color_palette'] → automatic fallback.
"""

import math
import warnings
from typing import Dict, List, Literal, Optional, Tuple, Union

import anndata
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse
import scipy.stats
import seaborn as sns

from mito_marker.analysis._plot_context import get_species_label
from mito_marker.analysis.colors import get_subject_colors, sort_values_for_legend
from mito_marker.controlled_vocabulary import PREFERRED_CONDITION_COLORS

# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _get_group_labels(
    anndata_object: anndata.AnnData,
    group_by: Union[str, List[str]],
) -> pd.Series:
    """Return one group-label string per observation.

    When *group_by* is a list, values from each column are concatenated with
    ``" / "`` so that ``["specie", "condition"]`` produces labels such as
    ``"Zebrafish / Old"``.

    Args:
        anndata_object: AnnData whose ``.obs`` contains the requested columns.
        group_by: Single column name or list of column names in ``.obs``.

    Returns:
        A :class:`pandas.Series` of string labels aligned with ``.obs``.

    Raises:
        ValueError: If any requested column is missing from ``.obs``.
    """
    columns = [group_by] if isinstance(group_by, str) else list(group_by)

    for column in columns:
        if column not in anndata_object.obs.columns:
            raise ValueError(
                f"Column '{column}' not found in .obs. "
                f"Available columns: {list(anndata_object.obs.columns)}"
            )

    if len(columns) == 1:
        column = anndata_object.obs[columns[0]]
        # Cast to string for consistent color-map lookups, but keep missing
        # values as real NaN so downstream .dropna() calls still exclude them
        # (astype(str) would otherwise turn NaN into the literal string "nan").
        return column.astype(str).mask(column.isna())

    combined = anndata_object.obs[columns[0]].astype(str)
    for column in columns[1:]:
        combined = combined + " / " + anndata_object.obs[column].astype(str)
    combined.name = " / ".join(columns)
    return combined


def _resolve_group_order(
    present_values,
    order: Optional[List[str]],
) -> List[str]:
    """Resolve the display order of group values for one axis.

    Same 3-tier convention already used elsewhere in the codebase (see
    ``amhi.py``'s ``group_order`` handling and ``clustermap_plot.py``'s
    ``_resolve_display_order``): values listed in *order* and present in the
    data come first, in the given order; any other present values are
    appended afterwards via :func:`sort_values_for_legend`. No present value
    is ever dropped for being unlisted. Falls back to
    :func:`sort_values_for_legend` alone when *order* is ``None``.

    Args:
        present_values: Values actually present in the data (e.g. from
            ``group_labels.dropna().unique()``).
        order: Optional explicit order (e.g. a caller-supplied cell-type
            order). May include values absent from the data.

    Returns:
        Ordered list containing exactly the values in present_values.
    """
    present_list = list(present_values)
    if order is None:
        return sort_values_for_legend(present_list)
    listed = [value for value in order if value in present_list]
    unlisted = sort_values_for_legend(
        value for value in present_list if value not in order
    )
    return listed + unlisted


def _resolve_features(
    anndata_object: anndata.AnnData,
    features: Optional[List[str]],
) -> List[str]:
    """Return the list of features to plot.

    When *features* is ``None`` (default), all variables are returned except
    those flagged as non-analytical in ``.var['is_non_analytical']`` (if that
    column exists) or listed in ``TEM_NON_ANALYTICAL_FEATURES``.

    Args:
        anndata_object: Source AnnData.
        features: Explicit list of feature names, or ``None`` to use all
            analytical features.

    Returns:
        List of feature names present in ``anndata_object.var_names``.

    Raises:
        ValueError: If any explicitly requested feature is absent from
            ``anndata_object.var_names``.
    """
    if features is not None:
        missing = [f for f in features if f not in anndata_object.var_names]
        if missing:
            raise ValueError(
                f"Features not found in .var_names: {missing}. "
                f"Available: {list(anndata_object.var_names)}"
            )
        return features

    all_features = list(anndata_object.var_names)

    # Prefer the flag stored in .var when available (set by ingestion pipeline)
    if "is_non_analytical" in anndata_object.var.columns:
        return [
            f for f in all_features
            if not anndata_object.var.loc[f, "is_non_analytical"]
        ]

    # Fallback: use the controlled-vocabulary list of known non-analytical columns
    try:
        from mito_marker.controlled_vocabulary import TEM_NON_ANALYTICAL_FEATURES
        return [f for f in all_features if f not in TEM_NON_ANALYTICAL_FEATURES]
    except ImportError:
        return all_features


def _build_color_map(
    anndata_object: anndata.AnnData,
    group_labels: pd.Series,
) -> Dict[str, str]:
    """Map each unique group label to a hex color string.

    Priority order (same as radar_plot):
      1. PREFERRED_CONDITION_COLORS — researcher-defined overrides.
      2. .uns['color_palette']      — palette built by assign_color_palette().
      3. Auto-generated Set2 colors — fallback for values absent from both.

    Args:
        anndata_object: AnnData that may contain ``.uns['color_palette']``.
        group_labels: Series of group label strings (one per observation).

    Returns:
        Dictionary ``{label: hex_color}``.
    """
    unique_labels = sort_values_for_legend(group_labels.dropna().unique())
    stored_palette: Dict[str, str] = anndata_object.uns.get("color_palette", {})
    color_map: Dict[str, str] = {}
    missing_labels = []

    for label in unique_labels:
        key = str(label)
        if key in PREFERRED_CONDITION_COLORS:
            color_map[key] = PREFERRED_CONDITION_COLORS[key]
        elif key in stored_palette:
            color_map[key] = stored_palette[key]
        else:
            missing_labels.append(key)

    if missing_labels:
        auto_colors = get_subject_colors(missing_labels, palette="Set2")
        color_map.update(auto_colors)

    return color_map


def _get_data_matrix(anndata_object: anndata.AnnData) -> np.ndarray:
    """Return ``.X`` as a dense float32 numpy array.

    Handles both dense and sparse representations transparently.

    Args:
        anndata_object: Source AnnData.

    Returns:
        2-D numpy array of shape ``(n_obs, n_vars)``.
    """
    if scipy.sparse.issparse(anndata_object.X):
        return anndata_object.X.toarray().astype(np.float32)
    return np.asarray(anndata_object.X, dtype=np.float32)


def _make_grid(
    n_features: int,
    n_cols: int,
    figsize_per_plot: Tuple[float, float],
) -> Tuple[plt.Figure, np.ndarray]:
    """Create a matplotlib subplot grid for *n_features* plots.

    Args:
        n_features: Total number of subplots needed.
        n_cols: Number of columns in the grid.
        figsize_per_plot: ``(width, height)`` in inches for a single subplot.

    Returns:
        ``(figure, axes_array)`` where *axes_array* has shape
        ``(n_rows, n_cols)`` and is always 2-D (``squeeze=False``).
    """
    n_rows = math.ceil(n_features / n_cols)
    figure, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(figsize_per_plot[0] * n_cols, figsize_per_plot[1] * n_rows),
        squeeze=False,
    )
    return figure, axes


def _hide_empty_axes(axes: np.ndarray, n_features: int) -> None:
    """Turn off any unused axes at the end of the grid.

    Args:
        axes: 2-D axes array returned by :func:`_make_grid`.
        n_features: Number of subplots that were actually filled.
    """
    total_cells = axes.size
    for cell_index in range(n_features, total_cells):
        axes.flat[cell_index].set_visible(False)


def _sort_features_by_category(feature_names: List[str]) -> List[str]:
    """Reorder feature names into contiguous TEM_FEATURE_CATEGORY_ORDER blocks.

    Mirrors ``radar_plot.py``'s ``_sort_tem_channels_by_category`` — same
    category source (``TEM_FEATURE_CATEGORIES``), same "Other" fallback for
    an uncategorised feature, same warning behavior. Returns names directly
    (not indices) since callers here already hold a plain feature-name list.
    Import is local to this function because feature categorization is a
    TEM-specific concept, not something every caller of this generic
    distribution-plot module needs to depend on.

    Args:
        feature_names: Feature names to reorder.

    Returns:
        The same names, reordered so all features of one category are
        contiguous, in ``TEM_FEATURE_CATEGORY_ORDER`` (then "Other"). Order
        within a category is preserved (stable sort).
    """
    from mito_marker.controlled_vocabulary import (
        TEM_FEATURE_CATEGORIES,
        TEM_FEATURE_CATEGORY_ORDER,
    )

    category_rank_order = [*TEM_FEATURE_CATEGORY_ORDER, "Other"]
    unmapped_names = [name for name in feature_names if name not in TEM_FEATURE_CATEGORIES]
    if unmapped_names:
        warnings.warn(
            f"Feature(s) not found in TEM_FEATURE_CATEGORIES, grouped as "
            f"'Other': {unmapped_names}. Add them to TEM_FEATURE_SUBSETS in "
            f"controlled_vocabulary.py if they are genuine TEM features.",
            stacklevel=2,
        )

    def _category_rank(name: str) -> int:
        category = TEM_FEATURE_CATEGORIES.get(name, "Other")
        return category_rank_order.index(category)

    return sorted(feature_names, key=_category_rank)


def _get_feature_category_color(feature_name: str) -> str:
    """Return the ``TEM_FEATURE_CATEGORY_COLORS`` hex color for one feature.

    Falls back to the "Other" color for a feature absent from
    ``TEM_FEATURE_CATEGORIES``. Import is local — same reasoning as
    :func:`_sort_features_by_category`.

    Args:
        feature_name: Feature name to look up.

    Returns:
        A ``"#rrggbb"`` hex color string.
    """
    from mito_marker.controlled_vocabulary import (
        TEM_FEATURE_CATEGORIES,
        TEM_FEATURE_CATEGORY_COLORS,
    )

    category = TEM_FEATURE_CATEGORIES.get(feature_name, "Other")
    return TEM_FEATURE_CATEGORY_COLORS.get(category, TEM_FEATURE_CATEGORY_COLORS["Other"])


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------


def plot_histogram(
    anndata_object: anndata.AnnData,
    group_by: Union[str, List[str]],
    features: Optional[List[str]] = None,
    bins: int = 50,
    alpha: float = 0.6,
    n_cols: int = 3,
    figsize_per_plot: Tuple[float, float] = (5, 3),
    title: str = "",
    max_points_per_group: Optional[int] = None,
) -> None:
    """Plot overlapping density-normalised histograms for each morphological feature.

    One subplot is drawn per feature, arranged in a grid with *n_cols* columns.
    Each group defined by *group_by* is drawn in a distinct colour as a
    semi-transparent overlapping histogram.

    Args:
        anndata_object: AnnData containing feature matrix in ``.X`` and
            grouping metadata in ``.obs``.
        group_by: Single ``.obs`` column name or list of column names whose
            combined values define the groups (e.g. ``"condition"`` or
            ``["specie", "condition"]``).
        features: List of feature names from ``.var_names`` to plot. When
            ``None`` (default), all analytical features are plotted (non-
            analytical features declared in ``TEM_NON_ANALYTICAL_FEATURES``
            or flagged in ``.var['is_non_analytical']`` are excluded).
        bins: Number of histogram bins. Default ``50``.
        alpha: Opacity of each histogram bar (0 = transparent, 1 = opaque).
            Default ``0.6``.
        n_cols: Number of subplot columns in the grid. Default ``3``.
        figsize_per_plot: ``(width, height)`` in inches for each individual
            subplot. Default ``(5, 3)``.
        title: Optional super-title displayed above the full figure.
        max_points_per_group: Maximum number of observations sampled per group
            before plotting. When ``None`` (default), all observations are used.

    Returns:
        None. The figure is displayed inline (Jupyter / Colab compatible).

    Example::

        plot_histogram(tem_anndata, group_by="condition")
        plot_histogram(tem_anndata, group_by=["specie", "condition"],
                       features=["Mito_Area", "Mito_AR"], bins=30,
                       max_points_per_group=1000)
    """
    print("--- plot_histogram ---")
    print(f"  group_by            : {group_by}")
    print(f"  bins                : {bins}")
    print(f"  alpha               : {alpha}")
    print(f"  max_points_per_group: {max_points_per_group}")

    feature_list = _resolve_features(anndata_object, features)
    group_labels = _get_group_labels(anndata_object, group_by)
    color_map = _build_color_map(anndata_object, group_labels)
    data_matrix = _get_data_matrix(anndata_object)
    feature_index = {name: idx for idx, name in enumerate(anndata_object.var_names)}

    n_features = len(feature_list)
    print(f"  features to plot    : {n_features}")
    print(f"  groups              : {list(color_map.keys())}")

    figure, axes = _make_grid(n_features, n_cols, figsize_per_plot)

    unique_labels = sort_values_for_legend(group_labels.dropna().unique())

    for plot_index, feature_name in enumerate(feature_list):
        ax = axes.flat[plot_index]
        var_column_index = feature_index[feature_name]

        for label in unique_labels:
            mask = group_labels == label
            values = data_matrix[mask, var_column_index]
            # Remove NaN and infinite values before plotting
            values = values[np.isfinite(values)]
            if values.size == 0:
                continue
            if max_points_per_group is not None and values.size > max_points_per_group:
                rng = np.random.default_rng(seed=42)
                values = rng.choice(values, size=max_points_per_group, replace=False)
            ax.hist(
                values,
                bins=bins,
                density=True,
                alpha=alpha,
                color=color_map[str(label)],
                label=label,
            )

        ax.set_xlabel(feature_name)
        ax.set_ylabel("Density")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # Add legend only on the first subplot to avoid clutter
        if plot_index == 0:
            legend_title = (
                group_by if isinstance(group_by, str) else " × ".join(group_by)
            )
            ax.legend(title=legend_title, fontsize=8, title_fontsize=8)

    _hide_empty_axes(axes, n_features)

    species = get_species_label(anndata_object)
    species_prefix = f"[{species}]  " if species else ""
    group_by_label = group_by if isinstance(group_by, str) else " × ".join(group_by)
    auto_title = f"{species_prefix}Histogram — grouped by '{group_by_label}'"
    figure.suptitle(title if title else auto_title, fontsize=14, y=1.01)

    plt.tight_layout()
    plt.show()
    print("  Done.")


def plot_density(
    anndata_object: anndata.AnnData,
    group_by: Union[str, List[str]],
    features: Optional[List[str]] = None,
    fill: bool = True,
    fill_alpha: float = 0.2,
    line_alpha: float = 0.9,
    n_cols: int = 3,
    figsize_per_plot: Tuple[float, float] = (5, 3),
    title: str = "",
    max_points_per_group: Optional[int] = None,
    order: Optional[List[str]] = None,
) -> None:
    """Plot overlapping Gaussian KDE density curves for each morphological feature.

    One subplot is drawn per feature, arranged in a grid with *n_cols* columns.
    Each group defined by *group_by* is drawn as a smooth KDE line with an
    optional shaded fill below the curve.

    Args:
        anndata_object: AnnData containing feature matrix in ``.X`` and
            grouping metadata in ``.obs``.
        group_by: Single ``.obs`` column name or list of column names whose
            combined values define the groups.
        features: List of feature names from ``.var_names`` to plot. When
            ``None`` (default), all analytical features are plotted.
        fill: Whether to shade the area under each KDE curve. Default ``True``.
        fill_alpha: Opacity of the filled area (0–1). Default ``0.2``.
        line_alpha: Opacity of the KDE line (0–1). Default ``0.9``.
        n_cols: Number of subplot columns in the grid. Default ``3``.
        figsize_per_plot: ``(width, height)`` in inches for each individual
            subplot. Default ``(5, 3)``.
        title: Optional super-title displayed above the full figure.
        max_points_per_group: Maximum number of observations sampled per group
            before computing the KDE. When ``None`` (default), all observations
            are used.
        order: Optional explicit display order for the group values (e.g.
            ``["Dis", "Loop", "Pro", "Emb", "Sp", "Mu"]``). Values listed here
            and present in the data are drawn/legended first, in this order;
            any other present values are appended afterwards. Since a density
            plot has no categorical x-axis (groups are overlaid KDE curves,
            not separate x positions), this controls legend order and
            drawing order (later curves are drawn on top), not axis
            placement. When ``None`` (default), falls back to the automatic
            numeric-aware sort used everywhere else in this codebase.

    Returns:
        None. The figure is displayed inline (Jupyter / Colab compatible).

    Example::

        plot_density(tem_anndata, group_by="condition")
        plot_density(tem_anndata, group_by="condition", fill=False,
                     features=["Mito_Circularity", "Mito_Roundness"],
                     max_points_per_group=1000)
        plot_density(tem_anndata, group_by="cell_type",
                     order=["Dis", "Loop", "Pro", "Emb", "Sp", "Mu"])
    """
    print("--- plot_density ---")
    print(f"  group_by            : {group_by}")
    print(f"  fill                : {fill}")
    print(f"  fill_alpha          : {fill_alpha}")
    print(f"  max_points_per_group: {max_points_per_group}")
    print(f"  order               : {order}")

    feature_list = _resolve_features(anndata_object, features)
    group_labels = _get_group_labels(anndata_object, group_by)
    color_map = _build_color_map(anndata_object, group_labels)
    data_matrix = _get_data_matrix(anndata_object)
    feature_index = {name: idx for idx, name in enumerate(anndata_object.var_names)}

    n_features = len(feature_list)
    print(f"  features to plot    : {n_features}")
    print(f"  groups              : {list(color_map.keys())}")

    figure, axes = _make_grid(n_features, n_cols, figsize_per_plot)

    unique_labels = _resolve_group_order(group_labels.dropna().unique(), order)

    for plot_index, feature_name in enumerate(feature_list):
        ax = axes.flat[plot_index]
        var_column_index = feature_index[feature_name]

        for label in unique_labels:
            mask = group_labels == label
            values = data_matrix[mask, var_column_index]
            values = values[np.isfinite(values)]
            if max_points_per_group is not None and values.size > max_points_per_group:
                rng = np.random.default_rng(seed=42)
                values = rng.choice(values, size=max_points_per_group, replace=False)
            if values.size < 2:
                # KDE requires at least 2 data points
                continue

            kde = scipy.stats.gaussian_kde(values)
            # Build a fine x-grid spanning the data range with a small margin
            x_min = float(values.min())
            x_max = float(values.max())
            margin = (x_max - x_min) * 0.05 if x_max > x_min else 1.0
            x_grid = np.linspace(x_min - margin, x_max + margin, 300)
            kde_values = kde(x_grid)

            color = color_map[str(label)]
            ax.plot(x_grid, kde_values, color=color, alpha=line_alpha, label=label)
            if fill:
                ax.fill_between(x_grid, kde_values, alpha=fill_alpha, color=color)

        ax.set_xlabel(feature_name)
        ax.set_ylabel("Density")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        if plot_index == 0:
            legend_title = (
                group_by if isinstance(group_by, str) else " × ".join(group_by)
            )
            ax.legend(title=legend_title, fontsize=8, title_fontsize=8)

    _hide_empty_axes(axes, n_features)

    species = get_species_label(anndata_object)
    species_prefix = f"[{species}]  " if species else ""
    group_by_label = group_by if isinstance(group_by, str) else " × ".join(group_by)
    auto_title = f"{species_prefix}Density — grouped by '{group_by_label}'"
    figure.suptitle(title if title else auto_title, fontsize=14, y=1.01)

    plt.tight_layout()
    plt.show()
    print("  Done.")


def plot_violin(
    anndata_object: anndata.AnnData,
    group_by: Union[str, List[str]],
    features: Optional[List[str]] = None,
    style: Literal["split", "grouped"] = "grouped",
    x_by: Optional[str] = None,
    n_cols: int = 3,
    figsize_per_plot: Tuple[float, float] = (5, 4),
    title: str = "",
    max_points_per_group: Optional[int] = None,
    order: Optional[List[str]] = None,
    group_label_map: Optional[Dict[str, str]] = None,
    group_features_by_category: bool = False,
) -> None:
    """Plot violin distributions for each morphological feature.

    Two styles are available, controlled by the *style* parameter:

    **"grouped"** (default):
        One violin per group value on the x-axis. Each group defined by
        *group_by* appears as a separate, coloured violin. Suitable for any
        number of groups. When *x_by* is also provided, the x-axis shows the
        *x_by* categories instead and violins are coloured by *group_by* — use
        this when each x-axis category belongs to exactly one group (e.g.
        species coloured by lifespan class).

    **"split"**:
        Requires *x_by* (an ``.obs`` column for the x-axis categories, e.g.
        ``"specie"``) and *group_by* must resolve to **exactly 2 unique
        categories** (e.g. ``"Old"`` and ``"Young"``). Each violin is split
        down the middle: the left half represents one condition and the right
        half the other. Only use this style when every x-axis category has
        observations in **both** groups; otherwise use ``"grouped"`` with
        *x_by*.

    Args:
        anndata_object: AnnData containing feature matrix in ``.X`` and
            grouping metadata in ``.obs``.
        group_by: Single ``.obs`` column name or list of column names whose
            combined values define the colour / split groups (e.g.
            ``"condition"`` or ``["specie", "condition"]``).
        features: List of feature names from ``.var_names`` to plot. When
            ``None`` (default), all analytical features are plotted.
        style: ``"grouped"`` (default) or ``"split"``. See descriptions above.
        x_by: Optional ``.obs`` column used as the x-axis. In ``"split"``
            style, required (must resolve to exactly 2 groups). In
            ``"grouped"`` style, optional — when provided, the x-axis shows
            *x_by* categories and violins are coloured by *group_by*; when
            omitted, the x-axis shows the group values directly.
        n_cols: Number of subplot columns in the grid. Default ``3``.
        figsize_per_plot: ``(width, height)`` in inches for each individual
            subplot. Default ``(5, 4)``.
        title: Optional super-title displayed above the full figure.
        max_points_per_group: Maximum number of observations sampled per group
            before plotting. When ``None`` (default), all observations are used.
        order: Optional explicit display order for the *group_by* values
            (e.g. ``["Dis", "Loop", "Pro", "Emb", "Sp", "Mu"]``). Values
            listed here and present in the data come first, in this order
            (and are placed on the x-axis / split sides accordingly); any
            other present values are appended afterwards. Same 3-tier
            convention used in ``amhi.py`` / ``clustermap_plot.py`` — no
            present value is ever dropped for being unlisted. When ``None``
            (default), falls back to the automatic numeric-aware sort used
            everywhere else in this codebase.
        group_label_map: Optional ``{original_value: display_label}`` remap
            applied only to what is drawn (x-axis ticks, legend, split
            sides) — internal filtering/coloring still resolves through the
            original values, so *order* and ``PREFERRED_CONDITION_COLORS``
            keys should still use the original values. Use this to make a
            single-condition panel self-explanatory without reading a
            separate legend (e.g. ``{"Dis": "Dis-S", "Pro": "Pro-S"}`` on a
            Chemical-Stress-only subset, so the x-axis itself shows "-S").
            Values absent from the map are shown unchanged.
        group_features_by_category: When ``True``, reorders *feature_list*
            into contiguous ``TEM_FEATURE_CATEGORY_ORDER`` blocks (Size,
            Shape, Intensity, Cristae Orientation, then "Other") and colors
            each subplot's title by its category
            (``TEM_FEATURE_CATEGORY_COLORS``), with a shared category legend
            — the same categorization already used by ``plot_radar()``,
            applied here to a multi-feature grid instead of a single polar
            plot. Intended for the "all metrics" grid views (one subplot per
            feature) — has no effect when *features* names a single feature.
            Default ``False`` (grid order unchanged, as before).

    Returns:
        None. The figure is displayed inline (Jupyter / Colab compatible).

    Raises:
        ValueError: If *style* is ``"split"`` and *x_by* is not provided.
        ValueError: If *style* is ``"split"`` and *group_by* does not resolve
            to exactly 2 unique categories.

    Example::

        # Grouped — condition on x-axis, one violin per condition
        plot_violin(tem_anndata, group_by="condition", style="grouped")

        # Grouped with x_by — species on x-axis, coloured by lifespan group
        # (use when each species belongs to exactly one group)
        plot_violin(tem_anndata, group_by="lifespan", style="grouped", x_by="specie")

        # Split — species on x-axis, violin split by condition (exactly 2)
        plot_violin(tem_anndata, group_by="condition",
                    style="split", x_by="specie")

        # Explicit order + self-explanatory labels on a single-condition subset
        plot_violin(chemical_stress_only_anndata, group_by="cell_type",
                    order=["Dis", "Loop", "Pro", "Emb", "Sp", "Mu"],
                    group_label_map={"Dis": "Dis-S", "Pro": "Pro-S"})

        # "All metrics" grid, features grouped by category (Size/Shape/…)
        plot_violin(tem_anndata, group_by="condition",
                    group_features_by_category=True)
    """
    print("--- plot_violin ---")
    print(f"  group_by                   : {group_by}")
    print(f"  style                      : {style}")
    print(f"  x_by                       : {x_by}")
    print(f"  max_points_per_group       : {max_points_per_group}")
    print(f"  order                      : {order}")
    print(f"  group_label_map            : {group_label_map}")
    print(f"  group_features_by_category : {group_features_by_category}")

    # --- Validate x_by requirements ---
    if style == "split" and x_by is None:
        raise ValueError(
            "style='split' requires the x_by parameter (the .obs column to "
            "place on the x-axis, e.g. x_by='specie')."
        )
    if x_by is not None and x_by not in anndata_object.obs.columns:
        raise ValueError(
            f"x_by column '{x_by}' not found in .obs. "
            f"Available columns: {list(anndata_object.obs.columns)}"
        )

    feature_list = _resolve_features(anndata_object, features)
    if group_features_by_category:
        feature_list = _sort_features_by_category(feature_list)
    group_labels = _get_group_labels(anndata_object, group_by)

    # --- Down-sample each group to max_points_per_group rows ---
    if max_points_per_group is not None:
        rng = np.random.default_rng(seed=42)
        keep_indices: List[int] = []
        for group_value in group_labels.dropna().unique():
            group_indices = np.where(group_labels == group_value)[0]
            if group_indices.size > max_points_per_group:
                group_indices = rng.choice(
                    group_indices, size=max_points_per_group, replace=False
                )
            keep_indices.extend(group_indices.tolist())
        keep_indices = sorted(keep_indices)
        sampled_anndata = anndata_object[keep_indices, :]
        group_labels = group_labels.iloc[keep_indices].reset_index(drop=True)
    else:
        sampled_anndata = anndata_object

    color_map = _build_color_map(anndata_object, group_labels)
    data_matrix = _get_data_matrix(sampled_anndata)
    feature_index = {name: idx for idx, name in enumerate(anndata_object.var_names)}

    unique_groups = _resolve_group_order(group_labels.dropna().unique(), order)

    if style == "split" and len(unique_groups) != 2:
        raise ValueError(
            f"style='split' requires group_by to resolve to exactly 2 unique "
            f"categories, but found {len(unique_groups)}: {unique_groups}. "
            f"Use style='grouped' for more than 2 groups."
        )

    # Warn when split style is used but some x-categories have data for only
    # one group — those violins will appear as half-violins.
    if style == "split":
        x_by_values = sampled_anndata.obs[x_by].values
        for x_val in np.unique(x_by_values):
            mask = x_by_values == x_val
            groups_present = set(group_labels[mask].dropna().unique())
            if len(groups_present) < 2:
                warnings.warn(
                    f"x_by value '{x_val}' has data for only one group "
                    f"({groups_present}), which produces a half-violin. "
                    f"Consider using style='grouped' with x_by='{x_by}' "
                    f"for full violins when groups are mutually exclusive.",
                    stacklevel=2,
                )
                break  # One warning is enough

    # --- Apply the display-label remap (drawing only — filtering/coloring
    # above already resolved through the original values) ---
    if group_label_map is not None:
        group_labels_for_plot = group_labels.map(
            lambda value: group_label_map.get(value, value) if pd.notna(value) else value
        )
        unique_groups = [group_label_map.get(value, value) for value in unique_groups]
        color_map = {
            group_label_map.get(original_key, original_key): color_value
            for original_key, color_value in color_map.items()
        }
    else:
        group_labels_for_plot = group_labels

    n_features = len(feature_list)
    print(f"  features to plot: {n_features}")
    print(f"  groups          : {unique_groups}")

    group_label_column = (
        group_by if isinstance(group_by, str) else " / ".join(group_by)
    )
    # Ensure the column name is a plain string for seaborn
    group_label_column_name = str(group_label_column)

    figure, axes = _make_grid(n_features, n_cols, figsize_per_plot)

    for plot_index, feature_name in enumerate(feature_list):
        ax = axes.flat[plot_index]
        var_column_index = feature_index[feature_name]
        feature_values = data_matrix[:, var_column_index]

        if style == "grouped":
            if x_by is not None:
                # x-axis = x_by categories, violins coloured by group_by.
                # This produces full violins even when each x-category belongs
                # to only one group (avoids the half-violin issue of split style).
                plot_dataframe = pd.DataFrame(
                    {
                        feature_name: feature_values,
                        group_label_column_name: group_labels_for_plot.values,
                        x_by: sampled_anndata.obs[x_by].values,
                    }
                )
                sns.violinplot(
                    data=plot_dataframe,
                    x=x_by,
                    y=feature_name,
                    hue=group_label_column_name,
                    hue_order=unique_groups,
                    palette=color_map,
                    inner="box",
                    ax=ax,
                )
                ax.set_xlabel(x_by)
                # Remove per-subplot legend; shared figure legend added below
                legend = ax.get_legend()
                if legend is not None:
                    legend.remove()
            else:
                # Default: x-axis = group values, one violin per group.
                # seaborn >= 0.12 requires `hue` when `palette` is used;
                # setting hue=x and legend=False avoids the FutureWarning.
                # `order`/`hue_order` both pin down the same axis here, so the
                # x-axis actually follows `unique_groups` instead of seaborn's
                # own first-occurrence default.
                plot_dataframe = pd.DataFrame(
                    {
                        feature_name: feature_values,
                        group_label_column_name: group_labels_for_plot.values,
                    }
                )
                sns.violinplot(
                    data=plot_dataframe,
                    x=group_label_column_name,
                    y=feature_name,
                    order=unique_groups,
                    hue=group_label_column_name,
                    hue_order=unique_groups,
                    palette=color_map,
                    inner="box",
                    legend=False,
                    ax=ax,
                )
                ax.set_xlabel(group_label_column_name)
            ax.set_ylabel(feature_name)

        else:  # style == "split"
            plot_dataframe = pd.DataFrame(
                {
                    feature_name: feature_values,
                    group_label_column_name: group_labels_for_plot.values,
                    x_by: sampled_anndata.obs[x_by].values,
                }
            )
            # seaborn split violin: each half = one hue category — hue_order
            # controls which condition lands on the left vs. right half.
            sns.violinplot(
                data=plot_dataframe,
                x=x_by,
                y=feature_name,
                hue=group_label_column_name,
                hue_order=unique_groups,
                split=True,
                palette=color_map,
                inner="quartile",
                ax=ax,
            )
            ax.set_xlabel(x_by)
            ax.set_ylabel(feature_name)
            # Remove per-subplot legend; a single shared legend is added below
            legend = ax.get_legend()
            if legend is not None:
                legend.remove()

        if group_features_by_category:
            ax.set_title(
                feature_name,
                fontsize=9,
                color=_get_feature_category_color(feature_name),
                fontweight="bold",
            )
        else:
            ax.set_title(feature_name, fontsize=9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    _hide_empty_axes(axes, n_features)

    legend_title = group_by if isinstance(group_by, str) else " × ".join(group_by)
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=color_map[str(label)])
        for label in unique_groups
    ]

    # grouped + x_by: shared figure-level legend (same pattern as split)
    if style == "grouped" and x_by is not None and n_features > 0:
        figure.legend(
            handles,
            unique_groups,
            title=legend_title,
            fontsize=9,
            title_fontsize=9,
            loc="upper right",
        )
    # grouped without x_by: legend on first subplot only
    elif style == "grouped" and n_features > 0:
        axes.flat[0].legend(
            handles,
            unique_groups,
            title=legend_title,
            fontsize=8,
            title_fontsize=8,
            loc="upper right",
        )

    # For split style, add a single shared legend outside the grid
    if style == "split" and n_features > 0:
        legend_title = (
            group_by if isinstance(group_by, str) else " × ".join(group_by)
        )
        handles = [
            plt.Rectangle((0, 0), 1, 1, color=color_map[str(label)])
            for label in unique_groups
        ]
        figure.legend(
            handles,
            unique_groups,
            title=legend_title,
            fontsize=9,
            title_fontsize=9,
            loc="upper right",
        )

    # Category legend — separate from the group legend above (different
    # corner) so both can coexist without overlapping.
    if group_features_by_category and n_features > 0:
        from mito_marker.controlled_vocabulary import (
            TEM_FEATURE_CATEGORIES,
            TEM_FEATURE_CATEGORY_COLORS,
            TEM_FEATURE_CATEGORY_ORDER,
        )

        categories_present = {
            TEM_FEATURE_CATEGORIES.get(name, "Other") for name in feature_list
        }
        category_display_order = [
            category
            for category in [*TEM_FEATURE_CATEGORY_ORDER, "Other"]
            if category in categories_present
        ]
        category_handles = [
            plt.Rectangle(
                (0, 0), 1, 1,
                color=TEM_FEATURE_CATEGORY_COLORS.get(
                    category, TEM_FEATURE_CATEGORY_COLORS["Other"]
                ),
            )
            for category in category_display_order
        ]
        figure.legend(
            category_handles,
            category_display_order,
            title="Parameter category",
            fontsize=9,
            title_fontsize=9,
            loc="upper left",
        )

    species = get_species_label(anndata_object)
    species_prefix = f"[{species}]  " if species else ""
    group_by_label = group_by if isinstance(group_by, str) else " × ".join(group_by)
    auto_title = f"{species_prefix}Violin — grouped by '{group_by_label}'"
    figure.suptitle(title if title else auto_title, fontsize=14, y=1.01)

    plt.tight_layout()
    plt.show()
    print("  Done.")
