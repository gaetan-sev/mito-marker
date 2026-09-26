"""
age_plot.py

Scatter plot of a morphological feature against a numerical age-like .obs column.

Provides one public function:

* ``plot_feature_vs_age()`` — for a single feature and a single numerical .obs
  column (typically age in days, months, or years), aggregates all observations
  that share the same subject into one representative point (median by default),
  then draws a scatter plot with a linear regression trend line and a shaded 95 %
  confidence interval.

  Pearson r and Spearman ρ are annotated on each group sub-line and on the
  overall regression.

  An optional ``color_by`` parameter colours points by a categorical .obs column
  (e.g. ``"condition"``, ``"specie"``), using the same colour-priority order
  as every other plot in this package:
    1. PREFERRED_CONDITION_COLORS   — researcher-defined overrides.
    2. .uns['color_palette']        — palette built by assign_color_palette().
    3. Automatic Set2 fallback.

Three aggregation modes are available via ``subject_id_column``:

* ``subject_id_column=None``         → one dot per raw observation (no aggregation).
* ``subject_id_column="subject_ID"`` → one aggregated dot per unique subject.
* ``subject_id_column=["001","002"]``→ filter to those subject IDs (matched
  against the ``"subject_ID"`` column), then one aggregated dot per subject.

Typical usage in a notebook:
    from mito_marker.analysis import plot_feature_vs_age

    # All subjects, one dot each (median of all mitochondria per subject).
    plot_feature_vs_age(tem_anndata, "Mito_AR", "age_days", color_by="condition")

    # Only subjects 001 and 003.
    plot_feature_vs_age(tem_anndata, "Mito_AR", "age_days",
                        subject_id_column=["001", "003"])

    # No aggregation — one dot per mitochondrion.
    plot_feature_vs_age(tem_anndata, "Mito_AR", "age_days",
                        subject_id_column=None)
"""

from typing import Dict, List, Optional, Tuple, Union

import anndata
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse
import scipy.stats

from mito_marker.analysis._plot_context import (
    get_run_context_console_text,
    get_run_context_footer_text,
)
from mito_marker.analysis.colors import get_subject_colors, sort_values_for_legend
from mito_marker.controlled_vocabulary import PREFERRED_CONDITION_COLORS

# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _build_color_map(
    anndata_object: anndata.AnnData,
    unique_labels: List[str],
) -> Dict[str, str]:
    """Map each unique group label to a hex colour string.

    Priority order:
      1. PREFERRED_CONDITION_COLORS — researcher-defined overrides.
      2. .uns['color_palette']      — palette built by assign_color_palette().
      3. Automatic Set2 fallback.

    Args:
        anndata_object: AnnData that may contain ``.uns['color_palette']``.
        unique_labels: Sorted list of unique group label strings.

    Returns:
        Dictionary ``{label: hex_color}``.
    """
    stored_palette: Dict[str, str] = anndata_object.uns.get("color_palette", {})
    color_map: Dict[str, str] = {}
    missing_labels: List[str] = []

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


def _compute_regression_band(
    x_values: np.ndarray,
    y_values: np.ndarray,
    x_grid: np.ndarray,
    confidence_level: float = 0.95,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute linear regression prediction and confidence band over a grid.

    Uses the Student-t confidence interval on the mean prediction:
        y_hat ± t * sqrt( MSE * (1/n + (x - x_mean)^2 / SSD) )

    This is the confidence interval on the mean response, not a prediction
    interval for a single future observation.

    Args:
        x_values:         1-D predictor array (one value per data point).
        y_values:         1-D response array (one value per data point).
        x_grid:           1-D array of x positions at which to evaluate the band.
        confidence_level: Coverage probability (default 0.95 → 95 %).

    Returns:
        Tuple ``(y_hat, lower_bound, upper_bound)`` — each a 1-D array of
        length ``len(x_grid)``.
    """
    n = len(x_values)
    slope, intercept, _, _, _ = scipy.stats.linregress(x_values, y_values)
    y_hat = slope * x_grid + intercept

    residuals = y_values - (slope * x_values + intercept)
    mean_squared_error = float(np.sum(residuals ** 2)) / max(n - 2, 1)

    x_mean = float(np.mean(x_values))
    sum_of_squared_deviations = float(np.sum((x_values - x_mean) ** 2))

    if sum_of_squared_deviations > 0:
        standard_error_grid = np.sqrt(
            mean_squared_error * (
                1.0 / n + (x_grid - x_mean) ** 2 / sum_of_squared_deviations
            )
        )
    else:
        standard_error_grid = np.zeros_like(x_grid)

    alpha = 1.0 - confidence_level
    t_critical = scipy.stats.t.ppf(1.0 - alpha / 2.0, df=max(n - 2, 1))

    return y_hat, y_hat - t_critical * standard_error_grid, y_hat + t_critical * standard_error_grid


def _resolve_data_layer(
    anndata_object: anndata.AnnData,
    use_active_layer: bool,
) -> Optional[str]:
    """Return the layer name to read feature values from, or None for raw .X.

    When *use_active_layer* is True, the layer stored in
    ``.uns['analysis_config']['active_layer']`` is used.  If no active layer
    is configured, falls back to raw ``.X`` (returns None).

    Args:
        anndata_object:   Source AnnData.
        use_active_layer: Whether to honour the active normalization layer.

    Returns:
        Layer name string, or ``None`` to indicate raw ``.X``.
    """
    if not use_active_layer:
        return None
    config = anndata_object.uns.get("analysis_config", {})
    return config.get("active_layer", None)  # None → raw .X


def _get_feature_values(
    anndata_object: anndata.AnnData,
    feature_name: str,
    layer: Optional[str] = None,
) -> np.ndarray:
    """Extract a single feature column from .X or a named layer as a 1-D float64 array.

    Args:
        anndata_object: Source AnnData.
        feature_name:   Must exist in ``anndata_object.var_names``.
        layer:          Layer name to read from, or ``None`` for raw ``.X``.

    Returns:
        1-D numpy array of shape ``(n_obs,)``.
    """
    feature_index = list(anndata_object.var_names).index(feature_name)
    if layer is not None and layer in anndata_object.layers:
        data = anndata_object.layers[layer]
    else:
        data = anndata_object.X
    if scipy.sparse.issparse(data):
        column = data[:, feature_index].toarray().ravel()
    else:
        column = np.asarray(data[:, feature_index]).ravel()
    return column.astype(np.float64)


def _build_observation_dataframe(
    anndata_object: anndata.AnnData,
    feature_values: np.ndarray,
    age_obs_column: str,
    color_by: Optional[str],
    subject_id_column: Optional[str] = None,
) -> pd.DataFrame:
    """Assemble a tidy DataFrame with one row per AnnData observation.

    Args:
        anndata_object:    Source AnnData.
        feature_values:    1-D float array aligned with .obs rows.
        age_obs_column:    Column name in .obs for the x-axis values.
        color_by:          Optional .obs column for colour grouping, or ``None``.
        subject_id_column: Optional .obs column whose values identify subjects.
                           When provided, a ``subject_id`` column is included in
                           the output so that ``_aggregate_per_subject`` can
                           group without relying on index alignment.

    Returns:
        DataFrame with columns: ``age``, ``feature``, optionally
        ``color_group``, and optionally ``subject_id``.
        Rows with NaN in ``age`` or ``feature`` are dropped.
    """
    observation_dataframe = pd.DataFrame(
        {
            "age": pd.to_numeric(anndata_object.obs[age_obs_column].values, errors="coerce"),
            "feature": feature_values,
        }
    )
    if color_by is not None:
        observation_dataframe["color_group"] = anndata_object.obs[color_by].values
    if subject_id_column is not None and subject_id_column in anndata_object.obs.columns:
        observation_dataframe["subject_id"] = anndata_object.obs[subject_id_column].values

    return observation_dataframe.dropna(subset=["age", "feature"])


def _aggregate_per_subject(
    observation_dataframe: pd.DataFrame,
    aggregation: str,
) -> pd.DataFrame:
    """Reduce per-observation rows to one row per unique subject.

    Expects ``observation_dataframe`` to contain a ``subject_id`` column
    (produced by ``_build_observation_dataframe`` when ``subject_id_column``
    is supplied).  For each unique subject, the feature column is summarised
    using *aggregation* and the first age value is kept (age is assumed
    constant within a subject).

    Args:
        observation_dataframe: Must include columns ``subject_id``, ``age``,
                               and ``feature``, and optionally ``color_group``.
        aggregation:           ``"median"`` or ``"mean"``.

    Returns:
        DataFrame with columns ``age``, ``feature``, and optionally
        ``color_group``.  One row per unique subject.
    """
    aggregation_function = np.median if aggregation == "median" else np.mean

    per_subject_rows = []
    for subject_id, group in observation_dataframe.groupby("subject_id", sort=True):
        row: Dict = {
            "age": float(group["age"].iloc[0]),
            "feature": float(aggregation_function(group["feature"].values)),
        }
        if "color_group" in group.columns:
            row["color_group"] = str(group["color_group"].iloc[0])
        per_subject_rows.append(row)

    return pd.DataFrame(per_subject_rows).dropna(subset=["age", "feature"])


# ---------------------------------------------------------------------------
# Public function
# ---------------------------------------------------------------------------


def plot_feature_vs_age(
    anndata_object: anndata.AnnData,
    feature_name: str,
    age_obs_column: str,
    color_by: Optional[str] = None,
    subject_id_column: Optional[Union[str, List[str]]] = "subject_ID",
    aggregation: str = "median",
    use_active_layer: bool = True,
    show_points: bool = False,
    show_trend_line: bool = False,
    show_confidence_band: bool = True,
    show_average_curve: bool = True,
    group_x_offset_fraction: float = 0.3,
    y_clip_percentile: Tuple[float, float] = (1.0, 99.0),
    point_size: float = 3.0,
    point_alpha: float = 0.25,
    average_curve_linewidth: float = 2.5,
    figsize: Tuple[float, float] = (9.0, 6.0),
) -> matplotlib.figure.Figure:
    """Scatter plot of a morphological feature against a numerical age column.

    **What each point represents** depends on ``subject_id_column``:

    * ``subject_id_column=None``          — one dot per raw observation (one
      cytometry event or one mitochondrion).  No aggregation is applied.  With
      large datasets (> 100 k observations) points will overlap heavily; the
      mean curve (``show_mean_curve=True``) is then the most informative visual.
    * ``subject_id_column="subject_ID"``  — one aggregated dot per unique subject
      (all subjects included).
    * ``subject_id_column=["001", "003"]``— filter to those subject IDs, then
      one aggregated dot per subject.

    **Data source**: by default the function reads from the active normalisation
    layer stored in ``.uns['analysis_config']['active_layer']`` (set by
    ``transform_and_normalize()``).  Set ``use_active_layer=False`` to plot
    raw ``.X`` values instead.

    **Average curve**: when ``show_average_curve=True`` (default), a line is
    drawn through the **mean** y-value at each unique age.  One curve per group
    when ``color_by`` is set.

    **Points**: hidden by default (``show_points=False``) so the curve is the
    primary visual and the y-axis scales tightly around it.  Set
    ``show_points=True`` to overlay the individual scatter cloud; the y-axis
    is then clipped to ``y_clip_percentile`` so outliers do not crush the
    scale.

    **Side-by-side groups**: when ``color_by`` is provided, curves (and points
    when shown) for each group are horizontally offset from the true age value
    so that groups are clearly separated at each age tick.

    **X-axis ticks**: only the actual age values present in the data are shown.

    Arguments:
        anndata_object:          AnnData whose .X / layers contain *feature_name*
                                 and whose .obs contains *age_obs_column*.
        feature_name:            Feature to plot on the y-axis.  Must exist in
                                 ``anndata_object.var_names``.
        age_obs_column:          Numerical .obs column for the x-axis.
        color_by:                Optional categorical .obs column used to colour
                                 curves and points (e.g. ``"condition"``,
                                 ``"diet"``).
        subject_id_column:       Controls the aggregation mode — see above.
                                 Defaults to ``"subject_ID"``.
        aggregation:             ``"median"`` (default) or ``"mean"``.
                                 Only used when *subject_id_column* is not
                                 ``None``.
        use_active_layer:        When ``True`` (default), reads values from the
                                 active normalisation layer in
                                 ``.uns['analysis_config']['active_layer']``.
                                 When ``False``, reads raw ``.X``.
        show_points:             Whether to draw individual scatter points
                                 (default ``False`` — curves only).
        show_trend_line:         Whether to draw linear regression lines
                                 (default ``False``).
        show_confidence_band:    Whether to shade the 95 % confidence band.
                                 Only effective when *show_trend_line* is
                                 ``True``.
        show_average_curve:      Whether to draw a line through the mean
                                 y-value at each unique age (one per group).
        group_x_offset_fraction: Fraction of the minimum age gap used as total
                                 side-by-side spread when ``color_by`` is set
                                 (default 0.3).
        y_clip_percentile:       ``(low, high)`` percentiles used to set the
                                 y-axis limits when ``show_points=True``
                                 (default ``(1.0, 99.0)``).  Ignored when
                                 ``show_points=False`` (axis scales to curves).
                                 Pass ``(0.0, 100.0)`` to disable clipping.
        point_size:              Scatter marker area in matplotlib units
                                 (default 3).  Only used when
                                 ``show_points=True``.
        point_alpha:             Opacity of scatter markers, 0–1 (default 0.25).
                                 Only used when ``show_points=True``.
        average_curve_linewidth: Line width of the average curve (default 2.5).
        figsize:                 Figure size in inches ``(width, height)``.

    Returns:
        The :class:`matplotlib.figure.Figure` so the caller can save or further
        customise it.

    Raises:
        ValueError: If *feature_name* is not in ``anndata_object.var_names``.
        ValueError: If *age_obs_column* is not in ``anndata_object.obs.columns``.
        ValueError: If *color_by* is provided but absent from ``.obs``.
        ValueError: If *aggregation* is not ``"median"`` or ``"mean"``.
    """
    # ---- Input validation -----------------------------------------------------

    if feature_name not in anndata_object.var_names:
        raise ValueError(
            f"Feature '{feature_name}' not found in .var_names. "
            f"Available: {list(anndata_object.var_names)}"
        )
    if age_obs_column not in anndata_object.obs.columns:
        raise ValueError(
            f"'{age_obs_column}' not found in .obs. "
            f"Available columns: {list(anndata_object.obs.columns)}"
        )
    if color_by is not None and color_by not in anndata_object.obs.columns:
        raise ValueError(
            f"color_by column '{color_by}' not found in .obs. "
            f"Available columns: {list(anndata_object.obs.columns)}"
        )
    if aggregation not in ("median", "mean"):
        raise ValueError(f"aggregation must be 'median' or 'mean', got '{aggregation}'.")

    # Resolve the mode from subject_id_column type.
    # mode="none"    → no aggregation, one dot per observation
    # mode="column"  → aggregate by a named .obs column
    # mode="filter"  → filter to a list of subject ID values, then aggregate
    if subject_id_column is None:
        mode = "none"
    elif isinstance(subject_id_column, list):
        mode = "filter"
        subject_id_filter_values: List[str] = [str(v) for v in subject_id_column]
        grouping_column = "subject_ID"
    else:
        mode = "column"
        grouping_column = subject_id_column

    # ---- Console header -------------------------------------------------------

    print("=" * 60)
    print("PLOT: feature vs age")
    print("-" * 60)
    print(f"  Feature (y-axis)   : {feature_name}")
    print(f"  Age column (x-axis): {age_obs_column}")
    print(f"  Colour by          : {color_by if color_by else 'none'}")
    if mode == "none":
        print("  Aggregation mode   : none (one dot per observation)")
    elif mode == "column":
        print(f"  Aggregation mode   : per subject — column '{grouping_column}', {aggregation}")
    else:
        n_requested = len(subject_id_filter_values)
        print(
            f"  Aggregation mode   : filtered — {n_requested} subject(s) from"
            f" column '{grouping_column}', {aggregation}"
        )
    print(get_run_context_console_text(anndata_object))

    # ---- Resolve data source (active layer vs raw .X) -------------------------

    active_layer = _resolve_data_layer(anndata_object, use_active_layer)
    layer_label = active_layer if active_layer else "raw .X"
    print(f"  Data source        : {layer_label}")

    # ---- Extract feature values -----------------------------------------------

    feature_values = _get_feature_values(anndata_object, feature_name, layer=active_layer)

    # ---- Build per-observation DataFrame, then apply aggregation mode ---------

    if mode == "none":
        # One dot per raw observation — no subject filtering or aggregation.
        plot_dataframe = _build_observation_dataframe(
            anndata_object, feature_values, age_obs_column, color_by
        )
        n_points_label = f"{len(plot_dataframe):,} observations (no aggregation)"

    elif mode == "column":
        if grouping_column not in anndata_object.obs.columns:
            # Column absent — fall back to no aggregation with a warning.
            print(
                f"WARNING: subject_id_column='{grouping_column}' not found in .obs. "
                "Falling back to one dot per observation."
            )
            plot_dataframe = _build_observation_dataframe(
                anndata_object, feature_values, age_obs_column, color_by
            )
            n_points_label = f"{len(plot_dataframe):,} observations (column not found, no aggregation)"
        else:
            observation_dataframe = _build_observation_dataframe(
                anndata_object, feature_values, age_obs_column, color_by,
                subject_id_column=grouping_column,
            )
            plot_dataframe = _aggregate_per_subject(observation_dataframe, aggregation)
            n_unique_subjects = anndata_object.obs[grouping_column].nunique()
            n_points_label = (
                f"{anndata_object.n_obs:,} obs → {len(plot_dataframe)} subjects "
                f"({n_unique_subjects} unique, {aggregation})"
            )

    else:  # mode == "filter"
        if grouping_column not in anndata_object.obs.columns:
            raise ValueError(
                f"subject_id_column list provided but '{grouping_column}' not found in .obs. "
                f"Available columns: {list(anndata_object.obs.columns)}"
            )
        # Filter AnnData rows to the requested subject IDs before building the DataFrame.
        subject_id_obs_series = anndata_object.obs[grouping_column].astype(str)
        filter_mask = subject_id_obs_series.isin(subject_id_filter_values)
        n_matched = int(filter_mask.sum())
        if n_matched == 0:
            print(
                f"WARNING: none of the {len(subject_id_filter_values)} requested "
                f"subject IDs were found in .obs['{grouping_column}']. "
                "The plot will be empty."
            )
        filtered_anndata = anndata_object[filter_mask].copy()
        filtered_feature_values = _get_feature_values(
            filtered_anndata, feature_name, layer=active_layer
        )

        observation_dataframe = _build_observation_dataframe(
            filtered_anndata, filtered_feature_values, age_obs_column, color_by,
            subject_id_column=grouping_column,
        )
        plot_dataframe = _aggregate_per_subject(observation_dataframe, aggregation
        )
        n_points_label = (
            f"filtered to {len(subject_id_filter_values)} requested subjects → "
            f"{len(plot_dataframe)} matched, {aggregation}"
        )

    print(f"  Data points        : {n_points_label}")

    if len(plot_dataframe) < 3:
        print(
            "WARNING: fewer than 3 data points — trend line statistics will not "
            "be meaningful."
        )

    # ---- Build colour map -----------------------------------------------------

    if color_by is not None and "color_group" in plot_dataframe.columns:
        unique_groups: List[str] = sort_values_for_legend(
            str(v) for v in plot_dataframe["color_group"].dropna().unique()
        )
        color_map = _build_color_map(anndata_object, unique_groups)
    else:
        unique_groups = ["all"]
        color_map = {"all": "#4878CF"}

    # ---- Global arrays for regression ----------------------------------------

    all_x = plot_dataframe["age"].values.astype(float)
    all_y = plot_dataframe["feature"].values.astype(float)
    valid_mask = np.isfinite(all_x) & np.isfinite(all_y)
    all_x_valid = all_x[valid_mask]
    all_y_valid = all_y[valid_mask]

    global_pearson_r: Optional[float] = None
    global_pearson_p: Optional[float] = None
    global_spearman_r: Optional[float] = None
    global_spearman_p: Optional[float] = None
    global_slope: Optional[float] = None

    if len(all_x_valid) >= 3:
        slope, _, r_value, p_value, _ = scipy.stats.linregress(all_x_valid, all_y_valid)
        global_slope = float(slope)
        global_pearson_r = float(r_value)
        global_pearson_p = float(p_value)
        spearman_result = scipy.stats.spearmanr(all_x_valid, all_y_valid)
        global_spearman_r = float(spearman_result.statistic)
        global_spearman_p = float(spearman_result.pvalue)

    # ---- Pre-compute side-by-side x-offsets per group -------------------------
    # Each group is shifted horizontally so groups appear side-by-side at each
    # age value rather than overlapping.  The offset is proportional to the
    # minimum gap between two consecutive age values.

    unique_ages_sorted = np.sort(
        plot_dataframe["age"].dropna().unique().astype(float)
    )
    if len(unique_ages_sorted) > 1:
        min_age_gap = float(np.min(np.diff(unique_ages_sorted)))
    else:
        min_age_gap = 1.0

    n_groups = len(unique_groups)
    if n_groups > 1:
        total_dodge_width = min_age_gap * group_x_offset_fraction
        half_dodge = total_dodge_width / 2.0
        # Space groups evenly: outermost groups at ±half_dodge.
        group_x_offsets: List[float] = [
            -half_dodge + i * total_dodge_width / (n_groups - 1)
            for i in range(n_groups)
        ]
    else:
        group_x_offsets = [0.0]

    # ---- Draw figure ----------------------------------------------------------

    figure, ax = plt.subplots(figsize=figsize)

    # x-range for the regression grid spans from the leftmost to the rightmost
    # point accounting for group offsets.
    x_range = all_x_valid.max() - all_x_valid.min() if len(all_x_valid) >= 2 else 1.0
    x_margin = x_range * 0.05
    x_grid = np.linspace(all_x_valid.min() - x_margin, all_x_valid.max() + x_margin, 200)

    for group_index, group_label in enumerate(unique_groups):
        x_offset = group_x_offsets[group_index]

        if color_by is not None and "color_group" in plot_dataframe.columns:
            group_rows = plot_dataframe[plot_dataframe["color_group"] == group_label]
        else:
            group_rows = plot_dataframe

        group_x_raw = group_rows["age"].values.astype(float)
        group_y = group_rows["feature"].values.astype(float)
        valid_group = np.isfinite(group_x_raw) & np.isfinite(group_y)
        group_x_raw = group_x_raw[valid_group]
        group_y = group_y[valid_group]
        group_x_offset = group_x_raw + x_offset

        marker_color = color_map.get(group_label, "#888888")

        # Scatter points — only when explicitly requested.
        if show_points:
            ax.scatter(
                group_x_offset,
                group_y,
                color=marker_color,
                s=point_size,
                alpha=point_alpha,
                edgecolors="none",
                label=group_label if color_by is not None else "_nolegend_",
                zorder=3,
                rasterized=True,  # Rasterise dense clouds for faster rendering/export.
            )

        # Average curve: mean y at each unique age, drawn at the offset x positions.
        if show_average_curve and len(group_x_raw) >= 2:
            age_average_series = (
                pd.Series(group_y, index=group_x_raw)
                .groupby(level=0)
                .mean()
                .sort_index()
            )
            average_ages = age_average_series.index.values + x_offset
            average_values = age_average_series.values
            curve_label = f"{group_label} average" if color_by is not None else "average"
            ax.plot(
                average_ages,
                average_values,
                color=marker_color,
                linewidth=average_curve_linewidth,
                marker="o",
                markersize=5,
                markeredgecolor="white",
                markeredgewidth=0.8,
                label=curve_label,
                zorder=5,
            )
            # When points are hidden, add the group label to the scatter legend entry
            # via a proxy artist so the colour is still identified in the legend.
            if not show_points and color_by is not None:
                # The curve label already includes the group name — no extra entry needed.
                pass

        # Per-group regression line (only when color_by is set and explicitly enabled).
        if show_trend_line and color_by is not None and len(group_x_raw) >= 3:
            group_pearson_r, group_pearson_p = scipy.stats.pearsonr(group_x_raw, group_y)
            group_spearman = scipy.stats.spearmanr(group_x_raw, group_y)
            group_y_hat, group_lower, group_upper = _compute_regression_band(
                group_x_raw, group_y, x_grid
            )
            reg_label = (
                f"{group_label} trend  "
                f"r={group_pearson_r:.2f} (p={group_pearson_p:.3f})  "
                f"ρ={group_spearman.statistic:.2f}"
            )
            ax.plot(
                x_grid + x_offset, group_y_hat,
                color=marker_color, linewidth=1.5, linestyle="--",
                label=reg_label, zorder=4,
            )
            if show_confidence_band:
                ax.fill_between(
                    x_grid + x_offset, group_lower, group_upper,
                    color=marker_color, alpha=0.10,
                )

    # ---- Y-axis scaling -------------------------------------------------------
    # • show_points=True  → clip to percentile range so extreme outliers do not
    #                        crush the scale (y_clip_percentile is applied).
    # • show_points=False → let matplotlib auto-scale to the visible curves only;
    #                        y_clip_percentile is ignored.

    if show_points:
        y_low_pct, y_high_pct = y_clip_percentile
        if y_low_pct > 0.0 or y_high_pct < 100.0:
            y_clip_low = float(np.nanpercentile(all_y_valid, y_low_pct))
            y_clip_high = float(np.nanpercentile(all_y_valid, y_high_pct))
            y_margin = (y_clip_high - y_clip_low) * 0.03
            ax.set_ylim(y_clip_low - y_margin, y_clip_high + y_margin)
            print(
                f"  Y-axis clipped     : [{y_clip_low:.3g}, {y_clip_high:.3g}]  "
                f"(p{y_low_pct:.4g}–p{y_high_pct:.4g})"
            )
    # When show_points=False, matplotlib auto-fits to the curve values — nothing needed.

    # ---- X-axis: show only the actual age values present in the data ----------
    # Rotate labels if there are many distinct age values to avoid overlap.

    ax.set_xticks(unique_ages_sorted)
    if len(unique_ages_sorted) > 10:
        ax.set_xticklabels(
            [str(int(v)) if float(v).is_integer() else f"{v:.1f}" for v in unique_ages_sorted],
            rotation=45,
            ha="right",
            fontsize=8,
        )
    else:
        ax.set_xticklabels(
            [str(int(v)) if float(v).is_integer() else f"{v:.1f}" for v in unique_ages_sorted]
        )

    # Expand x-limits slightly so the outermost group offsets are not clipped.
    ax.set_xlim(
        unique_ages_sorted[0] - min_age_gap * 0.6,
        unique_ages_sorted[-1] + min_age_gap * 0.6,
    )

    # ---- Axes labels, title, legend, footer -----------------------------------

    ax.set_xlabel(age_obs_column, fontsize=12)
    ax.set_ylabel(
        f"{feature_name}  [{aggregation}]" if mode != "none" else feature_name,
        fontsize=12,
    )

    title_parts = [f"{feature_name} vs. {age_obs_column}"]
    if color_by:
        title_parts.append(f"coloured by {color_by}")
    ax.set_title("  —  ".join(title_parts), fontsize=13, fontweight="bold")

    handles, labels = ax.get_legend_handles_labels()
    visible = [(h, la) for h, la in zip(handles, labels) if not la.startswith("_")]
    if visible:
        ax.legend(*zip(*visible), fontsize=9, framealpha=0.85, loc="best")

    # Build the footer text, overriding the layer field when raw .X was used
    # (get_run_context_footer_text always shows the stored active_layer name,
    # which would be misleading when use_active_layer=False).
    footer_text = get_run_context_footer_text(anndata_object)
    if not use_active_layer:
        import re
        footer_text = re.sub(r"Layer: [^|]+", "Layer: raw .X", footer_text)

    # Place the footer below the figure using figure coordinates so it never
    # overlaps the x-axis ticks or label regardless of tick density.
    # tight_layout(rect) reserves the bottom margin for the footer line.
    plt.tight_layout(rect=[0, 0.04, 1, 1])
    figure.text(
        0.5, 0.01,
        footer_text,
        ha="center", fontsize=7, color="#666666",
        transform=figure.transFigure,
    )

    # Push the figure into the Jupyter cell output immediately, then close it.
    # This ensures that every call in the same cell produces visible output —
    # not just the last one (which is the only return value Jupyter shows via repr).
    # Outside Jupyter/IPython the import fails silently and the caller receives
    # the Figure object to display or save as they wish.
    try:
        from IPython.display import display as _ipy_display
        _ipy_display(figure)
    except ImportError:
        pass
    plt.close(figure)

    # ---- Console summary ------------------------------------------------------

    print("-" * 60)
    print("RESULTS")
    print(f"  Data points plotted : {len(all_x_valid):,}")
    print(f"  Unique age values   : {len(unique_ages_sorted)}")
    if global_pearson_r is not None:
        print(f"  Overall Pearson r   : {global_pearson_r:.3f}  (p={global_pearson_p:.4f})")
        print(f"  Overall Spearman ρ  : {global_spearman_r:.3f}  (p={global_spearman_p:.4f})")
        direction = "increases" if global_slope > 0 else "decreases"
        print(
            f"  Trend               : {feature_name} {direction} with {age_obs_column} "
            f"(slope = {global_slope:.4g})"
        )
    print("=" * 60)

    return figure
