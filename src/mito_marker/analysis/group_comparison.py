"""
group_comparison.py

Per-channel statistical comparison across groups defined by an .obs column.

Provides one public function:

compare_groups()
    For each channel (feature), tests whether measurements differ significantly
    across groups defined by a categorical .obs column:

    - 2 groups  → Mann-Whitney U test  + rank-biserial correlation (effect size)
    - ≥3 groups → Kruskal-Wallis H test + eta-squared             (effect size)

    Both tests are non-parametric and rank-based, making them well-suited for
    cytometry and morphology data that rarely follows a normal distribution.
    Rank-based tests produce identical p-values regardless of whether raw or
    normalized data is used (arcsinh, zscore and minmax are all monotonic
    transformations that preserve ranks), so .X is always used.

    Observations with a missing (NaN) group label are silently excluded.

    By default, only channels marked as selected by the active feature selection
    method (from .uns['analysis_config']['active_selection']) are tested.  Pass
    use_active_selection=False to test all analytical channels instead.

    Multiple-testing correction uses Benjamini-Hochberg FDR (scipy ≥ 1.11).

    Returns a DataFrame (one row per channel) and prints a console summary.
    A bar chart of –log10(adjusted p-value) is produced automatically.

Typical usage in a notebook:
    from mito_marker.analysis import compare_groups

    # Uses active feature selection (HighVariance, MIM, …) if one is set
    stats_df = compare_groups(anndata_object, group_by="physical_activity_steps_n_day_split")

    # Force all analytical channels
    stats_df = compare_groups(anndata_object, group_by="age_quartiles",
                              use_active_selection=False)
"""

from typing import Optional, Tuple

import anndata
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse
import scipy.stats

from mito_marker.analysis.colors import sort_values_for_legend

# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

_ANALYSIS_CONFIG_KEY = "analysis_config"


def _get_channel_mask(
    anndata_object: anndata.AnnData,
    use_active_selection: bool,
) -> Tuple[np.ndarray, str]:
    """Return (boolean mask of length n_vars, description string).

    When use_active_selection is True and an active selection exists in
    .uns['analysis_config']['active_selection'], the mask combines the
    feature-selection column with the non-analytical exclusion.

    When no active selection is set, falls back to all analytical channels.

    Args:
        anndata_object: Source AnnData.
        use_active_selection: Whether to restrict to selected channels.

    Returns:
        Tuple of (boolean mask, human-readable description).
    """
    # Non-analytical exclusion (Time, FlowAI …)
    if "is_non_analytical" in anndata_object.var.columns:
        analytical_mask = ~anndata_object.var["is_non_analytical"].values.astype(bool)
    else:
        analytical_mask = np.ones(anndata_object.n_vars, dtype=bool)

    if not use_active_selection:
        return analytical_mask, "all analytical"

    config = anndata_object.uns.get(_ANALYSIS_CONFIG_KEY, {})
    active_selection = config.get("active_selection")

    if active_selection is None:
        return analytical_mask, "all analytical (no active selection found)"

    selection_col = f"is_selected_{active_selection}"
    if selection_col not in anndata_object.var.columns:
        return analytical_mask, f"all analytical ('{selection_col}' column not found)"

    selection_mask = anndata_object.var[selection_col].values.astype(bool)
    combined_mask = analytical_mask & selection_mask
    return combined_mask, f"selected by {active_selection}"


def _dense_x(anndata_object: anndata.AnnData) -> np.ndarray:
    """Return .X as a dense float64 array (handles sparse matrices).

    float64 is required for tied-rank corrections inside scipy's rank tests.

    Args:
        anndata_object: Source AnnData.

    Returns:
        2-D numpy array of shape (n_obs, n_vars).
    """
    if scipy.sparse.issparse(anndata_object.X):
        return anndata_object.X.toarray().astype(np.float64)
    return np.asarray(anndata_object.X, dtype=np.float64)


def _rank_biserial_correlation(u_statistic: float, n1: int, n2: int) -> float:
    """Compute rank-biserial correlation from a Mann-Whitney U statistic.

    Formula: r = 1 - (2 * U) / (n1 * n2)

    Interpretation (Cohen 1988):
        |r| < 0.10 → negligible
        |r| < 0.30 → small
        |r| < 0.50 → medium
        |r| >= 0.50 → large

    Args:
        u_statistic: Mann-Whitney U statistic (for group 1).
        n1: Sample size of group 1.
        n2: Sample size of group 2.

    Returns:
        Rank-biserial correlation in [-1, 1].
    """
    return float(1.0 - (2.0 * u_statistic) / (n1 * n2))


def _eta_squared(h_statistic: float, n_total: int) -> float:
    """Compute eta-squared from a Kruskal-Wallis H statistic.

    Formula: η² = H / (n - 1)  (appropriate shorthand when k << n)

    Interpretation (Cohen 1988):
        η² < 0.01 → negligible
        η² < 0.06 → small
        η² < 0.14 → medium
        η² >= 0.14 → large

    Args:
        h_statistic: Kruskal-Wallis H statistic.
        n_total: Total number of observations with a valid group label.

    Returns:
        Eta-squared in [0, 1].
    """
    if n_total <= 1:
        return float("nan")
    return float(h_statistic / (n_total - 1))


def _label_effect_size(effect_size: float, metric: str) -> str:
    """Translate a numeric effect size to a human-readable label.

    Args:
        effect_size: Effect size value (sign is ignored).
        metric: 'rank_biserial_r' or 'eta_squared'.

    Returns:
        One of 'negligible', 'small', 'medium', 'large', 'unknown'.
    """
    abs_effect = abs(effect_size)
    if np.isnan(abs_effect):
        return "unknown"
    if metric == "rank_biserial_r":
        thresholds = [(0.10, "negligible"), (0.30, "small"), (0.50, "medium")]
    else:  # eta_squared
        thresholds = [(0.01, "negligible"), (0.06, "small"), (0.14, "medium")]
    for threshold, label in thresholds:
        if abs_effect < threshold:
            return label
    return "large"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compare_groups(
    anndata_object: anndata.AnnData,
    group_by: str,
    alpha: float = 0.05,
    use_active_selection: bool = True,
    figsize: Optional[Tuple[float, float]] = None,
) -> pd.DataFrame:
    """Test whether each channel differs significantly across groups.

    Observations with a NaN group label are excluded before testing.
    By default, only channels marked as selected by the active feature
    selection method are tested; pass use_active_selection=False to test
    all analytical channels.

    For each channel, runs either a Mann-Whitney U test (2 groups) or a
    Kruskal-Wallis H test (≥3 groups), then applies Benjamini-Hochberg FDR
    correction across all tested channels simultaneously.

    A horizontal bar chart of –log10(adjusted p-value) is produced, and a
    console summary reports the proportion of significant channels.

    Args:
        anndata_object: AnnData whose .obs contains the group_by column.
        group_by: Column name in .obs that defines the groups to compare.
            Typically produced by bin_obs_column() or split_obs_by_threshold().
        alpha: Significance threshold applied to BH-adjusted p-values.
            Default 0.05.
        use_active_selection: If True (default), restrict to channels selected
            by the active feature selection method stored in
            .uns['analysis_config']['active_selection'].
            If False, test all analytical channels.
        figsize: Optional (width, height) in inches for the bar chart.
            Defaults to (11, 0.35 × n_channels + 1.5).

    Returns:
        DataFrame with one row per tested channel, sorted by p_adj ascending:
            channel            — feature name (from .var index)
            statistic          — U (2-group) or H (≥3-group) test statistic
            p_value            — raw p-value
            p_adj              — Benjamini-Hochberg adjusted p-value
            significant        — bool, True if p_adj < alpha
            effect_size        — rank-biserial r (2-group) or eta-squared (≥3-group)
            effect_size_metric — 'rank_biserial_r' or 'eta_squared'
            effect_label       — 'negligible' / 'small' / 'medium' / 'large'

    Raises:
        ValueError: If group_by is not found in .obs, or fewer than 2 valid groups.
    """
    # ------------------------------------------------------------------
    # 1. Validate group_by and drop NaN labels
    # ------------------------------------------------------------------
    if group_by not in anndata_object.obs.columns:
        raise ValueError(
            f"Column '{group_by}' not found in .obs. "
            f"Available columns: {list(anndata_object.obs.columns)}"
        )

    raw_labels: pd.Series = anndata_object.obs[group_by]
    valid_obs_mask: np.ndarray = raw_labels.notna().values
    n_dropped = int((~valid_obs_mask).sum())

    # Work only with observations that have a valid group label
    group_labels: pd.Series = raw_labels[valid_obs_mask].astype(str)
    unique_groups = sort_values_for_legend(group_labels.unique())
    n_groups = len(unique_groups)

    if n_groups < 2:
        raise ValueError(
            f"Column '{group_by}' has only {n_groups} valid group(s) after "
            f"dropping NaN labels. At least 2 groups are required."
        )

    test_name = "Mann-Whitney U" if n_groups == 2 else "Kruskal-Wallis H"
    effect_metric = "rank_biserial_r" if n_groups == 2 else "eta_squared"

    # ------------------------------------------------------------------
    # 2. Select channels to test
    # ------------------------------------------------------------------
    channel_mask, channel_description = _get_channel_mask(
        anndata_object, use_active_selection
    )
    channel_names: list = list(anndata_object.var_names[channel_mask])
    n_channels = len(channel_names)

    # Extract data matrix for valid observations and selected channels only
    full_matrix = _dense_x(anndata_object)
    data_matrix = full_matrix[valid_obs_mask, :][:, channel_mask]
    n_valid_obs = int(valid_obs_mask.sum())

    # ------------------------------------------------------------------
    # 3. Print header
    # ------------------------------------------------------------------
    print(f"\n{'=' * 60}")
    print(f"Group comparison: {group_by}")
    print(f"  Groups ({n_groups}): {', '.join(str(g) for g in unique_groups)}")
    if n_dropped > 0:
        print(f"  Observations excluded (NaN label): {n_dropped}")
    print(f"  Observations used: {n_valid_obs}")
    print(f"  Statistical test: {test_name}")
    print(f"  Effect size metric: {effect_metric}")
    print("  Multiple-testing correction: Benjamini-Hochberg FDR")
    print(f"  Significance threshold (alpha): {alpha}")
    print(f"  Channels tested: {n_channels} ({channel_description})")
    print("  Data used: .X (raw — rank tests are invariant to monotonic transforms)")
    print(f"{'=' * 60}")

    # ------------------------------------------------------------------
    # 4. Precompute per-group row indices (relative to valid_obs_mask rows)
    # ------------------------------------------------------------------
    group_row_indices: dict = {
        group: np.where(group_labels.values == group)[0]
        for group in unique_groups
    }

    # ------------------------------------------------------------------
    # 5. Per-channel statistical tests
    # ------------------------------------------------------------------
    records = []
    for channel_index, channel_name in enumerate(channel_names):
        column_values = data_matrix[:, channel_index]

        # Split values by group and drop NaN measurements within each group
        group_vectors = []
        for group in unique_groups:
            values = column_values[group_row_indices[group]]
            group_vectors.append(values[~np.isnan(values)])

        # Skip channels where any group has no valid measurements at all
        if any(len(v) == 0 for v in group_vectors):
            records.append(
                {
                    "channel": channel_name,
                    "statistic": float("nan"),
                    "p_value": float("nan"),
                    "p_adj": float("nan"),
                    "significant": False,
                    "effect_size": float("nan"),
                    "effect_size_metric": effect_metric,
                    "effect_label": "unknown",
                }
            )
            continue

        if n_groups == 2:
            n1, n2 = len(group_vectors[0]), len(group_vectors[1])
            try:
                u_stat, p_val = scipy.stats.mannwhitneyu(
                    group_vectors[0],
                    group_vectors[1],
                    alternative="two-sided",
                    use_continuity=True,
                )
            except ValueError:
                u_stat, p_val = float("nan"), float("nan")
            effect = _rank_biserial_correlation(float(u_stat), n1, n2)
            test_stat = float(u_stat)
        else:
            try:
                h_stat, p_val = scipy.stats.kruskal(*group_vectors)
            except ValueError:
                h_stat, p_val = float("nan"), float("nan")
            effect = _eta_squared(float(h_stat), n_valid_obs)
            test_stat = float(h_stat)

        records.append(
            {
                "channel": channel_name,
                "statistic": test_stat,
                "p_value": float(p_val),
                "p_adj": float("nan"),  # filled after BH step
                "significant": False,
                "effect_size": effect,
                "effect_size_metric": effect_metric,
                "effect_label": _label_effect_size(effect, effect_metric),
            }
        )

    result_dataframe = pd.DataFrame(records)

    # ------------------------------------------------------------------
    # 6. Benjamini-Hochberg FDR correction across all channels at once
    # ------------------------------------------------------------------
    valid_p_mask = ~result_dataframe["p_value"].isna()
    raw_p_values = result_dataframe.loc[valid_p_mask, "p_value"].values

    if len(raw_p_values) > 0:
        # scipy.stats.false_discovery_control requires scipy >= 1.11
        adjusted_p_values = scipy.stats.false_discovery_control(
            raw_p_values, method="bh"
        )
        result_dataframe.loc[valid_p_mask, "p_adj"] = adjusted_p_values

    result_dataframe["significant"] = result_dataframe["p_adj"] < alpha
    result_dataframe = result_dataframe.sort_values("p_adj").reset_index(drop=True)

    # ------------------------------------------------------------------
    # 7. Console summary
    # ------------------------------------------------------------------
    n_significant = int(result_dataframe["significant"].sum())
    proportion_significant = n_significant / n_channels if n_channels > 0 else 0.0

    print("\nResults (sorted by adjusted p-value):")
    print(
        f"  Significant channels (p_adj < {alpha}): "
        f"{n_significant} / {n_channels} "
        f"({proportion_significant:.1%})"
    )

    if n_significant > 0:
        sig_df = result_dataframe[result_dataframe["significant"]].copy()
        effect_counts = sig_df["effect_label"].value_counts()
        print("  Effect size breakdown (significant channels only):")
        for label in ["large", "medium", "small", "negligible"]:
            count = int(effect_counts.get(label, 0))
            if count > 0:
                print(f"    {label:>12s}: {count} channel(s)")

    print("\n  Top 10 most significant channels:")
    display_cols = ["channel", "p_value", "p_adj", "effect_size", "effect_label"]
    top10 = result_dataframe[display_cols].head(10).copy()
    top10["p_value"] = top10["p_value"].map("{:.2e}".format)
    top10["p_adj"] = top10["p_adj"].map("{:.2e}".format)
    top10["effect_size"] = top10["effect_size"].map("{:.3f}".format)
    print(top10.to_string(index=False))

    # ------------------------------------------------------------------
    # 8. Bar chart
    # ------------------------------------------------------------------
    _plot_significance_bars(
        result_dataframe=result_dataframe,
        group_by=group_by,
        alpha=alpha,
        test_name=test_name,
        effect_metric=effect_metric,
        channel_description=channel_description,
        anndata_object=anndata_object,
        figsize=figsize,
    )

    print(
        f"\nReturning DataFrame with {len(result_dataframe)} rows × "
        f"{result_dataframe.shape[1]} columns."
    )

    return result_dataframe


# ---------------------------------------------------------------------------
# Plot helper (private)
# ---------------------------------------------------------------------------


def _plot_significance_bars(
    result_dataframe: pd.DataFrame,
    group_by: str,
    alpha: float,
    test_name: str,
    effect_metric: str,
    channel_description: str,
    anndata_object: anndata.AnnData,
    figsize: Optional[Tuple[float, float]],
) -> None:
    """Render a horizontal bar chart of –log10(adjusted p-value) per channel.

    Significant channels are drawn in the first palette color; non-significant
    channels are light grey. A vertical dashed red line marks –log10(α).
    Bars are ordered most significant at top, least significant at bottom.

    Args:
        result_dataframe: Per-channel results sorted by p_adj ascending.
        group_by: Name of the grouping .obs column (used in title).
        alpha: Significance threshold.
        test_name: Human-readable test name.
        effect_metric: 'rank_biserial_r' or 'eta_squared'.
        channel_description: Short string describing which channels were tested.
        anndata_object: Source AnnData (color palette lookup).
        figsize: Optional (width, height) override.
    """
    n_channels = len(result_dataframe)
    if figsize is None:
        figsize = (11, max(4.0, 0.35 * n_channels + 1.5))

    palette = anndata_object.uns.get("color_palette", {})
    accent_color = list(palette.values())[0] if palette else "#2196F3"

    # Reverse: most significant at the top of the horizontal chart
    plot_df = result_dataframe.iloc[::-1].copy()

    neg_log10_p = -np.log10(
        np.clip(plot_df["p_adj"].values.astype(float), a_min=1e-300, a_max=None)
    )
    threshold_line = -np.log10(alpha)

    bar_colors = [
        accent_color if sig else "#CCCCCC"
        for sig in plot_df["significant"].values
    ]

    fig, axis = plt.subplots(figsize=figsize)

    axis.barh(
        y=np.arange(n_channels),
        width=neg_log10_p,
        color=bar_colors,
        edgecolor="none",
        height=0.7,
    )

    axis.axvline(
        x=threshold_line,
        color="#E53935",
        linewidth=1.5,
        linestyle="--",
        label=f"p_adj = {alpha}  (–log₁₀ = {threshold_line:.2f})",
    )

    axis.set_yticks(np.arange(n_channels))
    axis.set_yticklabels(plot_df["channel"].values, fontsize=8)
    axis.set_xlabel("–log₁₀(adjusted p-value)  [BH FDR]", fontsize=10)
    axis.set_title(
        f"Per-channel group comparison: {group_by}\n"
        f"{test_name}  ·  {effect_metric}  ·  channels: {channel_description}  ·  α = {alpha}",
        fontsize=10,
    )
    axis.legend(fontsize=9, loc="lower right")

    n_sig = int(result_dataframe["significant"].sum())
    axis.text(
        0.98,
        0.02,
        f"{n_sig}/{n_channels} channels significant",
        transform=axis.transAxes,
        fontsize=9,
        ha="right",
        va="bottom",
        color=accent_color if n_sig > 0 else "#888888",
    )

    plt.tight_layout()
    plt.show()
