"""
pca_plot.py

PCA dimensionality reduction and visualization for SFC AnnData.

Functions:
  - compute_pca() fits PCA on all events and stores coordinates + loadings.
    Results are cached so a second call with the same data skips refitting.
  - plot_pca_scatter() draws a 2-D scatter of PC1 vs PC2 with group centroids.
  - plot_pca_biplot() draws the correlation circle with channel loading arrows
    and group centroids.
  - plot_pca_loadings_bar() prints a structured per-component and cross-component
    loading summary to the console and displays a horizontal bar chart of the
    top-N contributing channels.  Auto-captured by ReportBuilder.

Unlike UMAP, PCA is fast enough to run on all events — no subsampling is needed
for computation. Subsampling is still done at plot time for visual clarity.

compute_pca(weight_by=[...]) optionally fits a WEIGHTED PCA so that unequal
group/subject event counts (e.g. comparing species with very different
numbers of imaged mitochondria) do not bias the fitted axes toward
whichever group has the most events — see ADR-011 in docs/DECISIONS.md.

Typical usage:
    from mito_marker.analysis import compute_pca, plot_pca_scatter, plot_pca_biplot
    sfc_subset = compute_pca(sfc_subset)
    plot_pca_scatter(sfc_subset, group_by="subject_ID")
    plot_pca_scatter(sfc_subset, group_by="dilution")
    plot_pca_biplot(sfc_subset, group_by="dilution")
"""

from math import ceil
from typing import Dict, List, Optional, Tuple, Union

import anndata
import matplotlib.colors as mcolors
import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Ellipse, Polygon
from scipy.spatial import ConvexHull, QhullError
from scipy.stats import chi2

from mito_marker.analysis._fingerprint import compute_fingerprint
from mito_marker.analysis._gpu_utils import is_cuml_available
from mito_marker.analysis._plot_context import (
    get_run_context_console_text,
    get_run_context_footer_text,
    get_species_label,
)
from mito_marker.analysis.colors import (
    COLOR_PALETTE_KEY,
    _generate_auto_colors,
    get_subject_colors,
    sort_values_for_legend,
)

# .uns key where the analysis pipeline configuration is stored.
_ANALYSIS_CONFIG_KEY = "analysis_config"

# .obsm key for PCA coordinates.
_OBSM_PCA_KEY = "X_pca"

# .uns keys for PCA metadata.
_PCA_LOADINGS_KEY = "pca_loadings"
_PCA_VAR_RATIO_KEY = "pca_explained_variance_ratio"
_PCA_CHANNEL_NAMES_KEY = "pca_channel_names"
# .uns key recording which .obs columns (if any) were used to weight PCA —
# see compute_pca(weight_by=...) and ADR-011 in docs/DECISIONS.md.
_PCA_WEIGHT_BY_KEY = "pca_weight_by"
# .uns keys that make the PCA reusable as a frozen reference space (ADR-015).
_PCA_MEAN_KEY = "pca_mean"
_PCA_INPUT_LAYER_KEY = "pca_input_layer"
_PCA_INPUT_FINGERPRINT_KEY = "pca_input_fingerprint"
_PCA_N_OBS_FITTED_KEY = "pca_n_obs_fitted"
_PCA_FINGERPRINT_KEY = "pca_fingerprint"

# subject_ID is colored from tab10 at plot time.
_SUBJECT_ID_COLUMN = "subject_ID"


def _build_pca_model(n_components: int, random_state: int) -> object:
    """
    Return a PCA model, preferring cuML (GPU) when available.

    Falls back to sklearn PCA when cuML is absent or no CUDA device is found.
    Both objects expose the same .fit_transform(), .components_, and
    .explained_variance_ratio_ interface.

    Parameters
    ----------
    n_components : int
        Number of principal components to compute.
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    object
        A PCA model instance (cuML or sklearn).
    """
    if is_cuml_available():
        import cuml.decomposition  # noqa: PLC0415
        print("  [GPU] cuML detected — using GPU-accelerated PCA.")
        return cuml.decomposition.PCA(n_components=n_components)
    print("  [CPU] cuML not available — using sklearn PCA.")
    from sklearn.decomposition import PCA  # noqa: PLC0415
    return PCA(n_components=n_components, random_state=random_state)


def _compute_nested_equal_weights(
    obs_dataframe: pd.DataFrame,
    hierarchy_columns: List[str],
) -> np.ndarray:
    """
    Compute a per-event weight so every group at every hierarchy level counts equally.

    Plain-language example: with hierarchy_columns=["specie", "subject_ID"],
    every SPECIES ends up contributing the same total weight to the fitted
    PCA axes, and within a species, every SUBJECT contributes the same
    weight — no matter how many mitochondria (rows) that species or subject
    happens to have. A subject with 99 events and a subject with 1,600
    events end up counting equally, exactly like two biological replicates
    should. See ADR-011 in docs/DECISIONS.md for the full rationale and a
    worked example with real subject counts.

    How it's computed: start every row's weight at 1, divide by the number
    of events sharing its full path (its own subject), then divide again by
    the number of distinct subjects within its species (and so on for any
    additional, coarser hierarchy levels). Finally rescale so all weights
    sum to 1.

    Arguments:
        obs_dataframe: AnnData.obs (or a subset), one row per event. Must
                       contain every column in hierarchy_columns, with no
                       missing values.
        hierarchy_columns: .obs column names ordered from COARSEST (e.g.
                           "specie") to FINEST (e.g. "subject_ID"). A single
                           column is allowed — equal weight per group, no
                           nesting.

    Returns:
        1-D numpy array, shape (len(obs_dataframe),), non-negative, summing
        to 1 (within floating-point precision).

    Raises:
        ValueError: If any hierarchy_columns value is missing (NaN) — every
                    event must have a known place in the hierarchy for its
                    weight to be well defined.
    """
    missing_value_counts = {
        column: int(obs_dataframe[column].isna().sum())
        for column in hierarchy_columns
        if obs_dataframe[column].isna().any()
    }
    if missing_value_counts:
        raise ValueError(
            "weight_by requires every event to have a non-missing value for "
            f"each hierarchy column, but found missing values: {missing_value_counts}. "
            "Drop or impute these events before calling compute_pca(weight_by=...)."
        )

    weights = np.ones(len(obs_dataframe), dtype=np.float64)

    # Finest level: divide by the number of events sharing this row's full
    # path (all hierarchy columns) — i.e. how many events its own subject has.
    finest_group_sizes = obs_dataframe.groupby(hierarchy_columns, observed=True)[
        hierarchy_columns[0]
    ].transform("size")
    weights /= finest_group_sizes.to_numpy()

    # Each coarser level: divide by the number of distinct child groups that
    # share the same parent (e.g. distinct subjects within a species), so the
    # parent's total weight stays split evenly across its children.
    for level in range(len(hierarchy_columns) - 1, 0, -1):
        parent_columns = hierarchy_columns[:level]
        child_column = hierarchy_columns[level]
        distinct_children_per_parent = obs_dataframe.groupby(parent_columns, observed=True)[
            child_column
        ].transform("nunique")
        weights /= distinct_children_per_parent.to_numpy()

    weights /= weights.sum()
    return weights


def _fit_weighted_pca(
    data_matrix: np.ndarray,
    weights: np.ndarray,
    n_components: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Fit PCA using a weighted mean and a weighted covariance matrix.

    Standard PCA implicitly gives every row (every mitochondrion) equal say
    in the fitted axes, so a subject/species with more events pulls the axes
    toward itself. Here, each row instead contributes according to `weights`
    (see _compute_nested_equal_weights) — a standard, published technique
    known as weighted PCA / weighted covariance eigendecomposition
    (Delchambre 2015, MNRAS). Every event is still projected onto the
    resulting axes afterwards, so no data point is dropped from the plotted
    result — only the axis-*fitting* step is reweighted.

    Arguments:
        data_matrix: shape (n_obs, n_channels) — the layer/selection to use
                     for PCA (same input _build_pca_model would receive).
        weights: shape (n_obs,), non-negative, summing to 1 — one weight per
                 row of data_matrix, from _compute_nested_equal_weights.
        n_components: Number of components to keep (already capped by the
                      caller to min(requested, n_channels, n_obs)).

    Returns:
        Tuple of:
          pca_coords: shape (n_obs, n_components) — every event projected
                      onto the weighted axes.
          components: shape (n_components, n_channels) — same orientation as
                      sklearn's PCA.components_.
          explained_variance_ratio: shape (n_components,).
    """
    data_matrix = np.asarray(data_matrix, dtype=np.float64)
    weighted_mean = weights @ data_matrix  # shape (n_channels,)
    centered = data_matrix - weighted_mean

    # Weighted covariance: sum_i weights[i] * outer(centered[i], centered[i]).
    weighted_covariance = (centered * weights[:, None]).T @ centered

    # weighted_covariance is symmetric — eigh is faster and numerically more
    # stable than a general eig() for this case, and returns real eigenvalues.
    eigenvalues, eigenvectors = np.linalg.eigh(weighted_covariance)
    # eigh returns ascending order — flip so PC1 (largest variance) is first.
    descending_order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[descending_order]
    eigenvectors = eigenvectors[:, descending_order]

    total_variance = eigenvalues.sum()
    explained_variance_ratio = (
        eigenvalues[:n_components] / total_variance
        if total_variance > 0
        else np.zeros(n_components)
    )

    components = eigenvectors[:, :n_components].T.astype(np.float32)  # (n_components, n_channels)
    pca_coords = (centered @ eigenvectors[:, :n_components]).astype(np.float32)

    return pca_coords, components, explained_variance_ratio.astype(np.float32)


def _print_weighting_qc(
    obs_dataframe: pd.DataFrame,
    hierarchy_columns: List[str],
    weights: np.ndarray,
) -> None:
    """
    Print a QC block describing the weighting applied before a weighted PCA fit.

    A beginner should be able to read this block and understand, without
    looking at any code, how many groups exist at each hierarchy level and
    how unevenly the raw (unweighted) event counts were spread — that spread
    is exactly what weight_by is correcting for.

    Arguments:
        obs_dataframe: AnnData.obs used to build the weights.
        hierarchy_columns: Same list passed to _compute_nested_equal_weights.
        weights: The computed per-event weights (sums to 1).

    Returns:
        None. Prints to stdout.
    """
    print(f"[compute_pca] weight_by={hierarchy_columns} — fitting weighted PCA.")
    for column in hierarchy_columns:
        group_sizes = obs_dataframe[column].value_counts()
        print(
            f"    '{column}': {group_sizes.size} distinct groups | "
            f"raw event counts range {int(group_sizes.min()):,}–{int(group_sizes.max()):,} "
            f"(without weighting, {group_sizes.idxmax()!r} would dominate the fit)"
        )
    # Kish's effective sample size: how many "equally-weighted" events the
    # weighting scheme is effectively equivalent to — always <= n_obs, and
    # much smaller than n_obs when a few events carry most of the weight.
    effective_sample_size = 1.0 / np.sum(weights ** 2)
    print(
        f"    weights: min={weights.min():.2e}, max={weights.max():.2e}, "
        f"sum={weights.sum():.6f} (should be 1.0) | "
        f"effective sample size ≈ {effective_sample_size:.1f} of {len(weights):,} events"
    )


def compute_pca(
    anndata_object: anndata.AnnData,
    n_components: int = 50,
    random_state: int = 42,
    force_recompute: bool = False,
    weight_by: Optional[List[str]] = None,
) -> anndata.AnnData:
    """
    Fit PCA on all events and store coordinates, loadings, and explained variance.

    The result is cached: if .obsm['X_pca'] already exists, the function prints
    a message and returns without recalculating. Pass force_recompute=True to
    override this behaviour (e.g. when changing n_components or weight_by).

    IMPORTANT — mutates anndata_object IN PLACE and returns that SAME object
    (it is not copied). This matters when comparing several weight_by variants
    side by side (e.g. unweighted vs weight_by=["specie"] vs
    weight_by=["specie", "subject_ID"], as recommended in ADR-011): calling
    compute_pca() several times on the SAME source object just overwrites the
    same .obsm['X_pca'] each time — every variable name ends up pointing at
    one object holding only the LAST result, and all subsequent plots will
    look identical. Give each variant its own copy instead::

        sfc_unweighted = compute_pca(sfc_subset.copy(), n_components=50)
        sfc_weighted   = compute_pca(sfc_subset.copy(), n_components=50,
                                      weight_by=["specie", "subject_ID"])

    Results stored:
      .obsm['X_pca']                        shape (n_obs, n_components)
      .uns['pca_loadings']                  shape (n_channels, n_components)
      .uns['pca_explained_variance_ratio']  shape (n_components,)
      .uns['pca_channel_names']             list of channel names used
      .uns['pca_weight_by']                 weight_by as passed, or None
      .uns['pca_mean']                      shape (n_channels,) — centring vector
      .uns['pca_input_layer']               layer the PCA was fitted on ("" = raw .X)
      .uns['pca_input_fingerprint']         fingerprint of that layer's frozen scaling
      .uns['pca_fingerprint']               fingerprint of the whole fitted PCA

    With pca_mean and pca_loadings, any new dataset scaled with the same
    frozen layer parameters can be projected on the SAME axes:
        coordinates = (scaled_matrix - pca_mean) @ pca_loadings
    (see project_to_reference_space() in analysis/cluster_model.py, ADR-015).

    Arguments:
        anndata_object: SFC AnnData after selection and optional normalization.
        n_components: Number of PCA components to compute. Capped at
                      min(n_components, n_channels, n_obs).
        random_state: Random seed for reproducibility. Ignored when
                      weight_by is set (the weighted solver is a closed-form
                      eigendecomposition — no randomness involved).
        force_recompute: If True, delete any existing cached PCA results and
                         recompute from scratch.
        weight_by: Optional list of .obs column names, ordered from COARSEST
                   to FINEST group (e.g. ["specie", "subject_ID"]). When set,
                   PCA is fit with a per-mitochondrion weight so that every
                   value of the coarsest column (e.g. every species)
                   contributes equally to the fitted axes, and within it,
                   every value of the next column (e.g. every subject)
                   contributes equally — regardless of how many events each
                   one has. This fixes the bias where a species/subject with
                   more recorded mitochondria would otherwise dominate the
                   shared PCA axes. Every event is still projected onto the
                   resulting axes and appears in .obsm['X_pca'] afterwards —
                   only the axis-fitting step is reweighted, no event is
                   dropped or duplicated. When None (default), standard
                   (unweighted, cuML/sklearn) PCA is used — identical to
                   prior behavior. See ADR-011 in docs/DECISIONS.md for the
                   full rationale and a worked example.

    Returns:
        The same AnnData object with PCA results populated.
    """
    # If the user explicitly requests a fresh computation, drop the cached key.
    if force_recompute and _OBSM_PCA_KEY in anndata_object.obsm:
        print(f"force_recompute=True — deleting cached .obsm['{_OBSM_PCA_KEY}'].")
        del anndata_object.obsm[_OBSM_PCA_KEY]

    # Cache check.
    if _OBSM_PCA_KEY in anndata_object.obsm:
        n_stored = anndata_object.obsm[_OBSM_PCA_KEY].shape[1]
        total_var = anndata_object.uns.get(_PCA_VAR_RATIO_KEY, np.array([0])).sum()
        cached_weight_by = anndata_object.uns.get(_PCA_WEIGHT_BY_KEY)
        print(
            f"PCA coordinates already computed ({n_stored} components, "
            f"{total_var:.1%} variance explained, weight_by={cached_weight_by}) "
            "— skipping recomputation."
        )
        print("=> Pass force_recompute=True to recompute.")
        return anndata_object

    print("=" * 60)
    print("COMPUTING PCA")
    print("=" * 60)

    data_matrix, channel_names = _get_data_and_channels(anndata_object)
    n_obs, n_channels = data_matrix.shape
    print(f"Input matrix: {n_obs:,} events × {n_channels} channels")

    # Cap n_components to the maximum possible rank.
    actual_components = min(n_components, n_channels, n_obs)
    if actual_components < n_components:
        print(
            f"n_components capped to {actual_components} "
            f"(min of requested={n_components}, channels={n_channels}, obs={n_obs})."
        )

    if weight_by is not None:
        weights = _compute_nested_equal_weights(anndata_object.obs, weight_by)
        _print_weighting_qc(anndata_object.obs, weight_by, weights)
        pca_coords, components, explained_variance_ratio = _fit_weighted_pca(
            data_matrix, weights, actual_components
        )
        # Same weighted centre as the one used inside _fit_weighted_pca().
        centering_mean = weights @ np.asarray(data_matrix, dtype=np.float64)
    else:
        pca_model = _build_pca_model(actual_components, random_state)
        # np.asarray() transfers results back to CPU RAM when cuML returns CuPy arrays.
        pca_coords = np.asarray(pca_model.fit_transform(data_matrix), dtype=np.float32)
        components = np.asarray(pca_model.components_, dtype=np.float32)
        explained_variance_ratio = np.asarray(pca_model.explained_variance_ratio_, dtype=np.float32)
        centering_mean = np.asarray(pca_model.mean_, dtype=np.float64)

    anndata_object.obsm[_OBSM_PCA_KEY] = pca_coords
    # loadings shape: (n_channels, n_components)
    anndata_object.uns[_PCA_LOADINGS_KEY] = np.asarray(components.T, dtype=np.float32)
    anndata_object.uns[_PCA_VAR_RATIO_KEY] = np.asarray(explained_variance_ratio, dtype=np.float32)
    anndata_object.uns[_PCA_CHANNEL_NAMES_KEY] = channel_names
    anndata_object.uns[_PCA_WEIGHT_BY_KEY] = list(weight_by) if weight_by else None
    _store_pca_projection_parameters(anndata_object, centering_mean, data_matrix.shape[0])

    total_variance = float(np.sum(explained_variance_ratio))
    print(
        f"PCA complete: {actual_components} components explain "
        f"{total_variance:.1%} of total variance."
    )

    # Print per-component variance for the first 10.
    top_n = min(10, actual_components)
    var_parts = [
        f"PC{i + 1}: {explained_variance_ratio[i]:.1%}"
        for i in range(top_n)
    ]
    print(" | ".join(var_parts))
    if actual_components > top_n:
        print(f"  ... ({actual_components - top_n} more components)")
    print("=" * 60)

    return anndata_object


def plot_pca_scatter(
    anndata_object: anndata.AnnData,
    group_by: Union[str, List[str]],
    max_points_per_group: Optional[int] = None,
    n_events_per_group: int = 500,
    title: str = "",
    auto_zoom: bool = True,
    zoom_percentile: float = 99.0,
    point_size: int = 7,
    point_alpha: float = 0.25,
    show_ellipses: bool = False,
    ellipse_confidence: float = 0.95,
    show_centroid_cross: bool = False,
    show_individual_centroids: bool = False,
    nest_aggregate_by: Optional[str] = None,
) -> None:
    """
    Draw a PCA scatter plot (PC1 vs PC2) with group centroids marked.

    Subsamples max_points_per_group events per group, then shuffles all groups
    together before plotting in a single scatter call so no single group
    visually covers another.

    Centroids are plotted as 'X' markers (size=160, black edge) on top of
    the scatter cloud. Axis labels include the explained variance percentage.

    When group_by is a list of column names, values from each column are
    combined with " / " to form one group label per unique combination
    (e.g. ["specie", "age_group"] → "Human / young", "Mouse / old", …).

    Three mutually-exclusive rendering modes for the group's spatial extent
    (default: none — the point cloud is drawn; show_ellipses and
    show_individual_centroids cannot both be True):

    Confidence ellipses (show_ellipses=True):
        Each ellipse is a bivariate-normal confidence ellipse fitted on ALL
        events of the group (not the subsampled/plotted subset, and no outlier
        filtering — every event contributes to the mean and covariance exactly
        like an unweighted PCA). Its center is the group mean, its axes are the
        eigenvectors of the group's 2x2 covariance matrix (PC1, PC2), and its
        semi-axis lengths are sqrt(eigenvalue * chi2.ppf(ellipse_confidence, df=2)).
        This is the standard "confidence ellipse" construction used for
        ordination plots in bioinformatics/ecology (e.g. R's ggbiplot
        stat_ellipse(), factoextra::fviz_pca_ind(addEllipses=TRUE), vegan::
        ordiellipse(type="t")), analogous to a 2-D Hotelling's T^2 region. At
        ellipse_confidence=0.95 the ellipse is expected to enclose ~95% of the
        group's events under a bivariate-normal assumption — it does NOT mean
        95% of the actual points are guaranteed inside if the group is skewed
        or multimodal, and unlike a percentile/convex-hull approach, extreme
        points are never dropped before fitting: a few strong outliers can
        inflate the ellipse. Use a lower ellipse_confidence (e.g. 0.68 ~ 1 SD)
        for a tighter, more outlier-resistant view, or clean outliers upstream
        if they are not biologically meaningful.
        No group-centroid marker is drawn unless show_centroid_cross=True.

    Individual centroid hulls (show_individual_centroids=True):
        Draws neither the ellipse nor the raw point cloud. Instead, within
        each group, one centroid is computed per distinct value of
        nest_aggregate_by (e.g. one point per subject_ID), marked with a solid
        colored disc (no label). When a group has >= 3 individuals, their
        centroids are connected by their convex hull (scipy.spatial.ConvexHull)
        — a solid line in the group color outlining the hull, with the
        enclosed area filled at 30% opacity. This is a purely geometric hull
        (no statistical assumption): it always contains exactly 100% of the
        individual centroids, unlike the confidence ellipse. It is the same
        construction as R's vegan::ordihull(), commonly used in ordination
        plots to delineate group extent from replicate/subject centroids
        rather than from raw events. Groups with exactly 2 individuals are
        connected by a straight line; a single individual is left as a lone
        disc.

    Arguments:
        anndata_object: AnnData with .obsm['X_pca'] populated by compute_pca().
        group_by: Name of the .obs column to color by (e.g. "subject_ID"), or
                  a list of column names whose values are combined into a single
                  group label.
        max_points_per_group: Maximum number of events drawn per group.
                              Increase for denser scatter, decrease for speed.
                              Defaults to 500. Ignored when show_ellipses=True
                              or show_individual_centroids=True (no raw point
                              cloud is drawn in either mode).
        n_events_per_group: Deprecated alias for max_points_per_group.
                            Kept for backwards compatibility — prefer
                            max_points_per_group in new code.
        title: Optional figure title. When empty (default), an informative
               title is generated automatically from the grouping column and
               event count.
        auto_zoom: When True (default), clip axis limits to zoom_percentile of
                   the plotted data so outliers do not compress the main cluster.
                   When show_ellipses=True or show_individual_centroids=True,
                   zoom_percentile is not used for the window itself — since no
                   raw event is drawn in either mode, the window is instead
                   fit tightly around the ellipses / individual-centroid hulls
                   that are actually shown, with the usual 10% margin.
        zoom_percentile: Percentile used for auto-zoom (default 99). Raises the
                         upper limit and lowers the lower limit symmetrically to
                         (100 - zoom_percentile).
        point_size: Marker size for individual events (default: 7).
        point_alpha: Marker transparency, 0 (invisible) to 1 (opaque) (default: 0.25).
        show_ellipses: When True, do not draw the individual-event point cloud —
                       draw one confidence ellipse per group instead, outlined
                       with a solid line in the group color and filled with the
                       same color at 50% opacity. See "Confidence ellipses" above
                       for how the ellipse is computed. Default: False.
        ellipse_confidence: Confidence level for the ellipse, between 0 and 1
                            (default 0.95, i.e. 95% — the standard default in
                            ggbiplot/factoextra/vegan). Only used when
                            show_ellipses=True.
        show_centroid_cross: When True, mark each group centroid with the bold
                             'X' marker (size=160, group color, black edge) but
                             WITHOUT the group-name label box. When False
                             (default), the group centroid is only drawn (with
                             its label) in the default point-cloud view — it is
                             hidden in show_ellipses / show_individual_centroids
                             mode unless this is set to True, since the ellipse
                             or hull already conveys the group's location.
        show_individual_centroids: When True, do not draw the ellipse or the
                                   point cloud — draw one solid colored disc per
                                   individual (see nest_aggregate_by) connected by a
                                   convex hull filled at 30% opacity. See
                                   "Individual centroid hulls" above. Requires
                                   nest_aggregate_by. Mutually exclusive with
                                   show_ellipses. Default: False.
        nest_aggregate_by: Name of the .obs column identifying each individual
                       within a group (e.g. "subject_ID"). Required when
                       show_individual_centroids=True; ignored otherwise.

    Returns:
        None. The figure is displayed via plt.show().
    """
    # Resolve which limit to use: max_points_per_group takes priority;
    # fall back to n_events_per_group for backwards-compatible call sites.
    effective_max_points = max_points_per_group if max_points_per_group is not None else n_events_per_group

    _require_pca_computed(anndata_object)

    if show_ellipses and show_individual_centroids:
        raise ValueError(
            "show_ellipses and show_individual_centroids are mutually exclusive — "
            "choose one rendering mode for plot_pca_scatter()."
        )
    if show_individual_centroids:
        if nest_aggregate_by is None:
            raise ValueError(
                "show_individual_centroids=True requires nest_aggregate_by to be set to "
                "the .obs column identifying each individual within a group "
                "(e.g. 'subject_ID')."
            )
        if nest_aggregate_by not in anndata_object.obs.columns:
            raise ValueError(
                f"nest_aggregate_by='{nest_aggregate_by}' not found in .obs. "
                f"Available columns: {list(anndata_object.obs.columns)}"
            )

    # Validate and combine grouping columns into a single label Series.
    group_labels = _get_group_labels(anndata_object, group_by)
    group_by_label = " × ".join(group_by) if isinstance(group_by, list) else group_by

    print(get_run_context_console_text(anndata_object))
    if show_individual_centroids:
        print(
            f"[plot_pca_scatter] Grouping by: '{group_by_label}' | individual centroids "
            f"by '{nest_aggregate_by}' (convex hull per group, no point cloud drawn)"
        )
    elif show_ellipses:
        print(
            f"[plot_pca_scatter] Grouping by: '{group_by_label}' | "
            f"{ellipse_confidence:.0%} confidence ellipses (fit on full group, no point cloud drawn)"
        )
    else:
        print(f"[plot_pca_scatter] Grouping by: '{group_by_label}' | max {effective_max_points} events/group")

    pca_coords = anndata_object.obsm[_OBSM_PCA_KEY]
    var_ratios = anndata_object.uns[_PCA_VAR_RATIO_KEY]

    obs_df = anndata_object.obs.copy()
    obs_df["_pc1"] = pca_coords[:, 0]
    obs_df["_pc2"] = pca_coords[:, 1]
    obs_df["_group_label"] = group_labels.values

    unique_groups = sort_values_for_legend(obs_df["_group_label"].dropna().unique())

    if len(unique_groups) == 0:
        raise ValueError(
            f"Column '{group_by_label}' has no non-NaN values — no groups to plot. "
            "The column is likely entirely NaN (e.g. the clinical variable was not populated)."
        )

    color_map = _build_color_map(anndata_object, group_by, unique_groups)

    # Subsample per group.
    group_frames = []
    for group_value in unique_groups:
        group_df = obs_df[obs_df["_group_label"] == group_value]
        if len(group_df) > effective_max_points:
            group_df = group_df.sample(n=effective_max_points, random_state=42)
        group_frames.append(group_df)

    plot_df = pd.concat(group_frames, ignore_index=True)
    # Shuffle so groups are interleaved.
    plot_df = plot_df.sample(frac=1, random_state=42).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(10, 7))

    # Collects each ellipse's/hull's axis-aligned bounding box so auto-zoom
    # never clips a drawn shape.
    extra_bounding_boxes: List[Tuple[float, float, float, float]] = []

    if show_individual_centroids:
        # Neither the ellipse nor the raw point cloud is drawn — only one
        # cross per individual, connected by a per-group convex hull.
        extra_bounding_boxes = _draw_individual_centroid_hulls(
            ax, obs_df, unique_groups, color_map, nest_aggregate_by
        )
    elif show_ellipses:
        # Ellipses are fit on the FULL group (obs_df), not the subsampled
        # plot_df, so the confidence region reflects the true population.
        for group_value in unique_groups:
            color = color_map.get(str(group_value), "#999999")
            group_df = obs_df[obs_df["_group_label"] == group_value]
            ellipse_params = _compute_confidence_ellipse_params(
                group_df["_pc1"].values, group_df["_pc2"].values, ellipse_confidence
            )
            if ellipse_params is None:
                print(
                    f"WARNING [plot_pca_scatter]: group '{group_value}' has fewer than "
                    "2 events — skipping its confidence ellipse."
                )
                continue
            center_x, center_y, width, height, angle_degrees = ellipse_params
            ax.add_patch(Ellipse(
                (center_x, center_y), width, height, angle=angle_degrees,
                facecolor=mcolors.to_rgba(color, alpha=0.5),
                edgecolor=color, linewidth=2, zorder=3,
            ))
            extra_bounding_boxes.append(
                _ellipse_bounding_box(center_x, center_y, width, height, angle_degrees)
            )
    else:
        # Assign a per-row color vector so the single scatter call respects the
        # shuffled order — re-separating by group before each ax.scatter() call
        # would make the last group cover all others regardless of shuffling.
        row_colors = plot_df["_group_label"].astype(str).map(
            lambda g: color_map.get(g, "#999999")
        ).tolist()

        # Draw event scatter in one call (shuffled order, no group covers another).
        ax.scatter(
            plot_df["_pc1"].values,
            plot_df["_pc2"].values,
            c=row_colors,
            s=point_size,
            alpha=point_alpha,
            edgecolors="none",
        )

    # Add invisible per-group handles for the legend.
    for group_value in unique_groups:
        color = color_map.get(str(group_value), "#999999")
        ax.scatter([], [], c=color, s=point_size, label=str(group_value))

    # Draw the group centroid on top. In show_ellipses / show_individual_centroids
    # mode the shape itself already conveys the group center, so no marker is
    # drawn unless the user explicitly opts in via show_centroid_cross — which
    # then also drops the label box for a clean, minimal look.
    draw_centroid_marker = show_centroid_cross or not (show_ellipses or show_individual_centroids)
    if draw_centroid_marker:
        centroid_df = obs_df.groupby("_group_label")[["_pc1", "_pc2"]].mean()
        for group_value in centroid_df.index:
            color = color_map.get(str(group_value), "#999999")
            cx, cy = centroid_df.loc[group_value, "_pc1"], centroid_df.loc[group_value, "_pc2"]
            ax.scatter(cx, cy, color=color, s=160, marker="X", edgecolors="black",
                       linewidths=0.7, zorder=5)
            if not show_centroid_cross:
                ax.annotate(
                    str(group_value),
                    (cx, cy),
                    textcoords="offset points",
                    xytext=(0, 8),
                    ha="center",
                    fontsize=9,
                    fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7, edgecolor="none"),
                )

    if auto_zoom:
        if show_ellipses or show_individual_centroids:
            # Only the ellipses / individual-centroid hulls are drawn (no raw
            # event cloud) — base the zoom purely on their bounding boxes.
            # Raw-event percentiles would zoom out further than necessary
            # (the ellipse/hull is tighter than the full per-event spread,
            # especially with the default zoom_percentile=99 vs. a 95%
            # confidence ellipse), leaving a lot of empty margin.
            if extra_bounding_boxes:
                x_lo = min(box[0] for box in extra_bounding_boxes)
                x_hi = max(box[1] for box in extra_bounding_boxes)
                y_lo = min(box[2] for box in extra_bounding_boxes)
                y_hi = max(box[3] for box in extra_bounding_boxes)
            else:
                x_lo, x_hi, y_lo, y_hi = -1.0, 1.0, -1.0, 1.0
        else:
            low_pct = 100.0 - zoom_percentile
            x_lo = np.percentile(plot_df["_pc1"].values, low_pct)
            x_hi = np.percentile(plot_df["_pc1"].values, zoom_percentile)
            y_lo = np.percentile(plot_df["_pc2"].values, low_pct)
            y_hi = np.percentile(plot_df["_pc2"].values, zoom_percentile)
        x_margin = (x_hi - x_lo) * 0.1 or 1.0
        y_margin = (y_hi - y_lo) * 0.1 or 1.0
        ax.set_xlim(x_lo - x_margin, x_hi + x_margin)
        ax.set_ylim(y_lo - y_margin, y_hi + y_margin)

    pc1_label = f"PC1 ({var_ratios[0]:.1%})" if len(var_ratios) > 0 else "PC1"
    pc2_label = f"PC2 ({var_ratios[1]:.1%})" if len(var_ratios) > 1 else "PC2"
    ax.set_xlabel(pc1_label, fontsize=12)
    ax.set_ylabel(pc2_label, fontsize=12)
    species = get_species_label(anndata_object)
    species_prefix = f"[{species}]  " if species else ""
    weight_suffix = _pca_weight_title_suffix(anndata_object)
    if show_individual_centroids:
        total_individuals = obs_df.groupby(["_group_label", nest_aggregate_by]).ngroups
        auto_title = (
            f"{species_prefix}PCA Scatter — grouped by '{group_by_label}' | "
            f"{total_individuals} individual centroids (by '{nest_aggregate_by}'){weight_suffix}"
        )
    elif show_ellipses:
        auto_title = (
            f"{species_prefix}PCA Scatter — grouped by '{group_by_label}' | "
            f"{len(obs_df):,} events | {ellipse_confidence:.0%} confidence ellipses{weight_suffix}"
        )
    else:
        auto_title = (
            f"{species_prefix}PCA Scatter — grouped by '{group_by_label}' | "
            f"{len(plot_df):,} events{weight_suffix}"
        )
    ax.set_title(title if title else auto_title, fontsize=13)
    _apply_legend(ax, group_by_label, len(unique_groups))
    ax.grid(True, linestyle="--", alpha=0.3)
    fig.text(
        0.5, 0.01, get_run_context_footer_text(anndata_object),
        ha="center", va="bottom",
        fontsize=6, color="gray", style="italic",
    )
    plt.tight_layout()
    plt.show()


def plot_pca_biplot(
    anndata_object: anndata.AnnData,
    group_by: Union[str, List[str]],
    top_n_variables: int = 5,
    labels: Optional[Union[str, List[str]]] = None,
    max_points_per_group: Optional[int] = None,
    title: str = "",
) -> None:
    """
    Draw a PCA correlation circle (biplot) showing channel loadings and group centroids.

    Visual elements:
      - Unit circle (gray dashed line).
      - Loading arrows for the top_n_variables channels ranked by combined
        loading magnitude on PC1 and PC2 (dark red, labeled).
      - Group centroids plotted as 'X' markers, scaled to fit within the unit
        circle (80% of radius).

    Prints the top_n_variables channels with the highest absolute loading on
    PC1 and PC2 before showing the figure.

    When group_by is a list of column names, values from each column are
    combined with " / " to form one group label per unique combination
    (e.g. ["specie", "age_group"] → "Human / young", "Mouse / old", …).

    Arguments:
        anndata_object: AnnData with .obsm['X_pca'] and .uns['pca_loadings']
                        populated by compute_pca().
        group_by: Name of the .obs column to color by (e.g. "subject_ID"), or
                  a list of column names whose values are combined into a single
                  group label.
        top_n_variables: Number of top-loading channels to print for each axis
                         and to label in the plot.
        labels: Optional .obs column name(s) whose values are appended next to
                each centroid label, separated by " | ".  Accepts a single
                string (e.g. "diet") or a list (e.g. ["diet", "age"]).
                Columns that do not exist in .obs are silently skipped.
                Not applicable when group_by is a list (combined labels already
                encode the multi-column information).
        max_points_per_group: When set, draws a lightly-styled background scatter
                              of individual events (up to this many per group)
                              scaled into the unit-circle space.  Default is None
                              (no individual points, centroids only).
        title: Optional figure title. When empty (default), an informative
               title is generated automatically.

    Returns:
        None. The figure is displayed via plt.show().
    """
    _require_pca_computed(anndata_object)

    # Validate and combine grouping columns into a single label Series.
    group_labels = _get_group_labels(anndata_object, group_by)
    group_by_label = " × ".join(group_by) if isinstance(group_by, list) else group_by

    # Normalise `labels` to a list of strings (may be empty).
    # When group_by is a list, extra labels are irrelevant — combined label
    # already encodes all the multi-column information.
    if labels is None or isinstance(group_by, list):
        label_columns: List[str] = []
    elif isinstance(labels, str):
        label_columns = [labels]
    else:
        label_columns = list(labels)

    # Warn once for any requested label column that doesn't exist.
    missing_label_columns = [col for col in label_columns if col not in anndata_object.obs.columns]
    if missing_label_columns:
        print(
            f"WARNING [plot_pca_biplot]: label column(s) not found in .obs and will be "
            f"skipped: {missing_label_columns}"
        )

    loadings = anndata_object.uns[_PCA_LOADINGS_KEY]  # (n_channels, n_components)
    var_ratios = anndata_object.uns[_PCA_VAR_RATIO_KEY]
    channel_names = anndata_object.uns.get(_PCA_CHANNEL_NAMES_KEY, [])
    pca_coords = anndata_object.obsm[_OBSM_PCA_KEY]

    if loadings.shape[1] < 2:
        print("WARNING: PCA has fewer than 2 components — biplot requires at least 2.")
        return

    loadings_pc1 = loadings[:, 0]
    loadings_pc2 = loadings[:, 1]

    # --- Print run context then top variables before the figure ---
    print(get_run_context_console_text(anndata_object))
    pc1_var = var_ratios[0] if len(var_ratios) > 0 else 0
    pc2_var = var_ratios[1] if len(var_ratios) > 1 else 0
    _print_top_variables(channel_names, loadings_pc1, loadings_pc2,
                         pc1_var, pc2_var, top_n_variables)

    # --- Scaling: fit centroids and arrows to the unit circle ---
    obs_df = anndata_object.obs.copy()
    obs_df["_pc1"] = pca_coords[:, 0]
    obs_df["_pc2"] = pca_coords[:, 1]
    obs_df["_group_label"] = group_labels.values
    centroid_df = obs_df.groupby("_group_label")[["_pc1", "_pc2"]].mean()

    if len(centroid_df) == 0:
        raise ValueError(
            f"Column '{group_by_label}' has no non-NaN values — no centroids to plot. "
            "The column is likely entirely NaN (e.g. the clinical variable was not populated)."
        )

    max_centroid = np.max(np.abs(centroid_df.values)) if len(centroid_df) > 0 else 1.0
    max_loading = np.max(
        np.sqrt(loadings_pc1 ** 2 + loadings_pc2 ** 2)
    )

    # Scale arrows so the longest loading fits inside the unit circle.
    arrow_scale = 0.9 / max_loading if max_loading > 0 else 1.0
    # Scale centroids so the farthest centroid is at 80% of the circle.
    centroid_scale = 0.8 / max_centroid if max_centroid > 0 else 1.0

    unique_groups = sort_values_for_legend(obs_df["_group_label"].dropna().unique())
    color_map = _build_color_map(anndata_object, group_by, unique_groups)

    fig, ax = plt.subplots(figsize=(10, 10))

    # Unit circle.
    theta = np.linspace(0, 2 * np.pi, 300)
    ax.plot(np.cos(theta), np.sin(theta), color="gray", linestyle="--", alpha=0.5, linewidth=1)

    _draw_loading_arrows(ax, channel_names, loadings_pc1, loadings_pc2,
                         arrow_scale, top_n_variables)

    # Group centroids scaled to fit inside unit circle.
    for group_value in centroid_df.index:
        color = color_map.get(str(group_value), "#999999")
        cx = centroid_df.loc[group_value, "_pc1"] * centroid_scale
        cy = centroid_df.loc[group_value, "_pc2"] * centroid_scale
        ax.scatter(cx, cy, color=color, s=120, marker="X",
                   edgecolors="black", linewidths=0.7, zorder=5, label=str(group_value))
        centroid_text = _build_centroid_label(str(group_value), obs_df, "_group_label", label_columns)
        ax.text(cx, cy + 0.04, centroid_text, fontsize=9, fontweight="bold",
                ha="center", va="bottom", color=color,
                bbox=dict(facecolor="white", alpha=0.6, edgecolor="none", pad=1))

    ax.set_xlim(-1.25, 1.25)
    ax.set_ylim(-1.25, 1.25)
    ax.axhline(0, color="lightgray", linewidth=0.8, linestyle="-")
    ax.axvline(0, color="lightgray", linewidth=0.8, linestyle="-")
    ax.set_aspect("equal")
    ax.set_xlabel(f"PC1 ({pc1_var:.1%})", fontsize=12)
    ax.set_ylabel(f"PC2 ({pc2_var:.1%})", fontsize=12)
    species = get_species_label(anndata_object)
    species_prefix = f"[{species}]  " if species else ""
    weight_suffix = _pca_weight_title_suffix(anndata_object)
    auto_title = (
        f"{species_prefix}PCA Biplot — grouped by '{group_by_label}'\n"
        f"(Arrows = channel loadings | Markers = group centroids{weight_suffix})"
    )
    ax.set_title(title if title else auto_title, fontsize=13)
    _apply_legend(ax, group_by_label, len(unique_groups))
    ax.grid(True, linestyle="--", alpha=0.2)
    fig.text(
        0.5, 0.01, get_run_context_footer_text(anndata_object),
        ha="center", va="bottom",
        fontsize=6, color="gray", style="italic",
    )
    plt.tight_layout()
    plt.show()


def _generate_condition_gradient(base_color: str, n: int) -> List[str]:
    """Generate n colors from a pale version (young/early) to a dark version (old/late).

    The base_color defines the hue; brightness goes from 60% white-mixed (youngest)
    to the pure color (middle) to 25% black-mixed (oldest).
    """
    if n == 0:
        return []
    if n == 1:
        return [base_color]
    rgb = np.array(mcolors.to_rgb(base_color))
    white = np.ones(3)
    black = np.zeros(3)
    result: List[str] = []
    for i in range(n):
        t = i / (n - 1)  # 0 = youngest, 1 = oldest
        if t < 0.5:
            # Pale → full color
            pale = rgb * 0.4 + white * 0.6
            color = pale + (rgb - pale) * (t * 2)
        else:
            # Full color → darkened
            darken_factor = 0.3 * (t - 0.5) * 2
            color = rgb * (1 - darken_factor) + black * darken_factor
        result.append(mcolors.to_hex(np.clip(color, 0.0, 1.0)))
    return result


def plot_pca_trajectory(
    anndata_object: anndata.AnnData,
    condition_column: str,
    time_column: str,
    top_n_variables: int = 5,
    show_loading_arrows: bool = True,
    time_order: Optional[List] = None,
    title: str = "",
    color_mode: str = "gradient",
    show_labels: bool = True,
) -> None:
    """
    Draw a PCA trajectory plot showing how each condition's centroid evolves over time.

    For each (condition, timepoint) pair, one centroid is computed as the mean of all
    events belonging to that group.  Centroids of the same condition are connected by a
    dashed line in chronological order so the trajectory through PCA space is visible.

    Visual elements:
      - Unit circle (gray dashed line).
      - Loading arrows for the top_n_variables channels (when show_loading_arrows=True).
      - One marker per (condition, timepoint) centroid.
      - Dashed lines connecting each condition's centroids in time order.
      - Optional text labels near each centroid (show_labels=True).
      - Legend showing conditions and, in gradient mode, the young/old color endpoints.

    Arguments:
        anndata_object: AnnData with .obsm['X_pca'] populated by compute_pca().
        condition_column: Name of the .obs column that defines conditions
                          (e.g. "diet" → "AL" / "IF").
        time_column: Name of the .obs column that defines the time axis
                     (e.g. "age" → 0, 2, 8, 10, 14, 16).
        top_n_variables: Number of top-loading channels to draw as arrows.
        show_loading_arrows: When False, only centroids and trajectories are drawn
                             (no loading arrows).  Useful for a cleaner view when
                             the arrow labels clutter the plot.
        time_order: Explicit ordered list of timepoint values.  When None, timepoints
                    are sorted in ascending order (numeric or lexicographic).
        title: Optional figure title. When empty (default), an informative
               title is generated automatically.
        color_mode: "gradient" (default) — each condition's markers shade from pale
                    (youngest timepoint) to dark (oldest), encoding time in color so
                    labels can be suppressed.  "flat" — legacy behavior, single flat
                    color per condition.
        show_labels: When True (default), show "{condition} | {timepoint}" text near
                     each centroid.  Set to False to remove all text annotations —
                     most useful combined with color_mode="gradient".

    Returns:
        None. The figure is displayed via plt.show().
    """
    _require_pca_computed(anndata_object)

    if condition_column not in anndata_object.obs.columns:
        raise ValueError(
            f"condition_column '{condition_column}' not found in .obs. "
            f"Available: {list(anndata_object.obs.columns)}"
        )
    if time_column not in anndata_object.obs.columns:
        raise ValueError(
            f"time_column '{time_column}' not found in .obs. "
            f"Available: {list(anndata_object.obs.columns)}"
        )

    print(get_run_context_console_text(anndata_object))
    print(
        f"[plot_pca_trajectory] condition='{condition_column}' | "
        f"time='{time_column}' | arrows={show_loading_arrows}"
    )

    loadings = anndata_object.uns[_PCA_LOADINGS_KEY]
    var_ratios = anndata_object.uns[_PCA_VAR_RATIO_KEY]
    channel_names = anndata_object.uns.get(_PCA_CHANNEL_NAMES_KEY, [])
    pca_coords = anndata_object.obsm[_OBSM_PCA_KEY]

    loadings_pc1 = loadings[:, 0]
    loadings_pc2 = loadings[:, 1]
    pc1_var = var_ratios[0] if len(var_ratios) > 0 else 0.0
    pc2_var = var_ratios[1] if len(var_ratios) > 1 else 0.0

    # Print top variables so the user can understand the axes.
    _print_top_variables(channel_names, loadings_pc1, loadings_pc2,
                         pc1_var, pc2_var, top_n_variables)

    # Build working dataframe with PCA coordinates.
    obs_df = anndata_object.obs[[condition_column, time_column]].copy()
    obs_df["_pc1"] = pca_coords[:, 0]
    obs_df["_pc2"] = pca_coords[:, 1]

    # Determine timepoint order.
    available_timepoints = obs_df[time_column].dropna().unique()
    if time_order is not None:
        sorted_timepoints = [t for t in time_order if t in available_timepoints]
    else:
        # Sort numerically when possible, lexicographically otherwise.
        try:
            sorted_timepoints = sorted(available_timepoints, key=float)
        except (TypeError, ValueError):
            sorted_timepoints = sort_values_for_legend(available_timepoints)

    # Compute centroids: mean PC1/PC2 per (condition, timepoint).
    centroid_df = (
        obs_df.groupby([condition_column, time_column])[["_pc1", "_pc2"]]
        .mean()
        .reset_index()
    )

    print(
        f"[plot_pca_trajectory] {len(centroid_df)} centroids computed "
        f"({obs_df[condition_column].nunique()} conditions × "
        f"{len(sorted_timepoints)} timepoints)."
    )

    # Scale all centroids to fit within 80% of the unit circle.
    max_centroid = np.max(np.abs(centroid_df[["_pc1", "_pc2"]].values)) if len(centroid_df) > 0 else 1.0
    centroid_scale = 0.8 / max_centroid if max_centroid > 0 else 1.0

    # Scale arrows to fit within the unit circle.
    max_loading = np.max(np.sqrt(loadings_pc1 ** 2 + loadings_pc2 ** 2))
    arrow_scale = 0.9 / max_loading if max_loading > 0 else 1.0

    unique_conditions = sort_values_for_legend(obs_df[condition_column].dropna().unique())
    color_map = _build_color_map(anndata_object, condition_column, unique_conditions)

    # Pre-compute per-condition gradient palettes (one color per timepoint).
    # In "flat" mode each palette has a single repeated color for backward-compat.
    n_timepoints = len(sorted_timepoints)
    condition_palettes: Dict[str, List[str]] = {}
    for condition_value in unique_conditions:
        base_color = color_map.get(str(condition_value), "#999999")
        if color_mode == "gradient":
            condition_palettes[str(condition_value)] = _generate_condition_gradient(
                base_color, n_timepoints
            )
        else:
            condition_palettes[str(condition_value)] = [base_color] * n_timepoints

    fig, ax = plt.subplots(figsize=(10, 10))

    # Unit circle.
    theta = np.linspace(0, 2 * np.pi, 300)
    ax.plot(np.cos(theta), np.sin(theta), color="gray", linestyle="--", alpha=0.5, linewidth=1)

    if show_loading_arrows:
        _draw_loading_arrows(ax, channel_names, loadings_pc1, loadings_pc2,
                             arrow_scale, top_n_variables)

    # Collect all plotted centroid positions to compute axis bounds afterwards.
    all_plotted_x: List[float] = []
    all_plotted_y: List[float] = []

    for condition_value in unique_conditions:
        palette = condition_palettes[str(condition_value)]
        base_color = color_map.get(str(condition_value), "#999999")
        condition_rows = centroid_df[centroid_df[condition_column] == condition_value]

        # Build ordered (x, y, color) list for existing timepoints.
        trajectory_segments: List[Tuple[float, float, str]] = []
        for timepoint_index, timepoint in enumerate(sorted_timepoints):
            row = condition_rows[condition_rows[time_column] == timepoint]
            if len(row) == 0:
                continue
            cx = float(row["_pc1"].iloc[0]) * centroid_scale
            cy = float(row["_pc2"].iloc[0]) * centroid_scale
            point_color = palette[timepoint_index]
            trajectory_segments.append((cx, cy, point_color))

        # Draw trajectory line segments, each colored by the gradient midpoint.
        for seg_index in range(len(trajectory_segments) - 1):
            x0, y0, c0 = trajectory_segments[seg_index]
            x1, y1, c1 = trajectory_segments[seg_index + 1]
            # Color the segment with the midpoint color for a smooth gradient effect.
            mid_rgb = (
                np.array(mcolors.to_rgb(c0)) + np.array(mcolors.to_rgb(c1))
            ) / 2
            seg_color = mcolors.to_hex(mid_rgb)
            ax.plot([x0, x1], [y0, y1], color=seg_color,
                    linestyle="--", linewidth=1.5, alpha=0.8)

        # Draw a transparent polyline for the legend entry (flat base color).
        if trajectory_segments:
            xs = [s[0] for s in trajectory_segments]
            ys = [s[1] for s in trajectory_segments]
            ax.plot(xs, ys, color=base_color, linestyle="--", linewidth=1.5,
                    alpha=0, label=str(condition_value))

        # Draw individual centroid markers and optional labels.
        for timepoint_index, timepoint in enumerate(sorted_timepoints):
            row = condition_rows[condition_rows[time_column] == timepoint]
            if len(row) == 0:
                continue
            cx = float(row["_pc1"].iloc[0]) * centroid_scale
            cy = float(row["_pc2"].iloc[0]) * centroid_scale
            point_color = palette[timepoint_index]
            all_plotted_x.append(cx)
            all_plotted_y.append(cy)

            # Marker size grows slightly with age to reinforce temporal encoding.
            if color_mode == "gradient" and n_timepoints > 1:
                t = timepoint_index / (n_timepoints - 1)
                marker_size = 80 + 80 * t  # 80 (young) → 160 (old)
            else:
                marker_size = 120

            ax.scatter(cx, cy, color=point_color, s=marker_size, marker="X",
                       edgecolors="black", linewidths=0.7, zorder=5)

            if show_labels:
                label_text = f"{condition_value} | {timepoint}"
                ax.text(cx, cy + 0.04, label_text, fontsize=8, fontweight="bold",
                        ha="center", va="bottom", color=point_color,
                        bbox=dict(facecolor="white", alpha=0.6, edgecolor="none", pad=1))

    # Dynamic axis bounds: encompass all plotted centroids with a margin.
    # Fall back to ±1.0 so the unit circle always stays visible.
    label_padding = 0.25 if show_labels else 0.15
    if all_plotted_x:
        data_bound = max(
            1.0,
            abs(min(all_plotted_x)) + label_padding,
            max(all_plotted_x) + label_padding,
            abs(min(all_plotted_y)) + label_padding,
            max(all_plotted_y) + label_padding,
        )
    else:
        data_bound = 1.25
    ax.set_xlim(-data_bound, data_bound)
    ax.set_ylim(-data_bound, data_bound)

    ax.axhline(0, color="lightgray", linewidth=0.8, linestyle="-")
    ax.axvline(0, color="lightgray", linewidth=0.8, linestyle="-")
    ax.set_aspect("equal")
    ax.set_xlabel(f"PC1 ({pc1_var:.1%})", fontsize=12)
    ax.set_ylabel(f"PC2 ({pc2_var:.1%})", fontsize=12)
    species = get_species_label(anndata_object)
    species_prefix = f"[{species}]  " if species else ""
    gradient_subtitle = (
        "pale marker = young  |  dark marker = old"
        if color_mode == "gradient"
        else "Markers = group centroids | Lines = trajectory over time"
    )
    auto_title = (
        f"{species_prefix}PCA Trajectory — '{condition_column}' over '{time_column}'\n"
        f"({gradient_subtitle})"
    )
    ax.set_title(title if title else auto_title, fontsize=13)

    # Build legend: one entry per condition + gradient color endpoints when applicable.
    from matplotlib.patches import Patch
    legend_handles = []
    for condition_value in unique_conditions:
        palette = condition_palettes[str(condition_value)]
        if color_mode == "gradient" and len(palette) >= 2:
            legend_handles.append(
                Patch(facecolor=palette[0], edgecolor="black", linewidth=0.5,
                      label=f"{condition_value} (young)")
            )
            legend_handles.append(
                Patch(facecolor=palette[-1], edgecolor="black", linewidth=0.5,
                      label=f"{condition_value} (old)")
            )
        else:
            legend_handles.append(
                Patch(facecolor=palette[0], edgecolor="black", linewidth=0.5,
                      label=str(condition_value))
            )
    ax.legend(handles=legend_handles, title=condition_column, fontsize=9,
              title_fontsize=9, loc="upper right", framealpha=0.9)

    ax.grid(True, linestyle="--", alpha=0.2)
    fig.text(
        0.5, 0.01, get_run_context_footer_text(anndata_object),
        ha="center", va="bottom",
        fontsize=6, color="gray", style="italic",
    )
    plt.tight_layout()
    plt.show()


def get_top_loading_channels(
    anndata_object: anndata.AnnData,
    pc_index: int = 0,
    top_n: int = 5,
) -> List[Tuple[str, float]]:
    """
    Return the top_n channels with the highest absolute loading on a given PC.

    Arguments:
        anndata_object: AnnData with .uns['pca_loadings'] populated.
        pc_index: Zero-based index of the PC to rank (0 = PC1, 1 = PC2, …).
        top_n: Number of top channels to return.

    Returns:
        List of (channel_name, loading_value) tuples sorted by |loading| descending.
    """
    _require_pca_computed(anndata_object)
    loadings = anndata_object.uns[_PCA_LOADINGS_KEY]
    channel_names = anndata_object.uns.get(_PCA_CHANNEL_NAMES_KEY, [])

    if pc_index >= loadings.shape[1]:
        raise ValueError(
            f"pc_index={pc_index} out of range — PCA has {loadings.shape[1]} components."
        )

    pc_loadings = loadings[:, pc_index]
    top_indices = np.argsort(np.abs(pc_loadings))[::-1][:top_n]

    return [
        (channel_names[i] if i < len(channel_names) else f"CH{i}", float(pc_loadings[i]))
        for i in top_indices
    ]


def plot_pca_loadings_bar(
    anndata_object: anndata.AnnData,
    top_n: int = 5,
    n_components: int = 3,
    title: str = "",
) -> None:
    """
    Print a structured loading summary and display a horizontal bar chart of the
    top contributing channels for the first ``n_components`` PCA components.

    Two console blocks are printed:
      1. Per-component block — top ``top_n`` channels ranked by their percentage
         share of absolute loading on that component.
      2. Cross-component feature summary — every unique channel that appears in
         any per-component top-N list, showing its loading percentage across all
         components.  Sorted by total contribution (sum across components).

    The percentage shown is the channel's absolute loading divided by the sum of
    all absolute loadings for that component, expressed as a percentage.  It is
    **not** the explained variance ratio — it quantifies how dominant a channel is
    within a single PC.

    A matplotlib figure with one horizontal-bar subplot per component is produced
    and displayed via ``plt.show()``.  When called inside a ``ReportBuilder`` context
    manager the figure is captured automatically.

    Arguments:
        anndata_object: AnnData with ``compute_pca()`` already called.
        top_n: Number of top channels to show per component (default 5).
        n_components: Number of PCA components to include (default 3 → PC1/PC2/PC3).
        title: Optional figure super-title. When empty (default), an informative
               title is generated automatically.

    Returns:
        None.  Side effects: console output + figure displayed via ``plt.show()``.

    Typical usage::

        import mito_marker

        mito_marker.compute_pca(sfc_subset)

        # Stand-alone call
        mito_marker.plot_pca_loadings_bar(sfc_subset, top_n=5)

        # Inside a ReportBuilder — captured automatically
        with mito_marker.ReportBuilder() as report:
            mito_marker.plot_pca_loadings_bar(sfc_subset, top_n=5)
        report.save_pdf("report.pdf", anndata_object=sfc_subset)
    """
    _require_pca_computed(anndata_object)

    loadings_matrix = anndata_object.uns[_PCA_LOADINGS_KEY]       # (n_channels, n_pcs)
    var_ratios = anndata_object.uns[_PCA_VAR_RATIO_KEY]            # (n_pcs,)
    channel_names: List[str] = list(
        anndata_object.uns.get(_PCA_CHANNEL_NAMES_KEY, [])
    )

    # Cap n_components to what is actually available.
    n_available_components = loadings_matrix.shape[1]
    n_components = min(n_components, n_available_components)

    # -----------------------------------------------------------------------
    # Pre-compute loading percentages for all components.
    # Percentage = abs(loading) / sum(abs(loadings for this PC)) * 100.
    # -----------------------------------------------------------------------
    def _loading_pct_for_pc(pc_idx: int) -> np.ndarray:
        """Return loading percentages for every channel on one component."""
        absolute_loadings = np.abs(loadings_matrix[:, pc_idx])
        total_absolute = absolute_loadings.sum()
        if total_absolute == 0.0:
            return np.zeros_like(absolute_loadings)
        return absolute_loadings / total_absolute * 100.0

    loading_pct_per_pc: List[np.ndarray] = [
        _loading_pct_for_pc(pc_idx) for pc_idx in range(n_components)
    ]

    # Top-N indices per component (sorted by descending %).
    top_indices_per_pc: List[List[int]] = [
        list(np.argsort(loading_pct_per_pc[pc_idx])[::-1][:top_n])
        for pc_idx in range(n_components)
    ]

    # -----------------------------------------------------------------------
    # Console output — run context header.
    # -----------------------------------------------------------------------
    print(get_run_context_console_text(anndata_object))
    separator = "=" * 60
    divider = "-" * 60
    print(separator)
    print(f"PCA LOADINGS — TOP {top_n} CHANNELS PER COMPONENT")
    print(separator)

    # Per-component block.
    for pc_idx in range(n_components):
        var_pct = float(var_ratios[pc_idx]) * 100.0
        print(f"\nPC{pc_idx + 1} ({var_pct:.1f}%) — top {top_n} channels:")
        for rank, channel_idx in enumerate(top_indices_per_pc[pc_idx]):
            channel_name = (
                channel_names[channel_idx]
                if channel_idx < len(channel_names)
                else f"CH{channel_idx}"
            )
            contribution_pct = loading_pct_per_pc[pc_idx][channel_idx]
            print(f"  {rank + 1:>2}. {channel_name:<30s}  {contribution_pct:5.1f}%")

    # -----------------------------------------------------------------------
    # Cross-component feature summary.
    # Collect unique channels that appear in the top-N of ANY component, then
    # show their loading % across all components.  Sorted by total contribution.
    # -----------------------------------------------------------------------
    unique_top_channel_indices: List[int] = []
    seen: set = set()
    for pc_idx in range(n_components):
        for channel_idx in top_indices_per_pc[pc_idx]:
            if channel_idx not in seen:
                unique_top_channel_indices.append(channel_idx)
                seen.add(channel_idx)

    # Sort unique features by their sum of loading percentages across all components.
    unique_top_channel_indices.sort(
        key=lambda idx: sum(loading_pct_per_pc[pc][idx] for pc in range(n_components)),
        reverse=True,
    )

    print(f"\n{divider}")
    print(f"CROSS-COMPONENT FEATURE SUMMARY  (unique features across top {top_n})")
    print(divider)

    # Header row.
    pc_headers = "".join(f"  PC{pc_idx + 1:>2}  " for pc_idx in range(n_components))
    print(f"  {'Feature':<30s}{pc_headers}  Total")
    print(f"  {'-' * 30}" + "  ------" * n_components + "  --------")

    for channel_idx in unique_top_channel_indices:
        channel_name = (
            channel_names[channel_idx]
            if channel_idx < len(channel_names)
            else f"CH{channel_idx}"
        )
        pc_values = [loading_pct_per_pc[pc_idx][channel_idx] for pc_idx in range(n_components)]
        pc_cells = "".join(f"  {v:5.1f}%" for v in pc_values)
        total_contribution = sum(pc_values)
        print(f"  {channel_name:<30s}{pc_cells}  {total_contribution:6.1f}%")

    print()

    # -----------------------------------------------------------------------
    # Matplotlib figure — one horizontal bar subplot per component.
    # -----------------------------------------------------------------------
    figure, axes = plt.subplots(
        1,
        n_components,
        figsize=(6 * n_components, max(4, top_n * 0.55 + 2)),
        sharey=False,
    )

    # Ensure axes is always a list (plt.subplots returns a single Axes when n=1).
    if n_components == 1:
        axes = [axes]

    species = get_species_label(anndata_object)
    species_prefix = f"[{species}]  " if species else ""
    auto_title = f"{species_prefix}PCA loadings — top {top_n} channels per component"
    figure.suptitle(
        title if title else auto_title,
        fontsize=13,
        fontweight="bold",
        y=1.01,
    )

    for pc_idx, axis in enumerate(axes):
        var_pct = float(var_ratios[pc_idx]) * 100.0
        top_channel_indices = top_indices_per_pc[pc_idx]

        # Build ordered lists: highest contributor at the top (last in barh).
        ordered_indices = list(reversed(top_channel_indices))
        ordered_names = [
            channel_names[i] if i < len(channel_names) else f"CH{i}"
            for i in ordered_indices
        ]
        ordered_pct = [loading_pct_per_pc[pc_idx][i] for i in ordered_indices]

        # Raw signed loading values determine bar color (positive = blue, negative = tomato).
        ordered_sign = [
            1 if loadings_matrix[i, pc_idx] >= 0 else -1
            for i in ordered_indices
        ]
        bar_colors = [
            "#4878cf" if sign >= 0 else "#e84646"   # blue / red
            for sign in ordered_sign
        ]

        axis.barh(ordered_names, ordered_pct, color=bar_colors, edgecolor="white", linewidth=0.4)

        # Annotate each bar with its percentage value.
        for bar_idx, (bar_value, sign) in enumerate(zip(ordered_pct, ordered_sign)):
            axis.text(
                bar_value + 0.2,
                bar_idx,
                f"{bar_value:.1f}%",
                va="center",
                fontsize=8,
                color="#333333",
            )

        axis.set_title(
            f"PC{pc_idx + 1}  ({var_pct:.1f}% variance)",
            fontsize=11,
            fontweight="bold",
        )
        axis.set_xlabel("% of absolute loading", fontsize=9)
        axis.tick_params(axis="y", labelsize=9)
        axis.tick_params(axis="x", labelsize=8)

        # Extend x-axis slightly so the annotations don't get clipped.
        x_max = max(ordered_pct) if ordered_pct else 1.0
        axis.set_xlim(0, x_max * 1.22)
        axis.spines[["top", "right"]].set_visible(False)

    # Footer with run-context provenance.
    figure.text(
        0.5,
        -0.02,
        get_run_context_footer_text(anndata_object),
        ha="center",
        fontsize=7,
        color="#888888",
        style="italic",
    )

    # Use explicit figure.tight_layout() so the subsequent text figure is not affected.
    figure.tight_layout()

    # Build the cross-component summary as a second figure so it is captured
    # automatically by ReportBuilder (both figures are open when plt.show() fires).
    _build_cross_component_text_figure(
        channel_names=channel_names,
        unique_top_channel_indices=unique_top_channel_indices,
        loading_pct_per_pc=loading_pct_per_pc,
        n_components=n_components,
        top_n=top_n,
    )

    plt.show()


def _build_cross_component_text_figure(
    channel_names: List[str],
    unique_top_channel_indices: List[int],
    loading_pct_per_pc: List[np.ndarray],
    n_components: int,
    top_n: int,
) -> matplotlib.figure.Figure:
    """Create a text figure with the cross-component feature summary table.

    Renders the same information as the console cross-component block so it is
    captured automatically by ReportBuilder and embedded in the PDF right after
    the loading bar chart.

    Args:
        channel_names: Ordered list of channel/feature names.
        unique_top_channel_indices: Indices of features that appear in the top-N
            of at least one component, sorted by descending total contribution.
        loading_pct_per_pc: List of percentage arrays, one per component.
        n_components: Number of PCA components shown.
        top_n: The ``top_n`` value used to select the features (used in the title).

    Returns:
        The matplotlib Figure (also registered with pyplot, so it will be
        captured by ReportBuilder when ``plt.show()`` is next called).
    """
    lines: List[str] = []
    divider = "-" * 60

    lines.append(f"CROSS-COMPONENT FEATURE SUMMARY  (unique features across top {top_n})")
    lines.append(divider)

    # Header row.
    pc_headers = "".join(f"   PC{pc_idx + 1:>2} " for pc_idx in range(n_components))
    lines.append(f"  {'Feature':<30s}{pc_headers}   Total")
    lines.append(f"  {'-' * 30}" + "  ------" * n_components + "  --------")

    for channel_idx in unique_top_channel_indices:
        channel_name = (
            channel_names[channel_idx]
            if channel_idx < len(channel_names)
            else f"CH{channel_idx}"
        )
        pc_values = [loading_pct_per_pc[pc_idx][channel_idx] for pc_idx in range(n_components)]
        pc_cells = "".join(f"  {v:5.1f}%" for v in pc_values)
        total_contribution = sum(pc_values)
        lines.append(f"  {channel_name:<30s}{pc_cells}  {total_contribution:6.1f}%")

    text_content = "\n".join(lines)

    text_figure = plt.figure(figsize=(11, 8.5))
    text_figure.patch.set_facecolor("white")
    text_figure.text(
        0.04,
        0.96,
        text_content,
        transform=text_figure.transFigure,
        verticalalignment="top",
        fontfamily="monospace",
        fontsize=8,
        color="#1a1a1a",
    )
    return text_figure


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

_MANY_GROUPS_THRESHOLD = 15


def _compute_confidence_ellipse_params(
    x_values: np.ndarray,
    y_values: np.ndarray,
    confidence: float,
) -> Optional[Tuple[float, float, float, float, float]]:
    """
    Compute a bivariate-normal confidence ellipse from raw (x, y) samples.

    The ellipse is centered on the sample mean and oriented along the
    eigenvectors of the 2x2 covariance matrix of (x_values, y_values) — i.e.
    its own local principal axes, not necessarily aligned with the global
    PC1/PC2 axes. Semi-axis lengths are sqrt(eigenvalue * chi2_quantile),
    where chi2_quantile = chi2.ppf(confidence, df=2). This is the standard
    Hotelling's T^2 / Mahalanobis-distance confidence-region construction used
    for group ellipses on ordination plots (ggbiplot::stat_ellipse(),
    factoextra::fviz_pca_ind(addEllipses=TRUE), vegan::ordiellipse(type="t")).

    No outlier filtering is applied — every point in x_values/y_values
    contributes to the mean and covariance, exactly as in a standard PCA.

    Arguments:
        x_values: 1-D array of PC1 coordinates for one group.
        y_values: 1-D array of PC2 coordinates for one group.
        confidence: Confidence level in (0, 1), e.g. 0.95 for a 95% ellipse.

    Returns:
        (center_x, center_y, width, height, angle_degrees), where width/height
        are full axis lengths (diameters, ready for matplotlib's Ellipse), or
        None when the group has fewer than 2 points (covariance undefined).
    """
    if len(x_values) < 2:
        return None
    center_x, center_y = float(np.mean(x_values)), float(np.mean(y_values))
    covariance_matrix = np.cov(x_values, y_values)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
    # eigh() returns eigenvalues ascending — reverse so index 0 is the major axis.
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.clip(eigenvalues[order], a_min=0.0, a_max=None)
    eigenvectors = eigenvectors[:, order]
    chi2_quantile = chi2.ppf(confidence, df=2)
    width, height = 2.0 * np.sqrt(eigenvalues * chi2_quantile)
    angle_degrees = float(np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0])))
    return center_x, center_y, float(width), float(height), angle_degrees


def _ellipse_bounding_box(
    center_x: float,
    center_y: float,
    width: float,
    height: float,
    angle_degrees: float,
) -> Tuple[float, float, float, float]:
    """
    Return the axis-aligned bounding box (x_lo, x_hi, y_lo, y_hi) of a rotated ellipse.

    Used to expand auto-zoom axis limits so confidence ellipses are never
    clipped by the plot boundary.
    """
    angle_radians = np.radians(angle_degrees)
    semi_x, semi_y = width / 2.0, height / 2.0
    half_extent_x = np.sqrt(
        (semi_x * np.cos(angle_radians)) ** 2 + (semi_y * np.sin(angle_radians)) ** 2
    )
    half_extent_y = np.sqrt(
        (semi_x * np.sin(angle_radians)) ** 2 + (semi_y * np.cos(angle_radians)) ** 2
    )
    return (
        center_x - half_extent_x, center_x + half_extent_x,
        center_y - half_extent_y, center_y + half_extent_y,
    )


def _draw_individual_centroid_hulls(
    ax: plt.Axes,
    obs_df: pd.DataFrame,
    unique_groups: List,
    color_map: Dict[str, str],
    nest_aggregate_by: str,
) -> List[Tuple[float, float, float, float]]:
    """
    Draw one solid colored disc per individual centroid, connected by a
    per-group convex hull filled at 30% opacity.

    Within each group (a value of the plot's group_by column), one centroid
    is computed per distinct value of nest_aggregate_by (mean PC1/PC2 of that
    individual's events). This is a purely geometric summary — no raw events
    and no statistical ellipse are drawn, only the individual centroids and
    the convex hull that encloses them (scipy.spatial.ConvexHull), analogous
    to R's vegan::ordihull(). Unlike a confidence ellipse, the hull always
    contains exactly 100% of the individual centroids.

    Arguments:
        ax: Matplotlib Axes to draw on.
        obs_df: Full (non-subsampled) working dataframe with '_pc1', '_pc2',
                '_group_label' columns already populated.
        unique_groups: Ordered list of group label values to iterate.
        color_map: Mapping of group value (str) -> hex color.
        nest_aggregate_by: .obs column identifying the individual sub-entity
                       within each group (e.g. "subject_ID").

    Returns:
        List of (x_lo, x_hi, y_lo, y_hi) bounding boxes, one per group that
        has at least one individual, so auto-zoom can include the full hull.
    """
    bounding_boxes: List[Tuple[float, float, float, float]] = []

    for group_value in unique_groups:
        color = color_map.get(str(group_value), "#999999")
        group_df = obs_df[obs_df["_group_label"] == group_value]
        individual_centroids = group_df.groupby(nest_aggregate_by)[["_pc1", "_pc2"]].mean()

        if len(individual_centroids) == 0:
            continue

        points = individual_centroids[["_pc1", "_pc2"]].values

        # Solid colored disc at each individual's centroid — no label.
        ax.scatter(points[:, 0], points[:, 1], color=color, s=60, marker="o",
                   alpha=1.0, edgecolors="none", zorder=6)

        if len(points) >= 3:
            try:
                hull = ConvexHull(points)
                hull_points = points[hull.vertices]
                ax.add_patch(Polygon(
                    hull_points, closed=True,
                    facecolor=mcolors.to_rgba(color, alpha=0.3),
                    edgecolor=color, linewidth=1.5, zorder=4,
                ))
            except QhullError:
                # Individual centroids are collinear — no 2-D area to fill,
                # fall back to a simple connecting line.
                ax.plot(points[:, 0], points[:, 1], color=color, linewidth=1.5, zorder=4)
        elif len(points) == 2:
            ax.plot(points[:, 0], points[:, 1], color=color, linewidth=1.5, zorder=4)
        # A single individual has nothing to connect — only the cross above.

        bounding_boxes.append((
            float(points[:, 0].min()), float(points[:, 0].max()),
            float(points[:, 1].min()), float(points[:, 1].max()),
        ))

    return bounding_boxes


def _draw_loading_arrows(
    ax: plt.Axes,
    channel_names: List[str],
    loadings_pc1: np.ndarray,
    loadings_pc2: np.ndarray,
    arrow_scale: float,
    top_n: int,
) -> None:
    """
    Draw the top_n channel loading arrows onto a biplot axes.

    Channels are ranked by combined loading magnitude sqrt(PC1² + PC2²).
    Each arrow is drawn in dark red from the origin to the scaled loading coordinate,
    with the channel name as a text label just beyond the arrowhead.

    Arguments:
        ax: Matplotlib Axes to draw on.
        channel_names: Ordered list of channel name strings.
        loadings_pc1: 1-D array of loadings on PC1.
        loadings_pc2: 1-D array of loadings on PC2.
        arrow_scale: Scalar that maps raw loading values into the unit-circle space.
        top_n: Number of top channels to draw.
    """
    loading_magnitudes = np.sqrt(loadings_pc1 ** 2 + loadings_pc2 ** 2)
    top_indices = np.argsort(loading_magnitudes)[::-1][:top_n]

    for idx in top_indices:
        channel_name = channel_names[idx] if idx < len(channel_names) else f"CH{idx}"
        scaled_lx = loadings_pc1[idx] * arrow_scale
        scaled_ly = loadings_pc2[idx] * arrow_scale

        ax.annotate(
            "",
            xy=(scaled_lx, scaled_ly),
            xytext=(0, 0),
            arrowprops=dict(arrowstyle="-|>", color="#8b0000", alpha=0.8, lw=1.5),
        )
        ax.text(
            scaled_lx * 1.08,
            scaled_ly * 1.08,
            channel_name,
            color="#8b0000",
            fontsize=7,
            ha="center",
            va="center",
        )


def _apply_legend(ax: plt.Axes, group_by: str, n_unique_groups: int) -> None:
    """
    Place the legend below the plot when there are many groups, to the right otherwise.

    With many subjects (> _MANY_GROUPS_THRESHOLD), a single-column legend placed
    beside the plot is taller than the figure and overlaps the data area.  Splitting
    into multiple columns and anchoring below keeps the plot readable.

    Arguments:
        ax: The matplotlib Axes on which to add the legend.
        group_by: Label used as the legend title.
        n_unique_groups: Number of distinct groups — controls ncol and placement.
    """
    if n_unique_groups > _MANY_GROUPS_THRESHOLD:
        legend_ncol = max(1, ceil(n_unique_groups / _MANY_GROUPS_THRESHOLD))
        ax.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.08),
            ncol=legend_ncol,
            fontsize=7,
            title=group_by,
        )
    else:
        ax.legend(title=group_by, bbox_to_anchor=(1.05, 1), loc="upper left")


def _get_analytical_mask_for_channels(
    anndata_object: anndata.AnnData,
    channel_names: List[str],
) -> np.ndarray:
    """
    Return a boolean mask (length = len(channel_names)) that is True for
    analytically usable channels.

    Reads .var['is_non_analytical'] when the column is present, selecting only
    the rows corresponding to channel_names (which may be a subset of all vars
    after feature selection).  Falls back to SFC_NON_ANALYTICAL_CHANNELS for
    AnnData objects that pre-date the introduction of this column.

    Arguments:
        anndata_object: The AnnData whose .var is inspected.
        channel_names: Ordered list of channel names to mask (may be a
            subset of anndata_object.var_names after feature selection).

    Returns:
        Boolean numpy array of length len(channel_names).
    """
    if "is_non_analytical" in anndata_object.var.columns:
        return ~anndata_object.var.loc[channel_names, "is_non_analytical"].values.astype(bool)

    # Fallback for AnnData objects loaded from disk before the is_non_analytical
    # column was introduced.  Both SFC and TEM non-analytical names are checked so
    # that the fallback works regardless of which pipeline produced the file.
    from mito_marker.controlled_vocabulary import (
        SFC_NON_ANALYTICAL_CHANNELS,
        TEM_NON_ANALYTICAL_FEATURES,
    )
    non_analytical_set = set(SFC_NON_ANALYTICAL_CHANNELS) | set(TEM_NON_ANALYTICAL_FEATURES)
    return np.array([ch not in non_analytical_set for ch in channel_names], dtype=bool)


def _get_data_and_channels(
    anndata_object: anndata.AnnData,
) -> Tuple[np.ndarray, List[str]]:
    """
    Return the data matrix and channel name list for PCA fitting.

    Respects the active layer and active feature selection.

    Arguments:
        anndata_object: AnnData with optional .uns['analysis_config'].

    Returns:
        Tuple of (data_matrix, channel_names).
    """
    analysis_config = anndata_object.uns.get(_ANALYSIS_CONFIG_KEY, {})
    active_layer = analysis_config.get("active_layer")
    active_selection = analysis_config.get("active_selection")

    if active_layer is not None and active_layer in anndata_object.layers:
        full_matrix = anndata_object.layers[active_layer]
        print(f"[compute_pca] Using layer: '{active_layer}'")
    else:
        full_matrix = anndata_object.X
        print("[compute_pca] Using raw .X")

    if hasattr(full_matrix, "toarray"):
        full_matrix = full_matrix.toarray()

    # np.asarray avoids an unnecessary 6 GB copy when the matrix is already
    # float32 contiguous (the common case after normalisation).
    full_matrix = np.asarray(full_matrix, dtype=np.float32)

    if active_selection is not None:
        selection_column = f"is_selected_{active_selection}"
        if selection_column in anndata_object.var.columns:
            mask = anndata_object.var[selection_column].values
            full_matrix = full_matrix[:, mask]
            channel_names = anndata_object.var_names[mask].tolist()
            print(
                f"[compute_pca] Feature selection ({active_selection}): "
                f"{mask.sum()} of {anndata_object.n_vars} channels used."
            )
        else:
            channel_names = anndata_object.var_names.tolist()
    else:
        channel_names = anndata_object.var_names.tolist()

    # Always exclude non-analytical channels (Time, FlowAI) from PCA input.
    # The mask is read from .var['is_non_analytical'] when available (set at
    # ingestion time); falls back to SFC_NON_ANALYTICAL_CHANNELS for older files.
    analytical_mask = _get_analytical_mask_for_channels(anndata_object, channel_names)
    if not analytical_mask.all():
        excluded = [ch for ch, keep in zip(channel_names, analytical_mask) if not keep]
        full_matrix = full_matrix[:, analytical_mask]
        channel_names = [ch for ch, keep in zip(channel_names, analytical_mask) if keep]
        print(f"[compute_pca] Non-analytical channels excluded from PCA: {excluded}")

    return full_matrix, channel_names


def _store_pca_projection_parameters(
    anndata_object: anndata.AnnData,
    centering_mean: np.ndarray,
    n_obs_fitted: int,
) -> None:
    """
    Record what is needed to project a NEW dataset onto the fitted PCA axes.

    The PCA loadings alone are not enough: a new point is projected with
    (x - mean) @ loadings, so the centring mean must be kept too. The layer
    the PCA was fitted on, and that layer's frozen-scaling fingerprint, are
    recorded so a cluster model can check that the whole chain
    (scaling -> PCA) is consistent.

    Arguments:
        anndata_object: AnnData just processed by compute_pca(); updated in place.
        centering_mean: Mean vector subtracted before projection, shape (n_channels,).
        n_obs_fitted: Number of observations the PCA was fitted on.
    """
    # Imported here to avoid a circular import at module load.
    from mito_marker.analysis.normalization import get_layer_parameters  # noqa: PLC0415

    active_layer = anndata_object.uns.get(_ANALYSIS_CONFIG_KEY, {}).get("active_layer")
    layer_parameters = get_layer_parameters(anndata_object, active_layer)
    if active_layer is None:
        input_fingerprint = "raw_X"
    elif layer_parameters is None:
        # The layer exists but its scaling was not recorded: the PCA can be
        # plotted, but a new dataset cannot be scaled identically before projection.
        input_fingerprint = "unknown"
        print(
            f"WARNING [compute_pca]: layer '{active_layer}' has no frozen parameters "
            "— this PCA cannot be used as a reference space for new datasets."
        )
    else:
        input_fingerprint = str(layer_parameters["fingerprint"])

    loadings = anndata_object.uns[_PCA_LOADINGS_KEY]
    anndata_object.uns[_PCA_MEAN_KEY] = np.asarray(centering_mean, dtype=np.float64)
    anndata_object.uns[_PCA_INPUT_LAYER_KEY] = active_layer if active_layer is not None else ""
    anndata_object.uns[_PCA_INPUT_FINGERPRINT_KEY] = input_fingerprint
    anndata_object.uns[_PCA_N_OBS_FITTED_KEY] = int(n_obs_fitted)
    anndata_object.uns[_PCA_FINGERPRINT_KEY] = compute_fingerprint(
        input_fingerprint, anndata_object.uns[_PCA_CHANNEL_NAMES_KEY],
        np.asarray(centering_mean, dtype=np.float64), np.asarray(loadings, dtype=np.float64),
    )
    print(
        f"=> Projection parameters stored (.uns['{_PCA_MEAN_KEY}'], fitted on "
        f"{n_obs_fitted:,} observations) — PCA fingerprint "
        f"{anndata_object.uns[_PCA_FINGERPRINT_KEY]}, input layer fingerprint {input_fingerprint}."
    )


def _require_pca_computed(anndata_object: anndata.AnnData) -> None:
    """
    Raise a clear error if compute_pca() has not been run yet.

    Arguments:
        anndata_object: AnnData to check.

    Raises:
        ValueError: If .obsm['X_pca'] does not exist.
    """
    if _OBSM_PCA_KEY not in anndata_object.obsm:
        raise ValueError(
            f"PCA not yet computed. Run compute_pca() first to populate "
            f".obsm['{_OBSM_PCA_KEY}']."
        )


def _pca_weight_title_suffix(anndata_object: anndata.AnnData) -> str:
    """
    Return a short " | weighted PCA (...)" title fragment when compute_pca()
    was run with weight_by=..., or "" when it was not (unchanged titles).

    Arguments:
        anndata_object: AnnData with compute_pca() already run.

    Returns:
        A string fragment to append to a plot title.
    """
    weight_by = anndata_object.uns.get(_PCA_WEIGHT_BY_KEY)
    if not weight_by:
        return ""
    return f" | weighted PCA ({' > '.join(weight_by)})"


def _get_group_labels(
    anndata_object: anndata.AnnData,
    group_by: Union[str, List[str]],
) -> pd.Series:
    """
    Extract and combine grouping columns from .obs into a single label Series.

    When group_by is a string, the corresponding column is returned as-is.
    When group_by is a list, the values from each column are concatenated with
    " / " to form a combined label (e.g. "Human / young", "Mouse / old").

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
    unique_groups: List,
) -> Dict[str, str]:
    """
    Build a {group_value_string: hex_color} mapping.

    Color priority:
      1. PREFERRED_CONDITION_COLORS from controlled_vocabulary.py.
      2. Palette stored in .uns['color_palette'] by assign_color_palette().
      3. Auto-generated Set2 colors for any values not covered by 1 or 2.

    For combined group_by lists, combined labels are not in any stored palette
    so tab20 colors are generated automatically.

    Arguments:
        anndata_object: AnnData with optional .uns['color_palette'].
        group_by: Name of the grouping .obs column, or a list of column names.
        unique_groups: List of unique group values.

    Returns:
        Dict mapping each group value (as string) to a hex color.
    """
    from mito_marker.controlled_vocabulary import PREFERRED_CONDITION_COLORS

    # Multi-column combination: combined labels are not in any stored palette.
    if isinstance(group_by, list):
        return get_subject_colors([str(g) for g in unique_groups], palette="tab20")

    if group_by == _SUBJECT_ID_COLUMN:
        return get_subject_colors([str(g) for g in unique_groups], palette="tab10")

    stored_palette = anndata_object.uns.get(COLOR_PALETTE_KEY, {})

    # Identify values that have no color in either source.
    uncolored = [
        str(g) for g in unique_groups
        if str(g) not in stored_palette and str(g) not in PREFERRED_CONDITION_COLORS
    ]

    if uncolored:
        print(
            f"WARNING: {len(uncolored)} group value(s) have no color in the palette "
            f"({uncolored}). Auto-assigning Set2 colors. "
            "Call assign_color_palette() before plotting for consistent colors."
        )
        auto_colors = _generate_auto_colors(len(uncolored), "Set2")
        auto_map = dict(zip(uncolored, auto_colors))
    else:
        auto_map = {}

    color_map: Dict[str, str] = {}
    for g in unique_groups:
        g_str = str(g)
        if g_str in PREFERRED_CONDITION_COLORS:
            color_map[g_str] = PREFERRED_CONDITION_COLORS[g_str]
        elif g_str in stored_palette:
            color_map[g_str] = stored_palette[g_str]
        else:
            color_map[g_str] = auto_map.get(g_str, "#999999")

    return color_map


def _build_centroid_label(
    group_value: str,
    obs_dataframe: pd.DataFrame,
    group_by: str,
    label_columns: List[str],
) -> str:
    """
    Build the display label for a centroid by appending requested .obs column values.

    For each column in label_columns the unique non-null values for the given
    group are looked up.  Columns absent from obs_dataframe are silently skipped.
    When a group has multiple distinct values for a column (unexpected for
    per-subject metadata), only the first unique value is shown.

    The result is the group value followed by each label value, joined by " | ".
    Example: "S001 | Low-fat | 52"

    Arguments:
        group_value: The string representation of the group centroid identifier.
        obs_dataframe: The .obs DataFrame with an added "_pc1"/"_pc2" column.
        group_by: Column name used to identify the group.
        label_columns: List of .obs columns whose values to append.

    Returns:
        A formatted label string.
    """
    parts = [group_value]
    group_rows = obs_dataframe[obs_dataframe[group_by].astype(str) == group_value]
    for col in label_columns:
        if col not in obs_dataframe.columns:
            continue
        unique_values = group_rows[col].dropna().unique()
        if len(unique_values) == 0:
            continue
        parts.append(str(unique_values[0]))
    return " | ".join(parts)


def _print_top_variables(
    channel_names: List[str],
    loadings_pc1: np.ndarray,
    loadings_pc2: np.ndarray,
    pc1_var: float,
    pc2_var: float,
    top_n: int,
) -> None:
    """
    Print the top_n channels with highest absolute loading on PC1 and PC2.

    Arguments:
        channel_names: List of channel name strings.
        loadings_pc1: 1-D array of loadings on PC1.
        loadings_pc2: 1-D array of loadings on PC2.
        pc1_var: Explained variance ratio for PC1.
        pc2_var: Explained variance ratio for PC2.
        top_n: Number of top channels to print for each axis.
    """
    # Contribution (%): squared loading × 100 (loadings are unit-norm eigenvectors,
    # so their squared values already sum to 1 — no additional normalisation needed)
    contributions_pc1 = loadings_pc1 ** 2 * 100
    contributions_pc2 = loadings_pc2 ** 2 * 100

    print(f"\nPC1 ({pc1_var:.1%}) — top {top_n} channels:")
    top1_indices = np.argsort(contributions_pc1)[::-1][:top_n]
    for rank, idx in enumerate(top1_indices, start=1):
        name = channel_names[idx] if idx < len(channel_names) else f"CH{idx}"
        print(f"  {rank}. {name:<20} {contributions_pc1[idx]:5.1f}%")

    print(f"\nPC2 ({pc2_var:.1%}) — top {top_n} channels:")
    top2_indices = np.argsort(contributions_pc2)[::-1][:top_n]
    for rank, idx in enumerate(top2_indices, start=1):
        name = channel_names[idx] if idx < len(channel_names) else f"CH{idx}"
        print(f"  {rank}. {name:<20} {contributions_pc2[idx]:5.1f}%")
    print()


# ---------------------------------------------------------------------------
# 3D Interactive PCA helpers (Plotly)
# ---------------------------------------------------------------------------


def _import_plotly():
    """
    Lazily import plotly.graph_objects and return the module.

    Using a lazy import prevents the rest of pca_plot.py from failing when
    Plotly is not installed — only the 3D functions raise an ImportError.

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
            "plotly is required for 3D PCA plots. "
            "Install it with: pip install plotly"
        ) from error


def _validate_pc_indices(
    anndata_object: anndata.AnnData,
    pc_x: int,
    pc_y: int,
    pc_z: int,
) -> None:
    """
    Validate that pc_x, pc_y, pc_z are in range and all distinct.

    Arguments:
        anndata_object: AnnData with .obsm['X_pca'] already populated.
        pc_x: 1-based index of the component to use as X axis.
        pc_y: 1-based index of the component to use as Y axis.
        pc_z: 1-based index of the component to use as Z axis.

    Raises:
        ValueError: If any index is out of range or two indices are equal.
    """
    n_components = anndata_object.obsm[_OBSM_PCA_KEY].shape[1]
    for param_name, param_value in [("pc_x", pc_x), ("pc_y", pc_y), ("pc_z", pc_z)]:
        if not (1 <= param_value <= n_components):
            raise ValueError(
                f"{param_name}={param_value} is out of range — PCA has "
                f"{n_components} components. Valid range: 1 to {n_components}."
            )
    if len({pc_x, pc_y, pc_z}) < 3:
        raise ValueError(
            f"pc_x, pc_y, and pc_z must all be distinct. "
            f"Got: pc_x={pc_x}, pc_y={pc_y}, pc_z={pc_z}."
        )


def _build_hover_text(
    group_plot_df: pd.DataFrame,
    group_by: str,
    group_value: object,
    hover_columns: List[str],
    pc_x: int,
    pc_y: int,
    pc_z: int,
) -> List[str]:
    """
    Build a per-event HTML hover string for the 3D scatter plot.

    Each hover shows: group value, subject_ID (when available and different from
    group_by), the three PC coordinates, and any extra requested columns.

    Arguments:
        group_plot_df: Subsampled DataFrame for this group with _pc_x/_pc_y/_pc_z columns.
        group_by: Column name used for grouping.
        group_value: Value identifying this group.
        hover_columns: Additional .obs columns to display.
        pc_x: 1-based PC index for X axis (used for label).
        pc_y: 1-based PC index for Y axis.
        pc_z: 1-based PC index for Z axis.

    Returns:
        List of HTML strings, one per event row.
    """
    hover_texts = []
    for _, row in group_plot_df.iterrows():
        lines = [f"<b>{group_by}</b>: {group_value}"]
        if _SUBJECT_ID_COLUMN in group_plot_df.columns and group_by != _SUBJECT_ID_COLUMN:
            lines.append(f"subject_ID: {row[_SUBJECT_ID_COLUMN]}")
        lines.append(f"PC{pc_x}: {row['_pc_x']:.4f}")
        lines.append(f"PC{pc_y}: {row['_pc_y']:.4f}")
        lines.append(f"PC{pc_z}: {row['_pc_z']:.4f}")
        for col in hover_columns:
            lines.append(f"{col}: {row[col]}")
        hover_texts.append("<br>".join(lines))
    return hover_texts


def _build_arrow_traces_3d(
    go: object,
    channel_names: List[str],
    scaled_tip_vectors: np.ndarray,
    top_indices: np.ndarray,
) -> List:
    """
    Build Plotly 3D traces for loading arrows (lines from origin + text labels at tips).

    All coordinates must already be scaled by the caller so arrows and centroids
    share the same visual scale.  Both traces share a legendgroup so toggling
    'Loadings' in the legend hides/shows both lines and labels together.

    Arguments:
        go: The plotly.graph_objects module (lazily imported by the caller).
        channel_names: Ordered list of channel name strings.
        scaled_tip_vectors: Shape (top_n, 3) — pre-scaled (x, y, z) of each arrow tip.
        top_indices: Shape (top_n,) — indices into channel_names for each arrow.

    Returns:
        List of two go.Scatter3d traces: [line_trace, label_trace].
    """
    x_lines: List = []
    y_lines: List = []
    z_lines: List = []
    x_tips: List[float] = []
    y_tips: List[float] = []
    z_tips: List[float] = []
    tip_labels: List[str] = []

    for i, idx in enumerate(top_indices):
        channel_name = channel_names[int(idx)] if int(idx) < len(channel_names) else f"CH{idx}"
        tip_x, tip_y, tip_z = (
            float(scaled_tip_vectors[i, 0]),
            float(scaled_tip_vectors[i, 1]),
            float(scaled_tip_vectors[i, 2]),
        )
        # None breaks the line between arrows so they appear as separate segments.
        x_lines.extend([0, tip_x, None])
        y_lines.extend([0, tip_y, None])
        z_lines.extend([0, tip_z, None])
        x_tips.append(tip_x)
        y_tips.append(tip_y)
        z_tips.append(tip_z)
        tip_labels.append(channel_name)

    line_trace = go.Scatter3d(  # type: ignore[attr-defined]
        x=x_lines,
        y=y_lines,
        z=z_lines,
        mode="lines",
        line=dict(color="#8b0000", width=3),
        name="Loadings",
        legendgroup="loadings",
        showlegend=True,
        hoverinfo="skip",
    )
    label_trace = go.Scatter3d(  # type: ignore[attr-defined]
        x=x_tips,
        y=y_tips,
        z=z_tips,
        mode="text",
        text=tip_labels,
        textfont=dict(size=9, color="#8b0000"),
        name="Loading labels",
        legendgroup="loadings",
        showlegend=False,
        hovertemplate="%{text}<extra></extra>",
    )
    return [line_trace, label_trace]


# ---------------------------------------------------------------------------
# 3D Interactive PCA public functions
# ---------------------------------------------------------------------------


def plot_pca_3d_scatter(
    anndata_object: anndata.AnnData,
    group_by: str,
    n_events_per_group: int = 500,
    pc_x: int = 1,
    pc_y: int = 2,
    pc_z: int = 3,
    hover_columns: Optional[List[str]] = None,
    marker_size: int = 1,
    marker_opacity: float = 0.25,
    width: int = 900,
    height: int = 700,
    title: str = "",
) -> object:
    """
    Draw an interactive 3D PCA scatter plot with mouse rotation and hover tooltips.

    Each event is drawn as a dot in 3D PCA space (three principal components
    simultaneously).  The plot is fully interactive: rotate by dragging, zoom
    with the scroll wheel, and hover over any dot to see its metadata.

    One Plotly trace is created per group so each group can be shown/hidden
    independently by clicking its entry in the legend.

    This function returns a go.Figure rather than calling plt.show() internally.
    In a Jupyter or Colab notebook the figure auto-displays when it is the last
    expression in a cell; use figure.show() to force display anywhere.

    Arguments:
        anndata_object: AnnData with .obsm['X_pca'] populated by compute_pca().
        group_by: Name of the .obs column to color events by (e.g. "subject_ID").
        n_events_per_group: Maximum events drawn per group (subsampled randomly).
        pc_x: 1-based index of the PC to place on the X axis (default: 1).
        pc_y: 1-based index of the PC to place on the Y axis (default: 2).
        pc_z: 1-based index of the PC to place on the Z axis (default: 3).
        hover_columns: Additional .obs column names to include in hover tooltips.
                       subject_ID and group_by are always shown automatically.
                       Columns absent from .obs are skipped with a warning.
        marker_size: Dot radius in pixels (default: 3).
        marker_opacity: Dot transparency, 0 (invisible) to 1 (opaque) (default: 0.6).
        width: Figure width in pixels (default: 900).
        height: Figure height in pixels (default: 700).

    Returns:
        go.Figure: A Plotly Figure. Call figure.show() to display it in a notebook.
    """
    go = _import_plotly()
    _require_pca_computed(anndata_object)

    if group_by not in anndata_object.obs.columns:
        raise ValueError(
            f"Column '{group_by}' not found in .obs. "
            f"Available: {list(anndata_object.obs.columns)}"
        )

    _validate_pc_indices(anndata_object, pc_x, pc_y, pc_z)

    # Validate and filter hover_columns.
    if hover_columns is None:
        validated_hover_columns: List[str] = []
    else:
        validated_hover_columns = []
        missing_hover = []
        for col in hover_columns:
            if col in anndata_object.obs.columns:
                validated_hover_columns.append(col)
            else:
                missing_hover.append(col)
        if missing_hover:
            print(
                f"WARNING [plot_pca_3d_scatter]: hover column(s) not found in .obs "
                f"and will be skipped: {missing_hover}"
            )

    pca_coords = anndata_object.obsm[_OBSM_PCA_KEY]
    var_ratios = anndata_object.uns[_PCA_VAR_RATIO_KEY]

    obs_df = anndata_object.obs.copy()
    obs_df["_pc_x"] = pca_coords[:, pc_x - 1]
    obs_df["_pc_y"] = pca_coords[:, pc_y - 1]
    obs_df["_pc_z"] = pca_coords[:, pc_z - 1]

    unique_groups = sort_values_for_legend(obs_df[group_by].dropna().unique())

    if len(unique_groups) == 0:
        raise ValueError(
            f"Column '{group_by}' has no non-NaN values — no groups to plot. "
            "The column is likely entirely NaN (e.g. the clinical variable was not populated). "
            f"Tip: print(anndata_object.obs['{group_by}'].value_counts(dropna=False)) to inspect it."
        )

    color_map = _build_color_map(anndata_object, group_by, unique_groups)

    # Subsample per group (identical logic to plot_pca_scatter).
    group_frames = []
    for group_value in unique_groups:
        group_df = obs_df[obs_df[group_by] == group_value]
        if len(group_df) > n_events_per_group:
            group_df = group_df.sample(n=n_events_per_group, random_state=42)
        group_frames.append(group_df)

    plot_df = pd.concat(group_frames, ignore_index=True)
    # Shuffle ALL groups together so no single group is drawn last and covers others.
    # A single trace is used for the data so the shuffled order is preserved in rendering.
    plot_df = plot_df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Build per-row colors and hover texts in shuffled order.
    row_colors = [color_map.get(str(v), "#999999") for v in plot_df[group_by].values]
    all_hover_texts = []
    for _, row in plot_df.iterrows():
        grp_val = row[group_by]
        lines = [f"<b>{group_by}</b>: {grp_val}"]
        if _SUBJECT_ID_COLUMN in plot_df.columns and group_by != _SUBJECT_ID_COLUMN:
            lines.append(f"subject_ID: {row[_SUBJECT_ID_COLUMN]}")
        lines.append(f"PC{pc_x}: {row['_pc_x']:.4f}")
        lines.append(f"PC{pc_y}: {row['_pc_y']:.4f}")
        lines.append(f"PC{pc_z}: {row['_pc_z']:.4f}")
        for col in validated_hover_columns:
            if col in row.index:
                lines.append(f"{col}: {row[col]}")
        all_hover_texts.append("<br>".join(lines))

    # One trace for all shuffled points — preserves render order across all groups.
    traces = [go.Scatter3d(  # type: ignore[attr-defined]
        x=plot_df["_pc_x"].values,
        y=plot_df["_pc_y"].values,
        z=plot_df["_pc_z"].values,
        mode="markers",
        marker=dict(size=marker_size, color=row_colors, opacity=marker_opacity),
        text=all_hover_texts,
        hovertemplate="%{text}<extra></extra>",
        showlegend=False,
    )]

    # Invisible traces per group for the color legend.
    for group_value in unique_groups:
        color = color_map.get(str(group_value), "#999999")
        traces.append(go.Scatter3d(  # type: ignore[attr-defined]
            x=[None], y=[None], z=[None],
            mode="markers",
            name=str(group_value),
            marker=dict(size=marker_size, color=color),
            showlegend=True,
        ))

    pc_x_label = f"PC{pc_x} ({var_ratios[pc_x - 1]:.1%})" if pc_x - 1 < len(var_ratios) else f"PC{pc_x}"
    pc_y_label = f"PC{pc_y} ({var_ratios[pc_y - 1]:.1%})" if pc_y - 1 < len(var_ratios) else f"PC{pc_y}"
    pc_z_label = f"PC{pc_z} ({var_ratios[pc_z - 1]:.1%})" if pc_z - 1 < len(var_ratios) else f"PC{pc_z}"

    species = get_species_label(anndata_object)
    species_prefix = f"[{species}]  " if species else ""
    figure = go.Figure(  # type: ignore[attr-defined]
        data=traces,
        layout=go.Layout(  # type: ignore[attr-defined]
            title=title if title else f"{species_prefix}PCA 3D Scatter — grouped by '{group_by}' | {len(plot_df):,} events",
            scene=dict(
                xaxis_title=pc_x_label,
                yaxis_title=pc_y_label,
                zaxis_title=pc_z_label,
            ),
            legend_title_text=group_by,
            width=width,
            height=height,
            margin=dict(l=0, r=0, b=0, t=40),
        ),
    )

    figure.add_annotation(
        text=get_run_context_footer_text(anndata_object),
        xref="paper", yref="paper",
        x=0.5, y=-0.07,
        showarrow=False,
        font=dict(size=9, color="gray"),
        align="center",
    )
    print(get_run_context_console_text(anndata_object))
    print(
        f"[plot_pca_3d_scatter] group_by='{group_by}' | "
        f"axes=PC{pc_x}/PC{pc_y}/PC{pc_z} | "
        f"{len(unique_groups)} groups | {len(plot_df):,} events plotted"
    )
    return figure


def plot_pca_3d_biplot(
    anndata_object: anndata.AnnData,
    group_by: str,
    top_n_variables: int = 5,
    labels: Optional[Union[str, List[str]]] = None,
    pc_x: int = 1,
    pc_y: int = 2,
    pc_z: int = 3,
    width: int = 900,
    height: int = 700,
    title: str = "",
) -> object:
    """
    Draw an interactive 3D PCA biplot: group centroids and channel loading arrows.

    Centroids (group mean positions in PCA space) and loading arrows (direction
    each channel pulls in PCA space) are drawn at the same scale.  This means
    a centroid near an arrow tip indicates that group scores high on that channel.

    The scale is computed from the combined maximum 3D norm of all centroids and
    all arrow tips: scale = 0.9 / max_norm.  Both are then multiplied by this
    scale so their relative positions are preserved and directly comparable.

    Arguments:
        anndata_object: AnnData with .obsm['X_pca'] and .uns['pca_loadings']
                        populated by compute_pca().
        group_by: Name of the .obs column to color centroids by.
        top_n_variables: Number of top-loading channels to display as arrows,
                         ranked by 3D loading magnitude √(PCx² + PCy² + PCz²).
        labels: Optional .obs column name(s) appended to centroid labels
                separated by " | ".  Same behaviour as plot_pca_biplot().
        pc_x: 1-based index of the PC to place on the X axis (default: 1).
        pc_y: 1-based index of the PC to place on the Y axis (default: 2).
        pc_z: 1-based index of the PC to place on the Z axis (default: 3).
        width: Figure width in pixels (default: 900).
        height: Figure height in pixels (default: 700).

    Returns:
        go.Figure: A Plotly Figure. Call figure.show() to display it in a notebook.
    """
    go = _import_plotly()
    _require_pca_computed(anndata_object)

    if group_by not in anndata_object.obs.columns:
        raise ValueError(
            f"Column '{group_by}' not found in .obs. "
            f"Available: {list(anndata_object.obs.columns)}"
        )

    _validate_pc_indices(anndata_object, pc_x, pc_y, pc_z)

    if _PCA_LOADINGS_KEY not in anndata_object.uns:
        raise ValueError(
            f"PCA loadings not found in .uns['{_PCA_LOADINGS_KEY}']. "
            "Run compute_pca() first."
        )

    # Normalise labels parameter (same logic as plot_pca_biplot).
    if labels is None:
        label_columns: List[str] = []
    elif isinstance(labels, str):
        label_columns = [labels]
    else:
        label_columns = list(labels)

    missing_label_columns = [c for c in label_columns if c not in anndata_object.obs.columns]
    if missing_label_columns:
        print(
            f"WARNING [plot_pca_3d_biplot]: label column(s) not found in .obs "
            f"and will be skipped: {missing_label_columns}"
        )

    loadings = anndata_object.uns[_PCA_LOADINGS_KEY]   # (n_channels, n_components)
    var_ratios = anndata_object.uns[_PCA_VAR_RATIO_KEY]
    channel_names = anndata_object.uns.get(_PCA_CHANNEL_NAMES_KEY, [])
    pca_coords = anndata_object.obsm[_OBSM_PCA_KEY]

    # Extract the 3-column loading sub-matrix for the three requested PCs.
    loadings_3d = loadings[:, [pc_x - 1, pc_y - 1, pc_z - 1]]  # (n_channels, 3)

    # Select top_n channels ranked by 3D loading magnitude.
    loading_magnitudes = np.linalg.norm(loadings_3d, axis=1)
    top_indices = np.argsort(loading_magnitudes)[::-1][:top_n_variables]
    top_loading_vectors = loadings_3d[top_indices]  # (top_n, 3)

    # Compute group centroids on the full dataset (not subsampled).
    obs_df = anndata_object.obs.copy()
    obs_df["_pc_x"] = pca_coords[:, pc_x - 1]
    obs_df["_pc_y"] = pca_coords[:, pc_y - 1]
    obs_df["_pc_z"] = pca_coords[:, pc_z - 1]
    centroid_df = obs_df.groupby(group_by)[["_pc_x", "_pc_y", "_pc_z"]].mean()

    if len(centroid_df) == 0:
        raise ValueError(
            f"Column '{group_by}' has no non-NaN values — no centroids to plot. "
            "The column is likely entirely NaN (e.g. the clinical variable was not populated). "
            f"Tip: print(anndata_object.obs['{group_by}'].value_counts(dropna=False)) to inspect it."
        )

    centroid_matrix = centroid_df.values  # (n_groups, 3)

    # Shared scale: fit all elements (centroids + arrow tips) into a unit sphere.
    # Using one scale preserves the visual relationship between centroids and arrows.
    arrow_norms = np.linalg.norm(top_loading_vectors, axis=1)
    centroid_norms = np.linalg.norm(centroid_matrix, axis=1)
    max_norm = max(
        float(arrow_norms.max()) if len(arrow_norms) > 0 else 1.0,
        float(centroid_norms.max()) if len(centroid_norms) > 0 else 1.0,
    )
    scale = 0.9 / max_norm if max_norm > 0 else 1.0

    scaled_top_loading_vectors = top_loading_vectors * scale
    scaled_centroid_matrix = centroid_matrix * scale

    # Print top channels.
    print(f"\nTop {top_n_variables} channels by 3D loading magnitude "
          f"(PC{pc_x}/PC{pc_y}/PC{pc_z}):")
    for rank, idx in enumerate(top_indices, start=1):
        name = channel_names[int(idx)] if int(idx) < len(channel_names) else f"CH{idx}"
        print(f"  {rank}. {name:<20} magnitude={loading_magnitudes[idx]:.3f}")
    print()

    unique_groups = sort_values_for_legend(obs_df[group_by].dropna().unique())
    color_map = _build_color_map(anndata_object, group_by, unique_groups)

    traces = []

    # One centroid trace per group.
    for i, group_value in enumerate(centroid_df.index):
        color = color_map.get(str(group_value), "#999999")
        cx, cy, cz = (
            float(scaled_centroid_matrix[i, 0]),
            float(scaled_centroid_matrix[i, 1]),
            float(scaled_centroid_matrix[i, 2]),
        )
        # Raw (unscaled) coordinates in the hover for scientific readability.
        raw_cx, raw_cy, raw_cz = (
            float(centroid_matrix[i, 0]),
            float(centroid_matrix[i, 1]),
            float(centroid_matrix[i, 2]),
        )
        centroid_label = _build_centroid_label(
            str(group_value), obs_df, group_by, label_columns
        )
        hover_html = (
            f"<b>{group_value}</b><br>"
            f"PC{pc_x}: {raw_cx:.4f}<br>"
            f"PC{pc_y}: {raw_cy:.4f}<br>"
            f"PC{pc_z}: {raw_cz:.4f}"
        )
        traces.append(go.Scatter3d(  # type: ignore[attr-defined]
            x=[cx],
            y=[cy],
            z=[cz],
            mode="markers+text",
            name=str(group_value),
            marker=dict(
                size=12,
                color=color,
                symbol="cross",
                line=dict(color="black", width=1),
            ),
            text=[centroid_label],
            textposition="top center",
            textfont=dict(size=9, color=color),
            hovertemplate=hover_html + "<extra></extra>",
        ))

    # Loading arrow traces (lines + labels, shared legend group).
    traces.extend(_build_arrow_traces_3d(
        go, channel_names, scaled_top_loading_vectors, top_indices
    ))

    pc_x_label = f"PC{pc_x} ({var_ratios[pc_x - 1]:.1%})" if pc_x - 1 < len(var_ratios) else f"PC{pc_x}"
    pc_y_label = f"PC{pc_y} ({var_ratios[pc_y - 1]:.1%})" if pc_y - 1 < len(var_ratios) else f"PC{pc_y}"
    pc_z_label = f"PC{pc_z} ({var_ratios[pc_z - 1]:.1%})" if pc_z - 1 < len(var_ratios) else f"PC{pc_z}"

    species = get_species_label(anndata_object)
    species_prefix = f"[{species}]  " if species else ""
    figure = go.Figure(  # type: ignore[attr-defined]
        data=traces,
        layout=go.Layout(  # type: ignore[attr-defined]
            title=title if title else (
                f"{species_prefix}PCA 3D Biplot — grouped by '{group_by}' | "
                f"top {top_n_variables} loading channels"
            ),
            scene=dict(
                xaxis_title=pc_x_label,
                yaxis_title=pc_y_label,
                zaxis_title=pc_z_label,
            ),
            legend_title_text=group_by,
            width=width,
            height=height,
            margin=dict(l=0, r=0, b=0, t=40),
        ),
    )

    figure.add_annotation(
        text=get_run_context_footer_text(anndata_object),
        xref="paper", yref="paper",
        x=0.5, y=-0.07,
        showarrow=False,
        font=dict(size=9, color="gray"),
        align="center",
    )
    print(get_run_context_console_text(anndata_object))
    print(
        f"[plot_pca_3d_biplot] group_by='{group_by}' | "
        f"axes=PC{pc_x}/PC{pc_y}/PC{pc_z} | "
        f"{len(unique_groups)} groups | {top_n_variables} loading arrows"
    )
    return figure


def plot_pca_3d_trajectory(
    anndata_object: anndata.AnnData,
    condition_column: str,
    time_column: str,
    top_n_variables: int = 5,
    show_loading_arrows: bool = True,
    time_order: Optional[List] = None,
    pc_x: int = 1,
    pc_y: int = 2,
    pc_z: int = 3,
    width: int = 900,
    height: int = 700,
    title: str = "",
) -> object:
    """
    Draw an interactive 3D PCA trajectory plot showing condition centroids over time.

    For each (condition, timepoint) pair, one centroid is computed as the mean
    of all event PCA coordinates belonging to that group.  Centroids belonging to
    the same condition are connected by lines in chronological order so the
    trajectory through 3D PCA space is visible.  If a condition has no events at
    a timepoint, the line breaks and resumes at the next available timepoint.

    Centroids and loading arrows share the same scale (identical to
    plot_pca_3d_biplot) so their visual positions can be directly compared.

    Arguments:
        anndata_object: AnnData with .obsm['X_pca'] populated by compute_pca().
        condition_column: .obs column defining conditions (e.g. "diet" → "AL"/"IF").
        time_column: .obs column defining the time axis (e.g. "age" → 0, 2, 8).
        top_n_variables: Number of top-loading channels to draw as arrows.
        show_loading_arrows: When False, only centroids and lines are drawn.
        time_order: Explicit ordered list of timepoint values.  When None,
                    timepoints are sorted numerically when possible, else
                    lexicographically.
        pc_x: 1-based index of the PC to place on the X axis (default: 1).
        pc_y: 1-based index of the PC to place on the Y axis (default: 2).
        pc_z: 1-based index of the PC to place on the Z axis (default: 3).
        width: Figure width in pixels (default: 900).
        height: Figure height in pixels (default: 700).

    Returns:
        go.Figure: A Plotly Figure. Call figure.show() to display it in a notebook.
    """
    go = _import_plotly()
    _require_pca_computed(anndata_object)

    if condition_column not in anndata_object.obs.columns:
        raise ValueError(
            f"condition_column '{condition_column}' not found in .obs. "
            f"Available: {list(anndata_object.obs.columns)}"
        )
    if time_column not in anndata_object.obs.columns:
        raise ValueError(
            f"time_column '{time_column}' not found in .obs. "
            f"Available: {list(anndata_object.obs.columns)}"
        )

    _validate_pc_indices(anndata_object, pc_x, pc_y, pc_z)

    loadings = anndata_object.uns[_PCA_LOADINGS_KEY]  # (n_channels, n_components)
    var_ratios = anndata_object.uns[_PCA_VAR_RATIO_KEY]
    channel_names = anndata_object.uns.get(_PCA_CHANNEL_NAMES_KEY, [])
    pca_coords = anndata_object.obsm[_OBSM_PCA_KEY]

    # Build working dataframe.
    obs_df = anndata_object.obs[[condition_column, time_column]].copy()
    obs_df["_pc_x"] = pca_coords[:, pc_x - 1]
    obs_df["_pc_y"] = pca_coords[:, pc_y - 1]
    obs_df["_pc_z"] = pca_coords[:, pc_z - 1]

    # Determine timepoint order.
    available_timepoints = obs_df[time_column].dropna().unique()
    if time_order is not None:
        sorted_timepoints = [t for t in time_order if t in available_timepoints]
    else:
        try:
            sorted_timepoints = sorted(available_timepoints, key=float)
        except (TypeError, ValueError):
            sorted_timepoints = sort_values_for_legend(available_timepoints)

    # Compute centroids: mean position per (condition, timepoint).
    centroid_df = (
        obs_df.groupby([condition_column, time_column])[["_pc_x", "_pc_y", "_pc_z"]]
        .mean()
        .reset_index()
    )

    print(
        f"[plot_pca_3d_trajectory] condition='{condition_column}' | "
        f"time='{time_column}' | arrows={show_loading_arrows} | "
        f"timepoints={sorted_timepoints}"
    )
    print(f"  {len(centroid_df)} centroids computed.")

    # Loading vectors for the 3 PCs.
    loadings_3d = loadings[:, [pc_x - 1, pc_y - 1, pc_z - 1]]  # (n_channels, 3)
    loading_magnitudes = np.linalg.norm(loadings_3d, axis=1)
    top_indices = np.argsort(loading_magnitudes)[::-1][:top_n_variables]
    top_loading_vectors = loadings_3d[top_indices]  # (top_n, 3)

    # Shared scale: all centroids + all arrows fit within a unit sphere.
    all_centroid_positions = centroid_df[["_pc_x", "_pc_y", "_pc_z"]].values
    centroid_norms = np.linalg.norm(all_centroid_positions, axis=1)
    arrow_norms = np.linalg.norm(top_loading_vectors, axis=1)
    max_norm = max(
        float(centroid_norms.max()) if len(centroid_norms) > 0 else 1.0,
        float(arrow_norms.max()) if show_loading_arrows and len(arrow_norms) > 0 else 1.0,
    )
    scale = 0.9 / max_norm if max_norm > 0 else 1.0

    scaled_top_loading_vectors = top_loading_vectors * scale

    unique_conditions = sort_values_for_legend(obs_df[condition_column].dropna().unique())
    color_map = _build_color_map(anndata_object, condition_column, unique_conditions)

    traces = []

    # One trace per condition — lines + markers in chronological time order.
    for condition_value in unique_conditions:
        color = color_map.get(str(condition_value), "#999999")
        condition_rows = centroid_df[centroid_df[condition_column] == condition_value]

        trajectory_x: List = []
        trajectory_y: List = []
        trajectory_z: List = []
        label_texts: List[str] = []
        hover_texts: List[str] = []

        for timepoint in sorted_timepoints:
            row = condition_rows[condition_rows[time_column] == timepoint]
            if len(row) == 0:
                # Break the line when a timepoint is missing.
                trajectory_x.append(None)
                trajectory_y.append(None)
                trajectory_z.append(None)
                label_texts.append("")
                hover_texts.append("")
                continue
            raw_cx = float(row["_pc_x"].iloc[0])
            raw_cy = float(row["_pc_y"].iloc[0])
            raw_cz = float(row["_pc_z"].iloc[0])
            trajectory_x.append(raw_cx * scale)
            trajectory_y.append(raw_cy * scale)
            trajectory_z.append(raw_cz * scale)
            label_texts.append(f"{condition_value} | {timepoint}")
            hover_texts.append(
                f"<b>{condition_value} | {timepoint}</b><br>"
                f"PC{pc_x}: {raw_cx:.4f}<br>"
                f"PC{pc_y}: {raw_cy:.4f}<br>"
                f"PC{pc_z}: {raw_cz:.4f}"
            )

        traces.append(go.Scatter3d(  # type: ignore[attr-defined]
            x=trajectory_x,
            y=trajectory_y,
            z=trajectory_z,
            mode="lines+markers+text",
            name=str(condition_value),
            line=dict(color=color, width=3),
            marker=dict(
                size=8,
                color=color,
                symbol="cross",
                line=dict(color="black", width=1),
            ),
            text=label_texts,
            textposition="top center",
            textfont=dict(size=8, color=color),
            hovertext=hover_texts,
            hovertemplate="%{hovertext}<extra></extra>",
        ))

    # Loading arrow traces (optional).
    if show_loading_arrows:
        traces.extend(_build_arrow_traces_3d(
            go, channel_names, scaled_top_loading_vectors, top_indices
        ))

    pc_x_label = f"PC{pc_x} ({var_ratios[pc_x - 1]:.1%})" if pc_x - 1 < len(var_ratios) else f"PC{pc_x}"
    pc_y_label = f"PC{pc_y} ({var_ratios[pc_y - 1]:.1%})" if pc_y - 1 < len(var_ratios) else f"PC{pc_y}"
    pc_z_label = f"PC{pc_z} ({var_ratios[pc_z - 1]:.1%})" if pc_z - 1 < len(var_ratios) else f"PC{pc_z}"

    species = get_species_label(anndata_object)
    species_prefix = f"[{species}]  " if species else ""
    figure = go.Figure(  # type: ignore[attr-defined]
        data=traces,
        layout=go.Layout(  # type: ignore[attr-defined]
            title=title if title else f"{species_prefix}PCA 3D Trajectory — '{condition_column}' over '{time_column}'",
            scene=dict(
                xaxis_title=pc_x_label,
                yaxis_title=pc_y_label,
                zaxis_title=pc_z_label,
            ),
            legend_title_text=condition_column,
            width=width,
            height=height,
            margin=dict(l=0, r=0, b=0, t=40),
        ),
    )

    figure.add_annotation(
        text=get_run_context_footer_text(anndata_object),
        xref="paper", yref="paper",
        x=0.5, y=-0.07,
        showarrow=False,
        font=dict(size=9, color="gray"),
        align="center",
    )
    print(get_run_context_console_text(anndata_object))
    print(
        f"[plot_pca_3d_trajectory] condition='{condition_column}' | "
        f"time='{time_column}' | axes=PC{pc_x}/PC{pc_y}/PC{pc_z}"
    )
    return figure
