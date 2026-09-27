"""
umap_plot.py

UMAP dimensionality reduction and scatter visualization for SFC AnnData.

Two-function design separates computation from display:
  - compute_umap() fits UMAP on a stratified random sample and stores the
    2-D coordinates in .obsm. Results are cached by parameter key so
    re-running with the same parameters skips recomputation.
  - plot_umap() subsamples from the stored coordinates and draws a scatter
    plot, coloring by any .obs column.

Why subsample for computation?
  Large cytometry datasets (100k–500k events) make full-dataset UMAP slow in
  Colab. The max_total_events cap (default 10,000) keeps the first call to
  compute_umap() under 1 minute. Subsequent plot_umap() calls are instant
  since they only draw from stored coordinates.

Typical usage:
    from mito_marker.analysis import compute_umap, plot_umap
    sfc_subset = compute_umap(sfc_subset)
    plot_umap(sfc_subset, group_by="subject_ID")
    plot_umap(sfc_subset, group_by="dilution")
"""

from math import ceil
from typing import List, Optional, Union

import anndata
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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

# .uns key where UMAP parameters are tracked.
_UMAP_PARAMS_KEY = "umap_params"

# .uns key holding serialized UMAP reducers, one per .obsm key, when
# compute_umap(keep_reducer=True) is used (ADR-015).
UMAP_REDUCERS_KEY = "umap_reducers"

# Column to use for stratified sampling when splitting events across subjects.
_STRATIFY_COLUMN = "subject_ID"

# subject_ID is colored from tab10 at plot time, not from the condition palette.
_SUBJECT_ID_COLUMN = "subject_ID"


def _build_umap_reducer(
    n_neighbors: int,
    min_dist: float,
    random_state: int,
    force_cpu: bool = False,
) -> object:
    """
    Return a UMAP reducer object, preferring cuML (GPU) when available.

    Tries to import cuml.manifold.UMAP first. If unavailable (cuML not installed
    or no CUDA device), falls back to umap-learn's CPU implementation. Both objects
    expose the same .fit_transform() interface.

    Parameters
    ----------
    n_neighbors : int
        Number of nearest neighbors for the UMAP graph.
    min_dist : float
        Minimum distance between embedded points.
    random_state : int
        Random seed for reproducibility.
    force_cpu : bool
        When True, always use umap-learn (needed when the reducer must be
        serialized for later reuse).

    Returns
    -------
    object
        A UMAP reducer instance (cuML or umap-learn).
    """
    if is_cuml_available() and not force_cpu:
        import cuml.manifold  # noqa: PLC0415
        print("  [GPU] cuML detected — using GPU-accelerated UMAP.")
        return cuml.manifold.UMAP(
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            random_state=random_state,
            n_components=2,
            verbose=False,
        )
    # Import directly from the core module to avoid umap's __init__.py
    # attempting to import ParametricUMAP → torch, which causes a circular
    # import error on some PyTorch versions in Colab.
    print("  [CPU] cuML not available — using umap-learn (CPU).")
    from umap.umap_ import UMAP  # noqa: PLC0415
    return UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=random_state,
        n_components=2,
        verbose=False,
    )


def compute_umap(
    anndata_object: anndata.AnnData,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    random_state: int = 42,
    max_total_events: int = 10_000,
    n_pca_components: Optional[int] = None,
    force_recompute: bool = False,
    embed_all_events: bool = False,
    keep_reducer: bool = False,
) -> anndata.AnnData:
    """
    Fit UMAP on a stratified random sample and store 2-D coordinates in .obsm.

    By default the computation is performed on the active layer (or raw .X if
    no layer is active). If feature selection is active, only the selected
    channels are used.

    When n_pca_components is provided, UMAP is run on the first
    n_pca_components columns of .obsm['X_pca'] instead of the raw data matrix.
    This is the standard single-cell pipeline (PCA → UMAP): PCA denoises the
    data so UMAP works in a cleaner, lower-dimensional space. compute_pca()
    must be called before using this option.

    The UMAP coordinates are stored under the key:
        .obsm['X_umap_n{n_neighbors}_d{min_dist}']               (raw channels)
        .obsm['X_umap_pca{n_pca_components}_n{n_neighbors}_d{min_dist}']  (PCA space)
    Example: .obsm['X_umap_n15_d0.1'] or .obsm['X_umap_pca15_n15_d0.1']

    If that key already exists in .obsm, the function prints a cache-hit message
    and returns immediately without recalculating. Pass force_recompute=True to
    override this behaviour (e.g. when changing max_total_events with the same
    n_neighbors/min_dist).

    Different parameter combinations produce distinct keys and coexist in .obsm.

    Arguments:
        anndata_object: SFC AnnData after selection and optional normalization.
        n_neighbors: UMAP n_neighbors parameter (local neighborhood size).
        min_dist: UMAP min_dist parameter (minimum distance in embedded space).
        random_state: Random seed for reproducibility.
        max_total_events: Maximum number of events to include in the UMAP fit.
                          Events are stratified by subject_ID when that column
                          is present, so no single subject dominates.
        n_pca_components: If provided, run UMAP on the first n_pca_components
                          columns of .obsm['X_pca'] instead of the raw data
                          matrix. compute_pca() must have been called first.
        force_recompute: If True, delete any existing cached UMAP coordinates
                         for these parameters and recompute from scratch.
        embed_all_events: If False (default), only the sampled events get
                          coordinates and every other row stays NaN. If True,
                          the events left out of the fit are placed on the
                          map with reducer.transform() — needed to cluster in
                          UMAP space, where every mitochondrion needs a position.
        keep_reducer: If True, the fitted reducer is serialized into
                      .uns['umap_reducers'][<obsm key>] (uint8 bytes,
                      h5ad-compatible) so a new dataset can later be placed on
                      the SAME map (cluster models, ADR-015). The CPU backend
                      (umap-learn) is used in this case because the GPU
                      reducer cannot be serialized reliably. The stored
                      reducer can weigh tens of MB and is copied into every
                      subset of this AnnData — hence opt-in.

    Returns:
        The same AnnData object with UMAP coordinates added to .obsm.
    """
    obsm_key = _build_umap_key(n_neighbors, min_dist, n_pca_components)

    # If the user explicitly requests a fresh computation, drop the cached key.
    if force_recompute and obsm_key in anndata_object.obsm:
        print(f"force_recompute=True — deleting cached .obsm['{obsm_key}'].")
        del anndata_object.obsm[obsm_key]

    # Cache check: skip recomputation if coordinates already exist.
    if obsm_key in anndata_object.obsm:
        print(
            f"UMAP coordinates found for n_neighbors={n_neighbors}, "
            f"min_dist={min_dist} — skipping recomputation."
        )
        print(f"=> Stored key: .obsm['{obsm_key}'] "
              f"({anndata_object.obsm[obsm_key].shape[0]} events)")
        return anndata_object

    print("=" * 60)
    print("COMPUTING UMAP")
    print("=" * 60)

    if n_pca_components is not None:
        # PCA → UMAP pipeline: use PCA coordinates as UMAP input.
        assert "X_pca" in anndata_object.obsm, (
            "n_pca_components was set but .obsm['X_pca'] does not exist. "
            "Call compute_pca() before compute_umap() with n_pca_components."
        )
        available_components = anndata_object.obsm["X_pca"].shape[1]
        assert n_pca_components <= available_components, (
            f"n_pca_components={n_pca_components} exceeds the number of PCA "
            f"components available ({available_components}). "
            f"Re-run compute_pca() with n_components >= {n_pca_components}."
        )
        data_matrix = np.asarray(
            anndata_object.obsm["X_pca"][:, :n_pca_components], dtype=np.float32
        )
        input_channel_names = [f"PC{index + 1}" for index in range(n_pca_components)]
        input_fingerprint = str(anndata_object.uns.get("pca_fingerprint", "unknown"))
        print(
            f"Input: PCA space — first {n_pca_components} components "
            f"of .obsm['X_pca'] (denoised representation)."
        )
    else:
        # Default pipeline: use raw channels (active layer + feature selection).
        data_matrix, input_channel_names = _get_data_matrix_and_channels(anndata_object)
        input_fingerprint = _get_active_layer_fingerprint(anndata_object)

    total_events = data_matrix.shape[0]
    print(f"Full dataset: {total_events:,} events × {data_matrix.shape[1]} dimensions")

    # Stratified sample: balance across subjects to avoid dominance.
    sample_indices = _stratified_sample(
        anndata_object.obs, max_total_events, random_state
    )
    sampled_matrix = data_matrix[sample_indices]
    print(
        f"Sampling {len(sample_indices):,} events "
        f"(from {total_events:,}) for UMAP fit."
    )

    reducer = _build_umap_reducer(
        n_neighbors, min_dist, random_state, force_cpu=keep_reducer
    )
    embedding = reducer.fit_transform(sampled_matrix)
    backend_name = "cuml" if "cuml" in type(reducer).__module__ else "umap-learn"

    # Store coordinates only for the sampled events.
    # Create a full-length array of NaN, then fill in the sampled rows.
    full_embedding = np.full((total_events, 2), np.nan, dtype=np.float32)
    # np.asarray() transfers the result back to CPU RAM when cuML returns a CuPy array.
    full_embedding[sample_indices] = np.asarray(embedding, dtype=np.float32)

    n_embedded_by_transform = 0
    if embed_all_events and len(sample_indices) < total_events:
        # Events outside the fit sample are placed on the frozen map; they do
        # not move the map. This is the standard umap-learn out-of-sample use.
        unsampled_mask = np.ones(total_events, dtype=bool)
        unsampled_mask[sample_indices] = False
        print(
            f"Placing the {int(unsampled_mask.sum()):,} events left out of the fit "
            "on the map with reducer.transform()…"
        )
        full_embedding[unsampled_mask] = np.asarray(
            reducer.transform(data_matrix[unsampled_mask]), dtype=np.float32
        )
        n_embedded_by_transform = int(unsampled_mask.sum())
    anndata_object.obsm[obsm_key] = full_embedding

    reducer_fingerprint = ""
    if keep_reducer:
        reducer_fingerprint = _store_umap_reducer(
            anndata_object, obsm_key, reducer, input_fingerprint
        )

    # Record active UMAP key in analysis config so plot_umap() finds it.
    if _ANALYSIS_CONFIG_KEY not in anndata_object.uns:
        anndata_object.uns[_ANALYSIS_CONFIG_KEY] = {}
    anndata_object.uns[_ANALYSIS_CONFIG_KEY][_UMAP_PARAMS_KEY] = {
        "active_key": obsm_key,
        "n_neighbors": n_neighbors,
        "min_dist": min_dist,
        "random_state": random_state,
        "n_events_sampled": len(sample_indices),
        "backend": backend_name,
        "n_pca_components": n_pca_components,
        "n_events_embedded_by_transform": n_embedded_by_transform,
        "input_channel_names": list(input_channel_names),
        "input_fingerprint": input_fingerprint,
        "reducer_fingerprint": reducer_fingerprint,
    }

    n_valid = int(np.isfinite(full_embedding[:, 0]).sum())
    print(f"=> UMAP complete. Coordinates stored in .obsm['{obsm_key}'].")
    print(f"=> {n_valid:,} of {total_events:,} events have valid UMAP coordinates.")
    print("=" * 60)

    return anndata_object


def plot_umap(
    anndata_object: anndata.AnnData,
    group_by: Union[str, List[str]],
    max_points_per_group: Optional[int] = None,
    n_events_per_group: int = 500,
    title: str = "",
    point_size: int = 7,
    point_alpha: float = 0.25,
) -> None:
    """
    Draw a UMAP scatter plot, coloring events by a .obs grouping column.

    Reads UMAP coordinates from the active .obsm key stored by compute_umap().
    Only events with valid (non-NaN) UMAP coordinates are plotted. Subsamples
    max_points_per_group events per group for visual clarity, then shuffles all
    groups together before plotting in a single scatter call so no group
    systematically covers another.

    When group_by is a list of column names, values from each column are
    combined with " / " to form one group label per unique combination
    (e.g. ["specie", "age_group"] → "Human / young", "Mouse / old", …).

    Arguments:
        anndata_object: AnnData with .obsm populated by compute_umap().
        group_by: Name of the .obs column to color by (e.g. "subject_ID"), or
                  a list of column names whose values are combined into a single
                  group label (e.g. ["specie", "age_group"]).
        max_points_per_group: Maximum number of events drawn per group.
                              Increase for denser scatter, decrease for speed.
                              Defaults to 500.
        n_events_per_group: Deprecated alias for max_points_per_group.
                            Kept for backwards compatibility — prefer
                            max_points_per_group in new code.
        title: Optional figure title. When empty (default), an informative
               title is generated automatically from the grouping column and
               UMAP parameters.
        point_size: Marker size for individual events (default: 7).
        point_alpha: Marker transparency, 0 (invisible) to 1 (opaque) (default: 0.25).

    Returns:
        None. The figure is displayed via plt.show().
    """
    # Resolve which limit to use: max_points_per_group takes priority;
    # fall back to n_events_per_group for backwards-compatible call sites.
    effective_max_points = max_points_per_group if max_points_per_group is not None else n_events_per_group

    obsm_key = _get_active_umap_key(anndata_object)

    # Validate and combine grouping columns into a single label Series.
    group_labels = _get_group_labels(anndata_object, group_by)
    group_by_label = " × ".join(group_by) if isinstance(group_by, list) else group_by

    print(get_run_context_console_text(anndata_object))
    print(f"[plot_umap] Using UMAP coordinates from: .obsm['{obsm_key}']")
    print(f"[plot_umap] Grouping by: '{group_by_label}' | max {effective_max_points} events/group")

    if obsm_key not in anndata_object.obsm:
        raise ValueError(
            f"UMAP key '{obsm_key}' not found in .obsm. "
            "Run compute_umap() first."
        )

    # Extract UMAP coordinates.
    all_coords = anndata_object.obsm[obsm_key]  # shape (n_obs, 2)
    obs_df = anndata_object.obs.copy()
    obs_df["_umap_x"] = all_coords[:, 0]
    obs_df["_umap_y"] = all_coords[:, 1]
    obs_df["_group_label"] = group_labels.values

    # Retain only events with valid UMAP coordinates.
    valid_mask = np.isfinite(obs_df["_umap_x"]) & np.isfinite(obs_df["_umap_y"])
    obs_df = obs_df[valid_mask]

    # Subsample per group.
    group_frames = []
    unique_groups = sort_values_for_legend(obs_df["_group_label"].dropna().unique())
    for group_value in unique_groups:
        group_df = obs_df[obs_df["_group_label"] == group_value]
        if len(group_df) > effective_max_points:
            group_df = group_df.sample(n=effective_max_points, random_state=42)
        group_frames.append(group_df)

    if not group_frames:
        print("WARNING: No events with valid UMAP coordinates found.")
        return

    plot_df = pd.concat(group_frames, ignore_index=True)

    # Shuffle rows so groups are visually interleaved when drawn as a single
    # scatter call. Plotting group-by-group would make the last group cover all
    # others regardless of shuffling.
    plot_df = plot_df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Build color map.
    color_map = _build_color_map_umap(anndata_object, group_by, unique_groups)

    # Assign a per-row color vector so the single scatter call respects the
    # shuffled order (each row keeps its group color).
    row_colors = plot_df["_group_label"].astype(str).map(
        lambda g: color_map.get(g, "#999999")
    ).tolist()

    fig, ax = plt.subplots(figsize=(10, 7))

    # Single scatter call — draws points in shuffled order so no group
    # systematically covers another.
    ax.scatter(
        plot_df["_umap_x"].values,
        plot_df["_umap_y"].values,
        c=row_colors,
        s=point_size,
        alpha=point_alpha,
        edgecolors="none",
    )

    # Add invisible per-group scatter handles for the legend.
    for group_value in unique_groups:
        color = color_map.get(str(group_value), "#999999")
        ax.scatter([], [], c=color, s=point_size, label=str(group_value))

    ax.set_xlabel("UMAP 1", fontsize=12)
    ax.set_ylabel("UMAP 2", fontsize=12)

    umap_params = anndata_object.uns.get(_ANALYSIS_CONFIG_KEY, {}).get(_UMAP_PARAMS_KEY, {})
    n_neighbors = umap_params.get("n_neighbors", "?")
    min_dist = umap_params.get("min_dist", "?")
    species = get_species_label(anndata_object)
    species_prefix = f"[{species}]  " if species else ""
    auto_title = (
        f"{species_prefix}UMAP — grouped by '{group_by_label}'\n"
        f"n_neighbors={n_neighbors}, min_dist={min_dist} | "
        f"{len(plot_df):,} events plotted"
    )
    ax.set_title(title if title else auto_title, fontsize=13)
    _apply_legend(ax, group_by_label, len(unique_groups))
    fig.text(
        0.5, 0.01, get_run_context_footer_text(anndata_object),
        ha="center", va="bottom",
        fontsize=6, color="gray", style="italic",
    )
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

_MANY_GROUPS_THRESHOLD = 15


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


def _build_umap_key(
    n_neighbors: int,
    min_dist: float,
    n_pca_components: Optional[int] = None,
) -> str:
    """
    Build the .obsm key string for a given set of UMAP parameters.

    When n_pca_components is provided, the key encodes that UMAP was run on
    PCA space rather than raw channels, so both variants can coexist in .obsm.

    Arguments:
        n_neighbors: UMAP n_neighbors value.
        min_dist: UMAP min_dist value.
        n_pca_components: Number of PCA components used as UMAP input, or None
                          when UMAP is run directly on the data matrix.

    Returns:
        String key, e.g. 'X_umap_n15_d0.1' or 'X_umap_pca15_n15_d0.1'.
    """
    if n_pca_components is not None:
        return f"X_umap_pca{n_pca_components}_n{n_neighbors}_d{min_dist}"
    return f"X_umap_n{n_neighbors}_d{min_dist}"


def _get_active_umap_key(anndata_object: anndata.AnnData) -> str:
    """
    Read the active UMAP .obsm key from .uns['analysis_config']['umap_params'].

    Falls back to the default key (n_neighbors=15, min_dist=0.1) if no active
    key is recorded.

    Arguments:
        anndata_object: AnnData whose .uns is searched.

    Returns:
        String key to use for .obsm lookup.
    """
    umap_params = (
        anndata_object.uns
        .get(_ANALYSIS_CONFIG_KEY, {})
        .get(_UMAP_PARAMS_KEY, {})
    )
    return umap_params.get("active_key", _build_umap_key(15, 0.1))


def _get_analytical_mask_for_channels(
    anndata_object: anndata.AnnData,
    channel_names: List[str],
) -> np.ndarray:
    """
    Return a boolean mask (length = len(channel_names)) that is True for
    analytically usable channels.

    Reads .var['is_non_analytical'] when the column is present, selecting only
    the rows corresponding to channel_names.  Falls back to SFC_NON_ANALYTICAL_CHANNELS
    for AnnData objects that pre-date the introduction of this column.

    Arguments:
        anndata_object: The AnnData whose .var is inspected.
        channel_names: Ordered list of channel names to mask.

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


def _get_data_matrix(anndata_object: anndata.AnnData) -> np.ndarray:
    """
    Return the data matrix to use for UMAP fitting.

    Respects the active layer and active feature selection. Converts sparse
    matrices to dense arrays.

    Arguments:
        anndata_object: AnnData with optional .uns['analysis_config'].

    Returns:
        Dense numpy array of shape (n_obs, n_channels).
    """
    data_matrix, _ = _get_data_matrix_and_channels(anndata_object)
    return data_matrix


def _get_active_layer_fingerprint(anndata_object: anndata.AnnData) -> str:
    """
    Return the frozen-scaling fingerprint of the active layer.

    Arguments:
        anndata_object: AnnData to inspect.

    Returns:
        The fingerprint, "raw_X" when no layer is active, or "unknown" when the
        active layer was created without recorded parameters.
    """
    # Imported here to avoid a circular import at module load.
    from mito_marker.analysis.normalization import get_layer_parameters  # noqa: PLC0415

    active_layer = anndata_object.uns.get(_ANALYSIS_CONFIG_KEY, {}).get("active_layer")
    if active_layer is None:
        return "raw_X"
    layer_parameters = get_layer_parameters(anndata_object, active_layer)
    return "unknown" if layer_parameters is None else str(layer_parameters["fingerprint"])


def _store_umap_reducer(
    anndata_object: anndata.AnnData,
    obsm_key: str,
    reducer: object,
    input_fingerprint: str,
) -> str:
    """
    Serialize a fitted UMAP reducer into .uns['umap_reducers'][obsm_key].

    The pickled bytes are stored as a uint8 numpy array, which .h5ad files
    can hold. Only load reducers produced by your own team: unpickling runs code.

    Arguments:
        anndata_object: AnnData updated in place.
        obsm_key: .obsm key of the embedding this reducer produced.
        reducer: The fitted umap-learn reducer.
        input_fingerprint: Fingerprint of the space the reducer was fitted on.

    Returns:
        Fingerprint of the stored reducer.
    """
    import pickle  # noqa: PLC0415

    reducer_bytes = pickle.dumps(reducer)
    stored_reducers = dict(anndata_object.uns.get(UMAP_REDUCERS_KEY, {}))
    stored_reducers[obsm_key] = np.frombuffer(reducer_bytes, dtype=np.uint8).copy()
    anndata_object.uns[UMAP_REDUCERS_KEY] = stored_reducers
    reducer_fingerprint = compute_fingerprint(input_fingerprint, reducer_bytes)
    print(
        f"=> Reducer stored in .uns['{UMAP_REDUCERS_KEY}']['{obsm_key}'] "
        f"({len(reducer_bytes) / 1e6:.1f} MB) — fingerprint {reducer_fingerprint}."
    )
    return reducer_fingerprint


def _get_data_matrix_and_channels(anndata_object: anndata.AnnData) -> tuple:
    """
    Return the UMAP input matrix and the names of its columns.

    Respects the active layer and active feature selection, and always drops
    non-analytical channels.

    Arguments:
        anndata_object: AnnData with optional .uns['analysis_config'].

    Returns:
        Tuple (dense numpy array of shape (n_obs, n_channels), list of channel names).
    """
    analysis_config = anndata_object.uns.get(_ANALYSIS_CONFIG_KEY, {})
    active_layer = analysis_config.get("active_layer")
    active_selection = analysis_config.get("active_selection")

    # Choose the data source.
    if active_layer is not None and active_layer in anndata_object.layers:
        full_matrix = anndata_object.layers[active_layer]
        print(f"[compute_umap] Using layer: '{active_layer}'")
    else:
        full_matrix = anndata_object.X
        print("[compute_umap] Using raw .X")

    if hasattr(full_matrix, "toarray"):
        full_matrix = full_matrix.toarray()

    # np.asarray avoids an unnecessary 6 GB copy when the matrix is already
    # float32 contiguous (the common case after normalisation).
    full_matrix = np.asarray(full_matrix, dtype=np.float32)

    # Track which channel names remain after feature selection.
    channel_names = anndata_object.var_names.tolist()

    # Apply feature selection if active.
    if active_selection is not None:
        selection_column = f"is_selected_{active_selection}"
        if selection_column in anndata_object.var.columns:
            mask = anndata_object.var[selection_column].values
            full_matrix = full_matrix[:, mask]
            channel_names = [ch for ch, keep in zip(channel_names, mask) if keep]
            print(
                f"[compute_umap] Feature selection ({active_selection}): "
                f"{mask.sum()} of {anndata_object.n_vars} channels used."
            )

    # Always exclude non-analytical channels (Time, FlowAI) from UMAP input.
    # The mask is read from .var['is_non_analytical'] when available (set at
    # ingestion time); falls back to SFC_NON_ANALYTICAL_CHANNELS for older files.
    analytical_mask = _get_analytical_mask_for_channels(anndata_object, channel_names)
    if not analytical_mask.all():
        excluded = [ch for ch, keep in zip(channel_names, analytical_mask) if not keep]
        full_matrix = full_matrix[:, analytical_mask]
        channel_names = [ch for ch, keep in zip(channel_names, analytical_mask) if keep]
        print(f"[compute_umap] Non-analytical channels excluded from UMAP: {excluded}")

    return full_matrix, channel_names


def _stratified_sample(
    obs_dataframe: pd.DataFrame,
    max_total: int,
    random_state: int,
) -> np.ndarray:
    """
    Return row indices for a stratified random sample balanced across subjects.

    If subject_ID is present: allocates events equally across subjects
    (floor division), then fills remaining slots with random events.
    If subject_ID is absent: simple random sample.

    Arguments:
        obs_dataframe: .obs DataFrame from the AnnData.
        max_total: Maximum number of indices to return.
        random_state: Random seed for reproducibility.

    Returns:
        1-D integer array of row indices (into obs_dataframe).
    """
    rng = np.random.default_rng(random_state)
    n_total = len(obs_dataframe)

    if n_total <= max_total:
        return np.arange(n_total)

    if _STRATIFY_COLUMN not in obs_dataframe.columns:
        # Simple random sample if no subject_ID.
        return rng.choice(n_total, size=max_total, replace=False)

    unique_subjects = obs_dataframe[_STRATIFY_COLUMN].dropna().unique()
    n_subjects = len(unique_subjects)
    if n_subjects == 0:
        return rng.choice(n_total, size=max_total, replace=False)

    per_subject_quota = max_total // n_subjects
    selected_indices = []

    for subject in unique_subjects:
        subject_indices = np.where(obs_dataframe[_STRATIFY_COLUMN].values == subject)[0]
        quota = min(per_subject_quota, len(subject_indices))
        chosen = rng.choice(subject_indices, size=quota, replace=False)
        selected_indices.append(chosen)

    combined = np.concatenate(selected_indices)

    # Fill remaining slots from the whole pool if quota didn't reach max_total.
    remaining = max_total - len(combined)
    if remaining > 0:
        all_not_chosen = np.setdiff1d(np.arange(n_total), combined)
        if len(all_not_chosen) > 0:
            extra = rng.choice(
                all_not_chosen,
                size=min(remaining, len(all_not_chosen)),
                replace=False,
            )
            combined = np.concatenate([combined, extra])

    return combined


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


def _build_color_map_umap(
    anndata_object: anndata.AnnData,
    group_by: Union[str, List[str]],
    unique_groups: List,
) -> dict:
    """
    Build a {group_value_string: hex_color} mapping for UMAP scatter.

    Color priority:
      1. PREFERRED_CONDITION_COLORS from controlled_vocabulary.py.
      2. Palette stored in .uns['color_palette'] by assign_color_palette().
      3. Auto-generated Set2 colors for any values not covered by 1 or 2.

    For combined group_by lists, combined labels are not in any stored palette
    so tab20 colors are generated automatically.

    Arguments:
        anndata_object: AnnData with optional .uns['color_palette'].
        group_by: Name of the grouping .obs column, or a list of column names.
        unique_groups: Ordered list of unique group values to map.

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

    color_map: dict = {}
    for g in unique_groups:
        g_str = str(g)
        if g_str in PREFERRED_CONDITION_COLORS:
            color_map[g_str] = PREFERRED_CONDITION_COLORS[g_str]
        elif g_str in stored_palette:
            color_map[g_str] = stored_palette[g_str]
        else:
            color_map[g_str] = auto_map.get(g_str, "#999999")

    return color_map
