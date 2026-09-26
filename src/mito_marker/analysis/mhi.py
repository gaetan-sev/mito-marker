"""
mhi.py

Mitochondrial Heterogeneity Index (MHI) — three complementary per-individual
heterogeneity metrics quantifying the morphological diversity of mitochondria
within each subject (or any user-defined group).

Three metrics:
  MHI-D (Dispersion)
      Median Euclidean distance from each mitochondrion to the group centroid
      in standardized feature space.  Low = tight cluster, high = broad scatter.

  MHI-S (Structure / Multimodality)
      Detects whether the mitochondria of an individual form distinct sub-
      populations (bimodal or multimodal distribution) vs a single continuous
      cloud.  Two complementary tests:
        - Hartigan dip test (1D projection onto local PC1)
        - GMM-BIC comparison (Gaussian Mixture Model with k=1..3)

  MHI-E (Entropy)
      Information-theoretic measure of spatial distribution diversity.
        - KNN entropy estimator (Kozachenko-Leonenko)
        - Binning entropy (PCA-discretised Shannon entropy)

Validation tools:
  create_chimera()              build a synthetic individual mixing two subjects
  validate_mhi_with_chimeras()  verify monotone MHI increase with mixing ratio

Sensitivity analysis:
  analyze_mhi_sensitivity()     find minimum n for stable MHI-D estimates

Diagnostic plots:
  plot_mhi_distances()          per-individual distance-to-centroid distributions
  plot_mhi_pca()                PC1 vs PC2 scatter coloured by group or GMM cluster

Typical usage in a Colab notebook:
    from mito_marker import compute_all_mhi, plot_mhi_distances

    mhi_dataframe = compute_all_mhi(tem_anndata, group_by="unique_subject_ID")
    plot_mhi_distances(tem_anndata)
"""

import warnings
from typing import Dict, Generator, List, Optional, Tuple, Union

import anndata
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.spatial import cKDTree
from scipy.special import digamma
from scipy.stats import kruskal, mannwhitneyu
from sklearn.decomposition import PCA as SklearnPCA
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from mito_marker.analysis.colors import sort_values_for_legend
from mito_marker.analysis.pca_plot import (
    _get_data_and_channels,
    compute_pca,
)
from mito_marker.analysis.selection import _ANALYSIS_CONFIG_KEY, _get_subject_column

# Optional diptest dependency.  If absent, MHI_S_dip columns are returned as NaN.
try:
    import diptest as _diptest_module

    _DIPTEST_AVAILABLE = True
except ImportError:
    _DIPTEST_AVAILABLE = False

# Optional scikit-posthocs dependency.  If absent, Dunn test falls back to
# pairwise Mann-Whitney U with manual Bonferroni correction.
try:
    from scikit_posthocs import posthoc_dunn as _posthoc_dunn

    _SCIKIT_POSTHOCS_AVAILABLE = True
except ImportError:
    _SCIKIT_POSTHOCS_AVAILABLE = False

# ──────────────────────────────────────────────────────────────────────────────
# Module-level constants
# ──────────────────────────────────────────────────────────────────────────────

# .uns key where MHI results are persisted.
_MHI_RESULTS_KEY = "mhi_results"

# .obsm key expected for global PCA coordinates (from compute_pca).
_OBSM_PCA_KEY = "X_pca"

# .uns key for PCA explained variance ratio (set by compute_pca).
_PCA_VAR_RATIO_KEY = "pca_explained_variance_ratio"

# Groups with fewer mitochondria than this are skipped (can't compute statistics).
_MIN_MITOS_PER_GROUP = 5

# Maximum local PCA dimensions for GMM in MHI-S.
_MAX_LOCAL_PCA_DIMS_GMM = 10

# Maximum global PCA dimensions used for binning entropy in MHI-E (curse of
# dimensionality: too many bins × dims → empty cells → entropy collapses).
_MAX_BINNING_DIMS = 5

# Variance threshold for auto PCA dimension selection.
_PCA_VARIANCE_TARGET = 0.95


# ──────────────────────────────────────────────────────────────────────────────
# Private helpers (not exported)
# ──────────────────────────────────────────────────────────────────────────────


def _resolve_group_by(
    anndata_object: anndata.AnnData,
    group_by: Optional[str],
) -> str:
    """
    Resolve the grouping column name.

    If group_by is None, auto-detects the subject column via _get_subject_column()
    (prefers 'unique_subject_ID', falls back to 'subject_ID').

    Arguments:
        anndata_object: AnnData whose .obs is inspected.
        group_by: Explicit column name, or None for auto-detection.

    Returns:
        Column name string.

    Raises:
        ValueError: If the resolved column is not present in .obs.
    """
    if group_by is not None:
        if group_by not in anndata_object.obs.columns:
            raise ValueError(
                f"Column '{group_by}' not found in .obs. "
                f"Available columns: {sorted(anndata_object.obs.columns.tolist())}"
            )
        return group_by

    subject_column = _get_subject_column(anndata_object.obs)
    if subject_column is None:
        raise ValueError(
            "No subject column found in .obs. "
            "Expected 'unique_subject_ID' (TEM pipeline) or 'subject_ID' (SFC pipeline). "
            "Pass group_by='your_column' explicitly."
        )
    return subject_column


def _get_mhi_feature_matrix(
    anndata_object: anndata.AnnData,
    n_pca_components: Union[str, int, None],
) -> Tuple[np.ndarray, List[str]]:
    """
    Build the feature matrix used for MHI-D and MHI-E distance/entropy computation.

    n_pca_components controls the feature space:
      'auto'  — global PCA, auto-select k components explaining >= 95% variance.
      None    — raw feature space (respects active layer and feature selection).
      int > 0 — global PCA with exactly that many components.

    The global PCA is retrieved from .obsm['X_pca'] (computed by compute_pca()).
    If not yet computed, compute_pca() is called automatically (result is cached).

    Arguments:
        anndata_object: AnnData with optional analysis_config in .uns.
        n_pca_components: Feature space selector — 'auto', None, or positive int.

    Returns:
        Tuple of (feature_matrix, feature_labels) where feature_matrix has
        shape (n_obs, n_features) as float32.
    """
    # ── Raw feature space ──────────────────────────────────────────────────
    if n_pca_components is None:
        raw_matrix, channel_names = _get_data_and_channels(anndata_object)
        return raw_matrix.astype(np.float32), list(channel_names)

    # ── PCA space: ensure global PCA is computed ───────────────────────────
    if _OBSM_PCA_KEY not in anndata_object.obsm:
        print(
            "[MHI] Global PCA not found — calling compute_pca() "
            "(result will be cached for subsequent calls)."
        )
        compute_pca(anndata_object)

    pca_coords = anndata_object.obsm[_OBSM_PCA_KEY]  # shape (n_obs, n_stored)
    explained_variance_ratio = anndata_object.uns.get(_PCA_VAR_RATIO_KEY, None)

    # ── Auto: select minimum k for >= 95% explained variance ──────────────
    if n_pca_components == "auto":
        if explained_variance_ratio is None:
            warnings.warn(
                "[MHI] PCA variance ratio not found in .uns. "
                f"Using all {pca_coords.shape[1]} stored components.",
                stacklevel=4,
            )
            k = pca_coords.shape[1]
        else:
            cumulative_variance = np.cumsum(explained_variance_ratio)
            threshold_indices = np.where(cumulative_variance >= _PCA_VARIANCE_TARGET)[0]
            if len(threshold_indices) == 0:
                total_variance = cumulative_variance[-1]
                k = len(explained_variance_ratio)
                warnings.warn(
                    f"[MHI] Global PCA explains only {total_variance:.1%} of variance "
                    f"({k} components). Using all available components.",
                    stacklevel=4,
                )
            else:
                # +1 because searchsorted returns a 0-based index
                k = int(threshold_indices[0]) + 1

        k = min(k, pca_coords.shape[1])

    # ── Explicit integer: use exactly n components ─────────────────────────
    else:
        k = min(int(n_pca_components), pca_coords.shape[1])
        if k < int(n_pca_components):
            warnings.warn(
                f"[MHI] Requested {n_pca_components} PCA components but only "
                f"{pca_coords.shape[1]} are stored. Using {k}.",
                stacklevel=4,
            )

    feature_labels = [f"PC{i + 1}" for i in range(k)]
    return pca_coords[:, :k].astype(np.float32), feature_labels


def _iter_groups(
    feature_matrix: np.ndarray,
    obs_series: pd.Series,
) -> Generator[Tuple[str, np.ndarray, np.ndarray], None, None]:
    """
    Yield per-group slices of the feature matrix.

    Splits feature_matrix into one block per unique value of obs_series.
    obs_series must be aligned with the rows of feature_matrix (same index order
    as the AnnData .obs).

    For each unique group value (iterated in sorted order for determinism):
      1. Locate all row positions where obs_series == group_value.
      2. Extract the corresponding rows from feature_matrix.
      3. Yield (group_value, row_indices, submatrix).

    Groups with fewer than _MIN_MITOS_PER_GROUP rows are skipped — you cannot
    compute a meaningful centroid or PCA from 1-4 points.

    Example:
        obs_series = ['KFish_1', 'KFish_1', 'KFish_2', 'KFish_2', 'KFish_2']
        → yields ('KFish_1', array([0, 1]), submatrix_2rows),
                  ('KFish_2', array([2, 3, 4]), submatrix_3rows)

    Arguments:
        feature_matrix: shape (n_obs, n_features).
        obs_series: pd.Series of group labels, aligned with feature_matrix rows.

    Yields:
        (group_value, row_indices, submatrix) for each qualifying group.
    """
    obs_array = obs_series.values  # plain numpy array for fast boolean masking
    sorted_group_values = sort_values_for_legend(obs_series.unique())

    for group_value in sorted_group_values:
        row_indices = np.where(obs_array == group_value)[0]
        n_mitos = len(row_indices)

        if n_mitos < _MIN_MITOS_PER_GROUP:
            warnings.warn(
                f"[MHI] Group '{group_value}' has only {n_mitos} mitochondria "
                f"(minimum required: {_MIN_MITOS_PER_GROUP}). Skipping this group.",
                stacklevel=3,
            )
            continue

        submatrix = feature_matrix[row_indices, :]
        yield group_value, row_indices, submatrix


def _standardize_submatrix(
    full_matrix: np.ndarray,
    row_indices: np.ndarray,
    standardize_on: str,
) -> np.ndarray:
    """
    Standardize the per-group submatrix to zero mean, unit variance per feature.

    Why this is always applied:
        MHI-D measures Euclidean distances.  If raw features are used (active
        layer = None), Mito_Area (µm²) would dominate over Mito_Circularity
        ([0, 1]) simply because of scale.  Standardization puts all features on
        equal footing.

        If the user already applied z-score normalization via
        transform_and_normalize(), the data arriving here is already globally
        standardized.  Re-fitting a StandardScaler on already-standardized data
        returns ~the same values (idempotent), so this call is harmless in that
        case.

    standardize_on='all':
        Fit the StandardScaler on the FULL dataset, then transform only the
        per-group slice.  All groups share the same reference mean/std →
        MHI-D values are directly comparable between individuals.
        Recommended for comparing heterogeneity between groups.

    standardize_on='group':
        Fit and transform on the per-group submatrix only.  Each individual's
        heterogeneity is measured relative to its own feature variance.
        Values are NOT directly comparable across individuals, but this mode
        is useful to explore the within-individual structure independent of
        between-group feature shifts.

    Arguments:
        full_matrix: Full feature matrix of shape (n_obs, n_features).
        row_indices: Row indices of the group within full_matrix.
        standardize_on: 'all' or 'group'.

    Returns:
        Standardized submatrix as float32, shape (len(row_indices), n_features).
    """
    submatrix = full_matrix[row_indices, :]

    scaler = StandardScaler()
    if standardize_on == "all":
        # Fit on the whole dataset so all groups share the same reference scale.
        scaler.fit(full_matrix)
    else:
        # Fit on this group's data only.
        scaler.fit(submatrix)

    return scaler.transform(submatrix).astype(np.float32)


def _fit_local_pca(
    submatrix: np.ndarray,
    n_components_target: int,
    variance_target: float = _PCA_VARIANCE_TARGET,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Fit a PCA on a single individual's submatrix (local PCA, not stored in AnnData).

    This is used by MHI-S to find the individual's principal axis of variation.
    The submatrix is standardized per-individual before fitting so that all
    features contribute equally regardless of their absolute scale.

    Arguments:
        submatrix: Feature matrix for one individual, shape (n_mitos, n_features).
        n_components_target: Maximum number of components to fit.
        variance_target: Cumulative variance threshold for auto-mode.

    Returns:
        (pca_coords, explained_variance_ratio) or (None, None) if too few points.
    """
    n_mitos, n_features = submatrix.shape
    # PCA requires at least 2 samples and 1 feature.
    max_possible_components = min(n_mitos - 1, n_features)
    if max_possible_components < 1:
        return None, None

    n_components = min(n_components_target, max_possible_components)

    # Standardize per-individual before local PCA so features are comparable.
    scaler = StandardScaler()
    standardized_submatrix = scaler.fit_transform(submatrix)

    local_pca_model = SklearnPCA(n_components=n_components, random_state=42)
    pca_coords = local_pca_model.fit_transform(standardized_submatrix)

    return pca_coords.astype(np.float32), local_pca_model.explained_variance_ratio_


def _auto_local_pca_dims(
    explained_variance_ratio: np.ndarray,
    max_dims: int,
) -> int:
    """
    Select the minimum number of PCA components that explain >= 95% variance,
    capped at max_dims.

    Arguments:
        explained_variance_ratio: Per-component variance fractions.
        max_dims: Hard cap on the returned number of dimensions.

    Returns:
        Number of components to use.
    """
    cumulative = np.cumsum(explained_variance_ratio)
    threshold_indices = np.where(cumulative >= _PCA_VARIANCE_TARGET)[0]
    if len(threshold_indices) == 0:
        k = len(explained_variance_ratio)
    else:
        k = int(threshold_indices[0]) + 1
    return min(k, max_dims)


def _store_mhi_result(
    anndata_object: anndata.AnnData,
    group_column: str,
    metric_key: str,
    results_dataframe: pd.DataFrame,
) -> None:
    """
    Persist MHI results in .uns['mhi_results'][group_column][metric_key].

    When two-level computation is active (nest_aggregate_by != group_by), both the
    per-individual results (under individual_col) and the aggregated group-level
    results (under group_col) are stored separately via two calls to this function.

    Arguments:
        anndata_object: AnnData to update in place.
        group_column: The .obs column used for grouping (dict key level 1).
        metric_key: 'D', 'S', 'E', or 'all' (dict key level 2).
        results_dataframe: DataFrame to store.
    """
    if _MHI_RESULTS_KEY not in anndata_object.uns:
        anndata_object.uns[_MHI_RESULTS_KEY] = {}
    if group_column not in anndata_object.uns[_MHI_RESULTS_KEY]:
        anndata_object.uns[_MHI_RESULTS_KEY][group_column] = {}
    anndata_object.uns[_MHI_RESULTS_KEY][group_column][metric_key] = results_dataframe


def _resolve_nest_aggregate_by(
    anndata_object: anndata.AnnData,
    nest_aggregate_by: str,
) -> str:
    """
    Validate that nest_aggregate_by is a column present in .obs.

    This function is a pure validator: it is only called when nest_aggregate_by is
    explicitly provided by the caller (not None).  Auto-detection is intentionally
    NOT performed here — when nest_aggregate_by is None, the caller defaults to using
    group_col as individual_col, preserving single-level behaviour.

    Arguments:
        anndata_object: AnnData whose .obs is inspected.
        nest_aggregate_by: Explicit column name that identifies individuals.

    Returns:
        The same column name string (validated).

    Raises:
        ValueError: If nest_aggregate_by is not present in .obs.
    """
    if nest_aggregate_by not in anndata_object.obs.columns:
        raise ValueError(
            f"Column '{nest_aggregate_by}' not found in .obs (nest_aggregate_by). "
            f"Available columns: {sorted(anndata_object.obs.columns.tolist())}"
        )
    return nest_aggregate_by


def _aggregate_individual_to_group(
    individual_df: pd.DataFrame,
    obs: pd.DataFrame,
    individual_col: str,
    group_col: str,
    agg_cols: List[Tuple[str, str]],
) -> pd.DataFrame:
    """
    Aggregate per-individual MHI results to the group level.

    For each group value, summarises the distribution of per-individual MHI scores
    using median, std, q25, and q75.  This two-level aggregation separates
    intra-individual heterogeneity (the true MHI signal) from inter-individual
    variance (noise when comparing groups such as species or conditions).

    Arguments:
        individual_df: DataFrame indexed by individual values (e.g. subject IDs).
            Must contain a 'n_mitos' column and the source columns listed in agg_cols.
        obs: anndata_object.obs DataFrame (used to map individuals to groups).
        individual_col: Column in obs that identifies individuals (index of individual_df).
        group_col: Column in obs that identifies groups (species, condition, etc.).
        agg_cols: List of (source_col, output_base_name) tuples.
            source_col: column in individual_df to aggregate.
            output_base_name: prefix for output columns ({base}_median, {base}_std, etc.).

    Returns:
        pd.DataFrame indexed by group_col name, with columns:
            n_individuals   — number of individuals in this group.
            n_mitos_total   — total mitochondria across all individuals in group.
            {base}_median   — median of per-individual values.
            {base}_std      — std of per-individual values (NaN if n_individuals == 1).
            {base}_q25      — 25th percentile of per-individual values.
            {base}_q75      — 75th percentile of per-individual values.
    """
    # Build individual → group mapping from obs, one row per unique individual.
    mapping_df = obs[[individual_col, group_col]].drop_duplicates()
    # Each individual must map to exactly one group. If the same individual appears
    # in multiple groups (e.g. longitudinal subjects in both Young and Old), the caller
    # must use a finer nest_aggregate_by column (e.g. subject_condition_ID) that is unique
    # within each group_by value.
    duplicate_mask = mapping_df[individual_col].duplicated(keep=False)
    if duplicate_mask.any():
        ambiguous = mapping_df.loc[duplicate_mask, individual_col].unique().tolist()[:5]
        raise ValueError(
            f"nest_aggregate_by='{individual_col}' is not unique "
            f"within group_by='{group_col}': individuals {ambiguous} (and possibly others) "
            f"map to multiple group values. Use a finer nest_aggregate_by column that is unique "
            f"per group — e.g. combine subject ID and condition into a single column."
        ) from None
    mapping_series = mapping_df.set_index(individual_col)[group_col]

    # Attach group assignment to the per-individual results.
    individual_df = individual_df.copy()
    individual_df["_group"] = individual_df.index.map(mapping_series)

    # Every individual in individual_df must map to a known group.
    unmapped = individual_df["_group"].isna().sum()
    assert unmapped == 0, (
        f"{unmapped} individuals in individual_df could not be mapped to a group. "
        f"Check that '{individual_col}' and '{group_col}' share values in obs."
    )

    rows: List[Dict] = []
    group_values_list: List[str] = []

    for group_value in sort_values_for_legend(individual_df["_group"].unique()):
        sub = individual_df[individual_df["_group"] == group_value]
        rec: Dict = {
            "n_individuals": len(sub),
            "n_mitos_total": int(sub["n_mitos"].sum()),
        }
        for src_col, out_base in agg_cols:
            if src_col not in sub.columns:
                continue
            vals = sub[src_col].dropna()
            rec[f"{out_base}_median"] = float(vals.median()) if len(vals) > 0 else np.nan
            rec[f"{out_base}_std"] = (
                float(vals.std(ddof=1)) if len(vals) > 1 else np.nan
            )
            rec[f"{out_base}_q25"] = float(vals.quantile(0.25)) if len(vals) > 0 else np.nan
            rec[f"{out_base}_q75"] = float(vals.quantile(0.75)) if len(vals) > 0 else np.nan
        rows.append(rec)
        group_values_list.append(group_value)

    return pd.DataFrame(
        rows,
        index=pd.Index(group_values_list, name=group_col),
    )


# ──────────────────────────────────────────────────────────────────────────────
# Public metric functions
# ──────────────────────────────────────────────────────────────────────────────


def compute_mhi_d(
    anndata_object: anndata.AnnData,
    group_by: Optional[str] = None,
    nest_aggregate_by: Optional[str] = None,
    standardize_on: str = "all",
    n_pca_components: Union[str, int, None] = "auto",
) -> pd.DataFrame:
    """
    Compute MHI-D (Dispersion) — median distance to centroid per group.

    MHI-D measures how widely scattered the mitochondria of an individual are in
    morphological feature space.  It is computed in three steps:
      1. Build a standardized feature matrix (global or per-group z-score).
      2. For each individual, find the centroid (mean position of all its mitos).
      3. Compute the Euclidean distance from each mito to the centroid.
         MHI-D = median of these distances.

    The median is used instead of the mean for robustness against outlier
    mitochondria that may be artefacts or extreme morphologies.

    Two-level computation (nest_aggregate_by != group_by):
        When nest_aggregate_by is provided and differs from group_by, MHI-D is first
        computed per individual, then aggregated to the group level.  This correctly
        separates intra-individual variance (the MHI signal) from inter-individual
        variance (noise).  Example: nest_aggregate_by='unique_subject_ID',
        group_by='specie' computes per-fish MHI then averages per species.

    Interpretation:
        MHI_D_median close to 0  → all mitochondria morphologically similar (tight).
        MHI_D_median >> 0        → high morphological diversity within the individual.
        MHI_D_q75 - MHI_D_q25   → interquartile range; large IQR = heavy tails.
        Compare between groups (young vs old, control vs disease) to detect
        condition-driven shifts in within-individual heterogeneity.

    Arguments:
        anndata_object: AnnData object (TEM or SFC pipeline output).
        group_by: .obs column used to define the reporting level (individuals or
            coarser groups).  Default: auto-detect 'unique_subject_ID' (TEM) or
            'subject_ID' (SFC).
        nest_aggregate_by: .obs column that identifies individuals.  When provided and
            different from group_by, enables two-level computation: per-individual
            MHI is computed first, then aggregated to group_by level.
            Default: None — individual_col defaults to group_col (single-level,
            preserving backward-compatible behaviour).
        standardize_on: 'all' fits the StandardScaler on the full dataset
            (recommended — MHI-D values are directly comparable across individuals).
            'group' fits per-individual (relative dispersion, not inter-comparable).
        n_pca_components: Feature space to compute distances in.
            'auto' (default): global PCA, auto-select k components for 95% variance.
            None: raw feature space (all analytical channels after feature selection).
            int: global PCA with exactly that many components.

    Returns:
        Single-level (nest_aggregate_by is None or equal to group_by):
            pd.DataFrame indexed by group value (e.g. 'KFish_1'), with columns:
                n_mitos        — number of mitochondria for this individual.
                MHI_D_median   — median distance to centroid (main MHI-D score).
                MHI_D_q25      — 25th percentile distance (lower bound of spread).
                MHI_D_q75      — 75th percentile distance (upper bound of spread).
            Stored in .uns['mhi_results'][group_by]['D'].
        Two-level (nest_aggregate_by differs from group_by):
            pd.DataFrame indexed by group value (e.g. 'human'), with columns:
                n_individuals  — number of individuals in this group.
                n_mitos_total  — total mitochondria across all individuals.
                MHI_D_median   — median of per-individual MHI_D_median values.
                MHI_D_std      — std of per-individual MHI_D_median values.
                MHI_D_q25      — 25th percentile of per-individual MHI_D_median.
                MHI_D_q75      — 75th percentile of per-individual MHI_D_median.
            Per-individual results stored under .uns['mhi_results'][nest_aggregate_by]['D'].
            Group-level results stored under .uns['mhi_results'][group_by]['D'].
    """
    print("=" * 60)
    print("COMPUTING MHI-D (Dispersion)")
    print("=" * 60)

    group_column = _resolve_group_by(anndata_object, group_by)
    # When nest_aggregate_by is None, default to single-level (group_col == individual_col).
    if nest_aggregate_by is None:
        individual_col = group_column
    else:
        individual_col = _resolve_nest_aggregate_by(anndata_object, nest_aggregate_by)

    # ── TWO-LEVEL PATH: compute per-individual first, then aggregate ───────
    if individual_col != group_column:
        print(
            f"  Two-level mode: individual='{individual_col}', group='{group_column}'"
        )
        print()
        individual_df = compute_mhi_d(
            anndata_object,
            group_by=individual_col,
            nest_aggregate_by=individual_col,  # force single-level recursion
            standardize_on=standardize_on,
            n_pca_components=n_pca_components,
        )
        _store_mhi_result(anndata_object, individual_col, "D", individual_df)

        group_df = _aggregate_individual_to_group(
            individual_df,
            anndata_object.obs,
            individual_col,
            group_column,
            agg_cols=[("MHI_D_median", "MHI_D")],
        )
        _store_mhi_result(anndata_object, group_column, "D", group_df)

        print(f"MHI-D group-level results ({len(group_df)} groups):")
        print(group_df.round(3).to_string())
        print()
        return group_df

    # ── SINGLE-LEVEL PATH (default behaviour, unchanged) ──────────────────
    n_groups = anndata_object.obs[group_column].nunique()

    print(f"  Group column : '{group_column}'")
    print(f"  Standardize  : '{standardize_on}'")
    if n_pca_components == "auto":
        print(f"  Feature space: global PCA (auto — target {_PCA_VARIANCE_TARGET:.0%} variance)")
    elif n_pca_components is None:
        print("  Feature space: raw (all analytical channels)")
    else:
        print(f"  Feature space: global PCA ({n_pca_components} components)")
    print(f"  Groups found : {n_groups}")
    print()

    feature_matrix, feature_labels = _get_mhi_feature_matrix(anndata_object, n_pca_components)
    n_obs, n_features = feature_matrix.shape
    preview_labels = feature_labels[:5]
    suffix = "…" if len(feature_labels) > 5 else ""
    print(f"Feature matrix: {n_obs:,} mitochondria × {n_features} features")
    print(f"Features used : {preview_labels}{suffix}")
    print()

    obs_series = anndata_object.obs[group_column]

    group_values: List[str] = []
    records: List[Dict] = []

    for group_value, row_indices, _submatrix in _iter_groups(feature_matrix, obs_series):
        # Standardize the per-individual slice using the chosen reference.
        standardized_submatrix = _standardize_submatrix(
            feature_matrix, row_indices, standardize_on
        )

        # Centroid = mean position of all mitochondria in standardized space.
        centroid = standardized_submatrix.mean(axis=0)  # shape (n_features,)

        # Euclidean distance of each mitochondrion to the centroid.
        # np.linalg.norm over axis=1 computes row-wise L2 norms efficiently.
        distances = np.linalg.norm(standardized_submatrix - centroid, axis=1)

        group_values.append(group_value)
        records.append(
            {
                "n_mitos": len(row_indices),
                "MHI_D_median": float(np.median(distances)),
                "MHI_D_q25": float(np.percentile(distances, 25)),
                "MHI_D_q75": float(np.percentile(distances, 75)),
            }
        )

    if not records:
        print("WARNING: No groups had enough mitochondria to compute MHI-D.")
        return pd.DataFrame(
            columns=["n_mitos", "MHI_D_median", "MHI_D_q25", "MHI_D_q75"]
        )

    results_dataframe = pd.DataFrame(
        records,
        index=pd.Index(group_values, name=group_column),
    )

    # Sanity checks
    assert (results_dataframe["n_mitos"] > 0).all(), "n_mitos must be positive."
    assert (results_dataframe["MHI_D_q25"] <= results_dataframe["MHI_D_median"]).all()
    assert (results_dataframe["MHI_D_median"] <= results_dataframe["MHI_D_q75"]).all()

    # ── Print summary ──────────────────────────────────────────────────────
    print(f"MHI-D results ({len(results_dataframe)} groups):")
    print(
        results_dataframe[["n_mitos", "MHI_D_median", "MHI_D_q25", "MHI_D_q75"]]
        .round(3)
        .to_string()
    )
    print()
    print(
        f"  MHI_D_median range: "
        f"{results_dataframe['MHI_D_median'].min():.3f} – "
        f"{results_dataframe['MHI_D_median'].max():.3f}"
    )
    print(
        "  Interpretation: higher MHI_D_median = greater within-individual "
        "morphological heterogeneity."
    )
    print()

    _store_mhi_result(anndata_object, group_column, "D", results_dataframe)

    return results_dataframe


def compute_mhi_s(
    anndata_object: anndata.AnnData,
    group_by: Optional[str] = None,
    nest_aggregate_by: Optional[str] = None,
    method: str = "both",
    n_pca_components: Union[str, int, None] = "auto",
    max_gmm_k: int = 3,
) -> pd.DataFrame:
    """
    Compute MHI-S (Structure) — multimodality detection per group.

    MHI-S detects whether the mitochondria within an individual form distinct
    sub-populations (bimodal or multimodal distribution) rather than a single
    continuous cloud.  Two complementary approaches:

    Option A — Hartigan Dip Test (requires: pip install diptest):
        For each individual, a local PCA is fit on that individual's mitos only
        (the direction of maximum variance is specific to this individual).
        The mitos are projected onto the first 1–3 local principal components.
        The Hartigan dip test is applied to each 1-D projection and the maximum
        dip statistic is kept.
        MHI_S_dip = max dip statistic across PC1/PC2/PC3.
        MHI_S_dip_pval = corresponding p-value (small p = significant multimodality).

    Option B — GMM-BIC comparison:
        A local PCA reduces the individual's mitos to k dimensions explaining
        >= 95% of their variance (max 10 components).  Gaussian Mixture Models
        with 1, 2, and 3 components are fitted.  ΔBIC is computed as:
          ΔBIC = BIC(k=1) - min(BIC(k=2..max_gmm_k))
        Positive ΔBIC = evidence for multi-modal structure (the model with more
        components fits significantly better than the unimodal reference).
        MHI_S_gmm_delta_bic = ΔBIC score.
        MHI_S_gmm_k_opt = number of components chosen by BIC minimisation.

    Note: The local PCA used here is specific to each individual and is never
    stored in the AnnData (it is purely a transient computation).

    Arguments:
        anndata_object: AnnData object.
        group_by: .obs grouping column (default: auto-detect subject column).
        nest_aggregate_by: .obs column that identifies individuals.  When provided and
            different from group_by, enables two-level computation (per-individual
            first, then aggregated to group level).  Default: None (single-level).
        method: 'dip', 'gmm', or 'both'.
        n_pca_components: Local PCA dimension for GMM reduction.
            'auto': use components for 95% variance (max 10).
            None: use all analytical channels directly (no local PCA — GMM in
                  full feature space, may be slow for n_features > 15).
            int: exact number of local PCA components.
        max_gmm_k: Maximum number of GMM components to test (default 3).

    Returns:
        Single-level: pd.DataFrame indexed by group value, with columns:
            n_mitos, MHI_S_dip, MHI_S_dip_pval, MHI_S_gmm_delta_bic, MHI_S_gmm_k_opt.
        Two-level: pd.DataFrame indexed by group value, with columns:
            n_individuals, n_mitos_total, plus _median/_std/_q25/_q75 for each metric.
        Stored in .uns['mhi_results'][group_by]['S'].
    """
    print("=" * 60)
    print("COMPUTING MHI-S (Structure / Multimodality)")
    print("=" * 60)

    if method not in ("dip", "gmm", "both", "distances", "all"):
        raise ValueError(
            f"method must be 'dip', 'gmm', 'both', 'distances', or 'all'. Got: '{method}'"
        )

    group_column = _resolve_group_by(anndata_object, group_by)
    if nest_aggregate_by is None:
        individual_col = group_column
    else:
        individual_col = _resolve_nest_aggregate_by(anndata_object, nest_aggregate_by)

    # ── TWO-LEVEL PATH ────────────────────────────────────────────────────
    if individual_col != group_column:
        print(
            f"  Two-level mode: individual='{individual_col}', group='{group_column}'"
        )
        print()
        individual_df = compute_mhi_s(
            anndata_object,
            group_by=individual_col,
            nest_aggregate_by=individual_col,
            method=method,
            n_pca_components=n_pca_components,
            max_gmm_k=max_gmm_k,
        )
        _store_mhi_result(anndata_object, individual_col, "S", individual_df)

        agg_cols = [
            ("MHI_S_dip", "MHI_S_dip"),
            ("MHI_S_dip_pval", "MHI_S_dip_pval"),
            ("MHI_S_gmm_delta_bic", "MHI_S_gmm_delta_bic"),
            ("MHI_S_gmm_k_opt", "MHI_S_gmm_k_opt"),
            ("MHI_S_dip_dist", "MHI_S_dip_dist"),
            ("MHI_S_dip_dist_pval", "MHI_S_dip_dist_pval"),
        ]
        group_df = _aggregate_individual_to_group(
            individual_df,
            anndata_object.obs,
            individual_col,
            group_column,
            agg_cols=agg_cols,
        )
        _store_mhi_result(anndata_object, group_column, "S", group_df)

        print(f"MHI-S group-level results ({len(group_df)} groups):")
        print(group_df.round(4).to_string())
        print()
        return group_df

    # ── SINGLE-LEVEL PATH ─────────────────────────────────────────────────
    n_groups = anndata_object.obs[group_column].nunique()

    print(f"  Group column  : '{group_column}'")
    print(f"  Method        : '{method}'")
    print(f"  max_gmm_k     : {max_gmm_k}")
    print(f"  Groups found  : {n_groups}")

    compute_dip = method in ("dip", "both", "all")
    compute_gmm = method in ("gmm", "both", "all")
    compute_distances_dip = method in ("distances", "all")

    if (compute_dip or compute_distances_dip) and not _DIPTEST_AVAILABLE:
        warnings.warn(
            "[MHI-S] The 'diptest' package is not installed. "
            "MHI_S_dip, MHI_S_dip_pval, MHI_S_dip_dist, and MHI_S_dip_dist_pval "
            "will be NaN. Install with: pip install diptest",
            stacklevel=2,
        )
    print()

    # MHI-S always starts from raw (or active-layer) features, then fits a
    # local PCA per individual.  We never use the global PCA here so that the
    # local principal axes capture within-individual structure.
    raw_matrix, channel_names = _get_data_and_channels(anndata_object)
    raw_matrix = raw_matrix.astype(np.float32)
    n_obs, n_raw_features = raw_matrix.shape
    print(f"Raw feature matrix: {n_obs:,} mitochondria × {n_raw_features} features")

    # For the 'distances' method: also build the global feature matrix used by
    # MHI-D (global PCA or raw features, governed by n_pca_components).  The dip
    # test runs on the same per-mito distance-to-centroid distribution as MHI-D,
    # probing whether that distribution is unimodal or multimodal.
    if compute_distances_dip:
        distances_feature_matrix, _ = _get_mhi_feature_matrix(
            anndata_object, n_pca_components
        )
    else:
        distances_feature_matrix = None
    print()

    obs_series = anndata_object.obs[group_column]

    group_values: List[str] = []
    records: List[Dict] = []

    for group_value, row_indices, submatrix in _iter_groups(raw_matrix, obs_series):
        dip_stat = np.nan
        dip_pval = np.nan
        delta_bic = np.nan
        k_opt = np.nan
        dip_dist_stat = np.nan
        dip_dist_pval = np.nan

        # ── Dip test on distances to centroid (method='distances' or 'all') ─
        # Runs Hartigan dip test on the scalar distribution of per-mito Euclidean
        # distances to the group centroid — the same distances computed in MHI-D,
        # in the globally standardized feature space.  This probes whether the
        # dispersion distribution is unimodal (homogeneous morphology) or
        # multimodal (distinct morphological sub-populations).
        if compute_distances_dip and _DIPTEST_AVAILABLE:
            std_dist_submatrix = _standardize_submatrix(
                distances_feature_matrix, row_indices, "all"
            )
            centroid_for_dip = std_dist_submatrix.mean(axis=0)
            dists_for_dip = np.linalg.norm(std_dist_submatrix - centroid_for_dip, axis=1)
            _d_stat, _d_pval = _diptest_module.diptest(dists_for_dip)
            dip_dist_stat = float(_d_stat)
            dip_dist_pval = float(_d_pval)

        # ── Dip test on local PCA projections (method='dip' or 'both') ────
        if compute_dip and _DIPTEST_AVAILABLE:
            # Fit a local PCA on this individual's mitos (max 3 components for
            # PC1/PC2/PC3 projection).
            local_pca_coords, local_var_ratio = _fit_local_pca(submatrix, n_components_target=3)

            if local_pca_coords is not None:
                n_local_pcs = local_pca_coords.shape[1]
                best_dip = 0.0
                best_pval = 1.0
                for pc_index in range(n_local_pcs):
                    projection = local_pca_coords[:, pc_index]
                    stat, pval = _diptest_module.diptest(projection)
                    if stat > best_dip:
                        best_dip = stat
                        best_pval = pval
                dip_stat = float(best_dip)
                dip_pval = float(best_pval)

        # ── GMM-BIC ───────────────────────────────────────────────────────
        if compute_gmm:
            # Reduce to local PCA space before fitting GMM to avoid
            # over-parameterised covariance matrices in high dimensions.
            if n_pca_components is None:
                # Use raw features directly in GMM (user explicitly chose no PCA).
                gmm_input = submatrix
                # Per-individual standardization so features are on equal scale.
                scaler = StandardScaler()
                gmm_input = scaler.fit_transform(gmm_input).astype(np.float32)
            else:
                # Auto or explicit: reduce via local PCA.
                local_dim_target = (
                    _MAX_LOCAL_PCA_DIMS_GMM
                    if n_pca_components == "auto"
                    else int(n_pca_components)
                )
                local_dim_target = min(local_dim_target, _MAX_LOCAL_PCA_DIMS_GMM)
                local_pca_coords, local_var_ratio = _fit_local_pca(
                    submatrix, n_components_target=local_dim_target
                )

                if local_pca_coords is None:
                    _rec: Dict = {"n_mitos": len(row_indices)}
                    if method != "distances":
                        _rec["MHI_S_dip"] = dip_stat if compute_dip else np.nan
                        _rec["MHI_S_dip_pval"] = dip_pval if compute_dip else np.nan
                        _rec["MHI_S_gmm_delta_bic"] = np.nan
                        _rec["MHI_S_gmm_k_opt"] = np.nan
                    if compute_distances_dip:
                        _rec["MHI_S_dip_dist"] = dip_dist_stat
                        _rec["MHI_S_dip_dist_pval"] = dip_dist_pval
                    group_values.append(group_value)
                    records.append(_rec)
                    continue

                # Keep only dimensions needed for 95% variance (cap at _MAX_LOCAL_PCA_DIMS_GMM).
                n_gmm_dims = _auto_local_pca_dims(local_var_ratio, _MAX_LOCAL_PCA_DIMS_GMM)
                gmm_input = local_pca_coords[:, :n_gmm_dims]

            # Fit GMMs with k = 1 .. max_gmm_k and collect BIC scores.
            bic_scores: List[float] = []
            for k_gmm in range(1, max_gmm_k + 1):
                # Limit components to number of observations.
                if k_gmm >= len(row_indices):
                    bic_scores.append(np.inf)
                    continue
                gmm_model = GaussianMixture(
                    n_components=k_gmm,
                    covariance_type="full",
                    random_state=42,
                    n_init=3,
                )
                gmm_model.fit(gmm_input)
                bic_scores.append(gmm_model.bic(gmm_input))

            bic_array = np.array(bic_scores)
            # ΔBIC = BIC(k=1) - min(BIC for k >= 2).
            # Positive value = k > 1 model is better → evidence for sub-populations.
            if max_gmm_k >= 2:
                k_opt_index = int(np.argmin(bic_array))
                k_opt = k_opt_index + 1  # 1-based
                delta_bic = float(bic_array[0] - bic_array[1:].min())
            else:
                k_opt = 1
                delta_bic = 0.0

        main_record: Dict = {"n_mitos": len(row_indices)}
        # Legacy dip/gmm columns: always include for non-distances methods (NaN when not computed).
        if method != "distances":
            main_record["MHI_S_dip"] = dip_stat if compute_dip else np.nan
            main_record["MHI_S_dip_pval"] = dip_pval if compute_dip else np.nan
            main_record["MHI_S_gmm_delta_bic"] = delta_bic if compute_gmm else np.nan
            main_record["MHI_S_gmm_k_opt"] = k_opt if compute_gmm else np.nan
        if compute_distances_dip:
            main_record["MHI_S_dip_dist"] = dip_dist_stat
            main_record["MHI_S_dip_dist_pval"] = dip_dist_pval
        group_values.append(group_value)
        records.append(main_record)

    if not records:
        print("WARNING: No groups had enough mitochondria to compute MHI-S.")
        empty_cols = ["n_mitos"]
        if method != "distances":
            empty_cols += ["MHI_S_dip", "MHI_S_dip_pval",
                           "MHI_S_gmm_delta_bic", "MHI_S_gmm_k_opt"]
        if compute_distances_dip:
            empty_cols += ["MHI_S_dip_dist", "MHI_S_dip_dist_pval"]
        return pd.DataFrame(columns=empty_cols)

    results_dataframe = pd.DataFrame(
        records,
        index=pd.Index(group_values, name=group_column),
    )

    print(f"MHI-S results ({len(results_dataframe)} groups):")
    display_cols = [col for col in results_dataframe.columns if col in results_dataframe]
    print(results_dataframe[display_cols].round(4).to_string())
    print()
    if compute_dip and _DIPTEST_AVAILABLE:
        print(
            "  MHI_S_dip: dip statistic on local PCA projections — higher = more bimodal. "
            "MHI_S_dip_pval < 0.05 suggests significant multimodality."
        )
    if compute_gmm:
        print(
            "  MHI_S_gmm_delta_bic: positive values indicate sub-populations "
            "fit better than a single Gaussian."
        )
    if compute_distances_dip and _DIPTEST_AVAILABLE:
        print(
            "  MHI_S_dip_dist: dip statistic on the distance-to-centroid distribution — "
            "higher = more multimodal dispersion structure. "
            "MHI_S_dip_dist_pval < 0.05 suggests significant multimodality."
        )
    print()

    _store_mhi_result(anndata_object, group_column, "S", results_dataframe)

    return results_dataframe


def compute_mhi_e(
    anndata_object: anndata.AnnData,
    group_by: Optional[str] = None,
    nest_aggregate_by: Optional[str] = None,
    method: str = "both",
    n_pca_components: Union[str, int, None] = "auto",
    n_bins: int = 4,
    n_neighbors: Union[str, int] = 5,
) -> pd.DataFrame:
    """
    Compute MHI-E (Entropy) — information-theoretic heterogeneity per group.

    MHI-E captures both the spread and the fine-grained structure of the mito-
    chondrial distribution via entropy.  Two approaches are provided:

    Option A — KNN entropy (Kozachenko-Leonenko estimator):
        Uses the distances to the k-th nearest neighbour to estimate the
        differential entropy of the continuous distribution.  Works well in
        PCA-reduced space (5–10 dimensions).  The result is normalised by
        log(n) so that a uniform distribution over n points has entropy ≈ 1.

        Formula:
            H_KL = -ψ(k) + ψ(n) + (d/n) × Σ log(dist_k(i))
            MHI_E_knn = H_KL / log(n)
        where ψ = digamma function, k = n_neighbors, d = number of dimensions.

    Option B — Binning entropy (Shannon entropy on PCA-discretised histogram):
        Reduces mitos to at most _MAX_BINNING_DIMS PCA dimensions (default 5),
        discretises each dimension into n_bins equal-frequency quantile bins,
        and computes the Shannon entropy of the resulting multinomial histogram.
        Normalised by log(n_bins^d) so that a uniform distribution gives 1.
        Simpler and more interpretable than KNN, but limited to ≤5 dimensions
        to avoid the curse of dimensionality (most cells would be empty in 28D).

    Both methods return values in [0, 1] after normalisation.

    Interpretation:
        MHI_E close to 0 → mitos are concentrated in a small region of the
                            feature space (very homogeneous individual).
        MHI_E close to 1 → mitos spread uniformly over the feature space
                            (maximally heterogeneous individual).

    Arguments:
        anndata_object: AnnData object.
        group_by: .obs grouping column (default: auto-detect subject column).
        nest_aggregate_by: .obs column that identifies individuals.  When provided and
            different from group_by, enables two-level computation (per-individual
            first, then aggregated to group level).  Default: None (single-level).
        method: 'knn', 'binning', or 'both'.
        n_pca_components: PCA feature space for KNN entropy (same as compute_mhi_d).
            'auto' (default): global PCA, 95% variance.
            None: raw features.
            int: exact number of PCA components.
        n_bins: Number of equal-frequency bins per PCA dimension for the
            binning estimator (default 4 → 4×4×…×4 histogram).
        n_neighbors: Number of nearest neighbours k for the KNN estimator.
            Default 5.  Pass 'auto' to use k = round(sqrt(n_mitos)) per
            individual — adapts k to the group size, giving more stable estimates
            for both small and large groups.

    Returns:
        Single-level: pd.DataFrame indexed by group value, with columns:
            n_mitos, MHI_E_knn, MHI_E_binning.
        Two-level: pd.DataFrame indexed by group value, with columns:
            n_individuals, n_mitos_total, plus _median/_std/_q25/_q75 for each metric.
        Stored in .uns['mhi_results'][group_by]['E'].
    """
    print("=" * 60)
    print("COMPUTING MHI-E (Entropy)")
    print("=" * 60)

    if method not in ("knn", "binning", "both"):
        raise ValueError(f"method must be 'knn', 'binning', or 'both'. Got: '{method}'")

    group_column = _resolve_group_by(anndata_object, group_by)
    if nest_aggregate_by is None:
        individual_col = group_column
    else:
        individual_col = _resolve_nest_aggregate_by(anndata_object, nest_aggregate_by)

    # ── TWO-LEVEL PATH ────────────────────────────────────────────────────
    if individual_col != group_column:
        print(
            f"  Two-level mode: individual='{individual_col}', group='{group_column}'"
        )
        print()
        individual_df = compute_mhi_e(
            anndata_object,
            group_by=individual_col,
            nest_aggregate_by=individual_col,
            method=method,
            n_pca_components=n_pca_components,
            n_bins=n_bins,
            n_neighbors=n_neighbors,
        )
        _store_mhi_result(anndata_object, individual_col, "E", individual_df)

        agg_cols = [
            ("MHI_E_knn", "MHI_E_knn"),
            ("MHI_E_binning", "MHI_E_binning"),
        ]
        group_df = _aggregate_individual_to_group(
            individual_df,
            anndata_object.obs,
            individual_col,
            group_column,
            agg_cols=agg_cols,
        )
        _store_mhi_result(anndata_object, group_column, "E", group_df)

        print(f"MHI-E group-level results ({len(group_df)} groups):")
        print(group_df.round(4).to_string())
        print()
        return group_df

    # ── SINGLE-LEVEL PATH ─────────────────────────────────────────────────
    n_groups = anndata_object.obs[group_column].nunique()

    print(f"  Group column : '{group_column}'")
    print(f"  Method       : '{method}'")
    print(f"  n_bins       : {n_bins}  (for binning estimator)")
    n_neighbors_display = n_neighbors if n_neighbors != "auto" else "auto (= round(sqrt(n_mitos)) per group)"
    print(f"  n_neighbors  : {n_neighbors_display}  (for KNN estimator)")
    print(f"  Groups found : {n_groups}")
    print()

    compute_knn = method in ("knn", "both")
    compute_binning = method in ("binning", "both")

    feature_matrix, feature_labels = _get_mhi_feature_matrix(anndata_object, n_pca_components)
    n_obs, n_features = feature_matrix.shape
    preview_labels = feature_labels[:5]
    suffix = "…" if len(feature_labels) > 5 else ""
    print(f"Feature matrix : {n_obs:,} mitochondria × {n_features} features")
    print(f"Features used  : {preview_labels}{suffix}")
    print()

    obs_series = anndata_object.obs[group_column]

    group_values: List[str] = []
    records: List[Dict] = []

    for group_value, row_indices, submatrix in _iter_groups(feature_matrix, obs_series):
        n_mitos = len(row_indices)
        entropy_knn = np.nan
        entropy_binning = np.nan

        # ── KNN entropy (Kozachenko-Leonenko estimator) ───────────────────
        if compute_knn:
            # Determine k: 'auto' scales with sqrt(n_mitos) per group so that
            # both small and large groups get a reasonable number of neighbours.
            if n_neighbors == "auto":
                k = max(1, int(round(np.sqrt(n_mitos))))
            else:
                k = int(n_neighbors)

            # Need at least k + 2 points for a meaningful estimate.
            min_points_for_knn = k + 2
            if n_mitos < min_points_for_knn:
                warnings.warn(
                    f"[MHI-E KNN] Group '{group_value}' has only {n_mitos} mitos "
                    f"(need ≥ {min_points_for_knn} for n_neighbors={k}). "
                    "MHI_E_knn will be NaN.",
                    stacklevel=2,
                )
            else:
                d = submatrix.shape[1]  # number of dimensions

                # Build a kd-tree for efficient neighbour search.
                kd_tree = cKDTree(submatrix)
                # Query the (k + 1)-th nearest neighbour
                # (the first result is the point itself, distance = 0).
                distances, _ = kd_tree.query(submatrix, k=k + 1)
                kth_distances = distances[:, k]  # shape (n_mitos,)

                # Avoid log(0) for duplicate points.
                kth_distances = np.maximum(kth_distances, 1e-10)

                # Kozachenko-Leonenko differential entropy estimator.
                # H_KL = -ψ(k) + ψ(n) + (d/n) × Σ log(r_k(i))
                h_kl = (
                    -digamma(k)
                    + digamma(n_mitos)
                    + (d / n_mitos) * np.sum(np.log(kth_distances))
                )

                # Normalise by log(n) so a uniform distribution gives ≈ 1.
                if n_mitos > 1:
                    entropy_knn = float(h_kl / np.log(n_mitos))
                else:
                    entropy_knn = 0.0

        # ── Binning entropy (PCA-discretised Shannon entropy) ──────────────
        if compute_binning:
            # For binning entropy, reduce to at most _MAX_BINNING_DIMS global
            # PCA dimensions to avoid the curse of dimensionality.
            # If n_pca_components is None (raw features), fit a local PCA first.
            if n_pca_components is None:
                local_coords, local_var = _fit_local_pca(
                    submatrix, n_components_target=_MAX_BINNING_DIMS
                )
                if local_coords is None:
                    records.append(
                        {
                            "n_mitos": n_mitos,
                            "MHI_E_knn": entropy_knn,
                            "MHI_E_binning": np.nan,
                        }
                    )
                    group_values.append(group_value)
                    continue
                n_binning_dims = _auto_local_pca_dims(local_var, _MAX_BINNING_DIMS)
                binning_matrix = local_coords[:, :n_binning_dims]
            else:
                # Global PCA already in submatrix; cap at _MAX_BINNING_DIMS.
                n_binning_dims = min(submatrix.shape[1], _MAX_BINNING_DIMS)
                binning_matrix = submatrix[:, :n_binning_dims]

            d_binning = binning_matrix.shape[1]

            # Discretise each dimension into n_bins equal-frequency (quantile) bins.
            # pd.qcut assigns each mito a bin label (integer 0..n_bins-1).
            bin_indices_per_dim: List[np.ndarray] = []
            valid_binning = True
            for dim_index in range(d_binning):
                column_values = binning_matrix[:, dim_index]
                try:
                    bin_labels = pd.qcut(
                        column_values,
                        q=n_bins,
                        labels=False,
                        duplicates="drop",
                    )
                    bin_labels_array = np.asarray(bin_labels)
                    if bin_labels is None or np.isnan(bin_labels_array.astype(float)).any():
                        valid_binning = False
                        break
                    bin_indices_per_dim.append(bin_labels_array.astype(int))
                except ValueError:
                    valid_binning = False
                    break

            if not valid_binning or not bin_indices_per_dim:
                entropy_binning = np.nan
            else:
                # Encode each mito as a multi-dim bin index using a base-n_bins scheme.
                # e.g. in 2D with n_bins=4: bin (2, 3) → 2*4 + 3 = 11
                multipliers = n_bins ** np.arange(d_binning)
                flat_bin_indices = np.stack(bin_indices_per_dim, axis=1)
                combined_bin = (flat_bin_indices * multipliers).sum(axis=1)

                # Count how many mitos fall in each cell.
                _, bin_counts = np.unique(combined_bin, return_counts=True)
                probabilities = bin_counts / bin_counts.sum()

                # Shannon entropy H = -Σ p × log(p).
                with np.errstate(divide="ignore", invalid="ignore"):
                    shannon_h = -np.sum(probabilities * np.log(np.maximum(probabilities, 1e-300)))

                # Normalise by maximum possible entropy: log(n_bins^d).
                max_entropy = d_binning * np.log(n_bins)
                if max_entropy > 0:
                    entropy_binning = float(np.clip(shannon_h / max_entropy, 0.0, 1.0))
                else:
                    entropy_binning = 0.0

        group_values.append(group_value)
        records.append(
            {
                "n_mitos": n_mitos,
                "MHI_E_knn": entropy_knn if compute_knn else np.nan,
                "MHI_E_binning": entropy_binning if compute_binning else np.nan,
            }
        )

    if not records:
        print("WARNING: No groups had enough mitochondria to compute MHI-E.")
        return pd.DataFrame(columns=["n_mitos", "MHI_E_knn", "MHI_E_binning"])

    results_dataframe = pd.DataFrame(
        records,
        index=pd.Index(group_values, name=group_column),
    )

    print(f"MHI-E results ({len(results_dataframe)} groups):")
    print(results_dataframe.round(4).to_string())
    print()
    print(
        "  Interpretation: values in [0, 1]. "
        "0 = perfectly concentrated; 1 = maximally dispersed / uniform."
    )
    print()

    _store_mhi_result(anndata_object, group_column, "E", results_dataframe)

    return results_dataframe


def compute_all_mhi(
    anndata_object: anndata.AnnData,
    group_by: Optional[str] = None,
    nest_aggregate_by: Optional[str] = None,
    metrics: Optional[List[str]] = None,
    **kwargs,
) -> pd.DataFrame:
    """
    Compute all requested MHI metrics and return a single merged DataFrame.

    Calls compute_mhi_d(), compute_mhi_s(), and/or compute_mhi_e() depending on
    the metrics argument, then left-joins the results on the group index.

    Arguments:
        anndata_object: AnnData object.
        group_by: .obs grouping column (default: auto-detect subject column).
        nest_aggregate_by: .obs column that identifies individuals.  When provided and
            different from group_by, enables two-level computation for all metrics.
            Default: None (single-level, backward-compatible).
        metrics: List of metric codes to compute. Subset of ['D', 'S', 'E'].
            Default: ['D', 'S', 'E'] (all three).
        **kwargs: Keyword arguments forwarded to each compute_mhi_* call.
            Examples:
                standardize_on='group'    → forwarded to compute_mhi_d
                method='gmm'              → forwarded to compute_mhi_s and compute_mhi_e
                n_pca_components=10       → forwarded to all three
                max_gmm_k=4              → forwarded to compute_mhi_s
                n_bins=5                 → forwarded to compute_mhi_e
                n_neighbors=7            → forwarded to compute_mhi_e

    Returns:
        pd.DataFrame indexed by group value with all computed metric columns.
        Stored in .uns['mhi_results'][group_by]['all'].
    """
    print("=" * 60)
    print("COMPUTING ALL MHI METRICS")
    print("=" * 60)

    if metrics is None:
        metrics = ["D", "S", "E"]

    invalid_metrics = [m for m in metrics if m not in ("D", "S", "E")]
    if invalid_metrics:
        raise ValueError(
            f"Unknown metric codes: {invalid_metrics}. Valid codes: 'D', 'S', 'E'."
        )

    group_column = _resolve_group_by(anndata_object, group_by)
    print(f"Metrics to compute: {metrics}")
    print(f"Group column: '{group_column}'")
    if nest_aggregate_by is not None:
        print(f"Individual column: '{nest_aggregate_by}'")
    print()

    # Separate kwargs by target function to avoid unexpected-keyword errors.
    # nest_aggregate_by is passed explicitly (not via **kwargs) so it is never
    # accidentally captured or filtered out by the key allowlists below.
    mhi_d_kwargs = {
        key: value
        for key, value in kwargs.items()
        if key in ("standardize_on", "n_pca_components")
    }
    mhi_s_kwargs = {
        key: value
        for key, value in kwargs.items()
        if key in ("method", "n_pca_components", "max_gmm_k")
    }
    mhi_e_kwargs = {
        key: value
        for key, value in kwargs.items()
        if key in ("method", "n_pca_components", "n_bins", "n_neighbors")
    }

    merged_dataframe: Optional[pd.DataFrame] = None

    if "D" in metrics:
        df_d = compute_mhi_d(
            anndata_object,
            group_by=group_column,
            nest_aggregate_by=nest_aggregate_by,
            **mhi_d_kwargs,
        )
        merged_dataframe = df_d

    if "S" in metrics:
        df_s = compute_mhi_s(
            anndata_object,
            group_by=group_column,
            nest_aggregate_by=nest_aggregate_by,
            **mhi_s_kwargs,
        )
        if merged_dataframe is None:
            merged_dataframe = df_s
        else:
            # Drop columns already present in merged_dataframe to avoid duplicates
            # (handles both single-level 'n_mitos' and two-level 'n_individuals' /
            # 'n_mitos_total' column name variants).
            overlap_cols = [c for c in df_s.columns if c in merged_dataframe.columns]
            df_s_trimmed = df_s.drop(columns=overlap_cols, errors="ignore")
            merged_dataframe = merged_dataframe.join(df_s_trimmed, how="left")

    if "E" in metrics:
        df_e = compute_mhi_e(
            anndata_object,
            group_by=group_column,
            nest_aggregate_by=nest_aggregate_by,
            **mhi_e_kwargs,
        )
        if merged_dataframe is None:
            merged_dataframe = df_e
        else:
            overlap_cols = [c for c in df_e.columns if c in merged_dataframe.columns]
            df_e_trimmed = df_e.drop(columns=overlap_cols, errors="ignore")
            merged_dataframe = merged_dataframe.join(df_e_trimmed, how="left")

    if merged_dataframe is None:
        return pd.DataFrame()

    # Store the merged result.
    _store_mhi_result(anndata_object, group_column, "all", merged_dataframe)
    if _ANALYSIS_CONFIG_KEY not in anndata_object.uns:
        anndata_object.uns[_ANALYSIS_CONFIG_KEY] = {}
    anndata_object.uns[_ANALYSIS_CONFIG_KEY]["mhi"] = {
        "group_by": group_column,
        "nest_aggregate_by": nest_aggregate_by,
        "metrics_computed": metrics,
        "kwargs": kwargs,
    }

    print("=" * 60)
    print("MHI SUMMARY")
    print("=" * 60)
    print(merged_dataframe.round(4).to_string())
    print()

    return merged_dataframe


# ──────────────────────────────────────────────────────────────────────────────
# Chimera validation
# ──────────────────────────────────────────────────────────────────────────────


def create_chimera(
    anndata_object: anndata.AnnData,
    subject_id_1: str,
    subject_id_2: str,
    n_mitos: Optional[int] = None,
    ratio: float = 0.5,
    seed: int = 42,
) -> anndata.AnnData:
    """
    Build a synthetic 'chimeric' individual by randomly mixing mitochondria from
    two real subjects.

    The chimera contains:
        ratio × n_mitos  mitochondria sampled from subject_id_1
        (1-ratio) × n_mitos  mitochondria sampled from subject_id_2

    This is the key validation tool for MHI: a chimera at ratio=0.5 should have
    higher heterogeneity than either parent (it has mixed the morphological
    'dialects' of two individuals).  Computing MHI at ratios 0.1 … 0.9 and
    checking for a monotone increase up to 0.5 validates that the metric captures
    biological heterogeneity rather than noise.

    The chimera preserves all .obs, .var, and .layers from the original AnnData,
    so it can be passed directly to compute_mhi_d/s/e.

    Arguments:
        anndata_object: Original AnnData containing both subjects.
        subject_id_1: Value in the auto-detected subject column for individual 1.
        subject_id_2: Value in the auto-detected subject column for individual 2.
        n_mitos: Total number of mitochondria in the chimera.
            None (default): min(n_mitos_1, n_mitos_2) — equal budget.
        ratio: Fraction of mitochondria drawn from individual 1.  Must be in [0, 1].
            ratio=0.5 → equal mix.  ratio=1.0 → pure individual 1.
        seed: Random seed for reproducibility.

    Returns:
        New AnnData with n_mitos rows.  The subject column is set to
        f"chimera_{subject_id_1}x{subject_id_2}_r{ratio:.2f}" for all rows.
        The original anndata_object is NOT modified.

    Raises:
        ValueError: If a subject ID is not found, if ratio is out of [0, 1], or
                    if the requested n_mitos exceeds what is available.
    """
    if not 0.0 <= ratio <= 1.0:
        raise ValueError(f"ratio must be in [0, 1]. Got: {ratio}")

    subject_column = _get_subject_column(anndata_object.obs)
    if subject_column is None:
        raise ValueError(
            "No subject column found. "
            "Expected 'unique_subject_ID' or 'subject_ID' in .obs."
        )

    all_subject_ids = set(anndata_object.obs[subject_column].unique())
    if subject_id_1 not in all_subject_ids:
        raise ValueError(
            f"subject_id_1='{subject_id_1}' not found in .obs['{subject_column}']. "
            f"Available IDs: {sorted(all_subject_ids)}"
        )
    if subject_id_2 not in all_subject_ids:
        raise ValueError(
            f"subject_id_2='{subject_id_2}' not found in .obs['{subject_column}']. "
            f"Available IDs: {sorted(all_subject_ids)}"
        )

    # Subset AnnData for each individual.
    mask_1 = anndata_object.obs[subject_column] == subject_id_1
    mask_2 = anndata_object.obs[subject_column] == subject_id_2
    adata_1 = anndata_object[mask_1]
    adata_2 = anndata_object[mask_2]

    n_available_1 = adata_1.n_obs
    n_available_2 = adata_2.n_obs

    # Determine total budget.
    if n_mitos is None:
        n_total = min(n_available_1, n_available_2)
    else:
        n_total = int(n_mitos)

    n_from_1 = int(n_total * ratio)
    n_from_2 = n_total - n_from_1

    if n_from_1 > n_available_1:
        raise ValueError(
            f"Requested {n_from_1} mitos from '{subject_id_1}' "
            f"but only {n_available_1} are available."
        )
    if n_from_2 > n_available_2:
        raise ValueError(
            f"Requested {n_from_2} mitos from '{subject_id_2}' "
            f"but only {n_available_2} are available."
        )

    rng = np.random.default_rng(seed)

    # Sample row indices within each individual's AnnData.
    indices_from_1 = rng.choice(n_available_1, size=n_from_1, replace=False)
    indices_from_2 = rng.choice(n_available_2, size=n_from_2, replace=False)

    sample_1 = adata_1[indices_from_1]
    sample_2 = adata_2[indices_from_2]

    # Concatenate; merge=None drops conflicting .uns keys cleanly.
    chimera = anndata.concat(
        [sample_1, sample_2],
        axis=0,
        join="outer",
        merge="same",
    )

    # Copy PCA metadata from the source AnnData so that compute_mhi_* can
    # determine the 95%-variance threshold and use the same feature space as
    # the full-dataset analysis.  Without this, _get_mhi_feature_matrix() would
    # find no variance ratio in .uns and fall back to all raw features —
    # producing distances in a different space from the full-dataset table.
    for uns_key in (
        _PCA_VAR_RATIO_KEY,           # "pca_explained_variance_ratio"
        "pca_loadings",               # loading matrix (not required but consistent)
        _ANALYSIS_CONFIG_KEY,         # active layer, active selection
    ):
        if uns_key in anndata_object.uns and uns_key not in chimera.uns:
            chimera.uns[uns_key] = anndata_object.uns[uns_key]

    # Overwrite the subject column with the chimera label.
    chimera_label = f"chimera_{subject_id_1}x{subject_id_2}_r{ratio:.2f}"
    chimera.obs[subject_column] = chimera_label

    print(
        f"Chimera created: '{chimera_label}' — "
        f"{n_from_1} mitos from '{subject_id_1}' + "
        f"{n_from_2} mitos from '{subject_id_2}' "
        f"(total: {chimera.n_obs})."
    )

    return chimera


def validate_mhi_with_chimeras(
    anndata_object: anndata.AnnData,
    subject_id_1: str,
    subject_id_2: str,
    ratios: Optional[List[float]] = None,
    n_repetitions: int = 3,
    seed: int = 42,
    metrics: Optional[List[str]] = None,
    show_plot: bool = True,
) -> pd.DataFrame:
    """
    Validate MHI metrics by testing that heterogeneity increases with chimeric mixing.

    For a valid heterogeneity index:
        - A chimera at ratio=0.5 (equal mix of two different individuals) should
          have higher MHI than either pure individual.
        - MHI(ratio) should be roughly monotone increasing from ratio=0.0
          (pure individual 2) to ratio=0.5, then symmetric / decreasing back.

    This function:
      1. Computes MHI for the two real individuals (subject_id_1, subject_id_2).
      2. Creates chimeras at each ratio with n_repetitions random seeds.
      3. Computes MHI for each chimera.
      4. Plots MHI = f(ratio) curve with confidence bands (mean ± std over reps).
      5. Warns if the curve is not monotone up to ratio=0.5.

    Arguments:
        anndata_object: AnnData containing both individuals.
        subject_id_1: Subject ID for individual 1 (corresponds to ratio=1.0).
        subject_id_2: Subject ID for individual 2 (corresponds to ratio=0.0).
        ratios: List of mixing ratios to test.  Default: [0.1, 0.2, …, 0.9].
        n_repetitions: Number of random subsamples per ratio for estimating
            variability (default 3).
        seed: Base random seed; each repetition uses seed + repetition_index.
        metrics: List of metric codes to compute — subset of ['D', 'S', 'E'].
            Default: ['D', 'S', 'E'] (all three).
        show_plot: If True (default), display the monotonicity plot inline.
            Set to False when calling from validate_all_chimera_pairs() to
            suppress individual pair plots; the caller builds the grid instead.

    Returns:
        pd.DataFrame with columns:
            ratio, repetition, n_mitos,
            MHI_D_median, MHI_S_dip, MHI_S_gmm_delta_bic, MHI_E_knn, MHI_E_binning
        (columns not in metrics are absent).
        The real individuals are included as ratio=0.0 (subject_id_2) and
        ratio=1.0 (subject_id_1).
    """
    if ratios is None:
        ratios = [round(r * 0.1, 1) for r in range(1, 10)]  # [0.1, 0.2, …, 0.9]
    if metrics is None:
        metrics = ["D", "S", "E"]

    print("=" * 60)
    print("CHIMERA VALIDATION")
    print("=" * 60)
    print(f"  Individual 1 : '{subject_id_1}'  (ratio = 1.0)")
    print(f"  Individual 2 : '{subject_id_2}'  (ratio = 0.0)")
    print(f"  Ratios tested: {ratios}")
    print(f"  Repetitions  : {n_repetitions}")
    print(f"  Metrics      : {metrics}")
    print()

    subject_column = _get_subject_column(anndata_object.obs)
    if subject_column is None:
        raise ValueError("No subject column found.")

    all_rows: List[Dict] = []

    def _extract_mhi_row(mhi_df: pd.DataFrame, label: str) -> Dict:
        """Extract scalar MHI values for a single-individual DataFrame."""
        if mhi_df.empty or label not in mhi_df.index:
            return {}
        row = mhi_df.loc[label]
        return row.to_dict()

    # ── Real individuals (ratio = 0.0 and ratio = 1.0) ────────────────────
    for real_ratio, real_id in [(0.0, subject_id_2), (1.0, subject_id_1)]:
        # Subset to the single individual.
        mask = anndata_object.obs[subject_column] == real_id
        adata_single = anndata_object[mask].copy()
        # Copy PCA metadata so that the single-individual subset uses the same
        # feature space as the full-dataset analysis.
        for uns_key in (_PCA_VAR_RATIO_KEY, "pca_loadings", _ANALYSIS_CONFIG_KEY):
            if uns_key in anndata_object.uns and uns_key not in adata_single.uns:
                adata_single.uns[uns_key] = anndata_object.uns[uns_key]

        mhi_df = compute_all_mhi(adata_single, group_by=subject_column, metrics=metrics)
        mhi_values = _extract_mhi_row(mhi_df, real_id)
        row_dict = {"ratio": real_ratio, "repetition": 0}
        row_dict.update(mhi_values)
        all_rows.append(row_dict)

    # ── Chimeras ──────────────────────────────────────────────────────────
    for ratio in ratios:
        for rep_index in range(n_repetitions):
            chimera_adata = create_chimera(
                anndata_object,
                subject_id_1=subject_id_1,
                subject_id_2=subject_id_2,
                ratio=ratio,
                seed=seed + rep_index,
            )
            chimera_label = f"chimera_{subject_id_1}x{subject_id_2}_r{ratio:.2f}"

            # PCA metadata is already copied inside create_chimera().
            mhi_df = compute_all_mhi(
                chimera_adata, group_by=subject_column, metrics=metrics
            )
            mhi_values = _extract_mhi_row(mhi_df, chimera_label)
            row_dict = {"ratio": ratio, "repetition": rep_index + 1, "n_mitos": chimera_adata.n_obs}
            row_dict.update(mhi_values)
            all_rows.append(row_dict)

    results_dataframe = pd.DataFrame(all_rows)

    # ── Monotonicity check for MHI-D ──────────────────────────────────────
    if "D" in metrics and "MHI_D_median" in results_dataframe.columns:
        chimera_mask = results_dataframe["ratio"].between(0.1, 0.5)
        chimera_means = (
            results_dataframe[chimera_mask]
            .groupby("ratio")["MHI_D_median"]
            .mean()
            .sort_index()
        )
        if len(chimera_means) >= 2:
            diffs = np.diff(chimera_means.values)
            if not np.all(diffs >= -0.01):  # allow tiny numeric fluctuation
                warnings.warn(
                    "[Chimera Validation] MHI_D_median is not monotone increasing "
                    "from ratio=0.0 to ratio=0.5. This may indicate that MHI-D is "
                    "capturing noise rather than biological heterogeneity.",
                    stacklevel=2,
                )

    # ── Plot ──────────────────────────────────────────────────────────────
    metric_columns = {
        "D": ["MHI_D_median"],
        "S": ["MHI_S_dip", "MHI_S_gmm_delta_bic"],
        "E": ["MHI_E_knn", "MHI_E_binning"],
    }
    columns_to_plot = [
        col
        for metric_code in metrics
        for col in metric_columns.get(metric_code, [])
        if col in results_dataframe.columns
    ]

    if columns_to_plot:
        n_cols_to_plot = len(columns_to_plot)
        fig, axes = plt.subplots(
            1, n_cols_to_plot, figsize=(5 * n_cols_to_plot, 4), squeeze=False
        )

        chimera_df = results_dataframe[results_dataframe["repetition"] > 0]
        real_df = results_dataframe[results_dataframe["repetition"] == 0]

        for axis_index, column_name in enumerate(columns_to_plot):
            ax = axes[0, axis_index]

            if column_name not in chimera_df.columns:
                ax.set_visible(False)
                continue

            # Mean and std of the metric across repetitions per ratio.
            grouped = chimera_df.groupby("ratio")[column_name].agg(["mean", "std"])
            ax.plot(
                grouped.index,
                grouped["mean"],
                marker="o",
                color="#2171b5",
                label="Chimera (mean ± std)",
            )
            ax.fill_between(
                grouped.index,
                grouped["mean"] - grouped["std"].fillna(0),
                grouped["mean"] + grouped["std"].fillna(0),
                alpha=0.25,
                color="#2171b5",
            )

            # Real individuals as dashed horizontal reference lines.
            if column_name in real_df.columns:
                real_0 = real_df[real_df["ratio"] == 0.0][column_name].values
                real_1 = real_df[real_df["ratio"] == 1.0][column_name].values
                if len(real_0) > 0 and not np.isnan(real_0[0]):
                    ax.axhline(
                        real_0[0],
                        linestyle="--",
                        color="#d62728",
                        alpha=0.7,
                        label=f"Pure {subject_id_2}",
                    )
                if len(real_1) > 0 and not np.isnan(real_1[0]):
                    ax.axhline(
                        real_1[0],
                        linestyle="--",
                        color="#2ca02c",
                        alpha=0.7,
                        label=f"Pure {subject_id_1}",
                    )

            ax.set_xlabel("Mixing ratio (fraction from individual 1)", fontsize=10)
            ax.set_ylabel(column_name, fontsize=10)
            ax.set_title(f"Chimera validation — {column_name}", fontsize=11)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        if show_plot:
            plt.show()
        else:
            plt.close()

    print(f"Chimera validation complete. {len(results_dataframe)} rows in results DataFrame.")
    return results_dataframe


# ──────────────────────────────────────────────────────────────────────────────
# Sensitivity analysis
# ──────────────────────────────────────────────────────────────────────────────


def analyze_mhi_sensitivity(
    anndata_object: anndata.AnnData,
    group_by: Optional[str] = None,
    nest_aggregate_by: Optional[str] = None,
    n_subsamples: Optional[List[int]] = None,
    n_repetitions: int = 10,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Estimate the minimum number of mitochondria required for a stable MHI-D estimate.

    For each individual with enough mitochondria, this function repeatedly
    sub-samples n mitos (for n in n_subsamples) and computes MHI-D each time.
    The coefficient of variation (CV = std / mean) across repetitions is plotted
    vs n.  A stable estimate is reached when CV < 5%.

    This is important for study design: if you collect TEM images per individual,
    how many do you need for MHI-D to be reliable?

    Note: Only individuals with n_mitos > max(n_subsamples) are included.
    The sensitivity analysis always operates at the individual level — the
    subsample loop uses individual_col regardless of group_by.  This is because
    stability is a per-individual property, not a group property.

    Arguments:
        anndata_object: AnnData object.
        group_by: .obs grouping column (default: auto-detect subject column).
        nest_aggregate_by: .obs column that identifies individuals.  When provided,
            the subsample loop runs at this level.  When None, defaults to
            group_col (same behaviour as before).
        n_subsamples: List of subsample sizes to test.
            Default: [20, 30, 50, 80, 100, 150].
        n_repetitions: Number of random subsamples per size (default 10).
        seed: Base random seed for reproducibility.

    Returns:
        pd.DataFrame with columns:
            group_value, n_subsample, repetition, MHI_D_median.
        The 'group_value' column contains individual IDs (from individual_col).
        Also plots CV of MHI_D_median vs n_subsample for each eligible individual.
    """
    if n_subsamples is None:
        n_subsamples = [20, 30, 50, 80, 100, 150]

    n_subsamples_sorted = sorted(n_subsamples)
    min_required = max(n_subsamples_sorted) + 1

    print("=" * 60)
    print("MHI-D SENSITIVITY ANALYSIS")
    print("=" * 60)
    print(f"  Subsample sizes    : {n_subsamples_sorted}")
    print(f"  Repetitions each   : {n_repetitions}")
    print(f"  Min mitos required : > {max(n_subsamples_sorted)} per individual")
    print()

    group_column = _resolve_group_by(anndata_object, group_by)
    # Sensitivity always runs at individual level; default to group_col if not specified.
    if nest_aggregate_by is None:
        individual_col = group_column
    else:
        individual_col = _resolve_nest_aggregate_by(anndata_object, nest_aggregate_by)

    # Identify eligible individuals (enough mitos for all subsample sizes).
    group_counts = anndata_object.obs[individual_col].value_counts()
    eligible_groups = group_counts[group_counts > min_required].index.tolist()

    if not eligible_groups:
        print(
            f"WARNING: No individuals have more than {min_required} mitochondria. "
            "Try reducing n_subsamples."
        )
        return pd.DataFrame(
            columns=["group_value", "n_subsample", "repetition", "MHI_D_median"]
        )

    print(f"Eligible individuals ({len(eligible_groups)} of {len(group_counts)}): {eligible_groups}")
    print()

    # Get full feature matrix once; subsampling happens on row indices.
    feature_matrix, feature_labels = _get_mhi_feature_matrix(anndata_object, n_pca_components="auto")
    # Use individual_col to find each individual's row indices.
    individual_obs_series = anndata_object.obs[individual_col]

    all_rows: List[Dict] = []

    for group_value in eligible_groups:
        row_indices = np.where(individual_obs_series.values == group_value)[0]
        n_total = len(row_indices)

        for n_subsample in n_subsamples_sorted:
            if n_subsample >= n_total:
                continue

            for rep_index in range(n_repetitions):
                rng = np.random.default_rng(seed + rep_index)
                sampled_indices = rng.choice(row_indices, size=n_subsample, replace=False)

                submatrix = feature_matrix[sampled_indices, :]

                # Standardize against the full dataset for comparability.
                scaler = StandardScaler()
                scaler.fit(feature_matrix)
                standardized_submatrix = scaler.transform(submatrix).astype(np.float32)

                centroid = standardized_submatrix.mean(axis=0)
                distances = np.linalg.norm(standardized_submatrix - centroid, axis=1)

                all_rows.append(
                    {
                        "group_value": group_value,
                        "n_subsample": n_subsample,
                        "repetition": rep_index,
                        "MHI_D_median": float(np.median(distances)),
                    }
                )

    if not all_rows:
        return pd.DataFrame(
            columns=["group_value", "n_subsample", "repetition", "MHI_D_median"]
        )

    results_dataframe = pd.DataFrame(all_rows)

    # ── Plot CV vs n_subsample ─────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))

    colors = plt.cm.tab10(np.linspace(0, 1, len(eligible_groups)))

    for color_index, group_value in enumerate(eligible_groups):
        group_df = results_dataframe[results_dataframe["group_value"] == group_value]
        cv_by_n = group_df.groupby("n_subsample")["MHI_D_median"].agg(
            lambda values: (values.std() / values.mean() * 100)
            if values.mean() != 0
            else np.nan
        )
        ax.plot(
            cv_by_n.index,
            cv_by_n.values,
            marker="o",
            color=colors[color_index],
            label=str(group_value),
        )

    # Reference line at CV = 5%.
    ax.axhline(5.0, linestyle="--", color="red", alpha=0.7, label="CV = 5% threshold")
    ax.set_xlabel("Number of mitochondria sub-sampled", fontsize=11)
    ax.set_ylabel("CV of MHI_D_median across repetitions (%)", fontsize=11)
    ax.set_title("MHI-D Sensitivity: stability vs number of mitochondria", fontsize=12)
    n_legend_cols = max(1, len(eligible_groups) // 6 + 1)
    n_legend_rows = int(np.ceil((len(eligible_groups) + 1) / n_legend_cols))  # +1 for threshold line
    bottom_margin = 0.08 + 0.04 * n_legend_rows
    ax.legend(
        fontsize=8,
        title=individual_col,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
        ncols=n_legend_cols,
    )
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.subplots_adjust(bottom=bottom_margin)
    plt.show()

    print(f"Sensitivity analysis complete. {len(results_dataframe)} rows total.")
    print(
        "  Recommendation: choose n where CV drops below 5% for most individuals."
    )

    return results_dataframe


# ──────────────────────────────────────────────────────────────────────────────
# Diagnostic plots
# ──────────────────────────────────────────────────────────────────────────────


def plot_mhi_distances(
    anndata_object: anndata.AnnData,
    group_by: Optional[str] = None,
    nest_aggregate_by: Optional[str] = None,
    mhi_results: Optional[pd.DataFrame] = None,
    standardize_on: str = "all",
    n_pca_components: Union[str, int, None] = "auto",
) -> None:
    """
    Plot MHI-D distributions for each group.

    Single-level (nest_aggregate_by is None or equal to group_by):
        Each group gets one violin + box panel showing the distribution of
        per-mitochondrion distances to the group centroid.  Horizontal lines
        indicate MHI_D_q25, MHI_D_median, and MHI_D_q75.

    Two-level (nest_aggregate_by differs from group_by):
        Each group gets one panel showing the scatter of per-individual
        MHI_D_median values (one dot per individual within the group).
        A red horizontal line marks the group-level median.
        This plot correctly separates intra-individual heterogeneity from
        inter-individual variability.

    Arguments:
        anndata_object: AnnData object.
        group_by: .obs grouping column (default: auto-detect subject column).
        nest_aggregate_by: .obs column that identifies individuals.  When provided and
            different from group_by, switches to two-level plot mode.
        mhi_results: Pre-computed MHI-D DataFrame (from compute_mhi_d()).
            If None, compute_mhi_d() is called internally.
        standardize_on: Passed to compute_mhi_d() if called internally.
        n_pca_components: Passed to compute_mhi_d() if called internally.

    Returns:
        None.  Displays the figure inline.
    """
    group_column = _resolve_group_by(anndata_object, group_by)
    if nest_aggregate_by is None:
        individual_col = group_column
    else:
        individual_col = _resolve_nest_aggregate_by(anndata_object, nest_aggregate_by)

    # ── TWO-LEVEL PLOT ────────────────────────────────────────────────────
    if individual_col != group_column:
        # Ensure per-individual MHI-D is computed.
        if (
            _MHI_RESULTS_KEY in anndata_object.uns
            and individual_col in anndata_object.uns[_MHI_RESULTS_KEY]
            and "D" in anndata_object.uns[_MHI_RESULTS_KEY][individual_col]
        ):
            individual_mhi = anndata_object.uns[_MHI_RESULTS_KEY][individual_col]["D"]
        else:
            individual_mhi = compute_mhi_d(
                anndata_object,
                group_by=individual_col,
                nest_aggregate_by=individual_col,
                standardize_on=standardize_on,
                n_pca_components=n_pca_components,
            )

        # Ensure group-level MHI-D is computed (for the median reference line).
        if mhi_results is None:
            mhi_results = compute_mhi_d(
                anndata_object,
                group_by=group_column,
                nest_aggregate_by=individual_col,
                standardize_on=standardize_on,
                n_pca_components=n_pca_components,
            )

        print("Plotting MHI-D per-individual values by group (two-level mode)...")

        # Map each individual to its group.
        mapping_df = (
            anndata_object.obs[[individual_col, group_column]]
            .drop_duplicates()
            .set_index(individual_col)
        )
        individual_mhi_copy = individual_mhi.copy()
        individual_mhi_copy["_group"] = individual_mhi_copy.index.map(
            mapping_df[group_column]
        )

        sorted_group_values = sort_values_for_legend(individual_mhi_copy["_group"].dropna().unique())
        n_groups = len(sorted_group_values)
        if n_groups == 0:
            print("No groups to plot.")
            return

        fig, axes = plt.subplots(
            1, n_groups, figsize=(max(4, 3 * n_groups), 5), squeeze=False, sharey=True
        )

        for plot_index, group_value in enumerate(sorted_group_values):
            ax = axes[0, plot_index]
            sub = individual_mhi_copy[individual_mhi_copy["_group"] == group_value]
            individual_values = sub["MHI_D_median"].dropna().values
            n_individuals = len(individual_values)

            # Scatter of per-individual MHI_D_median values.
            x_positions = np.zeros(n_individuals)
            ax.scatter(
                x_positions,
                individual_values,
                color="#1f77b4",
                s=60,
                zorder=3,
                alpha=0.8,
            )

            # Group-level median reference line.
            if group_value in mhi_results.index:
                group_median = mhi_results.loc[group_value].get("MHI_D_median", np.nan)
                if not np.isnan(group_median):
                    ax.axhline(
                        group_median,
                        color="#d73027",
                        linestyle="--",
                        linewidth=1.5,
                        alpha=0.9,
                        label="group median",
                    )

            ax.set_title(f"{group_value}\n(n={n_individuals} individuals)", fontsize=9)
            ax.set_xticks([])
            ax.set_ylabel("MHI_D_median (per individual)" if plot_index == 0 else "")
            ax.legend(fontsize=7, loc="upper right")
            ax.grid(True, axis="y", alpha=0.3)

        title = (
            f"MHI-D Per-Individual Values by '{group_column}'\n"
            f"(individuals identified by '{individual_col}')"
        )
        fig.suptitle(title, fontsize=12, fontweight="bold")
        plt.tight_layout()
        plt.show()

        print(
            "  Each dot = one individual's MHI_D_median. "
            "Red dashed line = group-level median of individual values."
        )
        return

    # ── SINGLE-LEVEL PLOT (default, unchanged) ────────────────────────────
    if mhi_results is None:
        mhi_results = compute_mhi_d(
            anndata_object,
            group_by=group_column,
            standardize_on=standardize_on,
            n_pca_components=n_pca_components,
        )

    print("Plotting MHI-D distance distributions...")

    feature_matrix, _ = _get_mhi_feature_matrix(anndata_object, n_pca_components)
    obs_series = anndata_object.obs[group_column]

    # Collect per-mito distances for all eligible groups.
    group_distances: Dict[str, np.ndarray] = {}
    for group_value, row_indices, _submatrix in _iter_groups(feature_matrix, obs_series):
        std_submatrix = _standardize_submatrix(feature_matrix, row_indices, standardize_on)
        centroid = std_submatrix.mean(axis=0)
        distances = np.linalg.norm(std_submatrix - centroid, axis=1)
        group_distances[group_value] = distances

    n_groups = len(group_distances)
    if n_groups == 0:
        print("No groups to plot.")
        return

    fig, axes = plt.subplots(
        1, n_groups, figsize=(max(4, 3 * n_groups), 5), squeeze=False, sharey=True
    )

    sorted_groups = sort_values_for_legend(group_distances.keys())
    for plot_index, group_value in enumerate(sorted_groups):
        ax = axes[0, plot_index]
        distances = group_distances[group_value]

        # Violin plot for the full distribution.
        violin_parts = ax.violinplot(
            [distances],
            positions=[0],
            showmeans=False,
            showmedians=False,
            showextrema=False,
        )
        for body in violin_parts.get("bodies", []):
            body.set_facecolor("#aec7e8")
            body.set_alpha(0.7)

        # Overlay box plot.
        ax.boxplot(
            [distances],
            positions=[0],
            widths=0.15,
            patch_artist=True,
            boxprops={"facecolor": "#1f77b4", "alpha": 0.5},
            medianprops={"color": "white", "linewidth": 2},
            whiskerprops={"color": "#1f77b4"},
            capprops={"color": "#1f77b4"},
            flierprops={"marker": ".", "alpha": 0.3, "color": "grey"},
        )

        # Overlay the exact MHI-D quantile lines from mhi_results.
        if group_value in mhi_results.index:
            row = mhi_results.loc[group_value]
            for quantile_value, quantile_label, line_color in [
                (row.get("MHI_D_q25", np.nan), "q25", "#fdae61"),
                (row.get("MHI_D_median", np.nan), "median", "#d73027"),
                (row.get("MHI_D_q75", np.nan), "q75", "#fdae61"),
            ]:
                if not np.isnan(quantile_value):
                    ax.axhline(
                        quantile_value,
                        color=line_color,
                        linestyle="--",
                        linewidth=1.5,
                        alpha=0.9,
                        label=quantile_label,
                    )

        n_mitos_label = len(distances)
        ax.set_title(f"{group_value}\n(n={n_mitos_label})", fontsize=9)
        ax.set_xticks([])
        ax.set_ylabel("Distance to centroid" if plot_index == 0 else "")
        ax.legend(fontsize=7, loc="upper right")
        ax.grid(True, axis="y", alpha=0.3)

    # Build species label from .obs if the column exists.
    if "specie" in anndata_object.obs.columns:
        unique_species = sort_values_for_legend(anndata_object.obs["specie"].dropna().unique())
        species_label = ", ".join(str(s) for s in unique_species)
    else:
        species_label = None

    title = f"MHI-D Distance Distributions by '{group_column}'"
    if species_label:
        title += f"\n{species_label}"

    fig.suptitle(title, fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.show()

    print(
        "  How to read: the red dashed line is MHI_D_median. "
        "Taller violins = higher within-individual heterogeneity."
    )


def plot_mhi_pca(
    anndata_object: anndata.AnnData,
    group_by: Optional[str] = None,
    color_by_gmm: bool = False,
    max_points_per_group: int = 500,
) -> None:
    """
    Plot PC1 vs PC2 scatter coloured by group membership or GMM cluster assignment.

    Uses the global PCA stored in .obsm['X_pca'] (compute_pca() is called if not
    yet computed).  Sub-samples up to max_points_per_group points per group for
    visual clarity.

    Arguments:
        anndata_object: AnnData object.
        group_by: .obs column used to colour points (default: auto-detect subject).
        color_by_gmm: If True, re-fit a GMM per individual and colour each
            mitochondrion by its assigned Gaussian component (1, 2, or 3).
            If False (default), colour by group_by value.
        max_points_per_group: Maximum scatter points per group for visual clarity.

    Returns:
        None.  Displays the figure inline.
    """
    group_column = _resolve_group_by(anndata_object, group_by)

    # Ensure global PCA is available.
    if _OBSM_PCA_KEY not in anndata_object.obsm:
        print("[plot_mhi_pca] Computing global PCA first...")
        compute_pca(anndata_object)

    pca_coords = anndata_object.obsm[_OBSM_PCA_KEY]  # shape (n_obs, n_components)
    if pca_coords.shape[1] < 2:
        print("Not enough PCA components to plot PC1 vs PC2.")
        return

    obs_series = anndata_object.obs[group_column].values
    sorted_groups = sort_values_for_legend(set(obs_series))
    n_groups = len(sorted_groups)

    colormap = plt.cm.tab20(np.linspace(0, 1, max(n_groups, 1)))
    group_to_color = {group: colormap[idx] for idx, group in enumerate(sorted_groups)}

    fig, ax = plt.subplots(figsize=(8, 6))

    rng = np.random.default_rng(42)

    for group_value in sorted_groups:
        group_mask = obs_series == group_value
        group_indices = np.where(group_mask)[0]

        # Sub-sample for visual clarity.
        if len(group_indices) > max_points_per_group:
            sampled_indices = rng.choice(
                group_indices, size=max_points_per_group, replace=False
            )
        else:
            sampled_indices = group_indices

        x_coords = pca_coords[sampled_indices, 0]
        y_coords = pca_coords[sampled_indices, 1]

        if color_by_gmm:
            # Fit a GMM (k=2) on the individual's full data and colour by cluster.
            full_group_indices = group_indices
            if len(full_group_indices) < _MIN_MITOS_PER_GROUP:
                continue
            group_matrix = pca_coords[full_group_indices, :min(pca_coords.shape[1], 5)]
            gmm_model = GaussianMixture(n_components=2, random_state=42)
            gmm_model.fit(group_matrix)
            # Predict for the sampled points.
            sampled_group_matrix = pca_coords[
                sampled_indices, : min(pca_coords.shape[1], 5)
            ]
            gmm_labels = gmm_model.predict(sampled_group_matrix)
            scatter_colors = [
                "#1f77b4" if label == 0 else "#ff7f0e" for label in gmm_labels
            ]
            ax.scatter(
                x_coords,
                y_coords,
                c=scatter_colors,
                s=10,
                alpha=0.5,
                label=f"{group_value} (GMM)",
            )
        else:
            ax.scatter(
                x_coords,
                y_coords,
                c=[group_to_color[group_value]],
                s=10,
                alpha=0.5,
                label=str(group_value),
            )

    ax.set_xlabel("PC1", fontsize=11)
    ax.set_ylabel("PC2", fontsize=11)
    title_suffix = "(coloured by GMM cluster)" if color_by_gmm else f"(coloured by {group_column})"
    ax.set_title(f"MHI-PCA: PC1 vs PC2 {title_suffix}", fontsize=12)
    ax.legend(
        fontsize=8,
        markerscale=2,
        loc="best",
        title=group_column if not color_by_gmm else "GMM component",
    )
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    print(
        f"  Plot shows PC1 vs PC2 from global PCA. "
        f"Each point is one mitochondrion. "
        f"Up to {max_points_per_group} points per group are shown."
    )


# ──────────────────────────────────────────────────────────────────────────────
# MHI results → .obs persistence
# ──────────────────────────────────────────────────────────────────────────────


def save_mhi_to_obs(
    anndata_object: anndata.AnnData,
    nest_aggregate_by: Optional[str] = None,
) -> None:
    """
    Write per-individual MHI values from .uns into .obs columns.

    After calling compute_all_mhi() with nest_aggregate_by, the per-individual MHI
    results are stored in .uns['mhi_results'][individual_col].  This function
    maps those scalar values back into .obs so each mitochondrion row carries
    the MHI score of its parent individual.  This enables downstream use such
    as colouring UMAP plots by MHI-D or correlating MHI-E with clinical data.

    Each per-individual metric column (e.g. MHI_D_median, MHI_S_dip_dist,
    MHI_E_knn) is written as a float column in .obs, aligned by the individual
    column.  Rows whose individual ID is not found in the MHI results get NaN.

    Arguments:
        anndata_object: AnnData with .uns['mhi_results'] populated by at least
            one of compute_mhi_d/s/e or compute_all_mhi.
        nest_aggregate_by: .obs column identifying individuals.  If None,
            auto-detects 'unique_subject_ID' (TEM) or 'subject_ID' (SFC).

    Returns:
        None.  Modifies anndata_object.obs in place.

    Raises:
        ValueError: If no MHI results are found for the resolved column.
    """
    individual_col = _resolve_group_by(anndata_object, nest_aggregate_by)

    if _MHI_RESULTS_KEY not in anndata_object.uns:
        raise ValueError(
            f"No MHI results found in .uns['{_MHI_RESULTS_KEY}']. "
            "Call compute_all_mhi() first."
        )

    if individual_col not in anndata_object.uns[_MHI_RESULTS_KEY]:
        raise ValueError(
            f"No per-individual MHI results for column '{individual_col}' in "
            f".uns['{_MHI_RESULTS_KEY}']. "
            f"Available columns: {sorted(anndata_object.uns[_MHI_RESULTS_KEY].keys())}"
        )

    print("=" * 60)
    print("SAVING PER-INDIVIDUAL MHI VALUES TO .obs")
    print("=" * 60)
    print(f"  Individual column: '{individual_col}'")

    individual_results = anndata_object.uns[_MHI_RESULTS_KEY][individual_col]
    obs_individual_series = anndata_object.obs[individual_col].astype(str)

    # Determine the source DataFrame: use 'all' (merged) if available; otherwise
    # collect from D/S/E tables without duplicating shared columns.
    _COUNT_COLS = {"n_mitos", "n_individuals", "n_mitos_total"}

    if "all" in individual_results:
        tables_to_process: List[pd.DataFrame] = [individual_results["all"]]
    else:
        tables_to_process = [
            individual_results[key]
            for key in ("D", "S", "E")
            if key in individual_results
        ]

    already_saved: set = set()
    saved_count = 0

    for source_df in tables_to_process:
        metric_cols = [c for c in source_df.columns if c not in _COUNT_COLS]
        for metric_col in metric_cols:
            if metric_col in already_saved:
                continue
            metric_series = source_df[metric_col].astype(float)
            # pd.Series.map(series) uses the series index as a lookup key.
            # individual_results index = individual IDs; obs_individual_series values = IDs.
            mapped_values = obs_individual_series.map(metric_series)
            anndata_object.obs[metric_col] = mapped_values.values.astype(float)
            n_missing = mapped_values.isna().sum()
            if n_missing > 0:
                warnings.warn(
                    f"[save_mhi_to_obs] {n_missing} rows in .obs['{individual_col}'] "
                    f"did not match any individual for metric '{metric_col}'. "
                    "They will be NaN.",
                    stacklevel=2,
                )
            already_saved.add(metric_col)
            saved_count += 1

    print(f"  Saved {saved_count} metric column(s): {sorted(already_saved)}")
    print()


# ──────────────────────────────────────────────────────────────────────────────
# Statistical tests
# ──────────────────────────────────────────────────────────────────────────────


def run_mhi_statistics(
    anndata_object: anndata.AnnData,
    group_by: Optional[str] = None,
    nest_aggregate_by: Optional[str] = None,
    metric_cols: Optional[List[str]] = None,
    n_permutations: int = 10_000,
    seed: int = 42,
) -> Dict:
    """
    Run statistical tests comparing MHI values between groups.

    For each requested metric column, three tests are performed:
      1. Kruskal-Wallis H test across all groups (non-parametric ANOVA).
      2. Post-hoc pairwise comparisons (if KW is significant at p < 0.05):
           - Dunn test with Bonferroni correction (requires scikit-posthocs),
             or pairwise Mann-Whitney U + Bonferroni as fallback.
      3. Permutation test (n_permutations random label shuffles) as a robust
         confirmation of the KW result.

    The per-individual MHI values are read from
    .uns['mhi_results'][individual_col].  Call compute_all_mhi() with
    nest_aggregate_by before calling this function.

    Arguments:
        anndata_object: AnnData with per-individual MHI results in .uns.
        group_by: .obs column identifying groups (e.g. 'specie').
            Default: auto-detect subject column.
        nest_aggregate_by: .obs column identifying individuals
            (e.g. 'unique_subject_ID').  Default: None → group_by is used
            (single-level, no pairwise tests make sense with 1 value/group).
        metric_cols: Columns to test.  Default: all columns starting with 'MHI_'.
        n_permutations: Number of permutations for the permutation test.
        seed: Random seed.

    Returns:
        Dict keyed by metric column name, each value is a dict:
            'kw_stat'   — Kruskal-Wallis H statistic (float).
            'kw_pval'   — Kruskal-Wallis p-value (float).
            'perm_pval' — Permutation test p-value (float).
            'posthoc'   — pd.DataFrame of pairwise p-values (Bonferroni-
                          corrected), or None if KW was not significant.
        Results stored in .uns['mhi_results'][group_by]['statistics'].

    Raises:
        ValueError: If per-individual MHI results are not found.
    """
    group_column = _resolve_group_by(anndata_object, group_by)
    individual_col = (
        _resolve_nest_aggregate_by(anndata_object, nest_aggregate_by)
        if nest_aggregate_by is not None
        else group_column
    )

    if _MHI_RESULTS_KEY not in anndata_object.uns:
        raise ValueError(
            f"No MHI results found in .uns['{_MHI_RESULTS_KEY}']. "
            "Call compute_all_mhi() first."
        )
    if individual_col not in anndata_object.uns[_MHI_RESULTS_KEY]:
        raise ValueError(
            f"No per-individual MHI results for column '{individual_col}'. "
            f"Available: {sorted(anndata_object.uns[_MHI_RESULTS_KEY].keys())}"
        )

    individual_results = anndata_object.uns[_MHI_RESULTS_KEY][individual_col]

    # Merge all metric tables into a single DataFrame.
    if "all" in individual_results:
        individual_df = individual_results["all"].copy()
    else:
        individual_df = None
        for key in ("D", "S", "E"):
            if key not in individual_results:
                continue
            part = individual_results[key]
            if individual_df is None:
                individual_df = part.copy()
            else:
                overlap = [c for c in part.columns if c in individual_df.columns]
                individual_df = individual_df.join(
                    part.drop(columns=overlap, errors="ignore"), how="outer"
                )
    if individual_df is None or individual_df.empty:
        raise ValueError(
            "No per-individual MHI DataFrames found. "
            "Call compute_all_mhi() with the relevant metrics first."
        )

    # Map each individual to its group.
    individual_to_group = (
        anndata_object.obs[[individual_col, group_column]]
        .drop_duplicates()
        .set_index(individual_col)[group_column]
    )
    individual_df["_group"] = individual_df.index.map(individual_to_group)

    # Auto-detect metric columns: those starting with 'MHI_'.
    _COUNT_COLS = {"n_mitos", "n_individuals", "n_mitos_total", "_group"}
    if metric_cols is None:
        metric_cols = [
            c for c in individual_df.columns
            if c.startswith("MHI_") and c not in _COUNT_COLS
        ]
    metric_cols = [c for c in metric_cols if c in individual_df.columns]

    print("=" * 60)
    print("MHI STATISTICAL TESTS")
    print("=" * 60)
    print(f"  Individual column : '{individual_col}'")
    print(f"  Group column      : '{group_column}'")
    print(f"  Metric columns    : {metric_cols}")
    print(f"  Permutations      : {n_permutations:,}")
    print()

    rng = np.random.default_rng(seed)
    results_dict: Dict = {}

    for metric_col in metric_cols:
        print(f"  Testing '{metric_col}'...")

        # Collect per-individual values, grouped.
        groups_data: Dict[str, np.ndarray] = {}
        for group_val in sort_values_for_legend(individual_df["_group"].dropna().unique()):
            vals = (
                individual_df[individual_df["_group"] == group_val][metric_col]
                .dropna()
                .values.astype(float)
            )
            if len(vals) > 0:
                groups_data[group_val] = vals

        if len(groups_data) < 2:
            results_dict[metric_col] = {
                "kw_stat": np.nan,
                "kw_pval": np.nan,
                "perm_pval": np.nan,
                "posthoc": None,
                "note": "fewer than 2 groups with valid data",
            }
            print(f"    WARNING: fewer than 2 groups — skipping.")
            continue

        group_names = list(groups_data.keys())
        groups_list = list(groups_data.values())
        all_vals = np.concatenate(groups_list)
        # Integer label array: group 0 has len(groups_list[0]) items, etc.
        group_labels = np.concatenate(
            [np.full(len(g), i, dtype=int) for i, g in enumerate(groups_list)]
        )

        # ── 1. Kruskal-Wallis H test ───────────────────────────────────────
        try:
            kw_stat, kw_pval = kruskal(*groups_list)
        except Exception:
            kw_stat, kw_pval = np.nan, np.nan

        print(f"    Kruskal-Wallis: H={kw_stat:.3f}, p={kw_pval:.4f}")

        # ── 2. Permutation test ────────────────────────────────────────────
        shuffled_labels = group_labels.copy()
        perm_stats = np.empty(n_permutations)
        for perm_i in range(n_permutations):
            rng.shuffle(shuffled_labels)
            perm_groups = [
                all_vals[shuffled_labels == i] for i in range(len(groups_list))
            ]
            # Filter out empty groups that may arise after shuffle.
            perm_groups = [g for g in perm_groups if len(g) > 0]
            try:
                perm_stat, _ = kruskal(*perm_groups) if len(perm_groups) >= 2 else (0.0, 1.0)
            except Exception:
                perm_stat = 0.0
            perm_stats[perm_i] = perm_stat

        if not np.isnan(kw_stat):
            perm_pval = float((perm_stats >= kw_stat).sum() / n_permutations)
        else:
            perm_pval = np.nan

        print(f"    Permutation test: p={perm_pval:.4f}  ({n_permutations:,} shuffles)")

        # ── 3. Post-hoc pairwise comparisons ──────────────────────────────
        posthoc_df = None
        if not np.isnan(kw_pval) and kw_pval < 0.05:
            n_pairs = len(group_names) * (len(group_names) - 1) // 2
            if _SCIKIT_POSTHOCS_AVAILABLE:
                try:
                    dunn_matrix = _posthoc_dunn(groups_list, p_adjust="bonferroni")
                    # Relabel integer indices with group names.
                    dunn_matrix.index = group_names
                    dunn_matrix.columns = group_names
                    posthoc_df = dunn_matrix
                    print(f"    Post-hoc Dunn (Bonferroni): {n_pairs} pairs tested.")
                except Exception as exc:
                    warnings.warn(
                        f"[run_mhi_statistics] Dunn test failed ({exc}); "
                        "falling back to pairwise Mann-Whitney U.",
                        stacklevel=2,
                    )
                    posthoc_df = None

            if posthoc_df is None:
                # Manual pairwise Mann-Whitney U + Bonferroni correction.
                pairwise_records: List[Dict] = []
                for i, g1 in enumerate(group_names):
                    for j, g2 in enumerate(group_names):
                        if j <= i:
                            continue
                        try:
                            mwu_stat, raw_pval = mannwhitneyu(
                                groups_data[g1], groups_data[g2], alternative="two-sided"
                            )
                        except Exception:
                            mwu_stat, raw_pval = np.nan, np.nan
                        bonf_pval = float(min(1.0, raw_pval * n_pairs)) if not np.isnan(raw_pval) else np.nan
                        pairwise_records.append(
                            {
                                "group1": g1,
                                "group2": g2,
                                "mwu_stat": float(mwu_stat),
                                "pval_raw": float(raw_pval),
                                "pval_bonferroni": bonf_pval,
                            }
                        )
                posthoc_df = pd.DataFrame(pairwise_records)
                print(f"    Post-hoc MWU (Bonferroni): {n_pairs} pairs tested.")

        results_dict[metric_col] = {
            "kw_stat": float(kw_stat),
            "kw_pval": float(kw_pval),
            "perm_pval": perm_pval,
            "posthoc": posthoc_df,
        }

    print()
    print(f"Tests complete for {len(results_dict)} metric column(s).")

    # Persist results in .uns.
    if _MHI_RESULTS_KEY not in anndata_object.uns:
        anndata_object.uns[_MHI_RESULTS_KEY] = {}
    if group_column not in anndata_object.uns[_MHI_RESULTS_KEY]:
        anndata_object.uns[_MHI_RESULTS_KEY][group_column] = {}
    anndata_object.uns[_MHI_RESULTS_KEY][group_column]["statistics"] = results_dict

    return results_dict


def run_mhi_age_tests(
    anndata_object: anndata.AnnData,
    species_col: str,
    age_col: str,
    nest_aggregate_by: Optional[str] = None,
    metric_cols: Optional[List[str]] = None,
    n_permutations: int = 10_000,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Test young vs. old MHI differences within each species.

    For each species and each requested MHI metric column, two tests are run:
      1. Mann-Whitney U test (young vs. old individuals, two-sided).
      2. Permutation test (n_permutations shuffles of age labels within species).

    The function reads per-individual MHI values from
    .uns['mhi_results'][individual_col] and maps individuals to their age group
    and species using anndata_object.obs.

    Arguments:
        anndata_object: AnnData with per-individual MHI results in .uns.
        species_col: .obs column identifying species (e.g. 'specie').
        age_col: .obs column identifying age groups (e.g. 'age_group').
            Expected to have exactly two unique values per species (young/old).
        nest_aggregate_by: .obs column identifying individuals.  If None,
            auto-detects 'unique_subject_ID' or 'subject_ID'.
        metric_cols: Columns to test.  Default: all columns starting with 'MHI_'.
        n_permutations: Number of label-shuffle permutations per species × metric.
        seed: Random seed.

    Returns:
        pd.DataFrame with one row per (species, metric_col) combination, columns:
            species          — species name.
            metric_col       — metric tested.
            n_young          — number of young individuals with valid values.
            n_old            — number of old individuals with valid values.
            mwu_stat         — Mann-Whitney U statistic.
            mwu_pval         — Mann-Whitney p-value (two-sided).
            perm_pval        — Permutation test p-value.
        Results stored in .uns['mhi_results'][species_col]['age_tests'].

    Raises:
        ValueError: If no per-individual MHI results are found, or if
            species_col / age_col are not present in .obs.
    """
    individual_col = (
        _resolve_nest_aggregate_by(anndata_object, nest_aggregate_by)
        if nest_aggregate_by is not None
        else _resolve_group_by(anndata_object, None)
    )

    for col in (species_col, age_col):
        if col not in anndata_object.obs.columns:
            raise ValueError(
                f"Column '{col}' not found in .obs. "
                f"Available: {sorted(anndata_object.obs.columns.tolist())}"
            )

    if _MHI_RESULTS_KEY not in anndata_object.uns:
        raise ValueError(
            f"No MHI results in .uns['{_MHI_RESULTS_KEY}']. "
            "Call compute_all_mhi() first."
        )
    if individual_col not in anndata_object.uns[_MHI_RESULTS_KEY]:
        raise ValueError(
            f"No per-individual MHI results for '{individual_col}'. "
            f"Available: {sorted(anndata_object.uns[_MHI_RESULTS_KEY].keys())}"
        )

    individual_results = anndata_object.uns[_MHI_RESULTS_KEY][individual_col]
    if "all" in individual_results:
        individual_df = individual_results["all"].copy()
    else:
        individual_df = None
        for key in ("D", "S", "E"):
            if key not in individual_results:
                continue
            part = individual_results[key]
            if individual_df is None:
                individual_df = part.copy()
            else:
                overlap = [c for c in part.columns if c in individual_df.columns]
                individual_df = individual_df.join(
                    part.drop(columns=overlap, errors="ignore"), how="outer"
                )
    if individual_df is None or individual_df.empty:
        raise ValueError("No per-individual MHI DataFrames found.")

    _COUNT_COLS = {"n_mitos", "n_individuals", "n_mitos_total"}
    if metric_cols is None:
        metric_cols = [
            c for c in individual_df.columns
            if c.startswith("MHI_") and c not in _COUNT_COLS
        ]
    metric_cols = [c for c in metric_cols if c in individual_df.columns]

    # Build individual → species and individual → age mapping.
    meta_df = (
        anndata_object.obs[[individual_col, species_col, age_col]]
        .drop_duplicates()
        .set_index(individual_col)
    )
    individual_df["_species"] = individual_df.index.map(meta_df[species_col])
    individual_df["_age"] = individual_df.index.map(meta_df[age_col])

    print("=" * 60)
    print("MHI AGE TESTS (YOUNG vs. OLD per SPECIES)")
    print("=" * 60)
    print(f"  Individual column : '{individual_col}'")
    print(f"  Species column    : '{species_col}'")
    print(f"  Age column        : '{age_col}'")
    print(f"  Metric columns    : {metric_cols}")
    print(f"  Permutations      : {n_permutations:,}")
    print()

    all_species = sort_values_for_legend(individual_df["_species"].dropna().unique())
    rng = np.random.default_rng(seed)
    rows: List[Dict] = []

    for species_val in all_species:
        species_mask = individual_df["_species"] == species_val
        species_sub = individual_df[species_mask]
        age_values = sort_values_for_legend(species_sub["_age"].dropna().unique())

        if len(age_values) < 2:
            warnings.warn(
                f"[run_mhi_age_tests] Species '{species_val}' has fewer than "
                f"2 age groups ({age_values}). Skipping.",
                stacklevel=2,
            )
            continue

        # Use first two age values (sorted): treat them as group A and group B.
        age_a, age_b = age_values[0], age_values[1]
        print(f"  {species_val}: '{age_a}' vs '{age_b}'")

        for metric_col in metric_cols:
            vals_a = (
                species_sub[species_sub["_age"] == age_a][metric_col]
                .dropna()
                .values.astype(float)
            )
            vals_b = (
                species_sub[species_sub["_age"] == age_b][metric_col]
                .dropna()
                .values.astype(float)
            )

            if len(vals_a) == 0 or len(vals_b) == 0:
                rows.append({
                    "species": species_val,
                    "metric_col": metric_col,
                    "n_young": len(vals_a),
                    "n_old": len(vals_b),
                    "mwu_stat": np.nan,
                    "mwu_pval": np.nan,
                    "perm_pval": np.nan,
                })
                continue

            # Mann-Whitney U test.
            try:
                mwu_stat, mwu_pval = mannwhitneyu(vals_a, vals_b, alternative="two-sided")
            except Exception:
                mwu_stat, mwu_pval = np.nan, np.nan

            # Permutation test: shuffle age labels within this species × metric.
            all_combined = np.concatenate([vals_a, vals_b])
            n_a = len(vals_a)
            perm_stats = np.empty(n_permutations)
            for perm_i in range(n_permutations):
                shuffled = rng.permutation(all_combined)
                perm_a = shuffled[:n_a]
                perm_b = shuffled[n_a:]
                try:
                    p_stat, _ = mannwhitneyu(perm_a, perm_b, alternative="two-sided")
                except Exception:
                    p_stat = 0.0
                perm_stats[perm_i] = p_stat

            if not np.isnan(mwu_stat):
                perm_pval = float((perm_stats >= mwu_stat).sum() / n_permutations)
            else:
                perm_pval = np.nan

            rows.append({
                "species": species_val,
                "metric_col": metric_col,
                "n_young": len(vals_a),
                "n_old": len(vals_b),
                "mwu_stat": float(mwu_stat),
                "mwu_pval": float(mwu_pval),
                "perm_pval": perm_pval,
            })

    results_df = pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=["species", "metric_col", "n_young", "n_old",
                 "mwu_stat", "mwu_pval", "perm_pval"]
    )

    print()
    print(f"Age tests complete: {len(results_df)} (species × metric) combinations.")
    print(results_df.round(4).to_string(index=False))
    print()

    # Persist in .uns.
    if _MHI_RESULTS_KEY not in anndata_object.uns:
        anndata_object.uns[_MHI_RESULTS_KEY] = {}
    if species_col not in anndata_object.uns[_MHI_RESULTS_KEY]:
        anndata_object.uns[_MHI_RESULTS_KEY][species_col] = {}
    anndata_object.uns[_MHI_RESULTS_KEY][species_col]["age_tests"] = results_df

    return results_df


# ──────────────────────────────────────────────────────────────────────────────
# New visualizations
# ──────────────────────────────────────────────────────────────────────────────


def plot_mhi_barplot(
    anndata_object: anndata.AnnData,
    group_by: Optional[str] = None,
    nest_aggregate_by: Optional[str] = None,
    metric: str = "MHI_D_median",
    standardize_on: str = "all",
    n_pca_components: Union[str, int, None] = "auto",
) -> None:
    """
    Barplot of group-level MHI median with per-individual std error bars.

    Each bar represents one group (e.g. one species).  The bar height is the
    group-level median of per-individual MHI values.  The error bar shows the
    inter-individual standard deviation within the group.

    When nest_aggregate_by differs from group_by (two-level mode), the function
    reads from .uns['mhi_results'][group_by] directly.  If not yet computed,
    compute_mhi_d() is called automatically for metric='MHI_D_median'.

    Arguments:
        anndata_object: AnnData with MHI results in .uns.
        group_by: .obs column identifying groups.  Default: auto-detect.
        nest_aggregate_by: .obs column identifying individuals.  When provided and
            different from group_by, reads two-level aggregated results.
        metric: Column name to plot (default 'MHI_D_median').
        standardize_on: Passed to compute_mhi_d() if computation is triggered.
        n_pca_components: Passed to compute_mhi_d() if computation is triggered.

    Returns:
        None.  Displays the figure inline.
    """
    group_column = _resolve_group_by(anndata_object, group_by)
    if nest_aggregate_by is not None:
        individual_col = _resolve_nest_aggregate_by(anndata_object, nest_aggregate_by)
    else:
        individual_col = group_column

    # Retrieve or compute group-level results.
    metric_key = "D" if "MHI_D" in metric else ("S" if "MHI_S" in metric else "E")
    group_df = None
    if (
        _MHI_RESULTS_KEY in anndata_object.uns
        and group_column in anndata_object.uns[_MHI_RESULTS_KEY]
        and metric_key in anndata_object.uns[_MHI_RESULTS_KEY][group_column]
    ):
        group_df = anndata_object.uns[_MHI_RESULTS_KEY][group_column][metric_key]

    if group_df is None or metric not in group_df.columns:
        if individual_col != group_column:
            group_df = compute_mhi_d(
                anndata_object,
                group_by=group_column,
                nest_aggregate_by=individual_col,
                standardize_on=standardize_on,
                n_pca_components=n_pca_components,
            )
        else:
            group_df = compute_mhi_d(
                anndata_object,
                group_by=group_column,
                standardize_on=standardize_on,
                n_pca_components=n_pca_components,
            )

    if metric not in group_df.columns:
        print(f"WARNING: metric '{metric}' not found in group-level results. "
              f"Available columns: {group_df.columns.tolist()}")
        return

    print(f"Plotting MHI barplot for '{metric}' by '{group_column}'...")

    sorted_groups = sort_values_for_legend(group_df.index.tolist())
    medians = [group_df.loc[g, metric] for g in sorted_groups]

    # Standard deviation column: {metric_base}_std in two-level mode.
    std_col = metric.replace("_median", "_std") if "_median" in metric else f"{metric}_std"
    if std_col in group_df.columns:
        stds = [group_df.loc[g, std_col] for g in sorted_groups]
    else:
        stds = [np.nan] * len(sorted_groups)

    # Use color palette from .uns if available; fall back to a default palette.
    color_palette = anndata_object.uns.get("color_palette", {})
    bar_colors = [
        color_palette.get(g, f"C{i % 10}") for i, g in enumerate(sorted_groups)
    ]

    fig, ax = plt.subplots(figsize=(max(5, 1.5 * len(sorted_groups)), 5))

    x_positions = np.arange(len(sorted_groups))
    bars = ax.bar(
        x_positions,
        medians,
        color=bar_colors,
        width=0.6,
        alpha=0.85,
        edgecolor="white",
        linewidth=0.8,
    )
    # Error bars (std); NaN stds are silently ignored by matplotlib.
    ax.errorbar(
        x_positions,
        medians,
        yerr=stds,
        fmt="none",
        color="black",
        capsize=5,
        linewidth=1.5,
        zorder=5,
    )

    ax.set_xticks(x_positions)
    ax.set_xticklabels(sorted_groups, rotation=30, ha="right", fontsize=10)
    ax.set_ylabel(metric, fontsize=11)
    ax.set_title(
        f"{metric} by '{group_column}'\n(bar = group median, error bar = inter-individual std)",
        fontsize=12,
    )
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.show()

    print(f"  Groups shown: {sorted_groups}")


def plot_mhi_scatter_de(
    anndata_object: anndata.AnnData,
    group_by: Optional[str] = None,
    nest_aggregate_by: Optional[str] = None,
    standardize_on: str = "all",
    n_pca_components: Union[str, int, None] = "auto",
) -> None:
    """
    Scatter plot of MHI-D (dispersion) vs MHI-E (KNN entropy) per individual.

    Each dot represents one individual, coloured by its group (species,
    condition, etc.).  Group centroids are marked with a larger marker; error
    crosses show the inter-individual standard deviation per group.

    Arguments:
        anndata_object: AnnData with MHI results in .uns.
        group_by: .obs grouping column.  Default: auto-detect.
        nest_aggregate_by: .obs column for individuals.  When provided and different
            from group_by, uses two-level results.
        standardize_on: Passed to compute_mhi_d() if triggered.
        n_pca_components: Passed to compute_mhi_* if triggered.

    Returns:
        None.  Displays the figure inline.
    """
    group_column = _resolve_group_by(anndata_object, group_by)
    if nest_aggregate_by is not None:
        individual_col = _resolve_nest_aggregate_by(anndata_object, nest_aggregate_by)
    else:
        individual_col = group_column

    # Ensure per-individual D and E results exist.
    def _get_or_compute(metric_key: str, compute_fn) -> Optional[pd.DataFrame]:
        if (
            _MHI_RESULTS_KEY in anndata_object.uns
            and individual_col in anndata_object.uns[_MHI_RESULTS_KEY]
            and metric_key in anndata_object.uns[_MHI_RESULTS_KEY][individual_col]
        ):
            return anndata_object.uns[_MHI_RESULTS_KEY][individual_col][metric_key]
        return compute_fn()

    d_df = _get_or_compute(
        "D",
        lambda: compute_mhi_d(
            anndata_object,
            group_by=individual_col,
            standardize_on=standardize_on,
            n_pca_components=n_pca_components,
        ),
    )
    e_df = _get_or_compute(
        "E",
        lambda: compute_mhi_e(
            anndata_object,
            group_by=individual_col,
            n_pca_components=n_pca_components,
        ),
    )

    # Merge D and E on individual index.
    scatter_df = d_df[["MHI_D_median"]].join(e_df[["MHI_E_knn"]], how="inner")
    scatter_df.dropna(subset=["MHI_D_median", "MHI_E_knn"], inplace=True)

    # Attach group labels.  When individual and group are the same column,
    # the mapping is identity (each individual IS its own group).
    if individual_col == group_column:
        unique_vals = anndata_object.obs[individual_col].unique()
        individual_to_group = pd.Series(unique_vals, index=unique_vals)
    else:
        individual_to_group = (
            anndata_object.obs[[individual_col, group_column]]
            .drop_duplicates()
            .set_index(individual_col)[group_column]
        )
    scatter_df["_group"] = scatter_df.index.map(individual_to_group)
    scatter_df.dropna(subset=["_group"], inplace=True)

    sorted_groups = sort_values_for_legend(scatter_df["_group"].unique())
    colormap = plt.cm.tab10(np.linspace(0, 1, max(len(sorted_groups), 1)))
    group_to_color = {g: colormap[i] for i, g in enumerate(sorted_groups)}

    print(f"Plotting MHI-D vs MHI-E scatter for '{group_column}'...")

    fig, ax = plt.subplots(figsize=(7, 6))

    for group_val in sorted_groups:
        sub = scatter_df[scatter_df["_group"] == group_val]
        color = group_to_color[group_val]
        ax.scatter(
            sub["MHI_D_median"],
            sub["MHI_E_knn"],
            color=color,
            s=60,
            alpha=0.8,
            label=str(group_val),
            zorder=3,
        )
        # Group centroid with error cross.
        if len(sub) >= 2:
            cx = sub["MHI_D_median"].mean()
            cy = sub["MHI_E_knn"].mean()
            sx = sub["MHI_D_median"].std()
            sy = sub["MHI_E_knn"].std()
            ax.scatter(cx, cy, color=color, s=80, marker="D", edgecolors="black",
                       linewidths=0.7, zorder=5)
            ax.errorbar(cx, cy, xerr=sx, yerr=sy, fmt="none", color=color,
                        capsize=3, linewidth=1.0, zorder=4)

    ax.set_xlabel("MHI-D (Dispersion — median distance to centroid)", fontsize=11)
    ax.set_ylabel("MHI-E (KNN Entropy)", fontsize=11)
    ax.set_title(
        f"MHI-D vs MHI-E per individual\n(diamonds = group centroids, crosses = ±1 std)",
        fontsize=12,
    )
    ax.legend(fontsize=9, title=group_column, loc="best")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    print(f"  {len(scatter_df)} individuals plotted across {len(sorted_groups)} groups.")


def plot_mhi_stripplot(
    anndata_object: anndata.AnnData,
    group_by: Optional[str] = None,
    age_col: str = "age_group",
    nest_aggregate_by: Optional[str] = None,
    metrics: Optional[List[str]] = None,
    standardize_on: str = "all",
    n_pca_components: Union[str, int, None] = "auto",
) -> None:
    """
    Boxplot + stripplot of per-individual MHI values by group, coloured by age.

    Each panel shows one MHI metric.  Each dot is one individual; the box
    shows the quartile range across individuals within each group.  Dots are
    coloured by the age column (typically 'young' vs 'old'), making age-driven
    heterogeneity differences immediately visible.

    Arguments:
        anndata_object: AnnData with MHI results in .uns.
        group_by: .obs grouping column (e.g. 'specie').  Default: auto-detect.
        age_col: .obs column with age group labels (default 'age_group').
        nest_aggregate_by: .obs column for individuals.  When provided, reads per-
            individual results from .uns.
        metrics: List of metric column names to plot.  Default: all MHI_* columns
            found in per-individual results.
        standardize_on: Passed to compute_mhi_d() if triggered.
        n_pca_components: Passed to compute_mhi_* if triggered.

    Returns:
        None.  Displays the figure inline.
    """
    group_column = _resolve_group_by(anndata_object, group_by)
    if nest_aggregate_by is not None:
        individual_col = _resolve_nest_aggregate_by(anndata_object, nest_aggregate_by)
    else:
        individual_col = group_column

    if age_col not in anndata_object.obs.columns:
        raise ValueError(
            f"age_col='{age_col}' not found in .obs. "
            f"Available: {sorted(anndata_object.obs.columns.tolist())}"
        )

    # Build per-individual DataFrame with group and age metadata.
    if (
        _MHI_RESULTS_KEY in anndata_object.uns
        and individual_col in anndata_object.uns[_MHI_RESULTS_KEY]
    ):
        individual_results = anndata_object.uns[_MHI_RESULTS_KEY][individual_col]
        if "all" in individual_results:
            individual_df = individual_results["all"].copy()
        else:
            individual_df = None
            for key in ("D", "S", "E"):
                if key not in individual_results:
                    continue
                part = individual_results[key]
                if individual_df is None:
                    individual_df = part.copy()
                else:
                    overlap = [c for c in part.columns if c in individual_df.columns]
                    individual_df = individual_df.join(
                        part.drop(columns=overlap, errors="ignore"), how="outer"
                    )
    else:
        # Trigger computation.
        individual_df = compute_mhi_d(
            anndata_object,
            group_by=individual_col,
            standardize_on=standardize_on,
            n_pca_components=n_pca_components,
        )

    if individual_df is None or individual_df.empty:
        print("WARNING: No per-individual MHI data found.")
        return

    # Attach group and age labels.
    meta_df = (
        anndata_object.obs[[individual_col, group_column, age_col]]
        .drop_duplicates()
        .set_index(individual_col)
    )
    individual_df["_group"] = individual_df.index.map(meta_df[group_column])
    individual_df["_age"] = individual_df.index.map(meta_df[age_col])
    individual_df.dropna(subset=["_group", "_age"], inplace=True)

    _COUNT_COLS = {"n_mitos", "n_individuals", "n_mitos_total", "_group", "_age"}
    if metrics is None:
        metrics = [
            c for c in individual_df.columns
            if c.startswith("MHI_") and c not in _COUNT_COLS
        ]
    metrics = [c for c in metrics if c in individual_df.columns and c not in _COUNT_COLS]

    if not metrics:
        print("WARNING: No MHI metric columns found.")
        return

    print(f"Plotting MHI stripplot for {len(metrics)} metric(s) by '{group_column}'...")

    n_panels = len(metrics)
    fig, axes = plt.subplots(
        1, n_panels, figsize=(max(5, 4 * n_panels), 6), squeeze=False
    )

    # Assign age colours: at most 2 age groups (young / old).
    age_values = sort_values_for_legend(individual_df["_age"].dropna().unique())
    age_palette = {av: col for av, col in zip(age_values, ["#4daf4a", "#e41a1c"])}

    for panel_index, metric_col in enumerate(metrics):
        ax = axes[0, panel_index]
        plot_data = individual_df[["_group", "_age", metric_col]].dropna()

        if plot_data.empty:
            ax.set_visible(False)
            continue

        sorted_groups = sort_values_for_legend(plot_data["_group"].unique())

        # Seaborn boxplot + stripplot (jittered individual dots).
        sns.boxplot(
            data=plot_data,
            x="_group",
            y=metric_col,
            order=sorted_groups,
            color="white",
            linewidth=1.0,
            fliersize=0,
            ax=ax,
        )
        sns.stripplot(
            data=plot_data,
            x="_group",
            y=metric_col,
            hue="_age",
            order=sorted_groups,
            palette=age_palette,
            size=8,
            jitter=True,
            dodge=True,
            alpha=0.8,
            ax=ax,
        )

        ax.set_xlabel("")
        ax.set_xticklabels(sorted_groups, rotation=30, ha="right", fontsize=9)
        ax.set_ylabel(metric_col if panel_index == 0 else "", fontsize=10)
        ax.set_title(metric_col, fontsize=10)
        ax.grid(True, axis="y", alpha=0.3)

        # Only show legend on the last panel.
        handles, labels = ax.get_legend_handles_labels()
        if panel_index < n_panels - 1:
            if ax.get_legend():
                ax.get_legend().remove()
        else:
            ax.legend(
                handles, labels, title=age_col, fontsize=8, loc="upper right"
            )

    fig.suptitle(
        f"Per-Individual MHI Values by '{group_column}' (coloured by '{age_col}')",
        fontsize=12,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.show()

    print(f"  {len(individual_df)} individuals plotted.")


def plot_mhi_species_heatmap(
    anndata_object: anndata.AnnData,
    group_by: Optional[str] = None,
    nest_aggregate_by: Optional[str] = None,
    standardize_on: str = "all",
    n_pca_components: Union[str, int, None] = "auto",
) -> None:
    """
    Heatmap of pairwise Euclidean distances between groups in MHI space.

    Each group is represented by a vector of its median MHI component values
    (MHI_D_median, MHI_S_dip_dist if available, MHI_E_knn if available).
    The pairwise Euclidean distance between these vectors is displayed as an
    annotated heatmap, revealing which species/conditions are most similar or
    different in their mitochondrial heterogeneity profiles.

    Arguments:
        anndata_object: AnnData with MHI results in .uns.
        group_by: .obs grouping column (e.g. 'specie').  Default: auto-detect.
        nest_aggregate_by: .obs column for individuals.  When provided and different
            from group_by, uses group-level aggregated results.
        standardize_on: Passed to compute_mhi_d() if triggered.
        n_pca_components: Passed to compute_mhi_d() if triggered.

    Returns:
        None.  Displays the figure inline.
    """
    group_column = _resolve_group_by(anndata_object, group_by)
    if nest_aggregate_by is not None:
        individual_col = _resolve_nest_aggregate_by(anndata_object, nest_aggregate_by)
    else:
        individual_col = group_column

    # Collect available group-level results.
    group_results = {}
    if _MHI_RESULTS_KEY in anndata_object.uns:
        mhi_uns = anndata_object.uns[_MHI_RESULTS_KEY]
        if group_column in mhi_uns:
            for key in ("all", "D", "S", "E"):
                if key in mhi_uns[group_column]:
                    group_results[key] = mhi_uns[group_column][key]

    # Build a merged group-level DataFrame with the representative metrics.
    if "all" in group_results:
        group_df = group_results["all"].copy()
    elif group_results:
        group_df = None
        for key in ("D", "S", "E"):
            if key not in group_results:
                continue
            part = group_results[key]
            if group_df is None:
                group_df = part.copy()
            else:
                overlap = [c for c in part.columns if c in group_df.columns]
                group_df = group_df.join(
                    part.drop(columns=overlap, errors="ignore"), how="outer"
                )
    else:
        # Trigger computation.
        if individual_col != group_column:
            group_df = compute_mhi_d(
                anndata_object,
                group_by=group_column,
                nest_aggregate_by=individual_col,
                standardize_on=standardize_on,
                n_pca_components=n_pca_components,
            )
        else:
            group_df = compute_mhi_d(
                anndata_object,
                group_by=group_column,
                standardize_on=standardize_on,
                n_pca_components=n_pca_components,
            )

    if group_df is None or group_df.empty:
        print("WARNING: No group-level MHI data available.")
        return

    # Select the representative scalar metrics for distance computation.
    # Prefer the _median suffix (two-level output), fall back to direct names.
    candidate_cols = [
        "MHI_D_median",
        "MHI_S_dip_dist_median", "MHI_S_dip_dist",
        "MHI_E_knn_median", "MHI_E_knn",
    ]
    vector_cols = [c for c in candidate_cols if c in group_df.columns]
    if not vector_cols:
        print(
            f"WARNING: No MHI metric columns found in group-level results. "
            f"Available: {group_df.columns.tolist()}"
        )
        return

    print(f"Plotting MHI species heatmap using columns: {vector_cols}")

    vector_matrix_df = group_df[vector_cols].dropna(how="all")
    # Fill remaining NaN with column means so distance is defined everywhere.
    vector_matrix_df = vector_matrix_df.fillna(vector_matrix_df.mean())
    # Standardise each MHI dimension to [0, 1] range before computing distance.
    col_mins = vector_matrix_df.min()
    col_maxs = vector_matrix_df.max()
    col_range = (col_maxs - col_mins).replace(0, 1)  # avoid division by zero
    normalised = (vector_matrix_df - col_mins) / col_range

    sorted_groups = sort_values_for_legend(normalised.index.tolist())
    n_groups = len(sorted_groups)

    # Compute pairwise Euclidean distance matrix.
    distance_matrix = np.zeros((n_groups, n_groups))
    for i, g1 in enumerate(sorted_groups):
        for j, g2 in enumerate(sorted_groups):
            v1 = normalised.loc[g1].values.astype(float)
            v2 = normalised.loc[g2].values.astype(float)
            distance_matrix[i, j] = float(np.linalg.norm(v1 - v2))

    distance_df = pd.DataFrame(
        distance_matrix, index=sorted_groups, columns=sorted_groups
    )

    fig, ax = plt.subplots(figsize=(max(6, n_groups + 1), max(5, n_groups)))
    sns.heatmap(
        distance_df,
        annot=True,
        fmt=".2f",
        cmap="YlOrRd",
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"label": "Euclidean distance in MHI space"},
        ax=ax,
    )
    ax.set_title(
        f"Pairwise MHI Distance Between '{group_column}' Groups\n"
        f"(computed from: {vector_cols})",
        fontsize=12,
    )
    plt.tight_layout()
    plt.show()

    print(
        "  Interpretation: lower value = more similar MHI profiles; "
        "higher value = more divergent mitochondrial heterogeneity."
    )


# ──────────────────────────────────────────────────────────────────────────────
# All-pairs chimera validation
# ──────────────────────────────────────────────────────────────────────────────


def validate_all_chimera_pairs(
    anndata_object: anndata.AnnData,
    species_col: str = "specie",
    age_col: str = "age_group",
    subject_col: Optional[str] = None,
    ratios: Optional[List[float]] = None,
    n_repetitions: int = 3,
    seed: int = 42,
    metrics: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Validate MHI monotonicity for all young/old pairs within each species.

    For each species, enumerates all combinations of one young individual and
    one old individual.  For each pair, validate_mhi_with_chimeras() is called
    to verify that mixing the two individuals' mitochondria produces a
    monotonically increasing MHI curve up to ratio=0.5.

    After collecting all pair results, a grid figure is displayed with one
    subplot per pair, arranged by species.

    Arguments:
        anndata_object: AnnData containing mitochondria from all individuals.
        species_col: .obs column identifying species (default 'specie').
        age_col: .obs column identifying age groups (default 'age_group').
        subject_col: .obs column identifying individuals.  If None,
            auto-detects 'unique_subject_ID' or 'subject_ID'.
        ratios: Mixing ratios to test.  Default: [0.1, 0.2, …, 0.9].
        n_repetitions: Random subsamples per ratio (default 3).
        seed: Base random seed.
        metrics: Subset of ['D', 'S', 'E'].  Default: ['D'].

    Returns:
        pd.DataFrame combining all pair results, with added columns:
            species   — species name.
            pair_id   — '{young_id} × {old_id}'.

    Raises:
        ValueError: If species_col or age_col not found in .obs.
    """
    if ratios is None:
        ratios = [round(r * 0.1, 1) for r in range(1, 10)]
    if metrics is None:
        metrics = ["D"]

    for col in (species_col, age_col):
        if col not in anndata_object.obs.columns:
            raise ValueError(
                f"Column '{col}' not found in .obs. "
                f"Available: {sorted(anndata_object.obs.columns.tolist())}"
            )

    resolved_subject_col = subject_col or _get_subject_column(anndata_object.obs)
    if resolved_subject_col is None:
        raise ValueError(
            "No subject column found. Expected 'unique_subject_ID' or 'subject_ID'. "
            "Pass subject_col explicitly."
        )

    print("=" * 60)
    print("ALL-PAIRS CHIMERA VALIDATION")
    print("=" * 60)
    print(f"  Species column  : '{species_col}'")
    print(f"  Age column      : '{age_col}'")
    print(f"  Subject column  : '{resolved_subject_col}'")
    print(f"  Ratios          : {ratios}")
    print(f"  Repetitions     : {n_repetitions}")
    print(f"  Metrics         : {metrics}")
    print()

    # Build per-individual metadata: subject → species, age.
    individual_meta = (
        anndata_object.obs[[resolved_subject_col, species_col, age_col]]
        .drop_duplicates()
        .set_index(resolved_subject_col)
    )

    all_species = sort_values_for_legend(individual_meta[species_col].dropna().unique())
    all_pair_rows: List[Dict] = []
    pair_info: List[Dict] = []  # (species, pair_id, young_id, old_id)

    for species_val in all_species:
        species_inds = individual_meta[individual_meta[species_col] == species_val]
        age_values = sort_values_for_legend(species_inds[age_col].dropna().unique())

        if len(age_values) < 2:
            warnings.warn(
                f"[validate_all_chimera_pairs] Species '{species_val}' has fewer "
                f"than 2 age groups. Skipping.",
                stacklevel=2,
            )
            continue

        # Treat first age value as "young" and second as "old" (sorted order).
        young_age, old_age = age_values[0], age_values[1]
        young_ids = species_inds[species_inds[age_col] == young_age].index.tolist()
        old_ids = species_inds[species_inds[age_col] == old_age].index.tolist()

        for young_id in young_ids:
            for old_id in old_ids:
                pair_info.append({
                    "species": species_val,
                    "pair_id": f"{young_id} × {old_id}",
                    "young_id": young_id,
                    "old_id": old_id,
                })

    if not pair_info:
        print("WARNING: No valid young/old pairs found.")
        return pd.DataFrame()

    print(f"  Found {len(pair_info)} pair(s) across {len(all_species)} species.")
    print()

    # Compute chimera validation for each pair (suppress individual plots).
    for pair_dict in pair_info:
        pair_label = pair_dict["pair_id"]
        print(f"  Pair: {pair_label}  ({pair_dict['species']})")
        try:
            pair_result = validate_mhi_with_chimeras(
                anndata_object,
                subject_id_1=pair_dict["young_id"],
                subject_id_2=pair_dict["old_id"],
                ratios=ratios,
                n_repetitions=n_repetitions,
                seed=seed,
                metrics=metrics,
                show_plot=False,
            )
            pair_result = pair_result.copy()
            pair_result["species"] = pair_dict["species"]
            pair_result["pair_id"] = pair_label
            all_pair_rows.append(pair_result)
        except Exception as exc:
            warnings.warn(
                f"[validate_all_chimera_pairs] Pair '{pair_label}' failed: {exc}",
                stacklevel=2,
            )

    if not all_pair_rows:
        print("WARNING: No chimera results collected.")
        return pd.DataFrame()

    combined_df = pd.concat(all_pair_rows, ignore_index=True)

    # ── Grid plot ─────────────────────────────────────────────────────────
    # Metric columns to plot (same logic as validate_mhi_with_chimeras).
    metric_col_map = {
        "D": ["MHI_D_median"],
        "S": ["MHI_S_dip", "MHI_S_gmm_delta_bic"],
        "E": ["MHI_E_knn", "MHI_E_binning"],
    }
    metric_to_plot_cols = [
        col
        for code in metrics
        for col in metric_col_map.get(code, [])
        if col in combined_df.columns
    ]

    if metric_to_plot_cols:
        n_pairs = len(pair_info)
        n_plot_cols = len(metric_to_plot_cols)
        fig, axes = plt.subplots(
            n_pairs,
            n_plot_cols,
            figsize=(5 * n_plot_cols, 4 * n_pairs),
            squeeze=False,
        )

        chimera_df = combined_df[combined_df["repetition"] > 0]
        real_df = combined_df[combined_df["repetition"] == 0]

        for pair_row_idx, pair_dict in enumerate(pair_info):
            young_id = pair_dict["young_id"]
            old_id = pair_dict["old_id"]
            pair_label = pair_dict["pair_id"]

            pair_chimera = chimera_df[chimera_df["pair_id"] == pair_label]
            pair_real = real_df[real_df["pair_id"] == pair_label]

            for col_idx, plot_col in enumerate(metric_to_plot_cols):
                ax = axes[pair_row_idx, col_idx]

                if plot_col not in pair_chimera.columns or pair_chimera.empty:
                    ax.set_visible(False)
                    continue

                grouped = pair_chimera.groupby("ratio")[plot_col].agg(["mean", "std"])
                ax.plot(
                    grouped.index,
                    grouped["mean"],
                    marker="o",
                    color="#2171b5",
                    label="Chimera mean ± std",
                )
                ax.fill_between(
                    grouped.index,
                    grouped["mean"] - grouped["std"].fillna(0),
                    grouped["mean"] + grouped["std"].fillna(0),
                    alpha=0.25,
                    color="#2171b5",
                )

                if plot_col in pair_real.columns:
                    r0 = pair_real[pair_real["ratio"] == 0.0][plot_col].values
                    r1 = pair_real[pair_real["ratio"] == 1.0][plot_col].values
                    if len(r0) > 0 and not np.isnan(r0[0]):
                        ax.axhline(r0[0], linestyle="--", color="#d62728",
                                   alpha=0.7, label=f"Pure {old_id}")
                    if len(r1) > 0 and not np.isnan(r1[0]):
                        ax.axhline(r1[0], linestyle="--", color="#2ca02c",
                                   alpha=0.7, label=f"Pure {young_id}")

                row_label = (
                    f"{pair_dict['species']}\n{pair_label}"
                    if col_idx == 0
                    else ""
                )
                ax.set_ylabel(row_label, fontsize=8)
                ax.set_xlabel("Mixing ratio" if pair_row_idx == n_pairs - 1 else "")
                ax.set_title(plot_col if pair_row_idx == 0 else "", fontsize=9)
                ax.legend(fontsize=7, loc="upper left")
                ax.grid(True, alpha=0.3)

        fig.suptitle(
            "Chimera Validation — All Young/Old Pairs\n"
            f"(species column: '{species_col}', age column: '{age_col}')",
            fontsize=12,
            fontweight="bold",
        )
        plt.tight_layout()
        plt.show()

    print(f"All-pairs chimera validation complete. {len(combined_df)} total rows.")
    return combined_df
