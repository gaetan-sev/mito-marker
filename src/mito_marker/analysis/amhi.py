"""
amhi.py

Absolute Mitochondrial Heterogeneity Index (AMHI) — a cross-experiment reference
framework that makes mitochondrial heterogeneity metrics fully comparable between
datasets, time points, and new subjects.

Background
----------
The existing MHI-D is *relative*: it refits a StandardScaler and PCA on whatever
subjects are present in the AnnData at call time.  Adding one new subject shifts
the axes and all previous values change.  Nothing is comparable across experiments.

This module introduces a **fixed reference space**: fit once on a chosen population,
freeze the transforms, and project any current or future subject into the same
coordinate system without refitting.

Two orthogonal absolute metrics
--------------------------------
  MHI_D_absolute
      Median Euclidean distance from each mitochondrion to *its own subject's
      centroid* in the fixed reference space.
      Measures internal heterogeneity — spread of mitos around their own centre.
      Subject A (tight cluster, far from AllMitoMean): LOW.
      Subject B (dispersed cluster, near AllMitoMean): HIGH.
      Equivalent to MHI-D but on a fixed, absolute scale.

  AMHI_offset
      Euclidean distance from the *subject's centroid* to AllMitoMean (the origin
      of the fixed PCA space).
      Measures absolute position — how different is this subject's typical mito
      from the global reference.
      Independent of MHI_D_absolute.

  AMHI_D
      Median Euclidean distance from each mitochondrion to AllMitoMean (the origin).
      Combines offset + spread: AMHI_D² ≈ AMHI_offset² + MHI_D_absolute².

Fixed transform pipeline
------------------------
1. compute_all_mito_mean():   fits StandardScaler + PCA on the chosen reference
                               population and freezes them in .uns['amhi_reference'].
                               Optionally injects AllMitoMean as a synthetic obs row.

2. compute_amhi() /
   compute_mhi_d_absolute():  load the frozen transforms (never refit) and project
                               any selected mitos using transform() only.

Typical notebook usage
-----------------------
    from mito_marker import (
        ingest_tem_folder,
        compute_all_mito_mean,
        compute_amhi,
        compute_mhi_d_absolute,
        plot_amhi_distances,
        plot_amhi_2d,
        plot_radar,
    )

    adata = ingest_tem_folder("data/raw/tem/")

    # Freeze reference on all subjects (or a subset via obs_filter)
    adata, _ = compute_all_mito_mean(adata)

    # AllMitoMean now appears naturally in radar plots
    plot_radar(adata, group_by="unique_subject_ID")

    # Compute absolute metrics for every subject
    amhi_df  = compute_amhi(adata, group_by="unique_subject_ID")
    mhid_df  = compute_mhi_d_absolute(adata, group_by="unique_subject_ID")

    # Visualise
    plot_amhi_distances(adata, group_by="unique_subject_ID")
    plot_amhi_2d(adata, color_by="condition")
"""

import pickle
import warnings
from typing import Dict, List, Optional, Tuple, Union

import anndata
import matplotlib.lines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import umap as UmapLib
from sklearn.decomposition import PCA as SklearnPCA
from sklearn.preprocessing import StandardScaler

from mito_marker.analysis.colors import get_color_for_value, sort_values_for_legend
from mito_marker.analysis.pca_plot import (
    _build_color_map,
    _get_group_labels,
)
from mito_marker.analysis.selection import _get_subject_column

# ──────────────────────────────────────────────────────────────────────────────
# Module-level constants
# ──────────────────────────────────────────────────────────────────────────────

# .uns key where the frozen reference transforms and AllMitoMean are stored.
_AMHI_REFERENCE_KEY = "amhi_reference"

# .uns key where per-group AMHI results (AMHI_offset, AMHI_D_*) are stored.
_AMHI_RESULTS_KEY = "amhi_results"

# .uns key where per-group MHI_D_absolute results are stored.
_AMHI_MHID_RESULTS_KEY = "amhi_results_mhid"

# Variance threshold used by 'auto' PCA dimension selection.
_PCA_VARIANCE_TARGET = 0.95

# Groups with fewer mitochondria than this are skipped.
_MIN_MITOS_PER_GROUP = 5


# ──────────────────────────────────────────────────────────────────────────────
# Private helpers
# ──────────────────────────────────────────────────────────────────────────────


def _get_raw_feature_matrix(
    anndata_object: anndata.AnnData,
) -> Tuple[np.ndarray, List[str]]:
    """
    Extract the raw feature matrix from .X, excluding non-analytical channels.

    AMHI always reads .X (raw measurements) — never a pre-normalised layer —
    because it applies its own StandardScaler internally.  Passing already
    normalised data would result in double normalisation and distort the
    reference space.

    If a normalised layer is active in .uns['analysis_config'], a console
    message informs the user that it is intentionally ignored.

    Arguments:
        anndata_object: AnnData to extract features from.

    Returns:
        Tuple of (feature_matrix, feature_names).
        feature_matrix: float32 array, shape (n_obs, n_analytical_features).
        feature_names: list of variable names matching the matrix columns.
    """
    # Detect whether a pre-normalised layer is active and warn the user.
    active_layer = (
        anndata_object.uns
        .get("analysis_config", {})
        .get("active_layer", None)
    )
    if active_layer is not None:
        print(
            f"[AMHI] INFO: Active layer '{active_layer}' is set in "
            ".uns['analysis_config'] but AMHI always reads raw .X to avoid "
            "double normalisation. The layer is intentionally ignored. "
            "AMHI applies its own StandardScaler — no pre-normalisation needed."
        )
    else:
        print("[AMHI] Reading raw .X.")

    # Identify analytical channels (exclude spatial coords and similar).
    if "is_non_analytical" in anndata_object.var.columns:
        analytical_mask = (
            ~anndata_object.var["is_non_analytical"].fillna(False).values
        )
    else:
        # Fallback for AnnData files that pre-date the is_non_analytical flag.
        from mito_marker.controlled_vocabulary import SFC_NON_ANALYTICAL_CHANNELS
        analytical_mask = np.array(
            [
                name not in SFC_NON_ANALYTICAL_CHANNELS
                for name in anndata_object.var_names
            ],
            dtype=bool,
        )

    excluded_names = anndata_object.var_names[~analytical_mask].tolist()
    if excluded_names:
        print(f"[AMHI] Non-analytical channels excluded from feature matrix: {excluded_names}")

    feature_names: List[str] = anndata_object.var_names[analytical_mask].tolist()

    raw_x = anndata_object.X
    if hasattr(raw_x, "toarray"):
        raw_x = raw_x.toarray()
    feature_matrix = np.asarray(raw_x, dtype=np.float32)[:, analytical_mask]

    return feature_matrix, feature_names


def _apply_obs_filter(
    anndata_object: anndata.AnnData,
    obs_filter: Optional[Dict[str, Union[str, List[str]]]],
) -> anndata.AnnData:
    """
    Return a view of anndata_object restricted to rows matching obs_filter.

    obs_filter is a dict mapping .obs column names to allowed values.
    Multiple keys are AND-ed.  A single string value is treated identically
    to a one-element list.

    Examples:
        {"specie": "Human"}                     → only Human rows
        {"specie": ["Human", "Mouse"]}          → Human OR Mouse rows
        {"condition": "Young", "specie": "Droso"} → Young AND Droso rows

    Arguments:
        anndata_object: AnnData to filter.
        obs_filter: Filter dict, or None (returns original unmodified).

    Returns:
        AnnData view (or the original object if obs_filter is None).

    Raises:
        ValueError: If a column name in obs_filter is not in .obs.
    """
    if obs_filter is None:
        return anndata_object

    mask = np.ones(anndata_object.n_obs, dtype=bool)
    for column_name, allowed_values in obs_filter.items():
        if column_name not in anndata_object.obs.columns:
            raise ValueError(
                f"obs_filter key '{column_name}' not found in .obs. "
                f"Available columns: {sorted(anndata_object.obs.columns.tolist())}"
            )
        if isinstance(allowed_values, str):
            allowed_values = [allowed_values]
        column_mask = anndata_object.obs[column_name].isin(allowed_values).values
        mask &= column_mask

    n_selected = mask.sum()
    print(
        f"[AMHI] obs_filter applied: {n_selected} / {anndata_object.n_obs} "
        f"mitochondria selected."
    )
    return anndata_object[mask]


def _exclude_reference_rows(
    anndata_object: anndata.AnnData,
    reference_label: str,
) -> anndata.AnnData:
    """
    Return a view of anndata_object with AllMitoMean synthetic rows removed.

    Checks both 'unique_subject_ID' and 'subject_ID' columns for the
    reference_label.  If neither column is present the original object is
    returned unchanged.

    Arguments:
        anndata_object: AnnData that may contain AllMitoMean synthetic row(s).
        reference_label: Label used when creating the synthetic row.

    Returns:
        AnnData view excluding any synthetic reference rows.
    """
    mask = np.ones(anndata_object.n_obs, dtype=bool)

    for col in ("unique_subject_ID", "subject_ID"):
        if col in anndata_object.obs.columns:
            mask &= anndata_object.obs[col].astype(str) != reference_label

    n_excluded = (~mask).sum()
    if n_excluded > 0:
        print(
            f"[AMHI] Excluded {n_excluded} synthetic '{reference_label}' row(s) "
            f"from computation."
        )
    return anndata_object[mask]


def _load_reference_transforms(
    anndata_object: anndata.AnnData,
    reference_label: str,
) -> Tuple[StandardScaler, SklearnPCA, np.ndarray, List[str]]:
    """
    Load the frozen StandardScaler, PCA, mean vector, and feature names from
    .uns['amhi_reference'].

    Arguments:
        anndata_object: AnnData containing .uns['amhi_reference'].
        reference_label: Reference label used when calling compute_all_mito_mean().

    Returns:
        Tuple of (scaler, pca, mean_vector, feature_names).

    Raises:
        ValueError: If compute_all_mito_mean() has not been called yet, or if the
            stored reference_label does not match the requested one.
    """
    if _AMHI_REFERENCE_KEY not in anndata_object.uns:
        raise ValueError(
            f"No AMHI reference found in .uns['{_AMHI_REFERENCE_KEY}']. "
            "Run compute_all_mito_mean() first to freeze the reference transforms."
        )

    reference_dict = anndata_object.uns[_AMHI_REFERENCE_KEY]
    stored_label = reference_dict.get("reference_label", "")
    if stored_label != reference_label:
        raise ValueError(
            f"Stored reference label '{stored_label}' does not match "
            f"requested label '{reference_label}'. "
            "Re-run compute_all_mito_mean() with the correct reference_label."
        )

    scaler: StandardScaler = pickle.loads(reference_dict["scaler_bytes"])
    pca: SklearnPCA = pickle.loads(reference_dict["pca_bytes"])
    mean_vector: np.ndarray = reference_dict["mean_vector"]
    feature_names: List[str] = list(reference_dict["feature_names"])

    return scaler, pca, mean_vector, feature_names


def _project_into_reference_space(
    feature_matrix: np.ndarray,
    scaler: StandardScaler,
    pca: SklearnPCA,
) -> np.ndarray:
    """
    Project a raw feature matrix into the frozen reference PCA space.

    Uses transform() only — the scaler and PCA are never refitted.  This
    guarantees that the coordinate system is identical for every call,
    regardless of which or how many subjects are present.

    Arguments:
        feature_matrix: Raw features, shape (n_obs, n_features), float32.
        scaler: Fitted StandardScaler from compute_all_mito_mean().
        pca: Fitted PCA from compute_all_mito_mean().

    Returns:
        Projected matrix of shape (n_obs, n_components), float32.
    """
    standardized = scaler.transform(feature_matrix.astype(np.float64)).astype(np.float32)
    projected = pca.transform(standardized).astype(np.float32)
    return projected


def _auto_select_pca_components(
    pca: SklearnPCA,
    n_pca_components: Union[str, int, None],
) -> int:
    """
    Determine how many PCA components to use from the frozen PCA object.

    Arguments:
        pca: Fitted PCA object.
        n_pca_components: 'auto' (95% variance), None (all components), or int.

    Returns:
        Number of components k to use.
    """
    n_stored = pca.n_components_

    if n_pca_components is None:
        return n_stored

    if n_pca_components == "auto":
        cumulative = np.cumsum(pca.explained_variance_ratio_)
        indices = np.where(cumulative >= _PCA_VARIANCE_TARGET)[0]
        if len(indices) == 0:
            warnings.warn(
                f"[AMHI] PCA explains only {cumulative[-1]:.1%} variance with all "
                f"{n_stored} components. Using all.",
                stacklevel=4,
            )
            return n_stored
        return int(indices[0]) + 1  # +1 because index is 0-based

    # Explicit integer
    k = min(int(n_pca_components), n_stored)
    if k < int(n_pca_components):
        warnings.warn(
            f"[AMHI] Requested {n_pca_components} PCA components but only "
            f"{n_stored} stored. Using {k}.",
            stacklevel=4,
        )
    return k


def _resolve_group_column(
    anndata_object: anndata.AnnData,
    group_by: Optional[str],
) -> str:
    """
    Resolve the grouping column, with auto-detection when group_by is None.

    Arguments:
        anndata_object: AnnData whose .obs is inspected.
        group_by: Explicit column name, or None for auto-detection.

    Returns:
        Column name string.

    Raises:
        ValueError: If the column is not present in .obs.
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
            "Expected 'unique_subject_ID' (TEM) or 'subject_ID' (SFC). "
            "Pass group_by='your_column' explicitly."
        )
    return subject_column


def _print_pca_loadings(
    pca: SklearnPCA,
    feature_names: List[str],
    top_n: int = 10,
) -> None:
    """
    Print the top contributing features for PC1 and PC2.

    For each principal component, display each feature's raw loading value and
    its relative contribution (|loading| / sum(|loadings|) × 100).  A high
    contribution means that feature strongly drives the variance captured by
    that axis.

    Arguments:
        pca: Fitted sklearn PCA object.
        feature_names: Ordered list of feature names matching pca.components_ columns.
        top_n: Number of top features to display per component (default 10).
    """
    loadings_matrix = pca.components_.T  # shape (n_features, n_components)
    n_pcs_to_show = min(2, pca.n_components_)
    for pc_idx in range(n_pcs_to_show):
        pc_loadings = loadings_matrix[:, pc_idx]
        var_pct = float(pca.explained_variance_ratio_[pc_idx]) * 100
        abs_loadings = np.abs(pc_loadings)
        contributions = abs_loadings / abs_loadings.sum() * 100
        sorted_idx = np.argsort(contributions)[::-1][:top_n]
        print(
            f"\n[AMHI] PC{pc_idx + 1} ({var_pct:.1f}% variance) "
            f"— top {top_n} features by contribution:"
        )
        print(f"  {'Rank':<5} {'Feature':<35} {'Loading':>9} {'Contribution':>14}")
        print(f"  {'-' * 65}")
        for rank, feat_idx in enumerate(sorted_idx):
            name = feature_names[feat_idx]
            loading = float(pc_loadings[feat_idx])
            contribution = float(contributions[feat_idx])
            print(
                f"  {rank + 1:<5} {name:<35} {loading:>+9.4f} {contribution:>13.1f}%"
            )


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────


def compute_all_mito_mean(
    anndata_object: anndata.AnnData,
    obs_filter: Optional[Dict[str, Union[str, List[str]]]] = None,
    n_pca_components: Union[str, int, None] = "auto",
    add_to_anndata: bool = True,
    reference_label: str = "AllMitoMean",
) -> Tuple[anndata.AnnData, np.ndarray]:
    """
    One-time setup: fit and freeze the absolute reference transform pipeline.

    This function defines the coordinate system used by all AMHI metrics.
    Once called, the StandardScaler and PCA are frozen in .uns['amhi_reference']
    and never refitted — any future subject is projected into this fixed space.

    The AllMitoMean is the mean feature vector of the reference population.
    In PCA space it projects exactly to the origin (0, 0, …, 0).

    obs_filter restricts which mitos define the reference (AND-ed across keys):
        None                                      all subjects in the AnnData
        {"specie": "Human"}                       only Human mitos
        {"specie": ["Human", "Mouse"]}            only mammals
        {"condition": "Young", "specie": "Droso"} Young Drosophila only

    The filter affects ONLY the reference fitting.  All subjects remain in
    the returned AnnData and can be measured against this reference.

    Arguments:
        anndata_object: AnnData to fit the reference on (TEM or SFC pipeline).
        obs_filter: Optional dict to restrict the reference population.
        n_pca_components: 'auto' (95% variance, default), None (all components),
            or positive int (exact number of components).
        add_to_anndata: If True, inject AllMitoMean as a synthetic obs row
            (unique_subject_ID = reference_label) using anndata.concat().
            This allows plot_radar() and other group-based functions to include
            AllMitoMean automatically when grouping by subject ID.
        reference_label: Name used for the synthetic obs row and as the
            lookup key for subsequent compute_amhi() calls.

    Returns:
        Tuple of (updated_anndata, raw_mean_vector) where:
          - updated_anndata contains the frozen transforms in .uns and,
            if add_to_anndata=True, an appended AllMitoMean obs row.
          - raw_mean_vector is the mean feature vector (n_features,) in the
            original (non-standardized) feature space, useful for radar plots.

    Side effects:
        Stores in .uns['amhi_reference']:
            scaler_bytes      — pickled StandardScaler (bytes)
            pca_bytes         — pickled PCA object (bytes)
            mean_vector       — np.ndarray (n_features,) raw mean
            feature_names     — list[str] feature column names
            n_obs_used        — int number of reference mitos
            species_included  — list[str] distinct species in reference
            obs_filter        — the filter dict (or None)
            reference_label   — the reference_label string
            n_pca_components  — the n_pca_components parameter used
    """
    print(f"\n{'='*60}")
    print(f"[AMHI] compute_all_mito_mean(reference_label='{reference_label}')")
    print(f"{'='*60}")

    # ── Step 1: exclude any pre-existing AllMitoMean row ──────────────────
    working_anndata = _exclude_reference_rows(anndata_object, reference_label)

    # ── Step 2: apply obs_filter to select the reference population ────────
    if obs_filter is not None:
        print(f"[AMHI] Applying obs_filter: {obs_filter}")
    reference_anndata = _apply_obs_filter(working_anndata, obs_filter)

    n_obs_used = reference_anndata.n_obs
    print(f"[AMHI] Reference population: {n_obs_used} mitochondria")

    if n_obs_used < 10:
        raise ValueError(
            f"[AMHI] Reference population has only {n_obs_used} mitochondria. "
            "Need at least 10 to fit a reliable StandardScaler and PCA."
        )

    # ── Step 3: extract the raw feature matrix ────────────────────────────
    raw_matrix, feature_names = _get_raw_feature_matrix(reference_anndata)
    raw_matrix = raw_matrix.astype(np.float64)
    print(f"[AMHI] Feature matrix: {raw_matrix.shape[0]} mitos × {raw_matrix.shape[1]} features")
    print(f"[AMHI] Features: {feature_names}")

    # ── Step 4: compute AllMitoMean in raw feature space ─────────────────
    mean_vector = raw_matrix.mean(axis=0).astype(np.float32)
    print(f"[AMHI] AllMitoMean computed (raw space). Shape: {mean_vector.shape}")

    # ── Step 5: fit StandardScaler on reference population ────────────────
    scaler = StandardScaler()
    scaler.fit(raw_matrix)
    standardized_matrix = scaler.transform(raw_matrix).astype(np.float32)
    print(
        f"[AMHI] StandardScaler fitted. "
        f"Mean range: [{scaler.mean_.min():.3f}, {scaler.mean_.max():.3f}]"
    )

    # ── Step 6: determine n_components then fit PCA ────────────────────────
    # First fit PCA with max components to determine auto threshold.
    max_components = min(raw_matrix.shape[0] - 1, raw_matrix.shape[1])
    pca_full = SklearnPCA(n_components=max_components, random_state=42)
    pca_full.fit(standardized_matrix)

    k = _auto_select_pca_components(pca_full, n_pca_components)

    # Refit PCA with the chosen number of components (cleaner object to store).
    pca = SklearnPCA(n_components=k, random_state=42)
    pca.fit(standardized_matrix)

    total_variance = pca.explained_variance_ratio_.sum()
    print(
        f"[AMHI] PCA fitted: {k} components, "
        f"{total_variance:.1%} variance explained."
    )
    _print_pca_loadings(pca, feature_names)

    # Verify AllMitoMean projects to origin (sanity check).
    mean_projected = pca.transform(scaler.transform(mean_vector.reshape(1, -1).astype(np.float64)))
    max_offset = np.abs(mean_projected).max()
    if max_offset > 1e-4:
        warnings.warn(
            f"[AMHI] AllMitoMean projection deviates from origin by {max_offset:.2e}. "
            "This is unexpected — check for NaN values in the feature matrix.",
            stacklevel=2,
        )
    else:
        print(f"[AMHI] AllMitoMean projects to origin. Max deviation: {max_offset:.2e} ✓")

    # ── Step 7: collect species_included ──────────────────────────────────
    species_included: List[str] = []
    for specie_col in ("specie", "species"):
        if specie_col in reference_anndata.obs.columns:
            species_included = sorted(
                reference_anndata.obs[specie_col].dropna().astype(str).unique().tolist()
            )
            break
    print(f"[AMHI] Species in reference: {species_included}")

    # ── Step 8: store everything in .uns ──────────────────────────────────
    anndata_object.uns[_AMHI_REFERENCE_KEY] = {
        "scaler_bytes": pickle.dumps(scaler),
        "pca_bytes": pickle.dumps(pca),
        "mean_vector": mean_vector,
        "feature_names": list(feature_names),
        "n_obs_used": int(n_obs_used),
        "species_included": species_included,
        "obs_filter": obs_filter,
        "reference_label": reference_label,
        "n_pca_components": n_pca_components,
    }
    print(f"[AMHI] Reference transforms stored in .uns['{_AMHI_REFERENCE_KEY}'].")

    # ── Step 9: inject synthetic AllMitoMean obs row ──────────────────────
    if add_to_anndata:
        print(f"[AMHI] Injecting '{reference_label}' as a synthetic obs row...")

        # Build a 1-row AnnData with .X = raw mean vector.
        n_vars = anndata_object.n_vars
        mean_x = np.zeros((1, n_vars), dtype=np.float32)
        # Fill only the analytical feature columns used for computing the mean.
        for feature_index, feature_name in enumerate(feature_names):
            if feature_name in anndata_object.var_names:
                var_position = anndata_object.var_names.get_loc(feature_name)
                mean_x[0, var_position] = mean_vector[feature_index]

        # For each normalisation layer that exists in the AnnData, compute the
        # mean of the reference population's values in that layer and store it
        # for the synthetic row.  Without this, plot_radar() would get NaN for
        # AllMitoMean in every layer and silently fall back to raw .X, which can
        # have wildly different scales (e.g. pixel-unit TEM features, raw SFC
        # intensities) and breaks the radar plot axis.
        synthetic_layers: Dict[str, np.ndarray] = {}
        for layer_key in working_anndata.layers:
            # anndata >= 0.13 exposes .X as layers[None]; it is not a
            # normalisation layer and must not be copied as one.
            if layer_key is None or layer_key not in reference_anndata.layers:
                continue
            ref_layer = reference_anndata.layers[layer_key]
            if hasattr(ref_layer, "toarray"):
                ref_layer = ref_layer.toarray()
            ref_layer_array = np.asarray(ref_layer, dtype=np.float64)
            layer_mean_row = np.nanmean(ref_layer_array, axis=0).astype(np.float32).reshape(1, -1)
            synthetic_layers[layer_key] = layer_mean_row
        if synthetic_layers:
            print(
                f"[AMHI] AllMitoMean layer values computed for "
                f"{len(synthetic_layers)} layer(s): {list(synthetic_layers.keys())}"
            )

        # Build .obs row — fill all columns with NaN / reference_label.
        synthetic_obs: Dict = {}
        for col in anndata_object.obs.columns:
            dtype = anndata_object.obs[col].dtype
            if col in ("unique_subject_ID", "subject_ID", "specie", "condition",
                       "source_filename", "Image_Name"):
                synthetic_obs[col] = [reference_label]
            elif hasattr(dtype, "categories"):
                # Categorical: use reference_label as a new category string.
                synthetic_obs[col] = [reference_label]
            else:
                # Numeric columns: NaN.
                synthetic_obs[col] = [np.nan]

        synthetic_obs_df = pd.DataFrame(
            synthetic_obs,
            index=[reference_label],
        )

        synthetic_adata = anndata.AnnData(
            X=mean_x,
            obs=synthetic_obs_df,
            var=anndata_object.var.copy(),
            layers=synthetic_layers if synthetic_layers else None,
        )

        updated_anndata = anndata.concat(
            [anndata_object, synthetic_adata],
            merge="same",
        )
        # Carry over .uns (anndata.concat does not preserve .uns by default).
        updated_anndata.uns = dict(anndata_object.uns)

        print(
            f"[AMHI] AllMitoMean row injected. "
            f"AnnData now has {updated_anndata.n_obs} observations "
            f"({anndata_object.n_obs} original + 1 synthetic)."
        )
    else:
        updated_anndata = anndata_object

    print("\n[AMHI] QC summary:")
    print(f"  Reference label   : {reference_label}")
    print(f"  Mitos in reference: {n_obs_used}")
    print(f"  PCA components    : {k} ({total_variance:.1%} variance)")
    print(f"  Species           : {species_included}")
    print(f"  obs_filter        : {obs_filter}")
    print(f"{'='*60}\n")

    return updated_anndata, mean_vector


def compute_amhi(
    anndata_object: anndata.AnnData,
    group_by: Optional[str] = None,
    obs_filter: Optional[Dict[str, Union[str, List[str]]]] = None,
    reference_label: str = "AllMitoMean",
) -> pd.DataFrame:
    """
    Compute AMHI_offset and AMHI_D for each group using the frozen reference space.

    Works on any subjects — whether they were part of the reference population
    or not.  The frozen StandardScaler and PCA are loaded from .uns and applied
    with transform() only (never refitted), so values are invariant to which
    other subjects are present in the AnnData.

    AMHI_D (per-mito distance to AllMitoMean):
        For each mitochondrion: ‖z_i‖₂ in the fixed PCA space (AllMitoMean = origin).
        Combines the subject's offset from AllMitoMean AND its internal spread.

        AMHI_D is the recommended single-scalar composite heterogeneity index.
        It integrates both the population-level positional shift (AMHI_offset) and
        the within-group dispersion (MHI_D_absolute) in one geometrically principled
        Euclidean norm:

            AMHI_D² ≈ AMHI_offset² + MHI_D_absolute²

        This Pythagorean decomposition mirrors total/explained/residual variance in
        ANOVA and is inherited by construction from the frozen reference space,
        making AMHI_D directly comparable across experiments without refitting.

    AMHI_offset (centroid distance to AllMitoMean):
        ‖centroid_group − origin‖₂.
        Measures how far the subject's typical mito sits from AllMitoMean,
        independent of internal spread.

    obs_filter restricts WHICH subjects are measured (independent of which
    subjects defined the reference):
        None                    measure all subjects in the AnnData
        {"specie": "Human"}     measure only humans
        {"condition": "Old"}    measure only old subjects

    Arguments:
        anndata_object: AnnData with frozen transforms in .uns['amhi_reference'].
        group_by: .obs column to group by (auto-detects unique_subject_ID / subject_ID).
        obs_filter: Optional dict to restrict which mitos are measured.
        reference_label: Must match the label used in compute_all_mito_mean().

    Returns:
        DataFrame indexed by group value with columns:
            n_mitos         — number of mitochondria in the group
            AMHI_offset     — distance from group centroid to AllMitoMean
            AMHI_D_median   — median per-mito distance to AllMitoMean (recommended composite scalar)
            AMHI_D_mean     — mean per-mito distance to AllMitoMean
            AMHI_D_std      — standard deviation of per-mito distances
            AMHI_D_q25      — 25th percentile of per-mito distances
            AMHI_D_q75      — 75th percentile of per-mito distances

    Side effects:
        Stores result in .uns['amhi_results'][group_column].
    """
    print(f"\n[AMHI] compute_amhi(group_by='{group_by}', reference='{reference_label}')")

    # ── Load frozen transforms ─────────────────────────────────────────────
    scaler, pca, _mean_vector, feature_names = _load_reference_transforms(
        anndata_object, reference_label
    )

    # ── Prepare data: exclude synthetic row, apply obs_filter ─────────────
    working_anndata = _exclude_reference_rows(anndata_object, reference_label)
    working_anndata = _apply_obs_filter(working_anndata, obs_filter)

    group_column = _resolve_group_column(working_anndata, group_by)
    print(f"[AMHI] Grouping by: '{group_column}'")

    # ── Extract and align feature matrix ──────────────────────────────────
    # We need only the features that were used to fit the reference.
    # Subset .var to those feature names in the same order.
    available_features = [f for f in feature_names if f in working_anndata.var_names]
    if len(available_features) != len(feature_names):
        missing = set(feature_names) - set(available_features)
        raise ValueError(
            f"[AMHI] {len(missing)} reference features not found in AnnData.var: {missing}. "
            "Make sure this AnnData was produced by the same pipeline as the reference."
        )

    # Get the full data matrix then select reference features in order.
    full_matrix, all_channel_names = _get_raw_feature_matrix(working_anndata)
    feature_indices = [all_channel_names.index(f) for f in feature_names]
    raw_matrix = full_matrix[:, feature_indices].astype(np.float64)

    # ── Project into fixed reference space ────────────────────────────────
    projected_matrix = _project_into_reference_space(raw_matrix, scaler, pca)
    # AllMitoMean is at origin by construction — distances = norms.
    per_mito_distances = np.linalg.norm(projected_matrix, axis=1).astype(np.float32)

    group_labels = working_anndata.obs[group_column].astype(str)
    group_obs_array = group_labels.values

    # ── Compute per-group statistics ──────────────────────────────────────
    results_rows = []
    for group_value in sort_values_for_legend(group_labels.unique()):
        mask = group_obs_array == group_value
        n_mitos = int(mask.sum())

        if n_mitos < _MIN_MITOS_PER_GROUP:
            warnings.warn(
                f"[AMHI] Group '{group_value}' has only {n_mitos} mitochondria "
                f"(minimum: {_MIN_MITOS_PER_GROUP}). Skipping.",
                stacklevel=2,
            )
            continue

        group_projected = projected_matrix[mask]
        group_distances = per_mito_distances[mask]

        centroid = group_projected.mean(axis=0)
        amhi_offset = float(np.linalg.norm(centroid))

        results_rows.append(
            {
                "n_mitos": n_mitos,
                "AMHI_offset": amhi_offset,
                "AMHI_D_median": float(np.median(group_distances)),
                "AMHI_D_mean": float(np.mean(group_distances)),
                "AMHI_D_std": float(np.std(group_distances)),
                "AMHI_D_q25": float(np.percentile(group_distances, 25)),
                "AMHI_D_q75": float(np.percentile(group_distances, 75)),
            }
        )

    results_dataframe = pd.DataFrame(
        results_rows,
        index=[
            gv for gv in sort_values_for_legend(group_labels.unique())
            if (group_obs_array == gv).sum() >= _MIN_MITOS_PER_GROUP
        ],
    )
    results_dataframe.index.name = group_column

    # ── Store and print ────────────────────────────────────────────────────
    if _AMHI_RESULTS_KEY not in anndata_object.uns:
        anndata_object.uns[_AMHI_RESULTS_KEY] = {}
    anndata_object.uns[_AMHI_RESULTS_KEY][group_column] = results_dataframe

    print(
        f"\n[AMHI] Results for {len(results_dataframe)} groups "
        f"(column: '{group_column}'):"
    )
    print(results_dataframe.to_string())
    print()

    return results_dataframe


def compute_mhi_d_absolute(
    anndata_object: anndata.AnnData,
    group_by: Optional[str] = None,
    obs_filter: Optional[Dict[str, Union[str, List[str]]]] = None,
    reference_label: str = "AllMitoMean",
) -> pd.DataFrame:
    """
    Compute MHI-D in the fixed reference space — absolute internal heterogeneity.

    This is the absolute version of the existing compute_mhi_d(): same formula
    (median distance of each mito to its own group's centroid), but computed in
    the frozen reference PCA space instead of a freshly fitted one.

    Because the transform pipeline never changes, values are on the same absolute
    scale for any set of subjects.  Subject A computed today with 6 subjects
    gives exactly the same MHI_D_absolute as when computed again tomorrow with 7.

    Comparing two subjects on internal heterogeneity is now meaningful regardless
    of their AMHI_offset: Subject A (tight cluster far from AllMitoMean) has a
    lower MHI_D_absolute than Subject B (dispersed cluster near AllMitoMean).

    Arguments:
        anndata_object: AnnData with frozen transforms in .uns['amhi_reference'].
        group_by: .obs column to group by (auto-detects unique_subject_ID / subject_ID).
        obs_filter: Optional dict to restrict which mitos are measured.
        reference_label: Must match the label used in compute_all_mito_mean().

    Returns:
        DataFrame indexed by group value with columns:
            n_mitos                 — number of mitochondria in the group
            MHI_D_absolute_median   — median per-mito distance to group centroid
            MHI_D_absolute_mean     — mean per-mito distance to group centroid
            MHI_D_absolute_std      — standard deviation
            MHI_D_absolute_q25      — 25th percentile
            MHI_D_absolute_q75      — 75th percentile

    Side effects:
        Stores result in .uns['amhi_results_mhid'][group_column].
    """
    print(
        f"\n[AMHI] compute_mhi_d_absolute(group_by='{group_by}', "
        f"reference='{reference_label}')"
    )

    # ── Load frozen transforms ─────────────────────────────────────────────
    scaler, pca, _mean_vector, feature_names = _load_reference_transforms(
        anndata_object, reference_label
    )

    # ── Prepare data ───────────────────────────────────────────────────────
    working_anndata = _exclude_reference_rows(anndata_object, reference_label)
    working_anndata = _apply_obs_filter(working_anndata, obs_filter)

    group_column = _resolve_group_column(working_anndata, group_by)
    print(f"[AMHI] Grouping by: '{group_column}'")

    # ── Extract feature matrix aligned to reference features ──────────────
    full_matrix, all_channel_names = _get_raw_feature_matrix(working_anndata)
    available_features = [f for f in feature_names if f in working_anndata.var_names]
    if len(available_features) != len(feature_names):
        missing = set(feature_names) - set(available_features)
        raise ValueError(
            f"[AMHI] {len(missing)} reference features not found in AnnData.var: {missing}."
        )
    feature_indices = [all_channel_names.index(f) for f in feature_names]
    raw_matrix = full_matrix[:, feature_indices].astype(np.float64)

    # ── Project into fixed reference space ────────────────────────────────
    projected_matrix = _project_into_reference_space(raw_matrix, scaler, pca)

    group_labels = working_anndata.obs[group_column].astype(str)
    group_obs_array = group_labels.values

    # ── Compute per-group MHI-D in fixed space ────────────────────────────
    results_rows = []
    valid_group_values = []
    for group_value in sort_values_for_legend(group_labels.unique()):
        mask = group_obs_array == group_value
        n_mitos = int(mask.sum())

        if n_mitos < _MIN_MITOS_PER_GROUP:
            warnings.warn(
                f"[AMHI] Group '{group_value}' has only {n_mitos} mitochondria. Skipping.",
                stacklevel=2,
            )
            continue

        group_projected = projected_matrix[mask]
        centroid = group_projected.mean(axis=0)
        distances_to_centroid = np.linalg.norm(
            group_projected - centroid, axis=1
        ).astype(np.float32)

        results_rows.append(
            {
                "n_mitos": n_mitos,
                "MHI_D_absolute_median": float(np.median(distances_to_centroid)),
                "MHI_D_absolute_mean": float(np.mean(distances_to_centroid)),
                "MHI_D_absolute_std": float(np.std(distances_to_centroid)),
                "MHI_D_absolute_q25": float(np.percentile(distances_to_centroid, 25)),
                "MHI_D_absolute_q75": float(np.percentile(distances_to_centroid, 75)),
            }
        )
        valid_group_values.append(group_value)

    results_dataframe = pd.DataFrame(results_rows, index=valid_group_values)
    results_dataframe.index.name = group_column

    # ── Store and print ────────────────────────────────────────────────────
    if _AMHI_MHID_RESULTS_KEY not in anndata_object.uns:
        anndata_object.uns[_AMHI_MHID_RESULTS_KEY] = {}
    anndata_object.uns[_AMHI_MHID_RESULTS_KEY][group_column] = results_dataframe

    print(
        f"\n[AMHI] MHI_D_absolute results for {len(results_dataframe)} groups "
        f"(column: '{group_column}'):"
    )
    print(results_dataframe.to_string())
    print()

    return results_dataframe


def plot_amhi_distances(
    anndata_object: anndata.AnnData,
    group_by: Optional[str] = None,
    obs_filter: Optional[Dict[str, Union[str, List[str]]]] = None,
    reference_label: str = "AllMitoMean",
    amhi_results: Optional[pd.DataFrame] = None,
    title: str = "",
) -> None:
    """
    Violin + box plot of per-mito AMHI_D distributions, with AMHI_offset overlaid.

    For each group shows:
      - Violin: per-mito distance distribution to AllMitoMean (AMHI_D)
      - Box: quartiles of the same distribution
      - Red dot: AMHI_offset (distance of the group centroid to AllMitoMean)
      - Console: summary table (AMHI_offset, AMHI_D_median, n_mitos per group)

    The difference between the red dot (AMHI_offset) and the violin centre
    reveals how much of the total distance is due to internal spread vs offset.

    Arguments:
        anndata_object: AnnData with frozen transforms in .uns['amhi_reference'].
        group_by: .obs column to group by (auto-detects if None).
        obs_filter: Optional dict to restrict which mitos are measured.
        reference_label: Must match the label used in compute_all_mito_mean().
        amhi_results: Pre-computed DataFrame from compute_amhi() — if None,
            compute_amhi() is called automatically.
        title: Optional plot title prefix.
    """
    # ── Load frozen transforms ─────────────────────────────────────────────
    scaler, pca, _mean_vector, feature_names = _load_reference_transforms(
        anndata_object, reference_label
    )

    # ── Prepare data ───────────────────────────────────────────────────────
    working_anndata = _exclude_reference_rows(anndata_object, reference_label)
    working_anndata = _apply_obs_filter(working_anndata, obs_filter)
    group_column = _resolve_group_column(working_anndata, group_by)

    # ── Get or compute AMHI results ────────────────────────────────────────
    if amhi_results is None:
        amhi_results = compute_amhi(
            anndata_object,
            group_by=group_column,
            obs_filter=obs_filter,
            reference_label=reference_label,
        )

    # ── Project all mitos into fixed space ────────────────────────────────
    full_matrix, all_channel_names = _get_raw_feature_matrix(working_anndata)
    feature_indices = [all_channel_names.index(f) for f in feature_names]
    raw_matrix = full_matrix[:, feature_indices].astype(np.float64)
    projected_matrix = _project_into_reference_space(raw_matrix, scaler, pca)
    per_mito_distances = np.linalg.norm(projected_matrix, axis=1).astype(np.float32)

    group_labels = working_anndata.obs[group_column].astype(str)
    valid_groups = amhi_results.index.tolist()

    # ── Build long-format DataFrame for seaborn ───────────────────────────
    plot_rows = []
    for group_value in valid_groups:
        mask = group_labels.values == group_value
        for dist in per_mito_distances[mask]:
            plot_rows.append({"group": group_value, "AMHI_D": float(dist)})
    plot_dataframe = pd.DataFrame(plot_rows)

    # ── Draw plot ─────────────────────────────────────────────────────────
    color_map = _build_color_map(anndata_object, group_column, valid_groups)
    palette = {g: color_map.get(g, "#999999") for g in valid_groups}

    fig, axes = plt.subplots(figsize=(max(8, len(valid_groups) * 1.4), 6))

    sns.violinplot(
        data=plot_dataframe,
        x="group",
        y="AMHI_D",
        hue="group",
        palette=palette,
        inner="box",
        order=valid_groups,
        ax=axes,
        linewidth=0.8,
        legend=False,
    )

    # Overlay AMHI_offset as red dots.
    for x_position, group_value in enumerate(valid_groups):
        if group_value in amhi_results.index:
            offset = amhi_results.loc[group_value, "AMHI_offset"]
            axes.plot(
                x_position,
                offset,
                marker="o",
                color="crimson",
                markersize=8,
                zorder=5,
                label="AMHI_offset" if x_position == 0 else None,
            )

    axes.set_xlabel(group_column, fontsize=12)
    axes.set_ylabel("Distance to AllMitoMean (fixed PCA space)", fontsize=12)
    axes.set_xticks(range(len(valid_groups)))
    axes.set_xticklabels(valid_groups, rotation=45, ha="right")

    # Clip y-axis at the 99th percentile of all per-mito distances so that rare
    # extreme outliers do not collapse the violin body into an unreadable line.
    # The violin shape is still computed from the full distribution; only the
    # visible axis range is restricted.
    y_clip = float(np.percentile(per_mito_distances, 99)) * 1.05
    axes.set_ylim(bottom=0, top=y_clip)

    plot_title = title if title else f"AMHI Distances — {group_column}"
    axes.set_title(plot_title, fontsize=14, fontweight="bold")

    # Legend for AMHI_offset dot.
    handles, labels = axes.get_legend_handles_labels()
    if handles:
        axes.legend(handles, labels, loc="upper right", fontsize=10)

    plt.tight_layout()
    plt.show()

    print(f"  [Note] y-axis clipped at 99th percentile ({y_clip:.2f}) for readability.")

    # ── Console summary ────────────────────────────────────────────────────
    print("\n[AMHI] Distance summary:")
    print(f"  {'Group':<25} {'n_mitos':>8} {'AMHI_offset':>12} {'AMHI_D_median':>14}")
    print(f"  {'-'*25} {'-'*8} {'-'*12} {'-'*14}")
    for group_value in valid_groups:
        if group_value in amhi_results.index:
            row = amhi_results.loc[group_value]
            print(
                f"  {group_value:<25} {int(row['n_mitos']):>8} "
                f"{row['AMHI_offset']:>12.4f} {row['AMHI_D_median']:>14.4f}"
            )
    print()


def plot_amhi_2d(
    anndata_object: anndata.AnnData,
    color_by: Union[str, List[str]],
    obs_filter: Optional[Dict[str, Union[str, List[str]]]] = None,
    reference_label: str = "AllMitoMean",
    max_points_per_group: int = 500,
    show_distance_rings: bool = True,
    title: str = "",
) -> None:
    """
    2D scatter in the fixed reference PCA space, with AllMitoMean at (0, 0).

    Each dot is one mitochondrion.  Its radial distance from center equals its
    individual AMHI_D contribution.  The position of a subject's cluster center
    relative to the origin is its AMHI_offset.  The spread of the cluster is its
    MHI_D_absolute.  All three properties are visible simultaneously.

    AllMitoMean is shown as a gold star at (0, 0).

    Optional concentric rings mark 1× and 2× the median AMHI_D of the full
    population, giving a visual sense of 'normal' deviation from the reference.

    Arguments:
        anndata_object: AnnData with frozen transforms in .uns['amhi_reference'].
        color_by: .obs column name (or list of column names) to color the dots.
            Single string: color by that .obs column.
            List of strings: color by combined "col1 / col2" label.
        obs_filter: Optional dict to restrict which mitos are plotted.
        reference_label: Must match the label used in compute_all_mito_mean().
        max_points_per_group: Maximum dots per group (random subsample if exceeded).
        show_distance_rings: If True, draw concentric rings at 1× and 2× median
            AMHI_D for scale reference.
        title: Optional plot title.
    """
    # ── Load frozen transforms ─────────────────────────────────────────────
    scaler, pca, _mean_vector, feature_names = _load_reference_transforms(
        anndata_object, reference_label
    )

    # ── Prepare data ───────────────────────────────────────────────────────
    working_anndata = _exclude_reference_rows(anndata_object, reference_label)
    working_anndata = _apply_obs_filter(working_anndata, obs_filter)

    # ── Extract feature matrix aligned to reference features ──────────────
    full_matrix, all_channel_names = _get_raw_feature_matrix(working_anndata)
    feature_indices = [all_channel_names.index(f) for f in feature_names]
    raw_matrix = full_matrix[:, feature_indices].astype(np.float64)

    # ── Project into fixed reference space ────────────────────────────────
    projected_matrix = _project_into_reference_space(raw_matrix, scaler, pca)
    # AllMitoMean = origin, no translation needed.
    pca_x = projected_matrix[:, 0]
    pca_y = projected_matrix[:, 1]

    # ── Group labels for coloring ──────────────────────────────────────────
    if isinstance(color_by, str):
        if color_by not in working_anndata.obs.columns:
            raise ValueError(
                f"color_by='{color_by}' not found in .obs. "
                f"Available: {sorted(working_anndata.obs.columns.tolist())}"
            )
        group_labels = working_anndata.obs[color_by].astype(str)
        color_column = color_by
    else:
        group_labels = _get_group_labels(working_anndata, color_by)
        color_column = " / ".join(color_by)

    unique_groups = sort_values_for_legend(group_labels.unique().tolist())
    color_map = _build_color_map(anndata_object, color_by, unique_groups)

    # ── Distance rings (1× and 2× median AMHI_D of full population) ────────
    all_distances = np.linalg.norm(projected_matrix, axis=1)
    median_distance = float(np.median(all_distances))

    # ── Draw plot ─────────────────────────────────────────────────────────
    fig, axes = plt.subplots(figsize=(9, 8))

    if show_distance_rings:
        for ring_multiple, ring_style in [(1, "--"), (2, ":")]:
            radius = ring_multiple * median_distance
            circle = plt.Circle(
                (0, 0),
                radius,
                fill=False,
                color="lightgray",
                linestyle=ring_style,
                linewidth=1.2,
                zorder=1,
            )
            axes.add_patch(circle)
            axes.text(
                0,
                radius,
                f"  {ring_multiple}× median AMHI_D",
                color="lightgray",
                fontsize=8,
                va="bottom",
                ha="left",
                zorder=1,
            )

    rng = np.random.default_rng(seed=42)
    for group_value in unique_groups:
        mask = group_labels.values == group_value
        indices = np.where(mask)[0]

        if len(indices) > max_points_per_group:
            indices = rng.choice(indices, size=max_points_per_group, replace=False)

        group_color = color_map.get(group_value, "#999999")

        axes.scatter(
            pca_x[indices],
            pca_y[indices],
            color=group_color,
            label=group_value,
            alpha=0.6,
            s=15,
            linewidths=0,
            zorder=2,
        )

        # Centroid of the full group (all mitos, not just the subsample).
        centroid_x = float(pca_x[mask].mean())
        centroid_y = float(pca_y[mask].mean())
        axes.scatter(
            centroid_x,
            centroid_y,
            marker="D",
            color=group_color,
            edgecolors="black",
            s=45,
            linewidths=0.5,
            zorder=4,
        )

    # AllMitoMean at origin.
    axes.scatter(
        [0],
        [0],
        marker="*",
        color="gold",
        edgecolors="black",
        s=120,
        zorder=5,
        label=reference_label,
        linewidths=0.5,
    )

    var_pc1 = float(pca.explained_variance_ratio_[0]) * 100
    var_pc2 = float(pca.explained_variance_ratio_[1]) * 100
    axes.set_xlabel(f"PC1 ({var_pc1:.1f}% variance)", fontsize=12)
    axes.set_ylabel(f"PC2 ({var_pc2:.1f}% variance)", fontsize=12)
    axes.axhline(0, color="lightgray", linewidth=0.5, zorder=0)
    axes.axvline(0, color="lightgray", linewidth=0.5, zorder=0)

    plot_title = title if title else f"AMHI 2D — coloured by {color_column}"
    axes.set_title(plot_title, fontsize=14, fontweight="bold")
    axes.set_aspect("equal", adjustable="datalim")

    # Add a single proxy entry explaining the centroid diamonds.
    centroid_proxy = matplotlib.lines.Line2D(
        [], [],
        marker="D",
        color="w",
        markerfacecolor="gray",
        markeredgecolor="black",
        markersize=7,
        linewidth=0,
        label="Group centroid",
    )

    legend = axes.legend(
        loc="upper right",
        fontsize=9,
        framealpha=0.8,
    )
    # Add the centroid proxy as an extra entry after automatic legend creation.
    legend_handles, legend_labels = axes.get_legend_handles_labels()
    legend_handles.append(centroid_proxy)
    legend_labels.append("Group centroid")
    legend = axes.legend(
        legend_handles,
        legend_labels,
        loc="upper right",
        fontsize=9,
        framealpha=0.8,
    )

    # Reduce the AllMitoMean star size in the legend only (the plot star at
    # s=300 is intentionally large; the legend entry does not need to match).
    for legend_handle, legend_label in zip(legend.legend_handles, legend_labels):
        if legend_label == reference_label and hasattr(legend_handle, "_sizes"):
            legend_handle._sizes = [60]

    plt.tight_layout()
    plt.show()

    print(
        f"[AMHI] 2D plot: {len(unique_groups)} groups, "
        f"median AMHI_D = {median_distance:.4f}, "
        f"PCA: PC1={var_pc1:.1f}%, PC2={var_pc2:.1f}%"
    )


def summarize_amhi(
    anndata_object: anndata.AnnData,
    group_by: Optional[str] = None,
    obs_filter: Optional[Dict[str, Union[str, List[str]]]] = None,
    reference_label: str = "AllMitoMean",
    amhi_results: Optional[pd.DataFrame] = None,
    mhid_results: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Compute and print a clean summary table combining the three core AMHI metrics.

    Merges compute_amhi() and compute_mhi_d_absolute() output into one row per
    group, prints a human-readable table sorted by AMHI_offset (descending),
    and returns the merged DataFrame for use in plot_amhi_profile().

    The three metrics are orthogonal by construction (Pythagorean decomposition):

        AMHI_D_median² ≈ AMHI_offset² + MHI_D_absolute_median²

    Interpretation:
        AMHI_offset            — positional shift: how far the group's average
                                 mitochondrion sits from the pan-species mean.
                                 Captures inter-group divergence, not internal spread.

        MHI_D_absolute_median  — within-group dispersion: how spread out individual
                                 mitochondria are around their own group centroid.
                                 Captures intra-subject heterogeneity, independent
                                 of where the group sits relative to AllMitoMean.

        AMHI_D_median          — recommended single-scalar composite index: median
                                 total deviation of individual mitochondria from the
                                 pan-species reference origin.
                                 Integrates both AMHI_offset and MHI_D_absolute in
                                 one geometrically principled value. Use this as the
                                 primary summary metric when a single number is needed.
                                 Caution: do not combine all three in a further
                                 composite — AMHI_D already encodes the other two.

    Arguments:
        anndata_object: AnnData with frozen transforms in .uns['amhi_reference'].
        group_by: .obs column to group by (auto-detects unique_subject_ID if None).
        obs_filter: Optional dict to restrict which mitos are measured.
        reference_label: Must match the label used in compute_all_mito_mean().
        amhi_results: Pre-computed DataFrame from compute_amhi(). If None,
            compute_amhi() is called automatically.
        mhid_results: Pre-computed DataFrame from compute_mhi_d_absolute(). If None,
            compute_mhi_d_absolute() is called automatically.

    Returns:
        Merged DataFrame with one row per group and columns:
        n_mitos, AMHI_offset, AMHI_D_median, MHI_D_absolute_median.
    """
    working_anndata = _exclude_reference_rows(anndata_object, reference_label)
    working_anndata = _apply_obs_filter(working_anndata, obs_filter)
    group_column = _resolve_group_column(working_anndata, group_by)

    if amhi_results is None:
        # Use cached result when no filter is active — avoids redundant recomputation.
        # When obs_filter is set the cached result covers all subjects, so we must recompute.
        if (
            obs_filter is None
            and _AMHI_RESULTS_KEY in anndata_object.uns
            and group_column in anndata_object.uns[_AMHI_RESULTS_KEY]
        ):
            amhi_results = anndata_object.uns[_AMHI_RESULTS_KEY][group_column]
        else:
            amhi_results = compute_amhi(
                anndata_object,
                group_by=group_column,
                obs_filter=obs_filter,
                reference_label=reference_label,
            )

    if mhid_results is None:
        if (
            obs_filter is None
            and _AMHI_MHID_RESULTS_KEY in anndata_object.uns
            and group_column in anndata_object.uns[_AMHI_MHID_RESULTS_KEY]
        ):
            mhid_results = anndata_object.uns[_AMHI_MHID_RESULTS_KEY][group_column]
        else:
            mhid_results = compute_mhi_d_absolute(
                anndata_object,
                group_by=group_column,
                obs_filter=obs_filter,
                reference_label=reference_label,
            )

    # Merge on the shared group index.
    summary = amhi_results[["n_mitos", "AMHI_offset", "AMHI_D_median"]].join(
        mhid_results[["MHI_D_absolute_median"]],
        how="inner",
    )
    summary = summary.sort_values("AMHI_offset", ascending=False)

    # ── Console output ─────────────────────────────────────────────────────
    col_w = max(len(group_column), max(len(str(g)) for g in summary.index))
    header = (
        f"{'Group':<{col_w}}  {'AMHI_offset':>12}  "
        f"{'AMHI_D_median':>14}  {'MHI_D_absolute_median':>22}  {'n_mitos':>8}"
    )
    separator = "-" * len(header)
    print(f"\n[AMHI] Summary — {group_column}")
    print(separator)
    print(header)
    print(separator)
    for group_value, row in summary.iterrows():
        print(
            f"{str(group_value):<{col_w}}  "
            f"{row['AMHI_offset']:>12.3f}  "
            f"{row['AMHI_D_median']:>14.3f}  "
            f"{row['MHI_D_absolute_median']:>22.3f}  "
            f"{int(row['n_mitos']):>8}"
        )
    print(separator)
    print()

    return summary


def plot_amhi_profile(
    anndata_object: anndata.AnnData,
    group_by: Optional[str] = None,
    color_by: Optional[str] = None,
    condition_by: Optional[str] = None,
    x_metric: str = "MHI_D_absolute_median",
    y_metric: str = "AMHI_offset",
    obs_filter: Optional[Dict[str, Union[str, List[str]]]] = None,
    reference_label: str = "AllMitoMean",
    summary_dataframe: Optional[pd.DataFrame] = None,
    show_arrows: bool = True,
    condition_order: Optional[List[str]] = None,
    show_reference_lines: bool = True,
    title: str = "",
) -> None:
    """
    2D scatter of per-group AMHI metrics, styled like the MHI profile plot.

    Each dot is one group (typically one subject). The axes are freely
    configurable to any two columns from summarize_amhi() output. The default
    layout is:
        x = MHI_D_absolute_median  (internal heterogeneity / dispersion)
        y = AMHI_offset            (position offset from AllMitoMean)

    Visual encoding:
        Colour  — color_by .obs column (e.g. 'specie')
        Shape   — condition_by .obs column (first value → circle, second → square)
        Arrows  — direction of shift from condition_order[0] to condition_order[1]
                  per colour group (e.g. Young → Old per species)
        Labels  — subject ID next to each dot
        Annotation — colour group name at the centroid of that group's cluster

    Arguments:
        anndata_object: AnnData with frozen transforms in .uns['amhi_reference'].
        group_by: .obs column that identifies individual groups / subjects.
            Auto-detected if None.
        color_by: .obs column for colour grouping (e.g. 'specie'). If None,
            all dots are coloured the same.
        condition_by: .obs column for marker shape (e.g. 'condition').
            If None, all dots use circles.
        x_metric: Column from summarize_amhi() output to plot on the x-axis.
            Default: 'MHI_D_absolute_median'.
        y_metric: Column from summarize_amhi() output to plot on the y-axis.
            Default: 'AMHI_offset'.
        obs_filter: Optional dict to restrict which mitos are measured.
        reference_label: Must match the label used in compute_all_mito_mean().
        summary_dataframe: Pre-computed DataFrame from summarize_amhi(). If None,
            summarize_amhi() is called automatically (no extra console output).
        show_arrows: If True, draw an arrow from the condition_order[0] centroid
            to the condition_order[1] centroid for each colour group.
        condition_order: Two-element list specifying which condition is the arrow
            tail (index 0) and which is the arrow head (index 1).
            Default: ['Young', 'Old'].
        show_reference_lines: If True, draw dashed grey lines at the
            population-wide median of each axis.
        title: Optional plot title. Auto-generated if empty.
    """
    if condition_order is None:
        condition_order = ["Young", "Old"]

    # ── Resolve group column and prepare data ─────────────────────────────
    working_anndata = _exclude_reference_rows(anndata_object, reference_label)
    working_anndata = _apply_obs_filter(working_anndata, obs_filter)
    group_column = _resolve_group_column(working_anndata, group_by)

    if summary_dataframe is None:
        summary_dataframe = summarize_amhi(
            anndata_object,
            group_by=group_column,
            obs_filter=obs_filter,
            reference_label=reference_label,
        )

    # Validate requested metric columns.
    for metric in (x_metric, y_metric):
        if metric not in summary_dataframe.columns:
            raise ValueError(
                f"Metric '{metric}' not found in summary_dataframe. "
                f"Available: {summary_dataframe.columns.tolist()}"
            )

    # ── Attach colour and condition metadata to the summary ────────────────
    # For each group value look up its colour_by and condition_by from .obs.
    # Use the first occurrence — all mitos in one group share these values.
    obs = working_anndata.obs

    def _first_obs_value(group_value: str, column: str) -> Optional[str]:
        """Return the first .obs value for the given group and column."""
        mask = obs[group_column].astype(str) == str(group_value)
        subset = obs.loc[mask, column]
        return str(subset.iloc[0]) if len(subset) > 0 else None

    summary_dataframe = summary_dataframe.copy()

    if color_by is not None and color_by in obs.columns:
        summary_dataframe["_color_group"] = [
            _first_obs_value(g, color_by) for g in summary_dataframe.index
        ]
    else:
        summary_dataframe["_color_group"] = "all"
        color_by = "_color_group"

    if condition_by is not None and condition_by in obs.columns:
        summary_dataframe["_condition"] = [
            _first_obs_value(g, condition_by) for g in summary_dataframe.index
        ]
    else:
        summary_dataframe["_condition"] = condition_order[0]
        condition_by = "_condition"

    color_groups = sort_values_for_legend(summary_dataframe["_color_group"].dropna().unique().tolist())
    color_map = _build_color_map(anndata_object, color_by, color_groups)

    # Marker shape: first condition_order value → circle, second → square.
    _MARKERS = {condition_order[0]: "o", condition_order[1]: "s"}

    # ── Draw plot ─────────────────────────────────────────────────────────
    fig, axes = plt.subplots(figsize=(10, 8))

    # Reference lines at population-wide medians.
    if show_reference_lines:
        x_med = float(summary_dataframe[x_metric].median())
        y_med = float(summary_dataframe[y_metric].median())
        axes.axvline(x_med, color="lightgray", linewidth=1.0, linestyle="--", zorder=0)
        axes.axhline(y_med, color="lightgray", linewidth=1.0, linestyle="--", zorder=0)

    # Draw one scatter per (color_group × condition) combination.
    plotted_color_labels: set = set()
    plotted_condition_labels: set = set()

    for _, row in summary_dataframe.iterrows():
        color_group = row["_color_group"]
        condition = row["_condition"]
        group_label = str(row.name)

        dot_color = color_map.get(str(color_group), "#999999")
        marker = _MARKERS.get(str(condition), "o")
        x_val = float(row[x_metric])
        y_val = float(row[y_metric])

        # Build legend label flags.
        color_legend_label = str(color_group) if str(color_group) not in plotted_color_labels else None
        condition_legend_label = str(condition) if str(condition) not in plotted_condition_labels else None
        plotted_color_labels.add(str(color_group))
        plotted_condition_labels.add(str(condition))

        axes.scatter(
            x_val,
            y_val,
            color=dot_color,
            marker=marker,
            s=80,
            edgecolors="black",
            linewidths=0.5,
            zorder=3,
        )
        # Subject ID label.
        axes.annotate(
            group_label,
            xy=(x_val, y_val),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=7,
            color=dot_color,
            zorder=4,
        )

    # Arrows: per colour group, from condition_order[0] centroid to condition_order[1].
    # Special case: when color_by == condition_by, each color_group contains only one
    # condition, so the per-group loop never finds both conditions.  Draw one global
    # arrow from the condition_order[0] centroid to the condition_order[1] centroid.
    if show_arrows and len(condition_order) >= 2:
        if color_by == condition_by:
            cond_0_rows = summary_dataframe[summary_dataframe["_condition"] == condition_order[0]]
            cond_1_rows = summary_dataframe[summary_dataframe["_condition"] == condition_order[1]]
            if len(cond_0_rows) > 0 and len(cond_1_rows) > 0:
                x0 = float(cond_0_rows[x_metric].mean())
                y0 = float(cond_0_rows[y_metric].mean())
                x1 = float(cond_1_rows[x_metric].mean())
                y1 = float(cond_1_rows[y_metric].mean())
                axes.annotate(
                    "",
                    xy=(x1, y1),
                    xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="-|>", color="dimgray", lw=1.5),
                    zorder=2,
                )
        else:
            for color_group in color_groups:
                group_rows = summary_dataframe[summary_dataframe["_color_group"] == color_group]
                cond_0_rows = group_rows[group_rows["_condition"] == condition_order[0]]
                cond_1_rows = group_rows[group_rows["_condition"] == condition_order[1]]
                if len(cond_0_rows) == 0 or len(cond_1_rows) == 0:
                    continue
                x0 = float(cond_0_rows[x_metric].mean())
                y0 = float(cond_0_rows[y_metric].mean())
                x1 = float(cond_1_rows[x_metric].mean())
                y1 = float(cond_1_rows[y_metric].mean())
                dot_color = color_map.get(str(color_group), "#999999")
                axes.annotate(
                    "",
                    xy=(x1, y1),
                    xytext=(x0, y0),
                    arrowprops=dict(
                        arrowstyle="-|>",
                        color=dot_color,
                        lw=1.5,
                    ),
                    zorder=2,
                )

    # Annotate colour group name at the centroid of each colour cluster.
    for color_group in color_groups:
        group_rows = summary_dataframe[summary_dataframe["_color_group"] == color_group]
        cx = float(group_rows[x_metric].mean())
        cy = float(group_rows[y_metric].mean())
        dot_color = color_map.get(str(color_group), "#999999")
        axes.text(
            cx,
            cy,
            str(color_group),
            fontsize=9,
            fontweight="bold",
            color=dot_color,
            ha="center",
            va="center",
            alpha=0.4,
            zorder=1,
        )

    # ── Axes labels and title ─────────────────────────────────────────────
    _METRIC_LABELS = {
        "AMHI_offset": "AMHI_offset  (distance of centroid to AllMitoMean)",
        "AMHI_D_median": "AMHI_D median  (total per-mito deviation from AllMitoMean)",
        "MHI_D_absolute_median": "MHI_D_absolute median  (internal dispersion)",
    }
    axes.set_xlabel(_METRIC_LABELS.get(x_metric, x_metric), fontsize=11)
    axes.set_ylabel(_METRIC_LABELS.get(y_metric, y_metric), fontsize=11)

    if not title:
        arrow_note = (
            f"  Arrow = {color_by} centroid direction  "
            f"({condition_order[0]} → {condition_order[1]})"
            if show_arrows and condition_by != "_condition"
            else ""
        )
        title = (
            f"Individual AMHI profile — {x_metric} vs {y_metric}\n"
            f"○ = {condition_order[0]}   □ = {condition_order[1]}{arrow_note}"
        )
    axes.set_title(title, fontsize=12, fontweight="bold")

    # ── Legend ─────────────────────────────────────────────────────────────
    # Left legend: colour groups (species). Right legend: condition shapes.
    color_handles = [
        matplotlib.lines.Line2D(
            [], [],
            marker="o",
            color="w",
            markerfacecolor=color_map.get(str(cg), "#999999"),
            markeredgecolor="black",
            markersize=8,
            linewidth=0,
            label=str(cg),
        )
        for cg in color_groups
    ]
    condition_handles = [
        matplotlib.lines.Line2D(
            [], [],
            marker=_MARKERS.get(str(cond), "o"),
            color="w",
            markerfacecolor="gray",
            markeredgecolor="black",
            markersize=8,
            linewidth=0,
            label=str(cond),
        )
        for cond in condition_order
        if str(cond) in summary_dataframe["_condition"].astype(str).values
    ]

    if color_by != "_color_group":
        color_legend = axes.legend(
            handles=color_handles,
            title=str(color_by),
            loc="upper left",
            fontsize=8,
            title_fontsize=9,
            framealpha=0.8,
        )
        axes.add_artist(color_legend)

    if condition_by != "_condition" and condition_handles:
        axes.legend(
            handles=condition_handles,
            title="Condition",
            loc="lower right",
            fontsize=8,
            title_fontsize=9,
            framealpha=0.8,
        )

    plt.tight_layout()
    plt.show()

    print(
        f"[AMHI] Profile plot: {len(summary_dataframe)} groups, "
        f"x={x_metric}, y={y_metric}"
    )


def plot_amhi_bar(
    anndata_object: anndata.AnnData,
    group_by: Optional[str] = None,
    color_by: Optional[str] = None,
    metric: str = "AMHI_offset",
    obs_filter: Optional[Dict[str, Union[str, List[str]]]] = None,
    reference_label: str = "AllMitoMean",
    summary_dataframe: Optional[pd.DataFrame] = None,
    show_error_bars: bool = True,
    error_style: str = "iqr",
    show_reference_line: bool = True,
    title: str = "",
) -> None:
    """
    Horizontal bar chart ranking groups by a chosen AMHI heterogeneity metric.

    One bar per group (subject, species, or other grouping), sorted from highest
    to lowest metric value. Bars are coloured by an optional categorical column
    (e.g. ``specie`` or ``condition``). An optional vertical dashed line marks
    the population-wide median.

    Args:
        anndata_object: AnnData object containing the TEM or SFC data.
        group_by: Column in ``.obs`` used to define groups (e.g.
            ``"unique_subject_ID"``). If None the default subject column is used.
        color_by: Column in ``.obs`` used to assign bar colours (e.g. ``"specie"``
            or ``"condition"``). If None all bars are drawn in gray.
        metric: Which summary metric to plot. One of ``"AMHI_offset"``,
            ``"AMHI_D_median"``, or ``"MHI_D_absolute_median"``.
        obs_filter: Optional dict of ``{column: allowed_value(s)}`` to pre-filter
            observations before computing.
        reference_label: Label of the synthetic AllMitoMean row to exclude.
        summary_dataframe: Pre-computed summary from ``summarize_amhi()``. If
            None it is computed internally.
        show_error_bars: If True, draw horizontal error bars for metrics that
            have a per-mito distribution (``"AMHI_D_median"`` and
            ``"MHI_D_absolute_median"``). No error bars for ``"AMHI_offset"``
            (single centroid value, no distribution).
        error_style: Style of error bars. ``"iqr"`` (default) draws asymmetric
            whiskers from Q25 to Q75 — robust against outliers and always
            non-negative. ``"std"`` draws symmetric ± standard deviation (left
            arm clamped to zero).
        show_reference_line: If True draw a vertical dashed line at the
            population-wide median of the chosen metric.
        title: Custom plot title. If empty a default title is generated.

    Returns:
        None. Displays the plot.
    """
    if error_style not in ("std", "iqr"):
        raise ValueError(
            f"error_style must be 'std' or 'iqr', got '{error_style}'."
        )
    _METRIC_LABELS: Dict[str, str] = {
        "AMHI_offset": "AMHI_offset  (centroid distance to AllMitoMean)",
        "AMHI_D_median": "AMHI_D median  (per-mito distance to AllMitoMean)",
        "MHI_D_absolute_median": "MHI_D_absolute median  (internal dispersion)",
    }

    working_anndata = _exclude_reference_rows(anndata_object, reference_label)
    working_anndata = _apply_obs_filter(working_anndata, obs_filter)
    group_column = _resolve_group_column(working_anndata, group_by)

    # ── Retrieve or compute the summary DataFrame ──────────────────────────
    if summary_dataframe is None:
        summary_dataframe = summarize_amhi(
            anndata_object,
            group_by=group_column,
            obs_filter=obs_filter,
            reference_label=reference_label,
        )

    if metric not in summary_dataframe.columns:
        raise ValueError(
            f"Metric '{metric}' not found in summary_dataframe. "
            f"Available: {summary_dataframe.columns.tolist()}"
        )

    # ── Retrieve per-mito distribution columns for error bars ─────────────
    # AMHI_offset is a single centroid value — no per-mito distribution, no error bars.
    # AMHI_D_median and MHI_D_absolute_median are medians over distributions → IQR or std.
    _AMHI_D_COLS = {
        "std": "AMHI_D_std",
        "q25": "AMHI_D_q25",
        "q75": "AMHI_D_q75",
    }
    _MHID_COLS = {
        "std": "MHI_D_absolute_std",
        "q25": "MHI_D_absolute_q25",
        "q75": "MHI_D_absolute_q75",
    }

    xerr_left_values: Optional[List[float]] = None
    xerr_right_values: Optional[List[float]] = None

    if show_error_bars and metric in ("AMHI_D_median", "MHI_D_absolute_median"):
        if metric == "AMHI_D_median":
            cache_key = _AMHI_RESULTS_KEY
            col_map = _AMHI_D_COLS
            compute_fn = lambda: compute_amhi(
                anndata_object,
                group_by=group_column,
                obs_filter=obs_filter,
                reference_label=reference_label,
            )
        else:
            cache_key = _AMHI_MHID_RESULTS_KEY
            col_map = _MHID_COLS
            compute_fn = lambda: compute_mhi_d_absolute(
                anndata_object,
                group_by=group_column,
                obs_filter=obs_filter,
                reference_label=reference_label,
            )

        if (
            cache_key in anndata_object.uns
            and group_column in anndata_object.uns[cache_key]
        ):
            dist_results = anndata_object.uns[cache_key][group_column]
        else:
            dist_results = compute_fn()

        groups_index = list(summary_dataframe.index)

        if error_style == "std":
            std_col = col_map["std"]
            std_list = [
                float(dist_results.loc[g, std_col]) if g in dist_results.index else 0.0
                for g in groups_index
            ]
            # Clamp left arm so whiskers never cross zero (distance ≥ 0).
            median_list = [float(summary_dataframe.loc[g, metric]) for g in groups_index]
            xerr_left_values = [min(m, s) for m, s in zip(median_list, std_list)]
            xerr_right_values = std_list
        else:
            # IQR: asymmetric, always non-negative, robust against outliers.
            q25_col, q75_col = col_map["q25"], col_map["q75"]
            median_list = [float(summary_dataframe.loc[g, metric]) for g in groups_index]
            q25_list = [
                float(dist_results.loc[g, q25_col]) if g in dist_results.index else 0.0
                for g in groups_index
            ]
            q75_list = [
                float(dist_results.loc[g, q75_col]) if g in dist_results.index else 0.0
                for g in groups_index
            ]
            xerr_left_values = [max(0.0, m - q25) for m, q25 in zip(median_list, q25_list)]
            xerr_right_values = [max(0.0, q75 - m) for m, q75 in zip(median_list, q75_list)]

    # ── Sort groups by metric value descending ─────────────────────────────
    sorted_summary = summary_dataframe.sort_values(metric, ascending=True)
    group_values: List[str] = [str(g) for g in sorted_summary.index]
    metric_values: np.ndarray = sorted_summary[metric].values.astype(float)
    n_groups = len(group_values)

    # ── Build colour map ───────────────────────────────────────────────────
    obs = working_anndata.obs

    def _first_obs_value(group_value: str, column: str) -> Optional[str]:
        """Return the first .obs value for the given group and column."""
        mask = obs[group_column].astype(str) == str(group_value)
        subset = obs.loc[mask, column]
        return str(subset.iloc[0]) if len(subset) > 0 else None

    if color_by is not None and color_by in obs.columns:
        color_category_per_group: List[Optional[str]] = [
            _first_obs_value(g, color_by) for g in group_values
        ]
        unique_color_categories: List[str] = sorted(
            {c for c in color_category_per_group if c is not None}
        )
        color_map = _build_color_map(anndata_object, color_by, unique_color_categories)
        bar_colors: List[str] = [
            color_map.get(str(c), "#999999") if c is not None else "#999999"
            for c in color_category_per_group
        ]
    else:
        unique_color_categories = []
        color_map = {}
        bar_colors = ["#999999"] * n_groups

    # Reorder error values to match sorted group order.
    sorted_xerr_left: Optional[np.ndarray] = None
    sorted_xerr_right: Optional[np.ndarray] = None
    if xerr_left_values is not None:
        original_order = list(summary_dataframe.index)
        sorted_xerr_left = np.array([
            xerr_left_values[original_order.index(g)] if g in original_order else 0.0
            for g in group_values
        ])
        sorted_xerr_right = np.array([
            xerr_right_values[original_order.index(g)] if g in original_order else 0.0
            for g in group_values
        ])

    # ── Draw ───────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(figsize=(10, max(6, n_groups * 0.4)))
    y_positions = np.arange(n_groups)

    axes.barh(
        y_positions,
        metric_values,
        color=bar_colors,
        height=0.6,
        alpha=0.85,
        edgecolor="white",
        linewidth=0.8,
    )

    if sorted_xerr_left is not None:
        axes.errorbar(
            metric_values,
            y_positions,
            xerr=[sorted_xerr_left, sorted_xerr_right],
            fmt="none",
            color="black",
            capsize=3,
            linewidth=1.0,
            zorder=5,
        )

    if show_reference_line:
        reference_value = float(metric_values.median()
                                if hasattr(metric_values, "median")
                                else np.median(metric_values))
        axes.axvline(
            reference_value,
            color="dimgray",
            linewidth=1.2,
            linestyle="--",
            zorder=2,
            label=f"Median ({reference_value:.3f})",
        )

    axes.set_yticks(y_positions)
    axes.set_yticklabels(group_values, fontsize=9)
    axes.invert_yaxis()
    axes.set_xlabel(_METRIC_LABELS.get(metric, metric), fontsize=11)

    # ── Legend for color_by categories ────────────────────────────────────
    if color_by is not None and unique_color_categories:
        legend_handles = [
            matplotlib.lines.Line2D(
                [0], [0],
                marker="s",
                color="w",
                markerfacecolor=color_map.get(cat, "#999999"),
                markersize=10,
                label=cat,
            )
            for cat in unique_color_categories
        ]
        if show_reference_line:
            legend_handles.append(
                matplotlib.lines.Line2D(
                    [0], [0],
                    color="dimgray",
                    linewidth=1.2,
                    linestyle="--",
                    label=f"Median ({reference_value:.3f})",
                )
            )
        axes.legend(
            handles=legend_handles,
            title=color_by,
            loc="lower right",
            fontsize=8,
        )
    elif show_reference_line:
        axes.legend(loc="lower right", fontsize=8)

    plot_title = title if title else f"AMHI — {metric} by {group_column}"
    axes.set_title(plot_title, fontsize=12, fontweight="bold")

    plt.tight_layout()
    plt.show()

    print(
        f"[AMHI] plot_amhi_bar: {n_groups} groups, metric={metric}, "
        f"color_by={color_by}, error_style={error_style if show_error_bars else 'none'}"
    )
    print(
        f"  metric range: [{float(metric_values.min()):.4f}, "
        f"{float(metric_values.max()):.4f}]"
    )


def plot_amhi_species_comparison(
    anndata_object: anndata.AnnData,
    group_by: str = "specie",
    split_by: Optional[str] = None,
    metric: str = "AMHI_offset",
    obs_filter: Optional[Dict[str, Union[str, List[str]]]] = None,
    reference_label: str = "AllMitoMean",
    subject_column: str = "unique_subject_ID",
    show_subject_dots: bool = True,
    group_order: Optional[List[str]] = None,
    title: str = "",
) -> None:
    """
    Vertical grouped bar chart comparing AMHI heterogeneity across species or
    any coarse grouping, with individual subject dots overlaid.

    One bar per species (or ``group_by`` category) shows the median metric value
    across subjects in that group. Error bars represent the inter-quartile range
    (Q25 – Q75). Individual subject values are optionally overlaid as jittered
    dots. When ``split_by`` is provided (e.g. ``"condition"``), side-by-side bars
    are drawn for each condition within every species slot.

    Args:
        anndata_object: AnnData object containing the TEM or SFC data.
        group_by: Column in ``.obs`` used for the x-axis grouping (e.g.
            ``"specie"``).
        split_by: Optional column in ``.obs`` used to split bars within each
            group (e.g. ``"condition"`` for Young vs Old). If None one bar per
            group is drawn.
        metric: Which per-subject AMHI metric to aggregate. One of
            ``"AMHI_offset"``, ``"AMHI_D_median"``, or
            ``"MHI_D_absolute_median"``.
        obs_filter: Optional dict of ``{column: allowed_value(s)}`` to
            pre-filter observations before computing.
        reference_label: Label of the synthetic AllMitoMean row to exclude.
        subject_column: Column in ``.obs`` that identifies individual subjects
            (default ``"unique_subject_ID"``).
        show_subject_dots: If True, overlay individual subject values as
            jittered dots on top of the bars.
        group_order: Explicit list of group labels defining the left-to-right
            x-axis order (e.g. ``["Worm", "Droso", "ZFish", "KFish", "Mouse",
            "Human"]``). Groups absent from the data are silently ignored.
            If None and ``group_by="specie"``, the canonical phylogenetic order
            from ``TEM_SPECIES_PHYLOGENETIC_ORDER`` is used automatically.
            Otherwise falls back to alphabetical order.
        title: Custom plot title. If empty a default title is generated.

    Returns:
        None. Displays the plot.
    """
    _METRIC_LABELS: Dict[str, str] = {
        "AMHI_offset": "AMHI_offset  (centroid distance to AllMitoMean)",
        "AMHI_D_median": "AMHI_D median  (per-mito distance to AllMitoMean)",
        "MHI_D_absolute_median": "MHI_D_absolute median  (internal dispersion)",
    }

    working_anndata = _exclude_reference_rows(anndata_object, reference_label)
    working_anndata = _apply_obs_filter(working_anndata, obs_filter)

    if group_by not in working_anndata.obs.columns:
        raise ValueError(
            f"group_by column '{group_by}' not found in .obs. "
            f"Available: {working_anndata.obs.columns.tolist()}"
        )
    if subject_column not in working_anndata.obs.columns:
        raise ValueError(
            f"subject_column '{subject_column}' not found in .obs. "
            f"Available: {working_anndata.obs.columns.tolist()}"
        )
    if split_by is not None and split_by not in working_anndata.obs.columns:
        raise ValueError(
            f"split_by column '{split_by}' not found in .obs. "
            f"Available: {working_anndata.obs.columns.tolist()}"
        )

    # ── Compute per-subject AMHI metrics ───────────────────────────────────
    subject_amhi_results = compute_amhi(
        anndata_object,
        group_by=subject_column,
        obs_filter=obs_filter,
        reference_label=reference_label,
    )
    subject_mhid_results = compute_mhi_d_absolute(
        anndata_object,
        group_by=subject_column,
        obs_filter=obs_filter,
        reference_label=reference_label,
    )
    # Merge so all three metrics are available on one DataFrame.
    subject_metrics = subject_amhi_results[
        ["n_mitos", "AMHI_offset", "AMHI_D_median"]
    ].join(subject_mhid_results[["MHI_D_absolute_median"]], how="inner")

    if metric not in subject_metrics.columns:
        raise ValueError(
            f"Metric '{metric}' not in available columns. "
            f"Choose from: {subject_metrics.columns.tolist()}"
        )

    # ── Enrich per-subject metrics with group_by (and split_by) ───────────
    columns_to_extract = [subject_column, group_by]
    if split_by is not None:
        columns_to_extract.append(split_by)
    meta_dataframe = (
        working_anndata.obs[columns_to_extract]
        .drop_duplicates()
        .set_index(subject_column)
    )
    subject_metrics = subject_metrics.copy()
    subject_metrics["_group"] = subject_metrics.index.map(meta_dataframe[group_by])
    if split_by is not None:
        subject_metrics["_split"] = subject_metrics.index.map(meta_dataframe[split_by])
    subject_metrics = subject_metrics.dropna(subset=["_group"])

    present_groups: List[str] = subject_metrics["_group"].dropna().unique().tolist()

    if group_order is not None:
        # Keep only groups that actually exist in the data, in the requested order.
        x_axis_categories = [g for g in group_order if g in present_groups]
        # Append any groups not listed in group_order at the end (alphabetically).
        unlisted = sort_values_for_legend(g for g in present_groups if g not in group_order)
        x_axis_categories.extend(unlisted)
    elif group_by == "specie":
        # Use canonical phylogenetic order when grouping by species.
        from mito_marker.controlled_vocabulary import TEM_SPECIES_PHYLOGENETIC_ORDER
        x_axis_categories = [g for g in TEM_SPECIES_PHYLOGENETIC_ORDER if g in present_groups]
        unlisted = sort_values_for_legend(g for g in present_groups if g not in TEM_SPECIES_PHYLOGENETIC_ORDER)
        x_axis_categories.extend(unlisted)
    else:
        x_axis_categories = sort_values_for_legend(present_groups)
    n_species = len(x_axis_categories)
    global_median = float(subject_metrics[metric].median())

    # ── Build colour map ───────────────────────────────────────────────────
    if split_by is not None:
        split_categories: List[str] = sorted(
            subject_metrics["_split"].dropna().unique().tolist()
        )
        color_map = _build_color_map(anndata_object, split_by, split_categories)
    else:
        split_categories = []
        color_map = _build_color_map(anndata_object, group_by, x_axis_categories)

    # ── Draw ───────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(figsize=(max(8, n_species * 1.4), 6))
    x_positions = np.arange(n_species)
    rng = np.random.default_rng(seed=42)

    if split_by is None:
        # One bar per species.
        for idx, species_value in enumerate(x_axis_categories):
            group_mask = subject_metrics["_group"] == species_value
            group_values_array = subject_metrics.loc[group_mask, metric].values.astype(float)

            if len(group_values_array) == 0:
                continue

            bar_median = float(np.median(group_values_array))
            q25_value = float(np.percentile(group_values_array, 25))
            q75_value = float(np.percentile(group_values_array, 75))
            bar_color = color_map.get(species_value, "#999999")

            axes.bar(
                x_positions[idx],
                bar_median,
                color=bar_color,
                width=0.6,
                alpha=0.85,
                edgecolor="white",
                linewidth=0.8,
            )
            lower_arm = max(0.0, bar_median - q25_value)
            upper_arm = max(0.0, q75_value - bar_median)
            axes.errorbar(
                x_positions[idx],
                bar_median,
                yerr=[[lower_arm], [upper_arm]],
                fmt="none",
                color="black",
                capsize=4,
                linewidth=1.2,
                zorder=5,
            )

            if show_subject_dots and len(group_values_array) > 0:
                jitter = rng.uniform(-0.18, 0.18, len(group_values_array))
                axes.scatter(
                    x_positions[idx] + jitter,
                    group_values_array,
                    color="black",
                    alpha=0.6,
                    s=25,
                    zorder=6,
                )

        axes.set_xticks(x_positions)
        axes.set_xticklabels(x_axis_categories, rotation=30, ha="right", fontsize=10)

    else:
        # Side-by-side bars: one per (species × condition) pair.
        n_splits = len(split_categories)
        bar_width = 0.7 / max(n_splits, 1)
        offsets = np.linspace(
            -(n_splits - 1) * bar_width / 2,
            (n_splits - 1) * bar_width / 2,
            n_splits,
        )
        plotted_split_labels: set = set()

        for split_idx, split_value in enumerate(split_categories):
            bar_color = color_map.get(split_value, "#999999")
            x_offset = offsets[split_idx]

            for species_idx, species_value in enumerate(x_axis_categories):
                group_mask = (
                    (subject_metrics["_group"] == species_value)
                    & (subject_metrics["_split"] == split_value)
                )
                group_values_array = subject_metrics.loc[
                    group_mask, metric
                ].values.astype(float)

                if len(group_values_array) == 0:
                    continue

                bar_median = float(np.median(group_values_array))
                q25_value = float(np.percentile(group_values_array, 25))
                q75_value = float(np.percentile(group_values_array, 75))
                x_center = x_positions[species_idx] + x_offset

                label = split_value if split_value not in plotted_split_labels else None
                plotted_split_labels.add(split_value)

                axes.bar(
                    x_center,
                    bar_median,
                    color=bar_color,
                    width=bar_width * 0.85,
                    alpha=0.85,
                    edgecolor="white",
                    linewidth=0.8,
                    label=label,
                )
                lower_arm = max(0.0, bar_median - q25_value)
                upper_arm = max(0.0, q75_value - bar_median)
                axes.errorbar(
                    x_center,
                    bar_median,
                    yerr=[[lower_arm], [upper_arm]],
                    fmt="none",
                    color="black",
                    capsize=3,
                    linewidth=1.0,
                    zorder=5,
                )

                if show_subject_dots and len(group_values_array) > 0:
                    jitter = rng.uniform(
                        -bar_width * 0.35,
                        bar_width * 0.35,
                        len(group_values_array),
                    )
                    axes.scatter(
                        x_center + jitter,
                        group_values_array,
                        color="black",
                        alpha=0.6,
                        s=20,
                        zorder=6,
                    )

        axes.set_xticks(x_positions)
        axes.set_xticklabels(x_axis_categories, rotation=30, ha="right", fontsize=10)
        axes.legend(title=split_by, loc="upper right", fontsize=9)

    # ── Global reference line ──────────────────────────────────────────────
    axes.axhline(
        global_median,
        color="dimgray",
        linewidth=1.2,
        linestyle="--",
        zorder=2,
        label=f"Global median ({global_median:.3f})",
    )

    axes.set_ylabel(_METRIC_LABELS.get(metric, metric), fontsize=11)
    axes.set_xlabel(group_by, fontsize=11)

    plot_title = (
        title if title
        else f"AMHI species comparison — {metric}"
        + (f" split by {split_by}" if split_by else "")
    )
    axes.set_title(plot_title, fontsize=12, fontweight="bold")

    plt.tight_layout()
    plt.show()

    print(
        f"[AMHI] plot_amhi_species_comparison: {n_species} groups in '{group_by}', "
        f"metric={metric}, split_by={split_by}, "
        f"show_subject_dots={show_subject_dots}"
    )
    print(f"  Subject-level results: {len(subject_metrics)} subjects")
    print(f"  Global median {metric}: {global_median:.4f}")


def plot_amhi_umap(
    anndata_object: anndata.AnnData,
    color_by: str = "specie",
    obs_filter: Optional[Dict[str, Union[str, List[str]]]] = None,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    max_points_per_group: Optional[int] = 500,
    reference_label: str = "AllMitoMean",
    title: str = "",
) -> None:
    """
    Project all mitochondria into the frozen AMHI reference PCA space, then
    compute and display a UMAP of those projected coordinates.

    Each point represents one mitochondrion.  Coloring by 'specie' (default)
    places all species on a single figure so you can see whether they form
    distinct morphological clusters, overlap, or share regions of the common
    reference space.

    The UMAP runs on the PCA coordinates — not on raw features — so that
    inter-experiment scaling differences are already neutralised by the frozen
    StandardScaler + PCA from compute_all_mito_mean().

    Arguments:
        anndata_object: AnnData with .uns['amhi_reference'] set by
            compute_all_mito_mean().
        color_by: .obs column used to color the scatter (default 'specie').
        obs_filter: Optional dict to restrict which mitochondria are included
            (e.g. {'condition': 'Young'}).  Same format as compute_all_mito_mean().
        n_neighbors: UMAP n_neighbors parameter (default 15).
        min_dist: UMAP min_dist parameter (default 0.1).
        max_points_per_group: Maximum number of mitochondria to sample per
            group value (stratified sampling).  None → use all points.
        reference_label: Label of the synthetic AllMitoMean row to exclude
            (default 'AllMitoMean').
        title: Optional plot title suffix.
    """
    print("=" * 60)
    print("[AMHI] plot_amhi_umap()")
    print("=" * 60)

    # ── Step 1: exclude synthetic AllMitoMean row ─────────────────────────
    working_anndata = _exclude_reference_rows(anndata_object, reference_label)

    # ── Step 2: optional obs_filter ───────────────────────────────────────
    working_anndata = _apply_obs_filter(working_anndata, obs_filter)

    # ── Step 3: load frozen transforms ────────────────────────────────────
    scaler, pca, _mean_vector, feature_names = _load_reference_transforms(
        anndata_object, reference_label
    )

    # ── Step 4: extract feature matrix aligned to reference features ──────
    full_matrix, all_channel_names = _get_raw_feature_matrix(working_anndata)
    available_features = [f for f in feature_names if f in working_anndata.var_names]
    if len(available_features) != len(feature_names):
        missing = set(feature_names) - set(available_features)
        raise ValueError(
            f"[AMHI] {len(missing)} reference features not found in AnnData.var: "
            f"{missing}."
        )
    feature_indices = [all_channel_names.index(f) for f in feature_names]
    raw_matrix = full_matrix[:, feature_indices].astype(np.float64)

    # ── Step 5: project into fixed PCA space ──────────────────────────────
    projected_matrix = _project_into_reference_space(raw_matrix, scaler, pca)
    # projected_matrix shape: (n_mitos, n_pca_components)

    # ── Step 6: stratified sub-sampling ───────────────────────────────────
    obs_dataframe = working_anndata.obs.reset_index(drop=True)
    if color_by not in obs_dataframe.columns:
        raise ValueError(
            f"[AMHI] color_by='{color_by}' not found in .obs. "
            f"Available columns: {sorted(obs_dataframe.columns.tolist())}"
        )

    if max_points_per_group is not None:
        group_series = obs_dataframe[color_by].astype(str)
        sample_indices: List[int] = []
        for group_value in group_series.unique():
            group_row_indices = np.where(group_series == group_value)[0]
            if len(group_row_indices) > max_points_per_group:
                rng = np.random.default_rng(seed=42)
                group_row_indices = rng.choice(
                    group_row_indices,
                    size=max_points_per_group,
                    replace=False,
                )
            sample_indices.extend(group_row_indices.tolist())
        sample_indices_array = np.array(sample_indices)
        projected_sample = projected_matrix[sample_indices_array]
        obs_sample = obs_dataframe.iloc[sample_indices_array].reset_index(drop=True)
    else:
        projected_sample = projected_matrix
        obs_sample = obs_dataframe

    n_points = len(obs_sample)
    group_values = obs_sample[color_by].astype(str).values
    unique_groups = sort_values_for_legend(set(group_values))

    print(
        f"[AMHI] Projecting {n_points} mitochondria "
        f"({len(unique_groups)} groups in '{color_by}') "
        f"using {pca.n_components_} PCA components."
    )
    print(f"[AMHI] Running UMAP (n_neighbors={n_neighbors}, min_dist={min_dist})...")

    # ── Step 7: UMAP on projected PCA coordinates ─────────────────────────
    umap_model = UmapLib.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=42,
        n_jobs=1,
    )
    umap_coords = umap_model.fit_transform(projected_sample)
    # umap_coords shape: (n_points, 2)

    print("[AMHI] UMAP done.")

    # ── Step 8: build color map ────────────────────────────────────────────
    # Prefer colors already stored in .uns['color_palette'] (set by
    # assign_color_palette()), then fall back to matplotlib tab10.
    stored_palette = anndata_object.uns.get("color_palette", {})
    tab10_colormap = plt.cm.get_cmap("tab10")
    group_color_map: Dict[str, str] = {}
    for tab10_index, group_value in enumerate(unique_groups):
        if group_value in stored_palette:
            group_color_map[group_value] = stored_palette[group_value]
        else:
            r, g, b, _ = tab10_colormap(tab10_index % 10)
            group_color_map[group_value] = "#{:02x}{:02x}{:02x}".format(
                int(r * 255), int(g * 255), int(b * 255)
            )

    # ── Step 9: scatter plot ───────────────────────────────────────────────
    figure, axes = plt.subplots(figsize=(9, 7))

    for group_value in unique_groups:
        group_mask = group_values == group_value
        group_color = group_color_map[group_value]
        axes.scatter(
            umap_coords[group_mask, 0],
            umap_coords[group_mask, 1],
            c=group_color,
            label=group_value,
            s=6,
            alpha=0.5,
            linewidths=0,
            rasterized=True,
        )

    axes.set_xlabel("UMAP 1", fontsize=12)
    axes.set_ylabel("UMAP 2", fontsize=12)
    plot_title = (
        f"AMHI UMAP — colored by {color_by}"
        + (f"\n{title}" if title else "")
    )
    axes.set_title(plot_title, fontsize=13, fontweight="bold")
    axes.legend(
        title=color_by,
        markerscale=2,
        fontsize=9,
        title_fontsize=10,
        framealpha=0.8,
        loc="best",
    )
    axes.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    plt.show()

    print(
        f"[AMHI] UMAP plotted: {n_points} points, {len(unique_groups)} groups "
        f"in '{color_by}', n_neighbors={n_neighbors}, min_dist={min_dist}."
    )
