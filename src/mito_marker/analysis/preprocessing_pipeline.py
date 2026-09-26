"""
preprocessing_pipeline.py

Non-interactive preprocessing pipeline for the mito-marker analysis pipeline.

Applies the first three analysis steps — subset selection, feature selection,
and transformation/normalization — driven by a preprocessing_config dict
instead of console prompts.

Typical usage:
    from mito_marker.analysis.preprocessing_config import get_default_preprocessing_config
    from mito_marker.analysis.preprocessing_pipeline import run_preprocessing

    config = get_default_preprocessing_config()
    config["filters"]                         = {"condition": ["young"]}
    config["transform"]                       = "arcsinh"
    config["normalization"]                   = "zscore_col"
    config["feature_selection_methods"]       = ["MIM"]
    config["feature_selection_top_n"]         = 20
    config["feature_selection_target_obs_column"] = "condition"

    sfc_subset = run_preprocessing(sfc_anndata, config)
"""

from typing import Any, Dict, List, Optional

import anndata
import numpy as np

from mito_marker.analysis.preprocessing_config import _validate_preprocessing_config
from mito_marker.analysis.selection import (
    _ANALYSIS_CONFIG_KEY,
    _filter_obs_by_values,
    _get_subject_column,
    _subsample_obs_per_subject,
)
from mito_marker.controlled_vocabulary import FILTERABLE_OBS_COLUMNS


# ---------------------------------------------------------------------------
# Internal: dataset summary printer
# ---------------------------------------------------------------------------


def _print_dataset_summary(anndata_object: anndata.AnnData, label: str) -> None:
    """
    Print a structured summary of the dataset broken down by species, subjects,
    and conditions.

    Arguments:
        anndata_object: AnnData whose .obs is inspected.
        label: Header label printed above the summary (e.g. "BEFORE" / "AFTER").
    """
    obs = anndata_object.obs
    total_events = anndata_object.n_obs
    width = 60

    print("=" * width)
    print(f"  DATASET SUMMARY — {label}")
    print(f"  {total_events:,} events  ×  {anndata_object.n_vars} features")
    print("=" * width)

    # ── Species ──────────────────────────────────────────────
    specie_col = "specie" if "specie" in obs.columns else None
    subject_col = (
        "unique_subject_ID" if "unique_subject_ID" in obs.columns
        else "subject_ID" if "subject_ID" in obs.columns
        else None
    )
    condition_col = "condition" if "condition" in obs.columns else None

    if specie_col is not None:
        print()
        print("  SPECIES")
        print(f"  {'Species':<20} {'Subjects':>10} {'Events':>10}")
        print(f"  {'-'*20} {'-'*10} {'-'*10}")
        for specie_value, specie_group in obs.groupby(specie_col, observed=True):
            n_events = len(specie_group)
            if subject_col is not None:
                n_subjects = specie_group[subject_col].nunique()
            else:
                n_subjects = 0
            print(f"  {str(specie_value):<20} {n_subjects:>10,} {n_events:>10,}")
        print(f"  {'TOTAL':<20} {obs[subject_col].nunique() if subject_col else '—':>10} {total_events:>10,}")

    # ── Subjects ─────────────────────────────────────────────
    if subject_col is not None:
        print()
        print("  SUBJECTS")
        print(f"  {'Subject ID':<30} {'Events':>10}")
        print(f"  {'-'*30} {'-'*10}")
        subject_event_counts = (
            obs.groupby(subject_col, observed=True)
            .size()
            .sort_index()
        )
        for subject_id, n_events in subject_event_counts.items():
            print(f"  {str(subject_id):<30} {n_events:>10,}")
        print(f"  {'TOTAL':<30} {total_events:>10,}")

    # ── Conditions ───────────────────────────────────────────
    if condition_col is not None:
        print()
        print("  CONDITIONS")
        print(f"  {'Condition':<20} {'Subjects':>10} {'Events':>10}")
        print(f"  {'-'*20} {'-'*10} {'-'*10}")
        for condition_value, condition_group in obs.groupby(condition_col, observed=True):
            n_events = len(condition_group)
            if subject_col is not None:
                n_subjects = condition_group[subject_col].nunique()
            else:
                n_subjects = 0
            print(f"  {str(condition_value):<20} {n_subjects:>10,} {n_events:>10,}")
        print(f"  {'TOTAL':<20} {obs[subject_col].nunique() if subject_col else '—':>10} {total_events:>10,}")

    print("=" * width)
    print()


# ---------------------------------------------------------------------------
# Public: run_preprocessing
# ---------------------------------------------------------------------------


def run_preprocessing(
    anndata_object: anndata.AnnData,
    preprocessing_config: dict,
) -> anndata.AnnData:
    """
    Apply subset selection, feature selection, and normalization non-interactively.

    Reads all parameters from preprocessing_config (built by
    get_default_preprocessing_config() or configure_preprocessing()) and runs
    the three pipeline steps in order:
      1. Subset selection — filter .obs rows and optionally sub-sample per subject.
      2. Feature selection — score channels and flag the top-N as selected.
      3. Transformation + normalization — store result in .layers, set active layer.

    The original AnnData is never modified. A new AnnData is returned for
    step 1 (filtering creates a new object); steps 2 and 3 modify and return
    the same object in-place.

    Arguments:
        anndata_object: Input AnnData (SFC or TEM) to preprocess.
        preprocessing_config: Dict produced by get_default_preprocessing_config()
                              or configure_preprocessing(). Must pass validation.

    Returns:
        Preprocessed AnnData with:
          - .obs filtered to the selected subset
          - .var columns for feature selection scores and flags
          - .layers entry for the active transform/normalization
          - .uns['analysis_config'] updated with selection, subsampling,
            active_layer, and active_selection keys
    """
    _validate_preprocessing_config(preprocessing_config)

    _print_dataset_summary(anndata_object, "BEFORE PREPROCESSING")

    print("=" * 60)
    print("PREPROCESSING PIPELINE (non-interactive)")
    print("=" * 60)
    print()

    # Step 1 — subset selection
    print("[Step 1] Subset selection")
    result = _apply_subset_selection(anndata_object, preprocessing_config)
    print(
        f"=> After subset selection: {result.n_obs:,} observations "
        f"× {result.n_vars} features"
    )
    print()

    # Step 2 — feature selection
    print("[Step 2] Feature selection")
    result = _apply_feature_selection(result, preprocessing_config)
    print()

    # Step 3 — transformation + normalization
    print("[Step 3] Transformation + normalization")
    result = _apply_transform_and_normalize(result, preprocessing_config)
    print()

    active_layer = result.uns.get(_ANALYSIS_CONFIG_KEY, {}).get("active_layer")
    active_selection = result.uns.get(_ANALYSIS_CONFIG_KEY, {}).get("active_selection")

    _print_dataset_summary(result, "AFTER PREPROCESSING")

    print("=" * 60)
    print("PREPROCESSING COMPLETE")
    print(f"=> active_layer    : {active_layer!r}")
    print(f"=> active_selection: {active_selection!r}")
    print("=" * 60)

    return result


# ---------------------------------------------------------------------------
# Internal: step 1 — subset selection
# ---------------------------------------------------------------------------


def _apply_subset_selection(
    anndata_object: anndata.AnnData,
    preprocessing_config: dict,
) -> anndata.AnnData:
    """
    Filter .obs rows and optionally sub-sample per subject.

    Iterates through FILTERABLE_OBS_COLUMNS in canonical order. For each
    column present in preprocessing_config['filters'], keeps only the rows
    whose value is in the specified list. Columns absent from 'filters' are
    kept as-is (no filtering for that column).

    Arguments:
        anndata_object: Input AnnData.
        preprocessing_config: Must contain 'filters' (dict) and
                              'n_events_per_subject' (int).

    Returns:
        New AnnData containing only the rows that passed all filters.
    """
    filters: Dict[str, List] = preprocessing_config["filters"]
    n_events_per_subject: int = preprocessing_config["n_events_per_subject"]

    # Add a helper column to track original row positions across filtering steps.
    # Positional indexing is unambiguous even when obs_names are not unique.
    current_obs = anndata_object.obs.copy()
    current_obs["_row_position"] = np.arange(anndata_object.n_obs)

    for column_name in FILTERABLE_OBS_COLUMNS:
        if column_name not in current_obs.columns:
            continue
        if column_name not in filters:
            continue

        selected_values = filters[column_name]
        if not selected_values:
            continue

        current_obs = _filter_obs_by_values(current_obs, column_name, selected_values)
        n_remaining = len(current_obs)
        print(f"  '{column_name}' filter → {n_remaining:,} events remaining.")

        if n_remaining == 0:
            print(
                "  WARNING: No events remaining after this filter. "
                "Check that the values in 'filters' match actual .obs values."
            )
            break

    if n_events_per_subject > 0:
        current_obs = _subsample_obs_per_subject(current_obs, n_events_per_subject)
        print(
            f"  Sub-sampling → {len(current_obs):,} events "
            f"({n_events_per_subject} per subject)."
        )
    else:
        print("  Sub-sampling: disabled.")

    # Reconstruct AnnData from the kept row positions.
    kept_positions = current_obs["_row_position"].values
    filtered_anndata = anndata_object[kept_positions].copy()
    # With-replacement sampling duplicates rows → obs names are no longer unique.
    # Make them unique so downstream AnnData operations (slicing, concat) stay unambiguous.
    filtered_anndata.obs_names_make_unique()

    if "_row_position" in filtered_anndata.obs.columns:
        del filtered_anndata.obs["_row_position"]

    # Initialise the shared analysis config dict in .uns.
    if _ANALYSIS_CONFIG_KEY not in filtered_anndata.uns:
        filtered_anndata.uns[_ANALYSIS_CONFIG_KEY] = {}

    analysis_cfg: Dict[str, Any] = filtered_anndata.uns[_ANALYSIS_CONFIG_KEY]
    analysis_cfg["selection"] = {col: vals for col, vals in filters.items()}
    analysis_cfg["subsampling"] = {
        "n_events_per_subject": n_events_per_subject,
        "applied": n_events_per_subject > 0,
    }
    analysis_cfg.setdefault("active_layer", None)
    analysis_cfg.setdefault("active_selection", None)

    return filtered_anndata


# ---------------------------------------------------------------------------
# Internal: step 2 — feature selection
# ---------------------------------------------------------------------------


def _apply_feature_selection(
    anndata_object: anndata.AnnData,
    preprocessing_config: dict,
) -> anndata.AnnData:
    """
    Score channels and flag the top-N as selected, non-interactively.

    Replicates the core logic of select_channels() using parameters from
    preprocessing_config instead of console prompts. Writes per-channel
    scores and boolean selection flags into .var, and records the active
    selection key in .uns['analysis_config']['active_selection'].

    CorrFilter always runs before ranking methods when both are requested.
    With multiple ranking methods, the INTERSECTION of all selected sets
    becomes the active selection.

    Arguments:
        anndata_object: AnnData with .uns['analysis_config'] initialised.
        preprocessing_config: Must contain 'feature_selection_methods',
                              'feature_selection_top_n',
                              'feature_selection_target_obs_column',
                              'feature_selection_corr_filter_threshold',
                              'feature_selection_mi_events_per_subject'.

    Returns:
        The same AnnData with new .var columns and updated
        .uns['analysis_config']['active_selection'].
    """
    # Lazy imports to avoid circular dependencies at module load time.
    from mito_marker.analysis.feature_selection import (
        _METHOD_CMI,
        _METHOD_CORR_FILTER,
        _METHOD_HIGH_VARIANCE,
        _METHOD_MIM,
        _METHOD_PCA_LOADINGS,
        _ensure_analysis_config,
        _get_active_data_matrix,
        _get_analytical_mask,
        _score_cmi,
        _score_corr_filter,
        _score_high_variance,
        _score_mim,
        _score_pca_loadings,
        _subsample_by_subject,
    )

    method_names: List[str] = preprocessing_config["feature_selection_methods"]
    top_n: int = preprocessing_config["feature_selection_top_n"]
    target_obs_column: Optional[str] = preprocessing_config[
        "feature_selection_target_obs_column"
    ]
    corr_threshold: float = preprocessing_config["feature_selection_corr_filter_threshold"]
    mi_events_per_subject: int = preprocessing_config["feature_selection_mi_events_per_subject"]

    _ensure_analysis_config(anndata_object)

    if not method_names:
        anndata_object.uns[_ANALYSIS_CONFIG_KEY]["active_selection"] = None
        print("  No feature selection applied. All channels kept.")
        return anndata_object

    # Build the analytical mask (excludes non-analytical channels like Time, FlowAI).
    analytical_mask = _get_analytical_mask(anndata_object)
    n_excluded = int((~analytical_mask).sum())
    if n_excluded > 0:
        excluded_names = [
            name
            for name, keep in zip(anndata_object.var_names, analytical_mask)
            if not keep
        ]
        print(f"  Non-analytical channels excluded from scoring: {excluded_names}")

    full_data_matrix = _get_active_data_matrix(anndata_object)
    analytical_matrix = full_data_matrix[:, analytical_mask]
    analytical_var_names = anndata_object.var_names[analytical_mask].tolist()

    # Pool mask: starts as all-analytical; CorrFilter may narrow it so that
    # subsequent ranking methods score only the surviving channels.
    pool_mask = analytical_mask.copy()

    # Prepare the (possibly subsampled) matrix for MI methods.
    has_mi = any(m in method_names for m in (_METHOD_MIM, _METHOD_CMI))
    analytical_matrix_for_scoring = analytical_matrix
    obs_for_scoring = anndata_object.obs

    if has_mi and mi_events_per_subject > 0:
        print(f"  Subsampling to {mi_events_per_subject} events/subject for MI scoring…")
        analytical_matrix_for_scoring, obs_for_scoring = _subsample_by_subject(
            analytical_matrix, anndata_object.obs, mi_events_per_subject
        )
        print(
            f"  => Scoring on {analytical_matrix_for_scoring.shape[0]:,} events "
            f"(subsampled from {anndata_object.n_obs:,})."
        )

    # CorrFilter always runs before ranking methods.
    ordered_methods: List[str] = (
        [_METHOD_CORR_FILTER] + [m for m in method_names if m != _METHOD_CORR_FILTER]
        if _METHOD_CORR_FILTER in method_names
        else list(method_names)
    )

    for method_name in ordered_methods:
        print(f"  Running {method_name}…")

        if method_name == _METHOD_CORR_FILTER:
            corr_analytical_matrix = full_data_matrix[:, analytical_mask]
            corr_keep_mask = _score_corr_filter(
                corr_analytical_matrix, analytical_var_names, corr_threshold
            )

            scores = np.zeros(anndata_object.n_vars, dtype=np.float64)
            scores[analytical_mask] = corr_keep_mask.astype(np.float64)

            score_col = f"{_METHOD_CORR_FILTER}_score"
            selected_col = f"is_selected_{_METHOD_CORR_FILTER}"
            anndata_object.var[score_col] = scores
            anndata_object.var[selected_col] = scores.astype(bool)

            n_kept = int(corr_keep_mask.sum())
            n_dropped = int((~corr_keep_mask).sum())
            dropped_names = [
                name
                for name, keep in zip(analytical_var_names, corr_keep_mask)
                if not keep
            ]
            print(
                f"  => Correlation threshold {corr_threshold:.0%}: "
                f"{n_kept} channels kept, {n_dropped} dropped."
            )
            if dropped_names:
                print(f"  => Dropped: {dropped_names}")

            # Narrow the pool for subsequent ranking methods.
            pool_mask = np.zeros(anndata_object.n_vars, dtype=bool)
            pool_mask[np.where(analytical_mask)[0][corr_keep_mask]] = True
            continue

        # --- Ranking methods ---
        pool_matrix = full_data_matrix[:, pool_mask]
        pool_var_names = anndata_object.var_names[pool_mask].tolist()
        n_pool = int(pool_mask.sum())

        # For MI methods use the (possibly subsampled) matrix restricted to pool.
        if has_mi:
            pool_matrix_for_scoring = analytical_matrix_for_scoring[
                :, pool_mask[analytical_mask]
            ]
        else:
            pool_matrix_for_scoring = pool_matrix

        if method_name == _METHOD_MIM:
            pool_scores = _score_mim(
                pool_matrix_for_scoring, obs_for_scoring, target_obs_column
            )
        elif method_name == _METHOD_CMI:
            pool_scores = _score_cmi(
                pool_matrix_for_scoring, obs_for_scoring, target_obs_column
            )
        elif method_name == _METHOD_HIGH_VARIANCE:
            pool_scores = _score_high_variance(pool_matrix)
        elif method_name == _METHOD_PCA_LOADINGS:
            pool_scores = _score_pca_loadings(pool_matrix, n_pool)
        else:
            raise ValueError(f"Unknown feature selection method: '{method_name}'")

        # Expand pool scores back to full n_vars space; channels outside pool stay 0.
        scores = np.zeros(anndata_object.n_vars, dtype=np.float64)
        scores[pool_mask] = pool_scores

        score_col = f"{method_name}_score"
        selected_col = f"is_selected_{method_name}"
        anndata_object.var[score_col] = scores

        is_selected = np.zeros(anndata_object.n_vars, dtype=bool)
        pool_indices = np.where(pool_mask)[0]
        top_pool_positions = np.argsort(pool_scores)[::-1][:top_n]
        is_selected[pool_indices[top_pool_positions]] = True
        anndata_object.var[selected_col] = is_selected

        selected_names = anndata_object.var_names[is_selected].tolist()
        dropped_pool_names = [
            pool_var_names[i]
            for i in range(len(pool_var_names))
            if i not in set(top_pool_positions)
        ]
        print(
            f"  => {len(selected_names)} channels selected "
            f"(TOP_N={top_n} from {n_pool} pool channels)."
        )
        if dropped_pool_names:
            print(f"  => Dropped: {dropped_pool_names}")

    # Combine all applied methods into a single active_selection key.
    all_applied = [
        m for m in ordered_methods
        if f"is_selected_{m}" in anndata_object.var.columns
    ]

    if len(all_applied) == 1:
        pipeline_key = all_applied[0]
    elif len(all_applied) > 1:
        combined = np.ones(anndata_object.n_vars, dtype=bool)
        for m in all_applied:
            combined &= anndata_object.var[f"is_selected_{m}"].values
        pipeline_key = "__".join(all_applied)
        anndata_object.var[f"is_selected_{pipeline_key}"] = combined
    else:
        print("  WARNING: No feature selection method produced results.")
        anndata_object.uns[_ANALYSIS_CONFIG_KEY]["active_selection"] = None
        return anndata_object

    anndata_object.uns[_ANALYSIS_CONFIG_KEY]["active_selection"] = pipeline_key

    final_mask = anndata_object.var[f"is_selected_{pipeline_key}"].values
    final_channels = anndata_object.var_names[final_mask].tolist()
    print(f"  => Active selection: '{pipeline_key}'")
    print(f"  => {len(final_channels)} channels selected out of {anndata_object.n_vars}.")
    print(f"  => Selected: {final_channels}")

    return anndata_object


# ---------------------------------------------------------------------------
# Internal: step 3 — transformation + normalization
# ---------------------------------------------------------------------------


def _apply_transform_and_normalize(
    anndata_object: anndata.AnnData,
    preprocessing_config: dict,
) -> anndata.AnnData:
    """
    Apply transformation and normalization, store result in .layers.

    Replicates the logic of transform_and_normalize() without prompts.
    Raw .X is never modified. The processed matrix is stored in a named
    .layers entry whose key encodes both choices (e.g. 'arcsinh__zscore_col').

    When both transform and normalization are 'none', no layer is created
    and .uns['analysis_config']['active_layer'] is set to None.

    Arguments:
        anndata_object: AnnData with .uns['analysis_config'] initialised.
        preprocessing_config: Must contain 'transform' and 'normalization'.

    Returns:
        The same AnnData with the new .layers entry (if applicable) and
        .uns['analysis_config']['active_layer'] updated.
    """
    from mito_marker.analysis.normalization import (
        _build_layer_name,
        _fit_and_apply_layer,
        _get_raw_matrix,
        _print_layer_parameters,
        _print_layer_summary,
        _set_active_layer,
        _store_layer_parameters,
        _warn_if_layer_parameters_missing,
    )

    transform_key: str = preprocessing_config["transform"]
    norm_key: str = preprocessing_config["normalization"]

    if transform_key == "none" and norm_key == "none":
        print("  No transformation or normalization applied.")
        print("  Downstream functions will read directly from .X.")
        _set_active_layer(anndata_object, None)
        return anndata_object

    layer_name = _build_layer_name(transform_key, norm_key)

    if layer_name in anndata_object.layers:
        print(f"  Layer '{layer_name}' already exists — reusing cached result.")
        _set_active_layer(anndata_object, layer_name)
        _warn_if_layer_parameters_missing(anndata_object, layer_name)
        return anndata_object

    print(f"  Applying '{layer_name}'…")

    raw_matrix = _get_raw_matrix(anndata_object)
    # The fitted scaler parameters are kept in .uns['layer_parameters'] so the
    # same scaling can later be replayed on another dataset (ADR-015).
    processed_matrix, layer_parameters = _fit_and_apply_layer(
        raw_matrix, transform_key, norm_key,
        anndata_object.var_names.tolist(), layer_name,
    )

    anndata_object.layers[layer_name] = processed_matrix.astype(np.float32, copy=False)
    _store_layer_parameters(anndata_object, layer_parameters)
    _set_active_layer(anndata_object, layer_name)
    _print_layer_summary(anndata_object.layers[layer_name], layer_name)
    _print_layer_parameters(layer_parameters)

    return anndata_object
