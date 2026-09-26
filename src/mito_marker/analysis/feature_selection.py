"""
feature_selection.py

Interactive channel (feature) selection for SFC AnnData objects.

Offers four methods ranked from simplest to most sophisticated:
  1. None          — keep all channels, skip this step.
  2. MIM           — Mutual Information Maximization [SUPERVISED].
  3. CMI / mRMR    — Min. Redundancy Max. Relevance [SUPERVISED].
  4. High Variance — rank channels by variance [UNSUPERVISED].
  5. PCA Loadings  — rank channels by contribution to principal components
                     [UNSUPERVISED].

The AnnData shape (.n_obs × .n_vars) is never changed by this function.
Results are written into .var as method-prefixed columns:
  .var['MIM_score'], .var['is_selected_MIM']
  .var['CMI_score'], .var['is_selected_CMI']
  .var['HighVariance_score'], .var['is_selected_HighVariance']
  .var['PCALoadings_score'], .var['is_selected_PCALoadings']

The most recently run method is recorded in
.uns['analysis_config']['active_selection'] so downstream plotting functions
automatically use the correct channel subset.

Typical usage:
    from mito_marker.analysis import select_channels
    sfc_subset = select_channels(sfc_subset)
"""

from typing import List, Optional, Tuple

import anndata
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_classif

from mito_marker.analysis.selection import _ANALYSIS_CONFIG_KEY

# Internal method identifiers used as column prefixes in .var.
_METHOD_CORR_FILTER = "CorrFilter"
_METHOD_MIM = "MIM"
_METHOD_CMI = "CMI"
_METHOD_HIGH_VARIANCE = "HighVariance"
_METHOD_PCA_LOADINGS = "PCALoadings"
_METHOD_SHAP_IMPORTANCE = "SHAPImportance"

# Default correlation threshold for the CorrFilter method (95%).
_DEFAULT_CORR_FILTER_THRESHOLD = 0.95

# Number of PCA components used when scoring channels via PCA loadings.
_PCA_LOADINGS_N_COMPONENTS = 20

# .var column that flags channels carrying no biological information (Time,
# FlowAI).  Set at ingestion time by anndata_builder._build_var_dataframe().
_IS_NON_ANALYTICAL_COL = "is_non_analytical"

# Cached result of the one-time GPU availability check.  None means the check
# has not yet run; "cuml" means cupy is importable and a GPU is accessible;
# "sklearn" means we fall back to CPU.
_GPU_BACKEND_CACHE: Optional[str] = None

# .obs columns that are never meaningful supervised targets for feature
# selection (pure identifiers, instrument metadata, or date fields).
_NON_TARGET_OBS_COLUMNS: set[str] = {
    "subject_ID",
    "unique_subject_ID",
    "source_filename",
    "fcs_acquisition_date",
    "fcs_volume_uL",
    "fcs_total_events",
    "tube_name",
}


def _get_analytical_mask(anndata_object: anndata.AnnData) -> np.ndarray:
    """
    Return a boolean mask (length n_vars) that is True for analytically usable channels.

    Reads .var['is_non_analytical'] when the column is present (set at ingestion
    time by anndata_builder._build_var_dataframe()).  Falls back to checking channel
    names against SFC_NON_ANALYTICAL_CHANNELS for AnnData objects that pre-date
    the introduction of this column (e.g. loaded from an older .h5ad file).

    Arguments:
        anndata_object: Any SFC AnnData object.

    Returns:
        Boolean numpy array of length n_vars; True means the channel is eligible
        for scoring and selection.
    """
    if _IS_NON_ANALYTICAL_COL in anndata_object.var.columns:
        return ~anndata_object.var[_IS_NON_ANALYTICAL_COL].values.astype(bool)

    # Fallback: reconstruct the mask from both pipeline lists so that TEM
    # non-analytical features (Mito_CentroidX/Y, Mito_FeretX/Y) are excluded
    # from scoring just as reliably as SFC channels (Time, FlowAI).
    from mito_marker.controlled_vocabulary import (
        SFC_NON_ANALYTICAL_CHANNELS,
        TEM_NON_ANALYTICAL_FEATURES,
    )
    non_analytical_set = set(SFC_NON_ANALYTICAL_CHANNELS) | set(TEM_NON_ANALYTICAL_FEATURES)
    return np.array(
        [name not in non_analytical_set for name in anndata_object.var_names],
        dtype=bool,
    )


def select_channels(anndata_object: anndata.AnnData) -> anndata.AnnData:
    """
    Interactively select the most informative spectral channels in an SFC AnnData.

    Presents a console menu where the user can choose one or more feature
    selection methods and, for supervised methods, a target .obs column. Writes
    per-channel scores and selection flags into .var. The AnnData shape is never
    changed — downstream functions read .var['is_selected_<method>'] to know
    which channels to use.

    Supports multi-method pipelines:
      - Methods are applied in the order the user specifies.
      - If CorrFilter is included, it always runs first regardless of order.
      - Subsequent ranking methods score only the channels that survived
        the CorrFilter (when applicable).
      - With multiple methods, the INTERSECTION of all selected sets is stored
        and becomes the active selection.

    Post-bagging usage (run this cell after run_ml_analysis()):
        sfc_subset = select_channels(sfc_subset)   # pick [6] SHAPImportance
        # SHAP values live in .uns['ml_results']['shap_values']

    Arguments:
        anndata_object: SFC AnnData produced by select_sfc_subset() (or any
                        AnnData with .uns['analysis_config'] initialised).

    Returns:
        The same AnnData object with new .var columns and updated
        .uns['analysis_config']['active_selection'].
    """
    print("=" * 60)
    print("INTERACTIVE CHANNEL (FEATURE) SELECTION")
    print("=" * 60)
    print(
        f"Current dataset: {anndata_object.n_obs:,} events "
        f"× {anndata_object.n_vars} channels"
    )
    print()

    method_names = _prompt_feature_selection_options(anndata_object)

    if not method_names:
        print("=> No feature selection applied. All channels kept.")
        _ensure_analysis_config(anndata_object)
        anndata_object.uns[_ANALYSIS_CONFIG_KEY]["active_selection"] = None
        print("=" * 60)
        return anndata_object

    # Build the base analytical mask (excludes Time, FlowAI channels).
    analytical_mask = _get_analytical_mask(anndata_object)
    n_excluded = int((~analytical_mask).sum())
    if n_excluded > 0:
        excluded_names = [
            name for name, keep in zip(anndata_object.var_names, analytical_mask) if not keep
        ]
        print(f"Non-analytical channels excluded from scoring: {excluded_names}")

    full_data_matrix = _get_active_data_matrix(anndata_object)
    analytical_matrix = full_data_matrix[:, analytical_mask]
    analytical_var_names = anndata_object.var_names[analytical_mask].tolist()

    # Pool mask: starts as all-analytical; CorrFilter may narrow it.
    # downstream ranking methods score only channels in pool_mask.
    pool_mask = analytical_mask.copy()

    # Prompt shared inputs once for all methods.
    has_supervised = any(m in method_names for m in (_METHOD_MIM, _METHOD_CMI))
    target_obs_column: Optional[str] = None
    if has_supervised:
        target_obs_column = _prompt_target_column(anndata_object)

    has_ranking = any(m != _METHOD_CORR_FILTER for m in method_names)
    top_n = 0
    if has_ranking:
        top_n = _prompt_top_n(anndata_object.n_vars)

    # Subsample prompt for MI methods (expensive).
    has_mi = any(m in method_names for m in (_METHOD_MIM, _METHOD_CMI))
    analytical_matrix_for_scoring = analytical_matrix
    obs_for_scoring = anndata_object.obs
    if has_mi:
        use_subsample, events_per_subject = _prompt_subsample_options(
            anndata_object.obs, anndata_object.n_obs
        )
        if use_subsample:
            print("  Subsampling events for MI scoring…")
            analytical_matrix_for_scoring, obs_for_scoring = _subsample_by_subject(
                analytical_matrix, anndata_object.obs, events_per_subject
            )
            print(
                f"  => Scoring on {analytical_matrix_for_scoring.shape[0]:,} events "
                f"(subsampled from {anndata_object.n_obs:,})."
            )

    # Process methods in user order; CorrFilter is hoisted to run first.
    ordered_methods = (
        [_METHOD_CORR_FILTER] + [m for m in method_names if m != _METHOD_CORR_FILTER]
        if _METHOD_CORR_FILTER in method_names
        else list(method_names)
    )

    for method_name in ordered_methods:
        print(f"\nRunning {method_name} feature selection — please wait…")

        if method_name == _METHOD_CORR_FILTER:
            threshold = _prompt_corr_filter_threshold()
            # Score only analytical channels.
            corr_analytical_matrix = full_data_matrix[:, analytical_mask]
            corr_keep_mask = _score_corr_filter(
                corr_analytical_matrix, analytical_var_names, threshold
            )
            # Expand keep_mask to full n_vars; non-analytical stay False.
            scores = np.zeros(anndata_object.n_vars, dtype=np.float64)
            scores[analytical_mask] = corr_keep_mask.astype(np.float64)

            # Write scores and boolean selection (kept=True, dropped=False).
            score_col = f"{_METHOD_CORR_FILTER}_score"
            selected_col = f"is_selected_{_METHOD_CORR_FILTER}"
            anndata_object.var[score_col] = scores
            anndata_object.var[selected_col] = scores.astype(bool)

            n_kept = int(corr_keep_mask.sum())
            n_dropped = int((~corr_keep_mask).sum())
            dropped_names = [
                name for name, keep in zip(analytical_var_names, corr_keep_mask) if not keep
            ]
            print(
                f"  Correlation threshold: {threshold:.0%} — "
                f"{n_kept} channels kept, {n_dropped} dropped."
            )
            if dropped_names:
                print(f"  Dropped channels: {dropped_names}")

            # Narrow the pool for subsequent ranking methods.
            # pool_mask lives in full n_vars space: True = eligible for ranking.
            pool_mask = np.zeros(anndata_object.n_vars, dtype=bool)
            pool_mask[np.where(analytical_mask)[0][corr_keep_mask]] = True
            continue

        # --- Ranking methods ---
        # Score only channels in pool_mask (may be narrowed by CorrFilter).
        pool_matrix = full_data_matrix[:, pool_mask]
        pool_var_names = anndata_object.var_names[pool_mask].tolist()
        n_pool = int(pool_mask.sum())

        # For MI scoring, use the (possibly subsampled) matrix restricted to pool.
        pool_matrix_for_scoring = analytical_matrix_for_scoring[
            :, pool_mask[analytical_mask]
        ] if has_mi else pool_matrix

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
        elif method_name == _METHOD_SHAP_IMPORTANCE:
            all_scores = _score_shap_importance(anndata_object)
            if all_scores is None:
                print(
                    f"  WARNING: SHAPImportance skipped — no SHAP values found. "
                    "Run run_ml_analysis() first."
                )
                continue
            # Restrict to pool channels.
            pool_scores = all_scores[pool_mask]
        else:
            raise ValueError(f"Unknown feature selection method: '{method_name}'")

        # Expand pool scores back to full n_vars; channels outside pool stay 0.
        scores = np.zeros(anndata_object.n_vars, dtype=np.float64)
        scores[pool_mask] = pool_scores

        # Write scores and top_n selection flags (pool_mask used as the eligible set).
        score_col = f"{method_name}_score"
        selected_col = f"is_selected_{method_name}"
        anndata_object.var[score_col] = scores

        is_selected = np.zeros(anndata_object.n_vars, dtype=bool)
        pool_indices = np.where(pool_mask)[0]
        top_pool_positions = np.argsort(pool_scores)[::-1][:top_n]
        is_selected[pool_indices[top_pool_positions]] = True
        anndata_object.var[selected_col] = is_selected

        selected_names = anndata_object.var_names[is_selected].tolist()
        selected_pool_positions = set(top_pool_positions)
        dropped_pool_names = [
            pool_var_names[i]
            for i in range(len(pool_var_names))
            if i not in selected_pool_positions
        ]
        print(
            f"  => {len(selected_names)} channels selected (TOP_N={top_n} "
            f"from {n_pool} pool channels)."
        )
        if dropped_pool_names:
            print(f"  Dropped channels: {dropped_pool_names}")

    # Combine all selections into a single active_selection mask.
    _ensure_analysis_config(anndata_object)

    ranking_methods = [m for m in ordered_methods if m != _METHOD_CORR_FILTER]
    all_applied = [m for m in ordered_methods
                   if f"is_selected_{m}" in anndata_object.var.columns]

    if len(all_applied) == 1:
        pipeline_key = all_applied[0]
    elif len(all_applied) > 1:
        # Intersection: channel must be selected by every applied method.
        combined = np.ones(anndata_object.n_vars, dtype=bool)
        for method_name in all_applied:
            combined &= anndata_object.var[f"is_selected_{method_name}"].values
        pipeline_key = "__".join(all_applied)
        anndata_object.var[f"is_selected_{pipeline_key}"] = combined
    else:
        # No method actually ran (e.g. SHAPImportance skipped).
        print("=> No feature selection was applied.")
        anndata_object.uns[_ANALYSIS_CONFIG_KEY]["active_selection"] = None
        print("=" * 60)
        return anndata_object

    anndata_object.uns[_ANALYSIS_CONFIG_KEY]["active_selection"] = pipeline_key

    final_mask = anndata_object.var[f"is_selected_{pipeline_key}"].values
    final_channels = anndata_object.var_names[final_mask].tolist()
    print(f"\n=> Active selection: '{pipeline_key}'")
    print(f"=> {len(final_channels)} channels selected out of {anndata_object.n_vars}.")
    print(f"=> Selected channels: {final_channels}")
    print("=" * 60)

    return anndata_object


def get_selected_data_matrix(anndata_object: anndata.AnnData) -> np.ndarray:
    """
    Return the data matrix restricted to the currently selected channels.

    Reads the active selection method from .uns['analysis_config']['active_selection'].
    Non-analytical channels (Time, FlowAI) are always excluded regardless of
    whether feature selection is active.

    Arguments:
        anndata_object: AnnData with .uns['analysis_config'] populated.

    Returns:
        2D float numpy array of shape (n_obs, n_selected_channels).
    """
    full_matrix = _get_active_data_matrix(anndata_object)

    config = anndata_object.uns.get(_ANALYSIS_CONFIG_KEY, {})
    active_selection = config.get("active_selection")

    if active_selection is not None:
        selection_col = f"is_selected_{active_selection}"
        if selection_col in anndata_object.var.columns:
            selected_mask = anndata_object.var[selection_col].values
        else:
            selected_mask = np.ones(anndata_object.n_vars, dtype=bool)
    else:
        selected_mask = np.ones(anndata_object.n_vars, dtype=bool)

    # Always exclude non-analytical channels from the returned matrix.
    analytical_mask = _get_analytical_mask(anndata_object)
    combined_mask = selected_mask & analytical_mask
    return full_matrix[:, combined_mask]


# ---------------------------------------------------------------------------
# Console interaction helpers
# ---------------------------------------------------------------------------


def _prompt_feature_selection_options(
    anndata_object: anndata.AnnData,
) -> List[str]:
    """
    Display the feature selection menu and return the user's ordered method list.

    The user may select one or more methods by entering comma-separated numbers
    (e.g. "2", "1,2", "2,3,4"). Methods are applied in the specified order.
    If CorrFilter (1) is included it always runs before any ranking methods.

    Pressing Enter without input keeps all channels and skips feature selection.

    Option [6] (SHAPImportance) is only shown when SHAP values from a prior
    run_ml_analysis() call are present in .uns['ml_results']['shap_values'].

    Returns:
        List of method name strings in user-specified order.
        Empty list means skip (keep all channels).
    """
    shap_available = (
        anndata_object.uns.get("ml_results", {}).get("shap_values") is not None
    )

    print("Feature selection method:")
    print(
        f"  [1] {_METHOD_CORR_FILTER} — Correlation filter"
        "             [UNSUPERVISED — threshold needed]"
    )
    print(
        f"  [2] {_METHOD_MIM} — Mutual Information Maximization"
        "       [SUPERVISED — needs a target label]"
    )
    print(
        f"  [3] {_METHOD_CMI}/mRMR — Min. Redundancy Max. Relevance"
        "  [SUPERVISED — needs a target label]"
    )
    print(
        f"  [4] {_METHOD_HIGH_VARIANCE} — rank channels by variance"
        "             [UNSUPERVISED — no target needed]"
    )
    print(
        f"  [5] {_METHOD_PCA_LOADINGS} — rank channels by PCA contribution"
        "   [UNSUPERVISED — no target needed]"
    )
    if shap_available:
        print(
            f"  [6] {_METHOD_SHAP_IMPORTANCE} — rank by mean |SHAP| from ML run"
            "   [requires prior run_ml_analysis()]"
        )
    print("Default (press Enter): keep all channels, skip this step")

    n_options = 6 if shap_available else 5
    method_map = {
        "1": _METHOD_CORR_FILTER,
        "2": _METHOD_MIM,
        "3": _METHOD_CMI,
        "4": _METHOD_HIGH_VARIANCE,
        "5": _METHOD_PCA_LOADINGS,
        "6": _METHOD_SHAP_IMPORTANCE,
    }

    while True:
        raw_response = input(
            f"Select methods (e.g. 2 or 1,2 or 2,3,4), or press Enter to skip: "
        ).strip()

        if raw_response == "":
            return []

        try:
            chosen_numbers = [token.strip() for token in raw_response.split(",")]
        except ValueError:
            print(f"  Invalid input '{raw_response}'. Enter numbers separated by commas.")
            continue

        invalid = [n for n in chosen_numbers if n not in method_map]
        if invalid:
            print(
                f"  Invalid choices: {invalid}. "
                f"Enter numbers between 1 and {n_options}."
            )
            continue

        if not shap_available and "6" in chosen_numbers:
            print(
                "  Option [6] SHAPImportance is not available — "
                "run run_ml_analysis() first."
            )
            continue

        # Deduplicate while preserving order.
        seen: set[str] = set()
        selected_methods: List[str] = []
        for number in chosen_numbers:
            method = method_map[number]
            if method not in seen:
                selected_methods.append(method)
                seen.add(method)

        return selected_methods


def _prompt_corr_filter_threshold() -> float:
    """
    Ask the user for the Pearson correlation threshold used by CorrFilter.

    Any pair of channels with absolute Pearson correlation above this threshold
    is considered redundant; one of them is dropped.

    Returns:
        Float threshold in (0, 1]. Default is _DEFAULT_CORR_FILTER_THRESHOLD (0.95).
    """
    default_pct = int(_DEFAULT_CORR_FILTER_THRESHOLD * 100)
    while True:
        raw_response = input(
            f"Correlation threshold % (press Enter for {default_pct}%): "
        ).strip()
        if raw_response == "":
            return _DEFAULT_CORR_FILTER_THRESHOLD
        try:
            value = float(raw_response)
            # Accept both 0–1 and 0–100 input.
            if value > 1.0:
                value = value / 100.0
            if 0.0 < value <= 1.0:
                return value
            print(f"  Threshold must be between 1 and 100 (or 0.01 and 1.0).")
        except ValueError:
            print(f"  Invalid value '{raw_response}'. Enter a number like 95 or 0.95.")


def _prompt_target_column(anndata_object: anndata.AnnData) -> str:
    """
    Ask the user which .obs column to use as the classification target for
    supervised feature selection. Only categorical and boolean columns are
    offered; numerical columns are offered with a warning that they will be
    binned into 3 quantile groups.

    Returns:
        The name of the selected .obs column.
    """
    from mito_marker.controlled_vocabulary import FILTERABLE_OBS_COLUMNS

    # Build candidates in two passes so the menu is stable and predictable:
    #   1. Declared filterable columns first, in their canonical order.
    #   2. Any remaining .obs column not in the exclusion set with > 1 unique
    #      value — this captures dynamically derived groupings such as
    #      age_group, insulin_level_uU_per_mL_tertiles, _split columns, etc.
    seen: set[str] = set()
    candidate_columns: list[str] = []

    for col in FILTERABLE_OBS_COLUMNS:
        if col in anndata_object.obs.columns and col not in _NON_TARGET_OBS_COLUMNS:
            candidate_columns.append(col)
            seen.add(col)

    for col in anndata_object.obs.columns:
        if col not in seen and col not in _NON_TARGET_OBS_COLUMNS:
            if anndata_object.obs[col].nunique() > 1:
                candidate_columns.append(col)

    if not candidate_columns:
        raise ValueError(
            "No filterable .obs columns found to use as a supervised target. "
            "Run select_sfc_subset() first."
        )

    print("\nTarget label for supervised feature selection:")
    for idx, col in enumerate(candidate_columns, start=1):
        dtype_note = ""
        if pd.api.types.is_float_dtype(anndata_object.obs[col]):
            dtype_note = " [numerical → will be auto-binned into 3 groups]"
        n_unique = anndata_object.obs[col].nunique()
        print(f"  [{idx}] {col}  ({n_unique} unique values){dtype_note}")

    while True:
        raw_response = input("Select target column: ").strip()
        try:
            choice = int(raw_response)
            if 1 <= choice <= len(candidate_columns):
                selected_column = candidate_columns[choice - 1]
                break
        except ValueError:
            pass
        print(
            f"  Invalid choice '{raw_response}'. "
            f"Enter a number between 1 and {len(candidate_columns)}."
        )

    return selected_column


def _prompt_top_n(n_vars: int) -> int:
    """
    Ask the user how many channels to keep (TOP_N).

    Returns:
        Integer number of channels to select (between 1 and n_vars).
    """
    default = min(40, n_vars)
    while True:
        raw_response = input(
            f"Enter TOP_N — number of channels to select (default {default}): "
        ).strip()
        if raw_response == "":
            return default
        try:
            top_n = int(raw_response)
            if 1 <= top_n <= n_vars:
                return top_n
            print(f"  TOP_N must be between 1 and {n_vars}.")
        except ValueError:
            print(f"  Invalid value '{raw_response}'. Enter a whole number.")


def _prompt_subsample_options(
    obs_dataframe: pd.DataFrame,
    n_obs: int,
) -> Tuple[bool, int]:
    """
    Ask the user whether to subsample events before mutual information scoring.

    Subsampling draws an equal, random number of events from each subject_ID,
    keeping class balance intact while reducing the wall time of the KNN or
    histogram-based MI computation on large datasets (>100k events).

    Arguments:
        obs_dataframe: The .obs DataFrame of the AnnData.
        n_obs: Total number of events currently in the dataset.

    Returns:
        Tuple of (use_subsample, events_per_subject).
        events_per_subject is 0 when use_subsample is False.
    """
    print("\nSubsampling (recommended for datasets > 100k events):")

    if "subject_ID" in obs_dataframe.columns:
        n_subjects = int(obs_dataframe["subject_ID"].nunique())
        avg_events_per_subject = n_obs // max(n_subjects, 1)
        default_per_subject = min(5_000, avg_events_per_subject)
        upper_bound = avg_events_per_subject
        print(
            f"  {n_obs:,} events across {n_subjects} subjects "
            f"(~{avg_events_per_subject:,} events / subject on average)."
        )
        print(
            "  Subsampling draws the same number of events from each subject, "
            "chosen at random (equal + random per subject)."
        )
    else:
        n_subjects = 1
        default_per_subject = min(50_000, n_obs)
        upper_bound = n_obs
        print(
            f"  {n_obs:,} events. No 'subject_ID' column found — "
            "subsampling will be a simple random draw."
        )

    while True:
        raw = input("  Enable subsampling? [y/n] (default: n): ").strip().lower()
        if raw in ("", "n", "no"):
            print("  => No subsampling. MI will be computed on all events.")
            return False, 0
        if raw in ("y", "yes"):
            break
        print(f"  Invalid input '{raw}'. Enter 'y' or 'n'.")

    while True:
        raw = input(
            f"  Events per subject (1 – {upper_bound:,}, "
            f"default {default_per_subject:,}): "
        ).strip()
        if raw == "":
            events_per_subject = default_per_subject
            break
        try:
            events_per_subject = int(raw)
            if 1 <= events_per_subject <= upper_bound:
                break
            print(f"  Enter a value between 1 and {upper_bound:,}.")
        except ValueError:
            print(f"  Invalid value '{raw}'. Enter a whole number.")

    estimated_total = events_per_subject * n_subjects
    print(
        f"  => Subsampling to ~{estimated_total:,} events "
        f"({events_per_subject:,} per subject)."
    )
    return True, events_per_subject


def _subsample_by_subject(
    data_matrix: np.ndarray,
    obs_dataframe: pd.DataFrame,
    events_per_subject: int,
    random_seed: int = 42,
) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Draw a stratified random subsample — equal events per subject_ID.

    Each subject contributes at most `events_per_subject` rows, chosen
    uniformly at random without replacement.  Subjects with fewer events
    than the requested quota contribute all of their rows.  The returned
    rows are sorted in ascending row order to preserve the original event
    ordering.

    Arguments:
        data_matrix: 2D float array (n_obs, n_vars).
        obs_dataframe: The .obs DataFrame; should contain a 'subject_ID' column.
        events_per_subject: Maximum number of events to draw per subject.
        random_seed: Random seed passed to numpy for reproducibility.

    Returns:
        Tuple of (subsampled_data_matrix, subsampled_obs_dataframe).
        Both are restricted to the sampled rows in ascending row order.
    """
    rng = np.random.default_rng(random_seed)

    if "subject_ID" not in obs_dataframe.columns:
        # Graceful fallback: simple random draw when there is no subject column.
        n_take = min(events_per_subject * 10, len(obs_dataframe))
        sampled_positions = np.sort(
            rng.choice(len(obs_dataframe), size=n_take, replace=False)
        )
        return data_matrix[sampled_positions], obs_dataframe.iloc[sampled_positions]

    # Reset index to obtain clean 0-based integer positions into data_matrix.
    obs_reset = obs_dataframe.reset_index(drop=True)
    sampled_positions_list: List[int] = []

    for _subject_id, subject_group in obs_reset.groupby("subject_ID"):
        n_available = len(subject_group)
        n_take = min(events_per_subject, n_available)
        chosen = rng.choice(
            subject_group.index.to_numpy(), size=n_take, replace=False
        )
        sampled_positions_list.extend(chosen.tolist())

    sampled_positions = np.sort(sampled_positions_list)
    return data_matrix[sampled_positions], obs_dataframe.iloc[sampled_positions]


# ---------------------------------------------------------------------------
# Scoring functions (one per method)
# ---------------------------------------------------------------------------


def _score_corr_filter(
    analytical_matrix: np.ndarray,
    var_names: List[str],
    threshold: float,
) -> np.ndarray:
    """
    Build a boolean keep-mask by removing highly correlated channels.

    Computes the absolute Pearson correlation matrix over the analytical channels,
    then iteratively marks one channel from each correlated pair (above threshold)
    for removal — the convention is to keep the first (left) column and drop the
    second (right) column, matching the upper-triangle scan order.

    Arguments:
        analytical_matrix: 2D float array (n_obs, n_analytical_channels).
        var_names: Channel names in column order, used as DataFrame index.
        threshold: Absolute Pearson correlation above which a channel is dropped.
                   Must be in (0, 1].

    Returns:
        Boolean numpy array of length n_analytical_channels.
        True = channel is kept; False = channel is dropped as redundant.
    """
    import warnings

    channel_dataframe = pd.DataFrame(analytical_matrix, columns=var_names)
    # Suppress potential warnings from correlating constant columns.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        corr_matrix = channel_dataframe.corr().abs()

    # Replace NaN correlations (constant columns) with 0 — a constant channel
    # has no correlation with anything and should be kept (or dropped by
    # HighVariance), not incorrectly labelled as redundant.
    corr_matrix = corr_matrix.fillna(0.0)

    # Scan the upper triangle: if corr(i, j) > threshold, mark j for removal.
    upper_triangle = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )
    to_drop: set[str] = {
        col for col in upper_triangle.columns
        if any(upper_triangle[col] > threshold)
    }

    keep_mask = np.array([name not in to_drop for name in var_names], dtype=bool)
    return keep_mask


def _score_shap_importance(
    anndata_object: anndata.AnnData,
) -> Optional[np.ndarray]:
    """
    Compute per-channel importance scores from previously computed SHAP values.

    Reads .uns['ml_results']['shap_values'] (stored by run_ml_analysis()) and
    computes mean(|SHAP|) per channel. For multiclass problems (3D shap_matrix),
    the absolute values are averaged first across classes, then across events.

    The returned scores are in full .var order (length = n_vars). Channels that
    were not included in the ML run (because they were excluded from the active
    selection at that time) receive a score of 0.

    Arguments:
        anndata_object: AnnData with .uns['ml_results'] populated by
                        run_ml_analysis().

    Returns:
        1D float64 array of length n_vars, or None if SHAP values are unavailable.
    """
    import warnings as _warnings

    ml_results = anndata_object.uns.get("ml_results", {})
    shap_matrix = ml_results.get("shap_values")
    shap_feature_names = ml_results.get("shap_feature_names")

    if shap_matrix is None:
        _warnings.warn(
            "SHAPImportance: no SHAP values found in .uns['ml_results']. "
            "Run run_ml_analysis() with compute_shap=True before calling "
            "select_channels() with this method.",
            UserWarning,
            stacklevel=3,
        )
        return None

    if shap_feature_names is None:
        _warnings.warn(
            "SHAPImportance: 'shap_feature_names' missing from .uns['ml_results']. "
            "Cannot map SHAP scores back to channels.",
            UserWarning,
            stacklevel=3,
        )
        return None

    shap_array = np.asarray(shap_matrix, dtype=np.float64)

    # 3D shap_matrix: (n_obs, n_features, n_classes) — multiclass.
    # Average |SHAP| across classes first, then across events.
    if shap_array.ndim == 3:
        feature_importance = np.mean(np.abs(shap_array), axis=(0, 2))
    elif shap_array.ndim == 2:
        # 2D: (n_obs, n_features) — binary or regression.
        feature_importance = np.mean(np.abs(shap_array), axis=0)
    else:
        _warnings.warn(
            f"SHAPImportance: unexpected shap_matrix shape {shap_array.shape}. "
            "Expected 2D (n_obs, n_features) or 3D (n_obs, n_features, n_classes).",
            UserWarning,
            stacklevel=3,
        )
        return None

    # Map SHAP feature names back to full .var_names positions.
    var_name_to_idx = {name: idx for idx, name in enumerate(anndata_object.var_names)}
    scores = np.zeros(anndata_object.n_vars, dtype=np.float64)
    for feat_idx, feat_name in enumerate(shap_feature_names):
        if feat_name in var_name_to_idx:
            scores[var_name_to_idx[feat_name]] = feature_importance[feat_idx]

    return scores


def _score_mim(
    data_matrix: np.ndarray,
    obs_dataframe: pd.DataFrame,
    target_obs_column: str,
) -> np.ndarray:
    """
    Score each channel using Mutual Information Maximization (MIM).

    Computes mutual_info_classif(X, y) where y is the target .obs column.
    If y is numerical (float dtype), it is auto-binned into 3 quantile groups
    before scoring.

    Arguments:
        data_matrix: 2D float array (n_obs, n_vars).
        obs_dataframe: The .obs DataFrame of the AnnData.
        target_obs_column: Name of the .obs column used as the class label.

    Returns:
        1D float array of MI scores, one per channel, in .var order.
    """
    target_labels = _prepare_target_labels(obs_dataframe, target_obs_column)

    backend = _detect_gpu_backend()
    if backend == "cuml":
        print("  GPU detected — using cupy histogram-based MI (fast).")
        return _score_mim_cuml(data_matrix, target_labels)

    print(
        f"  Falling back to sklearn KNN-based MI on CPU "
        f"(n_jobs=-1, {data_matrix.shape[0]:,} events)."
    )
    scores = mutual_info_classif(
        data_matrix, target_labels, random_state=42, n_jobs=-1
    )
    return scores


def _score_cmi(
    data_matrix: np.ndarray,
    obs_dataframe: pd.DataFrame,
    target_obs_column: str,
) -> np.ndarray:
    """
    Score each channel using Conditional Mutual Information / mRMR.

    Implements a greedy mRMR (minimum Redundancy Maximum Relevance) loop:
      score(candidate) = MI(candidate, target) − mean_correlation(candidate, selected)
    The channel with the highest score is selected at each iteration.

    Dispatches to a cupy GPU path when a GPU is available; falls back to a
    vectorized numpy CPU path otherwise.

    Inspired by select_features_cmi() in the Stage_M2_4_Restore_Data_Mito.ipynb
    reference notebook; adapted to return per-channel scores for all channels
    (not just the top_n), enabling storage in .var.

    Arguments:
        data_matrix: 2D float array (n_obs, n_vars).
        obs_dataframe: The .obs DataFrame of the AnnData.
        target_obs_column: Name of the .obs column used as the class label.

    Returns:
        1D float array of CMI/mRMR selection order scores (higher = selected
        earlier). Channels never reached by the greedy loop receive a score of 0.
    """
    target_labels = _prepare_target_labels(obs_dataframe, target_obs_column)

    backend = _detect_gpu_backend()
    if backend == "cuml":
        print("  GPU detected — using cupy for CMI relevance, correlation, and selection.")
        return _score_cmi_cuml(data_matrix, target_labels)

    print(
        f"  Falling back to numpy/sklearn CMI on CPU "
        f"(n_jobs=-1, {data_matrix.shape[0]:,} events)."
    )

    n_vars = data_matrix.shape[1]

    # Step 1: Relevance — MI(feature_i, target) for all channels.
    relevance = mutual_info_classif(
        data_matrix, target_labels, random_state=42, n_jobs=-1
    )

    # Step 2: Absolute Pearson correlation matrix; NaN for constant columns → 0.
    channel_dataframe = pd.DataFrame(data_matrix)
    correlation_matrix = np.nan_to_num(
        channel_dataframe.corr().abs().values, nan=0.0
    )

    # Step 3: Greedy mRMR loop — vectorized over candidates with numpy.
    # For each rank, score all remaining candidates at once instead of looping
    # over them in Python; then pick the argmax.
    remaining_mask = np.ones(n_vars, dtype=bool)
    order_scores = np.zeros(n_vars, dtype=np.float64)
    selected_indices: List[int] = []

    for rank in range(n_vars, 0, -1):
        if not selected_indices:
            mrmr_scores = relevance.copy()
        else:
            selected_arr = np.array(selected_indices)
            # Mean absolute correlation with all already-selected features.
            redundancy = correlation_matrix[:, selected_arr].mean(axis=1)  # (n_vars,)
            mrmr_scores = relevance - redundancy

        # Mask out already-selected channels before argmax.
        mrmr_scores[~remaining_mask] = -np.inf
        best_idx = int(np.argmax(mrmr_scores))

        order_scores[best_idx] = rank
        selected_indices.append(best_idx)
        remaining_mask[best_idx] = False

    return order_scores


def _score_cmi_cuml(
    data_matrix: np.ndarray,
    target_labels: np.ndarray,
) -> np.ndarray:
    """
    GPU-accelerated CMI/mRMR scoring using cupy.

    Three steps run entirely on the GPU:
      1. Relevance — MI(feature_i, target) via histogram-based estimation
         (reuses _score_mim_cuml, same method as the GPU MIM path).
      2. Correlation matrix — absolute Pearson correlations computed via a
         single matrix multiplication (much faster than pandas .corr()).
      3. Greedy mRMR selection loop — vectorized over all candidate channels
         using cupy argmax, eliminating the O(n_features²) Python inner loop.

    Arguments:
        data_matrix: 2D float array (n_obs, n_vars), on CPU.
        target_labels: 1D string array of class labels, length n_obs.

    Returns:
        1D float64 numpy array of mRMR selection order scores (on CPU).
    """
    import cupy as cp

    n_obs, n_vars = data_matrix.shape

    # Step 1: Relevance on GPU — reuse the histogram MI scorer.
    print("    [CMI 1/3] MI relevance on GPU…")
    relevance = cp.asarray(_score_mim_cuml(data_matrix, target_labels), dtype=cp.float64)

    # Step 2: Absolute Pearson correlation matrix on GPU via matrix multiply.
    # Standardise each column (zero mean, unit std) then compute X^T X / n.
    print("    [CMI 2/3] Correlation matrix on GPU…")
    X_gpu = cp.asarray(data_matrix, dtype=cp.float32)
    col_mean = X_gpu.mean(axis=0)                       # (n_vars,)
    col_std = X_gpu.std(axis=0)                         # (n_vars,)
    # Constant columns (std=0) contribute no redundancy — set std to 1 so
    # the standardised column becomes all-zeros, giving correlation = 0.
    col_std = cp.where(col_std == 0, cp.float32(1.0), col_std)
    X_standardized = (X_gpu - col_mean) / col_std      # (n_obs, n_vars)
    correlation_matrix = cp.abs(
        X_standardized.T @ X_standardized
    ).astype(cp.float64) / n_obs                        # (n_vars, n_vars)

    # Step 3: Greedy mRMR selection loop, vectorized over candidates.
    print("    [CMI 3/3] Greedy mRMR selection loop on GPU…")
    remaining_mask = cp.ones(n_vars, dtype=bool)
    order_scores = np.zeros(n_vars, dtype=np.float64)
    selected_indices: List[int] = []

    for rank in range(n_vars, 0, -1):
        if not selected_indices:
            mrmr_scores = relevance.copy()
        else:
            selected_arr = cp.array(selected_indices, dtype=cp.int32)
            # Mean absolute correlation with all already-selected features.
            redundancy = correlation_matrix[:, selected_arr].mean(axis=1)  # (n_vars,)
            mrmr_scores = relevance - redundancy

        # Mask out already-selected channels before argmax.
        mrmr_scores = cp.where(remaining_mask, mrmr_scores, cp.float64(-cp.inf))
        best_idx = int(cp.argmax(mrmr_scores).item())

        order_scores[best_idx] = rank
        selected_indices.append(best_idx)
        remaining_mask[best_idx] = False

    return order_scores


def _detect_gpu_backend() -> str:
    """
    Check whether cupy is installed and a GPU is accessible.

    The result is cached in _GPU_BACKEND_CACHE so the probe runs only once per
    Python session.

    Returns:
        'cuml' when cupy is importable and can allocate a GPU array;
        'sklearn' otherwise (no cupy, or no GPU driver).
    """
    global _GPU_BACKEND_CACHE
    if _GPU_BACKEND_CACHE is not None:
        return _GPU_BACKEND_CACHE
    try:
        import cupy as cp  # noqa: F401

        cp.array([1.0])  # Raises if no GPU driver is present.
        _GPU_BACKEND_CACHE = "cuml"
    except ImportError:
        # cupy is not installed — GPU hardware may still be present.
        print(
            "  cupy is not installed. To enable GPU-accelerated MI on Colab:\n"
            "    !pip install cupy-cuda12x   # H100 / A100 / L4 (CUDA 12)\n"
            "    !pip install cupy-cuda11x   # T4 / older (CUDA 11)\n"
            "  Then restart the runtime and re-run this cell."
        )
        _GPU_BACKEND_CACHE = "sklearn"
    except Exception:
        # cupy is installed but no GPU is accessible (CPU-only runtime).
        _GPU_BACKEND_CACHE = "sklearn"
    return _GPU_BACKEND_CACHE


def _score_mim_cuml(
    data_matrix: np.ndarray,
    target_labels: np.ndarray,
    n_bins: int = 50,
) -> np.ndarray:
    """
    GPU-accelerated MI scoring using cupy and equal-width histogram binning.

    Computes MI(X_i, Y) = H(X_i) + H(Y) - H(X_i, Y) for each feature column,
    estimating all distributions from joint and marginal histograms.  Absolute
    MI values differ from sklearn's KNN-based estimator, but the ranking across
    channels is equivalent for feature selection purposes.

    Runs entirely on GPU memory; transfers only the final scores array back to
    the CPU.

    Arguments:
        data_matrix: 2D float array (n_obs, n_vars), on CPU.
        target_labels: 1D string array of class labels, length n_obs.
        n_bins: Number of equal-width bins used to discretize each feature
                column.  50 is a good default for cytometry data.

    Returns:
        1D float64 numpy array of MI scores, one per channel, in .var order.
    """
    import cupy as cp

    X_gpu = cp.asarray(data_matrix, dtype=cp.float32)
    n_obs, n_features = X_gpu.shape

    # Encode string class labels to consecutive integers on the CPU, then send
    # to the GPU once (reused across all features).
    unique_labels, label_integers = np.unique(target_labels, return_inverse=True)
    n_classes = len(unique_labels)
    y_gpu = cp.asarray(label_integers, dtype=cp.int32)

    # Marginal distribution of Y — identical for every feature; compute once.
    marginal_y_gpu = cp.asarray(
        np.bincount(label_integers, minlength=n_classes) / n_obs,
        dtype=cp.float64,
    )

    # Bin ALL features at once: shape (n_obs, n_features).
    col_min = X_gpu.min(axis=0)                       # (n_features,)
    col_max = X_gpu.max(axis=0)                       # (n_features,)
    col_range = col_max - col_min
    # Constant features get range=0 → set to 1 so division is safe; their MI
    # will naturally come out as 0 because all events fall into the same bin.
    col_range = cp.where(col_range == 0, cp.float32(1.0), col_range)

    bin_indices = cp.floor(
        (X_gpu - col_min) / col_range * (n_bins - 1)
    ).clip(0, n_bins - 1).astype(cp.int32)            # (n_obs, n_features)

    scores = np.zeros(n_features, dtype=np.float64)

    for feature_idx in range(n_features):
        feat_bins = bin_indices[:, feature_idx]        # (n_obs,) on GPU

        # Flatten (bin, class) into a single integer for a fast GPU bincount.
        # Values range from 0 to n_bins * n_classes - 1.
        joint_flat = feat_bins * n_classes + y_gpu    # (n_obs,)
        joint_counts = cp.bincount(joint_flat, minlength=n_bins * n_classes)
        joint_proba = (
            joint_counts.reshape(n_bins, n_classes).astype(cp.float64) / n_obs
        )                                              # (n_bins, n_classes)

        marginal_x = joint_proba.sum(axis=1)           # (n_bins,)
        outer = cp.outer(marginal_x, marginal_y_gpu)   # (n_bins, n_classes)

        # MI = Σ_{x,y} p(x,y) · log[ p(x,y) / (p(x)·p(y)) ]
        # Skip zero-probability cells to avoid log(0).
        nonzero = (joint_proba > 0) & (outer > 0)
        mi_value = float(
            cp.sum(
                joint_proba[nonzero] * cp.log(joint_proba[nonzero] / outer[nonzero])
            ).item()
        )
        # MI is non-negative by definition; clamp floating-point underflow.
        scores[feature_idx] = max(mi_value, 0.0)

    return scores


def _score_high_variance(data_matrix: np.ndarray) -> np.ndarray:
    """
    Score each channel by its variance across all events.

    Higher variance means the channel spreads events more — useful for
    distinguishing subpopulations without needing a class label.

    Arguments:
        data_matrix: 2D float array (n_obs, n_vars).

    Returns:
        1D float array of per-channel variance values.
    """
    # Compute variance in the input dtype (typically float32), then upcast only
    # the 1-D result.  Using dtype=np.float64 here would force numpy to
    # materialise a full float64 copy of the matrix (~12 GB for 10 M events),
    # which causes OOM on Colab and can corrupt AnnData's cached .obs arrays.
    return np.var(data_matrix, axis=0).astype(np.float64)


def _score_pca_loadings(data_matrix: np.ndarray, n_vars: int) -> np.ndarray:
    """
    Score each channel by its summed absolute contribution to the top PCA components.

    Fits PCA on the data matrix, then for each channel sums the absolute values
    of its loadings across all retained components. Channels that drive the
    directions of maximum variance receive the highest scores.

    Arguments:
        data_matrix: 2D float array (n_obs, n_vars).
        n_vars: Total number of channels (used to cap n_components).

    Returns:
        1D float array of summed absolute loading scores, one per channel.
    """
    n_components = min(_PCA_LOADINGS_N_COMPONENTS, n_vars, data_matrix.shape[0])
    pca = PCA(n_components=n_components, random_state=42)
    pca.fit(data_matrix)
    # pca.components_ shape: (n_components, n_vars)
    # Weight each component's absolute loadings by its explained variance ratio
    # so that channels driving high-variance directions score higher than
    # channels only present in low-variance residual components.
    weighted_loadings = (
        np.abs(pca.components_)
        * pca.explained_variance_ratio_[:, np.newaxis]
    )
    loading_scores = weighted_loadings.sum(axis=0)
    return loading_scores


# ---------------------------------------------------------------------------
# Target label preparation
# ---------------------------------------------------------------------------


def _prepare_target_labels(
    obs_dataframe: pd.DataFrame,
    target_obs_column: str,
) -> np.ndarray:
    """
    Extract and prepare the class label array from .obs for supervised methods.

    If the column has a float dtype (e.g. age), auto-bins into 3 quantile groups
    and prints a warning. Categorical and boolean columns are used directly as
    string labels.

    Arguments:
        obs_dataframe: The .obs DataFrame of the AnnData.
        target_obs_column: Name of the .obs column to use as y.

    Returns:
        1D string array of class labels, length n_obs.
    """
    column = obs_dataframe[target_obs_column]

    if pd.api.types.is_float_dtype(column):
        print(
            f"  WARNING: '{target_obs_column}' is numerical. "
            "Auto-binning into 3 quantile groups for mutual information scoring."
        )
        # pd.qcut creates equal-frequency bins; labels become group identifiers.
        binned = pd.qcut(column.rank(method="first"), q=3, labels=["low", "mid", "high"])
        return binned.astype(str).values

    return column.astype(str).values


# ---------------------------------------------------------------------------
# .var writing and helpers
# ---------------------------------------------------------------------------


def _write_selection_to_var(
    anndata_object: anndata.AnnData,
    method_name: str,
    scores: np.ndarray,
    top_n: int,
    analytical_mask: Optional[np.ndarray] = None,
) -> None:
    """
    Write per-channel scores and selection flags into .var.

    Adds two method-prefixed columns:
      .var['{method}_score']       — float score for each channel.
      .var['is_selected_{method}'] — bool True for the top_n channels.

    Non-analytical channels (Time, FlowAI) are forced to is_selected=False
    regardless of their score, even when analytical_mask is not provided.

    Arguments:
        anndata_object: AnnData to update in-place.
        method_name: One of the _METHOD_* constants.
        scores: 1D float array of scores, one per channel in .var order.
        top_n: Number of channels to mark as selected.
        analytical_mask: Optional boolean array (length n_vars) that is True
            for channels eligible for selection. When provided, only those
            channels are ranked — non-analytical positions are skipped
            regardless of their score value.
    """
    score_col = f"{method_name}_score"
    selected_col = f"is_selected_{method_name}"

    anndata_object.var[score_col] = scores.astype(np.float64)

    is_selected = np.zeros(anndata_object.n_vars, dtype=bool)

    if analytical_mask is not None:
        # Rank only analytical channels; non-analytical are never selected.
        analytical_indices = np.where(analytical_mask)[0]
        analytical_scores = scores[analytical_mask]
        top_analytical = analytical_indices[np.argsort(analytical_scores)[::-1][:top_n]]
        is_selected[top_analytical] = True
    else:
        top_indices = np.argsort(scores)[::-1][:top_n]
        is_selected[top_indices] = True
        # Belt-and-suspenders: force non-analytical channels to False even if
        # their score accidentally ranked them in the top_n.
        fallback_analytical_mask = _get_analytical_mask(anndata_object)
        is_selected &= fallback_analytical_mask

    anndata_object.var[selected_col] = is_selected


def _get_active_data_matrix(anndata_object: anndata.AnnData) -> np.ndarray:
    """
    Return the data matrix from the active layer, or .X if none is set.

    Reads .uns['analysis_config']['active_layer']; falls back to .X when the
    key is absent or None.

    Arguments:
        anndata_object: AnnData with optional .uns['analysis_config'].

    Returns:
        2D float numpy array (n_obs, n_vars).
    """
    config = anndata_object.uns.get(_ANALYSIS_CONFIG_KEY, {})
    active_layer = config.get("active_layer")

    if active_layer is not None and active_layer in anndata_object.layers:
        layer_data = anndata_object.layers[active_layer]
        if hasattr(layer_data, "toarray"):
            return layer_data.toarray()
        return np.asarray(layer_data)

    # Ensure we always return a dense numpy array (not a sparse matrix).
    x_data = anndata_object.X
    if hasattr(x_data, "toarray"):
        return x_data.toarray()
    return np.asarray(x_data)


def _ensure_analysis_config(anndata_object: anndata.AnnData) -> None:
    """Initialise .uns['analysis_config'] if not already present."""
    if _ANALYSIS_CONFIG_KEY not in anndata_object.uns:
        anndata_object.uns[_ANALYSIS_CONFIG_KEY] = {}
    anndata_object.uns[_ANALYSIS_CONFIG_KEY].setdefault("active_layer", None)
    anndata_object.uns[_ANALYSIS_CONFIG_KEY].setdefault("active_selection", None)
