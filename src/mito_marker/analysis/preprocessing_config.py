"""
preprocessing_config.py

Preprocessing configuration for the mito-marker analysis pipeline.

Provides two ways to build the preprocessing_config dict that controls the
first three steps of the analysis pipeline — subset selection, feature
selection, and transformation/normalization:

  get_default_preprocessing_config()
      Returns a fully-populated dict with safe default values.
      Intended for programmatic use: call it, then override the keys you need.
      Pass the result to run_preprocessing() to apply all three steps at once
      without any interactive prompts.

  configure_preprocessing(anndata_object)
      Interactive console wizard (same numbered-menu UX as select_sfc_subset(),
      select_channels(), and transform_and_normalize()). Presents the same
      menus in sequence, then returns a validated preprocessing_config dict.

Typical usage:
    from mito_marker.analysis.preprocessing_config import get_default_preprocessing_config
    from mito_marker.analysis.preprocessing_pipeline import run_preprocessing

    # Option A: programmatic (no prompts)
    config = get_default_preprocessing_config()
    config["filters"]                         = {"condition": ["young"]}
    config["transform"]                       = "arcsinh"
    config["normalization"]                   = "zscore_col"
    config["feature_selection_methods"]       = ["MIM"]
    config["feature_selection_top_n"]         = 20
    config["feature_selection_target_obs_column"] = "condition"

    sfc_subset = run_preprocessing(sfc_anndata, config)

    # Option B: interactive wizard
    config = configure_preprocessing(sfc_anndata)
    sfc_subset = run_preprocessing(sfc_anndata, config)
"""

from typing import Dict, List, Optional

import anndata

# ---------------------------------------------------------------------------
# Allowed values — used both for validation and for displaying menus
# ---------------------------------------------------------------------------

ALLOWED_TRANSFORMS: List[str] = ["none", "arcsinh", "logicle"]

ALLOWED_NORMALIZATIONS: List[str] = [
    "none",
    "zscore_col",
    "minmax_col",
    "l2norm_row",
    "zscore_col__l2norm_row",
]

ALLOWED_FEATURE_SELECTION_METHODS: List[str] = [
    "CorrFilter",
    "MIM",
    "CMI",
    "HighVariance",
    "PCALoadings",
]

# One-line description for each transform shown in the interactive wizard.
_TRANSFORM_DESCRIPTIONS: Dict[str, str] = {
    "none":     "raw values as-is",
    "arcsinh":  "Arcsinh (cofactor=150) — cytometry standard, compresses extreme values",
    "logicle":  "Logicle — adaptive, best for well-calibrated panels",
}

# One-line description for each normalization shown in the interactive wizard.
_NORM_DESCRIPTIONS: Dict[str, str] = {
    "none":                 "no normalization",
    "zscore_col":           "Z-score per column — mean=0, std=1 per channel",
    "minmax_col":           "Min-Max per column — scales each channel to [0, 1]",
    "l2norm_row":           "L2 Norm per row — each event profile becomes a unit vector",
    "zscore_col__l2norm_row": "Z-score per column + L2 Norm per row",
}

# One-line description for each feature selection method.
_FS_METHOD_DESCRIPTIONS: Dict[str, str] = {
    "CorrFilter":   "Correlation filter             [UNSUPERVISED — threshold needed]",
    "MIM":          "Mutual Information Maximization [SUPERVISED   — needs a target label]",
    "CMI":          "CMI/mRMR — Min. Redundancy Max. Relevance [SUPERVISED — needs a target]",
    "HighVariance": "Rank channels by variance       [UNSUPERVISED — no target needed]",
    "PCALoadings":  "Rank channels by PCA contribution [UNSUPERVISED — no target needed]",
}


# ---------------------------------------------------------------------------
# Public: get_default_preprocessing_config
# ---------------------------------------------------------------------------


def get_default_preprocessing_config() -> dict:
    """
    Return a fully-populated preprocessing_config dict with safe default values.

    All keys are present so that _validate_preprocessing_config() passes.
    Override only the keys you need before passing the dict to run_preprocessing().

    Returns:
        dict with keys:
          Step 1 — subset selection:
            filters, n_events_per_subject
          Step 2 — feature selection:
            feature_selection_methods, feature_selection_top_n,
            feature_selection_target_obs_column,
            feature_selection_corr_filter_threshold,
            feature_selection_mi_events_per_subject
          Step 3 — transformation + normalization:
            transform, normalization
    """
    return {
        # --- Step 1: subset selection ---
        # Dict mapping .obs column names to lists of raw values to keep.
        # An empty dict keeps all events (no filtering).
        # Example: {"condition": ["young"], "staining": ["DeepRed"]}
        "filters": {},
        # Number of events to sample per subject after filtering.
        # 0 means skip sub-sampling and keep all filtered events.
        "n_events_per_subject": 0,

        # --- Step 2: feature selection ---
        # Ordered list of methods to apply. Empty list keeps all channels.
        # Allowed values: "CorrFilter", "MIM", "CMI", "HighVariance", "PCALoadings"
        "feature_selection_methods": [],
        # Number of top-ranked channels to keep (ignored when methods is empty
        # or when the only method is CorrFilter).
        "feature_selection_top_n": 40,
        # .obs column used as classification target for supervised methods
        # (MIM, CMI). Must be set when those methods are requested.
        "feature_selection_target_obs_column": None,
        # Pearson correlation threshold for CorrFilter. Pairs above this value
        # are considered redundant; one channel from each pair is dropped.
        "feature_selection_corr_filter_threshold": 0.95,
        # Events per subject drawn for MI scoring. 0 = use all events.
        # Subsampling speeds up MIM/CMI on large datasets (> 100k events).
        "feature_selection_mi_events_per_subject": 0,

        # --- Step 3: transformation + normalization ---
        # Transformation to apply before normalization.
        # "none" | "arcsinh" | "logicle"
        "transform": "none",
        # Normalization to apply after transformation.
        # "none" | "zscore_col" | "minmax_col" | "l2norm_row" | "zscore_col__l2norm_row"
        "normalization": "zscore_col",
    }


# ---------------------------------------------------------------------------
# Public: configure_preprocessing (interactive console wizard)
# ---------------------------------------------------------------------------


def configure_preprocessing(anndata_object: anndata.AnnData) -> dict:
    """
    Interactively build a preprocessing_config dict via numbered console menus.

    Runs the same interactive menus as select_sfc_subset(), select_channels(),
    and transform_and_normalize() — but instead of immediately modifying the
    AnnData, this wizard records all choices into a config dict and returns it.

    Presents menus in this order:
      [1] Subset filters    — one menu per filterable .obs column
      [2] Sub-sampling      — events per subject (0 = skip)
      [3] Feature selection — method(s), top-N, target column, thresholds
      [4] Transformation    — none / arcsinh / logicle
      [5] Normalization     — none / zscore_col / minmax_col / l2norm_row / combined

    Pass the returned dict to run_preprocessing() to apply all steps without
    any further interactive prompts.

    Arguments:
        anndata_object: AnnData used to list available .obs columns and values
                        in the menus. The object is never modified.

    Returns:
        Validated preprocessing_config dict.
    """
    print("=" * 60)
    print("INTERACTIVE PREPROCESSING CONFIGURATION")
    print("=" * 60)
    print(
        f"Dataset: {anndata_object.n_obs:,} observations "
        f"× {anndata_object.n_vars} features"
    )
    print()

    # ------------------------------------------------------------------
    # [1] Subset filters
    # ------------------------------------------------------------------
    filters = _wizard_subset_filters(anndata_object)

    # ------------------------------------------------------------------
    # [2] Sub-sampling
    # ------------------------------------------------------------------
    n_events_per_subject = _wizard_subsampling(anndata_object)

    # ------------------------------------------------------------------
    # [3] Feature selection
    # ------------------------------------------------------------------
    (
        feature_selection_methods,
        feature_selection_top_n,
        feature_selection_target_obs_column,
        feature_selection_corr_filter_threshold,
        feature_selection_mi_events_per_subject,
    ) = _wizard_feature_selection(anndata_object)

    # ------------------------------------------------------------------
    # [4 + 5] Transformation + Normalization
    # ------------------------------------------------------------------
    transform, normalization = _wizard_transform_and_norm()

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print()
    print("=> Configuration summary:")
    print(f"   filters                              : {filters}")
    print(f"   n_events_per_subject                 : {n_events_per_subject}")
    print(f"   feature_selection_methods            : {feature_selection_methods}")
    if feature_selection_methods:
        print(f"   feature_selection_top_n              : {feature_selection_top_n}")
        print(f"   feature_selection_target_obs_column  : {feature_selection_target_obs_column}")
        print(f"   feature_selection_corr_filter_threshold: {feature_selection_corr_filter_threshold}")
        print(f"   feature_selection_mi_events_per_subject: {feature_selection_mi_events_per_subject}")
    print(f"   transform                            : {transform}")
    print(f"   normalization                        : {normalization}")
    print("=" * 60)

    preprocessing_config = {
        "filters":                                  filters,
        "n_events_per_subject":                     n_events_per_subject,
        "feature_selection_methods":                feature_selection_methods,
        "feature_selection_top_n":                  feature_selection_top_n,
        "feature_selection_target_obs_column":      feature_selection_target_obs_column,
        "feature_selection_corr_filter_threshold":  feature_selection_corr_filter_threshold,
        "feature_selection_mi_events_per_subject":  feature_selection_mi_events_per_subject,
        "transform":                                transform,
        "normalization":                            normalization,
    }

    _validate_preprocessing_config(preprocessing_config)
    return preprocessing_config


# ---------------------------------------------------------------------------
# Internal: validation
# ---------------------------------------------------------------------------


def _validate_preprocessing_config(preprocessing_config: dict) -> None:
    """
    Assert that preprocessing_config contains all required keys with valid types
    and values.

    Arguments:
        preprocessing_config: Dict to validate.

    Raises:
        KeyError:   A required key is missing.
        TypeError:  A value has the wrong Python type.
        ValueError: A value is outside its allowed set.
    """
    required_keys: Dict[str, type] = {
        "filters":                                  dict,
        "n_events_per_subject":                     int,
        "feature_selection_methods":                list,
        "feature_selection_top_n":                  int,
        "feature_selection_corr_filter_threshold":  float,
        "feature_selection_mi_events_per_subject":  int,
        "transform":                                str,
        "normalization":                            str,
    }

    for key, expected_type in required_keys.items():
        if key not in preprocessing_config:
            raise KeyError(
                f"preprocessing_config is missing required key: '{key}'. "
                "Use get_default_preprocessing_config() to build a complete config."
            )
        value = preprocessing_config[key]
        if not isinstance(value, expected_type):
            raise TypeError(
                f"preprocessing_config['{key}'] must be {expected_type.__name__}, "
                f"got {type(value).__name__}: {value!r}"
            )

    # feature_selection_target_obs_column: str or None
    target_col = preprocessing_config.get("feature_selection_target_obs_column")
    if target_col is not None and not isinstance(target_col, str):
        raise TypeError(
            "preprocessing_config['feature_selection_target_obs_column'] must be "
            f"a str or None, got {type(target_col).__name__}: {target_col!r}"
        )

    # feature_selection_methods: list of allowed method names
    for method in preprocessing_config["feature_selection_methods"]:
        if method not in ALLOWED_FEATURE_SELECTION_METHODS:
            raise ValueError(
                f"Unknown feature_selection_methods entry: {method!r}. "
                f"Allowed: {ALLOWED_FEATURE_SELECTION_METHODS}"
            )

    # Supervised methods require a target column
    supervised_methods = {"MIM", "CMI"}
    requested_supervised = [
        m for m in preprocessing_config["feature_selection_methods"]
        if m in supervised_methods
    ]
    if requested_supervised and target_col is None:
        raise ValueError(
            f"feature_selection_target_obs_column must be set when using supervised "
            f"methods: {requested_supervised}"
        )

    # transform
    if preprocessing_config["transform"] not in ALLOWED_TRANSFORMS:
        raise ValueError(
            f"preprocessing_config['transform'] must be one of {ALLOWED_TRANSFORMS}, "
            f"got {preprocessing_config['transform']!r}"
        )

    # normalization
    if preprocessing_config["normalization"] not in ALLOWED_NORMALIZATIONS:
        raise ValueError(
            f"preprocessing_config['normalization'] must be one of {ALLOWED_NORMALIZATIONS}, "
            f"got {preprocessing_config['normalization']!r}"
        )

    # n_events_per_subject must be >= 0
    if preprocessing_config["n_events_per_subject"] < 0:
        raise ValueError(
            "preprocessing_config['n_events_per_subject'] must be >= 0 "
            f"(0 = skip sub-sampling), got {preprocessing_config['n_events_per_subject']}"
        )

    # feature_selection_corr_filter_threshold must be in (0, 1]
    threshold = preprocessing_config["feature_selection_corr_filter_threshold"]
    if not (0.0 < threshold <= 1.0):
        raise ValueError(
            "preprocessing_config['feature_selection_corr_filter_threshold'] must be "
            f"in (0, 1], got {threshold}"
        )

    # feature_selection_mi_events_per_subject must be >= 0
    if preprocessing_config["feature_selection_mi_events_per_subject"] < 0:
        raise ValueError(
            "preprocessing_config['feature_selection_mi_events_per_subject'] must be "
            f">= 0 (0 = no subsampling), got "
            f"{preprocessing_config['feature_selection_mi_events_per_subject']}"
        )

    # feature_selection_top_n must be >= 1 when ranking methods are requested
    ranking_methods = [
        m for m in preprocessing_config["feature_selection_methods"]
        if m != "CorrFilter"
    ]
    if ranking_methods and preprocessing_config["feature_selection_top_n"] < 1:
        raise ValueError(
            "preprocessing_config['feature_selection_top_n'] must be >= 1 "
            f"when ranking methods are requested, got "
            f"{preprocessing_config['feature_selection_top_n']}"
        )


# ---------------------------------------------------------------------------
# Internal: interactive wizard helpers
# ---------------------------------------------------------------------------


def _wizard_subset_filters(anndata_object: anndata.AnnData) -> Dict[str, List]:
    """
    Present one filter menu per filterable .obs column and return the choices dict.

    Columns with only one unique value are auto-skipped (no user choice needed).
    Columns absent from .obs are silently skipped.

    Arguments:
        anndata_object: AnnData whose .obs is used to build the menus.

    Returns:
        Dict mapping column names to lists of selected raw values.
        Columns for which the user chose 'All' are not included in the dict.
    """
    from mito_marker.analysis.selection import (
        _get_display_values,
        _get_subject_column,
        _prompt_column_selection,
    )
    from mito_marker.controlled_vocabulary import FILTERABLE_OBS_COLUMNS

    print("[1] SUBSET FILTERS")
    print(
        "    For each column, select which values to keep.\n"
        "    Press 'A' to keep all values for that column.\n"
    )

    obs_dataframe = anndata_object.obs
    subject_col = _get_subject_column(obs_dataframe)
    filters: Dict[str, List] = {}

    for column_name in FILTERABLE_OBS_COLUMNS:
        if column_name not in obs_dataframe.columns:
            continue
        display_values = _get_display_values(obs_dataframe[column_name])
        if len(display_values) <= 1:
            continue

        selected_raw_values = _prompt_column_selection(
            column_name, display_values, obs_dataframe, subject_col
        )
        if selected_raw_values is not None:
            filters[column_name] = selected_raw_values

        print()

    if not filters:
        print("=> No filters applied — all events will be kept.")
    else:
        print(f"=> Filters set for columns: {list(filters.keys())}")
    print()
    return filters


def _wizard_subsampling(anndata_object: anndata.AnnData) -> int:
    """
    Ask how many events per subject to sample and return the chosen value.

    Arguments:
        anndata_object: AnnData whose .obs is used to show per-subject counts.

    Returns:
        Number of events per subject (0 = skip sub-sampling).
    """
    from mito_marker.analysis.selection import _prompt_subsampling_size

    print("[2] SUB-SAMPLING")
    n_events_per_subject = _prompt_subsampling_size(anndata_object.obs)
    if n_events_per_subject > 0:
        print(f"=> Sub-sampling set to {n_events_per_subject} events per subject.")
    else:
        print("=> Sub-sampling disabled — all filtered events will be kept.")
    print()
    return n_events_per_subject


def _wizard_feature_selection(
    anndata_object: anndata.AnnData,
) -> tuple:
    """
    Present the feature selection menus and return all collected parameters.

    Arguments:
        anndata_object: AnnData used to list available .obs columns for
                        supervised target selection and to determine top-N bounds.

    Returns:
        Tuple of:
          (methods, top_n, target_obs_column, corr_filter_threshold,
           mi_events_per_subject)
    """
    from mito_marker.analysis.feature_selection import (
        _prompt_corr_filter_threshold,
        _prompt_feature_selection_options,
        _prompt_subsample_options,
        _prompt_target_column,
        _prompt_top_n,
    )

    print("[3] FEATURE SELECTION")
    method_names = _prompt_feature_selection_options(anndata_object)

    corr_filter_threshold: float = 0.95
    target_obs_column: Optional[str] = None
    top_n: int = min(40, anndata_object.n_vars)
    mi_events_per_subject: int = 0

    if method_names:
        if "CorrFilter" in method_names:
            corr_filter_threshold = _prompt_corr_filter_threshold()

        has_supervised = any(m in method_names for m in ("MIM", "CMI"))
        if has_supervised:
            target_obs_column = _prompt_target_column(anndata_object)

        has_ranking = any(m != "CorrFilter" for m in method_names)
        if has_ranking:
            top_n = _prompt_top_n(anndata_object.n_vars)

        has_mi = any(m in method_names for m in ("MIM", "CMI"))
        if has_mi:
            use_subsample, events_per_subject = _prompt_subsample_options(
                anndata_object.obs, anndata_object.n_obs
            )
            if use_subsample:
                mi_events_per_subject = events_per_subject

    print()
    return (
        method_names,
        top_n,
        target_obs_column,
        corr_filter_threshold,
        mi_events_per_subject,
    )


def _wizard_transform_and_norm() -> tuple:
    """
    Present the transformation + normalization combined menu.

    Reuses _prompt_transform_and_norm() from normalization.py, which returns
    the internal key strings (e.g. "arcsinh", "zscore_col").

    Returns:
        Tuple of (transform_key, norm_key).
    """
    from mito_marker.analysis.normalization import _prompt_transform_and_norm

    print("[4+5] TRANSFORMATION + NORMALIZATION")
    transform_key, norm_key = _prompt_transform_and_norm()
    print()
    return transform_key, norm_key
