"""
ml_pipeline.py

Main machine learning pipeline for the mito-marker analysis package.

This module ties together configuration (ml_config.py), data preparation
(ml_bagging.py), model training, evaluation, and SHAP explainability.

Two public entry points:

  run_ml_analysis(anndata_object, ml_config=None, feature_names_override=None)
      Train and evaluate one ML model on the AnnData.
      Returns a results dict and stores it in .uns['ml_results'].

  run_feature_subset_challenge(anndata_object, feature_subsets_dict, ml_config, ...)
      Run run_ml_analysis() once per named feature subset and compare results
      in a summary DataFrame.  Useful for deciding which feature group (e.g.
      Size vs Shape vs Intensity vs Cristae Orientation) carries the most
      predictive signal.

  get_sfc_feature_subsets(anndata_object)
      Build a feature subset dict for SFC data by matching channel names against
      the morpho and spectral prefix lists in controlled_vocabulary.py.

Typical usage (programmatic):
    from mito_marker.analysis.ml_config import get_default_ml_config, DEFAULT_MODEL_PARAMS
    from mito_marker.analysis.ml_pipeline import run_ml_analysis

    config = get_default_ml_config()
    config["task_type"]        = "classification"
    config["target_obs_column"] = "age_group"
    config["model_params"]     = DEFAULT_MODEL_PARAMS["RandomForest"]

    results = run_ml_analysis(my_anndata, ml_config=config)

Typical usage (interactive):
    from mito_marker.analysis import configure_ml, run_ml_analysis
    config  = configure_ml(my_anndata)       # numbered console menus
    results = run_ml_analysis(my_anndata, ml_config=config)
"""

import copy
import pprint
import warnings
from typing import Dict, List, Optional, Tuple

import anndata
import numpy as np
import pandas as pd
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import ElasticNet as SklearnElasticNet
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    auc,
    classification_report,
    confusion_matrix,
    mean_absolute_error,
    r2_score,
    roc_auc_score,
    roc_curve,
    root_mean_squared_error,
)
from sklearn.model_selection import KFold, LeaveOneGroupOut, StratifiedKFold, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC, SVR

from mito_marker.analysis._gpu_utils import is_cuml_available
from mito_marker.analysis.feature_selection import (
    _get_active_data_matrix,
    _score_cmi,
    _score_corr_filter,
    _score_high_variance,
    _score_mim,
    _score_pca_loadings,
    get_selected_data_matrix,
)
from mito_marker.analysis.ml_bagging import create_bags
from mito_marker.analysis.ml_config import _validate_ml_config, configure_ml
from mito_marker.analysis.selection import _ANALYSIS_CONFIG_KEY

# Key under which ML results are stored in .uns.
_ML_RESULTS_KEY = "ml_results"
_ML_CHALLENGE_KEY = "ml_subset_challenge_results"


# ---------------------------------------------------------------------------
# Public: get_sfc_feature_subsets
# ---------------------------------------------------------------------------


def get_sfc_feature_subsets(anndata_object: anndata.AnnData) -> Dict[str, List[str]]:
    """
    Build a feature subset dict for SFC data by matching channel names against
    predefined morpho and spectral prefix patterns.

    Morpho channels (scatter): names that start with "FSC" or "SSC".
    Spectral channels (fluorescence): names that start with "UV", "V", "B",
    "YG", or "R" (followed by a digit or hyphen, to avoid false matches).

    Arguments:
        anndata_object: SFC AnnData whose .var_names are cytometry channel names.

    Returns:
        dict with keys "Morpho" and "Spectral", each mapping to a list of
        matching channel names found in anndata_object.var_names.

    Example:
        subsets = get_sfc_feature_subsets(sfc_anndata)
        # → {"Morpho": ["FSC-A", "FSC-H", "SSC-A", ...],
        #    "Spectral": ["UV1-A", "V1-A", "B1-A", ...]}
        results = run_feature_subset_challenge(sfc_anndata, subsets, ml_config=config)
    """
    from mito_marker.controlled_vocabulary import (
        SFC_MORPHO_CHANNEL_PREFIXES,
        SFC_SPECTRAL_CHANNEL_PREFIXES,
    )

    all_channel_names = list(anndata_object.var_names)

    morpho_channels = [
        name for name in all_channel_names
        if any(name.startswith(prefix) for prefix in SFC_MORPHO_CHANNEL_PREFIXES)
    ]

    spectral_channels = [
        name for name in all_channel_names
        if any(name.startswith(prefix) for prefix in SFC_SPECTRAL_CHANNEL_PREFIXES)
        and name not in morpho_channels
    ]

    return {"Morpho": morpho_channels, "Spectral": spectral_channels}


# ---------------------------------------------------------------------------
# Internal: species label helper
# ---------------------------------------------------------------------------


def _build_species_label(anndata_object: anndata.AnnData) -> str:
    """
    Build a human-readable species label from .obs['specie'] if present.

    Returns a string like "Specie: Mouse" (one species) or
    "Species: [Mouse, Droso, Human]" (multiple), or "" if the column is absent.
    """
    if "specie" not in anndata_object.obs.columns:
        return ""
    unique_species = sorted(anndata_object.obs["specie"].dropna().unique().tolist())
    if not unique_species:
        return ""
    if len(unique_species) == 1:
        return f"Specie: {unique_species[0]}"
    species_list = ", ".join(str(s) for s in unique_species)
    return f"Species: [{species_list}]"


# ---------------------------------------------------------------------------
# Public: run_ml_analysis
# ---------------------------------------------------------------------------


def run_ml_analysis(
    anndata_object: anndata.AnnData,
    ml_config: Optional[dict] = None,
    feature_names_override: Optional[List[str]] = None,
) -> dict:
    """
    Train and evaluate one ML model on the AnnData, return structured results.

    Reads the current AnnData state to build the data matrix:
      - If feature_names_override is None: calls get_selected_data_matrix(),
        which respects the active layer and the active feature selection.
      - If feature_names_override is given: reads the active data layer directly
        (bypassing any feature selection mask) and filters to the specified
        feature names.  This allows the feature subset challenge to work
        independently of select_channels().

    If ml_config is None, configure_ml() is called interactively.

    Execution flow:
      1. Validate config.
      2. Extract data matrix and feature names.
      3. If strategy == "Bags": call create_bags() to aggregate rows.
      4. Evaluate with StandardCV or LOGO.
      5. If compute_shap: compute SHAP values and produce plots.
      6. Print QC block to console.
      7. Store results in .uns['ml_results'] and return.

    Arguments:
        anndata_object:        AnnData with .uns['analysis_config'] populated.
        ml_config:             Config dict from get_default_ml_config() or
                               configure_ml().  If None, prompts interactively.
        feature_names_override: List of feature names (from .var_names) to use
                               instead of the active feature selection.  When
                               provided, feature selection is bypassed entirely.

    Returns:
        Results dict (also stored in anndata_object.uns['ml_results']).
    """
    if ml_config is None:
        ml_config = configure_ml(anndata_object)

    _validate_ml_config(ml_config)

    task_type = ml_config["task_type"]
    strategy = ml_config["strategy"]
    target_obs_column = ml_config["target_obs_column"]
    subject_id_column = ml_config["subject_id_column"]
    random_state = ml_config["random_state"]
    model_name = ml_config["model_name"]

    print("=" * 60)
    print("ML ANALYSIS")
    print("=" * 60)
    print(f"  Task               : {task_type}")
    print(f"  Target column      : {target_obs_column}")
    print(f"  Strategy           : {strategy}")
    print(f"  Model              : {model_name}")
    model_params = ml_config["model_params"]
    if model_params:
        print("  Model params       :")
        for line in pprint.pformat(model_params, indent=4).splitlines():
            print(f"    {line}")
    else:
        print("  Model params       : sklearn defaults (empty dict)")
    print(f"  Evaluation         : {ml_config['evaluation_strategy']}")
    if ml_config.get("logo_group_by_column"):
        print(f"  LAVO group column  : {ml_config['logo_group_by_column']}")
    print(f"  Random state       : {random_state}")
    print()

    species_label = _build_species_label(anndata_object)

    # ------------------------------------------------------------------
    # Step 1: Build data matrix and feature name list
    # ------------------------------------------------------------------
    if feature_names_override is not None:
        # Bypass feature selection — read active layer directly.
        full_matrix = _get_active_data_matrix(anndata_object)
        full_feature_names = list(anndata_object.var_names)
        data_matrix, feature_names = _apply_feature_name_filter(
            full_matrix, full_feature_names, feature_names_override
        )
    else:
        # Respect the active feature selection (get_selected_data_matrix()
        # excludes non-analytical channels and applies is_selected_* mask).
        data_matrix = get_selected_data_matrix(anndata_object)
        config = anndata_object.uns.get(_ANALYSIS_CONFIG_KEY, {})
        active_selection = config.get("active_selection")
        feature_names = _get_selected_feature_names(anndata_object, active_selection)

    n_obs_original = data_matrix.shape[0]
    print(f"  Observations       : {n_obs_original:,}")
    print(f"  Features           : {len(feature_names)}")

    # ------------------------------------------------------------------
    # Step 2: Extract target labels and subject IDs from .obs
    # ------------------------------------------------------------------
    if target_obs_column not in anndata_object.obs.columns:
        raise ValueError(
            f"target_obs_column '{target_obs_column}' not found in .obs. "
            f"Available columns: {list(anndata_object.obs.columns)}"
        )
    if subject_id_column not in anndata_object.obs.columns:
        raise ValueError(
            f"subject_id_column '{subject_id_column}' not found in .obs. "
            f"Available columns: {list(anndata_object.obs.columns)}"
        )

    target_labels = anndata_object.obs[target_obs_column].values
    subject_ids = anndata_object.obs[subject_id_column].values

    # ------------------------------------------------------------------
    # Step 2b: Drop observations whose target value is NaN.
    # ------------------------------------------------------------------
    # Clinical metadata may be missing for some subjects.  sklearn rejects NaN
    # labels for both classifiers and regressors, so we remove those rows here.
    valid_mask = ~pd.isna(target_labels)
    n_nan = int((~valid_mask).sum())
    if n_nan > 0:
        nan_subjects = np.unique(subject_ids[~valid_mask])
        print(
            f"\n  WARNING: {n_nan:,} observations ({len(nan_subjects)} subject(s)) "
            f"have NaN in target column '{target_obs_column}' and will be excluded "
            f"from training."
        )
        print(f"  Excluded subject(s): {', '.join(str(s) for s in nan_subjects)}")
        data_matrix = data_matrix[valid_mask]
        target_labels = target_labels[valid_mask]
        subject_ids = subject_ids[valid_mask]
        filtered_obs = anndata_object.obs.iloc[np.where(valid_mask)[0]].copy()
    else:
        filtered_obs = anndata_object.obs

    # ------------------------------------------------------------------
    # Step 3: Apply Bags strategy if requested
    # ------------------------------------------------------------------
    if strategy == "Bags":
        bag_n_clusters = ml_config.get("bag_n_clusters")
        print()
        cluster_suffix = f" + cluster fractions (k={bag_n_clusters})" if bag_n_clusters else ""
        print(
            f"  Creating bags: {ml_config['bags_per_subject']} bags/subject, "
            f"{ml_config['mitos_per_bag']} mitos/bag{cluster_suffix} …"
        )
        data_matrix, target_labels, subject_ids, feature_names = create_bags(
            data_matrix=data_matrix,
            obs_dataframe=filtered_obs,
            subject_id_column=subject_id_column,
            bags_per_subject=ml_config["bags_per_subject"],
            mitos_per_bag=ml_config["mitos_per_bag"],
            bag_statistics=ml_config["bag_statistics"],
            target_obs_column=target_obs_column,
            task_type=task_type,
            random_state=random_state,
            feature_names=feature_names,
            n_clusters=bag_n_clusters,
            sampling_mode=ml_config.get("bag_sampling_mode", "auto"),
            max_overlap_fraction=ml_config.get("bag_max_overlap_fraction", 0.2),
        )
        print(f"  Bags created       : {data_matrix.shape[0]:,} total")
        print(f"  Bag features ({data_matrix.shape[1]:>3}) : {', '.join(feature_names)}")

    # ------------------------------------------------------------------
    # Step 3b: Post-bagging feature selection (optional pipeline)
    # ------------------------------------------------------------------
    bag_fs_methods = ml_config.get("bag_feature_selection_methods", [])
    if strategy == "Bags" and bag_fs_methods:
        print()
        print(f"  Post-bag feature selection: {' → '.join(bag_fs_methods)} …")
        print(f"  Features before: {data_matrix.shape[1]}")
        data_matrix, feature_names = _select_bag_features_pipeline(
            bag_matrix=data_matrix,
            bag_targets=target_labels,
            bag_feature_names=feature_names,
            methods=bag_fs_methods,
            top_k=ml_config["bag_feature_selection_top_k"],
            corr_threshold=ml_config["bag_feature_selection_corr_threshold"],
        )
        print(f"  Features after : {data_matrix.shape[1]}")
        print(f"  Selected       : {', '.join(feature_names)}")

    # ------------------------------------------------------------------
    # Step 3c: Build group_ids for LOGO/LAVO fold splits
    # ------------------------------------------------------------------
    # group_ids drives LeaveOneGroupOut splits. For standard LOGO it equals
    # subject_ids. For LAVO it is the value of logo_group_by_column per sample
    # (e.g. age), so all subjects sharing the same value are left out together.
    # Must be built after create_bags() so it aligns with the post-bag subject_ids.
    logo_group_by_column = ml_config.get("logo_group_by_column", None)
    if logo_group_by_column is not None:
        if logo_group_by_column not in anndata_object.obs.columns:
            raise ValueError(
                f"logo_group_by_column '{logo_group_by_column}' not found in .obs. "
                f"Available columns: {list(anndata_object.obs.columns)}"
            )
        # Build subject → group mapping from pre-bag obs (one unique row per subject).
        subject_to_group: dict = (
            filtered_obs[[subject_id_column, logo_group_by_column]]
            .drop_duplicates(subset=subject_id_column)
            .set_index(subject_id_column)[logo_group_by_column]
            .to_dict()
        )
        # Map each bag/event's subject_id to its group value (str for LOGO compatibility).
        group_ids = np.array(
            [str(subject_to_group[sid]) for sid in subject_ids], dtype=object
        )
    else:
        group_ids = subject_ids

    # ------------------------------------------------------------------
    # Step 4: Build and evaluate model
    # ------------------------------------------------------------------
    if strategy == "MIL":
        # MIL handles its own training loop — skip model build, LOGO split, SHAP.
        try:
            import torch  # noqa: F401 — ensure PyTorch is available before running folds
        except ImportError as mil_import_error:
            raise ImportError(
                "MIL strategy requires PyTorch. Install it with: pip install torch>=2.0"
            ) from mil_import_error

        from sklearn.preprocessing import LabelEncoder as _LabelEncoder

        from mito_marker.analysis.ml_mil import run_mil_logo_cv

        print()
        print("  Training model: AttentionMILModel (ABMIL) — LOGO …")

        mil_label_encoder = None if task_type == "regression" else _LabelEncoder()
        evaluation_results = run_mil_logo_cv(
            data_matrix=data_matrix,
            obs_dataframe=filtered_obs,
            ml_config=ml_config,
            feature_names=feature_names,
            label_encoder=mil_label_encoder,
        )

        # MIL does not produce SHAP values — emit warning if requested.
        if ml_config.get("compute_shap", False):
            warnings.warn(
                "MIL strategy does not support SHAP. "
                "Use plot_mil_attention() and get_mil_attention_dataframe() "
                "to identify the discriminative mitochondrial sub-population.",
                UserWarning,
                stacklevel=2,
            )

        shap_results: dict = {
            "shap_values": None,
            "shap_explainer_type": None,
            "shap_feature_names": None,
        }

        results = {
            "ml_config":           copy.deepcopy(ml_config),
            # For MIL, each subject is one training example (one bag). Report the
            # number of subjects, not mito count, as n_samples.
            "n_samples":           len(evaluation_results["logo_per_subject_scores"]),
            "n_features":          int(data_matrix.shape[1]),
            "feature_names":       feature_names,
            "evaluation_strategy": ml_config["evaluation_strategy"],
            "task_type":           task_type,
            **evaluation_results,
            **shap_results,
        }

        _print_evaluation_qc_block(results, task_type, "MIL (ABMIL)", strategy)

        if task_type == "classification":
            _plot_confusion_matrix(results, "MIL", ml_config["evaluation_strategy"], target_obs_column, species_label)
        else:
            _plot_regression_scatter(results, "MIL", ml_config["evaluation_strategy"], target_obs_column, species_label)

        anndata_object.uns[_ML_RESULTS_KEY] = results
        print(
            f"=> Results stored in .uns['{_ML_RESULTS_KEY}']. "
            "Use plot_mil_attention() for attention visualisation."
        )
        print("=" * 60)
        return results

    model = _build_model(
        model_name=model_name,
        task_type=task_type,
        model_params=ml_config["model_params"],
        random_state=random_state,
    )

    print()
    print(f"  Training model: {type(model).__name__} …")

    if ml_config["evaluation_strategy"] == "StandardCV":
        evaluation_results = _evaluate_standard_cv(
            model=model,
            feature_matrix=data_matrix,
            target_labels=target_labels,
            subject_ids=subject_ids,
            ml_config=ml_config,
        )
    else:  # LOGO / LAVO
        evaluation_results = _evaluate_logo_cv(
            model=model,
            feature_matrix=data_matrix,
            target_labels=target_labels,
            subject_ids=subject_ids,
            group_ids=group_ids,
            ml_config=ml_config,
            obs_dataframe=filtered_obs,
            compute_shap=ml_config["compute_shap"],
            model_name=model_name,
        )

    # ------------------------------------------------------------------
    # Step 5: SHAP explainability
    # ------------------------------------------------------------------
    shap_results: dict = {
        "shap_values": None,
        "shap_explainer_type": None,
        "shap_feature_names": None,
    }
    # Derive class names from original string labels (used by both branches).
    shap_class_names = [str(c) for c in np.unique(target_labels)]

    if ml_config["compute_shap"]:
        if ml_config["evaluation_strategy"] == "LOGO":
            # SHAP was already collected fold-by-fold inside _evaluate_logo_cv.
            # Just plot and package — no additional training needed.
            logo_importance = evaluation_results.get("logo_shap_importance")
            n_folds_used = evaluation_results.get("logo_shap_n_folds_used", 0)
            shap_per_class = ml_config.get("shap_per_class_plots", False)
            if logo_importance is not None:
                _plot_logo_shap_bar(
                    averaged_importance=logo_importance,
                    feature_names=feature_names,
                    n_folds_used=n_folds_used,
                    class_names=shap_class_names,
                    model_name=model_name,
                    target_obs_column=target_obs_column,
                    show_per_class=shap_per_class,
                    species_label=species_label,
                )
            # Beeswarm from OOF per-sample SHAP values (one dot per left-out sample).
            logo_oof_shap = evaluation_results.get("logo_shap_oof_values")
            logo_oof_X = evaluation_results.get("logo_shap_oof_features")
            if logo_oof_shap is not None and logo_oof_X is not None:
                _draw_logo_shap_beeswarm(
                    oof_shap=logo_oof_shap,
                    oof_features=logo_oof_X,
                    feature_names=feature_names,
                    n_folds_used=n_folds_used,
                    class_names=shap_class_names,
                    model_name=model_name,
                    target_obs_column=target_obs_column,
                    show_per_class=shap_per_class,
                    species_label=species_label,
                )
            shap_results = {
                "shap_values":         logo_importance,
                "shap_explainer_type": evaluation_results.get("logo_shap_explainer_type", "Unknown"),
                "shap_feature_names":  feature_names,
            }
        else:
            # StandardCV: fit a fresh model on an 80/20 split to produce SHAP
            # explanations on a held-out test set.
            # Encode string labels to integers so MLP with early_stopping=True can
            # call np.isnan on predictions without a TypeError.
            if task_type == "classification":
                shap_label_encoder = LabelEncoder()
                shap_encoded_labels: np.ndarray = shap_label_encoder.fit_transform(
                    target_labels.astype(str)
                )
            else:
                shap_label_encoder = None
                shap_encoded_labels = target_labels
            X_tr, X_te, y_tr, _ = train_test_split(
                data_matrix,
                shap_encoded_labels,
                test_size=ml_config.get("test_size", 0.2),
                random_state=random_state,
            )
            # Mirror the in-fold scaling used during evaluation so the SHAP model
            # sees the same feature distribution as the CV models.
            if ml_config.get("scale_features_in_fold", False):
                shap_scaler = StandardScaler()
                X_tr = shap_scaler.fit_transform(X_tr)
                X_te = shap_scaler.transform(X_te)
            shap_model = _build_model(model_name, task_type, ml_config["model_params"], random_state)
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="Got `batch_size`.*", category=UserWarning)
                shap_model.fit(X_tr, y_tr)
            # Derive class names from original string labels (not encoded integers).
            shap_class_names_cv = (
                [str(c) for c in shap_label_encoder.classes_]
                if shap_label_encoder is not None
                else shap_class_names
            )
            shap_results = _run_shap(
                model=shap_model,
                model_name=model_name,
                X_train=X_tr,
                X_test=X_te,
                feature_names=feature_names,
                ml_config=ml_config,
                class_names=shap_class_names_cv,
                evaluation_strategy=ml_config["evaluation_strategy"],
                target_obs_column=target_obs_column,
                species_label=species_label,
            )

    # ------------------------------------------------------------------
    # Step 6: Assemble results dict
    # ------------------------------------------------------------------
    results = {
        "ml_config":          copy.deepcopy(ml_config),
        "n_samples":          int(data_matrix.shape[0]),
        "n_features":         int(data_matrix.shape[1]),
        "feature_names":      feature_names,
        "evaluation_strategy": ml_config["evaluation_strategy"],
        "task_type":          task_type,
        **evaluation_results,
        **shap_results,
    }

    _print_evaluation_qc_block(results, task_type, model_name, strategy)

    # Visual outputs depending on task type.
    if task_type == "classification":
        _plot_confusion_matrix(results, model_name, ml_config["evaluation_strategy"], target_obs_column, species_label)
        _plot_roc_curve(results, model_name, ml_config["evaluation_strategy"], target_obs_column, species_label)
    else:
        _plot_regression_scatter(results, model_name, ml_config["evaluation_strategy"], target_obs_column, species_label)

    # Step 7: Persist in .uns (overwritten each call — same pattern as analysis_config).
    anndata_object.uns[_ML_RESULTS_KEY] = results
    print(
        f"=> Results stored in .uns['{_ML_RESULTS_KEY}']. "
        f"Capture the returned dict to compare multiple runs."
    )
    print("=" * 60)
    return results


# ---------------------------------------------------------------------------
# Public: run_feature_subset_challenge
# ---------------------------------------------------------------------------


def run_feature_subset_challenge(
    anndata_object: anndata.AnnData,
    feature_subsets_dict: Dict[str, Optional[List[str]]],
    ml_config: Optional[dict] = None,
    include_bags_trend_heterogeneity: bool = False,
) -> pd.DataFrame:
    """
    Run run_ml_analysis() for each named feature subset and compare performance.

    Useful for deciding which feature group (e.g. Size vs Shape vs Intensity vs
    Cristae Orientation) carries the most predictive information for the chosen
    target.

    Arguments:
        anndata_object:           AnnData with active layer and .obs populated.
        feature_subsets_dict:     Keys are subset labels; values are feature name
                                  lists or None (None = use all available features).
                                  Example: {"All": None, "Size": ["Mito_Area", ...]}
        ml_config:                Shared config for all runs.  If None, calls
                                  configure_ml() once interactively.
        include_bags_trend_heterogeneity:
                                  When True and strategy == "Bags", automatically
                                  adds two variants per subset:
                                    "{label}__Trend"         — mean + median features only
                                    "{label}__Heterogeneity" — std + skew features only
                                  Emits a warning and skips if strategy == "SingleMito".

    Returns:
        pd.DataFrame with one row per subset run and columns:
          subset_name, n_features, primary_metric, std, roc_auc
        Also printed as a formatted table and stored in
        anndata_object.uns['ml_subset_challenge_results'].
        Note: .uns['ml_results'] contains only the LAST run's results.
    """
    if ml_config is None:
        ml_config = configure_ml(anndata_object)
    _validate_ml_config(ml_config)

    is_bags = ml_config["strategy"] == "Bags"

    if include_bags_trend_heterogeneity and not is_bags:
        warnings.warn(
            "include_bags_trend_heterogeneity=True has no effect when "
            "strategy='SingleMito'. Skipping Trend/Heterogeneity variants.",
            UserWarning,
            stacklevel=2,
        )
        include_bags_trend_heterogeneity = False

    print("=" * 60)
    print("FEATURE SUBSET CHALLENGE")
    print("=" * 60)
    print(f"  Subsets to compare : {list(feature_subsets_dict.keys())}")
    print(f"  Model              : {ml_config['model_name']}")
    print(f"  Evaluation         : {ml_config['evaluation_strategy']}")
    print()

    task_type = ml_config["task_type"]
    challenge_rows: List[dict] = []

    for subset_label, feature_list in feature_subsets_dict.items():
        print(f"  Running subset: {subset_label!r} …")
        results = run_ml_analysis(
            anndata_object=anndata_object,
            ml_config=ml_config,
            feature_names_override=feature_list,
        )
        challenge_rows.append(
            _extract_challenge_row(subset_label, results, task_type)
        )

        # Bags-specific: also run Trend and Heterogeneity sub-subsets.
        if include_bags_trend_heterogeneity:
            bag_feature_names = results["feature_names"]

            trend_features = _filter_bag_features_by_stat(
                bag_feature_names, keep_stats={"mean", "median"}
            )
            if trend_features:
                print(f"  Running subset: {subset_label}__Trend …")
                trend_results = run_ml_analysis(
                    anndata_object=anndata_object,
                    ml_config=ml_config,
                    feature_names_override=_get_original_names_from_bag_features(
                        trend_features, feature_list, anndata_object
                    ),
                )
                # Manually filter to trend columns after bagging.
                challenge_rows.append(
                    _extract_challenge_row(
                        f"{subset_label}__Trend", trend_results, task_type
                    )
                )

            hetero_features = _filter_bag_features_by_stat(
                bag_feature_names, keep_stats={"std", "skew"}
            )
            if hetero_features:
                print(f"  Running subset: {subset_label}__Heterogeneity …")
                hetero_results = run_ml_analysis(
                    anndata_object=anndata_object,
                    ml_config=ml_config,
                    feature_names_override=_get_original_names_from_bag_features(
                        hetero_features, feature_list, anndata_object
                    ),
                )
                challenge_rows.append(
                    _extract_challenge_row(
                        f"{subset_label}__Heterogeneity", hetero_results, task_type
                    )
                )

    summary_dataframe = pd.DataFrame(challenge_rows)

    print()
    print("=" * 60)
    print("CHALLENGE SUMMARY")
    print("=" * 60)
    metric_label = "accuracy" if task_type == "classification" else "R²"
    print(
        summary_dataframe.to_string(
            index=False,
            float_format=lambda value: f"{value:.3f}",
        )
    )
    print()
    if task_type == "classification":
        print(f"  primary_metric = mean {metric_label} across folds (0=worst, 1=best)")
    else:
        print(
            "  primary_metric = mean R² across folds "
            "(1=perfect, 0=no better than mean, negative=worse)"
        )
    print("=" * 60)

    anndata_object.uns[_ML_CHALLENGE_KEY] = summary_dataframe
    return summary_dataframe


# ---------------------------------------------------------------------------
# Internal: model factory
# ---------------------------------------------------------------------------


# Maps a (model_name, task_type) key to the cuML module path and class name.
# Models not listed here (MLP) have no cuML equivalent and always use sklearn.
_CUML_MODEL_MAP: dict[str, tuple[str, str]] = {
    "ElasticNet_classification": ("cuml.linear_model", "LogisticRegression"),
    "ElasticNet_regression":     ("cuml.linear_model", "ElasticNet"),
    "RandomForest_classification": ("cuml.ensemble",   "RandomForestClassifier"),
    "RandomForest_regression":     ("cuml.ensemble",   "RandomForestRegressor"),
    "SVM_classification":          ("cuml.svm",        "SVC"),
    "SVM_regression":              ("cuml.svm",        "SVR"),
}


def _try_cuml_model(
    model_name: str,
    task_type: str,
    params: dict,
    random_state: int,
) -> object | None:
    """
    Attempt to build a cuML GPU model for the given model_name and task_type.

    Returns the instantiated model if cuML is available and a mapping exists,
    otherwise returns None so the caller falls back to sklearn.

    cuML models do not support the n_jobs argument — it is stripped from params
    before instantiation.

    Arguments:
        model_name:   Model family name (e.g. "RandomForest").
        task_type:    "classification" or "regression".
        params:       Dict of constructor kwargs (will not be mutated).
        random_state: Seed forwarded to the cuML constructor.

    Returns:
        Unfitted cuML estimator, or None if unavailable.
    """
    if not is_cuml_available():
        return None
    cuml_key = f"{model_name}_{task_type}"
    if cuml_key not in _CUML_MODEL_MAP:
        return None
    import importlib  # noqa: PLC0415
    module_path, class_name = _CUML_MODEL_MAP[cuml_key]
    try:
        module = importlib.import_module(module_path)
        cuml_class = getattr(module, class_name)
        # n_jobs is a CPU-parallelism argument not supported by cuML.
        safe_params = {k: v for k, v in params.items() if k != "n_jobs"}
        print(f"  [GPU] Using cuML {class_name}.")
        return cuml_class(random_state=random_state, **safe_params)
    except Exception:
        # If cuML instantiation fails for any reason, return None to use sklearn.
        return None


def _build_model(
    model_name: str,
    task_type: str,
    model_params: dict,
    random_state: int,
):
    """
    Instantiate the correct sklearn / xgboost model class.

    For ElasticNet:
      - Classification → LogisticRegression(penalty="elasticnet", solver="saga")
      - Regression     → sklearn.linear_model.ElasticNet
    All other models dispatch to the Classifier / Regressor variant as appropriate.

    GaussianNB is classification-only (sklearn has no Gaussian Naive Bayes
    regressor) — requesting task_type="regression" raises ValueError.
    GaussianNB and KNeighbors have no random_state argument (both are
    deterministic given the data), so random_state is accepted but ignored
    for them.

    Arguments:
        model_name:    One of ALLOWED_MODEL_NAMES.
        task_type:     "classification" or "regression".
        model_params:  Dict of kwargs forwarded to the sklearn constructor.
                       Empty dict → sklearn defaults.
        random_state:  Seed propagated to models that accept it.

    Returns:
        Unfitted sklearn / xgboost estimator instance.

    Raises:
        ValueError: model_name is not supported, or GaussianNB + regression.
        ImportError: XGBoost is requested but not installed.
    """
    params = dict(model_params)  # copy so we don't mutate the config

    if model_name == "ElasticNet":
        if task_type == "classification":
            # For classification, ElasticNet penalty is exposed through
            # LogisticRegression with penalty="elasticnet" and solver="saga".
            params.setdefault("penalty", "elasticnet")
            params.setdefault("solver", "saga")
            params.setdefault("l1_ratio", 0.5)
            params.setdefault("max_iter", 2500)
            model = _try_cuml_model("ElasticNet", "classification", params, random_state)
            if model is not None:
                return model
            return LogisticRegression(random_state=random_state, n_jobs=-1, **params)
        else:
            params.pop("penalty", None)
            params.pop("solver", None)
            params.pop("C", None)
            params.pop("class_weight", None)
            model = _try_cuml_model("ElasticNet", "regression", params, random_state)
            if model is not None:
                return model
            return SklearnElasticNet(**params)

    if model_name == "RandomForest":
        if task_type == "classification":
            model = _try_cuml_model("RandomForest", "classification", params, random_state)
            if model is not None:
                return model
            return RandomForestClassifier(random_state=random_state, n_jobs=-1, **params)
        else:
            params.pop("class_weight", None)
            model = _try_cuml_model("RandomForest", "regression", params, random_state)
            if model is not None:
                return model
            return RandomForestRegressor(random_state=random_state, n_jobs=-1, **params)

    if model_name == "XGBoost":
        try:
            import xgboost as xgb  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(
                "XGBoost is not installed. Install it with: pip install xgboost>=2.0"
            ) from exc
        params.pop("class_weight", None)
        # XGBoost has its own native GPU support — no cuML needed.
        xgb_device = "cuda" if is_cuml_available() else "cpu"
        if task_type == "classification":
            return xgb.XGBClassifier(random_state=random_state, device=xgb_device, **params)
        else:
            return xgb.XGBRegressor(random_state=random_state, device=xgb_device, **params)

    if model_name == "MLP":
        # No cuML equivalent for MLP — always use sklearn.
        if task_type == "classification":
            return MLPClassifier(random_state=random_state, **params)
        else:
            return MLPRegressor(random_state=random_state, **params)

    if model_name == "SVM":
        params.setdefault("probability", True)
        params.pop("class_weight", None)
        if task_type == "classification":
            model = _try_cuml_model("SVM", "classification", params, random_state)
            if model is not None:
                return model
            return SVC(random_state=random_state, **params)
        else:
            params.pop("probability", None)
            model = _try_cuml_model("SVM", "regression", params, random_state)
            if model is not None:
                return model
            return SVR(**params)

    if model_name == "ExtraTrees":
        # No cuML equivalent — always use sklearn.
        if task_type == "classification":
            return ExtraTreesClassifier(random_state=random_state, n_jobs=-1, **params)
        else:
            params.pop("class_weight", None)
            return ExtraTreesRegressor(random_state=random_state, n_jobs=-1, **params)

    if model_name == "GradientBoosting":
        # sklearn's GradientBoosting has no n_jobs or class_weight argument.
        params.pop("class_weight", None)
        if task_type == "classification":
            return GradientBoostingClassifier(random_state=random_state, **params)
        else:
            return GradientBoostingRegressor(random_state=random_state, **params)

    if model_name == "GaussianNB":
        if task_type != "classification":
            raise ValueError(
                "GaussianNB is a classification-only model (sklearn has no Gaussian "
                "Naive Bayes regressor). Choose a different model_name for regression."
            )
        # GaussianNB has no random_state argument — it is deterministic given the data.
        return GaussianNB(**params)

    if model_name == "KNeighbors":
        # No random_state argument — predictions are deterministic given the data.
        params.pop("class_weight", None)
        if task_type == "classification":
            return KNeighborsClassifier(**params)
        else:
            return KNeighborsRegressor(**params)

    raise ValueError(
        f"Unknown model_name: '{model_name}'. "
        f"Supported models: ElasticNet, RandomForest, XGBoost, MLP, SVM, "
        f"ExtraTrees, GradientBoosting, GaussianNB, KNeighbors"
    )


# ---------------------------------------------------------------------------
# Internal: feature name helpers
# ---------------------------------------------------------------------------


def _get_selected_feature_names(
    anndata_object: anndata.AnnData,
    active_selection: Optional[str],
) -> List[str]:
    """
    Return the list of feature names that match the currently active selection mask.

    When active_selection is None, returns all analytical (non-excluded) feature names.
    """
    from mito_marker.analysis.feature_selection import (
        _get_analytical_mask,
    )

    analytical_mask = _get_analytical_mask(anndata_object)

    if active_selection is not None:
        selection_col = f"is_selected_{active_selection}"
        if selection_col in anndata_object.var.columns:
            selection_mask = anndata_object.var[selection_col].values.astype(bool)
            combined_mask = analytical_mask & selection_mask
        else:
            combined_mask = analytical_mask
    else:
        combined_mask = analytical_mask

    return [
        name
        for name, keep in zip(anndata_object.var_names, combined_mask)
        if keep
    ]


def _apply_feature_name_filter(
    full_matrix: np.ndarray,
    full_feature_names: List[str],
    requested_names: List[str],
) -> Tuple[np.ndarray, List[str]]:
    """
    Filter a data matrix to the columns matching requested_names.

    Arguments:
        full_matrix:       2D array (n_obs, n_all_features).
        full_feature_names: Feature names aligned with full_matrix columns.
        requested_names:   Ordered subset of feature names to keep.

    Returns:
        (filtered_matrix, filtered_feature_names)

    Raises:
        ValueError: A name in requested_names is not found in full_feature_names.
    """
    name_to_index = {name: idx for idx, name in enumerate(full_feature_names)}
    missing = [n for n in requested_names if n not in name_to_index]
    if missing:
        raise ValueError(
            f"The following feature names were not found in the AnnData: {missing}. "
            f"Available features: {full_feature_names}"
        )
    column_indices = [name_to_index[n] for n in requested_names]
    return full_matrix[:, column_indices], list(requested_names)


# ---------------------------------------------------------------------------
# Internal: Post-bagging feature selection pipeline
# ---------------------------------------------------------------------------


def _select_bag_features_pipeline(
    bag_matrix: np.ndarray,
    bag_targets: np.ndarray,
    bag_feature_names: List[str],
    methods: List[str],
    top_k: int,
    corr_threshold: float,
) -> Tuple[np.ndarray, List[str]]:
    """
    Apply a sequential feature selection pipeline on a bag feature matrix.

    Each method in `methods` narrows the active feature set. Methods run in
    order; each one operates only on the features that survived previous steps.

    Supervised methods (MIM, CMI) use `bag_targets` as labels.
    Unsupervised methods (CorrFilter, HighVariance, PCALoadings) ignore targets.

    Arguments:
        bag_matrix:        shape (n_bags, n_bag_features)
        bag_targets:       shape (n_bags,) — target values (raw, unencoded)
        bag_feature_names: list of length n_bag_features
        methods:           ordered list of method names to apply
        top_k:             for ranking methods — number of features to retain
        corr_threshold:    for CorrFilter — Pearson correlation threshold

    Returns:
        (filtered_matrix, filtered_feature_names)
    """
    # Build a temporary obs DataFrame for supervised scoring functions, which
    # expect a DataFrame with a column named "target". The bag_targets array
    # already reflects bag-level aggregation (mode for classification, mean for
    # regression) so it is directly usable as the target signal.
    target_obs_dataframe = pd.DataFrame({"target": bag_targets})

    # active_mask tracks which columns (features) are still in the running.
    active_mask = np.ones(len(bag_feature_names), dtype=bool)

    for method in methods:
        active_indices = np.where(active_mask)[0]
        if active_indices.size == 0:
            print("    Warning: no features remaining before applying {method} — stopping.")
            break

        active_matrix = bag_matrix[:, active_indices]
        active_names = [bag_feature_names[i] for i in active_indices]

        if method == "CorrFilter":
            # Returns a boolean mask: True = keep, False = drop as redundant.
            keep_mask = _score_corr_filter(active_matrix, active_names, corr_threshold)
            active_mask[active_indices[~keep_mask]] = False

        else:
            # Ranking methods: score each feature, keep top-K.
            if method == "MIM":
                scores = _score_mim(active_matrix, target_obs_dataframe, "target")
            elif method == "CMI":
                scores = _score_cmi(active_matrix, target_obs_dataframe, "target")
            elif method == "HighVariance":
                scores = _score_high_variance(active_matrix)
            else:  # PCALoadings
                scores = _score_pca_loadings(active_matrix, len(active_names))

            k = min(top_k, len(active_indices))
            top_local_indices = np.argsort(scores)[::-1][:k]

            # Reset mask: only the top-K survivors from this step are kept.
            new_mask = np.zeros(len(bag_feature_names), dtype=bool)
            new_mask[active_indices[top_local_indices]] = True
            active_mask = new_mask

        print(f"    {method}: {int(active_mask.sum())} features remaining")

    final_indices = np.where(active_mask)[0]
    filtered_matrix = bag_matrix[:, final_indices]
    filtered_names = [bag_feature_names[i] for i in final_indices]
    return filtered_matrix, filtered_names


# ---------------------------------------------------------------------------
# Internal: StandardCV evaluation
# ---------------------------------------------------------------------------


def _evaluate_standard_cv(
    model,
    feature_matrix: np.ndarray,
    target_labels: np.ndarray,
    subject_ids: np.ndarray,
    ml_config: dict,
) -> dict:
    """
    Run a stratified train/test split and optional k-fold cross-validation.

    Returns a dict of metric keys merged into the final results dict.
    """
    task_type = ml_config["task_type"]
    test_size = ml_config["test_size"]
    n_folds = ml_config["n_folds"]
    random_state = ml_config["random_state"]

    # Convert labels to string for classification to avoid dtype issues.
    # LabelEncoder converts strings to integers so that models with early_stopping=True
    # (e.g. MLP) can call np.isnan on predictions without a TypeError.
    label_encoder: Optional[LabelEncoder] = None
    if task_type == "classification":
        target_labels = target_labels.astype(str)
        label_encoder = LabelEncoder()
        encoded_labels: np.ndarray = label_encoder.fit_transform(target_labels)
    else:
        encoded_labels = target_labels

    X_train, X_test, y_train_enc, y_test_enc = train_test_split(
        feature_matrix,
        encoded_labels,
        test_size=test_size,
        random_state=random_state,
        stratify=encoded_labels if task_type == "classification" else None,
    )

    # Apply in-fold scaling: fit StandardScaler on train, transform both splits.
    if ml_config.get("scale_features_in_fold", False):
        main_scaler = StandardScaler()
        X_train = main_scaler.fit_transform(X_train)
        X_test = main_scaler.transform(X_test)

    # Decode back to original string labels for human-readable metrics.
    y_test = label_encoder.inverse_transform(y_test_enc) if label_encoder is not None else y_test_enc

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Got `batch_size`.*", category=UserWarning)
        model.fit(X_train, y_train_enc)
    y_pred_enc = model.predict(X_test)
    y_pred = label_encoder.inverse_transform(y_pred_enc) if label_encoder is not None else y_pred_enc

    # Capture probabilities and class names for later confusion matrix / ROC plots.
    y_pred_proba: Optional[np.ndarray] = None
    if task_type == "classification" and hasattr(model, "predict_proba"):
        try:
            y_pred_proba = model.predict_proba(X_test)
        except Exception:
            pass
    class_names_list = _sort_class_names_numerically(
        [str(c) for c in label_encoder.classes_]
        if label_encoder is not None
        else [str(c) for c in np.unique(target_labels)]
    )

    # ------------------------------------------------------------------
    # K-fold cross-validation scores (if requested)
    # ------------------------------------------------------------------
    cv_scores: List[float] = []
    train_cv_scores: List[float] = []
    if n_folds > 0:
        if task_type == "classification":
            cv_splitter = StratifiedKFold(
                n_splits=n_folds, shuffle=True, random_state=random_state
            )
        else:
            cv_splitter = KFold(
                n_splits=n_folds, shuffle=True, random_state=random_state
            )

        from sklearn.base import clone
        for train_idx, val_idx in cv_splitter.split(feature_matrix, encoded_labels):
            fold_model = clone(model)

            # Apply in-fold scaling independently for each k-fold split.
            if ml_config.get("scale_features_in_fold", False):
                kfold_scaler = StandardScaler()
                X_kf_train = kfold_scaler.fit_transform(feature_matrix[train_idx])
                X_kf_val = kfold_scaler.transform(feature_matrix[val_idx])
            else:
                X_kf_train = feature_matrix[train_idx]
                X_kf_val = feature_matrix[val_idx]

            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="Got `batch_size`.*", category=UserWarning)
                fold_model.fit(X_kf_train, encoded_labels[train_idx])
            if task_type == "classification":
                fold_score = float(fold_model.score(X_kf_val, encoded_labels[val_idx]))
                fold_train_score = float(fold_model.score(X_kf_train, encoded_labels[train_idx]))
            else:
                fold_pred = fold_model.predict(X_kf_val)
                fold_score = float(r2_score(target_labels[val_idx], fold_pred))
                fold_train_pred = fold_model.predict(X_kf_train)
                fold_train_score = float(r2_score(target_labels[train_idx], fold_train_pred))
            cv_scores.append(fold_score)
            train_cv_scores.append(fold_train_score)

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------
    if task_type == "classification":
        roc_auc = _compute_roc_auc(model, X_test, y_test_enc)
        return {
            "classification_report": classification_report(y_test, y_pred),
            "confusion_matrix":      confusion_matrix(y_test, y_pred, labels=class_names_list),
            "roc_auc":               roc_auc,
            "cv_scores_accuracy":    cv_scores,
            "mean_accuracy":         float(np.mean(cv_scores)) if cv_scores else float(model.score(X_test, y_test_enc)),
            "std_accuracy":          float(np.std(cv_scores)) if cv_scores else 0.0,
            "train_cv_scores_accuracy": train_cv_scores,
            "mean_train_accuracy":   float(np.mean(train_cv_scores)) if train_cv_scores else None,
            "std_train_accuracy":    float(np.std(train_cv_scores)) if train_cv_scores else None,
            "class_names":           class_names_list,
            "roc_y_true":            y_test,
            "roc_y_pred_proba":      y_pred_proba,
            # Regression keys absent for classification
            "r2_score": None, "mae": None, "rmse": None,
            "cv_scores_r2": None, "mean_r2": None, "std_r2": None,
            "train_cv_scores_r2": None, "mean_train_r2": None, "std_train_r2": None,
            "logo_per_subject_scores": None,
        }
    else:
        y_pred_float = y_test.astype(float) if hasattr(y_test, "astype") else np.array(y_test, dtype=float)
        y_pred_arr = np.array(y_pred, dtype=float)
        r2 = float(r2_score(y_pred_float, y_pred_arr))
        mae = float(mean_absolute_error(y_pred_float, y_pred_arr))
        rmse = float(root_mean_squared_error(y_pred_float, y_pred_arr))
        return {
            "r2_score":           r2,
            "mae":                mae,
            "rmse":               rmse,
            "cv_scores_r2":       cv_scores,
            "mean_r2":            float(np.mean(cv_scores)) if cv_scores else r2,
            "std_r2":             float(np.std(cv_scores)) if cv_scores else 0.0,
            "train_cv_scores_r2":  train_cv_scores,
            "mean_train_r2":       float(np.mean(train_cv_scores)) if train_cv_scores else None,
            "std_train_r2":        float(np.std(train_cv_scores)) if train_cv_scores else None,
            "class_names":        None,
            "roc_y_true":         None,
            "roc_y_pred_proba":   None,
            # Sample-level true/predicted for scatter plot (StandardCV has no subject structure).
            "logo_subject_true":  y_pred_float,
            "logo_subject_pred":  y_pred_arr,
            "logo_subject_ids":   None,
            # Classification keys absent for regression
            "classification_report": None, "confusion_matrix": None,
            "roc_auc": None, "cv_scores_accuracy": None,
            "mean_accuracy": None, "std_accuracy": None,
            "train_cv_scores_accuracy": None, "mean_train_accuracy": None, "std_train_accuracy": None,
            "logo_per_subject_scores": None,
        }


# ---------------------------------------------------------------------------
# Internal: SHAP helpers (shared by _run_shap and _compute_logo_shap)
# ---------------------------------------------------------------------------


def _build_fold_shap_explainer(
    model,
    model_name: str,
    X_train_fold: np.ndarray,
    task_type: str,
):
    """
    Build the appropriate SHAP explainer for a single LOGO fold model.

    Routes to TreeExplainer for tree-based models, LinearExplainer for
    ElasticNet, and KernelExplainer for everything else. KernelExplainer
    background is capped at 50 samples to keep per-fold computation tractable.

    Arguments:
        model:        Fitted sklearn/xgboost estimator for this fold.
        model_name:   One of "RandomForest", "XGBoost", "ExtraTrees",
                      "GradientBoosting", "ElasticNet", "MLP", "SVM",
                      "GaussianNB", "KNeighbors".
        X_train_fold: Training feature matrix for this fold (used as background
                      for LinearExplainer and KernelExplainer).
        task_type:    "classification" or "regression".

    Returns:
        Tuple of (explainer, explainer_type_string).
    """
    try:
        import shap
    except ImportError as exc:
        raise ImportError(
            "SHAP is not installed. Install it with: pip install shap>=0.44"
        ) from exc

    if model_name in ("RandomForest", "XGBoost", "ExtraTrees", "GradientBoosting"):
        explainer = shap.TreeExplainer(model)
        explainer_type = "TreeExplainer"
    elif model_name == "ElasticNet":
        explainer = shap.LinearExplainer(model, X_train_fold)
        explainer_type = "LinearExplainer"
    else:
        # KernelExplainer is slow — use a small background to stay tractable per fold.
        # Used for MLP, SVM, GaussianNB, KNeighbors — none has a dedicated exact
        # SHAP explainer, so a model-agnostic approximation is the correct choice.
        n_background = min(50, X_train_fold.shape[0])
        background = shap.sample(X_train_fold, n_background)
        predict_fn = (
            model.predict_proba if task_type == "classification" else model.predict
        )
        explainer = shap.KernelExplainer(predict_fn, background)
        explainer_type = "KernelExplainer"

    return explainer, explainer_type


def _extract_shap_as_class_arrays(shap_raw) -> List[np.ndarray]:
    """
    Convert raw SHAP output into a list of 2D arrays (n_samples, n_features), one per class.

    Preserves sign and per-sample variation — suitable for beeswarm plots.
    Binary classification returns a list of length 1 (class-1 slice).
    Same type-routing as _normalize_shap_to_importance but without collapsing samples.

    Arguments:
        shap_raw: Raw output of explainer.shap_values().

    Returns:
        List of 2D numpy arrays of shape (n_samples, n_features), one per class.
    """
    if isinstance(shap_raw, list):
        if len(shap_raw) > 2:
            return [np.asarray(arr) for arr in shap_raw]
        else:
            return [np.asarray(shap_raw[1] if len(shap_raw) > 1 else shap_raw[0])]
    elif isinstance(shap_raw, np.ndarray) and shap_raw.ndim == 3:
        n_classes = shap_raw.shape[2]
        if n_classes > 2:
            return [shap_raw[:, :, c] for c in range(n_classes)]
        else:
            return [shap_raw[:, :, 1]]
    else:
        return [np.asarray(shap_raw)]


def _normalize_shap_to_importance(
    shap_raw,
    n_features: int,
) -> List[np.ndarray]:
    """
    Convert raw SHAP output into a list of normalised importance vectors.

    Raw SHAP output format varies by explainer and class count:
      - TreeExplainer (multi-class)  → list of 2D arrays, one per class
      - LinearExplainer (multi-class) → 3D array (n_samples, n_features, n_classes)
      - Binary / regression           → 2D array or list of 2 arrays

    For each class this function computes mean(|SHAP|) per feature, then
    divides by the sum so the resulting vector sums to 1.  Folds where the
    model produces near-zero SHAP magnitudes (e.g. completely wrong folds)
    contribute a uniform vector rather than skewing the average.

    Arguments:
        shap_raw:   Raw output of explainer.shap_values().
        n_features: Number of features (used to build the uniform fallback).

    Returns:
        List of 1D numpy arrays of shape (n_features,), one per class.
        Binary classification and regression return a list of length 1
        (class index 1 for binary, single output for regression).
    """
    # Normalise into a list of 2D arrays (n_samples, n_features), one per class.
    if isinstance(shap_raw, list):
        if len(shap_raw) > 2:
            # Multi-class: one 2D array per class.
            class_arrays: List[np.ndarray] = shap_raw
        else:
            # Binary: use class-1 slice only.
            class_arrays = [shap_raw[1] if len(shap_raw) > 1 else shap_raw[0]]
    elif isinstance(shap_raw, np.ndarray) and shap_raw.ndim == 3:
        # LinearExplainer multi-class: shape (n_samples, n_features, n_classes).
        n_classes = shap_raw.shape[2]
        if n_classes > 2:
            class_arrays = [shap_raw[:, :, c] for c in range(n_classes)]
        else:
            class_arrays = [shap_raw[:, :, 1]]
    else:
        # 2D array (binary or regression).
        class_arrays = [shap_raw]

    importance_list: List[np.ndarray] = []
    for class_array in class_arrays:
        raw_importance = np.mean(np.abs(class_array), axis=0)   # shape (n_features,)
        total = raw_importance.sum()
        # Avoid division by zero when the fold model produces all-zero SHAP values.
        normalized = raw_importance / total if total > 0 else np.ones(n_features) / n_features
        importance_list.append(normalized.astype(np.float32))

    return importance_list


# ---------------------------------------------------------------------------
# Internal: LOGO evaluation
# ---------------------------------------------------------------------------


def _evaluate_logo_cv(
    model,
    feature_matrix: np.ndarray,
    target_labels: np.ndarray,
    subject_ids: np.ndarray,
    group_ids: np.ndarray,
    ml_config: dict,
    obs_dataframe: Optional[pd.DataFrame] = None,
    compute_shap: bool = False,
    model_name: str = "",
) -> dict:
    """
    Run Leave-One-Group-Out cross-validation with subject_ids as groups.

    Each fold: train on all subjects except one, test on the left-out subject.
    Reports per-subject score and overall mean ± std.

    When obs_dataframe is provided and contains a 'unique_subject_ID' column,
    the per-subject print uses that column for display instead of the raw
    subject_id_column value.

    When compute_shap=True, SHAP values are computed on each left-out subject
    directly after fitting the fold model (no second pass over the data).
    Per-fold importance vectors are normalised to sum to 1 before averaging,
    so every fold votes equally regardless of model confidence.  Folds whose
    accuracy is below ml_config["shap_logo_min_fold_score"] are skipped.

    Arguments:
        model:        Base (unfitted) sklearn/xgboost estimator — cloned per fold.
        feature_matrix: Full feature matrix (n_obs, n_features).
        target_labels:  1D array of label strings or floats.
        subject_ids:    1D array of subject identifiers. Used for per-subject
                        regression aggregation and single-subject display labels.
                        NOT used for fold splits (see group_ids).
        group_ids:      1D array that drives LeaveOneGroupOut splits. For standard
                        LOGO this equals subject_ids (one fold per subject). For
                        LAVO (logo_group_by_column set) this is the age/group value
                        per sample — all samples sharing the same value are left out
                        together in one fold.
        ml_config:      Full ml_config dict; reads task_type, subject_id_column,
                        shap_logo_min_fold_score.
        obs_dataframe:  Optional AnnData .obs DataFrame for display label lookup.
        compute_shap:   If True, collect normalised SHAP importance per fold.
        model_name:     Model name string, passed to _build_fold_shap_explainer.

    Returns:
        Dict of metric keys merged into the final results dict. When
        compute_shap=True, also contains logo_shap_* keys.
    """
    task_type = ml_config["task_type"]

    # LabelEncoder converts string labels to integers so that models with early_stopping=True
    # (e.g. MLP) can call np.isnan on predictions without a TypeError.
    label_encoder: Optional[LabelEncoder] = None
    if task_type == "classification":
        target_labels = target_labels.astype(str)
        label_encoder = LabelEncoder()
        encoded_labels: np.ndarray = label_encoder.fit_transform(target_labels)
        print(f"  [LabelEncoder] classes_: {label_encoder.classes_}  "
              f"(encoding: {dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))})")
    else:
        encoded_labels = target_labels

    # Build a mapping from raw subject_id → display label.
    # If the obs contains a 'unique_subject_ID' column, use it for display;
    # otherwise fall back to the raw subject_id value.
    subject_id_column = ml_config["subject_id_column"]
    display_label_for_subject: Dict[str, str] = {}
    if obs_dataframe is not None and "unique_subject_ID" in obs_dataframe.columns:
        for raw_id in np.unique(subject_ids):
            mask = obs_dataframe[subject_id_column] == raw_id
            unique_ids = obs_dataframe.loc[mask, "unique_subject_ID"].unique()
            display_label_for_subject[str(raw_id)] = str(unique_ids[0]) if len(unique_ids) > 0 else str(raw_id)

    scale_in_fold = ml_config.get("scale_features_in_fold", False)

    logo = LeaveOneGroupOut()
    per_subject_scores: Dict[str, float] = {}
    cv_scores: List[float] = []
    train_cv_scores: List[float] = []

    # Accumulate out-of-fold predictions across all LOGO folds so that
    # the summary metrics (classification_report, ROC AUC, r2…) are computed
    # on genuinely held-out data — not on the final model's training set.
    oof_true: List[np.ndarray] = []
    oof_pred: List[np.ndarray] = []
    oof_proba: List[np.ndarray] = []   # classification only; may stay empty

    # For regression LOGO: accumulate one (true_age, mean_pred_age) per subject.
    # R² computed per fold is always 0 when test bags all share the same age
    # (SS_tot=0). The correct metric is R² across subjects.
    subject_level_true: List[float] = []
    subject_level_pred: List[float] = []
    subject_level_ids: List[str] = []

    # SHAP accumulation — filled only when compute_shap=True.
    per_fold_importances: List[List[np.ndarray]] = []
    shap_fold_subjects: List[str] = []
    fold_explainer_type: str = "Unknown"
    shap_logo_min_fold_score: Optional[float] = ml_config.get("shap_logo_min_fold_score", None)
    # Per-sample OOF SHAP values for beeswarm — filled alongside per_fold_importances.
    oof_shap_beeswarm: List[List[np.ndarray]] = []
    oof_X_beeswarm: List[np.ndarray] = []
    # Per-row subject and condition metadata for SHAP endotype analysis.
    oof_unique_subject_ids_beeswarm: List[np.ndarray] = []
    oof_conditions_beeswarm: List[np.ndarray] = []

    from sklearn.base import clone

    print()
    print("  LOGO/LAVO-CV per-fold results:")
    for train_idx, test_idx in logo.split(feature_matrix, target_labels, groups=group_ids):
        test_subjects = np.unique(subject_ids[test_idx])
        subject_label = str(test_subjects[0])

        # Fold metadata: how many unique subjects, which group value, display label.
        test_unique_subjects = np.unique(subject_ids[test_idx])
        n_test_subjects = len(test_unique_subjects)
        fold_group_value = str(np.unique(group_ids[test_idx])[0])
        if n_test_subjects == 1:
            display_label = display_label_for_subject.get(subject_label, subject_label)
        else:
            display_label = f"Age group {fold_group_value} [{n_test_subjects} subjects]"

        fold_model = clone(model)

        # Apply in-fold scaling: fit StandardScaler on train, transform both splits.
        # This prevents data leakage and ensures all features are on comparable scales.
        if scale_in_fold:
            fold_scaler = StandardScaler()
            X_train_fold = fold_scaler.fit_transform(feature_matrix[train_idx])
            X_test_fold = fold_scaler.transform(feature_matrix[test_idx])
        else:
            X_train_fold = feature_matrix[train_idx]
            X_test_fold = feature_matrix[test_idx]

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Got `batch_size`.*", category=UserWarning)
            fold_model.fit(X_train_fold, encoded_labels[train_idx])

        fold_pred = fold_model.predict(X_test_fold)
        if label_encoder is not None:
            fold_pred = label_encoder.inverse_transform(fold_pred)
        oof_true.append(target_labels[test_idx])
        oof_pred.append(fold_pred)

        if task_type == "classification":
            fold_score = float(fold_model.score(X_test_fold, encoded_labels[test_idx]))
            fold_train_score = float(fold_model.score(X_train_fold, encoded_labels[train_idx]))
            if hasattr(fold_model, "predict_proba"):
                oof_proba.append(fold_model.predict_proba(X_test_fold))
        else:
            # Regression: accumulate one (true_age, mean_pred_age) per subject in fold.
            # Standard LOGO: one subject per fold — loop runs once, backward-compatible.
            # LAVO: multiple subjects per fold — loop runs once per subject.
            # fold_score = -mean_fold_MAE so "higher = better" convention is preserved.
            fold_mae_list: List[float] = []
            for subj in test_unique_subjects:
                subj_mask = subject_ids[test_idx] == subj
                subj_true = float(np.unique(target_labels[test_idx][subj_mask].astype(float))[0])
                subj_pred_mean = float(np.mean(fold_pred[subj_mask].astype(float)))
                subject_level_true.append(subj_true)
                subject_level_pred.append(subj_pred_mean)
                subject_level_ids.append(str(subj))
                fold_mae_list.append(abs(subj_pred_mean - subj_true))
            fold_abs_error = float(np.mean(fold_mae_list))
            fold_score = -fold_abs_error
            fold_train_pred = fold_model.predict(X_train_fold)
            # Train R² is valid: train set spans many subjects with varied ages.
            fold_train_score = float(
                r2_score(target_labels[train_idx].astype(float), fold_train_pred.astype(float))
            )

        per_subject_scores[display_label] = fold_score
        cv_scores.append(fold_score)
        train_cv_scores.append(fold_train_score)
        if task_type == "classification":
            subject_target_class = str(np.unique(target_labels[test_idx])[0])
            print(f"    {display_label:<42} {subject_target_class:<15} accuracy: {fold_score:.3f}")
        else:
            fold_mean_true = float(np.mean(subject_level_true[-n_test_subjects:]))
            fold_mean_pred = float(np.mean(subject_level_pred[-n_test_subjects:]))
            print(
                f"    {display_label:<42} "
                f"true: {fold_mean_true:<8.1f} "
                f"pred: {fold_mean_pred:<8.1f} "
                f"MAE: {fold_abs_error:.3f}"
            )
            # For multi-subject folds (LAVO), print per-subject detail lines.
            if n_test_subjects > 1:
                for subj, subj_true, subj_pred_mean in zip(
                    test_unique_subjects,
                    subject_level_true[-n_test_subjects:],
                    subject_level_pred[-n_test_subjects:],
                ):
                    subj_display = display_label_for_subject.get(str(subj), str(subj))
                    subj_mae = abs(subj_pred_mean - subj_true)
                    print(f"        {subj_display:<30} pred: {subj_pred_mean:<8.1f} MAE: {subj_mae:.3f}")

        # ---- SHAP collection (inside the fold, no extra training) ----
        if compute_shap:
            # For regression LOGO, fold_score is -MAE (not R²), so the
            # shap_logo_min_fold_score threshold (designed for accuracy/R²) does
            # not apply — include every fold in the SHAP average.
            shap_threshold_exceeded = (
                task_type == "classification"
                and shap_logo_min_fold_score is not None
                and fold_score < shap_logo_min_fold_score
            )
            if shap_threshold_exceeded:
                print(
                    f"    [SHAP-LOGO] Fold {display_label} skipped "
                    f"(accuracy {fold_score:.3f} < threshold {shap_logo_min_fold_score})"
                )
            else:
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore")
                    fold_explainer, fold_explainer_type = _build_fold_shap_explainer(
                        fold_model, model_name, X_train_fold, task_type
                    )
                    # For KernelExplainer cap evaluation set to keep it tractable.
                    X_test_fold_shap = X_test_fold
                    if fold_explainer_type == "KernelExplainer":
                        X_test_fold_shap = X_test_fold_shap[: min(20, X_test_fold_shap.shape[0])]
                    fold_shap_raw = fold_explainer.shap_values(X_test_fold_shap)
                fold_importance = _normalize_shap_to_importance(
                    fold_shap_raw, n_features=feature_matrix.shape[1]
                )
                per_fold_importances.append(fold_importance)
                shap_fold_subjects.append(display_label)

                # Collect per-sample SHAP values for the OOF beeswarm.
                # Scale-normalise per fold (divide by mean absolute SHAP sum) so all
                # folds contribute equally to the beeswarm, regardless of model confidence.
                fold_class_arrays = _extract_shap_as_class_arrays(fold_shap_raw)
                fold_scale = float(
                    np.mean([np.sum(np.mean(np.abs(arr), axis=0)) for arr in fold_class_arrays])
                )
                if fold_scale > 0:
                    fold_class_arrays = [arr / fold_scale for arr in fold_class_arrays]
                oof_shap_beeswarm.append(fold_class_arrays)
                oof_X_beeswarm.append(X_test_fold_shap)
                # Store subject and condition for each OOF row so SHAP endotype
                # clustering can label which subject belongs to which cluster.
                n_test_shap = X_test_fold_shap.shape[0]
                oof_unique_subject_ids_beeswarm.append(
                    np.full(n_test_shap, display_label, dtype=object)
                )
                oof_conditions_beeswarm.append(target_labels[test_idx[:n_test_shap]])

    mean_score = float(np.mean(cv_scores))
    std_score = float(np.std(cv_scores))
    if task_type == "regression" and subject_level_true:
        # Subject-level R²: the only valid global metric for LOGO regression.
        # Each subject contributes one (true_age, mean_predicted_age) pair.
        subject_r2 = float(r2_score(subject_level_true, subject_level_pred))
        subject_mae = float(mean_absolute_error(subject_level_true, subject_level_pred))
        pearson_r = float(np.corrcoef(subject_level_true, subject_level_pred)[0, 1])
        print(
            f"  => Subject-level R²: {subject_r2:.3f}  "
            f"MAE: {subject_mae:.3f}  "
            f"Pearson r: {pearson_r:.3f}"
        )
        print(f"  => Mean per-subject MAE: {-mean_score:.3f} ± {std_score:.3f}")
    else:
        print(f"  => Mean accuracy: {mean_score:.3f} ± {std_score:.3f}")

    # Aggregate out-of-fold arrays for summary metrics.
    all_true = np.concatenate(oof_true)
    all_pred = np.concatenate(oof_pred)

    # Fit a final model on all data so downstream callers can access it.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Got `batch_size`.*", category=UserWarning)
        model.fit(feature_matrix, encoded_labels)

    # Average per-fold SHAP importance vectors collected inside the fold loop.
    logo_shap_importance: Optional[List[np.ndarray]] = None
    if compute_shap and per_fold_importances:
        n_classes_shap = len(per_fold_importances[0])
        logo_shap_importance = [
            np.mean([fold[c] for fold in per_fold_importances], axis=0)
            for c in range(n_classes_shap)
        ]
        n_used = len(per_fold_importances)
        n_total = len(cv_scores)
        print(
            f"  [SHAP-LOGO] Averaged normalised importance across "
            f"{n_used}/{n_total} folds."
        )
    elif compute_shap:
        print(
            "  [SHAP-LOGO] No folds passed the accuracy threshold — "
            "SHAP importance is unavailable."
        )

    # Concatenate per-sample OOF SHAP arrays for beeswarm plot.
    logo_oof_shap: Optional[List[np.ndarray]] = None
    logo_oof_X: Optional[np.ndarray] = None
    logo_oof_unique_subject_ids: Optional[np.ndarray] = None
    logo_oof_conditions: Optional[np.ndarray] = None
    if compute_shap and oof_shap_beeswarm:
        n_classes_beeswarm = len(oof_shap_beeswarm[0])
        logo_oof_shap = [
            np.concatenate([fold[c] for fold in oof_shap_beeswarm], axis=0)
            for c in range(n_classes_beeswarm)
        ]
        logo_oof_X = np.concatenate(oof_X_beeswarm, axis=0)
        logo_oof_unique_subject_ids = np.concatenate(oof_unique_subject_ids_beeswarm)
        logo_oof_conditions = np.concatenate(oof_conditions_beeswarm)

    logo_shap_dict: dict = {
        "logo_shap_importance":    logo_shap_importance,
        "logo_shap_n_folds_used":  len(per_fold_importances),
        "logo_shap_subjects_used": shap_fold_subjects,
        "logo_shap_explainer_type": fold_explainer_type,
        "logo_shap_oof_values":           logo_oof_shap,
        "logo_shap_oof_features":         logo_oof_X,
        "logo_shap_oof_unique_subject_ids": logo_oof_unique_subject_ids,
        "logo_shap_oof_conditions":       logo_oof_conditions,
    }

    if task_type == "classification":
        # ROC AUC from OOF probabilities when shapes are consistent across folds.
        # Shapes can mismatch when a fold's training set contains only 1 class
        # (predict_proba returns 1 column instead of n_classes) — in that case
        # we fall back to NaN rather than crashing.
        roc_auc = float("nan")
        all_proba_concat: Optional[np.ndarray] = None
        if oof_proba:
            proba_shapes = [p.shape[1] for p in oof_proba]
            if len(set(proba_shapes)) == 1:
                all_proba_concat = np.concatenate(oof_proba, axis=0)
                classes = np.unique(all_true)
                try:
                    if len(classes) == 2:
                        roc_auc = float(roc_auc_score(all_true, all_proba_concat[:, 1]))
                    else:
                        roc_auc = float(
                            roc_auc_score(all_true, all_proba_concat, multi_class="ovr", average="macro")
                        )
                except Exception:
                    roc_auc = float("nan")
        class_names_logo = _sort_class_names_numerically([str(c) for c in np.unique(all_true)])
        return {
            "classification_report": classification_report(all_true, all_pred),
            "confusion_matrix":      confusion_matrix(all_true, all_pred, labels=class_names_logo),
            "roc_auc":               roc_auc,
            "cv_scores_accuracy":    cv_scores,
            "mean_accuracy":         mean_score,
            "std_accuracy":          std_score,
            "train_cv_scores_accuracy": train_cv_scores,
            "mean_train_accuracy":   float(np.mean(train_cv_scores)) if train_cv_scores else None,
            "std_train_accuracy":    float(np.std(train_cv_scores)) if train_cv_scores else None,
            "logo_per_subject_scores": per_subject_scores,
            "class_names":           class_names_logo,
            "roc_y_true":            all_true,
            "roc_y_pred_proba":      all_proba_concat,
            "r2_score": None, "mae": None, "rmse": None,
            "cv_scores_r2": None, "mean_r2": None, "std_r2": None,
            "train_cv_scores_r2": None, "mean_train_r2": None, "std_train_r2": None,
            "logo_group_by_column":  ml_config.get("logo_group_by_column", None),
            **logo_shap_dict,
        }
    else:
        # Subject-level metrics: aggregate mean prediction per subject, then evaluate.
        # This is the correct metric for LOGO regression — bag-level R² is inflated
        # because all bags in one fold share the same true age (SS_tot = 0 per fold).
        subj_true = np.array(subject_level_true, dtype=float)
        subj_pred = np.array(subject_level_pred, dtype=float)
        subject_r2 = float(r2_score(subj_true, subj_pred))
        subject_mae = float(mean_absolute_error(subj_true, subj_pred))
        subject_rmse = float(root_mean_squared_error(subj_true, subj_pred))
        subject_pearson_r = float(np.corrcoef(subj_true, subj_pred)[0, 1]) if len(subj_true) > 1 else float("nan")
        return {
            # Primary metrics (subject-level — one point per subject).
            "r2_score":           subject_r2,
            "mae":                subject_mae,
            "rmse":               subject_rmse,
            "pearson_r":          subject_pearson_r,
            # Per-fold scores are -MAE (higher = better) to preserve "higher is better" convention.
            "cv_scores_r2":       cv_scores,
            "mean_r2":            subject_r2,
            "std_r2":             std_score,
            "train_cv_scores_r2":  train_cv_scores,
            "mean_train_r2":       float(np.mean(train_cv_scores)) if train_cv_scores else None,
            "std_train_r2":        float(np.std(train_cv_scores)) if train_cv_scores else None,
            "logo_per_subject_scores": per_subject_scores,
            # Subject-level true/predicted (+ IDs) for scatter plot downstream.
            "logo_subject_true":  subj_true,
            "logo_subject_pred":  subj_pred,
            "logo_subject_ids":   subject_level_ids,
            "class_names":        None,
            "roc_y_true":         None,
            "roc_y_pred_proba":   None,
            "classification_report": None, "confusion_matrix": None,
            "roc_auc": None, "cv_scores_accuracy": None,
            "mean_accuracy": None, "std_accuracy": None,
            "train_cv_scores_accuracy": None, "mean_train_accuracy": None, "std_train_accuracy": None,
            "logo_group_by_column":  ml_config.get("logo_group_by_column", None),
            **logo_shap_dict,
        }


# ---------------------------------------------------------------------------
# Internal: LOGO SHAP plot
# ---------------------------------------------------------------------------


def _plot_logo_shap_bar(
    averaged_importance: List[np.ndarray],
    feature_names: List[str],
    n_folds_used: int,
    class_names: List[str],
    model_name: str,
    target_obs_column: str,
    max_display: int = 15,
    show_per_class: bool = False,
    species_label: str = "",
) -> None:
    """
    Plot normalised mean |SHAP| importance averaged across LOGO folds.

    Uses matplotlib directly (not shap.summary_plot) because the input is
    already aggregated — one importance value per feature per class.

    For multiclass models a single global chart is produced first (mean |SHAP|
    across all classes) — this answers "which features matter most overall?".
    When show_per_class=True, one additional chart per class is appended.

    For binary classification or regression (one class vector): only one chart
    is produced (the global chart and the single-class chart are identical).

    Arguments:
        averaged_importance: List of 1D arrays (n_features,), one per class,
                             as returned by _evaluate_logo_cv logo_shap_importance.
        feature_names:       Feature column names, length must match arrays.
        n_folds_used:        Number of folds that contributed to the average.
        class_names:         Class label strings, same order as averaged_importance.
        model_name:          Model name for the plot title.
        target_obs_column:   Prediction target column name for the plot title.
        max_display:         Maximum number of features to display per chart.
        show_per_class:      When True, plot one additional chart per class after
                             the global chart (default False).
    """
    import matplotlib.pyplot as plt

    feature_names_array = np.array(feature_names)

    print()
    print(
        f"  [SHAP-LOGO] Bar chart: normalised mean |SHAP| averaged across {n_folds_used} folds."
    )
    print("  [SHAP-LOGO] X-axis: normalised mean |SHAP| (each fold sums to 1 before averaging).")
    print("  [SHAP-LOGO] Features sorted by descending average importance.")

    species_prefix = f"{species_label}\n" if species_label else ""
    title_suffix = (
        f"{species_prefix}Normalised mean |SHAP| — {n_folds_used} folds used\n"
        f"{model_name} | LOGO | Target: {target_obs_column}"
    )

    def _draw_bar(importance_vector: np.ndarray, class_label: str) -> None:
        """Draw one horizontal bar chart for a single class importance vector."""
        sorted_indices = np.argsort(importance_vector)[::-1]
        top_indices = sorted_indices[:max_display]
        top_importance = importance_vector[top_indices]
        top_names = feature_names_array[top_indices]

        # If there are more features than max_display, add a "Sum of N other features" row
        # at the bottom — matching shap.summary_plot convention.
        n_remaining = len(importance_vector) - max_display
        if n_remaining > 0:
            remaining_sum = importance_vector[sorted_indices[max_display:]].sum()
            top_importance = np.append(remaining_sum, top_importance[::-1])
            top_names = np.append(f"Sum of {n_remaining} other features", top_names[::-1])
        else:
            # Reverse so highest bar appears at top of chart.
            top_importance = top_importance[::-1]
            top_names = top_names[::-1]

        fig, ax = plt.subplots(figsize=(8, max(4, len(top_names) * 0.4 + 1)))
        # Colour the "Sum" row differently so it is visually distinct.
        bar_colors = [
            "#aec7e8" if str(name).startswith("Sum of") else "#1f77b4"
            for name in top_names
        ]
        ax.barh(top_names, top_importance, color=bar_colors)
        ax.set_xlabel("Normalised mean |SHAP| (averaged across folds)")
        ax.set_xlim(left=0)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        class_line = f"Class: {class_label}\n" if class_label else ""
        ax.set_title(
            f"SHAP Feature Importance (LOGO)\n{class_line}{title_suffix}",
            fontsize=9,
            pad=10,
        )
        plt.tight_layout()
        plt.show()

    is_multiclass = len(averaged_importance) > 1

    if is_multiclass:
        # Global chart: average |SHAP| across all classes → one unified ranking.
        # This answers "which features matter most for discriminating any class?"
        global_importance = np.mean(np.stack(averaged_importance), axis=0)
        all_class_labels = ", ".join(class_names) if class_names else "all classes"
        _draw_bar(global_importance, f"Global (mean across: {all_class_labels})")

        if show_per_class:
            for class_idx, importance_vector in enumerate(averaged_importance):
                class_label = (
                    class_names[class_idx] if class_idx < len(class_names) else str(class_idx)
                )
                _draw_bar(importance_vector, class_label)
    else:
        # Binary or regression: global and per-class are the same — one chart only.
        binary_label = class_names[1] if len(class_names) > 1 else (class_names[0] if class_names else "")
        _draw_bar(averaged_importance[0], binary_label)


def _draw_beeswarm_with_sum_row(
    sv: np.ndarray,
    feature_df: pd.DataFrame,
    max_display: int,
    title: str,
) -> None:
    """
    Draw a SHAP beeswarm plot with a guaranteed 'Sum of N other features' row at the bottom.

    When n_features > max_display, the top (max_display - 1) features are shown
    (sorted by mean |SHAP| descending, highest at top) and the last row displays
    the row-wise sum of the remaining SHAP values.  When all features fit within
    max_display, the plot is drawn normally with no sum row (nothing to aggregate).

    The sum row is forced to the bottom by passing the augmented array with the most
    important feature at index 0 and the sum row as the last column, then calling
    shap.summary_plot with sort=False, which renders index 0 at the top and the last
    index at the bottom.

    Arguments:
        sv:          2-D SHAP array (n_samples, n_features).
        feature_df:  Original feature values (n_samples, n_features), used to colour
                     beeswarm dots (red = high value, blue = low value).
        max_display: Maximum number of rows to render, including the sum row.
        title:       Figure suptitle.
    """
    import matplotlib.pyplot as plt
    import shap

    # Ensure sv is 2-D. TreeExplainer on some GPU models (e.g. cuML RandomForest)
    # can return (n_samples, n_features, 2) for binary classification; take class-1.
    if sv.ndim == 3:
        sv = sv[:, :, 1] if sv.shape[2] >= 2 else sv[:, :, 0]

    n_features = sv.shape[1]

    if n_features <= max_display:
        # Every feature fits — nothing left to aggregate.
        plt.figure()
        shap.summary_plot(sv, feature_df, max_display=max_display, show=False)
        fig = plt.gcf()
        for ax in fig.get_axes():
            ax.set_title("")
        fig.suptitle(title, fontsize=10)
        plt.subplots_adjust(top=0.88)
        plt.show()
        return

    # Rank all features by mean absolute SHAP value (descending).
    mean_abs_shap = np.mean(np.abs(sv), axis=0)
    sorted_descending = np.argsort(mean_abs_shap)[::-1]

    n_show = max_display - 1          # slots for real features
    top_idx = sorted_descending[:n_show]   # most important n_show features (descending)
    rest_idx = sorted_descending[n_show:]  # features that will be aggregated
    n_rest = len(rest_idx)
    sum_col_name = f"Sum of {n_rest} other features"

    # With sort=False, shap.summary_plot renders array column 0 at the TOP and the
    # last column at the BOTTOM.
    # Desired display order: top = most important feature, bottom = sum row.
    # → array order (index 0 first): [most_important, ..., least_shown, sum]
    sv_sum = sv[:, rest_idx].sum(axis=1, keepdims=True)  # (n_samples, 1)
    sv_display = np.concatenate([sv[:, top_idx], sv_sum], axis=1)

    feat_sum_values = feature_df.iloc[:, rest_idx].sum(axis=1).values
    top_cols = [feature_df.columns[i] for i in top_idx]
    feat_display = pd.DataFrame(
        np.column_stack([feature_df.iloc[:, top_idx].values, feat_sum_values]),
        columns=top_cols + [sum_col_name],
    )

    plt.figure()
    shap.summary_plot(
        sv_display,
        feat_display,
        max_display=n_show + 1,
        sort=False,   # preserve our pre-sorted order so the sum row stays at the bottom
        show=False,
    )
    fig = plt.gcf()
    for ax in fig.get_axes():
        ax.set_title("")
    fig.suptitle(title, fontsize=10)
    plt.subplots_adjust(top=0.88)
    plt.show()


def _draw_logo_shap_beeswarm(
    oof_shap: List[np.ndarray],
    oof_features: np.ndarray,
    feature_names: List[str],
    n_folds_used: int,
    class_names: Optional[List[str]],
    model_name: str,
    target_obs_column: str,
    max_display: int = 15,
    show_per_class: bool = False,
    species_label: str = "",
) -> None:
    """
    Draw a beeswarm summary plot from LOGO out-of-fold SHAP values.

    Each dot is one out-of-fold test sample (one observation left out per fold).
    SHAP values are scale-normalised per fold so all folds contribute equally.
    The last row ('Sum of N other features') is rendered automatically by
    shap.summary_plot when the feature count exceeds max_display.

    For multiclass models a single global beeswarm is produced first (mean SHAP
    across all classes, per sample per feature) so dot colour and position reflect
    the average effect across all classes.  When show_per_class=True, one
    additional beeswarm per class is appended.

    Arguments:
        oof_shap:          List of 2D arrays (n_oof_samples, n_features), one per class,
                           as returned by _evaluate_logo_cv logo_shap_oof_values.
        oof_features:      2D array (n_oof_samples, n_features) of feature values
                           used to colour the dots (red = high, blue = low).
        feature_names:     Feature column names, length must match n_features.
        n_folds_used:      Number of LOGO folds that contributed to the plot.
        class_names:       Class label strings in model class order.
        model_name:        Model name shown in the plot title.
        target_obs_column: Prediction target column name shown in the plot title.
        max_display:       Max features shown per chart; remaining are summed.
        show_per_class:    When True, plot one additional beeswarm per class after
                           the global beeswarm (default False).
    """
    feature_df = pd.DataFrame(oof_features, columns=feature_names)
    is_multiclass = len(oof_shap) > 1
    species_prefix = f"{species_label}\n" if species_label else ""
    title_prefix = (
        f"{species_prefix}SHAP Beeswarm (LOGO — {n_folds_used} folds, "
        f"{oof_features.shape[0]} OOF samples) | "
        f"{model_name} | Target: {target_obs_column}"
    )

    print()
    print(
        f"  [SHAP-LOGO] Beeswarm: {oof_features.shape[0]} out-of-fold samples "
        f"across {n_folds_used} folds."
    )
    print("  [SHAP-LOGO] Colour: red = high feature value, blue = low feature value.")
    print("  [SHAP-LOGO] Last row aggregates all features beyond the top displayed.")

    if is_multiclass:
        # Global beeswarm: mean signed SHAP across all classes per sample.
        # Dot position = average influence direction; colour = feature value.
        all_class_labels = ", ".join(class_names) if class_names else "all classes"
        _draw_beeswarm_with_sum_row(
            np.mean(np.stack(oof_shap), axis=0),
            feature_df, max_display,
            f"SHAP Beeswarm — Global (mean across: {all_class_labels})\n{title_prefix}",
        )

        # Per-class beeswarms: always shown for multiclass so each class's
        # directional signal is readable without the cross-class averaging artefact.
        print(
            f"  [SHAP-LOGO] Drawing one per-class beeswarm for each of the "
            f"{len(oof_shap)} classes."
        )
        for class_index, class_sv in enumerate(oof_shap):
            class_label = (
                class_names[class_index]
                if class_names and class_index < len(class_names)
                else f"Class {class_index}"
            )
            _draw_beeswarm_with_sum_row(
                class_sv, feature_df, max_display,
                f"SHAP Beeswarm — Target class: {class_label}\n{title_prefix}",
            )
    else:
        # Binary: oof_shap[0] contains SHAP values for the positive class (class index 1).
        # Positive SHAP => pushes toward this class; negative SHAP => pushes toward class 0.
        binary_class_label = (
            class_names[1] if class_names and len(class_names) > 1 else "class 1"
        )
        binary_class_other_label = (
            class_names[0] if class_names and len(class_names) > 0 else "class 0"
        )
        print(
            f"  [SHAP-LOGO] Binary beeswarm — SHAP values computed for class: {binary_class_label}. "
            f"Positive SHAP => predicts {binary_class_label} | "
            f"Negative SHAP => predicts {binary_class_other_label}."
        )
        _draw_beeswarm_with_sum_row(
            oof_shap[0], feature_df, max_display,
            f"Class: {binary_class_label}\n{title_prefix}",
        )


# ---------------------------------------------------------------------------
# Internal: SHAP
# ---------------------------------------------------------------------------


def _run_shap(
    model,
    model_name: str,
    X_train: np.ndarray,
    X_test: np.ndarray,
    feature_names: List[str],
    ml_config: dict,
    class_names: Optional[List[str]] = None,
    evaluation_strategy: str = "",
    target_obs_column: str = "",
    species_label: str = "",
) -> dict:
    """
    Compute SHAP values and produce summary and waterfall plots.

    Explainer routing:
      TreeExplainer   → RandomForest, XGBoost, ExtraTrees, GradientBoosting (exact, fast)
      LinearExplainer → ElasticNet (exact, fast)
      KernelExplainer → MLP, SVM, GaussianNB, KNeighbors (slow — background
                        capped at 100 samples, evaluation capped at 50 samples)

    Arguments:
        model:               Fitted sklearn / xgboost estimator.
        model_name:          One of ALLOWED_MODEL_NAMES.
        X_train:             Training data (used as background for KernelExplainer).
        X_test:              Test data to explain.
        feature_names:       Names aligned with X_test columns.
        ml_config:           Full config dict.
        class_names:         Optional list of class label strings in the order the
                             model's classes_ attribute.  Used to label the bars in
                             multi-class summary plots instead of "class 0", "class 1".
        evaluation_strategy: Evaluation strategy label (e.g. "StandardCV" or "LOGO")
                             shown in the plot title.
        target_obs_column:   Name of the .obs column used as the prediction target,
                             shown in the plot title so the reader knows which
                             biological outcome the SHAP values explain.

    Returns:
        dict with keys: shap_values, shap_explainer_type, shap_feature_names.
    """
    try:
        import shap
    except ImportError as exc:
        raise ImportError(
            "SHAP is not installed. Install it with: pip install shap>=0.44"
        ) from exc

    import matplotlib.pyplot as plt

    task_type = ml_config["task_type"]
    waterfall_index = ml_config.get("shap_waterfall_sample_index", 0)
    shap_per_class_plots = ml_config.get("shap_per_class_plots", False)

    if model_name in ("RandomForest", "XGBoost", "ExtraTrees", "GradientBoosting"):
        explainer = shap.TreeExplainer(model)
        explainer_type = "TreeExplainer"
    elif model_name == "ElasticNet":
        explainer = shap.LinearExplainer(model, X_train)
        explainer_type = "LinearExplainer"
    else:
        # KernelExplainer is slow — cap background and evaluation sets.
        n_background = min(100, X_train.shape[0])
        background = shap.sample(X_train, n_background)
        print(
            f"  [SHAP] KernelExplainer ({model_name}): using {n_background} background "
            f"samples. Evaluating on max 50 test samples — this may take a moment …"
        )
        explainer = shap.KernelExplainer(model.predict_proba if task_type == "classification" else model.predict, background)
        explainer_type = "KernelExplainer"
        X_test = X_test[: min(50, X_test.shape[0])]

    shap_values = explainer.shap_values(X_test)

    # Normalise shap_values into a consistent shap_matrix and an is_multiclass flag.
    #
    # TreeExplainer  → list of 2D arrays, one per class, length = n_classes
    # LinearExplainer (multi-class LogisticRegression) → 3D array (n_samples, n_features, n_classes)
    # Binary/regression → 2D array OR list of 2 arrays
    #
    # For bar-plot summary_plot we want:
    #   multi-class : pass the full list / 3D array so SHAP draws one bar per class
    #   binary      : pass class-1 2D array only (standard convention)
    feature_array = pd.DataFrame(X_test, columns=feature_names)
    is_multiclass = False

    if isinstance(shap_values, list):
        if len(shap_values) > 2:
            is_multiclass = True
            shap_matrix = shap_values          # list of 2D arrays
        else:
            shap_matrix = shap_values[1] if len(shap_values) > 1 else shap_values[0]
    elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
        # 3-D array: shape (n_samples, n_features, n_classes)
        # Produced by LinearExplainer (multi-class) or TreeExplainer on cuML models (binary).
        if shap_values.shape[2] > 2:
            is_multiclass = True
            shap_matrix = shap_values
        else:
            # Binary: keep only the class-1 slice → 2-D (n_samples, n_features).
            # Mirrors the list branch above so downstream code always receives a 2-D array.
            shap_matrix = shap_values[:, :, 1]
    else:
        shap_matrix = shap_values

    species_prefix = f"{species_label}\n" if species_label else ""
    shap_title_prefix = f"{species_prefix}{model_name} | {evaluation_strategy} | Target: {target_obs_column}"
    SHAP_MAX_DISPLAY = 15

    # Determine the named positive/target class once so every plot can use it.
    # For binary classification this is class index 1 (the "positive" class).
    # For multi-class or regression there is no single target class value.
    binary_class_label: str = (
        class_names[1]
        if (not is_multiclass) and class_names and len(class_names) > 1 and task_type == "classification"
        else ""
    )

    print()
    print("  [SHAP] Interpreting the SHAP Feature Importance plots:")
    print("    Bar chart  — mean(|SHAP|) per feature; shows overall importance across classes.")
    print("    Beeswarm   — each dot is one test sample; shows direction and spread of impact.")
    print("    Colour (beeswarm): pink/red = high feature value, blue/purple = low feature value.")
    print("    Features are sorted top-to-bottom by mean absolute SHAP value.")
    if binary_class_label:
        print(f"    Positive SHAP pushes prediction TOWARD '{binary_class_label}'; negative pushes AWAY from it.")
    elif is_multiclass:
        print("    Positive SHAP pushes prediction TOWARD the labelled class; negative pushes AWAY.")
    else:
        print("    Positive SHAP pushes prediction TOWARD higher values; negative pushes toward lower values.")
    print("    The last beeswarm row ('Sum of N other features') aggregates all remaining features.")

    # ---------------------------------------------------------------
    # Plot 1: Bar chart — mean(|SHAP|) overview, all classes at once.
    # ---------------------------------------------------------------
    plt.figure()
    shap.summary_plot(
        shap_matrix,
        feature_array,
        plot_type="bar",
        max_display=SHAP_MAX_DISPLAY,
        class_names=class_names if is_multiclass else None,
        show=False,
    )
    # SHAP may label legend entries "Class 0", "Class 1" etc. in older versions.
    # Patch them to use the actual class names when available.
    if class_names is not None and is_multiclass:
        bar_legend = plt.gca().get_legend()
        if bar_legend is not None:
            for legend_text in bar_legend.get_texts():
                raw_label = legend_text.get_text()
                if raw_label.lower().startswith("class "):
                    try:
                        class_index = int(raw_label.split(" ")[-1])
                        if class_index < len(class_names):
                            legend_text.set_text(class_names[class_index])
                    except ValueError:
                        pass
    bar_fig = plt.gcf()
    for bar_ax in bar_fig.get_axes():
        bar_ax.set_title("")
    if is_multiclass:
        bar_title_line1 = "SHAP Feature Importance — All classes"
    elif binary_class_label:
        bar_title_line1 = f"SHAP Feature Importance — Target class: {binary_class_label}"
    else:
        bar_title_line1 = "SHAP Feature Importance"
    bar_fig.suptitle(f"{bar_title_line1}\n{shap_title_prefix}", fontsize=10)
    plt.subplots_adjust(top=0.88)
    plt.show()

    # Per-class bar charts (one figure per class) when shap_per_class_plots=True.
    if is_multiclass and shap_per_class_plots:
        n_classes_bar = (
            len(shap_matrix) if isinstance(shap_matrix, list) else shap_matrix.shape[2]
        )
        for class_index in range(n_classes_bar):
            class_label = (
                class_names[class_index]
                if class_names and class_index < len(class_names)
                else f"Class {class_index}"
            )
            class_sv = (
                shap_matrix[class_index]
                if isinstance(shap_matrix, list)
                else shap_matrix[:, :, class_index]
            )
            plt.figure()
            shap.summary_plot(
                class_sv,
                feature_array,
                plot_type="bar",
                max_display=SHAP_MAX_DISPLAY,
                show=False,
            )
            per_class_bar_fig = plt.gcf()
            for ax in per_class_bar_fig.get_axes():
                ax.set_title("")
            per_class_bar_fig.suptitle(
                f"SHAP Feature Importance — Class: {class_label}\n{shap_title_prefix}",
                fontsize=10,
            )
            plt.subplots_adjust(top=0.88)
            plt.show()

    # ---------------------------------------------------------------
    # Plot 2+: Beeswarm/dot plots — direction and spread, per class.
    # ---------------------------------------------------------------
    # Summary beeswarm/dot plot — one figure per class.
    #
    # Multi-class  → one beeswarm per class so each is readable and the title
    #               never collides with SHAP's feature-value colorbar.
    # Binary       → single beeswarm for the positive class (class index 1),
    #               with the class name embedded in the title.
    # The High→Low feature-value colorbar on the right is rendered by SHAP
    # automatically when plot_type="dot" (the default for 2-D shap_values).
    # SHAP also adds the "Sum of N other features" row automatically when
    # the number of features exceeds SHAP_MAX_DISPLAY.

    if is_multiclass:
        # Global beeswarm (mean SHAP across all classes) — always shown.
        global_sv = (
            np.mean(np.stack(shap_matrix), axis=0)
            if isinstance(shap_matrix, list)
            else np.mean(shap_matrix, axis=2)
        )
        _draw_beeswarm_with_sum_row(
            global_sv, feature_array, SHAP_MAX_DISPLAY,
            f"SHAP Beeswarm — Global (mean across all classes)\n{shap_title_prefix}",
        )
        # Per-class beeswarms: always shown for multiclass so each class's
        # directional signal is readable without the cross-class averaging artefact.
        n_classes_bee = (
            len(shap_matrix) if isinstance(shap_matrix, list) else shap_matrix.shape[2]
        )
        for class_index in range(n_classes_bee):
            class_label = (
                class_names[class_index]
                if class_names and class_index < len(class_names)
                else f"Class {class_index}"
            )
            class_shap_values = (
                shap_matrix[class_index]
                if isinstance(shap_matrix, list)
                else shap_matrix[:, :, class_index]
            )
            _draw_beeswarm_with_sum_row(
                class_shap_values, feature_array, SHAP_MAX_DISPLAY,
                f"SHAP Beeswarm — Target class: {class_label}\n{shap_title_prefix}",
            )
    else:
        # Binary classification or regression.
        # binary_class_label is already computed above (before the bar chart).
        title_line1 = "SHAP Feature Importance"
        if binary_class_label:
            title_line1 += f" — Target class: {binary_class_label}"
        _draw_beeswarm_with_sum_row(
            shap_matrix, feature_array, SHAP_MAX_DISPLAY,
            f"{title_line1}\n{shap_title_prefix}",
        )

    # Waterfall plot for one sample.
    if waterfall_index is not None and waterfall_index < X_test.shape[0]:
        try:
            explanation = shap.Explanation(
                values=shap_matrix[waterfall_index],
                base_values=explainer.expected_value[1]
                if isinstance(explainer.expected_value, (list, np.ndarray))
                else explainer.expected_value,
                data=X_test[waterfall_index],
                feature_names=feature_names,
            )
            plt.figure()
            shap.plots.waterfall(explanation, max_display=15, show=False)
            print()
            print(f"  [SHAP] Interpreting the Waterfall plot (sample index {waterfall_index}):")
            print("    - Each bar represents one feature's contribution for this specific sample.")
            if binary_class_label:
                print(f"    - Red bars push the model output HIGHER (toward '{binary_class_label}').")
                print(f"    - Blue bars push the model output LOWER (away from '{binary_class_label}').")
            else:
                print("    - Red bars push the model output HIGHER (toward the positive class /")
                print(f"      higher value of '{target_obs_column}').")
                print("    - Blue bars push the model output LOWER (toward the negative class /")
                print(f"      lower value of '{target_obs_column}').")
            print("    - E[f(x)] is the average model output across the training set (baseline).")
            print("    - f(x) is the final model output for this sample.")
            waterfall_title_line1 = f"SHAP Waterfall — sample index {waterfall_index}"
            if binary_class_label:
                waterfall_title_line1 += f" | Target class: {binary_class_label}"
            plt.title(
                f"{waterfall_title_line1}\n{shap_title_prefix}"
            )
            plt.tight_layout()
            plt.show()
        except Exception:
            # Waterfall is optional; don't crash if the explainer format differs.
            pass

    return {
        "shap_values":        shap_matrix,
        "shap_explainer_type": explainer_type,
        "shap_feature_names": feature_names,
    }


# ---------------------------------------------------------------------------
# Internal: Regression scatter plot
# ---------------------------------------------------------------------------


def _plot_regression_scatter(
    results: dict,
    model_name: str,
    evaluation_strategy: str,
    target_obs_column: str,
    species_label: str = "",
) -> None:
    """
    Draw two diagnostic plots for regression results.

    Figure 1 — Predicted vs True scatter:
        One point per subject (LOGO) or per sample (StandardCV).
        Points are colored by absolute error. A diagonal dashed line marks
        perfect prediction. R², MAE, and RMSE are shown in the title.

    Figure 2 — Absolute error vs True value:
        Each subject/sample is a vertical bar sorted by true value along x.
        A horizontal dashed line marks the global MAE.
        Reveals whether prediction quality degrades at the extremes of the range.

    Arguments:
        results:             Results dict from run_ml_analysis() with task_type='regression'.
        model_name:          Model name shown in the plot title.
        evaluation_strategy: "StandardCV" or "LOGO" — shown in the title.
        target_obs_column:   Name of the prediction target (e.g. "age_days").
        species_label:       Optional species label as the first title line.
    """
    import matplotlib.pyplot as plt

    true_values: Optional[np.ndarray] = results.get("logo_subject_true")
    pred_values: Optional[np.ndarray] = results.get("logo_subject_pred")

    if true_values is None or pred_values is None or len(true_values) == 0:
        return

    true_arr = np.asarray(true_values, dtype=float)
    pred_arr = np.asarray(pred_values, dtype=float)
    abs_errors = np.abs(pred_arr - true_arr)

    subject_ids: Optional[List[str]] = results.get("logo_subject_ids")

    r2 = results.get("r2_score")
    mae = results.get("mae")
    rmse = results.get("rmse")

    species_prefix = f"{species_label}\n" if species_label else ""
    base_title = (
        f"{species_prefix}{model_name} | {evaluation_strategy} | Target: {target_obs_column}\n"
        f"R²={r2:.3f}  MAE={mae:.2f}  RMSE={rmse:.2f}"
        if r2 is not None else
        f"{species_prefix}{model_name} | {evaluation_strategy} | Target: {target_obs_column}"
    )

    # ------------------------------------------------------------------
    # Figure 1: Predicted vs True
    # ------------------------------------------------------------------
    fig1, ax1 = plt.subplots(figsize=(6, 5))

    scatter = ax1.scatter(
        true_arr,
        pred_arr,
        c=abs_errors,
        cmap="YlOrRd",
        s=60,
        edgecolors="black",
        linewidths=0.4,
        zorder=3,
        vmin=0,
    )
    fig1.colorbar(scatter, ax=ax1, label="Absolute error")

    # Identity line: perfect prediction
    value_min = float(min(true_arr.min(), pred_arr.min()))
    value_max = float(max(true_arr.max(), pred_arr.max()))
    ax1.plot(
        [value_min, value_max],
        [value_min, value_max],
        linestyle="--",
        color="steelblue",
        linewidth=1.2,
        label="Perfect prediction",
        zorder=2,
    )

    # Label points with subject IDs when available (LOGO mode)
    if subject_ids is not None:
        for x_val, y_val, label in zip(true_arr, pred_arr, subject_ids):
            ax1.annotate(
                label,
                xy=(x_val, y_val),
                xytext=(3, 3),
                textcoords="offset points",
                fontsize=6,
                color="dimgray",
            )

    ax1.set_xlabel(f"True {target_obs_column}")
    ax1.set_ylabel(f"Predicted {target_obs_column}")
    ax1.set_title(f"Predicted vs True\n{base_title}", fontsize=9)
    ax1.legend(fontsize=8)
    fig1.tight_layout()
    plt.show()

    # ------------------------------------------------------------------
    # Figure 2: Absolute error vs True value (sorted by true value)
    # ------------------------------------------------------------------
    sort_order = np.argsort(true_arr)
    sorted_true = true_arr[sort_order]
    sorted_errors = abs_errors[sort_order]
    sorted_ids = (
        [subject_ids[i] for i in sort_order]
        if subject_ids is not None else
        [str(i) for i in range(len(sorted_true))]
    )

    n_points = len(sorted_true)
    fig2, ax2 = plt.subplots(figsize=(max(6, n_points * 0.55), 4))

    norm_errors = sorted_errors / (sorted_errors.max() if sorted_errors.max() > 0 else 1.0)
    bar_colors = plt.get_cmap("YlOrRd")(norm_errors)

    ax2.bar(range(n_points), sorted_errors, color=bar_colors, edgecolor="black", linewidth=0.4)

    if mae is not None:
        ax2.axhline(mae, color="steelblue", linewidth=1.2, linestyle="--", label=f"MAE = {mae:.2f}")
        ax2.legend(fontsize=8)

    x_labels = [f"{sid}\n({tv:.0f})" for sid, tv in zip(sorted_ids, sorted_true)]
    ax2.set_xticks(range(n_points))
    ax2.set_xticklabels(x_labels, rotation=45, ha="right", fontsize=7)
    ax2.set_ylabel("Absolute error")
    ax2.set_title(
        f"Prediction Error per Subject (sorted by true {target_obs_column})\n{base_title}",
        fontsize=9,
    )
    fig2.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Internal: Confusion matrix plot
# ---------------------------------------------------------------------------


def _plot_confusion_matrix(
    results: dict,
    model_name: str,
    evaluation_strategy: str,
    target_obs_column: str,
    species_label: str = "",
) -> None:
    """
    Draw a normalised (%) confusion matrix heatmap for classification results.

    Uses a blue colour map for StandardCV and a green colour map for LOGO,
    to visually distinguish the two evaluation strategies at a glance.

    Arguments:
        results:             Results dict from run_ml_analysis().
        model_name:          Model name shown in the plot title.
        evaluation_strategy: "StandardCV" or "LOGO" — determines colour map.
        target_obs_column:   Name of the prediction target for the title.
        species_label:       Species label shown as the first title line (e.g. "Specie: Mouse").
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    raw_cm = results.get("confusion_matrix")
    class_names = results.get("class_names")
    if raw_cm is None or class_names is None:
        return

    # Normalise rows so each cell is the fraction of true-class samples
    # predicted as each class — makes classes with unequal sizes comparable.
    row_sums = raw_cm.sum(axis=1, keepdims=True)
    # Avoid division by zero for empty rows (can happen in tiny test sets).
    row_sums = np.where(row_sums == 0, 1, row_sums)
    normalised_cm = raw_cm / row_sums

    # 95 % confidence interval computed from the raw count matrix (normal approximation:
    # p ± 1.96 × sqrt(p × (1−p) / n)).  For StandardCV, n is the held-out test-set size;
    # for LOGO, n is the total number of OOF (out-of-fold) predictions — both are the
    # population that the confusion matrix was built from, so the formula is identical.
    n_total = int(raw_cm.sum())
    n_correct = int(raw_cm.diagonal().sum())
    accuracy_point = n_correct / n_total if n_total > 0 else 0.0
    ci_half_width = (
        1.96 * np.sqrt(accuracy_point * (1.0 - accuracy_point) / n_total)
        if n_total > 0 else 0.0
    )
    ci_low = max(0.0, accuracy_point - ci_half_width)
    ci_high = min(1.0, accuracy_point + ci_half_width)

    mean_accuracy = results.get("mean_accuracy")
    std_accuracy = results.get("std_accuracy")

    colour_map = "Blues" if evaluation_strategy == "StandardCV" else "Greens"
    accuracy_subtitle = (
        f"Mean accuracy: {mean_accuracy:.3f} ± {std_accuracy:.3f}"
        if mean_accuracy is not None else ""
    )
    species_prefix = f"{species_label}\n" if species_label else ""
    title = (
        f"{species_prefix}Confusion Matrix\n"
        f"{model_name} | {evaluation_strategy} | Target: {target_obs_column}\n"
        f"{accuracy_subtitle}"
    )

    # Label the sample source depending on the evaluation strategy so the console
    # message is unambiguous for both StandardCV and LOGO.
    sample_source = "test set" if evaluation_strategy == "StandardCV" else "OOF predictions (all subjects)"
    print()
    print("  [Confusion Matrix] Interpreting the confusion matrix:")
    print(f"    - Overall accuracy on the {sample_source}:")
    print(f"      {accuracy_point:.1%}  [95% CI: {ci_low:.1%} – {ci_high:.1%}]  (n = {n_total})")
    print(
        f"      → We are 95% confident the model's real accuracy lies between "
        f"{ci_low:.1%} and {ci_high:.1%}."
    )
    print("        A wide interval means the test set is small — don't over-interpret the exact number.")
    print("    - Each cell shows the % of true-class samples predicted as each class.")
    print("    - Diagonal cells (top-left to bottom-right) represent correct predictions")
    print("      (true positives for each class) — aim for high % here.")
    print("    - Off-diagonal cells are misclassifications — check which classes are")
    print("      most often confused with each other.")
    print("    - Rows sum to 100 % (one row per true class).")

    # Per-cell std: binomial SE = sqrt(p * (1-p) / n_row).
    # Same formula as the global CI, applied cell by cell.
    # Answers: "by how much would this % vary if evaluated on a different same-size sample?"
    # per_cell_std = np.sqrt(
    #     normalised_cm * (1.0 - normalised_cm) / np.where(row_sums == 0, 1, row_sums)
    # )
    # Custom annotation: "XX.X%" on the first line, "±YY.Y%" on the second.
    # annot_array = np.array([
    #     [
    #         f"{normalised_cm[i, j]:.1%}\n±{per_cell_std[i, j]:.1%}"
    #         for j in range(len(class_names))
    #     ]
    #     for i in range(len(class_names))
    # ])
    annot_array = np.array([
        [
            f"{normalised_cm[i, j]:.1%}"
            for j in range(len(class_names))
        ]
        for i in range(len(class_names))
    ])

    # Fixed cell size so the heatmap stays square regardless of class count.
    # +1.5 width for the colorbar; +1.8 height for the 3-line title.
    n = len(class_names)
    cell_size = 1.6

    # Font sizes must account for the y-axis margin consumed by long class labels.
    # Long labels (e.g. "Bin 4 (7214.8 – 9895.6)" = 24 chars) shrink the actual
    # heatmap cell width well below cell_size, causing overflow at a naive fixed size.
    max_label_chars = max(len(str(cn)) for cn in class_names) if class_names else 8
    # Estimate y-axis label width: base tick size × 0.55 (proportional char width) / 72.
    base_tick_pts = cell_size * 72 * 0.14
    char_width_inches = base_tick_pts * 0.55 / 72
    y_label_margin_inches = max_label_chars * char_width_inches
    # Effective cell width = (figure width − y-label margin − colorbar) / n.
    figure_width_inches = n * cell_size + 1.5
    actual_cell_width_inches = max(0.4, (figure_width_inches - y_label_margin_inches - 0.9) / n)
    # Annotation text is "XX.X%" (5 chars) but "100.0%" (6 chars) also occurs;
    # use the actual longest annotation string so 100% cells are sized correctly too.
    # target ≤ 55% fill so text never touches the cell border.
    # constraint: fontsize × max_annot_chars × 0.55 / 72 ≤ 0.55 × actual_cell_width_inches
    max_annot_chars = max(len(cell_text) for row in annot_array for cell_text in row)
    annot_fontsize = max(8,  int(actual_cell_width_inches * 72 * 0.55 / (max_annot_chars * 0.55)))
    tick_fontsize  = max(6,  int(annot_fontsize * 0.65))
    label_fontsize = max(8,  int(annot_fontsize * 0.75))
    title_fontsize = max(7,  int(annot_fontsize * 0.60))

    fig, axis = plt.subplots(figsize=(n * cell_size + 1.5, n * cell_size + 1.8))
    heatmap = sns.heatmap(
        normalised_cm,
        annot=annot_array,
        fmt="",
        cmap=colour_map,
        xticklabels=class_names,
        yticklabels=class_names,
        ax=axis,
        linewidths=0.5,
        linecolor="white",
        annot_kws={"size": annot_fontsize, "weight": "bold"},
    )
    axis.set_xlabel("Predicted label", fontsize=label_fontsize, labelpad=8)
    axis.set_ylabel("True label",      fontsize=label_fontsize, labelpad=8)
    axis.set_title(title, fontsize=title_fontsize)
    axis.tick_params(axis="both", labelsize=tick_fontsize)
    # 45° rotation uses less vertical space than seaborn's 90° default for long labels.
    axis.set_xticklabels(axis.get_xticklabels(), rotation=45, ha="right", fontsize=tick_fontsize)
    axis.set_yticklabels(axis.get_yticklabels(), rotation=0,  fontsize=tick_fontsize)
    # Scale the colorbar tick labels to match.
    colorbar = heatmap.collections[0].colorbar
    colorbar.ax.tick_params(labelsize=tick_fontsize)
    plt.tight_layout()
    plt.show()
    plt.close(fig)  # prevent the cleared figure repr from leaking into Jupyter cell output


# ---------------------------------------------------------------------------
# Internal: ROC curve plot
# ---------------------------------------------------------------------------


def _plot_roc_curve(
    results: dict,
    model_name: str,
    evaluation_strategy: str,
    target_obs_column: str,
    species_label: str = "",
) -> None:
    """
    Draw a ROC curve (or one-vs-rest curves for multiclass) for classification results.

    For binary classification: one curve with the AUC in the legend.
    For multiclass (3+ classes): one curve per class using a one-vs-rest approach,
    plus a macro-average curve.

    Arguments:
        results:             Results dict from run_ml_analysis().
        model_name:          Model name shown in the plot title.
        evaluation_strategy: "StandardCV" or "LOGO" — shown in the title.
        target_obs_column:   Name of the prediction target for the title.
        species_label:       Species label shown as the first title line (e.g. "Specie: Mouse").
    """
    import matplotlib.pyplot as plt

    y_true = results.get("roc_y_true")
    y_pred_proba = results.get("roc_y_pred_proba")
    class_names = results.get("class_names")

    if y_true is None or y_pred_proba is None or class_names is None:
        return

    y_true_str = np.array(y_true, dtype=str)
    species_prefix = f"{species_label}\n" if species_label else ""
    title = (
        f"{species_prefix}ROC Curve\n"
        f"{model_name} | {evaluation_strategy} | Target: {target_obs_column}"
    )

    print()
    print("  [ROC Curve] Interpreting the ROC curve:")
    print("    - The x-axis is False Positive Rate (FPR): fraction of negatives wrongly")
    print("      classified as positive.  The y-axis is True Positive Rate (TPR): fraction")
    print("      of positives correctly classified.")
    print("    - AUC (Area Under Curve) ranges from 0.5 (random) to 1.0 (perfect).")
    print("    - A curve close to the top-left corner indicates a strong classifier.")
    print("    - The dashed diagonal (AUC = 0.5) is a random classifier baseline.")
    if len(class_names) > 2:
        print("    - In multiclass mode, each curve is one-vs-rest (OvR).")
        print("      The macro-average curve averages AUC across all classes equally.")

    fig, axis = plt.subplots(figsize=(6, 5))

    unique_classes = np.unique(y_true_str)

    if len(unique_classes) == 2:
        # Binary: class_names[1] is the positive class (index 1 in predict_proba output).
        fpr, tpr, _ = roc_curve(
            y_true_str,
            y_pred_proba[:, 1],
            pos_label=class_names[1],
        )
        roc_auc_value = auc(fpr, tpr)
        axis.plot(fpr, tpr, lw=2, label=f"AUC = {roc_auc_value:.3f}")
    else:
        # Multiclass one-vs-rest.
        from sklearn.preprocessing import label_binarize
        y_bin = label_binarize(y_true_str, classes=class_names)
        mean_fpr = np.linspace(0, 1, 200)
        tpr_interpolated_list: List[np.ndarray] = []

        for class_index, class_label in enumerate(class_names):
            if class_index >= y_pred_proba.shape[1]:
                break
            fpr_c, tpr_c, _ = roc_curve(y_bin[:, class_index], y_pred_proba[:, class_index])
            roc_auc_c = auc(fpr_c, tpr_c)
            axis.plot(fpr_c, tpr_c, lw=1.5, label=f"{class_label} (AUC = {roc_auc_c:.3f})")
            tpr_interpolated_list.append(np.interp(mean_fpr, fpr_c, tpr_c))

        # Macro-average curve.
        mean_tpr = np.mean(tpr_interpolated_list, axis=0)
        mean_tpr[0] = 0.0
        macro_auc = auc(mean_fpr, mean_tpr)
        axis.plot(
            mean_fpr, mean_tpr,
            lw=2.5, linestyle="--", color="black",
            label=f"Macro-avg (AUC = {macro_auc:.3f})",
        )

    # Random classifier baseline.
    axis.plot([0, 1], [0, 1], linestyle=":", color="grey", lw=1.5, label="Random (AUC = 0.5)")
    axis.set_xlabel("False Positive Rate")
    axis.set_ylabel("True Positive Rate")
    axis.set_xlim([0.0, 1.0])
    axis.set_ylim([0.0, 1.05])
    axis.legend(loc="lower right", fontsize=9)
    axis.set_title(title)
    plt.tight_layout()
    plt.show()
    plt.close(fig)  # prevent the cleared figure repr from leaking into Jupyter cell output


# ---------------------------------------------------------------------------
# Internal: QC print block
# ---------------------------------------------------------------------------


def _print_evaluation_qc_block(
    results: dict,
    task_type: str,
    model_name: str,
    strategy: str,
) -> None:
    """
    Print a structured QC block to the console after model evaluation.

    Follows the project convention: a beginner must be able to follow the
    full execution and verify correctness without reading Python code.
    """
    print()
    print("  ─── Evaluation Results ───────────────────────────")
    print(f"  Model              : {model_name}")
    print(f"  Strategy           : {strategy}")
    print(f"  Samples used       : {results['n_samples']:,}")
    print(f"  Features used      : {results['n_features']}")
    print()

    if task_type == "classification":
        if results.get("mean_accuracy") is not None:
            print(f"  Mean accuracy (val): {results['mean_accuracy']:.3f} ± {results['std_accuracy']:.3f}")
        if results.get("mean_train_accuracy") is not None:
            gap = results["mean_train_accuracy"] - results["mean_accuracy"]
            print(f"  Mean accuracy (train): {results['mean_train_accuracy']:.3f} ± {results['std_train_accuracy']:.3f}  (train−val gap: {gap:+.3f})")
        if results.get("roc_auc") is not None:
            print(f"  ROC AUC            : {results['roc_auc']:.3f}")
        if results.get("classification_report"):
            print()
            print("  Classification report (validation folds):")
            for line in results["classification_report"].splitlines():
                print(f"    {line}")
        if results.get("logo_per_subject_scores") is not None:
            logo_mode = results.get("logo_group_by_column")
            logo_label = f"LAVO by {logo_mode}" if logo_mode else "LOGO"
            print()
            print(f"  Per-subject accuracy ({logo_label}):")
            for subject, score in results["logo_per_subject_scores"].items():
                print(f"    {subject:<40}: {score:.3f}")
    else:
        is_logo = results.get("logo_per_subject_scores") is not None
        if results.get("r2_score") is not None:
            print(f"  R² (subject-level) : {results['r2_score']:.3f}")
            print(f"  MAE (subject-level): {results['mae']:.4f}")
            print(f"  RMSE (subject-lvl) : {results['rmse']:.4f}")
            if results.get("pearson_r") is not None:
                print(f"  Pearson r          : {results['pearson_r']:.3f}")
        if not is_logo and results.get("mean_r2") is not None:
            # StandardCV: mean_r2 is per-fold R², meaningful to show separately.
            print(f"  Mean R² (val folds): {results['mean_r2']:.3f} ± {results['std_r2']:.3f}")
        if results.get("mean_train_r2") is not None:
            print(f"  Mean R² (train)    : {results['mean_train_r2']:.3f} ± {results['std_train_r2']:.3f}")
        if is_logo and results.get("logo_per_subject_scores") is not None:
            logo_mode = results.get("logo_group_by_column")
            logo_label = f"LAVO by {logo_mode}" if logo_mode else "LOGO"
            print()
            print(f"  Per-fold MAE ({logo_label}):")
            for subject, neg_mae in results["logo_per_subject_scores"].items():
                print(f"    {subject:<40}: MAE {-neg_mae:.3f}")

    if results.get("shap_explainer_type") is not None:
        print()
        print(f"  SHAP explainer     : {results['shap_explainer_type']}")
        print(f"  SHAP features      : {len(results['shap_feature_names'])}")


# ---------------------------------------------------------------------------
# Internal: ROC AUC helper
# ---------------------------------------------------------------------------


def _sort_class_names_numerically(class_names: List[str]) -> List[str]:
    """Sort class name strings numerically when all values are numeric, alphabetically otherwise."""
    try:
        return sorted(class_names, key=lambda x: float(x))
    except (ValueError, TypeError):
        return sorted(class_names)


def _compute_roc_auc(model, X_test: np.ndarray, y_test: np.ndarray) -> Optional[float]:
    """
    Compute ROC AUC score, handling binary and multiclass cases gracefully.

    Returns None when the model cannot produce probability estimates or when
    there is only one class in the test set (can happen with LOGO on tiny datasets).
    """
    try:
        y_prob = model.predict_proba(X_test)
        unique_classes = np.unique(y_test)
        if len(unique_classes) < 2:
            return None
        if len(unique_classes) == 2:
            return float(roc_auc_score(y_test, y_prob[:, 1]))
        # Multiclass
        return float(
            roc_auc_score(y_test, y_prob, multi_class="ovr", average="macro")
        )
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Internal: challenge helpers
# ---------------------------------------------------------------------------


def _extract_challenge_row(
    subset_label: str,
    results: dict,
    task_type: str,
) -> dict:
    """Extract a summary row dict from a run_ml_analysis() results dict."""
    if task_type == "classification":
        primary_metric = results.get("mean_accuracy")
        std = results.get("std_accuracy")
        roc_auc = results.get("roc_auc")
    else:
        primary_metric = results.get("mean_r2")
        std = results.get("std_r2")
        roc_auc = None

    return {
        "subset_name":    subset_label,
        "n_features":     results["n_features"],
        "primary_metric": primary_metric,
        "std":            std,
        "roc_auc":        roc_auc,
    }


def _filter_bag_features_by_stat(
    bag_feature_names: List[str],
    keep_stats: set,
) -> List[str]:
    """
    Return bag feature names whose statistic suffix is in keep_stats.

    Example:
        _filter_bag_features_by_stat(["Mito_Area__mean", "Mito_Area__std"], {"mean"})
        → ["Mito_Area__mean"]
    """
    return [
        name for name in bag_feature_names
        if any(name.endswith(f"__{stat}") for stat in keep_stats)
    ]


def _get_original_names_from_bag_features(
    bag_feature_names: List[str],
    original_feature_list: Optional[List[str]],
    anndata_object: anndata.AnnData,
) -> Optional[List[str]]:
    """
    Recover the original (pre-bagging) feature names from bag feature names.

    The bag pipeline creates features like "Mito_Area__mean" from "Mito_Area".
    To run bagging again on a subset of statistics, we need the original feature names.

    Returns None if the original feature list was None (meaning "use all features").
    """
    if original_feature_list is None:
        return None

    # Extract original names by stripping the statistic suffix.
    original_names_from_bag = set()
    for bag_name in bag_feature_names:
        for stat in ("__mean", "__std", "__median", "__skew"):
            if bag_name.endswith(stat):
                original_names_from_bag.add(bag_name[: -len(stat)])
                break

    # Intersect with the requested original feature list.
    filtered = [n for n in original_feature_list if n in original_names_from_bag]
    return filtered if filtered else None


# ---------------------------------------------------------------------------
# Public: raw signal diagnostic plot
# ---------------------------------------------------------------------------


def plot_feature_vs_target(
    anndata_object: anndata.AnnData,
    target_obs_column: str,
    subject_id_column: str = "subject_ID",
    feature_names: Optional[List[str]] = None,
    top_n: int = 6,
) -> None:
    """
    Scatter plot of per-subject mean(feature) vs target value — a raw-signal
    sanity check before interpreting ML results.

    For each selected feature, all events belonging to the same subject are
    averaged to produce one data point per subject, then plotted against the
    target column (e.g. age).  A Pearson r and Spearman ρ annotation is added
    to each panel.

    If feature_names is None, the function reads the top SHAP features from
    the last ML run stored in .uns['ml_results'].  Bag feature names like
    "UV3-H__skew" are resolved back to their base channel ("UV3-H") so that
    the raw (pre-bag) signal is used.

    Arguments:
        anndata_object:    AnnData whose .obs contains subject_id_column and
                           target_obs_column.
        target_obs_column: .obs column to use as the x-axis (e.g. "age").
        subject_id_column: .obs column identifying individual subjects.
        feature_names:     Explicit list of feature names to plot.  If None,
                           the top_n SHAP features from .uns['ml_results'] are
                           used.
        top_n:             Number of top SHAP features to visualise when
                           feature_names is None.
    """
    import matplotlib.pyplot as plt
    from scipy.stats import pearsonr, spearmanr

    # ---- Resolve feature names -----------------------------------------------
    if feature_names is None:
        ml_results = anndata_object.uns.get(_ML_RESULTS_KEY)
        if ml_results is None:
            print(
                "plot_feature_vs_target: no ML results found in .uns['ml_results']. "
                "Run run_ml_analysis() first, or pass feature_names explicitly."
            )
            return
        shap_feature_names = ml_results.get("shap_feature_names")
        shap_values = ml_results.get("shap_values")
        if shap_feature_names is None or shap_values is None:
            print(
                "plot_feature_vs_target: no SHAP data found in ML results. "
                "Re-run with compute_shap=True, or pass feature_names explicitly."
            )
            return

        # Average |SHAP| across classes (regression → single array, classification → list).
        if isinstance(shap_values, list):
            mean_abs_shap = np.mean(
                [np.abs(arr) for arr in shap_values], axis=0
            )
        else:
            mean_abs_shap = np.abs(np.asarray(shap_values))

        # Sort descending and take top_n.
        sorted_indices = np.argsort(mean_abs_shap)[::-1]
        top_bag_features = [shap_feature_names[i] for i in sorted_indices[:top_n]]

        # Strip bag statistic suffix to recover the original channel name.
        _BAG_SUFFIXES = ("__mean", "__std", "__median", "__skew")
        feature_names_resolved = []
        for bag_name in top_bag_features:
            base = bag_name
            for suffix in _BAG_SUFFIXES:
                if bag_name.endswith(suffix):
                    base = bag_name[: -len(suffix)]
                    break
            feature_names_resolved.append(base)

        # Label shown in the plot title: "channel (stat)"
        feature_display_labels = [
            f"{base} ({bag_name[len(base) + 2:]})"  # +2 for the double underscore
            if bag_name != base
            else base
            for base, bag_name in zip(feature_names_resolved, top_bag_features)
        ]
    else:
        feature_names_resolved = list(feature_names)
        feature_display_labels = list(feature_names)

    # ---- Validate target column -----------------------------------------------
    if target_obs_column not in anndata_object.obs.columns:
        print(f"plot_feature_vs_target: '{target_obs_column}' not found in .obs.")
        return
    if subject_id_column not in anndata_object.obs.columns:
        print(f"plot_feature_vs_target: '{subject_id_column}' not found in .obs.")
        return

    target_series = anndata_object.obs[target_obs_column]
    subject_series = anndata_object.obs[subject_id_column]

    # One target value per subject (must be unique — warn if not).
    subject_target = (
        target_series.groupby(subject_series).first().astype(float)
    )

    # ---- Get data matrix (active layer, all events) ---------------------------
    data_matrix = _get_active_data_matrix(anndata_object)
    all_var_names = list(anndata_object.var_names)

    # ---- Build subplot grid ---------------------------------------------------
    n_panels = len(feature_names_resolved)
    n_cols = min(3, n_panels)
    n_rows = (n_panels + n_cols - 1) // n_cols
    figure, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    if n_panels == 1:
        axes = np.array([axes])
    axes = np.array(axes).flatten()

    for panel_index, (channel_name, display_label) in enumerate(
        zip(feature_names_resolved, feature_display_labels)
    ):
        ax = axes[panel_index]

        # ---- Extract per-event values for this channel -----------------------
        if channel_name not in all_var_names:
            ax.set_title(f"{display_label}\n(not found in .var)")
            ax.axis("off")
            continue

        channel_col_index = all_var_names.index(channel_name)
        channel_values = data_matrix[:, channel_col_index]  # shape (n_events,)

        # Mean per subject.
        channel_series = pd.Series(channel_values, index=anndata_object.obs_names)
        subject_channel_mean = channel_series.groupby(subject_series).mean()

        # Align on subjects present in both.
        common_subjects = subject_target.index.intersection(subject_channel_mean.index)
        x_values = subject_target.loc[common_subjects].values
        y_values = subject_channel_mean.loc[common_subjects].values

        # ---- Statistics -------------------------------------------------------
        if len(x_values) >= 3:
            pearson_r_value, pearson_p = pearsonr(x_values, y_values)
            spearman_rho, spearman_p = spearmanr(x_values, y_values)
        else:
            pearson_r_value = pearson_p = spearman_rho = spearman_p = float("nan")

        # ---- Scatter + regression line ----------------------------------------
        ax.scatter(x_values, y_values, alpha=0.7, s=40)

        # Regression line (ordinary least squares).
        if len(x_values) >= 2 and not np.isnan(pearson_r_value):
            slope, intercept = np.polyfit(x_values, y_values, 1)
            x_line = np.linspace(x_values.min(), x_values.max(), 100)
            ax.plot(x_line, slope * x_line + intercept, color="tomato", linewidth=1.5)

        # ---- Labels and annotation -------------------------------------------
        ax.set_xlabel(target_obs_column)
        ax.set_ylabel(f"mean({channel_name})")
        ax.set_title(display_label, fontsize=9)
        annotation_text = (
            f"Pearson r={pearson_r_value:.2f} (p={pearson_p:.3f})\n"
            f"Spearman ρ={spearman_rho:.2f} (p={spearman_p:.3f})"
        )
        ax.annotate(
            annotation_text,
            xy=(0.05, 0.95),
            xycoords="axes fraction",
            va="top",
            fontsize=7.5,
            bbox={"boxstyle": "round,pad=0.3", "facecolor": "lightyellow", "alpha": 0.8},
        )

    # Hide empty panels.
    for empty_index in range(n_panels, len(axes)):
        axes[empty_index].axis("off")

    figure.suptitle(
        f"Raw signal per subject: mean(channel) vs {target_obs_column}\n"
        f"(active layer — {anndata_object.n_obs:,} events, "
        f"{subject_target.shape[0]} subjects)",
        fontsize=10,
    )
    plt.tight_layout()
    plt.show()
