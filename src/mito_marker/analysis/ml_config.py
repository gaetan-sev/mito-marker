"""
ml_config.py

Machine learning configuration for the mito-marker analysis pipeline.

Provides two ways to build the ml_config dict that controls all ML behaviour:

  get_default_ml_config()
      Returns a fully-populated dict with safe default values.
      Intended for programmatic use: call it, then override the keys you need.

  configure_ml(anndata_object)
      Interactive console wizard (same numbered-menu UX as select_channels()).
      Lists available .obs columns, prompts task type, strategy, model, and
      evaluation method.  Returns a validated ml_config dict.

  DEFAULT_MODEL_PARAMS
      Expert-tuned hyperparameter dicts for each supported model.
      Import and assign to config["model_params"] to use them.
      Biologists unfamiliar with hyperparameters can start with sklearn defaults
      (config["model_params"] = {}) and switch to these later.

Typical usage:
    from mito_marker.analysis.ml_config import get_default_ml_config, DEFAULT_MODEL_PARAMS

    # Option A: sklearn defaults
    config = get_default_ml_config()
    config["target_obs_column"] = "age_group"

    # Option B: expert-tuned defaults
    config = get_default_ml_config()
    config["target_obs_column"] = "age_group"
    config["model_name"]        = "RandomForest"
    config["model_params"]      = DEFAULT_MODEL_PARAMS["RandomForest"]

    # Option C: interactive console wizard
    config = configure_ml(my_anndata)
"""

import warnings
from typing import List, Optional

import anndata
import pandas as pd

from mito_marker.analysis.ml_bagging import ALLOWED_BAG_SAMPLING_MODES

# ---------------------------------------------------------------------------
# Allowed values — used both for validation and for displaying menus
# ---------------------------------------------------------------------------

ALLOWED_TASK_TYPES: List[str] = ["classification", "regression"]
ALLOWED_STRATEGIES: List[str] = ["SingleMito", "Bags", "MIL"]
ALLOWED_MODEL_NAMES: List[str] = [
    "ElasticNet", "RandomForest", "XGBoost", "MLP", "SVM",
    "ExtraTrees", "GradientBoosting", "GaussianNB", "KNeighbors",
    "MIL",
]
ALLOWED_EVALUATION_STRATEGIES: List[str] = ["StandardCV", "LOGO"]
ALLOWED_BAG_STATISTICS: List[str] = ["mean", "std", "median", "skew"]
ALLOWED_BAG_FS_METHODS: List[str] = [
    "CorrFilter", "MIM", "CMI", "HighVariance", "PCALoadings"
]

# One-line description shown in the interactive menu for each model.
_MODEL_DESCRIPTIONS: dict[str, str] = {
    "ElasticNet":   "linear model with L1+L2 regularization (fast, interpretable)",
    "RandomForest": "ensemble of decision trees — robust, handles non-linearity",
    "XGBoost":      "gradient boosting — usually a strong baseline",
    "MLP":          "multi-layer perceptron (neural network)",
    "SVM":          "support vector machine with RBF kernel",
    "ExtraTrees":       "more randomized bagged trees — often more stable than RandomForest",
    "GradientBoosting": "sklearn's sequential-tree boosting — simpler cousin of XGBoost",
    "GaussianNB":       "probabilistic baseline (Gaussian Naive Bayes) — classification only",
    "KNeighbors":       "proximity vote — captures local structure other models miss",
    "MIL":          "attention-based MIL — built-in neural network (only for MIL strategy)",
}

# ---------------------------------------------------------------------------
# Expert-tuned hyperparameter presets
# ---------------------------------------------------------------------------

DEFAULT_MODEL_PARAMS: dict[str, dict] = {
    "ElasticNet": {
        # penalty="elasticnet" mixes L1 (Lasso) and L2 (Ridge) regularization.
        # l1_ratio=0 → pure Ridge (keeps all correlated features, reduces their weight)
        # l1_ratio=1 → pure Lasso (keeps one of each correlated pair, zeros the rest)
        # 0.5 is a balanced mix — good when some features are correlated (e.g. Area + Perimeter).
        "penalty": "elasticnet",
        "l1_ratio": 0.5,
        # C is the inverse of regularization strength: larger C = less constrained model.
        # C=100 is fairly permissive — suitable when features are already normalized.
        "C": 100,
        "solver": "saga",  # required for elasticnet penalty
        "max_iter": 2500,
    },
    "RandomForest": {
        # n_estimators=1000: more trees → more stable predictions, diminishing returns beyond ~500.
        "n_estimators": 1000,
        # max_depth=10: prevents trees from memorizing individual subjects.
        "max_depth": 10,
        # class_weight="balanced": compensates for unequal group sizes (e.g. more Old than Young).
        "class_weight": "balanced",
    },
    "XGBoost": {
        # n_estimators=300 with a slow learning_rate=0.05 is more robust than
        # fewer trees with a fast rate — the model learns subtleties step by step.
        "n_estimators": 300,
        "learning_rate": 0.05,
        # max_depth=4: deep enough to see feature interactions, shallow enough to generalise.
        "max_depth": 4,
        # gamma=2.0: a node is only split if the gain exceeds 2.0 — suppresses noise splits.
        "gamma": 2.0,
        # reg_alpha=0.5: L1 regularization pushes irrelevant feature weights toward zero.
        "reg_alpha": 0.5,
        # subsample=0.7: each tree sees 70% of subjects → diversity, prevents overfitting.
        "subsample": 0.7,
        # colsample_bytree=0.5: each tree sees 50% of features → further diversity.
        "colsample_bytree": 0.5,
        "eval_metric": "logloss",
        "n_jobs": -1,
    },
    "MLP": {
        # (128, 64): funnel architecture — compresses from 128 to 64 neurons.
        "hidden_layer_sizes": (128, 64),
        "activation": "relu",
        "solver": "adam",
        # adaptive learning_rate: slows down as the model approaches convergence.
        "learning_rate": "adaptive",
        "momentum": 0.9,
        # alpha=0.05: L2 regularization weight — prevents overfitting on small datasets.
        "alpha": 0.05,
        "batch_size": 300,
        "learning_rate_init": 0.0005,
        "max_iter": 1000,
        # early_stopping: halts training when validation score stops improving.
        "early_stopping": True,
    },
    "SVM": {
        "kernel": "rbf",
        "C": 1.0,
        "gamma": "scale",
        # probability=True is required for ROC curves and SHAP KernelExplainer.
        "probability": True,
    },
    "ExtraTrees": {
        # More randomized split thresholds than RandomForest — trades a little bias
        # for lower variance, which tends to help on small, noisy subject counts.
        "n_estimators": 200,
        "max_depth": 8,
        "min_samples_leaf": 5,
        "class_weight": "balanced",
    },
    "GradientBoosting": {
        # Shallow trees (max_depth=3) with a slow learning_rate: sklearn's simpler,
        # CPU-only cousin of XGBoost — no GPU/cuML path, but no extra dependency either.
        "n_estimators": 150,
        "max_depth": 3,
        "learning_rate": 0.08,
        "subsample": 0.85,
    },
    "GaussianNB": {
        # var_smoothing adds a small fraction of the largest feature variance to
        # every feature's variance — avoids division-by-zero on near-constant features.
        # Classification only — sklearn has no Gaussian Naive Bayes regressor.
        "var_smoothing": 1e-9,
    },
    "KNeighbors": {
        # n_neighbors=3, weights="distance": validated as a strong general default
        # across all 6 TEM species in this project's own controlled model comparison
        # (mito-marker-lab, EXP-005 controlled cluster grid) — closer neighbors get more
        # vote weight, which matters when a subject's bags are not evenly spread.
        "n_neighbors": 3,
        "weights": "distance",
    },
    # MIL strategy — parameters are passed as mil_* keys in ml_config, not model_params.
    # Assign DEFAULT_MODEL_PARAMS["MIL"] to ml_config["model_params"] to use these defaults.
    "MIL": {
        "encoder_dim": 32,       # embedding dimension per mitochondrion
        "attention_dim": 16,     # hidden dimension of the gated attention network
        "dropout": 0.3,          # dropout after instance encoder
        "lr": 1e-3,              # Adam learning rate
        "weight_decay": 1e-4,    # L2 regularization (Adam weight_decay)
        "max_epochs": 300,       # maximum training epochs per fold
        "patience": 30,          # early stopping patience (validation loss)
        "min_mitos_per_bag": 5,  # subjects with fewer mitos are skipped
    },
}

# ---------------------------------------------------------------------------
# Public: get_default_ml_config
# ---------------------------------------------------------------------------


def get_default_ml_config() -> dict:
    """
    Return a fully-populated ml_config dict with safe default values.

    All keys are present so that _validate_ml_config() passes immediately.
    Override only the keys you need before passing the dict to run_ml_analysis().

    Returns:
        dict with keys:
          task_type, target_obs_column, subject_id_column,
          strategy, bags_per_subject, mitos_per_bag, bag_statistics,
          bag_sampling_mode, bag_max_overlap_fraction,
          model_name, model_params,
          evaluation_strategy, test_size, n_folds,
          compute_shap, shap_waterfall_sample_index,
          random_state
    """
    return {
        # Task
        "task_type":                  "classification",
        "target_obs_column":          "age_group",
        "subject_id_column":          "unique_subject_ID",
        # Data preparation strategy
        "strategy":                   "SingleMito",
        "bags_per_subject":           10,
        "mitos_per_bag":              30,
        "bag_statistics":             ["mean", "std", "median", "skew"],
        # None = disabled (bag features are only the statistics above).
        # int  = also fit a global KMeans(n_clusters=bag_n_clusters) once on all
        #        subjects' mitochondria (unsupervised, no label involved) and append
        #        one extra "fraction of this bag's mitos in cluster k" column per
        #        cluster. Captures multi-modal sub-population structure that mean/
        #        std/median/skew alone collapse away — see
        #        mito-marker-lab, EXP-005 (controlled cluster grid).
        "bag_n_clusters":             None,
        # How each subject's mitos are drawn into a bag. See ALLOWED_BAG_SAMPLING_MODES.
        # "auto" keeps the historical behaviour: distinct rows when the subject has
        # enough of them, bootstrap only for subjects smaller than mitos_per_bag.
        "bag_sampling_mode":          "auto",
        # QC threshold: flag any subject whose two bags are expected to share more than
        # this fraction of their distinct rows. Such bags are near-duplicates and inflate
        # the apparent sample size — 0.2 means "mitos_per_bag must stay under 20% of the
        # subject's row count". Set to 1.0 to disable flagging.
        "bag_max_overlap_fraction":   0.2,
        # Post-bagging feature selection (only applied when strategy == "Bags")
        "bag_feature_selection_methods":        [],    # e.g. ["CorrFilter", "HighVariance"]
        "bag_feature_selection_top_k":          20,
        "bag_feature_selection_corr_threshold": 0.95,
        # Model
        "model_name":                 "RandomForest",
        "model_params":               {},
        # Evaluation
        "evaluation_strategy":        "StandardCV",
        "logo_group_by_column":       None,   # None = standard LOGO (one subject out)
                                               # str  = LAVO (leave all subjects of same value out)
        "test_size":                  0.2,
        "n_folds":                    5,
        # Feature scaling inside each CV fold (recommended for ElasticNet/SVM)
        "scale_features_in_fold":     False,
        # SHAP
        "compute_shap":               True,
        "shap_waterfall_sample_index": 0,
        "shap_logo_min_fold_score":   None,
        # When True, plot one SHAP chart per class in addition to the global chart.
        # For multiclass models the global chart (mean |SHAP| across classes) is always
        # shown first.  Per-class charts add detail but multiply the number of figures
        # by the number of classes — set to False (default) to suppress them.
        "shap_per_class_plots":       False,
        # Reproducibility
        "random_state":               42,
        # MIL strategy parameters (only used when strategy == "MIL")
        # These override the DEFAULT_MODEL_PARAMS["MIL"] values above when set.
        "mil_encoder_dim":    32,
        "mil_attention_dim":  16,
        "mil_dropout":        0.3,
        "mil_lr":             1e-3,
        "mil_weight_decay":   1e-4,
        "mil_max_epochs":     300,
        "mil_patience":       30,
        "mil_min_mitos":      5,
        # None = use all mitos; int = subsample each bag for speed.
        # Recommended: 500–1000 for SFC (reduces forward-pass time proportionally).
        "mil_max_mitos_per_subject": None,
    }


# ---------------------------------------------------------------------------
# Public: configure_ml (interactive console wizard)
# ---------------------------------------------------------------------------


def configure_ml(anndata_object: anndata.AnnData) -> dict:
    """
    Interactively build an ml_config dict via numbered console menus.

    Presents menus in this order:
      [1] Task type        — classification or regression
      [2] Target column    — .obs column to predict (lists available columns)
      [3] Strategy         — SingleMito or Bags
          If Bags: bags_per_subject and mitos_per_bag (prompted with defaults)
      [4] Model            — ElasticNet / RandomForest / XGBoost / MLP / SVM
      [5] Evaluation       — StandardCV or LOGO
          If StandardCV: n_folds
      [6] SHAP             — yes or no
      [7] random_state     — integer seed (default 42)

    Note: model_params is NOT asked interactively — sklearn defaults (empty dict)
    are used.  To customise hyperparameters, set them after this call:
        config = configure_ml(my_anndata)
        config["model_params"] = DEFAULT_MODEL_PARAMS["RandomForest"]

    Arguments:
        anndata_object: Used to list available .obs columns in step [2].

    Returns:
        Validated ml_config dict.
    """
    print("=" * 60)
    print("INTERACTIVE ML CONFIGURATION")
    print("=" * 60)
    print(
        f"Current dataset: {anndata_object.n_obs:,} observations "
        f"× {anndata_object.n_vars} features"
    )
    print()

    # ------------------------------------------------------------------
    # [1] Task type
    # ------------------------------------------------------------------
    print("[1] Task type:")
    print("  [1] Classification — predict a categorical .obs column")
    print("  [2] Regression     — predict a continuous .obs column")
    task_type = _prompt_choice(
        prompt="Select task type",
        choices=["classification", "regression"],
        display_indices=["1", "2"],
    )
    print()

    # ------------------------------------------------------------------
    # [2] Target .obs column
    # ------------------------------------------------------------------
    obs_columns = list(anndata_object.obs.columns)
    print("[2] Target column to predict:")
    for index, column_name in enumerate(obs_columns, start=1):
        column_data = anndata_object.obs[column_name]
        n_unique = column_data.nunique()
        is_numeric = pd.api.types.is_numeric_dtype(column_data)
        numeric_flag = "  [numerical]" if is_numeric else ""
        print(f"  [{index}] {column_name}  ({n_unique} unique values){numeric_flag}")

    target_obs_column = _prompt_indexed_choice(
        prompt="Select target column",
        options=obs_columns,
    )
    print()

    # ------------------------------------------------------------------
    # [3] Strategy
    # ------------------------------------------------------------------
    print("[3] Data preparation strategy:")
    print("  [1] SingleMito — each row = one mitochondrion / cell (fastest, most data)")
    print("  [2] Bags       — aggregate N mitos into bags (reduces noise, recommended for LOGO)")
    print("  [3] MIL        — Attention-Based MIL: model sees ALL mitos per subject,")
    print("                   learns which morphological sub-population is discriminative")
    print("                   (requires PyTorch; forces LOGO evaluation)")
    strategy = _prompt_choice(
        prompt="Select strategy",
        choices=["SingleMito", "Bags", "MIL"],
        display_indices=["1", "2", "3"],
    )

    bags_per_subject: int = 10
    mitos_per_bag: int = 30
    bag_n_clusters: Optional[int] = None
    bag_sampling_mode: str = "auto"
    bag_max_overlap_fraction: float = 0.2
    bag_feature_selection_methods: List[str] = []
    bag_feature_selection_top_k: int = 20
    bag_feature_selection_corr_threshold: float = 0.95
    if strategy == "Bags":
        bags_per_subject = _prompt_positive_integer(
            prompt="  Bags per subject", default=10
        )
        mitos_per_bag = _prompt_positive_integer(
            prompt="  Mitos per bag", default=30
        )
        print()
        print("  [3-] Sampling mode — how each bag's mitos are drawn from the subject.")
        print("  Two bags of the same subject overlap by roughly mitos_per_bag / n_mitos;")
        print("  when that gets high the bags are near-duplicates and inflate the sample")
        print("  count without adding information. A QC table is printed either way.")
        for mode_name, mode_description in ALLOWED_BAG_SAMPLING_MODES.items():
            print(f"    {mode_name:<20} → {mode_description}.")
        bag_sampling_mode = _prompt_choice(
            prompt="  Sampling mode",
            choices=list(ALLOWED_BAG_SAMPLING_MODES),
            display_indices=[str(i + 1) for i in range(len(ALLOWED_BAG_SAMPLING_MODES))],
        )
        print()
        print("  [3a] Cluster-fraction features — optional. Adds, for each bag, the")
        print("  fraction of its mitos falling into each of k unsupervised morphology")
        print("  clusters (fit once via KMeans on all subjects pooled, no label used).")
        print("  Captures sub-population structure that mean/std/median/skew alone miss.")
        cluster_input = input(
            "  Number of clusters k (e.g. 6, or blank to disable): "
        ).strip()
        if cluster_input:
            try:
                bag_n_clusters = int(cluster_input)
                if bag_n_clusters < 2:
                    print("  => k must be >= 2 — cluster-fraction features disabled.")
                    bag_n_clusters = None
                else:
                    print(f"  => Cluster-fraction features enabled (k={bag_n_clusters}).")
            except ValueError:
                print("  => Invalid value — cluster-fraction features disabled.")
        print()
        print("  [3b] Post-bag feature selection — optional. Press Enter to skip.")
        print("  Methods available (enter comma-separated numbers, or blank to skip):")
        print("    1 → CorrFilter   [unsupervised — correlation threshold]")
        print("    2 → MIM          [supervised   — top-K by mutual information]")
        print("    3 → CMI/mRMR     [supervised   — top-K by min-redundancy max-relevance]")
        print("    4 → HighVariance [unsupervised — top-K by variance]")
        print("    5 → PCALoadings  [unsupervised — top-K by PCA contribution]")
        print("  Order matters — methods run sequentially on the bag feature matrix.")
        fs_input = input("  Selection (e.g. '1,4' or blank to skip): ").strip()
        if fs_input:
            method_map = {
                "1": "CorrFilter",
                "2": "MIM",
                "3": "CMI",
                "4": "HighVariance",
                "5": "PCALoadings",
            }
            chosen_numbers = [token.strip() for token in fs_input.split(",")]
            for token in chosen_numbers:
                if token in method_map:
                    bag_feature_selection_methods.append(method_map[token])
                else:
                    print(f"  => Unknown selection '{token}' — ignored.")
            if bag_feature_selection_methods:
                print(f"  => Methods selected: {bag_feature_selection_methods}")
                bag_feature_selection_top_k = _prompt_positive_integer(
                    prompt="  Top-K features to keep (for ranking methods)", default=20
                )
                if "CorrFilter" in bag_feature_selection_methods:
                    corr_input = input(
                        "  Correlation threshold for CorrFilter (default 0.95): "
                    ).strip()
                    if corr_input:
                        try:
                            bag_feature_selection_corr_threshold = float(corr_input)
                        except ValueError:
                            print("  => Invalid value — using default 0.95.")
            else:
                print("  => No valid methods selected — post-bag feature selection disabled.")
    print()

    # ------------------------------------------------------------------
    # [4] Model (skipped for MIL — model is built internally)
    # ------------------------------------------------------------------
    mil_max_epochs: int = 300
    mil_patience: int = 30
    if strategy == "MIL":
        chosen_model = "RandomForest"   # placeholder — unused by MIL
        print("[4] Model family: skipped — MIL uses its own attention neural network.")
        mil_max_epochs = _prompt_positive_integer(
            prompt="  Max training epochs per fold", default=300
        )
        mil_patience = _prompt_positive_integer(
            prompt="  Early stopping patience (epochs without improvement)", default=30
        )
    else:
        print("[4] Model family:")
        for index, model_name in enumerate(ALLOWED_MODEL_NAMES, start=1):
            description = _MODEL_DESCRIPTIONS[model_name]
            print(f"  [{index}] {model_name:<14} — {description}")

        chosen_model = _prompt_choice(
            prompt="Select model",
            choices=ALLOWED_MODEL_NAMES,
            display_indices=[str(i) for i in range(1, len(ALLOWED_MODEL_NAMES) + 1)],
        )
    print()

    # ------------------------------------------------------------------
    # [5] Evaluation strategy (forced to LOGO for MIL)
    # ------------------------------------------------------------------
    n_folds: int = 5
    if strategy == "MIL":
        evaluation_strategy = "LOGO"
        print("[5] Evaluation strategy: forced to LOGO (required for MIL bags).")
    else:
        print("[5] Evaluation strategy:")
        print("  [1] StandardCV — stratified 80/20 split + optional k-fold cross-validation")
        print("  [2] LOGO       — Leave-One-Group-Out: train on all subjects except one,")
        print("                   test on the left-out subject (recommended for few subjects)")
        evaluation_strategy = _prompt_choice(
            prompt="Select evaluation strategy",
            choices=["StandardCV", "LOGO"],
            display_indices=["1", "2"],
        )

        if evaluation_strategy == "StandardCV":
            n_folds = _prompt_positive_integer(
                prompt="  Number of k-fold CV folds (0 to skip k-fold)",
                default=5,
                allow_zero=True,
            )
    print()

    # ------------------------------------------------------------------
    # [6] In-fold feature scaling (skipped for MIL)
    # ------------------------------------------------------------------
    scale_features_in_fold: bool = False
    if strategy == "MIL":
        print("[6] Feature scaling: skipped — MIL uses its own internal normalization.")
    else:
        print("[6] Scale features inside each CV fold?")
        print("  Fits a StandardScaler on the training split of each fold, then applies")
        print("  it to both train and test. Prevents data leakage.")
        print("  Recommended for ElasticNet, SVM, and KNeighbors (distance-based).")
        scale_features_in_fold = _prompt_yes_no(prompt="  Scale features in fold", default=False)
    print()

    # ------------------------------------------------------------------
    # [7] SHAP explainability (skipped for MIL — use plot_mil_attention() instead)
    # ------------------------------------------------------------------
    shap_logo_min_fold_score: Optional[float] = None
    if strategy == "MIL":
        compute_shap = False
        print("[7] SHAP: disabled for MIL strategy.")
        print("  Use plot_mil_attention() to visualise attention weights instead.")
    else:
        print("[7] SHAP feature importance plots (yes/no, default yes):")
        compute_shap = _prompt_yes_no(prompt="  Compute SHAP", default=True)
    if compute_shap and evaluation_strategy == "LOGO":
        print(
            "  [LOGO] SHAP fold accuracy threshold (optional).\n"
            "  Folds below this accuracy are excluded from the SHAP average.\n"
            "  Example: 0.8 keeps only folds with accuracy >= 80%.\n"
            "  Press Enter to use all folds (no threshold):"
        )
        threshold_input = input("  Threshold (e.g. 0.8, or Enter to skip): ").strip()
        if threshold_input:
            try:
                shap_logo_min_fold_score = float(threshold_input)
                print(f"  => SHAP threshold set to {shap_logo_min_fold_score}")
            except ValueError:
                print("  => Invalid value — no threshold applied (using all folds).")
        else:
            print("  => No threshold — all folds will be included in SHAP average.")

    shap_per_class_plots: bool = False
    if compute_shap and task_type == "classification":
        print(
            "  Show separate SHAP plots per target class?\n"
            "  (Global chart always shown first; per-class charts add one figure per class.)\n"
            "  (1) No  — global chart only [default]\n"
            "  (2) Yes — global chart + one chart per class"
        )
        per_class_answer = input("  Choice [1]: ").strip() or "1"
        shap_per_class_plots = per_class_answer == "2"
        if shap_per_class_plots:
            print("  => Per-class SHAP charts enabled.")
        else:
            print("  => Global chart only.")
    print()

    # ------------------------------------------------------------------
    # [8] Random state
    # ------------------------------------------------------------------
    random_state = _prompt_positive_integer(
        prompt="[8] Random state (for full reproducibility)", default=42
    )
    print()

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("=> Configuration summary:")
    print(f"   task_type            : {task_type}")
    print(f"   target_obs_column    : {target_obs_column}")
    print(f"   strategy             : {strategy}", end="")
    if strategy == "Bags":
        print(f" ({bags_per_subject} bags/subject, {mitos_per_bag} mitos/bag", end="")
        print(f", sampling: {bag_sampling_mode})", end="")
        if bag_n_clusters:
            print(f" + cluster fractions (k={bag_n_clusters})", end="")
        if bag_feature_selection_methods:
            print(f" — post-bag FS: {bag_feature_selection_methods}", end="")
    print()
    if strategy == "MIL":
        print(f"   model_name           : MIL (internal ABMIL — encoder_dim=32, attention_dim=16)")
        print(f"   mil_max_epochs       : {mil_max_epochs}")
        print(f"   mil_patience         : {mil_patience}")
    else:
        print(f"   model_name           : {chosen_model}  (model_params={{}}, sklearn defaults)")
    print(f"   evaluation_strategy  : {evaluation_strategy}", end="")
    if evaluation_strategy == "StandardCV":
        folds_str = f"{n_folds}-fold CV" if n_folds > 0 else "single split, no k-fold"
        print(f" (80/20 split, {folds_str})", end="")
    print()
    print(f"   scale_features_in_fold: {scale_features_in_fold}")
    print(f"   compute_shap         : {compute_shap}")
    if strategy == "MIL":
        print("   Tip: use plot_mil_attention() and get_mil_attention_dataframe() for results.")
    else:
        if compute_shap and evaluation_strategy == "LOGO":
            print(f"   shap_logo_min_fold_score: {shap_logo_min_fold_score}")
        print()
        print("   Tip: to use expert hyperparameters, add after this call:")
        print("        from mito_marker.analysis.ml_config import DEFAULT_MODEL_PARAMS")
        print(f"        config['model_params'] = DEFAULT_MODEL_PARAMS['{chosen_model}']")
    print(f"   random_state         : {random_state}")
    print("=" * 60)

    ml_config = {
        "task_type":                  task_type,
        "target_obs_column":          target_obs_column,
        "subject_id_column":          "unique_subject_ID",
        "strategy":                   strategy,
        "bags_per_subject":           bags_per_subject,
        "mitos_per_bag":              mitos_per_bag,
        "bag_statistics":             ["mean", "std", "median", "skew"],
        "bag_n_clusters":             bag_n_clusters,
        "bag_sampling_mode":          bag_sampling_mode,
        "bag_max_overlap_fraction":   bag_max_overlap_fraction,
        "bag_feature_selection_methods":        bag_feature_selection_methods,
        "bag_feature_selection_top_k":          bag_feature_selection_top_k,
        "bag_feature_selection_corr_threshold": bag_feature_selection_corr_threshold,
        "model_name":                 chosen_model,
        "model_params":               {},
        "evaluation_strategy":        evaluation_strategy,
        "logo_group_by_column":       None,   # not prompted interactively — set manually after call
        "test_size":                  0.2,
        "n_folds":                    n_folds,
        "scale_features_in_fold":     scale_features_in_fold,
        "compute_shap":               compute_shap,
        "shap_waterfall_sample_index": 0,
        "shap_logo_min_fold_score":   shap_logo_min_fold_score,
        "shap_per_class_plots":       shap_per_class_plots,
        "random_state":               random_state,
        # MIL parameters (passthrough into ml_mil.train_mil_model via ml_config)
        "mil_encoder_dim":    32,
        "mil_attention_dim":  16,
        "mil_dropout":        0.3,
        "mil_lr":             1e-3,
        "mil_weight_decay":   1e-4,
        "mil_max_epochs":     mil_max_epochs,
        "mil_patience":       mil_patience,
        "mil_min_mitos":      5,
        "mil_max_mitos_per_subject": None,
    }

    _validate_ml_config(ml_config)
    return ml_config


# ---------------------------------------------------------------------------
# Internal: validation
# ---------------------------------------------------------------------------


def _validate_ml_config(ml_config: dict) -> None:
    """
    Assert that ml_config contains all required keys with valid types and values.

    Arguments:
        ml_config: Dict to validate.

    Raises:
        KeyError:   A required key is missing.
        TypeError:  A value has the wrong Python type.
        ValueError: A value is outside its allowed set.
    """
    required_keys: dict[str, type] = {
        "task_type":                  str,
        "target_obs_column":          str,
        "subject_id_column":          str,
        "strategy":                   str,
        "bags_per_subject":           int,
        "mitos_per_bag":              int,
        "bag_statistics":             list,
        "model_name":                 str,
        "model_params":               dict,
        "evaluation_strategy":        str,
        "test_size":                  float,
        "n_folds":                    int,
        "scale_features_in_fold":     bool,
        "bag_feature_selection_top_k":          int,
        "bag_feature_selection_corr_threshold": float,
        "compute_shap":               bool,
        "shap_waterfall_sample_index": int,
        "random_state":               int,
    }

    for key, expected_type in required_keys.items():
        if key not in ml_config:
            raise KeyError(
                f"ml_config is missing required key: '{key}'. "
                f"Use get_default_ml_config() to build a complete config."
            )
        value = ml_config[key]
        if not isinstance(value, expected_type):
            raise TypeError(
                f"ml_config['{key}'] must be {expected_type.__name__}, "
                f"got {type(value).__name__}: {value!r}"
            )

    # bag_feature_selection_methods is optional — list of method names or absent.
    bag_fs_methods = ml_config.get("bag_feature_selection_methods", [])
    if not isinstance(bag_fs_methods, list):
        raise TypeError(
            f"ml_config['bag_feature_selection_methods'] must be a list, "
            f"got {type(bag_fs_methods).__name__}: {bag_fs_methods!r}"
        )
    for method in bag_fs_methods:
        if method not in ALLOWED_BAG_FS_METHODS:
            raise ValueError(
                f"Unknown bag_feature_selection_methods entry: {method!r}. "
                f"Allowed: {sorted(ALLOWED_BAG_FS_METHODS)}"
            )

    # shap_per_class_plots is optional — bool, defaults to False.
    per_class_value = ml_config.get("shap_per_class_plots", False)
    if not isinstance(per_class_value, bool):
        raise TypeError(
            f"ml_config['shap_per_class_plots'] must be a bool, "
            f"got {type(per_class_value).__name__}: {per_class_value!r}"
        )

    # shap_logo_min_fold_score is optional — float in (0, 1] or None.
    threshold_value = ml_config.get("shap_logo_min_fold_score", None)
    if threshold_value is not None:
        if not isinstance(threshold_value, (int, float)):
            raise TypeError(
                f"ml_config['shap_logo_min_fold_score'] must be a float or None, "
                f"got {type(threshold_value).__name__}: {threshold_value!r}"
            )
        if float(threshold_value) <= 0.0:
            raise ValueError(
                f"ml_config['shap_logo_min_fold_score'] must be > 0, "
                f"got {threshold_value!r}"
            )

    # logo_group_by_column is optional — None or a non-empty string.
    logo_group_col = ml_config.get("logo_group_by_column", None)
    if logo_group_col is not None:
        if not isinstance(logo_group_col, str) or not logo_group_col.strip():
            raise TypeError(
                "ml_config['logo_group_by_column'] must be a non-empty str or None, "
                f"got {logo_group_col!r}"
            )

    if ml_config["task_type"] not in ALLOWED_TASK_TYPES:
        raise ValueError(
            f"ml_config['task_type'] must be one of {ALLOWED_TASK_TYPES}, "
            f"got {ml_config['task_type']!r}"
        )
    if ml_config["strategy"] not in ALLOWED_STRATEGIES:
        raise ValueError(
            f"ml_config['strategy'] must be one of {ALLOWED_STRATEGIES}, "
            f"got {ml_config['strategy']!r}"
        )
    # MIL has its own internal model — model_name is unused for this strategy.
    # For all other strategies, validate model_name against the allowed list.
    if ml_config["strategy"] != "MIL":
        if ml_config["model_name"] not in ALLOWED_MODEL_NAMES:
            raise ValueError(
                f"ml_config['model_name'] must be one of {ALLOWED_MODEL_NAMES}, "
                f"got {ml_config['model_name']!r}"
            )
    # MIL is designed for LOGO (one subject = one bag → can't split a bag into train/test).
    # StandardCV would require splitting a single subject's mitos, which defeats the purpose.
    if ml_config["strategy"] == "MIL" and ml_config["evaluation_strategy"] != "LOGO":
        warnings.warn(
            "MIL strategy is designed for LOGO evaluation (one subject left out per fold). "
            "StandardCV splits individual mitochondria — not subjects — which breaks "
            "the MIL bag structure. Set evaluation_strategy='LOGO' for correct results.",
            UserWarning,
            stacklevel=3,
        )
    if ml_config["evaluation_strategy"] not in ALLOWED_EVALUATION_STRATEGIES:
        raise ValueError(
            f"ml_config['evaluation_strategy'] must be one of "
            f"{ALLOWED_EVALUATION_STRATEGIES}, "
            f"got {ml_config['evaluation_strategy']!r}"
        )
    for stat in ml_config["bag_statistics"]:
        if stat not in ALLOWED_BAG_STATISTICS:
            raise ValueError(
                f"bag_statistics entry {stat!r} is not allowed. "
                f"Allowed values: {ALLOWED_BAG_STATISTICS}"
            )

    # bag_n_clusters is optional — None (disabled) or a positive int >= 2.
    bag_n_clusters = ml_config.get("bag_n_clusters", None)
    if bag_n_clusters is not None:
        if not isinstance(bag_n_clusters, int) or isinstance(bag_n_clusters, bool):
            raise TypeError(
                f"ml_config['bag_n_clusters'] must be an int or None, "
                f"got {type(bag_n_clusters).__name__}: {bag_n_clusters!r}"
            )
        if bag_n_clusters < 2:
            raise ValueError(
                f"ml_config['bag_n_clusters'] must be >= 2 (need at least 2 clusters "
                f"for fractions to be meaningful), got {bag_n_clusters}"
            )
    # bag_sampling_mode / bag_max_overlap_fraction are optional — absent means the
    # create_bags() defaults ("auto", 0.2) apply, so old configs keep working.
    bag_sampling_mode = ml_config.get("bag_sampling_mode", "auto")
    if not isinstance(bag_sampling_mode, str):
        raise TypeError(
            f"ml_config['bag_sampling_mode'] must be str, "
            f"got {type(bag_sampling_mode).__name__}: {bag_sampling_mode!r}"
        )
    if bag_sampling_mode not in ALLOWED_BAG_SAMPLING_MODES:
        raise ValueError(
            f"ml_config['bag_sampling_mode'] must be one of "
            f"{sorted(ALLOWED_BAG_SAMPLING_MODES)}, got {bag_sampling_mode!r}"
        )

    bag_max_overlap_fraction = ml_config.get("bag_max_overlap_fraction", 0.2)
    if isinstance(bag_max_overlap_fraction, bool) or not isinstance(
        bag_max_overlap_fraction, (int, float)
    ):
        raise TypeError(
            f"ml_config['bag_max_overlap_fraction'] must be a float, "
            f"got {type(bag_max_overlap_fraction).__name__}: {bag_max_overlap_fraction!r}"
        )
    if not 0.0 < float(bag_max_overlap_fraction) <= 1.0:
        raise ValueError(
            f"ml_config['bag_max_overlap_fraction'] must be in (0, 1], "
            f"got {bag_max_overlap_fraction}"
        )

    # A Bags run needs at least one feature source: raw statistics, cluster
    # fractions, or both — never neither.
    if (
        ml_config["strategy"] == "Bags"
        and not ml_config["bag_statistics"]
        and not bag_n_clusters
    ):
        raise ValueError(
            "Bags strategy requires at least one feature source: a non-empty "
            "'bag_statistics' list, a 'bag_n_clusters' value, or both."
        )

    if not 0.0 < ml_config["test_size"] < 1.0:
        raise ValueError(
            f"ml_config['test_size'] must be between 0 and 1, "
            f"got {ml_config['test_size']}"
        )


# ---------------------------------------------------------------------------
# Internal: prompt helpers
# ---------------------------------------------------------------------------


def _prompt_choice(
    prompt: str,
    choices: List[str],
    display_indices: List[str],
) -> str:
    """
    Prompt until the user enters a valid index and return the corresponding choice.
    """
    while True:
        raw = input(f"{prompt} ({'/'.join(display_indices)}): ").strip()
        if raw in display_indices:
            return choices[display_indices.index(raw)]
        print(
            f"  Invalid choice '{raw}'. "
            f"Please enter one of: {', '.join(display_indices)}"
        )


def _prompt_indexed_choice(prompt: str, options: List[str]) -> str:
    """
    Display options numbered 1..N and return the chosen option string.
    """
    valid_indices = [str(i) for i in range(1, len(options) + 1)]
    while True:
        raw = input(f"{prompt} (1–{len(options)}): ").strip()
        if raw in valid_indices:
            return options[int(raw) - 1]
        print(
            f"  Invalid choice '{raw}'. "
            f"Please enter a number between 1 and {len(options)}."
        )


def _prompt_positive_integer(
    prompt: str,
    default: int,
    allow_zero: bool = False,
) -> int:
    """
    Prompt for a positive integer; return default when the user presses Enter.
    """
    while True:
        raw = input(f"{prompt} (default {default}): ").strip()
        if raw == "":
            return default
        try:
            value = int(raw)
            if allow_zero and value == 0:
                return value
            if value > 0:
                return value
            print(f"  Please enter a positive integer (got {value}).")
        except ValueError:
            print(f"  Please enter a valid integer (got '{raw}').")


def _prompt_yes_no(prompt: str, default: bool) -> bool:
    """
    Prompt for yes/no; return default when the user presses Enter.
    """
    default_str = "yes" if default else "no"
    while True:
        raw = input(f"{prompt} (yes/no, default {default_str}): ").strip().lower()
        if raw == "":
            return default
        if raw in ("y", "yes"):
            return True
        if raw in ("n", "no"):
            return False
        print(f"  Please enter 'yes' or 'no' (got '{raw}').")
