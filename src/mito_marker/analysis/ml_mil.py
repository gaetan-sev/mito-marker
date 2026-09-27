"""
ml_mil.py

Attention-Based Multiple Instance Learning (ABMIL) for mitochondria classification.

In MIL, each subject is a "bag" of mitochondria. The model classifies subjects
(Young/Old, species…) by looking at ALL mitochondria from a subject together,
without collapsing them into fixed statistics first.

Each mitochondrion receives a scalar attention weight (between 0 and 1, all
weights in a bag sum to 1). Scientific interpretation:
  - Attention weight ≠ "this specific mito has the phenotype"
  - Attention weight = "mitos with THIS morphological profile drove the
    classification for this subject"
  - Secondary analysis (plot_mil_attention, get_mil_attention_dataframe):
    compare feature values of high-attention vs low-attention mitos to identify
    the discriminative morphological sub-population.

Architecture — gated attention MIL (Ilse et al. 2018):

    Input: bag tensor of shape (n_mitos, n_features)
           ↓
    Instance encoder: Linear(n_features → encoder_dim) → ReLU → Dropout
           ↓  one embedding per mito
    Gated attention: tanh-branch × sigmoid-branch → Linear → softmax over mitos
           ↓  one scalar per mito, all sum to 1
    Bag representation: z = Σ(attention_i × embedding_i)
           ↓  single vector of shape (encoder_dim,)
    Classifier: Linear(encoder_dim → n_classes)

Reference:
    Ilse, M., Tomczak, J., & Welling, M. (2018).
    Attention-based deep multiple instance learning. ICML 2018. arXiv:1802.04712

Dependencies:
    torch>=2.0 is required for training and inference. It is NOT imported at
    module level — only when a function that trains or predicts is called.
    This allows the package to load without PyTorch for users who only use
    SingleMito or Bags strategies.

Typical usage:
    from mito_marker.analysis.ml_config import get_default_ml_config
    from mito_marker.analysis.ml_pipeline import run_ml_analysis
    from mito_marker.analysis.ml_mil import plot_mil_attention, get_mil_attention_dataframe

    config = get_default_ml_config()
    config["strategy"]            = "MIL"
    config["evaluation_strategy"] = "LOGO"
    config["target_obs_column"]   = "condition"
    config["model_name"]          = "MIL"
    results = run_ml_analysis(my_anndata, ml_config=config)
    plot_mil_attention(results, my_anndata, config)
    attention_df = get_mil_attention_dataframe(results, my_anndata, config)
"""

import copy
import warnings
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import anndata
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import LabelEncoder

if TYPE_CHECKING:
    # Annotations only: torch stays an optional runtime dependency.
    import torch

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _sort_class_names_numerically(class_names: List[str]) -> List[str]:
    """Sort class name strings numerically when all values are numeric, alphabetically otherwise."""
    try:
        return sorted(class_names, key=lambda x: float(x))
    except (ValueError, TypeError):
        return sorted(class_names)


# ---------------------------------------------------------------------------
# Internal: PyTorch model factory (lazy import)
# ---------------------------------------------------------------------------


def _build_attention_mil_model(
    n_features: int,
    encoder_dim: int,
    attention_dim: int,
    n_classes: int,
    dropout: float,
) -> object:
    """
    Instantiate an AttentionMILModel after importing PyTorch.

    The class is defined here rather than at module level so that importing
    ml_mil.py does not require PyTorch. The class is created fresh on each call
    to avoid stale references across different PyTorch versions.

    Arguments:
        n_features:    Number of input features per mitochondrion.
        encoder_dim:   Embedding dimension from the instance encoder.
        attention_dim: Hidden dimension of the gated attention network.
        n_classes:     Number of classification output classes.
        dropout:       Dropout probability applied after the instance encoder.

    Returns:
        Unfitted AttentionMILModel instance.

    Raises:
        ImportError: torch is not installed.
    """
    try:
        import torch.nn as nn
    except ImportError as exc:
        raise ImportError(
            "MIL strategy requires PyTorch. Install it with: pip install torch>=2.0"
        ) from exc

    class AttentionMILModel(nn.Module):
        """
        Gated attention MIL model (Ilse et al. 2018).

        Encodes each instance (mitochondrion) independently, then computes
        a set of attention weights via a two-branch (tanh × sigmoid) gate.
        The bag representation is the attention-weighted sum of instance
        embeddings. A linear classifier maps it to class logits.

        The gated design (product of tanh and sigmoid) is more selective than
        a single-branch attention — important for small datasets where the model
        must converge quickly on a discriminative subset of mitos.
        """

        def __init__(self) -> None:
            super().__init__()
            # Instance encoder: one embedding per mitochondrion
            self.instance_encoder = nn.Sequential(
                nn.Linear(n_features, encoder_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            # Gated attention — tanh branch models feature content
            self.attention_V = nn.Linear(encoder_dim, attention_dim)
            # Gated attention — sigmoid branch controls how much to let through
            self.attention_U = nn.Linear(encoder_dim, attention_dim)
            # Scalar attention score per instance
            self.attention_w = nn.Linear(attention_dim, 1, bias=False)
            # Bag-level classifier
            self.classifier = nn.Linear(encoder_dim, n_classes)

        def forward(
            self, bag_tensor: "torch.Tensor"
        ) -> Tuple["torch.Tensor", "torch.Tensor"]:
            """
            Forward pass: encode instances, pool with gated attention, classify.

            Arguments:
                bag_tensor: Float tensor of shape (n_instances, n_features).

            Returns:
                logits:            Float tensor of shape (1, n_classes).
                attention_weights: Float tensor of shape (n_instances,), sums to 1.
            """
            import torch

            h = self.instance_encoder(bag_tensor)       # (n_instances, encoder_dim)
            # Gated attention: element-wise product of the two branches
            a_V = torch.tanh(self.attention_V(h))       # (n_instances, attention_dim)
            a_U = torch.sigmoid(self.attention_U(h))    # (n_instances, attention_dim)
            a = self.attention_w(a_V * a_U)             # (n_instances, 1)
            a = torch.softmax(a, dim=0)                 # (n_instances, 1), sums to 1
            # Weighted sum of instance embeddings → bag representation
            z = torch.sum(a * h, dim=0, keepdim=True)  # (1, encoder_dim)
            logits = self.classifier(z)                 # (1, n_classes)
            attention_weights = a.squeeze(1)            # (n_instances,)
            return logits, attention_weights

    return AttentionMILModel()


# ---------------------------------------------------------------------------
# Internal: model training
# ---------------------------------------------------------------------------


def train_mil_model(
    bags_dict: Dict[str, Tuple[np.ndarray, float]],
    ml_config: dict,
    label_encoder: Optional[LabelEncoder],
    random_state: int,
) -> object:
    """
    Train an AttentionMILModel on a collection of subject bags.

    Supports both classification (CrossEntropyLoss) and regression (MSELoss).
    Training runs bag-by-bag (batch_size=1) with Adam optimizer.
    One subject is held out from the training set as a validation bag for early
    stopping. The model weights that achieved the lowest validation loss are
    restored before the function returns.

    Arguments:
        bags_dict:     {subject_id: (instance_matrix, label_value)} where
                       instance_matrix has shape (n_mitos, n_features), float32.
                       label_value is an int (class index) for classification
                       or a float (target value) for regression.
        ml_config:     Full ml_config dict. Reads task_type and mil_* keys.
        label_encoder: Pre-fitted LabelEncoder for classification; None for regression.
        random_state:  Integer seed for torch and numpy RNGs.

    Returns:
        Fitted AttentionMILModel in eval mode.

    Raises:
        ImportError: torch is not installed.
        ValueError:  bags_dict is empty.
    """
    try:
        import torch
        import torch.nn as nn
    except ImportError as exc:
        raise ImportError(
            "MIL strategy requires PyTorch. Install it with: pip install torch>=2.0"
        ) from exc

    if not bags_dict:
        raise ValueError(
            "bags_dict is empty — at least one training bag is required."
        )

    encoder_dim = int(ml_config.get("mil_encoder_dim", 32))
    attention_dim = int(ml_config.get("mil_attention_dim", 16))
    dropout = float(ml_config.get("mil_dropout", 0.3))
    lr = float(ml_config.get("mil_lr", 1e-3))
    weight_decay = float(ml_config.get("mil_weight_decay", 1e-4))
    max_epochs = int(ml_config.get("mil_max_epochs", 300))
    patience = int(ml_config.get("mil_patience", 30))

    task_type = ml_config.get("task_type", "classification")

    # Infer model dimensions from the first bag
    first_matrix = next(iter(bags_dict.values()))[0]
    n_features = first_matrix.shape[1]
    # Regression uses a single output neuron; classification uses one per class.
    if task_type == "regression":
        n_classes = 1
    else:
        n_classes = len(label_encoder.classes_)

    # Seed for full reproducibility
    torch.manual_seed(random_state)
    np.random.seed(random_state)

    model = _build_attention_mil_model(
        n_features=n_features,
        encoder_dim=encoder_dim,
        attention_dim=attention_dim,
        n_classes=n_classes,
        dropout=dropout,
    )

    # Use GPU when available — significant speedup on Colab / cloud instances.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss() if task_type == "regression" else nn.CrossEntropyLoss()

    # Pre-convert all bag matrices to device tensors once before the training loop.
    # Avoids repeated numpy→tensor conversion every epoch (saves significant time
    # with many subjects and hundreds of epochs).
    bag_tensors: Dict[str, "torch.Tensor"] = {
        subj_id: torch.from_numpy(matrix.astype(np.float32)).to(device)
        for subj_id, (matrix, _) in bags_dict.items()
    }

    subject_ids = list(bags_dict.keys())

    # Hold out one validation subject for early stopping (reproducible permutation)
    rng = np.random.default_rng(random_state)
    shuffled_ids = rng.permutation(subject_ids).tolist()
    if len(shuffled_ids) >= 2:
        val_subject = shuffled_ids[-1]
        train_subjects = shuffled_ids[:-1]
    else:
        # Too few subjects to hold out one — disable early stopping
        val_subject = None
        train_subjects = shuffled_ids

    best_val_loss = float("inf")
    best_weights = copy.deepcopy(model.state_dict())
    epochs_no_improve = 0

    model.train()
    for epoch in range(max_epochs):
        # Shuffle training bag order each epoch to diversify gradient updates
        epoch_rng = np.random.default_rng(random_state + epoch + 1)
        epoch_order = epoch_rng.permutation(train_subjects).tolist()

        for subj_id in epoch_order:
            _, label_val = bags_dict[subj_id]
            bag_tensor = bag_tensors[subj_id]  # already on device

            optimizer.zero_grad()
            logits, _ = model(bag_tensor)
            if task_type == "regression":
                # Shape (1, 1) matches logits shape for MSELoss.
                label_tensor = torch.tensor(
                    [[float(label_val)]], dtype=torch.float32, device=device
                )
            else:
                label_tensor = torch.tensor(
                    [int(label_val)], dtype=torch.long, device=device
                )
            loss = criterion(logits, label_tensor)
            loss.backward()
            optimizer.step()

        # Early stopping: evaluate on the held-out validation subject
        if val_subject is not None:
            model.eval()
            with torch.no_grad():
                _, val_label = bags_dict[val_subject]
                val_tensor = bag_tensors[val_subject]  # already on device
                val_logits, _ = model(val_tensor)
                if task_type == "regression":
                    val_pred = float(val_logits.squeeze().item())
                    val_loss = float((val_pred - float(val_label)) ** 2)
                else:
                    val_label_tensor = torch.tensor(
                        [int(val_label)], dtype=torch.long, device=device
                    )
                    val_loss = float(criterion(val_logits, val_label_tensor).item())
            model.train()

            if val_loss < best_val_loss - 1e-7:
                best_val_loss = val_loss
                best_weights = copy.deepcopy(model.state_dict())
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    break

    # Restore the weights from the epoch with lowest validation loss
    model.load_state_dict(best_weights)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Public: LOGO cross-validation
# ---------------------------------------------------------------------------


def run_mil_logo_cv(
    data_matrix: np.ndarray,
    obs_dataframe: pd.DataFrame,
    ml_config: dict,
    feature_names: List[str],
    label_encoder: Optional[LabelEncoder],
) -> dict:
    """
    Run Leave-One-Subject-Out cross-validation with the MIL attention model.

    Each LOGO fold:
      - Train an AttentionMILModel on all subjects except one.
      - Test on the left-out subject's bag (all its mitochondria at once).
      - Collect the test subject's attention weights for interpretability.

    The returned dict is schema-compatible with _evaluate_logo_cv() so that
    the rest of run_ml_analysis() (QC printing, confusion matrix, ROC curve,
    .uns storage) works unchanged.
    The extra key "mil_attention_per_subject" carries the scientific output.

    Arguments:
        data_matrix:    Float array of shape (n_mitos, n_features).
        obs_dataframe:  AnnData .obs aligned row-by-row with data_matrix.
                        Must contain subject_id_column and target_obs_column.
        ml_config:      Full ml_config dict.
        feature_names:  Feature name list aligned with data_matrix columns.
        label_encoder:  LabelEncoder — fitted inside this function.

    Returns:
        Results dict with the following MIL-specific key:
        "mil_attention_per_subject": {subject_id: np.ndarray shape (n_mitos,)}
          Each array sums to 1 and shows how much each mitochondrion
          contributed to classifying that subject.

    Raises:
        ImportError: torch is not installed.
        ValueError:  Fewer than 2 valid subjects after applying min_mitos filter.
    """
    try:
        import torch
    except ImportError as exc:
        raise ImportError(
            "MIL strategy requires PyTorch. Install it with: pip install torch>=2.0"
        ) from exc

    task_type = ml_config["task_type"]
    subject_id_column = ml_config["subject_id_column"]
    target_obs_column = ml_config["target_obs_column"]
    random_state = int(ml_config.get("random_state", 42))
    min_mitos = int(ml_config.get("mil_min_mitos", 5))
    max_mitos_raw = ml_config.get("mil_max_mitos_per_subject", None)
    max_mitos: Optional[int] = int(max_mitos_raw) if max_mitos_raw is not None else None

    subject_id_values = obs_dataframe[subject_id_column].values
    target_values = obs_dataframe[target_obs_column].values
    unique_subject_ids = obs_dataframe[subject_id_column].unique()

    # Build per-subject bags: {subject_id: (float32_matrix, label_str)}
    all_bags_str: Dict[str, Tuple[np.ndarray, str]] = {}
    for subj_id in unique_subject_ids:
        mask = subject_id_values == subj_id
        subj_matrix = data_matrix[mask].astype(np.float32)
        subj_targets = target_values[mask]

        if subj_matrix.shape[0] < min_mitos:
            warnings.warn(
                f"Subject '{subj_id}' has only {subj_matrix.shape[0]} mitochondria "
                f"(< mil_min_mitos={min_mitos}). This subject will be skipped.",
                UserWarning,
                stacklevel=2,
            )
            continue

        if task_type == "classification":
            # Assign the majority class label among this subject's mitos
            unique_labels, counts = np.unique(
                subj_targets.astype(str), return_counts=True
            )
            subj_label_str = str(unique_labels[np.argmax(counts)])
        else:
            subj_label_str = str(float(np.mean(subj_targets.astype(float))))

        all_bags_str[str(subj_id)] = (subj_matrix, subj_label_str)

    n_valid_subjects = len(all_bags_str)
    if n_valid_subjects < 2:
        raise ValueError(
            f"MIL LOGO requires at least 2 valid subjects, but only "
            f"{n_valid_subjects} passed the mil_min_mitos={min_mitos} filter. "
            "Lower mil_min_mitos in your config or check your data."
        )

    # Optional: subsample each bag to at most max_mitos mitochondria.
    # Applied once before the fold loop so every fold sees identical subsets.
    # Reduces forward-pass time proportionally (e.g. 2000→500 = 4× faster per bag).
    if max_mitos is not None:
        subsample_rng = np.random.default_rng(random_state + 999)
        n_subsampled = sum(
            1 for matrix, _ in all_bags_str.values() if matrix.shape[0] > max_mitos
        )
        for subj_id in list(all_bags_str.keys()):
            matrix, label_str = all_bags_str[subj_id]
            if matrix.shape[0] > max_mitos:
                indices = subsample_rng.choice(matrix.shape[0], size=max_mitos, replace=False)
                all_bags_str[subj_id] = (matrix[indices], label_str)
        if n_subsampled > 0:
            print(
                f"  MIL: subsampled {n_subsampled} bag(s) to {max_mitos} mitos "
                "(mil_max_mitos_per_subject)."
            )

    # Report compute device (detected from the first fold's model — printed once here).
    if torch.cuda.is_available():
        device_label = f"GPU — {torch.cuda.get_device_name(0)}"
    else:
        device_label = "CPU  (set Colab runtime to GPU for faster training)"
    print(f"  MIL: compute device — {device_label}")

    # Fit the label encoder on all available labels before the fold loop
    if task_type == "classification":
        all_label_strings = [label for _, label in all_bags_str.values()]
        label_encoder.fit(np.unique(all_label_strings))

    # Encode labels: class integers for classification, floats for regression.
    bags_dict_encoded: Dict[str, Tuple[np.ndarray, float]] = {}
    for subj_id, (matrix, label_str) in all_bags_str.items():
        if task_type == "classification":
            encoded_label = float(int(label_encoder.transform([label_str])[0]))
        else:
            encoded_label = float(label_str)
        bags_dict_encoded[subj_id] = (matrix, encoded_label)

    subject_list = list(bags_dict_encoded.keys())

    # Build fold list: [(fold_label, [test_subject_ids])]
    # Standard LOGO: one subject per fold (backward-compatible).
    # LAVO (logo_group_by_column set): all subjects sharing the same group value
    # are left out together in a single fold — one model trained per group.
    logo_group_by_column = ml_config.get("logo_group_by_column", None)
    if logo_group_by_column is not None:
        if logo_group_by_column not in obs_dataframe.columns:
            raise ValueError(
                f"logo_group_by_column '{logo_group_by_column}' not found in .obs. "
                f"Available columns: {list(obs_dataframe.columns)}"
            )
        # Build {subject_id: group_value} from a single row per subject.
        subject_to_group: dict = (
            obs_dataframe[[subject_id_column, logo_group_by_column]]
            .drop_duplicates(subset=subject_id_column)
            .set_index(subject_id_column)[logo_group_by_column]
            .to_dict()
        )
        group_to_subjects: Dict[str, List[str]] = {}
        for subj_id in subject_list:
            group_key = str(subject_to_group.get(subj_id, subj_id))
            if group_key not in group_to_subjects:
                group_to_subjects[group_key] = []
            group_to_subjects[group_key].append(subj_id)
        # Stable alphabetical order so fold indices are reproducible across runs.
        fold_list: List[Tuple[str, List[str]]] = sorted(group_to_subjects.items())
        cv_mode_label = "LAVO"
    else:
        fold_list = [(subj, [subj]) for subj in subject_list]
        cv_mode_label = "LOGO"

    # LOGO / LAVO loop
    per_subject_scores: Dict[str, float] = {}
    cv_scores: List[float] = []
    oof_true: List[str] = []
    oof_pred: List[str] = []
    oof_proba: List[np.ndarray] = []
    mil_attention_per_subject: Dict[str, np.ndarray] = {}
    # Regression accumulators — populated only when task_type == "regression"
    oof_true_floats: List[float] = []
    oof_pred_floats: List[float] = []
    oof_regression_subject_ids: List[str] = []
    per_fold_abs_errors: List[float] = []

    print()
    print(f"  MIL {cv_mode_label}-CV per-fold results:")

    for fold_label, test_subjects_in_fold in fold_list:
        test_subjects_set = set(test_subjects_in_fold)
        # Train on all subjects NOT in this fold's group.
        train_bags = {
            s: v for s, v in bags_dict_encoded.items() if s not in test_subjects_set
        }

        # Use a fold-specific but deterministic seed so folds are independent.
        # Use modular arithmetic to stay within int32 range for torch.manual_seed.
        fold_seed = (random_state + abs(hash(fold_label))) % (2 ** 31)
        fold_model = train_mil_model(
            bags_dict=train_bags,
            ml_config=ml_config,
            label_encoder=label_encoder,
            random_state=fold_seed,
        )

        # Run inference on each test subject in this fold (one bag per subject).
        infer_device = next(fold_model.parameters()).device
        fold_scores_per_subject: List[float] = []
        fold_pred_floats_per_subject: List[float] = []   # regression only

        for test_subject in test_subjects_in_fold:
            test_matrix, _ = bags_dict_encoded[test_subject]
            test_label_str = all_bags_str[test_subject][1]

            with torch.no_grad():
                test_tensor = torch.from_numpy(test_matrix.astype(np.float32)).to(infer_device)
                logits, attention_weights = fold_model(test_tensor)
                # Move results back to CPU before numpy/Python operations.
                logits = logits.cpu()
                attention_weights = attention_weights.cpu()

            if task_type == "classification":
                proba = torch.softmax(logits, dim=1).numpy()[0]   # (n_classes,)
                pred_int = int(torch.argmax(logits, dim=1).item())
                pred_str = str(label_encoder.inverse_transform([pred_int])[0])
                subject_score = 1.0 if pred_str == test_label_str else 0.0
                oof_true.append(test_label_str)
                oof_pred.append(pred_str)
                oof_proba.append(proba)
            else:
                pred_float = float(logits.squeeze().item())
                true_float = float(bags_dict_encoded[test_subject][1])
                subject_score = abs(pred_float - true_float)
                oof_true_floats.append(true_float)
                oof_pred_floats.append(pred_float)
                oof_regression_subject_ids.append(str(test_subject))
                per_fold_abs_errors.append(subject_score)
                fold_pred_floats_per_subject.append(pred_float)

            per_subject_scores[test_subject] = subject_score
            fold_scores_per_subject.append(subject_score)
            # Attention weights: shape (n_mitos_for_this_subject,), sums to 1
            mil_attention_per_subject[test_subject] = attention_weights.numpy()

        # Fold-level score = mean across all test subjects in this fold.
        fold_score = float(np.mean(fold_scores_per_subject))
        cv_scores.append(fold_score)

        # Print fold summary: header line for multi-subject folds (LAVO),
        # then one detail line per subject.
        n_test = len(test_subjects_in_fold)
        if n_test > 1:
            if task_type == "classification":
                print(
                    f"    [group={fold_label}]  accuracy: {fold_score:.3f}"
                    f"  ({n_test} subjects)"
                )
            else:
                print(
                    f"    [group={fold_label}]  MAE: {fold_score:.3f}"
                    f"  ({n_test} subjects)"
                )

        indent = "        " if n_test > 1 else "    "
        for subj_idx, test_subject in enumerate(test_subjects_in_fold):
            test_label_str = all_bags_str[test_subject][1]
            s_score = fold_scores_per_subject[subj_idx]
            if task_type == "classification":
                print(
                    f"{indent}{test_subject:<42} {test_label_str:<15} "
                    f"accuracy: {s_score:.3f}"
                )
            else:
                pred_val = fold_pred_floats_per_subject[subj_idx]
                true_val = float(bags_dict_encoded[test_subject][1])
                print(
                    f"{indent}{test_subject:<42} true={true_val:.2f}  "
                    f"pred={pred_val:.2f}  |error|={s_score:.2f}"
                )

    mean_score = float(np.mean(cv_scores)) if cv_scores else 0.0
    std_score = float(np.std(cv_scores)) if cv_scores else 0.0

    # Compute regression metrics from accumulated OOF predictions.
    # R² is computed globally over all subjects (not averaged per fold) because
    # per-fold R² with a single test point is mathematically undefined.
    overall_r2: Optional[float] = None
    overall_mae: Optional[float] = None
    overall_rmse: Optional[float] = None
    if task_type == "regression" and len(oof_true_floats) > 0:
        from sklearn.metrics import (
            mean_absolute_error as _mae_score,
        )
        from sklearn.metrics import (
            mean_squared_error as _mse_score,
        )
        from sklearn.metrics import (
            r2_score as _r2_score,
        )
        overall_r2 = float(_r2_score(oof_true_floats, oof_pred_floats))
        overall_mae = float(_mae_score(oof_true_floats, oof_pred_floats))
        overall_rmse = float(np.sqrt(_mse_score(oof_true_floats, oof_pred_floats)))

    print()
    if task_type == "classification":
        print(f"  MIL LOGO mean accuracy: {mean_score:.3f} ± {std_score:.3f}")
    else:
        print(
            f"  MIL LOGO regression (OOF): "
            f"R²={overall_r2:.3f}  MAE={overall_mae:.3f}  RMSE={overall_rmse:.3f}"
        )

    # Build aggregate metrics from all out-of-fold predictions (classification only)
    all_true = np.array(oof_true)
    all_pred = np.array(oof_pred)
    all_proba: Optional[np.ndarray] = np.vstack(oof_proba) if oof_proba else None

    roc_auc: float = float("nan")
    if task_type == "classification" and all_proba is not None:
        classes = np.unique(all_true)
        try:
            if len(classes) == 2:
                roc_auc = float(roc_auc_score(all_true, all_proba[:, 1]))
            else:
                roc_auc = float(
                    roc_auc_score(
                        all_true, all_proba,
                        multi_class="ovr", average="macro",
                    )
                )
        except Exception:
            roc_auc = float("nan")

    class_names: Optional[List[str]] = (
        _sort_class_names_numerically([str(c) for c in label_encoder.classes_])
        if task_type == "classification" else None
    )

    return {
        # Classification metrics
        "classification_report": (
            classification_report(all_true, all_pred)
            if task_type == "classification" else None
        ),
        "confusion_matrix": (
            confusion_matrix(all_true, all_pred, labels=class_names)
            if task_type == "classification" else None
        ),
        "roc_auc":               roc_auc if task_type == "classification" else None,
        "cv_scores_accuracy":    cv_scores if task_type == "classification" else None,
        "mean_accuracy":         mean_score if task_type == "classification" else None,
        "std_accuracy":          std_score if task_type == "classification" else None,
        "train_cv_scores_accuracy": [],
        "mean_train_accuracy":   None,
        "std_train_accuracy":    None,
        "class_names":           class_names,
        "roc_y_true":            all_true if task_type == "classification" else None,
        "roc_y_pred_proba":      all_proba if task_type == "classification" else None,
        # Per-subject LOGO scores
        "logo_per_subject_scores": per_subject_scores,
        "logo_group_by_column":    ml_config.get("logo_group_by_column", None),
        # Regression metrics
        "r2_score":        overall_r2,
        "mae":             overall_mae,
        "rmse":            overall_rmse,
        "cv_scores_r2":    per_fold_abs_errors if task_type == "regression" else None,
        "mean_r2":         overall_r2,
        "std_r2":          float(np.std(per_fold_abs_errors)) if task_type == "regression" else None,
        "train_cv_scores_r2": None, "mean_train_r2": None, "std_train_r2": None,
        # Subject-level true/predicted for scatter plot — same keys as _evaluate_logo_cv().
        "logo_subject_true": np.array(oof_true_floats, dtype=float) if task_type == "regression" else None,
        "logo_subject_pred": np.array(oof_pred_floats, dtype=float) if task_type == "regression" else None,
        "logo_subject_ids":  oof_regression_subject_ids if task_type == "regression" else None,
        # SHAP fields — MIL uses attention weights instead (always None here)
        "logo_shap_importance":             None,
        "logo_shap_n_folds_used":           0,
        "logo_shap_subjects_used":          [],
        "logo_shap_explainer_type":         None,
        "logo_shap_oof_values":             None,
        "logo_shap_oof_features":           None,
        "logo_shap_oof_unique_subject_ids": None,
        "logo_shap_oof_conditions":         None,
        # MIL-specific: primary scientific output
        "mil_attention_per_subject": mil_attention_per_subject,
    }


# ---------------------------------------------------------------------------
# Public: attention weight DataFrame
# ---------------------------------------------------------------------------


def get_mil_attention_dataframe(
    results: dict,
    anndata_object: anndata.AnnData,
    ml_config: dict,
) -> pd.DataFrame:
    """
    Build a tidy DataFrame of per-mitochondrion attention weights + feature values.

    This is the primary scientific output of MIL:
      - Sort by attention_weight descending to find the most discriminative mitos
      - Compare feature values between high-attention and low-attention mitos
        to identify which morphological sub-population drives the classification

    DataFrame columns:
        {subject_id_column}, {target_obs_column}, mito_index,
        attention_weight, {feature_1}, ..., {feature_N}

    The result is stored in anndata_object.uns["mil_attention"] and returned.

    Arguments:
        results:         Dict returned by run_ml_analysis(strategy="MIL").
                         Must contain "mil_attention_per_subject".
        anndata_object:  AnnData used during training, with .obs and .X populated.
        ml_config:       ml_config dict used during run_ml_analysis().

    Returns:
        DataFrame with one row per mitochondrion, for all subjects that were
        left out in LOGO folds (i.e. all subjects in the dataset).

    Raises:
        KeyError: "mil_attention_per_subject" not in results.
    """
    mil_attention = results.get("mil_attention_per_subject")
    if mil_attention is None:
        raise KeyError(
            "'mil_attention_per_subject' not found in results. "
            "Run run_ml_analysis() with strategy='MIL' first."
        )

    subject_id_column = ml_config["subject_id_column"]
    target_obs_column = ml_config["target_obs_column"]
    feature_names: List[str] = results.get("feature_names", list(anndata_object.var_names))

    # Get the feature matrix — same path as run_ml_analysis() uses
    from mito_marker.analysis.feature_selection import get_selected_data_matrix

    data_matrix = get_selected_data_matrix(anndata_object)

    rows: List[dict] = []
    for subj_id, attention_weights in mil_attention.items():
        mask = anndata_object.obs[subject_id_column].astype(str) == str(subj_id)
        subject_obs = anndata_object.obs[mask]
        subject_matrix = data_matrix[mask.values]

        # Use the smaller of the two sizes to handle any length mismatch
        n_mitos = min(len(attention_weights), subject_matrix.shape[0])

        condition_values = subject_obs[target_obs_column].unique()
        condition = str(condition_values[0]) if len(condition_values) > 0 else "Unknown"

        obs_index_values = list(subject_obs.index)

        for mito_idx in range(n_mitos):
            row: dict = {
                subject_id_column:  subj_id,
                target_obs_column:  condition,
                "mito_index":       obs_index_values[mito_idx] if mito_idx < len(obs_index_values) else mito_idx,
                "attention_weight": float(attention_weights[mito_idx]),
            }
            for feat_idx, feat_name in enumerate(feature_names):
                if feat_idx < subject_matrix.shape[1]:
                    row[feat_name] = float(subject_matrix[mito_idx, feat_idx])
            rows.append(row)

    attention_dataframe = pd.DataFrame(rows)
    anndata_object.uns["mil_attention"] = attention_dataframe

    print(
        f"  Attention DataFrame: {attention_dataframe.shape[0]:,} mitochondria "
        f"× {attention_dataframe.shape[1]} columns. "
        "Stored in .uns['mil_attention']."
    )
    return attention_dataframe


# ---------------------------------------------------------------------------
# Public: visualization
# ---------------------------------------------------------------------------


def plot_mil_attention(
    results: dict,
    anndata_object: anndata.AnnData,
    ml_config: dict,
    top_n_subjects: int = 5,
    attention_high_quantile: float = 0.9,
    attention_low_quantile: float = 0.1,
) -> List[object]:
    """
    Visualize MIL attention weights to identify the discriminative mito sub-population.

    Produces three figures that together answer the question:
    "Which mitochondria, and which of their features, drove the classification?"

    Figure 1 — Strip plot:
        Attention weight distribution per subject, colored by condition.
        Concentrated distributions (one mito gets most of the weight) indicate
        a dominant sub-population; uniform distributions indicate diffuse signal.

    Figure 2 — PCA scatter:
        All mitochondria in PCA space, colored by attention weight.
        The "hot" (high-attention) region = the morphological sub-population
        the model relied on. This directly identifies WHERE in feature space
        the discriminative mitos live.

    Figure 3 — Feature comparison bar chart:
        Mean normalized feature values for high-attention mitos (top quantile)
        vs low-attention mitos (bottom quantile), pooled across all subjects.
        Bars above zero = features enriched in the discriminative sub-population.
        This directly answers: "which morphological features characterize the
        sub-population that drives the Young/Old or species classification?"

    Arguments:
        results:                   Dict from run_ml_analysis(strategy="MIL").
        anndata_object:            AnnData used during training.
        ml_config:                 ml_config used during training.
        top_n_subjects:            Maximum subjects to include in the legend.
        attention_high_quantile:   Quantile threshold for "high-attention" mitos.
        attention_low_quantile:    Quantile threshold for "low-attention" mitos.

    Returns:
        List of matplotlib Figure objects produced (strip_fig, pca_fig, feat_fig).

    Raises:
        ImportError: matplotlib is not installed.
        KeyError:    results does not contain "mil_attention_per_subject".
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "Matplotlib is required for plot_mil_attention. "
            "Install it with: pip install matplotlib>=3.8"
        ) from exc

    mil_attention = results.get("mil_attention_per_subject")
    if mil_attention is None:
        raise KeyError(
            "'mil_attention_per_subject' not found in results. "
            "Run run_ml_analysis() with strategy='MIL' first."
        )

    subject_id_column = ml_config["subject_id_column"]
    target_obs_column = ml_config["target_obs_column"]
    feature_names: List[str] = results.get("feature_names", list(anndata_object.var_names))

    # Build the attention DataFrame if not already cached
    if "mil_attention" in anndata_object.uns and isinstance(
        anndata_object.uns["mil_attention"], pd.DataFrame
    ):
        attention_dataframe = anndata_object.uns["mil_attention"]
    else:
        attention_dataframe = get_mil_attention_dataframe(results, anndata_object, ml_config)

    conditions = attention_dataframe[target_obs_column].unique()

    # Use the project color palette if available, otherwise fall back to Set2
    color_palette: Dict[str, object] = anndata_object.uns.get("color_palette", {})
    if not color_palette:
        default_colors = plt.get_cmap("Set2").colors
        color_palette = {
            str(cond): default_colors[i % len(default_colors)]
            for i, cond in enumerate(sorted(str(c) for c in conditions))
        }

    figures: List[object] = []

    # ----------------------------------------------------------------
    # Figure 1: Strip plot — attention weight distribution per subject
    # ----------------------------------------------------------------
    all_subjects = list(mil_attention.keys())
    n_subjects = len(all_subjects)

    strip_fig, ax_strip = plt.subplots(figsize=(max(6, n_subjects * 0.7), 5))

    legend_conditions_seen: set = set()
    for subj_idx, subj_id in enumerate(all_subjects):
        weights = mil_attention[subj_id]
        subj_mask = attention_dataframe[subject_id_column].astype(str) == str(subj_id)
        cond = str(
            attention_dataframe.loc[subj_mask, target_obs_column].iloc[0]
        )
        color = color_palette.get(cond, "#888888")

        x_jitter = np.random.default_rng(42 + subj_idx).uniform(
            -0.2, 0.2, size=len(weights)
        )
        ax_strip.scatter(
            np.full(len(weights), subj_idx) + x_jitter,
            weights,
            c=[color],
            alpha=0.35,
            s=6,
            linewidths=0,
        )
        # Diamond marker for the mean attention weight per subject
        legend_label = str(cond) if cond not in legend_conditions_seen else None
        legend_conditions_seen.add(cond)
        ax_strip.scatter(
            subj_idx,
            float(np.mean(weights)),
            c=[color],
            s=60,
            marker="D",
            zorder=5,
            edgecolors="black",
            linewidths=0.5,
            label=legend_label,
        )

    ax_strip.set_xticks(range(n_subjects))
    ax_strip.set_xticklabels(all_subjects, rotation=45, ha="right", fontsize=8)
    ax_strip.set_ylabel("Attention weight")
    ax_strip.set_title(
        "MIL Attention Weight Distribution per Subject\n"
        "(diamond = mean; each dot = one mitochondrion)",
        fontsize=10,
    )
    ax_strip.legend(title="Condition", loc="upper right", fontsize=8)
    strip_fig.tight_layout()
    plt.show()
    figures.append(strip_fig)

    # ----------------------------------------------------------------
    # Figure 2: PCA scatter — mitos colored by attention weight
    # ----------------------------------------------------------------
    feature_cols_in_df = [f for f in feature_names if f in attention_dataframe.columns]
    if len(feature_cols_in_df) >= 2 and attention_dataframe.shape[0] > 2:
        from sklearn.decomposition import PCA

        feature_matrix_for_pca = attention_dataframe[feature_cols_in_df].values.astype(float)
        valid_row_mask = ~np.isnan(feature_matrix_for_pca).any(axis=1)

        if valid_row_mask.sum() >= 3:
            pca_model = PCA(n_components=2, random_state=42)
            pca_coords = pca_model.fit_transform(feature_matrix_for_pca[valid_row_mask])
            attention_values = attention_dataframe["attention_weight"].values[valid_row_mask]

            pca_fig, ax_pca = plt.subplots(figsize=(7, 6))
            scatter = ax_pca.scatter(
                pca_coords[:, 0],
                pca_coords[:, 1],
                c=attention_values,
                cmap="hot",
                alpha=0.5,
                s=8,
                linewidths=0,
                vmin=0,
            )
            pca_fig.colorbar(scatter, ax=ax_pca, label="Attention weight")
            ax_pca.set_xlabel(
                f"PC1 ({pca_model.explained_variance_ratio_[0]:.1%} variance)"
            )
            ax_pca.set_ylabel(
                f"PC2 ({pca_model.explained_variance_ratio_[1]:.1%} variance)"
            )
            ax_pca.set_title(
                "Mitochondria in PCA Space Colored by Attention Weight\n"
                "'Hot' region = discriminative morphological sub-population",
                fontsize=10,
            )
            pca_fig.tight_layout()
            plt.show()
            figures.append(pca_fig)

    # ----------------------------------------------------------------
    # Figure 3: Feature comparison — high-attention vs low-attention mitos
    # ----------------------------------------------------------------
    if feature_cols_in_df:
        all_weights = attention_dataframe["attention_weight"].values
        high_threshold = float(np.quantile(all_weights, attention_high_quantile))
        low_threshold = float(np.quantile(all_weights, attention_low_quantile))

        high_mask = all_weights >= high_threshold
        low_mask = all_weights <= low_threshold

        if high_mask.sum() > 0 and low_mask.sum() > 0:
            high_means = attention_dataframe.loc[high_mask, feature_cols_in_df].mean()
            low_means = attention_dataframe.loc[low_mask, feature_cols_in_df].mean()

            # Normalize by overall feature std for a fair, dimensionless comparison
            overall_std = attention_dataframe[feature_cols_in_df].std().replace(0, 1.0)
            high_means_norm = (high_means / overall_std).values
            low_means_norm = (low_means / overall_std).values

            n_features_plot = len(feature_cols_in_df)
            feat_fig, ax_feat = plt.subplots(
                figsize=(max(6, n_features_plot * 0.55), 5)
            )
            x_positions = np.arange(n_features_plot)
            bar_width = 0.35

            ax_feat.bar(
                x_positions - bar_width / 2,
                high_means_norm,
                bar_width,
                label=f"High attention (top {int((1 - attention_high_quantile) * 100)}%)",
                color="#e74c3c",
                alpha=0.8,
            )
            ax_feat.bar(
                x_positions + bar_width / 2,
                low_means_norm,
                bar_width,
                label=f"Low attention (bottom {int(attention_low_quantile * 100)}%)",
                color="#3498db",
                alpha=0.8,
            )

            ax_feat.set_xticks(x_positions)
            ax_feat.set_xticklabels(feature_cols_in_df, rotation=60, ha="right", fontsize=7)
            ax_feat.set_ylabel("Normalized mean feature value (÷ std)")
            ax_feat.set_title(
                "Morphological Profile: High-Attention vs Low-Attention Mitochondria\n"
                "Bars diverging from zero = features enriched in discriminative sub-population",
                fontsize=9,
            )
            ax_feat.axhline(0, color="black", linewidth=0.5, linestyle="--")
            ax_feat.legend(fontsize=8)
            feat_fig.tight_layout()
            plt.show()
            figures.append(feat_fig)

    return figures
