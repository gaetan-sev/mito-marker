"""
test_analysis_ml_mil.py

Unit and integration tests for mito_marker.analysis.ml_mil.

All tests in this file require PyTorch (torch>=2.0). If torch is not installed,
the entire module is skipped gracefully via pytest.importorskip at the top.

Covers:
  _build_attention_mil_model():
    - Returns an nn.Module instance.
    - Forward pass produces logits (1, n_classes) and attention weights (n_instances,).
    - Attention weights sum to 1 (softmax property).
    - Gradient flows through the network without error.

  train_mil_model():
    - Trains without raising on a minimal bags_dict.
    - Returns a model in eval mode.
    - Determinism: same random_state → same predictions on same test bag.
    - ValueError when bags_dict is empty.

  run_mil_logo_cv():
    - Returns a dict with the same schema as _evaluate_logo_cv() classification output.
    - "mil_attention_per_subject" key is present; each array sums to 1.
    - Attention array shape matches the number of mitos for each test subject.
    - Multi-class (3-class) case runs without error.
    - ValueError when fewer than 2 subjects pass the min_mitos filter.
    - UserWarning when a subject has fewer mitos than mil_min_mitos.

  get_mil_attention_dataframe():
    - Returns a DataFrame with the expected columns.
    - One row per mitochondrion (across all test subjects).
    - Stored in anndata_object.uns["mil_attention"].
    - KeyError when results dict lacks "mil_attention_per_subject".

  Integration via run_ml_analysis(strategy="MIL"):
    - Completes without error and returns a results dict.
    - Results stored in .uns["ml_results"].
    - "mil_attention_per_subject" is present in results.
    - n_samples equals the number of unique subjects (not mito count).
    - UserWarning emitted when evaluation_strategy="StandardCV" is used with MIL.
"""

import warnings

import anndata
import numpy as np
import pandas as pd
import pytest

# Skip this entire module if PyTorch is not installed.
torch = pytest.importorskip("torch")


from mito_marker.analysis.ml_mil import (
    _build_attention_mil_model,
    get_mil_attention_dataframe,
    plot_mil_attention,
    run_mil_logo_cv,
    train_mil_model,
)
from sklearn.preprocessing import LabelEncoder


# ---------------------------------------------------------------------------
# Shared fixture factories
# ---------------------------------------------------------------------------


def _make_mil_anndata(
    n_subjects: int = 6,
    mitos_per_subject: int = 30,
    n_features: int = 8,
    n_classes: int = 2,
    condition_labels: list = None,
    seed: int = 0,
) -> anndata.AnnData:
    """
    Build a minimal AnnData for MIL tests.

    Each subject has mitos_per_subject rows with n_features features.
    Subjects are split evenly across n_classes conditions.
    The unique_subject_ID column is set so that the MIL pipeline can find it.
    """
    rng = np.random.default_rng(seed)
    subject_ids = [f"S{i:03d}" for i in range(n_subjects)]

    if condition_labels is not None:
        assert len(condition_labels) == n_subjects
        conditions = condition_labels
    else:
        class_names = [f"Class{c}" for c in range(n_classes)]
        conditions = [class_names[i % n_classes] for i in range(n_subjects)]

    n_obs = n_subjects * mitos_per_subject
    feature_matrix = rng.standard_normal((n_obs, n_features)).astype(np.float32)
    subject_id_col = np.repeat(subject_ids, mitos_per_subject)
    condition_col = np.repeat(conditions, mitos_per_subject)

    obs_df = pd.DataFrame(
        {
            "unique_subject_ID": subject_id_col,
            "condition":         condition_col,
        },
        index=[f"obs_{i}" for i in range(n_obs)],
    )
    var_df = pd.DataFrame(index=[f"feature_{i}" for i in range(n_features)])
    return anndata.AnnData(X=feature_matrix, obs=obs_df, var=var_df)


def _make_minimal_bags_dict(
    n_subjects: int = 4,
    mitos_per_subject: int = 20,
    n_features: int = 5,
    n_classes: int = 2,
    seed: int = 1,
) -> tuple:
    """
    Build a minimal bags_dict and a fitted LabelEncoder for training tests.

    Returns:
        (bags_dict, label_encoder)
        bags_dict: {subject_id: (float32_matrix, label_int)}
    """
    rng = np.random.default_rng(seed)
    class_names = [f"Class{c}" for c in range(n_classes)]
    label_encoder = LabelEncoder()
    label_encoder.fit(class_names)

    bags_dict = {}
    for i in range(n_subjects):
        subj_id = f"S{i:03d}"
        matrix = rng.standard_normal((mitos_per_subject, n_features)).astype(np.float32)
        label_int = i % n_classes
        bags_dict[subj_id] = (matrix, label_int)

    return bags_dict, label_encoder


def _make_minimal_ml_config(
    strategy: str = "MIL",
    evaluation_strategy: str = "LOGO",
    mil_max_epochs: int = 5,
    mil_patience: int = 3,
) -> dict:
    """
    Build a minimal ml_config suitable for MIL tests.

    Uses very small epoch counts so tests complete quickly.
    """
    return {
        "task_type":             "classification",
        "target_obs_column":     "condition",
        "subject_id_column":     "unique_subject_ID",
        "strategy":              strategy,
        "model_name":            "MIL",
        "model_params":          {},
        "evaluation_strategy":   evaluation_strategy,
        "test_size":             0.2,
        "n_folds":               3,
        "bags_per_subject":      5,
        "mitos_per_bag":         10,
        "bag_statistics":        ["mean"],
        "bag_feature_selection_methods":        [],
        "bag_feature_selection_top_k":          10,
        "bag_feature_selection_corr_threshold": 0.95,
        "scale_features_in_fold":  False,
        "compute_shap":            False,
        "shap_waterfall_sample_index": 0,
        "shap_logo_min_fold_score":    None,
        "random_state":            42,
        # MIL hyperparameters — small values for fast tests
        "mil_encoder_dim":   16,
        "mil_attention_dim":  8,
        "mil_dropout":        0.1,
        "mil_lr":             1e-3,
        "mil_weight_decay":   1e-4,
        "mil_max_epochs":     mil_max_epochs,
        "mil_patience":       mil_patience,
        "mil_min_mitos":      5,
    }


# ---------------------------------------------------------------------------
# Tests: _build_attention_mil_model
# ---------------------------------------------------------------------------


class TestBuildAttentionMILModel:
    """Tests for the model architecture: forward pass, shapes, gradients."""

    def test_returns_nn_module(self):
        import torch.nn as nn
        model = _build_attention_mil_model(
            n_features=10, encoder_dim=16, attention_dim=8, n_classes=2, dropout=0.1
        )
        assert isinstance(model, nn.Module)

    def test_forward_logits_shape(self):
        model = _build_attention_mil_model(
            n_features=8, encoder_dim=16, attention_dim=8, n_classes=2, dropout=0.0
        )
        model.eval()
        bag = torch.randn(20, 8)
        logits, _ = model(bag)
        assert logits.shape == (1, 2)

    def test_forward_attention_shape(self):
        n_instances = 35
        model = _build_attention_mil_model(
            n_features=5, encoder_dim=16, attention_dim=8, n_classes=3, dropout=0.0
        )
        model.eval()
        bag = torch.randn(n_instances, 5)
        _, attention_weights = model(bag)
        assert attention_weights.shape == (n_instances,)

    def test_attention_sums_to_one(self):
        model = _build_attention_mil_model(
            n_features=6, encoder_dim=16, attention_dim=8, n_classes=2, dropout=0.0
        )
        model.eval()
        bag = torch.randn(50, 6)
        _, attention_weights = model(bag)
        total = float(attention_weights.sum().item())
        assert abs(total - 1.0) < 1e-5, f"Attention sum should be 1, got {total}"

    def test_gradient_flows(self):
        import torch.nn as nn
        model = _build_attention_mil_model(
            n_features=4, encoder_dim=8, attention_dim=4, n_classes=2, dropout=0.0
        )
        bag = torch.randn(10, 4)
        label = torch.tensor([0], dtype=torch.long)
        logits, _ = model(bag)
        loss = nn.CrossEntropyLoss()(logits, label)
        loss.backward()
        for param in model.parameters():
            assert param.grad is not None, "All params should have gradients after backward()"

    def test_single_instance_bag(self):
        model = _build_attention_mil_model(
            n_features=5, encoder_dim=16, attention_dim=8, n_classes=2, dropout=0.0
        )
        model.eval()
        bag = torch.randn(1, 5)
        logits, attention_weights = model(bag)
        assert logits.shape == (1, 2)
        assert attention_weights.shape == (1,)
        assert abs(float(attention_weights.sum().item()) - 1.0) < 1e-5

    def test_multiclass_three_classes(self):
        model = _build_attention_mil_model(
            n_features=8, encoder_dim=16, attention_dim=8, n_classes=3, dropout=0.0
        )
        model.eval()
        bag = torch.randn(25, 8)
        logits, _ = model(bag)
        assert logits.shape == (1, 3)


# ---------------------------------------------------------------------------
# Tests: train_mil_model
# ---------------------------------------------------------------------------


class TestTrainMILModel:
    """Tests for the training loop: convergence, eval mode, determinism."""

    def test_trains_without_error(self):
        bags_dict, label_encoder = _make_minimal_bags_dict(n_subjects=4, n_classes=2)
        config = _make_minimal_ml_config()
        model = train_mil_model(bags_dict, config, label_encoder, random_state=0)
        assert model is not None

    def test_returns_model_in_eval_mode(self):
        bags_dict, label_encoder = _make_minimal_bags_dict(n_subjects=4, n_classes=2)
        config = _make_minimal_ml_config()
        model = train_mil_model(bags_dict, config, label_encoder, random_state=0)
        assert not model.training, "Model should be in eval mode after train_mil_model"

    def test_determinism_same_seed(self):
        bags_dict, label_encoder = _make_minimal_bags_dict(n_subjects=4, n_classes=2)
        config = _make_minimal_ml_config()
        test_bag = torch.from_numpy(
            np.random.default_rng(99).standard_normal((20, 5)).astype(np.float32)
        )

        model_a = train_mil_model(bags_dict, config, label_encoder, random_state=7)
        with torch.no_grad():
            logits_a, _ = model_a(test_bag)

        model_b = train_mil_model(bags_dict, config, label_encoder, random_state=7)
        with torch.no_grad():
            logits_b, _ = model_b(test_bag)

        assert torch.allclose(logits_a, logits_b, atol=1e-5), (
            "Same random_state should produce identical logits"
        )

    def test_raises_on_empty_bags_dict(self):
        _, label_encoder = _make_minimal_bags_dict(n_subjects=2, n_classes=2)
        config = _make_minimal_ml_config()
        with pytest.raises(ValueError, match="empty"):
            train_mil_model({}, config, label_encoder, random_state=0)

    def test_single_training_subject_no_crash(self):
        bags_dict, label_encoder = _make_minimal_bags_dict(n_subjects=1, n_classes=2)
        config = _make_minimal_ml_config(mil_max_epochs=3)
        model = train_mil_model(bags_dict, config, label_encoder, random_state=0)
        assert model is not None

    def test_multiclass_three_classes(self):
        bags_dict, label_encoder = _make_minimal_bags_dict(
            n_subjects=6, n_classes=3, mitos_per_subject=15
        )
        config = _make_minimal_ml_config()
        model = train_mil_model(bags_dict, config, label_encoder, random_state=0)
        assert model is not None


# ---------------------------------------------------------------------------
# Tests: run_mil_logo_cv
# ---------------------------------------------------------------------------


class TestRunMilLogoCv:
    """Tests for the LOGO evaluation loop: return schema, attention shapes."""

    def test_returns_dict_with_expected_keys(self):
        adata = _make_mil_anndata(n_subjects=4, mitos_per_subject=20, n_features=5)
        config = _make_minimal_ml_config()
        label_encoder = LabelEncoder()

        result = run_mil_logo_cv(
            data_matrix=adata.X.copy(),
            obs_dataframe=adata.obs,
            ml_config=config,
            feature_names=list(adata.var_names),
            label_encoder=label_encoder,
        )

        expected_keys = [
            "classification_report", "confusion_matrix", "roc_auc",
            "cv_scores_accuracy", "mean_accuracy", "std_accuracy",
            "logo_per_subject_scores", "class_names",
            "roc_y_true", "roc_y_pred_proba",
            "r2_score", "mae", "rmse",
            "mil_attention_per_subject",
        ]
        for key in expected_keys:
            assert key in result, f"Missing key: '{key}'"

    def test_mil_attention_per_subject_is_dict(self):
        adata = _make_mil_anndata(n_subjects=4, mitos_per_subject=20, n_features=5)
        config = _make_minimal_ml_config()
        label_encoder = LabelEncoder()

        result = run_mil_logo_cv(
            data_matrix=adata.X.copy(),
            obs_dataframe=adata.obs,
            ml_config=config,
            feature_names=list(adata.var_names),
            label_encoder=label_encoder,
        )

        attention_dict = result["mil_attention_per_subject"]
        assert isinstance(attention_dict, dict)
        unique_subjects = adata.obs["unique_subject_ID"].unique()
        assert len(attention_dict) == len(unique_subjects)

    def test_attention_arrays_sum_to_one(self):
        adata = _make_mil_anndata(n_subjects=4, mitos_per_subject=20, n_features=5)
        config = _make_minimal_ml_config()
        label_encoder = LabelEncoder()

        result = run_mil_logo_cv(
            data_matrix=adata.X.copy(),
            obs_dataframe=adata.obs,
            ml_config=config,
            feature_names=list(adata.var_names),
            label_encoder=label_encoder,
        )

        for subj_id, weights in result["mil_attention_per_subject"].items():
            total = float(weights.sum())
            assert abs(total - 1.0) < 1e-4, (
                f"Subject '{subj_id}': attention sum should be 1, got {total}"
            )

    def test_attention_array_shape_matches_mito_count(self):
        mitos_per_subject = 25
        adata = _make_mil_anndata(
            n_subjects=4, mitos_per_subject=mitos_per_subject, n_features=5
        )
        config = _make_minimal_ml_config()
        label_encoder = LabelEncoder()

        result = run_mil_logo_cv(
            data_matrix=adata.X.copy(),
            obs_dataframe=adata.obs,
            ml_config=config,
            feature_names=list(adata.var_names),
            label_encoder=label_encoder,
        )

        for subj_id, weights in result["mil_attention_per_subject"].items():
            assert weights.shape == (mitos_per_subject,), (
                f"Subject '{subj_id}': expected ({mitos_per_subject},), got {weights.shape}"
            )

    def test_mean_accuracy_is_float_in_zero_one(self):
        adata = _make_mil_anndata(n_subjects=4, mitos_per_subject=20, n_features=5)
        config = _make_minimal_ml_config()
        label_encoder = LabelEncoder()

        result = run_mil_logo_cv(
            data_matrix=adata.X.copy(),
            obs_dataframe=adata.obs,
            ml_config=config,
            feature_names=list(adata.var_names),
            label_encoder=label_encoder,
        )

        assert isinstance(result["mean_accuracy"], float)
        assert 0.0 <= result["mean_accuracy"] <= 1.0

    def test_multiclass_three_classes(self):
        adata = _make_mil_anndata(
            n_subjects=6, mitos_per_subject=20, n_features=5, n_classes=3
        )
        config = _make_minimal_ml_config()
        label_encoder = LabelEncoder()

        result = run_mil_logo_cv(
            data_matrix=adata.X.copy(),
            obs_dataframe=adata.obs,
            ml_config=config,
            feature_names=list(adata.var_names),
            label_encoder=label_encoder,
        )

        assert result["mean_accuracy"] is not None
        assert len(result["class_names"]) == 3

    def test_raises_when_fewer_than_two_subjects(self):
        adata = _make_mil_anndata(n_subjects=1, mitos_per_subject=20, n_features=5)
        config = _make_minimal_ml_config()
        label_encoder = LabelEncoder()

        with pytest.raises(ValueError, match="at least 2"):
            run_mil_logo_cv(
                data_matrix=adata.X.copy(),
                obs_dataframe=adata.obs,
                ml_config=config,
                feature_names=list(adata.var_names),
                label_encoder=label_encoder,
            )

    def test_warns_for_small_subject(self):
        """UserWarning emitted when a subject has fewer mitos than mil_min_mitos."""
        n_subjects = 4
        mitos_per_subject = 20
        n_features = 5

        rng = np.random.default_rng(42)
        # 4 normal subjects + 1 tiny subject with 3 mitos
        n_obs = n_subjects * mitos_per_subject + 3
        feature_matrix = rng.standard_normal((n_obs, n_features)).astype(np.float32)
        subject_ids = np.concatenate([
            np.repeat([f"S{i:03d}" for i in range(n_subjects)], mitos_per_subject),
            ["Tiny"] * 3,
        ])
        # Tile enough repetitions to cover all mito rows, then slice to exact length.
        n_mito_rows = n_subjects * mitos_per_subject
        conditions = np.concatenate([
            np.tile(["Class0", "Class1"], n_mito_rows // 2 + 1)[:n_mito_rows],
            ["Class0"] * 3,
        ])
        obs_df = pd.DataFrame(
            {"unique_subject_ID": subject_ids, "condition": conditions},
            index=[f"obs_{i}" for i in range(n_obs)],
        )
        adata = anndata.AnnData(
            X=feature_matrix,
            obs=obs_df,
            var=pd.DataFrame(index=[f"feature_{i}" for i in range(n_features)]),
        )
        config = _make_minimal_ml_config()
        config["mil_min_mitos"] = 5  # Tiny subject has 3 < 5 → warning

        label_encoder = LabelEncoder()
        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter("always")
            run_mil_logo_cv(
                data_matrix=adata.X.copy(),
                obs_dataframe=adata.obs,
                ml_config=config,
                feature_names=list(adata.var_names),
                label_encoder=label_encoder,
            )
        warning_messages = [str(w.message) for w in caught_warnings]
        assert any("Tiny" in msg for msg in warning_messages), (
            f"Expected a warning about 'Tiny' subject, got: {warning_messages}"
        )

    def test_logo_per_subject_scores_has_all_subjects(self):
        adata = _make_mil_anndata(n_subjects=6, mitos_per_subject=20, n_features=5)
        config = _make_minimal_ml_config()
        label_encoder = LabelEncoder()

        result = run_mil_logo_cv(
            data_matrix=adata.X.copy(),
            obs_dataframe=adata.obs,
            ml_config=config,
            feature_names=list(adata.var_names),
            label_encoder=label_encoder,
        )

        n_unique_subjects = adata.obs["unique_subject_ID"].nunique()
        assert len(result["logo_per_subject_scores"]) == n_unique_subjects

    def test_max_mitos_per_subject_subsampling(self):
        """Bags are subsampled when mil_max_mitos_per_subject is set."""
        mitos_per_subject = 50
        max_mitos = 20
        adata = _make_mil_anndata(n_subjects=4, mitos_per_subject=mitos_per_subject, n_features=5)
        config = _make_minimal_ml_config()
        config["mil_max_mitos_per_subject"] = max_mitos
        label_encoder = LabelEncoder()

        result = run_mil_logo_cv(
            data_matrix=adata.X.copy(),
            obs_dataframe=adata.obs,
            ml_config=config,
            feature_names=list(adata.var_names),
            label_encoder=label_encoder,
        )

        # Attention array shape should reflect the subsampled mito count, not the original.
        for subj_id, weights in result["mil_attention_per_subject"].items():
            assert weights.shape == (max_mitos,), (
                f"Subject '{subj_id}': expected ({max_mitos},) after subsampling, "
                f"got {weights.shape}"
            )


# ---------------------------------------------------------------------------
# Tests: get_mil_attention_dataframe
# ---------------------------------------------------------------------------


class TestGetMilAttentionDataframe:
    """Tests for the tidy DataFrame output."""

    def _run_and_get(self, n_subjects=4, mitos_per_subject=20, n_features=5):
        adata = _make_mil_anndata(
            n_subjects=n_subjects,
            mitos_per_subject=mitos_per_subject,
            n_features=n_features,
        )
        adata.uns["analysis_config"] = {"active_layer": None, "active_selection": None}
        adata.var["is_non_analytical"] = False
        config = _make_minimal_ml_config()
        label_encoder = LabelEncoder()
        results = run_mil_logo_cv(
            data_matrix=adata.X.copy(),
            obs_dataframe=adata.obs,
            ml_config=config,
            feature_names=list(adata.var_names),
            label_encoder=label_encoder,
        )
        results["feature_names"] = list(adata.var_names)
        return results, adata, config

    def test_returns_dataframe(self):
        results, adata, config = self._run_and_get()
        df = get_mil_attention_dataframe(results, adata, config)
        assert isinstance(df, pd.DataFrame)

    def test_expected_columns_present(self):
        results, adata, config = self._run_and_get(n_features=5)
        df = get_mil_attention_dataframe(results, adata, config)
        assert "unique_subject_ID" in df.columns
        assert "condition" in df.columns
        assert "mito_index" in df.columns
        assert "attention_weight" in df.columns
        for feat in adata.var_names:
            assert feat in df.columns, f"Feature column '{feat}' missing"

    def test_row_count_equals_total_mitos(self):
        n_subjects = 4
        mitos_per_subject = 20
        results, adata, config = self._run_and_get(
            n_subjects=n_subjects, mitos_per_subject=mitos_per_subject
        )
        df = get_mil_attention_dataframe(results, adata, config)
        expected_rows = n_subjects * mitos_per_subject
        assert len(df) == expected_rows, f"Expected {expected_rows} rows, got {len(df)}"

    def test_stored_in_uns(self):
        results, adata, config = self._run_and_get()
        get_mil_attention_dataframe(results, adata, config)
        assert "mil_attention" in adata.uns
        assert isinstance(adata.uns["mil_attention"], pd.DataFrame)

    def test_raises_when_no_mil_attention(self):
        adata = _make_mil_anndata()
        config = _make_minimal_ml_config()
        empty_results = {"feature_names": list(adata.var_names)}
        with pytest.raises(KeyError):
            get_mil_attention_dataframe(empty_results, adata, config)

    def test_attention_weight_values_are_non_negative(self):
        results, adata, config = self._run_and_get()
        df = get_mil_attention_dataframe(results, adata, config)
        assert (df["attention_weight"] >= 0).all()


# ---------------------------------------------------------------------------
# Integration tests: run_ml_analysis with strategy="MIL"
# ---------------------------------------------------------------------------


class TestRunMLAnalysisMIL:
    """End-to-end tests using run_ml_analysis() with strategy='MIL'."""

    def _build_adata(self, n_subjects=6, mitos_per_subject=20, n_features=8):
        adata = _make_mil_anndata(
            n_subjects=n_subjects,
            mitos_per_subject=mitos_per_subject,
            n_features=n_features,
            n_classes=2,
        )
        adata.uns["analysis_config"] = {"active_layer": None, "active_selection": None}
        adata.var["is_non_analytical"] = False
        return adata

    def test_run_ml_analysis_mil_completes(self):
        from mito_marker.analysis.ml_pipeline import run_ml_analysis

        adata = self._build_adata()
        config = _make_minimal_ml_config()
        results = run_ml_analysis(adata, ml_config=config)
        assert isinstance(results, dict)

    def test_results_stored_in_uns(self):
        from mito_marker.analysis.ml_pipeline import run_ml_analysis

        adata = self._build_adata()
        config = _make_minimal_ml_config()
        run_ml_analysis(adata, ml_config=config)
        assert "ml_results" in adata.uns

    def test_mil_attention_per_subject_in_results(self):
        from mito_marker.analysis.ml_pipeline import run_ml_analysis

        adata = self._build_adata()
        config = _make_minimal_ml_config()
        results = run_ml_analysis(adata, ml_config=config)
        assert "mil_attention_per_subject" in results
        assert isinstance(results["mil_attention_per_subject"], dict)

    def test_n_samples_equals_n_subjects(self):
        from mito_marker.analysis.ml_pipeline import run_ml_analysis

        n_subjects = 6
        adata = self._build_adata(n_subjects=n_subjects)
        config = _make_minimal_ml_config()
        results = run_ml_analysis(adata, ml_config=config)
        assert results["n_samples"] == n_subjects

    def test_standard_cv_with_mil_emits_warning(self):
        from mito_marker.analysis.ml_pipeline import run_ml_analysis

        adata = self._build_adata()
        config = _make_minimal_ml_config(evaluation_strategy="StandardCV")

        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter("always")
            run_ml_analysis(adata, ml_config=config)

        warning_messages = [str(w.message) for w in caught_warnings]
        assert any("MIL" in msg or "LOGO" in msg for msg in warning_messages), (
            f"Expected a warning about MIL + StandardCV, got: {warning_messages}"
        )

    def test_shap_disabled_for_mil(self):
        from mito_marker.analysis.ml_pipeline import run_ml_analysis

        adata = self._build_adata()
        config = _make_minimal_ml_config()
        config["compute_shap"] = True  # Attempt to enable SHAP

        results = run_ml_analysis(adata, ml_config=config)
        # SHAP is not applicable to MIL — should be None
        assert results.get("shap_values") is None


# ---------------------------------------------------------------------------
# Tests: plot_mil_attention (smoke tests)
# ---------------------------------------------------------------------------


class TestPlotMilAttention:
    """Smoke tests — verify the visualization function does not crash."""

    def _run_and_get(self, n_subjects=4, mitos_per_subject=20, n_features=5):
        adata = _make_mil_anndata(
            n_subjects=n_subjects,
            mitos_per_subject=mitos_per_subject,
            n_features=n_features,
        )
        adata.uns["analysis_config"] = {"active_layer": None, "active_selection": None}
        adata.var["is_non_analytical"] = False
        config = _make_minimal_ml_config()
        label_encoder = LabelEncoder()
        results = run_mil_logo_cv(
            data_matrix=adata.X.copy(),
            obs_dataframe=adata.obs,
            ml_config=config,
            feature_names=list(adata.var_names),
            label_encoder=label_encoder,
        )
        results["feature_names"] = list(adata.var_names)
        return results, adata, config

    def test_plot_returns_list_of_figures(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        results, adata, config = self._run_and_get()
        figures = plot_mil_attention(results, adata, config)
        plt.close("all")
        assert isinstance(figures, list)
        assert len(figures) >= 1

    def test_raises_when_no_mil_attention(self):
        adata = _make_mil_anndata()
        config = _make_minimal_ml_config()
        empty_results = {"feature_names": list(adata.var_names)}
        with pytest.raises(KeyError):
            plot_mil_attention(empty_results, adata, config)


# ---------------------------------------------------------------------------
# Tests: MIL regression support
# ---------------------------------------------------------------------------


def _make_regression_bags_dict(
    n_subjects: int = 6,
    mitos_per_subject: int = 20,
    n_features: int = 5,
    seed: int = 3,
) -> dict:
    """
    Build a bags_dict with continuous float labels (simulating age).

    Returns:
        {subject_id: (float32_matrix, label_float)}
        label_float is a simulated age value (20, 30, 40, ...).
    """
    rng = np.random.default_rng(seed)
    bags_dict: dict = {}
    for i in range(n_subjects):
        subj_id = f"R{i:03d}"
        matrix = rng.standard_normal((mitos_per_subject, n_features)).astype(np.float32)
        age = float(20 + i * 10)
        bags_dict[subj_id] = (matrix, age)
    return bags_dict


def _make_regression_mil_anndata(
    n_subjects: int = 6,
    mitos_per_subject: int = 20,
    n_features: int = 5,
    seed: int = 3,
) -> anndata.AnnData:
    """
    Build a minimal AnnData for MIL regression tests.

    The "age" obs column contains continuous float values (simulated ages).
    """
    rng = np.random.default_rng(seed)
    subject_ids = [f"R{i:03d}" for i in range(n_subjects)]
    ages = [float(20 + i * 10) for i in range(n_subjects)]

    n_obs = n_subjects * mitos_per_subject
    feature_matrix = rng.standard_normal((n_obs, n_features)).astype(np.float32)
    subject_id_col = np.repeat(subject_ids, mitos_per_subject)
    age_col = np.repeat(ages, mitos_per_subject)

    obs_df = pd.DataFrame(
        {
            "unique_subject_ID": subject_id_col,
            "age":               age_col,
        },
        index=[f"obs_{i}" for i in range(n_obs)],
    )
    var_df = pd.DataFrame(index=[f"feature_{i}" for i in range(n_features)])
    return anndata.AnnData(X=feature_matrix, obs=obs_df, var=var_df)


def _make_regression_ml_config(
    mil_max_epochs: int = 5,
    mil_patience: int = 3,
) -> dict:
    """Build a minimal ml_config for regression MIL tests."""
    config = _make_minimal_ml_config(
        mil_max_epochs=mil_max_epochs,
        mil_patience=mil_patience,
    )
    config["task_type"] = "regression"
    config["target_obs_column"] = "age"
    return config


class TestMILRegression:
    """Tests for MIL regression support (task_type='regression')."""

    def test_train_mil_model_regression_runs(self):
        """train_mil_model completes without error for regression bags."""
        bags_dict = _make_regression_bags_dict()
        config = _make_regression_ml_config()
        model = train_mil_model(bags_dict, config, label_encoder=None, random_state=0)
        assert model is not None

    def test_train_mil_model_regression_returns_eval_mode(self):
        """Model returned by train_mil_model is in eval mode."""
        bags_dict = _make_regression_bags_dict()
        config = _make_regression_ml_config()
        model = train_mil_model(bags_dict, config, label_encoder=None, random_state=0)
        assert not model.training

    def test_train_mil_model_regression_output_shape(self):
        """Regression model produces logits of shape (1, 1)."""
        bags_dict = _make_regression_bags_dict(n_features=5)
        config = _make_regression_ml_config()
        model = train_mil_model(bags_dict, config, label_encoder=None, random_state=0)
        test_bag = torch.from_numpy(
            np.random.default_rng(99).standard_normal((15, 5)).astype(np.float32)
        )
        with torch.no_grad():
            logits, _ = model(test_bag)
        assert logits.shape == (1, 1), f"Expected (1, 1), got {logits.shape}"

    def test_run_mil_logo_cv_regression_schema(self):
        """run_mil_logo_cv returns a dict with all required keys for regression."""
        adata = _make_regression_mil_anndata()
        config = _make_regression_ml_config()

        result = run_mil_logo_cv(
            data_matrix=adata.X.copy(),
            obs_dataframe=adata.obs,
            ml_config=config,
            feature_names=list(adata.var_names),
            label_encoder=None,
        )

        required_keys = [
            "r2_score", "mae", "rmse",
            "cv_scores_r2", "mean_r2", "std_r2",
            "mil_attention_per_subject",
            "logo_per_subject_scores",
            "classification_report",
            "confusion_matrix",
        ]
        for key in required_keys:
            assert key in result, f"Missing key: '{key}'"

    def test_regression_metrics_are_numeric(self):
        """r2_score, mae, and rmse are floats (not None) for regression."""
        adata = _make_regression_mil_anndata()
        config = _make_regression_ml_config()

        result = run_mil_logo_cv(
            data_matrix=adata.X.copy(),
            obs_dataframe=adata.obs,
            ml_config=config,
            feature_names=list(adata.var_names),
            label_encoder=None,
        )

        assert isinstance(result["r2_score"], float), "r2_score should be a float"
        assert isinstance(result["mae"], float), "mae should be a float"
        assert isinstance(result["rmse"], float), "rmse should be a float"
        assert result["rmse"] >= 0.0, "RMSE must be non-negative"
        assert result["mae"] >= 0.0, "MAE must be non-negative"

    def test_classification_fields_are_none_for_regression(self):
        """Classification-only fields are None when task_type='regression'."""
        adata = _make_regression_mil_anndata()
        config = _make_regression_ml_config()

        result = run_mil_logo_cv(
            data_matrix=adata.X.copy(),
            obs_dataframe=adata.obs,
            ml_config=config,
            feature_names=list(adata.var_names),
            label_encoder=None,
        )

        assert result["classification_report"] is None
        assert result["confusion_matrix"] is None
        assert result["mean_accuracy"] is None
        assert result["class_names"] is None

    def test_mil_attention_populated_for_regression(self):
        """mil_attention_per_subject is non-empty and arrays sum to 1 for regression."""
        adata = _make_regression_mil_anndata(n_subjects=4, mitos_per_subject=20)
        config = _make_regression_ml_config()

        result = run_mil_logo_cv(
            data_matrix=adata.X.copy(),
            obs_dataframe=adata.obs,
            ml_config=config,
            feature_names=list(adata.var_names),
            label_encoder=None,
        )

        attention_dict = result["mil_attention_per_subject"]
        assert isinstance(attention_dict, dict)
        assert len(attention_dict) == 4
        for subj_id, weights in attention_dict.items():
            total = float(weights.sum())
            assert abs(total - 1.0) < 1e-4, (
                f"Subject '{subj_id}': attention sum should be 1, got {total}"
            )

    def test_cv_scores_r2_is_list_of_floats(self):
        """cv_scores_r2 is a list of per-subject absolute errors."""
        n_subjects = 6
        adata = _make_regression_mil_anndata(n_subjects=n_subjects)
        config = _make_regression_ml_config()

        result = run_mil_logo_cv(
            data_matrix=adata.X.copy(),
            obs_dataframe=adata.obs,
            ml_config=config,
            feature_names=list(adata.var_names),
            label_encoder=None,
        )

        cv_scores = result["cv_scores_r2"]
        assert isinstance(cv_scores, list), "cv_scores_r2 should be a list"
        assert len(cv_scores) == n_subjects, (
            f"Expected {n_subjects} per-fold scores, got {len(cv_scores)}"
        )
        for score in cv_scores:
            assert isinstance(score, float), f"Each score should be float, got {type(score)}"


# ---------------------------------------------------------------------------
# Tests: LAVO (Leave All subjects Of same group Out) for MIL
# ---------------------------------------------------------------------------


def _make_lavo_anndata(
    n_subjects_per_group: int = 2,
    groups: list = None,
    mitos_per_subject: int = 20,
    n_features: int = 6,
    seed: int = 7,
) -> anndata.AnnData:
    """
    Build an AnnData where subjects are assigned to named age groups.
    Used to test LAVO: subjects sharing the same 'age' value are left out together.
    """
    if groups is None:
        groups = ["young", "middle", "old"]

    rng = np.random.default_rng(seed)
    subject_ids = []
    conditions = []
    age_groups = []

    class_names = ["ClassA", "ClassB"]
    for group_idx, group in enumerate(groups):
        for subj_idx in range(n_subjects_per_group):
            subject_ids.append(f"{group}_S{subj_idx:02d}")
            conditions.append(class_names[group_idx % len(class_names)])
            age_groups.append(group)

    n_subjects = len(subject_ids)
    n_obs = n_subjects * mitos_per_subject
    feature_matrix = rng.standard_normal((n_obs, n_features)).astype(np.float32)
    subject_id_col = np.repeat(subject_ids, mitos_per_subject)
    condition_col = np.repeat(conditions, mitos_per_subject)
    age_col = np.repeat(age_groups, mitos_per_subject)

    obs_df = pd.DataFrame(
        {
            "unique_subject_ID": subject_id_col,
            "condition":         condition_col,
            "age":               age_col,
        },
        index=[f"obs_{i}" for i in range(n_obs)],
    )
    var_df = pd.DataFrame(index=[f"feat_{i}" for i in range(n_features)])
    return anndata.AnnData(X=feature_matrix, obs=obs_df, var=var_df)


class TestMILLAVO:
    """Tests for LAVO (Leave All subjects Of same group Out) in run_mil_logo_cv."""

    def _make_config(self, logo_group_by_column=None):
        config = _make_minimal_ml_config(mil_max_epochs=5, mil_patience=3)
        config["logo_group_by_column"] = logo_group_by_column
        config["mil_min_mitos"] = 3
        return config

    def test_lavo_returns_one_cv_score_per_group(self):
        """LAVO produces one fold (one cv_score) per unique group value, not per subject."""
        groups = ["young", "old"]
        n_subjects_per_group = 2
        adata = _make_lavo_anndata(
            n_subjects_per_group=n_subjects_per_group, groups=groups
        )
        config = self._make_config(logo_group_by_column="age")
        label_encoder = LabelEncoder()

        result = run_mil_logo_cv(
            data_matrix=adata.X.copy(),
            obs_dataframe=adata.obs,
            ml_config=config,
            feature_names=list(adata.var_names),
            label_encoder=label_encoder,
        )

        # Two groups → two folds → two cv_scores
        assert len(result["cv_scores_accuracy"]) == len(groups), (
            f"Expected {len(groups)} cv_scores (one per group), "
            f"got {len(result['cv_scores_accuracy'])}"
        )

    def test_lavo_attention_indexed_by_subject_not_group(self):
        """mil_attention_per_subject is keyed by subject_id, not group label."""
        groups = ["young", "old"]
        n_subjects_per_group = 2
        adata = _make_lavo_anndata(
            n_subjects_per_group=n_subjects_per_group, groups=groups
        )
        config = self._make_config(logo_group_by_column="age")
        label_encoder = LabelEncoder()

        result = run_mil_logo_cv(
            data_matrix=adata.X.copy(),
            obs_dataframe=adata.obs,
            ml_config=config,
            feature_names=list(adata.var_names),
            label_encoder=label_encoder,
        )

        expected_subjects = set(adata.obs["unique_subject_ID"].unique())
        actual_subjects = set(result["mil_attention_per_subject"].keys())
        assert actual_subjects == expected_subjects, (
            f"Attention keys should be subject IDs. "
            f"Expected {expected_subjects}, got {actual_subjects}"
        )

    def test_lavo_per_subject_scores_covers_all_subjects(self):
        """per_subject_scores has one entry per subject (not per group)."""
        groups = ["young", "middle", "old"]
        adata = _make_lavo_anndata(
            n_subjects_per_group=2, groups=groups
        )
        config = self._make_config(logo_group_by_column="age")
        label_encoder = LabelEncoder()

        result = run_mil_logo_cv(
            data_matrix=adata.X.copy(),
            obs_dataframe=adata.obs,
            ml_config=config,
            feature_names=list(adata.var_names),
            label_encoder=label_encoder,
        )

        n_subjects = len(adata.obs["unique_subject_ID"].unique())
        assert len(result["logo_per_subject_scores"]) == n_subjects

    def test_lavo_attention_sums_to_one_per_subject(self):
        """Each subject's attention weights sum to 1 (softmax property)."""
        groups = ["young", "old"]
        adata = _make_lavo_anndata(
            n_subjects_per_group=3, groups=groups
        )
        config = self._make_config(logo_group_by_column="age")
        label_encoder = LabelEncoder()

        result = run_mil_logo_cv(
            data_matrix=adata.X.copy(),
            obs_dataframe=adata.obs,
            ml_config=config,
            feature_names=list(adata.var_names),
            label_encoder=label_encoder,
        )

        for subj_id, weights in result["mil_attention_per_subject"].items():
            total = float(weights.sum())
            assert abs(total - 1.0) < 1e-4, (
                f"Subject '{subj_id}': attention sum should be 1, got {total}"
            )

    def test_lavo_logo_group_by_column_stored_in_result(self):
        """logo_group_by_column is propagated to the result dict."""
        adata = _make_lavo_anndata(n_subjects_per_group=2, groups=["young", "old"])
        config = self._make_config(logo_group_by_column="age")
        label_encoder = LabelEncoder()

        result = run_mil_logo_cv(
            data_matrix=adata.X.copy(),
            obs_dataframe=adata.obs,
            ml_config=config,
            feature_names=list(adata.var_names),
            label_encoder=label_encoder,
        )

        assert result["logo_group_by_column"] == "age"

    def test_lavo_missing_column_raises_value_error(self):
        """ValueError when logo_group_by_column is not in obs_dataframe columns."""
        adata = _make_lavo_anndata(n_subjects_per_group=2, groups=["young", "old"])
        config = self._make_config(logo_group_by_column="nonexistent_column")
        label_encoder = LabelEncoder()

        with pytest.raises(ValueError, match="nonexistent_column"):
            run_mil_logo_cv(
                data_matrix=adata.X.copy(),
                obs_dataframe=adata.obs,
                ml_config=config,
                feature_names=list(adata.var_names),
                label_encoder=label_encoder,
            )

    def test_lavo_none_behaves_as_standard_logo(self):
        """With logo_group_by_column=None, one cv_score per subject (standard LOGO)."""
        n_subjects = 4
        adata = _make_mil_anndata(n_subjects=n_subjects, mitos_per_subject=20)
        config = self._make_config(logo_group_by_column=None)
        label_encoder = LabelEncoder()

        result = run_mil_logo_cv(
            data_matrix=adata.X.copy(),
            obs_dataframe=adata.obs,
            ml_config=config,
            feature_names=list(adata.var_names),
            label_encoder=label_encoder,
        )

        assert len(result["cv_scores_accuracy"]) == n_subjects, (
            "Without LAVO, should have one cv_score per subject"
        )
