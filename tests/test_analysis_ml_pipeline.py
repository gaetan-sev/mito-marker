"""
test_analysis_ml_pipeline.py

Unit tests for mito_marker.analysis.ml_pipeline.

Covers:
  _build_model():
    - Returns correct sklearn class for each (model_name, task_type) pair.
    - XGBoost returns XGBClassifier / XGBRegressor.
    - Unknown model_name raises ValueError.

  _apply_feature_name_filter():
    - None override → returns full matrix and names unchanged.
    - Subset list → correct column selection.
    - Non-existent name → raises ValueError.

  run_ml_analysis():
    - SingleMito + StandardCV + RandomForest + classification:
        results dict contains all expected keys.
        results stored in .uns["ml_results"].
        n_samples equals n_obs.
    - LOGO evaluation: logo_per_subject_scores has one key per unique subject.
    - Bags strategy: n_samples == n_subjects * bags_per_subject;
        feature_names contain "__mean" suffixes.
    - Regression task: r2_score present, classification_report absent (None).
    - compute_shap=False: shap_values is None.
    - feature_names_override bypasses get_selected_data_matrix().

  run_feature_subset_challenge():
    - Returns a DataFrame with one row per subset.
    - "All" subset (None value) runs on all features.
    - include_bags_trend_heterogeneity=True with SingleMito emits UserWarning.
    - Result stored in .uns["ml_subset_challenge_results"].

  get_sfc_feature_subsets():
    - Returns dict with "Morpho" and "Spectral" keys.
    - FSC/SSC channels go to "Morpho".
    - UV/V/B channels go to "Spectral".
"""

import warnings

import anndata
import numpy as np
import pandas as pd
import pytest
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
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.svm import SVC, SVR

from mito_marker.analysis.ml_config import get_default_ml_config
from mito_marker.analysis.ml_pipeline import (
    _apply_feature_name_filter,
    _build_fold_shap_explainer,
    _build_model,
    _normalize_shap_to_importance,
    _select_bag_features_pipeline,
    get_sfc_feature_subsets,
    run_feature_subset_challenge,
    run_ml_analysis,
)
from mito_marker.analysis.selection import _ANALYSIS_CONFIG_KEY


# ---------------------------------------------------------------------------
# Shared fixture factory
# ---------------------------------------------------------------------------


def _make_test_anndata(
    n_obs: int = 120,
    n_vars: int = 10,
    n_subjects: int = 3,
) -> anndata.AnnData:
    """
    Build a minimal AnnData for ML pipeline tests.

    3 subjects × 40 observations each (default n_obs=120).
    .obs has: subject_ID, age_group (binary), age (continuous).
    .var has: is_non_analytical=False for all features.
    """
    assert n_obs % n_subjects == 0, "n_obs must be divisible by n_subjects"
    mitos_per_subject = n_obs // n_subjects

    rng = np.random.default_rng(0)
    data_matrix = rng.random((n_obs, n_vars)).astype(np.float32)

    subjects = [f"S{i:02d}" for i in range(n_subjects)]
    subject_ids = np.repeat(subjects, mitos_per_subject)
    age_group_labels = np.where(subject_ids == subjects[0], "Young", "Old")
    # Assign distinct ages per subject for regression.
    age_values = np.where(subject_ids == subjects[0], 5.0, 20.0)

    obs_dataframe = pd.DataFrame(
        {
            "subject_ID": subject_ids,
            "age_group":  age_group_labels,
            "age":        age_values,
        },
        index=[f"obs_{i}" for i in range(n_obs)],
    )
    var_dataframe = pd.DataFrame(
        {
            "is_non_analytical": [False] * n_vars,
        },
        index=[f"feature_{i}" for i in range(n_vars)],
    )
    test_anndata = anndata.AnnData(X=data_matrix, obs=obs_dataframe, var=var_dataframe)
    # Initialise analysis_config so get_selected_data_matrix() works without error.
    test_anndata.uns[_ANALYSIS_CONFIG_KEY] = {"active_layer": None, "active_selection": None}
    return test_anndata


def _base_classification_config() -> dict:
    """Return a minimal classification config suitable for fast tests."""
    config = get_default_ml_config()
    config["task_type"] = "classification"
    config["target_obs_column"] = "age_group"
    config["subject_id_column"] = "subject_ID"
    config["strategy"] = "SingleMito"
    config["model_name"] = "RandomForest"
    config["model_params"] = {"n_estimators": 10, "max_depth": 3}
    config["evaluation_strategy"] = "StandardCV"
    config["n_folds"] = 2
    config["compute_shap"] = False
    config["random_state"] = 42
    return config


def _base_regression_config() -> dict:
    config = _base_classification_config()
    config["task_type"] = "regression"
    config["target_obs_column"] = "age"
    config["model_params"] = {"n_estimators": 10, "max_depth": 3}
    return config


# ---------------------------------------------------------------------------
# Tests: _build_model
# ---------------------------------------------------------------------------


class TestBuildModel:
    def test_randomforest_classification(self):
        model = _build_model("RandomForest", "classification", {}, 42)
        assert isinstance(model, RandomForestClassifier)

    def test_randomforest_regression(self):
        model = _build_model("RandomForest", "regression", {}, 42)
        assert isinstance(model, RandomForestRegressor)

    def test_elasticnet_classification(self):
        model = _build_model("ElasticNet", "classification", {}, 42)
        assert isinstance(model, LogisticRegression)

    def test_elasticnet_regression(self):
        model = _build_model("ElasticNet", "regression", {}, 42)
        assert isinstance(model, SklearnElasticNet)

    def test_mlp_classification(self):
        model = _build_model("MLP", "classification", {}, 42)
        assert isinstance(model, MLPClassifier)

    def test_mlp_regression(self):
        model = _build_model("MLP", "regression", {}, 42)
        assert isinstance(model, MLPRegressor)

    def test_svm_classification(self):
        model = _build_model("SVM", "classification", {}, 42)
        assert isinstance(model, SVC)

    def test_svm_regression(self):
        model = _build_model("SVM", "regression", {}, 42)
        assert isinstance(model, SVR)

    def test_xgboost_classification(self):
        xgb = pytest.importorskip("xgboost")
        model = _build_model("XGBoost", "classification", {}, 42)
        assert isinstance(model, xgb.XGBClassifier)

    def test_xgboost_regression(self):
        xgb = pytest.importorskip("xgboost")
        model = _build_model("XGBoost", "regression", {}, 42)
        assert isinstance(model, xgb.XGBRegressor)

    def test_extratrees_classification(self):
        model = _build_model("ExtraTrees", "classification", {}, 42)
        assert isinstance(model, ExtraTreesClassifier)

    def test_extratrees_regression(self):
        model = _build_model("ExtraTrees", "regression", {}, 42)
        assert isinstance(model, ExtraTreesRegressor)

    def test_gradientboosting_classification(self):
        model = _build_model("GradientBoosting", "classification", {}, 42)
        assert isinstance(model, GradientBoostingClassifier)

    def test_gradientboosting_regression(self):
        model = _build_model("GradientBoosting", "regression", {}, 42)
        assert isinstance(model, GradientBoostingRegressor)

    def test_gaussiannb_classification(self):
        model = _build_model("GaussianNB", "classification", {}, 42)
        assert isinstance(model, GaussianNB)

    def test_gaussiannb_regression_raises_value_error(self):
        with pytest.raises(ValueError, match="classification-only"):
            _build_model("GaussianNB", "regression", {}, 42)

    def test_kneighbors_classification(self):
        model = _build_model("KNeighbors", "classification", {}, 42)
        assert isinstance(model, KNeighborsClassifier)

    def test_kneighbors_regression(self):
        model = _build_model("KNeighbors", "regression", {}, 42)
        assert isinstance(model, KNeighborsRegressor)

    def test_unknown_model_raises_value_error(self):
        with pytest.raises(ValueError, match="Unknown model_name"):
            _build_model("LightGBM", "classification", {}, 42)

    def test_model_params_forwarded(self):
        model = _build_model(
            "RandomForest", "classification", {"n_estimators": 7, "max_depth": 2}, 42
        )
        assert model.n_estimators == 7
        assert model.max_depth == 2

    def test_random_state_propagated(self):
        model_42 = _build_model("RandomForest", "classification", {}, 42)
        model_99 = _build_model("RandomForest", "classification", {}, 99)
        assert model_42.random_state == 42
        assert model_99.random_state == 99


# ---------------------------------------------------------------------------
# Tests: _build_fold_shap_explainer — routes each model family to the
# explainer type that "makes the most sense" for it (exact TreeExplainer for
# every tree ensemble, exact LinearExplainer for ElasticNet, model-agnostic
# KernelExplainer as the fallback for everything without a dedicated one).
# ---------------------------------------------------------------------------


class TestBuildFoldShapExplainer:
    def _make_fit_data(self):
        rng = np.random.default_rng(0)
        X_train = rng.random((30, 4))
        y_train = np.array([0, 1] * 15)
        return X_train, y_train

    @pytest.mark.parametrize("model_name", ["RandomForest", "ExtraTrees", "GradientBoosting"])
    def test_tree_models_use_tree_explainer(self, model_name):
        shap = pytest.importorskip("shap")
        X_train, y_train = self._make_fit_data()
        model = _build_model(model_name, "classification", {"n_estimators": 5, "max_depth": 2}, 42)
        model.fit(X_train, y_train)

        explainer, explainer_type = _build_fold_shap_explainer(model, model_name, X_train, "classification")
        assert explainer_type == "TreeExplainer"
        assert isinstance(explainer, shap.TreeExplainer)

    def test_elasticnet_uses_linear_explainer(self):
        shap = pytest.importorskip("shap")
        X_train, y_train = self._make_fit_data()
        model = _build_model("ElasticNet", "classification", {}, 42)
        model.fit(X_train, y_train)

        explainer, explainer_type = _build_fold_shap_explainer(model, "ElasticNet", X_train, "classification")
        assert explainer_type == "LinearExplainer"
        assert isinstance(explainer, shap.LinearExplainer)

    @pytest.mark.parametrize("model_name", ["GaussianNB", "KNeighbors", "MLP", "SVM"])
    def test_no_dedicated_explainer_models_use_kernel_explainer(self, model_name):
        shap = pytest.importorskip("shap")
        X_train, y_train = self._make_fit_data()
        model = _build_model(model_name, "classification", {}, 42)
        model.fit(X_train, y_train)

        explainer, explainer_type = _build_fold_shap_explainer(model, model_name, X_train, "classification")
        assert explainer_type == "KernelExplainer"
        assert isinstance(explainer, shap.KernelExplainer)


# ---------------------------------------------------------------------------
# Tests: _apply_feature_name_filter
# ---------------------------------------------------------------------------


class TestApplyFeatureNameFilter:
    def setup_method(self):
        rng = np.random.default_rng(1)
        self.matrix = rng.random((10, 4))
        self.names = ["CH0", "CH1", "CH2", "CH3"]

    def test_none_override_returns_full_matrix(self):
        filtered_matrix, filtered_names = _apply_feature_name_filter(
            self.matrix, self.names, ["CH0", "CH1", "CH2", "CH3"]
        )
        assert filtered_matrix.shape == (10, 4)
        assert filtered_names == self.names

    def test_subset_returns_correct_columns(self):
        filtered_matrix, filtered_names = _apply_feature_name_filter(
            self.matrix, self.names, ["CH0", "CH2"]
        )
        assert filtered_matrix.shape == (10, 2)
        assert filtered_names == ["CH0", "CH2"]
        # Verify the actual column values match.
        np.testing.assert_array_equal(filtered_matrix[:, 0], self.matrix[:, 0])
        np.testing.assert_array_equal(filtered_matrix[:, 1], self.matrix[:, 2])

    def test_nonexistent_name_raises_value_error(self):
        with pytest.raises(ValueError, match="NONEXISTENT"):
            _apply_feature_name_filter(self.matrix, self.names, ["CH0", "NONEXISTENT"])


# ---------------------------------------------------------------------------
# Tests: run_ml_analysis — StandardCV + SingleMito
# ---------------------------------------------------------------------------


class TestRunMlAnalysisStandardCV:
    def test_results_dict_has_all_classification_keys(self):
        test_anndata = _make_test_anndata()
        config = _base_classification_config()

        results = run_ml_analysis(test_anndata, ml_config=config)

        expected_keys = [
            "ml_config", "n_samples", "n_features", "feature_names",
            "evaluation_strategy", "task_type",
            "classification_report", "confusion_matrix", "roc_auc",
            "cv_scores_accuracy", "mean_accuracy", "std_accuracy",
            "shap_values", "shap_explainer_type", "shap_feature_names",
        ]
        for key in expected_keys:
            assert key in results, f"Missing key in results: '{key}'"

    def test_results_stored_in_uns(self):
        test_anndata = _make_test_anndata()
        config = _base_classification_config()

        run_ml_analysis(test_anndata, ml_config=config)

        assert "ml_results" in test_anndata.uns

    def test_n_samples_equals_n_obs_for_singlemito(self):
        test_anndata = _make_test_anndata(n_obs=120)
        config = _base_classification_config()
        config["strategy"] = "SingleMito"

        results = run_ml_analysis(test_anndata, ml_config=config)

        assert results["n_samples"] == 120

    def test_mean_accuracy_between_zero_and_one(self):
        test_anndata = _make_test_anndata()
        config = _base_classification_config()

        results = run_ml_analysis(test_anndata, ml_config=config)

        assert 0.0 <= results["mean_accuracy"] <= 1.0

    def test_compute_shap_false_returns_none(self):
        test_anndata = _make_test_anndata()
        config = _base_classification_config()
        config["compute_shap"] = False

        results = run_ml_analysis(test_anndata, ml_config=config)

        assert results["shap_values"] is None
        assert results["shap_explainer_type"] is None
        assert results["shap_feature_names"] is None

    def test_feature_names_list_correct_length(self):
        test_anndata = _make_test_anndata(n_vars=10)
        config = _base_classification_config()

        results = run_ml_analysis(test_anndata, ml_config=config)

        assert results["n_features"] == len(results["feature_names"])


# ---------------------------------------------------------------------------
# Tests: run_ml_analysis — LOGO evaluation
# ---------------------------------------------------------------------------


class TestRunMlAnalysisLogo:
    def test_logo_per_subject_scores_keys(self):
        test_anndata = _make_test_anndata(n_obs=120, n_subjects=3)
        config = _base_classification_config()
        config["evaluation_strategy"] = "LOGO"

        results = run_ml_analysis(test_anndata, ml_config=config)

        per_subject_scores = results.get("logo_per_subject_scores")
        assert per_subject_scores is not None
        assert len(per_subject_scores) == 3  # 3 subjects

    def test_logo_cv_scores_length(self):
        test_anndata = _make_test_anndata(n_obs=120, n_subjects=3)
        config = _base_classification_config()
        config["evaluation_strategy"] = "LOGO"

        results = run_ml_analysis(test_anndata, ml_config=config)

        assert len(results["cv_scores_accuracy"]) == 3


# ---------------------------------------------------------------------------
# Tests: run_ml_analysis — Bags strategy
# ---------------------------------------------------------------------------


class TestRunMlAnalysisBags:
    def test_bags_n_samples(self):
        n_subjects = 3
        bags_per_subject = 5
        test_anndata = _make_test_anndata(n_obs=120, n_subjects=n_subjects)
        config = _base_classification_config()
        config["strategy"] = "Bags"
        config["bags_per_subject"] = bags_per_subject
        config["mitos_per_bag"] = 10
        config["bag_statistics"] = ["mean", "std"]

        results = run_ml_analysis(test_anndata, ml_config=config)

        assert results["n_samples"] == n_subjects * bags_per_subject

    def test_bags_feature_names_have_stat_suffixes(self):
        test_anndata = _make_test_anndata()
        config = _base_classification_config()
        config["strategy"] = "Bags"
        config["bags_per_subject"] = 4
        config["mitos_per_bag"] = 10
        config["bag_statistics"] = ["mean", "std"]

        results = run_ml_analysis(test_anndata, ml_config=config)

        assert any("__mean" in name for name in results["feature_names"])
        assert any("__std" in name for name in results["feature_names"])


# ---------------------------------------------------------------------------
# Tests: run_ml_analysis — regression
# ---------------------------------------------------------------------------


class TestRunMlAnalysisRegression:
    def test_regression_keys_present(self):
        test_anndata = _make_test_anndata()
        config = _base_regression_config()

        results = run_ml_analysis(test_anndata, ml_config=config)

        assert "r2_score" in results
        assert results["r2_score"] is not None
        assert "mae" in results
        assert "rmse" in results

    def test_classification_keys_absent_for_regression(self):
        test_anndata = _make_test_anndata()
        config = _base_regression_config()

        results = run_ml_analysis(test_anndata, ml_config=config)

        # Classification-specific keys must be None for regression.
        assert results.get("classification_report") is None
        assert results.get("mean_accuracy") is None


# ---------------------------------------------------------------------------
# Tests: run_ml_analysis — feature_names_override
# ---------------------------------------------------------------------------


class TestFeatureNamesOverride:
    def test_override_restricts_features(self):
        test_anndata = _make_test_anndata(n_vars=10)
        config = _base_classification_config()
        override = ["feature_0", "feature_1", "feature_2"]

        results = run_ml_analysis(
            test_anndata, ml_config=config, feature_names_override=override
        )

        assert results["n_features"] == 3
        assert results["feature_names"] == override

    def test_override_nonexistent_name_raises(self):
        test_anndata = _make_test_anndata()
        config = _base_classification_config()

        with pytest.raises(ValueError, match="NONEXISTENT"):
            run_ml_analysis(
                test_anndata,
                ml_config=config,
                feature_names_override=["feature_0", "NONEXISTENT"],
            )


# ---------------------------------------------------------------------------
# Tests: run_feature_subset_challenge
# ---------------------------------------------------------------------------


class TestRunFeatureSubsetChallenge:
    def test_returns_dataframe(self):
        test_anndata = _make_test_anndata()
        config = _base_classification_config()

        result_table = run_feature_subset_challenge(
            test_anndata,
            feature_subsets_dict={"All": None},
            ml_config=config,
        )

        assert isinstance(result_table, pd.DataFrame)

    def test_one_row_per_subset(self):
        test_anndata = _make_test_anndata(n_vars=6)
        config = _base_classification_config()
        subsets = {
            "All":     None,
            "Group_A": ["feature_0", "feature_1", "feature_2"],
            "Group_B": ["feature_3", "feature_4", "feature_5"],
        }

        result_table = run_feature_subset_challenge(
            test_anndata,
            feature_subsets_dict=subsets,
            ml_config=config,
        )

        assert len(result_table) == 3
        assert set(result_table["subset_name"]) == {"All", "Group_A", "Group_B"}

    def test_stored_in_uns(self):
        test_anndata = _make_test_anndata()
        config = _base_classification_config()

        run_feature_subset_challenge(
            test_anndata,
            feature_subsets_dict={"All": None},
            ml_config=config,
        )

        assert "ml_subset_challenge_results" in test_anndata.uns

    def test_trend_heterogeneity_singlemito_emits_warning(self):
        test_anndata = _make_test_anndata()
        config = _base_classification_config()
        config["strategy"] = "SingleMito"

        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter("always")
            run_feature_subset_challenge(
                test_anndata,
                feature_subsets_dict={"All": None},
                ml_config=config,
                include_bags_trend_heterogeneity=True,
            )

        user_warnings = [w for w in caught_warnings if issubclass(w.category, UserWarning)]
        assert len(user_warnings) >= 1

    def test_expected_columns_in_result(self):
        test_anndata = _make_test_anndata()
        config = _base_classification_config()

        result_table = run_feature_subset_challenge(
            test_anndata,
            feature_subsets_dict={"All": None},
            ml_config=config,
        )

        for column in ["subset_name", "n_features", "primary_metric", "std"]:
            assert column in result_table.columns, f"Missing column: '{column}'"


# ---------------------------------------------------------------------------
# Tests: get_sfc_feature_subsets
# ---------------------------------------------------------------------------


class TestGetSfcFeatureSubsets:
    def _make_sfc_anndata(self) -> anndata.AnnData:
        channel_names = [
            "FSC-A", "FSC-H", "FSC-W",
            "SSC-A", "SSC-B-A",
            "UV1-A", "UV2-A",
            "V1-A", "V2-A",
            "B1-A",
            "YG1-A",
            "R1-A",
            "Time",
        ]
        n_obs = 10
        data_matrix = np.ones((n_obs, len(channel_names)), dtype=np.float32)
        obs_df = pd.DataFrame(index=[f"obs_{i}" for i in range(n_obs)])
        var_df = pd.DataFrame(index=channel_names)
        return anndata.AnnData(X=data_matrix, obs=obs_df, var=var_df)

    def test_returns_dict_with_morpho_and_spectral(self):
        sfc_anndata = self._make_sfc_anndata()
        subsets = get_sfc_feature_subsets(sfc_anndata)
        assert "Morpho" in subsets
        assert "Spectral" in subsets

    def test_fsc_ssc_in_morpho(self):
        sfc_anndata = self._make_sfc_anndata()
        subsets = get_sfc_feature_subsets(sfc_anndata)
        for channel in ["FSC-A", "FSC-H", "FSC-W", "SSC-A", "SSC-B-A"]:
            assert channel in subsets["Morpho"], f"'{channel}' should be in Morpho"

    def test_uv_v_b_in_spectral(self):
        sfc_anndata = self._make_sfc_anndata()
        subsets = get_sfc_feature_subsets(sfc_anndata)
        for channel in ["UV1-A", "UV2-A", "V1-A", "V2-A", "B1-A", "YG1-A", "R1-A"]:
            assert channel in subsets["Spectral"], f"'{channel}' should be in Spectral"

    def test_time_not_in_any_subset(self):
        sfc_anndata = self._make_sfc_anndata()
        subsets = get_sfc_feature_subsets(sfc_anndata)
        assert "Time" not in subsets["Morpho"]
        assert "Time" not in subsets["Spectral"]

    def test_no_channel_in_both_subsets(self):
        sfc_anndata = self._make_sfc_anndata()
        subsets = get_sfc_feature_subsets(sfc_anndata)
        overlap = set(subsets["Morpho"]) & set(subsets["Spectral"])
        assert len(overlap) == 0, f"Channels in both subsets: {overlap}"


# ---------------------------------------------------------------------------
# Tests: LOGO-consistent SHAP
# ---------------------------------------------------------------------------


class TestLogoShap:
    """Tests for the LOGO per-fold SHAP collection and normalisation."""

    def _make_logo_anndata(self) -> anndata.AnnData:
        """4 subjects × 30 obs each, binary classification target."""
        return _make_test_anndata(n_obs=120, n_subjects=4)

    def _logo_shap_config(self) -> dict:
        config = _base_classification_config()
        config["evaluation_strategy"] = "LOGO"
        config["compute_shap"] = True
        config["model_name"] = "RandomForest"
        config["model_params"] = {"n_estimators": 5, "max_depth": 2}
        return config

    def test_logo_shap_returns_importance_list_of_arrays(self):
        """shap_values must be a list of 1D arrays, each of length n_features."""
        test_anndata = self._make_logo_anndata()
        config = self._logo_shap_config()

        results = run_ml_analysis(test_anndata, ml_config=config)

        shap_values = results["shap_values"]
        assert shap_values is not None, "shap_values should not be None when compute_shap=True"
        assert isinstance(shap_values, list), "shap_values must be a list"
        n_features = test_anndata.n_vars
        for importance_vector in shap_values:
            assert isinstance(importance_vector, np.ndarray), (
                "Each entry in shap_values must be a numpy array"
            )
            assert importance_vector.shape == (n_features,), (
                f"Expected shape ({n_features},), got {importance_vector.shape}"
            )

    def test_logo_shap_importance_sums_approx_one(self):
        """Each averaged importance vector must sum to approximately 1 (normalised)."""
        test_anndata = self._make_logo_anndata()
        config = self._logo_shap_config()

        results = run_ml_analysis(test_anndata, ml_config=config)

        shap_values = results["shap_values"]
        assert shap_values is not None
        for importance_vector in shap_values:
            total = float(importance_vector.sum())
            assert abs(total - 1.0) < 0.05, (
                f"Importance vector should sum to ~1, got {total:.4f}"
            )

    def test_logo_shap_threshold_excludes_all_folds_when_impossible(self):
        """Setting threshold > 1 must exclude every fold — shap_values is None or empty."""
        test_anndata = self._make_logo_anndata()
        config = self._logo_shap_config()
        # Threshold above 1 is impossible for any fold — all folds must be skipped.
        config["shap_logo_min_fold_score"] = 1.1

        results = run_ml_analysis(test_anndata, ml_config=config)

        # When no folds pass the threshold, logo_shap_importance is None,
        # so shap_values stored in results must also be None.
        assert results["shap_values"] is None, (
            "shap_values must be None when all folds are excluded by the threshold"
        )

    def test_normalize_shap_to_importance_2d_input(self):
        """_normalize_shap_to_importance normalises a 2D array to one vector summing to 1."""
        rng = np.random.default_rng(7)
        shap_raw = rng.random((20, 5)).astype(np.float32)   # (n_samples, n_features)

        result = _normalize_shap_to_importance(shap_raw, n_features=5)

        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0].shape == (5,)
        assert abs(result[0].sum() - 1.0) < 1e-5

    def test_normalize_shap_to_importance_list_binary(self):
        """Binary list input: uses class-1 slice and returns one normalised vector."""
        rng = np.random.default_rng(8)
        shap_raw = [rng.random((10, 4)), rng.random((10, 4))]

        result = _normalize_shap_to_importance(shap_raw, n_features=4)

        assert len(result) == 1
        assert result[0].shape == (4,)
        assert abs(result[0].sum() - 1.0) < 1e-5

    def test_normalize_shap_to_importance_list_multiclass(self):
        """Multi-class list input (3 classes): returns 3 normalised vectors."""
        rng = np.random.default_rng(9)
        shap_raw = [rng.random((10, 6)) for _ in range(3)]

        result = _normalize_shap_to_importance(shap_raw, n_features=6)

        assert len(result) == 3
        for vec in result:
            assert vec.shape == (6,)
            assert abs(vec.sum() - 1.0) < 1e-5


# ---------------------------------------------------------------------------
# Tests: _select_bag_features_pipeline
# ---------------------------------------------------------------------------


def _make_bag_matrix(
    n_bags: int = 60,
    n_features: int = 12,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, list]:
    """
    Build a synthetic bag matrix for feature selection tests.

    Returns (bag_matrix, bag_targets, feature_names).
    bag_matrix shape: (n_bags, n_features).
    bag_targets: alternating "Young" / "Old" labels.
    feature_names: ["f0__mean", "f0__std", ..., "f2__skew"] style names.
    """
    rng = np.random.default_rng(seed)
    bag_matrix = rng.random((n_bags, n_features)).astype(np.float32)
    bag_targets = np.array(["Young" if i % 2 == 0 else "Old" for i in range(n_bags)])
    n_base = n_features // 4
    stats = ["mean", "std", "median", "skew"]
    feature_names = [
        f"feat_{base}__{stat}"
        for base in range(n_base)
        for stat in stats
    ]
    return bag_matrix, bag_targets, feature_names


class TestSelectBagFeaturesPipeline:
    def test_empty_methods_returns_unchanged_matrix(self):
        bag_matrix, bag_targets, feature_names = _make_bag_matrix()

        filtered_matrix, filtered_names = _select_bag_features_pipeline(
            bag_matrix=bag_matrix,
            bag_targets=bag_targets,
            bag_feature_names=feature_names,
            methods=[],
            top_k=5,
            corr_threshold=0.95,
        )

        assert filtered_matrix.shape == bag_matrix.shape
        assert filtered_names == feature_names

    def test_high_variance_returns_top_k_features(self):
        bag_matrix, bag_targets, feature_names = _make_bag_matrix(n_features=12)

        filtered_matrix, filtered_names = _select_bag_features_pipeline(
            bag_matrix=bag_matrix,
            bag_targets=bag_targets,
            bag_feature_names=feature_names,
            methods=["HighVariance"],
            top_k=5,
            corr_threshold=0.95,
        )

        assert filtered_matrix.shape[1] == 5
        assert len(filtered_names) == 5
        assert filtered_matrix.shape[0] == bag_matrix.shape[0]

    def test_high_variance_top_k_larger_than_features_returns_all(self):
        """top_k larger than feature count → all features returned."""
        bag_matrix, bag_targets, feature_names = _make_bag_matrix(n_features=8)

        filtered_matrix, filtered_names = _select_bag_features_pipeline(
            bag_matrix=bag_matrix,
            bag_targets=bag_targets,
            bag_feature_names=feature_names,
            methods=["HighVariance"],
            top_k=100,
            corr_threshold=0.95,
        )

        assert filtered_matrix.shape[1] == 8

    def test_pca_loadings_returns_top_k_features(self):
        bag_matrix, bag_targets, feature_names = _make_bag_matrix(n_features=12)

        filtered_matrix, filtered_names = _select_bag_features_pipeline(
            bag_matrix=bag_matrix,
            bag_targets=bag_targets,
            bag_feature_names=feature_names,
            methods=["PCALoadings"],
            top_k=4,
            corr_threshold=0.95,
        )

        assert filtered_matrix.shape[1] == 4
        assert len(filtered_names) == 4

    def test_mim_classification_returns_top_k(self):
        bag_matrix, bag_targets, feature_names = _make_bag_matrix(n_features=8)

        filtered_matrix, filtered_names = _select_bag_features_pipeline(
            bag_matrix=bag_matrix,
            bag_targets=bag_targets,
            bag_feature_names=feature_names,
            methods=["MIM"],
            top_k=3,
            corr_threshold=0.95,
        )

        assert filtered_matrix.shape[1] == 3
        assert len(filtered_names) == 3

    def test_cmi_classification_returns_top_k(self):
        bag_matrix, bag_targets, feature_names = _make_bag_matrix(n_features=8)

        filtered_matrix, filtered_names = _select_bag_features_pipeline(
            bag_matrix=bag_matrix,
            bag_targets=bag_targets,
            bag_feature_names=feature_names,
            methods=["CMI"],
            top_k=3,
            corr_threshold=0.95,
        )

        assert filtered_matrix.shape[1] == 3

    def test_corr_filter_drops_perfectly_correlated_feature(self):
        """CorrFilter must remove a feature that is an exact copy of another."""
        rng = np.random.default_rng(1)
        n_bags = 60
        base_features = rng.random((n_bags, 3)).astype(np.float32)
        # Fourth column is a perfect copy of the first → correlation = 1.0.
        duplicate_column = base_features[:, 0:1]
        bag_matrix = np.hstack([base_features, duplicate_column])
        bag_targets = np.array(["A" if i % 2 == 0 else "B" for i in range(n_bags)])
        feature_names = ["f0", "f1", "f2", "f0_duplicate"]

        filtered_matrix, filtered_names = _select_bag_features_pipeline(
            bag_matrix=bag_matrix,
            bag_targets=bag_targets,
            bag_feature_names=feature_names,
            methods=["CorrFilter"],
            top_k=10,
            corr_threshold=0.95,
        )

        assert filtered_matrix.shape[1] == 3
        assert "f0_duplicate" not in filtered_names

    def test_sequential_pipeline_corr_then_high_variance(self):
        """CorrFilter runs first, HighVariance operates only on survivors."""
        rng = np.random.default_rng(2)
        n_bags = 60
        base = rng.random((n_bags, 3)).astype(np.float32)
        duplicate = base[:, 0:1]
        bag_matrix = np.hstack([base, duplicate])  # 4 columns, 1 duplicate
        bag_targets = np.array(["A" if i % 2 == 0 else "B" for i in range(n_bags)])
        feature_names = ["f0", "f1", "f2", "f0_dup"]

        # CorrFilter drops the duplicate → 3 features. HighVariance keeps top 2.
        filtered_matrix, filtered_names = _select_bag_features_pipeline(
            bag_matrix=bag_matrix,
            bag_targets=bag_targets,
            bag_feature_names=feature_names,
            methods=["CorrFilter", "HighVariance"],
            top_k=2,
            corr_threshold=0.95,
        )

        assert filtered_matrix.shape[1] == 2
        assert "f0_dup" not in filtered_names

    def test_order_matters_high_variance_then_corr(self):
        """Different method order → potentially different result (no error)."""
        bag_matrix, bag_targets, feature_names = _make_bag_matrix(n_features=12)

        result_a, names_a = _select_bag_features_pipeline(
            bag_matrix=bag_matrix,
            bag_targets=bag_targets,
            bag_feature_names=feature_names,
            methods=["CorrFilter", "HighVariance"],
            top_k=4,
            corr_threshold=0.95,
        )
        result_b, names_b = _select_bag_features_pipeline(
            bag_matrix=bag_matrix,
            bag_targets=bag_targets,
            bag_feature_names=feature_names,
            methods=["HighVariance", "CorrFilter"],
            top_k=4,
            corr_threshold=0.95,
        )

        # Both return valid matrices; order is allowed to differ.
        assert result_a.shape[0] == bag_matrix.shape[0]
        assert result_b.shape[0] == bag_matrix.shape[0]
        assert result_a.shape[1] <= 4
        assert result_b.shape[1] <= 4

    def test_returned_feature_names_are_subset_of_input(self):
        bag_matrix, bag_targets, feature_names = _make_bag_matrix(n_features=12)

        _, filtered_names = _select_bag_features_pipeline(
            bag_matrix=bag_matrix,
            bag_targets=bag_targets,
            bag_feature_names=feature_names,
            methods=["HighVariance"],
            top_k=5,
            corr_threshold=0.95,
        )

        for name in filtered_names:
            assert name in feature_names


# ---------------------------------------------------------------------------
# Tests: run_ml_analysis with post-bagging feature selection
# ---------------------------------------------------------------------------


class TestRunMlAnalysisWithBagFeatureSelection:
    def _bags_classification_config(self) -> dict:
        config = get_default_ml_config()
        config["task_type"] = "classification"
        config["target_obs_column"] = "age_group"
        config["subject_id_column"] = "subject_ID"
        config["strategy"] = "Bags"
        config["bags_per_subject"] = 5
        config["mitos_per_bag"] = 10
        config["bag_statistics"] = ["mean", "std"]
        config["model_name"] = "RandomForest"
        config["model_params"] = {"n_estimators": 5, "max_depth": 2}
        config["evaluation_strategy"] = "StandardCV"
        config["n_folds"] = 2
        config["compute_shap"] = False
        config["random_state"] = 0
        return config

    def test_no_post_bag_selection_runs_on_all_bag_features(self):
        """Empty bag_feature_selection_methods → model trained on all bag features."""
        test_anndata = _make_test_anndata(n_obs=120, n_vars=4)
        config = self._bags_classification_config()
        config["bag_feature_selection_methods"] = []

        results = run_ml_analysis(test_anndata, ml_config=config)

        # 4 features × 2 statistics = 8 bag features
        assert results["n_features"] == 8

    def test_high_variance_post_bag_selection_reduces_features(self):
        """HighVariance with top_k=3 → model trained on 3 features."""
        test_anndata = _make_test_anndata(n_obs=120, n_vars=4)
        config = self._bags_classification_config()
        config["bag_feature_selection_methods"] = ["HighVariance"]
        config["bag_feature_selection_top_k"] = 3

        results = run_ml_analysis(test_anndata, ml_config=config)

        assert results["n_features"] == 3

    def test_mim_post_bag_selection_uses_target_column(self):
        """MIM (supervised) runs without error using the configured target column."""
        test_anndata = _make_test_anndata(n_obs=120, n_vars=4)
        config = self._bags_classification_config()
        config["bag_feature_selection_methods"] = ["MIM"]
        config["bag_feature_selection_top_k"] = 4

        results = run_ml_analysis(test_anndata, ml_config=config)

        assert results["n_features"] == 4

    def test_pipeline_stored_in_uns(self):
        """Results including feature names are stored in .uns['ml_results']."""
        test_anndata = _make_test_anndata(n_obs=120, n_vars=4)
        config = self._bags_classification_config()
        config["bag_feature_selection_methods"] = ["HighVariance"]
        config["bag_feature_selection_top_k"] = 3

        run_ml_analysis(test_anndata, ml_config=config)

        assert "ml_results" in test_anndata.uns


# ---------------------------------------------------------------------------
# Shared fixture factory for LAVO tests
# ---------------------------------------------------------------------------


def _make_lavo_anndata(
    n_subjects_per_age: int = 3,
    ages: tuple = (0, 4, 8, 12),
    mitos_per_subject: int = 40,
    n_vars: int = 8,
) -> anndata.AnnData:
    """
    Build an AnnData for LAVO (Leave-All-Age-Value-Out) tests.

    Creates n_subjects_per_age subjects for each age in ages.
    .obs has: subject_ID (str), age (float — the LAVO grouping column).
    """
    n_subjects = n_subjects_per_age * len(ages)
    n_obs = n_subjects * mitos_per_subject

    rng = np.random.default_rng(99)
    data_matrix = rng.random((n_obs, n_vars)).astype(np.float32)

    subjects = [f"S{i:03d}" for i in range(n_subjects)]
    age_list = [float(a) for a in ages for _ in range(n_subjects_per_age)]
    subject_ids = np.repeat(subjects, mitos_per_subject)
    age_values = np.repeat(age_list, mitos_per_subject)

    obs_dataframe = pd.DataFrame(
        {"subject_ID": subject_ids, "age": age_values},
        index=[f"obs_{i}" for i in range(n_obs)],
    )
    var_dataframe = pd.DataFrame(
        {"is_non_analytical": [False] * n_vars},
        index=[f"feature_{i}" for i in range(n_vars)],
    )
    lavo_anndata = anndata.AnnData(X=data_matrix, obs=obs_dataframe, var=var_dataframe)
    lavo_anndata.uns[_ANALYSIS_CONFIG_KEY] = {"active_layer": None, "active_selection": None}
    return lavo_anndata


# ---------------------------------------------------------------------------
# Tests: LAVO (Leave-All-Age-Value-Out) evaluation strategy
# ---------------------------------------------------------------------------


class TestLavoEvaluation:
    """Tests for logo_group_by_column — the LAVO configuration key."""

    def _lavo_regression_config(self) -> dict:
        """Minimal regression config with LAVO enabled on the 'age' column."""
        config = get_default_ml_config()
        config["task_type"] = "regression"
        config["target_obs_column"] = "age"
        config["subject_id_column"] = "subject_ID"
        config["strategy"] = "SingleMito"
        config["model_name"] = "RandomForest"
        config["model_params"] = {"n_estimators": 5, "max_depth": 2}
        config["evaluation_strategy"] = "LOGO"
        config["logo_group_by_column"] = "age"
        config["compute_shap"] = False
        config["random_state"] = 42
        return config

    def test_lavo_n_folds_equals_n_unique_ages(self):
        """Number of folds must equal number of unique age values."""
        ages = (0, 4, 8, 12)
        lavo_anndata = _make_lavo_anndata(ages=ages)
        config = self._lavo_regression_config()

        results = run_ml_analysis(lavo_anndata, ml_config=config)

        assert len(results["cv_scores_r2"]) == len(ages)

    def test_lavo_subject_level_accumulation(self):
        """logo_subject_true must contain one entry per subject across all folds."""
        n_subjects_per_age = 3
        ages = (0, 4, 8)
        lavo_anndata = _make_lavo_anndata(n_subjects_per_age=n_subjects_per_age, ages=ages)
        config = self._lavo_regression_config()

        results = run_ml_analysis(lavo_anndata, ml_config=config)

        n_subjects_total = n_subjects_per_age * len(ages)
        assert len(results["logo_subject_true"]) == n_subjects_total
        assert len(results["logo_subject_pred"]) == n_subjects_total

    def test_lavo_none_gives_one_fold_per_subject(self):
        """logo_group_by_column=None must produce one fold per subject (standard LOGO)."""
        n_subjects = 5
        lavo_anndata = _make_lavo_anndata(n_subjects_per_age=1, ages=tuple(range(n_subjects)))
        config = self._lavo_regression_config()
        config["logo_group_by_column"] = None  # disable LAVO → standard LOGO

        results = run_ml_analysis(lavo_anndata, ml_config=config)

        assert len(results["cv_scores_r2"]) == n_subjects

    def test_lavo_invalid_column_raises_valueerror(self):
        """Non-existent logo_group_by_column must raise ValueError."""
        lavo_anndata = _make_lavo_anndata()
        config = self._lavo_regression_config()
        config["logo_group_by_column"] = "nonexistent_column"

        with pytest.raises(ValueError, match="nonexistent_column"):
            run_ml_analysis(lavo_anndata, ml_config=config)

    def test_lavo_with_bags_strategy(self):
        """LAVO must work when strategy='Bags' — fold count equals n_unique_ages."""
        ages = (0, 4, 8)
        lavo_anndata = _make_lavo_anndata(
            n_subjects_per_age=3, ages=ages, mitos_per_subject=60
        )
        config = self._lavo_regression_config()
        config["strategy"] = "Bags"
        config["bags_per_subject"] = 5
        config["mitos_per_bag"] = 10
        config["bag_statistics"] = ["mean", "std"]

        results = run_ml_analysis(lavo_anndata, ml_config=config)

        assert len(results["cv_scores_r2"]) == len(ages)

    def test_lavo_classification(self):
        """LAVO must work for classification — fold count equals n_unique_group_values.

        Uses 3 ages so each fold's training set still contains both classes
        (leaving one age out never leaves only a single class in training).
        """
        # ages 0 and 4 → "Young", age 8 → "Old"
        lavo_anndata = _make_lavo_anndata(n_subjects_per_age=3, ages=(0, 4, 8))
        lavo_anndata.obs["age_class"] = lavo_anndata.obs["age"].map(
            {0.0: "Young", 4.0: "Young", 8.0: "Old"}
        )
        config = get_default_ml_config()
        config["task_type"] = "classification"
        config["target_obs_column"] = "age_class"
        config["subject_id_column"] = "subject_ID"
        config["evaluation_strategy"] = "LOGO"
        config["logo_group_by_column"] = "age"
        config["model_name"] = "RandomForest"
        config["model_params"] = {"n_estimators": 5, "max_depth": 2}
        config["compute_shap"] = False

        results = run_ml_analysis(lavo_anndata, ml_config=config)

        assert len(results["cv_scores_accuracy"]) == 3  # 3 age groups → 3 folds

    def test_lavo_logo_group_by_column_echoed_in_results(self):
        """logo_group_by_column must be present in the returned results dict."""
        lavo_anndata = _make_lavo_anndata(ages=(0, 4))
        config = self._lavo_regression_config()

        results = run_ml_analysis(lavo_anndata, ml_config=config)

        assert results.get("logo_group_by_column") == "age"
