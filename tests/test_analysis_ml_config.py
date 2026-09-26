"""
test_analysis_ml_config.py

Unit tests for mito_marker.analysis.ml_config.

Covers:
  get_default_ml_config():
    - Returns a dict with all 15 required keys.
    - All values have the correct Python types.
    - "bag_statistics" list contains only valid entries.

  _validate_ml_config():
    - Passes silently on a valid config.
    - Raises KeyError on a missing required key.
    - Raises TypeError on a value with the wrong type.
    - Raises ValueError on an unknown task_type.
    - Raises ValueError on an unknown model_name.
    - Raises ValueError on an unknown evaluation_strategy.
    - Raises ValueError on a bag_statistics entry that is not allowed.
    - Raises ValueError when test_size is out of (0, 1) range.

  configure_ml():
    - Happy path (classification + SingleMito + RandomForest + StandardCV)
      produces a correct validated dict.
    - Regression path sets task_type="regression".
    - Bags path sets strategy="Bags" and bags_per_subject / mitos_per_bag.
    - LOGO path sets evaluation_strategy="LOGO".
    - Invalid menu choice retries until a valid one is entered.
    - model_params is always {} (not asked interactively).
    - bag_statistics is always the default list (not asked interactively).
"""

import anndata
import numpy as np
import pandas as pd
import pytest

from mito_marker.analysis.ml_bagging import ALLOWED_BAG_SAMPLING_MODES
from mito_marker.analysis.ml_config import (
    ALLOWED_BAG_STATISTICS,
    ALLOWED_EVALUATION_STRATEGIES,
    ALLOWED_MODEL_NAMES,
    ALLOWED_STRATEGIES,
    ALLOWED_TASK_TYPES,
    DEFAULT_MODEL_PARAMS,
    _validate_ml_config,
    configure_ml,
    get_default_ml_config,
)

# ---------------------------------------------------------------------------
# Shared fixture factory
# ---------------------------------------------------------------------------


def _make_test_anndata(n_obs: int = 120, n_vars: int = 10) -> anndata.AnnData:
    """
    Build a minimal AnnData with the .obs columns expected by configure_ml().
    """
    rng = np.random.default_rng(0)
    data_matrix = rng.random((n_obs, n_vars)).astype(np.float32)
    subjects = ["S01", "S02", "S03"]
    obs_dataframe = pd.DataFrame(
        {
            "subject_ID": np.tile(subjects, n_obs // len(subjects) + 1)[:n_obs],
            "age_group":  np.tile(["Young", "Old"], n_obs // 2 + 1)[:n_obs],
            "age":        rng.uniform(1, 30, n_obs),
        },
        index=[f"obs_{i}" for i in range(n_obs)],
    )
    var_dataframe = pd.DataFrame(
        index=[f"feature_{i}" for i in range(n_vars)]
    )
    return anndata.AnnData(X=data_matrix, obs=obs_dataframe, var=var_dataframe)


def _make_input_sequence(responses: list):
    """Return a mock input() callable that returns responses in order."""
    responses_iter = iter(responses)

    def mock_input(prompt: str = "") -> str:
        return next(responses_iter)

    return mock_input


# ---------------------------------------------------------------------------
# Tests: get_default_ml_config
# ---------------------------------------------------------------------------


class TestGetDefaultMlConfig:
    def test_returns_dict(self):
        config = get_default_ml_config()
        assert isinstance(config, dict)

    def test_all_required_keys_present(self):
        config = get_default_ml_config()
        required_keys = [
            "task_type", "target_obs_column", "subject_id_column",
            "strategy", "bags_per_subject", "mitos_per_bag", "bag_statistics",
            "bag_feature_selection_methods", "bag_feature_selection_top_k",
            "bag_feature_selection_corr_threshold",
            "model_name", "model_params",
            "evaluation_strategy", "test_size", "n_folds",
            "compute_shap", "shap_waterfall_sample_index",
            "random_state",
        ]
        for key in required_keys:
            assert key in config, f"Missing required key: {key}"

    def test_correct_types(self):
        config = get_default_ml_config()
        assert isinstance(config["task_type"], str)
        assert isinstance(config["target_obs_column"], str)
        assert isinstance(config["subject_id_column"], str)
        assert isinstance(config["strategy"], str)
        assert isinstance(config["bags_per_subject"], int)
        assert isinstance(config["mitos_per_bag"], int)
        assert isinstance(config["bag_statistics"], list)
        assert isinstance(config["bag_feature_selection_methods"], list)
        assert isinstance(config["bag_feature_selection_top_k"], int)
        assert isinstance(config["bag_feature_selection_corr_threshold"], float)
        assert isinstance(config["model_name"], str)
        assert isinstance(config["model_params"], dict)
        assert isinstance(config["evaluation_strategy"], str)
        assert isinstance(config["test_size"], float)
        assert isinstance(config["n_folds"], int)
        assert isinstance(config["compute_shap"], bool)
        assert isinstance(config["shap_waterfall_sample_index"], int)
        assert isinstance(config["random_state"], int)

    def test_bag_statistics_valid(self):
        config = get_default_ml_config()
        for stat in config["bag_statistics"]:
            assert stat in ALLOWED_BAG_STATISTICS

    def test_default_passes_validation(self):
        config = get_default_ml_config()
        # Should not raise.
        _validate_ml_config(config)

    def test_model_params_empty_by_default(self):
        config = get_default_ml_config()
        assert config["model_params"] == {}


# ---------------------------------------------------------------------------
# Tests: DEFAULT_MODEL_PARAMS
# ---------------------------------------------------------------------------


class TestDefaultModelParams:
    def test_all_models_have_params(self):
        for model_name in ALLOWED_MODEL_NAMES:
            assert model_name in DEFAULT_MODEL_PARAMS, (
                f"DEFAULT_MODEL_PARAMS missing entry for '{model_name}'"
            )

    def test_all_entries_are_dicts(self):
        for model_name, params in DEFAULT_MODEL_PARAMS.items():
            assert isinstance(params, dict), (
                f"DEFAULT_MODEL_PARAMS['{model_name}'] is not a dict"
            )


# ---------------------------------------------------------------------------
# Tests: _validate_ml_config
# ---------------------------------------------------------------------------


class TestValidateMlConfig:
    def test_valid_config_does_not_raise(self):
        config = get_default_ml_config()
        _validate_ml_config(config)  # no exception

    def test_missing_key_raises_key_error(self):
        config = get_default_ml_config()
        del config["task_type"]
        with pytest.raises(KeyError, match="task_type"):
            _validate_ml_config(config)

    def test_wrong_type_raises_type_error(self):
        config = get_default_ml_config()
        config["random_state"] = "not_an_int"
        with pytest.raises(TypeError):
            _validate_ml_config(config)

    def test_default_config_has_bag_sampling_keys(self):
        config = get_default_ml_config()
        assert config["bag_sampling_mode"] == "auto"
        assert config["bag_max_overlap_fraction"] == 0.2

    def test_invalid_bag_sampling_mode_raises_value_error(self):
        config = get_default_ml_config()
        config["bag_sampling_mode"] = "bootstrap"
        with pytest.raises(ValueError, match="bag_sampling_mode"):
            _validate_ml_config(config)

    def test_all_bag_sampling_modes_are_accepted(self):
        for mode_name in ALLOWED_BAG_SAMPLING_MODES:
            config = get_default_ml_config()
            config["bag_sampling_mode"] = mode_name
            _validate_ml_config(config)  # must not raise

    def test_bag_max_overlap_fraction_out_of_range_raises(self):
        config = get_default_ml_config()
        config["bag_max_overlap_fraction"] = 0.0
        with pytest.raises(ValueError, match="bag_max_overlap_fraction"):
            _validate_ml_config(config)

    def test_bag_max_overlap_fraction_wrong_type_raises(self):
        config = get_default_ml_config()
        config["bag_max_overlap_fraction"] = "0.2"
        with pytest.raises(TypeError, match="bag_max_overlap_fraction"):
            _validate_ml_config(config)

    def test_bag_sampling_keys_are_optional_for_old_configs(self):
        """Configs written before these keys existed must still validate."""
        config = get_default_ml_config()
        del config["bag_sampling_mode"]
        del config["bag_max_overlap_fraction"]
        _validate_ml_config(config)  # must not raise

    def test_invalid_task_type_raises_value_error(self):
        config = get_default_ml_config()
        config["task_type"] = "clustering"
        with pytest.raises(ValueError, match="task_type"):
            _validate_ml_config(config)

    def test_invalid_model_name_raises_value_error(self):
        config = get_default_ml_config()
        config["model_name"] = "LightGBM"
        with pytest.raises(ValueError, match="model_name"):
            _validate_ml_config(config)

    def test_invalid_evaluation_strategy_raises_value_error(self):
        config = get_default_ml_config()
        config["evaluation_strategy"] = "LeaveOneOut"
        with pytest.raises(ValueError, match="evaluation_strategy"):
            _validate_ml_config(config)

    def test_invalid_bag_statistic_raises_value_error(self):
        config = get_default_ml_config()
        config["bag_statistics"] = ["mean", "variance"]  # "variance" not allowed
        with pytest.raises(ValueError, match="variance"):
            _validate_ml_config(config)

    def test_test_size_zero_raises_value_error(self):
        config = get_default_ml_config()
        config["test_size"] = 0.0
        with pytest.raises(ValueError, match="test_size"):
            _validate_ml_config(config)

    def test_test_size_one_raises_value_error(self):
        config = get_default_ml_config()
        config["test_size"] = 1.0
        with pytest.raises(ValueError, match="test_size"):
            _validate_ml_config(config)

    def test_all_task_types_pass(self):
        for task_type in ALLOWED_TASK_TYPES:
            config = get_default_ml_config()
            config["task_type"] = task_type
            _validate_ml_config(config)

    def test_all_model_names_pass(self):
        for model_name in ALLOWED_MODEL_NAMES:
            config = get_default_ml_config()
            config["model_name"] = model_name
            _validate_ml_config(config)

    def test_all_evaluation_strategies_pass(self):
        for strategy in ALLOWED_EVALUATION_STRATEGIES:
            config = get_default_ml_config()
            config["evaluation_strategy"] = strategy
            _validate_ml_config(config)


# ---------------------------------------------------------------------------
# Tests: configure_ml (interactive console)
# ---------------------------------------------------------------------------


class TestConfigureMl:
    def test_happy_path_classification_singlemito_randomforest_standardcv(
        self, monkeypatch
    ):
        """
        Full happy path: classification, SingleMito, RandomForest, StandardCV.
        Input sequence (one response per input() call):
          [1] task_type              → "1" (classification)
          [2] target col             → age_group index
          [3] strategy               → "1" (SingleMito)
          [4] model                  → "2" (RandomForest)
          [5] eval                   → "1" (StandardCV)
          [5b] n_folds               → "" (default 5)
          [6] scale_features_in_fold → "" (default no)
          [7] SHAP                   → "" (default yes)
          [8] random_state           → "" (default 42)
        """
        test_anndata = _make_test_anndata()
        obs_columns = list(test_anndata.obs.columns)
        # Find index of "age_group"
        age_group_index = str(obs_columns.index("age_group") + 1)

        responses = [
            "1",             # task_type: classification
            age_group_index, # target column: age_group
            "1",             # strategy: SingleMito
            "2",             # model: RandomForest (index 2)
            "1",             # evaluation: StandardCV
            "",              # n_folds: default 5
            "",              # scale_features_in_fold: default no
            "",              # SHAP: default yes
            "",              # shap_per_class_plots: default (global only)
            "",              # random_state: default 42
        ]
        monkeypatch.setattr("builtins.input", _make_input_sequence(responses))

        config = configure_ml(test_anndata)

        assert config["task_type"] == "classification"
        assert config["target_obs_column"] == "age_group"
        assert config["strategy"] == "SingleMito"
        assert config["model_name"] == "RandomForest"
        assert config["evaluation_strategy"] == "StandardCV"
        assert config["n_folds"] == 5
        assert config["compute_shap"] is True
        assert config["random_state"] == 42
        assert config["model_params"] == {}

    def test_regression_path(self, monkeypatch):
        test_anndata = _make_test_anndata()
        obs_columns = list(test_anndata.obs.columns)
        age_index = str(obs_columns.index("age") + 1)

        responses = [
            "2",        # task_type: regression
            age_index,  # target: age
            "1",        # strategy: SingleMito
            "1",        # model: ElasticNet
            "1",        # evaluation: StandardCV
            "",         # n_folds: default
            "",         # scale_features_in_fold: default
            "",         # SHAP: default
            "",         # random_state: default
        ]
        monkeypatch.setattr("builtins.input", _make_input_sequence(responses))

        config = configure_ml(test_anndata)
        assert config["task_type"] == "regression"
        assert config["target_obs_column"] == "age"

    def test_bags_path_sets_bags_params(self, monkeypatch):
        test_anndata = _make_test_anndata()
        obs_columns = list(test_anndata.obs.columns)
        age_group_index = str(obs_columns.index("age_group") + 1)

        responses = [
            "1",             # classification
            age_group_index,
            "2",             # strategy: Bags
            "20",            # bags_per_subject
            "50",            # mitos_per_bag
            "1",             # sampling mode: auto
            "",              # [3a] cluster-fraction k: skip (blank)
            "",              # [3b] post-bag FS: skip (blank)
            "2",             # RandomForest
            "2",             # LOGO
            "",              # scale_features_in_fold: default
            "",              # SHAP: default yes (True)
            "",              # SHAP LOGO threshold: skip
            "",              # shap_per_class_plots: default (global only)
            "",              # random_state: default
        ]
        monkeypatch.setattr("builtins.input", _make_input_sequence(responses))

        config = configure_ml(test_anndata)
        assert config["strategy"] == "Bags"
        assert config["bags_per_subject"] == 20
        assert config["mitos_per_bag"] == 50
        assert config["evaluation_strategy"] == "LOGO"
        assert config["bag_feature_selection_methods"] == []
        assert config["bag_n_clusters"] is None
        assert config["bag_sampling_mode"] == "auto"
        assert config["bag_max_overlap_fraction"] == 0.2

    def test_logo_path(self, monkeypatch):
        test_anndata = _make_test_anndata()
        obs_columns = list(test_anndata.obs.columns)
        age_group_index = str(obs_columns.index("age_group") + 1)

        responses = [
            "1",             # classification
            age_group_index,
            "1",             # SingleMito
            "2",             # RandomForest
            "2",             # LOGO (no n_folds prompt)
            "",              # scale_features_in_fold: default
            "",              # SHAP: default yes
            "",              # SHAP LOGO threshold: skip
            "",              # shap_per_class_plots: default (global only)
            "",              # random_state
        ]
        monkeypatch.setattr("builtins.input", _make_input_sequence(responses))

        config = configure_ml(test_anndata)
        assert config["evaluation_strategy"] == "LOGO"

    def test_invalid_menu_choice_retries(self, monkeypatch):
        """An invalid choice at step [1] should prompt again until valid."""
        test_anndata = _make_test_anndata()
        obs_columns = list(test_anndata.obs.columns)
        age_group_index = str(obs_columns.index("age_group") + 1)

        responses = [
            "9",             # invalid task_type choice
            "0",             # still invalid
            "1",             # valid: classification
            age_group_index,
            "1",             # SingleMito
            "2",             # RandomForest
            "1",             # StandardCV
            "",              # n_folds
            "",              # scale_features_in_fold: default
            "",              # SHAP
            "",              # shap_per_class_plots: default (global only)
            "",              # random_state
        ]
        monkeypatch.setattr("builtins.input", _make_input_sequence(responses))

        config = configure_ml(test_anndata)
        assert config["task_type"] == "classification"

    def test_returned_config_passes_validation(self, monkeypatch):
        test_anndata = _make_test_anndata()
        obs_columns = list(test_anndata.obs.columns)
        age_group_index = str(obs_columns.index("age_group") + 1)

        responses = ["1", age_group_index, "1", "2", "1", "", "", "", "", ""]
        monkeypatch.setattr("builtins.input", _make_input_sequence(responses))

        config = configure_ml(test_anndata)
        _validate_ml_config(config)  # must not raise

    def test_model_params_always_empty(self, monkeypatch):
        """configure_ml() must never set model_params to anything other than {}."""
        test_anndata = _make_test_anndata()
        obs_columns = list(test_anndata.obs.columns)
        age_group_index = str(obs_columns.index("age_group") + 1)

        responses = ["1", age_group_index, "1", "3", "1", "", "", "", "", ""]
        monkeypatch.setattr("builtins.input", _make_input_sequence(responses))

        config = configure_ml(test_anndata)
        assert config["model_params"] == {}

    def test_bags_path_with_post_bag_feature_selection(self, monkeypatch):
        """Bags + StandardCV + HighVariance post-bag FS: methods and top_k stored."""
        test_anndata = _make_test_anndata()
        obs_columns = list(test_anndata.obs.columns)
        age_group_index = str(obs_columns.index("age_group") + 1)

        responses = [
            "1",             # classification
            age_group_index,
            "2",             # strategy: Bags
            "10",            # bags_per_subject
            "30",            # mitos_per_bag
            "1",             # sampling mode: auto
            "",              # [3a] cluster-fraction k: skip (blank)
            "4",             # [3b] post-bag FS: HighVariance (index 4)
            "15",            # top_k: 15
            "2",             # RandomForest
            "1",             # StandardCV
            "",              # n_folds: default
            "",              # scale_features_in_fold: default
            "",              # SHAP: default
            "",              # shap_per_class_plots: default (global only)
            "",              # random_state: default
        ]
        monkeypatch.setattr("builtins.input", _make_input_sequence(responses))

        config = configure_ml(test_anndata)
        assert config["bag_feature_selection_methods"] == ["HighVariance"]
        assert config["bag_feature_selection_top_k"] == 15
        assert config["strategy"] == "Bags"

    def test_bags_path_with_corr_filter_sets_threshold(self, monkeypatch):
        """Bags + CorrFilter: threshold prompt is shown and stored."""
        test_anndata = _make_test_anndata()
        obs_columns = list(test_anndata.obs.columns)
        age_group_index = str(obs_columns.index("age_group") + 1)

        responses = [
            "1",             # classification
            age_group_index,
            "2",             # strategy: Bags
            "10",            # bags_per_subject
            "30",            # mitos_per_bag
            "1",             # sampling mode: auto
            "",              # [3a] cluster-fraction k: skip (blank)
            "1",             # [3b] post-bag FS: CorrFilter (index 1)
            "20",            # top_k
            "0.90",          # corr_threshold
            "2",             # RandomForest
            "1",             # StandardCV
            "",              # n_folds: default
            "",              # scale_features_in_fold: default
            "",              # SHAP: default
            "",              # shap_per_class_plots: default (global only)
            "",              # random_state: default
        ]
        monkeypatch.setattr("builtins.input", _make_input_sequence(responses))

        config = configure_ml(test_anndata)
        assert config["bag_feature_selection_methods"] == ["CorrFilter"]
        assert config["bag_feature_selection_corr_threshold"] == 0.90


# ---------------------------------------------------------------------------
# Tests: _validate_ml_config — new bag_feature_selection_methods validation
# ---------------------------------------------------------------------------


class TestValidateBagFeatureSelection:
    def test_empty_methods_list_passes(self):
        config = get_default_ml_config()
        config["bag_feature_selection_methods"] = []
        _validate_ml_config(config)  # must not raise

    def test_all_valid_methods_pass(self):
        from mito_marker.analysis.ml_config import ALLOWED_BAG_FS_METHODS

        config = get_default_ml_config()
        config["bag_feature_selection_methods"] = list(ALLOWED_BAG_FS_METHODS)
        _validate_ml_config(config)  # must not raise

    def test_unknown_method_raises_value_error(self):
        config = get_default_ml_config()
        config["bag_feature_selection_methods"] = ["HighVariance", "FakeMethod"]
        with pytest.raises(ValueError, match="FakeMethod"):
            _validate_ml_config(config)

    def test_methods_not_a_list_raises_type_error(self):
        config = get_default_ml_config()
        config["bag_feature_selection_methods"] = "HighVariance"
        with pytest.raises(TypeError, match="bag_feature_selection_methods"):
            _validate_ml_config(config)

    def test_top_k_wrong_type_raises_type_error(self):
        config = get_default_ml_config()
        config["bag_feature_selection_top_k"] = 20.0  # float, not int
        with pytest.raises(TypeError, match="bag_feature_selection_top_k"):
            _validate_ml_config(config)

    def test_corr_threshold_wrong_type_raises_type_error(self):
        config = get_default_ml_config()
        config["bag_feature_selection_corr_threshold"] = "0.95"  # str, not float
        with pytest.raises(TypeError, match="bag_feature_selection_corr_threshold"):
            _validate_ml_config(config)


# ---------------------------------------------------------------------------
# Tests: _validate_ml_config — bag_n_clusters (cluster-fraction bagging option)
# ---------------------------------------------------------------------------


class TestBagNClustersValidation:
    def test_none_by_default_passes(self):
        config = get_default_ml_config()
        assert config["bag_n_clusters"] is None
        _validate_ml_config(config)  # must not raise

    def test_valid_int_passes(self):
        config = get_default_ml_config()
        config["strategy"] = "Bags"
        config["bag_n_clusters"] = 6
        _validate_ml_config(config)  # must not raise

    def test_wrong_type_raises_type_error(self):
        config = get_default_ml_config()
        config["bag_n_clusters"] = "6"
        with pytest.raises(TypeError, match="bag_n_clusters"):
            _validate_ml_config(config)

    def test_bool_raises_type_error(self):
        """bool is a subclass of int in Python — must be rejected explicitly."""
        config = get_default_ml_config()
        config["bag_n_clusters"] = True
        with pytest.raises(TypeError, match="bag_n_clusters"):
            _validate_ml_config(config)

    def test_less_than_two_raises_value_error(self):
        config = get_default_ml_config()
        config["bag_n_clusters"] = 1
        with pytest.raises(ValueError, match="bag_n_clusters"):
            _validate_ml_config(config)

    def test_bags_strategy_requires_statistics_or_clusters(self):
        config = get_default_ml_config()
        config["strategy"] = "Bags"
        config["bag_statistics"] = []
        config["bag_n_clusters"] = None
        with pytest.raises(ValueError, match="at least one feature source"):
            _validate_ml_config(config)

    def test_bags_strategy_with_only_clusters_passes(self):
        config = get_default_ml_config()
        config["strategy"] = "Bags"
        config["bag_statistics"] = []
        config["bag_n_clusters"] = 4
        _validate_ml_config(config)  # must not raise
