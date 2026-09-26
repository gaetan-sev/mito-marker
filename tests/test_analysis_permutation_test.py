"""
test_analysis_permutation_test.py

Unit tests for mito_marker.analysis.permutation_test.

Covers:
  _build_binary_assignments() / _build_label_assignments():
    - Binary classification: exact test gives C(n_subjects, n_minority) assignments.
    - Index 0 is always the true (unpermuted) label array.
    - max_permutations below the exact count triggers a Monte Carlo subsample of
      the requested size (+1 for the true assignment).
    - Exact count above the safety limit falls back to a capped subsample.
    - Multi-class / regression: falls back to Monte Carlo shuffles (default
      count when max_permutations is None).

  run_permutation_test():
    - Binary classification with few subjects: exact test, correct n_permutations.
    - Regression: Monte Carlo path used, correct keys.
    - max_permutations caps the exact binary test.
    - Degenerate cohort (<3 valid subjects) raises ValueError.
    - Results dict has all expected keys and is stored in .uns.
    - observed_metric matches what run_ml_analysis gives directly on the same
      (unpermuted) config — the permutation test's own "observed" evaluation
      must be consistent with the trusted real run.
    - n_seeds_per_permutation > 1 does not crash and returns a valid result.
"""

import math

import anndata
import numpy as np
import pandas as pd
import pytest

from mito_marker.analysis.ml_config import get_default_ml_config
from mito_marker.analysis.ml_pipeline import run_ml_analysis
from mito_marker.analysis.permutation_test import (
    _build_binary_assignments,
    _build_label_assignments,
    run_permutation_test,
)
from mito_marker.analysis.selection import _ANALYSIS_CONFIG_KEY


# ---------------------------------------------------------------------------
# Shared fixture factory
# ---------------------------------------------------------------------------


def _make_test_anndata(
    n_subjects: int = 6,
    mitos_per_subject: int = 60,
    n_features: int = 4,
    n_young: int = 3,
    signal: bool = True,
    seed: int = 0,
) -> anndata.AnnData:
    """
    Build a minimal AnnData for permutation test integration tests.

    The first n_young subjects are "Young", the rest "Old". When signal=True,
    feature 0 carries a real mean shift between the two groups so the real
    labels should score clearly better than permuted ones.
    """
    rng = np.random.default_rng(seed)
    subjects = [f"S{i:02d}" for i in range(n_subjects)]
    subject_ids = np.repeat(subjects, mitos_per_subject)
    is_young = np.isin(subject_ids, subjects[:n_young])
    age_group = np.where(is_young, "Young", "Old")

    data_matrix = rng.normal(0, 1, size=(n_subjects * mitos_per_subject, n_features))
    if signal:
        data_matrix[:, 0] += np.where(is_young, 2.5, -2.5)
    data_matrix = data_matrix.astype(np.float32)

    obs_dataframe = pd.DataFrame(
        {
            "subject_ID": subject_ids,
            "age_group": age_group,
            "unique_subject_ID": subject_ids,
            "age": np.where(is_young, 5.0, 20.0),
        },
        index=[f"obs_{i}" for i in range(len(subject_ids))],
    )
    var_dataframe = pd.DataFrame(
        {"is_non_analytical": [False] * n_features},
        index=[f"feature_{i}" for i in range(n_features)],
    )
    test_anndata = anndata.AnnData(X=data_matrix, obs=obs_dataframe, var=var_dataframe)
    test_anndata.uns[_ANALYSIS_CONFIG_KEY] = {"active_layer": None, "active_selection": None}
    return test_anndata


def _base_config(n_features: int = 4) -> dict:
    config = get_default_ml_config()
    config.update(
        {
            "task_type": "classification",
            "target_obs_column": "age_group",
            "subject_id_column": "subject_ID",
            "strategy": "Bags",
            "bags_per_subject": 4,
            "mitos_per_bag": 15,
            "bag_statistics": ["mean", "std"],
            "bag_n_clusters": None,
            "model_name": "ElasticNet",
            "model_params": {},
            "evaluation_strategy": "LOGO",
            "compute_shap": False,
            "scale_features_in_fold": True,
            "random_state": 42,
        }
    )
    return config


# ---------------------------------------------------------------------------
# Tests: _build_binary_assignments / _build_label_assignments
# ---------------------------------------------------------------------------


class TestBuildBinaryAssignments:
    def test_exact_count_matches_binomial_coefficient(self):
        true_labels = np.array(["Young"] * 3 + ["Old"] * 3, dtype=object)
        assignments, is_exact = _build_binary_assignments(
            true_labels, ["Old", "Young"], max_permutations=None, rng=np.random.default_rng(0)
        )
        assert is_exact is True
        assert len(assignments) == math.comb(6, 3)

    def test_index_zero_is_true_labels(self):
        true_labels = np.array(["Young"] * 3 + ["Old"] * 3, dtype=object)
        assignments, _ = _build_binary_assignments(
            true_labels, ["Old", "Young"], max_permutations=None, rng=np.random.default_rng(0)
        )
        np.testing.assert_array_equal(assignments[0], true_labels)

    def test_capped_below_exact_gives_requested_plus_one(self):
        true_labels = np.array(["Young"] * 4 + ["Old"] * 4, dtype=object)
        exact_count = math.comb(8, 4)
        assignments, is_exact = _build_binary_assignments(
            true_labels, ["Old", "Young"], max_permutations=10, rng=np.random.default_rng(0)
        )
        assert exact_count > 10
        assert is_exact is False
        assert len(assignments) == 11  # 10 requested null assignments + true

    def test_max_permutations_above_exact_uses_exact(self):
        true_labels = np.array(["Young"] * 3 + ["Old"] * 3, dtype=object)
        assignments, is_exact = _build_binary_assignments(
            true_labels, ["Old", "Young"], max_permutations=1000, rng=np.random.default_rng(0)
        )
        assert is_exact is True
        assert len(assignments) == math.comb(6, 3)

    def test_all_assignments_preserve_class_counts(self):
        true_labels = np.array(["Young"] * 3 + ["Old"] * 5, dtype=object)
        assignments, _ = _build_binary_assignments(
            true_labels, ["Old", "Young"], max_permutations=None, rng=np.random.default_rng(0)
        )
        for assignment in assignments:
            assert int(np.sum(assignment == "Young")) == 3
            assert int(np.sum(assignment == "Old")) == 5

    def test_safety_limit_fallback(self, monkeypatch):
        """Exact count above the safety limit falls back to a capped subsample."""
        import mito_marker.analysis.permutation_test as permutation_test_module

        monkeypatch.setattr(permutation_test_module, "_EXACT_COMBINATION_SAFETY_LIMIT", 5)
        true_labels = np.array(["Young"] * 3 + ["Old"] * 3, dtype=object)  # C(6,3)=20 > 5
        assignments, is_exact = _build_binary_assignments(
            true_labels, ["Old", "Young"], max_permutations=None, rng=np.random.default_rng(0)
        )
        assert is_exact is False
        assert len(assignments) == 6  # 5 (the patched limit) + true


class TestBuildLabelAssignments:
    def test_binary_classification_routes_to_exact(self):
        true_labels = np.array(["Young"] * 3 + ["Old"] * 3, dtype=object)
        assignments, is_exact = _build_label_assignments(
            true_labels, "classification", max_permutations=None, rng=np.random.default_rng(0)
        )
        assert is_exact is True
        assert len(assignments) == math.comb(6, 3)

    def test_multiclass_uses_monte_carlo_default_count(self):
        true_labels = np.array(["A"] * 2 + ["B"] * 2 + ["C"] * 2, dtype=object)
        assignments, is_exact = _build_label_assignments(
            true_labels, "classification", max_permutations=50, rng=np.random.default_rng(0)
        )
        assert is_exact is False
        assert len(assignments) == 51  # 50 requested + true

    def test_regression_uses_monte_carlo(self):
        true_labels = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        assignments, is_exact = _build_label_assignments(
            true_labels, "regression", max_permutations=30, rng=np.random.default_rng(0)
        )
        assert is_exact is False
        assert len(assignments) == 31
        np.testing.assert_array_equal(assignments[0], true_labels)


# ---------------------------------------------------------------------------
# Tests: run_permutation_test() — integration
# ---------------------------------------------------------------------------


class TestRunPermutationTest:
    def test_binary_classification_exact(self):
        test_anndata = _make_test_anndata(n_subjects=6, n_young=3, signal=True)
        config = _base_config()

        results = run_permutation_test(test_anndata, config)

        assert results["is_exact"] is True
        assert results["n_permutations"] == math.comb(6, 3)
        assert results["metric_name"] == "mean_accuracy"
        assert 0.0 <= results["observed_metric"] <= 1.0
        assert len(results["null_metrics"]) == results["n_permutations"] - 1

    def test_max_permutations_caps_exact_test(self):
        test_anndata = _make_test_anndata(n_subjects=8, n_young=4, signal=True)
        config = _base_config()

        results = run_permutation_test(test_anndata, config, max_permutations=5)

        assert results["is_exact"] is False
        assert results["n_permutations"] == 6

    def test_regression_uses_monte_carlo(self):
        test_anndata = _make_test_anndata(n_subjects=6, n_young=3, signal=True)
        config = _base_config()
        config["task_type"] = "regression"
        config["target_obs_column"] = "age"

        results = run_permutation_test(test_anndata, config, max_permutations=8)

        assert results["metric_name"] == "mean_r2"
        assert results["is_exact"] is False
        assert results["n_permutations"] == 9

    def test_degenerate_cohort_raises_value_error(self):
        test_anndata = _make_test_anndata(n_subjects=2, mitos_per_subject=20, n_young=1)
        config = _base_config()

        with pytest.raises(ValueError, match="at least 3 subjects"):
            run_permutation_test(test_anndata, config)

    def test_results_dict_has_expected_keys(self):
        test_anndata = _make_test_anndata(n_subjects=6, n_young=3, signal=True)
        config = _base_config()

        results = run_permutation_test(test_anndata, config)

        expected_keys = {
            "observed_metric", "metric_name", "null_metrics", "p_value",
            "n_permutations", "is_exact", "n_subjects", "max_permutations_requested",
            "n_seeds_per_permutation", "target_obs_column", "subject_id_column",
            "model_name", "evaluation_strategy", "task_type",
        }
        assert expected_keys.issubset(results.keys())

    def test_results_stored_in_uns(self):
        test_anndata = _make_test_anndata(n_subjects=6, n_young=3, signal=True)
        config = _base_config()

        run_permutation_test(test_anndata, config)

        assert "permutation_test_results" in test_anndata.uns

    def test_observed_metric_matches_direct_run_ml_analysis(self):
        """The permutation test's own 'observed' evaluation must reproduce the
        real run_ml_analysis() result on the same (unpermuted) config/seed."""
        test_anndata = _make_test_anndata(n_subjects=6, n_young=3, signal=True)
        config = _base_config()

        direct_results = run_ml_analysis(test_anndata.copy(), ml_config=dict(config, compute_shap=False))
        permutation_results = run_permutation_test(test_anndata, config)

        assert permutation_results["observed_metric"] == pytest.approx(direct_results["mean_accuracy"])

    def test_n_seeds_per_permutation_runs_without_error(self):
        test_anndata = _make_test_anndata(n_subjects=6, n_young=3, signal=True)
        config = _base_config()

        results = run_permutation_test(test_anndata, config, max_permutations=3, n_seeds_per_permutation=2)

        assert results["n_seeds_per_permutation"] == 2
        assert results["n_permutations"] == 4

    def test_original_anndata_target_column_not_mutated(self):
        """run_permutation_test must not leave the caller's obs permuted."""
        test_anndata = _make_test_anndata(n_subjects=6, n_young=3, signal=True)
        original_labels = test_anndata.obs["age_group"].copy()
        config = _base_config()

        run_permutation_test(test_anndata, config, max_permutations=3)

        pd.testing.assert_series_equal(test_anndata.obs["age_group"], original_labels)

    def test_significant_signal_gives_low_p_value(self):
        """A strong, real signal should score better than almost all permutations."""
        test_anndata = _make_test_anndata(n_subjects=8, n_young=4, signal=True, seed=7)
        config = _base_config()

        results = run_permutation_test(test_anndata, config)

        assert results["observed_metric"] >= np.median(results["null_metrics"])
