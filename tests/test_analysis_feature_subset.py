"""
test_analysis_feature_subset.py

Unit tests for mito_marker.analysis.feature_subset.

Tests verify:
  - Fraction modes keep an exact count (25%, 8%), pooled or per group.
  - Value modes respect inclusive / strict bounds.
  - Features are read from raw .X first, then from numeric .obs columns.
  - Invalid parameter combinations raise clear errors.
  - The subset history records thresholds and per-subject counts, in order,
    and survives an .h5ad round trip.
  - Subsets keep .layers / .obsm / .uns of the input (frozen reference space).
  - subset_by_obs_values() combines columns with AND and values with OR.
"""

import anndata
import numpy as np
import pandas as pd
import pytest

from mito_marker.analysis.feature_subset import (
    get_subset_history,
    resolve_subject_column,
    subset_by_feature_values,
    subset_by_obs_values,
)
from mito_marker.analysis.selection import _ANALYSIS_CONFIG_KEY


def _make_test_anndata(n_obs_per_subject: int = 100) -> anndata.AnnData:
    """
    Four subjects (2 Young, 2 Old) with Mito_Area = 1..n inside each subject,
    so every quantile is known exactly.
    """
    subjects = ["Y1", "Y2", "O1", "O2"]
    conditions = ["Young", "Young", "Old", "Old"]
    area_values = np.concatenate([
        np.arange(1, n_obs_per_subject + 1, dtype=np.float32) * (index + 1)
        for index in range(len(subjects))
    ])
    other_values = np.random.default_rng(0).normal(size=area_values.size).astype(np.float32)
    x_matrix = np.column_stack([area_values, other_values])
    obs = pd.DataFrame({
        "unique_subject_ID": np.repeat(subjects, n_obs_per_subject),
        "condition": np.repeat(conditions, n_obs_per_subject),
        "age_days": np.tile(np.arange(n_obs_per_subject, dtype=float), len(subjects)),
    })
    obs.index = [f"mito_{index}" for index in range(len(obs))]
    tem_anndata = anndata.AnnData(
        X=x_matrix, obs=obs, var=pd.DataFrame(index=["Mito_Area", "Mito_Circularity"])
    )
    tem_anndata.uns[_ANALYSIS_CONFIG_KEY] = {"active_layer": None, "active_selection": None}
    return tem_anndata


@pytest.fixture
def tem_anndata() -> anndata.AnnData:
    return _make_test_anndata()


class TestFractionModes:
    """lowest_fraction / highest_fraction."""

    def test_lowest_25_percent_pooled_exact_count(self, tem_anndata):
        subset = subset_by_feature_values(tem_anndata, "Mito_Area", "lowest_fraction", fraction=0.25)
        assert subset.n_obs == 100
        kept = subset[:, "Mito_Area"].X.ravel()
        rejected = np.setdiff1d(tem_anndata[:, "Mito_Area"].X.ravel(), kept)
        assert kept.max() <= rejected.min()

    def test_highest_8_percent_pooled_exact_count(self, tem_anndata):
        subset = subset_by_feature_values(tem_anndata, "Mito_Area", "highest_fraction", fraction=0.08)
        assert subset.n_obs == 32
        assert subset[:, "Mito_Area"].X.min() >= np.sort(tem_anndata[:, "Mito_Area"].X.ravel())[-32]

    def test_within_group_keeps_same_share_per_subject(self, tem_anndata):
        subset = subset_by_feature_values(
            tem_anndata, "Mito_Area", "lowest_fraction", fraction=0.25,
            within_group="unique_subject_ID",
        )
        counts = subset.obs["unique_subject_ID"].value_counts()
        assert (counts == 25).all() and len(counts) == 4
        # Subject Y1 has areas 1..100 -> its 25 smallest are 1..25.
        y1_values = subset[subset.obs["unique_subject_ID"] == "Y1", "Mito_Area"].X.ravel()
        np.testing.assert_array_equal(np.sort(y1_values), np.arange(1, 26))

    def test_pooled_fraction_is_dominated_by_small_subjects(self, tem_anndata):
        # Without within_group, the subject with the smallest values (Y1) takes
        # most of the pooled 25% — the reason within_group exists.
        subset = subset_by_feature_values(tem_anndata, "Mito_Area", "lowest_fraction", fraction=0.25)
        counts = subset.obs["unique_subject_ID"].value_counts()
        assert counts["Y1"] > counts.get("O2", 0)

    def test_per_group_thresholds_recorded(self, tem_anndata):
        subset = subset_by_feature_values(
            tem_anndata, "Mito_Area", "lowest_fraction", fraction=0.25,
            within_group="unique_subject_ID",
        )
        entry = get_subset_history(subset)[-1]
        assert entry["per_group_thresholds"]["Y1"] == 25.0
        assert entry["per_group_thresholds"]["Y2"] == 50.0

    def test_nan_rows_never_kept(self, tem_anndata):
        tem_anndata.X[:10, 0] = np.nan
        subset = subset_by_feature_values(tem_anndata, "Mito_Area", "lowest_fraction", fraction=0.5)
        assert not np.isnan(subset[:, "Mito_Area"].X).any()


class TestValueModes:
    """between / below / above."""

    def test_between_inclusive(self, tem_anndata):
        subset = subset_by_feature_values(
            tem_anndata, "Mito_Area", "between", lower_value=10, upper_value=20
        )
        values = subset[:, "Mito_Area"].X.ravel()
        assert values.min() == 10 and values.max() == 20

    def test_between_strict(self, tem_anndata):
        subset = subset_by_feature_values(
            tem_anndata, "Mito_Area", "between", lower_value=10, upper_value=20, inclusive=False
        )
        values = subset[:, "Mito_Area"].X.ravel()
        assert values.min() > 10 and values.max() < 20

    def test_below_and_above_partition(self, tem_anndata):
        below = subset_by_feature_values(tem_anndata, "Mito_Area", "below", upper_value=50)
        above = subset_by_feature_values(
            tem_anndata, "Mito_Area", "above", lower_value=50, inclusive=False
        )
        assert below.n_obs + above.n_obs == tem_anndata.n_obs

    def test_threshold_from_one_dataset_applied_to_another(self, tem_anndata):
        young = subset_by_obs_values(tem_anndata, {"condition": ["Young"]})
        old = subset_by_obs_values(tem_anndata, {"condition": ["Old"]})
        young_small = subset_by_feature_values(young, "Mito_Area", "lowest_fraction", fraction=0.25)
        young_threshold = get_subset_history(young_small)[-1]["upper_value_applied"]
        old_small = subset_by_feature_values(old, "Mito_Area", "below", upper_value=young_threshold)
        assert old_small[:, "Mito_Area"].X.max() <= young_threshold

    def test_obs_numeric_column(self, tem_anndata):
        subset = subset_by_feature_values(tem_anndata, "age_days", "below", upper_value=9)
        assert subset.n_obs == 40
        assert get_subset_history(subset)[-1]["feature_source"] == "obs"


class TestValidation:
    """Invalid parameters raise explicit errors."""

    @pytest.mark.parametrize("kwargs,error", [
        ({"mode": "unknown", "fraction": 0.2}, ValueError),
        ({"mode": "lowest_fraction"}, ValueError),
        ({"mode": "lowest_fraction", "fraction": 1.5}, ValueError),
        ({"mode": "lowest_fraction", "fraction": 0.0}, ValueError),
        ({"mode": "lowest_fraction", "fraction": 0.2, "upper_value": 3}, ValueError),
        ({"mode": "lowest_fraction", "fraction": 0.2, "within_group": "missing"}, KeyError),
        ({"mode": "between", "lower_value": 5}, ValueError),
        ({"mode": "between", "lower_value": 5, "upper_value": 1}, ValueError),
        ({"mode": "below", "lower_value": 5}, ValueError),
        ({"mode": "above", "upper_value": 5}, ValueError),
        ({"mode": "above", "lower_value": 5, "fraction": 0.1}, ValueError),
        ({"mode": "above", "lower_value": 5, "within_group": "condition"}, ValueError),
    ])
    def test_invalid_parameters(self, tem_anndata, kwargs, error):
        with pytest.raises(error):
            subset_by_feature_values(tem_anndata, "Mito_Area", **kwargs)

    def test_unknown_feature(self, tem_anndata):
        with pytest.raises(KeyError):
            subset_by_feature_values(tem_anndata, "Nope", "below", upper_value=1)

    def test_non_numeric_obs_column(self, tem_anndata):
        with pytest.raises(TypeError):
            subset_by_feature_values(tem_anndata, "condition", "below", upper_value=1)


class TestHistoryAndInheritance:
    """Subset history and preservation of the reference space."""

    def test_input_not_modified(self, tem_anndata):
        subset_by_feature_values(tem_anndata, "Mito_Area", "below", upper_value=10)
        assert tem_anndata.n_obs == 400
        assert get_subset_history(tem_anndata) == []

    def test_history_chained_in_order(self, tem_anndata):
        young = subset_by_obs_values(tem_anndata, {"condition": ["Young"]})
        young_small = subset_by_feature_values(young, "Mito_Area", "lowest_fraction", fraction=0.25)
        history = get_subset_history(young_small)
        assert [entry["subset_type"] for entry in history] == ["obs_values", "feature_values"]
        assert history[0]["n_obs_before"] == 400 and history[0]["n_obs_after"] == 200
        assert history[1]["n_obs_before"] == 200 and history[1]["n_obs_after"] == 50
        assert history[1]["n_obs_before_by_subject"] == {"Y1": 100, "Y2": 100}
        assert history[1]["subject_column"] == "unique_subject_ID"

    def test_layers_obsm_uns_inherited(self, tem_anndata):
        tem_anndata.layers["none__zscore_col"] = tem_anndata.X.copy()
        tem_anndata.obsm["X_pca"] = np.ones((tem_anndata.n_obs, 3), dtype=np.float32)
        tem_anndata.uns["pca_fingerprint"] = "abc123abc123"
        subset = subset_by_feature_values(tem_anndata, "Mito_Area", "below", upper_value=10)
        assert "none__zscore_col" in subset.layers
        assert subset.obsm["X_pca"].shape == (subset.n_obs, 3)
        assert subset.uns["pca_fingerprint"] == "abc123abc123"

    def test_history_survives_h5ad(self, tem_anndata, tmp_path):
        subset = subset_by_feature_values(
            tem_anndata, "Mito_Area", "lowest_fraction", fraction=0.25,
            within_group="unique_subject_ID",
        )
        subset.write_h5ad(tmp_path / "subset.h5ad")
        reloaded = anndata.read_h5ad(tmp_path / "subset.h5ad")
        entry = get_subset_history(reloaded)[-1]
        assert entry["mode"] == "lowest_fraction"
        assert entry["n_obs_after"] == 100


class TestSubsetByObsValues:
    """subset_by_obs_values()."""

    def test_or_within_column(self, tem_anndata):
        subset = subset_by_obs_values(tem_anndata, {"unique_subject_ID": ["Y1", "O1"]})
        assert subset.n_obs == 200

    def test_and_across_columns(self, tem_anndata):
        subset = subset_by_obs_values(
            tem_anndata, {"condition": ["Young"], "unique_subject_ID": ["Y1", "O1"]}
        )
        assert set(subset.obs["unique_subject_ID"]) == {"Y1"}

    def test_scalar_value_accepted(self, tem_anndata):
        subset = subset_by_obs_values(tem_anndata, {"condition": "Old"})
        assert set(subset.obs["condition"]) == {"Old"}

    def test_duplicate_obs_index_is_safe(self, tem_anndata):
        tem_anndata.obs.index = ["same"] * tem_anndata.n_obs
        subset = subset_by_obs_values(tem_anndata, {"unique_subject_ID": ["Y1"]})
        assert subset.n_obs == 100

    def test_missing_column_raises(self, tem_anndata):
        with pytest.raises(KeyError):
            subset_by_obs_values(tem_anndata, {"nope": ["x"]})

    def test_empty_filters_raise(self, tem_anndata):
        with pytest.raises(ValueError):
            subset_by_obs_values(tem_anndata, {})

    def test_resolve_subject_column_prefers_unique(self, tem_anndata):
        tem_anndata.obs["subject_ID"] = "x"
        assert resolve_subject_column(tem_anndata) == "unique_subject_ID"
