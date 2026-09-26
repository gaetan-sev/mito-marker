"""
test_analysis_ml_bagging.py

Unit tests for mito_marker.analysis.ml_bagging.

Covers:
  _build_bag_feature_names():
    - Output length = n_features × n_statistics.
    - Feature names follow the "{feature}__{stat}" double-underscore pattern.
    - Order: all stats for feature 0 first, then all stats for feature 1, etc.

  create_bags():
    - Output shapes: n_bags_total == n_subjects * bags_per_subject.
    - Feature matrix width == n_features * n_statistics.
    - bag_feature_names has the correct length.
    - bag_subject_ids has one entry per bag with the correct subject ID.
    - Results are deterministic: two runs with the same random_state are identical.
    - Different random_state values produce different bags (non-determinism check).
    - bag_statistics=["mean"] only → correct reduced column count.
    - Classification mode: bag target = subject's class label (str).
    - Regression mode: bag target is a float.
    - Small-N subject (fewer mitos than mitos_per_bag): no error,
      warnings.warn() is emitted.
    - Returned bag_feature_names contain "__mean" suffix (when mean is requested).

  create_bags() with n_clusters (cluster-fraction option):
    - Adds n_clusters extra columns, named "cluster_{k}__fraction".
    - Each bag's cluster fractions sum to 1.
    - bag_statistics=[] + n_clusters set → cluster-fraction-only bags.
    - bag_statistics=[] + n_clusters=None → raises ValueError.
    - Same random_state → deterministic (KMeans fit + sampling both seeded).

  _build_sampling_plan() (sampling mode + overlap diagnostics):
    - "auto" falls back to replacement only for undersized subjects.
    - "with_replacement" / "without_replacement" apply to every subject.
    - "without_replacement" raises when a subject is too small.
    - Overlap fraction equals mitos_per_bag / n_rows without replacement.
    - Overlap fraction equals 1 − (1 − 1/n)**k with replacement.
    - max_disjoint_bags == n_rows // mitos_per_bag.

  create_bags() sampling QC:
    - Invalid sampling_mode / max_overlap_fraction raise ValueError.
    - Bags drawn without replacement contain only distinct rows.
    - High overlap emits a UserWarning naming the subject.
    - Low overlap emits no overlap warning.
    - verbose=False silences the printed table but keeps the warning.
    - The QC table is printed with the mode and per-subject overlap.
"""

import warnings

import anndata
import numpy as np
import pandas as pd
import pytest

from mito_marker.analysis.ml_bagging import (
    ALLOWED_BAG_SAMPLING_MODES,
    _build_bag_feature_names,
    _build_sampling_plan,
    create_bags,
)

# ---------------------------------------------------------------------------
# Shared fixture factory
# ---------------------------------------------------------------------------


def _make_bag_anndata(
    n_subjects: int = 3,
    mitos_per_subject: int = 50,
    n_features: int = 5,
    add_small_subject: bool = False,
) -> anndata.AnnData:
    """
    Build a minimal AnnData suitable for bagging tests.

    Each subject has mitos_per_subject rows. An optional 4th "small" subject
    can be added with only 5 mitos (useful for testing the small-N warning).
    """
    rng = np.random.default_rng(7)
    subjects = [f"S{i:02d}" for i in range(n_subjects)]
    n_obs = n_subjects * mitos_per_subject

    data_matrix = rng.random((n_obs, n_features)).astype(np.float32)
    subject_ids = np.repeat(subjects, mitos_per_subject)
    age_group_labels = np.where(
        subject_ids == subjects[0], "Young", "Old"
    )
    age_values = np.where(subject_ids == subjects[0], 5.0, 20.0)

    if add_small_subject:
        small_subject_matrix = rng.random((5, n_features)).astype(np.float32)
        data_matrix = np.vstack([data_matrix, small_subject_matrix])
        subject_ids = np.concatenate([subject_ids, ["SmallSubject"] * 5])
        age_group_labels = np.concatenate([age_group_labels, ["Young"] * 5])
        age_values = np.concatenate([age_values, [5.0] * 5])

    obs_dataframe = pd.DataFrame(
        {
            "subject_ID": subject_ids,
            "age_group":  age_group_labels,
            "age":        age_values,
        },
        index=[f"obs_{i}" for i in range(len(subject_ids))],
    )
    var_dataframe = pd.DataFrame(
        index=[f"feature_{i}" for i in range(n_features)]
    )
    return anndata.AnnData(X=data_matrix, obs=obs_dataframe, var=var_dataframe)


# ---------------------------------------------------------------------------
# Tests: _build_bag_feature_names
# ---------------------------------------------------------------------------


class TestBuildBagFeatureNames:
    def test_length(self):
        names = _build_bag_feature_names(["A", "B", "C"], ["mean", "std"])
        assert len(names) == 6  # 3 features × 2 stats

    def test_double_underscore_pattern(self):
        names = _build_bag_feature_names(["Mito_Area"], ["mean", "std", "median", "skew"])
        for name in names:
            assert "__" in name, f"Expected double underscore in '{name}'"

    def test_ordering_features_first(self):
        """All stats for feature 0 come before any stat for feature 1."""
        names = _build_bag_feature_names(["Mito_Area", "Mito_AR"], ["mean", "std"])
        assert names == [
            "Mito_Area__mean",
            "Mito_Area__std",
            "Mito_AR__mean",
            "Mito_AR__std",
        ]

    def test_single_stat(self):
        names = _build_bag_feature_names(["Mito_Area", "Mito_AR"], ["mean"])
        assert names == ["Mito_Area__mean", "Mito_AR__mean"]

    def test_empty_features(self):
        names = _build_bag_feature_names([], ["mean", "std"])
        assert names == []


# ---------------------------------------------------------------------------
# Tests: create_bags
# ---------------------------------------------------------------------------


class TestCreateBags:
    def test_output_shape_n_bags(self):
        test_anndata = _make_bag_anndata(n_subjects=3, mitos_per_subject=50)
        bags_per_subject = 5

        bag_matrix, bag_targets, bag_subject_ids, bag_feature_names = create_bags(
            data_matrix=test_anndata.X,
            obs_dataframe=test_anndata.obs,
            subject_id_column="subject_ID",
            bags_per_subject=bags_per_subject,
            mitos_per_bag=10,
            bag_statistics=["mean", "std"],
            target_obs_column="age_group",
            task_type="classification",
            random_state=42,
            n_jobs=1,
        )

        assert bag_matrix.shape[0] == 3 * bags_per_subject  # n_subjects * bags_per_subject
        assert len(bag_targets) == 3 * bags_per_subject
        assert len(bag_subject_ids) == 3 * bags_per_subject

    def test_output_feature_width(self):
        test_anndata = _make_bag_anndata(n_features=5)

        bag_matrix, _, _, bag_feature_names = create_bags(
            data_matrix=test_anndata.X,
            obs_dataframe=test_anndata.obs,
            subject_id_column="subject_ID",
            bags_per_subject=4,
            mitos_per_bag=10,
            bag_statistics=["mean", "std"],
            target_obs_column="age_group",
            task_type="classification",
            random_state=42,
            n_jobs=1,
        )

        assert bag_matrix.shape[1] == 5 * 2  # n_features × n_stats
        assert len(bag_feature_names) == 5 * 2

    def test_bag_feature_names_have_mean_suffix(self):
        test_anndata = _make_bag_anndata(n_features=3)

        _, _, _, bag_feature_names = create_bags(
            data_matrix=test_anndata.X,
            obs_dataframe=test_anndata.obs,
            subject_id_column="subject_ID",
            bags_per_subject=3,
            mitos_per_bag=5,
            bag_statistics=["mean"],
            target_obs_column="age_group",
            task_type="classification",
            random_state=42,
            n_jobs=1,
        )

        assert all(name.endswith("__mean") for name in bag_feature_names)

    def test_subject_ids_correct(self):
        test_anndata = _make_bag_anndata(n_subjects=2, mitos_per_subject=30)
        bags_per_subject = 4

        _, _, bag_subject_ids, _ = create_bags(
            data_matrix=test_anndata.X,
            obs_dataframe=test_anndata.obs,
            subject_id_column="subject_ID",
            bags_per_subject=bags_per_subject,
            mitos_per_bag=5,
            bag_statistics=["mean"],
            target_obs_column="age_group",
            task_type="classification",
            random_state=42,
            n_jobs=1,
        )

        unique_subjects = sorted(test_anndata.obs["subject_ID"].unique())
        for subject in unique_subjects:
            count = np.sum(bag_subject_ids == subject)
            assert count == bags_per_subject, (
                f"Subject '{subject}' should have {bags_per_subject} bags, got {count}"
            )

    def test_determinism_same_seed(self):
        test_anndata = _make_bag_anndata()

        bag_matrix_1, _, _, _ = create_bags(
            data_matrix=test_anndata.X,
            obs_dataframe=test_anndata.obs,
            subject_id_column="subject_ID",
            bags_per_subject=5,
            mitos_per_bag=10,
            bag_statistics=["mean", "std"],
            target_obs_column="age_group",
            task_type="classification",
            random_state=42,
            n_jobs=1,
        )
        bag_matrix_2, _, _, _ = create_bags(
            data_matrix=test_anndata.X,
            obs_dataframe=test_anndata.obs,
            subject_id_column="subject_ID",
            bags_per_subject=5,
            mitos_per_bag=10,
            bag_statistics=["mean", "std"],
            target_obs_column="age_group",
            task_type="classification",
            random_state=42,
            n_jobs=1,
        )

        np.testing.assert_array_equal(bag_matrix_1, bag_matrix_2)

    def test_different_seeds_produce_different_bags(self):
        test_anndata = _make_bag_anndata()

        bag_matrix_1, _, _, _ = create_bags(
            data_matrix=test_anndata.X,
            obs_dataframe=test_anndata.obs,
            subject_id_column="subject_ID",
            bags_per_subject=5,
            mitos_per_bag=10,
            bag_statistics=["mean"],
            target_obs_column="age_group",
            task_type="classification",
            random_state=42,
            n_jobs=1,
        )
        bag_matrix_2, _, _, _ = create_bags(
            data_matrix=test_anndata.X,
            obs_dataframe=test_anndata.obs,
            subject_id_column="subject_ID",
            bags_per_subject=5,
            mitos_per_bag=10,
            bag_statistics=["mean"],
            target_obs_column="age_group",
            task_type="classification",
            random_state=99,
            n_jobs=1,
        )

        assert not np.array_equal(bag_matrix_1, bag_matrix_2)

    def test_single_stat_column_count(self):
        test_anndata = _make_bag_anndata(n_features=4)

        bag_matrix, _, _, bag_feature_names = create_bags(
            data_matrix=test_anndata.X,
            obs_dataframe=test_anndata.obs,
            subject_id_column="subject_ID",
            bags_per_subject=3,
            mitos_per_bag=5,
            bag_statistics=["mean"],
            target_obs_column="age_group",
            task_type="classification",
            random_state=42,
            n_jobs=1,
        )

        assert bag_matrix.shape[1] == 4  # 4 features × 1 stat
        assert len(bag_feature_names) == 4

    def test_classification_target_is_string(self):
        test_anndata = _make_bag_anndata()

        _, bag_targets, _, _ = create_bags(
            data_matrix=test_anndata.X,
            obs_dataframe=test_anndata.obs,
            subject_id_column="subject_ID",
            bags_per_subject=3,
            mitos_per_bag=5,
            bag_statistics=["mean"],
            target_obs_column="age_group",
            task_type="classification",
            random_state=42,
            n_jobs=1,
        )

        # All targets should be strings for classification.
        for target in bag_targets:
            assert isinstance(target, str), f"Expected str, got {type(target)}: {target}"

    def test_regression_target_is_float(self):
        test_anndata = _make_bag_anndata()

        _, bag_targets, _, _ = create_bags(
            data_matrix=test_anndata.X,
            obs_dataframe=test_anndata.obs,
            subject_id_column="subject_ID",
            bags_per_subject=3,
            mitos_per_bag=5,
            bag_statistics=["mean"],
            target_obs_column="age",
            task_type="regression",
            random_state=42,
            n_jobs=1,
        )

        for target in bag_targets:
            assert isinstance(float(target), float)

    def test_small_n_subject_emits_warning(self):
        """Subject with fewer mitos than mitos_per_bag triggers a UserWarning."""
        test_anndata = _make_bag_anndata(add_small_subject=True)

        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter("always")
            create_bags(
                data_matrix=test_anndata.X,
                obs_dataframe=test_anndata.obs,
                subject_id_column="subject_ID",
                bags_per_subject=3,
                mitos_per_bag=20,  # SmallSubject has only 5 → triggers warning
                bag_statistics=["mean"],
                target_obs_column="age_group",
                task_type="classification",
                random_state=42,
                n_jobs=1,
            )

        user_warnings = [w for w in caught_warnings if issubclass(w.category, UserWarning)]
        assert len(user_warnings) >= 1, "Expected at least one UserWarning for small subject"
        warning_messages = " ".join(str(w.message) for w in user_warnings)
        assert "SmallSubject" in warning_messages or "replacement" in warning_messages

    def test_small_n_subject_no_error(self):
        """create_bags() must not raise an error for small subjects."""
        test_anndata = _make_bag_anndata(add_small_subject=True)

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            bag_matrix, _, _, _ = create_bags(
                data_matrix=test_anndata.X,
                obs_dataframe=test_anndata.obs,
                subject_id_column="subject_ID",
                bags_per_subject=3,
                mitos_per_bag=20,
                bag_statistics=["mean"],
                target_obs_column="age_group",
                task_type="classification",
                random_state=42,
                n_jobs=1,
            )

        # 3 original subjects + 1 small subject = 4, each with 3 bags.
        assert bag_matrix.shape[0] == 4 * 3

    def test_all_four_statistics(self):
        test_anndata = _make_bag_anndata(n_features=3)

        bag_matrix, _, _, bag_feature_names = create_bags(
            data_matrix=test_anndata.X,
            obs_dataframe=test_anndata.obs,
            subject_id_column="subject_ID",
            bags_per_subject=2,
            mitos_per_bag=10,
            bag_statistics=["mean", "std", "median", "skew"],
            target_obs_column="age_group",
            task_type="classification",
            random_state=42,
            n_jobs=1,
        )

        assert bag_matrix.shape[1] == 3 * 4  # n_features × 4 stats
        assert len(bag_feature_names) == 3 * 4
        for stat in ["mean", "std", "median", "skew"]:
            assert any(name.endswith(f"__{stat}") for name in bag_feature_names), (
                f"Expected '{stat}' suffix in bag feature names"
            )


# ---------------------------------------------------------------------------
# Tests: create_bags() with n_clusters (cluster-fraction option)
# ---------------------------------------------------------------------------


class TestCreateBagsClusterFractions:
    def test_adds_n_clusters_extra_columns(self):
        test_anndata = _make_bag_anndata(n_subjects=3, mitos_per_subject=50, n_features=4)

        bag_matrix, _, _, bag_feature_names = create_bags(
            data_matrix=test_anndata.X,
            obs_dataframe=test_anndata.obs,
            subject_id_column="subject_ID",
            bags_per_subject=4,
            mitos_per_bag=10,
            bag_statistics=["mean", "std"],
            target_obs_column="age_group",
            task_type="classification",
            random_state=42,
            n_jobs=1,
            n_clusters=3,
        )

        assert bag_matrix.shape[1] == 4 * 2 + 3  # (n_features x 2 stats) + 3 cluster fractions
        assert len(bag_feature_names) == 4 * 2 + 3
        cluster_names = [name for name in bag_feature_names if name.endswith("__fraction")]
        assert cluster_names == ["cluster_0__fraction", "cluster_1__fraction", "cluster_2__fraction"]

    def test_cluster_fractions_sum_to_one_per_bag(self):
        test_anndata = _make_bag_anndata(n_subjects=3, mitos_per_subject=50, n_features=4)

        bag_matrix, _, _, bag_feature_names = create_bags(
            data_matrix=test_anndata.X,
            obs_dataframe=test_anndata.obs,
            subject_id_column="subject_ID",
            bags_per_subject=4,
            mitos_per_bag=10,
            bag_statistics=["mean"],
            target_obs_column="age_group",
            task_type="classification",
            random_state=42,
            n_jobs=1,
            n_clusters=4,
        )

        cluster_column_indices = [
            index for index, name in enumerate(bag_feature_names) if name.endswith("__fraction")
        ]
        cluster_fractions = bag_matrix[:, cluster_column_indices]
        np.testing.assert_allclose(cluster_fractions.sum(axis=1), 1.0)

    def test_cluster_fractions_only_when_bag_statistics_empty(self):
        test_anndata = _make_bag_anndata(n_subjects=3, mitos_per_subject=50, n_features=4)

        bag_matrix, _, _, bag_feature_names = create_bags(
            data_matrix=test_anndata.X,
            obs_dataframe=test_anndata.obs,
            subject_id_column="subject_ID",
            bags_per_subject=3,
            mitos_per_bag=10,
            bag_statistics=[],
            target_obs_column="age_group",
            task_type="classification",
            random_state=42,
            n_jobs=1,
            n_clusters=5,
        )

        assert bag_matrix.shape[1] == 5
        assert bag_feature_names == [f"cluster_{i}__fraction" for i in range(5)]

    def test_empty_statistics_and_no_clusters_raises_value_error(self):
        test_anndata = _make_bag_anndata(n_features=4)

        with pytest.raises(ValueError, match="at least one feature source"):
            create_bags(
                data_matrix=test_anndata.X,
                obs_dataframe=test_anndata.obs,
                subject_id_column="subject_ID",
                bags_per_subject=3,
                mitos_per_bag=10,
                bag_statistics=[],
                target_obs_column="age_group",
                task_type="classification",
                random_state=42,
                n_jobs=1,
            )

    def test_determinism_same_seed_with_clusters(self):
        test_anndata = _make_bag_anndata(n_subjects=3, mitos_per_subject=50, n_features=4)

        bag_matrix_1, _, _, _ = create_bags(
            data_matrix=test_anndata.X,
            obs_dataframe=test_anndata.obs,
            subject_id_column="subject_ID",
            bags_per_subject=4,
            mitos_per_bag=10,
            bag_statistics=["mean"],
            target_obs_column="age_group",
            task_type="classification",
            random_state=42,
            n_jobs=1,
            n_clusters=3,
        )
        bag_matrix_2, _, _, _ = create_bags(
            data_matrix=test_anndata.X,
            obs_dataframe=test_anndata.obs,
            subject_id_column="subject_ID",
            bags_per_subject=4,
            mitos_per_bag=10,
            bag_statistics=["mean"],
            target_obs_column="age_group",
            task_type="classification",
            random_state=42,
            n_jobs=1,
            n_clusters=3,
        )

        np.testing.assert_array_equal(bag_matrix_1, bag_matrix_2)

    def test_n_clusters_none_gives_same_shape_as_before(self):
        """n_clusters=None (default) must not change existing behaviour."""
        test_anndata = _make_bag_anndata(n_subjects=3, mitos_per_subject=50, n_features=4)

        bag_matrix, _, _, bag_feature_names = create_bags(
            data_matrix=test_anndata.X,
            obs_dataframe=test_anndata.obs,
            subject_id_column="subject_ID",
            bags_per_subject=4,
            mitos_per_bag=10,
            bag_statistics=["mean", "std"],
            target_obs_column="age_group",
            task_type="classification",
            random_state=42,
            n_jobs=1,
        )

        assert bag_matrix.shape[1] == 4 * 2
        assert not any(name.endswith("__fraction") for name in bag_feature_names)


# ---------------------------------------------------------------------------
# Tests: _build_sampling_plan (sampling mode + overlap diagnostics)
# ---------------------------------------------------------------------------


class TestBuildSamplingPlan:
    def test_auto_uses_replacement_only_for_undersized_subjects(self):
        plan = _build_sampling_plan(
            subject_row_counts={"Big": 500, "Small": 5},
            mitos_per_bag=20,
            sampling_mode="auto",
        )

        assert plan["Big"]["use_replacement"] is False
        assert plan["Small"]["use_replacement"] is True

    def test_with_replacement_applies_to_every_subject(self):
        plan = _build_sampling_plan(
            subject_row_counts={"Big": 500, "Small": 5},
            mitos_per_bag=20,
            sampling_mode="with_replacement",
        )

        assert all(subject["use_replacement"] is True for subject in plan.values())

    def test_without_replacement_applies_to_every_subject(self):
        plan = _build_sampling_plan(
            subject_row_counts={"A": 500, "B": 300},
            mitos_per_bag=20,
            sampling_mode="without_replacement",
        )

        assert all(subject["use_replacement"] is False for subject in plan.values())

    def test_without_replacement_raises_for_undersized_subject(self):
        with pytest.raises(ValueError, match="without_replacement"):
            _build_sampling_plan(
                subject_row_counts={"Big": 500, "Small": 5},
                mitos_per_bag=20,
                sampling_mode="without_replacement",
            )

    def test_error_message_names_the_offending_subject(self):
        with pytest.raises(ValueError, match="Small"):
            _build_sampling_plan(
                subject_row_counts={"Big": 500, "Small": 5},
                mitos_per_bag=20,
                sampling_mode="without_replacement",
            )

    def test_overlap_fraction_without_replacement(self):
        """Two bags of k rows out of n share k/n of their rows on average."""
        plan = _build_sampling_plan(
            subject_row_counts={"Human_027": 268},
            mitos_per_bag=200,
            sampling_mode="auto",
        )

        assert plan["Human_027"]["overlap_fraction"] == pytest.approx(200 / 268)
        assert plan["Human_027"]["distinct_rows_per_bag"] == pytest.approx(200.0)

    def test_overlap_fraction_with_replacement(self):
        """With replacement the overlap equals the per-row inclusion probability."""
        n_rows = 268
        mitos_per_bag = 200
        expected_probability = 1.0 - (1.0 - 1.0 / n_rows) ** mitos_per_bag

        plan = _build_sampling_plan(
            subject_row_counts={"Human_027": n_rows},
            mitos_per_bag=mitos_per_bag,
            sampling_mode="with_replacement",
        )

        assert plan["Human_027"]["overlap_fraction"] == pytest.approx(expected_probability)
        assert plan["Human_027"]["distinct_rows_per_bag"] == pytest.approx(
            n_rows * expected_probability
        )
        # A bootstrap bag holds fewer distinct rows than it draws.
        assert plan["Human_027"]["distinct_rows_per_bag"] < mitos_per_bag

    def test_small_bag_relative_to_subject_gives_low_overlap(self):
        plan = _build_sampling_plan(
            subject_row_counts={"Droso_13": 1154},
            mitos_per_bag=50,
            sampling_mode="auto",
        )

        assert plan["Droso_13"]["overlap_fraction"] < 0.05

    def test_max_disjoint_bags(self):
        plan = _build_sampling_plan(
            subject_row_counts={"KFish_565": 227, "Droso_13": 1154},
            mitos_per_bag=200,
            sampling_mode="auto",
        )

        assert plan["KFish_565"]["max_disjoint_bags"] == 1
        assert plan["Droso_13"]["max_disjoint_bags"] == 5

    def test_overlap_fraction_never_exceeds_one(self):
        plan = _build_sampling_plan(
            subject_row_counts={"Tiny": 3},
            mitos_per_bag=200,
            sampling_mode="auto",
        )

        assert 0.0 <= plan["Tiny"]["overlap_fraction"] <= 1.0


# ---------------------------------------------------------------------------
# Tests: create_bags() sampling mode option and QC output
# ---------------------------------------------------------------------------


class TestCreateBagsSamplingMode:
    def _create(self, test_anndata, **overrides):
        """Call create_bags() with the shared defaults used across these tests."""
        call_kwargs = dict(
            data_matrix=test_anndata.X,
            obs_dataframe=test_anndata.obs,
            subject_id_column="subject_ID",
            bags_per_subject=3,
            mitos_per_bag=10,
            bag_statistics=["mean"],
            target_obs_column="age_group",
            task_type="classification",
            random_state=42,
            n_jobs=1,
        )
        call_kwargs.update(overrides)
        return create_bags(**call_kwargs)

    def test_all_documented_modes_are_accepted(self):
        test_anndata = _make_bag_anndata()

        for mode_name in ALLOWED_BAG_SAMPLING_MODES:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                bag_matrix, _, _, _ = self._create(test_anndata, sampling_mode=mode_name)
            assert bag_matrix.shape[0] == 3 * 3

    def test_unknown_sampling_mode_raises(self):
        test_anndata = _make_bag_anndata()

        with pytest.raises(ValueError, match="sampling_mode"):
            self._create(test_anndata, sampling_mode="bootstrap")

    def test_out_of_range_max_overlap_fraction_raises(self):
        test_anndata = _make_bag_anndata()

        with pytest.raises(ValueError, match="max_overlap_fraction"):
            self._create(test_anndata, max_overlap_fraction=1.5)

    def test_without_replacement_raises_for_small_subject(self):
        test_anndata = _make_bag_anndata(add_small_subject=True)

        with pytest.raises(ValueError, match="SmallSubject"):
            self._create(
                test_anndata,
                mitos_per_bag=20,
                sampling_mode="without_replacement",
            )

    def test_without_replacement_draws_distinct_rows(self):
        """A bag's mean of distinct rows can never equal a single row's value twice over."""
        # One subject, one feature holding a distinct power of two per row: the sum of a
        # sampled bag then identifies exactly which rows were drawn.
        n_rows = 12
        data_matrix = (2.0 ** np.arange(n_rows)).reshape(-1, 1)
        obs_dataframe = pd.DataFrame(
            {"subject_ID": ["S0"] * n_rows, "age_group": ["Young"] * n_rows},
            index=[f"obs_{i}" for i in range(n_rows)],
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            bag_matrix, _, _, _ = create_bags(
                data_matrix=data_matrix,
                obs_dataframe=obs_dataframe,
                subject_id_column="subject_ID",
                bags_per_subject=20,
                mitos_per_bag=4,
                bag_statistics=["mean"],
                target_obs_column="age_group",
                task_type="classification",
                random_state=42,
                n_jobs=1,
                sampling_mode="without_replacement",
            )

        # Sum of 4 distinct powers of two decodes to exactly 4 set bits.
        for bag_row in bag_matrix:
            bag_sum = int(round(bag_row[0] * 4))
            assert bin(bag_sum).count("1") == 4, (
                f"Bag sum {bag_sum} does not decode to 4 distinct rows"
            )

    def test_with_replacement_can_repeat_rows(self):
        """Bootstrap draws must sometimes repeat a row, unlike a distinct-row draw."""
        n_rows = 12
        data_matrix = (2.0 ** np.arange(n_rows)).reshape(-1, 1)
        obs_dataframe = pd.DataFrame(
            {"subject_ID": ["S0"] * n_rows, "age_group": ["Young"] * n_rows},
            index=[f"obs_{i}" for i in range(n_rows)],
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            bag_matrix, _, _, _ = create_bags(
                data_matrix=data_matrix,
                obs_dataframe=obs_dataframe,
                subject_id_column="subject_ID",
                bags_per_subject=30,
                mitos_per_bag=4,
                bag_statistics=["mean"],
                target_obs_column="age_group",
                task_type="classification",
                random_state=42,
                n_jobs=1,
                sampling_mode="with_replacement",
            )

        bag_sums = [int(round(bag_row[0] * 4)) for bag_row in bag_matrix]
        assert any(bin(bag_sum).count("1") < 4 for bag_sum in bag_sums), (
            "Expected at least one bootstrap bag to repeat a row"
        )

    def test_high_overlap_emits_warning_naming_the_subject(self):
        # 50 rows per subject, 40 per bag → 80% overlap, well above the 20% default.
        test_anndata = _make_bag_anndata(n_subjects=2, mitos_per_subject=50)

        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter("always")
            self._create(test_anndata, mitos_per_bag=40)

        messages = " ".join(str(w.message) for w in caught_warnings)
        assert "overlap" in messages.lower()
        assert "S00" in messages

    def test_low_overlap_emits_no_warning(self):
        # 500 rows per subject, 10 per bag → 2% overlap, below the threshold.
        test_anndata = _make_bag_anndata(n_subjects=2, mitos_per_subject=500)

        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter("always")
            self._create(test_anndata, mitos_per_bag=10)

        overlap_warnings = [
            w for w in caught_warnings if "overlap" in str(w.message).lower()
        ]
        assert overlap_warnings == []

    def test_max_overlap_fraction_of_one_disables_flagging(self):
        test_anndata = _make_bag_anndata(n_subjects=2, mitos_per_subject=50)

        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter("always")
            self._create(test_anndata, mitos_per_bag=40, max_overlap_fraction=1.0)

        overlap_warnings = [
            w for w in caught_warnings if "overlap" in str(w.message).lower()
        ]
        assert overlap_warnings == []

    def test_qc_table_is_printed(self, capsys):
        test_anndata = _make_bag_anndata(n_subjects=2, mitos_per_subject=50)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._create(test_anndata, mitos_per_bag=40)

        console_output = capsys.readouterr().out
        assert "[Bag sampling QC]" in console_output
        assert "auto" in console_output
        assert "Overlap" in console_output
        assert "S00" in console_output
        assert "80.0%" in console_output

    def test_qc_table_reports_replacement_explicitly(self, capsys):
        test_anndata = _make_bag_anndata(n_subjects=2, mitos_per_subject=50)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._create(test_anndata, mitos_per_bag=40, sampling_mode="with_replacement")

        console_output = capsys.readouterr().out
        assert "with replacement" in console_output
        assert "with_replacement" in console_output

    def test_verbose_false_silences_table_but_keeps_warning(self, capsys):
        test_anndata = _make_bag_anndata(n_subjects=2, mitos_per_subject=50)

        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter("always")
            self._create(test_anndata, mitos_per_bag=40, verbose=False)

        console_output = capsys.readouterr().out
        assert "[Bag sampling QC]" not in console_output
        messages = " ".join(str(w.message) for w in caught_warnings)
        assert "overlap" in messages.lower()

    def test_default_mode_reproduces_previous_bags(self):
        """The new arguments must not change the bags produced by the old defaults."""
        test_anndata = _make_bag_anndata(n_subjects=3, mitos_per_subject=50)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            bag_matrix_default, _, _, _ = self._create(test_anndata)
            bag_matrix_explicit, _, _, _ = self._create(
                test_anndata, sampling_mode="auto", max_overlap_fraction=0.2
            )

        np.testing.assert_array_equal(bag_matrix_default, bag_matrix_explicit)
