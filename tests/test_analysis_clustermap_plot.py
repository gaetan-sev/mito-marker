"""
test_analysis_clustermap_plot.py

Unit tests for mito_marker.analysis.clustermap_plot.

Covers:
  _get_data_and_channels():
    - Uses active layer when set, falls back to .X otherwise.
    - Excludes .var['is_non_analytical'] == True channels.

  _get_group_labels():
    - Single-column and multi-column (combined) grouping.
    - Raises ValueError for a missing .obs column.

  _resolve_display_order():
    - Listed-and-present values first (requested order), unlisted appended sorted.
    - Defaults to sorted order when no order is requested.

  _apply_node_rotations():
    - Swaps exactly the targeted row's children, no other row changes.
    - Out-of-range row index raises ValueError.
    - Double rotation of the same row is a no-op.
    - Leaf membership under a rotated node is unchanged (order-independent set).

  _can_cluster_axis():
    - False when show_dendrograms is False.
    - False when axis has < 2 leaves.
    - False for correlation metric when the other axis has < 1 dimension.

  plot_feature_clustermap():
    - Returns a dict with exactly the documented keys.
    - group_order / feature_order are permutations of the full label sets.
    - Non-analytical features never appear in feature_order.
    - Missing group_by column raises ValueError.
    - standardize_group_means True vs False.
    - Rotation changes group_linkage at the targeted row and reorders that subtree.
    - Degenerate inputs (1 group, 1 feature) skip clustering without crashing.
    - rotate_*_nodes combined with show_dendrograms=False raises ValueError.
    - Out-of-range rotation index raises ValueError.

  plot_feature_clustermap(nest_aggregate_by=...) — ADR-011 nested group means:
    - Nested and pooled matrices differ on an unbalanced fixture.
    - Nested result equals the hand-computed mean-of-per-subject-means.
    - Nested == pooled on a balanced fixture (equal rows per subject).
    - Two-level nesting works and differs from one-level nesting.
    - Missing nest_aggregate_by .obs column raises ValueError.
    - nest_aggregate_by=None reproduces the pre-change pooled behaviour exactly.
    - The returned dict records which aggregation mode was used.
"""

import numpy as np
import pandas as pd
import pytest
import anndata
import matplotlib

matplotlib.use("Agg")

from mito_marker.analysis.clustermap_plot import (
    _ANALYSIS_CONFIG_KEY,
    _apply_node_rotations,
    _can_cluster_axis,
    _get_data_and_channels,
    _get_group_labels,
    _resolve_display_order,
    plot_feature_clustermap,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_test_anndata(
    n_groups: int = 5,
    n_features: int = 8,
    n_obs_per_group: int = 20,
    include_non_analytical: bool = True,
    random_state: int = 0,
) -> anndata.AnnData:
    """Synthetic AnnData with separable group-mean structure for clustering tests."""
    rng = np.random.default_rng(random_state)
    group_names = [f"Group{i}" for i in range(n_groups)]

    # Each group has a distinct mean offset per feature so aggregated group
    # means are well separated and produce a deterministic clustering tree.
    group_offsets = rng.uniform(-10, 10, size=(n_groups, n_features))
    data_rows = []
    group_labels = []
    for group_index, group_name in enumerate(group_names):
        group_data = group_offsets[group_index] + rng.normal(
            scale=0.1, size=(n_obs_per_group, n_features)
        )
        data_rows.append(group_data)
        group_labels.extend([group_name] * n_obs_per_group)

    x_matrix = np.vstack(data_rows).astype(np.float32)
    feature_names = [f"Feature_{i}" for i in range(n_features)]

    obs = pd.DataFrame({"condition": group_labels})
    var = pd.DataFrame(index=feature_names)
    if include_non_analytical:
        var["is_non_analytical"] = False
        var.loc[feature_names[0], "is_non_analytical"] = True
        var.loc[feature_names[1], "is_non_analytical"] = True

    adata = anndata.AnnData(X=x_matrix, obs=obs, var=var)
    adata.uns[_ANALYSIS_CONFIG_KEY] = {"active_layer": None, "active_selection": None}
    return adata


@pytest.fixture
def test_anndata() -> anndata.AnnData:
    return _make_test_anndata()


# ---------------------------------------------------------------------------
# _get_data_and_channels
# ---------------------------------------------------------------------------


class TestGetDataAndChannels:
    def test_uses_raw_x_when_no_active_layer(self, test_anndata):
        matrix, channels = _get_data_and_channels(test_anndata)
        expected_channels = [c for c in test_anndata.var_names if not test_anndata.var.loc[c, "is_non_analytical"]]
        assert channels == expected_channels
        assert matrix.shape == (test_anndata.n_obs, len(expected_channels))

    def test_uses_active_layer_when_set(self, test_anndata):
        test_anndata.layers["scaled"] = test_anndata.X * 2.0
        test_anndata.uns[_ANALYSIS_CONFIG_KEY]["active_layer"] = "scaled"
        matrix, _ = _get_data_and_channels(test_anndata)
        non_analytical_mask = ~test_anndata.var["is_non_analytical"].values.astype(bool)
        np.testing.assert_allclose(matrix, test_anndata.layers["scaled"][:, non_analytical_mask])

    def test_falls_back_to_x_when_layer_key_missing(self, test_anndata):
        test_anndata.uns[_ANALYSIS_CONFIG_KEY]["active_layer"] = "nonexistent_layer"
        matrix, _ = _get_data_and_channels(test_anndata)
        non_analytical_mask = ~test_anndata.var["is_non_analytical"].values.astype(bool)
        np.testing.assert_allclose(matrix, test_anndata.X[:, non_analytical_mask])

    def test_excludes_non_analytical_channels(self, test_anndata):
        _, channels = _get_data_and_channels(test_anndata)
        assert "Feature_0" not in channels
        assert "Feature_1" not in channels

    def test_no_non_analytical_column_keeps_all_channels(self):
        adata = _make_test_anndata(include_non_analytical=False)
        _, channels = _get_data_and_channels(adata)
        assert channels == list(adata.var_names)


# ---------------------------------------------------------------------------
# _get_group_labels
# ---------------------------------------------------------------------------


class TestGetGroupLabels:
    def test_single_column(self, test_anndata):
        labels = _get_group_labels(test_anndata, "condition")
        pd.testing.assert_series_equal(labels, test_anndata.obs["condition"])

    def test_multi_column_combined_with_slash(self, test_anndata):
        test_anndata.obs["batch"] = "B1"
        labels = _get_group_labels(test_anndata, ["condition", "batch"])
        assert labels.iloc[0] == f"{test_anndata.obs['condition'].iloc[0]} / B1"

    def test_missing_column_raises_value_error(self, test_anndata):
        with pytest.raises(ValueError, match="not found in .obs"):
            _get_group_labels(test_anndata, "nonexistent")


# ---------------------------------------------------------------------------
# _resolve_display_order
# ---------------------------------------------------------------------------


class TestResolveDisplayOrder:
    def test_none_defaults_to_sorted(self):
        result = _resolve_display_order(None, ["C", "A", "B"])
        assert result == ["A", "B", "C"]

    def test_listed_values_come_first_in_requested_order(self):
        result = _resolve_display_order(["B", "A"], ["A", "B", "C"])
        assert result == ["B", "A", "C"]

    def test_no_present_value_is_dropped(self):
        result = _resolve_display_order(["Z"], ["A", "B"])
        assert set(result) == {"A", "B"}
        assert len(result) == 2

    def test_unlisted_values_absent_from_order_still_sorted(self):
        result = _resolve_display_order(["B"], ["A", "B", "C", "D"])
        assert result == ["B", "A", "C", "D"]


# ---------------------------------------------------------------------------
# _apply_node_rotations
# ---------------------------------------------------------------------------


def _hand_built_linkage() -> np.ndarray:
    """4 leaves (0,1,2,3): row0 merges 0+1 -> node4, row1 merges 2+3 -> node5,
    row2 merges node4+node5 -> node6."""
    return np.array(
        [
            [0.0, 1.0, 1.0, 2.0],
            [2.0, 3.0, 1.5, 2.0],
            [4.0, 5.0, 3.0, 4.0],
        ]
    )


class TestApplyNodeRotations:
    def test_none_leaves_matrix_unchanged(self):
        matrix = _hand_built_linkage()
        result = _apply_node_rotations(matrix, None)
        np.testing.assert_array_equal(result, matrix)

    def test_rotates_only_targeted_row(self):
        matrix = _hand_built_linkage()
        result = _apply_node_rotations(matrix, [1])
        np.testing.assert_array_equal(result[0], matrix[0])
        np.testing.assert_array_equal(result[2], matrix[2])
        assert result[1, 0] == matrix[1, 1]
        assert result[1, 1] == matrix[1, 0]

    def test_out_of_range_row_raises_value_error(self):
        matrix = _hand_built_linkage()
        with pytest.raises(ValueError, match="out of range"):
            _apply_node_rotations(matrix, [99])

    def test_double_rotation_is_no_op(self):
        matrix = _hand_built_linkage()
        once = _apply_node_rotations(matrix, [2])
        twice = _apply_node_rotations(once, [2])
        np.testing.assert_array_equal(twice, matrix)

    def test_leaf_membership_unchanged_under_rotation(self):
        """Rotating a node changes display order only, never which leaves belong under it."""
        from scipy.cluster.hierarchy import dendrogram

        matrix = _hand_built_linkage()
        rotated = _apply_node_rotations(matrix, [2])

        original_leaves = set(dendrogram(matrix, no_plot=True)["leaves"])
        rotated_leaves = set(dendrogram(rotated, no_plot=True)["leaves"])
        assert original_leaves == rotated_leaves == {0, 1, 2, 3}


# ---------------------------------------------------------------------------
# _can_cluster_axis
# ---------------------------------------------------------------------------


class TestCanClusterAxis:
    def test_false_when_dendrograms_disabled(self):
        assert _can_cluster_axis("group", 5, 5, False, "correlation") is False

    def test_false_when_fewer_than_two_leaves(self):
        assert _can_cluster_axis("group", 1, 5, True, "correlation") is False

    def test_false_for_correlation_with_single_dimension_other_axis(self):
        assert _can_cluster_axis("feature", 5, 1, True, "correlation") is False

    def test_true_for_euclidean_with_single_dimension_other_axis(self):
        assert _can_cluster_axis("feature", 5, 1, True, "euclidean") is True

    def test_true_when_enough_leaves_and_dimensions(self):
        assert _can_cluster_axis("group", 5, 5, True, "correlation") is True

    def test_false_with_two_leaves_and_minimum_three(self):
        """2 leaves only ever produce one trivial merge — group axis calls with minimum_leaf_count=3."""
        assert _can_cluster_axis("group", 2, 5, True, "correlation", minimum_leaf_count=3) is False

    def test_true_with_two_leaves_and_default_minimum(self):
        """Default minimum_leaf_count=2 still allows clustering exactly 2 leaves (used for features)."""
        assert _can_cluster_axis("feature", 2, 5, True, "correlation") is True


# ---------------------------------------------------------------------------
# plot_feature_clustermap — integration
# ---------------------------------------------------------------------------


class TestPlotFeatureClustermapStructure:
    def test_returns_dict_with_documented_keys(self, test_anndata):
        result = plot_feature_clustermap(test_anndata, group_by="condition")
        assert set(result.keys()) == {
            "figure",
            "group_order",
            "feature_order",
            "group_linkage",
            "feature_linkage",
            "standardized_group_means",
            "nest_aggregate_by",
        }

    def test_figure_is_matplotlib_figure(self, test_anndata):
        import matplotlib.figure

        result = plot_feature_clustermap(test_anndata, group_by="condition")
        assert isinstance(result["figure"], matplotlib.figure.Figure)

    def test_group_order_is_permutation_of_all_groups(self, test_anndata):
        result = plot_feature_clustermap(test_anndata, group_by="condition")
        assert sorted(result["group_order"]) == sorted(test_anndata.obs["condition"].unique())

    def test_feature_order_excludes_non_analytical(self, test_anndata):
        result = plot_feature_clustermap(test_anndata, group_by="condition")
        assert "Feature_0" not in result["feature_order"]
        assert "Feature_1" not in result["feature_order"]
        expected_features = [f for f in test_anndata.var_names if f not in ("Feature_0", "Feature_1")]
        assert sorted(result["feature_order"]) == sorted(expected_features)

    def test_missing_group_by_raises_value_error(self, test_anndata):
        with pytest.raises(ValueError, match="not found in .obs"):
            plot_feature_clustermap(test_anndata, group_by="nonexistent")


class TestTwoGroupsComparison:
    """With exactly 2 groups (e.g. "Young" vs "Old"), the plot should still render as a
    plain side-by-side comparison — group clustering is a trivial single merge and is
    skipped, but feature clustering still runs normally for a readable multi-metric view."""

    def test_group_clustering_skipped_with_warning(self):
        adata = _make_test_anndata(n_groups=2, n_features=8)
        with pytest.warns(UserWarning, match="Fewer than 3 groups"):
            result = plot_feature_clustermap(adata, group_by="condition")
        assert result["group_linkage"] is None

    def test_still_plots_a_figure(self):
        import matplotlib.figure

        adata = _make_test_anndata(n_groups=2, n_features=8)
        with pytest.warns(UserWarning):
            result = plot_feature_clustermap(adata, group_by="condition")
        assert isinstance(result["figure"], matplotlib.figure.Figure)

    def test_group_order_is_requested_order_not_dropped(self):
        adata = _make_test_anndata(n_groups=2, n_features=8)
        with pytest.warns(UserWarning):
            result = plot_feature_clustermap(adata, group_by="condition")
        assert sorted(result["group_order"]) == sorted(adata.obs["condition"].unique())

    def test_feature_clustering_still_runs(self):
        """Feature axis clustering is unaffected by the 2-group group-axis guard."""
        adata = _make_test_anndata(n_groups=2, n_features=8, include_non_analytical=False)
        with pytest.warns(UserWarning, match="Fewer than 3 groups"):
            result = plot_feature_clustermap(adata, group_by="condition")
        assert result["feature_linkage"] is not None

    def test_rotate_group_nodes_raises_since_group_axis_not_clustered(self):
        adata = _make_test_anndata(n_groups=2, n_features=8)
        with pytest.raises(ValueError, match="not being clustered"):
            plot_feature_clustermap(adata, group_by="condition", rotate_group_nodes=[0])


class TestClusterFeaturesToggle:
    """cluster_features independently enables/disables feature (row) clustering."""

    def test_false_disables_feature_linkage_with_no_warning(self, test_anndata, recwarn):
        result = plot_feature_clustermap(test_anndata, group_by="condition", cluster_features=False)
        assert result["feature_linkage"] is None
        assert not any(issubclass(w.category, UserWarning) for w in recwarn.list)

    def test_false_keeps_feature_order_unclustered(self, test_anndata):
        result = plot_feature_clustermap(test_anndata, group_by="condition", cluster_features=False)
        expected_features = sorted(f for f in test_anndata.var_names if f not in ("Feature_0", "Feature_1"))
        assert result["feature_order"] == expected_features

    def test_false_does_not_affect_group_clustering(self, test_anndata):
        result = plot_feature_clustermap(test_anndata, group_by="condition", cluster_features=False)
        assert result["group_linkage"] is not None

    def test_true_is_default_behavior(self, test_anndata):
        with_default = plot_feature_clustermap(test_anndata, group_by="condition")
        with_explicit_true = plot_feature_clustermap(test_anndata, group_by="condition", cluster_features=True)
        assert with_default["feature_order"] == with_explicit_true["feature_order"]

    def test_rotate_feature_nodes_raises_when_cluster_features_false(self, test_anndata):
        with pytest.raises(ValueError, match="not being clustered"):
            plot_feature_clustermap(
                test_anndata, group_by="condition", cluster_features=False, rotate_feature_nodes=[0]
            )


class TestStandardizeToggle:
    def test_standardize_true_gives_zero_mean_rows(self, test_anndata):
        result = plot_feature_clustermap(test_anndata, group_by="condition", standardize_group_means=True)
        row_means = result["standardized_group_means"].values.mean(axis=1)
        np.testing.assert_allclose(row_means, 0.0, atol=1e-6)

    def test_standardize_false_keeps_raw_means(self, test_anndata):
        result = plot_feature_clustermap(test_anndata, group_by="condition", standardize_group_means=False)
        row_means = result["standardized_group_means"].values.mean(axis=1)
        assert not np.allclose(row_means, 0.0, atol=1e-6)


class TestRotationIntegration:
    def test_rotation_changes_group_linkage_at_targeted_row_only(self, test_anndata):
        baseline = plot_feature_clustermap(test_anndata, group_by="condition")
        target_row = 0
        rotated = plot_feature_clustermap(
            test_anndata, group_by="condition", rotate_group_nodes=[target_row]
        )
        baseline_linkage = baseline["group_linkage"]
        rotated_linkage = rotated["group_linkage"]

        assert rotated_linkage[target_row, 0] == baseline_linkage[target_row, 1]
        assert rotated_linkage[target_row, 1] == baseline_linkage[target_row, 0]
        for row_index in range(baseline_linkage.shape[0]):
            if row_index != target_row:
                np.testing.assert_array_equal(rotated_linkage[row_index], baseline_linkage[row_index])

    def test_rotation_out_of_range_raises_value_error(self, test_anndata):
        with pytest.raises(ValueError, match="out of range"):
            plot_feature_clustermap(test_anndata, group_by="condition", rotate_group_nodes=[999])

    def test_rotation_with_dendrograms_disabled_raises_value_error(self, test_anndata):
        with pytest.raises(ValueError, match="not being clustered"):
            plot_feature_clustermap(
                test_anndata, group_by="condition", show_dendrograms=False, rotate_group_nodes=[0]
            )


class TestDegenerateInputs:
    def test_single_group_skips_group_clustering(self):
        adata = _make_test_anndata(n_groups=1, n_features=6)
        with pytest.warns(UserWarning):
            result = plot_feature_clustermap(adata, group_by="condition")
        assert result["group_linkage"] is None
        assert result["group_order"] == list(adata.obs["condition"].unique())

    def test_single_group_standardize_falls_back_to_raw_means(self):
        adata = _make_test_anndata(n_groups=1, n_features=6)
        with pytest.warns(UserWarning):
            result = plot_feature_clustermap(adata, group_by="condition", standardize_group_means=True)
        assert not np.isnan(result["standardized_group_means"].values).any()

    def test_single_feature_skips_feature_clustering(self):
        adata = _make_test_anndata(n_groups=5, n_features=1, include_non_analytical=False)
        with pytest.warns(UserWarning):
            result = plot_feature_clustermap(adata, group_by="condition")
        assert result["feature_linkage"] is None


# ---------------------------------------------------------------------------
# plot_feature_clustermap(nest_aggregate_by=...) — ADR-011 §3 nested group means
# ---------------------------------------------------------------------------


def _make_subject_anndata(
    subject_row_counts: dict,
    n_features: int = 6,
    random_state: int = 11,
) -> anndata.AnnData:
    """
    Synthetic AnnData with an explicit number of rows per subject.

    subject_row_counts maps "GroupName/SubjectID" -> number of rows. Each
    subject has its own constant feature offset, so the difference between a
    pooled and a nested group mean is exactly the difference between a
    row-count-weighted and an unweighted average of those offsets.
    """
    rng = np.random.default_rng(random_state)

    data_rows = []
    group_labels = []
    subject_labels = []
    for subject_index, (subject_key, n_rows) in enumerate(subject_row_counts.items()):
        group_name, subject_id = subject_key.split("/")
        subject_offset = np.full(n_features, float(subject_index) * 10.0)
        data_rows.append(subject_offset + rng.normal(scale=0.01, size=(n_rows, n_features)))
        group_labels.extend([group_name] * n_rows)
        subject_labels.extend([subject_id] * n_rows)

    x_matrix = np.vstack(data_rows).astype(np.float64)
    feature_names = [f"Feature_{i}" for i in range(n_features)]
    obs = pd.DataFrame({
        "condition": group_labels,
        "unique_subject_ID": subject_labels,
    })
    var = pd.DataFrame(index=feature_names)
    var["is_non_analytical"] = False

    adata = anndata.AnnData(X=x_matrix, obs=obs, var=var)
    adata.uns[_ANALYSIS_CONFIG_KEY] = {"active_layer": None, "active_selection": None}
    return adata


def _make_two_level_anndata(random_state: int = 13) -> anndata.AnnData:
    """
    AnnData with a species level above the subject level, both unbalanced.

    "SpA" contributes 1 subject with many rows; "SpB" contributes 3 subjects
    with few rows each. Pooled, one-level and two-level nesting therefore give
    three different answers (the same construction as the ADR-013 radar test).
    """
    n_features = 4
    rng = np.random.default_rng(random_state)
    specification = [
        # (condition, species, subject, n_rows, Feature_0 value)
        ("Old", "SpA", "SpA-1", 40, 0.0),
        ("Old", "SpB", "SpB-1", 5, 10.0),
        ("Old", "SpB", "SpB-2", 5, 10.0),
        ("Old", "SpB", "SpB-3", 5, 10.0),
        ("Young", "SpA", "SpA-2", 20, 4.0),
        ("Young", "SpB", "SpB-4", 20, 12.0),
        ("Mid", "SpA", "SpA-3", 12, 2.0),
        ("Mid", "SpB", "SpB-5", 12, 20.0),
    ]
    data_rows = []
    conditions, species, subjects = [], [], []
    for condition, specie, subject, n_rows, value in specification:
        # Feature_0 carries the hand-computable value the assertions below
        # check; the remaining features are random so that no feature row is
        # constant across groups (correlation distance is undefined for a
        # constant vector) whichever aggregation mode is used.
        subject_profile = np.concatenate([[value], rng.uniform(-5, 5, size=n_features - 1)])
        data_rows.append(np.tile(subject_profile, (n_rows, 1)))
        conditions.extend([condition] * n_rows)
        species.extend([specie] * n_rows)
        subjects.extend([subject] * n_rows)

    obs = pd.DataFrame({
        "condition": conditions,
        "specie": species,
        "unique_subject_ID": subjects,
    })
    var = pd.DataFrame(index=[f"Feature_{i}" for i in range(n_features)])
    var["is_non_analytical"] = False

    adata = anndata.AnnData(X=np.vstack(data_rows).astype(np.float64), obs=obs, var=var)
    adata.uns[_ANALYSIS_CONFIG_KEY] = {"active_layer": None, "active_selection": None}
    return adata


class TestNestAggregateBy:
    def test_nested_differs_from_pooled_on_unbalanced_fixture(self):
        # Group "A": one subject with 200 rows, one with 10 — a 20-to-1 imbalance.
        adata = _make_subject_anndata({
            "A/A_heavy": 200, "A/A_light": 10,
            "B/B_one": 50, "B/B_two": 50,
            "C/C_one": 30, "C/C_two": 30,
        })
        pooled = plot_feature_clustermap(adata, group_by="condition")["standardized_group_means"]
        nested = plot_feature_clustermap(
            adata, group_by="condition", nest_aggregate_by="unique_subject_ID"
        )["standardized_group_means"]

        assert not np.allclose(pooled.values, nested.values)
        # The imbalance is confined to group "A" — the balanced groups must match.
        np.testing.assert_allclose(pooled["B"].values, nested["B"].values)
        np.testing.assert_allclose(pooled["C"].values, nested["C"].values)

    def test_nested_equals_mean_of_per_subject_means(self):
        adata = _make_subject_anndata({"A/A_heavy": 200, "A/A_light": 10, "B/B_one": 40})
        nested = plot_feature_clustermap(
            adata, group_by="condition", nest_aggregate_by="unique_subject_ID"
        )["standardized_group_means"]

        data_matrix = np.asarray(adata.X)
        subject_ids = adata.obs["unique_subject_ID"].values
        expected_group_a = np.stack([
            data_matrix[subject_ids == "A_heavy"].mean(axis=0),
            data_matrix[subject_ids == "A_light"].mean(axis=0),
        ]).mean(axis=0)

        # standardized_group_means is feature x group, so column "A" is the profile.
        np.testing.assert_allclose(nested["A"].values, expected_group_a)

    def test_nested_equals_pooled_on_balanced_fixture(self):
        adata = _make_subject_anndata({
            "A/A_one": 50, "A/A_two": 50,
            "B/B_one": 50, "B/B_two": 50,
            "C/C_one": 50, "C/C_two": 50,
        })
        pooled = plot_feature_clustermap(adata, group_by="condition")["standardized_group_means"]
        nested = plot_feature_clustermap(
            adata, group_by="condition", nest_aggregate_by="unique_subject_ID"
        )["standardized_group_means"]
        np.testing.assert_allclose(pooled.values, nested.values)

    def test_two_level_nesting_differs_from_one_level(self):
        adata = _make_two_level_anndata()
        pooled = plot_feature_clustermap(adata, group_by="condition")["standardized_group_means"]
        one_level = plot_feature_clustermap(
            adata, group_by="condition", nest_aggregate_by="unique_subject_ID"
        )["standardized_group_means"]
        two_level = plot_feature_clustermap(
            adata, group_by="condition", nest_aggregate_by=["specie", "unique_subject_ID"]
        )["standardized_group_means"]

        # Group "Old": 40 rows at 0.0 (SpA, 1 subject) + 15 rows at 10.0 (SpB, 3 subjects).
        assert pooled.loc["Feature_0", "Old"] == pytest.approx(150 / 55)
        assert one_level.loc["Feature_0", "Old"] == pytest.approx(7.5)
        assert two_level.loc["Feature_0", "Old"] == pytest.approx(5.0)
        assert not np.allclose(one_level.values, two_level.values)

    def test_missing_nest_column_raises_value_error(self, test_anndata):
        with pytest.raises(ValueError, match="nest_aggregate_by column 'nope' not found in .obs"):
            plot_feature_clustermap(test_anndata, group_by="condition", nest_aggregate_by="nope")

    def test_missing_column_in_multi_level_list_raises_value_error(self):
        adata = _make_two_level_anndata()
        with pytest.raises(ValueError, match="nest_aggregate_by column 'missing' not found"):
            plot_feature_clustermap(
                adata, group_by="condition", nest_aggregate_by=["missing", "unique_subject_ID"]
            )

    def test_none_reproduces_pooled_groupby_mean_exactly(self):
        """nest_aggregate_by=None must be byte-identical to the pre-change behaviour."""
        adata = _make_subject_anndata({"A/A_heavy": 200, "A/A_light": 10, "B/B_one": 40})
        result = plot_feature_clustermap(adata, group_by="condition")

        # The historical implementation, reproduced literally.
        group_labels = adata.obs["condition"]
        valid_mask = group_labels.notna().values
        expected = (
            pd.DataFrame(
                np.asarray(adata.X)[valid_mask],
                columns=adata.var_names.tolist(),
                index=group_labels.values[valid_mask],
            )
            .groupby(level=0)
            .mean()
        ).T

        np.testing.assert_array_equal(
            result["standardized_group_means"].values,
            expected.loc[result["standardized_group_means"].index, result["standardized_group_means"].columns].values,
        )

    def test_returned_dict_records_aggregation_mode(self):
        adata = _make_subject_anndata({"A/A_one": 20, "B/B_one": 20, "C/C_one": 20})
        assert plot_feature_clustermap(adata, group_by="condition")["nest_aggregate_by"] is None
        assert plot_feature_clustermap(
            adata, group_by="condition", nest_aggregate_by="unique_subject_ID"
        )["nest_aggregate_by"] == ["unique_subject_ID"]
        assert plot_feature_clustermap(
            adata, group_by="condition", nest_aggregate_by=["condition", "unique_subject_ID"]
        )["nest_aggregate_by"] == ["condition", "unique_subject_ID"]
