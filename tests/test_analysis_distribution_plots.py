"""
Tests for mito_marker.analysis.distribution_plots

Covers:
  - _resolve_features()  — default exclusion of non-analytical features
  - _get_group_labels()  — single string, list of strings, missing column
  - _build_color_map()   — returns a dict with one entry per unique label
  - plot_histogram()     — smoke tests + parameter validation
  - plot_density()       — smoke tests + fill=False variant
  - plot_violin()        — grouped and split styles, error conditions
"""

import warnings

import anndata
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import scipy.sparse

matplotlib.use("Agg")  # Non-interactive backend — safe for CI and Colab

from mito_marker.analysis.distribution_plots import (
    _build_color_map,
    _get_data_matrix,
    _get_feature_category_color,
    _get_group_labels,
    _resolve_features,
    _resolve_group_order,
    _sort_features_by_category,
    plot_density,
    plot_histogram,
    plot_violin,
)
from mito_marker.controlled_vocabulary import (
    TEM_FEATURE_CATEGORIES,
    TEM_FEATURE_CATEGORY_COLORS,
    TEM_FEATURE_CATEGORY_ORDER,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_anndata(
    n_obs: int = 80,
    n_vars: int = 5,
    conditions: list | None = None,
    species: list | None = None,
    include_non_analytical_flag: bool = False,
    sparse: bool = False,
) -> anndata.AnnData:
    """Build a minimal AnnData for testing distribution plot functions.

    Args:
        n_obs: Number of observations (mitochondria).
        n_vars: Number of feature variables.
        conditions: List of condition labels, cycled across obs. Defaults to
            ["Old", "Young"].
        species: List of species labels, cycled across obs. Defaults to
            ["Zebrafish", "Killifish"].
        include_non_analytical_flag: If True, marks the last variable as
            non-analytical via .var['is_non_analytical'].
        sparse: If True, store .X as a scipy sparse matrix.

    Returns:
        A small AnnData suitable for unit tests.
    """
    rng = np.random.default_rng(42)

    if conditions is None:
        conditions = ["Old", "Young"]
    if species is None:
        species = ["Zebrafish", "Killifish"]

    feature_names = [f"Feature_{i}" for i in range(n_vars)]
    x_data = rng.random((n_obs, n_vars)).astype(np.float32)

    if sparse:
        x_matrix = scipy.sparse.csr_matrix(x_data)
    else:
        x_matrix = x_data

    # Decouple the cycling so that all combinations of condition × species
    # are produced (condition cycles every row, species cycles every block).
    obs_dataframe = pd.DataFrame(
        {
            "condition": [conditions[i % len(conditions)] for i in range(n_obs)],
            "specie": [species[(i // len(conditions)) % len(species)] for i in range(n_obs)],
        },
        index=[f"obs_{i}" for i in range(n_obs)],
    )

    var_dataframe = pd.DataFrame(index=feature_names)
    if include_non_analytical_flag:
        var_dataframe["is_non_analytical"] = [False] * (n_vars - 1) + [True]

    return anndata.AnnData(X=x_matrix, obs=obs_dataframe, var=var_dataframe)


@pytest.fixture
def basic_anndata() -> anndata.AnnData:
    """Standard 5-feature, 80-obs AnnData with Old/Young conditions."""
    return _make_anndata()


@pytest.fixture
def anndata_with_flag() -> anndata.AnnData:
    """AnnData where the last feature is flagged as non-analytical."""
    return _make_anndata(include_non_analytical_flag=True)


@pytest.fixture
def sparse_anndata() -> anndata.AnnData:
    """AnnData with a sparse .X matrix."""
    return _make_anndata(sparse=True)


@pytest.fixture
def three_condition_anndata() -> anndata.AnnData:
    """AnnData with three distinct conditions (Young/Middle/Old)."""
    return _make_anndata(conditions=["Young", "Middle", "Old"])


@pytest.fixture
def cell_type_anndata() -> anndata.AnnData:
    """AnnData whose group values are NOT already in alphabetical order.

    Values ["Mu", "Dis", "Pro"] alphabetize to ["Dis", "Mu", "Pro"] — useful
    to tell apart "the default numeric-aware sort" from "an explicit
    caller-supplied order" such as ["Dis", "Pro", "Mu"].
    """
    return _make_anndata(conditions=["Mu", "Dis", "Pro"])


@pytest.fixture
def tem_like_anndata() -> anndata.AnnData:
    """AnnData using real TEM feature names spanning all 4 categories.

    One feature from Size, Shape, Intensity, Cristae Orientation, plus one
    unmapped name — deliberately shuffled so the un-grouped order does not
    already match TEM_FEATURE_CATEGORY_ORDER.
    """
    rng = np.random.default_rng(42)
    feature_names = [
        "Intensity_SD",              # Intensity
        "Mito_Area",                 # Size
        "CristaeOrientation_Angle",  # Cristae Orientation
        "Mito_AR",                   # Shape
        "Not_A_Real_Feature",        # unmapped -> "Other"
    ]
    n_obs = 40
    obs_dataframe = pd.DataFrame(
        {"condition": ["Old", "Young"] * (n_obs // 2)},
        index=[f"obs_{i}" for i in range(n_obs)],
    )
    return anndata.AnnData(
        X=rng.random((n_obs, len(feature_names))).astype(np.float32),
        obs=obs_dataframe,
        var=pd.DataFrame(index=feature_names),
    )


# ---------------------------------------------------------------------------
# _resolve_features
# ---------------------------------------------------------------------------


class TestResolveFeatures:
    def test_explicit_list_returned_unchanged(self, basic_anndata):
        result = _resolve_features(basic_anndata, features=["Feature_0", "Feature_2"])
        assert result == ["Feature_0", "Feature_2"]

    def test_none_returns_all_features_no_flag(self, basic_anndata):
        result = _resolve_features(basic_anndata, features=None)
        assert set(result) == set(basic_anndata.var_names)

    def test_none_excludes_flagged_non_analytical(self, anndata_with_flag):
        result = _resolve_features(anndata_with_flag, features=None)
        assert "Feature_4" not in result
        assert len(result) == 4

    def test_none_includes_all_when_no_flag_column(self, basic_anndata):
        # basic_anndata has no 'is_non_analytical' column and no matching
        # TEM_NON_ANALYTICAL_FEATURES names, so all features are returned
        result = _resolve_features(basic_anndata, features=None)
        assert len(result) == 5

    def test_invalid_feature_raises_value_error(self, basic_anndata):
        with pytest.raises(ValueError, match="not found in .var_names"):
            _resolve_features(basic_anndata, features=["Feature_0", "DoesNotExist"])


# ---------------------------------------------------------------------------
# _get_group_labels
# ---------------------------------------------------------------------------


class TestGetGroupLabels:
    def test_single_string_column(self, basic_anndata):
        labels = _get_group_labels(basic_anndata, "condition")
        assert list(labels.unique()) == sorted(["Old", "Young"])

    def test_list_of_one_column(self, basic_anndata):
        labels = _get_group_labels(basic_anndata, ["condition"])
        assert set(labels.unique()) == {"Old", "Young"}

    def test_list_of_two_columns_combined(self, basic_anndata):
        labels = _get_group_labels(basic_anndata, ["specie", "condition"])
        # Should produce combined labels like "Zebrafish / Old"
        assert all(" / " in label for label in labels.unique())
        assert len(labels) == len(basic_anndata)

    def test_missing_column_raises_value_error(self, basic_anndata):
        with pytest.raises(ValueError, match="not found in .obs"):
            _get_group_labels(basic_anndata, "nonexistent_column")

    def test_missing_column_in_list_raises_value_error(self, basic_anndata):
        with pytest.raises(ValueError, match="not found in .obs"):
            _get_group_labels(basic_anndata, ["condition", "missing"])

    def test_combined_series_name(self, basic_anndata):
        labels = _get_group_labels(basic_anndata, ["specie", "condition"])
        assert labels.name == "specie / condition"


# ---------------------------------------------------------------------------
# _build_color_map
# ---------------------------------------------------------------------------


class TestBuildColorMap:
    def test_returns_dict_with_correct_keys(self, basic_anndata):
        labels = _get_group_labels(basic_anndata, "condition")
        color_map = _build_color_map(basic_anndata, labels)
        assert set(color_map.keys()) == {"Old", "Young"}

    def test_values_are_non_empty_strings(self, basic_anndata):
        labels = _get_group_labels(basic_anndata, "condition")
        color_map = _build_color_map(basic_anndata, labels)
        for color_value in color_map.values():
            assert isinstance(color_value, str) and len(color_value) > 0

    def test_combined_group_by(self, basic_anndata):
        labels = _get_group_labels(basic_anndata, ["specie", "condition"])
        color_map = _build_color_map(basic_anndata, labels)
        # 2 species × 2 conditions = 4 combined labels
        assert len(color_map) == 4


# ---------------------------------------------------------------------------
# _get_data_matrix
# ---------------------------------------------------------------------------


class TestGetDataMatrix:
    def test_dense_returns_numpy_array(self, basic_anndata):
        matrix = _get_data_matrix(basic_anndata)
        assert isinstance(matrix, np.ndarray)
        assert matrix.shape == (80, 5)

    def test_sparse_returns_numpy_array(self, sparse_anndata):
        matrix = _get_data_matrix(sparse_anndata)
        assert isinstance(matrix, np.ndarray)
        assert matrix.shape == (80, 5)

    def test_dtype_is_float32(self, basic_anndata):
        matrix = _get_data_matrix(basic_anndata)
        assert matrix.dtype == np.float32


# ---------------------------------------------------------------------------
# _resolve_group_order
# ---------------------------------------------------------------------------


class TestResolveGroupOrder:
    def test_none_order_falls_back_to_sorted(self):
        result = _resolve_group_order(["Mu", "Dis", "Pro"], order=None)
        assert result == ["Dis", "Mu", "Pro"]

    def test_explicit_order_respected(self):
        result = _resolve_group_order(["Mu", "Dis", "Pro"], order=["Dis", "Pro", "Mu"])
        assert result == ["Dis", "Pro", "Mu"]

    def test_unlisted_present_values_appended_sorted(self):
        # "Emb" is present but not in the requested order — must still
        # appear (never silently dropped), sorted after the listed ones.
        result = _resolve_group_order(
            ["Mu", "Dis", "Emb"], order=["Dis", "Pro", "Mu"]
        )
        assert result == ["Dis", "Mu", "Emb"]

    def test_order_values_absent_from_data_are_ignored(self):
        # "Pro" is requested but not present — must not appear in the result.
        result = _resolve_group_order(["Dis", "Mu"], order=["Dis", "Pro", "Mu"])
        assert result == ["Dis", "Mu"]

    def test_no_value_ever_dropped(self):
        present = ["Mu", "Dis", "Pro", "Emb", "Sp"]
        result = _resolve_group_order(present, order=["Dis", "Loop", "Pro"])
        assert set(result) == set(present)


# ---------------------------------------------------------------------------
# _sort_features_by_category / _get_feature_category_color
# ---------------------------------------------------------------------------


class TestSortFeaturesByCategory:
    def test_reorders_into_contiguous_category_blocks(self):
        shuffled = [
            "Intensity_SD", "Mito_Area", "CristaeOrientation_Angle", "Mito_AR",
        ]
        result = _sort_features_by_category(shuffled)
        categories_in_result = [TEM_FEATURE_CATEGORIES[name] for name in result]
        # Category order must follow TEM_FEATURE_CATEGORY_ORDER and each
        # category's features must be contiguous (no interleaving).
        seen_categories = []
        for category in categories_in_result:
            if category not in seen_categories:
                seen_categories.append(category)
        assert seen_categories == sorted(
            seen_categories, key=TEM_FEATURE_CATEGORY_ORDER.index
        )

    def test_unmapped_feature_grouped_as_other_with_warning(self):
        with pytest.warns(UserWarning, match="Other"):
            result = _sort_features_by_category(["Mito_Area", "Not_A_Real_Feature"])
        assert "Not_A_Real_Feature" in result
        assert result[-1] == "Not_A_Real_Feature"  # "Other" sorts last

    def test_no_feature_ever_dropped(self):
        features = ["Intensity_SD", "Mito_Area", "CristaeOrientation_Angle", "Mito_AR"]
        assert set(_sort_features_by_category(features)) == set(features)

    def test_stable_within_category(self):
        # Two Size features, given in reverse alphabetical order — the sort
        # is stable, so their relative order must be preserved.
        result = _sort_features_by_category(["Mito_Perimeter", "Mito_Area"])
        assert result == ["Mito_Perimeter", "Mito_Area"]


class TestGetFeatureCategoryColor:
    def test_known_feature_returns_its_category_color(self):
        assert _get_feature_category_color("Mito_Area") == TEM_FEATURE_CATEGORY_COLORS["Size"]

    def test_unknown_feature_returns_other_color(self):
        assert (
            _get_feature_category_color("Not_A_Real_Feature")
            == TEM_FEATURE_CATEGORY_COLORS["Other"]
        )


# ---------------------------------------------------------------------------
# plot_histogram
# ---------------------------------------------------------------------------


class TestPlotHistogram:
    def test_runs_without_error(self, basic_anndata):
        plot_histogram(basic_anndata, group_by="condition")
        plt.close("all")

    def test_runs_with_list_group_by(self, basic_anndata):
        plot_histogram(basic_anndata, group_by=["specie", "condition"])
        plt.close("all")

    def test_explicit_features_subset(self, basic_anndata):
        plot_histogram(
            basic_anndata,
            group_by="condition",
            features=["Feature_0", "Feature_1"],
        )
        plt.close("all")

    def test_bins_parameter(self, basic_anndata):
        # Should not raise; fewer bins produce wider bars
        plot_histogram(basic_anndata, group_by="condition", bins=10)
        plt.close("all")

    def test_alpha_parameter(self, basic_anndata):
        plot_histogram(basic_anndata, group_by="condition", alpha=0.3)
        plt.close("all")

    def test_three_conditions(self, three_condition_anndata):
        plot_histogram(three_condition_anndata, group_by="condition")
        plt.close("all")

    def test_custom_title(self, basic_anndata):
        plot_histogram(basic_anndata, group_by="condition", title="My Title")
        plt.close("all")

    def test_sparse_data(self, sparse_anndata):
        plot_histogram(sparse_anndata, group_by="condition")
        plt.close("all")

    def test_missing_group_column_raises(self, basic_anndata):
        with pytest.raises(ValueError):
            plot_histogram(basic_anndata, group_by="missing_col")

    def test_invalid_feature_raises(self, basic_anndata):
        with pytest.raises(ValueError):
            plot_histogram(basic_anndata, group_by="condition", features=["Bad"])

    def test_returns_none(self, basic_anndata):
        result = plot_histogram(basic_anndata, group_by="condition")
        plt.close("all")
        assert result is None


# ---------------------------------------------------------------------------
# plot_density
# ---------------------------------------------------------------------------


class TestPlotDensity:
    def test_runs_without_error(self, basic_anndata):
        plot_density(basic_anndata, group_by="condition")
        plt.close("all")

    def test_runs_with_list_group_by(self, basic_anndata):
        plot_density(basic_anndata, group_by=["specie", "condition"])
        plt.close("all")

    def test_fill_false(self, basic_anndata):
        plot_density(basic_anndata, group_by="condition", fill=False)
        plt.close("all")

    def test_fill_true_default(self, basic_anndata):
        plot_density(basic_anndata, group_by="condition", fill=True)
        plt.close("all")

    def test_explicit_features_subset(self, basic_anndata):
        plot_density(
            basic_anndata,
            group_by="condition",
            features=["Feature_0", "Feature_2"],
        )
        plt.close("all")

    def test_custom_alpha(self, basic_anndata):
        plot_density(
            basic_anndata,
            group_by="condition",
            fill_alpha=0.5,
            line_alpha=0.7,
        )
        plt.close("all")

    def test_three_conditions(self, three_condition_anndata):
        plot_density(three_condition_anndata, group_by="condition")
        plt.close("all")

    def test_sparse_data(self, sparse_anndata):
        plot_density(sparse_anndata, group_by="condition")
        plt.close("all")

    def test_missing_group_column_raises(self, basic_anndata):
        with pytest.raises(ValueError):
            plot_density(basic_anndata, group_by="missing_col")

    def test_returns_none(self, basic_anndata):
        result = plot_density(basic_anndata, group_by="condition")
        plt.close("all")
        assert result is None

    def test_custom_title(self, basic_anndata):
        plot_density(basic_anndata, group_by="condition", title="KDE Plot")
        plt.close("all")

    def test_explicit_order_runs_without_error(self, cell_type_anndata):
        plot_density(cell_type_anndata, group_by="condition", order=["Dis", "Pro", "Mu"])
        plt.close("all")

    def test_explicit_order_never_drops_unlisted_group(self, cell_type_anndata):
        # "Pro" is present but not requested — must still be drawn (legend
        # entry present), not silently dropped.
        plot_density(cell_type_anndata, group_by="condition", order=["Dis", "Mu"])
        legend_texts = [text.get_text() for text in plt.gcf().axes[0].get_legend().get_texts()]
        plt.close("all")
        assert "Pro" in legend_texts


# ---------------------------------------------------------------------------
# plot_violin — "grouped" style
# ---------------------------------------------------------------------------


class TestPlotViolinGrouped:
    def test_runs_without_error(self, basic_anndata):
        plot_violin(basic_anndata, group_by="condition", style="grouped")
        plt.close("all")

    def test_default_style_is_grouped(self, basic_anndata):
        # Should not raise — grouped is the default
        plot_violin(basic_anndata, group_by="condition")
        plt.close("all")

    def test_three_conditions(self, three_condition_anndata):
        plot_violin(three_condition_anndata, group_by="condition", style="grouped")
        plt.close("all")

    def test_explicit_features(self, basic_anndata):
        plot_violin(
            basic_anndata,
            group_by="condition",
            style="grouped",
            features=["Feature_0"],
        )
        plt.close("all")

    def test_list_group_by(self, basic_anndata):
        plot_violin(basic_anndata, group_by=["specie", "condition"], style="grouped")
        plt.close("all")

    def test_returns_none(self, basic_anndata):
        result = plot_violin(basic_anndata, group_by="condition", style="grouped")
        plt.close("all")
        assert result is None

    def test_missing_group_column_raises(self, basic_anndata):
        with pytest.raises(ValueError):
            plot_violin(basic_anndata, group_by="missing_col", style="grouped")

    def test_custom_title(self, basic_anndata):
        plot_violin(basic_anndata, group_by="condition", style="grouped", title="T")
        plt.close("all")

    def test_sparse_data(self, sparse_anndata):
        plot_violin(sparse_anndata, group_by="condition", style="grouped")
        plt.close("all")

    def test_grouped_with_x_by_runs(self, basic_anndata):
        # x-axis = specie, violins coloured by condition
        plot_violin(
            basic_anndata,
            group_by="condition",
            style="grouped",
            x_by="specie",
        )
        plt.close("all")

    def test_grouped_with_x_by_invalid_column_raises(self, basic_anndata):
        with pytest.raises(ValueError, match="x_by column"):
            plot_violin(
                basic_anndata,
                group_by="condition",
                style="grouped",
                x_by="nonexistent_col",
            )

    def test_grouped_with_x_by_exclusive_groups(self):
        # Each specie belongs to exactly one group — the half-violin scenario.
        # Must produce a figure without raising.
        rng = np.random.default_rng(0)
        n_obs = 60
        # KFish → Short life, ZFish → Long life (mutually exclusive)
        species = ["KFish"] * 30 + ["ZFish"] * 30
        groups = ["Short life"] * 30 + ["Long life"] * 30
        obs = pd.DataFrame(
            {"specie": species, "lifespan": groups},
            index=[f"obs_{i}" for i in range(n_obs)],
        )
        exclusive_anndata = anndata.AnnData(
            X=rng.random((n_obs, 3)).astype(np.float32),
            obs=obs,
            var=pd.DataFrame(index=["F0", "F1", "F2"]),
        )
        plot_violin(
            exclusive_anndata,
            group_by="lifespan",
            style="grouped",
            x_by="specie",
        )
        plt.close("all")

    def test_explicit_order_controls_x_axis(self, cell_type_anndata):
        plot_violin(
            cell_type_anndata,
            group_by="condition",
            style="grouped",
            order=["Dis", "Pro", "Mu"],
        )
        x_tick_labels = [
            label.get_text() for label in plt.gcf().axes[0].get_xticklabels()
        ]
        plt.close("all")
        assert x_tick_labels == ["Dis", "Pro", "Mu"]

    def test_explicit_order_never_drops_unlisted_group(self, cell_type_anndata):
        plot_violin(
            cell_type_anndata,
            group_by="condition",
            style="grouped",
            order=["Dis", "Mu"],  # "Pro" is present but unlisted
        )
        x_tick_labels = [
            label.get_text() for label in plt.gcf().axes[0].get_xticklabels()
        ]
        plt.close("all")
        assert set(x_tick_labels) == {"Dis", "Mu", "Pro"}

    def test_group_label_map_relabels_x_axis(self, basic_anndata):
        plot_violin(
            basic_anndata,
            group_by="condition",
            style="grouped",
            group_label_map={"Old": "Old-S", "Young": "Young-S"},
        )
        x_tick_labels = {
            label.get_text() for label in plt.gcf().axes[0].get_xticklabels()
        }
        plt.close("all")
        assert x_tick_labels == {"Old-S", "Young-S"}

    def test_group_label_map_partial_leaves_others_unchanged(self, basic_anndata):
        plot_violin(
            basic_anndata,
            group_by="condition",
            style="grouped",
            group_label_map={"Old": "Old-S"},  # "Young" left as-is
        )
        x_tick_labels = {
            label.get_text() for label in plt.gcf().axes[0].get_xticklabels()
        }
        plt.close("all")
        assert x_tick_labels == {"Old-S", "Young"}

    def test_group_features_by_category_reorders_subplots(self, tem_like_anndata):
        plot_violin(
            tem_like_anndata,
            group_by="condition",
            style="grouped",
            group_features_by_category=True,
        )
        figure = plt.gcf()
        subplot_titles = [ax.get_title() for ax in figure.axes if ax.get_title()]
        plt.close("all")
        categories_in_order = [
            TEM_FEATURE_CATEGORIES.get(name, "Other") for name in subplot_titles
        ]
        seen = []
        for category in categories_in_order:
            if category not in seen:
                seen.append(category)
        assert seen == sorted(seen, key=[*TEM_FEATURE_CATEGORY_ORDER, "Other"].index)

    def test_group_features_by_category_false_keeps_original_order(self, tem_like_anndata):
        # Default behavior must be unchanged — same order as .var_names.
        plot_violin(
            tem_like_anndata,
            group_by="condition",
            style="grouped",
            group_features_by_category=False,
        )
        figure = plt.gcf()
        subplot_titles = [ax.get_title() for ax in figure.axes if ax.get_title()]
        plt.close("all")
        assert subplot_titles == list(tem_like_anndata.var_names)


# ---------------------------------------------------------------------------
# plot_violin — "split" style
# ---------------------------------------------------------------------------


class TestPlotViolinSplit:
    def test_runs_without_error(self, basic_anndata):
        # Old and Young are exactly 2 conditions → valid split
        plot_violin(
            basic_anndata,
            group_by="condition",
            style="split",
            x_by="specie",
        )
        plt.close("all")

    def test_missing_x_by_raises(self, basic_anndata):
        with pytest.raises(ValueError, match="x_by"):
            plot_violin(basic_anndata, group_by="condition", style="split")

    def test_invalid_x_by_column_raises(self, basic_anndata):
        with pytest.raises(ValueError, match="x_by column"):
            plot_violin(
                basic_anndata,
                group_by="condition",
                style="split",
                x_by="nonexistent_col",
            )

    def test_more_than_two_groups_raises(self, three_condition_anndata):
        with pytest.raises(ValueError, match="exactly 2 unique"):
            plot_violin(
                three_condition_anndata,
                group_by="condition",
                style="split",
                x_by="specie",
            )

    def test_explicit_features(self, basic_anndata):
        plot_violin(
            basic_anndata,
            group_by="condition",
            style="split",
            x_by="specie",
            features=["Feature_0", "Feature_1"],
        )
        plt.close("all")

    def test_returns_none(self, basic_anndata):
        result = plot_violin(
            basic_anndata,
            group_by="condition",
            style="split",
            x_by="specie",
        )
        plt.close("all")
        assert result is None

    def test_custom_title(self, basic_anndata):
        plot_violin(
            basic_anndata,
            group_by="condition",
            style="split",
            x_by="specie",
            title="Split Violin",
        )
        plt.close("all")

    def test_sparse_data_split(self, sparse_anndata):
        plot_violin(
            sparse_anndata,
            group_by="condition",
            style="split",
            x_by="specie",
        )
        plt.close("all")

    def test_split_warns_for_exclusive_groups(self):
        # Each specie belongs to exactly one group — seaborn would draw half
        # violins; the function must emit a UserWarning.
        rng = np.random.default_rng(0)
        n_obs = 60
        obs = pd.DataFrame(
            {
                "specie": ["KFish"] * 30 + ["ZFish"] * 30,
                "lifespan": ["Short life"] * 30 + ["Long life"] * 30,
            },
            index=[f"obs_{i}" for i in range(n_obs)],
        )
        exclusive_anndata = anndata.AnnData(
            X=rng.random((n_obs, 2)).astype(np.float32),
            obs=obs,
            var=pd.DataFrame(index=["F0", "F1"]),
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            plot_violin(
                exclusive_anndata,
                group_by="lifespan",
                style="split",
                x_by="specie",
            )
            plt.close("all")
        assert any("half-violin" in str(w.message) for w in caught)

    def test_explicit_order_runs_without_error(self, basic_anndata):
        # order is meaningful in split style too: it picks which condition
        # lands on the left vs. right half of each violin.
        plot_violin(
            basic_anndata,
            group_by="condition",
            style="split",
            x_by="specie",
            order=["Young", "Old"],
        )
        plt.close("all")
