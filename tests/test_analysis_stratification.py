"""
test_analysis_stratification.py

Unit tests for mito_marker.analysis.stratification.

Covers:
  bin_obs_column():
    - Default tertile binning: correct column name, 3 bins, all events assigned.
    - Quartile binning (n_bins=4): correct auto column name, 4 bins.
    - Custom n_bins=5: column name uses "_5bins" suffix.
    - Custom labels: applied verbatim; wrong length raises ValueError.
    - NaN handling: NaN source values remain NaN; non-NaN events correctly binned.
    - Color palette injection: all bin labels appear in .uns['color_palette'].
    - .uns['analysis_config']['stratification'] is populated correctly.
    - Overwriting an existing column emits a UserWarning.
    - obs_column not found raises KeyError.
    - obs_column not numeric raises TypeError.
    - n_bins < 2 raises ValueError.
    - Custom new_column_name is used verbatim.

  split_obs_by_threshold():
    - Basic split: correct column name, exactly 2 groups, correct membership.
    - Auto-generated labels include the threshold value.
    - Custom labels applied verbatim; wrong count raises ValueError.
    - NaN handling: NaN source values remain NaN.
    - Color palette injection: both labels in .uns['color_palette'].
    - .uns['analysis_config']['threshold_split'] is populated correctly.
    - All events below threshold → below group only (empty above group).
    - All events at/above threshold → above group only (empty below group).
    - Overwriting an existing column emits a UserWarning.
    - obs_column not found raises KeyError.
    - obs_column not numeric raises TypeError.

  map_obs_values():
    - Returns the new column name string.
    - Default column name: "{source_column}_group".
    - Custom new_column_name used verbatim.
    - Mapped values appear in the new column with correct labels.
    - Source column is left unchanged after mapping.
    - All declared category keys appear in .uns['color_palette'].
    - Colors are valid hex strings.
    - Unmapped values receive the unmapped_label.
    - unmapped_label does not appear in palette when no row is unmapped.
    - unmapped_label appears in palette when at least one row is unmapped.
    - Custom unmapped_label is used.
    - Duplicate source value across two keys emits a UserWarning (last key wins).
    - Overwriting an existing column emits a UserWarning.
    - Missing source_column raises KeyError.
    - Two independent mappings coexist in .uns['analysis_config']['obs_value_map'].
    - .uns['analysis_config']['obs_value_map'] stores source_column and value_map.
    - n_obs is unchanged after mapping.
"""

import warnings

import anndata
import numpy as np
import pandas as pd
import pytest

from mito_marker.analysis.colors import COLOR_PALETTE_KEY
from mito_marker.analysis.selection import _ANALYSIS_CONFIG_KEY
from mito_marker.analysis.stratification import bin_obs_column, map_obs_values, split_obs_by_threshold


# ---------------------------------------------------------------------------
# Shared fixture factory
# ---------------------------------------------------------------------------


def _make_test_anndata(
    n_obs: int = 120,
    include_age: bool = True,
    include_glucose: bool = True,
    age_with_nan: bool = False,
) -> anndata.AnnData:
    """
    Build a minimal AnnData with numerical .obs columns suitable for stratification tests.

    Arguments:
        n_obs: Number of observations.
        include_age: Include a float 'age' column (0–99).
        include_glucose: Include a float 'glucose' column (0.5–2.5).
        age_with_nan: Replace the first 10 age values with NaN.

    Returns:
        AnnData with shape (n_obs, 5).
    """
    rng = np.random.default_rng(42)
    x_matrix = rng.standard_normal((n_obs, 5)).astype(np.float32)
    obs_dict: dict = {}

    if include_age:
        age_values = rng.uniform(low=0.0, high=99.0, size=n_obs).astype(np.float32)
        if age_with_nan:
            age_values[:10] = np.nan
        obs_dict["age"] = age_values

    if include_glucose:
        obs_dict["glucose"] = rng.uniform(low=0.5, high=2.5, size=n_obs).astype(np.float32)

    obs_dataframe = pd.DataFrame(obs_dict, index=[f"cell_{i}" for i in range(n_obs)])
    var_dataframe = pd.DataFrame(index=[f"ch_{i}" for i in range(5)])

    return anndata.AnnData(X=x_matrix, obs=obs_dataframe, var=var_dataframe)


# ===========================================================================
# Tests for bin_obs_column()
# ===========================================================================


class TestBinObsColumnBasic:
    def test_default_tertiles_column_name(self) -> None:
        anndata_object = _make_test_anndata()
        bin_obs_column(anndata_object, obs_column="age")
        assert "age_tertiles" in anndata_object.obs.columns

    def test_default_tertiles_produces_three_groups(self) -> None:
        anndata_object = _make_test_anndata()
        bin_obs_column(anndata_object, obs_column="age")
        unique_values = anndata_object.obs["age_tertiles"].dropna().unique()
        assert len(unique_values) == 3

    def test_all_events_assigned_no_nan(self) -> None:
        anndata_object = _make_test_anndata()
        bin_obs_column(anndata_object, obs_column="age")
        assert anndata_object.obs["age_tertiles"].isna().sum() == 0

    def test_quartiles_column_name(self) -> None:
        anndata_object = _make_test_anndata()
        bin_obs_column(anndata_object, obs_column="age", n_bins=4)
        assert "age_quartiles" in anndata_object.obs.columns

    def test_quartiles_produces_four_groups(self) -> None:
        anndata_object = _make_test_anndata()
        bin_obs_column(anndata_object, obs_column="age", n_bins=4)
        unique_values = anndata_object.obs["age_quartiles"].dropna().unique()
        assert len(unique_values) == 4

    def test_custom_n_bins_uses_bins_suffix(self) -> None:
        anndata_object = _make_test_anndata()
        bin_obs_column(anndata_object, obs_column="age", n_bins=5)
        assert "age_5bins" in anndata_object.obs.columns

    def test_custom_new_column_name(self) -> None:
        anndata_object = _make_test_anndata()
        bin_obs_column(anndata_object, obs_column="age", new_column_name="my_age_bins")
        assert "my_age_bins" in anndata_object.obs.columns

    def test_returns_same_anndata_object(self) -> None:
        anndata_object = _make_test_anndata()
        returned = bin_obs_column(anndata_object, obs_column="age")
        assert returned is anndata_object

    def test_n_obs_unchanged(self) -> None:
        anndata_object = _make_test_anndata(n_obs=120)
        bin_obs_column(anndata_object, obs_column="age")
        assert anndata_object.n_obs == 120

    def test_n_vars_unchanged(self) -> None:
        anndata_object = _make_test_anndata()
        bin_obs_column(anndata_object, obs_column="age")
        assert anndata_object.n_vars == 5


class TestBinObsColumnLabels:
    def test_auto_labels_contain_t1_t2_t3_for_tertiles(self) -> None:
        anndata_object = _make_test_anndata()
        bin_obs_column(anndata_object, obs_column="age", n_bins=3)
        labels = sorted(anndata_object.obs["age_tertiles"].dropna().unique())
        # Labels are sorted alphabetically; check prefix presence
        all_labels_joined = " ".join(labels)
        assert "T1" in all_labels_joined
        assert "T2" in all_labels_joined
        assert "T3" in all_labels_joined

    def test_auto_labels_contain_q_prefix_for_quartiles(self) -> None:
        anndata_object = _make_test_anndata()
        bin_obs_column(anndata_object, obs_column="age", n_bins=4)
        labels = sorted(anndata_object.obs["age_quartiles"].dropna().unique())
        all_labels_joined = " ".join(labels)
        assert "Q1" in all_labels_joined
        assert "Q4" in all_labels_joined

    def test_custom_labels_applied_verbatim(self) -> None:
        anndata_object = _make_test_anndata()
        custom_labels = ["Young", "Middle-aged", "Old"]
        bin_obs_column(anndata_object, obs_column="age", n_bins=3, labels=custom_labels)
        unique_values = set(anndata_object.obs["age_tertiles"].dropna().unique())
        assert unique_values == set(custom_labels)

    def test_wrong_labels_length_raises_value_error(self) -> None:
        anndata_object = _make_test_anndata()
        with pytest.raises(ValueError, match="labels has"):
            bin_obs_column(
                anndata_object, obs_column="age", n_bins=3, labels=["Only", "Two"]
            )


class TestBinObsColumnNaN:
    def test_nan_source_values_remain_nan(self) -> None:
        anndata_object = _make_test_anndata(age_with_nan=True)
        bin_obs_column(anndata_object, obs_column="age")
        nan_count = anndata_object.obs["age_tertiles"].isna().sum()
        assert nan_count == 10

    def test_non_nan_events_correctly_binned_with_nan_present(self) -> None:
        anndata_object = _make_test_anndata(n_obs=120, age_with_nan=True)
        bin_obs_column(anndata_object, obs_column="age")
        # 110 events should be assigned (120 - 10 NaN)
        non_nan_count = anndata_object.obs["age_tertiles"].notna().sum()
        assert non_nan_count == 110


class TestBinObsColumnColors:
    def test_bin_labels_appear_in_color_palette(self) -> None:
        anndata_object = _make_test_anndata()
        bin_obs_column(anndata_object, obs_column="age", n_bins=3)
        palette = anndata_object.uns.get(COLOR_PALETTE_KEY, {})
        bin_labels = list(anndata_object.obs["age_tertiles"].dropna().unique())
        for label in bin_labels:
            assert label in palette, f"Label '{label}' missing from color_palette"

    def test_colors_are_valid_hex_strings(self) -> None:
        import re

        anndata_object = _make_test_anndata()
        bin_obs_column(anndata_object, obs_column="age", n_bins=3)
        palette = anndata_object.uns[COLOR_PALETTE_KEY]
        hex_pattern = re.compile(r"^#[0-9a-fA-F]{6}$")
        for label in anndata_object.obs["age_tertiles"].dropna().unique():
            assert hex_pattern.match(palette[label]), (
                f"Color for '{label}' is not a valid hex string: {palette[label]}"
            )

    def test_color_injection_is_idempotent(self) -> None:
        anndata_object = _make_test_anndata()
        bin_obs_column(anndata_object, obs_column="age", n_bins=3)
        palette_after_first = dict(anndata_object.uns[COLOR_PALETTE_KEY])
        bin_obs_column(anndata_object, obs_column="glucose", n_bins=3)
        # First call's colors must not be overwritten
        for label, color in palette_after_first.items():
            assert anndata_object.uns[COLOR_PALETTE_KEY][label] == color


class TestBinObsColumnMetadata:
    def test_analysis_config_stratification_key_exists(self) -> None:
        anndata_object = _make_test_anndata()
        bin_obs_column(anndata_object, obs_column="age")
        assert _ANALYSIS_CONFIG_KEY in anndata_object.uns
        assert "stratification" in anndata_object.uns[_ANALYSIS_CONFIG_KEY]

    def test_stratification_metadata_source_column(self) -> None:
        anndata_object = _make_test_anndata()
        bin_obs_column(anndata_object, obs_column="age")
        metadata = anndata_object.uns[_ANALYSIS_CONFIG_KEY]["stratification"]
        assert metadata["source_column"] == "age"

    def test_stratification_metadata_n_bins_requested(self) -> None:
        anndata_object = _make_test_anndata()
        bin_obs_column(anndata_object, obs_column="age", n_bins=4)
        metadata = anndata_object.uns[_ANALYSIS_CONFIG_KEY]["stratification"]
        assert metadata["n_bins_requested"] == 4

    def test_stratification_metadata_bin_edges_length(self) -> None:
        anndata_object = _make_test_anndata()
        bin_obs_column(anndata_object, obs_column="age", n_bins=3)
        metadata = anndata_object.uns[_ANALYSIS_CONFIG_KEY]["stratification"]
        # bin_edges has n_bins + 1 elements
        assert len(metadata["bin_edges"]) == 4

    def test_stratification_metadata_labels_match_column(self) -> None:
        anndata_object = _make_test_anndata()
        bin_obs_column(anndata_object, obs_column="age", n_bins=3)
        metadata = anndata_object.uns[_ANALYSIS_CONFIG_KEY]["stratification"]
        column_labels = set(anndata_object.obs["age_tertiles"].dropna().unique())
        assert set(metadata["labels"]) == column_labels


class TestBinObsColumnErrors:
    def test_missing_obs_column_raises_key_error(self) -> None:
        anndata_object = _make_test_anndata()
        with pytest.raises(KeyError, match="nonexistent"):
            bin_obs_column(anndata_object, obs_column="nonexistent")

    def test_non_numeric_column_raises_type_error(self) -> None:
        anndata_object = _make_test_anndata()
        anndata_object.obs["category"] = "label_a"
        with pytest.raises(TypeError, match="not numerical"):
            bin_obs_column(anndata_object, obs_column="category")

    def test_n_bins_less_than_2_raises_value_error(self) -> None:
        anndata_object = _make_test_anndata()
        with pytest.raises(ValueError, match="n_bins must be >= 2"):
            bin_obs_column(anndata_object, obs_column="age", n_bins=1)

    def test_overwriting_existing_column_emits_warning(self) -> None:
        anndata_object = _make_test_anndata()
        bin_obs_column(anndata_object, obs_column="age")
        with pytest.warns(UserWarning, match="already exists"):
            bin_obs_column(anndata_object, obs_column="age")


# ===========================================================================
# Tests for split_obs_by_threshold()
# ===========================================================================


class TestSplitObsByThresholdBasic:
    def test_default_column_name(self) -> None:
        anndata_object = _make_test_anndata()
        split_obs_by_threshold(anndata_object, obs_column="glucose", threshold=1.5)
        assert "glucose_split" in anndata_object.obs.columns

    def test_produces_exactly_two_groups(self) -> None:
        anndata_object = _make_test_anndata()
        split_obs_by_threshold(anndata_object, obs_column="glucose", threshold=1.5)
        unique_values = anndata_object.obs["glucose_split"].dropna().unique()
        assert len(unique_values) == 2

    def test_below_threshold_group_is_correct(self) -> None:
        anndata_object = _make_test_anndata()
        threshold = 1.5
        split_obs_by_threshold(anndata_object, obs_column="glucose", threshold=threshold)
        below_label = f"Low (< {threshold:.1f})"
        below_mask = anndata_object.obs["glucose_split"] == below_label
        # Every event in the below group must have glucose < threshold
        assert (anndata_object.obs.loc[below_mask, "glucose"] < threshold).all()

    def test_above_threshold_group_is_correct(self) -> None:
        anndata_object = _make_test_anndata()
        threshold = 1.5
        split_obs_by_threshold(anndata_object, obs_column="glucose", threshold=threshold)
        above_label = f"High (≥ {threshold:.1f})"
        above_mask = anndata_object.obs["glucose_split"] == above_label
        # Every event in the above group must have glucose >= threshold
        assert (anndata_object.obs.loc[above_mask, "glucose"] >= threshold).all()

    def test_all_events_covered_no_nan(self) -> None:
        anndata_object = _make_test_anndata()
        split_obs_by_threshold(anndata_object, obs_column="glucose", threshold=1.5)
        assert anndata_object.obs["glucose_split"].isna().sum() == 0

    def test_below_plus_above_equals_total(self) -> None:
        anndata_object = _make_test_anndata()
        split_obs_by_threshold(anndata_object, obs_column="glucose", threshold=1.5)
        value_counts = anndata_object.obs["glucose_split"].value_counts()
        assert value_counts.sum() == anndata_object.n_obs

    def test_returns_same_anndata_object(self) -> None:
        anndata_object = _make_test_anndata()
        returned = split_obs_by_threshold(anndata_object, obs_column="glucose", threshold=1.5)
        assert returned is anndata_object

    def test_custom_new_column_name(self) -> None:
        anndata_object = _make_test_anndata()
        split_obs_by_threshold(
            anndata_object, obs_column="glucose", threshold=1.5, new_column_name="diabetic_status"
        )
        assert "diabetic_status" in anndata_object.obs.columns


class TestSplitObsByThresholdLabels:
    def test_auto_labels_contain_threshold_value(self) -> None:
        anndata_object = _make_test_anndata()
        split_obs_by_threshold(anndata_object, obs_column="glucose", threshold=1.26)
        labels = list(anndata_object.obs["glucose_split"].dropna().unique())
        all_labels_joined = " ".join(labels)
        assert "1.3" in all_labels_joined  # threshold 1.26 formatted as 1.3

    def test_custom_labels_applied_verbatim(self) -> None:
        anndata_object = _make_test_anndata()
        custom_labels = ["Normoglycaemia", "T2D range"]
        split_obs_by_threshold(
            anndata_object,
            obs_column="glucose",
            threshold=1.26,
            labels=custom_labels,
        )
        unique_values = set(anndata_object.obs["glucose_split"].dropna().unique())
        assert unique_values == set(custom_labels)

    def test_wrong_labels_count_raises_value_error(self) -> None:
        anndata_object = _make_test_anndata()
        with pytest.raises(ValueError, match="exactly 2 elements"):
            split_obs_by_threshold(
                anndata_object,
                obs_column="glucose",
                threshold=1.5,
                labels=["Only one label"],
            )


class TestSplitObsByThresholdNaN:
    def test_nan_source_values_remain_nan(self) -> None:
        anndata_object = _make_test_anndata(age_with_nan=True)
        split_obs_by_threshold(anndata_object, obs_column="age", threshold=50.0)
        nan_count = anndata_object.obs["age_split"].isna().sum()
        assert nan_count == 10

    def test_non_nan_events_correctly_split_with_nan_present(self) -> None:
        anndata_object = _make_test_anndata(n_obs=120, age_with_nan=True)
        split_obs_by_threshold(anndata_object, obs_column="age", threshold=50.0)
        non_nan_count = anndata_object.obs["age_split"].notna().sum()
        assert non_nan_count == 110


class TestSplitObsByThresholdEdgeCases:
    def test_threshold_below_all_values_puts_all_in_above_group(self) -> None:
        anndata_object = _make_test_anndata()
        # glucose is in [0.5, 2.5]; threshold of 0.0 puts everyone in above group
        split_obs_by_threshold(anndata_object, obs_column="glucose", threshold=0.0)
        above_label = "High (≥ 0.0)"
        above_count = (anndata_object.obs["glucose_split"] == above_label).sum()
        assert above_count == anndata_object.n_obs

    def test_threshold_above_all_values_puts_all_in_below_group(self) -> None:
        anndata_object = _make_test_anndata()
        # glucose is in [0.5, 2.5]; threshold of 100.0 puts everyone in below group
        split_obs_by_threshold(anndata_object, obs_column="glucose", threshold=100.0)
        below_label = "Low (< 100.0)"
        below_count = (anndata_object.obs["glucose_split"] == below_label).sum()
        assert below_count == anndata_object.n_obs


class TestSplitObsByThresholdColors:
    def test_both_labels_appear_in_color_palette(self) -> None:
        anndata_object = _make_test_anndata()
        threshold = 1.5
        split_obs_by_threshold(anndata_object, obs_column="glucose", threshold=threshold)
        palette = anndata_object.uns.get(COLOR_PALETTE_KEY, {})
        below_label = f"Low (< {threshold:.1f})"
        above_label = f"High (≥ {threshold:.1f})"
        assert below_label in palette
        assert above_label in palette

    def test_colors_are_valid_hex_strings(self) -> None:
        import re

        anndata_object = _make_test_anndata()
        split_obs_by_threshold(anndata_object, obs_column="glucose", threshold=1.5)
        palette = anndata_object.uns[COLOR_PALETTE_KEY]
        hex_pattern = re.compile(r"^#[0-9a-fA-F]{6}$")
        for label in anndata_object.obs["glucose_split"].dropna().unique():
            assert hex_pattern.match(palette[label])


class TestSplitObsByThresholdMetadata:
    def test_analysis_config_threshold_split_key_exists(self) -> None:
        anndata_object = _make_test_anndata()
        split_obs_by_threshold(anndata_object, obs_column="glucose", threshold=1.5)
        assert _ANALYSIS_CONFIG_KEY in anndata_object.uns
        assert "threshold_split" in anndata_object.uns[_ANALYSIS_CONFIG_KEY]

    def test_metadata_threshold_value_stored(self) -> None:
        anndata_object = _make_test_anndata()
        split_obs_by_threshold(anndata_object, obs_column="glucose", threshold=1.26)
        metadata = anndata_object.uns[_ANALYSIS_CONFIG_KEY]["threshold_split"]
        assert metadata["threshold"] == pytest.approx(1.26)

    def test_metadata_source_column_stored(self) -> None:
        anndata_object = _make_test_anndata()
        split_obs_by_threshold(anndata_object, obs_column="glucose", threshold=1.5)
        metadata = anndata_object.uns[_ANALYSIS_CONFIG_KEY]["threshold_split"]
        assert metadata["source_column"] == "glucose"

    def test_metadata_new_column_stored(self) -> None:
        anndata_object = _make_test_anndata()
        split_obs_by_threshold(
            anndata_object, obs_column="glucose", threshold=1.5, new_column_name="my_split"
        )
        metadata = anndata_object.uns[_ANALYSIS_CONFIG_KEY]["threshold_split"]
        assert metadata["new_column"] == "my_split"


class TestSplitObsByThresholdErrors:
    def test_missing_obs_column_raises_key_error(self) -> None:
        anndata_object = _make_test_anndata()
        with pytest.raises(KeyError, match="nonexistent"):
            split_obs_by_threshold(anndata_object, obs_column="nonexistent", threshold=1.0)

    def test_non_numeric_column_raises_type_error(self) -> None:
        anndata_object = _make_test_anndata()
        anndata_object.obs["category"] = "label_a"
        with pytest.raises(TypeError, match="not numerical"):
            split_obs_by_threshold(anndata_object, obs_column="category", threshold=1.0)

    def test_overwriting_existing_column_emits_warning(self) -> None:
        anndata_object = _make_test_anndata()
        split_obs_by_threshold(anndata_object, obs_column="glucose", threshold=1.5)
        with pytest.warns(UserWarning, match="already exists"):
            split_obs_by_threshold(anndata_object, obs_column="glucose", threshold=1.5)


# ===========================================================================
# Helpers for map_obs_values tests
# ===========================================================================


def _make_species_anndata(n_per_species: int = 20) -> anndata.AnnData:
    """
    Build a minimal AnnData with a categorical 'specie' .obs column.

    Six species are included (n_per_species rows each):
      worm, fly, K fish, Z fish, mouse, human
    Total observations: 6 * n_per_species.
    """
    species = ["worm", "fly", "K fish", "Z fish", "mouse", "human"]
    specie_column = []
    for species_name in species:
        specie_column.extend([species_name] * n_per_species)

    total_obs = len(specie_column)
    rng = np.random.default_rng(0)
    x_matrix = rng.standard_normal((total_obs, 3)).astype(np.float32)

    obs_dataframe = pd.DataFrame(
        {"specie": specie_column},
        index=[f"obs_{i}" for i in range(total_obs)],
    )
    var_dataframe = pd.DataFrame(index=[f"ch_{i}" for i in range(3)])
    return anndata.AnnData(X=x_matrix, obs=obs_dataframe, var=var_dataframe)


_LIFESPAN_VALUE_MAP = {
    "Long life": ["worm", "fly", "K fish"],
    "Short life": ["Z fish", "mouse", "human"],
}


# ===========================================================================
# Tests for map_obs_values()
# ===========================================================================


class TestMapObsValuesBasic:
    def test_returns_new_column_name_string(self) -> None:
        anndata_object = _make_species_anndata()
        result = map_obs_values(anndata_object, "specie", _LIFESPAN_VALUE_MAP)
        assert isinstance(result, str)

    def test_default_column_name(self) -> None:
        anndata_object = _make_species_anndata()
        result = map_obs_values(anndata_object, "specie", _LIFESPAN_VALUE_MAP)
        assert result == "specie_group"
        assert "specie_group" in anndata_object.obs.columns

    def test_custom_new_column_name(self) -> None:
        anndata_object = _make_species_anndata()
        result = map_obs_values(
            anndata_object, "specie", _LIFESPAN_VALUE_MAP, new_column_name="lifespan"
        )
        assert result == "lifespan"
        assert "lifespan" in anndata_object.obs.columns

    def test_mapped_values_are_correct(self) -> None:
        anndata_object = _make_species_anndata(n_per_species=10)
        map_obs_values(anndata_object, "specie", _LIFESPAN_VALUE_MAP)
        long_life_rows = anndata_object.obs[
            anndata_object.obs["specie"].isin(["worm", "fly", "K fish"])
        ]
        assert (long_life_rows["specie_group"] == "Long life").all()
        short_life_rows = anndata_object.obs[
            anndata_object.obs["specie"].isin(["Z fish", "mouse", "human"])
        ]
        assert (short_life_rows["specie_group"] == "Short life").all()

    def test_source_column_is_unchanged(self) -> None:
        anndata_object = _make_species_anndata(n_per_species=5)
        original_species = anndata_object.obs["specie"].tolist()
        map_obs_values(anndata_object, "specie", _LIFESPAN_VALUE_MAP)
        assert anndata_object.obs["specie"].tolist() == original_species

    def test_n_obs_unchanged(self) -> None:
        anndata_object = _make_species_anndata()
        original_n_obs = anndata_object.n_obs
        map_obs_values(anndata_object, "specie", _LIFESPAN_VALUE_MAP)
        assert anndata_object.n_obs == original_n_obs

    def test_unique_group_labels_match_value_map_keys(self) -> None:
        anndata_object = _make_species_anndata()
        map_obs_values(anndata_object, "specie", _LIFESPAN_VALUE_MAP)
        unique_labels = set(anndata_object.obs["specie_group"].unique())
        assert unique_labels == {"Long life", "Short life"}


class TestMapObsValuesUnmapped:
    def test_unmapped_values_receive_default_other_label(self) -> None:
        anndata_object = _make_species_anndata(n_per_species=5)
        # Only map "worm"; all others → "Other"
        map_obs_values(
            anndata_object,
            "specie",
            {"Only worms": ["worm"]},
        )
        unmapped_rows = anndata_object.obs[anndata_object.obs["specie"] != "worm"]
        assert (unmapped_rows["specie_group"] == "Other").all()

    def test_custom_unmapped_label(self) -> None:
        anndata_object = _make_species_anndata(n_per_species=5)
        map_obs_values(
            anndata_object,
            "specie",
            {"Only worms": ["worm"]},
            unmapped_label="Unknown species",
        )
        unmapped_rows = anndata_object.obs[anndata_object.obs["specie"] != "worm"]
        assert (unmapped_rows["specie_group"] == "Unknown species").all()

    def test_unmapped_label_not_in_palette_when_all_mapped(self) -> None:
        anndata_object = _make_species_anndata()
        map_obs_values(anndata_object, "specie", _LIFESPAN_VALUE_MAP)
        palette = anndata_object.uns.get(COLOR_PALETTE_KEY, {})
        assert "Other" not in palette

    def test_unmapped_label_in_palette_when_some_rows_unmapped(self) -> None:
        anndata_object = _make_species_anndata(n_per_species=5)
        map_obs_values(
            anndata_object,
            "specie",
            {"Only worms": ["worm"]},
        )
        palette = anndata_object.uns.get(COLOR_PALETTE_KEY, {})
        assert "Other" in palette


class TestMapObsValuesColors:
    def test_all_declared_keys_appear_in_color_palette(self) -> None:
        anndata_object = _make_species_anndata()
        map_obs_values(anndata_object, "specie", _LIFESPAN_VALUE_MAP)
        palette = anndata_object.uns.get(COLOR_PALETTE_KEY, {})
        for label in _LIFESPAN_VALUE_MAP.keys():
            assert label in palette

    def test_colors_are_valid_hex_strings(self) -> None:
        import re

        anndata_object = _make_species_anndata()
        map_obs_values(anndata_object, "specie", _LIFESPAN_VALUE_MAP)
        palette = anndata_object.uns[COLOR_PALETTE_KEY]
        hex_pattern = re.compile(r"^#[0-9a-fA-F]{6}$")
        for label in _LIFESPAN_VALUE_MAP.keys():
            assert hex_pattern.match(palette[label])

    def test_three_group_mapping_all_colors_injected(self) -> None:
        anndata_object = _make_species_anndata()
        three_group_map = {
            "Invertebrate": ["worm", "fly"],
            "Vertebrate": ["K fish", "Z fish"],
            "Mammal": ["mouse", "human"],
        }
        map_obs_values(anndata_object, "specie", three_group_map)
        palette = anndata_object.uns.get(COLOR_PALETTE_KEY, {})
        for label in three_group_map.keys():
            assert label in palette


class TestMapObsValuesMetadata:
    def test_analysis_config_obs_value_map_key_exists(self) -> None:
        anndata_object = _make_species_anndata()
        col = map_obs_values(anndata_object, "specie", _LIFESPAN_VALUE_MAP)
        assert _ANALYSIS_CONFIG_KEY in anndata_object.uns
        assert "obs_value_map" in anndata_object.uns[_ANALYSIS_CONFIG_KEY]
        assert col in anndata_object.uns[_ANALYSIS_CONFIG_KEY]["obs_value_map"]

    def test_metadata_source_column_stored(self) -> None:
        anndata_object = _make_species_anndata()
        col = map_obs_values(anndata_object, "specie", _LIFESPAN_VALUE_MAP)
        metadata = anndata_object.uns[_ANALYSIS_CONFIG_KEY]["obs_value_map"][col]
        assert metadata["source_column"] == "specie"

    def test_metadata_value_map_stored(self) -> None:
        anndata_object = _make_species_anndata()
        col = map_obs_values(anndata_object, "specie", _LIFESPAN_VALUE_MAP)
        metadata = anndata_object.uns[_ANALYSIS_CONFIG_KEY]["obs_value_map"][col]
        assert metadata["value_map"] == {k: list(v) for k, v in _LIFESPAN_VALUE_MAP.items()}

    def test_two_independent_mappings_coexist(self) -> None:
        anndata_object = _make_species_anndata()
        col1 = map_obs_values(
            anndata_object, "specie", _LIFESPAN_VALUE_MAP, new_column_name="lifespan"
        )
        col2 = map_obs_values(
            anndata_object,
            "specie",
            {"Invertebrate": ["worm", "fly"], "Vertebrate": ["K fish", "Z fish", "mouse", "human"]},
            new_column_name="phylogeny",
        )
        obs_value_map = anndata_object.uns[_ANALYSIS_CONFIG_KEY]["obs_value_map"]
        assert col1 in obs_value_map
        assert col2 in obs_value_map
        assert "lifespan" in anndata_object.obs.columns
        assert "phylogeny" in anndata_object.obs.columns


class TestMapObsValuesWarningsAndErrors:
    def test_missing_source_column_raises_key_error(self) -> None:
        anndata_object = _make_species_anndata()
        with pytest.raises(KeyError, match="nonexistent"):
            map_obs_values(anndata_object, "nonexistent", {"A": ["worm"]})

    def test_overwriting_existing_column_emits_warning(self) -> None:
        anndata_object = _make_species_anndata()
        map_obs_values(anndata_object, "specie", _LIFESPAN_VALUE_MAP)
        with pytest.warns(UserWarning, match="already exists"):
            map_obs_values(anndata_object, "specie", _LIFESPAN_VALUE_MAP)

    def test_duplicate_source_value_emits_warning_last_key_wins(self) -> None:
        anndata_object = _make_species_anndata(n_per_species=5)
        overlapping_map = {
            "Group A": ["worm", "fly"],
            "Group B": ["fly", "K fish"],  # "fly" appears in both A and B
        }
        with pytest.warns(UserWarning, match="multiple category keys"):
            map_obs_values(anndata_object, "specie", overlapping_map)
        # "fly" rows should be assigned to "Group B" (last key wins)
        fly_rows = anndata_object.obs[anndata_object.obs["specie"] == "fly"]
        assert (fly_rows["specie_group"] == "Group B").all()
