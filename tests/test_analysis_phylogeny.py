"""
test_analysis_phylogeny.py

Unit tests for mito_marker.analysis.phylogeny and for the phylogeny/feature
declarations it reads from mito_marker.controlled_vocabulary.

Covers:
  TEM_SPECIES_DIVERGENCE_TIME_MYA:
    - Covers every species in ALLOWED_SPECIES_TEM.
    - Has a zero diagonal and is symmetric.
    - Satisfies the ultrametric (three-point) condition.
    - Encodes the intended node heights (87 / 230 / 429 / 682 / 685 Mya).

  assert_ultrametric_divergence_times():
    - Passes on the declared matrix.
    - Raises on a non-zero diagonal, on asymmetry, and on an ultrametric
      violation, naming the offending species.
    - Raises when a requested species or pair is missing.

  build_species_phylogeny_linkage():
    - Recovers ((Worm, Droso), ((ZFish, KFish), (Mouse, Human))).
    - branch_scale="time" gives node heights equal to divergence times.
    - branch_scale="cladogram" gives the same topology with uniform,
      one-unit-per-level heights.
    - Leaf order follows TEM_SPECIES_PHYLOGENETIC_ORDER.
    - Works on a subset of species.
    - Missing species, unknown branch_scale and fewer than 2 species raise
      ValueError.

  linkage_to_newick_string() / get_linkage_clades():
    - Render topology and enumerate clades; the root is excluded by default.

  TEM_FEATURE_SUBSETS / TEM_FEATURE_CATEGORIES / TEM_FEATURE_CATEGORY_COLORS:
    - TEM_FEATURE_SUBSETS is the single source of truth; the other two are
      derived from it and stay consistent.
    - Every analytical TEM feature has exactly one category.
    - No non-analytical feature has a category.
    - Every category used has a declared color and a place in the display order.
"""

import numpy as np
import pytest

from mito_marker.analysis.phylogeny import (
    ALLOWED_BRANCH_SCALES,
    assert_ultrametric_divergence_times,
    build_species_phylogeny_linkage,
    get_linkage_clades,
    linkage_to_newick_string,
)
from mito_marker.controlled_vocabulary import (
    ALLOWED_SPECIES_TEM,
    TEM_EXPECTED_PHYLOGENETIC_CLADES,
    TEM_FEATURE_CATEGORIES,
    TEM_FEATURE_CATEGORY_COLORS,
    TEM_FEATURE_CATEGORY_ORDER,
    TEM_FEATURE_COLUMNS,
    TEM_FEATURE_SUBSETS,
    TEM_NON_ANALYTICAL_FEATURES,
    TEM_SPECIES_DIVERGENCE_TIME_MYA,
    TEM_SPECIES_PHYLOGENETIC_ORDER,
)

ALL_TEM_SPECIES = list(TEM_SPECIES_PHYLOGENETIC_ORDER)
EXPECTED_NEWICK = "((Worm,Droso),((ZFish,KFish),(Mouse,Human)))"


# ---------------------------------------------------------------------------
# TEM_SPECIES_DIVERGENCE_TIME_MYA
# ---------------------------------------------------------------------------


class TestDivergenceTimeMatrix:
    def test_covers_every_allowed_tem_species(self):
        assert set(TEM_SPECIES_DIVERGENCE_TIME_MYA) == set(ALLOWED_SPECIES_TEM)
        for species_row in TEM_SPECIES_DIVERGENCE_TIME_MYA.values():
            assert set(species_row) == set(ALLOWED_SPECIES_TEM)

    def test_diagonal_is_zero(self):
        for species in TEM_SPECIES_DIVERGENCE_TIME_MYA:
            assert TEM_SPECIES_DIVERGENCE_TIME_MYA[species][species] == 0.0

    def test_matrix_is_symmetric(self):
        for species_a in TEM_SPECIES_DIVERGENCE_TIME_MYA:
            for species_b in TEM_SPECIES_DIVERGENCE_TIME_MYA:
                assert (
                    TEM_SPECIES_DIVERGENCE_TIME_MYA[species_a][species_b]
                    == TEM_SPECIES_DIVERGENCE_TIME_MYA[species_b][species_a]
                )

    def test_matrix_is_ultrametric(self):
        # No exception means the three-point condition holds for every triple.
        assert_ultrametric_divergence_times(TEM_SPECIES_DIVERGENCE_TIME_MYA)

    def test_encodes_the_intended_node_heights(self):
        assert TEM_SPECIES_DIVERGENCE_TIME_MYA["Human"]["Mouse"] == 87.0
        assert TEM_SPECIES_DIVERGENCE_TIME_MYA["ZFish"]["KFish"] == 224.0
        assert TEM_SPECIES_DIVERGENCE_TIME_MYA["Human"]["ZFish"] == 429.0
        assert TEM_SPECIES_DIVERGENCE_TIME_MYA["Worm"]["Droso"] == 572.0
        assert TEM_SPECIES_DIVERGENCE_TIME_MYA["Worm"]["Human"] == 708.0

    def test_ecdysozoa_and_bilaterian_nodes_are_clearly_separated(self):
        """
        With the sourced timetree.org values the Ecdysozoa node (572 Mya) and
        the bilaterian root (708 Mya) are 136 Mya apart — a clear separation,
        not the near-polytomy the original provisional placeholders (682 /
        685 Mya) produced. See the vocabulary comment for that history.
        """
        ecdysozoa_node = TEM_SPECIES_DIVERGENCE_TIME_MYA["Worm"]["Droso"]
        bilaterian_root = TEM_SPECIES_DIVERGENCE_TIME_MYA["Worm"]["Human"]
        assert bilaterian_root > ecdysozoa_node
        assert bilaterian_root - ecdysozoa_node > 50.0

    def test_expected_clades_match_the_declared_topology(self):
        expected_clades = set(TEM_EXPECTED_PHYLOGENETIC_CLADES.values())
        recovered = set(
            get_linkage_clades(
                build_species_phylogeny_linkage(ALL_TEM_SPECIES)["linkage"], ALL_TEM_SPECIES
            )
        )
        assert expected_clades <= recovered


# ---------------------------------------------------------------------------
# assert_ultrametric_divergence_times
# ---------------------------------------------------------------------------


def _valid_three_species_matrix() -> dict:
    """A minimal valid ultrametric matrix: ((A, B):10, C):20."""
    return {
        "A": {"A": 0.0, "B": 10.0, "C": 20.0},
        "B": {"A": 10.0, "B": 0.0, "C": 20.0},
        "C": {"A": 20.0, "B": 20.0, "C": 0.0},
    }


class TestAssertUltrametric:
    def test_accepts_a_valid_matrix(self):
        assert_ultrametric_divergence_times(_valid_three_species_matrix())

    def test_rejects_non_zero_diagonal(self):
        matrix = _valid_three_species_matrix()
        matrix["B"]["B"] = 3.0
        with pytest.raises(ValueError, match="non-zero diagonal"):
            assert_ultrametric_divergence_times(matrix)

    def test_rejects_asymmetric_matrix(self):
        matrix = _valid_three_species_matrix()
        matrix["A"]["B"] = 11.0
        with pytest.raises(ValueError, match="not symmetric"):
            assert_ultrametric_divergence_times(matrix)

    def test_rejects_ultrametric_violation_and_names_the_triple(self):
        # Making all three pairwise distances distinct breaks the three-point
        # condition: the two largest are no longer equal.
        matrix = _valid_three_species_matrix()
        matrix["A"]["C"] = 15.0
        matrix["C"]["A"] = 15.0
        with pytest.raises(ValueError, match="ultrametric") as error:
            assert_ultrametric_divergence_times(matrix)
        assert "'A'" in str(error.value) and "'C'" in str(error.value)

    def test_rejects_missing_species(self):
        with pytest.raises(ValueError, match="missing from the divergence-time matrix"):
            assert_ultrametric_divergence_times(
                _valid_three_species_matrix(), species_list=["A", "B", "Unknown"]
            )

    def test_rejects_missing_pair(self):
        matrix = _valid_three_species_matrix()
        del matrix["A"]["C"]
        with pytest.raises(ValueError, match="is missing"):
            assert_ultrametric_divergence_times(matrix)

    def test_can_check_a_subset_only(self):
        matrix = _valid_three_species_matrix()
        matrix["C"]["C"] = 99.0  # broken, but excluded from the requested subset
        assert_ultrametric_divergence_times(matrix, species_list=["A", "B"])


# ---------------------------------------------------------------------------
# build_species_phylogeny_linkage
# ---------------------------------------------------------------------------


class TestBuildSpeciesPhylogenyLinkage:
    def test_recovers_the_expected_topology(self):
        result = build_species_phylogeny_linkage(ALL_TEM_SPECIES)
        assert linkage_to_newick_string(result["linkage"], ALL_TEM_SPECIES) == EXPECTED_NEWICK

    def test_topology_is_independent_of_input_order(self):
        shuffled = ["Human", "Worm", "Mouse", "KFish", "Droso", "ZFish"]
        result = build_species_phylogeny_linkage(shuffled)
        assert linkage_to_newick_string(result["linkage"], shuffled) == EXPECTED_NEWICK

    def test_leaf_order_follows_the_reference_phylogenetic_order(self):
        result = build_species_phylogeny_linkage(ALL_TEM_SPECIES)
        assert result["leaf_order"] == ALL_TEM_SPECIES

    def test_returns_the_documented_keys(self):
        result = build_species_phylogeny_linkage(ALL_TEM_SPECIES)
        assert set(result.keys()) == {"linkage", "leaf_order", "distance_matrix"}

    def test_time_scale_node_heights_are_divergence_times(self):
        result = build_species_phylogeny_linkage(ALL_TEM_SPECIES)
        node_heights = sorted(result["linkage"][:, 2].tolist())
        assert node_heights == [87.0, 224.0, 429.0, 572.0, 708.0]

    def test_distance_matrix_is_returned_in_requested_order(self):
        result = build_species_phylogeny_linkage(ALL_TEM_SPECIES)
        assert result["distance_matrix"].index.tolist() == ALL_TEM_SPECIES
        assert result["distance_matrix"].loc["Human", "Mouse"] == 87.0

    def test_cladogram_keeps_topology_with_uniform_heights(self):
        time_scaled = build_species_phylogeny_linkage(ALL_TEM_SPECIES, branch_scale="time")
        cladogram = build_species_phylogeny_linkage(ALL_TEM_SPECIES, branch_scale="cladogram")

        assert linkage_to_newick_string(cladogram["linkage"], ALL_TEM_SPECIES) == EXPECTED_NEWICK
        assert cladogram["leaf_order"] == time_scaled["leaf_order"]
        # Only the heights (column 2) differ — the merge structure is identical.
        np.testing.assert_array_equal(
            cladogram["linkage"][:, [0, 1, 3]], time_scaled["linkage"][:, [0, 1, 3]]
        )
        # Every merge sits exactly one unit above its deepest child.
        assert sorted(cladogram["linkage"][:, 2].tolist()) == [1.0, 1.0, 1.0, 2.0, 3.0]

    def test_works_on_a_subset_of_species(self):
        subset = ["Worm", "Mouse", "Human"]
        result = build_species_phylogeny_linkage(subset)
        assert linkage_to_newick_string(result["linkage"], subset) == "(Worm,(Mouse,Human))"

    def test_missing_species_raises_value_error(self):
        with pytest.raises(ValueError, match="No divergence times declared for"):
            build_species_phylogeny_linkage(["Human", "Mouse", "Axolotl"])

    def test_unknown_branch_scale_raises_value_error(self):
        with pytest.raises(ValueError, match="branch_scale must be one of"):
            build_species_phylogeny_linkage(ALL_TEM_SPECIES, branch_scale="logarithmic")

    def test_fewer_than_two_species_raises_value_error(self):
        with pytest.raises(ValueError, match="at least 2 species"):
            build_species_phylogeny_linkage(["Human"])

    def test_allowed_branch_scales_are_documented(self):
        assert set(ALLOWED_BRANCH_SCALES) == {"time", "cladogram"}


# ---------------------------------------------------------------------------
# linkage_to_newick_string / get_linkage_clades
# ---------------------------------------------------------------------------


class TestLinkageDescription:
    def test_newick_string_renders_nested_topology(self):
        linkage_matrix = np.array([
            [0.0, 1.0, 1.0, 2.0],
            [3.0, 2.0, 2.0, 3.0],
        ])
        # Row 1 merges node 3 (the A+B cluster) with leaf 2, in that order.
        assert linkage_to_newick_string(linkage_matrix, ["A", "B", "C"]) == "((A,B),C)"

    def test_clades_exclude_the_root_by_default(self):
        result = build_species_phylogeny_linkage(ALL_TEM_SPECIES)
        clades = get_linkage_clades(result["linkage"], ALL_TEM_SPECIES)
        assert frozenset(ALL_TEM_SPECIES) not in clades
        assert len(clades) == len(ALL_TEM_SPECIES) - 2

    def test_clades_include_the_root_when_requested(self):
        result = build_species_phylogeny_linkage(ALL_TEM_SPECIES)
        clades = get_linkage_clades(result["linkage"], ALL_TEM_SPECIES, include_root=True)
        assert frozenset(ALL_TEM_SPECIES) in clades

    def test_expected_clades_are_recovered_from_the_phylogeny(self):
        result = build_species_phylogeny_linkage(ALL_TEM_SPECIES)
        clades = set(get_linkage_clades(result["linkage"], ALL_TEM_SPECIES))
        assert frozenset({"Worm", "Droso"}) in clades
        assert frozenset({"ZFish", "KFish"}) in clades
        assert frozenset({"Mouse", "Human"}) in clades


# ---------------------------------------------------------------------------
# TEM_FEATURE_SUBSETS / TEM_FEATURE_CATEGORIES / TEM_FEATURE_CATEGORY_COLORS
# ---------------------------------------------------------------------------


class TestFeatureCategories:
    EXPECTED_CATEGORIES = {"Size", "Shape", "Intensity", "Cristae Orientation"}

    def test_subsets_hold_exactly_the_four_declared_categories(self):
        assert set(TEM_FEATURE_SUBSETS) == self.EXPECTED_CATEGORIES

    def test_subsets_only_reference_analytical_feature_columns(self):
        for category, feature_names in TEM_FEATURE_SUBSETS.items():
            for feature_name in feature_names:
                assert feature_name in TEM_FEATURE_COLUMNS, (category, feature_name)
                assert feature_name not in TEM_NON_ANALYTICAL_FEATURES

    def test_subsets_partition_every_analytical_feature_without_overlap(self):
        analytical_features = [
            feature for feature in TEM_FEATURE_COLUMNS
            if feature not in TEM_NON_ANALYTICAL_FEATURES
        ]
        assigned = [name for names in TEM_FEATURE_SUBSETS.values() for name in names]
        assert sorted(assigned) == sorted(analytical_features)
        assert len(assigned) == len(set(assigned))  # no feature in two categories

    def test_categories_is_the_inverse_of_subsets(self):
        expected_inverse = {
            feature_name: category
            for category, feature_names in TEM_FEATURE_SUBSETS.items()
            for feature_name in feature_names
        }
        assert TEM_FEATURE_CATEGORIES == expected_inverse

    def test_category_order_matches_subset_insertion_order(self):
        assert TEM_FEATURE_CATEGORY_ORDER == list(TEM_FEATURE_SUBSETS)

    def test_every_analytical_feature_has_a_category(self):
        analytical_features = [
            feature for feature in TEM_FEATURE_COLUMNS
            if feature not in TEM_NON_ANALYTICAL_FEATURES
        ]
        missing = [f for f in analytical_features if f not in TEM_FEATURE_CATEGORIES]
        assert missing == []

    def test_no_non_analytical_feature_has_a_category(self):
        wrongly_categorised = [
            feature for feature in TEM_NON_ANALYTICAL_FEATURES
            if feature in TEM_FEATURE_CATEGORIES
        ]
        assert wrongly_categorised == []

    def test_no_unknown_feature_is_categorised(self):
        unknown = [f for f in TEM_FEATURE_CATEGORIES if f not in TEM_FEATURE_COLUMNS]
        assert unknown == []

    def test_categories_are_exactly_the_four_declared_ones(self):
        assert set(TEM_FEATURE_CATEGORIES.values()) == self.EXPECTED_CATEGORIES

    def test_every_category_has_a_color_and_a_display_rank(self):
        for category in set(TEM_FEATURE_CATEGORIES.values()):
            assert category in TEM_FEATURE_CATEGORY_COLORS
            assert category in TEM_FEATURE_CATEGORY_ORDER

    def test_colors_are_hex_strings(self):
        for color in TEM_FEATURE_CATEGORY_COLORS.values():
            assert color.startswith("#") and len(color) == 7

    def test_other_fallback_color_exists_for_uncategorised_features(self):
        assert "Other" in TEM_FEATURE_CATEGORY_COLORS
