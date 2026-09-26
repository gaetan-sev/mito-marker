"""
test_analysis_phylo_tanglegram.py

Unit tests for mito_marker.analysis.phylo_tanglegram.

Covers:
  plot_phylo_tanglegram():
    - Returns a dict with exactly the documented keys.
    - Heatmap row order equals the phylogeny leaf order.
    - Columns are grouped into contiguous, deterministic category blocks.
    - An explicit feature_order wins and drops nothing.
    - optimize_leaf_order=True changes only leaf ORDER, never cluster
      membership, and never baker_gamma.
    - nest_aggregate_by is recorded in the returned dict and changes the
      plotted matrix on an unbalanced fixture.
    - standardize_group_means True vs False produce DIFFERENT species
      linkages — under this figure's transposed orientation it is an
      analytical lever, not a cosmetic one.
    - Degenerate input (fewer than 3 species) fails with a clear message.
    - Missing group_by / nest_aggregate_by column raises ValueError.
    - branch_scale changes baker_gamma (cophenetic distances include node
      heights), so it is an analytical choice, not a display option.

  Agreement statistics:
    - baker_gamma is 1.0 when the morphology linkage IS the phylogeny's own.
    - baker_gamma is invariant to node rotation.
    - n_crossings DOES change with node rotation — which is exactly why it is
      descriptive only and baker_gamma is the headline statistic.
    - n_crossings is 0 for identical leaf orders, and counts inverted pairs.
    - The exact permutation p-value enumerates all 6! = 720 permutations, so
      every p-value is a multiple of 1/720 and never below that floor.
    - leaf_order_agreement wording for perfect, inverted-pair and general
      disagreement.

  compute_tanglegram_sensitivity():
    - Returns one row per method x metric x standardization combination.
    - Flags (never drops) method/metric pairs that are not strictly valid.
    - Sorted by baker_gamma descending.
    - Reports which expected clades each combination recovered.

  compute_clade_support():
    - Requires nest_aggregate_by — subject-level resampling is impossible
      without it.
    - Resamples SUBJECTS, not rows: a fixture with one dominant outlier
      subject shows visibly lower support than a balanced fixture.
    - Deterministic for a fixed seed, including n_bootstrap=1.
    - A clade absent from the point estimate is returned as 0.0, not omitted.
    - Support values are fractions in [0, 1].
"""

import numpy as np
import pandas as pd
import pytest
import anndata
import matplotlib

matplotlib.use("Agg")

from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform

from mito_marker.analysis.clustermap_plot import _ANALYSIS_CONFIG_KEY, _apply_node_rotations
from mito_marker.analysis.phylogeny import build_species_phylogeny_linkage, get_linkage_clades
from mito_marker.analysis.phylo_tanglegram import (
    _compute_baker_gamma_with_exact_p_value,
    _count_leaf_order_crossings,
    _describe_leaf_order_agreement,
    _resolve_feature_column_order,
    compute_clade_support,
    compute_tanglegram_sensitivity,
    plot_phylo_tanglegram,
)
from mito_marker.controlled_vocabulary import (
    TEM_FEATURE_CATEGORIES,
    TEM_FEATURE_COLUMNS,
    TEM_NON_ANALYTICAL_FEATURES,
    TEM_SPECIES_PHYLOGENETIC_ORDER,
)

ALL_TEM_SPECIES = list(TEM_SPECIES_PHYLOGENETIC_ORDER)
DOCUMENTED_RESULT_KEYS = {
    "figure",
    "phylogeny_linkage",
    "phylogeny_leaf_order",
    "morphology_linkage",
    "morphology_leaf_order",
    "species_feature_matrix",
    "feature_order",
    "baker_gamma",
    "baker_gamma_p_value",
    "cophenetic_correlation",
    "n_crossings_raw",
    "n_crossings_after_leaf_optimization",
    "leaf_order_agreement",
    "nest_aggregate_by",
    "standardize_group_means",
    "clade_support",
}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


# Species profiles are built so that the three phylogenetic sister pairs are
# also each other's nearest neighbours in morphology space: each pair shares a
# strong random "clade direction", and the two sisters differ only by a small
# random offset. The tree the tests check against is therefore known in
# advance, without relying on real data.
_SISTER_PAIRS = [("Worm", "Droso"), ("ZFish", "KFish"), ("Mouse", "Human")]


def _build_species_profiles(n_features: int, sister_separation: float = 0.18) -> dict:
    """
    Build one feature profile per species with a known three-pair structure.

    Each sister pair gets its own random direction in feature space; the two
    sisters sit on that direction, separated by sister_separation. Random
    directions in many dimensions are nearly orthogonal, so between-pair
    distances are large and within-pair distances are small under both
    correlation and euclidean metrics.
    """
    profile_generator = np.random.default_rng(101)
    profiles = {}
    for pair_index, (first_species, second_species) in enumerate(_SISTER_PAIRS):
        clade_direction = profile_generator.normal(size=n_features) + pair_index * 0.0
        sister_direction = profile_generator.normal(size=n_features) * sister_separation
        profiles[first_species] = clade_direction + sister_direction
        profiles[second_species] = clade_direction - sister_direction
    return profiles


def _make_species_anndata(
    subjects_per_species: int = 5,
    rows_per_subject: int = 8,
    within_subject_noise: float = 0.02,
    between_subject_noise: float = 0.05,
    outlier_species: str = "",
    outlier_shift: float = 0.0,
    outlier_extra_rows: int = 0,
    outlier_toward_species: str = "",
    species: list = None,
    random_state: int = 5,
) -> anndata.AnnData:
    """
    Synthetic multi-species TEM-shaped AnnData with a known morphology tree.

    Subjects scatter around their species profile (see
    _build_species_profiles). outlier_species / outlier_shift /
    outlier_extra_rows introduce ONE dominant subject: displaced along its own
    random direction (not by a constant, which a correlation metric would
    ignore) and optionally contributing many more rows. That is the fixture
    the subject-level bootstrap and the nested-aggregation tests need.
    """
    rng = np.random.default_rng(random_state)
    species_list = list(species) if species is not None else list(ALL_TEM_SPECIES)
    feature_names = list(TEM_FEATURE_COLUMNS)
    n_features = len(feature_names)

    species_profiles = _build_species_profiles(n_features)

    # Every subject profile is drawn BEFORE any row is generated, and each
    # subject's rows come from its own seeded generator. Row counts therefore
    # never shift the random stream, so a fixture built with an outlier is
    # bit-identical to the balanced one everywhere except at that outlier —
    # which is what makes the balanced/dominant comparison below a controlled
    # experiment rather than two unrelated random datasets.
    subject_profiles: dict = {}
    for species_name in species_list:
        for subject_index in range(subjects_per_species):
            subject_profiles[(species_name, subject_index)] = species_profiles[
                species_name
            ] + rng.normal(scale=between_subject_noise, size=n_features)

    if outlier_species:
        if outlier_toward_species:
            # Drag the outlier subject along the line toward another clade.
            # outlier_shift=1.0 puts it exactly on that clade's profile, 3.0
            # well past it. This is the displacement that can actually break a
            # sister pair: a random direction mostly adds distance in an unused
            # dimension, which barely moves a correlation-based tree.
            displacement = (
                species_profiles[outlier_toward_species] - species_profiles[outlier_species]
            )
        else:
            # No target clade: displace along a fixed random direction. A
            # constant offset would be invisible to the correlation metric,
            # which reads only the SHAPE of a profile.
            displacement = np.random.default_rng(909).normal(size=n_features)
            displacement = displacement / np.linalg.norm(displacement)
        subject_profiles[(outlier_species, 0)] = (
            subject_profiles[(outlier_species, 0)] + outlier_shift * displacement
        )

    data_rows = []
    species_labels = []
    subject_labels = []
    for species_index, species_name in enumerate(species_list):
        for subject_index in range(subjects_per_species):
            is_outlier = species_name == outlier_species and subject_index == 0
            n_rows = rows_per_subject + (outlier_extra_rows if is_outlier else 0)
            row_generator = np.random.default_rng([random_state, species_index, subject_index])
            data_rows.append(
                subject_profiles[(species_name, subject_index)]
                + row_generator.normal(scale=within_subject_noise, size=(n_rows, n_features))
            )
            species_labels.extend([species_name] * n_rows)
            subject_labels.extend([f"{species_name}_{subject_index:02d}"] * n_rows)

    obs = pd.DataFrame({
        "specie": species_labels,
        "unique_subject_ID": subject_labels,
    })
    var = pd.DataFrame(index=feature_names)
    var["is_non_analytical"] = [
        feature in TEM_NON_ANALYTICAL_FEATURES for feature in feature_names
    ]

    species_anndata = anndata.AnnData(X=np.vstack(data_rows).astype(np.float64), obs=obs, var=var)
    species_anndata.uns[_ANALYSIS_CONFIG_KEY] = {"active_layer": None, "active_selection": None}
    return species_anndata


@pytest.fixture
def species_anndata() -> anndata.AnnData:
    return _make_species_anndata()


# ---------------------------------------------------------------------------
# plot_phylo_tanglegram — structure
# ---------------------------------------------------------------------------


class TestTanglegramStructure:
    def test_returns_dict_with_documented_keys(self, species_anndata):
        result = plot_phylo_tanglegram(species_anndata, show_merge_diagnostic=False)
        assert set(result.keys()) == DOCUMENTED_RESULT_KEYS

    def test_figure_is_matplotlib_figure(self, species_anndata):
        import matplotlib.figure

        result = plot_phylo_tanglegram(species_anndata, show_merge_diagnostic=False)
        assert isinstance(result["figure"], matplotlib.figure.Figure)

    def test_heatmap_row_order_equals_phylogeny_leaf_order(self, species_anndata):
        result = plot_phylo_tanglegram(species_anndata, show_merge_diagnostic=False)
        expected_leaf_order = build_species_phylogeny_linkage(
            result["species_feature_matrix"].index.tolist()
        )["leaf_order"]
        assert result["phylogeny_leaf_order"] == expected_leaf_order
        assert result["phylogeny_leaf_order"] == ALL_TEM_SPECIES

    def test_non_analytical_features_are_excluded(self, species_anndata):
        result = plot_phylo_tanglegram(species_anndata, show_merge_diagnostic=False)
        for feature in TEM_NON_ANALYTICAL_FEATURES:
            assert feature not in result["feature_order"]

    def test_columns_form_contiguous_category_blocks(self, species_anndata):
        result = plot_phylo_tanglegram(species_anndata, show_merge_diagnostic=False)
        categories = [TEM_FEATURE_CATEGORIES[f] for f in result["feature_order"]]
        seen_blocks = []
        for category in categories:
            if not seen_blocks or seen_blocks[-1] != category:
                seen_blocks.append(category)
        assert len(seen_blocks) == len(set(seen_blocks))
        assert seen_blocks == ["Size", "Shape", "Intensity", "Cristae Orientation"]

    def test_category_ordering_is_deterministic(self, species_anndata):
        first = plot_phylo_tanglegram(species_anndata, show_merge_diagnostic=False)
        second = plot_phylo_tanglegram(species_anndata, show_merge_diagnostic=False)
        assert first["feature_order"] == second["feature_order"]

    def test_explicit_feature_order_wins_and_drops_nothing(self, species_anndata):
        analytical = [f for f in TEM_FEATURE_COLUMNS if f not in TEM_NON_ANALYTICAL_FEATURES]
        requested = [analytical[3], analytical[0]]
        result = plot_phylo_tanglegram(
            species_anndata, feature_order=requested, show_merge_diagnostic=False
        )
        assert result["feature_order"][:2] == requested
        assert sorted(result["feature_order"]) == sorted(analytical)

    def test_missing_group_by_raises_value_error(self, species_anndata):
        with pytest.raises(ValueError, match="not found in .obs"):
            plot_phylo_tanglegram(species_anndata, group_by="nonexistent")

    def test_missing_nest_column_raises_value_error(self, species_anndata):
        with pytest.raises(ValueError, match="nest_aggregate_by column 'nope' not found"):
            plot_phylo_tanglegram(species_anndata, nest_aggregate_by="nope")

    def test_fewer_than_three_species_raises_clear_error(self):
        two_species = _make_species_anndata(species=["Mouse", "Human"])
        with pytest.raises(ValueError, match="at least 3 species"):
            plot_phylo_tanglegram(two_species, show_merge_diagnostic=False)

    def test_cladogram_branch_scale_renders(self, species_anndata):
        result = plot_phylo_tanglegram(
            species_anndata, branch_scale="cladogram", show_merge_diagnostic=False
        )
        assert sorted(result["phylogeny_linkage"][:, 2].tolist()) == [1.0, 1.0, 1.0, 2.0, 3.0]

    def test_branch_scale_changes_baker_gamma(self, species_anndata):
        """
        baker_gamma is built from cophenetic distances, which include node
        HEIGHTS — so the time-scaled and cladogram phylogenies score
        differently even though their topology is identical. branch_scale is
        therefore part of the statistic and must be reported with it, not
        treated as a display option.
        """
        time_scaled = plot_phylo_tanglegram(
            species_anndata, branch_scale="time", show_merge_diagnostic=False
        )
        cladogram = plot_phylo_tanglegram(
            species_anndata, branch_scale="cladogram", show_merge_diagnostic=False
        )
        # Same topology on both sides ...
        assert time_scaled["phylogeny_leaf_order"] == cladogram["phylogeny_leaf_order"]
        np.testing.assert_array_equal(
            time_scaled["morphology_linkage"], cladogram["morphology_linkage"]
        )
        # ... but a different agreement score, because node heights differ.
        assert time_scaled["baker_gamma"] != pytest.approx(cladogram["baker_gamma"])


# ---------------------------------------------------------------------------
# plot_phylo_tanglegram — analytical levers
# ---------------------------------------------------------------------------


class TestAnalyticalLevers:
    def test_nest_aggregate_by_is_recorded_in_the_result(self, species_anndata):
        pooled = plot_phylo_tanglegram(species_anndata, show_merge_diagnostic=False)
        nested = plot_phylo_tanglegram(
            species_anndata, nest_aggregate_by="unique_subject_ID", show_merge_diagnostic=False
        )
        assert pooled["nest_aggregate_by"] is None
        assert nested["nest_aggregate_by"] == ["unique_subject_ID"]

    def test_nest_aggregate_by_changes_the_matrix_on_an_unbalanced_fixture(self):
        unbalanced = _make_species_anndata(
            outlier_species="Mouse", outlier_shift=6.0, outlier_extra_rows=200
        )
        pooled = plot_phylo_tanglegram(unbalanced, show_merge_diagnostic=False)
        nested = plot_phylo_tanglegram(
            unbalanced, nest_aggregate_by="unique_subject_ID", show_merge_diagnostic=False
        )
        assert not np.allclose(
            pooled["species_feature_matrix"].values,
            nested["species_feature_matrix"].values,
        )

    def test_standardization_changes_the_species_tree(self, species_anndata):
        """
        This is the assertion that documents the transpose.

        plot_feature_clustermap() has species as COLUMNS, where z-scoring each
        feature's row cannot affect the feature tree. Here species are the ROWS
        being clustered, so the same z-score reweights every feature to equal
        influence and DOES change the species tree — an analytical choice that
        must be declared, not a display option.
        """
        standardized = plot_phylo_tanglegram(
            species_anndata, standardize_group_means=True, show_merge_diagnostic=False
        )
        unstandardized = plot_phylo_tanglegram(
            species_anndata, standardize_group_means=False, show_merge_diagnostic=False
        )
        assert not np.allclose(
            standardized["species_feature_matrix"].values,
            unstandardized["species_feature_matrix"].values,
        )
        assert not np.allclose(
            standardized["morphology_linkage"][:, 2],
            unstandardized["morphology_linkage"][:, 2],
        )
        assert standardized["standardize_group_means"] is True
        assert unstandardized["standardize_group_means"] is False

    def test_optimize_leaf_order_preserves_cluster_membership(self, species_anndata):
        without = plot_phylo_tanglegram(
            species_anndata, optimize_leaf_order=False, show_merge_diagnostic=False
        )
        with_optimization = plot_phylo_tanglegram(
            species_anndata, optimize_leaf_order=True, show_merge_diagnostic=False
        )
        species_list = without["species_feature_matrix"].index.tolist()
        assert set(get_linkage_clades(without["morphology_linkage"], species_list)) == set(
            get_linkage_clades(with_optimization["morphology_linkage"], species_list)
        )

    def test_optimize_leaf_order_never_changes_baker_gamma(self, species_anndata):
        without = plot_phylo_tanglegram(
            species_anndata, optimize_leaf_order=False, show_merge_diagnostic=False
        )
        with_optimization = plot_phylo_tanglegram(
            species_anndata, optimize_leaf_order=True, show_merge_diagnostic=False
        )
        assert without["baker_gamma"] == pytest.approx(with_optimization["baker_gamma"])

    def test_both_crossing_counts_are_reported(self, species_anndata):
        result = plot_phylo_tanglegram(
            species_anndata, optimize_leaf_order=True, show_merge_diagnostic=False
        )
        assert isinstance(result["n_crossings_raw"], int)
        assert isinstance(result["n_crossings_after_leaf_optimization"], int)

    def test_rotation_out_of_range_raises_value_error(self, species_anndata):
        with pytest.raises(ValueError, match="out of range"):
            plot_phylo_tanglegram(
                species_anndata, rotate_morphology_nodes=[99], show_merge_diagnostic=False
            )


# ---------------------------------------------------------------------------
# Agreement statistics
# ---------------------------------------------------------------------------


def _phylogeny_linkage_for_all_species() -> np.ndarray:
    return build_species_phylogeny_linkage(ALL_TEM_SPECIES)["linkage"]


class TestBakerGamma:
    def test_is_one_when_morphology_linkage_is_the_phylogenys_own(self):
        phylogeny = build_species_phylogeny_linkage(ALL_TEM_SPECIES)
        # Feed the morphology side the phylogeny's OWN distance matrix.
        morphology_linkage = linkage(
            squareform(phylogeny["distance_matrix"].values, checks=False), method="average"
        )
        baker_gamma, _ = _compute_baker_gamma_with_exact_p_value(
            phylogeny["linkage"], morphology_linkage
        )
        assert baker_gamma == pytest.approx(1.0)

    def test_is_invariant_to_node_rotation(self):
        phylogeny_linkage = _phylogeny_linkage_for_all_species()
        morphology_linkage = linkage(
            np.array([
                [0.0, 0.0], [0.2, 0.1], [5.0, 0.0], [5.2, 0.1], [9.0, 0.0], [9.3, 0.1],
            ]),
            method="average",
        )
        baseline_gamma, baseline_p = _compute_baker_gamma_with_exact_p_value(
            phylogeny_linkage, morphology_linkage
        )
        for rotated_row in range(morphology_linkage.shape[0]):
            rotated_gamma, rotated_p = _compute_baker_gamma_with_exact_p_value(
                phylogeny_linkage,
                _apply_node_rotations(morphology_linkage, [rotated_row]),
            )
            assert rotated_gamma == pytest.approx(baseline_gamma)
            assert rotated_p == pytest.approx(baseline_p)

    def test_exact_p_value_enumerates_all_720_permutations(self):
        """
        With 6 species there are 6! = 720 label permutations and every one is
        enumerated, so any p-value is an exact multiple of 1/720 and can never
        fall below that floor (~0.0014).
        """
        phylogeny_linkage = _phylogeny_linkage_for_all_species()
        _, p_value = _compute_baker_gamma_with_exact_p_value(
            phylogeny_linkage, phylogeny_linkage
        )
        assert p_value >= 1 / 720 - 1e-12
        assert p_value * 720 == pytest.approx(round(p_value * 720))

    def test_identical_trees_give_p_equal_to_the_symmetry_count(self):
        """
        Comparing the phylogeny with itself, gamma is at its maximum, so the
        p-value counts exactly the label permutations that leave the tree's
        cophenetic distances unchanged: one swap inside each of the three
        sister pairs, 2 x 2 x 2 = 8 out of 720.
        """
        phylogeny_linkage = _phylogeny_linkage_for_all_species()
        gamma, p_value = _compute_baker_gamma_with_exact_p_value(
            phylogeny_linkage, phylogeny_linkage
        )
        assert gamma == pytest.approx(1.0)
        assert p_value == pytest.approx(8 / 720)


class TestCrossings:
    def test_zero_for_identical_leaf_orders(self):
        assert _count_leaf_order_crossings(ALL_TEM_SPECIES, ALL_TEM_SPECIES) == 0

    def test_counts_a_single_inverted_pair(self):
        inverted = ["Worm", "Droso", "KFish", "ZFish", "Mouse", "Human"]
        assert _count_leaf_order_crossings(ALL_TEM_SPECIES, inverted) == 1

    def test_fully_reversed_order_gives_every_pair(self):
        reversed_order = list(reversed(ALL_TEM_SPECIES))
        n_leaves = len(ALL_TEM_SPECIES)
        assert _count_leaf_order_crossings(ALL_TEM_SPECIES, reversed_order) == (
            n_leaves * (n_leaves - 1) // 2
        )

    def test_changes_with_node_rotation(self, species_anndata):
        """
        Rotating a node leaves the tree identical but changes the drawn leaf
        order, so the crossing count moves. That is precisely why n_crossings
        is descriptive only and baker_gamma is the headline statistic.
        """
        baseline = plot_phylo_tanglegram(
            species_anndata, optimize_leaf_order=False, show_merge_diagnostic=False
        )
        rotated = plot_phylo_tanglegram(
            species_anndata,
            optimize_leaf_order=False,
            rotate_morphology_nodes=[baseline["morphology_linkage"].shape[0] - 1],
            show_merge_diagnostic=False,
        )
        assert rotated["n_crossings_after_leaf_optimization"] != (
            baseline["n_crossings_after_leaf_optimization"]
        )
        # ... while baker_gamma, which reads only topology, does not move.
        assert rotated["baker_gamma"] == pytest.approx(baseline["baker_gamma"])


class TestLeafOrderAgreement:
    def test_perfect_agreement_wording(self):
        summary = _describe_leaf_order_agreement(ALL_TEM_SPECIES, ALL_TEM_SPECIES)
        assert summary == "6/6 leaves in phylogenetic order (perfect agreement)"

    def test_inverted_pair_wording(self):
        inverted = ["Worm", "Droso", "KFish", "ZFish", "Mouse", "Human"]
        summary = _describe_leaf_order_agreement(ALL_TEM_SPECIES, inverted)
        assert summary == "4/6 leaves in phylogenetic order (ZFish and KFish inverted)"

    def test_general_disagreement_lists_the_misplaced_leaves(self):
        shuffled = ["Droso", "ZFish", "Worm", "KFish", "Mouse", "Human"]
        summary = _describe_leaf_order_agreement(ALL_TEM_SPECIES, shuffled)
        assert summary.startswith("3/6 leaves in phylogenetic order (out of place: ")
        assert "Worm" in summary and "Droso" in summary and "ZFish" in summary


# ---------------------------------------------------------------------------
# _resolve_feature_column_order
# ---------------------------------------------------------------------------


class TestResolveFeatureColumnOrder:
    def test_groups_by_category_in_declared_order(self):
        present = ["Kurtosis", "CristaeOrientation_Area", "Mito_Area", "Mito_AR"]
        assert _resolve_feature_column_order(present, None, True) == [
            "Mito_Area", "Mito_AR", "Kurtosis", "CristaeOrientation_Area"
        ]

    def test_explicit_order_wins_and_appends_the_rest(self):
        present = ["Mito_Area", "Mito_AR", "Kurtosis"]
        assert _resolve_feature_column_order(present, ["Kurtosis"], True) == [
            "Kurtosis", "Mito_Area", "Mito_AR"
        ]

    def test_grouping_disabled_keeps_the_incoming_order(self):
        present = ["Kurtosis", "Mito_Area"]
        assert _resolve_feature_column_order(present, None, False) == present

    def test_uncategorised_feature_is_appended_last(self):
        present = ["Unknown_Feature", "Mito_Area"]
        assert _resolve_feature_column_order(present, None, True) == [
            "Mito_Area", "Unknown_Feature"
        ]


# ---------------------------------------------------------------------------
# compute_tanglegram_sensitivity
# ---------------------------------------------------------------------------


class TestTanglegramSensitivity:
    def test_returns_one_row_per_combination(self, species_anndata):
        table = compute_tanglegram_sensitivity(
            species_anndata,
            cluster_methods=("average", "ward"),
            cluster_metrics=("correlation", "euclidean"),
            standardize_options=(True, False),
        )
        assert len(table) == 2 * 2 * 2

    def test_has_the_documented_columns(self, species_anndata):
        table = compute_tanglegram_sensitivity(
            species_anndata,
            cluster_methods=("average",),
            cluster_metrics=("correlation",),
            standardize_options=(True,),
        )
        for column in [
            "cluster_method", "cluster_metric", "standardize_group_means",
            "is_valid_combination", "baker_gamma", "baker_gamma_p_value",
            "n_crossings_raw", "newick_topology", "n_expected_clades_recovered", "note",
            "recovered_(Worm,Droso)", "recovered_(ZFish,KFish)", "recovered_(Mouse,Human)",
        ]:
            assert column in table.columns

    def test_flags_rather_than_drops_invalid_method_metric_pairs(self, species_anndata):
        table = compute_tanglegram_sensitivity(
            species_anndata,
            cluster_methods=("ward",),
            cluster_metrics=("correlation", "euclidean"),
            standardize_options=(True,),
        )
        assert len(table) == 2  # nothing dropped
        invalid = table[table["cluster_metric"] == "correlation"].iloc[0]
        assert bool(invalid["is_valid_combination"]) is False
        assert "euclidean" in invalid["note"]
        valid = table[table["cluster_metric"] == "euclidean"].iloc[0]
        assert bool(valid["is_valid_combination"]) is True

    def test_sorted_by_baker_gamma_descending(self, species_anndata):
        table = compute_tanglegram_sensitivity(
            species_anndata,
            cluster_methods=("average", "single"),
            cluster_metrics=("correlation", "euclidean"),
            standardize_options=(True, False),
        )
        scored = table["baker_gamma"].dropna().tolist()
        assert scored == sorted(scored, reverse=True)

    def test_recovers_the_expected_clades_on_the_designed_fixture(self, species_anndata):
        table = compute_tanglegram_sensitivity(
            species_anndata,
            cluster_methods=("average",),
            cluster_metrics=("correlation",),
            standardize_options=(True,),
        )
        row = table.iloc[0]
        assert int(row["n_expected_clades_recovered"]) == 3
        assert row["newick_topology"] != ""


# ---------------------------------------------------------------------------
# compute_clade_support
# ---------------------------------------------------------------------------


class TestCladeSupport:
    def test_requires_nest_aggregate_by(self, species_anndata):
        with pytest.raises(ValueError, match="requires nest_aggregate_by"):
            compute_clade_support(species_anndata, n_bootstrap=5)

    def test_support_values_are_fractions(self, species_anndata):
        support = compute_clade_support(
            species_anndata, nest_aggregate_by="unique_subject_ID", n_bootstrap=20
        )
        assert all(0.0 <= value <= 1.0 for value in support.values())

    def test_is_deterministic_for_a_fixed_seed(self, species_anndata):
        first = compute_clade_support(
            species_anndata, nest_aggregate_by="unique_subject_ID",
            n_bootstrap=25, random_state=7,
        )
        second = compute_clade_support(
            species_anndata, nest_aggregate_by="unique_subject_ID",
            n_bootstrap=25, random_state=7,
        )
        assert first == second

    def test_single_replicate_is_deterministic(self, species_anndata):
        first = compute_clade_support(
            species_anndata, nest_aggregate_by="unique_subject_ID",
            n_bootstrap=1, random_state=3,
        )
        second = compute_clade_support(
            species_anndata, nest_aggregate_by="unique_subject_ID",
            n_bootstrap=1, random_state=3,
        )
        assert first == second
        # With one replicate every observed clade is at 100% or absent.
        assert set(first.values()) <= {0.0, 1.0}

    def test_expected_clade_absent_from_the_estimate_is_reported_as_zero(self):
        """A clade that never appears must be reported at 0.0, not omitted."""
        # Anchors deliberately scrambled so the sister pairs are NOT nearest
        # neighbours: the phylogenetic clades should essentially never form.
        scrambled = _make_species_anndata(between_subject_noise=0.02)
        scrambled_x = scrambled.X.copy()
        species_values = scrambled.obs["specie"].values
        for species_name, shift in zip(ALL_TEM_SPECIES, [0, 40, 5, 45, 10, 50]):
            scrambled_x[species_values == species_name] += shift
        scrambled.X = scrambled_x

        support = compute_clade_support(
            scrambled, nest_aggregate_by="unique_subject_ID",
            n_bootstrap=30, random_state=1, cluster_metric="euclidean",
        )
        for clade_key in ["(Worm,Droso)", "(ZFish,KFish)", "(Mouse,Human)"]:
            assert clade_key in support
        assert min(support[k] for k in ["(Worm,Droso)", "(ZFish,KFish)", "(Mouse,Human)"]) == 0.0

    def test_resamples_subjects_so_one_dominant_subject_lowers_support(self):
        """
        Resampling happens over SUBJECTS, so a species carrying one atypical
        subject has an unstable mean: whenever that subject is drawn more than
        once the species migrates and its clade breaks. A balanced fixture,
        where every subject sits near its species profile, must show visibly
        higher support.

        If resampling were done over rows instead, the outlier's contribution
        would be fixed at its share of the rows and the species mean would
        barely move between replicates — the two fixtures would then look
        almost identical, which is what this test exists to rule out. The
        dominant fixture deliberately also gives the outlier 400 extra rows,
        so a row-level bootstrap would make it MORE stable, not less.

        The untouched (Worm,Droso) clade must keep full support in both
        fixtures, showing the effect is localised to the species carrying the
        outlier rather than a global loss of resolution.
        """
        balanced = _make_species_anndata(random_state=11)
        dominant = _make_species_anndata(
            outlier_species="Mouse",
            outlier_toward_species="ZFish",
            outlier_shift=3.0,
            outlier_extra_rows=400,
            random_state=11,
        )

        balanced_support = compute_clade_support(
            balanced, nest_aggregate_by="unique_subject_ID", n_bootstrap=150, random_state=2
        )
        dominant_support = compute_clade_support(
            dominant, nest_aggregate_by="unique_subject_ID", n_bootstrap=150, random_state=2
        )

        assert balanced_support["(Mouse,Human)"] > dominant_support["(Mouse,Human)"] + 0.30
        # The clade with no outlier is unaffected — this is not a global effect.
        assert balanced_support["(Worm,Droso)"] == pytest.approx(1.0)
        assert dominant_support["(Worm,Droso)"] == pytest.approx(1.0)
