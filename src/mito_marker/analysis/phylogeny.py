"""
phylogeny.py

Builds the KNOWN species phylogeny as a scipy linkage matrix, so it can be
drawn by exactly the same scipy.cluster.hierarchy.dendrogram() machinery
already used for data-derived trees elsewhere in this package.

The trick that keeps this dependency-free
-----------------------------------------
Divergence times are ULTRAMETRIC: every living species has had exactly the
same amount of time to evolve since any shared ancestor, so every leaf sits
at the same distance from the root. For an ultrametric distance matrix,
average linkage (UPGMA) is not an approximation — it reconstructs the
generating tree EXACTLY, both its topology and its node heights. So

    linkage(squareform(divergence_time_matrix), method="average")

is the true phylogeny, with linkage[:, 2] holding divergence times in Mya.
No phytools, ete3, Bio.Phylo, dendropy or R dependency is required, and the
package's Google Colab compatibility (CLAUDE.md §1) is preserved.

That exactness is entirely conditional on the matrix really being
ultrametric. A matrix that is not produces a silently DIFFERENT tree rather
than an error, so assert_ultrametric_divergence_times() checks the property
and fails loudly before any tree is built.

Public API:
    build_species_phylogeny_linkage(species_list, branch_scale="time")
    assert_ultrametric_divergence_times(divergence_time_matrix)

Typical usage:
    from mito_marker.analysis import build_species_phylogeny_linkage
    phylogeny = build_species_phylogeny_linkage(["Worm", "Droso", "ZFish",
                                                 "KFish", "Mouse", "Human"])
    print(phylogeny["leaf_order"])
"""

from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import squareform

from mito_marker.controlled_vocabulary import (
    TEM_SPECIES_DIVERGENCE_TIME_MYA,
    TEM_SPECIES_PHYLOGENETIC_ORDER,
)

ALLOWED_BRANCH_SCALES: Dict[str, str] = {
    "time": "Node heights are divergence times in millions of years (Mya).",
    "cladogram": "Topology only — every merge step is one unit high.",
}

# Tolerance (in Mya) used when checking symmetry and the ultrametric
# three-point condition. Published divergence-time estimates are quoted to
# whole millions of years, so anything above this is a real inconsistency and
# not a floating-point artifact.
_ULTRAMETRIC_TOLERANCE_MYA: float = 1e-6


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def assert_ultrametric_divergence_times(
    divergence_time_matrix: Dict[str, Dict[str, float]],
    species_list: Optional[Sequence[str]] = None,
    tolerance: float = _ULTRAMETRIC_TOLERANCE_MYA,
) -> None:
    """
    Fail loudly unless a divergence-time table is a valid ultrametric matrix.

    Three properties are required, and each one is checked separately so the
    error message says exactly which pair of species broke it:

      1. Zero diagonal — a species diverged from itself 0 Mya ago.
      2. Symmetry — d(a, b) == d(b, a).
      3. The ultrametric (three-point) condition — for EVERY triple of
         species, the two largest of the three pairwise distances must be
         equal. In tree terms: of any three species, two are always each
         other's closer relatives and both are equally distant from the
         third, because they meet it at the same ancestral node.

    Property 3 is the one that matters. It is what guarantees that average
    linkage reconstructs the intended tree exactly instead of a plausible
    but different one — and it is the one an innocent-looking manual edit to
    a single number will silently break.

    Arguments:
        divergence_time_matrix: Nested dict of pairwise divergence times in
            Mya, e.g. TEM_SPECIES_DIVERGENCE_TIME_MYA.
        species_list: Optional subset of species to check. Defaults to every
            species present as a top-level key.
        tolerance: Absolute tolerance in Mya for all three checks.

    Returns:
        None. Raises on the first violation found.

    Raises:
        ValueError: If a species is missing from the matrix, or if the zero
            diagonal, symmetry, or ultrametric condition is violated.
    """
    species = list(species_list) if species_list is not None else list(divergence_time_matrix.keys())

    for species_a in species:
        if species_a not in divergence_time_matrix:
            raise ValueError(
                f"Species '{species_a}' is missing from the divergence-time matrix. "
                f"Available species: {sorted(divergence_time_matrix.keys())}"
            )
        for species_b in species:
            if species_b not in divergence_time_matrix[species_a]:
                raise ValueError(
                    f"Divergence time for the pair ('{species_a}', '{species_b}') is missing "
                    f"from the divergence-time matrix. Every pair must be declared."
                )

    for species_a in species:
        diagonal_value = float(divergence_time_matrix[species_a][species_a])
        if abs(diagonal_value) > tolerance:
            raise ValueError(
                f"Divergence-time matrix has a non-zero diagonal: "
                f"d('{species_a}', '{species_a}') = {diagonal_value} Mya, expected 0."
            )

    for species_a in species:
        for species_b in species:
            forward = float(divergence_time_matrix[species_a][species_b])
            backward = float(divergence_time_matrix[species_b][species_a])
            if abs(forward - backward) > tolerance:
                raise ValueError(
                    f"Divergence-time matrix is not symmetric: "
                    f"d('{species_a}', '{species_b}') = {forward} but "
                    f"d('{species_b}', '{species_a}') = {backward}."
                )

    for index_a, species_a in enumerate(species):
        for index_b in range(index_a + 1, len(species)):
            species_b = species[index_b]
            for index_c in range(index_b + 1, len(species)):
                species_c = species[index_c]
                pairwise = sorted([
                    float(divergence_time_matrix[species_a][species_b]),
                    float(divergence_time_matrix[species_a][species_c]),
                    float(divergence_time_matrix[species_b][species_c]),
                ])
                # Ultrametric: the two LARGEST of the three must be equal.
                if abs(pairwise[1] - pairwise[2]) > tolerance:
                    raise ValueError(
                        f"Divergence-time matrix violates the ultrametric (three-point) "
                        f"condition for the triple ('{species_a}', '{species_b}', "
                        f"'{species_c}'): sorted pairwise distances are {pairwise} Mya — "
                        f"the two largest must be equal. Average linkage would silently "
                        f"build a DIFFERENT tree from this matrix. Fix the table in "
                        f"controlled_vocabulary.py before using it."
                    )


def build_species_phylogeny_linkage(
    species_list: List[str],
    branch_scale: str = "time",
) -> Dict[str, object]:
    """
    Build the known species phylogeny as a scipy linkage matrix.

    Reads pairwise divergence times from TEM_SPECIES_DIVERGENCE_TIME_MYA
    (controlled_vocabulary.py), validates that they are ultrametric, and runs
    average linkage (UPGMA) on them. Because the input is ultrametric this
    returns the true phylogeny exactly — topology and node heights — not an
    approximation of it. See the module docstring.

    Display order: the returned tree's nodes are rotated so its leaves follow
    TEM_SPECIES_PHYLOGENETIC_ORDER as closely as the topology allows. That is
    a FIXED, a-priori reference order declared in controlled_vocabulary.py —
    it is never derived from measured data, so this rotation cannot tune the
    figure toward any hypothesis under test. Rotating a node only swaps which
    child is drawn first; it never changes which species belong to which
    clade.

    Arguments:
        species_list: Species to include. Every one must be present in
            TEM_SPECIES_DIVERGENCE_TIME_MYA. Order is irrelevant to the
            result — the tree, not the input order, determines leaf order.
        branch_scale: "time" (default) draws node heights as divergence times
            in Mya, which is the honest scale but renders the Ecdysozoa node
            (682 Mya) and the bilaterian root (685 Mya) almost on top of each
            other. "cladogram" keeps the identical topology but sets every
            merge one unit above its deepest child, so all branch lengths are
            equal and the deep structure stays readable. Only the heights
            (column 2 of the linkage) differ between the two.

    Returns:
        Dict with keys:
          "linkage":         np.ndarray, scipy linkage matrix of shape
                              (n_species - 1, 4), indexed against the ORDER
                              OF species_list as passed in.
          "leaf_order":      List[str], species in dendrogram leaf order.
          "distance_matrix": pd.DataFrame, the square divergence-time matrix
                              (Mya) actually used, rows/columns in
                              species_list order.

    Raises:
        ValueError: If branch_scale is not "time" or "cladogram", if fewer
            than 2 species are given, if any species is absent from
            TEM_SPECIES_DIVERGENCE_TIME_MYA, or if the divergence-time matrix
            is not ultrametric.
    """
    if branch_scale not in ALLOWED_BRANCH_SCALES:
        raise ValueError(
            f"branch_scale must be one of {sorted(ALLOWED_BRANCH_SCALES)}, got '{branch_scale}'."
        )

    if len(species_list) < 2:
        raise ValueError(
            f"build_species_phylogeny_linkage() needs at least 2 species to build a tree, "
            f"got {len(species_list)}: {list(species_list)}."
        )

    missing_species = [
        species for species in species_list if species not in TEM_SPECIES_DIVERGENCE_TIME_MYA
    ]
    if missing_species:
        raise ValueError(
            f"No divergence times declared for {missing_species}. Available species: "
            f"{sorted(TEM_SPECIES_DIVERGENCE_TIME_MYA.keys())}. Add the missing rows and "
            f"columns to TEM_SPECIES_DIVERGENCE_TIME_MYA in controlled_vocabulary.py "
            f"(the matrix must stay symmetric, zero-diagonal and ultrametric)."
        )

    assert_ultrametric_divergence_times(TEM_SPECIES_DIVERGENCE_TIME_MYA, species_list)

    distance_matrix = pd.DataFrame(
        [
            [float(TEM_SPECIES_DIVERGENCE_TIME_MYA[row][column]) for column in species_list]
            for row in species_list
        ],
        index=list(species_list),
        columns=list(species_list),
    )

    phylogeny_linkage = linkage(squareform(distance_matrix.values, checks=False), method="average")

    if branch_scale == "cladogram":
        phylogeny_linkage = _rescale_linkage_to_cladogram(phylogeny_linkage)

    phylogeny_linkage = _rotate_linkage_towards_reference_order(
        phylogeny_linkage, list(species_list), TEM_SPECIES_PHYLOGENETIC_ORDER
    )

    leaf_order = [species_list[index] for index in leaves_list(phylogeny_linkage)]

    print("-" * 60)
    print(f"KNOWN PHYLOGENY — {len(species_list)} species, branch_scale='{branch_scale}'")
    print(f"  {ALLOWED_BRANCH_SCALES[branch_scale]}")
    print(f"  Divergence times (Mya): min={distance_matrix.values[distance_matrix.values > 0].min():.1f}, "
          f"max={distance_matrix.values.max():.1f}")
    print(f"  Ultrametric check: PASSED ({len(species_list)} species)")
    print(f"  Leaf order: {leaf_order}")
    print(f"  Newick: {linkage_to_newick_string(phylogeny_linkage, list(species_list))}")
    print("-" * 60)

    return {
        "linkage": phylogeny_linkage,
        "leaf_order": leaf_order,
        "distance_matrix": distance_matrix,
    }


def linkage_to_newick_string(
    linkage_matrix: np.ndarray,
    leaf_labels: List[str],
) -> str:
    """
    Render a linkage matrix's topology as a parenthesised (Newick-ish) string.

    Branch lengths are omitted — this is a compact, printable, comparable
    description of TOPOLOGY only, used in QC output and in the
    compute_tanglegram_sensitivity() table where two trees must be compared
    at a glance.

    Arguments:
        linkage_matrix: scipy linkage matrix of shape (n_leaves - 1, 4).
        leaf_labels: Leaf labels indexed as the linkage matrix indexes them.

    Returns:
        A string such as "((Worm,Droso),((ZFish,KFish),(Mouse,Human)))".
    """
    node_strings: List[str] = list(leaf_labels)
    for row in linkage_matrix:
        left_index, right_index = int(row[0]), int(row[1])
        node_strings.append(f"({node_strings[left_index]},{node_strings[right_index]})")
    return node_strings[-1]


def get_linkage_clades(
    linkage_matrix: np.ndarray,
    leaf_labels: List[str],
    include_root: bool = False,
) -> List[frozenset]:
    """
    List the clade (leaf set) produced by every internal node of a tree.

    A "clade" here is simply the set of leaves sitting under one internal
    node. Two trees can be compared clade by clade regardless of how their
    nodes happen to be drawn left-to-right, which is what makes this the
    right unit for bootstrap support and for checking whether an expected
    grouping was recovered.

    Arguments:
        linkage_matrix: scipy linkage matrix of shape (n_leaves - 1, 4).
        leaf_labels: Leaf labels indexed as the linkage matrix indexes them.
        include_root: When False (default), the final merge is omitted — the
            root clade contains every leaf in every possible tree, so it
            carries no information.

    Returns:
        List of frozensets of leaf labels, one per retained internal node.
    """
    node_clades: List[frozenset] = [frozenset({label}) for label in leaf_labels]
    for row in linkage_matrix:
        left_index, right_index = int(row[0]), int(row[1])
        node_clades.append(node_clades[left_index] | node_clades[right_index])

    internal_clades = node_clades[len(leaf_labels):]
    if not include_root and internal_clades:
        internal_clades = internal_clades[:-1]
    return internal_clades


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _rescale_linkage_to_cladogram(linkage_matrix: np.ndarray) -> np.ndarray:
    """
    Replace node heights with level ranks, keeping the topology untouched.

    Every leaf sits at height 0 and every internal node is placed exactly one
    unit above its deepest child, so all branch steps are the same length.
    This is the standard cladogram rendering: it says "A and B are each
    other's closest relatives" without claiming anything about when they
    split. Only column 2 of the linkage matrix changes.

    Arguments:
        linkage_matrix: scipy linkage matrix of shape (n_leaves - 1, 4).

    Returns:
        A copy of linkage_matrix with level-rank heights.
    """
    rescaled_matrix = linkage_matrix.copy()
    n_leaves = rescaled_matrix.shape[0] + 1
    node_heights = np.zeros(n_leaves + rescaled_matrix.shape[0], dtype=np.float64)

    for row_index, row in enumerate(rescaled_matrix):
        left_index, right_index = int(row[0]), int(row[1])
        merged_height = max(node_heights[left_index], node_heights[right_index]) + 1.0
        node_heights[n_leaves + row_index] = merged_height
        rescaled_matrix[row_index, 2] = merged_height

    return rescaled_matrix


def _rotate_linkage_towards_reference_order(
    linkage_matrix: np.ndarray,
    leaf_labels: List[str],
    reference_order: Sequence[str],
) -> np.ndarray:
    """
    Rotate nodes so leaves follow a FIXED reference order where the tree allows.

    At each node, the child whose earliest-ranked leaf comes first in
    reference_order is drawn first. This is a pure display rotation — leaf
    membership is untouched — and it is driven by a hard-coded, a-priori
    order (TEM_SPECIES_PHYLOGENETIC_ORDER), never by measured data, so it
    cannot tune the figure toward a result.

    Arguments:
        linkage_matrix: scipy linkage matrix of shape (n_leaves - 1, 4).
        leaf_labels: Leaf labels indexed as the linkage matrix indexes them.
        reference_order: The desired leaf order. Labels absent from it are
            ranked after every listed label, in leaf_labels order, so the
            result stays deterministic.

    Returns:
        A rotated copy of linkage_matrix.
    """
    rotated_matrix = linkage_matrix.copy()
    n_leaves = len(leaf_labels)

    reference_rank = {label: rank for rank, label in enumerate(reference_order)}
    node_rank = np.array([
        reference_rank.get(label, len(reference_order) + index)
        for index, label in enumerate(leaf_labels)
    ], dtype=np.float64)
    node_rank = np.concatenate([node_rank, np.zeros(rotated_matrix.shape[0])])

    for row_index, row in enumerate(rotated_matrix):
        left_index, right_index = int(row[0]), int(row[1])
        if node_rank[left_index] > node_rank[right_index]:
            rotated_matrix[row_index, [0, 1]] = rotated_matrix[row_index, [1, 0]]
            left_index, right_index = right_index, left_index
        node_rank[n_leaves + row_index] = min(node_rank[left_index], node_rank[right_index])

    return rotated_matrix
