"""
phylo_tanglegram.py

Dual-tree (tanglegram) figure comparing the KNOWN species phylogeny with the
morphology tree derived from the data, with the species x feature heatmap
between them and connector lines joining matching leaves.

    +----------+  +---------------------------+  +----------+  +-+
    | phylogeny|  |  species x features       |  |morphology|  |c|
    |  (known) |--|         heatmap           |--|dendrogram|  |b|
    +----------+  +---------------------------+  +----------+  +-+
      leaves in       rows locked to the           leaves in     colour
      tree order      LEFT tree's leaf order       its own       bar
                                                   cluster order

WHAT THIS FIGURE IS FOR
-----------------------
This figure TESTS the hypothesis that unsupervised clustering of
mitochondrial morphology recovers the known phylogeny — and it must be
equally capable of showing that the hypothesis is false. If the morphology
tree does not match the phylogeny, the correct deliverable is a figure that
shows the mismatch clearly. Do not re-parameterise the analysis to improve
agreement.

That is also why the data-derived dendrogram is NOT replaced by the known
phylogeny: locking the row order to the phylogeny and calling the result a
recovery would assume the very thing the panel claims to demonstrate.
Crossings in the connector lines are a feature, not a defect — they show
honestly where morphology departs from phylogeny.

READING THE STATISTICS
----------------------
baker_gamma is the headline number. It is computed from the two trees'
COPHENETIC distances, so it depends only on topology and node heights and is
completely invariant to how the leaves happen to be drawn left-to-right. It
cannot be improved by rotating a node.

n_crossings is descriptive only. It depends on display order, so a purely
cosmetic rotation changes it. Both the raw value and the value after the
optional leaf-order optimisation are returned, so the effect of that
cosmetic step stays visible instead of being absorbed.

One caveat on baker_gamma: cophenetic distances include NODE HEIGHTS, so
branch_scale changes it. With branch_scale="time" it compares topology and
divergence times; with "cladogram" it compares topology and depth ranks
only, which usually scores higher because the near-polytomy at the
bilaterian root stops penalising the comparison. Report which scale was
used alongside the number — it is not a free display choice.

Two companion functions guard against reading too much into a 6-leaf tree:
    compute_tanglegram_sensitivity()  — does the result survive other
                                        clustering method/metric choices, or
                                        does it exist only under one?
    compute_clade_support()           — bootstrap over SUBJECTS: is each
                                        clade stable, or is it noise?

Public API:
    plot_phylo_tanglegram(anndata_object, ...)
    compute_tanglegram_sensitivity(anndata_object, ...)
    compute_clade_support(anndata_object, ...)

Typical usage:
    from mito_marker.analysis import plot_phylo_tanglegram, compute_clade_support
    support = compute_clade_support(tem_anndata, nest_aggregate_by="unique_subject_ID")
    result = plot_phylo_tanglegram(
        tem_anndata,
        nest_aggregate_by="unique_subject_ID",
        support_values=support,
    )
    print(result["baker_gamma"], result["baker_gamma_p_value"])
"""

from itertools import permutations
from typing import Dict, List, Optional, Sequence, Tuple, Union

import anndata
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle
from scipy.cluster.hierarchy import cophenet, dendrogram, linkage, optimal_leaf_ordering
from scipy.spatial.distance import pdist, squareform
from scipy.stats import rankdata

from mito_marker.analysis._aggregation import (
    _compute_group_mean_matrix,
    _nested_group_aggregate,
    _resolve_nest_aggregate_columns,
)
from mito_marker.analysis.clustermap_plot import (
    _apply_node_rotations,
    _get_data_and_channels,
    _get_group_labels,
    _print_merge_diagnostic,
    _standardize_group_means,
)
from mito_marker.analysis.colors import get_color_for_value
from mito_marker.analysis.phylogeny import (
    build_species_phylogeny_linkage,
    get_linkage_clades,
    linkage_to_newick_string,
)
from mito_marker.controlled_vocabulary import (
    PREFERRED_CONDITION_COLORS,
    TEM_EXPECTED_PHYLOGENETIC_CLADES,
    TEM_FEATURE_CATEGORIES,
    TEM_FEATURE_CATEGORY_COLORS,
    TEM_FEATURE_CATEGORY_ORDER,
    TEM_SPECIES_PHYLOGENETIC_ORDER,
)

# scipy's dendrogram() places leaf k of its returned 'ivl' list at
# coordinate 10*k + 5 on the leaf axis, and sets that axis to [0, 10*n].
# Every panel of this figure is drawn in that same coordinate system so the
# two trees, the heatmap rows and the connector lines line up exactly.
_DENDROGRAM_LEAF_SPACING: float = 10.0

# Minimum number of species below which a tanglegram carries no information:
# with 2 leaves both trees have exactly one possible topology, so agreement
# between them is guaranteed and meaningless.
_MINIMUM_SPECIES_FOR_TANGLEGRAM: int = 3

# Above this many species, enumerating every label permutation for the exact
# Baker's gamma p-value stops being tractable (9! = 362,880).
_MAXIMUM_SPECIES_FOR_EXACT_PERMUTATION: int = 8

# Bootstrap clade support below this fraction is drawn de-emphasised, because
# a branch supported by fewer than half the replicates should not be read as
# a finding.
_WEAK_SUPPORT_THRESHOLD: float = 0.50

# Fraction of the left tree panel's width left empty past its leaf tips, so
# the tree does not collide with the species names drawn on the heatmap.
_LEAF_LABEL_GUTTER_FRACTION: float = 0.45

# Fraction of the connector panel's width the connector lines actually span.
# The remainder is left blank for the morphology tree's leaf labels, which
# matplotlib anchors on that tree's left spine and draws leftward.
_CONNECTOR_LINE_SPAN: float = 0.62


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def plot_phylo_tanglegram(
    anndata_object: anndata.AnnData,
    group_by: str = "specie",
    nest_aggregate_by: Optional[Union[str, List[str]]] = None,
    standardize_group_means: bool = True,
    cluster_method: str = "average",
    cluster_metric: str = "correlation",
    branch_scale: str = "time",
    feature_order: Optional[List[str]] = None,
    group_features_by_category: bool = True,
    optimize_leaf_order: bool = True,
    rotate_morphology_nodes: Optional[List[int]] = None,
    show_merge_diagnostic: bool = True,
    support_values: Optional[Dict[str, float]] = None,
    title: str = "",
    figsize: Tuple[float, float] = (14.0, 5.0),
    colormap: str = "RdBu_r",
) -> Dict[str, object]:
    """
    Draw a tanglegram: known phylogeny | heatmap | data-derived morphology tree.

    The left tree is the known species phylogeny, reconstructed exactly from
    divergence times (see analysis/phylogeny.py). The right tree is computed
    from the measured morphology with cluster_method / cluster_metric. The
    heatmap rows are locked to the LEFT tree's leaf order, and connector
    lines join each heatmap row to that species' leaf in the right tree, so
    crossings show exactly where morphology and phylogeny disagree.

    This figure tests the "morphology recovers the phylogeny" hypothesis; it
    is meant to be equally able to falsify it. A low baker_gamma is a valid
    result, not a problem to be parameterised away.

    Arguments:
        anndata_object: AnnData with .obs[group_by] and, optionally,
            .uns['analysis_config']['active_layer'] and
            .uns['color_palette'] (used for the leaf label colors).
        group_by: .obs column holding the species (default "specie").
        nest_aggregate_by: Optional .obs column, or list of columns ordered
            COARSEST to FINEST, giving every subject one vote in its species
            profile instead of letting the most heavily imaged subject
            dominate. Same argument, semantics and ordering convention as
            plot_radar() and plot_feature_clustermap(). ADR-011 §3 makes this
            the recommended default for any multi-species comparison with
            unequal event counts; it is left at None only so that omitting it
            never silently changes an existing call's output.
        standardize_group_means: THIS IS AN ANALYTICAL CHOICE HERE, NOT A
            COSMETIC ONE — declare it in the Methods section. In
            plot_feature_clustermap() species are COLUMNS, so z-scoring each
            feature's row across species cannot affect the feature tree. This
            figure TRANSPOSES that: species are the observations being
            clustered, so z-scoring each feature across species reweights
            every feature to equal influence and therefore DOES change the
            species tree and baker_gamma. True (default) gives every
            morphological feature the same say regardless of its raw scale or
            effect size; False lets features with large absolute spread
            dominate the distance. Report which one was used — the returned
            dict records it under "standardize_group_means".
        cluster_method: scipy linkage method for the morphology tree
            (default "average"). See compute_tanglegram_sensitivity() before
            treating any single choice as the result.
        cluster_metric: scipy distance metric for the morphology tree
            (default "correlation").
        branch_scale: "time" (default) draws the phylogeny's node heights as
            divergence times in Mya; "cladogram" draws topology only with
            equal branch lengths, which is more readable because the
            Ecdysozoa node (682 Mya) and the bilaterian root (685 Mya) are
            nearly a polytomy. NOTE: this is not purely a display option —
            baker_gamma is computed from cophenetic distances, which include
            node heights, so switching scale changes it. Report the scale
            together with the statistic.
        feature_order: Optional explicit column order. When given it wins
            over the category ordering; the category strip then simply
            reflects whatever category runs appear in that order.
        group_features_by_category: When True (default), columns are ordered
            by TEM_FEATURE_CATEGORIES (Size, Shape, Intensity, Cristae
            Orientation) and a colored category strip is drawn above them.
            Features are NOT clustered in this figure — the biological
            grouping is the point — but the order within each category is
            deterministic (the order declared in TEM_FEATURE_SUBSETS).
        optimize_leaf_order: When True (default), runs scipy's
            optimal_leaf_ordering() on the morphology linkage. Read this
            carefully: optimal_leaf_ordering maximises similarity between
            ADJACENT LEAVES using the MORPHOLOGY distance matrix itself. It
            knows nothing about the phylogeny and does NOT minimise crossings
            against it — any reduction in crossings is incidental. Like a
            manual rotation it only swaps the two children of nodes, so
            cluster membership, node heights and baker_gamma are all
            unchanged; only the drawn order differs.
        rotate_morphology_nodes: Optional list of linkage row indices (as
            printed by the merge-order diagnostic) whose two children are
            swapped in the morphology tree. Same semantics as
            rotate_group_nodes in plot_feature_clustermap(). FOR LEGIBILITY
            ONLY: never rotate to make the morphology tree agree with the
            phylogeny. That would not change the topology, so it is not
            wrong as such, but it tunes the figure toward the hypothesis
            under test and turns n_crossings into a post-hoc optimised number
            instead of an observation.
        show_merge_diagnostic: When True (default), prints each morphology
            linkage row's two merged children and their distance — this is
            how a row index for rotate_morphology_nodes is chosen.
        support_values: Optional bootstrap clade support, as returned by
            compute_clade_support(). When given, each internal node of the
            morphology tree is annotated with its support. Support below 50%
            is drawn grey and in parentheses so an unsupported branch is not
            read as a finding.
        title: Figure title. Defaults to an auto-generated one.
        figsize: Figure size in inches (width, height).
        colormap: Diverging matplotlib colormap, centered at 0. Keep this
            diverging. The plotted values are z-scored group means (when
            standardize_group_means=True) — a SIGNED quantity where zero (the
            cross-species average) is meaningful. A sequential palette such
            as "mako" or "viridis" destroys the above-average/below-average
            reading, so this default is deliberate and should not be
            "fixed" later.

    Returns:
        Dict with keys:
          "figure":                              matplotlib.figure.Figure
          "phylogeny_linkage":                   np.ndarray
          "phylogeny_leaf_order":                List[str], top to bottom
          "morphology_linkage":                  np.ndarray, as drawn
          "morphology_leaf_order":               List[str], top to bottom
          "species_feature_matrix":              pd.DataFrame actually plotted
          "feature_order":                       List[str], final column order
          "baker_gamma":                         float, headline agreement stat
          "baker_gamma_p_value":                 float, exact permutation p
          "cophenetic_correlation":              float
          "n_crossings_raw":                     int, descriptive only
          "n_crossings_after_leaf_optimization": int, descriptive only
          "leaf_order_agreement":                str
          "nest_aggregate_by":                   Optional[List[str]]
          "standardize_group_means":             bool
          "clade_support":                       Optional[Dict[str, float]]

    Raises:
        ValueError: If group_by or nest_aggregate_by names a column absent
            from .obs, if fewer than 3 species are present, if a species has
            no declared divergence times, or if a rotate_morphology_nodes
            index is out of range.
    """
    print("=" * 70)
    print(f"PHYLOGENY / MORPHOLOGY TANGLEGRAM — grouped by '{group_by}'")
    print("=" * 70)

    nest_aggregate_by_columns = _resolve_nest_aggregate_columns(
        anndata_object, nest_aggregate_by, caller_name="plot_phylo_tanglegram"
    )
    species_feature_matrix = _build_species_feature_matrix(
        anndata_object=anndata_object,
        group_by=group_by,
        nest_aggregate_by_columns=nest_aggregate_by_columns,
        standardize_group_means=standardize_group_means,
        caller_name="plot_phylo_tanglegram",
    )
    species_list = species_feature_matrix.index.tolist()

    if len(species_list) < _MINIMUM_SPECIES_FOR_TANGLEGRAM:
        raise ValueError(
            f"plot_phylo_tanglegram() needs at least {_MINIMUM_SPECIES_FOR_TANGLEGRAM} "
            f"species to be informative, found {len(species_list)}: {species_list}. "
            "With 2 leaves both trees have the only possible topology, so their "
            "agreement is guaranteed and says nothing. Use plot_feature_clustermap() "
            "for a small side-by-side comparison instead."
        )

    resolved_feature_order = _resolve_feature_column_order(
        present_features=species_feature_matrix.columns.tolist(),
        feature_order=feature_order,
        group_features_by_category=group_features_by_category,
    )
    species_feature_matrix = species_feature_matrix[resolved_feature_order]

    phylogeny = build_species_phylogeny_linkage(species_list, branch_scale=branch_scale)
    phylogeny_linkage = phylogeny["linkage"]

    raw_morphology_linkage = linkage(
        species_feature_matrix.values, method=cluster_method, metric=cluster_metric
    )
    morphology_linkage = raw_morphology_linkage
    if optimize_leaf_order:
        morphology_linkage = optimal_leaf_ordering(
            morphology_linkage, pdist(species_feature_matrix.values, metric=cluster_metric)
        )
    morphology_linkage = _apply_node_rotations(morphology_linkage, rotate_morphology_nodes)

    if show_merge_diagnostic:
        _print_merge_diagnostic("MORPHOLOGY", morphology_linkage, species_list)

    phylogeny_leaf_order = _leaf_order_from_linkage(phylogeny_linkage, species_list)
    morphology_leaf_order = _leaf_order_from_linkage(morphology_linkage, species_list)
    raw_morphology_leaf_order = _leaf_order_from_linkage(raw_morphology_linkage, species_list)

    baker_gamma, baker_gamma_p_value = _compute_baker_gamma_with_exact_p_value(
        phylogeny_linkage, morphology_linkage
    )
    cophenetic_correlation = float(
        cophenet(morphology_linkage, pdist(species_feature_matrix.values, metric=cluster_metric))[0]
    )
    n_crossings_raw = _count_leaf_order_crossings(phylogeny_leaf_order, raw_morphology_leaf_order)
    n_crossings_after = _count_leaf_order_crossings(phylogeny_leaf_order, morphology_leaf_order)
    leaf_order_agreement = _describe_leaf_order_agreement(
        phylogeny_leaf_order, morphology_leaf_order
    )

    figure = _render_tanglegram_figure(
        anndata_object=anndata_object,
        species_feature_matrix=species_feature_matrix,
        species_list=species_list,
        phylogeny_linkage=phylogeny_linkage,
        morphology_linkage=morphology_linkage,
        group_features_by_category=group_features_by_category,
        support_values=support_values,
        title=title or _build_default_title(
            group_by, cluster_method, cluster_metric,
            standardize_group_means, nest_aggregate_by_columns, baker_gamma,
        ),
        figsize=figsize,
        colormap=colormap,
    )

    _print_tanglegram_qc(
        species_feature_matrix=species_feature_matrix,
        phylogeny_leaf_order=phylogeny_leaf_order,
        morphology_leaf_order=morphology_leaf_order,
        morphology_linkage=morphology_linkage,
        species_list=species_list,
        baker_gamma=baker_gamma,
        baker_gamma_p_value=baker_gamma_p_value,
        cophenetic_correlation=cophenetic_correlation,
        n_crossings_raw=n_crossings_raw,
        n_crossings_after=n_crossings_after,
        leaf_order_agreement=leaf_order_agreement,
        optimize_leaf_order=optimize_leaf_order,
        standardize_group_means=standardize_group_means,
        cluster_method=cluster_method,
        cluster_metric=cluster_metric,
        branch_scale=branch_scale,
        support_values=support_values,
    )

    return {
        "figure": figure,
        "phylogeny_linkage": phylogeny_linkage,
        "phylogeny_leaf_order": phylogeny_leaf_order,
        "morphology_linkage": morphology_linkage,
        "morphology_leaf_order": morphology_leaf_order,
        "species_feature_matrix": species_feature_matrix,
        "feature_order": resolved_feature_order,
        "baker_gamma": baker_gamma,
        "baker_gamma_p_value": baker_gamma_p_value,
        "cophenetic_correlation": cophenetic_correlation,
        "n_crossings_raw": n_crossings_raw,
        "n_crossings_after_leaf_optimization": n_crossings_after,
        "leaf_order_agreement": leaf_order_agreement,
        "nest_aggregate_by": nest_aggregate_by_columns,
        "standardize_group_means": standardize_group_means,
        "clade_support": support_values,
    }


def compute_tanglegram_sensitivity(
    anndata_object: anndata.AnnData,
    group_by: str = "specie",
    nest_aggregate_by: Optional[Union[str, List[str]]] = None,
    cluster_methods: Sequence[str] = ("average", "complete", "ward", "single"),
    cluster_metrics: Sequence[str] = ("correlation", "euclidean", "cosine"),
    standardize_options: Sequence[bool] = (True, False),
) -> pd.DataFrame:
    """
    Re-run the morphology tree under every clustering choice and tabulate it.

    With only 6 leaves, cluster_method x cluster_metric x standardization is a
    garden of forking paths: roughly two dozen combinations, several of which
    give different topologies. Nothing in the analysis itself stops a default
    from being picked after seeing which one matches the phylogeny, so this
    table makes the whole space visible at once. It is a supplementary figure,
    and its point is explicit: if the phylogeny is recovered only under one
    specific combination, that is a fragile result and the paper must say so.

    Arguments:
        anndata_object: AnnData with .obs[group_by].
        group_by: .obs column holding the species (default "specie").
        nest_aggregate_by: Passed through unchanged to the species-mean
            aggregation — see plot_phylo_tanglegram().
        cluster_methods: scipy linkage methods to sweep.
        cluster_metrics: scipy distance metrics to sweep.
        standardize_options: Whether to sweep the z-score lever (see
            plot_phylo_tanglegram(standardize_group_means=...) — under this
            figure's transposed orientation it changes the species tree).

    Returns:
        DataFrame with one row per combination, sorted by baker_gamma
        descending, with columns: cluster_method, cluster_metric,
        standardize_group_means, is_valid_combination, baker_gamma,
        baker_gamma_p_value, n_crossings_raw, newick_topology,
        recovered_(Worm,Droso), recovered_(ZFish,KFish),
        recovered_(Mouse,Human), n_expected_clades_recovered, note.

    Raises:
        ValueError: If group_by or nest_aggregate_by names a column absent
            from .obs, or if fewer than 3 species are present.
    """
    print("=" * 70)
    print("TANGLEGRAM SENSITIVITY — clustering method x metric x standardization")
    print("=" * 70)

    nest_aggregate_by_columns = _resolve_nest_aggregate_columns(
        anndata_object, nest_aggregate_by, caller_name="compute_tanglegram_sensitivity"
    )

    matrix_by_standardization = {
        standardize: _build_species_feature_matrix(
            anndata_object=anndata_object,
            group_by=group_by,
            nest_aggregate_by_columns=nest_aggregate_by_columns,
            standardize_group_means=standardize,
            caller_name="compute_tanglegram_sensitivity",
            verbose=False,
        )
        for standardize in standardize_options
    }
    species_list = next(iter(matrix_by_standardization.values())).index.tolist()

    if len(species_list) < _MINIMUM_SPECIES_FOR_TANGLEGRAM:
        raise ValueError(
            f"compute_tanglegram_sensitivity() needs at least "
            f"{_MINIMUM_SPECIES_FOR_TANGLEGRAM} species, found {len(species_list)}: "
            f"{species_list}."
        )

    phylogeny_linkage = build_species_phylogeny_linkage(species_list)["linkage"]
    phylogeny_leaf_order = _leaf_order_from_linkage(phylogeny_linkage, species_list)

    rows: List[Dict[str, object]] = []
    for standardize in standardize_options:
        species_feature_matrix = matrix_by_standardization[standardize]
        for cluster_method in cluster_methods:
            for cluster_metric in cluster_metrics:
                rows.append(
                    _evaluate_one_clustering_combination(
                        species_feature_matrix=species_feature_matrix,
                        species_list=species_list,
                        phylogeny_linkage=phylogeny_linkage,
                        phylogeny_leaf_order=phylogeny_leaf_order,
                        cluster_method=cluster_method,
                        cluster_metric=cluster_metric,
                        standardize=bool(standardize),
                    )
                )

    sensitivity_table = pd.DataFrame(rows).sort_values(
        "baker_gamma", ascending=False, na_position="last"
    ).reset_index(drop=True)

    _print_sensitivity_qc(sensitivity_table)
    return sensitivity_table


def compute_clade_support(
    anndata_object: anndata.AnnData,
    group_by: str = "specie",
    nest_aggregate_by: Optional[Union[str, List[str]]] = None,
    n_bootstrap: int = 1000,
    random_state: int = 42,
    **clustering_kwargs: object,
) -> Dict[str, float]:
    """
    Bootstrap support for every clade of the morphology tree, resampling SUBJECTS.

    A tree built from 6 points in ~24 dimensions is unstable. Without a
    stability estimate there is no way to tell "morphology recovers the
    phylogeny" from "this tree is noise that happens to look right" — and,
    just as importantly, no way to tell a real disagreement from a fragile
    one. This protects against a false bad result as much as a false good one.

    Resampling happens WITH REPLACEMENT at the level of the finest
    nest_aggregate_by unit (the subject), never at the level of individual
    mitochondria. This is the same reasoning already documented at the top of
    analysis/permutation_test.py: all of one subject's mitochondria share one
    biological label, so resampling mitochondria would destroy the subject
    structure and inflate apparent support. Each resampled draw is given its
    own identity, so a subject drawn twice counts twice — which is what makes
    it a bootstrap rather than a re-shuffle.

    Arguments:
        anndata_object: AnnData with .obs[group_by].
        group_by: .obs column holding the species (default "specie").
        nest_aggregate_by: REQUIRED here — a .obs column, or list ordered
            COARSEST to FINEST, whose finest level identifies the subject.
            Subject-level resampling is impossible without it.
        n_bootstrap: Number of bootstrap replicates (default 1000).
        random_state: Seed, so a given call is reproducible.
        **clustering_kwargs: Passed to the morphology linkage — accepts
            cluster_method, cluster_metric and standardize_group_means, whose
            defaults match plot_phylo_tanglegram().

    Returns:
        Dict mapping a clade string (e.g. "(Mouse,Human)") to the fraction of
        bootstrap replicates in which that clade appeared, for every clade
        observed at least once, PLUS the three clades of the reference
        phylogeny, which are always present in the result even when they were
        never recovered — a support of 0.0 is a valid and informative answer,
        and omitting it would hide it.

    Raises:
        ValueError: If nest_aggregate_by is None, if a named column is absent
            from .obs, if fewer than 3 species are present, or if an
            unexpected clustering keyword is given.
    """
    cluster_method = str(clustering_kwargs.pop("cluster_method", "average"))
    cluster_metric = str(clustering_kwargs.pop("cluster_metric", "correlation"))
    standardize_group_means = bool(clustering_kwargs.pop("standardize_group_means", True))
    if clustering_kwargs:
        raise ValueError(
            f"Unexpected clustering keyword(s): {sorted(clustering_kwargs)}. "
            "Supported: cluster_method, cluster_metric, standardize_group_means."
        )

    print("=" * 70)
    print(f"BOOTSTRAP CLADE SUPPORT — {n_bootstrap} replicates, resampled at the SUBJECT level")
    print("=" * 70)

    nest_aggregate_by_columns = _resolve_nest_aggregate_columns(
        anndata_object, nest_aggregate_by, caller_name="compute_clade_support"
    )
    if nest_aggregate_by_columns is None:
        raise ValueError(
            "compute_clade_support() requires nest_aggregate_by so that bootstrap "
            "resampling happens at the SUBJECT level (e.g. "
            "nest_aggregate_by='unique_subject_ID'). Resampling individual "
            "mitochondria instead would destroy the subject structure and inflate "
            "apparent support — see the design notes in analysis/permutation_test.py."
        )

    subject_profiles = _build_subject_profile_table(
        anndata_object=anndata_object,
        group_by=group_by,
        nest_aggregate_by_columns=nest_aggregate_by_columns,
    )
    species_list = sorted(subject_profiles["group_labels"].unique().tolist())

    if len(species_list) < _MINIMUM_SPECIES_FOR_TANGLEGRAM:
        raise ValueError(
            f"compute_clade_support() needs at least {_MINIMUM_SPECIES_FOR_TANGLEGRAM} "
            f"species, found {len(species_list)}: {species_list}."
        )

    for species in species_list:
        n_subjects = int((subject_profiles["group_labels"] == species).sum())
        print(f"    {species}: {n_subjects} resampling unit(s) at the finest level")

    clade_counts: Dict[str, int] = {}
    random_generator = np.random.default_rng(random_state)
    n_completed = 0

    for _ in range(n_bootstrap):
        replicate_matrix = _resample_species_feature_matrix(
            subject_profiles=subject_profiles,
            species_list=species_list,
            nest_aggregate_by_columns=nest_aggregate_by_columns,
            standardize_group_means=standardize_group_means,
            random_generator=random_generator,
        )
        try:
            replicate_linkage = linkage(
                replicate_matrix.values, method=cluster_method, metric=cluster_metric
            )
        except ValueError:
            # A degenerate replicate (e.g. every drawn subject identical, so a
            # constant profile makes correlation distance undefined) carries no
            # information about clade stability — skip it rather than counting
            # it as evidence either way.
            continue

        n_completed += 1
        for clade in get_linkage_clades(replicate_linkage, species_list):
            clade_key = _format_clade_key(clade)
            clade_counts[clade_key] = clade_counts.get(clade_key, 0) + 1

    if n_completed == 0:
        raise ValueError(
            "Every bootstrap replicate was degenerate — no clade support could be "
            f"estimated with cluster_metric='{cluster_metric}'. Try "
            "cluster_metric='euclidean', or check that the species profiles are "
            "not constant across features."
        )

    clade_support = {
        clade_key: count / n_completed for clade_key, count in clade_counts.items()
    }
    # The reference phylogeny's clades are always reported, even at 0.0 —
    # "this expected clade never appeared" is the informative answer, and
    # dropping the key would hide it.
    for clade_key, clade in TEM_EXPECTED_PHYLOGENETIC_CLADES.items():
        if clade.issubset(set(species_list)):
            clade_support.setdefault(_format_clade_key(clade), 0.0)

    _print_clade_support_qc(
        clade_support, n_completed, n_bootstrap, cluster_method, cluster_metric,
        standardize_group_means, nest_aggregate_by_columns,
    )
    return clade_support


# ---------------------------------------------------------------------------
# Private helpers — data preparation
# ---------------------------------------------------------------------------


def _build_species_feature_matrix(
    anndata_object: anndata.AnnData,
    group_by: str,
    nest_aggregate_by_columns: Optional[List[str]],
    standardize_group_means: bool,
    caller_name: str,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Build the species x feature matrix that every tree in this module clusters.

    Reuses plot_feature_clustermap()'s data-access and standardization
    helpers so the two figures cannot drift apart, then transposes the result
    to species-as-rows, which is the orientation the tanglegram clusters in.

    Arguments:
        anndata_object: AnnData with .obs[group_by].
        group_by: .obs column holding the species.
        nest_aggregate_by_columns: Validated nesting columns, or None to pool.
        standardize_group_means: Whether to z-score each feature across
            species before clustering (see plot_phylo_tanglegram()).
        caller_name: Public function name shown in the printed QC lines.
        verbose: When False, the per-group aggregation QC lines are suppressed
            (used by the sensitivity sweep, which builds the matrix twice).

    Returns:
        DataFrame indexed by species, one column per analytical feature.
    """
    import contextlib  # noqa: PLC0415
    import io  # noqa: PLC0415

    output_sink = contextlib.redirect_stdout(io.StringIO()) if not verbose else contextlib.nullcontext()

    with output_sink:
        data_matrix, channel_names = _get_data_and_channels(anndata_object, caller_name=caller_name)
        group_labels = _get_group_labels(anndata_object, group_by)

        valid_mask = group_labels.notna().values
        nest_hierarchy_values = (
            anndata_object.obs[nest_aggregate_by_columns].values[valid_mask]
            if nest_aggregate_by_columns is not None
            else None
        )
        group_mean_dataframe = _compute_group_mean_matrix(
            data_matrix=data_matrix[valid_mask],
            channel_names=channel_names,
            group_label_values=group_labels.values[valid_mask],
            nest_hierarchy_values=nest_hierarchy_values,
            nest_aggregate_by_columns=nest_aggregate_by_columns,
            caller_name=caller_name,
        )

        # _standardize_group_means() z-scores each ROW of a feature x group
        # matrix, so the frame is transposed into that orientation, scaled,
        # and transposed back to species-as-rows.
        standardized_dataframe = _standardize_group_means(
            group_mean_dataframe.T, standardize_group_means, caller_name=caller_name
        ).T

    return standardized_dataframe


def _build_subject_profile_table(
    anndata_object: anndata.AnnData,
    group_by: str,
    nest_aggregate_by_columns: List[str],
) -> Dict[str, object]:
    """
    Collapse the raw observations to one mean profile per finest-level unit.

    This is the table the bootstrap resamples from: one row per subject, so
    that drawing rows with replacement is subject-level resampling by
    construction.

    Arguments:
        anndata_object: AnnData with .obs[group_by].
        group_by: .obs column holding the species.
        nest_aggregate_by_columns: Nesting columns, coarsest to finest; the
            last one identifies the subject.

    Returns:
        Dict with "profiles" (np.ndarray, one row per subject),
        "group_labels" (pd.Series of the species per subject),
        "hierarchy_values" (np.ndarray of the nesting columns per subject) and
        "channel_names" (List[str]).
    """
    data_matrix, channel_names = _get_data_and_channels(
        anndata_object, caller_name="compute_clade_support"
    )
    group_labels = _get_group_labels(anndata_object, group_by)

    hierarchy_frame = anndata_object.obs[nest_aggregate_by_columns]
    valid_mask = group_labels.notna().values & ~hierarchy_frame.isnull().any(axis=1).values

    valid_matrix = data_matrix[valid_mask]
    valid_groups = pd.Series(group_labels.values[valid_mask]).astype(str)
    valid_hierarchy = hierarchy_frame.values[valid_mask]
    finest_values = valid_hierarchy[:, -1]

    subject_rows: List[np.ndarray] = []
    subject_groups: List[str] = []
    subject_hierarchy: List[np.ndarray] = []
    for subject_id in pd.unique(finest_values):
        subject_mask = finest_values == subject_id
        subject_rows.append(np.nanmean(valid_matrix[subject_mask], axis=0))
        subject_groups.append(valid_groups.values[subject_mask][0])
        subject_hierarchy.append(valid_hierarchy[subject_mask][0])

    print(
        f"[compute_clade_support] Collapsed {int(valid_mask.sum()):,} observations to "
        f"{len(subject_rows)} unit(s) at level '{nest_aggregate_by_columns[-1]}'."
    )

    return {
        "profiles": np.vstack(subject_rows),
        "group_labels": pd.Series(subject_groups),
        "hierarchy_values": np.array(subject_hierarchy, dtype=object),
        "channel_names": channel_names,
    }


def _resample_species_feature_matrix(
    subject_profiles: Dict[str, object],
    species_list: List[str],
    nest_aggregate_by_columns: List[str],
    standardize_group_means: bool,
    random_generator: np.random.Generator,
) -> pd.DataFrame:
    """
    Draw one bootstrap replicate of the species x feature matrix.

    Within each species, its subjects are drawn WITH REPLACEMENT to the same
    count. Each draw is relabelled with a unique synthetic identity at the
    finest hierarchy level so that a subject drawn twice really does count
    twice — without that relabelling the nested aggregate would collapse the
    duplicate back to one and the bootstrap would degenerate into sampling
    without replacement.

    Arguments:
        subject_profiles: Output of _build_subject_profile_table().
        species_list: Species to include, in the desired row order.
        nest_aggregate_by_columns: The nesting columns, coarsest to finest.
        standardize_group_means: Whether to z-score each feature across
            species after aggregation.
        random_generator: Seeded numpy Generator.

    Returns:
        DataFrame indexed by species, one column per feature.
    """
    profiles = subject_profiles["profiles"]
    group_labels = subject_profiles["group_labels"]
    hierarchy_values = subject_profiles["hierarchy_values"]
    channel_names = subject_profiles["channel_names"]

    replicate_rows: List[np.ndarray] = []
    for species in species_list:
        species_indices = np.flatnonzero((group_labels == species).values)
        drawn_indices = random_generator.choice(
            species_indices, size=len(species_indices), replace=True
        )
        drawn_profiles = profiles[drawn_indices]
        drawn_hierarchy = hierarchy_values[drawn_indices].copy()
        # Unique identity per draw — see the docstring.
        for draw_index in range(drawn_hierarchy.shape[0]):
            drawn_hierarchy[draw_index, -1] = f"{drawn_hierarchy[draw_index, -1]}#{draw_index}"

        replicate_rows.append(
            _nested_group_aggregate(
                drawn_profiles, drawn_hierarchy, np.nanmean, caller_name="compute_clade_support"
            )
        )

    replicate_matrix = pd.DataFrame(
        np.vstack(replicate_rows), index=species_list, columns=channel_names
    )
    if standardize_group_means:
        replicate_matrix = _standardize_group_means(
            replicate_matrix.T, True, caller_name="compute_clade_support"
        ).T
    return replicate_matrix


def _resolve_feature_column_order(
    present_features: List[str],
    feature_order: Optional[List[str]],
    group_features_by_category: bool,
) -> List[str]:
    """
    Decide the heatmap's column order.

    An explicit feature_order always wins (any present feature it omits is
    appended afterwards, so nothing is silently dropped). Otherwise, when
    group_features_by_category is True, features are grouped into contiguous
    biological blocks in TEM_FEATURE_CATEGORY_ORDER, and within each block
    they keep the order declared in TEM_FEATURE_CATEGORIES — deterministic,
    and independent of the data. Features with no declared category are
    appended last as "Other".

    Arguments:
        present_features: Features actually available in the matrix.
        feature_order: Optional explicit order.
        group_features_by_category: Whether to group by biological category.

    Returns:
        Ordered list containing exactly the values in present_features.
    """
    if feature_order is not None:
        listed = [feature for feature in feature_order if feature in present_features]
        unlisted = [feature for feature in present_features if feature not in listed]
        return listed + unlisted

    if not group_features_by_category:
        return list(present_features)

    declared_rank = {feature: rank for rank, feature in enumerate(TEM_FEATURE_CATEGORIES)}
    category_rank = {category: rank for rank, category in enumerate(TEM_FEATURE_CATEGORY_ORDER)}

    def _sort_key(feature: str) -> Tuple[int, int]:
        category = TEM_FEATURE_CATEGORIES.get(feature, "Other")
        return (
            category_rank.get(category, len(category_rank)),
            declared_rank.get(feature, len(declared_rank)),
        )

    return sorted(present_features, key=_sort_key)


# ---------------------------------------------------------------------------
# Private helpers — agreement statistics
# ---------------------------------------------------------------------------


def _leaf_order_from_linkage(linkage_matrix: np.ndarray, leaf_labels: List[str]) -> List[str]:
    """
    Return the leaf labels in the order the dendrogram draws them.

    Arguments:
        linkage_matrix: scipy linkage matrix.
        leaf_labels: Leaf labels indexed as the linkage matrix indexes them.

    Returns:
        List of labels in dendrogram leaf order (first = top of this figure).
    """
    from scipy.cluster.hierarchy import leaves_list  # noqa: PLC0415

    return [leaf_labels[index] for index in leaves_list(linkage_matrix)]


def _compute_baker_gamma_with_exact_p_value(
    linkage_a: np.ndarray,
    linkage_b: np.ndarray,
) -> Tuple[float, float]:
    """
    Baker's gamma between two trees, plus its EXACT permutation p-value.

    Baker's gamma is the Spearman correlation between the two trees'
    cophenetic distances — for every pair of leaves, how far apart the tree
    says they are. Because it reads only topology and node heights it is
    completely invariant to leaf display order: no rotation can change it.

    The p-value is exact, not Monte Carlo. With n species there are only n!
    ways to relabel the leaves of one tree (720 for the 6 TEM species), so
    every one of them is enumerated and the p-value is the fraction whose
    gamma is at least the observed one. The observed labelling is itself one
    of the permutations, so the p-value can never be smaller than 1/n! —
    a floor of 1/720 ~= 0.0014 for 6 species. Above
    _MAXIMUM_SPECIES_FOR_EXACT_PERMUTATION species, enumeration stops being
    tractable and the p-value is returned as NaN with a printed warning
    rather than silently switching to an approximation.

    Arguments:
        linkage_a: Reference linkage (the phylogeny).
        linkage_b: Comparison linkage (the morphology tree). Must index the
            same leaves in the same order as linkage_a.

    Returns:
        Tuple of (baker_gamma, baker_gamma_p_value).
    """
    n_leaves = linkage_a.shape[0] + 1

    # Spearman = Pearson on ranks. Ranking each cophenetic vector once is
    # enough: permuting leaf labels permutes the entries of the distance
    # matrix, which leaves each entry's rank untouched.
    reference_ranks = rankdata(cophenet(linkage_a))
    comparison_rank_square = squareform(rankdata(cophenet(linkage_b)))

    observed_gamma = _pearson_correlation(
        reference_ranks, squareform(comparison_rank_square, checks=False)
    )

    if n_leaves > _MAXIMUM_SPECIES_FOR_EXACT_PERMUTATION:
        print(
            f"WARNING [plot_phylo_tanglegram]: {n_leaves} species means {n_leaves}! label "
            "permutations — too many to enumerate exactly, so baker_gamma_p_value is NaN. "
            "No Monte Carlo approximation is substituted."
        )
        return observed_gamma, float("nan")

    n_at_least_as_extreme = 0
    n_permutations = 0
    for permutation in permutations(range(n_leaves)):
        permuted_index = np.array(permutation)
        permuted_ranks = squareform(
            comparison_rank_square[np.ix_(permuted_index, permuted_index)], checks=False
        )
        n_permutations += 1
        if _pearson_correlation(reference_ranks, permuted_ranks) >= observed_gamma - 1e-12:
            n_at_least_as_extreme += 1

    return observed_gamma, n_at_least_as_extreme / n_permutations


def _pearson_correlation(first_vector: np.ndarray, second_vector: np.ndarray) -> float:
    """
    Pearson correlation of two 1-D arrays, returning 0.0 for a constant input.

    A constant vector has zero variance, so the correlation is undefined;
    0.0 ("no relationship measurable") is returned instead of NaN so that a
    degenerate tree does not poison the permutation loop.

    Arguments:
        first_vector: 1-D numeric array.
        second_vector: 1-D numeric array of the same length.

    Returns:
        Correlation coefficient as a float.
    """
    first_centered = first_vector - first_vector.mean()
    second_centered = second_vector - second_vector.mean()
    denominator = np.sqrt((first_centered ** 2).sum() * (second_centered ** 2).sum())
    if denominator == 0:
        return 0.0
    return float((first_centered * second_centered).sum() / denominator)


def _count_leaf_order_crossings(
    reference_leaf_order: List[str],
    comparison_leaf_order: List[str],
) -> int:
    """
    Count how many connector lines cross between two leaf orders.

    Two species' connectors cross exactly when their relative vertical order
    is reversed between the two trees, so this is the number of inverted
    pairs. Purely a property of how the leaves are DRAWN — a cosmetic
    rotation changes it, which is why it is reported as descriptive context
    and never as the headline agreement statistic (that is baker_gamma).

    Arguments:
        reference_leaf_order: Leaf labels in the left tree's draw order.
        comparison_leaf_order: Leaf labels in the right tree's draw order.

    Returns:
        Number of crossing connector pairs.
    """
    comparison_position = {label: index for index, label in enumerate(comparison_leaf_order)}
    positions = [comparison_position[label] for label in reference_leaf_order]
    return int(sum(
        1
        for first in range(len(positions))
        for second in range(first + 1, len(positions))
        if positions[first] > positions[second]
    ))


def _describe_leaf_order_agreement(
    phylogeny_leaf_order: List[str],
    morphology_leaf_order: List[str],
) -> str:
    """
    Summarize leaf-order agreement in one human-readable sentence.

    Arguments:
        phylogeny_leaf_order: Leaf labels in the phylogeny's draw order.
        morphology_leaf_order: Leaf labels in the morphology tree's draw order.

    Returns:
        A string such as
        "5/6 leaves in phylogenetic order (ZFish and KFish inverted)".
    """
    n_leaves = len(phylogeny_leaf_order)
    morphology_position = {label: index for index, label in enumerate(morphology_leaf_order)}

    misplaced = [
        label
        for index, label in enumerate(phylogeny_leaf_order)
        if morphology_position[label] != index
    ]
    n_matched = n_leaves - len(misplaced)

    if not misplaced:
        return f"{n_leaves}/{n_leaves} leaves in phylogenetic order (perfect agreement)"

    if len(misplaced) == 2:
        first_label, second_label = misplaced
        first_index = phylogeny_leaf_order.index(first_label)
        second_index = phylogeny_leaf_order.index(second_label)
        if (
            morphology_position[first_label] == second_index
            and morphology_position[second_label] == first_index
        ):
            return (
                f"{n_matched}/{n_leaves} leaves in phylogenetic order "
                f"({first_label} and {second_label} inverted)"
            )

    return (
        f"{n_matched}/{n_leaves} leaves in phylogenetic order "
        f"(out of place: {', '.join(misplaced)})"
    )


def _format_clade_key(clade: frozenset) -> str:
    """
    Render a clade as a stable, readable key such as "(ZFish,KFish)".

    Members are ordered by TEM_SPECIES_PHYLOGENETIC_ORDER so the same clade
    always produces the same key regardless of how it was discovered.

    Arguments:
        clade: Set of species names under one internal node.

    Returns:
        Parenthesised, comma-separated species string.
    """
    phylogenetic_rank = {
        species: rank for rank, species in enumerate(TEM_SPECIES_PHYLOGENETIC_ORDER)
    }
    ordered = sorted(clade, key=lambda species: (phylogenetic_rank.get(species, 999), species))
    return "(" + ",".join(ordered) + ")"


def _evaluate_one_clustering_combination(
    species_feature_matrix: pd.DataFrame,
    species_list: List[str],
    phylogeny_linkage: np.ndarray,
    phylogeny_leaf_order: List[str],
    cluster_method: str,
    cluster_metric: str,
    standardize: bool,
) -> Dict[str, object]:
    """
    Score one (method, metric, standardization) combination for the sensitivity table.

    Combinations that are not strictly valid are FLAGGED, never silently
    dropped: "ward", "centroid" and "median" linkage are defined only for
    Euclidean distance, so pairing them with another metric produces a tree
    whose branch lengths have no defined meaning. Hiding those rows would
    make the parameter space look smaller than it is.

    Arguments:
        species_feature_matrix: Species x feature matrix to cluster.
        species_list: Species in matrix row order.
        phylogeny_linkage: The reference phylogeny linkage.
        phylogeny_leaf_order: Reference leaf order for the crossing count.
        cluster_method: scipy linkage method.
        cluster_metric: scipy distance metric.
        standardize: Which standardization the matrix was built with.

    Returns:
        One row of the sensitivity table, as a dict.
    """
    euclidean_only_methods = {"ward", "centroid", "median"}
    is_valid_combination = not (
        cluster_method in euclidean_only_methods and cluster_metric != "euclidean"
    )
    note = (
        ""
        if is_valid_combination
        else f"'{cluster_method}' linkage is only strictly defined for euclidean distance"
    )

    row: Dict[str, object] = {
        "cluster_method": cluster_method,
        "cluster_metric": cluster_metric,
        "standardize_group_means": standardize,
        "is_valid_combination": is_valid_combination,
        "baker_gamma": float("nan"),
        "baker_gamma_p_value": float("nan"),
        "n_crossings_raw": -1,
        "newick_topology": "",
        "n_expected_clades_recovered": 0,
        "note": note,
    }
    for clade_key in TEM_EXPECTED_PHYLOGENETIC_CLADES:
        row[f"recovered_{clade_key}"] = False

    try:
        morphology_linkage = linkage(
            species_feature_matrix.values, method=cluster_method, metric=cluster_metric
        )
    except (ValueError, TypeError) as error:
        row["note"] = f"{note + '; ' if note else ''}linkage failed: {error}"
        return row

    baker_gamma, baker_gamma_p_value = _compute_baker_gamma_with_exact_p_value(
        phylogeny_linkage, morphology_linkage
    )
    observed_clades = set(get_linkage_clades(morphology_linkage, species_list))

    row["baker_gamma"] = baker_gamma
    row["baker_gamma_p_value"] = baker_gamma_p_value
    row["n_crossings_raw"] = _count_leaf_order_crossings(
        phylogeny_leaf_order, _leaf_order_from_linkage(morphology_linkage, species_list)
    )
    row["newick_topology"] = linkage_to_newick_string(morphology_linkage, species_list)

    n_recovered = 0
    for clade_key, clade in TEM_EXPECTED_PHYLOGENETIC_CLADES.items():
        recovered = clade in observed_clades
        row[f"recovered_{clade_key}"] = recovered
        n_recovered += int(recovered)
    row["n_expected_clades_recovered"] = n_recovered

    return row


# ---------------------------------------------------------------------------
# Private helpers — figure rendering
# ---------------------------------------------------------------------------


def _get_species_color(anndata_object: anndata.AnnData, species: str) -> str:
    """
    Look up one species' display color.

    Prefers the palette stored in .uns['color_palette'] by
    assign_color_palette(), then the project-wide species colors declared in
    PREFERRED_CONDITION_COLORS, then a neutral dark grey.

    Arguments:
        anndata_object: AnnData whose .uns may hold a color palette.
        species: Species label to look up.

    Returns:
        Hex color string.
    """
    return get_color_for_value(
        anndata_object,
        species,
        fallback_color=PREFERRED_CONDITION_COLORS.get(species, "#444444"),
    )


def _compute_dendrogram_node_positions(
    linkage_matrix: np.ndarray,
    drawn_leaf_indices: List[int],
) -> np.ndarray:
    """
    Compute every node's position on the dendrogram's leaf axis.

    Reproduces scipy's own rule: the k-th drawn leaf sits at
    10*k + 5, and an internal node sits midway between its two children.
    Needed because scipy returns its drawn coordinates in DFS order, which
    cannot be mapped back to linkage row numbers reliably.

    Arguments:
        linkage_matrix: scipy linkage matrix of shape (n_leaves - 1, 4).
        drawn_leaf_indices: Leaf indices in the order the dendrogram draws
            them (scipy's leaves_list output).

    Returns:
        Array of length 2*n_leaves - 1 holding each node's leaf-axis position,
        indexed the way the linkage matrix indexes nodes.
    """
    n_leaves = len(drawn_leaf_indices)
    node_positions = np.zeros(2 * n_leaves - 1, dtype=np.float64)
    for draw_index, leaf_index in enumerate(drawn_leaf_indices):
        node_positions[leaf_index] = _DENDROGRAM_LEAF_SPACING * draw_index + (
            _DENDROGRAM_LEAF_SPACING / 2.0
        )
    for row_index, row in enumerate(linkage_matrix):
        left_index, right_index = int(row[0]), int(row[1])
        node_positions[n_leaves + row_index] = (
            node_positions[left_index] + node_positions[right_index]
        ) / 2.0
    return node_positions


def _draw_tree_panel(
    axes: "plt.Axes",
    linkage_matrix: np.ndarray,
    species_list: List[str],
    orientation: str,
    anndata_object: anndata.AnnData,
    n_species: int,
    show_leaf_labels: bool = True,
) -> List[str]:
    """
    Draw one dendrogram panel and color its leaf labels by species.

    Arguments:
        axes: Target matplotlib axes.
        linkage_matrix: Linkage to draw.
        species_list: Leaf labels indexed as the linkage indexes them.
        orientation: "left" (root at left, leaves at right — used for the
            phylogeny) or "right" (root at right, leaves at left — used for
            the morphology tree).
        anndata_object: Source of the species colors.
        n_species: Number of leaves, used to set the shared leaf-axis limits.
        show_leaf_labels: When False, the panel draws the tree only. Used for
            the phylogeny, whose labels would land underneath the heatmap
            axes — the heatmap draws them on its own left edge instead.

    Returns:
        The leaf labels in draw order (top to bottom of the figure).
    """
    dendrogram_result = dendrogram(
        linkage_matrix,
        labels=species_list,
        orientation=orientation,
        ax=axes,
        color_threshold=0,
        above_threshold_color="#444444",
        no_labels=not show_leaf_labels,
    )
    # Inverting the leaf axis puts leaf 0 at the TOP, so the figure reads
    # top-to-bottom in the same order the returned leaf lists are printed.
    axes.set_ylim(_DENDROGRAM_LEAF_SPACING * n_species, 0)

    # Pull the left panel's leaf tips back from its right edge, so the tree
    # does not run into the species names the heatmap draws on its own left
    # edge. The right panel keeps scipy's limits: matplotlib anchors its leaf
    # labels at the axes spine, so widening there would push the labels
    # outward into the connector lines instead of away from them.
    if orientation == "left":
        root_distance, leaf_distance = axes.get_xlim()
        label_gutter = abs(root_distance - leaf_distance) * _LEAF_LABEL_GUTTER_FRACTION
        axes.set_xlim(root_distance, leaf_distance - label_gutter)

    axes.set_xticks([])
    for spine in axes.spines.values():
        spine.set_visible(False)
    axes.tick_params(axis="y", length=0, labelsize=8)
    for tick_label in axes.get_yticklabels():
        tick_label.set_color(_get_species_color(anndata_object, tick_label.get_text()))
        tick_label.set_fontweight("bold")
    return list(dendrogram_result["ivl"])


def _draw_category_strip(
    axes: "plt.Axes",
    feature_order: List[str],
) -> List[int]:
    """
    Draw the colored biological-category strip above the heatmap columns.

    One colored block per contiguous run of the same category in the final
    column order, labelled with the category name. Returns the block
    boundaries so the heatmap can draw matching white gaps.

    Arguments:
        axes: Target matplotlib axes, sitting directly above the heatmap.
        feature_order: Final heatmap column order.

    Returns:
        Column indices where one category block ends and the next begins.
    """
    categories = [TEM_FEATURE_CATEGORIES.get(feature, "Other") for feature in feature_order]
    boundaries: List[int] = []

    block_start = 0
    for column_index in range(1, len(categories) + 1):
        is_last = column_index == len(categories)
        if is_last or categories[column_index] != categories[block_start]:
            category = categories[block_start]
            block_color = TEM_FEATURE_CATEGORY_COLORS.get(
                category, TEM_FEATURE_CATEGORY_COLORS["Other"]
            )
            axes.add_patch(
                Rectangle(
                    (block_start, 0.0), column_index - block_start, 1.0,
                    facecolor=block_color, edgecolor="white", linewidth=1.0,
                )
            )
            axes.text(
                # Wrap a multi-word category name ("Cristae Orientation") onto
                # two lines so it fits inside its narrow column block instead
                # of spilling over the neighbouring panel.
                (block_start + column_index) / 2.0, 0.5, category.replace(" ", "\n"),
                ha="center", va="center", fontsize=8, color="white", fontweight="bold",
            )
            if not is_last:
                boundaries.append(column_index)
            block_start = column_index

    axes.set_xlim(0, len(feature_order))
    axes.set_ylim(0, 1)
    axes.axis("off")
    return boundaries


def _draw_connector_lines(
    axes: "plt.Axes",
    phylogeny_leaf_order: List[str],
    morphology_leaf_order: List[str],
    anndata_object: anndata.AnnData,
) -> None:
    """
    Draw one line per species, from its heatmap row to its morphology leaf.

    Every crossing is a species pair whose relative order differs between the
    two trees — that is exactly the disagreement the figure exists to show,
    so nothing here is smoothed or hidden.

    Arguments:
        axes: The narrow axes between the heatmap and the morphology tree.
        phylogeny_leaf_order: Species top to bottom on the heatmap side.
        morphology_leaf_order: Species top to bottom on the tree side.
        anndata_object: Source of the species colors.
    """
    n_species = len(phylogeny_leaf_order)
    morphology_position = {label: index for index, label in enumerate(morphology_leaf_order)}
    half_step = _DENDROGRAM_LEAF_SPACING / 2.0

    for phylogeny_index, species in enumerate(phylogeny_leaf_order):
        left_y = _DENDROGRAM_LEAF_SPACING * phylogeny_index + half_step
        right_y = _DENDROGRAM_LEAF_SPACING * morphology_position[species] + half_step
        axes.plot(
            [0.0, _CONNECTOR_LINE_SPAN], [left_y, right_y],
            color=_get_species_color(anndata_object, species),
            linewidth=1.4, alpha=0.9, solid_capstyle="round",
        )

    axes.set_xlim(0, 1)
    axes.set_ylim(_DENDROGRAM_LEAF_SPACING * n_species, 0)
    axes.axis("off")


def _annotate_clade_support(
    axes: "plt.Axes",
    linkage_matrix: np.ndarray,
    species_list: List[str],
    support_values: Dict[str, float],
) -> None:
    """
    Write bootstrap support next to each internal node of the morphology tree.

    Support below _WEAK_SUPPORT_THRESHOLD is drawn grey and in parentheses,
    so a branch that most bootstrap replicates did not reproduce is visibly
    marked as such instead of reading like a finding.

    Arguments:
        axes: The morphology tree's axes.
        linkage_matrix: The morphology linkage as drawn.
        species_list: Leaf labels indexed as the linkage indexes them.
        support_values: Clade key -> support fraction, from
            compute_clade_support().
    """
    from scipy.cluster.hierarchy import leaves_list  # noqa: PLC0415

    node_positions = _compute_dendrogram_node_positions(
        linkage_matrix, list(leaves_list(linkage_matrix))
    )
    node_clades = get_linkage_clades(linkage_matrix, species_list, include_root=True)
    n_leaves = len(species_list)

    for row_index, row in enumerate(linkage_matrix[:-1]):  # the root is always 100%
        clade_key = _format_clade_key(node_clades[row_index])
        if clade_key not in support_values:
            continue
        support = support_values[clade_key]
        is_weak = support < _WEAK_SUPPORT_THRESHOLD
        axes.text(
            float(row[2]),
            node_positions[n_leaves + row_index],
            f"({support * 100:.0f})" if is_weak else f"{support * 100:.0f}",
            fontsize=6.5,
            color="#999999" if is_weak else "#222222",
            ha="center",
            va="bottom",
        )


def _render_tanglegram_figure(
    anndata_object: anndata.AnnData,
    species_feature_matrix: pd.DataFrame,
    species_list: List[str],
    phylogeny_linkage: np.ndarray,
    morphology_linkage: np.ndarray,
    group_features_by_category: bool,
    support_values: Optional[Dict[str, float]],
    title: str,
    figsize: Tuple[float, float],
    colormap: str,
) -> "plt.Figure":
    """
    Assemble the five-panel tanglegram figure.

    seaborn.clustermap cannot produce this layout — it owns its own gridspec
    and supports only one dendrogram per axis — so the figure is built
    directly with matplotlib.gridspec, one scipy dendrogram per tree and a
    pcolormesh heatmap between them.

    Arguments:
        anndata_object: Source of the species colors.
        species_feature_matrix: Species x feature matrix to plot.
        species_list: Species in matrix row order (the linkage indexing).
        phylogeny_linkage: Known phylogeny, drawn on the left.
        morphology_linkage: Data-derived tree, drawn on the right.
        group_features_by_category: Whether to draw the category strip and
            the white gaps between category blocks.
        support_values: Optional bootstrap clade support to annotate.
        title: Figure title.
        figsize: Figure size in inches.
        colormap: Diverging colormap name, centered at 0.

    Returns:
        The assembled matplotlib Figure.
    """
    n_species = len(species_list)
    feature_order = species_feature_matrix.columns.tolist()

    figure = plt.figure(figsize=figsize)
    grid = GridSpec(
        2, 5,
        figure=figure,
        width_ratios=[1.0, 3.8, 0.9, 1.0, 0.08],
        height_ratios=[0.10, 1.0],
        wspace=0.10,
        hspace=0.04,
    )

    axes_phylogeny = figure.add_subplot(grid[1, 0])
    axes_heatmap = figure.add_subplot(grid[1, 1])
    axes_connectors = figure.add_subplot(grid[1, 2])
    axes_morphology = figure.add_subplot(grid[1, 3])
    axes_colorbar = figure.add_subplot(grid[1, 4])

    phylogeny_leaf_order = _draw_tree_panel(
        axes_phylogeny, phylogeny_linkage, species_list, "left", anndata_object, n_species,
        show_leaf_labels=False,
    )
    morphology_leaf_order = _draw_tree_panel(
        axes_morphology, morphology_linkage, species_list, "right", anndata_object, n_species
    )
    axes_phylogeny.set_title("Known phylogeny", fontsize=9, pad=6)
    axes_morphology.set_title("Morphology (from data)", fontsize=9, pad=6)

    plotted_matrix = species_feature_matrix.loc[phylogeny_leaf_order, feature_order]
    color_limit = float(np.nanmax(np.abs(plotted_matrix.values))) or 1.0
    mesh = axes_heatmap.pcolormesh(
        np.arange(len(feature_order) + 1, dtype=np.float64),
        np.arange(n_species + 1, dtype=np.float64) * _DENDROGRAM_LEAF_SPACING,
        plotted_matrix.values,
        cmap=colormap,
        vmin=-color_limit,
        vmax=color_limit,
        edgecolors="white",
        linewidth=0.5,
    )
    axes_heatmap.set_xlim(0, len(feature_order))
    axes_heatmap.set_ylim(_DENDROGRAM_LEAF_SPACING * n_species, 0)
    axes_heatmap.set_yticks(
        np.arange(n_species) * _DENDROGRAM_LEAF_SPACING + _DENDROGRAM_LEAF_SPACING / 2.0
    )
    axes_heatmap.set_yticklabels(phylogeny_leaf_order, fontsize=8)
    axes_heatmap.tick_params(axis="y", length=0, pad=4)
    for tick_label in axes_heatmap.get_yticklabels():
        tick_label.set_color(_get_species_color(anndata_object, tick_label.get_text()))
        tick_label.set_fontweight("bold")
    axes_heatmap.set_xticks(np.arange(len(feature_order)) + 0.5)
    axes_heatmap.set_xticklabels(feature_order, rotation=45, ha="right", fontsize=7)
    axes_heatmap.tick_params(axis="x", length=0)
    for spine in axes_heatmap.spines.values():
        spine.set_visible(False)

    if group_features_by_category:
        axes_category_strip = figure.add_subplot(grid[0, 1])
        block_boundaries = _draw_category_strip(axes_category_strip, feature_order)
        # A wider white line at each category boundary separates the blocks
        # visually without hiding any cell.
        for boundary in block_boundaries:
            axes_heatmap.axvline(boundary, color="white", linewidth=3.0)

    _draw_connector_lines(
        axes_connectors, phylogeny_leaf_order, morphology_leaf_order, anndata_object
    )

    if support_values:
        _annotate_clade_support(
            axes_morphology, morphology_linkage, species_list, support_values
        )

    colorbar = figure.colorbar(mesh, cax=axes_colorbar)
    colorbar.ax.tick_params(labelsize=7)
    colorbar.set_label("Group mean (z-scored)", fontsize=7)

    figure.suptitle(title, fontsize=11, y=1.03)
    return figure


def _build_default_title(
    group_by: str,
    cluster_method: str,
    cluster_metric: str,
    standardize_group_means: bool,
    nest_aggregate_by_columns: Optional[List[str]],
    baker_gamma: float,
) -> str:
    """
    Build the auto-generated figure title.

    Every analytical choice that changes the result is named in the title, so
    a figure pasted into a slide still carries its own provenance.

    Arguments:
        group_by: The .obs column grouped on.
        cluster_method: scipy linkage method used for the morphology tree.
        cluster_metric: scipy distance metric used for the morphology tree.
        standardize_group_means: Whether the z-score lever was on.
        nest_aggregate_by_columns: Nesting columns, or None when pooled.
        baker_gamma: The headline agreement statistic.

    Returns:
        Two-line title string.
    """
    aggregation = (
        f"nested by {' > '.join(nest_aggregate_by_columns)}"
        if nest_aggregate_by_columns
        else "pooled events"
    )
    return (
        f"Phylogeny vs mitochondrial morphology — grouped by '{group_by}'\n"
        f"(linkage: {cluster_method}/{cluster_metric} | "
        f"z-scored across {group_by}: {standardize_group_means} | {aggregation} | "
        f"Baker's gamma = {baker_gamma:.3f})"
    )


# ---------------------------------------------------------------------------
# Private helpers — QC output (CLAUDE.md section 6)
# ---------------------------------------------------------------------------


def _print_tanglegram_qc(
    species_feature_matrix: pd.DataFrame,
    phylogeny_leaf_order: List[str],
    morphology_leaf_order: List[str],
    morphology_linkage: np.ndarray,
    species_list: List[str],
    baker_gamma: float,
    baker_gamma_p_value: float,
    cophenetic_correlation: float,
    n_crossings_raw: int,
    n_crossings_after: int,
    leaf_order_agreement: str,
    optimize_leaf_order: bool,
    standardize_group_means: bool,
    cluster_method: str,
    cluster_metric: str,
    branch_scale: str,
    support_values: Optional[Dict[str, float]],
) -> None:
    """
    Print the tanglegram QC block.

    Everything a reader needs to check the figure without opening the code:
    matrix dimensions and value range, NaN/Inf counts, the exact parameters
    used, both leaf orders, and every agreement statistic that will appear in
    the figure legend.

    Arguments:
        species_feature_matrix: The matrix actually plotted.
        phylogeny_leaf_order: Species order of the left tree.
        morphology_leaf_order: Species order of the right tree.
        morphology_linkage: The morphology tree as drawn.
        species_list: Species in linkage index order.
        baker_gamma: Headline agreement statistic.
        baker_gamma_p_value: Its exact permutation p-value.
        cophenetic_correlation: How faithfully the morphology dendrogram
            represents the raw morphological distances.
        n_crossings_raw: Crossings before any leaf-order optimisation.
        n_crossings_after: Crossings as drawn.
        leaf_order_agreement: Human-readable leaf-order summary.
        optimize_leaf_order: Whether optimal_leaf_ordering() was applied.
        standardize_group_means: Which standardization was used.
        cluster_method: scipy linkage method used.
        cluster_metric: scipy distance metric used.
        branch_scale: Which phylogeny branch scale baker_gamma was computed
            against — cophenetic distances include node heights, so this is
            part of the statistic, not just a display choice.
        support_values: Bootstrap clade support, when supplied.
    """
    values = species_feature_matrix.values
    n_species, n_features = species_feature_matrix.shape

    print("-" * 70)
    print(f"Species: {n_species} | Features: {n_features}")
    print(
        f"=> Species-mean matrix: min={np.nanmin(values):.4f}, max={np.nanmax(values):.4f}, "
        f"mean={np.nanmean(values):.4f}"
    )
    print(f"=> NaN count: {int(np.isnan(values).sum())} | Inf count: {int(np.isinf(values).sum())}")
    print(
        f"Parameters — cluster_method='{cluster_method}', cluster_metric='{cluster_metric}', "
        f"standardize_group_means={standardize_group_means}, "
        f"optimize_leaf_order={optimize_leaf_order}, branch_scale='{branch_scale}'"
    )
    print(
        "  standardize_group_means is an ANALYTICAL choice in this figure: species are the\n"
        "  observations being clustered here, so z-scoring each feature across species\n"
        "  changes the species tree and Baker's gamma. Declare it in the Methods section."
    )
    print(f"Phylogeny leaf order (top to bottom):  {phylogeny_leaf_order}")
    print(f"Morphology leaf order (top to bottom): {morphology_leaf_order}")
    print(f"Morphology topology: {linkage_to_newick_string(morphology_linkage, species_list)}")
    print("-" * 70)
    print("AGREEMENT STATISTICS (these belong in the figure legend)")
    print(f"  Baker's gamma                    : {baker_gamma:.4f}   <- headline statistic")
    print(f"  Baker's gamma exact p-value      : {baker_gamma_p_value:.4f}")
    print("    Baker's gamma reads only topology and node heights, so no leaf rotation")
    print("    can change it. The p-value enumerates every label permutation exactly.")
    print(f"    Computed against the phylogeny at branch_scale='{branch_scale}' — node heights")
    print("    are part of the comparison, so the other scale gives a different number.")
    print(f"  Cophenetic correlation           : {cophenetic_correlation:.4f}")
    print("    (how faithfully the morphology dendrogram represents the raw distances)")
    print(f"  Connector crossings, raw         : {n_crossings_raw}   <- descriptive only")
    print(f"  Connector crossings, as drawn    : {n_crossings_after}   <- descriptive only")
    print("    Crossings depend on display order, so a cosmetic rotation changes them.")
    print(f"  Leaf order agreement             : {leaf_order_agreement}")

    if support_values:
        print("-" * 70)
        print("BOOTSTRAP CLADE SUPPORT (as annotated on the morphology tree)")
        for clade_key, support in sorted(
            support_values.items(), key=lambda item: item[1], reverse=True
        ):
            flag = "  <- weak (<50%)" if support < _WEAK_SUPPORT_THRESHOLD else ""
            print(f"  {clade_key:<32} {support * 100:5.1f}%{flag}")
    print("=" * 70)


def _print_sensitivity_qc(sensitivity_table: pd.DataFrame) -> None:
    """
    Print the sensitivity sweep table and its summary statistics.

    The median and full range of Baker's gamma across every valid combination
    are the numbers that say whether the result is robust or exists only
    under one parameter choice.

    Arguments:
        sensitivity_table: The table returned by
            compute_tanglegram_sensitivity().
    """
    display_columns = [
        "cluster_method", "cluster_metric", "standardize_group_means",
        "is_valid_combination", "baker_gamma", "baker_gamma_p_value",
        "n_crossings_raw", "n_expected_clades_recovered", "newick_topology",
    ]
    print("-" * 70)
    print("SENSITIVITY TABLE — sorted by Baker's gamma, descending")
    with pd.option_context("display.width", 200, "display.max_columns", 50):
        print(sensitivity_table[display_columns].to_string(index=False))

    valid_rows = sensitivity_table[
        sensitivity_table["is_valid_combination"] & sensitivity_table["baker_gamma"].notna()
    ]
    print("-" * 70)
    print(f"Combinations swept: {len(sensitivity_table)} "
          f"({len(valid_rows)} strictly valid method/metric pairs)")
    if len(valid_rows) > 0:
        gamma_values = valid_rows["baker_gamma"]
        print(
            f"Baker's gamma across valid combinations: median={gamma_values.median():.4f}, "
            f"range=[{gamma_values.min():.4f}, {gamma_values.max():.4f}]"
        )
        n_all_three = int((valid_rows["n_expected_clades_recovered"] == 3).sum())
        print(
            f"Valid combinations recovering all 3 expected clades: {n_all_three}/{len(valid_rows)}"
        )
        if n_all_three <= 1:
            print(
                "  => The phylogeny is recovered under at most ONE parameter combination. "
                "That is a FRAGILE result and the paper must say so."
            )
    invalid_rows = sensitivity_table[~sensitivity_table["is_valid_combination"]]
    if len(invalid_rows) > 0:
        print(
            f"Flagged (not dropped): {len(invalid_rows)} combination(s) whose linkage method "
            "is only strictly defined for euclidean distance."
        )
    print("=" * 70)


def _print_clade_support_qc(
    clade_support: Dict[str, float],
    n_completed: int,
    n_bootstrap: int,
    cluster_method: str,
    cluster_metric: str,
    standardize_group_means: bool,
    nest_aggregate_by_columns: List[str],
) -> None:
    """
    Print the bootstrap clade support QC block.

    Arguments:
        clade_support: Clade key -> support fraction.
        n_completed: Replicates that produced a usable tree.
        n_bootstrap: Replicates requested.
        cluster_method: scipy linkage method used.
        cluster_metric: scipy distance metric used.
        standardize_group_means: Which standardization was used.
        nest_aggregate_by_columns: The nesting columns; the last one is the
            unit that was resampled.
    """
    print("-" * 70)
    print(
        f"Parameters — resampled unit='{nest_aggregate_by_columns[-1]}', "
        f"cluster_method='{cluster_method}', cluster_metric='{cluster_metric}', "
        f"standardize_group_means={standardize_group_means}"
    )
    print(f"Replicates: {n_completed}/{n_bootstrap} usable "
          f"({n_bootstrap - n_completed} degenerate, skipped)")
    print("Clade support (fraction of bootstrap replicates containing the clade):")
    for clade_key, support in sorted(clade_support.items(), key=lambda item: item[1], reverse=True):
        is_expected = clade_key in TEM_EXPECTED_PHYLOGENETIC_CLADES
        marker = "  [expected by phylogeny]" if is_expected else ""
        flag = "  <- weak (<50%)" if support < _WEAK_SUPPORT_THRESHOLD else ""
        print(f"  {clade_key:<32} {support * 100:5.1f}%{marker}{flag}")
    print("-" * 70)
    print("Support of the three clades the reference phylogeny predicts:")
    for clade_key in TEM_EXPECTED_PHYLOGENETIC_CLADES:
        support = clade_support.get(clade_key)
        if support is None:
            print(f"  {clade_key:<32}  n/a (species not present in this dataset)")
        else:
            print(f"  {clade_key:<32} {support * 100:5.1f}%")
    print("=" * 70)
