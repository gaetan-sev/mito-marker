"""
ml_bagging.py

Bag creation for the mito-marker ML pipeline.

The "Bags" strategy aggregates individual mitochondria (or cells) from each
biological subject into statistical summary rows before model training.  This
reduces the problem from "thousands of correlated mitos per subject" to "N bags
per subject", where each bag captures the distribution of the sampled mitos via
mean, std, median, and skew statistics.

Why bags instead of sklearn.ensemble.BaggingClassifier:
    BaggingClassifier trains multiple models on random subsets and combines their
    predictions.  Our bags are a *data transformation step*: we create new
    training examples (one bag = one row of statistics) BEFORE any model is
    trained.  The resulting dataset is then passed to a single model.

Optional cluster-fraction features (n_clusters argument):
    Mean/std/median/skew summarise a bag's mitochondria as if they came from one
    population.  A subject whose mitochondria actually form two distinct
    sub-populations (e.g. 30% fragmented, 70% intact) looks the same as a subject
    whose whole population shifted 30% of the way there once only the mean is
    kept.  When n_clusters is set, a KMeans(n_clusters=n_clusters) is fit ONCE on
    every subject's mitochondria pooled together (unsupervised — no target label
    involved), and each bag gets one extra column per cluster: the fraction of
    that bag's sampled mitochondria falling in that cluster (n_clusters numbers
    per bag, summing to 1). Validated across all 6 TEM species in
    mito-marker-lab, EXP-005 (controlled cluster grid) and EXP-006 (n_clusters sweep).

Sampling mode and bag overlap (sampling_mode / max_overlap_fraction arguments):
    Two bags drawn from the same subject share a predictable fraction of that
    subject's mitochondria.  When mitos_per_bag approaches the subject's total
    row count, that fraction approaches 1: the bags become near-duplicates of
    each other.  They still count as separate rows in the model's training set
    and in every downstream sample count, so a subject with few mitos can be
    counted N times while contributing roughly one independent observation.
    This silently inflates apparent sample size and any confidence interval
    derived from it.

    Expected overlap = expected fraction of one bag's distinct rows that also
    appear in another bag of the same subject:
      - without replacement: mitos_per_bag / n_rows_for_this_subject
      - with replacement:    1 − (1 − 1/n_rows) ** mitos_per_bag

    sampling_mode makes the draw explicit instead of implicit, and every mode
    prints a per-subject QC table naming the mode actually used, the expected
    overlap, and how many fully disjoint bags the subject could support.
    Subjects above max_overlap_fraction are flagged in that table and also raise
    a UserWarning so programmatic callers can catch them.

Public API:
    create_bags(data_matrix, obs_dataframe, subject_id_column, ...)
        → (bag_feature_matrix, bag_targets, bag_subject_ids, bag_feature_names)
    ALLOWED_BAG_SAMPLING_MODES — allowed values for sampling_mode.

Internal helpers:
    _build_bag_feature_names(original_names, statistics)
    _build_sampling_plan(subject_row_counts, mitos_per_bag, sampling_mode)
    _print_bag_sampling_qc(sampling_plan, ...)
    _bag_one_subject(subject_matrix, ...)
"""

import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import skew as scipy_skew
from sklearn.cluster import KMeans

# Allowed values for the sampling_mode argument of create_bags().
# Keys are the accepted strings, values are plain-English descriptions printed
# back to the user in the QC block so the console is self-documenting.
ALLOWED_BAG_SAMPLING_MODES: Dict[str, str] = {
    "auto": (
        "Draw without replacement when the subject has enough rows, fall back to "
        "with replacement only for subjects that have fewer rows than mitos_per_bag"
    ),
    "without_replacement": (
        "Always draw distinct rows — every bag is a true subset of the subject. "
        "Raises an error if any subject has fewer rows than mitos_per_bag"
    ),
    "with_replacement": (
        "Always draw with replacement (bootstrap) — a row may appear several times "
        "in the same bag, and every subject is sampled identically regardless of size"
    ),
}

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def create_bags(
    data_matrix: np.ndarray,
    obs_dataframe: pd.DataFrame,
    subject_id_column: str,
    bags_per_subject: int,
    mitos_per_bag: int,
    bag_statistics: List[str],
    target_obs_column: str,
    task_type: str,
    random_state: int = 42,
    n_jobs: int = -1,
    feature_names: Optional[List[str]] = None,
    n_clusters: Optional[int] = None,
    sampling_mode: str = "auto",
    max_overlap_fraction: float = 0.2,
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """
    Aggregate individual rows (mitos or cells) into statistical bags, one subject at a time.

    For each subject, draws `bags_per_subject` independent random samples of
    `mitos_per_bag` rows.  Each sample becomes one bag row: the requested
    statistics (mean, std, median, skew) of each original feature are computed
    and concatenated.  bag_statistics may be an empty list when n_clusters is
    set, for a bag made of cluster fractions only.

    The target label for each bag is inherited from the subject:
      - Classification: the subject's group value (e.g. "Young").
      - Regression:     the subject's numeric target value (e.g. age = 32.5).
    This is valid because all mitos from one subject always share the same
    biological label.

    Execution is parallelised with joblib over subjects.  Each worker receives
    a distinct seed derived from random_state, guaranteeing bit-identical results
    when random_state is the same across runs.

    Optional cluster-fraction features (n_clusters):
      When n_clusters is not None, a single KMeans(n_clusters=n_clusters) is fit
      here — once, before the per-subject parallel loop, on data_matrix pooled
      across every subject (unsupervised, no target label involved). Each bag
      then gets n_clusters extra columns appended after the statistics columns:
      the fraction of that bag's sampled rows falling in each cluster (the same
      sampled rows used for the mean/std/median/skew columns, so both describe
      the identical draw). This fits the same "compute once, reuse per bag"
      simplification the rest of this pipeline already makes for feature
      selection — it is not refit per subject or per bag.

    Sampling mode:
      The draw is resolved once per subject, in this function (not inside the
      joblib workers), so the QC block below is printed from the parent process
      and always reaches the console in the order the user expects.  Whatever
      the mode, a per-subject table reports the mode actually used, the expected
      overlap between two bags of that subject, and how many fully disjoint bags
      that subject could support.  Subjects whose expected overlap exceeds
      max_overlap_fraction are flagged in the table and raise a UserWarning.

    Arguments:
        data_matrix:        2D float array of shape (n_obs, n_features).
        obs_dataframe:      DataFrame aligned with data_matrix rows (.obs).
        subject_id_column:  Column in obs_dataframe used to identify subjects.
        bags_per_subject:   Number of bags to draw per subject.
        mitos_per_bag:      Number of rows to aggregate per bag.
        bag_statistics:     Subset of ["mean", "std", "median", "skew"] to compute.
                            May be empty only if n_clusters is set.
        target_obs_column:  .obs column providing the bag target label.
        task_type:          "classification" or "regression".
        random_state:       Master seed; per-subject seeds = (random_state + index) % (2^31-1).
                            Also seeds the KMeans fit when n_clusters is set.
        n_jobs:             Passed to joblib.Parallel (-1 uses all available cores).
        feature_names:      Optional list of original feature names aligned with data_matrix
                            columns.  When provided and data_matrix is a numpy array, these
                            names are used instead of the generic "feature_i" fallback.
        n_clusters:         None (default) disables cluster-fraction features. An int >= 2
                            fits one global KMeans and appends that many fraction columns
                            per bag, named "cluster_0__fraction" ... "cluster_{k-1}__fraction".
        sampling_mode:      One of ALLOWED_BAG_SAMPLING_MODES:
                              "auto" (default)      — without replacement where possible,
                                                      with replacement for subjects too small.
                                                      This is the historical behaviour.
                              "without_replacement" — every bag is a true subset of distinct
                                                      rows; raises ValueError if any subject
                                                      has fewer rows than mitos_per_bag.
                              "with_replacement"    — bootstrap draw for every subject, so all
                                                      subjects are sampled identically no
                                                      matter how many rows they have.
        max_overlap_fraction: QC threshold in (0, 1]. A subject whose two bags are expected to
                            share more than this fraction of their distinct rows is flagged in
                            the QC table and raises a UserWarning. Default 0.2 — i.e. flag as
                            soon as mitos_per_bag exceeds 20% of a subject's row count. Set to
                            1.0 to disable the flagging (the QC table is still printed).
        verbose:            True (default) prints the per-subject sampling QC block. Set False
                            for loops that call create_bags() many times (e.g. the permutation
                            test). The UserWarning is emitted either way.

    Raises:
        ValueError: sampling_mode is not in ALLOWED_BAG_SAMPLING_MODES, max_overlap_fraction is
                    outside (0, 1], or sampling_mode="without_replacement" was requested while
                    at least one subject has fewer rows than mitos_per_bag.

    Returns:
        Tuple of four elements:
          bag_feature_matrix  — 2D float array (n_bags_total, n_bag_features)
          bag_target_labels   — 1D array (n_bags_total,) — str for classification, float for regression
          bag_subject_ids     — 1D str array (n_bags_total,) — subject of origin per bag
          bag_feature_names   — list[str] of feature names, e.g. "Mito_Area__mean",
                                "cluster_0__fraction"
    """
    if not bag_statistics and not n_clusters:
        raise ValueError(
            "create_bags() needs at least one feature source: a non-empty "
            "bag_statistics list, an n_clusters value, or both."
        )
    if sampling_mode not in ALLOWED_BAG_SAMPLING_MODES:
        raise ValueError(
            f"create_bags() sampling_mode={sampling_mode!r} is not allowed. "
            f"Allowed values: {sorted(ALLOWED_BAG_SAMPLING_MODES)}"
        )
    if not 0.0 < max_overlap_fraction <= 1.0:
        raise ValueError(
            f"create_bags() max_overlap_fraction must be in (0, 1], "
            f"got {max_overlap_fraction}"
        )

    # Sort subjects alphabetically so that per-subject seeds are deterministic
    # regardless of insertion order in the original DataFrame.
    unique_subjects = sorted(obs_dataframe[subject_id_column].unique())

    # Resolve the draw for every subject here, in the parent process, rather than
    # inside each joblib worker: a worker's print()/warnings.warn() is not reliably
    # forwarded to the notebook, so QC emitted there can silently vanish. Deciding
    # up front also lets the QC block report the whole cohort in one table.
    subject_row_counts = {
        subject_id: int((obs_dataframe[subject_id_column].values == subject_id).sum())
        for subject_id in unique_subjects
    }
    sampling_plan = _build_sampling_plan(
        subject_row_counts=subject_row_counts,
        mitos_per_bag=mitos_per_bag,
        sampling_mode=sampling_mode,
    )
    _report_bag_sampling_qc(
        sampling_plan=sampling_plan,
        bags_per_subject=bags_per_subject,
        mitos_per_bag=mitos_per_bag,
        sampling_mode=sampling_mode,
        max_overlap_fraction=max_overlap_fraction,
        verbose=verbose,
    )

    original_feature_names: List[str]
    if hasattr(data_matrix, "columns"):
        # Accept DataFrames as well as numpy arrays.
        original_feature_names = list(data_matrix.columns)
        data_matrix = np.asarray(data_matrix)
    elif feature_names is not None:
        # Caller provided the actual feature names (e.g. from anndata_object.var_names).
        original_feature_names = list(feature_names)
    else:
        # Fallback: generic names when no names are available.
        original_feature_names = [f"feature_{i}" for i in range(data_matrix.shape[1])]

    bag_feature_names = _build_bag_feature_names(original_feature_names, bag_statistics)

    # Fit the global cluster assignment ONCE, before the per-subject parallel loop —
    # unsupervised, on every subject's mitochondria pooled together. Passing per-subject
    # slices of this single label array into each worker keeps the clustering itself
    # outside the parallelised, per-subject sampling loop.
    cluster_labels: Optional[np.ndarray] = None
    if n_clusters:
        kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
        cluster_labels = kmeans.fit_predict(data_matrix)
        bag_feature_names = bag_feature_names + [
            f"cluster_{cluster_index}__fraction" for cluster_index in range(n_clusters)
        ]

    # Parallel execution: one job per subject.
    per_subject_results = Parallel(n_jobs=n_jobs)(
        delayed(_bag_one_subject)(
            subject_matrix=data_matrix[
                obs_dataframe[subject_id_column].values == subject_id
            ],
            bags_per_subject=bags_per_subject,
            mitos_per_bag=mitos_per_bag,
            bag_statistics=bag_statistics,
            subject_target_labels=obs_dataframe.loc[
                obs_dataframe[subject_id_column] == subject_id, target_obs_column
            ].values,
            subject_id=subject_id,
            task_type=task_type,
            use_replacement=sampling_plan[subject_id]["use_replacement"],
            seed=(random_state + subject_index) % (2**31 - 1),
            subject_cluster_labels=(
                cluster_labels[obs_dataframe[subject_id_column].values == subject_id]
                if cluster_labels is not None
                else None
            ),
            n_clusters=n_clusters or 0,
        )
        for subject_index, subject_id in enumerate(unique_subjects)
    )

    # Concatenate results from all subjects.
    all_bag_matrices = [result[0] for result in per_subject_results]
    all_bag_targets = [result[1] for result in per_subject_results]
    all_bag_subject_ids = [result[2] for result in per_subject_results]

    bag_feature_matrix = np.vstack(all_bag_matrices)
    bag_target_labels = np.concatenate(all_bag_targets)
    bag_subject_ids = np.concatenate(all_bag_subject_ids)

    return bag_feature_matrix, bag_target_labels, bag_subject_ids, bag_feature_names


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _build_bag_feature_names(
    original_feature_names: List[str],
    bag_statistics: List[str],
) -> List[str]:
    """
    Build the ordered list of bag feature column names.

    For each original feature, all requested statistics are appended in the
    order they appear in bag_statistics.  Double underscores match the layer
    naming convention used elsewhere in the package.

    Arguments:
        original_feature_names: Feature names from anndata_object.var_names.
        bag_statistics:         Ordered list of statistics to compute.

    Returns:
        list[str] of length len(original_feature_names) * len(bag_statistics),
        e.g. ["Mito_Area__mean", "Mito_Area__std", "Mito_AR__mean", "Mito_AR__std"]
    """
    bag_names: List[str] = []
    for feature_name in original_feature_names:
        for stat in bag_statistics:
            bag_names.append(f"{feature_name}__{stat}")
    return bag_names


def _build_sampling_plan(
    subject_row_counts: Dict[str, int],
    mitos_per_bag: int,
    sampling_mode: str,
) -> Dict[str, Dict[str, float]]:
    """
    Decide how each subject will be sampled and quantify how redundant its bags will be.

    Two bags drawn from the same subject overlap by a predictable amount, and that
    amount is what decides whether N bags carry N observations' worth of information
    or one observation repeated N times. Both formulas below express the same
    quantity — the expected fraction of one bag's DISTINCT rows that also appear in
    another bag of the same subject — so the number is comparable across modes:

      - without replacement: a bag holds k distinct rows out of n, and a second
        independent bag hits each of them with probability k/n, so the expected
        shared count is k²/n and the overlap fraction is k/n.
      - with replacement: a given row is missed by all k draws with probability
        (1 − 1/n)**k, so it appears in a bag with probability p = 1 − (1 − 1/n)**k.
        A bag then holds n·p distinct rows on average, two bags share n·p² of them,
        and the ratio is again p. The two formulas agree when k << n and diverge
        only in the regime this QC exists to catch.

    Arguments:
        subject_row_counts: Mapping subject_id → number of rows available for that subject.
        mitos_per_bag:      Number of rows drawn per bag.
        sampling_mode:      One of ALLOWED_BAG_SAMPLING_MODES.

    Raises:
        ValueError: sampling_mode is "without_replacement" and at least one subject
                    has fewer rows than mitos_per_bag, which makes a distinct-row
                    draw impossible.

    Returns:
        dict keyed by subject_id, each value a dict with keys:
          n_available            — rows available for this subject (int)
          use_replacement        — whether this subject is drawn with replacement (bool)
          distinct_rows_per_bag  — expected number of distinct rows in one bag (float)
          overlap_fraction       — expected overlap between two bags, in [0, 1] (float)
          max_disjoint_bags      — how many fully non-overlapping bags fit in this
                                   subject, i.e. n_available // mitos_per_bag (int)
    """
    if sampling_mode == "without_replacement":
        too_small = {
            subject_id: n_rows
            for subject_id, n_rows in subject_row_counts.items()
            if n_rows < mitos_per_bag
        }
        if too_small:
            detail = ", ".join(
                f"'{subject_id}' ({n_rows} rows)"
                for subject_id, n_rows in sorted(too_small.items())
            )
            raise ValueError(
                f"sampling_mode='without_replacement' requires every subject to have at "
                f"least mitos_per_bag={mitos_per_bag} rows, but "
                f"{len(too_small)} subject(s) have fewer: {detail}. "
                f"Lower mitos_per_bag to at most {min(subject_row_counts.values())}, drop "
                f"the undersized subjects, or use sampling_mode='with_replacement' "
                f"(bootstrap) or 'auto' (per-subject fallback) instead."
            )

    sampling_plan: Dict[str, Dict[str, float]] = {}
    for subject_id, n_available in subject_row_counts.items():
        if sampling_mode == "with_replacement":
            use_replacement = True
        elif sampling_mode == "without_replacement":
            use_replacement = False
        else:
            # "auto": replacement is a fallback for subjects that cannot supply
            # mitos_per_bag distinct rows.
            use_replacement = n_available < mitos_per_bag

        if use_replacement:
            probability_row_appears = 1.0 - (1.0 - 1.0 / n_available) ** mitos_per_bag
            distinct_rows_per_bag = n_available * probability_row_appears
            overlap_fraction = probability_row_appears
        else:
            distinct_rows_per_bag = float(mitos_per_bag)
            overlap_fraction = mitos_per_bag / n_available

        sampling_plan[subject_id] = {
            "n_available":           n_available,
            "use_replacement":       use_replacement,
            "distinct_rows_per_bag": distinct_rows_per_bag,
            "overlap_fraction":      min(1.0, overlap_fraction),
            "max_disjoint_bags":     n_available // mitos_per_bag,
        }

    return sampling_plan


def _report_bag_sampling_qc(
    sampling_plan: Dict[str, Dict[str, float]],
    bags_per_subject: int,
    mitos_per_bag: int,
    sampling_mode: str,
    max_overlap_fraction: float,
    verbose: bool,
) -> None:
    """
    Print the per-subject sampling QC table and warn about near-duplicate bags.

    Printed for every sampling mode, not only the degenerate ones, so a reader of
    the console log can always tell exactly how each subject was drawn without
    reading Python code. Subjects above max_overlap_fraction are additionally
    raised as a UserWarning so non-interactive callers and tests can catch them.

    Arguments:
        sampling_plan:        Output of _build_sampling_plan().
        bags_per_subject:     Number of bags requested per subject.
        mitos_per_bag:        Number of rows drawn per bag.
        sampling_mode:        One of ALLOWED_BAG_SAMPLING_MODES.
        max_overlap_fraction: Threshold above which a subject is flagged.
        verbose:              False silences the printed table (the warning still fires).

    Returns:
        None.
    """
    flagged_subjects = [
        subject_id
        for subject_id, plan in sampling_plan.items()
        if plan["overlap_fraction"] > max_overlap_fraction
    ]
    replacement_subjects = [
        subject_id
        for subject_id, plan in sampling_plan.items()
        if plan["use_replacement"]
    ]
    smallest_subject_id = min(
        sampling_plan, key=lambda subject_id: sampling_plan[subject_id]["n_available"]
    )
    smallest_row_count = int(sampling_plan[smallest_subject_id]["n_available"])

    if verbose:
        print()
        print(f"  [Bag sampling QC] mode: '{sampling_mode}'")
        print(f"    {ALLOWED_BAG_SAMPLING_MODES[sampling_mode]}.")
        print(
            "    Overlap = expected % of one bag's distinct rows also present in another"
        )
        print(
            "    bag of the SAME subject. High overlap means near-duplicate bags: they"
        )
        print(
            "    inflate the row count fed to the model without adding new information,"
        )
        print(
            "    so accuracy and confidence intervals computed per bag become optimistic."
        )
        print()
        print(
            f"    {'Subject':<24} {'Rows':>8} {'Sampling':<20} "
            f"{'Distinct/bag':>12} {'Overlap':>8} {'Disjoint bags':>14}"
        )
        print(f"    {'-' * 24} {'-' * 8} {'-' * 20} {'-' * 12} {'-' * 8} {'-' * 14}")
        for subject_id, plan in sorted(sampling_plan.items()):
            draw_label = "with replacement" if plan["use_replacement"] else "without replacement"
            flag = "  <-- OVERLAP" if plan["overlap_fraction"] > max_overlap_fraction else ""
            print(
                f"    {str(subject_id):<24} {int(plan['n_available']):>8,} {draw_label:<20} "
                f"{plan['distinct_rows_per_bag']:>12.0f} "
                f"{plan['overlap_fraction']:>7.1%} "
                f"{int(plan['max_disjoint_bags']):>14,}{flag}"
            )
        print()
        print(
            f"    => Requested {bags_per_subject} bags/subject × {mitos_per_bag} rows/bag."
        )
        if replacement_subjects:
            print(
                f"    => Drawn WITH replacement: {len(replacement_subjects)}/{len(sampling_plan)} "
                f"subject(s) — {', '.join(str(s) for s in sorted(replacement_subjects))}. "
                f"Rows repeat inside a bag; 'Distinct/bag' is the real information content."
            )
        else:
            print(
                f"    => Drawn WITHOUT replacement: all {len(sampling_plan)} subject(s). "
                f"Every bag holds {mitos_per_bag} distinct rows."
            )
        if flagged_subjects:
            largest_safe_bag_size = max(1, int(max_overlap_fraction * smallest_row_count))
            print(
                f"    => {len(flagged_subjects)}/{len(sampling_plan)} subject(s) exceed the "
                f"max_overlap_fraction={max_overlap_fraction:.0%} threshold: "
                f"{', '.join(str(s) for s in sorted(flagged_subjects))}."
            )
            print(
                f"    => Their bags are largely the same rows re-drawn, so they contribute "
                f"far fewer than {bags_per_subject} independent observations each."
            )
            print(
                f"    => To respect the threshold for every subject, use "
                f"mitos_per_bag <= {largest_safe_bag_size} "
                f"({max_overlap_fraction:.0%} of the {smallest_row_count:,} rows in the "
                f"smallest subject '{smallest_subject_id}')."
            )
            print(
                "    => Report sample size and confidence intervals per SUBJECT, not per bag."
            )
        else:
            print(
                f"    => All subjects are at or below the "
                f"max_overlap_fraction={max_overlap_fraction:.0%} threshold."
            )

    # Warn outside the verbose guard: a silenced console must not silence the
    # signal that the bags are redundant.
    if flagged_subjects:
        flagged_detail = ", ".join(
            f"'{subject_id}' ({sampling_plan[subject_id]['overlap_fraction']:.0%} overlap, "
            f"{int(sampling_plan[subject_id]['n_available'])} rows)"
            for subject_id in sorted(flagged_subjects)
        )
        warnings.warn(
            f"Bag overlap above max_overlap_fraction={max_overlap_fraction:.0%} for "
            f"{len(flagged_subjects)}/{len(sampling_plan)} subject(s) with "
            f"mitos_per_bag={mitos_per_bag}: {flagged_detail}. "
            f"Bags from these subjects are near-duplicates and inflate the apparent "
            f"sample size — lower mitos_per_bag or report metrics per subject.",
            UserWarning,
            stacklevel=3,
        )
    if replacement_subjects:
        warnings.warn(
            f"Sampling WITH replacement for {len(replacement_subjects)}/{len(sampling_plan)} "
            f"subject(s) with mitos_per_bag={mitos_per_bag}: "
            f"{', '.join(str(s) for s in sorted(replacement_subjects))}. "
            f"Rows repeat within a bag, so a bag holds fewer distinct rows than "
            f"mitos_per_bag suggests.",
            UserWarning,
            stacklevel=3,
        )


def _bag_one_subject(
    subject_matrix: np.ndarray,
    bags_per_subject: int,
    mitos_per_bag: int,
    bag_statistics: List[str],
    subject_target_labels: np.ndarray,
    subject_id: str,
    task_type: str,
    use_replacement: bool,
    seed: int,
    subject_cluster_labels: Optional[np.ndarray] = None,
    n_clusters: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create bags for a single subject.  Intended to run in a joblib worker.

    Whether this subject is drawn with replacement is decided by the caller
    (_build_sampling_plan, run from the parent process) and passed in, so no QC
    decision or QC message originates inside a worker — worker output is not
    reliably forwarded to the notebook console.

    Arguments:
        subject_matrix:       2D float array (n_mitos_for_this_subject, n_features).
        bags_per_subject:     Number of bags to draw.
        mitos_per_bag:        Number of rows per bag.
        bag_statistics:       Statistics to compute per feature.
        subject_target_labels: 1D array of target values for this subject's rows.
        subject_id:           Subject identifier string (used in the returned id array).
        task_type:            "classification" or "regression".
        use_replacement:      Resolved by _build_sampling_plan(); True draws rows with
                              replacement, False draws distinct rows.
        seed:                 Per-subject random seed (derived from master random_state).
        subject_cluster_labels: Optional 1D int array, aligned with subject_matrix rows,
                                giving each row's pre-fitted global cluster index. When
                                given, n_clusters extra fraction columns are appended.
        n_clusters:           Number of clusters (0 disables the fraction columns).

    Returns:
        Tuple of:
          (n_bags, n_bag_features) float array,
          (n_bags,) target array,
          (n_bags,) subject_id array
    """
    n_available = subject_matrix.shape[0]

    rng = np.random.default_rng(seed)
    n_features = subject_matrix.shape[1]
    n_stats = len(bag_statistics)
    n_stat_columns = n_features * n_stats
    bag_matrix = np.empty((bags_per_subject, n_stat_columns + n_clusters), dtype=np.float64)

    for bag_index in range(bags_per_subject):
        sampled_indices = rng.choice(
            n_available,
            size=mitos_per_bag,
            replace=use_replacement,
        )
        sampled_rows = subject_matrix[sampled_indices]

        # Compute each requested statistic across the sampled rows.
        stat_arrays: List[np.ndarray] = []
        for stat in bag_statistics:
            if stat == "mean":
                stat_arrays.append(sampled_rows.mean(axis=0))
            elif stat == "std":
                stat_arrays.append(sampled_rows.std(axis=0, ddof=1))
            elif stat == "median":
                stat_arrays.append(np.median(sampled_rows, axis=0))
            elif stat == "skew":
                stat_arrays.append(scipy_skew(sampled_rows, axis=0))

        # Interleave statistics: [f0_mean, f0_std, f1_mean, f1_std, ...]
        # This matches the order produced by _build_bag_feature_names.
        for feature_index in range(n_features):
            for stat_index, stat_array in enumerate(stat_arrays):
                bag_matrix[bag_index, feature_index * n_stats + stat_index] = (
                    stat_array[feature_index]
                )

        # Cluster fractions, appended after the statistics columns. Uses the SAME
        # sampled_indices as the statistics above, so both describe the identical draw.
        if n_clusters:
            sampled_cluster_labels = subject_cluster_labels[sampled_indices]
            for cluster_index in range(n_clusters):
                bag_matrix[bag_index, n_stat_columns + cluster_index] = float(
                    np.mean(sampled_cluster_labels == cluster_index)
                )

    # Determine the target label for this subject's bags.
    # All mitos from one subject share the same biological label, so we take the
    # first non-NaN value as the subject's representative label.
    valid_labels = subject_target_labels[
        ~pd.isnull(subject_target_labels)
    ]
    if len(valid_labels) == 0:
        representative_label = np.nan
    elif task_type == "regression":
        representative_label = float(np.mean(valid_labels.astype(float)))
    else:
        # Classification: use the most frequent label (mode).
        unique_labels, counts = np.unique(valid_labels.astype(str), return_counts=True)
        representative_label = unique_labels[np.argmax(counts)]

    bag_targets = np.full(bags_per_subject, representative_label)
    bag_subject_ids = np.array([subject_id] * bags_per_subject)

    return bag_matrix, bag_targets, bag_subject_ids
