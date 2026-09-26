"""
permutation_test.py

Subject-level exact / Monte Carlo permutation test for run_ml_analysis() results.

A cross-validated accuracy (or R²) can look good by chance alone, especially
with the small subject counts typical of TEM/SFC cohorts (often 6-12 subjects).
This module answers: "out of every way the Young/Old (or other) labels could
have been assigned to these subjects, how many would have scored as well as
the real labels did?" That fraction is the p-value.

Public API:
    run_permutation_test(anndata_object, ml_config, ...)
        → dict with observed_metric, null_metrics, p_value, ... Also stores
          the same dict in anndata_object.uns['permutation_test_results'] and
          draws a publication-ready histogram of the null distribution with
          the observed value marked.

Design notes
------------
Permutation happens at the SUBJECT level, never at the individual
mitochondrion/cell level: every row belonging to a subject always gets that
subject's (permuted) label, exactly like the real data (all of one subject's
mitos share one biological label). Permuting individual rows instead would
destroy the subject structure the whole LOGO/Bags pipeline is built around
and silently inflate the null distribution's spread.

Binary classification (exactly 2 unique subject-level label values) uses the
EXACT test: every way to choose which subjects get the minority label
(C(n_subjects, n_minority) assignments — see itertools.combinations). This is
tractable for the subject counts this package is designed for (a few hundred
assignments at most) and gives a mathematically exact p-value, not an
approximation.

Multi-class classification and regression targets have no such compact exact
form (the number of distinct label arrangements grows combinatorially, or is
simply the sample space of a continuous variable) — these fall back to a
Monte Carlo permutation test (random re-shuffles of the subject-level label
array), which is the standard approach in that setting.

Every run re-evaluates the full ml_config pipeline (Bags/SingleMito/MIL,
StandardCV/LOGO, feature selection) unchanged, via run_ml_analysis() itself
— only the labels differ between permutations — so the null distribution is
directly comparable to the real, trusted result. SHAP and plotting are
suppressed during the permutation loop itself for speed and console
cleanliness; only the final null-distribution figure is shown.
"""

import contextlib
import io
import itertools
import math
import warnings
from typing import Any, Dict, List, Optional

import anndata
import numpy as np
import pandas as pd

from mito_marker.analysis.ml_config import _validate_ml_config
from mito_marker.analysis.ml_pipeline import _build_species_label, run_ml_analysis

_PERMUTATION_RESULTS_KEY = "permutation_test_results"

# Above this many exact combinations, exact enumeration is not attempted by
# default (it would be intractable) — a Monte Carlo subsample is used instead.
_EXACT_COMBINATION_SAFETY_LIMIT = 5000
_DEFAULT_MONTE_CARLO_PERMUTATIONS = 1000


def run_permutation_test(
    anndata_object: anndata.AnnData,
    ml_config: dict,
    max_permutations: Optional[int] = None,
    n_seeds_per_permutation: int = 1,
    random_state: int = 42,
) -> dict:
    """
    Run a subject-level permutation test for the model/strategy described by ml_config.

    Re-runs run_ml_analysis() once per label permutation (SHAP and plots
    suppressed) to build a null distribution of the primary metric
    (mean_accuracy for classification, mean_r2 for regression), then compares
    the real labels' score against it.

    Arguments:
        anndata_object:          Same AnnData already used with run_ml_analysis()
                                 for the "real" result. Its .obs must contain
                                 ml_config['target_obs_column'] and
                                 ml_config['subject_id_column']. Not mutated —
                                 all work happens on an internal copy.
        ml_config:               Same config dict used for the real run. compute_shap
                                 is forced to False for every permutation (SHAP on
                                 randomised labels is meaningless and would be slow).
        max_permutations:        Caps the number of label assignments evaluated.
                                 None (default):
                                   - binary classification: exact test if
                                     C(n_subjects, n_minority) <=
                                     _EXACT_COMBINATION_SAFETY_LIMIT, else a
                                     capped Monte Carlo subsample of that size.
                                   - multi-class / regression: 1000 Monte Carlo
                                     shuffles (no compact exact form exists).
                                 An int: used as the cap directly — e.g. for a
                                 cohort where the exact count is large (a 12-subject
                                 6-vs-6 split has C(12,6)=924 assignments), pass
                                 max_permutations=200 to trade some p-value
                                 precision for a ~4-5x faster run.
        n_seeds_per_permutation: Evaluate each permutation this many times (different
                                 model random_state each time) and average the metric.
                                 Default 1 (fastest). Increase to 2-3 for models with
                                 real fit-to-fit randomness (RandomForest, ExtraTrees,
                                 GradientBoosting, MLP) on very small cohorts, where a
                                 single seed's score can be unrepresentative — this
                                 project's own experiments (EXP-004) found a single-seed
                                 permutation test could look misleadingly significant
                                 until averaged over multiple seeds.
        random_state:            Seed for both the Monte Carlo sampling of label
                                 assignments and (combined with a running offset) the
                                 per-permutation model seeds.

    Returns:
        dict with keys: observed_metric, metric_name, null_metrics, p_value,
        n_permutations, is_exact, n_subjects, max_permutations_requested,
        n_seeds_per_permutation, target_obs_column, subject_id_column,
        model_name, evaluation_strategy, task_type. Also stored in
        anndata_object.uns['permutation_test_results'].
    """
    _validate_ml_config(ml_config)

    task_type = ml_config["task_type"]
    target_obs_column = ml_config["target_obs_column"]
    subject_id_column = ml_config["subject_id_column"]
    model_name = "MIL" if ml_config["strategy"] == "MIL" else ml_config["model_name"]
    evaluation_strategy = ml_config["evaluation_strategy"]
    metric_name = "mean_accuracy" if task_type == "classification" else "mean_r2"
    species_label = _build_species_label(anndata_object)

    print("=" * 60)
    print("PERMUTATION TEST")
    print("=" * 60)
    print(f"  Target column      : {target_obs_column}")
    print(f"  Subject ID column  : {subject_id_column}")
    print(f"  Model              : {model_name}")
    print(f"  Evaluation         : {evaluation_strategy}")
    print(f"  Metric             : {metric_name}")

    # ------------------------------------------------------------------
    # Step 1: subject-level true labels (mirrors run_ml_analysis's own NaN drop).
    # ------------------------------------------------------------------
    valid_obs = anndata_object.obs[~pd.isna(anndata_object.obs[target_obs_column])]
    subject_to_true_label = (
        valid_obs[[subject_id_column, target_obs_column]]
        .drop_duplicates(subset=subject_id_column)
        .set_index(subject_id_column)[target_obs_column]
    )
    subject_ids_sorted: List = sorted(subject_to_true_label.index.tolist(), key=str)
    n_subjects = len(subject_ids_sorted)
    if n_subjects < 3:
        raise ValueError(
            f"Permutation test needs at least 3 subjects with a valid "
            f"'{target_obs_column}' value, found {n_subjects}."
        )
    true_label_array = subject_to_true_label.loc[subject_ids_sorted].to_numpy()
    print(f"  Subjects           : {n_subjects}")

    rng = np.random.default_rng(random_state)

    # ------------------------------------------------------------------
    # Step 2: build the list of label assignments to evaluate (index 0 = true labels).
    # ------------------------------------------------------------------
    assignments, is_exact = _build_label_assignments(
        true_label_array=true_label_array,
        task_type=task_type,
        max_permutations=max_permutations,
        rng=rng,
    )
    n_permutations = len(assignments)
    exact_word = "exact" if is_exact else "Monte Carlo subsample"
    print(f"  Permutations       : {n_permutations} ({exact_word})")
    if n_seeds_per_permutation > 1:
        print(f"  Seeds/permutation  : {n_seeds_per_permutation} (averaged)")

    # ------------------------------------------------------------------
    # Step 3: evaluate every assignment, quietly, reusing run_ml_analysis().
    # ------------------------------------------------------------------
    working_anndata = anndata_object.copy()
    working_ml_config = dict(ml_config)
    working_ml_config["compute_shap"] = False

    observed_metric: Optional[float] = None
    null_metrics: List[float] = []
    print()
    print("  Evaluating permutations …")
    for assignment_index, label_array in enumerate(assignments):
        subject_to_permuted_label = dict(zip(subject_ids_sorted, label_array))
        permuted_column = (
            working_anndata.obs[subject_id_column].map(subject_to_permuted_label)
        )
        working_anndata.obs[target_obs_column] = permuted_column.values

        seed_metrics = [
            _evaluate_quietly(
                working_anndata, working_ml_config, metric_name,
                seed=random_state + assignment_index * n_seeds_per_permutation + seed_offset,
            )
            for seed_offset in range(n_seeds_per_permutation)
        ]
        metric_value = float(np.mean(seed_metrics))

        if assignment_index == 0:
            observed_metric = metric_value
        else:
            null_metrics.append(metric_value)

        if (assignment_index + 1) % max(1, n_permutations // 10) == 0 or assignment_index + 1 == n_permutations:
            print(f"    … {assignment_index + 1}/{n_permutations}")

    assert observed_metric is not None
    n_at_or_above = sum(1 for value in null_metrics if value >= observed_metric) + 1
    p_value = n_at_or_above / n_permutations
    p_floor = 1.0 / n_permutations

    print()
    print(f"  Observed {metric_name} (true labels) : {observed_metric:.4f}")
    print(f"  Null distribution ({len(null_metrics)} other assignments): "
          f"mean={np.mean(null_metrics):.4f}  max={np.max(null_metrics):.4f}")
    print(f"  p-value            : {p_value:.4f}  (floor = 1/{n_permutations} = {p_floor:.4f})")
    print("=" * 60)

    results: Dict[str, Any] = {
        "observed_metric":            observed_metric,
        "metric_name":                metric_name,
        "null_metrics":               null_metrics,
        "p_value":                    p_value,
        "n_permutations":             n_permutations,
        "is_exact":                   is_exact,
        "n_subjects":                 n_subjects,
        "max_permutations_requested": max_permutations,
        "n_seeds_per_permutation":    n_seeds_per_permutation,
        "target_obs_column":          target_obs_column,
        "subject_id_column":          subject_id_column,
        "model_name":                 model_name,
        "evaluation_strategy":        evaluation_strategy,
        "task_type":                  task_type,
    }

    _plot_permutation_test_result(results, species_label=species_label)

    anndata_object.uns[_PERMUTATION_RESULTS_KEY] = results
    print(f"=> Results stored in .uns['{_PERMUTATION_RESULTS_KEY}'].")
    return results


# ---------------------------------------------------------------------------
# Internal: label assignment enumeration
# ---------------------------------------------------------------------------


def _build_label_assignments(
    true_label_array: np.ndarray,
    task_type: str,
    max_permutations: Optional[int],
    rng: np.random.Generator,
):
    """
    Build the list of subject-level label assignments to evaluate.

    Index 0 of the returned list is always the true (unpermuted) label array.

    Returns:
        Tuple of (assignments: List[np.ndarray], is_exact: bool).
    """
    unique_labels = sorted(set(str(label) for label in true_label_array))

    if task_type == "classification" and len(unique_labels) == 2:
        return _build_binary_assignments(true_label_array, unique_labels, max_permutations, rng)

    # Multi-class classification or regression: no compact exact form — Monte Carlo.
    n_draws = max_permutations if max_permutations is not None else _DEFAULT_MONTE_CARLO_PERMUTATIONS
    if max_permutations is None:
        target_kind = "multi-class" if task_type == "classification" else "regression"
        print(
            f"  No max_permutations given for a {target_kind} target — exact enumeration "
            f"is intractable here, defaulting to {n_draws} Monte Carlo shuffles."
        )
    assignments = [true_label_array.copy()]
    for _ in range(n_draws):
        assignments.append(rng.permutation(true_label_array))
    return assignments, False


def _build_binary_assignments(
    true_label_array: np.ndarray,
    unique_labels: List[str],
    max_permutations: Optional[int],
    rng: np.random.Generator,
):
    """Exact (or Monte Carlo-subsampled) assignments for binary classification."""
    n_subjects = len(true_label_array)
    label_a, label_b = unique_labels
    string_labels = np.array([str(label) for label in true_label_array])
    n_a = int(np.sum(string_labels == label_a))
    exact_count = math.comb(n_subjects, n_a)

    use_exact = max_permutations is None and exact_count <= _EXACT_COMBINATION_SAFETY_LIMIT
    if max_permutations is not None and max_permutations >= exact_count:
        use_exact = True
    if max_permutations is None and exact_count > _EXACT_COMBINATION_SAFETY_LIMIT:
        print(
            f"  C({n_subjects},{n_a}) = {exact_count:,} exact assignments exceeds the "
            f"safety limit ({_EXACT_COMBINATION_SAFETY_LIMIT:,}) — subsampling "
            f"{_EXACT_COMBINATION_SAFETY_LIMIT} instead. Pass max_permutations explicitly "
            f"to control this directly."
        )
        max_permutations = _EXACT_COMBINATION_SAFETY_LIMIT

    true_combo = frozenset(int(i) for i in np.where(string_labels == label_a)[0])
    all_indices = list(range(n_subjects))

    if use_exact:
        combos = [frozenset(combo) for combo in itertools.combinations(all_indices, n_a)]
    else:
        seen = {true_combo}
        combos = [true_combo]
        max_attempts = max(max_permutations * 50, 1000)
        attempts = 0
        while len(combos) < max_permutations + 1 and attempts < max_attempts:
            candidate = frozenset(rng.choice(all_indices, size=n_a, replace=False).tolist())
            attempts += 1
            if candidate not in seen:
                seen.add(candidate)
                combos.append(candidate)

    # Ensure the true assignment is index 0, whatever order combos came in.
    combos = [true_combo] + [combo for combo in combos if combo != true_combo]

    assignments = []
    for combo in combos:
        labels = np.full(n_subjects, label_b, dtype=object)
        for index in combo:
            labels[index] = label_a
        assignments.append(labels)

    return assignments, use_exact


# ---------------------------------------------------------------------------
# Internal: one quiet evaluation
# ---------------------------------------------------------------------------


def _evaluate_quietly(
    working_anndata: anndata.AnnData,
    working_ml_config: dict,
    metric_name: str,
    seed: int,
) -> float:
    """Run run_ml_analysis() once, with console output and plots suppressed."""
    import matplotlib.pyplot as plt

    per_call_config = dict(working_ml_config)
    per_call_config["random_state"] = seed

    original_show = plt.show
    plt.show = lambda *args, **kwargs: None
    try:
        with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()):
            warnings.simplefilter("ignore")
            results = run_ml_analysis(working_anndata, ml_config=per_call_config)
    finally:
        plt.show = original_show
        plt.close("all")
    return float(results[metric_name])


# ---------------------------------------------------------------------------
# Internal: figure
# ---------------------------------------------------------------------------


def _plot_permutation_test_result(results: dict, species_label: str = "") -> None:
    """
    Draw a publication-ready histogram of the null distribution with the
    observed (true-label) metric marked, and print a short interpretation.
    """
    import matplotlib.pyplot as plt

    null_metrics = results["null_metrics"]
    observed_metric = results["observed_metric"]
    metric_name = results["metric_name"]
    metric_label = "Accuracy" if metric_name == "mean_accuracy" else "R²"

    print()
    print("  [Permutation test] Interpreting the null-distribution plot:")
    print("    Each bar is how many of the OTHER possible label assignments scored")
    print("    in that range — i.e. what this model/data could achieve on labels")
    print("    that carry no real Young/Old (or target) signal.")
    print("    The dashed line is the score obtained with the REAL labels.")
    print("    The further right of the null distribution it sits, the less likely")
    print("    the real result is a coincidence of this specific small sample.")

    figure, axis = plt.subplots(figsize=(7.5, 4.5))
    n_bins = int(np.clip(np.sqrt(len(null_metrics)) * 2, 8, 30))
    axis.hist(
        null_metrics, bins=n_bins, color="#4C72B0", alpha=0.85,
        edgecolor="white", linewidth=0.6, zorder=2,
    )
    axis.axvline(observed_metric, color="#C44E52", linewidth=2.5, linestyle="--", zorder=3)

    y_top = axis.get_ylim()[1]
    median_null = float(np.median(null_metrics))
    label_ha = "left" if observed_metric >= median_null else "right"
    label_offset = 6 if label_ha == "left" else -6
    axis.annotate(
        f"Observed\n{metric_label} = {observed_metric:.3f}",
        xy=(observed_metric, y_top * 0.96),
        xytext=(label_offset, 0), textcoords="offset points",
        color="#C44E52", fontsize=10, fontweight="bold",
        ha=label_ha, va="top",
    )

    exact_word = "exact" if results["is_exact"] else "Monte Carlo"
    p_text = (
        f"p = {results['p_value']:.4g}\n"
        f"({results['n_permutations']} {exact_word} permutations)"
    )
    axis.text(
        0.02, 0.96, p_text, transform=axis.transAxes, fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", edgecolor="#4C72B0", alpha=0.95),
    )

    species_prefix = f"{species_label}\n" if species_label else ""
    axis.set_title(
        f"{species_prefix}Permutation Test\n"
        f"{results['model_name']} | {results['evaluation_strategy']} | "
        f"Target: {results['target_obs_column']}",
        fontsize=11,
    )
    axis.set_xlabel(metric_label)
    axis.set_ylabel("Number of label permutations")
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.show()
    plt.close(figure)
