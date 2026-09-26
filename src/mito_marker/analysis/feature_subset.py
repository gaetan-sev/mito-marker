"""
feature_subset.py

Non-interactive subsets of an AnnData, by feature value or by .obs metadata.

Two functions, designed to be chained from a notebook:

  - subset_by_obs_values()     keeps rows whose .obs values match a filter,
                               e.g. {"condition": ["Young"]}.
  - subset_by_feature_values() keeps rows by the value of ONE feature:
                               the 25% lowest, the 8% highest, between two
                               values, below or above a value.

Both return a COPY: the input AnnData is never modified. The copy keeps every
.layers, .obsm and .uns entry of the input. This matters for cluster models
(ADR-015): when the normalization and the PCA were computed on the full
reference dataset BEFORE subsetting, every subset already lives in that same
frozen reference space.

Every call appends one entry to the subset history,
.uns['analysis_config']['subset_history'], recording what was done, the
absolute thresholds actually applied, and the number of mitochondria before
and after — overall and per subject. Read it back in order with
get_subset_history(). compute_cluster_proportions() uses it to find the size
of the population a subset was drawn from.

Typical usage:
    from mito_marker.analysis import subset_by_obs_values, subset_by_feature_values
    young_anndata = subset_by_obs_values(tem_anndata, {"condition": ["Young"]})
    young_small_anndata = subset_by_feature_values(
        young_anndata, "Mito_Area", mode="lowest_fraction", fraction=0.25,
        within_group="unique_subject_ID",
    )
"""

import gc
from typing import Any, Dict, List, Optional, Tuple

import anndata
import numpy as np
import pandas as pd

from mito_marker.analysis.selection import _ANALYSIS_CONFIG_KEY, _filter_obs_by_values
from mito_marker.controlled_vocabulary import ALLOWED_FEATURE_SUBSET_MODES

# Key of the subset history inside .uns['analysis_config'].
SUBSET_HISTORY_KEY = "subset_history"

# Subject columns, in order of preference, used for per-subject counts.
# unique_subject_ID comes first because plain subject_ID values repeat across
# species in the multi-species TEM dataset ("01" exists in several species).
_SUBJECT_COLUMN_CANDIDATES: List[str] = ["unique_subject_ID", "subject_ID"]

_FRACTION_MODES = ("lowest_fraction", "highest_fraction")


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------


def subset_by_feature_values(
    anndata_object: anndata.AnnData,
    feature_name: str,
    mode: str,
    fraction: Optional[float] = None,
    lower_value: Optional[float] = None,
    upper_value: Optional[float] = None,
    within_group: Optional[str] = None,
    inclusive: bool = True,
) -> anndata.AnnData:
    """
    Keep the mitochondria selected by the value of one feature.

    Modes (see ALLOWED_FEATURE_SUBSET_MODES):
      - "lowest_fraction":  the `fraction` of rows with the lowest values
                            (fraction=0.25 -> the 25% smallest).
      - "highest_fraction": the `fraction` of rows with the highest values.
      - "between":          lower_value <= value <= upper_value.
      - "below":            value <= upper_value.
      - "above":            value >= lower_value.
    With inclusive=False the bounds are strict (< and > instead of <= and >=).

    The feature is read from the RAW values of .X when feature_name is a
    variable, otherwise from a numeric .obs column. Raw values are used on
    purpose: a quantile does not change under a monotone transformation
    (arcsinh, z-score), so "the 25% smallest" is the same set of mitochondria
    whatever layer is active, and a value threshold stays in physical units.

    Fraction modes keep an EXACT count: round(fraction × n) rows, taken in
    sorted order (ties broken by row order), rather than "every value below
    the quantile", which could keep more rows than asked when values repeat.

    within_group (fraction modes only): with within_group=None the fraction
    is taken over all mitochondria pooled, so a subject with small or many
    mitochondria contributes more of the subset. With
    within_group="unique_subject_ID", the fraction is taken inside each
    subject, so every subject contributes the same share of its own
    mitochondria (ADR-004 / ADR-011).

    Rows whose feature value is NaN are never kept, and their count is printed.

    Arguments:
        anndata_object: Input AnnData (not modified).
        feature_name: Variable name (.var_names) or numeric .obs column.
        mode: One of ALLOWED_FEATURE_SUBSET_MODES.
        fraction: Share of rows to keep, strictly between 0 and 1
                  (fraction modes only).
        lower_value: Lower bound ("between" and "above" only).
        upper_value: Upper bound ("between" and "below" only).
        within_group: .obs column inside which the fraction is computed
                      (fraction modes only), e.g. "unique_subject_ID".
        inclusive: Whether bounds are included (value modes only).

    Returns:
        A new AnnData holding the kept rows, with one entry appended to
        .uns['analysis_config']['subset_history'].

    Raises:
        KeyError: Unknown feature or within_group column.
        ValueError: Unknown mode, or parameters inconsistent with the mode.
    """
    _validate_feature_subset_arguments(
        anndata_object, mode, fraction, lower_value, upper_value, within_group
    )
    feature_values, feature_source = _get_feature_values(anndata_object, feature_name)

    print("=" * 60)
    print("SUBSET BY FEATURE VALUE")
    print("=" * 60)
    print(
        f"Parameters: feature='{feature_name}' (from {feature_source}), mode='{mode}', "
        f"fraction={fraction}, lower_value={lower_value}, upper_value={upper_value}, "
        f"within_group={within_group}, inclusive={inclusive}"
    )
    print(f"Mode meaning: {ALLOWED_FEATURE_SUBSET_MODES[mode]}")

    is_missing = np.isnan(feature_values)
    if is_missing.any():
        print(
            f"WARNING: {int(is_missing.sum()):,} rows have a NaN '{feature_name}' "
            "and can never be kept."
        )

    per_group_thresholds: Dict[str, float] = {}
    if mode in _FRACTION_MODES:
        keep_mask, per_group_thresholds = _select_fraction(
            anndata_object.obs, feature_values, mode, float(fraction), within_group
        )
    else:
        keep_mask = _select_by_value(feature_values, mode, lower_value, upper_value, inclusive)

    kept_values = feature_values[keep_mask]
    lower_value_applied = float(np.min(kept_values)) if kept_values.size else float("nan")
    upper_value_applied = float(np.max(kept_values)) if kept_values.size else float("nan")

    history_entry: Dict[str, Any] = {
        "subset_type": "feature_values",
        "feature_name": feature_name,
        "feature_source": feature_source,
        "mode": mode,
        "fraction": float(fraction) if fraction is not None else float("nan"),
        "lower_value_requested": float(lower_value) if lower_value is not None else float("nan"),
        "upper_value_requested": float(upper_value) if upper_value is not None else float("nan"),
        "inclusive": bool(inclusive),
        "within_group": within_group if within_group is not None else "",
        # The actual range of kept values: reuse upper_value_applied as an
        # absolute threshold on another dataset (e.g. Old below the Young cut).
        "lower_value_applied": lower_value_applied,
        "upper_value_applied": upper_value_applied,
        "per_group_thresholds": per_group_thresholds,
    }

    subset_anndata = _build_subset(anndata_object, keep_mask, history_entry)
    _print_feature_subset_qc(
        anndata_object, subset_anndata, feature_values, kept_values,
        feature_name, per_group_thresholds,
    )
    return subset_anndata


def subset_by_obs_values(
    anndata_object: anndata.AnnData,
    obs_filters: Dict[str, List[Any]],
) -> anndata.AnnData:
    """
    Keep the rows whose .obs values match every filter.

    Values listed for one column are combined with OR; different columns are
    combined with AND. Example: {"condition": ["Young"], "specie": ["Mouse",
    "Human"]} keeps young mice and young humans.

    Arguments:
        anndata_object: Input AnnData (not modified).
        obs_filters: Dict mapping an .obs column name to the list of values to
                     keep. A single value may be passed without a list.

    Returns:
        A new AnnData holding the kept rows, with one entry appended to
        .uns['analysis_config']['subset_history'].

    Raises:
        KeyError: A filter column is not in .obs.
        ValueError: obs_filters is empty.
    """
    if not obs_filters:
        raise ValueError("obs_filters is empty — give at least one column and its values.")

    print("=" * 60)
    print("SUBSET BY .obs VALUES")
    print("=" * 60)

    # A positional column is carried along so the final mask does not depend
    # on the .obs index being unique.
    filtered_obs = anndata_object.obs.assign(_row_position=np.arange(anndata_object.n_obs))
    normalized_filters: Dict[str, List[str]] = {}
    for column_name, wanted_values in obs_filters.items():
        if column_name not in anndata_object.obs.columns:
            raise KeyError(
                f"Column '{column_name}' not found in .obs. "
                f"Available columns: {list(anndata_object.obs.columns)}"
            )
        if not isinstance(wanted_values, (list, tuple, set, np.ndarray)):
            wanted_values = [wanted_values]
        wanted_values = list(wanted_values)
        present_values = set(anndata_object.obs[column_name].dropna().astype(str))
        absent_values = [value for value in wanted_values if str(value) not in present_values]
        if absent_values:
            print(f"WARNING: values {absent_values} do not occur in .obs['{column_name}'].")
        filtered_obs = _filter_obs_by_values(filtered_obs, column_name, wanted_values)
        normalized_filters[column_name] = [str(value) for value in wanted_values]
        print(f"  {column_name} in {wanted_values}: {len(filtered_obs):,} rows left")

    keep_mask = np.zeros(anndata_object.n_obs, dtype=bool)
    keep_mask[filtered_obs["_row_position"].to_numpy()] = True
    history_entry = {"subset_type": "obs_values", "obs_filters": normalized_filters}
    subset_anndata = _build_subset(anndata_object, keep_mask, history_entry)
    _print_subset_counts_qc(anndata_object, subset_anndata, flag_lost_subjects=False)
    print_anndata_qc(subset_anndata, "subset_by_obs_values")
    return subset_anndata


def get_subset_history(anndata_object: anndata.AnnData) -> List[Dict[str, Any]]:
    """
    Return every subset step applied to this AnnData, oldest first.

    Arguments:
        anndata_object: AnnData produced by the subset functions.

    Returns:
        List of history entries (dicts); empty when no subset was recorded.
    """
    history = anndata_object.uns.get(_ANALYSIS_CONFIG_KEY, {}).get(SUBSET_HISTORY_KEY, {})
    return [history[key] for key in sorted(history)]


def resolve_subject_column(anndata_object: anndata.AnnData) -> Optional[str]:
    """
    Return the .obs column identifying subjects, preferring unique_subject_ID.

    Arguments:
        anndata_object: AnnData to inspect.

    Returns:
        Column name, or None when no subject column exists.
    """
    for column_name in _SUBJECT_COLUMN_CANDIDATES:
        if column_name in anndata_object.obs.columns:
            return column_name
    return None


def print_anndata_qc(anndata_object: anndata.AnnData, step_name: str) -> None:
    """
    Print the standard AnnData QC block (dimensions, heads, keys, .X summary).

    Arguments:
        anndata_object: AnnData to describe.
        step_name: Name of the step, shown in the header.
    """
    print(f"--- QC after {step_name} ---")
    print(f"Observations: {anndata_object.n_obs}, Variables: {anndata_object.n_vars}")
    print(".obs.head():")
    print(anndata_object.obs.head())
    print(".var.head():")
    print(anndata_object.var.head())
    print(f".obsm keys: {list(anndata_object.obsm.keys())}")
    print(f".uns keys: {list(anndata_object.uns.keys())}")
    print(f".layers keys: {list(anndata_object.layers.keys())}")
    if anndata_object.n_obs > 0:
        x_matrix = anndata_object.X
        if hasattr(x_matrix, "toarray"):
            x_matrix = x_matrix.toarray()
        x_matrix = np.asarray(x_matrix, dtype=np.float64)
        finite_values = x_matrix[np.isfinite(x_matrix)]
        print(
            f".X: NaN={int(np.isnan(x_matrix).sum())}, infinite={int(np.isinf(x_matrix).sum())}"
            + (
                f", min={finite_values.min():.4g}, max={finite_values.max():.4g}, "
                f"mean={finite_values.mean():.4g}"
                if finite_values.size else ""
            )
        )
    print("=" * 60)


# ---------------------------------------------------------------------------
# Selection logic
# ---------------------------------------------------------------------------


def _validate_feature_subset_arguments(
    anndata_object: anndata.AnnData,
    mode: str,
    fraction: Optional[float],
    lower_value: Optional[float],
    upper_value: Optional[float],
    within_group: Optional[str],
) -> None:
    """
    Check that the parameters passed to subset_by_feature_values() fit the mode.

    Arguments:
        anndata_object: Input AnnData.
        mode, fraction, lower_value, upper_value, within_group: As in
            subset_by_feature_values().

    Raises:
        ValueError / KeyError with a message naming the offending parameter.
    """
    if mode not in ALLOWED_FEATURE_SUBSET_MODES:
        raise ValueError(
            f"Unknown mode '{mode}'. Allowed: {list(ALLOWED_FEATURE_SUBSET_MODES)}"
        )
    if mode in _FRACTION_MODES:
        if fraction is None or not 0.0 < float(fraction) < 1.0:
            raise ValueError(
                f"mode='{mode}' needs fraction strictly between 0 and 1 "
                f"(e.g. 0.25 for 25%), got {fraction}."
            )
        if lower_value is not None or upper_value is not None:
            raise ValueError(f"mode='{mode}' uses fraction only; do not pass lower_value/upper_value.")
        if within_group is not None and within_group not in anndata_object.obs.columns:
            raise KeyError(f"within_group column '{within_group}' not found in .obs.")
        return

    if fraction is not None:
        raise ValueError(f"mode='{mode}' uses values, not fraction.")
    if within_group is not None:
        raise ValueError(
            f"within_group only applies to fraction modes; mode='{mode}' uses the "
            "same absolute threshold for every row."
        )
    if mode == "between":
        if lower_value is None or upper_value is None:
            raise ValueError("mode='between' needs both lower_value and upper_value.")
        if lower_value > upper_value:
            raise ValueError(
                f"lower_value ({lower_value}) is greater than upper_value ({upper_value})."
            )
    if mode == "below" and (upper_value is None or lower_value is not None):
        raise ValueError("mode='below' needs upper_value only.")
    if mode == "above" and (lower_value is None or upper_value is not None):
        raise ValueError("mode='above' needs lower_value only.")


def _get_feature_values(
    anndata_object: anndata.AnnData,
    feature_name: str,
) -> Tuple[np.ndarray, str]:
    """
    Return the raw values of a feature, from .X first, then from .obs.

    Arguments:
        anndata_object: AnnData to read.
        feature_name: Variable name or numeric .obs column.

    Returns:
        Tuple (float64 array of length n_obs, source label "var" or "obs").

    Raises:
        KeyError: feature_name is neither a variable nor an .obs column.
        TypeError: the .obs column is not numeric.
    """
    if feature_name in anndata_object.var_names:
        column_index = anndata_object.var_names.get_loc(feature_name)
        column = anndata_object.X[:, column_index]
        if hasattr(column, "toarray"):
            column = column.toarray()
        return np.asarray(column, dtype=np.float64).ravel(), "var"

    if feature_name in anndata_object.obs.columns:
        obs_column = anndata_object.obs[feature_name]
        if not pd.api.types.is_numeric_dtype(obs_column):
            raise TypeError(f".obs['{feature_name}'] is not numeric (dtype {obs_column.dtype}).")
        return obs_column.to_numpy(dtype=np.float64), "obs"

    raise KeyError(
        f"'{feature_name}' is neither a variable (.var_names) nor an .obs column."
    )


def _select_fraction(
    obs_dataframe: pd.DataFrame,
    feature_values: np.ndarray,
    mode: str,
    fraction: float,
    within_group: Optional[str],
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Build the keep-mask for the fraction modes, pooled or per group.

    Arguments:
        obs_dataframe: .obs of the input AnnData.
        feature_values: Feature value per row (NaN rows are never kept).
        mode: "lowest_fraction" or "highest_fraction".
        fraction: Share of rows to keep in each group.
        within_group: Grouping .obs column, or None for one pooled group.

    Returns:
        Tuple (boolean keep-mask, {group label: boundary value applied}).
        For lowest_fraction the boundary is the largest kept value; for
        highest_fraction it is the smallest kept value.
    """
    keep_mask = np.zeros(len(feature_values), dtype=bool)
    thresholds: Dict[str, float] = {}

    if within_group is None:
        group_positions = {"all": np.arange(len(feature_values))}
    else:
        group_labels = obs_dataframe[within_group].astype(str).to_numpy()
        group_positions = {
            label: np.flatnonzero(group_labels == label)
            for label in pd.unique(group_labels)
        }

    for group_label, positions in group_positions.items():
        valid_positions = positions[~np.isnan(feature_values[positions])]
        n_keep = int(round(fraction * len(valid_positions)))
        if n_keep == 0:
            print(
                f"WARNING: group '{group_label}' has {len(valid_positions)} valid rows — "
                f"{fraction:.0%} of that rounds to 0, nothing kept for this group."
            )
            continue
        # A stable sort breaks ties by row order, which makes the result
        # reproducible and the kept count exact.
        order = np.argsort(feature_values[valid_positions], kind="stable")
        if mode == "highest_fraction":
            order = order[::-1]
        chosen_positions = valid_positions[order[:n_keep]]
        keep_mask[chosen_positions] = True
        thresholds[str(group_label)] = float(feature_values[chosen_positions[-1]])

    return keep_mask, thresholds


def _select_by_value(
    feature_values: np.ndarray,
    mode: str,
    lower_value: Optional[float],
    upper_value: Optional[float],
    inclusive: bool,
) -> np.ndarray:
    """
    Build the keep-mask for the value modes ("between", "below", "above").

    Arguments:
        feature_values: Feature value per row.
        mode: "between", "below" or "above".
        lower_value: Lower bound (or None).
        upper_value: Upper bound (or None).
        inclusive: Whether the bounds themselves are kept.

    Returns:
        Boolean keep-mask (NaN rows are False because NaN comparisons are False).
    """
    keep_mask = np.ones(len(feature_values), dtype=bool)
    with np.errstate(invalid="ignore"):
        if lower_value is not None:
            keep_mask &= (feature_values >= lower_value) if inclusive else (feature_values > lower_value)
        if upper_value is not None:
            keep_mask &= (feature_values <= upper_value) if inclusive else (feature_values < upper_value)
    return keep_mask


# ---------------------------------------------------------------------------
# Subset construction and QC
# ---------------------------------------------------------------------------


def _build_subset(
    anndata_object: anndata.AnnData,
    keep_mask: np.ndarray,
    history_entry: Dict[str, Any],
) -> anndata.AnnData:
    """
    Copy the kept rows and append the history entry to the copy.

    Arguments:
        anndata_object: Input AnnData (not modified).
        keep_mask: Boolean mask of rows to keep.
        history_entry: Description of the step; counts are added here.

    Returns:
        The new AnnData.
    """
    subject_column = resolve_subject_column(anndata_object)
    history_entry["subject_column"] = subject_column if subject_column is not None else ""
    history_entry["n_obs_before"] = int(anndata_object.n_obs)
    history_entry["n_obs_after"] = int(np.sum(keep_mask))
    history_entry["n_obs_before_by_subject"] = _count_by_subject(anndata_object.obs, subject_column)
    history_entry["n_obs_after_by_subject"] = _count_by_subject(
        anndata_object.obs[keep_mask], subject_column
    )

    subset_anndata = anndata_object[keep_mask].copy()

    analysis_config = dict(subset_anndata.uns.get(_ANALYSIS_CONFIG_KEY, {}))
    history = dict(analysis_config.get(SUBSET_HISTORY_KEY, {}))
    # Zero-padded keys keep the history in order once sorted, and a dict of
    # dicts (unlike a list of dicts) can be written to .h5ad.
    history[f"{len(history):03d}"] = history_entry
    analysis_config[SUBSET_HISTORY_KEY] = history
    subset_anndata.uns[_ANALYSIS_CONFIG_KEY] = analysis_config

    gc.collect()
    return subset_anndata


def _count_by_subject(obs_dataframe: pd.DataFrame, subject_column: Optional[str]) -> Dict[str, int]:
    """
    Count rows per subject.

    Arguments:
        obs_dataframe: .obs rows to count.
        subject_column: Subject column, or None.

    Returns:
        {subject: count} with string keys (empty when no subject column).
    """
    if subject_column is None:
        return {}
    counts = obs_dataframe[subject_column].astype(str).value_counts()
    return {str(subject): int(count) for subject, count in counts.items()}


def _print_subset_counts_qc(
    anndata_object: anndata.AnnData,
    subset_anndata: anndata.AnnData,
    flag_lost_subjects: bool = True,
) -> None:
    """
    Print counts before/after, per subject and per condition, and flag
    subjects that lost every mitochondrion.

    Arguments:
        anndata_object: Input AnnData.
        subset_anndata: Resulting subset.
        flag_lost_subjects: When False (filters on .obs values, which remove
                            whole subjects on purpose), only the subjects kept
                            are listed and no warning is printed.
    """
    n_before, n_after = anndata_object.n_obs, subset_anndata.n_obs
    kept_share = n_after / n_before if n_before else 0.0
    print(f"Rows: {n_before:,} -> {n_after:,} ({kept_share:.1%} kept)")

    for column_name in ("condition", "diet", "specie"):
        if column_name in anndata_object.obs.columns:
            before = anndata_object.obs[column_name].astype(str).value_counts()
            after = subset_anndata.obs[column_name].astype(str).value_counts()
            summary = {value: f"{int(after.get(value, 0))}/{int(count)}" for value, count in before.items()}
            print(f"  kept per {column_name} (after/before): {summary}")

    subject_column = resolve_subject_column(anndata_object)
    if subject_column is None:
        return
    before = anndata_object.obs[subject_column].astype(str).value_counts()
    after = subset_anndata.obs[subject_column].astype(str).value_counts()
    counts_table = pd.DataFrame({"before": before, "after": after}).fillna(0).astype(int)
    counts_table["kept_share"] = (counts_table["after"] / counts_table["before"]).round(3)
    if not flag_lost_subjects:
        counts_table = counts_table[counts_table["after"] > 0]
    print(f"  kept per {subject_column}:")
    print(counts_table.sort_index().to_string())
    lost_subjects = counts_table.index[counts_table["after"] == 0].tolist()
    if lost_subjects and flag_lost_subjects:
        print(f"WARNING: these subjects have NO mitochondria left: {lost_subjects}")


def _print_feature_subset_qc(
    anndata_object: anndata.AnnData,
    subset_anndata: anndata.AnnData,
    feature_values: np.ndarray,
    kept_values: np.ndarray,
    feature_name: str,
    per_group_thresholds: Dict[str, float],
) -> None:
    """
    Print the QC block of subset_by_feature_values().

    Arguments:
        anndata_object: Input AnnData.
        subset_anndata: Resulting subset.
        feature_values: Feature values of every input row.
        kept_values: Feature values of the kept rows.
        feature_name: Name of the feature.
        per_group_thresholds: Boundary value applied in each group (fraction modes).
    """
    def describe(values: np.ndarray) -> str:
        finite_values = values[np.isfinite(values)]
        if finite_values.size == 0:
            return "no finite value"
        return (
            f"min={finite_values.min():.4g}, max={finite_values.max():.4g}, "
            f"mean={finite_values.mean():.4g}, median={np.median(finite_values):.4g}"
        )

    print(f"'{feature_name}' before: {describe(feature_values)}")
    print(f"'{feature_name}' after:  {describe(kept_values)}")
    if kept_values.size:
        print(
            f"Absolute range kept: [{np.min(kept_values):.6g}, {np.max(kept_values):.6g}] "
            "(recorded as lower/upper_value_applied in the subset history)"
        )
    if len(per_group_thresholds) > 1:
        threshold_values = np.array(list(per_group_thresholds.values()))
        print(
            f"Boundary value per group: min={threshold_values.min():.4g}, "
            f"median={np.median(threshold_values):.4g}, max={threshold_values.max():.4g}"
        )
    _print_subset_counts_qc(anndata_object, subset_anndata)
    print_anndata_qc(subset_anndata, "subset_by_feature_values")
