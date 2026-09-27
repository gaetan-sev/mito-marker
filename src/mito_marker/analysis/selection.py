"""
selection.py

Interactive subset selection for SFC AnnData objects.

Provides a console-driven menu that allows non-programmer users to filter an
AnnData object by any combination of .obs metadata columns defined in
FILTERABLE_OBS_COLUMNS (controlled_vocabulary.py). Each filter step presents
only the values present in the current subset — preventing impossible
combinations and zero-event results.

Typical usage in a Jupyter / Colab notebook cell:
    from mito_marker.analysis import select_sfc_subset
    sfc_subset = select_sfc_subset(sfc_anndata)
"""

from typing import Any, List, Optional, Tuple

import anndata
import numpy as np
import pandas as pd

from mito_marker.controlled_vocabulary import FILTERABLE_OBS_COLUMNS

# Key under which all analysis pipeline choices are persisted in .uns.
_ANALYSIS_CONFIG_KEY = "analysis_config"

# Preferred subject grouping column. Falls back to _SUBJECT_ID_FALLBACK when
# unique_subject_ID is not present (e.g. SFC AnnData objects).
_SUBJECT_ID_COLUMN = "unique_subject_ID"
_SUBJECT_ID_FALLBACK = "subject_ID"


def _get_subject_column(obs_dataframe: pd.DataFrame) -> Optional[str]:
    """
    Return the subject column name to use for grouping and display.

    Prefers 'unique_subject_ID' (the cross-species key created by the TEM
    pipeline as '{specie}_{subject_ID}'). Falls back to 'subject_ID' for
    SFC AnnData objects that do not carry the combined key.

    Arguments:
        obs_dataframe: The .obs DataFrame of the current subset.

    Returns:
        Column name string, or None if neither column is present.
    """
    if _SUBJECT_ID_COLUMN in obs_dataframe.columns:
        return _SUBJECT_ID_COLUMN
    if _SUBJECT_ID_FALLBACK in obs_dataframe.columns:
        return _SUBJECT_ID_FALLBACK
    return None


def select_sfc_subset(anndata_object: anndata.AnnData) -> anndata.AnnData:
    """
    Interactively filter an SFC AnnData to a user-defined subset of events.

    For each column listed in FILTERABLE_OBS_COLUMNS the function:
      - Skips columns absent from .obs.
      - Auto-skips columns with only one unique value in the current subset
        (no user choice is needed).
      - Displays a numbered menu of available values with event counts.
      - Reads user input from the console: comma-separated numbers or "A" for all.
      - Progressively narrows the subset — each column menu shows only values
        present among events that passed all previous filters.

    The raw .X matrix and .var DataFrame are never modified.
    Selection choices are recorded in .uns['analysis_config']['selection'].

    Arguments:
        anndata_object: Input SFC AnnData produced by ingest_sfc_folder() or
                        loaded from a saved .h5ad file.

    Returns:
        A new AnnData containing only the events matching all selected filters.
        .uns['analysis_config'] is initialised with 'selection', 'active_layer',
        and 'active_selection' keys for use by downstream analysis functions.
    """
    print("=" * 60)
    print("INTERACTIVE SUBSET SELECTION")
    print("=" * 60)
    print(
        f"Full dataset: {anndata_object.n_obs:,} events "
        f"× {anndata_object.n_vars} channels"
    )
    print()

    # Work with a copy of .obs so we can progressively filter its rows without
    # touching the original AnnData.  A synthetic integer column tracks the
    # original row position so the final slice uses positional indexing —
    # index-based isin() is unreliable when obs_names are not unique.
    current_obs_dataframe = anndata_object.obs.copy()
    current_obs_dataframe["_row_position"] = np.arange(anndata_object.n_obs)
    selection_log: dict[str, Any] = {}

    # Determine the subject column once — it stays the same throughout the loop
    # even as rows are narrowed down.
    subject_col = _get_subject_column(current_obs_dataframe)

    for column_name in FILTERABLE_OBS_COLUMNS:
        if column_name not in current_obs_dataframe.columns:
            continue

        column_series = current_obs_dataframe[column_name]
        display_values = _get_display_values(column_series)

        # Auto-skip when only one distinct non-empty value remains — the user
        # has no meaningful choice to make for this column.
        if len(display_values) <= 1:
            if display_values:
                selection_log[column_name] = "auto (only one value present)"
            continue

        selected_raw_values = _prompt_column_selection(
            column_name, display_values, current_obs_dataframe, subject_col
        )

        if selected_raw_values is None:
            # User chose "A" — keep all values for this column.
            selection_log[column_name] = "all"
            continue

        current_obs_dataframe = _filter_obs_by_values(
            current_obs_dataframe, column_name, selected_raw_values
        )
        # Store the display strings (not raw values) for human-readable config.
        selected_display = [
            entry[1]
            for entry in display_values
            if entry[0] in selected_raw_values
            or (
                isinstance(entry[0], float)
                and not np.isnan(entry[0])
                and entry[0] in selected_raw_values
            )
        ]
        selection_log[column_name] = selected_display

        remaining = len(current_obs_dataframe)
        print(
            f"=> {remaining:,} events remaining after '{column_name}' filter."
        )
        print()

        if remaining == 0:
            print("WARNING: No events remaining. Selection produced an empty subset.")
            break

    # Sub-sampling step: offer to draw a fixed number of events per subject.
    subsampling_n = _prompt_subsampling_size(current_obs_dataframe)
    subsampling_log: dict[str, Any] = {
        "n_events_per_subject": subsampling_n,
        "applied": subsampling_n > 0,
    }

    if subsampling_n > 0:
        current_obs_dataframe = _subsample_obs_per_subject(
            current_obs_dataframe, subsampling_n
        )
        print(
            f"=> {len(current_obs_dataframe):,} events after sub-sampling "
            f"({subsampling_n} per subject)."
        )
        print()
    else:
        print("=> Sub-sampling skipped — all filtered events retained.")
        print()

    # Apply the final row selection using the stored integer positions.
    # Positional indexing is unambiguous even when obs_names are not unique.
    # With-replacement sampling produces duplicate positions — AnnData handles
    # this by creating duplicate rows, which is the intended behaviour.
    kept_positions = current_obs_dataframe["_row_position"].values
    filtered_anndata = anndata_object[kept_positions].copy()
    # Remove the helper column — it must not persist in the output .obs.
    if "_row_position" in filtered_anndata.obs.columns:
        del filtered_anndata.obs["_row_position"]

    # Initialise the shared analysis config dict in .uns.
    if _ANALYSIS_CONFIG_KEY not in filtered_anndata.uns:
        filtered_anndata.uns[_ANALYSIS_CONFIG_KEY] = {}

    analysis_config: dict[str, Any] = filtered_anndata.uns[_ANALYSIS_CONFIG_KEY]
    analysis_config["selection"] = selection_log
    analysis_config["subsampling"] = subsampling_log
    # Placeholders consumed by downstream steps (normalization, feature selection).
    analysis_config.setdefault("active_layer", None)
    analysis_config.setdefault("active_selection", None)

    subject_col = _get_subject_column(filtered_anndata.obs)
    n_subjects = (
        filtered_anndata.obs[subject_col].nunique()
        if subject_col is not None
        else "?"
    )

    print("=" * 60)
    print("SELECTION COMPLETE")
    print(
        f"=> {filtered_anndata.n_obs:,} events "
        f"| {n_subjects} subject(s) "
        f"| {filtered_anndata.n_vars} channels"
    )
    print("=" * 60)

    return filtered_anndata


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _get_display_values(
    column_series: pd.Series,
) -> List[Tuple[Any, str, int]]:
    """
    Extract (raw_value, display_string, event_count) tuples for a .obs column.

    Handles four column types:
      - Boolean (e.g. FlowAI_Pass): True listed before False.
      - Float (e.g. age): ascending numeric order; NaN grouped as "Unknown".
      - Categorical or object string (e.g. dilution, unique_subject_ID): alphabetical.
    Empty strings ("") are treated as missing, same as NaN, and displayed as
    "Unknown (not recorded)".

    Arguments:
        column_series: A pandas Series from .obs (one column of the current
                       subset being filtered).

    Returns:
        List of (raw_value, display_string, count) tuples. An empty list means
        no meaningful values were found (column is all-missing).
    """
    # --- Boolean columns ---
    if pd.api.types.is_bool_dtype(column_series):
        results = []
        for raw_val in [True, False]:
            count = int((column_series == raw_val).sum())
            if count > 0:
                results.append((raw_val, str(raw_val), count))
        return results

    # --- Float columns (e.g. age stored as float32 with NaN for missing) ---
    if pd.api.types.is_float_dtype(column_series):
        results = []
        non_nan_values = sorted(column_series.dropna().unique())
        for raw_val in non_nan_values:
            count = int((column_series == raw_val).sum())
            # Show as integer when the value is a whole number (3.0 → "3").
            display = (
                str(int(raw_val))
                if raw_val == int(raw_val)
                else f"{raw_val:.3g}"
            )
            results.append((raw_val, display, count))

        nan_count = int(column_series.isna().sum())
        if nan_count > 0:
            results.append((np.nan, "Unknown (not recorded)", nan_count))
        return results

    # --- Categorical and object-string columns ---
    # value_counts(dropna=False) also counts NaN/None/pd.NA entries.
    value_counts = column_series.value_counts(dropna=False)
    results = []
    unknown_count = 0

    for raw_val, count in value_counts.items():
        # Skip categories that have no events in the current subset.
        # Categorical dtype retains all declared categories even after filtering,
        # so value_counts() can return zeros for values not present.
        if count == 0:
            continue

        is_missing = pd.isna(raw_val) or (
            isinstance(raw_val, str) and raw_val == ""
        )
        if is_missing:
            unknown_count += int(count)
        else:
            results.append((raw_val, str(raw_val), int(count)))

    # Alphabetical sort for readability.
    results.sort(key=lambda entry: entry[1])

    if unknown_count > 0:
        # Raw value "" is used as the sentinel so _filter_obs_by_values can
        # correctly match both "" strings and NaN values.
        results.append(("", "Unknown (not recorded)", unknown_count))

    return results


def _prompt_column_selection(
    column_name: str,
    display_values: List[Tuple[Any, str, int]],
    obs_dataframe: Optional[pd.DataFrame] = None,
    subject_col: Optional[str] = None,
) -> Optional[List[Any]]:
    """
    Display a numbered menu for one .obs column and return the user's selection.

    Loops until the user enters a valid response:
      - "A" or "a": keep all values (returns None).
      - Comma-separated integers (e.g. "1", "1,3"): select those entries.

    When obs_dataframe and subject_col are provided, each menu row also shows
    the number of unique subjects that have that column value, e.g.:
      [1] Human   - 6 subjects -   (6,956 events)

    Arguments:
        column_name: Name of the .obs column being filtered.
        display_values: List of (raw_value, display_string, count) tuples.
        obs_dataframe: The current filtered .obs DataFrame. When provided together
                       with subject_col, per-value subject counts are displayed.
        subject_col: Name of the subject column used to count unique subjects.

    Returns:
        None when the user chose all values.
        A list of raw values corresponding to the numbered selections otherwise.
    """
    total_events = sum(count for _, _, count in display_values)
    print(f"--- Filtering column: {column_name} ---")

    # Precompute per-value subject counts when subject information is available.
    show_subject_counts = obs_dataframe is not None and subject_col is not None
    per_value_subject_counts: List[int] = []
    if show_subject_counts:
        for raw_val, _, _ in display_values:
            is_missing_sentinel = raw_val == "" or (
                isinstance(raw_val, float) and np.isnan(raw_val)
            )
            if is_missing_sentinel:
                mask = obs_dataframe[column_name].isna() | (
                    obs_dataframe[column_name] == ""
                )
            else:
                mask = obs_dataframe[column_name] == raw_val
            per_value_subject_counts.append(
                int(obs_dataframe.loc[mask, subject_col].nunique())
            )

    for idx, (_, display_string, count) in enumerate(display_values, start=1):
        if show_subject_counts:
            n_subjects = per_value_subject_counts[idx - 1]
            subject_label = f"- {n_subjects} subject{'s' if n_subjects != 1 else ''} -"
            print(f"  [{idx}] {display_string:<30} {subject_label:<20} ({count:,} events)")
        else:
            print(f"  [{idx}] {display_string:<40} ({count:,} events)")

    if show_subject_counts:
        total_subjects = int(obs_dataframe[subject_col].nunique())
        subject_label = f"- {total_subjects} subject{'s' if total_subjects != 1 else ''} -"
        print(f"  [A] All                        {subject_label:<20} ({total_events:,} events)")
    else:
        print(f"  [A] All                                    ({total_events:,} events)")

    while True:
        raw_response = input(
            f"Select {column_name} values (e.g. 1,3 or A): "
        ).strip()

        if raw_response.upper() == "A":
            return None

        try:
            chosen_numbers = [
                int(token.strip()) for token in raw_response.split(",")
            ]
        except ValueError:
            print(
                f"  Invalid input: '{raw_response}'. "
                f"Enter numbers separated by commas (e.g. 1,3) or A for all."
            )
            continue

        valid_range = range(1, len(display_values) + 1)
        invalid_numbers = [n for n in chosen_numbers if n not in valid_range]
        if invalid_numbers:
            print(
                f"  Invalid numbers: {invalid_numbers}. "
                f"Choose between 1 and {len(display_values)}."
            )
            continue

        selected_raw_values = [display_values[n - 1][0] for n in chosen_numbers]
        return selected_raw_values


def _prompt_subsampling_size(
    obs_dataframe: pd.DataFrame,
    default_n: int = 0,
) -> int:
    """
    Ask the user how many events per subject to sample, then return the chosen value.

    Displays current per-subject event counts so the user can make an informed
    decision. Loops until the user enters a valid non-negative integer or presses
    Enter to accept the default (skip sub-sampling).

    Arguments:
        obs_dataframe: The .obs DataFrame after all column filters have been applied.
                       Used only to display event counts.
        default_n: Events per subject returned when the user presses Enter without
                   typing a value. Defaults to 0 (skip sub-sampling).

    Returns:
        Number of events per subject to sample. 0 means skip sub-sampling.
    """
    print("--- Per-subject event sub-sampling ---")
    subject_col = _get_subject_column(obs_dataframe)
    if subject_col is not None:
        # Sort subjects from most to least events, then display with 1-based index
        # numbers — same visual format as the other filter menus.
        subject_counts = obs_dataframe[subject_col].value_counts()
        subject_counts = subject_counts[subject_counts > 0]
        for idx, (subject_id, count) in enumerate(subject_counts.items(), start=1):
            print(f"  [{idx}] {str(subject_id):<40} ({count:,} events)")
    else:
        print(f"  (no subject column — total: {len(obs_dataframe):,} events)")

    print("  Default (press Enter): skip sub-sampling — keep all events.")
    print()

    while True:
        raw_response = input(
            "Events per subject (enter a number to subsample, or press Enter to skip): "
        ).strip()

        if raw_response == "":
            return default_n

        try:
            n_events = int(raw_response)
        except ValueError:
            print(
                f"  Invalid input: '{raw_response}'. "
                f"Enter a whole number or press Enter to skip."
            )
            continue

        if n_events < 0:
            print("  Value must be 0 or greater. Enter 0 or press Enter to skip.")
            continue

        return n_events


def _subsample_obs_per_subject(
    obs_dataframe: pd.DataFrame,
    n_events_per_subject: int,
    random_seed: int = 0,
) -> pd.DataFrame:
    """
    Return a sub-sampled version of obs_dataframe with at most n_events_per_subject
    rows per subject.

    Groups by 'unique_subject_ID' when available (cross-species unique key),
    falling back to 'subject_ID' for AnnData objects that do not carry the
    combined key (e.g. SFC data).

    Sampling is performed without replacement when the subject has enough unique
    events. When a subject has fewer events than requested, sampling is performed
    with replacement (allowing the same event to appear multiple times) and a
    WARNING is printed to the console.

    Arguments:
        obs_dataframe: The .obs DataFrame after filtering (includes '_row_position').
        n_events_per_subject: Target number of events to sample per subject.
        random_seed: Integer seed passed to numpy.random.default_rng for
                     reproducibility across runs.

    Returns:
        Sub-sampled DataFrame. Rows may be duplicated for subjects that had
        insufficient events (with-replacement case).
    """
    rng = np.random.default_rng(random_seed)

    subject_col = _get_subject_column(obs_dataframe)
    if subject_col is None:
        # No subject column — treat the entire dataset as one group.
        n_available = len(obs_dataframe)
        if n_available >= n_events_per_subject:
            chosen_indices = rng.choice(n_available, size=n_events_per_subject, replace=False)
        else:
            print(
                f"WARNING: Dataset has only {n_available} events but "
                f"{n_events_per_subject} were requested. "
                f"Sampling with replacement — some events will appear more than once."
            )
            chosen_indices = rng.choice(n_available, size=n_events_per_subject, replace=True)
        return obs_dataframe.iloc[chosen_indices]

    sampled_parts: List[pd.DataFrame] = []
    for subject_id, subject_group in obs_dataframe.groupby(subject_col, sort=False, observed=True):
        n_available = len(subject_group)
        if n_available >= n_events_per_subject:
            chosen_indices = rng.choice(n_available, size=n_events_per_subject, replace=False)
        else:
            print(
                f"WARNING: Subject '{subject_id}' has only {n_available} events "
                f"but {n_events_per_subject} were requested. "
                f"Sampling with replacement — some events will appear more than once."
            )
            chosen_indices = rng.choice(n_available, size=n_events_per_subject, replace=True)
        sampled_parts.append(subject_group.iloc[chosen_indices])

    return pd.concat(sampled_parts)


def _filter_obs_by_values(
    obs_dataframe: pd.DataFrame,
    column_name: str,
    selected_raw_values: List[Any],
) -> pd.DataFrame:
    """
    Return rows of obs_dataframe where column_name matches any selected value.

    Handles NaN and empty strings as "missing" — both are matched when the
    sentinel value "" is in selected_raw_values.

    Arguments:
        obs_dataframe: The current .obs DataFrame being progressively filtered.
        column_name: The column to filter on.
        selected_raw_values: Raw values to retain (may include np.nan or "").

    Returns:
        Filtered DataFrame with only the matching rows.
    """
    column = obs_dataframe[column_name]
    inclusion_mask = pd.Series(False, index=obs_dataframe.index)

    for raw_val in selected_raw_values:
        # Detect the "missing" sentinel: np.nan, "" string, or actual NaN float.
        is_missing_sentinel = raw_val == "" or (
            isinstance(raw_val, float) and np.isnan(raw_val)
        )
        if is_missing_sentinel:
            # Match both NaN values and the "" empty-string sentinel used by
            # anndata_builder.py for absent FCS metadata fields.
            inclusion_mask |= column.isna() | (column == "")
        else:
            inclusion_mask |= column == raw_val

    return obs_dataframe[inclusion_mask]
