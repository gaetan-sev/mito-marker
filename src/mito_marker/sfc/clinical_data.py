"""
clinical_data.py

Merges per-subject clinical variables from a CSV file into an existing AnnData
object created by the SFC ingestion pipeline.

The CSV is expected to have one row per subject. The first column is the subject
identifier (e.g. "MNMS 005"). Remaining columns may be numeric (float) or
categorical (string). Every event (row) in the AnnData receives the clinical values
for its subject_ID. Events from subjects not found in the CSV receive NaN (numeric
columns) or empty string (string columns).
"""

import re
import unicodedata
from typing import Optional

import anndata
import numpy as np
import pandas as pd

from mito_marker.controlled_vocabulary import (
    MNMS_CLINICAL_CSV_MISSING_VALUE_CODE,
    MNMS_CLINICAL_CSV_SUBJECT_ID_PATTERN,
)


def _sanitize_column_name(raw_name: str) -> str:
    """Convert a raw CSV column name to a valid, Python-friendly .obs key.

    Steps applied in order:
    1. Normalize unicode to ASCII-compatible form (é → e, ç → c, etc.).
    2. Replace newlines, tabs, and common special characters with underscores.
    3. Strip leading/trailing underscores.
    4. Collapse consecutive underscores into one.
    5. Convert to lowercase.

    Args:
        raw_name: Original column name from the CSV header.

    Returns:
        A sanitized string suitable for use as a pandas column key.
    """
    # Decompose accented characters and drop the diacritic combining marks
    normalized = unicodedata.normalize("NFD", raw_name)
    ascii_safe = "".join(
        character for character in normalized
        if unicodedata.category(character) != "Mn"
    )

    # Replace any character that is not alphanumeric or underscore with "_"
    # This covers spaces, newlines, parentheses, slashes, %, +, -, ., *, etc.
    result = re.sub(r"[^a-zA-Z0-9]+", "_", ascii_safe)

    # Strip edge underscores and collapse internal runs of underscores
    result = result.strip("_")
    result = re.sub(r"_+", "_", result)

    return result.lower()


def _parse_subject_id_from_csv_index(raw_id: str) -> Optional[str]:
    """Extract a zero-padded 3-digit subject_ID from a CSV index value.

    Examples:
        "MNMS 005" → "005"
        "MNMS 67"  → "067"
        "MNMS005"  → "005"
        "hello"    → None  (no digits found)

    Args:
        raw_id: The raw index value from the clinical CSV (first column).

    Returns:
        A zero-padded 3-digit string (e.g. "005"), or None if no numeric
        ID can be extracted.
    """
    match = re.search(MNMS_CLINICAL_CSV_SUBJECT_ID_PATTERN, str(raw_id).strip())
    if match is None:
        return None
    return f"{int(match.group(1)):03d}"


def enrich_with_clinical_data(
    anndata_object: anndata.AnnData,
    clinical_csv_path: str,
    missing_value_code: float = MNMS_CLINICAL_CSV_MISSING_VALUE_CODE,
) -> anndata.AnnData:
    """Merge per-subject clinical variables from a CSV into AnnData .obs.

    Each event (row in .obs) receives the clinical values for its subject_ID.
    Events whose subject_ID is not found in the CSV receive NaN for every
    clinical column. The function modifies the AnnData in place and also
    returns it for convenient chaining.

    CSV format expected:
    - First column: subject identifier, e.g. "MNMS 005" (used as index).
    - Remaining columns: numeric (float) or categorical (string) clinical variables.
    - Numeric missing values encoded as `missing_value_code` (default -1.0) → NaN.
    - String columns are stored as object dtype; unmatched subjects get empty string "".

    Column names are sanitized to ASCII-safe, underscore-separated lowercase
    strings before being stored in .obs. The mapping from sanitized name back
    to the original CSV column header is stored in .uns["clinical_column_name_mapping"].

    Args:
        anndata_object: AnnData produced by ingest_sfc_folder(), must contain
            .obs["subject_ID"] with zero-padded 3-digit strings (e.g. "005").
        clinical_csv_path: Path to the clinical CSV file.
        missing_value_code: Float value used in the CSV to represent missing
            data. All occurrences are replaced with NaN before merging.

    Returns:
        The same AnnData object with clinical columns added to .obs and
        the original↔sanitized column name mapping stored in .uns.
    """
    print("=" * 60)
    print("Clinical data enrichment — START")
    print(f"  CSV path       : {clinical_csv_path}")
    print(f"  Missing code   : {missing_value_code} → NaN")
    print(f"  AnnData events : {anndata_object.n_obs}")
    print("=" * 60)

    # --- 1. Load the clinical CSV ---
    print("\nStep 1: Loading clinical CSV...")
    clinical_dataframe = pd.read_csv(clinical_csv_path, index_col=0)
    print(f"  Loaded {len(clinical_dataframe)} rows × {len(clinical_dataframe.columns)} columns")

    # --- 2. Replace missing value code with NaN ---
    print(f"\nStep 2: Replacing {missing_value_code} → NaN...")
    clinical_dataframe = clinical_dataframe.replace(missing_value_code, np.nan)

    # --- 3. Build subject_ID → DataFrame row mapping ---
    print("\nStep 3: Parsing subject IDs from CSV index...")
    subject_id_to_clinical_row: dict[str, pd.Series] = {}
    skipped_rows: list[str] = []

    for raw_index_value, row in clinical_dataframe.iterrows():
        parsed_id = _parse_subject_id_from_csv_index(str(raw_index_value))
        if parsed_id is None:
            skipped_rows.append(str(raw_index_value))
        else:
            subject_id_to_clinical_row[parsed_id] = row

    print(f"  Parsed successfully : {len(subject_id_to_clinical_row)} subjects")
    if skipped_rows:
        print(f"  Skipped (no ID)    : {len(skipped_rows)} rows → {skipped_rows}")

    # --- 4. Sanitize column names and handle duplicates ---
    print("\nStep 4: Sanitizing column names...")
    raw_to_sanitized: dict[str, str] = {}
    seen_sanitized_names: dict[str, int] = {}  # sanitized_name → count of occurrences

    for raw_column_name in clinical_dataframe.columns:
        base_sanitized = _sanitize_column_name(str(raw_column_name))
        if base_sanitized not in seen_sanitized_names:
            seen_sanitized_names[base_sanitized] = 1
            final_sanitized = base_sanitized
        else:
            seen_sanitized_names[base_sanitized] += 1
            final_sanitized = f"{base_sanitized}_{seen_sanitized_names[base_sanitized]}"
        raw_to_sanitized[raw_column_name] = final_sanitized

    # Store the reverse mapping (sanitized → original) in .uns for user reference
    sanitized_to_original: dict[str, str] = {v: k for k, v in raw_to_sanitized.items()}
    anndata_object.uns["clinical_column_name_mapping"] = sanitized_to_original
    print(f"  {len(raw_to_sanitized)} columns sanitized")

    # --- 5. Determine which AnnData subjects match the clinical CSV ---
    anndata_subject_ids: pd.Series = anndata_object.obs["subject_ID"]
    unique_anndata_subjects: set[str] = set(anndata_subject_ids.unique())
    unique_clinical_subjects: set[str] = set(subject_id_to_clinical_row.keys())

    matched_subjects = unique_anndata_subjects & unique_clinical_subjects
    unmatched_subjects = unique_anndata_subjects - unique_clinical_subjects

    print("\nStep 5: Subject ID matching")
    print(f"  Subjects in AnnData            : {len(unique_anndata_subjects)}")
    print(f"  Subjects in CSV                : {len(unique_clinical_subjects)}")
    print(f"  Matched (will get data)        : {len(matched_subjects)}")
    if unmatched_subjects:
        print(f"  Unmatched (will get NaN)       : {len(unmatched_subjects)}")
        for unmatched_subject_id in sorted(unmatched_subjects):
            print(f"    - {unmatched_subject_id}")
    events_with_clinical_data = anndata_subject_ids.isin(matched_subjects).sum()
    print(f"  Events with clinical data      : {events_with_clinical_data} / {anndata_object.n_obs}")

    # --- 6. Broadcast clinical values to .obs — build all columns first, join once ---
    # Inserting columns one by one into an existing DataFrame causes pandas to
    # allocate a new internal memory block per column, which fragments the backing
    # numpy arrays and triggers PerformanceWarning at ~10+ columns. The fix is to
    # accumulate all new columns in a plain dict, build a single DataFrame from them,
    # and join with the existing .obs in one pd.concat call.
    print(f"\nStep 6: Adding {len(raw_to_sanitized)} clinical columns to .obs...")
    numeric_column_count = 0
    string_column_count = 0
    new_columns_dict: dict[str, np.ndarray | pd.Categorical] = {}

    for raw_column_name, sanitized_column_name in raw_to_sanitized.items():
        # Build a Series mapping each AnnData event to its clinical value
        # by looking up the event's subject_ID in the clinical data dict.
        # Unmatched subjects receive np.nan as a sentinel for "no data".
        raw_values_per_event = anndata_subject_ids.map(
            lambda subject_id, col=raw_column_name: (
                subject_id_to_clinical_row[subject_id][col]
                if subject_id in subject_id_to_clinical_row
                else np.nan
            )
        )

        # Try numeric first. If the column contains non-numeric strings (e.g.
        # "inactive", "active"), the cast fails and we store as Categorical instead.
        try:
            new_columns_dict[sanitized_column_name] = raw_values_per_event.astype(np.float32).values
            numeric_column_count += 1
        except (ValueError, TypeError):
            # String column: replace the np.nan sentinel (used for unmatched subjects)
            # with empty string "" so the category set is well-defined and h5py-safe.
            # Then cast to pd.Categorical so the unique strings are stored once in a
            # lookup table and each event stores only a small integer code — far cheaper
            # than repeating full string pointers across millions of rows (object dtype).
            string_values = raw_values_per_event.apply(
                lambda value: "" if (isinstance(value, float) and np.isnan(value)) else str(value)
            )
            # Collect all unique string values from the CSV for this column.
            # Using the full CSV rather than only subjects present in AnnData keeps
            # the category set stable when working on subsets or concatenating datasets.
            # "" is always included as the category for unmatched subjects.
            all_categories = sorted(
                {
                    str(subject_id_to_clinical_row[sid][raw_column_name])
                    for sid in subject_id_to_clinical_row
                    if not (
                        isinstance(subject_id_to_clinical_row[sid][raw_column_name], float)
                        and np.isnan(subject_id_to_clinical_row[sid][raw_column_name])
                    )
                }
                | {""}
            )
            new_columns_dict[sanitized_column_name] = pd.Categorical(
                string_values.values, categories=all_categories
            )
            string_column_count += 1

    # Drop any .obs columns that would collide with incoming clinical columns.
    # Keeping both would produce duplicate column names, which crashes write_h5ad
    # (h5py requires unique dataset names within a group).
    columns_to_replace = [col for col in new_columns_dict if col in anndata_object.obs.columns]
    if columns_to_replace:
        for col in columns_to_replace:
            print(f"  WARNING: .obs column '{col}' already exists and will be replaced by the clinical CSV value.")
        anndata_object.obs = anndata_object.obs.drop(columns=columns_to_replace)

    # Join all new columns at once to avoid DataFrame fragmentation.
    # pd.concat produces a fresh, contiguous DataFrame — no PerformanceWarning.
    new_clinical_dataframe = pd.DataFrame(new_columns_dict, index=anndata_object.obs.index)
    anndata_object.obs = pd.concat([anndata_object.obs, new_clinical_dataframe], axis=1)

    print(f"  Done. Added {numeric_column_count} numeric (float32) + {string_column_count} categorical (Categorical) columns.")

    # --- 7. QC block ---
    print("\n" + "=" * 60)
    print("QC — Clinical data enrichment")
    print("=" * 60)
    print(f"  AnnData dimensions     : {anndata_object.n_obs} events × {anndata_object.n_vars} channels")
    print(f"  New .obs columns added : {len(raw_to_sanitized)}")
    print(f"  Total .obs columns     : {len(anndata_object.obs.columns)}")
    print(f"  Subjects matched       : {len(matched_subjects)} / {len(unique_anndata_subjects)}")

    # Verify NaN / missing counts for first few clinical columns
    first_sanitized_names = list(raw_to_sanitized.values())[:3]
    print(f"\n  Column types: {numeric_column_count} numeric (float32), {string_column_count} categorical (Categorical)")
    print("\n  Missing value counts in first 3 clinical columns:")
    for sanitized_column_name in first_sanitized_names:
        col = anndata_object.obs[sanitized_column_name]
        if hasattr(col, "cat"):
            missing_count = (col == "").sum()
            n_categories = len(col.cat.categories)
            print(f"    {sanitized_column_name} [Categorical, {n_categories} cats]: {missing_count} empty / {anndata_object.n_obs} events")
        else:
            nan_count = col.isna().sum()
            print(f"    {sanitized_column_name} [float32]: {nan_count} NaN / {anndata_object.n_obs} events")

    print("\n  Sample clinical data (first event with valid data):")
    valid_event_mask = anndata_subject_ids.isin(matched_subjects)
    if valid_event_mask.any():
        first_valid_index = anndata_object.obs.index[valid_event_mask][0]
        sample_row = anndata_object.obs.loc[first_valid_index, first_sanitized_names]
        print(f"    {sample_row.to_dict()}")

    print("\n  .uns keys after enrichment:")
    print(f"    {list(anndata_object.uns.keys())}")
    print("=" * 60)
    print("Clinical data enrichment — DONE")
    print("=" * 60)

    return anndata_object
