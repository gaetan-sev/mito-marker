"""
filename_columns.py

.obs columns declared in the notebook and read from words of the FCS filename
(ADR-021).

A colleague may write in the filenames a description of the sample that the
package does not know, e.g. Smurf and non-Smurf flies:

    FLY_001_D30_S_DR, FLY_002_D30_NonS_DR

The notebook names the .obs column and the value each word gives,

    filename_columns = {"smurf_status": {"S": "Smurf", "NonS": "Non_Smurf"}}

so that a new description needs no change to the package. The declaration is
saved with the dataset in .uns["sfc_filename_columns"].

Public API:
    annotate_sfc_filename_columns(spectral_cytometry_anndata, filename_columns)
        Write one categorical .obs column per declared column.
"""

from typing import Optional

import anndata
import pandas as pd

from mito_marker.controlled_vocabulary import (
    SFC_CONDITION_TOKENS_UNS_KEY,
    SFC_FILENAME_COLUMNS_UNS_KEY,
    SFC_OBS_COLUMNS_EXTENDABLE_FROM_FILENAME,
)
from mito_marker.sfc.filename_parser import (
    _filename_column_categories,
    _split_filename_into_words,
    _validate_filename_columns,
    parse_filename_columns,
)

# Number of conflicting files listed in full in the error message.
_MAX_CONFLICTS_SHOWN = 10


def annotate_sfc_filename_columns(
    spectral_cytometry_anndata: anndata.AnnData,
    filename_columns: Optional[dict[str, dict[str, str]]] = None,
) -> anndata.AnnData:
    """
    Add to .obs the columns declared in the notebook, read from FCS filename words.

    Each declared column lists the filename words that give it a value (ADR-021):

        filename_columns = {
            "smurf_status": {"S": "Smurf", "NonS": "Non_Smurf"},
        }

        FLY_001_D30_S_DR     → smurf_status = "Smurf"
        FLY_002_D30_NonS_DR  → smurf_status = "Non_Smurf"
        FLY_003_D30_DR       → smurf_status = NaN (no word of the column)

    The values are read from .obs["source_filename"], so this works both during
    ingestion (ingest_sfc_folder calls it) and on a .h5ad loaded later: declaring
    a new column does not require reading the .fcs files again.

    The column can be:
      - a new column (e.g. 'smurf_status');
      - a column created earlier in the notebook: it is replaced;
      - one of the columns the package already reads and that accept new words
        (SFC_OBS_COLUMNS_EXTENDABLE_FROM_FILENAME: diet, age_group, treatment).
        The built-in words keep their value (e.g. 'AL' stays 'AL') and the
        declared words add new values (e.g. {"diet": {"CR": "Caloric_Restriction"}}).
    The columns computed by the ingestion (specie, subject_ID, age, marker,
    condition, …) cannot be declared.

    Every declared column is written as a categorical column, NaN for a file
    that holds no word of the column. Categories: the built-in values first for
    an extendable column, then the declared values in alphabetical order (the
    same before and after an .h5ad round trip).

    The columns declared here are merged with those already saved in
    .uns["sfc_filename_columns"] (a value given here replaces the saved one for
    the same word), and the result is saved back to .uns, so the dataset records
    which word gives which value. The words of the declared columns are then
    recognised by annotate_sfc_conditions(), which no longer reports them as
    unrecognised: call this function first.

    Arguments:
        spectral_cytometry_anndata: SFC AnnData with a 'source_filename' .obs column
                                    (every AnnData built by ingest_sfc_folder has one).
        filename_columns: Columns to declare, {column name: {word as written in
                          filenames: value stored in .obs}}, e.g.
                          {"smurf_status": {"S": "Smurf", "NonS": "Non_Smurf"}}.
                          A word matches only as a whole word between '_', and
                          case matters. Several words may give the same value.
                          None declares nothing new (the saved columns still apply).

    Returns:
        The same AnnData, modified in place and returned for chaining.

    Raises:
        ValueError: If .obs has no 'source_filename' column, if a declared column,
                    word or value is invalid, or if a filename holds words giving
                    two different values to one column (all such files are listed).
        TypeError: If filename_columns is not a dict of dicts.
    """
    observation_dataframe = spectral_cytometry_anndata.obs
    if "source_filename" not in observation_dataframe.columns:
        raise ValueError(
            "annotate_sfc_filename_columns() reads the values from "
            ".obs['source_filename'], which this AnnData does not have. Available "
            f".obs columns: {sorted(observation_dataframe.columns)}"
        )
    if filename_columns is not None:
        _validate_filename_columns(filename_columns)

    print("\n--- SFC .obs columns read from filename words (ADR-021) ---")

    # STEP 1 — Columns: those saved with the dataset, completed by this call.
    # The saved products are checked too, so that a word never has two meanings.
    saved_columns = _read_saved_filename_columns(spectral_cytometry_anndata)
    declared_columns = _merge_filename_columns(saved_columns, filename_columns or {})
    saved_products = {
        str(token): str(description)
        for token, description in dict(
            spectral_cytometry_anndata.uns.get(SFC_CONDITION_TOKENS_UNS_KEY, {})
        ).items()
    }
    _validate_filename_columns(declared_columns, condition_tokens=saved_products)

    print(f"Columns saved with the dataset : {sorted(saved_columns) if saved_columns else 'none'}")
    print(
        "Columns declared in this call  : "
        f"{sorted(filename_columns) if filename_columns else 'none'}"
    )
    if not declared_columns:
        print("No .obs column declared from filename words: nothing to write.")
        return spectral_cytometry_anndata

    print(f"Columns used ({len(declared_columns)}):")
    for column_name, value_by_token in declared_columns.items():
        word_list = ", ".join(f"{token} → {value}" for token, value in value_by_token.items())
        print(f"  {column_name:<20}: {word_list}")

    # STEP 2 — Parse each source file once (events of one file share its values).
    # Every conflicting file is collected before raising, so that one run lists
    # all the filenames to correct.
    source_filenames = observation_dataframe["source_filename"].astype(str)
    values_by_filename: dict[str, dict[str, Optional[str]]] = {}
    conflict_messages: list[str] = []
    for filename in source_filenames.unique():
        try:
            values_by_filename[filename] = parse_filename_columns(filename, declared_columns)
        except ValueError as conflict_error:
            conflict_messages.append(str(conflict_error))
    if conflict_messages:
        raise ValueError(_build_conflict_message(conflict_messages))

    # STEP 3 — Write one categorical column per declared column
    existing_columns = set(observation_dataframe.columns)
    for column_name, value_by_token in declared_columns.items():
        value_by_filename = {
            filename: values[column_name] for filename, values in values_by_filename.items()
        }
        # A file without a word of the column maps to None, stored as NaN.
        observation_dataframe[column_name] = pd.Categorical(
            source_filenames.map(value_by_filename).to_numpy(),
            categories=_filename_column_categories(column_name, value_by_token),
        )

    spectral_cytometry_anndata.uns[SFC_FILENAME_COLUMNS_UNS_KEY] = {
        column_name: dict(value_by_token) for column_name, value_by_token in declared_columns.items()
    }

    # STEP 4 — QC block
    _print_filename_columns_qc(
        spectral_cytometry_anndata,
        declared_columns=declared_columns,
        values_by_filename=values_by_filename,
        replaced_columns=[name for name in declared_columns if name in existing_columns],
    )
    return spectral_cytometry_anndata


def _read_saved_filename_columns(
    spectral_cytometry_anndata: anndata.AnnData,
) -> dict[str, dict[str, str]]:
    """
    Return the .obs columns saved in .uns["sfc_filename_columns"], as plain strings.

    Arguments:
        spectral_cytometry_anndata: An SFC AnnData, possibly read from .h5ad.

    Returns:
        {column name: {filename word: value}}, empty when nothing is saved.
    """
    saved_columns = spectral_cytometry_anndata.uns.get(SFC_FILENAME_COLUMNS_UNS_KEY, {})
    return {
        str(column_name): {str(token): str(value) for token, value in dict(value_by_token).items()}
        for column_name, value_by_token in dict(saved_columns).items()
    }


def _merge_filename_columns(
    saved_columns: dict[str, dict[str, str]],
    new_columns: dict[str, dict[str, str]],
) -> dict[str, dict[str, str]]:
    """
    Merge newly declared columns into saved ones, word by word.

    A new column is added; for a column already saved, new words are added and a
    word declared again takes its new value.

    Arguments:
        saved_columns: Columns saved with the dataset.
        new_columns: Columns declared in this call.

    Returns:
        The merged declaration; saved columns first, in their saved order.
    """
    merged_columns = {
        column_name: dict(value_by_token) for column_name, value_by_token in saved_columns.items()
    }
    for column_name, value_by_token in new_columns.items():
        merged_columns[column_name] = {**merged_columns.get(column_name, {}), **value_by_token}
    return merged_columns


def _build_conflict_message(conflict_messages: list[str]) -> str:
    """
    Build the error text listing the files whose words give two values to one column.

    Arguments:
        conflict_messages: One message per conflicting file (parse_filename_columns).

    Returns:
        The error message.
    """
    message_lines = [
        f"{len(conflict_messages)} file(s) give two values to one declared .obs column. "
        "Nothing was written. Correct these filenames or the declaration:",
    ]
    message_lines += [f"  - {message}" for message in conflict_messages[:_MAX_CONFLICTS_SHOWN]]
    if len(conflict_messages) > _MAX_CONFLICTS_SHOWN:
        message_lines.append(
            f"  … and {len(conflict_messages) - _MAX_CONFLICTS_SHOWN} more file(s)."
        )
    return "\n".join(message_lines)


def _print_filename_columns_qc(
    spectral_cytometry_anndata: anndata.AnnData,
    declared_columns: dict[str, dict[str, str]],
    values_by_filename: dict[str, dict[str, Optional[str]]],
    replaced_columns: list[str],
) -> None:
    """
    Print the QC block after the declared columns are written.

    Shows dimensions, one table per column (value, words, files, events, and the
    files without a value), a preview of the columns, the .uns keys, and a check
    that every declared word appears in at least one filename: a word found
    nowhere is usually a spelling or case mistake in the declaration.

    Arguments:
        spectral_cytometry_anndata: The annotated AnnData.
        declared_columns: {column name: {filename word: value}} used for the call.
        values_by_filename: {source filename: parse_filename_columns() result}.
        replaced_columns: Declared columns that existed in .obs before the call.

    Returns:
        None
    """
    separator = "=" * 60
    observation_dataframe = spectral_cytometry_anndata.obs

    print(f"\n{separator}")
    print("QC CHECKPOINT: SFC .obs columns from filename words")
    print(separator)
    print(f"    Observations (events) : {spectral_cytometry_anndata.n_obs:,}")
    print(f"    Variables (channels)  : {spectral_cytometry_anndata.n_vars}")
    print(f"    Source files          : {len(values_by_filename)}")

    print(f"\n[1] Values per column ({len(declared_columns)} column(s))")
    files_without_value_by_column: dict[str, list[str]] = {}
    for column_name, value_by_token in declared_columns.items():
        if column_name in SFC_OBS_COLUMNS_EXTENDABLE_FROM_FILENAME:
            column_kind = "built-in column, declared words added"
        elif column_name in replaced_columns:
            column_kind = "replaced"
        else:
            column_kind = "new column"
        print(f"    {column_name} ({column_kind})")
        print(f"      {'value':<26}{'words':<22}{'files':>7}{'events':>12}")

        column_values = observation_dataframe[column_name]
        for value in column_values.cat.categories:
            words = [token for token, declared_value in value_by_token.items() if declared_value == value]
            file_count = sum(
                1 for values in values_by_filename.values() if values[column_name] == value
            )
            if not words and file_count == 0:
                continue  # built-in value absent from these files: not worth a line
            event_count = int((column_values == value).sum())
            words_display = ", ".join(words) if words else "(built-in)"
            print(f"      {str(value):<26}{words_display:<22}{file_count:>7}{event_count:>12,}")

        files_without_value = [
            filename for filename, values in values_by_filename.items() if values[column_name] is None
        ]
        files_without_value_by_column[column_name] = files_without_value
        missing_event_count = int(column_values.isna().sum())
        print(f"      {'(no value: NaN)':<26}{'':<22}{len(files_without_value):>7}{missing_event_count:>12,}")

    print("\n[2] Columns written to .obs")
    for column_name in declared_columns:
        replaced_note = " (replaced)" if column_name in replaced_columns else ""
        print(f"    {column_name}{replaced_note}")
    print("\n    .obs preview (first 5 rows):")
    print(observation_dataframe[list(declared_columns)].head(5).to_string())

    uns_keys = list(spectral_cytometry_anndata.uns.keys())
    print(f"\n[3] .uns keys : {uns_keys if uns_keys else 'none'}")

    print("\n[4] Sanity check: every declared word appears in at least one filename")
    words_in_filenames: set[str] = set()
    for filename in values_by_filename:
        words_in_filenames.update(_split_filename_into_words(filename))
    words_never_found = [
        f"'{token}' ({column_name})"
        for column_name, value_by_token in declared_columns.items()
        for token in value_by_token
        if token not in words_in_filenames
    ]
    if not words_never_found:
        print("    PASS — every declared word appears in at least one filename")
    else:
        print(
            f"    CHECK — {len(words_never_found)} declared word(s) appear in no filename "
            f"(spelling? case matters): {', '.join(words_never_found)}"
        )
    for column_name, files_without_value in files_without_value_by_column.items():
        if files_without_value:
            print(
                f"    INFO — '{column_name}' is NaN for {len(files_without_value)} file(s), "
                f"e.g. {files_without_value[0]}"
            )
    print(separator + "\n")
