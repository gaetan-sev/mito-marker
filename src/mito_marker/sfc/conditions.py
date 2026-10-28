"""
conditions.py

Experimental conditions of SFC events: the products added to the tube and the
waiting times before acquisition, read from the FCS filename (ADR-019).

Some protocols add products to the same sample one after the other (e.g. the
substrates PGM, then ADP) and wait a given time before each acquisition. Each
filename lists every product present in the tube, in order of addition:

    WORM_001_D08_AL_DR, WORM_001_D08_AL_PGM_DR,
    WORM_001_D08_AL_PGM_ADP_1min_DR, WORM_001_D08_AL_PGM_ADP_30min_DR

The products are declared in the notebook (condition_tokens), not in
controlled_vocabulary.py, so that a new product needs no change to the package.
Any filename word that no rule recognises triggers a warning explaining how to
declare it.

Public API:
    annotate_sfc_conditions(spectral_cytometry_anndata, condition_tokens)
        Write condition, number_of_additions, minutes_after_last_addition and one
        with_<product> column per declared product to .obs.
"""

import difflib
import math
import re
import warnings
from typing import Optional

import anndata
import numpy as np
import pandas as pd

from mito_marker.controlled_vocabulary import (
    SFC_CONDITION_OBS_FIELDS,
    SFC_CONDITION_TOKENS_UNS_KEY,
    SFC_PRODUCT_PRESENCE_OBS_COLUMN_PREFIX,
    SFC_PRODUCT_TOKEN_PATTERN,
)
from mito_marker.sfc.filename_parser import (
    _validate_condition_tokens,
    parse_condition_from_filename,
)


class UnrecognisedFilenameTokenWarning(UserWarning):
    """Warning raised when an FCS filename holds a word that no rule recognises."""


# By default Python shows a given warning only once per session, so re-running a
# notebook cell would hide it the second time. "always" for this category only;
# a notebook can still silence it with
# warnings.filterwarnings("ignore", category=UnrecognisedFilenameTokenWarning).
warnings.filterwarnings("always", category=UnrecognisedFilenameTokenWarning)


def annotate_sfc_conditions(
    spectral_cytometry_anndata: anndata.AnnData,
    condition_tokens: Optional[dict[str, str]] = None,
) -> anndata.AnnData:
    """
    Add the experimental condition of every event to .obs, read from its FCS filename.

    The condition is read from .obs["source_filename"], so this works both during
    ingestion (ingest_sfc_folder calls it) and on a .h5ad loaded later: declaring a
    new product does not require reading the .fcs files again.

    The products declared here are merged with those already saved in
    .uns["sfc_condition_tokens"] (a description given here replaces the saved one),
    and the result is saved back to .uns, so the dataset records what each token
    means. Columns written to .obs:
      - 'condition' (categorical): products in order of addition with waiting
        times, e.g. 'PGM_ADP_30min'; 'No_addition' when no product is declared.
        Categories are in protocol order: fewer additions first, then shorter
        waiting time, so plots and groupby follow the protocol.
      - 'number_of_additions' (int): number of products added.
      - 'minutes_after_last_addition' (float32): waiting time after the last
        product, in minutes; NaN when no time follows the last product.
      - 'with_<product>' (bool), one per declared product, e.g. 'with_ADP'.

    Filename words that no rule recognises are reported in one
    UnrecognisedFilenameTokenWarning (a UserWarning, shown at every call) that
    shows how to declare a product in the notebook. They are not stored anywhere.

    Arguments:
        spectral_cytometry_anndata: SFC AnnData with a 'source_filename' .obs column
                                    (every AnnData built by ingest_sfc_folder has one).
        condition_tokens: Products to declare, {token as written in filenames:
                          plain-English description}, e.g.
                          {"PGM": "Pyruvate + glutamate + malate (complex I substrates)",
                           "ADP": "Adenosine diphosphate"}.
                          None declares nothing new (the saved products still apply).

    Returns:
        The same AnnData, modified in place and returned for chaining.

    Raises:
        ValueError: If .obs has no 'source_filename' column, or if a declared
                    product token or description is invalid.
        TypeError: If condition_tokens is not a dict.
    """
    if "source_filename" not in spectral_cytometry_anndata.obs.columns:
        raise ValueError(
            "annotate_sfc_conditions() reads the condition from .obs['source_filename'], "
            "which this AnnData does not have. Available .obs columns: "
            f"{sorted(spectral_cytometry_anndata.obs.columns)}"
        )
    if condition_tokens is not None:
        _validate_condition_tokens(condition_tokens)

    print("\n--- SFC conditions: products added to the tube (ADR-019) ---")

    # STEP 1 — Products: those saved with the dataset, completed by this call
    saved_tokens = {
        str(token): str(description)
        for token, description in dict(
            spectral_cytometry_anndata.uns.get(SFC_CONDITION_TOKENS_UNS_KEY, {})
        ).items()
    }
    declared_tokens = {**saved_tokens, **(condition_tokens or {})}
    _validate_condition_tokens(declared_tokens)

    print(f"Products saved with the dataset : {sorted(saved_tokens) if saved_tokens else 'none'}")
    print(
        "Products declared in this call  : "
        f"{sorted(condition_tokens) if condition_tokens else 'none'}"
    )
    print(f"Products used ({len(declared_tokens)}):")
    for token, description in declared_tokens.items():
        print(f"  {token:<14}: {description}")

    # STEP 2 — Parse each source file once (events of one file share its condition)
    source_filenames = spectral_cytometry_anndata.obs["source_filename"].astype(str)
    parsed_by_filename = {
        filename: parse_condition_from_filename(filename, declared_tokens)
        for filename in source_filenames.unique()
    }

    # STEP 3 — Write the .obs columns
    condition_order = _order_conditions(list(parsed_by_filename.values()))
    existing_columns = set(spectral_cytometry_anndata.obs.columns)
    written_columns: list[str] = []

    condition_by_filename = {name: parsed["condition"] for name, parsed in parsed_by_filename.items()}
    spectral_cytometry_anndata.obs["condition"] = pd.Categorical(
        source_filenames.map(condition_by_filename).to_numpy(),
        categories=condition_order,
    )
    written_columns.append("condition")

    additions_by_filename = {
        name: parsed["number_of_additions"] for name, parsed in parsed_by_filename.items()
    }
    spectral_cytometry_anndata.obs["number_of_additions"] = (
        source_filenames.map(additions_by_filename).to_numpy(dtype=np.int64)
    )
    written_columns.append("number_of_additions")

    minutes_by_filename = {
        name: parsed["minutes_after_last_addition"] for name, parsed in parsed_by_filename.items()
    }
    spectral_cytometry_anndata.obs["minutes_after_last_addition"] = (
        source_filenames.map(minutes_by_filename).to_numpy(dtype=np.float32)
    )
    written_columns.append("minutes_after_last_addition")

    for token in declared_tokens:
        presence_column = f"{SFC_PRODUCT_PRESENCE_OBS_COLUMN_PREFIX}{token}"
        presence_by_filename = {
            name: token in parsed["products"] for name, parsed in parsed_by_filename.items()
        }
        spectral_cytometry_anndata.obs[presence_column] = (
            source_filenames.map(presence_by_filename).to_numpy(dtype=bool)
        )
        written_columns.append(presence_column)

    spectral_cytometry_anndata.uns[SFC_CONDITION_TOKENS_UNS_KEY] = dict(declared_tokens)

    # STEP 4 — QC block
    _print_condition_qc(
        spectral_cytometry_anndata,
        parsed_by_filename=parsed_by_filename,
        condition_order=condition_order,
        written_columns=written_columns,
        replaced_columns=[name for name in written_columns if name in existing_columns],
    )

    # STEP 5 — Warn about filename words that no rule recognises
    files_by_unrecognised_token: dict[str, list[str]] = {}
    for filename, parsed in parsed_by_filename.items():
        for token in parsed["unrecognised_tokens"]:
            files_by_unrecognised_token.setdefault(token, []).append(filename)
    if files_by_unrecognised_token:
        warnings.warn(
            _build_unrecognised_token_message(files_by_unrecognised_token, declared_tokens),
            UnrecognisedFilenameTokenWarning,
            stacklevel=2,
        )

    return spectral_cytometry_anndata


def _order_conditions(parsed_conditions: list[dict]) -> list[str]:
    """
    Sort condition labels in protocol order.

    Fewer additions first, then shorter waiting time after the last addition (no
    waiting time written comes before any written time), then alphabetical order
    to break ties. Example: No_addition, PGM, PGM_ADP_1min, PGM_ADP_30min.

    Arguments:
        parsed_conditions: Dicts returned by parse_condition_from_filename().

    Returns:
        Unique condition labels in protocol order.
    """
    def protocol_order_key(parsed: dict) -> tuple[int, float, str]:
        minutes = parsed["minutes_after_last_addition"]
        return (
            parsed["number_of_additions"],
            -1.0 if math.isnan(minutes) else minutes,
            parsed["condition"],
        )

    ordered_labels: list[str] = []
    for parsed in sorted(parsed_conditions, key=protocol_order_key):
        if parsed["condition"] not in ordered_labels:
            ordered_labels.append(parsed["condition"])
    return ordered_labels


def _print_condition_qc(
    spectral_cytometry_anndata: anndata.AnnData,
    parsed_by_filename: dict[str, dict],
    condition_order: list[str],
    written_columns: list[str],
    replaced_columns: list[str],
) -> None:
    """
    Print the QC block after the condition columns are written.

    Shows dimensions, one line per condition (files, events, additions, waiting
    time), a preview of the new columns, the .uns keys, and a PASS/FAIL check
    that every event received a condition.

    Arguments:
        spectral_cytometry_anndata: The annotated AnnData.
        parsed_by_filename: {source filename: parse_condition_from_filename() result}.
        condition_order: Condition labels in protocol order.
        written_columns: .obs columns written by annotate_sfc_conditions().
        replaced_columns: Those of written_columns that existed before the call.

    Returns:
        None
    """
    separator = "=" * 60
    observation_dataframe = spectral_cytometry_anndata.obs

    print(f"\n{separator}")
    print("QC CHECKPOINT: SFC conditions from filenames")
    print(separator)
    print(f"    Observations (events) : {spectral_cytometry_anndata.n_obs:,}")
    print(f"    Variables (channels)  : {spectral_cytometry_anndata.n_vars}")
    print(f"    Source files          : {len(parsed_by_filename)}")

    print(f"\n[1] Conditions in protocol order ({len(condition_order)})")
    print(f"    {'condition':<32}{'files':>7}{'events':>12}{'additions':>11}{'minutes':>9}")
    for condition in condition_order:
        files_with_condition = [
            parsed for parsed in parsed_by_filename.values() if parsed["condition"] == condition
        ]
        event_count = int((observation_dataframe["condition"] == condition).sum())
        minutes = files_with_condition[0]["minutes_after_last_addition"]
        minutes_display = "—" if math.isnan(minutes) else f"{minutes:g}"
        print(
            f"    {condition:<32}{len(files_with_condition):>7}{event_count:>12,}"
            f"{files_with_condition[0]['number_of_additions']:>11}{minutes_display:>9}"
        )

    print("\n[2] Columns written to .obs")
    for column_name in written_columns:
        description = SFC_CONDITION_OBS_FIELDS.get(
            column_name, "True when the product is in the tube"
        )
        replaced_note = " (replaced)" if column_name in replaced_columns else ""
        print(f"    {column_name:<30}: {description}{replaced_note}")
    print("\n    .obs preview (first 5 rows):")
    print(observation_dataframe[written_columns].head(5).to_string())

    uns_keys = list(spectral_cytometry_anndata.uns.keys())
    print(f"\n[3] .uns keys : {uns_keys if uns_keys else 'none'}")

    missing_condition_count = int(observation_dataframe["condition"].isna().sum())
    print("\n[4] Sanity check: every event has a condition")
    if missing_condition_count == 0:
        print("    PASS — no event without a condition")
    else:
        print(f"    FAIL — {missing_condition_count:,} events without a condition")
    print(separator + "\n")


def _build_unrecognised_token_message(
    files_by_unrecognised_token: dict[str, list[str]],
    declared_tokens: dict[str, str],
) -> str:
    """
    Build the warning text for filename words that no rule recognises.

    The text lists each word with one example file, suggests a declared product
    when the word looks like a typing error, and shows the notebook code that
    declares a new product.

    Arguments:
        files_by_unrecognised_token: {unrecognised word: filenames containing it}.
        declared_tokens: The products currently declared.

    Returns:
        The warning message.
    """
    message_lines = [
        f"{len(files_by_unrecognised_token)} word(s) of the FCS filenames were not "
        "recognised. They were ignored: they appear in no .obs column.",
    ]
    for token, filenames in sorted(files_by_unrecognised_token.items()):
        message_lines.append(f"  - '{token}' in {len(filenames)} file(s), e.g. {filenames[0]}")
        suggested_token = _suggest_declared_token(token, declared_tokens)
        if suggested_token is not None:
            message_lines.append(
                f"      Did you mean '{suggested_token}'? If so, correct the filename."
            )

    declarable_tokens = [
        token for token in sorted(files_by_unrecognised_token)
        if re.fullmatch(SFC_PRODUCT_TOKEN_PATTERN, token)
    ]
    message_lines += [
        "",
        "If a word is a product added to the tube, declare it in your notebook with a",
        "short description, then run ingest_sfc_folder() again with condition_tokens:",
        "",
        "    condition_tokens = {",
    ]
    for token, description in declared_tokens.items():
        message_lines.append(f'        "{token}": "{description}",')
    for token in declarable_tokens:
        message_lines.append(f'        "{token}": "<what this product is>",')
    message_lines += [
        "    }",
        "    ingest_sfc_folder(..., condition_tokens=condition_tokens)",
        "",
        "or update an AnnData already loaded, without reading the .fcs files again:",
        "",
        "    annotate_sfc_conditions(spectral_cytometry_anndata, condition_tokens=condition_tokens)",
        "    spectral_cytometry_anndata.write_h5ad(output_file_path)",
        "",
        "A waiting time (e.g. 30min) must come right after the product it refers to.",
        "If a word is not a product (e.g. a FlowJo gate name), you can ignore this warning.",
    ]
    return "\n".join(message_lines)


def _suggest_declared_token(token: str, declared_tokens: dict[str, str]) -> Optional[str]:
    """
    Return the declared product closest to an unrecognised word, if any is close.

    Case is ignored and letters may be swapped (e.g. 'pgm' or 'PMG' → 'PGM').

    Arguments:
        token: The unrecognised word.
        declared_tokens: The products currently declared.

    Returns:
        The closest declared token, or None when none is similar enough.
    """
    declared_by_lowercase = {declared.lower(): declared for declared in declared_tokens}
    close_matches = difflib.get_close_matches(
        token.lower(), list(declared_by_lowercase), n=1, cutoff=0.6
    )
    if not close_matches:
        return None
    return declared_by_lowercase[close_matches[0]]
