"""
qc.py

Quality-control printing functions for the SFC ingestion pipeline.

All functions in this module are stateless: they print information to the console
and return None. They never modify any data structure. Their purpose is to make
the full execution log readable by a beginner scientist without requiring them to
open a Python interpreter or inspect variables manually.

Every major pipeline step calls at least one function from this module so the
execution log serves as a complete, self-contained audit trail.
"""

from typing import List

import anndata
import numpy as np


def print_fcs_load_summary(
    fcs_file_path: str,
    channel_names_before_exclusion: List[str],
    channel_names_after_exclusion: List[str],
    event_count: int,
) -> None:
    """
    Print a load summary after a single FCS file is read by FlowIO.

    Shows the filename, total channels in the file, which channels were excluded,
    how many channels were retained, and the number of events.

    Arguments:
        fcs_file_path: Path to the FCS file that was loaded.
        channel_names_before_exclusion: Full channel name list before filtering.
        channel_names_after_exclusion: Channel name list after applying exclusions.
        event_count: Number of events (rows) in the loaded file.

    Returns:
        None
    """
    import os
    filename = os.path.basename(fcs_file_path)
    n_before = len(channel_names_before_exclusion)
    n_after = len(channel_names_after_exclusion)
    n_excluded = n_before - n_after
    excluded_names = sorted(
        set(channel_names_before_exclusion) - set(channel_names_after_exclusion)
    )

    print("  ┌─ FCS LOAD SUMMARY " + "─" * 50)
    print(f"  │ File              : {filename}")
    print(f"  │ Events            : {event_count:,}")
    print(f"  │ Channels in file  : {n_before}")
    print(f"  │ Channels excluded : {n_excluded}  {excluded_names if excluded_names else ''}")
    print(f"  │ Channels retained : {n_after}")
    print("  └" + "─" * 67)


def print_obs_metadata_summary(obs_metadata: dict[str, str | bool | None]) -> None:
    """
    Print the obs metadata dict extracted from a filename, field by field.

    Each field is printed on its own line. Optional fields that were absent from the
    filename are shown explicitly as 'NOT PRESENT IN FILENAME' to avoid any ambiguity
    between a missing value and a None that slipped through unexpectedly.

    Arguments:
        obs_metadata: Dict from parse_fcs_filename(), with keys: 'specie',
                      'subject_ID', 'age', 'diet', 'dilution', 'marker',
                      'FlowAI_Pass', 'source_filename'.

    Returns:
        None
    """
    print("  ┌─ FILENAME METADATA " + "─" * 49)
    field_display_order = [
        "specie",
        "subject_ID",
        "age",
        "diet",
        "dilution",
        "marker",
        "FlowAI_Pass",
        "source_filename",
    ]
    for field_name in field_display_order:
        value = obs_metadata.get(field_name)
        display_value = "NOT PRESENT IN FILENAME" if value is None else str(value)
        print(f"  │ {field_name:<20}: {display_value}")
    print("  └" + "─" * 67)


def print_anndata_qc(
    sfc_anndata: anndata.AnnData,
    step_label: str,
) -> None:
    """
    Print the full QC block for an AnnData object after a named pipeline step.

    Prints dimensions, .obs and .var previews, .obsm and .uns key listings,
    NaN and Inf counts in .X, and the min/max/mean of .X values. Ends with an
    explicit PASS or FAIL sanity check for data integrity.

    This output is designed so a beginner can verify at a glance that data loaded
    correctly and that no corrupted values (NaN, Inf) entered the matrix.

    Arguments:
        sfc_anndata: The AnnData object to inspect.
        step_label: Human-readable label for this QC checkpoint,
                    e.g. 'After building from single FCS file'.

    Returns:
        None
    """
    separator = "=" * 60

    print(f"\n{separator}")
    print(f"QC CHECKPOINT: {step_label}")
    print(separator)

    # [1] Dimensions
    print("\n[1] Dimensions")
    print(f"    Observations (events) : {sfc_anndata.n_obs:,}")
    print(f"    Variables (channels)  : {sfc_anndata.n_vars}")

    # [2] .obs preview
    print("\n[2] .obs preview (first 5 rows)")
    print(sfc_anndata.obs.head(5).to_string())

    # [3] .var preview
    print("\n[3] .var preview (first 5 rows)")
    print(sfc_anndata.var.head(5).to_string())

    # [4] .obsm keys
    obsm_keys = list(sfc_anndata.obsm.keys()) if sfc_anndata.obsm else []
    print(f"\n[4] .obsm keys : {obsm_keys if obsm_keys else 'none'}")

    # [5] .uns keys
    uns_keys = list(sfc_anndata.uns.keys()) if sfc_anndata.uns else []
    print(f"[5] .uns keys  : {uns_keys if uns_keys else 'none'}")

    # [6] Data quality in .X
    x_matrix = sfc_anndata.X
    nan_count = int(np.sum(np.isnan(x_matrix)))
    inf_count = int(np.sum(np.isinf(x_matrix)))
    x_min = float(np.min(x_matrix))
    x_max = float(np.max(x_matrix))
    x_mean = float(np.mean(x_matrix))

    print("\n[6] Data quality (.X)")
    print(f"    NaN values   : {nan_count:,}")
    print(f"    Inf values   : {inf_count:,}")
    print(f"    Min value    : {x_min:.4f}")
    print(f"    Max value    : {x_max:.4f}")
    print(f"    Mean value   : {x_mean:.4f}")

    # [7] Sanity check
    all_finite = (nan_count == 0) and (inf_count == 0)
    print("\n[7] Sanity check: all .X values are finite")
    if all_finite:
        print("    PASS — no NaN or Inf detected")
    else:
        print(f"    FAIL — {nan_count} NaN and {inf_count} Inf values detected in .X")

    print(separator + "\n")


def print_append_summary(
    existing_n_obs: int,
    new_n_obs: int,
    combined_n_obs: int,
    output_file_path: str,
) -> None:
    """
    Print a summary after appending a new AnnData to an existing .h5ad file.

    Arguments:
        existing_n_obs: Number of events in the pre-existing .h5ad.
        new_n_obs: Number of events added from the new FCS file(s).
        combined_n_obs: Total events after concatenation.
        output_file_path: Path where the combined file was saved.

    Returns:
        None
    """
    print("  ┌─ APPEND SUMMARY " + "─" * 51)
    print(f"  │ Events before append : {existing_n_obs:,}")
    print(f"  │ Events in new data   : {new_n_obs:,}")
    print(f"  │ Events after append  : {combined_n_obs:,}")
    print(f"  │ Saved to             : {output_file_path}")
    print("  └" + "─" * 67)


def print_folder_ingestion_summary(
    fcs_files_found: List[str],
    fcs_files_successfully_loaded: List[str],
    fcs_files_skipped: List[str],
    fcs_files_failed: List[str],
    total_events: int,
    output_file_path: str,
) -> None:
    """
    Print the final summary after processing an entire folder of FCS files.

    Arguments:
        fcs_files_found: All .fcs filenames discovered in the folder.
        fcs_files_successfully_loaded: Filenames that loaded without error.
        fcs_files_skipped: Filenames that were already present in the existing
                           .h5ad and were skipped to avoid duplication.
        fcs_files_failed: Filenames that raised an exception during loading.
        total_events: n_obs of the final combined AnnData.
        output_file_path: Path where the .h5ad was saved.

    Returns:
        None
    """
    separator = "=" * 60
    print(separator)
    print("FOLDER INGESTION COMPLETE")
    print(separator)
    print(f"  .fcs files found       : {len(fcs_files_found)}")
    print(f"  Successfully loaded    : {len(fcs_files_successfully_loaded)}")
    for filename in fcs_files_successfully_loaded:
        print(f"    - {filename}")
    print(f"  Skipped (duplicate)    : {len(fcs_files_skipped)}")
    for filename in fcs_files_skipped:
        print(f"    - {filename}")
    print(f"  Failed                 : {len(fcs_files_failed)}")
    for filename in fcs_files_failed:
        print(f"    - {filename}")
    print()
    print(f"  Total events in output : {total_events:,}")
    print(f"  Output file            : {output_file_path}")
    print(separator)
