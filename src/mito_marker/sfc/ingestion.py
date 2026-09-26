"""
ingestion.py

Orchestrator for the SFC data ingestion pipeline.

This module contains the single public function that end users call from a notebook:
    ingest_sfc_folder()

It discovers all .fcs files in a directory, processes each one through the full
pipeline (filename parsing → FCS loading → AnnData construction → QC), concatenates
the per-file AnnData objects into a single combined dataset, and saves the result
as an .h5ad file. It also supports an append mode that extends an existing .h5ad
file with new events, with strict channel-compatibility checking.
"""

import gc
import os
from typing import List, Optional

import anndata
from flowio.exceptions import FlowIOException

from mito_marker.controlled_vocabulary import SFC_CHANNELS_TO_EXCLUDE
from mito_marker.sfc.anndata_builder import build_sfc_anndata
from mito_marker.sfc.fcs_loader import load_fcs_file
from mito_marker.sfc.filename_parser import parse_fcs_filename
from mito_marker.sfc.qc import (
    print_anndata_qc,
    print_append_summary,
    print_fcs_load_summary,
    print_folder_ingestion_summary,
    print_obs_metadata_summary,
)


def ingest_sfc_folder(
    data_directory_path: str,
    output_file_path: str,
    channels_to_exclude: Optional[List[str]] = None,
    append_to_existing: bool = False,
) -> anndata.AnnData:
    """
    Ingest all .fcs files in a directory into a single AnnData object and save as .h5ad.

    This is the primary public entry point for the SFC pipeline. For each .fcs file
    found in data_directory_path, it: parses the filename to extract .obs metadata,
    loads the event matrix with FlowIO, builds a per-file AnnData, then concatenates
    all per-file objects into one combined dataset.

    If append_to_existing=True and the output file already exists, the existing .h5ad
    is loaded first and new events are appended after verifying that channel names
    match exactly. If append_to_existing=False and the file already exists, it is
    silently overwritten.

    Arguments:
        data_directory_path: Path to a directory containing .fcs files. Does not
                             recurse into subdirectories. Works identically in both
                             Google Colab and GitHub Codespaces — never hardcode paths.
        output_file_path: Path where the resulting AnnData will be written as .h5ad.
                          Parent directory is created automatically if it does not exist.
        channels_to_exclude: Optional list of channel names to drop from .X
                             (e.g. ['Time', 'FlowAI_RG']). Applied identically to every
                             file. Unrecognised names produce a warning but do not fail.
        append_to_existing: If True, load the existing .h5ad at output_file_path and
                            concatenate new events to it. Raises ValueError if the channel
                            names in the new files do not exactly match the existing .var.
                            If the output file does not exist, behaves like False.

    Returns:
        The combined anndata.AnnData object (same data as saved to output_file_path).

    Raises:
        FileNotFoundError: If data_directory_path does not exist.
        ValueError: If no .fcs files are found in data_directory_path, or if
                    channel names mismatch when appending to an existing .h5ad.
    """
    # SFC_CHANNELS_TO_EXCLUDE (FJComp-* channels and other non-informative columns) are always dropped.
    # Any caller-supplied list is applied on top.
    effective_channels_to_exclude: List[str] = list(
        dict.fromkeys(SFC_CHANNELS_TO_EXCLUDE + (channels_to_exclude or []))
    )

    print("\n--- SFC Ingestion Pipeline Starting ---")
    print(f"Scanning directory       : {data_directory_path}")
    print(f"Always excluded          : {len(SFC_CHANNELS_TO_EXCLUDE)} channels")
    if channels_to_exclude:
        print(f"Additional exclusions    : {channels_to_exclude}")
    else:
        print("Additional exclusions    : none")
    print(f"Append to existing       : {append_to_existing}")
    print(f"Output file              : {output_file_path}")

    # STEP 1 — Discover FCS files
    fcs_file_paths = _discover_fcs_files(data_directory_path)
    fcs_basenames = [os.path.basename(p) for p in fcs_file_paths]
    print(f"\nFound {len(fcs_file_paths)} .fcs file(s):")
    for basename in fcs_basenames:
        print(f"  - {basename}")

    # STEP 2 — Load existing .h5ad if in append mode
    anndata_objects_to_combine: List[anndata.AnnData] = []
    existing_n_obs = 0
    already_ingested_filenames: set = set()

    if append_to_existing and os.path.exists(output_file_path):
        print(f"\nAppend mode: loading existing .h5ad from {output_file_path}")
        existing_sfc_anndata = anndata.read_h5ad(output_file_path)
        existing_n_obs = existing_sfc_anndata.n_obs
        print(
            f"Existing file: {existing_n_obs:,} events × "
            f"{existing_sfc_anndata.n_vars} channels"
        )
        # Collect filenames already present so we can skip them in STEP 3.
        # source_filename uniquely identifies the originating .fcs file for every event.
        already_ingested_filenames = set(
            existing_sfc_anndata.obs["source_filename"].unique()
        )
        print(
            f"Already ingested source files: {len(already_ingested_filenames)} "
            f"— duplicates will be skipped."
        )
        anndata_objects_to_combine.append(existing_sfc_anndata)
    elif append_to_existing and not os.path.exists(output_file_path):
        print(
            "\nAppend mode requested but no existing file found at "
            f"{output_file_path}. Creating new file."
        )

    # STEP 3 — Process each FCS file
    successfully_loaded: List[str] = []
    failed_filenames: List[str] = []
    skipped_filenames: List[str] = []

    for file_index, fcs_file_path in enumerate(fcs_file_paths):
        basename = os.path.basename(fcs_file_path)
        print(f"\n[{file_index + 1}/{len(fcs_file_paths)}] Processing: {basename}")

        if basename in already_ingested_filenames:
            print(
                f"  SKIP: '{basename}' is already present in the existing .h5ad "
                f"— skipping to avoid duplication."
            )
            skipped_filenames.append(basename)
            continue

        try:
            per_file_anndata = _ingest_single_fcs_file(fcs_file_path, effective_channels_to_exclude)
        except (ValueError, FileNotFoundError, FlowIOException) as ingestion_error:
            print(f"  ERROR loading {basename}: {ingestion_error}")
            failed_filenames.append(basename)
            continue

        # Channel compatibility check is deliberately OUTSIDE the try/except above.
        # A mismatch means the entire batch is inconsistent — it must propagate
        # immediately rather than being silently caught and added to failed_filenames.
        if anndata_objects_to_combine:
            _validate_var_compatibility(anndata_objects_to_combine[0], per_file_anndata)

        anndata_objects_to_combine.append(per_file_anndata)
        successfully_loaded.append(basename)

    # STEP 4 — Concatenate
    if not anndata_objects_to_combine:
        raise ValueError(
            "No FCS files were successfully loaded. Cannot create an AnnData object."
        )

    print(f"\nConcatenating {len(anndata_objects_to_combine)} AnnData object(s)...")
    combined_sfc_anndata = _concatenate_anndata_objects(anndata_objects_to_combine)
    del anndata_objects_to_combine
    gc.collect()
    print(
        f"Combined AnnData: {combined_sfc_anndata.n_obs:,} events × "
        f"{combined_sfc_anndata.n_vars} channels"
    )

    # Log append summary if we added to existing data
    if append_to_existing and existing_n_obs > 0:
        new_n_obs = combined_sfc_anndata.n_obs - existing_n_obs
        print_append_summary(
            existing_n_obs=existing_n_obs,
            new_n_obs=new_n_obs,
            combined_n_obs=combined_sfc_anndata.n_obs,
            output_file_path=output_file_path,
        )

    # STEP 5 — QC block on combined object
    print_anndata_qc(combined_sfc_anndata, step_label="After full folder ingestion")

    # STEP 6 — Save to disk
    print(f"Saving to {output_file_path} ...")
    output_parent_directory = os.path.dirname(output_file_path)
    if output_parent_directory:
        os.makedirs(output_parent_directory, exist_ok=True)
    combined_sfc_anndata.write_h5ad(output_file_path)
    print("Saved successfully.")

    # STEP 7 — Final summary
    print_folder_ingestion_summary(
        fcs_files_found=fcs_basenames,
        fcs_files_successfully_loaded=successfully_loaded,
        fcs_files_skipped=skipped_filenames,
        fcs_files_failed=failed_filenames,
        total_events=combined_sfc_anndata.n_obs,
        output_file_path=output_file_path,
    )

    return combined_sfc_anndata


def _ingest_single_fcs_file(
    fcs_file_path: str,
    channels_to_exclude: Optional[List[str]],
) -> anndata.AnnData:
    """
    Run the full ingestion pipeline for a single FCS file.

    Orchestrates: parse_fcs_filename → load_fcs_file → build_sfc_anndata.
    Calls QC printing functions after each step. If any step raises an exception,
    the filename is appended to the error message for clearer diagnostics.

    Arguments:
        fcs_file_path: Path to a single .fcs file.
        channels_to_exclude: Channel names to exclude (passed through to load_fcs_file).

    Returns:
        An AnnData object for this single file.

    Raises:
        ValueError: Propagated from the filename parser or FCS loader, with the
                    filename appended to the message for context.
        FileNotFoundError: If the file does not exist.
    """
    basename = os.path.basename(fcs_file_path)

    # STEP A — Parse filename
    print(f"  Parsing filename: {basename}")
    obs_metadata = parse_fcs_filename(basename)
    print_obs_metadata_summary(obs_metadata)

    # STEP B — Load FCS
    print("  Loading FCS data with FlowIO...")
    # Load without exclusions first to report before/after channel counts
    (
        channel_names_all,
        channel_descriptions_all,
        event_matrix,
        instrument_metadata,
        fcs_obs_metadata,
    ) = load_fcs_file(fcs_file_path, channels_to_exclude=None)
    channel_names_all = list(channel_names_all)  # copy before potential mutation

    # Apply exclusions separately so we can show before/after in the load summary
    if channels_to_exclude:
        from mito_marker.sfc.fcs_loader import _apply_channel_exclusions
        channel_names, channel_descriptions, event_matrix = _apply_channel_exclusions(
            channel_names_all, channel_descriptions_all, event_matrix, channels_to_exclude
        )
    else:
        channel_names = channel_names_all
        channel_descriptions = channel_descriptions_all

    print_fcs_load_summary(
        fcs_file_path=fcs_file_path,
        channel_names_before_exclusion=channel_names_all,
        channel_names_after_exclusion=channel_names,
        event_count=event_matrix.shape[0],
    )

    # STEP C — Build AnnData
    print("  Building AnnData object...")
    per_file_anndata = build_sfc_anndata(
        obs_metadata=obs_metadata,
        channel_names=channel_names,
        channel_descriptions=channel_descriptions,
        event_matrix=event_matrix,
        fcs_instrument_metadata=instrument_metadata,
        fcs_obs_metadata=fcs_obs_metadata,
    )
    print_anndata_qc(per_file_anndata, step_label=f"After building from {basename}")

    del event_matrix
    gc.collect()

    return per_file_anndata


def _validate_var_compatibility(
    existing_anndata: anndata.AnnData,
    new_anndata: anndata.AnnData,
) -> None:
    """
    Check that the .var index of new_anndata matches exactly that of existing_anndata.

    Channel names must be identical in both content and order. A mismatch raises a
    detailed ValueError listing the channels that differ, so the user can diagnose
    whether the issue is a different instrument panel or a forgotten exclusion.

    Arguments:
        existing_anndata: The AnnData loaded from the existing .h5ad (or the first
                          per-file AnnData built in this run).
        new_anndata: The newly built AnnData to be appended.

    Returns:
        None

    Raises:
        ValueError: If .var indices differ, with a detailed list of discrepancies.
    """
    existing_channels = list(existing_anndata.var_names)
    new_channels = list(new_anndata.var_names)

    if existing_channels == new_channels:
        return  # Perfect match — nothing to do

    only_in_existing = sorted(set(existing_channels) - set(new_channels))
    only_in_new = sorted(set(new_channels) - set(existing_channels))

    error_lines = [
        "Channel names do not match between existing data and new file.",
        f"  Channels in existing data but NOT in new file ({len(only_in_existing)}): "
        f"{only_in_existing}",
        f"  Channels in new file but NOT in existing data ({len(only_in_new)}): "
        f"{only_in_new}",
        "Ensure that the same channels_to_exclude list is used for all files.",
    ]
    raise ValueError("\n".join(error_lines))


def _concatenate_anndata_objects(
    anndata_list: List[anndata.AnnData],
) -> anndata.AnnData:
    """
    Concatenate a list of AnnData objects along the observation axis.

    Uses anndata.concat() with join='inner' (keeps only channels present in all
    objects — since _validate_var_compatibility has already ensured they match,
    this is equivalent to an outer join). The obs index is reset to a clean integer
    range after concatenation to avoid duplicate index values across files.

    Arguments:
        anndata_list: Non-empty list of AnnData objects with identical .var indices.

    Returns:
        A single concatenated AnnData object with a clean integer obs index.
    """
    if len(anndata_list) == 1:
        # No need to concatenate — just reset the index
        single_anndata = anndata_list[0].copy()
        single_anndata.obs_names = [str(i) for i in range(single_anndata.n_obs)]
        return single_anndata

    combined = anndata.concat(
        anndata_list,
        join="inner",       # inner join on .var (channels must already match)
        merge="first",      # keep the first .uns value for each key; avoids errors
                            # when per-file keys (e.g. fcs_instrument_metadata) differ
        label=None,         # do not add a batch label column to .obs
    )

    # Reset obs index to a clean sequential integer range to avoid duplicate indices
    combined.obs_names = [str(i) for i in range(combined.n_obs)]

    return combined


def _discover_fcs_files(data_directory_path: str) -> List[str]:
    """
    Return a sorted list of absolute paths to all .fcs files in data_directory_path.

    Does not recurse into subdirectories. Matching is case-insensitive on the
    '.fcs' extension to handle both '.fcs' and '.FCS' files.

    Arguments:
        data_directory_path: Directory to scan.

    Returns:
        Sorted list of absolute .fcs file paths.

    Raises:
        FileNotFoundError: If data_directory_path does not exist.
        ValueError: If no .fcs files are found in the directory.
    """
    if not os.path.isdir(data_directory_path):
        raise FileNotFoundError(
            f"Data directory not found: {data_directory_path}"
        )

    all_entries = os.listdir(data_directory_path)
    fcs_filenames = sorted(
        entry for entry in all_entries
        if entry.lower().endswith(".fcs") and
        os.path.isfile(os.path.join(data_directory_path, entry))
    )

    if not fcs_filenames:
        raise ValueError(
            f"No .fcs files found in directory: {data_directory_path}"
        )

    return [os.path.abspath(os.path.join(data_directory_path, f)) for f in fcs_filenames]
