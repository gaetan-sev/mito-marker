"""
tem_ingestion.py

Entry point for TEM pipeline ingestion.

Discovers all *_MITO_measurements.txt files in a directory, processes each one
into an AnnData object, concatenates them into a single combined AnnData, runs a
QC block, and saves the result to an .h5ad file.

The pipeline mirrors ingest_sfc_folder from the SFC pipeline:
  - Supports appending new files to an existing .h5ad.
  - Validates that all files expose the same 28 feature columns.
  - Resets the .obs index to clean integers after concatenation.
  - Prints status messages at every step so a beginner can follow execution.

Two ingestion modes:

1. Standard mode (use_filename_for_metadata=True, default):
   Parses each filename to extract species, condition, and subject_ID.
   Use this for files whose names follow a known naming convention
   (KFish, ZFish, Mouse, MNMS/Human).

2. Override mode (use_filename_for_metadata=False):
   Skips filename parsing entirely. Use this when filenames carry no
   biological metadata (e.g. generic export names). In this mode you
   must supply the species via the `specie` argument; subject IDs are
   assigned automatically as "001", "002", … in sorted file order, and
   the condition is read from the Condition_Name column inside each file.

Usage (Google Colab or Codespaces):
    from mito_marker.tem import ingest_tem_folder

    # Standard mode
    tem_anndata = ingest_tem_folder(
        data_directory_path="data/raw/tem/",
        output_file_path="data/processed/tem_measurements.h5ad",
    )

    # Override mode (meaningless filenames)
    tem_anndata = ingest_tem_folder(
        data_directory_path="data/raw/tem/knockout/",
        output_file_path="data/processed/tem_ko.h5ad",
        specie="Mouse",
        use_filename_for_metadata=False,
    )
"""

import gc
import glob
import os
from typing import Optional

import anndata
import numpy as np
import pandas as pd

from mito_marker.controlled_vocabulary import ALLOWED_SPECIES_TEM
from mito_marker.tem.tem_anndata_builder import build_tem_anndata
from mito_marker.tem.tem_file_loader import load_tem_file
from mito_marker.tem.tem_filename_parser import parse_tem_filename


def ingest_tem_folder(
    data_directory_path: str,
    output_file_path: str,
    append_to_existing: bool = False,
    specie: Optional[str] = None,
    use_filename_for_metadata: bool = True,
) -> anndata.AnnData:
    """
    Ingest all TEM measurement files in a directory into a single AnnData.

    Discovers every file matching '*_MITO_measurements.txt' in
    data_directory_path (non-recursive), processes each file individually,
    then concatenates all per-file AnnData objects into one combined object.

    Parameters
    ----------
    data_directory_path : str
        Path to the directory containing *_MITO_measurements.txt files.
        In Google Colab: pass the mounted Google Drive path.
        In Codespaces: pass the local data/ directory path.
    output_file_path : str
        Path where the combined AnnData will be saved as an .h5ad file.
    append_to_existing : bool, optional
        If True and output_file_path already exists, load the existing AnnData
        and append the new files to it. Default is False.
    specie : Optional[str], optional
        Canonical species name to assign to all files when
        use_filename_for_metadata=False. Must be a key in ALLOWED_SPECIES_TEM.
        Ignored when use_filename_for_metadata=True. Default is None.
    use_filename_for_metadata : bool, optional
        If True (default), parse each filename to extract species, condition,
        and subject_ID using the standard token-based convention.
        If False, skip filename parsing entirely: the species is taken from
        the `specie` argument, subject IDs are assigned as "001", "002", …
        in sorted file order, and condition is read from Condition_Name
        inside each file.

    Returns
    -------
    anndata.AnnData
        Combined AnnData with all mitochondria from all processed files.
        .obs columns: specie, condition, subject_ID, unique_subject_ID,
                      Image_Name, source_filename
        .X: float32 matrix of shape (total_mitochondria, 28)

    Raises
    ------
    ValueError
        If use_filename_for_metadata=False and specie is None or not a
        recognised value in ALLOWED_SPECIES_TEM.
    FileNotFoundError
        If no *_MITO_measurements.txt files are found in data_directory_path.
    RuntimeError
        If all discovered files fail to process.
    """
    print("=" * 60)
    print("TEM INGESTION PIPELINE")
    print("=" * 60)
    print(f"Data directory          : {data_directory_path}")
    print(f"Output file             : {output_file_path}")
    print(f"Append mode             : {append_to_existing}")
    print(f"Use filename metadata   : {use_filename_for_metadata}")

    if not use_filename_for_metadata:
        if specie is None:
            raise ValueError(
                "use_filename_for_metadata=False requires a specie argument. "
                f"Allowed values: {list(ALLOWED_SPECIES_TEM.keys())}"
            )
        if specie not in ALLOWED_SPECIES_TEM:
            raise ValueError(
                f"Unknown specie '{specie}'. "
                f"Allowed values: {list(ALLOWED_SPECIES_TEM.keys())}. "
                f"Add the new species to ALLOWED_SPECIES_TEM in controlled_vocabulary.py "
                f"if it is genuinely new."
            )
        print(f"Override specie         : {specie}")

    txt_file_paths = _discover_tem_files(data_directory_path)
    if not txt_file_paths:
        raise FileNotFoundError(
            f"No *_MITO_measurements.txt files found in '{data_directory_path}'."
        )

    existing_anndata: Optional[anndata.AnnData] = None
    if append_to_existing and os.path.exists(output_file_path):
        print(f"\nLoading existing AnnData from '{output_file_path}' for append...")
        existing_anndata = anndata.read_h5ad(output_file_path)
        print(
            f"  Existing object: {existing_anndata.n_obs} mitochondria × "
            f"{existing_anndata.n_vars} features"
        )

    per_file_anndata_list: list[anndata.AnnData] = []
    successful_files: list[str] = []
    failed_files: list[str] = []

    for file_index, file_path in enumerate(txt_file_paths):
        filename = os.path.basename(file_path)
        print(f"\n{'─' * 40}")
        print(f"Processing: {filename}")
        try:
            if use_filename_for_metadata:
                per_file_anndata = _ingest_single_tem_file(file_path)
            else:
                per_file_anndata = _ingest_single_tem_file_no_filename_metadata(
                    file_path=file_path,
                    specie_override=specie,
                    file_index=file_index,
                )
            per_file_anndata_list.append(per_file_anndata)
            successful_files.append(filename)
        except Exception as processing_error:
            print(f"  [ERROR] Failed to process '{filename}': {processing_error}")
            failed_files.append(filename)

    if not per_file_anndata_list:
        raise RuntimeError(
            "All TEM files failed to process. "
            f"Failed files: {failed_files}"
        )

    # Add existing AnnData at the front of the list if appending
    if existing_anndata is not None:
        per_file_anndata_list.insert(0, existing_anndata)

    print(f"\n{'─' * 40}")
    print(f"Concatenating {len(per_file_anndata_list)} AnnData objects...")
    combined_anndata = _concatenate_anndata_objects(per_file_anndata_list)

    print("\n" + "=" * 60)
    print("QC SUMMARY — Combined AnnData")
    print("=" * 60)
    _print_anndata_qc(combined_anndata)

    print(f"\nSaving combined AnnData to '{output_file_path}'...")
    os.makedirs(os.path.dirname(os.path.abspath(output_file_path)), exist_ok=True)
    combined_anndata.write_h5ad(output_file_path)
    print("  Saved successfully.")

    print("\n" + "=" * 60)
    print("INGESTION COMPLETE")
    print(f"  Successful files : {len(successful_files)}")
    if failed_files:
        print(f"  Failed files     : {len(failed_files)}")
        for failed_filename in failed_files:
            print(f"    - {failed_filename}")
    print(f"  Total mitochondria: {combined_anndata.n_obs}")
    print(f"  Total features    : {combined_anndata.n_vars}")
    print("=" * 60)

    # Free per-file objects from memory before returning
    del per_file_anndata_list
    gc.collect()

    return combined_anndata


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _discover_tem_files(data_directory_path: str) -> list[str]:
    """
    Find all *_MITO_measurements.txt files in the given directory (non-recursive).

    Parameters
    ----------
    data_directory_path : str
        Path to the directory to search.

    Returns
    -------
    list[str]
        Sorted list of absolute file paths.
    """
    pattern = os.path.join(data_directory_path, "*_MITO_measurements.txt")
    found_paths = sorted(glob.glob(pattern))
    print(f"\nDiscovered {len(found_paths)} TEM measurement file(s):")
    for file_path in found_paths:
        print(f"  - {os.path.basename(file_path)}")
    return found_paths


def _ingest_single_tem_file(file_path: str) -> anndata.AnnData:
    """
    Process a single TEM .txt file using standard filename parsing.

    Extracts species, condition, and subject_ID from the filename, then
    loads the file and builds an AnnData object.

    Parameters
    ----------
    file_path : str
        Path to the *_MITO_measurements.txt file.

    Returns
    -------
    anndata.AnnData
        Per-file AnnData (n_mitochondria_in_file × 28).
    """
    filename = os.path.basename(file_path)

    # Step 1: parse the filename for biological metadata
    filename_metadata = parse_tem_filename(filename)
    print(
        f"  Filename metadata: "
        f"specie={filename_metadata['specie']} | "
        f"condition={filename_metadata['condition']} | "
        f"diet={filename_metadata['diet']} | "
        f"subject_ID={filename_metadata['subject_ID']}"
    )

    # Step 2: load the file and validate its contents
    feature_matrix, image_name, condition_name_from_file = load_tem_file(file_path)

    # Step 3: build AnnData
    per_file_anndata = build_tem_anndata(
        feature_matrix=feature_matrix,
        image_name=image_name,
        condition_name_from_file=condition_name_from_file,
        filename_metadata=filename_metadata,
        source_filename=filename,
        source_file_path=file_path,
    )

    return per_file_anndata


def _ingest_single_tem_file_no_filename_metadata(
    file_path: str,
    specie_override: str,
    file_index: int,
) -> anndata.AnnData:
    """
    Process a single TEM .txt file without parsing the filename.

    Used when filenames carry no biological metadata. The species is supplied
    by the caller; the subject_ID is assigned as a zero-padded integer based
    on the file's position in the sorted list; the condition is read from the
    Condition_Name column inside the file.

    Parameters
    ----------
    file_path : str
        Path to the *_MITO_measurements.txt file.
    specie_override : str
        Canonical species name provided by the caller (validated upstream).
    file_index : int
        Zero-based position of this file in the sorted file list, used to
        generate the subject_ID ("001", "002", …).

    Returns
    -------
    anndata.AnnData
        Per-file AnnData (n_mitochondria_in_file × 28).
    """
    filename = os.path.basename(file_path)
    subject_id = f"{file_index + 1:03d}"

    # Synthetic metadata: species from caller, condition=None (read from file
    # content by _resolve_condition), subject_ID auto-incremented.
    filename_metadata: dict[str, Optional[str]] = {
        "specie": specie_override,
        "condition": None,
        "diet": None,
        "subject_ID": subject_id,
    }
    print(
        f"  Override metadata: "
        f"specie={specie_override} | "
        f"condition=from_file | "
        f"subject_ID={subject_id}"
    )

    feature_matrix, image_name, condition_name_from_file = load_tem_file(file_path)

    per_file_anndata = build_tem_anndata(
        feature_matrix=feature_matrix,
        image_name=image_name,
        condition_name_from_file=condition_name_from_file,
        filename_metadata=filename_metadata,
        source_filename=filename,
        source_file_path=file_path,
    )

    return per_file_anndata


def _concatenate_anndata_objects(
    anndata_list: list[anndata.AnnData],
) -> anndata.AnnData:
    """
    Concatenate a list of per-file AnnData objects into one combined object.

    Validates that all objects share the same .var index (same 28 features).
    Resets the .obs index to clean integers after concatenation.

    Parameters
    ----------
    anndata_list : list[anndata.AnnData]
        Per-file AnnData objects to concatenate.

    Returns
    -------
    anndata.AnnData
        Combined AnnData with a clean integer .obs index.

    Raises
    ------
    ValueError
        If the .var index differs between any two files (feature column mismatch).
    """
    reference_var_index = anndata_list[0].var_names
    for anndata_object in anndata_list[1:]:
        if not reference_var_index.equals(anndata_object.var_names):
            raise ValueError(
                "Feature column mismatch between files. "
                "All TEM files must have the same 28 morphological features. "
                f"Expected: {list(reference_var_index)}, "
                f"Got: {list(anndata_object.var_names)}"
            )

    # merge="same" instructs anndata to preserve .var columns that are
    # identical across all objects (feature_description, is_non_analytical).
    # Without it, anndata.concat drops all .var columns silently — only
    # .var_names (the index) is kept by default.
    combined = anndata.concat(anndata_list, join="inner", merge="same")
    # Reset .obs index to clean integers (0, 1, 2, ...) for consistency
    combined.obs_names = pd.RangeIndex(combined.n_obs).astype(str)
    return combined


def _print_anndata_qc(combined_anndata: anndata.AnnData) -> None:
    """
    Print a QC summary for the combined AnnData object.

    Covers dimensions, .obs column preview, .var preview, NaN/Inf check
    in .X, and basic statistics of .X values.

    Parameters
    ----------
    combined_anndata : anndata.AnnData
        The combined AnnData object to audit.
    """
    print(f"Observations (mitochondria): {combined_anndata.n_obs}")
    print(f"Variables (features)        : {combined_anndata.n_vars}")

    print("\n.obs head (first 3 rows):")
    print(combined_anndata.obs.head(3).to_string())

    print("\n.var head (first 5 features):")
    print(combined_anndata.var.head(5).to_string())

    print("\n.obs value counts:")
    for column in ["specie", "condition", "diet"]:
        if column in combined_anndata.obs.columns:
            print(f"  {column}:")
            print(combined_anndata.obs[column].value_counts(dropna=False).to_string())

    feature_matrix = combined_anndata.X
    nan_count = int(np.isnan(feature_matrix).sum())
    inf_count = int(np.isinf(feature_matrix).sum())
    print(f"\n.X NaN values : {nan_count}")
    print(f".X Inf values : {inf_count}")
    print(f".X min        : {float(feature_matrix.min()):.4f}")
    print(f".X max        : {float(feature_matrix.max()):.4f}")
    print(f".X mean       : {float(feature_matrix.mean()):.4f}")

    print(f"\n.uns keys: {list(combined_anndata.uns.keys())}")
