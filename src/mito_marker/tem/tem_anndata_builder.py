"""
tem_anndata_builder.py

Assembles a per-file AnnData object from the outputs of tem_file_loader and
tem_filename_parser.

AnnData structure:
  .X    — float32 matrix, shape (n_mitochondria, 28), the morphological features.
           Never modified after creation. Normalized/transformed copies go in .layers.
  .obs  — one row per mitochondrion, columns:
            specie            : canonical species (from filename, categorical)
            condition         : canonical condition (from filename or file content, categorical)
            diet              : canonical diet value ("AL"/"IF"), or None if absent (categorical)
            subject_ID        : numeric subject identifier as a string
            unique_subject_ID : "{specie}_{subject_ID}", guaranteed unique across species
            Image_Name        : field-of-view identifier from the file itself
            source_filename   : basename of the source .txt file
  .var  — 28 rows indexed by feature name, one column 'feature_description' (empty)
  .uns['source_file'] — absolute path of the .txt file loaded

The 'condition' parsed from the filename is cross-checked against the
'Condition_Name' column read from inside the file. A mismatch emits a warning
but does not prevent the AnnData from being built — the filename value is always
used as the authoritative source for .obs when available.

When the filename carries no condition token (e.g. Human/MNMS files, or files
ingested with use_filename_for_metadata=False), the Condition_Name column inside
the file is used instead and validated against ALLOWED_CONDITIONS_TEM.
"""

import os
import warnings
from typing import Optional

import anndata
import numpy as np
import pandas as pd

from mito_marker.controlled_vocabulary import (
    ALLOWED_CONDITIONS_TEM,
    ALLOWED_DIET_TEM,
    ALLOWED_SPECIES_TEM,
    TEM_FEATURE_COLUMNS,
    TEM_NON_ANALYTICAL_FEATURES,
)


def build_tem_anndata(
    feature_matrix: np.ndarray,
    image_name: str,
    condition_name_from_file: str,
    filename_metadata: dict[str, Optional[str]],
    source_filename: str,
    source_file_path: str,
) -> anndata.AnnData:
    """
    Build an AnnData object for a single TEM measurement file.

    Parameters
    ----------
    feature_matrix : np.ndarray
        Float32 array of shape (n_mitochondria, 28). The values of .X.
    image_name : str
        Value of the Image_Name column from the file (same for every row).
    condition_name_from_file : str
        Value of the Condition_Name column from the file, used as fallback
        when filename_metadata["condition"] is None.
    filename_metadata : dict[str, Optional[str]]
        Output of parse_tem_filename or a synthetic dict for the
        use_filename_for_metadata=False mode.
        Keys: "specie", "condition", "diet", "subject_ID".
        Values are canonical strings or None when parsing failed / not applicable.
    source_filename : str
        Basename of the .txt file (stored in .obs for traceability).
    source_file_path : str
        Absolute path of the .txt file (stored in .uns).

    Returns
    -------
    anndata.AnnData
        AnnData with .X, .obs, .var, and .uns populated as described above.
    """
    n_mitochondria = feature_matrix.shape[0]

    resolved_condition = _resolve_condition(
        filename_metadata.get("condition"),
        condition_name_from_file,
        source_filename,
    )

    obs_dataframe = _build_obs_dataframe(
        specie=filename_metadata.get("specie"),
        condition=resolved_condition,
        diet=filename_metadata.get("diet"),
        subject_id=filename_metadata.get("subject_ID"),
        image_name=image_name,
        source_filename=source_filename,
        n_mitochondria=n_mitochondria,
    )
    var_dataframe = _build_var_dataframe()

    tem_anndata = anndata.AnnData(
        X=feature_matrix.copy(),
        obs=obs_dataframe,
        var=var_dataframe,
    )
    tem_anndata.uns["source_file"] = os.path.abspath(source_file_path)

    print(
        f"[TEM builder] AnnData built: {tem_anndata.n_obs} mitochondria × "
        f"{tem_anndata.n_vars} features | "
        f"specie={filename_metadata.get('specie')} | "
        f"condition={resolved_condition} | "
        f"diet={filename_metadata.get('diet')} | "
        f"subject_ID={filename_metadata.get('subject_ID')}"
    )

    return tem_anndata


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _resolve_condition(
    condition_from_filename: Optional[str],
    condition_name_from_file: str,
    source_filename: str,
) -> Optional[str]:
    """
    Determine the authoritative condition value to store in .obs.

    Two cases:
    - Filename provides a condition token (e.g. KFish, ZFish, Mouse files):
      use that value and cross-check it against Condition_Name in the file.
      A mismatch emits a warning — the filename value always wins.
    - Filename provides no condition token (e.g. Human/MNMS files, or files
      ingested with use_filename_for_metadata=False):
      use Condition_Name from inside the file. If that value is not in
      ALLOWED_CONDITIONS_TEM, emit a warning and return None.

    Parameters
    ----------
    condition_from_filename : Optional[str]
        Canonical condition extracted from the filename, or None if the parser
        found no condition token (expected for Human/MNMS files and for the
        use_filename_for_metadata=False ingestion mode).
    condition_name_from_file : str
        Raw Condition_Name value read from the file's data rows.
    source_filename : str
        Basename of the source file, used in warning messages.

    Returns
    -------
    Optional[str]
        Resolved condition canonical string, or None if it cannot be determined.
    """
    if condition_from_filename is not None:
        # Filename wins — cross-check against the file content for consistency.
        if condition_name_from_file != condition_from_filename:
            warnings.warn(
                f"Condition mismatch in '{source_filename}': "
                f"filename parser found '{condition_from_filename}' but the file's "
                f"Condition_Name column says '{condition_name_from_file}'. "
                f"The filename value ('{condition_from_filename}') will be used in .obs. "
                f"Check that the filename and the ImageJ export are consistent.",
                stacklevel=3,
            )
        return condition_from_filename

    # No condition token in filename — use the file's Condition_Name column.
    if condition_name_from_file in ALLOWED_CONDITIONS_TEM:
        print(
            f"[TEM builder] condition not in filename — "
            f"using Condition_Name from file: '{condition_name_from_file}'"
        )
        return condition_name_from_file

    warnings.warn(
        f"'{source_filename}': Condition_Name in file is '{condition_name_from_file}', "
        f"which is not in ALLOWED_CONDITIONS_TEM "
        f"(allowed: {list(ALLOWED_CONDITIONS_TEM.keys())}). "
        f"condition will be set to None.",
        stacklevel=3,
    )
    return None


def _build_obs_dataframe(
    specie: Optional[str],
    condition: Optional[str],
    diet: Optional[str],
    subject_id: Optional[str],
    image_name: str,
    source_filename: str,
    n_mitochondria: int,
) -> pd.DataFrame:
    """
    Build the .obs DataFrame, one row per mitochondrion.

    Categorical columns (specie, condition, diet) are declared with the full
    category axis from controlled_vocabulary so that files from different
    experiments can be concatenated without category mismatches.

    Parameters
    ----------
    specie : Optional[str]
        Canonical species name, or None if parsing failed.
    condition : Optional[str]
        Resolved canonical condition (from filename or file content), or None.
    diet : Optional[str]
        Canonical diet value ("AL" or "IF"), or None if absent from the filename.
    subject_id : Optional[str]
        Subject ID string, or None if parsing failed.
        A ``unique_subject_ID`` column is derived as ``"{specie}_{subject_id}"``
        so that subjects from different species with the same short ID remain
        distinguishable after merging.
    image_name : str
        Image_Name value broadcast to all rows.
    source_filename : str
        Source basename broadcast to all rows.
    n_mitochondria : int
        Number of rows to create.

    Returns
    -------
    pd.DataFrame
        DataFrame indexed by integers 0..n_mitochondria-1.
    """
    specie_value = specie
    condition_value = condition
    diet_value = diet
    subject_id_value = subject_id or ""

    # Categorical columns: declare the full category axis so that concat works
    # across files from different species/conditions without a join='outer' mismatch.
    specie_series = pd.Categorical(
        [specie_value] * n_mitochondria,
        categories=list(ALLOWED_SPECIES_TEM.keys()),
    )
    condition_series = pd.Categorical(
        [condition_value] * n_mitochondria,
        categories=list(ALLOWED_CONDITIONS_TEM.keys()),
    )
    diet_series = pd.Categorical(
        [diet_value] * n_mitochondria,
        categories=list(ALLOWED_DIET_TEM.keys()),
    )

    # Build a cross-species unique subject identifier so that subjects with the
    # same short numeric ID (e.g. Mouse "1" vs Fly "1") remain distinguishable
    # when multiple per-species AnnData objects are merged in a notebook.
    specie_part = specie_value if specie_value else "Unknown"
    id_part = subject_id_value if subject_id_value else "Unknown"
    unique_subject_id_value = f"{specie_part}_{id_part}"

    obs_dataframe = pd.DataFrame(
        {
            "specie": specie_series,
            "condition": condition_series,
            "diet": diet_series,
            "subject_ID": [subject_id_value] * n_mitochondria,
            "unique_subject_ID": [unique_subject_id_value] * n_mitochondria,
            "Image_Name": [image_name] * n_mitochondria,
            "source_filename": [source_filename] * n_mitochondria,
        },
        index=pd.RangeIndex(n_mitochondria),
    )
    return obs_dataframe


def _build_var_dataframe() -> pd.DataFrame:
    """
    Build the .var DataFrame, one row per morphological feature.

    Parameters
    ----------
    None

    Returns
    -------
    pd.DataFrame
        DataFrame indexed by TEM_FEATURE_COLUMNS with one column
        'feature_description' (empty strings — no descriptions available
        from the ImageJ output).
    """
    non_analytical_set = set(TEM_NON_ANALYTICAL_FEATURES)
    var_dataframe = pd.DataFrame(
        {
            "feature_description": [""] * len(TEM_FEATURE_COLUMNS),
            "is_non_analytical": [
                col in non_analytical_set for col in TEM_FEATURE_COLUMNS
            ],
        },
        index=pd.Index(TEM_FEATURE_COLUMNS, name="feature_name"),
    )
    return var_dataframe
