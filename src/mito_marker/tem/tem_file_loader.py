"""
tem_file_loader.py

Reads a single TEM measurement .txt file (tab-separated, one row per mitochondrion)
and returns:
  - The feature matrix as a float32 NumPy array (28 morphological features → .X)
  - The Image_Name string from the file (saved to .obs for traceability)
  - The Condition_Name string from the file (used as a double-check against the
    age group parsed from the filename — not saved to .obs itself)

Validation rules applied on load:
  - All columns listed in TEM_EXPECTED_FILE_COLUMNS must be present.
  - Mito_Area must be strictly positive for every row (areas cannot be zero or negative).
  - No NaN is allowed in any of the 28 feature columns.
  - Image_Name must be the same in every row of a single file.
  - Condition_Name must be the same in every row of a single file.
"""

import os
import warnings

import numpy as np
import pandas as pd

from mito_marker.controlled_vocabulary import (
    TEM_EXPECTED_FILE_COLUMNS,
    TEM_FEATURE_COLUMNS,
)


def load_tem_file(
    file_path: str,
) -> tuple[np.ndarray, str, str]:
    """
    Read a TEM measurement .txt file and return its morphological data.

    The file must be tab-separated with a header row. All columns listed in
    TEM_EXPECTED_FILE_COLUMNS must be present. Mito_Area must be > 0 for every
    row. No NaN is allowed in the 28 feature columns.

    Parameters
    ----------
    file_path : str
        Absolute or relative path to the _MITO_measurements.txt file.

    Returns
    -------
    tuple[np.ndarray, str, str]
        - feature_matrix : float32 ndarray of shape (n_mitochondria, 28)
        - image_name     : value of the Image_Name column (same for every row)
        - condition_name_from_file : value of the Condition_Name column (same
          for every row), used by the caller for a double-check against the
          filename-parsed age group
    """
    print(f"[TEM loader] Reading file: {os.path.basename(file_path)}")

    raw_dataframe = _read_tab_separated_file(file_path)
    _validate_required_columns(raw_dataframe, file_path)
    _validate_mito_area_positive(raw_dataframe, file_path)
    _validate_no_nan_in_features(raw_dataframe, file_path)

    image_name = _extract_single_value_column(
        raw_dataframe, "Image_Name", file_path
    )
    condition_name_from_file = _extract_single_value_column(
        raw_dataframe, "Condition_Name", file_path
    )

    feature_matrix = raw_dataframe[TEM_FEATURE_COLUMNS].to_numpy(dtype=np.float32)

    n_mitochondria = feature_matrix.shape[0]
    print(
        f"[TEM loader] Loaded {n_mitochondria} mitochondria, "
        f"{len(TEM_FEATURE_COLUMNS)} features. "
        f"Image: {image_name} | Condition: {condition_name_from_file}"
    )

    return feature_matrix, image_name, condition_name_from_file


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _read_tab_separated_file(file_path: str) -> pd.DataFrame:
    """
    Read the tab-separated .txt file into a DataFrame.

    Parameters
    ----------
    file_path : str
        Path to the file.

    Returns
    -------
    pd.DataFrame
        Raw DataFrame with all columns from the file.
    """
    return pd.read_csv(file_path, sep="\t")


def _validate_required_columns(
    raw_dataframe: pd.DataFrame, file_path: str
) -> None:
    """
    Assert that every column in TEM_EXPECTED_FILE_COLUMNS is present.

    Parameters
    ----------
    raw_dataframe : pd.DataFrame
        DataFrame loaded from the file.
    file_path : str
        Path used in the error message.

    Raises
    ------
    ValueError
        If one or more expected columns are missing.
    """
    missing_columns = [
        col for col in TEM_EXPECTED_FILE_COLUMNS if col not in raw_dataframe.columns
    ]
    if missing_columns:
        raise ValueError(
            f"File '{os.path.basename(file_path)}' is missing required columns: "
            f"{missing_columns}. "
            f"Expected all of: {TEM_EXPECTED_FILE_COLUMNS}"
        )


def _validate_mito_area_positive(
    raw_dataframe: pd.DataFrame, file_path: str
) -> None:
    """
    Assert that Mito_Area is strictly positive for every row.

    Negative or zero areas are physically impossible and indicate a corrupted
    measurement or a coordinate system error in the ImageJ output.

    Parameters
    ----------
    raw_dataframe : pd.DataFrame
        DataFrame loaded from the file.
    file_path : str
        Path used in the error message.

    Raises
    ------
    AssertionError
        If any Mito_Area value is <= 0.
    """
    invalid_rows = (raw_dataframe["Mito_Area"] <= 0).sum()
    assert invalid_rows == 0, (
        f"File '{os.path.basename(file_path)}': {invalid_rows} row(s) have "
        f"Mito_Area <= 0. All mitochondrial areas must be strictly positive."
    )


def _validate_no_nan_in_features(
    raw_dataframe: pd.DataFrame, file_path: str
) -> None:
    """
    Assert that no NaN values are present in any of the 28 feature columns.

    Parameters
    ----------
    raw_dataframe : pd.DataFrame
        DataFrame loaded from the file.
    file_path : str
        Path used in the error message.

    Raises
    ------
    AssertionError
        If any feature column contains NaN.
    """
    nan_counts = raw_dataframe[TEM_FEATURE_COLUMNS].isna().sum()
    columns_with_nan = nan_counts[nan_counts > 0]
    assert len(columns_with_nan) == 0, (
        f"File '{os.path.basename(file_path)}': NaN values found in feature "
        f"columns: {columns_with_nan.to_dict()}. All measurements must be present."
    )


def _extract_single_value_column(
    raw_dataframe: pd.DataFrame, column_name: str, file_path: str
) -> str:
    """
    Extract the unique value from a column that is expected to be the same
    for every row in the file.

    If multiple distinct values are found (which would indicate a file that
    mixes images), a warning is emitted and the first value is returned.

    Parameters
    ----------
    raw_dataframe : pd.DataFrame
        DataFrame loaded from the file.
    column_name : str
        Name of the column to extract.
    file_path : str
        Path used in the warning message.

    Returns
    -------
    str
        The (expected unique) value of that column.
    """
    unique_values = raw_dataframe[column_name].unique()
    if len(unique_values) > 1:
        warnings.warn(
            f"File '{os.path.basename(file_path)}': column '{column_name}' has "
            f"multiple distinct values {list(unique_values)}. "
            f"A single TEM file should describe a single image. "
            f"Using the first value: '{unique_values[0]}'.",
            stacklevel=3,
        )
    return str(unique_values[0])
