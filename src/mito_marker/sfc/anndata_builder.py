"""
anndata_builder.py

Assemble an AnnData object from parsed spectral flow cytometry data.

This module takes the outputs of filename_parser.py and fcs_loader.py and combines
them into a properly typed AnnData object where:
  - .X    holds the event × channel measurement matrix (float32)
  - .obs  holds per-event metadata extracted from the FCS filename
  - .var  holds per-channel metadata (channel names and descriptions)
  - .uns  holds instrument metadata extracted from the FCS TEXT segment
"""

from typing import List, Optional

import anndata
import numpy as np
import pandas as pd

from mito_marker.controlled_vocabulary import (
    ALLOWED_DIET_SFC,
    ALLOWED_DILUTION_SFC,
    ALLOWED_SPECIES_SFC,
    SFC_NON_ANALYTICAL_CHANNELS,
)

# Sentinel value for missing optional plain-string fields in .obs.
# Empty string "" is used because AnnData's h5py writer requires uniform string
# object arrays. np.nan (float) and all-None arrays both fail h5py's string encoder.
# An all-"" array is a valid uniform string array that h5py can encode correctly.
# Use pd.isna(value) or `value == ""` to test for absence downstream.
_MISSING_STRING_VALUE = ""


def build_sfc_anndata(
    obs_metadata: dict[str, str | bool | None],
    channel_names: List[str],
    channel_descriptions: List[str],
    event_matrix: np.ndarray,
    fcs_instrument_metadata: dict[str, str],
    fcs_obs_metadata: dict[str, str | None],
) -> anndata.AnnData:
    """
    Construct an AnnData object from parsed SFC data for a single FCS file.

    Each row in the resulting AnnData corresponds to one flow cytometry event (cell).
    All events from the same file share identical .obs metadata (extracted from the
    filename and from the FCS TEXT segment), which is broadcast across all rows.

    Structure:
      - .X    : float32 matrix, shape (n_events, n_channels)
      - .obs  : DataFrame with one row per event; includes both filename-parsed fields
                (specie, subject_ID, age, diet, dilution, marker, FlowAI_Pass,
                source_filename) and FCS file metadata fields (fcs_acquisition_date,
                fcs_tube_name, fcs_volume_uL, fcs_total_events)
      - .var  : DataFrame indexed by channel_names; column 'channel_description'
      - .uns['fcs_instrument_metadata'] : dict of FCS TEXT segment fields

    Arguments:
        obs_metadata: Dict from parse_fcs_filename() with keys: 'specie',
                      'subject_ID', 'age', 'diet', 'dilution', 'marker',
                      'FlowAI_Pass', 'source_filename'.
        channel_names: List of retained channel name strings (becomes .var index).
        channel_descriptions: List of channel description strings (same order).
        event_matrix: 2D float32 array of shape (n_events, n_channels).
        fcs_instrument_metadata: Dict from _extract_fcs_instrument_metadata() for .uns.
        fcs_obs_metadata: Dict from _extract_fcs_obs_metadata() with keys:
                          'fcs_acquisition_date', 'fcs_tube_name',
                          'fcs_volume_uL', 'fcs_total_events'.

    Returns:
        An anndata.AnnData object.
    """
    n_events = event_matrix.shape[0]

    observation_dataframe = _build_obs_dataframe(obs_metadata, fcs_obs_metadata, n_events)
    variable_dataframe = _build_var_dataframe(channel_names, channel_descriptions)

    sfc_anndata = anndata.AnnData(
        X=event_matrix,
        obs=observation_dataframe,
        var=variable_dataframe,
    )
    sfc_anndata.uns["fcs_instrument_metadata"] = fcs_instrument_metadata

    return sfc_anndata


def _build_obs_dataframe(
    obs_metadata: dict[str, str | bool | None],
    fcs_obs_metadata: dict[str, str | None],
    n_events: int,
) -> pd.DataFrame:
    """
    Broadcast per-file metadata dicts into a DataFrame with n_events rows.

    Combines filename-parsed metadata (obs_metadata) and FCS TEXT segment metadata
    (fcs_obs_metadata) into a single DataFrame. Each key becomes a column; the
    value is repeated for every event row since all events in a file share the
    same file-level attributes.

    Categorical fields (specie, diet, dilution, marker) are cast to pandas
    Categorical, using the full set of allowed values from controlled_vocabulary.py
    as the declared categories. This ensures that even if a given file only contains
    one specie, the category axis knows all possible species — important for
    downstream groupby and concat operations.

    FlowAI_Pass is stored as numpy bool (dtype=np.bool_). Optional fields (age, diet)
    that are None produce a column of np.nan in an object-dtype array. FCS metadata
    fields that are None (absent from the TEXT segment) also produce np.nan.

    All string columns use object dtype (not pd.StringDtype) because AnnData's h5py
    writer does not support pandas nullable string arrays. np.nan is the standard
    sentinel for missing values in object-dtype AnnData .obs columns.

    Arguments:
        obs_metadata: Dict from parse_fcs_filename() with filename-derived fields.
        fcs_obs_metadata: Dict from _extract_fcs_obs_metadata() with FCS TEXT fields.
        n_events: Number of rows to produce (one per flow cytometry event).

    Returns:
        pd.DataFrame with n_events rows and one column per obs field.
    """
    # Build a dict of column_name → array/Series of length n_events
    obs_columns: dict[str, pd.Series] = {}

    # --- Categorical fields with declared categories ---
    # pd.Categorical with string categories is supported by AnnData's h5py writer.
    obs_columns["specie"] = pd.Categorical(
        [obs_metadata["specie"]] * n_events,
        categories=list(ALLOWED_SPECIES_SFC.keys()),
    )

    obs_columns["dilution"] = pd.Categorical(
        [obs_metadata["dilution"]] * n_events,
        categories=list(ALLOWED_DILUTION_SFC.keys()),
    )

    # marker is stored as object dtype rather than pd.Categorical because combined
    # values (e.g. "TMRM+MtDeepRed") cannot be pre-enumerated as fixed categories.
    obs_columns["marker"] = np.array(
        [obs_metadata["marker"]] * n_events,
        dtype=object,
    )

    # --- Optional categorical field: diet ---
    # When absent from filename, store np.nan (not pd.NA) within the Categorical.
    # np.nan in a pd.Categorical is represented as a missing value, and is written
    # correctly by AnnData/h5py. pd.StringDtype (pd.NA) is NOT supported.
    diet_value: Optional[str] = obs_metadata.get("diet")  # type: ignore[assignment]
    obs_columns["diet"] = pd.Categorical(
        [diet_value] * n_events,  # None when absent; pd.Categorical stores None as code=-1
        categories=list(ALLOWED_DIET_SFC.keys()),
    )

    # --- Optional plain string fields: age_group and treatment ---
    # Stored as object dtype (not pd.Categorical) because these fields are absent for
    # non-mouse cohorts (MNMS, WORM, FLY) and None → "" sentinel in those cases.
    age_group_value: Optional[str] = obs_metadata.get("age_group")  # type: ignore[assignment]
    obs_columns["age_group"] = np.array(
        [age_group_value if age_group_value is not None else _MISSING_STRING_VALUE] * n_events,
        dtype=object,
    )

    treatment_value: Optional[str] = obs_metadata.get("treatment")  # type: ignore[assignment]
    obs_columns["treatment"] = np.array(
        [treatment_value if treatment_value is not None else _MISSING_STRING_VALUE] * n_events,
        dtype=object,
    )

    # --- Optional numeric field: age ---
    # _extract_age returns an int (the numeric part of the token, e.g. 3 from 'D03'),
    # or None when no age token is present in the filename.
    # Stored as float32 with np.nan for missing values — float32 supports NaN natively,
    # whereas numpy integer dtypes do not. h5py writes float32 arrays correctly.
    age_value: Optional[int] = obs_metadata.get("age")  # type: ignore[assignment]
    obs_columns["age"] = np.array(
        [float(age_value) if age_value is not None else np.nan] * n_events,
        dtype=np.float32,
    )

    # --- Boolean field: FlowAI_Pass ---
    # Regular numpy bool array — AnnData's h5py writer handles np.bool_ correctly.
    obs_columns["FlowAI_Pass"] = np.array(
        [bool(obs_metadata["FlowAI_Pass"])] * n_events, dtype=np.bool_
    )

    # --- Plain string fields from filename (object dtype for h5py compatibility) ---
    obs_columns["subject_ID"] = np.array(
        [obs_metadata["subject_ID"]] * n_events, dtype=object
    )
    obs_columns["source_filename"] = np.array(
        [obs_metadata["source_filename"]] * n_events, dtype=object
    )

    # --- FCS file metadata fields (from TEXT segment, object dtype) ---
    # Grouped under the 'fcs_' prefix per controlled_vocabulary.FCS_OBS_METADATA_FIELDS.
    # Values are None when absent from the instrument's TEXT segment → stored as np.nan.
    for fcs_field_name in ["fcs_acquisition_date", "fcs_tube_name",
                           "fcs_volume_uL", "fcs_total_events"]:
        field_value = fcs_obs_metadata.get(fcs_field_name, None)
        obs_columns[fcs_field_name] = np.array(
            [field_value if field_value is not None else _MISSING_STRING_VALUE] * n_events,
            dtype=object,
        )

    observation_dataframe = pd.DataFrame(obs_columns)
    return observation_dataframe


def _build_var_dataframe(
    channel_names: List[str],
    channel_descriptions: List[str],
) -> pd.DataFrame:
    """
    Build the .var DataFrame for an SFC AnnData object.

    The index is set to channel_names (e.g. 'FSC-A', 'B1-A', 'Time').
    Columns:
      - 'channel_description' : $PnS label from the FCS TEXT segment (may be empty).
      - 'is_non_analytical'   : True for channels that carry no biological information
                                and must be excluded from all analytical computations
                                (feature selection, PCA, UMAP, ML). Currently marks
                                'Time' and 'FlowAI'.  Analysis functions read this
                                column rather than hard-coding channel names.

    Arguments:
        channel_names: Channel name strings (becomes the .var index).
        channel_descriptions: Channel description strings (becomes 'channel_description').

    Returns:
        pd.DataFrame indexed by channel_names with columns 'channel_description'
        and 'is_non_analytical'.
    """
    non_analytical_set = set(SFC_NON_ANALYTICAL_CHANNELS)
    variable_dataframe = pd.DataFrame(
        {
            "channel_description": channel_descriptions,
            "is_non_analytical": [name in non_analytical_set for name in channel_names],
        },
        index=channel_names,
    )
    variable_dataframe.index.name = "channel_name"
    return variable_dataframe
