"""
fcs_loader.py

Load a single FCS (Flow Cytometry Standard) file using FlowIO and return its
channel metadata and event data matrix.

FlowIO reads the binary FCS file and exposes:
- .text          : dict of all TEXT segment key/value pairs (e.g. '$PAR', '$TOT', '$P1N')
- .event_count   : number of events (rows)
- .channel_count : number of channels (columns)
- .events        : flat 1D numpy array of all event values (length = event_count * channel_count)

This module reshapes that flat array into a 2D matrix (events × channels), extracts
channel names ($PnN) and optional descriptions ($PnS), and applies any user-requested
channel exclusions.
"""

import os
from typing import List, Optional

import flowio
import numpy as np

from mito_marker.controlled_vocabulary import SFC_CHANNELS_TO_EXCLUDE, SFC_CHANNELS_TO_KEEP


def load_fcs_file(
    fcs_file_path: str,
    channels_to_exclude: Optional[List[str]] = None,
) -> tuple[List[str], List[str], np.ndarray, dict[str, str], dict[str, str | None]]:
    """
    Load a single FCS file using FlowIO and return channel metadata + event matrix.

    Opens the file with flowio.FlowData, extracts channel names ($PnN) and channel
    descriptions ($PnS) from the TEXT segment, reshapes the flat event array into a
    2D matrix, applies any requested channel exclusions, reorders the retained channels
    into canonical spectral order (FSC/SSC → UV → V → B → YG → R, as declared in
    SFC_CHANNELS_TO_KEEP), and extracts both instrument metadata (for .uns) and
    per-file obs metadata (for .obs).

    Arguments:
        fcs_file_path: Absolute or relative path to the .fcs file.
        channels_to_exclude: Optional list of channel name strings to drop from the
                             output (e.g. ['Time', 'FlowAI_RG']). Matching is
                             case-sensitive. Unrecognised names are ignored with a warning.

    Returns:
        A 5-tuple:
          - channel_names: List[str] of retained channel names (from $PnN).
          - channel_descriptions: List[str] of retained channel descriptions
                                  (from $PnS; empty string '' if the key is absent).
          - event_matrix: np.ndarray of shape (n_events, n_retained_channels),
                          dtype float32.
          - instrument_metadata: dict[str, str] of selected FCS TEXT fields for .uns.
          - fcs_obs_metadata: dict[str, str | None] with per-file metadata for .obs:
                              'fcs_acquisition_date', 'fcs_tube_name',
                              'fcs_volume_uL', 'fcs_total_events'.
                              Values are None when the TEXT field is absent.

    Raises:
        FileNotFoundError: If fcs_file_path does not exist.
        ValueError: If the file is not a valid FCS file or contains zero events.
    """
    if not os.path.exists(fcs_file_path):
        raise FileNotFoundError(f"FCS file not found: {fcs_file_path}")

    fcs_data = flowio.FlowData(fcs_file_path)

    if fcs_data.event_count == 0:
        raise ValueError(f"FCS file contains zero events: {fcs_file_path}")

    channel_count = fcs_data.channel_count
    fcs_text = fcs_data.text

    all_channel_names = _extract_channel_names(fcs_text, channel_count)
    all_channel_descriptions = _extract_channel_descriptions(fcs_text, channel_count)

    # Warn about channels that are unknown to the controlled vocabulary.
    _warn_unlisted_channels(all_channel_names, fcs_file_path)

    event_matrix = _reshape_event_array(
        np.array(fcs_data.events),
        fcs_data.event_count,
        channel_count,
    )
    instrument_metadata = _extract_fcs_instrument_metadata(fcs_text)
    fcs_obs_metadata = _extract_fcs_obs_metadata(fcs_text)

    if channels_to_exclude:
        channel_names, channel_descriptions, event_matrix = _apply_channel_exclusions(
            all_channel_names,
            all_channel_descriptions,
            event_matrix,
            channels_to_exclude,
        )
    else:
        channel_names = all_channel_names
        channel_descriptions = all_channel_descriptions

    # Always reorder into canonical spectral order regardless of which channels were excluded.
    # This guarantees that .X columns are in the same biological order across all files.
    channel_names, channel_descriptions, event_matrix = _reorder_channels_by_spectral_group(
        channel_names, channel_descriptions, event_matrix
    )

    return channel_names, channel_descriptions, event_matrix, instrument_metadata, fcs_obs_metadata


def _extract_channel_names(fcs_text: dict[str, str], channel_count: int) -> List[str]:
    """
    Extract ordered list of channel names from the FCS TEXT segment dictionary.

    Reads keys p1n, p2n, ..., p{channel_count}n from fcs_text.

    FlowIO normalises all FCS TEXT segment keys by stripping the leading '$' and
    converting to lowercase before storing them. So the FCS standard key '$P1N'
    becomes 'p1n', '$P208N' becomes 'p208n', etc. This function uses the FlowIO
    key format directly.

    Returns an empty string for any missing key (the FCS 3.1 spec makes $PnN
    optional, though in practice it is always present).

    Arguments:
        fcs_text: The FCS TEXT segment as a dict (from flowio.FlowData.text).
        channel_count: Total number of channels ($PAR value).

    Returns:
        List of channel name strings, length == channel_count.
    """
    channel_names = []
    for channel_index in range(1, channel_count + 1):
        # FlowIO stores '$P1N' as 'p1n' (stripped '$', lowercased)
        name_key = f"p{channel_index}n"
        channel_names.append(fcs_text.get(name_key, ""))
    return channel_names


def _extract_channel_descriptions(fcs_text: dict[str, str], channel_count: int) -> List[str]:
    """
    Extract ordered list of channel descriptions from the FCS TEXT segment dictionary.

    Reads keys p1s, p2s, ..., p{channel_count}s from fcs_text.

    FlowIO normalises all FCS TEXT segment keys by stripping the leading '$' and
    converting to lowercase. So '$P1S' becomes 'p1s', '$P208S' becomes 'p208s'.

    $PnS is optional in FCS 3.1 — returns empty string '' for any missing key.
    Many instruments (including the Cytek Aurora used for the project fixtures) do
    not populate $PnS, so empty strings are the expected common case.

    Arguments:
        fcs_text: The FCS TEXT segment as a dict (from flowio.FlowData.text).
        channel_count: Total number of channels ($PAR value).

    Returns:
        List of channel description strings, length == channel_count.
    """
    channel_descriptions = []
    for channel_index in range(1, channel_count + 1):
        # FlowIO stores '$P1S' as 'p1s' (stripped '$', lowercased)
        description_key = f"p{channel_index}s"
        channel_descriptions.append(fcs_text.get(description_key, ""))
    return channel_descriptions


def _reshape_event_array(
    flat_events: np.ndarray,
    event_count: int,
    channel_count: int,
) -> np.ndarray:
    """
    Reshape the flat FlowIO event array into a 2D matrix and cast to float32.

    FlowIO returns events as a flat 1D array of length event_count * channel_count,
    ordered as [event_0_ch_0, event_0_ch_1, ..., event_0_ch_n, event_1_ch_0, ...].
    Reshaping to (event_count, channel_count) gives the standard (observations × variables)
    orientation used by AnnData.

    float32 is used instead of float64 to reduce memory usage, which is important
    when running in Google Colab where RAM is limited.

    Arguments:
        flat_events: 1D numpy array from flowio.FlowData.events.
        event_count: Number of events ($TOT value from flowio.FlowData.event_count).
        channel_count: Number of channels ($PAR value from flowio.FlowData.channel_count).

    Returns:
        np.ndarray of shape (event_count, channel_count), dtype float32.
    """
    return flat_events.reshape(event_count, channel_count).astype(np.float32)


def _apply_channel_exclusions(
    channel_names: List[str],
    channel_descriptions: List[str],
    event_matrix: np.ndarray,
    channels_to_exclude: List[str],
) -> tuple[List[str], List[str], np.ndarray]:
    """
    Remove columns corresponding to the specified channel names from the event matrix.

    Channels are matched by their name string (case-sensitive). Any name in
    channels_to_exclude that does not match a channel in the file is ignored
    with a printed warning — this prevents hard failures when the exclusion list
    is reused across files with slightly different channel sets.

    Arguments:
        channel_names: Full list of channel names before exclusion.
        channel_descriptions: Full list of channel descriptions (same order).
        event_matrix: 2D float32 array of shape (n_events, n_channels).
        channels_to_exclude: List of channel names to drop.

    Returns:
        3-tuple: (filtered_names, filtered_descriptions, filtered_matrix)
    """
    exclusion_set = set(channels_to_exclude)

    # Warn about any requested exclusions that do not match any channel
    unrecognised_exclusions = exclusion_set - set(channel_names)
    if unrecognised_exclusions:
        print(
            f"  WARNING: The following channels_to_exclude were not found in this "
            f"FCS file and will be ignored: {sorted(unrecognised_exclusions)}"
        )

    # Build a boolean mask: True = keep this channel, False = exclude
    columns_to_keep = [name not in exclusion_set for name in channel_names]
    keep_indices = [i for i, keep in enumerate(columns_to_keep) if keep]

    filtered_names = [channel_names[i] for i in keep_indices]
    filtered_descriptions = [channel_descriptions[i] for i in keep_indices]
    filtered_matrix = event_matrix[:, keep_indices]

    return filtered_names, filtered_descriptions, filtered_matrix


def _warn_unlisted_channels(channel_names: List[str], fcs_file_path: str) -> None:
    """
    Print a warning for any channel present in the FCS file but absent from both
    SFC_CHANNELS_TO_KEEP and SFC_CHANNELS_TO_EXCLUDE in controlled_vocabulary.py.

    A channel absent from both lists is genuinely unknown — it was not anticipated
    when the controlled vocabulary was written. Such a channel will be RETAINED in
    the output (it is not in the exclusion list), which means it will appear as a
    column in .X and in the final AnnData. The user should decide whether to add it
    to SFC_CHANNELS_TO_KEEP (to include it in the canonical order) or to
    SFC_CHANNELS_TO_EXCLUDE (to drop it silently in future runs).

    The check is performed on the raw full channel list, before any exclusions, so
    every channel the instrument produced is inspected regardless of what the caller
    requested to exclude.

    Arguments:
        channel_names: Full list of channel names read directly from the FCS file.
        fcs_file_path: Path to the FCS file, used in the warning message for traceability.

    Returns:
        None
    """
    import os
    known_channels: set[str] = set(SFC_CHANNELS_TO_KEEP) | set(SFC_CHANNELS_TO_EXCLUDE)
    unlisted_channels = sorted(set(channel_names) - known_channels)

    if unlisted_channels:
        filename = os.path.basename(fcs_file_path)
        print(
            f"  WARNING: {len(unlisted_channels)} channel(s) in '{filename}' are not listed "
            f"in SFC_CHANNELS_TO_KEEP or SFC_CHANNELS_TO_EXCLUDE:\n"
            f"    {unlisted_channels}\n"
            f"  These channels will be RETAINED in .X as-is (they are not excluded).\n"
            f"  To suppress this warning, add each channel to the appropriate list "
            f"in controlled_vocabulary.py."
        )


def _reorder_channels_by_spectral_group(
    channel_names: List[str],
    channel_descriptions: List[str],
    event_matrix: np.ndarray,
) -> tuple[List[str], List[str], np.ndarray]:
    """
    Reorder channels into canonical spectral order using SFC_CHANNELS_TO_KEEP as reference.

    SFC_CHANNELS_TO_KEEP in controlled_vocabulary.py lists all expected channels in the
    exact biological order: FSC/SSC → UV → V → B → YG → R → quality/timing channels.
    This function sorts the retained channels so their column position in the output
    matches their position in that canonical list.

    Channels absent from SFC_CHANNELS_TO_KEEP (e.g. unexpected instrument channels) are
    appended at the tail in their original relative order — they are not silently dropped.

    This guarantees that .X column 0 is always FSC-A, column 1 is always FSC-H, etc.,
    regardless of how the instrument or FlowJo exported the file.

    Arguments:
        channel_names: Channel names after exclusions have been applied.
        channel_descriptions: Corresponding descriptions (same order as channel_names).
        event_matrix: 2D float32 array of shape (n_events, n_channels).

    Returns:
        3-tuple: (reordered_names, reordered_descriptions, reordered_matrix),
                 where column order follows SFC_CHANNELS_TO_KEEP.
    """
    # Build a lookup: channel_name → position in the canonical list.
    canonical_position: dict[str, int] = {
        name: position for position, name in enumerate(SFC_CHANNELS_TO_KEEP)
    }
    n_canonical = len(SFC_CHANNELS_TO_KEEP)

    # Assign each channel a sort key.
    # Channels in the canonical list: sort key = their canonical position (0, 1, 2, …).
    # Channels NOT in the canonical list: sort key = n_canonical + original_index,
    # which places them after all canonical channels while preserving their mutual order.
    def sort_key(index_and_name: tuple[int, str]) -> int:
        original_index, name = index_and_name
        return canonical_position.get(name, n_canonical + original_index)

    sorted_indexed = sorted(enumerate(channel_names), key=sort_key)
    sorted_indices = [original_index for original_index, _ in sorted_indexed]

    reordered_names = [channel_names[i] for i in sorted_indices]
    reordered_descriptions = [channel_descriptions[i] for i in sorted_indices]
    reordered_matrix = event_matrix[:, sorted_indices]

    return reordered_names, reordered_descriptions, reordered_matrix


def _extract_fcs_instrument_metadata(fcs_text: dict[str, str]) -> dict[str, str]:
    """
    Extract instrument and acquisition metadata from the FCS TEXT segment for .uns storage.

    Copies a fixed set of informational keys from the TEXT segment. Missing keys are
    silently skipped — not all instruments populate all fields. The result is stored
    under AnnData.uns['fcs_instrument_metadata'] for traceability.

    Arguments:
        fcs_text: Full FCS TEXT segment dict from flowio.FlowData.text.

    Returns:
        Dict mapping TEXT key → value (all strings). Only keys that are present in
        fcs_text are included.
    """
    # FlowIO normalises all FCS TEXT keys: strips leading '$' and lowercases everything.
    # So '$CYT' → 'cyt', '$DATE' → 'date', 'APPLY COMPENSATION' → 'apply compensation'.
    keys_of_interest = [
        "cyt",               # Cytometer type (instrument model name) — from $CYT
        "cytsn",             # Cytometer serial number — from $CYTSN
        "inst",              # Institution — from $INST
        "date",              # Acquisition date — from $DATE
        "btim",              # Begin time of acquisition — from $BTIM
        "etim",              # End time of acquisition — from $ETIM
        "op",                # Operator name — from $OP
        "mode",              # Acquisition mode — from $MODE
        "byteord",           # Byte order — from $BYTEORD
        "datatype",          # Data type — from $DATATYPE
        "tot",               # Total number of events — from $TOT
        "par",               # Number of parameters (channels) — from $PAR
        "fj_fcs_version",    # FlowJo internal FCS version tag
        "timestep",          # Time resolution (seconds per time unit) — from $TIMESTEP
        "apply compensation",    # Whether compensation was applied (Cytek tag)
        "laser1name",        # Laser names (Cytek Aurora vendor extension)
        "laser2name",
        "laser3name",
        "laser4name",
        "laser5name",
    ]

    instrument_metadata = {}
    for key in keys_of_interest:
        if key in fcs_text:
            instrument_metadata[key] = fcs_text[key]

    return instrument_metadata


def _extract_fcs_obs_metadata(fcs_text: dict[str, str]) -> dict[str, str | None]:
    """
    Extract per-file metadata from the FCS TEXT segment for storage in AnnData .obs.

    These four fields describe the acquisition file as a whole and are broadcast to
    every event row (same value for all events from the same file). They are stored
    as separate .obs columns, named with the 'fcs_' prefix to make their origin clear.

    Fields extracted:
      - 'fcs_acquisition_date' : from $DATE  (acquisition date)
      - 'fcs_tube_name'        : from TUBENAME (vendor extension, Cytek Aurora)
      - 'fcs_volume_uL'        : from $Vol (volume acquired in µL)
      - 'fcs_total_events'     : from $TOT (total events in the file)

    All values are returned as strings. None is returned for any field absent from
    the FCS TEXT segment.

    Arguments:
        fcs_text: Full FCS TEXT segment dict from flowio.FlowData.text.

    Returns:
        Dict with exactly four keys: 'fcs_acquisition_date', 'fcs_tube_name',
        'fcs_volume_uL', 'fcs_total_events'. Values are str or None.
    """
    # FlowIO normalises all FCS TEXT keys: strips all '$' chars and lowercases.
    # '$DATE' → 'date', 'TUBENAME' → 'tubename', '$Vol' → 'vol', '$TOT' → 'tot'
    return {
        "fcs_acquisition_date": fcs_text.get("date", None),
        "fcs_tube_name": fcs_text.get("tubename", None),
        "fcs_volume_uL": fcs_text.get("vol", None),
        "fcs_total_events": fcs_text.get("tot", None),
    }
