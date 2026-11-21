"""
fcs_io.py

Read every channel of an .fcs file as it is, and write .fcs files that FlowJo can
reopen like its own exports.

WHY A SECOND FCS READER

`load_fcs_file()` (fcs_loader.py) prepares the channels for analysis: it drops
the FlowJo compensation duplicates and the non-analytical channels and reorders
the rest. The preprocessing steps (flowAI, P1, DR+; mito-marker-lab, EXP-067)
need the opposite: every channel in its original order, the original TEXT
keywords (channel ranges, acquisition date, time step), and the events in their
acquisition order, so that the file written after a step is the same file as
before, minus the removed events.

WHAT IS KEPT WHEN WRITING

- Every channel and its values (32-bit float, as written by the Aurora).
- Every TEXT keyword of the source file except the structural ones (data
  offsets, number of events and channels, byte order, per-channel bit
  widths), which FlowIO computes again. Channel ranges ($PnR) and labels
  ($PnS) are kept, so FlowJo shows the same axes.
- `$FIL` is set to the new file name, as FlowJo does for its exports.

Extra channels (e.g. the `FlowAI` label channel) are appended after the
original ones, so the original channel numbers do not change.
"""

import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import flowio
import numpy as np

# TEXT keywords that describe the layout of the file. FlowIO writes them again
# from the data, so the values of the source file must not be copied.
FCS_STRUCTURAL_KEYWORDS: set[str] = {
    "begindata", "enddata", "beginanalysis", "endanalysis", "beginstext", "endstext",
    "mode", "byteord", "datatype", "par", "tot", "nextdata", "fil",
}

# Per-channel keywords written again by FlowIO ($PnB bits, $PnE log decades,
# $PnN name, $PnS label). $PnR (range) and $PnG (gain) are passed through.
FCS_PER_CHANNEL_REWRITTEN_PATTERN = re.compile(r"^p\d+[bens]$")


@dataclass
class FcsEvents:
    """Events and keywords of one .fcs file, every channel in its original order.

    Attributes:
        event_matrix: Event values, shape (n_events, n_channels), float64.
        channel_names: $PnN of each channel, in file order.
        channel_descriptions: $PnS of each channel ('' when absent).
        text_keywords: Every TEXT keyword of the file, lower-case names without '$'
            (FlowIO convention).
        source_file_path: Path of the file the events were read from.
    """

    event_matrix: np.ndarray
    channel_names: List[str]
    channel_descriptions: List[str]
    text_keywords: Dict[str, str] = field(default_factory=dict)
    source_file_path: str = ""

    @property
    def n_events(self) -> int:
        """Number of events (rows)."""
        return int(self.event_matrix.shape[0])

    def channel_values(self, channel_name: str) -> np.ndarray:
        """Return the values of one channel; raises KeyError with the available names."""
        if channel_name not in self.channel_names:
            raise KeyError(
                f"Channel '{channel_name}' not found in {os.path.basename(self.source_file_path)}. "
                f"Available: {self.channel_names}"
            )
        return self.event_matrix[:, self.channel_names.index(channel_name)]

    def channel_ranges(self) -> np.ndarray:
        """$PnR of each channel as floats (NaN when the keyword is missing)."""
        ranges = []
        for channel_number in range(1, len(self.channel_names) + 1):
            value = self.text_keywords.get(f"p{channel_number}r", "")
            ranges.append(float(value) if value.strip() else np.nan)
        return np.array(ranges)


def read_fcs_events(fcs_file_path: str) -> FcsEvents:
    """Read every channel of an .fcs file in its original order, with its TEXT keywords.

    Args:
        fcs_file_path: Path to the .fcs file.

    Returns:
        FcsEvents with float64 values (no channel excluded, no reordering, no scaling).

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file is empty or its header cannot be read.
    """
    if not os.path.isfile(fcs_file_path):
        raise FileNotFoundError(f"FCS file not found: {fcs_file_path}")
    if os.path.getsize(fcs_file_path) == 0:
        raise ValueError(f"FCS file is empty (0 bytes): {fcs_file_path}")
    try:
        flow_data = flowio.FlowData(fcs_file_path, ignore_offset_error=True)
    except ValueError as read_error:
        raise ValueError(f"FCS header cannot be read: {fcs_file_path} ({read_error})") from read_error
    channel_count = flow_data.channel_count
    channel_names = [flow_data.text[f"p{number}n"] for number in range(1, channel_count + 1)]
    channel_descriptions = [flow_data.text.get(f"p{number}s", "") for number in range(1, channel_count + 1)]
    event_matrix = flow_data.as_array(preprocess=False).astype(np.float64)
    return FcsEvents(
        event_matrix=event_matrix,
        channel_names=channel_names,
        channel_descriptions=channel_descriptions,
        text_keywords=dict(flow_data.text),
        source_file_path=fcs_file_path,
    )


def write_fcs_events(
    fcs_events: FcsEvents,
    output_file_path: str,
    event_mask: Optional[np.ndarray] = None,
    extra_channels: Optional[Dict[str, np.ndarray]] = None,
    extra_channel_range: float = 20000.0,
) -> str:
    """Write events to a new .fcs file that FlowJo can open like one of its exports.

    Args:
        fcs_events: Source events and keywords (from read_fcs_events()).
        output_file_path: Path of the file to write (its folder must exist).
        event_mask: Optional boolean array (n_events,): only True events are written,
            in their original order. None writes every event.
        extra_channels: Optional {channel name: values (n_events,)} appended after the
            original channels, e.g. {"FlowAI": labels}. Values are masked like the events.
        extra_channel_range: $PnR written for the extra channels. 20000 is the range
            the FlowJo flowAI plugin gives its label channel.

    Returns:
        The path of the written file.
    """
    if event_mask is None:
        event_mask = np.ones(fcs_events.n_events, dtype=bool)
    event_mask = np.asarray(event_mask, dtype=bool)
    if event_mask.shape != (fcs_events.n_events,):
        raise ValueError(f"event_mask has shape {event_mask.shape}, expected ({fcs_events.n_events},)")

    channel_names = list(fcs_events.channel_names)
    channel_descriptions = list(fcs_events.channel_descriptions)
    columns = [fcs_events.event_matrix[event_mask]]
    for extra_name, extra_values in (extra_channels or {}).items():
        extra_values = np.asarray(extra_values, dtype=np.float64)
        if extra_values.shape != (fcs_events.n_events,):
            raise ValueError(f"Extra channel '{extra_name}' has shape {extra_values.shape}, expected ({fcs_events.n_events},)")
        if extra_name in channel_names:
            raise ValueError(f"Extra channel '{extra_name}' already exists in the file")
        channel_names.append(extra_name)
        channel_descriptions.append("")
        columns.append(extra_values[event_mask][:, None])
    output_matrix = np.hstack(columns).astype(np.float32)

    metadata = {}
    for keyword, value in fcs_events.text_keywords.items():
        # FCS does not allow empty values, and the structural keywords are rebuilt by FlowIO.
        if keyword in FCS_STRUCTURAL_KEYWORDS or FCS_PER_CHANNEL_REWRITTEN_PATTERN.match(keyword):
            continue
        if not str(value).strip():
            continue
        metadata[keyword] = str(value).strip()
    for channel_number in range(len(fcs_events.channel_names) + 1, len(channel_names) + 1):
        metadata[f"p{channel_number}r"] = str(int(extra_channel_range))
    metadata["fil"] = os.path.basename(output_file_path)

    with open(output_file_path, "wb") as file_handle:
        flowio.create_fcs(
            file_handle,
            output_matrix.flatten(),
            channel_names,
            opt_channel_names=[description if description else None for description in channel_descriptions],
            metadata_dict=metadata,
        )
    return output_file_path


def list_fcs_files(input_directory_path: str, file_pattern: str = "*.fcs") -> List[str]:
    """Sorted list of the .fcs files of a folder (not recursive) matching a glob pattern.

    Args:
        input_directory_path: Folder holding the .fcs files.
        file_pattern: Glob pattern, e.g. '*_DR*.fcs'.

    Returns:
        Sorted list of file paths.

    Raises:
        FileNotFoundError: If the folder does not exist.
    """
    import glob

    if not os.path.isdir(input_directory_path):
        raise FileNotFoundError(f"Input folder not found: {input_directory_path}")
    return sorted(path for path in glob.glob(os.path.join(input_directory_path, file_pattern)) if path.lower().endswith(".fcs"))


def fcs_file_stem(fcs_file_path: str) -> str:
    """File name without the folder and the .fcs extension."""
    return os.path.splitext(os.path.basename(fcs_file_path))[0]
