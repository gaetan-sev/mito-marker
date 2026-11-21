"""
conftest.py

Shared pytest fixtures.

write_synthetic_fcs: write a small .fcs file whose channels and values are known by
construction (used by the SFC preprocessing tests: fcs_io, flowai, gating,
flowjo_workspace). No real acquisition is needed.
"""

import os
from typing import Callable, Dict, Optional

import flowio
import numpy as np
import pytest

AURORA_CHANNEL_RANGE = 4194304  # $PnR of the Cytek Aurora channels


@pytest.fixture
def write_synthetic_fcs(tmp_path) -> Callable[..., str]:
    """Return a function writing a synthetic .fcs file and returning its path.

    The function takes: file_name, channels ({name: values}), optional keywords
    (extra TEXT keywords), optional ranges ({name: $PnR}, default 4194304) and an
    optional folder (default tmp_path).
    """

    def write(
        file_name: str,
        channels: Dict[str, np.ndarray],
        keywords: Optional[Dict[str, str]] = None,
        ranges: Optional[Dict[str, float]] = None,
        folder: Optional[str] = None,
    ) -> str:
        folder = str(folder or tmp_path)
        os.makedirs(folder, exist_ok=True)
        channel_names = list(channels)
        event_matrix = np.column_stack([np.asarray(channels[name], dtype=np.float64) for name in channel_names]).astype(np.float32)
        metadata = {"date": "01-Jan-2026", "btim": "10:00:00", "cyt": "Aurora", "timestep": "0.0001"}
        for channel_number, name in enumerate(channel_names, start=1):
            metadata[f"p{channel_number}r"] = str(int((ranges or {}).get(name, AURORA_CHANNEL_RANGE)))
        metadata.update(keywords or {})
        file_path = os.path.join(folder, file_name)
        with open(file_path, "wb") as file_handle:
            flowio.create_fcs(file_handle, event_matrix.flatten(), channel_names, metadata_dict=metadata)
        return file_path

    return write
