"""
channel_aggregation.py

Aggregate SFC channels by laser colour and pulse type (-A / -H / -W).

For a Cytek Aurora AnnData with 142 channels spread across six laser groups
(FSC, SSC, UV, V, B, YG, R), this module reduces the data to one variable per
(laser_group, pulse_type) pair by averaging all detector channels within that
group.  The result is a new AnnData with typically ~18 variables instead of 142,
suitable for radar plots and longitudinal curves.

Example output variable names (in canonical laser order)::

    FSC-A, FSC-H, FSC-W,
    SSC-A, SSC-H, SSC-W,
    UV-A,  UV-H,  UV-W,
    V-A,   V-H,   V-W,
    B-A,   B-H,   B-W,
    YG-A,  YG-H,  YG-W,
    R-A,   R-H,   R-W

Typical usage::

    from mito_marker.analysis import aggregate_sfc_channels_by_color

    color_anndata = aggregate_sfc_channels_by_color(sfc_anndata)
    # color_anndata.n_vars ≈ 18

    from mito_marker.analysis import plot_radar
    plot_radar(color_anndata, group_by="condition")

    from mito_marker.analysis import plot_time_curve
    plot_time_curve(color_anndata, x_column="timepoint",
                    y_channels=["UV-A", "YG-A", "R-A"], group_by="condition")
"""

import copy
import re
import warnings
from collections import defaultdict
from typing import Dict, List, Tuple

import anndata
import numpy as np
import pandas as pd

_ANALYSIS_CONFIG_KEY = "analysis_config"

# Canonical laser group order — same as radar_plot._LASER_GROUP_ORDER.
_LASER_GROUP_ORDER: Dict[str, int] = {
    "FSC": 0,
    "SSC": 1,
    "UV":  2,
    "V":   3,
    "B":   4,
    "YG":  5,
    "R":   6,
}

# Pattern matching Cytek Aurora channel names: laser_prefix + optional_detector + '-' + pulse.
# Examples: "UV16-A", "FSC-H", "YG10-W", "B14-A"
_CHANNEL_PATTERN = re.compile(r"^([A-Za-z]+)\d*-([AHW])$")


def aggregate_sfc_channels_by_color(
    anndata_object: anndata.AnnData,
) -> anndata.AnnData:
    """
    Return a new AnnData where each variable is the mean of all channels from
    the same laser group and same pulse type (-A, -H, -W).

    The source data is read from the active layer if one is set in
    .uns['analysis_config']['active_layer'], otherwise from raw .X.  The
    returned AnnData's .X contains the aggregated values; active_layer is
    reset to None (the .X already IS the relevant data).

    Variable names in the result follow the pattern ``"{LASER}-{PULSE}"``
    (e.g. ``"UV-A"``, ``"R-H"``), ordered by canonical laser group
    (FSC → SSC → UV → V → B → YG → R) then pulse type (A → H → W).

    Channels whose names do not match the expected pattern
    ``{laser_prefix}{optional_digits}-{A|H|W}`` are excluded with a warning.

    Arguments:
        anndata_object: SFC AnnData (Cytek Aurora or compatible).  .obs,
                        .uns, and layer information are copied to the result.

    Returns:
        A new AnnData with:
          - .X          — float32 aggregated matrix, shape (n_obs, n_groups)
          - .obs        — copy of the original .obs
          - .var        — DataFrame indexed by aggregated channel names
          - .uns        — copy of the original .uns with active_layer set to None
          - .layers     — empty (the aggregated values live in .X)
          - .obsm/.obsp — empty

    Raises:
        ValueError: If no channels match the expected naming pattern.
    """
    source_matrix, channel_names = _get_data_matrix_and_names(anndata_object)

    # Group channels: {(laser, pulse): [col_index, ...]}
    laser_pulse_groups: Dict[Tuple[str, str], List[int]] = defaultdict(list)
    unmatched: List[str] = []

    for col_index, channel_name in enumerate(channel_names):
        match = _CHANNEL_PATTERN.match(channel_name)
        if match:
            laser = match.group(1).upper()
            pulse = match.group(2).upper()
            laser_pulse_groups[(laser, pulse)].append(col_index)
        else:
            unmatched.append(channel_name)

    if unmatched:
        warnings.warn(
            f"[aggregate_sfc_channels_by_color] {len(unmatched)} channel(s) did not "
            f"match the expected laser-pulse pattern and were excluded: {unmatched}",
            stacklevel=2,
        )

    if not laser_pulse_groups:
        raise ValueError(
            "No channels matched the expected SFC naming pattern "
            "'{laser}{digits}-{A|H|W}'. "
            f"Variable names found: {channel_names[:10]}..."
        )

    # Sort groups by canonical laser order, then pulse type order (A, H, W).
    pulse_order = {"A": 0, "H": 1, "W": 2}
    sorted_groups = sorted(
        laser_pulse_groups.items(),
        key=lambda kv: (
            _LASER_GROUP_ORDER.get(kv[0][0], 999),
            pulse_order.get(kv[0][1], 9),
        ),
    )

    # Build aggregated matrix column by column.
    aggregated_columns: List[np.ndarray] = []
    aggregated_var_names: List[str] = []
    aggregated_source_counts: List[int] = []

    for (laser, pulse), col_indices in sorted_groups:
        sub_matrix = source_matrix[:, col_indices]
        aggregated_col = np.mean(sub_matrix, axis=1).astype(np.float32)
        aggregated_columns.append(aggregated_col)
        aggregated_var_names.append(f"{laser}-{pulse}")
        aggregated_source_counts.append(len(col_indices))

    aggregated_matrix = np.column_stack(aggregated_columns).astype(np.float32)

    # .var DataFrame: one row per aggregated channel.
    var_dataframe = pd.DataFrame(
        {"n_source_channels": aggregated_source_counts},
        index=aggregated_var_names,
    )
    var_dataframe.index.name = None

    # .uns: deep-copy original, then reset active_layer to None.
    uns_copy = copy.deepcopy(dict(anndata_object.uns))
    if _ANALYSIS_CONFIG_KEY in uns_copy:
        uns_copy[_ANALYSIS_CONFIG_KEY] = dict(uns_copy[_ANALYSIS_CONFIG_KEY])
        uns_copy[_ANALYSIS_CONFIG_KEY]["active_layer"] = None
        uns_copy[_ANALYSIS_CONFIG_KEY]["active_selection"] = None
    uns_copy["aggregation_source"] = {
        "method": "aggregate_sfc_channels_by_color",
        "n_source_channels": len(channel_names),
        "n_aggregated_channels": len(aggregated_var_names),
    }

    result = anndata.AnnData(
        X=aggregated_matrix,
        obs=anndata_object.obs.copy(),
        var=var_dataframe,
        uns=uns_copy,
    )

    active_layer_label = (
        anndata_object.uns.get(_ANALYSIS_CONFIG_KEY, {}).get("active_layer") or "raw .X"
    )
    print(
        f"[aggregate_sfc_channels_by_color] Source: '{active_layer_label}' | "
        f"{len(channel_names)} channels → {len(aggregated_var_names)} colour-aggregated channels"
    )
    print(f"  Aggregated variables: {aggregated_var_names}")
    if unmatched:
        print(f"  Excluded (no pattern match): {unmatched}")
    print(f"  Result shape: {result.n_obs} obs × {result.n_vars} vars")

    return result


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _get_data_matrix_and_names(
    anndata_object: anndata.AnnData,
) -> Tuple[np.ndarray, List[str]]:
    """
    Return (data_matrix, channel_names) from the active layer or .X.

    Does NOT apply feature selection (aggregation should operate on all
    channels before any selection filter to preserve the laser-group
    averages).  Non-analytical channels (Time, FlowAI, spatial coords) are
    excluded so they do not pollute the averages.
    """
    analysis_config = anndata_object.uns.get(_ANALYSIS_CONFIG_KEY, {})
    active_layer = analysis_config.get("active_layer")

    if active_layer is not None and active_layer in anndata_object.layers:
        full_matrix = anndata_object.layers[active_layer]
        layer_label = active_layer
    else:
        full_matrix = anndata_object.X
        layer_label = "raw .X"

    if hasattr(full_matrix, "toarray"):
        full_matrix = full_matrix.toarray()

    full_matrix = np.asarray(full_matrix, dtype=np.float32)

    print(f"[aggregate_sfc_channels_by_color] Reading from: '{layer_label}'")

    if "is_non_analytical" in anndata_object.var.columns:
        analytical_mask = ~anndata_object.var["is_non_analytical"].values.astype(bool)
        channel_names = anndata_object.var_names[analytical_mask].tolist()
        return full_matrix[:, analytical_mask], channel_names

    channel_names = anndata_object.var_names.tolist()
    return full_matrix, channel_names
