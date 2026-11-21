"""
flowjo_workspace.py

Read the gates drawn by hand in a FlowJo workspace (.wsp) and apply them in
Python, without FlowJo (mito-marker-lab, EXP-067).

WHY

A .wsp file stores, for every sample, the exact shape of each gate drawn by the
operator (polygon vertices, rectangle bounds), the axis scale of each channel and
the number of events FlowJo found in each population. Reading it allows:

  - to reproduce exactly the populations of experiments already gated by hand;
  - to compare a rule (gate_p1(), gate_deep_red()) with the hand on the same tubes;
  - to keep a hand-drawn gate for the tubes a rule flags for a human look.

HOW FLOWJO DRAWS A GATE, AND WHAT IS REPRODUCED

FlowJo draws polygon edges as straight lines on the DISPLAYED axes, not on the raw
values. On a log axis (log10, floor at 1, values <= 1 shown on the floor) this is
reproduced exactly: over 443 samples, rectangles give FlowJo's counts exactly and
log/log polygons within 0.09 %. On a biexponential axis, FlowJo's transform is
approximated by asinh(x / 250); such gates are flagged `approximate` (within
0.33 % for P1 polygons in EXP-067). Rectangles do not depend on the axis scale,
except for the log floor, which is reproduced.

Only polygon and rectangle gates are supported (the gates used by the lab).
"""

import os
import xml.etree.ElementTree as ElementTree
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from matplotlib.path import Path as PolygonPath

from mito_marker.sfc.fcs_io import fcs_file_stem, read_fcs_events, write_fcs_events

GATING_NAMESPACE = "{http://www.isac-net.org/std/Gating-ML/v2.0/gating}"
DATATYPE_NAMESPACE = "{http://www.isac-net.org/std/Gating-ML/v2.0/datatypes}"
TRANSFORM_NAMESPACE = "{http://www.isac-net.org/std/Gating-ML/v2.0/transformations}"
BIEXPONENTIAL_COFACTOR = 250.0  # asinh(x / 250) approximates FlowJo's biex axis with width -250
COMPENSATION_PREFIXES: tuple[str, ...] = ("FJComp-", "Comp-")


def _strip_compensation_prefix(channel_name: str) -> str:
    """'Comp-R2-A' -> 'R2-A'. The lab's acquisition-defined matrix is the identity (EXP-067)."""
    for prefix in COMPENSATION_PREFIXES:
        if channel_name.startswith(prefix):
            return channel_name[len(prefix):]
    return channel_name


def _parse_gate(population: ElementTree.Element) -> Optional[Dict]:
    gate_element = population.find("Gate")
    if gate_element is None or not len(gate_element):
        return None
    shape = list(gate_element)[0]
    shape_name = shape.tag.replace(GATING_NAMESPACE, "")
    channels = [dimension.get(DATATYPE_NAMESPACE + "name") for dimension in shape.iter(DATATYPE_NAMESPACE + "fcs-dimension")]
    if shape_name == "PolygonGate":
        vertices = [[float(coordinate.get(DATATYPE_NAMESPACE + "value")) for coordinate in vertex.findall(GATING_NAMESPACE + "coordinate")]
                    for vertex in shape.findall(GATING_NAMESPACE + "vertex")]
        return {"gate_type": "polygon", "channels": channels, "vertices": vertices, "bounds": None}
    if shape_name == "RectangleGate":
        bounds = []
        for dimension in shape.findall(GATING_NAMESPACE + "dimension"):
            low, high = dimension.get(GATING_NAMESPACE + "min"), dimension.get(GATING_NAMESPACE + "max")
            bounds.append((float(low) if low else -np.inf, float(high) if high else np.inf))
        return {"gate_type": "rectangle", "channels": channels, "vertices": None, "bounds": bounds}
    return {"gate_type": shape_name, "channels": channels, "vertices": None, "bounds": None}


def _axis_scales(sample: ElementTree.Element) -> Dict[str, str]:
    scales = {}
    transformations = sample.find("Transformations")
    if transformations is None:
        return scales
    for transform in transformations:
        parameter = transform.find(DATATYPE_NAMESPACE + "parameter")
        if parameter is not None:
            scales[parameter.get(DATATYPE_NAMESPACE + "name")] = transform.tag.replace(TRANSFORM_NAMESPACE, "")
    return scales


def read_flowjo_gates(workspace_file_path: str) -> pd.DataFrame:
    """List every gated population of every sample of a FlowJo workspace.

    Args:
        workspace_file_path: Path to the .wsp file.

    Returns:
        DataFrame, one row per (sample, population): sample_file, population_path
        (e.g. '/P1/DR+'), population_name, parent_path, gate_type ('polygon',
        'rectangle', ...), channels, vertices (polygon) or bounds (rectangle),
        axis_scales (channel -> 'log', 'biex', 'linear'), flowjo_count, parent_count.
    """
    if not os.path.isfile(workspace_file_path):
        raise FileNotFoundError(f"Workspace not found: {workspace_file_path}")
    rows: List[Dict] = []

    def walk(node: ElementTree.Element, path: str, parent_count: int, base: Dict) -> None:
        for population in node.findall("Subpopulations/Population"):
            population_path = f"{path}/{population.get('name')}"
            gate = _parse_gate(population)
            count = int(population.get("count") or -1)
            if gate is not None:
                rows.append({**base, "population_path": population_path, "population_name": population.get("name"),
                             "parent_path": path or "root", **gate, "flowjo_count": count, "parent_count": parent_count})
            walk(population, population_path, count, base)

    for sample in ElementTree.parse(workspace_file_path).getroot().iter("Sample"):
        dataset, sample_node = sample.find("DataSet"), sample.find("SampleNode")
        if dataset is None or sample_node is None:
            continue
        base = {"sample_file": os.path.basename(dataset.get("uri").replace("%20", " ")), "axis_scales": _axis_scales(sample)}
        walk(sample_node, "", int(sample_node.get("count") or -1), base)
    return pd.DataFrame(rows)


def _to_display(values: np.ndarray, scale: str) -> np.ndarray:
    if scale == "log":
        return np.log10(np.clip(values, 1.0, None))
    if scale == "biex":
        return np.arcsinh(values / BIEXPONENTIAL_COFACTOR)
    return values


def gate_mask(gate_row: pd.Series, channel_values: Dict[str, np.ndarray]) -> np.ndarray:
    """Events inside one FlowJo gate (not including its parents), as FlowJo draws it.

    Args:
        gate_row: One row of read_flowjo_gates().
        channel_values: {channel name: values} of the sample (raw names, no 'Comp-' prefix needed).

    Returns:
        Boolean mask of the events inside the gate.
    """
    channels = [_strip_compensation_prefix(name) for name in gate_row["channels"]]
    scales = [gate_row["axis_scales"].get(name, gate_row["axis_scales"].get(_strip_compensation_prefix(name), "linear"))
              for name in gate_row["channels"]]
    if gate_row["gate_type"] == "rectangle":
        inside = np.ones(len(channel_values[channels[0]]), dtype=bool)
        for channel, scale, (low, high) in zip(channels, scales, gate_row["bounds"]):
            values = channel_values[channel]
            if scale == "log":
                values = np.clip(values, 1.0, None)  # FlowJo shows values <= 1 on the log-axis floor
            inside &= (values >= low) & (values < high)
        return inside
    if gate_row["gate_type"] == "polygon":
        vertices = np.asarray(gate_row["vertices"], dtype=np.float64)
        shown_vertices = np.c_[_to_display(vertices[:, 0], scales[0]), _to_display(vertices[:, 1], scales[1])]
        shown_points = np.c_[_to_display(channel_values[channels[0]], scales[0]), _to_display(channel_values[channels[1]], scales[1])]
        return PolygonPath(shown_vertices).contains_points(shown_points)
    raise NotImplementedError(f"Gate type '{gate_row['gate_type']}' is not supported (polygon and rectangle only)")


def _population_mask(population_path: str, sample_gates: pd.DataFrame, channel_values: Dict[str, np.ndarray],
                     mask_cache: Dict[str, np.ndarray]) -> np.ndarray:
    """Events inside a population and all its parents (FlowJo gates are nested); results are cached."""
    if population_path not in mask_cache:
        gate_row = sample_gates.loc[population_path]
        parent_mask = _population_mask(gate_row["parent_path"], sample_gates, channel_values, mask_cache)
        mask_cache[population_path] = parent_mask & gate_mask(gate_row, channel_values)
    return mask_cache[population_path]


def apply_flowjo_gates(
    workspace_file_path: str,
    fcs_directory_path: str,
    populations: Optional[Sequence[str]] = None,
    output_directory_path: Optional[str] = None,
    overwrite: bool = False,
) -> pd.DataFrame:
    """Apply the gates of a FlowJo workspace to its .fcs files, and optionally export the populations.

    Args:
        workspace_file_path: Path to the .wsp file.
        fcs_directory_path: Folder holding the samples' .fcs files (searched recursively by file name).
        populations: Population names or paths to apply (e.g. ['P1', 'DR+'] or ['/P1/DR+']).
            None applies every gated population. A population is always taken inside its parents.
        output_directory_path: When given, each population is written as '<sample>_<population>.fcs'
            (like a FlowJo export).
        overwrite: Replace existing exports.

    Returns:
        DataFrame, one row per (sample, population): FlowJo count, Python count, difference
        (events and % of the parent), whether the reproduction is approximate (biexponential
        polygon), output file.
    """
    gates = read_flowjo_gates(workspace_file_path)
    if gates.empty:
        raise ValueError(f"No gate found in {workspace_file_path}")
    if populations is not None:
        wanted = set(populations)
        gates = gates[gates.population_name.isin(wanted) | gates.population_path.isin(wanted)]
    file_index: Dict[str, str] = {}
    for folder, _, file_names in os.walk(fcs_directory_path):
        for file_name in file_names:
            if file_name.lower().endswith(".fcs"):
                file_index.setdefault(file_name, os.path.join(folder, file_name))
    if output_directory_path:
        os.makedirs(output_directory_path, exist_ok=True)
    all_gates = read_flowjo_gates(workspace_file_path)

    print("=" * 70)
    print("Applying the gates of a FlowJo workspace")
    print("=" * 70)
    print(f"  Workspace : {workspace_file_path}")
    print(f"  FCS folder: {fcs_directory_path} ({len(file_index)} .fcs files found)")
    print(f"  Populations: {sorted(gates.population_path.unique())}")

    rows = []
    for sample_file, sample_gates in gates.groupby("sample_file", sort=True):
        if sample_file not in file_index:
            print(f"  SKIPPED {sample_file}: .fcs file not found")
            rows.append({"sample_file": sample_file, "status": "fcs file not found"})
            continue
        fcs_events = read_fcs_events(file_index[sample_file])
        channel_values = {name: fcs_events.event_matrix[:, index] for index, name in enumerate(fcs_events.channel_names)}
        sample_all_gates = all_gates[all_gates.sample_file == sample_file].set_index("population_path")
        mask_cache: Dict[str, np.ndarray] = {"root": np.ones(fcs_events.n_events, dtype=bool)}
        for gate_row in sample_gates.itertuples():
            gate_series = sample_all_gates.loc[gate_row.population_path]
            mask = _population_mask(gate_row.population_path, sample_all_gates, channel_values, mask_cache)
            parent_count = int(_population_mask(gate_row.parent_path, sample_all_gates, channel_values, mask_cache).sum())
            approximate = gate_row.gate_type == "polygon" and any(
                gate_row.axis_scales.get(name, gate_row.axis_scales.get(_strip_compensation_prefix(name))) == "biex"
                for name in gate_row.channels)
            output_file = ""
            if output_directory_path:
                output_path = os.path.join(output_directory_path, f"{fcs_file_stem(sample_file)}_{gate_row.population_name}.fcs")
                if os.path.exists(output_path) and not overwrite:
                    raise FileExistsError(f"{output_path} exists; pass overwrite=True to replace it")
                write_fcs_events(fcs_events, output_path, event_mask=mask)
                output_file = os.path.basename(output_path)
            python_count = int(mask.sum())
            rows.append({"sample_file": sample_file, "status": "ok", "population_path": gate_row.population_path,
                         "gate_type": gate_series["gate_type"], "channels": "|".join(gate_series["channels"]),
                         "flowjo_count": gate_row.flowjo_count, "python_count": python_count,
                         "difference": python_count - gate_row.flowjo_count,
                         "difference_percent_of_parent": round(100 * (python_count - gate_row.flowjo_count) / max(parent_count, 1), 4),
                         "approximate": bool(approximate), "output_file": output_file})
        del fcs_events, channel_values, mask_cache

    result = pd.DataFrame(rows)
    applied = result[result.status == "ok"] if "status" in result else result.iloc[0:0]
    print("-" * 70)
    print("QC — FlowJo gates applied in Python")
    print(f"  Samples: {result.sample_file.nunique()}, populations computed: {len(applied)}")
    if len(applied):
        exact = applied.difference == 0
        print(f"  Same count as FlowJo: {int(exact.sum())} / {len(applied)}; "
              f"largest |difference|: {applied.difference_percent_of_parent.abs().max():.3f}% of the parent")
        print(f"  Approximate (polygon on a biexponential axis): {int(applied.approximate.sum())}")
    print(result.head().to_string())
    print("=" * 70)
    return result
