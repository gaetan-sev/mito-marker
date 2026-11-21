"""
gating.py

The two gates the lab draws by hand in FlowJo after flowAI, as fixed rules
(ADR-023, ADR-024; mito-marker-lab, EXP-067):

  P1  — keep the events whose FSC-A and SSC-A areas are above 1 and below
        saturation. By hand, P1 is a large polygon on the FSC-A / SSC-A log plot;
        in practice it removes the events piled on the axes, whose area is zero
        or negative and cannot be placed on a log scale (93 % of the removed
        events have FSC-A <= 1). The rule keeps 99.9 % of the same events as the
        hand-drawn P1 (median Jaccard index 0.9993 over 443 samples).

  DR+ — keep the events of a Deep Red stained tube that are brighter, in R2-A,
        than 99.9 % of the events of the unstained (NM) tube of the same
        subject. This is the operator's rule ("everything outside the
        brightness zone of the unstained reference") written as a number: the
        manual thresholds let through 0.12 % of the NM events (median). A rule
        cannot reproduce the hand exactly, because the hand is not constant
        (0 to 12.6 % of the NM let through, 10.6 points between the gates of
        one experiment); it is about 5 points from the hand in a typical tube.

TWO ALERTS FOR A HUMAN LOOK (DR+)

Where the stained and unstained clouds overlap, any threshold is a trade-off
between unstained events let through and stained events lost, and a rule and a
person can disagree by 40 points (Smurf, EXP-067). Two numbers flag those tubes;
neither assumes a shape for the clouds (a Smurf NM tube can hold two populations):

  - unstained bright tail: how stretched the top of the NM tube is, i.e. the
    distance between its 99th and 99.9th percentiles on the display scale
    asinh(R2-A / 250), which FlowJo's biexponential axis resembles. A clean NM
    has a compact top; a bright tail or a second population stretches it.
  - threshold sensitivity: the % of the stained events between the 99th and the
    99.99th percentiles of the NM tube, i.e. how many points of DR+ would change
    if 1 % of the NM were let through instead of 0.01 %. A large value means
    the result depends on the trade-off, not on the staining.

A tube with an alert is gated like the others and flagged in the summary
table and on its QC figure (grey band = that 99–99.99 % zone).
"""

import gc
import os
import re
from typing import Dict, List, Sequence, Union

import numpy as np
import pandas as pd

from mito_marker.controlled_vocabulary import MARKER_TOKEN_TO_CANONICAL_SFC
from mito_marker.sfc.fcs_io import (
    FcsEvents,
    fcs_file_stem,
    list_fcs_files,
    read_fcs_events,
    write_fcs_events,
)

# ---------------------------------------------------------------- defaults
DEFAULT_P1_SCATTER_CHANNELS: tuple[str, str] = ("FSC-A", "SSC-A")
DEFAULT_P1_MINIMUM_AREA = 1.0
DEFAULT_DEEP_RED_CHANNEL = "R2-A"
DEFAULT_DEEP_RED_QUANTILE = 0.999
# Display scale: asinh(x / 250) follows FlowJo's biexponential axis (width -250) closely
# enough to describe where the clouds sit; it is used for the alerts and figures only.
DISPLAY_SCALE_COFACTOR = 250.0
TAIL_LOWER_QUANTILE, TAIL_UPPER_QUANTILE = 0.99, 0.999             # NM top spread, alert 1
SENSITIVITY_LOWER_QUANTILE, SENSITIVITY_UPPER_QUANTILE = 0.99, 0.9999  # threshold band, alert 2
DEFAULT_UNSTAINED_TAIL_ALERT_SPREAD = 0.867        # display units; EXP-067 amendment 1, see ADR-024
DEFAULT_THRESHOLD_SENSITIVITY_ALERT_POINTS = 44.8  # % of stained events; EXP-067 amendment 1, see ADR-024
GATE_INSIDE_LABEL = 2000.0
GATE_OUTSIDE_LABEL = 6000.0
OUTPUT_MODES: tuple[str, ...] = ("kept_events", "all_events_flagged", "both")
R_MAD_SCALE = 1.4826

# Filename words removed before pairing tubes: export prefixes and step suffixes.
PAIRING_IGNORED_TOKENS: set[str] = {"export", "expor", "good", "events", "goodevents", "FlowAIGoodEvents",
                                    "P1", "DR+", "FlowAIFlagged", "P1Flagged", "DRFlagged"}
DEEP_RED_TOKENS: set[str] = {token for token, marker in MARKER_TOKEN_TO_CANONICAL_SFC.items() if marker == "MtDeepRed"}
UNSTAINED_TOKENS: set[str] = {token for token, marker in MARKER_TOKEN_TO_CANONICAL_SFC.items() if marker == "Not_Marked"}


def _saturation_values(fcs_events: FcsEvents, channel_names: Sequence[str]) -> np.ndarray:
    """Top of the range ($PnR - 1) of each channel; values there are saturated.

    Some FlowJo exports write $PnR = 1 for FSC-A / SSC-A (EXP-067, D08-D12). A range
    below the largest value of the channel cannot be the instrument range. The most
    common plausible range of the file's other channels is used instead (on the
    Aurora every detector shares one range, 4194304), with a printed note; when no
    channel has a plausible range, the saturation check is skipped (NaN).
    """
    ranges = fcs_events.channel_ranges()
    plausible = [channel_range for channel_range, values in zip(ranges, fcs_events.event_matrix.T)
                 if not np.isnan(channel_range) and (values.size == 0 or channel_range >= values.max())]
    usual_range = max(set(plausible), key=plausible.count) if plausible else np.nan
    saturation = []
    for name in channel_names:
        channel_range = ranges[fcs_events.channel_names.index(name)]
        values = fcs_events.channel_values(name)
        if np.isnan(channel_range) or (values.size and channel_range < values.max()):
            print(f"  Note: {os.path.basename(fcs_events.source_file_path)}: $PnR of {name} ({channel_range}) is below its "
                  f"largest value; using the file's usual range ({usual_range}) to find saturated events")
            saturation.append(usual_range - 1)
        else:
            saturation.append(channel_range - 1)
    return np.array(saturation)


def _check_output_mode(output_mode: str) -> None:
    if output_mode not in OUTPUT_MODES:
        raise ValueError(f"output_mode must be one of {OUTPUT_MODES}, got '{output_mode}'")


def _planned_output_paths(input_file: str, output_directory_path: str, kept_suffix: str, flagged_suffix: str, output_mode: str) -> List[str]:
    stem = fcs_file_stem(input_file)
    outputs = []
    if output_mode in ("kept_events", "both"):
        outputs.append(os.path.join(output_directory_path, f"{stem}_{kept_suffix}.fcs"))
    if output_mode in ("all_events_flagged", "both"):
        outputs.append(os.path.join(output_directory_path, f"{stem}_{flagged_suffix}.fcs"))
    return outputs


def _refuse_existing_outputs(planned_paths: Sequence[str], overwrite: bool) -> None:
    existing = [path for path in planned_paths if os.path.exists(path)]
    if existing and not overwrite:
        raise FileExistsError(f"{len(existing)} output file(s) already exist (e.g. {existing[0]}); pass overwrite=True to replace them")


def _write_gate_outputs(fcs_events: FcsEvents, inside_gate: np.ndarray, output_paths: Sequence[str], kept_suffix: str, gate_channel_name: str) -> List[str]:
    """Write the kept events, and/or every event with a gate channel (2000 inside, 6000 outside)."""
    written = []
    for output_path in output_paths:
        if output_path.endswith(f"_{kept_suffix}.fcs"):
            write_fcs_events(fcs_events, output_path, event_mask=inside_gate)
        else:
            write_fcs_events(fcs_events, output_path,
                             extra_channels={gate_channel_name: np.where(inside_gate, GATE_INSIDE_LABEL, GATE_OUTSIDE_LABEL)})
        written.append(os.path.basename(output_path))
    return written


# ---------------------------------------------------------------- P1
def compute_p1_mask(
    fcs_events: FcsEvents,
    scatter_channels: Sequence[str] = DEFAULT_P1_SCATTER_CHANNELS,
    minimum_area: float = DEFAULT_P1_MINIMUM_AREA,
    exclude_saturated: bool = True,
) -> np.ndarray:
    """P1 rule: True for the events whose scatter areas are above minimum_area (and not saturated).

    Args:
        fcs_events: Events of one tube.
        scatter_channels: Channels checked (default FSC-A and SSC-A).
        minimum_area: An event needs a value above this in every scatter channel. 1 is the
            floor of the FlowJo log axis: lower values cannot be placed on it.
        exclude_saturated: Also remove events at the top of the range ($PnR - 1).

    Returns:
        Boolean mask (n_events,), file order.
    """
    inside = np.ones(fcs_events.n_events, dtype=bool)
    saturation = _saturation_values(fcs_events, scatter_channels)
    for channel_name, saturation_value in zip(scatter_channels, saturation):
        values = fcs_events.channel_values(channel_name)
        inside &= values > minimum_area
        if exclude_saturated and not np.isnan(saturation_value):
            inside &= values < saturation_value
    return inside


def _save_p1_qc_figure(fcs_events: FcsEvents, inside_gate: np.ndarray, scatter_channels: Sequence[str], minimum_area: float, figure_path: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x_values = fcs_events.channel_values(scatter_channels[0])
    y_values = fcs_events.channel_values(scatter_channels[1])
    shown = (x_values > 0) & (y_values > 0)
    figure, axis = plt.subplots(figsize=(5.2, 4.6))
    axis.hist2d(np.log10(x_values[shown]), np.log10(y_values[shown]), bins=150, cmap="Greys", cmin=1)
    axis.axvline(np.log10(minimum_area), color="tab:red", linewidth=1)
    axis.axhline(np.log10(minimum_area), color="tab:red", linewidth=1)
    axis.set_xlabel(f"log10 {scatter_channels[0]}")
    axis.set_ylabel(f"log10 {scatter_channels[1]}")
    not_shown = int((~shown).sum())
    axis.set_title(f"P1 — {os.path.basename(fcs_events.source_file_path)}\n"
                   f"kept {int(inside_gate.sum())} of {fcs_events.n_events} ({100 * inside_gate.mean():.1f}%); "
                   f"{not_shown} events have an area <= 0 and cannot be drawn on log axes", fontsize=8)
    figure.tight_layout()
    figure.savefig(figure_path, dpi=110)
    plt.close(figure)


def gate_p1(
    input_directory_path: str,
    output_directory_path: str,
    file_pattern: str = "*.fcs",
    scatter_channels: Sequence[str] = DEFAULT_P1_SCATTER_CHANNELS,
    minimum_area: float = DEFAULT_P1_MINIMUM_AREA,
    exclude_saturated: bool = True,
    output_mode: str = "kept_events",
    save_qc_figures: bool = True,
    overwrite: bool = False,
) -> pd.DataFrame:
    """Apply the P1 rule to every .fcs file of a folder and write the results as .fcs files.

    Outputs in output_directory_path ('<tube>' = input file name without .fcs):
      - 'kept_events': '<tube>_P1.fcs', the events inside P1 (like a FlowJo P1 export);
      - 'all_events_flagged': '<tube>_P1Flagged.fcs', every event plus a 'P1_Gate' channel
        (2000 inside, 6000 outside), to see the removed events in FlowJo;
      - 'both'. Plus 'p1_summary.csv' and, if asked, one QC figure per tube.

    Args:
        input_directory_path: Folder of .fcs files, usually the run_flowai() output.
        output_directory_path: Folder for the outputs (created if missing).
        file_pattern: Glob pattern selecting the files.
        scatter_channels, minimum_area, exclude_saturated: see compute_p1_mask().
        output_mode: 'kept_events', 'all_events_flagged' or 'both'.
        save_qc_figures: Write one PNG per tube.
        overwrite: Replace existing outputs; False raises before writing anything.

    Returns:
        DataFrame, one row per tube: events in and kept, % kept, output file(s).
    """
    _check_output_mode(output_mode)
    input_files = list_fcs_files(input_directory_path, file_pattern)
    if not input_files:
        raise FileNotFoundError(f"No .fcs file matching '{file_pattern}' in {input_directory_path}")
    os.makedirs(output_directory_path, exist_ok=True)
    if save_qc_figures:
        os.makedirs(os.path.join(output_directory_path, "qc_figures"), exist_ok=True)
    planned = {path: _planned_output_paths(path, output_directory_path, "P1", "P1Flagged", output_mode) for path in input_files}
    _refuse_existing_outputs([path for outputs in planned.values() for path in outputs], overwrite)

    print("=" * 70)
    print("P1 gate (fixed rule, ADR-023)")
    print("=" * 70)
    print(f"  Input folder : {input_directory_path} ({len(input_files)} files matching '{file_pattern}')")
    print(f"  Output folder: {output_directory_path}")
    print(f"  Rule         : {' and '.join(f'{name} > {minimum_area}' for name in scatter_channels)}"
          f"{', unsaturated' if exclude_saturated else ''}")
    print(f"  Output mode  : {output_mode}")

    summary_rows = []
    for file_number, input_file in enumerate(input_files, start=1):
        fcs_events = read_fcs_events(input_file)
        inside = compute_p1_mask(fcs_events, scatter_channels, minimum_area, exclude_saturated)
        written = _write_gate_outputs(fcs_events, inside, planned[input_file], "P1", "P1_Gate")
        if save_qc_figures:
            _save_p1_qc_figure(fcs_events, inside, scatter_channels, minimum_area,
                               os.path.join(output_directory_path, "qc_figures", f"{fcs_file_stem(input_file)}_p1.png"))
        summary_rows.append({"input_file": os.path.basename(input_file), "events_in": fcs_events.n_events,
                             "events_kept": int(inside.sum()), "percent_kept": round(100 * float(inside.mean()), 3),
                             "output_files": ";".join(written)})
        print(f"  [{file_number}/{len(input_files)}] {os.path.basename(input_file)}: kept {int(inside.sum())} of {fcs_events.n_events} ({100 * inside.mean():.1f}%)")
        del fcs_events, inside
        gc.collect()

    summary = pd.DataFrame(summary_rows)
    summary_path = os.path.join(output_directory_path, "p1_summary.csv")
    summary.to_csv(summary_path, index=False)
    print("-" * 70)
    print("QC — P1")
    print(f"  Tubes: {len(summary)}; % kept per tube: median {summary.percent_kept.median():.1f}, "
          f"min {summary.percent_kept.min():.1f}, max {summary.percent_kept.max():.1f}")
    assert (summary.events_kept <= summary.events_in).all(), "More events kept than read"
    print(f"  Summary table: {summary_path}")
    print(summary.head().to_string())
    print("=" * 70)
    return summary


# ---------------------------------------------------------------- pairing stained / unstained tubes
def _pairing_tokens(file_name: str) -> List[str]:
    stem = re.sub(r"\.fcs$", "", os.path.basename(file_name), flags=re.IGNORECASE)
    return [token for token in stem.split("_") if token and token not in PAIRING_IGNORED_TOKENS]


def pair_stained_unstained(input_directory_path: str, file_pattern: str = "*.fcs") -> pd.DataFrame:
    """Pair each Deep Red stained tube with the unstained (NM) tube of the same subject.

    A tube is stained when a filename word means Deep Red ('DR', 'DeepRed', 'DeepRed+')
    and unstained when a word is 'NM' (MARKER_TOKEN_TO_CANONICAL_SFC). The NM tube must
    share the first two words (e.g. 'WORM_005', 'MOUSE_001', 'KF_001'); among those, the
    one sharing the longest run of leading words is taken. Check the table: when names
    are not consistent (e.g. a reference NM from another animal), edit 'unstained_file'
    by hand and pass the edited table to gate_deep_red().

    Args:
        input_directory_path: Folder of .fcs files (usually the gate_p1() output).
        file_pattern: Glob pattern selecting the files.

    Returns:
        DataFrame with columns stained_file, unstained_file (NaN when none), match
        ('same subject', 'several candidates, first taken' or 'no unstained tube found').
    """
    file_names = [os.path.basename(path) for path in list_fcs_files(input_directory_path, file_pattern)]
    stained = [name for name in file_names if set(_pairing_tokens(name)) & DEEP_RED_TOKENS]
    unstained = [name for name in file_names if set(_pairing_tokens(name)) & UNSTAINED_TOKENS]
    rows = []
    for stained_name in stained:
        stained_tokens = _pairing_tokens(stained_name)
        best_names, best_length = [], 0
        for unstained_name in unstained:
            unstained_tokens = _pairing_tokens(unstained_name)
            if unstained_tokens[:2] != stained_tokens[:2]:
                continue
            common = 0
            for stained_token, unstained_token in zip(stained_tokens, unstained_tokens):
                if stained_token != unstained_token:
                    break
                common += 1
            if common > best_length:
                best_names, best_length = [unstained_name], common
            elif common == best_length:
                best_names.append(unstained_name)
        if not best_names:
            rows.append({"stained_file": stained_name, "unstained_file": np.nan, "match": "no unstained tube found"})
        else:
            rows.append({"stained_file": stained_name, "unstained_file": sorted(best_names)[0],
                         "match": "same subject" if len(best_names) == 1 else "several candidates, first taken"})
    pairs = pd.DataFrame(rows, columns=["stained_file", "unstained_file", "match"])
    print("=" * 70)
    print("Pairing Deep Red stained tubes with their unstained (NM) tube")
    print("=" * 70)
    print(f"  Folder: {input_directory_path}; stained tubes: {len(stained)}, unstained tubes: {len(unstained)}")
    print(f"  Matches: {pairs.match.value_counts().to_dict()}")
    print(pairs.to_string())
    print("  Check this table; edit 'unstained_file' where needed before gate_deep_red().")
    print("=" * 70)
    return pairs


# ---------------------------------------------------------------- DR+
def _display_scale(values: np.ndarray) -> np.ndarray:
    return np.arcsinh(np.asarray(values, dtype=np.float64) / DISPLAY_SCALE_COFACTOR)


def compute_deep_red_threshold(unstained_values: np.ndarray, quantile: float = DEFAULT_DEEP_RED_QUANTILE) -> float:
    """DR+ threshold: the given quantile of the unstained tube's values (default 99.9th percentile)."""
    if unstained_values.size == 0:
        raise ValueError("The unstained tube has no event")
    if not 0 < quantile <= 1:
        raise ValueError(f"quantile must be in (0, 1], got {quantile}")
    return float(np.quantile(unstained_values, quantile))


def compute_deep_red_alerts(
    unstained_values: np.ndarray,
    stained_values: np.ndarray,
    unstained_tail_alert_spread: float = DEFAULT_UNSTAINED_TAIL_ALERT_SPREAD,
    threshold_sensitivity_alert_points: float = DEFAULT_THRESHOLD_SENSITIVITY_ALERT_POINTS,
) -> Dict[str, Union[float, bool]]:
    """The two DR+ alerts (see the module docstring).

    Args:
        unstained_values: Deep Red channel of the unstained (NM) tube.
        stained_values: Deep Red channel of the stained tube.
        unstained_tail_alert_spread: Alert 1 when asinh(q99.9 / 250) - asinh(q99 / 250) of the NM
            exceeds this.
        threshold_sensitivity_alert_points: Alert 2 when more than this % of the stained events lie
            between the NM's 99th and 99.99th percentiles.

    Returns:
        dict with unstained_tail_spread, threshold_sensitivity_points, band_low and band_high
        (raw values of the NM's 99th and 99.99th percentiles), unstained_tail_alert,
        threshold_sensitivity_alert.
    """
    tail_low, tail_high = np.quantile(unstained_values, [TAIL_LOWER_QUANTILE, TAIL_UPPER_QUANTILE])
    tail_spread = float(_display_scale(np.array([tail_high]))[0] - _display_scale(np.array([tail_low]))[0])
    band_low, band_high = np.quantile(unstained_values, [SENSITIVITY_LOWER_QUANTILE, SENSITIVITY_UPPER_QUANTILE])
    sensitivity = 100.0 * float(np.mean((stained_values > band_low) & (stained_values <= band_high))) if stained_values.size else float("nan")
    return {
        "unstained_tail_spread": tail_spread,
        "threshold_sensitivity_points": sensitivity,
        "band_low": float(band_low),
        "band_high": float(band_high),
        "unstained_tail_alert": bool(tail_spread > unstained_tail_alert_spread),
        "threshold_sensitivity_alert": bool(sensitivity > threshold_sensitivity_alert_points),
    }


def _save_deep_red_qc_figure(stained_values: np.ndarray, unstained_values: np.ndarray, threshold: float, alerts: Dict,
                             channel: str, stained_name: str, unstained_name: str, figure_path: str) -> None:
    """Histograms of the NM and stained tubes on the display scale, with the threshold and the 99-99.99 % band."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(7.8, 4.0))
    all_shown = _display_scale(np.concatenate([stained_values, unstained_values]))
    bins = np.linspace(np.percentile(all_shown, 0.1), np.percentile(all_shown, 99.9), 160)
    band = _display_scale(np.array([alerts["band_low"], alerts["band_high"]]))
    axis.axvspan(band[0], band[1], color="0.85", label="NM 99th to 99.99th percentile: DR+ here depends on the trade-off")
    axis.hist(_display_scale(unstained_values), bins=bins, density=True, histtype="step", color="0.3", label=f"unstained (NM): {unstained_name}")
    axis.hist(_display_scale(stained_values), bins=bins, density=True, histtype="step", color="tab:red", label=f"stained (DR): {stained_name}")
    axis.axvline(_display_scale(np.array([threshold]))[0], color="black", linewidth=1.2, label="DR+ threshold (99.9 % of the NM below)")
    axis.set_xlabel(f"asinh({channel} / {DISPLAY_SCALE_COFACTOR:.0f})  (display scale, like FlowJo's biexponential axis)")
    axis.set_ylabel("density")
    alert_text = []
    if alerts["unstained_tail_alert"]:
        alert_text.append(f"NM bright tail (spread {alerts['unstained_tail_spread']:.2f})")
    if alerts["threshold_sensitivity_alert"]:
        alert_text.append(f"{alerts['threshold_sensitivity_points']:.1f}% of DR in the grey band")
    axis.set_title(f"DR+ = {100 * np.mean(stained_values > threshold):.1f}% of the stained events"
                   + (" — ALERT: " + "; ".join(alert_text) + " → look at this tube" if alert_text else ""), fontsize=8)
    axis.legend(fontsize=6, loc="upper left")
    figure.tight_layout()
    figure.savefig(figure_path, dpi=110)
    plt.close(figure)


def gate_deep_red(
    input_directory_path: str,
    output_directory_path: str,
    pairs: Union[pd.DataFrame, str, None] = None,
    channel: str = DEFAULT_DEEP_RED_CHANNEL,
    quantile: float = DEFAULT_DEEP_RED_QUANTILE,
    unstained_tail_alert_spread: float = DEFAULT_UNSTAINED_TAIL_ALERT_SPREAD,
    threshold_sensitivity_alert_points: float = DEFAULT_THRESHOLD_SENSITIVITY_ALERT_POINTS,
    output_mode: str = "kept_events",
    save_qc_figures: bool = True,
    overwrite: bool = False,
) -> pd.DataFrame:
    """Apply the DR+ rule to every stained tube of a folder, using its paired unstained tube.

    For each pair, threshold = quantile of `channel` in the unstained tube (default the
    99.9th percentile of R2-A); the stained tube's events above the threshold are DR+.
    Outputs in output_directory_path ('<tube>' = stained file name without .fcs):
      - 'kept_events': '<tube>_DR+.fcs', the DR+ events (like a FlowJo DR+ export);
      - 'all_events_flagged': '<tube>_DRFlagged.fcs', every event plus a 'DR_Gate' channel
        (2000 above the threshold, 6000 below);
      - 'both'. Plus 'deep_red_summary.csv' and, if asked, one QC figure per stained tube.

    Args:
        input_directory_path: Folder of .fcs files after P1 (gate_p1() output).
        output_directory_path: Folder for the outputs (created if missing).
        pairs: Output of pair_stained_unstained(), possibly edited, or the path of that
            table saved as CSV. None pairs the tubes automatically. 'unstained_file' may list
            several NM files separated by ';': their events are pooled into one reference
            (when the NM tubes are other animals than the stained ones).
        channel: Deep Red channel (default 'R2-A').
        quantile: Share of the unstained events left below the threshold (default 0.999).
        unstained_tail_alert_spread, threshold_sensitivity_alert_points: alert limits, see
            compute_deep_red_alerts().
        output_mode: 'kept_events', 'all_events_flagged' or 'both'.
        save_qc_figures: Write one PNG per stained tube.
        overwrite: Replace existing outputs; False raises before writing anything.

    Returns:
        DataFrame, one row per stained tube: unstained tube, threshold, % DR+, % of the NM
        above the threshold, alert values and flags, output file(s).
    """
    _check_output_mode(output_mode)
    if pairs is None:
        pairs = pair_stained_unstained(input_directory_path)
    elif isinstance(pairs, str):
        pairs = pd.read_csv(pairs)
    missing_columns = {"stained_file", "unstained_file"} - set(pairs.columns)
    if missing_columns:
        raise ValueError(f"pairs is missing the column(s) {sorted(missing_columns)}")
    os.makedirs(output_directory_path, exist_ok=True)
    if save_qc_figures:
        os.makedirs(os.path.join(output_directory_path, "qc_figures"), exist_ok=True)
    planned = {row.stained_file: _planned_output_paths(row.stained_file, output_directory_path, "DR+", "DRFlagged", output_mode)
               for row in pairs.itertuples() if isinstance(row.unstained_file, str)}
    _refuse_existing_outputs([path for outputs in planned.values() for path in outputs], overwrite)

    print("=" * 70)
    print("DR+ gate (threshold from the paired unstained tube, ADR-024)")
    print("=" * 70)
    print(f"  Input folder : {input_directory_path}")
    print(f"  Output folder: {output_directory_path}")
    print(f"  Rule         : {channel} > {100 * quantile:g}th percentile of the paired NM tube")
    print(f"  Alerts       : NM tail spread > {unstained_tail_alert_spread} ; DR between NM 99-99.99 % > {threshold_sensitivity_alert_points} points")
    print(f"  Pairs        : {len(pairs)} stained tubes, {int(pairs.unstained_file.notna().sum())} with an unstained tube")

    unstained_cache: Dict[str, np.ndarray] = {}
    summary_rows = []
    for row_number, pair in enumerate(pairs.itertuples(), start=1):
        row = {"stained_file": pair.stained_file, "unstained_file": pair.unstained_file}
        if not isinstance(pair.unstained_file, str):
            print(f"  [{row_number}/{len(pairs)}] SKIPPED {pair.stained_file}: no unstained tube")
            summary_rows.append({**row, "status": "no unstained tube"})
            continue
        if pair.unstained_file not in unstained_cache:
            # Several NM tubes separated by ';' are pooled (a group reference, e.g. NM tubes of other animals).
            unstained_cache[pair.unstained_file] = np.concatenate([
                read_fcs_events(os.path.join(input_directory_path, unstained_name.strip())).channel_values(channel)
                for unstained_name in pair.unstained_file.split(";") if unstained_name.strip()])
        unstained_values = unstained_cache[pair.unstained_file]
        if unstained_values.size == 0:
            print(f"  [{row_number}/{len(pairs)}] SKIPPED {pair.stained_file}: the unstained tube {pair.unstained_file} has no event")
            summary_rows.append({**row, "status": "unstained tube empty"})
            continue
        stained_events = read_fcs_events(os.path.join(input_directory_path, pair.stained_file))
        stained_values = stained_events.channel_values(channel)
        threshold = compute_deep_red_threshold(unstained_values, quantile)
        inside = stained_values > threshold
        alerts = compute_deep_red_alerts(unstained_values, stained_values,
                                         unstained_tail_alert_spread, threshold_sensitivity_alert_points)
        written = _write_gate_outputs(stained_events, inside, planned[pair.stained_file], "DR+", "DR_Gate")
        if save_qc_figures:
            _save_deep_red_qc_figure(stained_values, unstained_values, threshold, alerts, channel, pair.stained_file, pair.unstained_file,
                                     os.path.join(output_directory_path, "qc_figures", f"{fcs_file_stem(pair.stained_file)}_deep_red.png"))
        row.update({
            "status": "ok",
            "threshold": round(threshold, 3),
            "events_in": stained_events.n_events,
            "events_dr_positive": int(inside.sum()),
            "percent_dr_positive": round(100 * float(inside.mean()), 3),
            "percent_unstained_above_threshold": round(100 * float(np.mean(unstained_values > threshold)), 3),
            "unstained_tail_spread": round(alerts["unstained_tail_spread"], 3),
            "threshold_sensitivity_points": round(alerts["threshold_sensitivity_points"], 3),
            "unstained_tail_alert": alerts["unstained_tail_alert"],
            "threshold_sensitivity_alert": alerts["threshold_sensitivity_alert"],
            "output_files": ";".join(written),
        })
        summary_rows.append(row)
        flags = [name for name, raised in (("NM bright tail", alerts["unstained_tail_alert"]), ("threshold sensitivity", alerts["threshold_sensitivity_alert"])) if raised]
        print(f"  [{row_number}/{len(pairs)}] {pair.stained_file} vs {pair.unstained_file}: threshold {threshold:.0f}, "
              f"DR+ {100 * inside.mean():.1f}%" + (f" | ALERT: {', '.join(flags)} → look at this tube" if flags else ""))
        del stained_events, stained_values, inside
        gc.collect()

    summary = pd.DataFrame(summary_rows)
    summary_path = os.path.join(output_directory_path, "deep_red_summary.csv")
    summary.to_csv(summary_path, index=False)
    gated = summary[summary.status == "ok"] if "status" in summary else summary.iloc[0:0]
    print("-" * 70)
    print("QC — DR+")
    print(f"  Stained tubes gated: {len(gated)} / {len(summary)}")
    if len(gated):
        print(f"  % DR+ per tube: median {gated.percent_dr_positive.median():.1f}, min {gated.percent_dr_positive.min():.1f}, "
              f"max {gated.percent_dr_positive.max():.1f}")
        print(f"  % of NM above its own threshold: median {gated.percent_unstained_above_threshold.median():.3f} (expected about {100 * (1 - quantile):g})")
        alerted = gated[gated.unstained_tail_alert.astype(bool) | gated.threshold_sensitivity_alert.astype(bool)]
        print(f"  Tubes with an alert (look at them): {len(alerted)}" + (f" — {list(alerted.stained_file)}" if len(alerted) else ""))
        assert (gated.events_dr_positive <= gated.events_in).all(), "More DR+ events than events"
    print(f"  Summary table: {summary_path}")
    print(summary.head().to_string())
    print("=" * 70)
    return summary
