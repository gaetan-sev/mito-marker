"""
flowai.py

flowAI quality control of spectral flow cytometry acquisitions, ported to Python
from the R code shipped with the FlowJo flowAI plugin (ADR-022).

WHY A PORT AND NOT A CALL TO R

The lab runs flowAI through the FlowJo plugin, by hand, tube by tube. The R code
the plugin runs is shipped with it (RScript.flowAI.*.R: flowAI's functions, the
Christiano–Fitzgerald filter of mFilter). This module follows that code line by
line. On 468 acquisitions processed by the plugin it gives the same good/bad
label for every event in 464 files (mito-marker-lab, EXP-067); the other 4 are
raw files rewritten after the plugin ran. No R installation is needed.

WHAT flowAI CHECKS (Monaco et al., Bioinformatics 2016)

1. Flow rate. The acquisition is cut into 0.1-second bins and the events of each
   bin are counted. A bin whose count jumps away from the others (a clog, a
   bubble) is an anomaly: the count series is split into trend and cycle by the
   Christiano–Fitzgerald filter, then a generalised ESD test (alpha 0.01) flags
   the outlying bins. Every event of a flagged bin is removed, whatever its
   values.
2. Signal acquisition. The events, in time order, are cut into about 100 bins of
   equal size and the median of every channel is taken in each bin. A binary
   segmentation (normal mean and variance, penalty 200, at most 3 changepoints)
   finds where a channel's median changes level. Only the longest stretch
   without a change in any channel is kept.
3. Dynamic range. An event is removed if, in any fluorescence channel (FSC and
   SSC are not checked), its value reaches the top of the range ($PnR - 1,
   saturation) or falls below the median of the channel's negative values minus
   3.5 robust standard deviations of them.

The plugin defaults are this module's defaults. An event passes flowAI when it
passes the three checks (plugin option "remove from: all").

ONE DIFFERENCE WITH THE PLUGIN, ON PURPOSE

flowAI sorts the events by time before checking them, and the plugin writes its
labels in that sorted order. FlowJo then gives label i to event i of the file.
When the Time channel is not increasing (a tube acquired in two runs), FlowJo
therefore puts labels on the wrong events (1–28 % of the events in 8 such files,
EXP-067). This module gives each event its own label. Set
`match_flowjo_plugin_order=True` to reproduce the FlowJo exports exactly.
"""

import gc
import os
from dataclasses import dataclass, field
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
from scipy.stats import t as student_t

from mito_marker.sfc.fcs_io import (
    FcsEvents,
    fcs_file_stem,
    list_fcs_files,
    read_fcs_events,
    write_fcs_events,
)

# ---------------------------------------------------------------- plugin defaults
FLOWAI_CHECKS: tuple[str, ...] = ("flow_rate", "signal_acquisition", "dynamic_range")
DEFAULT_SECOND_FRACTION = 0.1          # width of a flow-rate bin, in seconds
DEFAULT_ESD_ALPHA = 0.01               # significance level of the generalised ESD test
DEFAULT_ESD_MAX_ANOMALY_SHARE = 0.49   # at most 49 % of the bins can be flagged (flowAI max_anoms)
DEFAULT_MAX_CHANGEPOINTS = 3           # per channel, signal acquisition check
DEFAULT_CHANGEPOINT_PENALTY = 200.0    # manual penalty of the binary segmentation
DEFAULT_CHANGEPOINT_MIN_SEGMENT = 2    # changepoint minseglen; each side of a split keeps >= minseglen + 1 bins
DEFAULT_NEGATIVE_OUTLIER_MADS = 3.5    # dynamic range, lower limit
# Label values written in the FlowAI channel, as the FlowJo plugin does
# (good around 2000, bad around 6000), so that the same FlowJo gate 1500–2500 works.
FLOWAI_GOOD_LABEL = 2000.0
FLOWAI_BAD_LABEL = 6000.0
FLOWAI_LABEL_CHANNEL = "FlowAI"
OUTPUT_MODES: tuple[str, ...] = ("kept_events", "all_events_flagged", "both")

R_MAD_SCALE = 1.4826  # constant of R's mad(): makes the MAD a standard deviation for normal data


@dataclass
class FlowAIQualityResult:
    """Outcome of flowAI on one acquisition. Every mask is in the file's event order.

    Attributes:
        flow_rate_pass: True for the events kept by the flow rate check.
        signal_acquisition_pass: True for the events kept by the signal acquisition check.
        dynamic_range_pass: True for the events kept by the dynamic range check.
        overall_pass: True for the events kept by every check that was run.
        checks_run: Names of the checks that were run.
        time_is_increasing: False when the Time channel goes back at least once.
        time_reset_count: Number of times the Time channel goes back.
        flow_rate_bin_counts: Events per flow-rate bin (for the QC figure).
        anomalous_bins: Positions (1-based, flowAI convention) of the flagged bins.
        signal_segment: First and last signal bin of the stretch kept.
        parameters: Parameter values used.
    """

    flow_rate_pass: np.ndarray
    signal_acquisition_pass: np.ndarray
    dynamic_range_pass: np.ndarray
    overall_pass: np.ndarray
    checks_run: List[str]
    time_is_increasing: bool
    time_reset_count: int
    flow_rate_bin_counts: np.ndarray = field(default_factory=lambda: np.zeros(0))
    anomalous_bins: List[int] = field(default_factory=list)
    signal_segment: tuple[int, int] = (0, 0)
    parameters: Dict[str, object] = field(default_factory=dict)

    def percent_removed(self) -> Dict[str, float]:
        """Percentage of events removed by each check and by flowAI as a whole."""
        return {
            "flow_rate": 100.0 * float(np.mean(~self.flow_rate_pass)),
            "signal_acquisition": 100.0 * float(np.mean(~self.signal_acquisition_pass)),
            "dynamic_range": 100.0 * float(np.mean(~self.dynamic_range_pass)),
            "overall": 100.0 * float(np.mean(~self.overall_pass)),
        }


# ---------------------------------------------------------------- shared helpers
def _r_mad(values: np.ndarray) -> float:
    """R's mad(): 1.4826 x median(|x - median(x)|). NaN for an empty vector, like R."""
    if values.size == 0:
        return float("nan")
    return R_MAD_SCALE * float(np.median(np.abs(values - np.median(values))))


# ---------------------------------------------------------------- 1. flow rate
def _flow_rate_bins(time_values: np.ndarray, time_step: float, second_fraction: float) -> tuple[np.ndarray, np.ndarray]:
    """Count the events of each time bin, as flowAI's flow_rate_bin().

    R's hist() uses intervals closed on the right, (a, b], the first one also
    holding its lower limit; flowAI puts a 0 in front of the counts. Returns the
    counts (with the leading 0) and, for every event, its position in that
    vector (1-based, as in R).
    """
    total_seconds = np.ceil(time_step * time_values.max())
    breaks = np.arange(0.0, total_seconds / time_step + 1e-9, second_fraction / time_step)
    interval_of_event = np.searchsorted(breaks, time_values, side="left")
    interval_of_event[interval_of_event == 0] = 1
    bin_count = len(breaks) - 1
    counts = np.bincount(interval_of_event, minlength=bin_count + 1)[1:bin_count + 1]
    return np.concatenate([[0], counts]), interval_of_event + 1


def _christiano_fitzgerald_cycle(series: np.ndarray, lower_period: float = 2.0, upper_period: float = 8.0) -> np.ndarray:
    """Cycle part of the Christiano–Fitzgerald asymmetric filter, as called by flowAI.

    flowAI calls mFilter's cffilter() with its defaults: periods 2 to 8 bins,
    theta = 1, no unit root, no drift. With theta = 1 the filter matrix is the
    symmetric Toeplitz matrix of the ideal band-pass weights B_0, B_1, ...
    """
    length = len(series)
    high_frequency, low_frequency = 2 * np.pi / lower_period, 2 * np.pi / upper_period
    lags = np.arange(1, 2 * length + 1)
    weights = np.concatenate([
        [(high_frequency - low_frequency) / np.pi],
        (np.sin(lags * high_frequency) - np.sin(lags * low_frequency)) / (lags * np.pi),
    ])
    lag_matrix = np.abs(np.subtract.outer(np.arange(length), np.arange(length)))
    return weights[lag_matrix] @ series


def _generalised_esd_anomalies(bin_counts: np.ndarray, alpha: float, max_anomaly_share: float, use_decomposition: bool) -> List[int]:
    """flowAI's anomaly_detection(): bins whose count is an outlier, by generalised ESD.

    Empty bins are left out. The tested value combines the distance of each count
    to the median count and the size of the Christiano–Fitzgerald cycle, signed by
    the side of the trend. At each round the most extreme bin is removed; the
    number of anomalies is the last round whose statistic exceeds the critical
    value lambda (Rosner 1983). Note: flowAI scales by the MAD of the raw counts.
    """
    bin_positions = np.flatnonzero(bin_counts != 0) + 1
    counts = bin_counts[bin_counts != 0].astype(np.float64)
    median_count = np.trunc(np.median(counts))
    if use_decomposition:
        cycle = _christiano_fitzgerald_cycle(counts)
        trend = counts - cycle
        trend_side = np.sign(trend - median_count)
        trend_side[trend_side == 0] = 1
        tested_values = np.trunc(np.abs(counts - median_count) + np.abs(cycle)) * trend_side
    else:
        tested_values = counts - np.median(counts)
    n_bins = len(tested_values)
    remaining_positions, remaining_values, remaining_counts = bin_positions.copy(), tested_values.copy(), counts.copy()
    flagged_positions: List[int] = []
    anomaly_count = 0
    for round_number in range(1, int(np.trunc(n_bins * max_anomaly_share)) + 1):
        spread = _r_mad(remaining_counts)
        if spread == 0:
            break
        scores = np.abs(remaining_values - np.median(remaining_values)) / spread
        most_extreme = int(np.argmax(scores))
        statistic = scores[most_extreme]
        flagged_positions.append(int(remaining_positions[most_extreme]))
        keep = remaining_positions != remaining_positions[most_extreme]
        remaining_positions, remaining_values, remaining_counts = remaining_positions[keep], remaining_values[keep], remaining_counts[keep]
        probability = 1 - alpha / (2 * (n_bins - round_number + 1))
        t_quantile = student_t.ppf(probability, n_bins - round_number - 1)
        critical_value = t_quantile * (n_bins - round_number) / np.sqrt(
            (n_bins - round_number - 1 + t_quantile ** 2) * (n_bins - round_number + 1)
        )
        if statistic > critical_value:
            anomaly_count = round_number
    return flagged_positions[:anomaly_count]


def _flow_rate_check(time_values: np.ndarray, time_step: float, second_fraction: float, alpha: float,
                     max_anomaly_share: float, use_decomposition: bool) -> tuple[np.ndarray, np.ndarray, List[int]]:
    """Mask of the events kept by the flow rate check (events in time order), bin counts, flagged bins."""
    bin_counts, bin_of_event = _flow_rate_bins(time_values, time_step, second_fraction)
    if len(np.unique(bin_counts)) == 1:
        return np.ones(len(time_values), dtype=bool), bin_counts, []
    anomalous_bins = _generalised_esd_anomalies(bin_counts, alpha, max_anomaly_share, use_decomposition)
    return ~np.isin(bin_of_event, anomalous_bins), bin_counts, anomalous_bins


# ---------------------------------------------------------------- 2. signal acquisition
def _normal_meanvar_cost(value_sum: float, square_sum: float, length: int) -> float:
    """Twice the negative log-likelihood of a normal segment with its own mean and variance."""
    variance = (square_sum - value_sum * value_sum / length) / length
    if variance <= 0:
        variance = 1e-11  # changepoint's floor for a constant segment
    return length * (np.log(2 * np.pi) + np.log(variance) + 1)


def _binary_segmentation(series: np.ndarray, penalty: float, max_changepoints: int, min_segment: int) -> List[int]:
    """changepoint::cpt.meanvar(method='BinSeg', test.stat='Normal', penalty='Manual').

    At each round the split that most lowers the cost, over every current
    segment, is added. A changepoint is kept when its cost gain (made
    non-increasing across rounds) reaches the penalty. Each side of a split keeps
    at least min_segment + 1 values (found to reproduce the plugin, EXP-067).
    Returns the last index (1-based) of each kept segment, then the series length.
    """
    length = len(series)
    cumulative_sum = np.concatenate([[0.0], np.cumsum(series)])
    cumulative_square = np.concatenate([[0.0], np.cumsum(series ** 2)])

    def segment_cost(start: int, end: int) -> float:  # half-open [start, end)
        return _normal_meanvar_cost(cumulative_sum[end] - cumulative_sum[start],
                                    cumulative_square[end] - cumulative_square[start], end - start)

    boundaries = [0, length]
    found: List[tuple[int, float]] = []
    previous_gain = np.inf
    for _ in range(max_changepoints):
        best_gain, best_split = -np.inf, None
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            whole_cost = segment_cost(start, end)
            for split in range(start + min_segment + 1, end - min_segment):
                gain = whole_cost - segment_cost(start, split) - segment_cost(split, end)
                if gain > best_gain:
                    best_gain, best_split = gain, split
        if best_split is None:
            break
        previous_gain = min(previous_gain, best_gain)
        found.append((best_split, previous_gain))
        boundaries = sorted(boundaries + [best_split])
    return sorted(split for split, gain in found if gain >= penalty) + [length]


def _signal_acquisition_check(event_matrix: np.ndarray, penalty: float, max_changepoints: int, min_segment: int) -> tuple[np.ndarray, tuple[int, int]]:
    """Mask of the events kept by the signal acquisition check (events in time order)."""
    n_events = event_matrix.shape[0]
    bin_size = int(min(max(1, np.ceil(n_events / 100)), 500))
    full_bin_count = n_events // bin_size
    bin_of_event = np.concatenate([
        np.repeat(np.arange(1, full_bin_count + 1), bin_size),
        np.full(n_events - full_bin_count * bin_size, full_bin_count + 1),
    ])
    bin_ids = np.unique(bin_of_event)
    bin_medians = np.vstack([np.median(event_matrix[bin_of_event == bin_id], axis=0) for bin_id in bin_ids])
    changepoints = {1}
    for channel_index in range(bin_medians.shape[1]):
        changepoints.update(_binary_segmentation(bin_medians[:, channel_index], penalty, max_changepoints, min_segment))
    ordered = np.array(sorted(changepoints))
    gaps = np.diff(ordered)
    longest = np.flatnonzero(gaps == gaps.max())
    # flowAI keeps c(starts, ends) of every longest stretch and uses its first two
    # numbers; with a tie this spans from the first tie to the start of the second.
    segment_bounds = np.concatenate([ordered[longest], ordered[longest + 1]])
    first_bin, last_bin = int(segment_bounds[0]), int(segment_bounds[1])
    return (bin_of_event >= first_bin) & (bin_of_event <= last_bin), (first_bin, last_bin)


# ---------------------------------------------------------------- 3. dynamic range
def _dynamic_range_check(event_matrix: np.ndarray, channel_names: Sequence[str], range_maximum: np.ndarray,
                         side: str, negative_outlier_mads: float) -> np.ndarray:
    """Mask of the events kept by the dynamic range check (FSC and SSC channels excluded, flowAI default)."""
    removed = np.zeros(event_matrix.shape[0], dtype=bool)
    for channel_index, channel_name in enumerate(channel_names):
        if "FSC" in channel_name or "SSC" in channel_name:
            continue
        values = event_matrix[:, channel_index]
        if side in ("lower", "both"):
            negative_values = values[values < 0]
            # 0.6745 converts the MAD into the scale flowAI uses (its own convention, kept as is).
            lower_limit = -negative_outlier_mads * _r_mad(negative_values) / 0.6745 + (np.median(negative_values) if negative_values.size else np.nan)
            if np.isnan(lower_limit):
                lower_limit = values.min() - 1
            removed |= values <= lower_limit
        if side in ("upper", "both"):
            removed |= values >= range_maximum[channel_index]
    return ~removed


# ---------------------------------------------------------------- public API
def compute_flowai_quality(
    fcs_events: FcsEvents,
    checks: Sequence[str] = FLOWAI_CHECKS,
    second_fraction: float = DEFAULT_SECOND_FRACTION,
    alpha: float = DEFAULT_ESD_ALPHA,
    use_decomposition: bool = True,
    max_changepoints: int = DEFAULT_MAX_CHANGEPOINTS,
    changepoint_penalty: float = DEFAULT_CHANGEPOINT_PENALTY,
    dynamic_range_side: str = "both",
    time_channel: str = "Time",
    match_flowjo_plugin_order: bool = False,
) -> FlowAIQualityResult:
    """Run the flowAI checks on one acquisition, as the FlowJo plugin does with its defaults.

    Args:
        fcs_events: Raw acquisition from read_fcs_events() (every channel, file order).
        checks: Checks to run, any of 'flow_rate', 'signal_acquisition', 'dynamic_range'.
            An event passes when it passes every check that was run.
        second_fraction: Width of a flow-rate bin, in seconds (plugin: 0.1).
        alpha: Significance level of the generalised ESD test (plugin: 0.01).
        use_decomposition: Decompose the flow rate with the Christiano–Fitzgerald filter (plugin: True).
        max_changepoints: Most changepoints per channel in the signal check (plugin: 3).
        changepoint_penalty: Penalty of the binary segmentation (plugin: 200).
        dynamic_range_side: 'both', 'upper' or 'lower' (plugin: 'both').
        time_channel: Name of the time channel.
        match_flowjo_plugin_order: True gives event i the plugin's label i, as FlowJo does.
            This only differs from the default when the Time channel is not increasing,
            and it then labels the wrong events (see the module docstring).

    Returns:
        FlowAIQualityResult with one mask per check and the overall mask, in file order.
    """
    unknown_checks = set(checks) - set(FLOWAI_CHECKS)
    if unknown_checks:
        raise ValueError(f"Unknown flowAI checks {sorted(unknown_checks)}; choose from {FLOWAI_CHECKS}")
    if dynamic_range_side not in ("both", "upper", "lower"):
        raise ValueError(f"dynamic_range_side must be 'both', 'upper' or 'lower', got '{dynamic_range_side}'")

    raw_time_values = fcs_events.channel_values(time_channel)
    # flowCore's read.FCS() cuts every channel at $PnR - 1, Time included, before flowAI sees it;
    # some Aurora files hold Time values above their declared range (EXP-067: 0.5-1 % of a tube).
    time_range = fcs_events.channel_ranges()[fcs_events.channel_names.index(time_channel)]
    time_values = np.minimum(raw_time_values, time_range - 1) if not np.isnan(time_range) else raw_time_values
    time_step = float(fcs_events.text_keywords.get("timestep", "0.01"))
    analysed_indices = [index for index, name in enumerate(fcs_events.channel_names) if name != time_channel]
    analysed_names = [fcs_events.channel_names[index] for index in analysed_indices]
    # flowCore's read.FCS() cuts values at the top of the range ($PnR - 1) before flowAI sees them.
    range_maximum = fcs_events.channel_ranges()[analysed_indices] - 1
    implausible = [name for name, top, values in zip(analysed_names, range_maximum, fcs_events.event_matrix[:, analysed_indices].T)
                   if values.size and top + 1 < values.max()]
    if implausible:
        # flowAI (and the plugin) trust $PnR; a range below the data means the file is not a raw
        # acquisition (e.g. a FlowJo export with $PnR = 1), and the dynamic range check will remove almost everything.
        print(f"  WARNING: {os.path.basename(fcs_events.source_file_path)}: $PnR below the largest value for {implausible[:5]}"
              f"{'...' if len(implausible) > 5 else ''}; run flowAI on raw acquisition files")
    analysed_matrix = np.minimum(fcs_events.event_matrix[:, analysed_indices], range_maximum)

    time_order = np.argsort(time_values, kind="stable")
    sorted_time = time_values[time_order]
    sorted_matrix = analysed_matrix[time_order]
    n_events = fcs_events.n_events
    all_true = np.ones(n_events, dtype=bool)

    flow_rate_sorted, bin_counts, anomalous_bins = all_true, np.zeros(0), []
    if "flow_rate" in checks:
        flow_rate_sorted, bin_counts, anomalous_bins = _flow_rate_check(
            sorted_time, time_step, second_fraction, alpha, DEFAULT_ESD_MAX_ANOMALY_SHARE, use_decomposition)
    signal_sorted, segment = all_true, (0, 0)
    if "signal_acquisition" in checks:
        signal_sorted, segment = _signal_acquisition_check(
            sorted_matrix, changepoint_penalty, max_changepoints, DEFAULT_CHANGEPOINT_MIN_SEGMENT)
    dynamic_sorted = all_true
    if "dynamic_range" in checks:
        dynamic_sorted = _dynamic_range_check(sorted_matrix, analysed_names, range_maximum,
                                              dynamic_range_side, DEFAULT_NEGATIVE_OUTLIER_MADS)

    def to_file_order(mask_in_time_order: np.ndarray) -> np.ndarray:
        if match_flowjo_plugin_order:
            return mask_in_time_order.copy()
        mask_in_file_order = np.empty(n_events, dtype=bool)
        mask_in_file_order[time_order] = mask_in_time_order
        return mask_in_file_order

    flow_rate_pass = to_file_order(flow_rate_sorted)
    signal_pass = to_file_order(signal_sorted)
    dynamic_pass = to_file_order(dynamic_sorted)
    reset_count = int(np.sum(np.diff(raw_time_values) < 0))
    return FlowAIQualityResult(
        flow_rate_pass=flow_rate_pass,
        signal_acquisition_pass=signal_pass,
        dynamic_range_pass=dynamic_pass,
        overall_pass=flow_rate_pass & signal_pass & dynamic_pass,
        checks_run=list(checks),
        time_is_increasing=reset_count == 0,
        time_reset_count=reset_count,
        flow_rate_bin_counts=bin_counts,
        anomalous_bins=anomalous_bins,
        signal_segment=segment,
        parameters={"checks": list(checks), "second_fraction": second_fraction, "alpha": alpha,
                    "use_decomposition": use_decomposition, "max_changepoints": max_changepoints,
                    "changepoint_penalty": changepoint_penalty, "dynamic_range_side": dynamic_range_side,
                    "match_flowjo_plugin_order": match_flowjo_plugin_order},
    )


def flowai_label_channel(quality_result: FlowAIQualityResult) -> np.ndarray:
    """FlowAI channel values as the plugin writes them: 2000 for a good event, 6000 for a bad one."""
    return np.where(quality_result.overall_pass, FLOWAI_GOOD_LABEL, FLOWAI_BAD_LABEL)


def _save_flowai_qc_figure(quality_result: FlowAIQualityResult, fcs_events: FcsEvents, figure_path: str) -> None:
    """One QC figure per tube: events per time bin with the flagged bins, and what each check removed."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, (rate_axis, bar_axis) = plt.subplots(1, 2, figsize=(11, 3.6), gridspec_kw={"width_ratios": [3, 1]})
    counts = quality_result.flow_rate_bin_counts
    if counts.size:
        seconds = np.arange(len(counts)) * quality_result.parameters["second_fraction"]
        rate_axis.plot(seconds, counts, color="0.35", linewidth=0.8)
        flagged = np.array(quality_result.anomalous_bins, dtype=int) - 1
        if flagged.size:
            rate_axis.scatter(seconds[flagged], counts[flagged], color="tab:red", s=10, label="anomalous bin (removed)")
            rate_axis.legend(loc="upper right", fontsize=8)
    rate_axis.set_xlabel("Acquisition time (s)")
    rate_axis.set_ylabel(f"Events per {quality_result.parameters['second_fraction']} s")
    rate_axis.set_title("Flow rate: a red bin is a jump in the event rate (clog, bubble)", fontsize=9)
    percent = quality_result.percent_removed()
    names = ["flow rate", "signal\nacquisition", "dynamic\nrange", "overall"]
    values = [percent["flow_rate"], percent["signal_acquisition"], percent["dynamic_range"], percent["overall"]]
    bar_axis.bar(names, values, color=["0.6", "0.6", "0.6", "tab:red"])
    for position, value in enumerate(values):
        bar_axis.text(position, value, f"{value:.1f}%", ha="center", va="bottom", fontsize=8)
    bar_axis.set_ylabel("% of events removed")
    bar_axis.set_title("Removed by each check\n(an event can fail several)", fontsize=9)
    bar_axis.tick_params(axis="x", labelsize=7)
    figure.suptitle(f"flowAI — {os.path.basename(fcs_events.source_file_path)} ({fcs_events.n_events} events)", fontsize=10)
    figure.tight_layout()
    figure.savefig(figure_path, dpi=110)
    plt.close(figure)


def run_flowai(
    input_directory_path: str,
    output_directory_path: str,
    file_pattern: str = "*.fcs",
    checks: Sequence[str] = FLOWAI_CHECKS,
    output_mode: str = "kept_events",
    second_fraction: float = DEFAULT_SECOND_FRACTION,
    alpha: float = DEFAULT_ESD_ALPHA,
    use_decomposition: bool = True,
    max_changepoints: int = DEFAULT_MAX_CHANGEPOINTS,
    changepoint_penalty: float = DEFAULT_CHANGEPOINT_PENALTY,
    dynamic_range_side: str = "both",
    match_flowjo_plugin_order: bool = False,
    save_qc_figures: bool = True,
    overwrite: bool = False,
) -> pd.DataFrame:
    """Run flowAI on every raw .fcs file of a folder and write the results as .fcs files.

    Each tube is read, checked (compute_flowai_quality()) and written to
    output_directory_path, the input files are never modified:
      - output_mode 'kept_events': '<tube>_FlowAIGoodEvents.fcs' with the good events only,
        every channel, plus a 'FlowAI' channel (2000), like a FlowJo export of the plugin's
        good population;
      - 'all_events_flagged': '<tube>_FlowAIFlagged.fcs' with every event and the 'FlowAI' channel
        (2000 good, 6000 bad), to look at the removed events in FlowJo;
      - 'both': the two files.
    A table 'flowai_summary.csv' (one row per tube) is written too, and one QC figure
    per tube in 'qc_figures/' when save_qc_figures is True.

    Args:
        input_directory_path: Folder of raw .fcs files (not recursive).
        output_directory_path: Folder for the outputs (created if missing).
        file_pattern: Glob pattern selecting the files, e.g. 'WORM_*.fcs'.
        checks: flowAI checks to run (see compute_flowai_quality()).
        output_mode: 'kept_events', 'all_events_flagged' or 'both'.
        second_fraction, alpha, use_decomposition, max_changepoints, changepoint_penalty,
        dynamic_range_side, match_flowjo_plugin_order: see compute_flowai_quality().
        save_qc_figures: Write one PNG per tube.
        overwrite: Replace existing output files; False raises before writing anything.

    Returns:
        DataFrame, one row per tube: file names, events in and kept, % removed per check,
        Time channel status, output file(s).
    """
    if output_mode not in OUTPUT_MODES:
        raise ValueError(f"output_mode must be one of {OUTPUT_MODES}, got '{output_mode}'")
    input_files = list_fcs_files(input_directory_path, file_pattern)
    os.makedirs(output_directory_path, exist_ok=True)
    if save_qc_figures:
        os.makedirs(os.path.join(output_directory_path, "qc_figures"), exist_ok=True)

    print("=" * 70)
    print("flowAI quality control (Python port of the FlowJo plugin, ADR-022)")
    print("=" * 70)
    print(f"  Input folder : {input_directory_path}")
    print(f"  Output folder: {output_directory_path}")
    print(f"  Files        : {len(input_files)} matching '{file_pattern}'")
    print(f"  Checks       : {list(checks)}")
    print(f"  Parameters   : second_fraction={second_fraction}, alpha={alpha}, decomposition={use_decomposition}, "
          f"max_changepoints={max_changepoints}, penalty={changepoint_penalty}, side='{dynamic_range_side}'")
    print(f"  Output mode  : {output_mode}; FlowJo plugin label order: {match_flowjo_plugin_order}")
    if not input_files:
        raise FileNotFoundError(f"No .fcs file matching '{file_pattern}' in {input_directory_path}")

    planned_outputs = {}
    for input_file in input_files:
        stem = fcs_file_stem(input_file)
        outputs = []
        if output_mode in ("kept_events", "both"):
            outputs.append(os.path.join(output_directory_path, f"{stem}_FlowAIGoodEvents.fcs"))
        if output_mode in ("all_events_flagged", "both"):
            outputs.append(os.path.join(output_directory_path, f"{stem}_FlowAIFlagged.fcs"))
        planned_outputs[input_file] = outputs
    existing = [path for outputs in planned_outputs.values() for path in outputs if os.path.exists(path)]
    if existing and not overwrite:
        raise FileExistsError(f"{len(existing)} output file(s) already exist (e.g. {existing[0]}); pass overwrite=True to replace them")

    summary_rows = []
    for file_number, input_file in enumerate(input_files, start=1):
        row = {"input_file": os.path.basename(input_file)}
        try:
            fcs_events = read_fcs_events(input_file)
        except ValueError as read_error:
            print(f"  [{file_number}/{len(input_files)}] SKIPPED {os.path.basename(input_file)}: {read_error}")
            summary_rows.append({**row, "status": f"unreadable: {read_error}"})
            continue
        quality = compute_flowai_quality(
            fcs_events, checks=checks, second_fraction=second_fraction, alpha=alpha,
            use_decomposition=use_decomposition, max_changepoints=max_changepoints,
            changepoint_penalty=changepoint_penalty, dynamic_range_side=dynamic_range_side,
            match_flowjo_plugin_order=match_flowjo_plugin_order)
        labels = flowai_label_channel(quality)
        written = []
        for output_path in planned_outputs[input_file]:
            mask = quality.overall_pass if output_path.endswith("_FlowAIGoodEvents.fcs") else None
            written.append(os.path.basename(write_fcs_events(fcs_events, output_path, event_mask=mask,
                                                             extra_channels={FLOWAI_LABEL_CHANNEL: labels})))
        if save_qc_figures:
            _save_flowai_qc_figure(quality, fcs_events, os.path.join(output_directory_path, "qc_figures", f"{fcs_file_stem(input_file)}_flowai.png"))
        percent = quality.percent_removed()
        row.update({
            "status": "ok",
            "events_in": fcs_events.n_events,
            "events_kept": int(quality.overall_pass.sum()),
            "percent_removed": round(percent["overall"], 3),
            "percent_removed_flow_rate": round(percent["flow_rate"], 3),
            "percent_removed_signal_acquisition": round(percent["signal_acquisition"], 3),
            "percent_removed_dynamic_range": round(percent["dynamic_range"], 3),
            "time_is_increasing": quality.time_is_increasing,
            "time_reset_count": quality.time_reset_count,
            "output_files": ";".join(written),
        })
        summary_rows.append(row)
        time_note = "" if quality.time_is_increasing else f" | WARNING: Time goes back {quality.time_reset_count}x (tube acquired in several runs?)"
        print(f"  [{file_number}/{len(input_files)}] {os.path.basename(input_file)}: {fcs_events.n_events} events, "
              f"kept {row['events_kept']} ({100 - percent['overall']:.1f}%) | removed: flow rate {percent['flow_rate']:.1f}%, "
              f"signal {percent['signal_acquisition']:.1f}%, dynamic range {percent['dynamic_range']:.1f}%{time_note}")
        del fcs_events, quality, labels
        gc.collect()

    summary = pd.DataFrame(summary_rows)
    summary_path = os.path.join(output_directory_path, "flowai_summary.csv")
    summary.to_csv(summary_path, index=False)

    # ---- QC block
    processed = summary[summary.status == "ok"]
    print("-" * 70)
    print("QC — flowAI")
    print(f"  Tubes processed: {len(processed)} / {len(summary)} (skipped: {len(summary) - len(processed)})")
    if len(processed):
        print(f"  Events in: {int(processed.events_in.sum())}, kept: {int(processed.events_kept.sum())}")
        print(f"  % removed per tube: median {processed.percent_removed.median():.1f}, "
              f"min {processed.percent_removed.min():.1f}, max {processed.percent_removed.max():.1f}")
        assert (processed.events_kept <= processed.events_in).all(), "More events kept than read"
        not_increasing = processed[~processed.time_is_increasing.astype(bool)]
        if len(not_increasing):
            print(f"  WARNING: {len(not_increasing)} tube(s) with a Time channel that goes back: {list(not_increasing.input_file)}")
    print(f"  Summary table: {summary_path}")
    print(summary.head().to_string())
    print("=" * 70)
    return summary
