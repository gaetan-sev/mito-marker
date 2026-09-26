"""
channel_intensity_plot.py

Per-sample fluorescence intensity profiles across SFC channels.

Draws one series per sample (one SFC / FCS file) with the channels on the X-axis
and the fluorescence intensity on the Y-axis, as a line plot or as a bar plot.
This is the plot to reach for when comparing the spectral signature of individual
samples side by side — for example to spot a file whose violet detectors sit far
above the rest of the cohort.

Why a statistic is involved
---------------------------
``.X`` holds one row per mitochondrion, not one row per sample: a single FCS file
contributes tens of thousands of rows.  For one (sample, channel) pair there is
therefore not one pulse value but tens of thousands of them.  Since the plot draws
a single point per (sample, channel), those events must be collapsed into one
number — that is what ``statistic`` controls (``"mean"`` by default, ``"median"``
for the outlier-robust value usually reported as "MFI" in cytometry).

Channel selection
-----------------
The ``channels`` argument accepts three token forms, resolved in this order:

  1. An exact channel name          → that single channel        ``"V1-A"``
  2. ``"{LASER}-{PULSE}"``          → every channel of that laser with that
     that is not an exact name         pulse, as separate X ticks   ``"V-A"``
  3. A bare laser name              → every channel of that laser with pulse
                                       ``pulse_type``               ``"V"``

Group tokens **expand** the X-axis; they never average channels together.  Use
``aggregate_sfc_channels_by_color()`` when you do want laser-group means.

Note: this module parses ``SSC-B-A`` / ``SSC-B-H`` / ``SSC-B-W`` as side-scatter
channels, whereas ``channel_aggregation.py`` drops them with a warning because its
regex does not allow a second hyphen.  The two modules therefore treat SSC-B
differently by design; that is not a bug.

Typical usage::

    from mito_marker import plot_channel_intensity_line, plot_channel_intensity_bar

    # All violet and yellow-green area channels, one line per FCS file
    plot_channel_intensity_line(sfc_anndata, channels=["V-A", "YG-A"])

    # Hand-picked channels, three samples only, as bars
    plot_channel_intensity_bar(
        sfc_anndata,
        channels=["V1-A", "V2-A", "YG3-A"],
        samples=["good_events_MNMS_020_FlowAIGoodEvents_DeepRed.fcs"],
    )

    # Height pulses instead of area
    plot_channel_intensity_line(sfc_anndata, channels=["V-H", "YG-H"])
"""

from math import ceil
from typing import Dict, List, Optional, Tuple

import anndata
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from mito_marker.analysis._plot_context import (
    get_run_context_console_text,
    get_run_context_footer_text,
    get_species_label,
)

# Imported rather than copied: this dict already exists twice (here and in
# radar_plot.py).  Adding a third copy would make future edits even harder.
from mito_marker.analysis.channel_aggregation import _LASER_GROUP_ORDER
from mito_marker.analysis.colors import get_subject_colors, sort_values_for_legend
from mito_marker.controlled_vocabulary import PREFERRED_CONDITION_COLORS

_ANALYSIS_CONFIG_KEY = "analysis_config"

_ALLOWED_PULSE_TYPES = ("A", "H", "W")
_ALLOWED_STATISTICS = ("mean", "median")

# Cluster of grouped bars occupies 80% of the tick spacing, leaving a constant
# 20% gap between adjacent channels whatever the number of samples.
_BAR_CLUSTER_WIDTH = 0.8

# Above this many samples grouped bars become hairlines and the figure stops
# being readable — the user is told, but never blocked.
_MAX_READABLE_BAR_SAMPLES = 15

# Above this many X ticks the labels only fit when rotated vertically.
_ROTATE_LABELS_ABOVE_N_TICKS = 20


def plot_channel_intensity_line(
    anndata_object: anndata.AnnData,
    channels: Optional[List[str]] = None,
    pulse_type: str = "A",
    sample_column: str = "source_filename",
    samples: Optional[List[str]] = None,
    statistic: str = "mean",
    show_error_bars: bool = False,
    log_scale: bool = False,
    title: str = "",
) -> None:
    """
    Plot the per-sample fluorescence intensity profile across channels as lines.

    One line is drawn per sample.  Each point is the aggregate (mean or median)
    of all that sample's mitochondria for one channel, optionally with a
    vertical +/-1 std cap.

    The active layer is read from .uns['analysis_config']['active_layer'].
    When active_layer is None, raw .X is used.

    Arguments:
        anndata_object: SFC AnnData after optional normalization and feature
                        selection.
        channels: Channels to place on the X-axis.  Each entry is either an
                  exact channel name (``"V1-A"``), a laser-and-pulse group token
                  (``"V-A"``, ``"YG-H"``) which expands to every matching channel
                  as separate X ticks, or a bare laser name (``"V"``) which
                  expands using ``pulse_type``.  Group tokens never average
                  channels together.  The caller's order is preserved and
                  duplicates are removed.  When None, every analytical channel
                  matching ``pulse_type`` is used.
        pulse_type: Pulse measurement to use — ``"A"`` (area, the default),
                    ``"H"`` (height) or ``"W"`` (width).  It applies only to
                    tokens that do not state a pulse themselves, i.e. bare laser
                    names and ``channels=None``.  A token that already carries a
                    pulse always wins, so ``channels=["V-A", "V-H"]`` mixes both.
        sample_column: .obs column identifying one sample.  Defaults to
                       ``"source_filename"``, the unique per-FCS-file key.
        samples: Values of ``sample_column`` to draw.  When None, every sample
                 present is drawn.  The list order fixes the legend order.
        statistic: How the events of one sample are collapsed into one point —
                   ``"mean"`` (default) or ``"median"``.  Fluorescence is
                   log-normally distributed, so a few very bright mitochondria
                   pull the mean above the population centre; ``"median"`` is the
                   robust value usually reported as "MFI" in cytometry.
        show_error_bars: When True, a +/-1 std cap over the sample's events is
                         drawn on each point.  Defaults to False: with many
                         channels or many samples the caps overlap and hide the
                         profile shape, which is what this plot is read for.
        log_scale: When True, the Y-axis uses a logarithmic scale.  Useful on raw
                   .X, where intensities span several orders of magnitude and a
                   linear axis flattens every dim laser.  Leave it False when an
                   arcsinh / z-score layer is active — those values are already
                   compressed and may be negative.
        title: Optional figure title.  Defaults to an auto-generated description.

    Returns:
        None.  The figure is displayed via plt.show().

    Raises:
        ValueError: If ``pulse_type`` or ``statistic`` is invalid, if
                    ``sample_column`` is not in .obs, if a ``channels`` token is
                    neither a channel name nor a laser group, if a token matches
                    no channel, if a requested sample is absent, or if the
                    AnnData holds no SFC channels at all.
    """
    _render_channel_intensity(
        anndata_object=anndata_object,
        channels=channels,
        pulse_type=pulse_type,
        sample_column=sample_column,
        samples=samples,
        statistic=statistic,
        show_error_bars=show_error_bars,
        log_scale=log_scale,
        title=title,
        plot_kind="line",
        function_name="plot_channel_intensity_line",
    )


def plot_channel_intensity_bar(
    anndata_object: anndata.AnnData,
    channels: Optional[List[str]] = None,
    pulse_type: str = "A",
    sample_column: str = "source_filename",
    samples: Optional[List[str]] = None,
    statistic: str = "mean",
    show_error_bars: bool = False,
    log_scale: bool = False,
    title: str = "",
) -> None:
    """
    Plot the per-sample fluorescence intensity profile across channels as bars.

    Identical to plot_channel_intensity_line() in every respect except the mark:
    channels are grouped clusters on the X-axis and each sample contributes one
    coloured bar per cluster.  Bars become very thin above roughly 15 samples;
    a console note then suggests the line variant or a ``samples`` subset.

    Arguments:
        anndata_object: SFC AnnData after optional normalization and feature
                        selection.
        channels: Channels to place on the X-axis.  See
                  plot_channel_intensity_line() for the accepted token forms.
        pulse_type: ``"A"`` (default), ``"H"`` or ``"W"``.
        sample_column: .obs column identifying one sample.  Defaults to
                       ``"source_filename"``.
        samples: Values of ``sample_column`` to draw.  When None, all are drawn.
        statistic: ``"mean"`` (default) or ``"median"``.
        show_error_bars: When True, a +/-1 std cap is drawn on each bar.
                         Defaults to False.
        log_scale: When True, the Y-axis uses a logarithmic scale.
        title: Optional figure title.  Defaults to an auto-generated description.

    Returns:
        None.  The figure is displayed via plt.show().

    Raises:
        ValueError: Same conditions as plot_channel_intensity_line().
    """
    _render_channel_intensity(
        anndata_object=anndata_object,
        channels=channels,
        pulse_type=pulse_type,
        sample_column=sample_column,
        samples=samples,
        statistic=statistic,
        show_error_bars=show_error_bars,
        log_scale=log_scale,
        title=title,
        plot_kind="bar",
        function_name="plot_channel_intensity_bar",
    )


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _parse_channel_name(channel_name: str) -> Optional[Tuple[str, str, int]]:
    """
    Split an SFC channel name into (laser, pulse, detector_number).

    Examples::

        "V1-A"     -> ("V",   "A", 1)
        "UV16-H"   -> ("UV",  "H", 16)
        "YG10-W"   -> ("YG",  "W", 10)
        "FSC-A"    -> ("FSC", "A", 0)     # scatter channels have no detector number
        "SSC-B-A"  -> ("SSC", "A", 0)     # second side-scatter detector
        "Time"     -> None
        "Mito_Area"-> None

    The laser is read as the leading alphabetic run of the name, never with
    ``startswith``.  That removes the "UV vs V" prefix-collision problem by
    construction: "UV1-A" parses to laser "UV", which simply is not "V".

    Arguments:
        channel_name: A value taken from .var_names.

    Returns:
        The (laser, pulse, detector_number) triple, or None when the name is not
        an SFC channel (no recognised pulse suffix).
    """
    head, separator, pulse = channel_name.rpartition("-")
    if not separator:
        return None
    pulse = pulse.upper()
    if pulse not in _ALLOWED_PULSE_TYPES:
        return None

    # "SSC-B" keeps only "SSC": the extra token is a detector label, not a laser.
    first_token = head.split("-")[0]
    laser = first_token.rstrip("0123456789").upper()
    if not laser:
        return None

    detector_digits = first_token[len(laser):]
    detector_number = int(detector_digits) if detector_digits.isdigit() else 0
    return laser, pulse, detector_number


def _get_data_and_channels(
    anndata_object: anndata.AnnData,
    function_name: str,
) -> Tuple[np.ndarray, List[str], str]:
    """
    Return (data_matrix, channel_names, layer_label) for the active layer.

    Follows the same resolution chain as radar_plot, time_curve, pca_plot and
    umap_plot: active_layer -> feature selection -> non-analytical exclusion.

    Arguments:
        anndata_object: AnnData carrying .uns['analysis_config'].
        function_name: Name used to prefix console messages.

    Returns:
        The dense float32 matrix restricted to usable channels, the matching
        channel names, and the human-readable layer label used in axis titles.
    """
    analysis_config = anndata_object.uns.get(_ANALYSIS_CONFIG_KEY, {})
    active_layer = analysis_config.get("active_layer")
    active_selection = analysis_config.get("active_selection")

    if active_layer is not None and active_layer in anndata_object.layers:
        full_matrix = anndata_object.layers[active_layer]
        layer_label = active_layer
    else:
        full_matrix = anndata_object.X
        layer_label = "raw .X"

    if hasattr(full_matrix, "toarray"):
        full_matrix = full_matrix.toarray()
    full_matrix = np.asarray(full_matrix, dtype=np.float32)

    print(f"[{function_name}] Using data from: '{layer_label}'")

    if active_selection is not None:
        selection_column = f"is_selected_{active_selection}"
        if selection_column in anndata_object.var.columns:
            selection_mask = anndata_object.var[selection_column].values.astype(bool)
            channel_names = anndata_object.var_names[selection_mask].tolist()
            print(
                f"[{function_name}] Feature selection active ({active_selection}): "
                f"{len(channel_names)} of {anndata_object.n_vars} channels available."
            )
            return full_matrix[:, selection_mask], channel_names, layer_label

    if "is_non_analytical" in anndata_object.var.columns:
        analytical_mask = ~anndata_object.var["is_non_analytical"].values.astype(bool)
        channel_names = anndata_object.var_names[analytical_mask].tolist()
        return full_matrix[:, analytical_mask], channel_names, layer_label

    return full_matrix, anndata_object.var_names.tolist(), layer_label


def _resolve_channels(
    available_channels: List[str],
    channels: Optional[List[str]],
    pulse_type: str,
    function_name: str,
) -> List[str]:
    """
    Expand the ``channels`` tokens into the ordered list of channels to plot.

    Resolution per token, in order:
      1. An exact entry of ``available_channels`` -> that single channel.  This
         rule matters for "FSC-A" and "SSC-A", which are real channel names even
         though they look like laser-group tokens.
      2. "{LASER}-{PULSE}" -> every channel of that laser with that pulse.
      3. A bare laser name -> every channel of that laser with ``pulse_type``.

    Expanded channels are sorted by detector number numerically, so "V2-A"
    precedes "V10-A".  The caller's token order is preserved across tokens and
    duplicates are dropped, keeping the first occurrence.

    Arguments:
        available_channels: Channel names surviving the layer / selection chain.
        channels: The caller's token list, or None for "all channels of
                  ``pulse_type``".
        pulse_type: Pulse used for bare laser tokens and for ``channels=None``.
        function_name: Name used in error messages and console notes.

    Returns:
        The ordered, de-duplicated list of channel names to place on the X-axis.

    Raises:
        ValueError: If no SFC channel exists at all, if a token is unknown, or
                    if a token matches no channel.
    """
    parsed_channels: Dict[str, Tuple[str, str, int]] = {}
    unparsed: List[str] = []
    for channel_name in available_channels:
        parsed = _parse_channel_name(channel_name)
        if parsed is None:
            unparsed.append(channel_name)
        else:
            parsed_channels[channel_name] = parsed

    if not parsed_channels:
        raise ValueError(
            f"No spectral flow cytometry channels were found. {function_name}() expects "
            "an SFC AnnData whose channel names end in '-A', '-H' or '-W' "
            "(for example 'V1-A'). The first channel names found were: "
            f"{available_channels[:5]}. For TEM data use plot_radar() or "
            "plot_histogram() instead."
        )

    if unparsed:
        print(
            f"[{function_name}] NOTE: {len(unparsed)} channel name(s) are not SFC "
            f"channels and were ignored: {unparsed[:5]}"
        )

    def _channels_for_group(laser: str, pulse: str) -> List[str]:
        """Channel names of one laser and pulse, ordered by detector number."""
        matching = [
            (detector_number, name)
            for name, (channel_laser, channel_pulse, detector_number) in parsed_channels.items()
            if channel_laser == laser and channel_pulse == pulse
        ]
        return [name for _, name in sorted(matching)]

    if channels is None:
        resolved = [
            name
            for name, (_, channel_pulse, _) in parsed_channels.items()
            if channel_pulse == pulse_type
        ]
        if not resolved:
            present_pulses = sorted({pulse for _, pulse, _ in parsed_channels.values()})
            raise ValueError(
                f"No channel with pulse type '{pulse_type}' was found. "
                f"Pulse types present: {present_pulses}."
            )
        return resolved

    known_lasers = list(_LASER_GROUP_ORDER.keys())
    resolved: List[str] = []

    for token in channels:
        # Rule 1 — an exact channel name always wins over group expansion.
        if token in parsed_channels:
            expanded = [token]
        else:
            token_parsed = _parse_channel_name(token)
            if token_parsed is not None and token_parsed[0] in _LASER_GROUP_ORDER:
                # Rule 2 — "{LASER}-{PULSE}" such as "V-A" or "YG-H".
                laser, pulse, _ = token_parsed
                expanded = _channels_for_group(laser, pulse)
            elif token.upper() in _LASER_GROUP_ORDER:
                # Rule 3 — a bare laser name such as "V"; pulse comes from pulse_type.
                laser, pulse = token.upper(), pulse_type
                expanded = _channels_for_group(laser, pulse)
            else:
                raise ValueError(
                    f"'{token}' is neither a channel name nor a laser group. "
                    f"Laser groups: {known_lasers}. "
                    f"Example channel names: {list(parsed_channels.keys())[:5]}"
                )

            if not expanded:
                same_laser = sorted(
                    name
                    for name, (channel_laser, _, _) in parsed_channels.items()
                    if channel_laser == laser
                )
                raise ValueError(
                    f"Laser group '{token}' matched no channel. Channels available "
                    f"for laser '{laser}': {same_laser if same_laser else 'none'}"
                )

        for channel_name in expanded:
            if channel_name not in resolved:
                resolved.append(channel_name)

    return resolved


def _resolve_samples(
    anndata_object: anndata.AnnData,
    sample_column: str,
    samples: Optional[List[str]],
) -> Tuple[pd.Series, List[str]]:
    """
    Return the string-cast sample column and the ordered list of samples to draw.

    Arguments:
        anndata_object: AnnData whose .obs holds ``sample_column``.
        sample_column: The .obs column identifying one sample.
        samples: The caller's subset, or None for every sample present.

    Returns:
        (sample_series, sample_values) where sample_series is .obs[sample_column]
        cast to str and sample_values is the ordered list of values to draw.

    Raises:
        ValueError: If the column holds no usable value, or if a requested sample
                    is absent from it.
    """
    sample_series = anndata_object.obs[sample_column].astype(str)
    present_values = sample_series.unique().tolist()

    if not present_values:
        raise ValueError(
            f".obs['{sample_column}'] contains no usable values — nothing to plot."
        )

    if samples is None:
        return sample_series, [str(value) for value in sort_values_for_legend(present_values)]

    requested = [str(value) for value in samples]
    missing = [value for value in requested if value not in present_values]
    if missing:
        raise ValueError(
            f"The following samples were not found in .obs['{sample_column}']: "
            f"{missing}. Available: {sorted(present_values)}"
        )
    return sample_series, requested


def _aggregate_values(values: np.ndarray, statistic: str) -> Tuple[float, float]:
    """
    Collapse one sample's events for one channel into (value, std).

    Arguments:
        values: 1-D array of per-event intensities, possibly containing NaN.
        statistic: ``"mean"`` or ``"median"``.

    Returns:
        (aggregate_value, standard_deviation), or (nan, nan) when no finite value
        is available.
    """
    finite_values = values[np.isfinite(values)]
    if finite_values.size == 0:
        return np.nan, np.nan
    aggregate_value = (
        float(np.mean(finite_values)) if statistic == "mean" else float(np.median(finite_values))
    )
    return aggregate_value, float(np.std(finite_values))


def _compute_intensity_table(
    data_matrix: np.ndarray,
    sample_series: pd.Series,
    sample_values: List[str],
    statistic: str,
    function_name: str,
    channel_names: List[str],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build the (n_samples, n_channels) value and error tables.

    Arguments:
        data_matrix: Event-by-channel matrix already restricted to the resolved
                     channels, in the same order as ``channel_names``.
        sample_series: .obs sample column cast to str, one entry per event.
        sample_values: Ordered sample values, one row of the output per value.
        statistic: ``"mean"`` or ``"median"``.
        function_name: Name used in console notes.
        channel_names: Resolved channel names, used only for the empty-slice note.

    Returns:
        (values, errors), both float arrays of shape (n_samples, n_channels).
    """
    values = np.full((len(sample_values), data_matrix.shape[1]), np.nan, dtype=float)
    errors = np.full_like(values, np.nan)

    sample_array = sample_series.values
    for row_index, sample_value in enumerate(sample_values):
        row_mask = sample_array == sample_value
        sample_matrix = data_matrix[row_mask]
        for column_index in range(data_matrix.shape[1]):
            aggregate_value, standard_deviation = _aggregate_values(
                sample_matrix[:, column_index], statistic
            )
            values[row_index, column_index] = aggregate_value
            errors[row_index, column_index] = standard_deviation

    # One summary line rather than one message per empty cell: with 40 channels a
    # per-cell note would bury the rest of the console output.
    n_empty_cells = int(np.sum(~np.isfinite(values)))
    if n_empty_cells:
        print(
            f"[{function_name}] NOTE: {n_empty_cells} of {values.size} "
            f"(sample × channel) cell(s) had no finite value and are left blank."
        )

    return values, errors


def _build_color_map(
    anndata_object: anndata.AnnData,
    sample_values: List[str],
) -> Dict[str, str]:
    """
    Map sample values to hex colours.

    Priority:
      1. PREFERRED_CONDITION_COLORS (researcher-declared colours always win)
      2. .uns['color_palette'] (populated by assign_color_palette())
      3. Auto-generated Set2 colours for anything left

    Step 3 matters: without it every sample would fall back to the same grey.

    Arguments:
        anndata_object: AnnData possibly carrying .uns['color_palette'].
        sample_values: Sample values needing a colour.

    Returns:
        A {sample_value: hex_color} dict covering every requested value.
    """
    stored_palette = anndata_object.uns.get("color_palette", {})
    colors: Dict[str, str] = {}
    unassigned: List[str] = []

    for sample_value in sample_values:
        if sample_value in PREFERRED_CONDITION_COLORS:
            colors[sample_value] = PREFERRED_CONDITION_COLORS[sample_value]
        elif sample_value in stored_palette:
            colors[sample_value] = stored_palette[sample_value]
        else:
            unassigned.append(sample_value)

    if unassigned:
        colors.update(get_subject_colors(unassigned, palette="Set2"))

    return colors


def _shorten_sample_label(sample_value: str, max_length: int = 28) -> str:
    """
    Shorten a sample value for the legend.

    FCS filenames routinely exceed 50 characters, which would swallow the figure.
    The full value is always printed to the console, so nothing is hidden.

    Arguments:
        sample_value: The raw .obs value.
        max_length: Maximum length of the returned label.

    Returns:
        The value without a trailing ".fcs", middle-ellipsised when still too long.
    """
    label = sample_value[:-4] if sample_value.lower().endswith(".fcs") else sample_value
    if len(label) <= max_length:
        return label
    head_length = (max_length - 3) // 2
    tail_length = max_length - 3 - head_length
    return f"{label[:head_length]}...{label[-tail_length:]}"


def _clamp_lower_error(values: np.ndarray, errors: np.ndarray) -> np.ndarray:
    """
    Build an asymmetric (2, n) error array whose lower arm stays positive.

    A log axis cannot draw a whisker reaching zero or below, so the lower arm is
    shortened to stop just above the smallest positive plotted value.  The upper
    arm is never modified.

    Arguments:
        values: Plotted values for one sample.
        errors: Matching +/-1 std values.

    Returns:
        A (2, n) array of [lower_arm, upper_arm] lengths.
    """
    positive_values = values[np.isfinite(values) & (values > 0)]
    floor = float(np.min(positive_values)) / 10.0 if positive_values.size else 1e-9
    lower_arm = np.minimum(errors, np.maximum(values - floor, 0.0))
    return np.vstack([lower_arm, errors])


def _draw_lines(
    axes: plt.Axes,
    values: np.ndarray,
    errors: np.ndarray,
    sample_values: List[str],
    color_map: Dict[str, str],
    show_error_bars: bool,
    log_scale: bool,
) -> None:
    """
    Draw one line per sample across the channel positions.

    Error bars are drawn as vertical caps rather than a shaded band: the X-axis
    holds unrelated detectors, so a filled band between them would suggest an
    interpolation that has no meaning.

    ``errorbar`` is used even when ``show_error_bars`` is False (with yerr=None)
    so that the axes always carry exactly one container per sample, whatever the
    arguments.

    Arguments:
        axes: Target axes.
        values: (n_samples, n_channels) plotted values.
        errors: (n_samples, n_channels) +/-1 std values.
        sample_values: Ordered sample values, one line each.
        color_map: {sample_value: hex_color}.
        show_error_bars: Whether to draw the caps.
        log_scale: Whether the Y-axis is logarithmic (clamps the lower cap).
    """
    x_positions = np.arange(values.shape[1])

    for row_index, sample_value in enumerate(sample_values):
        row_values = values[row_index]

        error_argument = None
        if show_error_bars:
            row_errors = errors[row_index]
            error_argument = (
                _clamp_lower_error(row_values, row_errors) if log_scale else row_errors
            )

        axes.errorbar(
            x_positions,
            row_values,
            yerr=error_argument,
            color=color_map.get(sample_value, "#999999"),
            linewidth=2,
            marker="o",
            markersize=5,
            capsize=3,
            elinewidth=0.8,
            label=_shorten_sample_label(sample_value),
        )


def _draw_bars(
    axes: plt.Axes,
    values: np.ndarray,
    errors: np.ndarray,
    sample_values: List[str],
    color_map: Dict[str, str],
    show_error_bars: bool,
    log_scale: bool,
) -> None:
    """
    Draw one grouped bar per sample at each channel position.

    Each channel occupies a cluster of ``_BAR_CLUSTER_WIDTH`` centred on its tick;
    the cluster is split evenly between samples so the gap between channels stays
    constant whatever the sample count.

    Arguments:
        axes: Target axes.
        values: (n_samples, n_channels) plotted values.
        errors: (n_samples, n_channels) +/-1 std values.
        sample_values: Ordered sample values, one bar colour each.
        color_map: {sample_value: hex_color}.
        show_error_bars: Whether to draw the caps.
        log_scale: Whether the Y-axis is logarithmic (clamps the lower cap).
    """
    n_samples = len(sample_values)
    x_positions = np.arange(values.shape[1])
    bar_width = _BAR_CLUSTER_WIDTH / n_samples

    for row_index, sample_value in enumerate(sample_values):
        offset = (row_index - (n_samples - 1) / 2) * bar_width
        row_values = values[row_index]

        error_argument = None
        if show_error_bars:
            row_errors = errors[row_index]
            error_argument = (
                _clamp_lower_error(row_values, row_errors) if log_scale else row_errors
            )

        axes.bar(
            x_positions + offset,
            row_values,
            width=bar_width,
            yerr=error_argument,
            color=color_map.get(sample_value, "#999999"),
            edgecolor="white",
            linewidth=0.5,
            alpha=0.9,
            capsize=3,
            error_kw={"linewidth": 0.8, "ecolor": "black"},
            label=_shorten_sample_label(sample_value),
        )


def _finalize_axes(
    figure: plt.Figure,
    axes: plt.Axes,
    anndata_object: anndata.AnnData,
    channel_names: List[str],
    sample_values: List[str],
    layer_label: str,
    statistic: str,
    log_scale: bool,
    title: str,
    function_name: str,
) -> None:
    """
    Apply ticks, labels, title, scale, legend and provenance footer.

    Arguments:
        figure: The figure being rendered.
        axes: The axes holding the marks.
        anndata_object: Source AnnData, used for the species label and footer.
        channel_names: Resolved channels, used as X tick labels.
        sample_values: Ordered sample values, used for the auto title.
        layer_label: Layer name shown on the Y-axis and in the title.
        statistic: ``"mean"`` or ``"median"``, shown in the title.
        log_scale: Whether to switch the Y-axis to a logarithmic scale.
        title: Caller-supplied title, or "" for the auto title.
        function_name: Name used in console notes.
    """
    axes.set_xticks(np.arange(len(channel_names)))
    if len(channel_names) > _ROTATE_LABELS_ABOVE_N_TICKS:
        axes.set_xticklabels(channel_names, rotation=90, fontsize=7)
    else:
        axes.set_xticklabels(channel_names, rotation=0, fontsize=10)

    axes.set_xlabel("Channel", fontsize=12)
    axes.set_ylabel(f"Fluorescence intensity ({layer_label})", fontsize=12)

    if log_scale:
        axes.set_yscale("log")

    if not title:
        species = get_species_label(anndata_object)
        species_prefix = f"[{species}]  " if species else ""
        title = (
            f"{species_prefix}Channel Intensity — {len(sample_values)} sample(s)\n"
            f"(Layer: {layer_label} | {statistic} per sample)"
        )
    axes.set_title(title, fontsize=11, pad=12)

    axes.spines["top"].set_visible(False)
    axes.spines["right"].set_visible(False)

    handles, labels = axes.get_legend_handles_labels()
    legend_kwargs = {"fontsize": 9, "title": "Sample", "title_fontsize": 10}
    if len(labels) > 12:
        axes.legend(
            handles,
            labels,
            ncol=ceil(len(labels) / 12),
            loc="upper center",
            bbox_to_anchor=(0.5, -0.20),
            **legend_kwargs,
        )
        plt.tight_layout(rect=[0, 0.12, 1, 1])
    else:
        axes.legend(handles, labels, loc="upper left", bbox_to_anchor=(1.02, 1), **legend_kwargs)
        plt.tight_layout()

    footer_text = get_run_context_footer_text(anndata_object)
    if footer_text:
        figure.text(0.5, -0.02, footer_text, ha="center", fontsize=7, color="#666666")


def _render_channel_intensity(
    anndata_object: anndata.AnnData,
    channels: Optional[List[str]],
    pulse_type: str,
    sample_column: str,
    samples: Optional[List[str]],
    statistic: str,
    show_error_bars: bool,
    log_scale: bool,
    title: str,
    plot_kind: str,
    function_name: str,
) -> None:
    """
    Shared renderer behind plot_channel_intensity_line() and _bar().

    Everything except the drawing step is identical between the two public
    functions, so they differ only by ``plot_kind``.

    Arguments:
        anndata_object: SFC AnnData to plot.
        channels: Channel tokens, or None for all channels of ``pulse_type``.
        pulse_type: ``"A"``, ``"H"`` or ``"W"``.
        sample_column: .obs column identifying one sample.
        samples: Sample subset, or None for all.
        statistic: ``"mean"`` or ``"median"``.
        show_error_bars: Whether to draw +/-1 std caps.
        log_scale: Whether the Y-axis is logarithmic.
        title: Caller-supplied title, or "" for the auto title.
        plot_kind: ``"line"`` or ``"bar"``.
        function_name: Public function name, used in every console message.

    Returns:
        None.  The figure is displayed via plt.show().
    """
    pulse_type = pulse_type.upper() if isinstance(pulse_type, str) else pulse_type
    if pulse_type not in _ALLOWED_PULSE_TYPES:
        raise ValueError(f"pulse_type must be 'A', 'H', or 'W', got '{pulse_type}'.")
    if statistic not in _ALLOWED_STATISTICS:
        raise ValueError(f"statistic must be 'mean' or 'median', got '{statistic}'.")
    if sample_column not in anndata_object.obs.columns:
        raise ValueError(
            f"sample_column '{sample_column}' not found in .obs. "
            f"Available columns: {list(anndata_object.obs.columns)}"
        )

    print(get_run_context_console_text(anndata_object))

    data_matrix, available_channels, layer_label = _get_data_and_channels(
        anndata_object, function_name
    )
    channel_names = _resolve_channels(
        available_channels, channels, pulse_type, function_name
    )
    sample_series, sample_values = _resolve_samples(anndata_object, sample_column, samples)

    column_indices = [available_channels.index(name) for name in channel_names]
    data_matrix = data_matrix[:, column_indices]

    print(
        f"[{function_name}] pulse_type='{pulse_type}' | "
        f"{len(channel_names)} channel(s) | {len(sample_values)} sample(s) | "
        f"statistic={statistic}"
    )
    print(f"    channels : {', '.join(channel_names)}")
    print("    samples  : " + "\n               ".join(sample_values))

    if plot_kind == "bar" and len(sample_values) > _MAX_READABLE_BAR_SAMPLES:
        print(
            f"[{function_name}] NOTE: {len(sample_values)} samples — bars become very "
            "thin. Consider plot_channel_intensity_line() or the samples=[...] argument."
        )

    values, errors = _compute_intensity_table(
        data_matrix, sample_series, sample_values, statistic, function_name, channel_names
    )

    if log_scale:
        n_non_positive = int(np.sum(np.isfinite(values) & (values <= 0)))
        if n_non_positive:
            print(
                f"[{function_name}] NOTE: log_scale=True but {n_non_positive} value(s) "
                "are <= 0 and cannot be drawn on a log axis."
            )

    color_map = _build_color_map(anndata_object, sample_values)

    figure_width = max(10.0, 0.35 * len(channel_names) + 0.2 * len(sample_values))
    figure, axes = plt.subplots(figsize=(figure_width, 5))

    draw_function = _draw_bars if plot_kind == "bar" else _draw_lines
    draw_function(
        axes=axes,
        values=values,
        errors=errors,
        sample_values=sample_values,
        color_map=color_map,
        show_error_bars=show_error_bars,
        log_scale=log_scale,
    )

    _finalize_axes(
        figure=figure,
        axes=axes,
        anndata_object=anndata_object,
        channel_names=channel_names,
        sample_values=sample_values,
        layer_label=layer_label,
        statistic=statistic,
        log_scale=log_scale,
        title=title,
        function_name=function_name,
    )

    plt.show()
