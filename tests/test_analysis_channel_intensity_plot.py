"""
test_analysis_channel_intensity_plot.py

Tests for mito_marker.analysis.channel_intensity_plot.

Tests verify:
  - _parse_channel_name() splits SFC channel names into (laser, pulse, detector),
    including the two-hyphen "SSC-B-A" form that channel_aggregation.py drops.
  - _resolve_channels() expands laser-group tokens into individual channels,
    preserves caller order, sorts detectors numerically, and never lets the "UV"
    and "V" prefixes collide.
  - The plotted values equal the per-sample mean / median over that sample's
    events, and the error bars equal the standard deviation.
  - Line rendering: one container per sample, correct Y data, ticks, log scale,
    active layer and active feature selection handling.
  - Bar rendering: one BarContainer per sample, correct heights, bar width and
    cluster centring.
  - Every documented ValueError, parametrised over both public functions.
  - Colour resolution priority and the console QC output.
"""

import matplotlib

matplotlib.use("Agg")  # Non-interactive backend — no window required.

import anndata
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.container import BarContainer

from mito_marker.analysis.channel_intensity_plot import (
    _build_color_map,
    _parse_channel_name,
    _resolve_channels,
    _shorten_sample_label,
    plot_channel_intensity_bar,
    plot_channel_intensity_line,
)
from mito_marker.analysis.selection import _ANALYSIS_CONFIG_KEY

N_EVENTS_PER_SAMPLE = 20
SAMPLE_COLUMN = "source_filename"

_BASE_CHANNELS = [
    "FSC-A", "FSC-H",
    "SSC-A", "SSC-H",
    "UV1-A", "UV2-A", "UV1-H", "UV2-H",
    "V1-A", "V2-A", "V10-A", "V1-H", "V2-H", "V10-H",
    "YG1-A", "YG2-A", "YG1-H", "YG2-H",
    "R1-A", "R1-H",
]


def _make_sfc_anndata(
    include_layer: bool = False,
    include_selection: bool = False,
    include_non_analytical: bool = False,
    include_ssc_b: bool = False,
    include_w_pulse: bool = False,
    n_samples: int = 3,
) -> anndata.AnnData:
    """Build a synthetic SFC AnnData with realistic Cytek Aurora channel names."""
    channel_names = list(_BASE_CHANNELS)
    if include_ssc_b:
        channel_names += ["SSC-B-A", "SSC-B-H"]
    if include_w_pulse:
        channel_names += ["V8-W", "YG5-W"]
    if include_non_analytical:
        channel_names += ["Time"]

    n_obs = N_EVENTS_PER_SAMPLE * n_samples
    random_generator = np.random.default_rng(42)
    data_matrix = np.abs(
        random_generator.normal(500.0, 100.0, (n_obs, len(channel_names)))
    ).astype(np.float32)

    sample_names = [f"good_events_MNMS_{index:03d}_DeepRed.fcs" for index in range(n_samples)]
    observation_dataframe = pd.DataFrame(
        {
            SAMPLE_COLUMN: np.repeat(sample_names, N_EVENTS_PER_SAMPLE),
            "subject_ID": np.repeat([f"{index:03d}" for index in range(n_samples)],
                                    N_EVENTS_PER_SAMPLE),
            "diet": np.tile(["AL", "IF"], n_obs // 2),
            "specie": ["MNMS"] * n_obs,
        }
    )

    variable_dataframe = pd.DataFrame(index=channel_names)
    if include_non_analytical:
        variable_dataframe["is_non_analytical"] = [
            name == "Time" for name in channel_names
        ]

    sfc_anndata = anndata.AnnData(
        X=data_matrix, obs=observation_dataframe, var=variable_dataframe
    )
    sfc_anndata.uns[_ANALYSIS_CONFIG_KEY] = {"active_layer": None, "active_selection": None}

    if include_layer:
        sfc_anndata.layers["arcsinh__zscore_col"] = np.arcsinh(
            data_matrix / 150.0
        ).astype(np.float32)
        sfc_anndata.uns[_ANALYSIS_CONFIG_KEY]["active_layer"] = "arcsinh__zscore_col"

    if include_selection:
        # Keep V1-A and V2-A but drop V10-A, so an expanded "V-A" token is visibly
        # narrowed by the active selection.
        kept = {"FSC-A", "SSC-A", "V1-A", "V2-A", "YG1-A"}
        variable_extended = sfc_anndata.var.copy()
        variable_extended["is_selected_HighVariance"] = [
            name in kept for name in channel_names
        ]
        sfc_anndata.var = variable_extended
        sfc_anndata.uns[_ANALYSIS_CONFIG_KEY]["active_selection"] = "HighVariance"

    return sfc_anndata


def _make_tem_anndata() -> anndata.AnnData:
    """Build a TEM AnnData, whose feature names carry no pulse suffix."""
    feature_names = ["Mito_Area", "Mito_Perimeter", "Mito_Circularity", "Mito_AR"]
    data_matrix = np.abs(
        np.random.default_rng(0).normal(1.0, 0.2, (20, len(feature_names)))
    ).astype(np.float32)
    observation_dataframe = pd.DataFrame({SAMPLE_COLUMN: ["image_a"] * 10 + ["image_b"] * 10})
    tem_anndata = anndata.AnnData(
        X=data_matrix,
        obs=observation_dataframe,
        var=pd.DataFrame(index=feature_names),
    )
    tem_anndata.uns[_ANALYSIS_CONFIG_KEY] = {"active_layer": None, "active_selection": None}
    return tem_anndata


def _sample_names(n_samples: int = 3) -> list[str]:
    """Return the sample values produced by _make_sfc_anndata()."""
    return [f"good_events_MNMS_{index:03d}_DeepRed.fcs" for index in range(n_samples)]


def _tick_labels(axes: plt.Axes) -> list[str]:
    """Return the X tick label texts of an axes."""
    return [tick.get_text() for tick in axes.get_xticklabels()]


def _bar_containers(axes: plt.Axes) -> list[BarContainer]:
    """Return only the bar containers — ax.bar(yerr=...) also adds an errorbar one."""
    return [container for container in axes.containers if isinstance(container, BarContainer)]


@pytest.fixture(autouse=True)
def _close_figures():
    """Close every figure after each test so figure counts stay independent."""
    yield
    plt.close("all")


@pytest.fixture
def sfc_anndata() -> anndata.AnnData:
    return _make_sfc_anndata()


# ---------------------------------------------------------------------------
# _parse_channel_name
# ---------------------------------------------------------------------------


class TestParseChannelName:
    """Channel-name parsing into (laser, pulse, detector_number)."""

    def test_spectral_channel(self) -> None:
        assert _parse_channel_name("V1-A") == ("V", "A", 1)

    def test_two_digit_detector(self) -> None:
        assert _parse_channel_name("UV16-H") == ("UV", "H", 16)

    def test_width_pulse(self) -> None:
        assert _parse_channel_name("YG10-W") == ("YG", "W", 10)

    def test_scatter_channel_has_no_detector_number(self) -> None:
        assert _parse_channel_name("FSC-A") == ("FSC", "A", 0)

    def test_second_side_scatter_detector_is_parsed(self) -> None:
        """SSC-B-A has two hyphens; channel_aggregation.py drops it, this module keeps it."""
        assert _parse_channel_name("SSC-B-A") == ("SSC", "A", 0)
        assert _parse_channel_name("SSC-B-W") == ("SSC", "W", 0)

    def test_uv_is_not_read_as_violet(self) -> None:
        """The laser is the full alphabetic run, so 'UV' can never collide with 'V'."""
        assert _parse_channel_name("UV1-A")[0] == "UV"
        assert _parse_channel_name("V1-A")[0] == "V"

    def test_lowercase_is_accepted(self) -> None:
        assert _parse_channel_name("v1-a") == ("V", "A", 1)

    @pytest.mark.parametrize("channel_name", ["Time", "FlowAI", "Mito_Area", "V1-X", "V1"])
    def test_non_sfc_names_return_none(self, channel_name: str) -> None:
        assert _parse_channel_name(channel_name) is None


# ---------------------------------------------------------------------------
# _resolve_channels
# ---------------------------------------------------------------------------


class TestResolveChannels:
    """Expansion of the channels tokens into the ordered X-axis channel list."""

    def test_group_token_expands_to_individual_channels(self) -> None:
        resolved = _resolve_channels(_BASE_CHANNELS, ["V-A"], "A", "fn")
        assert resolved == ["V1-A", "V2-A", "V10-A"]

    def test_expansion_sorts_detectors_numerically(self) -> None:
        """V2-A must precede V10-A — a lexicographic sort would invert them."""
        resolved = _resolve_channels(_BASE_CHANNELS, ["V-A"], "A", "fn")
        assert resolved.index("V2-A") < resolved.index("V10-A")

    def test_group_token_honours_its_own_pulse(self) -> None:
        resolved = _resolve_channels(_BASE_CHANNELS, ["YG-H"], "A", "fn")
        assert resolved == ["YG1-H", "YG2-H"]

    def test_uv_token_does_not_pick_up_violet_channels(self) -> None:
        resolved = _resolve_channels(_BASE_CHANNELS, ["UV-A"], "A", "fn")
        assert resolved == ["UV1-A", "UV2-A"]

    def test_violet_token_does_not_pick_up_uv_channels(self) -> None:
        resolved = _resolve_channels(_BASE_CHANNELS, ["V-A"], "A", "fn")
        assert all(not name.startswith("UV") for name in resolved)

    def test_exact_channel_name_wins_over_group_expansion(self) -> None:
        """SSC-A is a real channel name even though it looks like a group token."""
        available = _BASE_CHANNELS + ["SSC-B-A"]
        assert _resolve_channels(available, ["SSC-A"], "A", "fn") == ["SSC-A"]

    def test_bare_laser_token_uses_pulse_type(self) -> None:
        available = _BASE_CHANNELS + ["SSC-B-A"]
        assert _resolve_channels(available, ["SSC"], "A", "fn") == ["SSC-A", "SSC-B-A"]

    def test_caller_token_order_is_preserved(self) -> None:
        resolved = _resolve_channels(_BASE_CHANNELS, ["YG-A", "V-A"], "A", "fn")
        assert resolved == ["YG1-A", "YG2-A", "V1-A", "V2-A", "V10-A"]

    def test_duplicates_are_removed_keeping_first_occurrence(self) -> None:
        resolved = _resolve_channels(_BASE_CHANNELS, ["V1-A", "V-A"], "A", "fn")
        assert resolved == ["V1-A", "V2-A", "V10-A"]

    def test_mixed_pulses_are_allowed(self) -> None:
        resolved = _resolve_channels(_BASE_CHANNELS, ["V-A", "V-H"], "A", "fn")
        assert resolved == ["V1-A", "V2-A", "V10-A", "V1-H", "V2-H", "V10-H"]

    def test_none_returns_every_channel_of_the_pulse(self) -> None:
        resolved = _resolve_channels(_BASE_CHANNELS, None, "H", "fn")
        assert all(name.endswith("-H") for name in resolved)
        assert "V1-A" not in resolved

    def test_single_channel_token(self) -> None:
        assert _resolve_channels(_BASE_CHANNELS, ["V2-A"], "A", "fn") == ["V2-A"]


# ---------------------------------------------------------------------------
# Plotted values
# ---------------------------------------------------------------------------


class TestIntensityValues:
    """The plotted value must be the per-sample statistic over that sample's events."""

    def test_line_values_equal_per_sample_mean(self, sfc_anndata: anndata.AnnData) -> None:
        plot_channel_intensity_line(sfc_anndata, channels=["V-A"])
        axes = plt.gcf().get_axes()[0]

        channel_indices = [
            list(sfc_anndata.var_names).index(name) for name in ["V1-A", "V2-A", "V10-A"]
        ]
        row_mask = (sfc_anndata.obs[SAMPLE_COLUMN] == _sample_names()[0]).values
        expected = sfc_anndata.X[row_mask][:, channel_indices].mean(axis=0)

        np.testing.assert_allclose(
            axes.containers[0].lines[0].get_ydata(), expected, rtol=1e-5
        )

    def test_median_statistic_equals_numpy_median(self, sfc_anndata: anndata.AnnData) -> None:
        plot_channel_intensity_line(sfc_anndata, channels=["V1-A"], statistic="median")
        axes = plt.gcf().get_axes()[0]

        channel_index = list(sfc_anndata.var_names).index("V1-A")
        row_mask = (sfc_anndata.obs[SAMPLE_COLUMN] == _sample_names()[0]).values
        expected = np.median(sfc_anndata.X[row_mask][:, channel_index])

        assert axes.containers[0].lines[0].get_ydata()[0] == pytest.approx(expected, rel=1e-5)

    def test_median_differs_from_mean_on_skewed_data(self) -> None:
        """Negative control: the two statistics must not be interchangeable."""
        skewed_anndata = _make_sfc_anndata(n_samples=1)
        channel_index = list(skewed_anndata.var_names).index("V1-A")
        skewed_matrix = skewed_anndata.X.copy()
        skewed_matrix[:5, channel_index] = 50_000.0  # a few very bright mitochondria
        skewed_anndata.X = skewed_matrix

        plot_channel_intensity_line(skewed_anndata, channels=["V1-A"], statistic="mean")
        mean_value = plt.gcf().get_axes()[0].containers[0].lines[0].get_ydata()[0]
        plt.close("all")

        plot_channel_intensity_line(skewed_anndata, channels=["V1-A"], statistic="median")
        median_value = plt.gcf().get_axes()[0].containers[0].lines[0].get_ydata()[0]

        assert mean_value > median_value * 2

    def test_error_bars_equal_standard_deviation(self, sfc_anndata: anndata.AnnData) -> None:
        plot_channel_intensity_bar(sfc_anndata, channels=["V1-A"], show_error_bars=True)
        axes = plt.gcf().get_axes()[0]

        channel_index = list(sfc_anndata.var_names).index("V1-A")
        row_mask = (sfc_anndata.obs[SAMPLE_COLUMN] == _sample_names()[0]).values
        expected_std = float(np.std(sfc_anndata.X[row_mask][:, channel_index]))

        bar_container = _bar_containers(axes)[0]
        segments = bar_container.errorbar.lines[2][0].get_segments()[0]
        assert (segments[1][1] - segments[0][1]) / 2 == pytest.approx(expected_std, rel=1e-4)

    def test_samples_argument_restricts_and_orders(self, sfc_anndata: anndata.AnnData) -> None:
        chosen = [_sample_names()[2], _sample_names()[0]]
        plot_channel_intensity_line(sfc_anndata, channels=["V1-A"], samples=chosen)
        axes = plt.gcf().get_axes()[0]

        assert len(axes.containers) == 2
        _, labels = axes.get_legend_handles_labels()
        assert labels == [_shorten_sample_label(value) for value in chosen]

    def test_nan_events_are_ignored(self) -> None:
        nan_anndata = _make_sfc_anndata(n_samples=1)
        channel_index = list(nan_anndata.var_names).index("V1-A")
        matrix_with_nan = nan_anndata.X.copy()
        matrix_with_nan[:5, channel_index] = np.nan
        nan_anndata.X = matrix_with_nan

        plot_channel_intensity_line(nan_anndata, channels=["V1-A"])
        plotted = plt.gcf().get_axes()[0].containers[0].lines[0].get_ydata()[0]
        expected = np.mean(matrix_with_nan[5:, channel_index])

        assert plotted == pytest.approx(expected, rel=1e-5)


# ---------------------------------------------------------------------------
# Line rendering
# ---------------------------------------------------------------------------


class TestLineRendering:
    """Figure structure produced by plot_channel_intensity_line()."""

    def test_one_container_per_sample(self, sfc_anndata: anndata.AnnData) -> None:
        plot_channel_intensity_line(sfc_anndata, channels=["V-A"])
        assert len(plt.gcf().get_axes()[0].containers) == 3

    def test_tick_labels_are_the_resolved_channels(self, sfc_anndata: anndata.AnnData) -> None:
        plot_channel_intensity_line(sfc_anndata, channels=["V-A", "YG-A"])
        assert _tick_labels(plt.gcf().get_axes()[0]) == [
            "V1-A", "V2-A", "V10-A", "YG1-A", "YG2-A"
        ]

    def test_error_bars_are_off_by_default(self, sfc_anndata: anndata.AnnData) -> None:
        plot_channel_intensity_line(sfc_anndata, channels=["V1-A"])
        assert plt.gcf().get_axes()[0].containers[0].has_yerr is False

    def test_error_bars_can_be_enabled(self, sfc_anndata: anndata.AnnData) -> None:
        plot_channel_intensity_line(sfc_anndata, channels=["V1-A"], show_error_bars=True)
        assert plt.gcf().get_axes()[0].containers[0].has_yerr is True

    def test_log_scale_sets_logarithmic_axis(self, sfc_anndata: anndata.AnnData) -> None:
        plot_channel_intensity_line(sfc_anndata, channels=["V1-A"], log_scale=True)
        assert plt.gcf().get_axes()[0].get_yscale() == "log"

    def test_active_layer_is_used(self) -> None:
        layered_anndata = _make_sfc_anndata(include_layer=True)
        plot_channel_intensity_line(layered_anndata, channels=["V1-A"])
        axes = plt.gcf().get_axes()[0]

        channel_index = list(layered_anndata.var_names).index("V1-A")
        row_mask = (layered_anndata.obs[SAMPLE_COLUMN] == _sample_names()[0]).values
        expected = layered_anndata.layers["arcsinh__zscore_col"][row_mask][
            :, channel_index
        ].mean()

        assert axes.containers[0].lines[0].get_ydata()[0] == pytest.approx(expected, rel=1e-5)
        assert "arcsinh__zscore_col" in axes.get_ylabel()

    def test_active_selection_narrows_the_expansion(self) -> None:
        selected_anndata = _make_sfc_anndata(include_selection=True)
        plot_channel_intensity_line(selected_anndata, channels=["V-A"])
        assert _tick_labels(plt.gcf().get_axes()[0]) == ["V1-A", "V2-A"]

    def test_non_analytical_channels_are_never_plotted(self) -> None:
        anndata_with_time = _make_sfc_anndata(include_non_analytical=True)
        plot_channel_intensity_line(anndata_with_time, channels=None)
        assert "Time" not in _tick_labels(plt.gcf().get_axes()[0])

    def test_custom_title_is_used_verbatim(self, sfc_anndata: anndata.AnnData) -> None:
        plot_channel_intensity_line(sfc_anndata, channels=["V1-A"], title="My title")
        assert plt.gcf().get_axes()[0].get_title() == "My title"

    def test_auto_title_mentions_layer_and_statistic(self, sfc_anndata: anndata.AnnData) -> None:
        plot_channel_intensity_line(sfc_anndata, channels=["V1-A"], statistic="median")
        title = plt.gcf().get_axes()[0].get_title()
        assert "raw .X" in title and "median" in title

    def test_width_pulse_channels_can_be_plotted(self) -> None:
        anndata_with_width = _make_sfc_anndata(include_w_pulse=True)
        plot_channel_intensity_line(anndata_with_width, channels=None, pulse_type="W")
        assert _tick_labels(plt.gcf().get_axes()[0]) == ["V8-W", "YG5-W"]


# ---------------------------------------------------------------------------
# Bar rendering
# ---------------------------------------------------------------------------


class TestBarRendering:
    """Figure structure produced by plot_channel_intensity_bar()."""

    def test_one_bar_container_per_sample(self, sfc_anndata: anndata.AnnData) -> None:
        plot_channel_intensity_bar(sfc_anndata, channels=["V-A"])
        assert len(_bar_containers(plt.gcf().get_axes()[0])) == 3

    def test_bar_heights_match_the_sample_means(self, sfc_anndata: anndata.AnnData) -> None:
        plot_channel_intensity_bar(sfc_anndata, channels=["V1-A", "V2-A"])
        bar_container = _bar_containers(plt.gcf().get_axes()[0])[1]

        channel_indices = [list(sfc_anndata.var_names).index(n) for n in ["V1-A", "V2-A"]]
        row_mask = (sfc_anndata.obs[SAMPLE_COLUMN] == _sample_names()[1]).values
        expected = sfc_anndata.X[row_mask][:, channel_indices].mean(axis=0)

        np.testing.assert_allclose(
            [bar.get_height() for bar in bar_container], expected, rtol=1e-5
        )

    def test_bar_width_splits_the_cluster_evenly(self, sfc_anndata: anndata.AnnData) -> None:
        plot_channel_intensity_bar(sfc_anndata, channels=["V1-A"])
        bar_container = _bar_containers(plt.gcf().get_axes()[0])[0]
        assert bar_container[0].get_width() == pytest.approx(0.8 / 3)

    def test_cluster_is_centred_on_the_tick(self, sfc_anndata: anndata.AnnData) -> None:
        plot_channel_intensity_bar(sfc_anndata, channels=["V1-A", "V2-A"])
        containers = _bar_containers(plt.gcf().get_axes()[0])
        first_cluster_centres = [
            container[0].get_x() + container[0].get_width() / 2 for container in containers
        ]
        assert float(np.mean(first_cluster_centres)) == pytest.approx(0.0)

    def test_error_bars_are_off_by_default(self, sfc_anndata: anndata.AnnData) -> None:
        plot_channel_intensity_bar(sfc_anndata, channels=["V1-A"])
        assert _bar_containers(plt.gcf().get_axes()[0])[0].errorbar is None

    def test_error_bars_can_be_enabled(self, sfc_anndata: anndata.AnnData) -> None:
        plot_channel_intensity_bar(sfc_anndata, channels=["V1-A"], show_error_bars=True)
        assert _bar_containers(plt.gcf().get_axes()[0])[0].errorbar is not None

    def test_many_samples_prints_a_readability_note(self, capsys: pytest.CaptureFixture) -> None:
        many_sample_anndata = _make_sfc_anndata(n_samples=20)
        plot_channel_intensity_bar(many_sample_anndata, channels=["V1-A"])
        assert "bars become very thin" in capsys.readouterr().out

    def test_log_scale_keeps_the_lower_error_arm_positive(
        self, sfc_anndata: anndata.AnnData
    ) -> None:
        plot_channel_intensity_bar(
            sfc_anndata, channels=["V1-A"], log_scale=True, show_error_bars=True
        )
        bar_container = _bar_containers(plt.gcf().get_axes()[0])[0]
        segments = bar_container.errorbar.lines[2][0].get_segments()[0]
        assert segments[0][1] > 0


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "plot_function", [plot_channel_intensity_line, plot_channel_intensity_bar]
)
class TestValidation:
    """Every documented ValueError, checked on both public functions."""

    def test_invalid_pulse_type(self, plot_function, sfc_anndata: anndata.AnnData) -> None:
        with pytest.raises(ValueError, match="pulse_type must be 'A', 'H', or 'W'"):
            plot_function(sfc_anndata, pulse_type="Z")

    def test_invalid_statistic(self, plot_function, sfc_anndata: anndata.AnnData) -> None:
        with pytest.raises(ValueError, match="statistic must be 'mean' or 'median'"):
            plot_function(sfc_anndata, statistic="geometric")

    def test_unknown_sample_column(self, plot_function, sfc_anndata: anndata.AnnData) -> None:
        with pytest.raises(ValueError, match="sample_column 'nope' not found in .obs"):
            plot_function(sfc_anndata, sample_column="nope")

    def test_unknown_channel_token(self, plot_function, sfc_anndata: anndata.AnnData) -> None:
        with pytest.raises(ValueError, match="neither a channel name nor a laser group"):
            plot_function(sfc_anndata, channels=["Purple"])

    def test_token_matching_no_channel(self, plot_function, sfc_anndata: anndata.AnnData) -> None:
        with pytest.raises(ValueError, match="matched no channel"):
            plot_function(sfc_anndata, channels=["R-W"])

    def test_unknown_sample(self, plot_function, sfc_anndata: anndata.AnnData) -> None:
        with pytest.raises(ValueError, match="samples were not found"):
            plot_function(sfc_anndata, channels=["V1-A"], samples=["absent.fcs"])

    def test_tem_anndata_is_rejected_with_a_helpful_message(self, plot_function) -> None:
        with pytest.raises(ValueError, match="spectral flow cytometry"):
            plot_function(_make_tem_anndata())

    def test_no_channel_for_the_requested_pulse(self, plot_function) -> None:
        """The base fixture has no -W channels, so pulse_type='W' must raise."""
        with pytest.raises(ValueError, match="No channel with pulse type 'W'"):
            plot_function(_make_sfc_anndata(), pulse_type="W")


# ---------------------------------------------------------------------------
# Colours and console output
# ---------------------------------------------------------------------------


class TestColorsAndConsole:
    """Colour resolution priority and the QC console block."""

    def test_stored_palette_is_used(self, sfc_anndata: anndata.AnnData) -> None:
        sfc_anndata.uns["color_palette"] = {_sample_names()[0]: "#123456"}
        color_map = _build_color_map(sfc_anndata, _sample_names())
        assert color_map[_sample_names()[0]] == "#123456"

    def test_preferred_condition_colors_win_over_stored_palette(
        self, sfc_anndata: anndata.AnnData
    ) -> None:
        from mito_marker.controlled_vocabulary import PREFERRED_CONDITION_COLORS

        preferred_key = next(iter(PREFERRED_CONDITION_COLORS))
        sfc_anndata.uns["color_palette"] = {preferred_key: "#000000"}
        color_map = _build_color_map(sfc_anndata, [preferred_key])
        assert color_map[preferred_key] == PREFERRED_CONDITION_COLORS[preferred_key]

    def test_unassigned_samples_get_distinct_non_grey_colors(
        self, sfc_anndata: anndata.AnnData
    ) -> None:
        color_map = _build_color_map(sfc_anndata, _sample_names())
        assert len(set(color_map.values())) == 3
        assert "#999999" not in color_map.values()

    def test_sample_labels_are_shortened_for_the_legend(self) -> None:
        assert _shorten_sample_label("short.fcs") == "short"
        long_label = _shorten_sample_label("g" * 60 + ".fcs")
        assert len(long_label) == 28 and "..." in long_label

    def test_console_reports_context_channels_and_samples(
        self, sfc_anndata: anndata.AnnData, capsys: pytest.CaptureFixture
    ) -> None:
        plot_channel_intensity_line(sfc_anndata, channels=["V-A"])
        console_output = capsys.readouterr().out

        assert "PLOT CONTEXT" in console_output
        assert "Using data from: 'raw .X'" in console_output
        assert "3 channel(s) | 3 sample(s) | statistic=mean" in console_output
        assert "V1-A, V2-A, V10-A" in console_output
        assert _sample_names()[0] in console_output

    def test_console_reports_the_active_layer(self, capsys: pytest.CaptureFixture) -> None:
        plot_channel_intensity_line(_make_sfc_anndata(include_layer=True), channels=["V1-A"])
        assert "Using data from: 'arcsinh__zscore_col'" in capsys.readouterr().out
