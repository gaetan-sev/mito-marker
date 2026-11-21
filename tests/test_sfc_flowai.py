"""
test_sfc_flowai.py

Tests for the flowAI port (mito_marker.sfc.flowai).

Each check is tested on an acquisition built so that the right answer is known:
  - flow rate: a steady (Poisson) event rate with one burst bin; the burst events
    must be removed and most of the others kept;
  - signal acquisition: a channel whose level jumps after 30 % of the events; the
    first 30 % must be removed, and nothing when there is no jump;
  - dynamic range: saturated and extremely negative values are removed, except on
    FSC / SSC channels, which flowAI does not check.
The R behaviours the port must keep (right-closed time bins, the Christiano–Fitzgerald
weights) are tested on hand-computed values. The agreement with the FlowJo plugin
itself (identical labels on 464 of 468 acquisitions) was checked in mito-marker-lab,
EXP-067, on raw data that cannot be committed here.
"""

import os

import numpy as np
import pandas as pd
import pytest

from mito_marker.sfc.fcs_io import read_fcs_events
from mito_marker.sfc.flowai import (
    _binary_segmentation,
    _christiano_fitzgerald_cycle,
    _flow_rate_bins,
    compute_flowai_quality,
    flowai_label_channel,
    run_flowai,
)

TIME_STEP = 0.0001
TICKS_PER_BIN = 1000  # 0.1 s at a time step of 0.0001 s


def steady_acquisition(seed: int = 1, n_bins: int = 200, mean_per_bin: float = 40.0, burst_bin: int | None = None) -> dict:
    """Events spread in time with a Poisson count per 0.1 s bin, optionally one burst bin (x10)."""
    rng = np.random.default_rng(seed)
    times = []
    for bin_index in range(n_bins):
        count = rng.poisson(mean_per_bin) * (10 if bin_index == burst_bin else 1)
        times.append(np.sort(rng.uniform(bin_index * TICKS_PER_BIN + 1, (bin_index + 1) * TICKS_PER_BIN, count)))
    time_values = np.concatenate(times)
    n_events = len(time_values)
    return {"FSC-A": rng.uniform(1e3, 1e5, n_events), "SSC-A": rng.uniform(1e3, 1e5, n_events),
            "R1-A": rng.normal(2000, 200, n_events), "Time": time_values}


def test_flow_rate_bins_follow_r_hist_right_closed_intervals():
    counts, bin_of_event = _flow_rate_bins(np.array([0.0, 10.0, 11.0, 20.0]), time_step=0.01, second_fraction=0.1)
    # (0, 10] holds 0 (lowest break included) and 10; (10, 20] holds 11 and 20; flowAI prepends a 0.
    assert counts[0] == 0 and counts[1] == 2 and counts[2] == 2
    assert bin_of_event.tolist() == [2, 2, 3, 3]


def test_christiano_fitzgerald_weights_are_the_ideal_band_pass_weights():
    unit = np.zeros(5)
    unit[2] = 1.0
    cycle = _christiano_fitzgerald_cycle(unit)
    # Periods 2 to 8: B0 = (pi - pi/4) / pi = 0.75; B1 = (sin(pi) - sin(pi/4)) / pi.
    assert cycle[2] == pytest.approx(0.75)
    assert cycle[3] == pytest.approx((np.sin(np.pi) - np.sin(np.pi / 4)) / np.pi)
    assert cycle[1] == pytest.approx(cycle[3])


def test_binary_segmentation_finds_a_level_change_and_respects_the_penalty():
    rng = np.random.default_rng(3)
    series = np.concatenate([rng.normal(0, 1, 50), rng.normal(8, 1, 50)])
    assert _binary_segmentation(series, penalty=200.0, max_changepoints=3, min_segment=2)[0] == 50
    flat = rng.normal(0, 1, 100)
    assert _binary_segmentation(flat, penalty=200.0, max_changepoints=3, min_segment=2) == [100]


def test_flow_rate_check_removes_a_burst_bin(write_synthetic_fcs):
    channels = steady_acquisition(burst_bin=120)
    events = read_fcs_events(write_synthetic_fcs("burst.fcs", channels))
    quality = compute_flowai_quality(events, checks=("flow_rate",))
    in_burst = (events.channel_values("Time") > 120 * TICKS_PER_BIN) & (events.channel_values("Time") <= 121 * TICKS_PER_BIN)
    assert not quality.flow_rate_pass[in_burst].any()
    assert quality.flow_rate_pass[~in_burst].mean() > 0.8
    assert quality.overall_pass.tolist() == quality.flow_rate_pass.tolist()


def test_signal_acquisition_check_keeps_the_longest_stable_stretch(write_synthetic_fcs):
    rng = np.random.default_rng(5)
    n_events = 10000
    level = np.where(np.arange(n_events) < 3000, 1000.0, 5000.0)
    channels = {"R1-A": level + rng.normal(0, 20, n_events), "R2-A": rng.normal(500, 20, n_events),
                "Time": np.arange(1, n_events + 1, dtype=float)}
    quality = compute_flowai_quality(read_fcs_events(write_synthetic_fcs("shift.fcs", channels)), checks=("signal_acquisition",))
    removed = 1 - quality.signal_acquisition_pass.mean()
    assert 0.28 <= removed <= 0.32
    assert quality.signal_acquisition_pass[-1] and not quality.signal_acquisition_pass[0]

    channels["R1-A"] = 1000.0 + rng.normal(0, 20, n_events)
    stable = compute_flowai_quality(read_fcs_events(write_synthetic_fcs("stable.fcs", channels)), checks=("signal_acquisition",))
    assert stable.signal_acquisition_pass.all()


def test_dynamic_range_removes_saturated_and_extreme_negative_values_but_not_scatter(write_synthetic_fcs):
    rng = np.random.default_rng(7)
    n_events = 2000
    fluorescence = rng.normal(-50, 10, n_events)       # negative values with a small spread
    fluorescence[10] = 4194303.0                        # saturated
    fluorescence[20] = -100000.0                        # far below the negative values
    forward_scatter = rng.uniform(1e3, 1e5, n_events)
    forward_scatter[30] = 4194303.0                     # saturated scatter: not checked by flowAI
    channels = {"FSC-A": forward_scatter, "R1-A": fluorescence, "Time": np.arange(1, n_events + 1, dtype=float)}
    quality = compute_flowai_quality(read_fcs_events(write_synthetic_fcs("range.fcs", channels)), checks=("dynamic_range",))
    assert not quality.dynamic_range_pass[10]
    assert not quality.dynamic_range_pass[20]
    assert quality.dynamic_range_pass[30]
    assert quality.dynamic_range_pass.mean() > 0.99


def test_plugin_order_only_differs_when_time_goes_back(write_synthetic_fcs):
    channels = steady_acquisition(seed=11, n_bins=100, burst_bin=40)
    events = read_fcs_events(write_synthetic_fcs("sorted.fcs", channels))
    own = compute_flowai_quality(events, checks=("flow_rate",))
    plugin = compute_flowai_quality(events, checks=("flow_rate",), match_flowjo_plugin_order=True)
    assert own.time_is_increasing and own.overall_pass.tolist() == plugin.overall_pass.tolist()

    # Second half acquired first: the Time channel goes back once.
    half = len(channels["Time"]) // 2
    reordered = {name: np.concatenate([values[half:], values[:half]]) for name, values in channels.items()}
    events = read_fcs_events(write_synthetic_fcs("reset.fcs", reordered))
    own = compute_flowai_quality(events, checks=("flow_rate",))
    plugin = compute_flowai_quality(events, checks=("flow_rate",), match_flowjo_plugin_order=True)
    assert not own.time_is_increasing and own.time_reset_count == 1
    order = np.argsort(events.channel_values("Time"), kind="stable")
    # The plugin's label i is the label of the i-th event in time order.
    assert plugin.overall_pass.tolist() == own.overall_pass[order].tolist()


def test_time_above_its_range_is_cut_like_flowcore(write_synthetic_fcs):
    channels = steady_acquisition(n_bins=50)
    channels["Time"][-20:] = 80 * TICKS_PER_BIN  # beyond the declared range, as in some Aurora files
    events = read_fcs_events(write_synthetic_fcs("late.fcs", channels, ranges={"Time": 50 * TICKS_PER_BIN}))
    quality = compute_flowai_quality(events, checks=("flow_rate",))
    # Cut at $PnR - 1 = 49999 ticks: the bins stop at 5 s instead of 8 s.
    assert len(quality.flow_rate_bin_counts) == 51


def test_unknown_check_and_side_raise(write_synthetic_fcs):
    events = read_fcs_events(write_synthetic_fcs("small.fcs", steady_acquisition(n_bins=20)))
    with pytest.raises(ValueError, match="Unknown flowAI checks"):
        compute_flowai_quality(events, checks=("flow_speed",))
    with pytest.raises(ValueError, match="dynamic_range_side"):
        compute_flowai_quality(events, dynamic_range_side="middle")


def test_label_channel_uses_plugin_values(write_synthetic_fcs):
    events = read_fcs_events(write_synthetic_fcs("labels.fcs", steady_acquisition(n_bins=50, burst_bin=25)))
    labels = flowai_label_channel(compute_flowai_quality(events))
    assert set(np.unique(labels)) <= {2000.0, 6000.0}


def test_run_flowai_writes_fcs_files_summary_and_figures(write_synthetic_fcs, tmp_path):
    input_folder = tmp_path / "raw"
    write_synthetic_fcs("WORM_001_D02_AL_DR.fcs", steady_acquisition(seed=1, burst_bin=50), folder=input_folder)
    write_synthetic_fcs("WORM_001_D02_AL_NM.fcs", steady_acquisition(seed=2), folder=input_folder)
    output_folder = tmp_path / "01_flowai"
    summary = run_flowai(str(input_folder), str(output_folder), output_mode="both")
    assert list(summary.status) == ["ok", "ok"]
    kept = read_fcs_events(str(output_folder / "WORM_001_D02_AL_DR_FlowAIGoodEvents.fcs"))
    flagged = read_fcs_events(str(output_folder / "WORM_001_D02_AL_DR_FlowAIFlagged.fcs"))
    assert kept.n_events == summary.events_kept.iloc[0]
    assert set(np.unique(kept.channel_values("FlowAI"))) == {2000.0}
    assert flagged.n_events == summary.events_in.iloc[0]
    assert int(np.sum(flagged.channel_values("FlowAI") == 2000.0)) == kept.n_events
    assert os.path.exists(output_folder / "flowai_summary.csv")
    assert len(os.listdir(output_folder / "qc_figures")) == 2
    pd.testing.assert_frame_equal(pd.read_csv(output_folder / "flowai_summary.csv")[["input_file", "events_kept"]],
                                  summary[["input_file", "events_kept"]])
    with pytest.raises(FileExistsError):
        run_flowai(str(input_folder), str(output_folder), output_mode="both")
    run_flowai(str(input_folder), str(output_folder), output_mode="both", overwrite=True, save_qc_figures=False)


def test_run_flowai_skips_an_empty_file_and_rejects_bad_mode(write_synthetic_fcs, tmp_path):
    input_folder = tmp_path / "raw"
    write_synthetic_fcs("good.fcs", steady_acquisition(n_bins=50), folder=input_folder)
    (input_folder / "empty.fcs").write_bytes(b"")
    summary = run_flowai(str(input_folder), str(tmp_path / "out"), save_qc_figures=False)
    assert summary.set_index("input_file").loc["empty.fcs", "status"].startswith("unreadable")
    with pytest.raises(ValueError, match="output_mode"):
        run_flowai(str(input_folder), str(tmp_path / "out2"), output_mode="everything")
