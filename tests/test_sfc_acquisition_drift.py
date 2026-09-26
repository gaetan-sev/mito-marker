"""
test_sfc_acquisition_drift.py

Tests for measure_acquisition_drift(), believable_effect_floor() and
print_acquisition_drift_report().

The strategy throughout is to build acquisitions whose drift is KNOWN by
construction and check the function recovers it:

  - a flat acquisition must report a drift of about 1.0;
  - an acquisition whose channel is multiplied by a known factor across the run
    must report that factor;
  - an acquisition that wanders up and comes back must report a full-range
    drift larger than its end-to-end drift, which is the whole reason both are
    reported;
  - non-positive values must be dropped and counted, not logged.

No real FCS file is needed: every AnnData is built in memory.
"""

import anndata
import numpy as np
import pandas as pd
import pytest

from mito_marker.sfc.acquisition_drift import (
    believable_effect_floor,
    measure_acquisition_drift,
    print_acquisition_drift_report,
)

RANDOM_SEED = 42
CHANNEL_NAMES = ["FSC-A", "SSC-A", "Time"]


def build_acquisition_anndata(
    subject_to_drift_factor: dict[str, float],
    n_events_per_subject: int = 5000,
    baseline_value: float = 1000.0,
    noise_multiplier: float = 0.0,
) -> anndata.AnnData:
    """
    Build an AnnData whose drift is known by construction.

    Within each subject the time channel runs from 0 to 1 and the FSC-A channel
    is multiplied smoothly from `baseline_value` to
    `baseline_value * drift_factor`, so the true end-to-end drift is exactly the
    drift factor up to the window discretisation.

    Arguments:
        subject_to_drift_factor: subject identifier -> the factor its channel is
            multiplied by across the acquisition.
        n_events_per_subject: events per subject.
        baseline_value: the channel value at the start of the acquisition.
        noise_multiplier: lognormal noise width; 0.0 means noiseless.

    Returns:
        AnnData with the three channels in `.var_names` and a subject column in
        `.obs`.
    """
    generator = np.random.default_rng(RANDOM_SEED)
    measurement_blocks: list[np.ndarray] = []
    subject_labels: list[str] = []
    for subject, drift_factor in subject_to_drift_factor.items():
        time_values = np.linspace(0.0, 1.0, n_events_per_subject)
        channel_values = baseline_value * drift_factor ** time_values
        if noise_multiplier > 0.0:
            channel_values = channel_values * generator.lognormal(
                mean=0.0, sigma=noise_multiplier, size=n_events_per_subject)
        side_scatter = channel_values * 0.5
        measurement_blocks.append(
            np.column_stack([channel_values, side_scatter, time_values]))
        subject_labels.extend([subject] * n_events_per_subject)
    measurements = np.vstack(measurement_blocks)
    observations = pd.DataFrame({"subject_ID": subject_labels})
    variables = pd.DataFrame(index=pd.Index(CHANNEL_NAMES, name="channel"))
    return anndata.AnnData(X=measurements, obs=observations, var=variables)


# ---------------------------------------------------------------------------
# measure_acquisition_drift
# ---------------------------------------------------------------------------
def test_flat_acquisition_reports_no_drift():
    """An acquisition whose channel never moves must report a drift of 1.0."""
    drift_table = measure_acquisition_drift(
        build_acquisition_anndata({"flat": 1.0}))
    assert len(drift_table) == 1
    assert drift_table["end_to_end_drift_ratio"].iloc[0] == pytest.approx(1.0, abs=1e-9)
    assert drift_table["full_range_drift_ratio"].iloc[0] == pytest.approx(1.0, abs=1e-9)


@pytest.mark.parametrize("drift_factor", [1.31, 2.2, 0.5])
def test_known_drift_is_recovered(drift_factor):
    """A planted multiplicative drift must be recovered to within the window width."""
    drift_table = measure_acquisition_drift(
        build_acquisition_anndata({"planted": drift_factor}))
    recovered = drift_table["end_to_end_drift_ratio"].iloc[0]
    # With ten windows the first and last medians sit at the centres of the
    # first and last window, so the recovered span is 9/10 of the full one.
    expected = drift_factor ** 0.9
    assert recovered == pytest.approx(expected, rel=0.02)


def test_full_range_exceeds_end_to_end_when_the_tube_wanders():
    """A tube that rises and returns must show range drift but no end-to-end drift."""
    n_events = 6000
    time_values = np.linspace(0.0, 1.0, n_events)
    # Up then back down: ends where it started, but visits a factor of two.
    channel_values = 1000.0 * 2.0 ** (1.0 - np.abs(2.0 * time_values - 1.0))
    measurements = np.column_stack(
        [channel_values, channel_values * 0.5, time_values])
    wandering = anndata.AnnData(
        X=measurements,
        obs=pd.DataFrame({"subject_ID": ["wanderer"] * n_events}),
        var=pd.DataFrame(index=pd.Index(CHANNEL_NAMES, name="channel")))

    drift_table = measure_acquisition_drift(wandering)
    end_to_end = drift_table["end_to_end_drift_ratio"].iloc[0]
    full_range = drift_table["full_range_drift_ratio"].iloc[0]
    assert end_to_end == pytest.approx(1.0, abs=0.05)
    assert full_range > 1.7
    assert full_range > end_to_end


def test_non_positive_values_are_dropped_and_counted():
    """Events at or below zero cannot be logged; they must be dropped and reported."""
    n_events = 4000
    time_values = np.linspace(0.0, 1.0, n_events)
    channel_values = np.full(n_events, 1000.0)
    channel_values[:500] = -5.0
    channel_values[500:600] = 0.0
    measurements = np.column_stack(
        [channel_values, np.abs(channel_values) * 0.5, time_values])
    with_bad_events = anndata.AnnData(
        X=measurements,
        obs=pd.DataFrame({"subject_ID": ["mixed"] * n_events}),
        var=pd.DataFrame(index=pd.Index(CHANNEL_NAMES, name="channel")))

    drift_table = measure_acquisition_drift(with_bad_events)
    assert drift_table["n_events_dropped_non_positive"].iloc[0] == 600
    assert drift_table["n_events_usable"].iloc[0] == n_events - 600
    assert np.isfinite(drift_table["end_to_end_drift_ratio"].iloc[0])


def test_every_subject_is_returned_in_input_order():
    """A caller must always get back every subject it passed in."""
    subjects = {"first": 1.0, "second": 1.5, "third": 0.8}
    drift_table = measure_acquisition_drift(build_acquisition_anndata(subjects))
    assert list(drift_table["subject_ID"]) == list(subjects)


def test_subject_with_too_few_events_returns_nan_rather_than_vanishing():
    """An unmeasurable acquisition must be reported as NaN, never silently dropped."""
    drift_table = measure_acquisition_drift(
        build_acquisition_anndata({"tiny": 1.5}, n_events_per_subject=50),
        minimum_events_per_window=100)
    assert len(drift_table) == 1
    assert drift_table["n_windows_usable"].iloc[0] == 0
    assert np.isnan(drift_table["end_to_end_drift_ratio"].iloc[0])


def test_the_side_scatter_channel_can_be_measured_too():
    """The channel is a parameter, not a hard-coded FSC-A."""
    drift_table = measure_acquisition_drift(
        build_acquisition_anndata({"planted": 2.0}), channel_name="SSC-A")
    assert drift_table["channel"].iloc[0] == "SSC-A"
    # SSC-A is FSC-A halved, so it carries exactly the same multiplicative drift.
    assert drift_table["end_to_end_drift_ratio"].iloc[0] == pytest.approx(
        2.0 ** 0.9, rel=0.02)


def test_noise_does_not_manufacture_drift():
    """Lognormal noise around a flat signal must not produce a systematic drift."""
    drift_table = measure_acquisition_drift(
        build_acquisition_anndata({"noisy": 1.0}, n_events_per_subject=20000,
                                  noise_multiplier=0.4))
    assert drift_table["end_to_end_drift_ratio"].iloc[0] == pytest.approx(1.0, abs=0.05)


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------
def test_missing_channel_raises():
    """A channel that is not in .var_names must fail loudly."""
    acquisitions = build_acquisition_anndata({"one": 1.0})
    with pytest.raises(AssertionError, match="not in .var_names"):
        measure_acquisition_drift(acquisitions, channel_name="NoSuchChannel")


def test_missing_subject_column_raises():
    """A subject column that is not in .obs must fail loudly."""
    acquisitions = build_acquisition_anndata({"one": 1.0})
    with pytest.raises(AssertionError, match="not a column of .obs"):
        measure_acquisition_drift(acquisitions, subject_column="no_such_column")


def test_one_window_is_refused():
    """A drift needs at least two windows to exist."""
    acquisitions = build_acquisition_anndata({"one": 1.0})
    with pytest.raises(AssertionError, match="at least two time windows"):
        measure_acquisition_drift(acquisitions, n_time_windows=1)


# ---------------------------------------------------------------------------
# believable_effect_floor
# ---------------------------------------------------------------------------
def test_floor_reports_the_worst_tube():
    """The conservative floor is the worst acquisition, not the average one."""
    drift_table = measure_acquisition_drift(
        build_acquisition_anndata({"calm": 1.05, "rough": 2.5}))
    floor = believable_effect_floor(drift_table)
    assert floor["n_acquisitions"] == 2.0
    assert floor["worst_full_range_ratio"] > floor["median_full_range_ratio"]
    assert floor["worst_full_range_ratio"] == pytest.approx(2.5 ** 0.9, rel=0.02)


def test_floor_on_an_empty_table_is_not_a_crash():
    """An unmeasurable set must return NaN, not raise."""
    floor = believable_effect_floor(pd.DataFrame({
        "full_range_drift_ratio": [], "end_to_end_drift_log10": []}))
    assert floor["n_acquisitions"] == 0.0
    assert np.isnan(floor["median_full_range_ratio"])


# ---------------------------------------------------------------------------
# print_acquisition_drift_report
# ---------------------------------------------------------------------------
def test_report_prints_the_floor_and_the_worst_tube(capsys):
    """The QC block must name the channel, the floor and the worst acquisition."""
    drift_table = measure_acquisition_drift(
        build_acquisition_anndata({"calm": 1.05, "rough": 2.5}))
    print_acquisition_drift_report(drift_table)
    printed = capsys.readouterr().out
    assert "ACQUISITION DRIFT REPORT" in printed
    assert "FSC-A" in printed
    assert "Worst acquisition: rough" in printed
    assert "not interpretable" in printed


def test_report_names_unmeasurable_acquisitions(capsys):
    """An acquisition that could not be measured must be named, not hidden."""
    drift_table = measure_acquisition_drift(
        build_acquisition_anndata({"tiny": 1.5}, n_events_per_subject=50),
        minimum_events_per_window=100)
    print_acquisition_drift_report(drift_table)
    printed = capsys.readouterr().out
    assert "tiny" in printed
    assert "not measurable" in printed
