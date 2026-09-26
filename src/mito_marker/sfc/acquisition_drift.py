"""
acquisition_drift.py

Measure how much a spectral flow cytometry measurement moves between the first
and the last events of a single acquisition.

WHY THIS IS A PACKAGE FUNCTION AND NOT AN EXPERIMENT SCRIPT

EXP-017 found, as a side observation, that the median event size moves by a
factor of 1.31 across one acquisition tube (median over 28 animals) and by a
factor of 2.2 in the worst tube. That number is not a result about
mitochondria: it is a property of the instrument and the sample as they were
run, and it sets a FLOOR on how small a between-group difference can be
believed. A group difference smaller than the drift inside a single tube is not
interpretable whatever its p-value.

Every SFC analysis in this project needs that floor before it reports an effect
size, so the measurement belongs in the package rather than in one experiment's
pipeline folder.

WHAT IS MEASURED

The acquisition is split into equal windows of the `Time` channel. Within each
window the median of a chosen channel is taken on a log scale, because scatter
and fluorescence intensities are multiplicative quantities: "drifted by a
factor of 1.3" is the meaningful statement, "drifted by 4,000 arbitrary units"
is not.

Two numbers per acquisition:

  - the END-TO-END drift, the last window's median over the first window's, which
    says whether the measurement moved in a consistent direction;
  - the FULL RANGE, the largest window median over the smallest, which catches a
    tube that wandered and came back, and is the conservative floor.

WHAT IT CANNOT TELL YOU

Whether the drift is the instrument (a clogging nozzle, a settling sheath
pressure, detector warm-up) or the sample (larger particles sedimenting in the
tube, aggregates forming over minutes). Both produce the same signature and
only a bead control run at the start and end of a session can separate them.
The function therefore reports the drift and never attributes it.
"""

from typing import List

import anndata
import numpy as np
import pandas as pd

# Below this many events a window median is too noisy to compare with another
# window's, so the window is dropped and the drop is reported.
DEFAULT_MINIMUM_EVENTS_PER_WINDOW = 100

# The number of equal time windows an acquisition is split into. Ten is the
# EXP-017 value: enough to see a trend, few enough that each window keeps
# hundreds of events in the smallest tube of that cohort.
DEFAULT_TIME_WINDOW_COUNT = 10


def measure_acquisition_drift(
    spectral_cytometry_anndata: anndata.AnnData,
    channel_name: str = "FSC-A",
    time_channel_name: str = "Time",
    subject_column: str = "subject_ID",
    n_time_windows: int = DEFAULT_TIME_WINDOW_COUNT,
    minimum_events_per_window: int = DEFAULT_MINIMUM_EVENTS_PER_WINDOW,
) -> pd.DataFrame:
    """
    Measure within-acquisition drift of one channel, per subject.

    The acquisition is split into equal windows of the time channel and the
    median of the chosen channel is taken in each, on a log scale. Events whose
    channel value is not strictly positive cannot be logged and are dropped per
    subject; the count is reported in the returned table.

    Arguments:
        spectral_cytometry_anndata: AnnData whose `.X` holds the channels and
            whose `.obs` holds one row per event.
        channel_name: the channel whose drift is measured, as named in `.var`.
        time_channel_name: the acquisition time channel, as named in `.var`.
        subject_column: the `.obs` column identifying one acquisition.
        n_time_windows: how many equal time windows to split each acquisition
            into.
        minimum_events_per_window: windows holding fewer events than this are
            dropped from that subject's drift.

    Returns:
        One row per subject with the number of usable events, the first and
        last window medians on the log scale, the end-to-end drift as a ratio,
        and the full-range drift as a ratio. Subjects with fewer than two
        usable windows are returned with NaN drifts rather than dropped, so a
        caller always sees every subject it passed in.
    """
    for required_channel in (channel_name, time_channel_name):
        assert required_channel in spectral_cytometry_anndata.var_names, (
            f"channel '{required_channel}' is not in .var_names")
    assert subject_column in spectral_cytometry_anndata.obs.columns, (
        f"'{subject_column}' is not a column of .obs")
    assert n_time_windows >= 2, "at least two time windows are needed for a drift"

    channel_values = np.asarray(
        spectral_cytometry_anndata[:, channel_name].X, dtype=np.float64).ravel()
    time_values = np.asarray(
        spectral_cytometry_anndata[:, time_channel_name].X, dtype=np.float64).ravel()
    subject_labels = spectral_cytometry_anndata.obs[subject_column].to_numpy()

    rows: List[dict] = []
    for subject in pd.unique(subject_labels):
        belongs_to_subject = subject_labels == subject
        subject_channel = channel_values[belongs_to_subject]
        subject_time = time_values[belongs_to_subject]
        # A scatter or fluorescence value of zero or below is a baseline
        # artefact, not a measurement, and cannot be put on a log scale.
        is_usable = np.isfinite(subject_channel) & (subject_channel > 0)
        is_usable &= np.isfinite(subject_time)
        n_dropped = int((~is_usable).sum())
        subject_channel = subject_channel[is_usable]
        subject_time = subject_time[is_usable]

        window_medians = _window_medians(
            subject_channel, subject_time, n_time_windows, minimum_events_per_window)
        if len(window_medians) < 2:
            rows.append({
                subject_column: subject, "channel": channel_name,
                "n_events_usable": int(len(subject_channel)),
                "n_events_dropped_non_positive": n_dropped,
                "n_windows_usable": len(window_medians),
                "first_window_log10_median": float("nan"),
                "last_window_log10_median": float("nan"),
                "end_to_end_drift_log10": float("nan"),
                "end_to_end_drift_ratio": float("nan"),
                "full_range_drift_log10": float("nan"),
                "full_range_drift_ratio": float("nan"),
            })
            continue

        end_to_end_log10 = float(window_medians[-1] - window_medians[0])
        full_range_log10 = float(np.max(window_medians) - np.min(window_medians))
        rows.append({
            subject_column: subject, "channel": channel_name,
            "n_events_usable": int(len(subject_channel)),
            "n_events_dropped_non_positive": n_dropped,
            "n_windows_usable": len(window_medians),
            "first_window_log10_median": float(window_medians[0]),
            "last_window_log10_median": float(window_medians[-1]),
            "end_to_end_drift_log10": end_to_end_log10,
            "end_to_end_drift_ratio": float(10.0 ** end_to_end_log10),
            "full_range_drift_log10": full_range_log10,
            "full_range_drift_ratio": float(10.0 ** full_range_log10),
        })
    return pd.DataFrame(rows)


def _window_medians(
    channel_values: np.ndarray,
    time_values: np.ndarray,
    n_time_windows: int,
    minimum_events_per_window: int,
) -> np.ndarray:
    """
    The log10 median of one channel in each equal window of the time axis.

    Windows are equal in TIME, not in event count, because an instrument that
    slows down mid-run spends more time producing fewer events and an
    equal-count split would hide exactly that.

    Arguments:
        channel_values: strictly positive channel values of one acquisition.
        time_values: the matching time values.
        n_time_windows: how many equal windows to cut the time axis into.
        minimum_events_per_window: windows below this are dropped.

    Returns:
        The log10 medians of the usable windows, in time order.
    """
    if len(channel_values) == 0:
        return np.asarray([], dtype=float)
    time_minimum, time_maximum = float(time_values.min()), float(time_values.max())
    if not np.isfinite(time_minimum) or time_maximum <= time_minimum:
        return np.asarray([], dtype=float)
    boundaries = np.linspace(time_minimum, time_maximum, n_time_windows + 1)
    log_channel = np.log10(channel_values)

    medians: List[float] = []
    for window_index in range(n_time_windows):
        lower = boundaries[window_index]
        # The last window includes its upper boundary so the final events are
        # not silently discarded.
        upper = boundaries[window_index + 1]
        in_window = (time_values >= lower) & (
            time_values <= upper if window_index == n_time_windows - 1
            else time_values < upper)
        if int(in_window.sum()) < minimum_events_per_window:
            continue
        medians.append(float(np.median(log_channel[in_window])))
    return np.asarray(medians, dtype=float)


def believable_effect_floor(drift_table: pd.DataFrame) -> dict[str, float]:
    """
    The smallest between-group ratio this acquisition set can support.

    A difference between two groups that is smaller than the movement inside a
    single tube cannot be attributed to the groups. The median full-range drift
    is the routine floor and the worst tube is the conservative one.

    Arguments:
        drift_table: the table returned by measure_acquisition_drift().

    Returns:
        Dict with the median and worst full-range drift ratios, the median
        end-to-end drift ratio, and how many acquisitions the floor rests on.
    """
    usable = drift_table[drift_table["full_range_drift_ratio"].notna()]
    if len(usable) == 0:
        return {"n_acquisitions": 0.0, "median_full_range_ratio": float("nan"),
                "worst_full_range_ratio": float("nan"),
                "median_end_to_end_ratio": float("nan")}
    return {
        "n_acquisitions": float(len(usable)),
        "median_full_range_ratio": float(usable["full_range_drift_ratio"].median()),
        "worst_full_range_ratio": float(usable["full_range_drift_ratio"].max()),
        "median_end_to_end_ratio": float(
            (10.0 ** usable["end_to_end_drift_log10"].abs()).median()),
    }


def print_acquisition_drift_report(
    drift_table: pd.DataFrame,
    subject_column: str = "subject_ID",
) -> None:
    """
    Print the drift QC block for an acquisition set.

    Arguments:
        drift_table: the table returned by measure_acquisition_drift().
        subject_column: the subject column name used when measuring.

    Returns:
        None
    """
    channel_name = (drift_table["channel"].iloc[0] if len(drift_table) > 0
                    else "unknown")
    floor = believable_effect_floor(drift_table)

    print("  ┌─ ACQUISITION DRIFT REPORT " + "─" * 42)
    print(f"  │ Channel measured        : {channel_name}")
    print(f"  │ Acquisitions            : {len(drift_table)}")
    print(f"  │ Acquisitions measurable : {int(floor['n_acquisitions'])}")
    if len(drift_table) > 0:
        print(f"  │ Events usable (total)   : "
              f"{int(drift_table['n_events_usable'].sum()):,}")
        n_dropped = int(drift_table["n_events_dropped_non_positive"].sum())
        print(f"  │ Events dropped (<= 0)   : {n_dropped:,}")
    print("  ├" + "─" * 67)
    print(f"  │ Median end-to-end drift : "
          f"x{floor['median_end_to_end_ratio']:.3f}")
    print(f"  │ Median full-range drift : "
          f"x{floor['median_full_range_ratio']:.3f}")
    print(f"  │ Worst full-range drift  : "
          f"x{floor['worst_full_range_ratio']:.3f}")
    print("  ├" + "─" * 67)
    print("  │ READING: a between-group difference smaller than the median")
    print("  │ full-range drift moved inside ONE tube. It is not interpretable")
    print("  │ as a group difference whatever its p-value.")
    print("  └" + "─" * 67)

    # idxmax() on an all-NaN column returns NaN, so the worst acquisition is
    # only named when at least one was measurable.
    measurable = drift_table[drift_table["full_range_drift_ratio"].notna()]
    if len(measurable) > 0:
        worst = measurable.loc[measurable["full_range_drift_ratio"].idxmax()]
        print(f"  Worst acquisition: {worst[subject_column]} "
              f"(x{worst['full_range_drift_ratio']:.3f} across "
              f"{int(worst['n_windows_usable'])} windows)")
    unmeasurable = drift_table[drift_table["full_range_drift_ratio"].isna()]
    for _, row in unmeasurable.iterrows():
        print(f"  ! {row[subject_column]}: not measurable "
              f"({int(row['n_windows_usable'])} usable windows, "
              f"{int(row['n_events_usable']):,} usable events)")
