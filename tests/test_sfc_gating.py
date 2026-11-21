"""
test_sfc_gating.py

Tests for the P1 and DR+ rules (mito_marker.sfc.gating).

P1: events with an FSC-A or SSC-A area <= 1, or saturated, are removed; a $PnR
keyword below the data (seen in FlowJo exports) disables the saturation check
instead of removing every event.
DR+: the threshold is the 99.9th percentile of the paired unstained (NM) tube; the
two alerts fire on a stretched NM top (bright tail or second population) and when
many stained events sit between the NM's 99th and 99.99th percentiles.
Pairing: NM and DR tubes of the same subject are matched by filename words.
"""

import os

import numpy as np
import pandas as pd
import pytest

from mito_marker.sfc.fcs_io import read_fcs_events
from mito_marker.sfc.gating import (
    compute_deep_red_alerts,
    compute_deep_red_threshold,
    compute_p1_mask,
    gate_deep_red,
    gate_p1,
    pair_stained_unstained,
)

SATURATED = 4194303.0


def scatter_channels(n_events: int, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    return {"FSC-A": rng.uniform(100, 1e5, n_events), "SSC-A": rng.uniform(100, 1e5, n_events),
            "Time": np.arange(1, n_events + 1, dtype=float)}


# ---------------------------------------------------------------- P1
def test_p1_removes_non_positive_and_saturated_areas(write_synthetic_fcs):
    channels = scatter_channels(100)
    channels["FSC-A"][:5] = [0.0, -20.0, 1.0, 0.5, SATURATED]
    channels["SSC-A"][5:7] = [-3.0, SATURATED]
    mask = compute_p1_mask(read_fcs_events(write_synthetic_fcs("p1.fcs", channels)))
    assert not mask[:7].any()
    assert mask[7:].all()


def test_p1_replaces_an_implausible_range_keyword_by_the_file_usual_range(write_synthetic_fcs, capsys):
    channels = scatter_channels(50)
    channels["FSC-A"][0] = -5.0
    channels["FSC-A"][1] = SATURATED
    channels["R2-A"] = np.full(50, 100.0)  # a detector with the usual Aurora range (4194304)
    events = read_fcs_events(write_synthetic_fcs("export.fcs", channels, ranges={"FSC-A": 1, "SSC-A": 1}))
    mask = compute_p1_mask(events)
    # Not every event removed: the non-positive and the saturated events only.
    assert mask.sum() == 48 and not mask[0] and not mask[1]
    assert "usual range (4194304.0)" in capsys.readouterr().out


def test_gate_p1_writes_kept_and_flagged_files(write_synthetic_fcs, tmp_path):
    channels = scatter_channels(200)
    channels["FSC-A"][:20] = -1.0
    input_folder = tmp_path / "flowai"
    write_synthetic_fcs("WORM_001_D02_AL_DR_FlowAIGoodEvents.fcs", channels, folder=input_folder)
    summary = gate_p1(str(input_folder), str(tmp_path / "p1"), output_mode="both")
    assert summary.events_kept.iloc[0] == 180
    kept = read_fcs_events(str(tmp_path / "p1" / "WORM_001_D02_AL_DR_FlowAIGoodEvents_P1.fcs"))
    flagged = read_fcs_events(str(tmp_path / "p1" / "WORM_001_D02_AL_DR_FlowAIGoodEvents_P1Flagged.fcs"))
    assert kept.n_events == 180 and flagged.n_events == 200
    assert int(np.sum(flagged.channel_values("P1_Gate") == 2000.0)) == 180
    assert os.path.exists(tmp_path / "p1" / "p1_summary.csv")


# ---------------------------------------------------------------- pairing
def test_pairing_matches_same_subject_and_reports_missing(write_synthetic_fcs, tmp_path):
    folder = tmp_path / "p1"
    channels = scatter_channels(10)
    for name in ["WORM_001_D02_AL_DR", "WORM_001_D02_AL_NM", "WORM_002_D02_AL_DR",
                 "KF_001_15W_M1_DR_TMRM", "KF_001_15W_M1_NM", "KF_001_15W_M1_TMRM"]:
        write_synthetic_fcs(f"export_{name}_FlowAIGoodEvents_P1.fcs", channels, folder=folder)
    pairs = pair_stained_unstained(str(folder)).set_index("stained_file")
    assert pairs.loc["export_WORM_001_D02_AL_DR_FlowAIGoodEvents_P1.fcs", "unstained_file"] == "export_WORM_001_D02_AL_NM_FlowAIGoodEvents_P1.fcs"
    assert pairs.loc["export_KF_001_15W_M1_DR_TMRM_FlowAIGoodEvents_P1.fcs", "unstained_file"] == "export_KF_001_15W_M1_NM_FlowAIGoodEvents_P1.fcs"
    assert pairs.loc["export_WORM_002_D02_AL_DR_FlowAIGoodEvents_P1.fcs", "match"] == "no unstained tube found"
    assert "export_KF_001_15W_M1_TMRM_FlowAIGoodEvents_P1.fcs" not in pairs.index  # TMRM only: not Deep Red


# ---------------------------------------------------------------- threshold and alerts
def test_threshold_is_the_unstained_quantile():
    values = np.arange(1, 10001, dtype=float)
    assert compute_deep_red_threshold(values, 0.999) == pytest.approx(np.quantile(values, 0.999))
    with pytest.raises(ValueError):
        compute_deep_red_threshold(np.array([]))
    with pytest.raises(ValueError):
        compute_deep_red_threshold(values, 1.5)


def test_alerts_fire_on_a_stretched_unstained_top_and_on_threshold_sensitivity():
    rng = np.random.default_rng(1)
    clean_unstained = rng.normal(0, 150, 20000)
    well_separated_stained = rng.normal(60000, 8000, 20000)
    clean = compute_deep_red_alerts(clean_unstained, well_separated_stained)
    assert not clean["unstained_tail_alert"] and not clean["threshold_sensitivity_alert"]

    # A bright tail spread over several decades in the NM tube (as in the Smurf NM tubes,
    # median spread 1.44 in EXP-067) stretches its top.
    tailed_unstained = np.concatenate([clean_unstained, rng.lognormal(np.log(3000), 1.5, 200)])
    tailed = compute_deep_red_alerts(tailed_unstained, well_separated_stained)
    assert tailed["unstained_tail_alert"]

    # Most stained events between the NM's 99th and 99.99th percentiles: the result depends on the trade-off.
    low, high = np.quantile(tailed_unstained, [0.99, 0.9999])
    overlapping_stained = rng.uniform(low, high, 20000)
    overlap = compute_deep_red_alerts(tailed_unstained, overlapping_stained)
    assert overlap["threshold_sensitivity_points"] > 90 and overlap["threshold_sensitivity_alert"]


# ---------------------------------------------------------------- gate_deep_red
def build_pair(write_synthetic_fcs, folder, subject: str, seed: int, bright_share: float = 0.5) -> None:
    rng = np.random.default_rng(seed)
    n_events = 4000
    unstained = scatter_channels(n_events, seed)
    unstained["R2-A"] = rng.normal(0, 150, n_events)
    stained = scatter_channels(n_events, seed + 100)
    bright = rng.random(n_events) < bright_share
    stained["R2-A"] = np.where(bright, rng.normal(60000, 8000, n_events), rng.normal(0, 150, n_events))
    write_synthetic_fcs(f"{subject}_AL_NM_FlowAIGoodEvents_P1.fcs", unstained, folder=folder)
    write_synthetic_fcs(f"{subject}_AL_DR_FlowAIGoodEvents_P1.fcs", stained, folder=folder)


def test_gate_deep_red_end_to_end(write_synthetic_fcs, tmp_path):
    folder = tmp_path / "p1"
    build_pair(write_synthetic_fcs, folder, "WORM_001_D02", seed=1, bright_share=0.5)
    build_pair(write_synthetic_fcs, folder, "WORM_002_D02", seed=2, bright_share=0.2)
    summary = gate_deep_red(str(folder), str(tmp_path / "dr"), output_mode="both").set_index("stained_file")
    first = summary.loc["WORM_001_D02_AL_DR_FlowAIGoodEvents_P1.fcs"]
    assert first.percent_dr_positive == pytest.approx(50, abs=3)
    assert summary.loc["WORM_002_D02_AL_DR_FlowAIGoodEvents_P1.fcs", "percent_dr_positive"] == pytest.approx(20, abs=3)
    assert first.percent_unstained_above_threshold <= 0.2
    kept = read_fcs_events(str(tmp_path / "dr" / "WORM_001_D02_AL_DR_FlowAIGoodEvents_P1_DR+.fcs"))
    assert kept.n_events == first.events_dr_positive
    flagged = read_fcs_events(str(tmp_path / "dr" / "WORM_001_D02_AL_DR_FlowAIGoodEvents_P1_DRFlagged.fcs"))
    assert int(np.sum(flagged.channel_values("DR_Gate") == 2000.0)) == first.events_dr_positive
    assert os.path.exists(tmp_path / "dr" / "deep_red_summary.csv")
    assert len(os.listdir(tmp_path / "dr" / "qc_figures")) == 2


def test_gate_deep_red_pools_several_unstained_tubes_and_reads_pairs_from_csv(write_synthetic_fcs, tmp_path):
    folder = tmp_path / "p1"
    build_pair(write_synthetic_fcs, folder, "WORM_001_D02", seed=1)
    build_pair(write_synthetic_fcs, folder, "WORM_002_D02", seed=2)
    pairs = pd.DataFrame({"stained_file": ["WORM_001_D02_AL_DR_FlowAIGoodEvents_P1.fcs"],
                          "unstained_file": ["WORM_001_D02_AL_NM_FlowAIGoodEvents_P1.fcs; WORM_002_D02_AL_NM_FlowAIGoodEvents_P1.fcs"]})
    pairs_path = tmp_path / "pairs.csv"
    pairs.to_csv(pairs_path, index=False)
    summary = gate_deep_red(str(folder), str(tmp_path / "dr"), pairs=str(pairs_path), save_qc_figures=False)
    pooled = np.concatenate([read_fcs_events(str(folder / f"WORM_00{number}_D02_AL_NM_FlowAIGoodEvents_P1.fcs")).channel_values("R2-A")
                             for number in (1, 2)])
    assert summary.threshold.iloc[0] == pytest.approx(np.quantile(pooled, 0.999), rel=1e-3)


def test_gate_deep_red_skips_missing_and_empty_references(write_synthetic_fcs, tmp_path):
    folder = tmp_path / "p1"
    build_pair(write_synthetic_fcs, folder, "WORM_001_D02", seed=1)
    empty = scatter_channels(0)
    empty["R2-A"] = np.zeros(0)
    write_synthetic_fcs("WORM_003_D02_AL_NM_FlowAIGoodEvents_P1.fcs", empty, folder=folder)
    pairs = pd.DataFrame({"stained_file": ["WORM_001_D02_AL_DR_FlowAIGoodEvents_P1.fcs"] * 2,
                          "unstained_file": [np.nan, "WORM_003_D02_AL_NM_FlowAIGoodEvents_P1.fcs"]})
    summary = gate_deep_red(str(folder), str(tmp_path / "dr"), pairs=pairs, save_qc_figures=False)
    assert list(summary.status) == ["no unstained tube", "unstained tube empty"]
    with pytest.raises(ValueError, match="missing the column"):
        gate_deep_red(str(folder), str(tmp_path / "dr2"), pairs=pd.DataFrame({"stained_file": ["x.fcs"]}))
