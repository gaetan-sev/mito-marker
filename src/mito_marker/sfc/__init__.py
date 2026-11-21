"""
mito_marker.sfc

Spectral flow cytometry (SFC) ingestion pipeline.

Public API:
    ingest_sfc_folder(data_directory_path, output_file_path, ...)
        Ingest all .fcs files in a directory into a single AnnData and save as .h5ad.

    enrich_with_clinical_data(anndata_object, clinical_csv_path, ...)
        Merge per-subject clinical variables from a CSV into AnnData .obs,
        linked via subject_ID.

    annotate_sfc_conditions(spectral_cytometry_anndata, condition_tokens)
        Add the products added to each tube and the waiting times (read from
        the FCS filename) to .obs, with products declared in the notebook.

    annotate_sfc_filename_columns(spectral_cytometry_anndata, filename_columns)
        Add to .obs the columns declared in the notebook, each value read from a
        word of the FCS filename (e.g. 'S' → smurf_status = 'Smurf').

Preprocessing steps, one call each, .fcs files in, .fcs files out (ADR-022):
    run_flowai(input_directory_path, output_directory_path, ...)
        flowAI quality control, Python port of the FlowJo plugin (ADR-022).
    gate_p1(input_directory_path, output_directory_path, ...)
        P1: FSC-A > 1 and SSC-A > 1, unsaturated (ADR-023).
    pair_stained_unstained(input_directory_path)
        Pair each Deep Red stained tube with its unstained (NM) tube.
    gate_deep_red(input_directory_path, output_directory_path, pairs, ...)
        DR+: R2-A above the 99.9th percentile of the paired NM tube, with alerts (ADR-024).
    read_flowjo_gates(workspace_file_path), apply_flowjo_gates(...)
        Read the gates of a FlowJo workspace and apply them in Python.
    read_fcs_events(fcs_file_path), write_fcs_events(...)
        Read every channel of an .fcs file; write .fcs files FlowJo can reopen.
"""

from mito_marker.sfc.clinical_data import enrich_with_clinical_data
from mito_marker.sfc.conditions import annotate_sfc_conditions
from mito_marker.sfc.fcs_io import FcsEvents, read_fcs_events, write_fcs_events
from mito_marker.sfc.filename_columns import annotate_sfc_filename_columns
from mito_marker.sfc.flowai import compute_flowai_quality, run_flowai
from mito_marker.sfc.flowjo_workspace import apply_flowjo_gates, read_flowjo_gates
from mito_marker.sfc.gating import (
    compute_deep_red_alerts,
    compute_deep_red_threshold,
    compute_p1_mask,
    gate_deep_red,
    gate_p1,
    pair_stained_unstained,
)
from mito_marker.sfc.ingestion import ingest_sfc_folder

__all__ = [
    "ingest_sfc_folder",
    "enrich_with_clinical_data",
    "annotate_sfc_conditions",
    "annotate_sfc_filename_columns",
    "run_flowai",
    "compute_flowai_quality",
    "gate_p1",
    "compute_p1_mask",
    "pair_stained_unstained",
    "gate_deep_red",
    "compute_deep_red_threshold",
    "compute_deep_red_alerts",
    "read_flowjo_gates",
    "apply_flowjo_gates",
    "FcsEvents",
    "read_fcs_events",
    "write_fcs_events",
]
