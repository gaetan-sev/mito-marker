"""
mito_marker.sfc

Spectral flow cytometry (SFC) ingestion pipeline.

Public API:
    ingest_sfc_folder(data_directory_path, output_file_path, ...)
        Ingest all .fcs files in a directory into a single AnnData and save as .h5ad.

    enrich_with_clinical_data(anndata_object, clinical_csv_path, ...)
        Merge per-subject clinical variables from a CSV into AnnData .obs,
        linked via subject_ID.
"""

from mito_marker.sfc.clinical_data import enrich_with_clinical_data
from mito_marker.sfc.ingestion import ingest_sfc_folder

__all__ = ["ingest_sfc_folder", "enrich_with_clinical_data"]
