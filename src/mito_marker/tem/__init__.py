"""
TEM pipeline public API.

Provides ingestion of TEM morphological measurement files (*_MITO_measurements.txt)
produced by ImageJ/Fiji into AnnData format.
"""

from mito_marker.tem.tem_ingestion import ingest_tem_folder

__all__ = ["ingest_tem_folder"]
