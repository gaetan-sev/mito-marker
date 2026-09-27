"""
mito_marker.inspector

Interactive exploration tools for non-code experts.

Provides a single entry point for visualising any major data structure used
in this package (AnnData objects, pandas DataFrames) without writing Python.

In a Jupyter or Google Colab notebook, functions in this module display
rich tabbed widgets powered by ipywidgets.  In a plain Python console
(scripts, pytest), the same functions fall back to formatted text output
that mirrors the QC style used throughout the package.

Public API:
    inspect_anndata(anndata_object, ...)
        Display a five-tab explorer for an AnnData object.
"""

from mito_marker.inspector.anndata_inspector import inspect_anndata

__all__ = ["inspect_anndata"]
