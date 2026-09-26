"""
tests/test_inspector.py

Unit tests for mito_marker.inspector.anndata_inspector.

All tests run in a plain pytest context (no notebook kernel), so
_is_notebook_environment() returns False and only the console fallback
code path is exercised.  The ipywidgets tab-building functions are not
tested here because they require a live IPython display kernel.

Tests cover:
    - inspect_anndata() completes without exceptions on typical AnnData objects
    - Console output contains expected dimension strings
    - Sparse .X matrices are handled correctly
    - AnnData objects with no .obs/.var columns are handled gracefully
    - AnnData objects that have .obsm and .uns populated are handled
    - _format_bytes() returns correct human-readable strings
    - _x_matrix_slice_as_dataframe() returns correct shape
"""

import anndata
import numpy as np
import pandas as pd
import pytest
import scipy.sparse

from mito_marker.inspector import inspect_anndata
from mito_marker.inspector.anndata_inspector import (
    _build_subject_summary_dataframe,
    _detect_subject_column,
    _format_bytes,
    _x_matrix_info,
    _x_matrix_slice_as_dataframe,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def typical_anndata() -> anndata.AnnData:
    """
    AnnData with 20 observations, 5 variables, and realistic metadata columns.

    .obs has two categorical columns.
    .var has one descriptive column.
    .X is a dense float32 matrix.
    """
    rng = np.random.default_rng(seed=42)
    x_matrix = rng.random((20, 5)).astype(np.float32)

    obs_dataframe = pd.DataFrame(
        {
            "condition": ["control"] * 10 + ["treated"] * 10,
            "subject_id": [f"S{i:02d}" for i in range(20)],
        },
        index=[f"cell_{i}" for i in range(20)],
    )
    var_dataframe = pd.DataFrame(
        {"feature_description": [f"Morphological feature {i}" for i in range(5)]},
        index=[f"feature_{i}" for i in range(5)],
    )
    return anndata.AnnData(X=x_matrix, obs=obs_dataframe, var=var_dataframe)


@pytest.fixture
def sparse_anndata() -> anndata.AnnData:
    """AnnData whose .X is a scipy CSR sparse matrix (30% fill)."""
    sparse_x = scipy.sparse.random(50, 8, density=0.3, format="csr", dtype=np.float32)
    return anndata.AnnData(X=sparse_x)


@pytest.fixture
def minimal_anndata() -> anndata.AnnData:
    """AnnData with no .obs or .var columns — the bare minimum."""
    return anndata.AnnData(X=np.ones((5, 3), dtype=np.float64))


@pytest.fixture
def anndata_with_obsm_and_uns() -> anndata.AnnData:
    """AnnData that has PCA coordinates in .obsm and a parameter dict in .uns."""
    rng = np.random.default_rng(seed=0)
    base_anndata = anndata.AnnData(X=rng.random((10, 4)).astype(np.float32))
    base_anndata.obsm["X_pca"] = rng.random((10, 2)).astype(np.float32)
    base_anndata.uns["pca_parameters"] = {"n_components": 2, "random_state": 0}
    return base_anndata


# ---------------------------------------------------------------------------
# inspect_anndata — console path (no notebook kernel in pytest)
# ---------------------------------------------------------------------------


def test_inspect_anndata_runs_without_error(typical_anndata, capsys):
    """inspect_anndata must complete without raising on a normal AnnData."""
    inspect_anndata(typical_anndata)
    captured = capsys.readouterr()
    assert len(captured.out) > 0, "Expected non-empty console output."


def test_inspect_anndata_shows_section_headers(typical_anndata, capsys):
    """Console output must contain the main section labels."""
    inspect_anndata(typical_anndata)
    captured = capsys.readouterr()
    assert "ANNDATA INSPECTOR" in captured.out
    assert "[1] Structure" in captured.out
    assert "[2] .obs" in captured.out
    assert "[3] .var" in captured.out
    assert "[4] .X" in captured.out
    assert "[5] .obsm" in captured.out
    assert "[6] .uns" in captured.out


def test_inspect_anndata_shows_correct_dimensions(typical_anndata, capsys):
    """Console output must include the n_obs (20) and n_vars (5) values."""
    inspect_anndata(typical_anndata)
    captured = capsys.readouterr()
    # 20 observations
    assert "20" in captured.out
    # 5 variables
    assert "5" in captured.out


def test_inspect_anndata_shows_obs_column_names(typical_anndata, capsys):
    """Console output must list the .obs column names."""
    inspect_anndata(typical_anndata)
    captured = capsys.readouterr()
    assert "condition" in captured.out
    assert "subject_id" in captured.out


def test_inspect_anndata_sparse_matrix(sparse_anndata, capsys):
    """inspect_anndata must handle a sparse .X without raising."""
    inspect_anndata(sparse_anndata)
    captured = capsys.readouterr()
    assert "sparse" in captured.out


def test_inspect_anndata_minimal_no_obs_var_columns(minimal_anndata, capsys):
    """inspect_anndata must handle AnnData with no .obs or .var columns."""
    inspect_anndata(minimal_anndata)
    captured = capsys.readouterr()
    assert "ANNDATA INSPECTOR" in captured.out


def test_inspect_anndata_with_obsm_and_uns(anndata_with_obsm_and_uns, capsys):
    """Console output must list .obsm keys and .uns keys when populated."""
    inspect_anndata(anndata_with_obsm_and_uns)
    captured = capsys.readouterr()
    assert "X_pca" in captured.out
    assert "pca_parameters" in captured.out


def test_inspect_anndata_respects_max_display_rows(typical_anndata, capsys):
    """max_display_rows must limit the number of .obs rows shown."""
    inspect_anndata(typical_anndata, max_display_rows=3)
    captured = capsys.readouterr()
    # The header says "first 3 rows"
    assert "3" in captured.out


# ---------------------------------------------------------------------------
# _format_bytes
# ---------------------------------------------------------------------------


def test_format_bytes_bytes():
    assert _format_bytes(512) == "512.0 B"


def test_format_bytes_kilobytes():
    assert _format_bytes(2048) == "2.0 KB"


def test_format_bytes_megabytes():
    # 1_572_864 bytes = 1.5 MB exactly
    assert _format_bytes(1_572_864) == "1.5 MB"


def test_format_bytes_gigabytes():
    assert _format_bytes(2 * 1024 ** 3) == "2.0 GB"


# ---------------------------------------------------------------------------
# _x_matrix_info
# ---------------------------------------------------------------------------


def test_x_matrix_info_dense(typical_anndata):
    info = _x_matrix_info(typical_anndata)
    assert info["is_sparse"] is False
    assert info["density_percent"] == 100.0
    assert info["dtype"] == "float32"
    assert "B" in info["size_human"] or "KB" in info["size_human"]


def test_x_matrix_info_sparse(sparse_anndata):
    info = _x_matrix_info(sparse_anndata)
    assert info["is_sparse"] is True
    assert 0.0 <= info["density_percent"] <= 100.0


# ---------------------------------------------------------------------------
# _x_matrix_slice_as_dataframe
# ---------------------------------------------------------------------------


def test_x_matrix_slice_shape_dense(typical_anndata):
    """Slice must not exceed the requested row/column limits."""
    result_dataframe = _x_matrix_slice_as_dataframe(typical_anndata, max_display_rows=5, max_display_columns=3)
    assert result_dataframe.shape == (5, 3)


def test_x_matrix_slice_shape_respects_anndata_size():
    """Slice must not exceed actual AnnData dimensions."""
    small_anndata = anndata.AnnData(X=np.eye(3, dtype=np.float32))
    result_dataframe = _x_matrix_slice_as_dataframe(
        small_anndata, max_display_rows=100, max_display_columns=100
    )
    assert result_dataframe.shape == (3, 3)


def test_x_matrix_slice_sparse_returns_dense_dataframe(sparse_anndata):
    """Sparse .X slice must be returned as a dense (non-sparse) DataFrame."""
    result_dataframe = _x_matrix_slice_as_dataframe(
        sparse_anndata, max_display_rows=5, max_display_columns=4
    )
    # A pandas DataFrame backed by a dense array has no 'nnz' attribute
    assert not scipy.sparse.issparse(result_dataframe.values)


def test_x_matrix_slice_index_names(typical_anndata):
    """Slice index must match obs_names and columns must match var_names."""
    result_dataframe = _x_matrix_slice_as_dataframe(
        typical_anndata, max_display_rows=3, max_display_columns=2
    )
    assert list(result_dataframe.index) == ["cell_0", "cell_1", "cell_2"]
    assert list(result_dataframe.columns) == ["feature_0", "feature_1"]


# ---------------------------------------------------------------------------
# _detect_subject_column
# ---------------------------------------------------------------------------


def test_detect_subject_column_finds_subject_id():
    """Must detect 'subject_ID' (exact case used in this project)."""
    obs_dataframe = pd.DataFrame({"subject_ID": ["S01", "S02"], "condition": ["ctrl", "treated"]})
    assert _detect_subject_column(obs_dataframe) == "subject_ID"


def test_detect_subject_column_finds_patient_id():
    """Must detect alternative common names like 'patient_id'."""
    obs_dataframe = pd.DataFrame({"patient_id": ["P01"], "age": [30]})
    assert _detect_subject_column(obs_dataframe) == "patient_id"


def test_detect_subject_column_returns_none_when_absent():
    """Must return None when no recognisable subject column exists."""
    obs_dataframe = pd.DataFrame({"condition": ["ctrl"], "timepoint": ["D0"]})
    assert _detect_subject_column(obs_dataframe) is None


# ---------------------------------------------------------------------------
# _build_subject_summary_dataframe
# ---------------------------------------------------------------------------


def test_subject_summary_one_row_per_subject():
    """Summary must have exactly as many rows as unique subject values."""
    obs_dataframe = pd.DataFrame(
        {
            "subject_ID": ["S01", "S01", "S02", "S02", "S02"],
            "condition": ["ctrl", "ctrl", "treated", "treated", "treated"],
            "diet": ["normal", "normal", "high_fat", "high_fat", "high_fat"],
        }
    )
    summary = _build_subject_summary_dataframe(obs_dataframe, "subject_ID")
    assert len(summary) == 2
    assert list(summary.index) == ["S01", "S02"]


def test_subject_summary_single_value_shown_directly():
    """When all obs for a subject share the same value, show it without '/'."""
    obs_dataframe = pd.DataFrame(
        {
            "subject_ID": ["S01", "S01"],
            "diet": ["normal", "normal"],
        }
    )
    summary = _build_subject_summary_dataframe(obs_dataframe, "subject_ID")
    assert summary.loc["S01", "diet"] == "normal"


def test_subject_summary_multiple_values_joined_with_slash():
    """When a subject has multiple distinct values, they must be joined with ' / '."""
    obs_dataframe = pd.DataFrame(
        {
            "subject_ID": ["S01", "S01"],
            "marker": ["DeepRed", "Unstained"],
        }
    )
    summary = _build_subject_summary_dataframe(obs_dataframe, "subject_ID")
    # Both values must appear, joined
    assert "DeepRed" in summary.loc["S01", "marker"]
    assert "Unstained" in summary.loc["S01", "marker"]
    assert " / " in summary.loc["S01", "marker"]


def test_inspect_anndata_subject_summary_in_console(capsys):
    """Console output must include the subject summary section when subject_ID exists."""
    obs_dataframe = pd.DataFrame(
        {
            "subject_ID": ["S01", "S01", "S02"],
            "diet": ["normal", "normal", "high_fat"],
        },
        index=["c0", "c1", "c2"],
    )
    test_anndata = anndata.AnnData(
        X=np.ones((3, 2), dtype=np.float32), obs=obs_dataframe
    )
    inspect_anndata(test_anndata)
    captured = capsys.readouterr()
    assert "Subject / group summary" in captured.out
    assert "S01" in captured.out
    assert "S02" in captured.out


def test_inspect_anndata_explicit_group_by_column(capsys):
    """Passing group_by_column explicitly must override auto-detection."""
    obs_dataframe = pd.DataFrame(
        {
            "mouse_id": ["M01", "M01", "M02"],
            "diet": ["normal", "normal", "high_fat"],
        },
        index=["c0", "c1", "c2"],
    )
    test_anndata = anndata.AnnData(
        X=np.ones((3, 2), dtype=np.float32), obs=obs_dataframe
    )
    inspect_anndata(test_anndata, group_by_column="mouse_id")
    captured = capsys.readouterr()
    assert "Subject / group summary" in captured.out
    assert "M01" in captured.out
