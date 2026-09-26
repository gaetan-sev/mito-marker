"""
test_integration_pipeline.py

End-to-end integration test for the full SFC analysis pipeline.

Chains all six analysis steps on a single synthetic AnnData and verifies that each
step writes its expected keys and that the raw .X matrix is never modified.

Pipeline order tested:
    1. select_sfc_subset()       — obs filtering
    2. transform_and_normalize() — arcsinh transform + zscore column normalization
    3. select_channels()         — HighVariance feature selection, top 5
    4. assign_color_palette()    — hex colors for condition values
    5. compute_pca()             — PCA on all events
    6. compute_umap()            — UMAP on a stratified sample
"""

import matplotlib
matplotlib.use("Agg")

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import anndata

from mito_marker.analysis import (
    assign_color_palette,
    compute_pca,
    compute_umap,
    select_channels,
    select_sfc_subset,
    transform_and_normalize,
)
from mito_marker.analysis.colors import COLOR_PALETTE_KEY
from mito_marker.analysis.selection import _ANALYSIS_CONFIG_KEY


# ---------------------------------------------------------------------------
# Input mock helper
# ---------------------------------------------------------------------------


def _make_input_sequence(responses: list[str]):
    """
    Return a callable that yields responses in order.

    Raises AssertionError if more input() calls are made than expected, which
    surfaces any unexpected interactive prompts immediately.
    """
    iterator = iter(responses)

    def _mock_input(prompt: str = "") -> str:
        try:
            return next(iterator)
        except StopIteration:
            raise AssertionError(
                f"More input() calls than expected. Last prompt was: {prompt!r}"
            )

    return _mock_input


# ---------------------------------------------------------------------------
# Shared pipeline fixture (class-scoped — runs once for all tests in the class)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="class")
def pipeline_result():
    """
    Run the full six-step analysis pipeline once on a synthetic AnnData.

    Returns a dict with:
        initial_x:     the raw .X matrix before any processing (preserved for comparison)
        final_anndata: the AnnData object after all six pipeline steps

    Input responses:
        select_sfc_subset:       4 × "A" — keep all events for each prompted column,
                                 then "" to skip sub-sampling.
            Prompted columns (>1 unique value): dilution, diet, FlowAI_Pass, age.
            Auto-skipped (single value): specie ("MNMS"), marker ("MtDeepRed").
            Absent from obs: unique_subject_ID, fcs_acquisition_date.
        transform_and_normalize: "2b" (arcsinh + zscore_col — combined prompt)
        select_channels:         "4" (HighVariance) → "5" (top 5 channels)
    """
    rng = np.random.default_rng(seed=77)
    n_obs = 150
    n_vars = 20
    x_matrix = np.abs(rng.random((n_obs, n_vars)) * 10_000).astype(np.float32)

    obs = pd.DataFrame(
        {
            "subject_ID": np.tile(["S001", "S002", "S003"], n_obs // 3 + 1)[:n_obs],
            "specie": ["MNMS"] * n_obs,            # single value → auto-skipped
            "marker": ["MtDeepRed"] * n_obs,        # single value → auto-skipped
            "dilution": np.tile(["Diluted", "Not_Diluted"], n_obs // 2 + 1)[:n_obs],
            "diet": np.tile(["AL", "IF"], n_obs // 2 + 1)[:n_obs],
            "FlowAI_Pass": np.tile([True, False], n_obs // 2 + 1)[:n_obs].astype(bool),
            "age": np.tile([3.0, 6.0], n_obs // 2 + 1)[:n_obs].astype(np.float32),
        }
    )
    var = pd.DataFrame(index=[f"CH{i:02d}" for i in range(n_vars)])
    base_anndata = anndata.AnnData(X=x_matrix, obs=obs, var=var)

    # Preserve the raw matrix to verify it is unchanged after all pipeline steps.
    initial_x = x_matrix.copy()

    input_responses = [
        # select_sfc_subset — 4 prompted columns, each answered with "A" (keep all).
        # specie auto-skipped (single value "MNMS"), unique_subject_ID absent from obs,
        # marker auto-skipped (single value). Prompted: dilution, diet, FlowAI_Pass, age.
        "A", "A", "A", "A",
        # sub-sampling prompt — press Enter to skip (default = 0, keep all events)
        "",
        # transform_and_normalize — arcsinh (2) + zscore column-wise (b) → "2b"
        "2b",
        # select_channels — HighVariance method (4) then top_n=5 (5)
        "4", "5",
    ]

    with patch("builtins.input", side_effect=_make_input_sequence(input_responses)):
        pipeline_anndata = select_sfc_subset(base_anndata)
        transform_and_normalize(pipeline_anndata)
        select_channels(pipeline_anndata)
        assign_color_palette(pipeline_anndata)
        compute_pca(pipeline_anndata, n_components=10)
        compute_umap(
            pipeline_anndata,
            n_neighbors=5,
            min_dist=0.1,
            max_total_events=30,
        )

    return {"initial_x": initial_x, "final_anndata": pipeline_anndata}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestEndToEndPipeline:
    """Verifies that each analysis step left its expected trace in the AnnData."""

    def test_analysis_config_created_after_selection(self, pipeline_result):
        """select_sfc_subset must initialise .uns['analysis_config']."""
        adata = pipeline_result["final_anndata"]
        assert _ANALYSIS_CONFIG_KEY in adata.uns

    def test_active_layer_set_after_normalization(self, pipeline_result):
        """transform_and_normalize must update active_layer to the new layer key."""
        adata = pipeline_result["final_anndata"]
        active_layer = adata.uns[_ANALYSIS_CONFIG_KEY].get("active_layer")
        assert active_layer == "arcsinh__zscore_col"

    def test_normalization_layer_exists(self, pipeline_result):
        """transform_and_normalize must add the named layer to .layers."""
        adata = pipeline_result["final_anndata"]
        assert "arcsinh__zscore_col" in adata.layers

    def test_feature_selection_var_column_exists(self, pipeline_result):
        """select_channels must write is_selected_HighVariance to .var."""
        adata = pipeline_result["final_anndata"]
        assert "is_selected_HighVariance" in adata.var.columns

    def test_feature_selection_exactly_five_selected(self, pipeline_result):
        """select_channels with top_n=5 must mark exactly 5 channels as selected."""
        adata = pipeline_result["final_anndata"]
        n_selected = adata.var["is_selected_HighVariance"].sum()
        assert n_selected == 5

    def test_color_palette_created(self, pipeline_result):
        """assign_color_palette must create a non-empty .uns['color_palette'] dict."""
        adata = pipeline_result["final_anndata"]
        assert COLOR_PALETTE_KEY in adata.uns
        assert isinstance(adata.uns[COLOR_PALETTE_KEY], dict)
        assert len(adata.uns[COLOR_PALETTE_KEY]) > 0

    def test_pca_coordinates_exist(self, pipeline_result):
        """compute_pca must populate .obsm['X_pca']."""
        adata = pipeline_result["final_anndata"]
        assert "X_pca" in adata.obsm

    def test_umap_key_exists(self, pipeline_result):
        """compute_umap must add at least one 'X_umap_*' key to .obsm."""
        adata = pipeline_result["final_anndata"]
        umap_keys = [key for key in adata.obsm.keys() if key.startswith("X_umap")]
        assert len(umap_keys) > 0

    def test_raw_x_unchanged(self, pipeline_result):
        """No analysis step must modify the raw .X matrix."""
        adata = pipeline_result["final_anndata"]
        np.testing.assert_array_equal(
            adata.X,
            pipeline_result["initial_x"],
            err_msg=".X was modified during the analysis pipeline.",
        )
