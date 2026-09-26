"""
test_analysis_mhi.py

Unit tests for mito_marker.analysis.mhi.

Covers:
  Private helpers:
    _resolve_group_by(): explicit column, None auto-detect, missing column error.
    _get_mhi_feature_matrix(): 'auto' PCA, None raw, int PCA.
    _iter_groups(): yields correct row indices, submatrices, skips small groups.
    _standardize_submatrix(): 'all' vs 'group' scaling.

  compute_mhi_d():
    - Returns DataFrame with correct index and columns.
    - n_mitos column matches actual group sizes.
    - q25 <= median <= q75 for all groups.
    - Directional sanity: wide-scatter group > tight-cluster group.
    - standardize_on='group' differs from 'all'.
    - n_pca_components=None (raw space) works.
    - n_pca_components=int (explicit PCA) works.
    - Unknown group_by raises ValueError.
    - Result stored in .uns['mhi_results'].

  compute_mhi_s():
    - method='dip', 'gmm', 'both' return correct column subsets.
    - NaN columns for inactive method.
    - MHI_S_dip_pval in [0, 1] when diptest available.
    - When _DIPTEST_AVAILABLE=False (monkeypatched), dip columns are NaN + warning.
    - k_opt in [1, max_gmm_k].
    - Result stored in .uns['mhi_results'].

  compute_mhi_e():
    - method='knn', 'binning', 'both' return correct column subsets.
    - KNN entropy non-negative and finite for normal distributions.
    - Binning entropy in [0, 1].
    - Entropy ordering: uniform > tight cluster.
    - Result stored in .uns['mhi_results'].

  compute_all_mhi():
    - Returns DataFrame with all columns when metrics=['D','S','E'].
    - Stores merged result in .uns['mhi_results'][group_by]['all'].
    - metrics=['D'] returns only MHI-D columns.
    - Unknown metric code raises ValueError.
    - n_mitos column present and consistent with individual metrics.

  create_chimera():
    - Returns AnnData with correct n_mitos.
    - Subject column is set to chimera label for all rows.
    - ratio=0.5 → approximately equal split.
    - ratio=1.0 → all rows from individual 1.
    - ratio=0.0 → all rows from individual 2.
    - Original AnnData not mutated.
    - Two calls with same seed yield identical rows.
    - Unknown subject_id_1 raises ValueError.
    - Unknown subject_id_2 raises ValueError.
    - ratio outside [0, 1] raises ValueError.
    - n_mitos exceeding available raises ValueError.

  validate_mhi_with_chimeras():
    - Returns DataFrame with expected columns.
    - DataFrame has correct number of rows.
    - Plot runs without exception (Agg backend).

  analyze_mhi_sensitivity():
    - Returns DataFrame with expected columns.
    - Excludes individuals with too few mitos.
    - Returns empty DataFrame if no eligible groups.
    - Plot runs without exception.

  plot_mhi_distances() / plot_mhi_pca():
    - Smoke tests: run without exception on Agg backend.
"""

import warnings

import anndata
import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")  # non-interactive backend for tests

from mito_marker.analysis.mhi import (
    _iter_groups,
    _resolve_group_by,
    _standardize_submatrix,
    analyze_mhi_sensitivity,
    compute_all_mhi,
    compute_mhi_d,
    compute_mhi_e,
    compute_mhi_s,
    create_chimera,
    plot_mhi_barplot,
    plot_mhi_distances,
    plot_mhi_pca,
    plot_mhi_scatter_de,
    plot_mhi_species_heatmap,
    plot_mhi_stripplot,
    run_mhi_age_tests,
    run_mhi_statistics,
    save_mhi_to_obs,
    validate_all_chimera_pairs,
    validate_mhi_with_chimeras,
)
import mito_marker.analysis.mhi as mhi_module
from mito_marker.analysis.selection import _ANALYSIS_CONFIG_KEY

# ─────────────────────────────────────────────────────────────────────────────
# Shared fixture factories
# ─────────────────────────────────────────────────────────────────────────────

N_FEATURES = 10
N_SUBJECTS = 4
N_MITOS_PER_SUBJECT = 60  # enough for all tests (> 150 for sensitivity tests → use 200)
N_MITOS_LARGE = 200  # for sensitivity tests


def _make_mhi_anndata(
    n_subjects: int = N_SUBJECTS,
    n_mitos_per_subject: int = N_MITOS_PER_SUBJECT,
    n_features: int = N_FEATURES,
    seed: int = 42,
    include_unique_subject_id: bool = True,
) -> anndata.AnnData:
    """
    Build a minimal AnnData with synthetic mitochondrial data.

    Subjects are designed with different heterogeneity levels:
      Subject 0 ('sub_0'): tight cluster — low MHI-D.
      Subject 1 ('sub_1'): wide spread   — high MHI-D.
      Subject 2 ('sub_2'): bimodal        — high MHI-S.
      Subject 3 ('sub_3'): uniform        — high MHI-E.
      (Remaining subjects: standard normal distribution.)

    Arguments:
        n_subjects: Number of subjects.
        n_mitos_per_subject: Rows per subject.
        n_features: Number of features (analogous to morphological channels).
        seed: Random seed.
        include_unique_subject_id: Whether to add 'unique_subject_ID' column.

    Returns:
        AnnData of shape (n_subjects * n_mitos_per_subject, n_features).
    """
    rng = np.random.default_rng(seed)
    all_x_blocks = []
    subject_ids = []

    for subject_index in range(n_subjects):
        subject_label = f"sub_{subject_index}"

        if subject_index == 0:
            # Tight cluster: very small variance around zero.
            block = rng.normal(loc=0.0, scale=0.1, size=(n_mitos_per_subject, n_features))
        elif subject_index == 1:
            # Wide spread: large variance.
            block = rng.normal(loc=0.0, scale=3.0, size=(n_mitos_per_subject, n_features))
        elif subject_index == 2:
            # Bimodal: two clusters along feature 0.
            n_half = n_mitos_per_subject // 2
            cluster_a = rng.normal(loc=-3.0, scale=0.5, size=(n_half, n_features))
            cluster_b = rng.normal(loc=3.0, scale=0.5, size=(n_mitos_per_subject - n_half, n_features))
            block = np.vstack([cluster_a, cluster_b])
        elif subject_index == 3:
            # Uniform distribution over the feature space.
            block = rng.uniform(low=-4.0, high=4.0, size=(n_mitos_per_subject, n_features))
        else:
            # Standard normal.
            block = rng.normal(loc=0.0, scale=1.0, size=(n_mitos_per_subject, n_features))

        all_x_blocks.append(block.astype(np.float32))
        subject_ids.extend([subject_label] * n_mitos_per_subject)

    x_matrix = np.vstack(all_x_blocks)
    n_total = len(subject_ids)

    obs_dataframe = pd.DataFrame(
        {"subject_ID": subject_ids},
        index=[f"mito_{i}" for i in range(n_total)],
    )
    if include_unique_subject_id:
        # Use the same IDs as subject_ID so test assertions are consistent.
        obs_dataframe["unique_subject_ID"] = subject_ids
    obs_dataframe["condition"] = [
        "young" if int(sid.split("_")[1]) < n_subjects // 2 else "old"
        for sid in subject_ids
    ]

    var_dataframe = pd.DataFrame(
        {"feature_description": [f"Feature {i}" for i in range(n_features)],
         "is_non_analytical": [False] * n_features},
        index=[f"feat_{i}" for i in range(n_features)],
    )

    adata = anndata.AnnData(X=x_matrix, obs=obs_dataframe, var=var_dataframe)
    adata.uns[_ANALYSIS_CONFIG_KEY] = {
        "active_layer": None,
        "active_selection": None,
    }
    return adata


def _make_mhi_anndata_no_unique_subject_id(
    n_mitos_per_subject: int = N_MITOS_PER_SUBJECT,
) -> anndata.AnnData:
    """AnnData without 'unique_subject_ID' — only 'subject_ID' (SFC style)."""
    return _make_mhi_anndata(
        n_mitos_per_subject=n_mitos_per_subject,
        include_unique_subject_id=False,
    )


def _make_mhi_anndata_large(seed: int = 42) -> anndata.AnnData:
    """AnnData with enough mitos per subject for sensitivity analysis (> 150)."""
    return _make_mhi_anndata(
        n_subjects=3, n_mitos_per_subject=N_MITOS_LARGE, seed=seed
    )


# ─────────────────────────────────────────────────────────────────────────────
# Tests: private helpers
# ─────────────────────────────────────────────────────────────────────────────


class TestResolveGroupBy:
    def test_explicit_column_returned(self) -> None:
        adata = _make_mhi_anndata()
        assert _resolve_group_by(adata, "condition") == "condition"

    def test_none_returns_unique_subject_id_when_present(self) -> None:
        adata = _make_mhi_anndata(include_unique_subject_id=True)
        assert _resolve_group_by(adata, None) == "unique_subject_ID"

    def test_none_falls_back_to_subject_id(self) -> None:
        adata = _make_mhi_anndata_no_unique_subject_id()
        assert _resolve_group_by(adata, None) == "subject_ID"

    def test_missing_explicit_column_raises_value_error(self) -> None:
        adata = _make_mhi_anndata()
        with pytest.raises(ValueError, match="not found in .obs"):
            _resolve_group_by(adata, "non_existent_column")

    def test_none_with_no_subject_column_raises_value_error(self) -> None:
        adata = _make_mhi_anndata()
        # Remove both subject columns.
        adata.obs.drop(columns=["subject_ID", "unique_subject_ID"], inplace=True)
        with pytest.raises(ValueError, match="No subject column found"):
            _resolve_group_by(adata, None)


class TestIterGroups:
    def test_yields_correct_number_of_groups(self) -> None:
        adata = _make_mhi_anndata(n_subjects=3)
        feature_matrix = adata.X.copy()
        obs_series = adata.obs["unique_subject_ID"]
        groups = list(_iter_groups(feature_matrix, obs_series))
        assert len(groups) == 3

    def test_submatrix_row_count_matches_group_size(self) -> None:
        adata = _make_mhi_anndata(n_subjects=2, n_mitos_per_subject=30)
        feature_matrix = adata.X.copy()
        obs_series = adata.obs["unique_subject_ID"]
        for group_value, row_indices, submatrix in _iter_groups(feature_matrix, obs_series):
            assert len(row_indices) == submatrix.shape[0]
            assert submatrix.shape[0] == 30

    def test_row_indices_select_correct_rows(self) -> None:
        adata = _make_mhi_anndata(n_subjects=2, n_mitos_per_subject=20)
        feature_matrix = adata.X.copy()
        obs_series = adata.obs["unique_subject_ID"]
        for _gv, row_indices, submatrix in _iter_groups(feature_matrix, obs_series):
            np.testing.assert_array_equal(submatrix, feature_matrix[row_indices])

    def test_groups_sorted_alphabetically(self) -> None:
        adata = _make_mhi_anndata(n_subjects=3)
        feature_matrix = adata.X.copy()
        obs_series = adata.obs["unique_subject_ID"]
        group_values = [gv for gv, _, _ in _iter_groups(feature_matrix, obs_series)]
        assert group_values == sorted(group_values)

    def test_small_group_skipped_with_warning(self) -> None:
        adata = _make_mhi_anndata(n_subjects=2, n_mitos_per_subject=30)
        # Add a tiny group (2 mitos) — below _MIN_MITOS_PER_GROUP=5.
        tiny_obs = pd.DataFrame(
            {"unique_subject_ID": ["tiny"] * 2, "subject_ID": ["tiny"] * 2,
             "condition": ["young"] * 2},
            index=[f"tiny_{i}" for i in range(2)],
        )
        rng = np.random.default_rng(0)
        tiny_x = rng.standard_normal((2, adata.n_vars)).astype(np.float32)
        extra = anndata.AnnData(
            X=tiny_x,
            obs=tiny_obs,
            var=adata.var.copy(),
        )
        combined = anndata.concat([adata, extra])
        feature_matrix = combined.X.copy()
        obs_series = combined.obs["unique_subject_ID"]
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            groups = list(_iter_groups(feature_matrix, obs_series))
        group_values = [gv for gv, _, _ in groups]
        assert "tiny" not in group_values
        assert any("tiny" in str(w.message) for w in caught)


class TestStandardizeSubmatrix:
    def test_all_mode_mean_near_zero_for_full_dataset(self) -> None:
        rng = np.random.default_rng(42)
        matrix = rng.standard_normal((100, 5)).astype(np.float32)
        indices = np.arange(100)
        standardized = _standardize_submatrix(matrix, indices, "all")
        np.testing.assert_allclose(standardized.mean(axis=0), 0.0, atol=1e-4)

    def test_group_mode_mean_near_zero_for_submatrix(self) -> None:
        rng = np.random.default_rng(42)
        matrix = rng.standard_normal((100, 5)).astype(np.float32)
        # Only standardize rows 0–49.
        indices = np.arange(50)
        standardized = _standardize_submatrix(matrix, indices, "group")
        np.testing.assert_allclose(standardized.mean(axis=0), 0.0, atol=1e-4)

    def test_all_and_group_differ_for_biased_submatrix(self) -> None:
        rng = np.random.default_rng(0)
        matrix = rng.standard_normal((200, 5)).astype(np.float32)
        # Shift the submatrix so it differs strongly from the global mean.
        matrix[:50] += 5.0
        indices = np.arange(50)
        std_all = _standardize_submatrix(matrix, indices, "all")
        std_group = _standardize_submatrix(matrix, indices, "group")
        # The two standardizations should give different results.
        assert not np.allclose(std_all, std_group, atol=0.1)

    def test_returns_float32(self) -> None:
        rng = np.random.default_rng(0)
        matrix = rng.standard_normal((50, 3)).astype(np.float64)  # float64 input
        indices = np.arange(50)
        result = _standardize_submatrix(matrix, indices, "all")
        assert result.dtype == np.float32


# ─────────────────────────────────────────────────────────────────────────────
# Tests: compute_mhi_d
# ─────────────────────────────────────────────────────────────────────────────


class TestComputeMhiD:
    def test_returns_dataframe(self) -> None:
        adata = _make_mhi_anndata()
        result = compute_mhi_d(adata, group_by="subject_ID")
        assert isinstance(result, pd.DataFrame)

    def test_index_name_matches_group_by(self) -> None:
        adata = _make_mhi_anndata()
        result = compute_mhi_d(adata, group_by="subject_ID")
        assert result.index.name == "subject_ID"

    def test_expected_columns_present(self) -> None:
        adata = _make_mhi_anndata()
        result = compute_mhi_d(adata, group_by="subject_ID")
        for col in ("n_mitos", "MHI_D_median", "MHI_D_q25", "MHI_D_q75"):
            assert col in result.columns

    def test_n_rows_equals_n_subjects(self) -> None:
        adata = _make_mhi_anndata(n_subjects=3)
        result = compute_mhi_d(adata, group_by="subject_ID")
        assert len(result) == 3

    def test_n_mitos_column_correct(self) -> None:
        n_mitos = 40
        adata = _make_mhi_anndata(n_subjects=2, n_mitos_per_subject=n_mitos)
        result = compute_mhi_d(adata, group_by="subject_ID")
        assert (result["n_mitos"] == n_mitos).all()

    def test_quantile_ordering(self) -> None:
        adata = _make_mhi_anndata()
        result = compute_mhi_d(adata, group_by="subject_ID")
        assert (result["MHI_D_q25"] <= result["MHI_D_median"]).all()
        assert (result["MHI_D_median"] <= result["MHI_D_q75"]).all()

    def test_values_non_negative(self) -> None:
        adata = _make_mhi_anndata()
        result = compute_mhi_d(adata, group_by="subject_ID")
        assert (result["MHI_D_median"] >= 0.0).all()

    def test_wide_subject_has_higher_mhi_d_than_tight(self) -> None:
        # Subject 0 = tight (scale=0.1), Subject 1 = wide (scale=3.0).
        adata = _make_mhi_anndata(n_subjects=2)
        result = compute_mhi_d(adata, group_by="subject_ID")
        mhi_tight = result.loc["sub_0", "MHI_D_median"]
        mhi_wide = result.loc["sub_1", "MHI_D_median"]
        assert mhi_wide > mhi_tight

    def test_standardize_on_group_differs_from_all(self) -> None:
        adata = _make_mhi_anndata()
        result_all = compute_mhi_d(adata, group_by="subject_ID", standardize_on="all")
        result_group = compute_mhi_d(adata, group_by="subject_ID", standardize_on="group")
        # The median values should not be identical.
        assert not result_all["MHI_D_median"].equals(result_group["MHI_D_median"])

    def test_raw_feature_space_n_pca_none(self) -> None:
        adata = _make_mhi_anndata()
        result = compute_mhi_d(adata, group_by="subject_ID", n_pca_components=None)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == N_SUBJECTS

    def test_explicit_pca_components(self) -> None:
        adata = _make_mhi_anndata()
        result = compute_mhi_d(adata, group_by="subject_ID", n_pca_components=3)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == N_SUBJECTS

    def test_auto_detects_unique_subject_id(self) -> None:
        adata = _make_mhi_anndata(include_unique_subject_id=True)
        result = compute_mhi_d(adata)  # group_by=None → auto-detect
        assert result.index.name == "unique_subject_ID"

    def test_unknown_group_by_raises_value_error(self) -> None:
        adata = _make_mhi_anndata()
        with pytest.raises(ValueError):
            compute_mhi_d(adata, group_by="no_such_column")

    def test_result_stored_in_uns(self) -> None:
        adata = _make_mhi_anndata()
        compute_mhi_d(adata, group_by="subject_ID")
        assert "mhi_results" in adata.uns
        assert "subject_ID" in adata.uns["mhi_results"]
        assert "D" in adata.uns["mhi_results"]["subject_ID"]


# ─────────────────────────────────────────────────────────────────────────────
# Tests: compute_mhi_s
# ─────────────────────────────────────────────────────────────────────────────


class TestComputeMhiS:
    def test_returns_dataframe(self) -> None:
        adata = _make_mhi_anndata()
        result = compute_mhi_s(adata, group_by="subject_ID", method="gmm")
        assert isinstance(result, pd.DataFrame)

    def test_expected_columns_both_methods(self) -> None:
        adata = _make_mhi_anndata()
        result = compute_mhi_s(adata, group_by="subject_ID", method="both")
        for col in ("n_mitos", "MHI_S_dip", "MHI_S_dip_pval",
                    "MHI_S_gmm_delta_bic", "MHI_S_gmm_k_opt"):
            assert col in result.columns

    def test_method_gmm_dip_columns_are_nan(self) -> None:
        adata = _make_mhi_anndata()
        result = compute_mhi_s(adata, group_by="subject_ID", method="gmm")
        assert result["MHI_S_dip"].isna().all()
        assert result["MHI_S_dip_pval"].isna().all()

    def test_method_dip_gmm_columns_are_nan(self) -> None:
        adata = _make_mhi_anndata()
        result = compute_mhi_s(adata, group_by="subject_ID", method="dip")
        assert result["MHI_S_gmm_delta_bic"].isna().all()
        assert result["MHI_S_gmm_k_opt"].isna().all()

    def test_gmm_k_opt_in_valid_range(self) -> None:
        adata = _make_mhi_anndata()
        result = compute_mhi_s(adata, group_by="subject_ID", method="gmm", max_gmm_k=3)
        valid_mask = result["MHI_S_gmm_k_opt"].notna()
        assert ((result.loc[valid_mask, "MHI_S_gmm_k_opt"] >= 1) &
                (result.loc[valid_mask, "MHI_S_gmm_k_opt"] <= 3)).all()

    def test_dip_pval_in_zero_one_when_available(self) -> None:
        if not mhi_module._DIPTEST_AVAILABLE:
            pytest.skip("diptest not installed")
        adata = _make_mhi_anndata()
        result = compute_mhi_s(adata, group_by="subject_ID", method="dip")
        valid_pvals = result["MHI_S_dip_pval"].dropna()
        assert ((valid_pvals >= 0.0) & (valid_pvals <= 1.0)).all()

    def test_bimodal_subject_has_higher_dip_than_unimodal(self) -> None:
        if not mhi_module._DIPTEST_AVAILABLE:
            pytest.skip("diptest not installed")
        adata = _make_mhi_anndata(n_subjects=3)
        result = compute_mhi_s(adata, group_by="subject_ID", method="dip")
        # sub_0 = tight cluster (unimodal), sub_2 = bimodal
        dip_tight = result.loc["sub_0", "MHI_S_dip"]
        dip_bimodal = result.loc["sub_2", "MHI_S_dip"]
        assert dip_bimodal > dip_tight

    def test_diptest_unavailable_produces_nan_and_warning(self, monkeypatch) -> None:
        monkeypatch.setattr(mhi_module, "_DIPTEST_AVAILABLE", False)
        adata = _make_mhi_anndata()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = compute_mhi_s(adata, group_by="subject_ID", method="dip")
        assert result["MHI_S_dip"].isna().all()
        assert any("diptest" in str(w.message).lower() for w in caught)

    def test_invalid_method_raises_value_error(self) -> None:
        adata = _make_mhi_anndata()
        with pytest.raises(ValueError, match="method must be"):
            compute_mhi_s(adata, group_by="subject_ID", method="invalid")

    def test_result_stored_in_uns(self) -> None:
        adata = _make_mhi_anndata()
        compute_mhi_s(adata, group_by="subject_ID", method="gmm")
        assert "S" in adata.uns["mhi_results"]["subject_ID"]

    def test_n_rows_equals_n_subjects(self) -> None:
        adata = _make_mhi_anndata(n_subjects=3)
        result = compute_mhi_s(adata, group_by="subject_ID", method="gmm")
        assert len(result) == 3


# ─────────────────────────────────────────────────────────────────────────────
# Tests: compute_mhi_e
# ─────────────────────────────────────────────────────────────────────────────


class TestComputeMhiE:
    def test_returns_dataframe(self) -> None:
        adata = _make_mhi_anndata()
        result = compute_mhi_e(adata, group_by="subject_ID")
        assert isinstance(result, pd.DataFrame)

    def test_expected_columns_both_methods(self) -> None:
        adata = _make_mhi_anndata()
        result = compute_mhi_e(adata, group_by="subject_ID", method="both")
        for col in ("n_mitos", "MHI_E_knn", "MHI_E_binning"):
            assert col in result.columns

    def test_method_knn_binning_column_nan(self) -> None:
        adata = _make_mhi_anndata()
        result = compute_mhi_e(adata, group_by="subject_ID", method="knn")
        assert result["MHI_E_binning"].isna().all()

    def test_method_binning_knn_column_nan(self) -> None:
        adata = _make_mhi_anndata()
        result = compute_mhi_e(adata, group_by="subject_ID", method="binning")
        assert result["MHI_E_knn"].isna().all()

    def test_knn_entropy_non_negative(self) -> None:
        adata = _make_mhi_anndata()
        result = compute_mhi_e(adata, group_by="subject_ID", method="knn")
        valid = result["MHI_E_knn"].dropna()
        # KNN entropy can be negative in theory but normalised / log(n) should be
        # meaningful.  We test that the values are finite.
        assert valid.apply(np.isfinite).all()

    def test_binning_entropy_in_zero_one(self) -> None:
        adata = _make_mhi_anndata()
        result = compute_mhi_e(adata, group_by="subject_ID", method="binning")
        valid = result["MHI_E_binning"].dropna()
        assert ((valid >= 0.0) & (valid <= 1.0)).all()

    def test_uniform_subject_higher_entropy_than_tight(self) -> None:
        # sub_0 = tight cluster, sub_3 = uniform distribution.
        adata = _make_mhi_anndata(n_subjects=4, n_mitos_per_subject=80)
        result = compute_mhi_e(adata, group_by="subject_ID", method="binning")
        e_tight = result.loc["sub_0", "MHI_E_binning"]
        e_uniform = result.loc["sub_3", "MHI_E_binning"]
        assert e_uniform > e_tight

    def test_result_stored_in_uns(self) -> None:
        adata = _make_mhi_anndata()
        compute_mhi_e(adata, group_by="subject_ID")
        assert "E" in adata.uns["mhi_results"]["subject_ID"]

    def test_invalid_method_raises_value_error(self) -> None:
        adata = _make_mhi_anndata()
        with pytest.raises(ValueError, match="method must be"):
            compute_mhi_e(adata, group_by="subject_ID", method="bad")

    def test_n_rows_equals_n_subjects(self) -> None:
        adata = _make_mhi_anndata(n_subjects=3)
        result = compute_mhi_e(adata, group_by="subject_ID")
        assert len(result) == 3


# ─────────────────────────────────────────────────────────────────────────────
# Tests: compute_all_mhi
# ─────────────────────────────────────────────────────────────────────────────


class TestComputeAllMhi:
    def test_returns_dataframe(self) -> None:
        adata = _make_mhi_anndata()
        result = compute_all_mhi(adata, group_by="subject_ID")
        assert isinstance(result, pd.DataFrame)

    def test_all_metrics_columns_present_by_default(self) -> None:
        adata = _make_mhi_anndata()
        result = compute_all_mhi(adata, group_by="subject_ID")
        expected = ["n_mitos", "MHI_D_median", "MHI_D_q25", "MHI_D_q75",
                    "MHI_S_gmm_delta_bic", "MHI_S_gmm_k_opt",
                    "MHI_E_knn", "MHI_E_binning"]
        for col in expected:
            assert col in result.columns, f"Missing column: {col}"

    def test_metrics_d_only_returns_mhi_d_columns(self) -> None:
        adata = _make_mhi_anndata()
        result = compute_all_mhi(adata, group_by="subject_ID", metrics=["D"])
        assert "MHI_D_median" in result.columns
        assert "MHI_S_dip" not in result.columns
        assert "MHI_E_knn" not in result.columns

    def test_metrics_d_e_returns_correct_columns(self) -> None:
        adata = _make_mhi_anndata()
        result = compute_all_mhi(adata, group_by="subject_ID", metrics=["D", "E"])
        assert "MHI_D_median" in result.columns
        assert "MHI_E_knn" in result.columns
        assert "MHI_S_dip" not in result.columns

    def test_n_rows_equals_n_subjects(self) -> None:
        adata = _make_mhi_anndata(n_subjects=3)
        result = compute_all_mhi(adata, group_by="subject_ID")
        assert len(result) == 3

    def test_stored_in_uns_all_key(self) -> None:
        adata = _make_mhi_anndata()
        compute_all_mhi(adata, group_by="subject_ID")
        assert "all" in adata.uns["mhi_results"]["subject_ID"]

    def test_unknown_metric_code_raises_value_error(self) -> None:
        adata = _make_mhi_anndata()
        with pytest.raises(ValueError, match="Unknown metric codes"):
            compute_all_mhi(adata, group_by="subject_ID", metrics=["X"])

    def test_n_mitos_column_consistent_with_individual_metrics(self) -> None:
        adata = _make_mhi_anndata(n_mitos_per_subject=50)
        result = compute_all_mhi(adata, group_by="subject_ID", metrics=["D"])
        assert (result["n_mitos"] == 50).all()

    def test_kwargs_forwarded_to_compute_mhi_d(self) -> None:
        adata = _make_mhi_anndata()
        # standardize_on is a compute_mhi_d kwarg — should not raise.
        result = compute_all_mhi(
            adata, group_by="subject_ID", metrics=["D"], standardize_on="group"
        )
        assert "MHI_D_median" in result.columns


# ─────────────────────────────────────────────────────────────────────────────
# Tests: create_chimera
# ─────────────────────────────────────────────────────────────────────────────


class TestCreateChimera:
    def test_returns_anndata(self) -> None:
        adata = _make_mhi_anndata()
        chimera = create_chimera(adata, "sub_0", "sub_1")
        assert isinstance(chimera, anndata.AnnData)

    def test_total_mitos_is_min_of_both(self) -> None:
        adata = _make_mhi_anndata(n_subjects=2, n_mitos_per_subject=60)
        chimera = create_chimera(adata, "sub_0", "sub_1", ratio=0.5)
        # n_mitos=None → min(60, 60) = 60
        assert chimera.n_obs == 60

    def test_explicit_n_mitos(self) -> None:
        adata = _make_mhi_anndata(n_subjects=2, n_mitos_per_subject=80)
        chimera = create_chimera(adata, "sub_0", "sub_1", n_mitos=40)
        assert chimera.n_obs == 40

    def test_subject_column_set_to_chimera_label(self) -> None:
        adata = _make_mhi_anndata()
        chimera = create_chimera(adata, "sub_0", "sub_1", ratio=0.5)
        labels = chimera.obs["unique_subject_ID"].unique()
        assert len(labels) == 1
        assert "chimera" in labels[0]

    def test_ratio_one_all_from_subject_1(self) -> None:
        adata = _make_mhi_anndata(n_subjects=2, n_mitos_per_subject=50)
        chimera = create_chimera(adata, "sub_0", "sub_1", ratio=1.0)
        # All mitos should come from sub_0.
        assert chimera.n_obs == 50

    def test_ratio_zero_all_from_subject_2(self) -> None:
        adata = _make_mhi_anndata(n_subjects=2, n_mitos_per_subject=50)
        chimera = create_chimera(adata, "sub_0", "sub_1", ratio=0.0)
        assert chimera.n_obs == 50

    def test_original_anndata_not_mutated(self) -> None:
        adata = _make_mhi_anndata()
        original_n_obs = adata.n_obs
        original_subject_values = adata.obs["unique_subject_ID"].tolist()
        create_chimera(adata, "sub_0", "sub_1")
        assert adata.n_obs == original_n_obs
        assert adata.obs["unique_subject_ID"].tolist() == original_subject_values

    def test_seed_reproducibility(self) -> None:
        adata = _make_mhi_anndata()
        chimera_a = create_chimera(adata, "sub_0", "sub_1", seed=99)
        chimera_b = create_chimera(adata, "sub_0", "sub_1", seed=99)
        np.testing.assert_array_equal(chimera_a.X, chimera_b.X)

    def test_different_seeds_give_different_results(self) -> None:
        adata = _make_mhi_anndata()
        chimera_a = create_chimera(adata, "sub_0", "sub_1", seed=0)
        chimera_b = create_chimera(adata, "sub_0", "sub_1", seed=1)
        assert not np.array_equal(chimera_a.X, chimera_b.X)

    def test_unknown_subject_id_1_raises_value_error(self) -> None:
        adata = _make_mhi_anndata()
        with pytest.raises(ValueError, match="not found"):
            create_chimera(adata, "no_such_id", "sub_1")

    def test_unknown_subject_id_2_raises_value_error(self) -> None:
        adata = _make_mhi_anndata()
        with pytest.raises(ValueError, match="not found"):
            create_chimera(adata, "sub_0", "no_such_id")

    def test_ratio_out_of_range_raises_value_error(self) -> None:
        adata = _make_mhi_anndata()
        with pytest.raises(ValueError, match="ratio must be in"):
            create_chimera(adata, "sub_0", "sub_1", ratio=1.5)

    def test_n_mitos_exceeding_available_raises_value_error(self) -> None:
        adata = _make_mhi_anndata(n_subjects=2, n_mitos_per_subject=30)
        with pytest.raises(ValueError, match="Requested"):
            create_chimera(adata, "sub_0", "sub_1", n_mitos=1000)


# ─────────────────────────────────────────────────────────────────────────────
# Tests: validate_mhi_with_chimeras
# ─────────────────────────────────────────────────────────────────────────────


class TestValidateMhiWithChimeras:
    def test_returns_dataframe(self) -> None:
        adata = _make_mhi_anndata(n_mitos_per_subject=60)
        result = validate_mhi_with_chimeras(
            adata, "sub_0", "sub_1",
            ratios=[0.3, 0.5, 0.7],
            n_repetitions=2,
            metrics=["D"],
        )
        assert isinstance(result, pd.DataFrame)

    def test_expected_columns_present(self) -> None:
        adata = _make_mhi_anndata(n_mitos_per_subject=60)
        result = validate_mhi_with_chimeras(
            adata, "sub_0", "sub_1",
            ratios=[0.5],
            n_repetitions=1,
            metrics=["D"],
        )
        assert "ratio" in result.columns
        assert "repetition" in result.columns
        assert "MHI_D_median" in result.columns

    def test_number_of_rows(self) -> None:
        adata = _make_mhi_anndata(n_mitos_per_subject=60)
        ratios = [0.3, 0.5, 0.7]
        n_reps = 2
        result = validate_mhi_with_chimeras(
            adata, "sub_0", "sub_1",
            ratios=ratios,
            n_repetitions=n_reps,
            metrics=["D"],
        )
        # 2 real individuals (rep=0) + len(ratios) * n_reps chimeras
        expected_rows = 2 + len(ratios) * n_reps
        assert len(result) == expected_rows

    def test_plot_runs_without_exception(self) -> None:
        adata = _make_mhi_anndata(n_mitos_per_subject=60)
        validate_mhi_with_chimeras(
            adata, "sub_0", "sub_1",
            ratios=[0.3, 0.7],
            n_repetitions=1,
            metrics=["D"],
        )


# ─────────────────────────────────────────────────────────────────────────────
# Tests: analyze_mhi_sensitivity
# ─────────────────────────────────────────────────────────────────────────────


class TestAnalyzeMhiSensitivity:
    def test_returns_dataframe(self) -> None:
        adata = _make_mhi_anndata_large()
        result = analyze_mhi_sensitivity(
            adata, group_by="subject_ID",
            n_subsamples=[20, 30], n_repetitions=3,
        )
        assert isinstance(result, pd.DataFrame)

    def test_expected_columns(self) -> None:
        adata = _make_mhi_anndata_large()
        result = analyze_mhi_sensitivity(
            adata, group_by="subject_ID",
            n_subsamples=[20, 30], n_repetitions=2,
        )
        for col in ("group_value", "n_subsample", "repetition", "MHI_D_median"):
            assert col in result.columns

    def test_excludes_groups_with_too_few_mitos(self) -> None:
        # n_mitos_per_subject=40, max(n_subsamples)=50 → all groups excluded.
        adata = _make_mhi_anndata(n_subjects=2, n_mitos_per_subject=40)
        result = analyze_mhi_sensitivity(
            adata, group_by="subject_ID",
            n_subsamples=[50], n_repetitions=2,
        )
        assert len(result) == 0

    def test_returns_empty_df_when_no_eligible_groups(self) -> None:
        adata = _make_mhi_anndata(n_subjects=2, n_mitos_per_subject=10)
        result = analyze_mhi_sensitivity(
            adata, group_by="subject_ID",
            n_subsamples=[20], n_repetitions=2,
        )
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0

    def test_n_repetitions_rows_per_subsample(self) -> None:
        adata = _make_mhi_anndata_large()
        n_reps = 5
        result = analyze_mhi_sensitivity(
            adata, group_by="subject_ID",
            n_subsamples=[20], n_repetitions=n_reps,
        )
        # Each eligible group × each subsample size × n_reps
        n_eligible = 3  # _make_mhi_anndata_large has 3 subjects, all > 150 mitos
        assert len(result) == n_eligible * n_reps

    def test_plot_runs_without_exception(self) -> None:
        adata = _make_mhi_anndata_large()
        analyze_mhi_sensitivity(
            adata, group_by="subject_ID",
            n_subsamples=[20, 30], n_repetitions=2,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Tests: diagnostic plots (smoke tests)
# ─────────────────────────────────────────────────────────────────────────────


class TestPlotMhiDistances:
    def test_runs_without_exception(self) -> None:
        adata = _make_mhi_anndata()
        plot_mhi_distances(adata, group_by="subject_ID")

    def test_with_precomputed_mhi_results(self) -> None:
        adata = _make_mhi_anndata()
        mhi_df = compute_mhi_d(adata, group_by="subject_ID")
        plot_mhi_distances(adata, group_by="subject_ID", mhi_results=mhi_df)

    def test_raw_feature_space(self) -> None:
        adata = _make_mhi_anndata()
        plot_mhi_distances(adata, group_by="subject_ID", n_pca_components=None)


class TestPlotMhiPca:
    def test_runs_without_exception(self) -> None:
        adata = _make_mhi_anndata()
        plot_mhi_pca(adata, group_by="subject_ID")

    def test_color_by_gmm(self) -> None:
        adata = _make_mhi_anndata()
        plot_mhi_pca(adata, group_by="subject_ID", color_by_gmm=True)


# ─────────────────────────────────────────────────────────────────────────────
# Tests: two-level computation (_resolve_nest_aggregate_by, _aggregate_individual_to_group)
# ─────────────────────────────────────────────────────────────────────────────


class TestResolveNestAggregateBy:
    """Tests for _resolve_nest_aggregate_by() — pure validator, no auto-detection."""

    def test_explicit_valid_column_returned(self) -> None:
        from mito_marker.analysis.mhi import _resolve_nest_aggregate_by

        adata = _make_mhi_anndata()
        assert _resolve_nest_aggregate_by(adata, "subject_ID") == "subject_ID"

    def test_explicit_unique_subject_id_returned(self) -> None:
        from mito_marker.analysis.mhi import _resolve_nest_aggregate_by

        adata = _make_mhi_anndata()
        assert _resolve_nest_aggregate_by(adata, "unique_subject_ID") == "unique_subject_ID"

    def test_missing_column_raises_value_error(self) -> None:
        from mito_marker.analysis.mhi import _resolve_nest_aggregate_by

        adata = _make_mhi_anndata()
        with pytest.raises(ValueError, match="not found in .obs"):
            _resolve_nest_aggregate_by(adata, "no_such_col")

    def test_condition_column_accepted(self) -> None:
        from mito_marker.analysis.mhi import _resolve_nest_aggregate_by

        adata = _make_mhi_anndata()
        # Any valid obs column is accepted (not just subject columns).
        assert _resolve_nest_aggregate_by(adata, "condition") == "condition"


class TestAggregateIndividualToGroup:
    """Tests for _aggregate_individual_to_group() — pure aggregation logic."""

    def _make_individual_df(self, n_subjects: int = 4, n_mitos_per_subject: int = 60) -> tuple:
        """Return (individual_df, obs) for a standard 4-subject fixture."""
        adata = _make_mhi_anndata(
            n_subjects=n_subjects, n_mitos_per_subject=n_mitos_per_subject
        )
        individual_df = compute_mhi_d(adata, group_by="subject_ID")
        return individual_df, adata.obs

    def test_returns_correct_n_individuals(self) -> None:
        from mito_marker.analysis.mhi import _aggregate_individual_to_group

        individual_df, obs = self._make_individual_df()
        result = _aggregate_individual_to_group(
            individual_df, obs,
            individual_col="subject_ID",
            group_col="condition",
            agg_cols=[("MHI_D_median", "MHI_D")],
        )
        # 4 subjects split 2+2 into young/old.
        assert (result["n_individuals"] == 2).all()

    def test_returns_correct_n_mitos_total(self) -> None:
        from mito_marker.analysis.mhi import _aggregate_individual_to_group

        n_mitos = 60
        individual_df, obs = self._make_individual_df(n_mitos_per_subject=n_mitos)
        result = _aggregate_individual_to_group(
            individual_df, obs,
            individual_col="subject_ID",
            group_col="condition",
            agg_cols=[("MHI_D_median", "MHI_D")],
        )
        assert (result["n_mitos_total"] == 2 * n_mitos).all()

    def test_output_index_name_is_group_col(self) -> None:
        from mito_marker.analysis.mhi import _aggregate_individual_to_group

        individual_df, obs = self._make_individual_df()
        result = _aggregate_individual_to_group(
            individual_df, obs,
            individual_col="subject_ID",
            group_col="condition",
            agg_cols=[("MHI_D_median", "MHI_D")],
        )
        assert result.index.name == "condition"

    def test_std_is_nan_for_single_individual_group(self) -> None:
        from mito_marker.analysis.mhi import _aggregate_individual_to_group

        # Use only 2 subjects → each condition has 1 individual → std = NaN.
        adata = _make_mhi_anndata(n_subjects=2, n_mitos_per_subject=60)
        individual_df = compute_mhi_d(adata, group_by="subject_ID")
        result = _aggregate_individual_to_group(
            individual_df, adata.obs,
            individual_col="subject_ID",
            group_col="condition",
            agg_cols=[("MHI_D_median", "MHI_D")],
        )
        assert result["MHI_D_std"].isna().all()

    def test_aggregated_median_within_individual_range(self) -> None:
        from mito_marker.analysis.mhi import _aggregate_individual_to_group

        individual_df, obs = self._make_individual_df()
        result = _aggregate_individual_to_group(
            individual_df, obs,
            individual_col="subject_ID",
            group_col="condition",
            agg_cols=[("MHI_D_median", "MHI_D")],
        )
        # Group median must lie within [min, max] of the constituent individuals.
        for group_value in result.index:
            members = obs[obs["condition"] == group_value]["subject_ID"].unique()
            individual_values = individual_df.loc[
                [m for m in members if m in individual_df.index], "MHI_D_median"
            ].values
            group_median = result.loc[group_value, "MHI_D_median"]
            assert individual_values.min() <= group_median <= individual_values.max()

    def test_q25_leq_median_leq_q75(self) -> None:
        from mito_marker.analysis.mhi import _aggregate_individual_to_group

        individual_df, obs = self._make_individual_df()
        result = _aggregate_individual_to_group(
            individual_df, obs,
            individual_col="subject_ID",
            group_col="condition",
            agg_cols=[("MHI_D_median", "MHI_D")],
        )
        assert (result["MHI_D_q25"] <= result["MHI_D_median"]).all()
        assert (result["MHI_D_median"] <= result["MHI_D_q75"]).all()


class TestComputeMhiDTwoLevel:
    """Tests for compute_mhi_d() with nest_aggregate_by != group_by."""

    def test_returns_dataframe(self) -> None:
        adata = _make_mhi_anndata()
        result = compute_mhi_d(adata, group_by="condition", nest_aggregate_by="subject_ID")
        assert isinstance(result, pd.DataFrame)

    def test_index_name_is_group_col(self) -> None:
        adata = _make_mhi_anndata()
        result = compute_mhi_d(adata, group_by="condition", nest_aggregate_by="subject_ID")
        assert result.index.name == "condition"

    def test_expected_columns_present(self) -> None:
        adata = _make_mhi_anndata()
        result = compute_mhi_d(adata, group_by="condition", nest_aggregate_by="subject_ID")
        for col in ("n_individuals", "n_mitos_total", "MHI_D_median", "MHI_D_std",
                    "MHI_D_q25", "MHI_D_q75"):
            assert col in result.columns, f"Missing column: {col}"

    def test_n_rows_equals_n_conditions(self) -> None:
        adata = _make_mhi_anndata()
        result = compute_mhi_d(adata, group_by="condition", nest_aggregate_by="subject_ID")
        n_conditions = adata.obs["condition"].nunique()
        assert len(result) == n_conditions

    def test_both_uns_keys_stored(self) -> None:
        adata = _make_mhi_anndata()
        compute_mhi_d(adata, group_by="condition", nest_aggregate_by="subject_ID")
        assert "subject_ID" in adata.uns["mhi_results"]
        assert "D" in adata.uns["mhi_results"]["subject_ID"]
        assert "condition" in adata.uns["mhi_results"]
        assert "D" in adata.uns["mhi_results"]["condition"]

    def test_individual_uns_has_single_level_columns(self) -> None:
        adata = _make_mhi_anndata()
        compute_mhi_d(adata, group_by="condition", nest_aggregate_by="subject_ID")
        individual_result = adata.uns["mhi_results"]["subject_ID"]["D"]
        assert "n_mitos" in individual_result.columns
        assert "n_individuals" not in individual_result.columns

    def test_n_individuals_correct(self) -> None:
        adata = _make_mhi_anndata()
        result = compute_mhi_d(adata, group_by="condition", nest_aggregate_by="subject_ID")
        # 4 subjects, 2 per condition.
        assert (result["n_individuals"] == 2).all()

    def test_n_mitos_total_correct(self) -> None:
        adata = _make_mhi_anndata()
        result = compute_mhi_d(adata, group_by="condition", nest_aggregate_by="subject_ID")
        assert (result["n_mitos_total"] == 2 * N_MITOS_PER_SUBJECT).all()

    def test_std_is_finite_or_nan(self) -> None:
        adata = _make_mhi_anndata()
        result = compute_mhi_d(adata, group_by="condition", nest_aggregate_by="subject_ID")
        assert not np.isinf(result["MHI_D_std"].dropna().values).any()

    def test_default_nest_aggregate_by_none_gives_single_level_schema(self) -> None:
        # When nest_aggregate_by=None, individual_col defaults to group_col → single-level.
        adata = _make_mhi_anndata()
        result = compute_mhi_d(adata, group_by="subject_ID")
        assert "n_mitos" in result.columns
        assert "n_individuals" not in result.columns

    def test_unknown_nest_aggregate_by_raises_value_error(self) -> None:
        adata = _make_mhi_anndata()
        with pytest.raises(ValueError, match="not found in .obs"):
            compute_mhi_d(adata, group_by="condition", nest_aggregate_by="no_such_col")

    def test_group_by_condition_without_nest_aggregate_by_gives_single_level(self) -> None:
        # group_by='condition', nest_aggregate_by=None → individual_col=group_col → single-level.
        adata = _make_mhi_anndata()
        result = compute_mhi_d(adata, group_by="condition")
        assert "n_mitos" in result.columns
        assert "n_individuals" not in result.columns


class TestComputeMhiSTwoLevel:
    """Tests for compute_mhi_s() with nest_aggregate_by != group_by."""

    def test_returns_dataframe(self) -> None:
        adata = _make_mhi_anndata()
        result = compute_mhi_s(adata, group_by="condition", nest_aggregate_by="subject_ID")
        assert isinstance(result, pd.DataFrame)

    def test_expected_columns_present(self) -> None:
        adata = _make_mhi_anndata()
        result = compute_mhi_s(
            adata, group_by="condition", nest_aggregate_by="subject_ID", method="gmm"
        )
        for col in ("n_individuals", "n_mitos_total",
                    "MHI_S_gmm_delta_bic_median", "MHI_S_gmm_delta_bic_std",
                    "MHI_S_gmm_k_opt_median"):
            assert col in result.columns, f"Missing column: {col}"

    def test_both_uns_keys_stored(self) -> None:
        adata = _make_mhi_anndata()
        compute_mhi_s(
            adata, group_by="condition", nest_aggregate_by="subject_ID", method="gmm"
        )
        assert "subject_ID" in adata.uns["mhi_results"]
        assert "S" in adata.uns["mhi_results"]["subject_ID"]
        assert "condition" in adata.uns["mhi_results"]
        assert "S" in adata.uns["mhi_results"]["condition"]

    def test_n_rows_equals_n_conditions(self) -> None:
        adata = _make_mhi_anndata()
        result = compute_mhi_s(
            adata, group_by="condition", nest_aggregate_by="subject_ID", method="gmm"
        )
        assert len(result) == adata.obs["condition"].nunique()


class TestComputeMhiETwoLevel:
    """Tests for compute_mhi_e() with nest_aggregate_by != group_by."""

    def test_returns_dataframe(self) -> None:
        adata = _make_mhi_anndata()
        result = compute_mhi_e(adata, group_by="condition", nest_aggregate_by="subject_ID")
        assert isinstance(result, pd.DataFrame)

    def test_expected_columns_present(self) -> None:
        adata = _make_mhi_anndata()
        result = compute_mhi_e(adata, group_by="condition", nest_aggregate_by="subject_ID")
        for col in ("n_individuals", "n_mitos_total",
                    "MHI_E_knn_median", "MHI_E_knn_std",
                    "MHI_E_binning_median", "MHI_E_binning_std"):
            assert col in result.columns, f"Missing column: {col}"

    def test_both_uns_keys_stored(self) -> None:
        adata = _make_mhi_anndata()
        compute_mhi_e(adata, group_by="condition", nest_aggregate_by="subject_ID")
        assert "subject_ID" in adata.uns["mhi_results"]
        assert "E" in adata.uns["mhi_results"]["subject_ID"]
        assert "condition" in adata.uns["mhi_results"]
        assert "E" in adata.uns["mhi_results"]["condition"]

    def test_n_rows_equals_n_conditions(self) -> None:
        adata = _make_mhi_anndata()
        result = compute_mhi_e(adata, group_by="condition", nest_aggregate_by="subject_ID")
        assert len(result) == adata.obs["condition"].nunique()


class TestComputeAllMhiTwoLevel:
    """Tests for compute_all_mhi() with nest_aggregate_by != group_by."""

    def test_returns_dataframe_with_all_metrics(self) -> None:
        adata = _make_mhi_anndata()
        result = compute_all_mhi(
            adata, group_by="condition", nest_aggregate_by="subject_ID",
            metrics=["D", "S", "E"],
        )
        assert isinstance(result, pd.DataFrame)
        assert "MHI_D_median" in result.columns
        assert "MHI_S_gmm_delta_bic_median" in result.columns
        assert "MHI_E_knn_median" in result.columns

    def test_index_is_group_col(self) -> None:
        adata = _make_mhi_anndata()
        result = compute_all_mhi(
            adata, group_by="condition", nest_aggregate_by="subject_ID", metrics=["D"],
        )
        assert result.index.name == "condition"

    def test_group_level_all_stored_in_uns(self) -> None:
        adata = _make_mhi_anndata()
        compute_all_mhi(
            adata, group_by="condition", nest_aggregate_by="subject_ID", metrics=["D"],
        )
        assert "all" in adata.uns["mhi_results"]["condition"]

    def test_individual_level_d_also_stored(self) -> None:
        adata = _make_mhi_anndata()
        compute_all_mhi(
            adata, group_by="condition", nest_aggregate_by="subject_ID", metrics=["D"],
        )
        assert "subject_ID" in adata.uns["mhi_results"]
        assert "D" in adata.uns["mhi_results"]["subject_ID"]

    def test_nest_aggregate_by_forwarded_without_error(self) -> None:
        adata = _make_mhi_anndata()
        # Should not raise even with all metrics.
        compute_all_mhi(
            adata, group_by="condition", nest_aggregate_by="subject_ID",
            metrics=["D", "S", "E"],
        )


class TestAnalyzeMhiSensitivityTwoLevel:
    """Tests for analyze_mhi_sensitivity() with nest_aggregate_by != group_by."""

    def test_group_value_column_contains_individual_ids(self) -> None:
        adata = _make_mhi_anndata(n_subjects=4, n_mitos_per_subject=N_MITOS_LARGE)
        result = analyze_mhi_sensitivity(
            adata,
            group_by="condition",
            nest_aggregate_by="subject_ID",
            n_subsamples=[20, 30],
            n_repetitions=2,
        )
        # group_value should contain individual IDs, not condition names.
        condition_values = set(adata.obs["condition"].unique())
        individual_values = set(adata.obs["subject_ID"].unique())
        result_values = set(result["group_value"].unique())
        # Must overlap with individuals, not conditions.
        assert result_values <= individual_values
        assert not result_values.issubset(condition_values)

    def test_runs_without_exception(self) -> None:
        adata = _make_mhi_anndata(n_subjects=4, n_mitos_per_subject=N_MITOS_LARGE)
        analyze_mhi_sensitivity(
            adata,
            group_by="condition",
            nest_aggregate_by="subject_ID",
            n_subsamples=[20, 30],
            n_repetitions=2,
        )


class TestPlotMhiDistancesTwoLevel:
    """Smoke tests for plot_mhi_distances() in two-level mode."""

    def test_runs_without_exception(self) -> None:
        adata = _make_mhi_anndata()
        plot_mhi_distances(
            adata, group_by="condition", nest_aggregate_by="subject_ID"
        )

    def test_with_precomputed_mhi_results(self) -> None:
        adata = _make_mhi_anndata()
        mhi_df = compute_mhi_d(
            adata, group_by="condition", nest_aggregate_by="subject_ID"
        )
        plot_mhi_distances(
            adata, group_by="condition", nest_aggregate_by="subject_ID",
            mhi_results=mhi_df,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Helper fixture: AnnData with specie + age_group columns
# ─────────────────────────────────────────────────────────────────────────────


def _make_species_anndata() -> anndata.AnnData:
    """
    Build AnnData with 'specie' and 'age_group' columns.

    4 subjects:
        sub_0 → SpeciesA, young
        sub_1 → SpeciesA, old
        sub_2 → SpeciesB, young
        sub_3 → SpeciesB, old
    """
    adata = _make_mhi_anndata(n_subjects=4)
    species_map = {
        "sub_0": "SpeciesA", "sub_1": "SpeciesA",
        "sub_2": "SpeciesB", "sub_3": "SpeciesB",
    }
    age_map = {
        "sub_0": "young", "sub_1": "old",
        "sub_2": "young", "sub_3": "old",
    }
    adata.obs["specie"] = adata.obs["subject_ID"].map(species_map)
    adata.obs["age_group"] = adata.obs["subject_ID"].map(age_map)
    return adata


# ─────────────────────────────────────────────────────────────────────────────
# Tests: compute_mhi_s — new method='distances' and method='all'
# ─────────────────────────────────────────────────────────────────────────────


class TestComputeMhiSDistancesMethod:
    """Tests for the new method='distances' and method='all' in compute_mhi_s."""

    def test_distances_method_adds_columns(self) -> None:
        adata = _make_mhi_anndata()
        result = compute_mhi_s(adata, group_by="subject_ID", method="distances")
        assert "MHI_S_dip_dist" in result.columns
        assert "MHI_S_dip_dist_pval" in result.columns

    def test_distances_method_gmm_columns_absent(self) -> None:
        adata = _make_mhi_anndata()
        result = compute_mhi_s(adata, group_by="subject_ID", method="distances")
        assert "MHI_S_gmm_delta_bic" not in result.columns
        assert "MHI_S_dip" not in result.columns

    def test_both_method_does_not_add_distance_columns(self) -> None:
        """Backward compatibility: method='both' must not include distance columns."""
        adata = _make_mhi_anndata()
        result = compute_mhi_s(adata, group_by="subject_ID", method="both")
        assert "MHI_S_dip_dist" not in result.columns
        assert "MHI_S_dip_dist_pval" not in result.columns

    def test_all_method_has_all_columns(self) -> None:
        adata = _make_mhi_anndata()
        result = compute_mhi_s(adata, group_by="subject_ID", method="all")
        for col in ("MHI_S_dip", "MHI_S_dip_pval",
                    "MHI_S_gmm_delta_bic", "MHI_S_gmm_k_opt",
                    "MHI_S_dip_dist", "MHI_S_dip_dist_pval"):
            assert col in result.columns, f"Missing column: {col}"

    def test_distances_pval_in_unit_interval(self) -> None:
        if not mhi_module._DIPTEST_AVAILABLE:
            pytest.skip("diptest not installed")
        adata = _make_mhi_anndata()
        result = compute_mhi_s(adata, group_by="subject_ID", method="distances")
        valid_pvals = result["MHI_S_dip_dist_pval"].dropna()
        assert ((valid_pvals >= 0.0) & (valid_pvals <= 1.0)).all()

    def test_distances_stat_non_negative(self) -> None:
        if not mhi_module._DIPTEST_AVAILABLE:
            pytest.skip("diptest not installed")
        adata = _make_mhi_anndata()
        result = compute_mhi_s(adata, group_by="subject_ID", method="distances")
        valid_stats = result["MHI_S_dip_dist"].dropna()
        assert (valid_stats >= 0.0).all()

    def test_distances_nan_when_diptest_missing(self) -> None:
        original = mhi_module._DIPTEST_AVAILABLE
        mhi_module._DIPTEST_AVAILABLE = False
        try:
            adata = _make_mhi_anndata()
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                result = compute_mhi_s(adata, group_by="subject_ID", method="distances")
            assert result["MHI_S_dip_dist"].isna().all()
            assert any("diptest" in str(w.message).lower() for w in caught)
        finally:
            mhi_module._DIPTEST_AVAILABLE = original

    def test_result_stored_in_uns(self) -> None:
        adata = _make_mhi_anndata()
        compute_mhi_s(adata, group_by="subject_ID", method="distances")
        assert "S" in adata.uns["mhi_results"]["subject_ID"]

    def test_two_level_distances_aggregated(self) -> None:
        """Two-level mode should aggregate MHI_S_dip_dist across individuals."""
        if not mhi_module._DIPTEST_AVAILABLE:
            pytest.skip("diptest not installed")
        adata = _make_mhi_anndata()
        result = compute_mhi_s(
            adata,
            group_by="condition",
            nest_aggregate_by="subject_ID",
            method="distances",
        )
        assert "MHI_S_dip_dist_median" in result.columns


# ─────────────────────────────────────────────────────────────────────────────
# Tests: compute_mhi_e — n_neighbors='auto'
# ─────────────────────────────────────────────────────────────────────────────


class TestComputeMhiEAutoNeighbors:
    """Tests for n_neighbors='auto' in compute_mhi_e."""

    def test_auto_runs_without_error(self) -> None:
        adata = _make_mhi_anndata()
        result = compute_mhi_e(adata, group_by="subject_ID", method="knn",
                               n_neighbors="auto")
        assert isinstance(result, pd.DataFrame)

    def test_auto_returns_expected_columns(self) -> None:
        adata = _make_mhi_anndata()
        result = compute_mhi_e(adata, group_by="subject_ID", method="knn",
                               n_neighbors="auto")
        assert "MHI_E_knn" in result.columns

    def test_auto_entropy_not_all_nan(self) -> None:
        adata = _make_mhi_anndata()
        result = compute_mhi_e(adata, group_by="subject_ID", method="knn",
                               n_neighbors="auto")
        assert result["MHI_E_knn"].notna().any()

    def test_auto_produces_different_values_than_fixed_k(self) -> None:
        """With 60 mitos, auto k=8 ≠ k=5, so entropy values must differ."""
        adata = _make_mhi_anndata()
        result_auto = compute_mhi_e(adata, group_by="subject_ID", method="knn",
                                    n_neighbors="auto")
        result_fixed = compute_mhi_e(adata, group_by="subject_ID", method="knn",
                                     n_neighbors=5)
        # At least one value should differ between auto and k=5.
        auto_vals = result_auto["MHI_E_knn"].dropna().values
        fixed_vals = result_fixed["MHI_E_knn"].dropna().values
        assert not np.allclose(auto_vals, fixed_vals), (
            "Expected auto-k and fixed-k=5 to give different entropy values "
            f"for n_mitos=60 (auto k=8 ≠ 5). Got auto={auto_vals}, fixed={fixed_vals}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Tests: save_mhi_to_obs
# ─────────────────────────────────────────────────────────────────────────────


class TestSaveMhiToObs:
    """Tests for save_mhi_to_obs()."""

    def test_adds_columns_to_obs(self) -> None:
        adata = _make_mhi_anndata()
        compute_all_mhi(adata, group_by="subject_ID", metrics=["D"])
        save_mhi_to_obs(adata, nest_aggregate_by="subject_ID")
        assert "MHI_D_median" in adata.obs.columns

    def test_values_aligned_with_subject(self) -> None:
        """Each row in .obs should carry its individual's MHI-D value."""
        adata = _make_mhi_anndata()
        compute_all_mhi(adata, group_by="subject_ID", metrics=["D"])
        save_mhi_to_obs(adata, nest_aggregate_by="subject_ID")
        # All mitos from sub_0 should share the same MHI_D_median value.
        sub_0_vals = adata.obs[adata.obs["subject_ID"] == "sub_0"]["MHI_D_median"]
        assert sub_0_vals.nunique() == 1

    def test_no_mhi_results_raises_value_error(self) -> None:
        adata = _make_mhi_anndata()
        with pytest.raises(ValueError, match="No MHI results"):
            save_mhi_to_obs(adata, nest_aggregate_by="subject_ID")

    def test_wrong_column_raises_value_error(self) -> None:
        adata = _make_mhi_anndata()
        compute_all_mhi(adata, group_by="subject_ID", metrics=["D"])
        with pytest.raises(ValueError):
            save_mhi_to_obs(adata, nest_aggregate_by="nonexistent_col")

    def test_missing_individual_produces_nan(self) -> None:
        """Rows for individuals not in MHI results receive NaN."""
        adata = _make_mhi_anndata(n_subjects=4)
        # Compute MHI only for subject_IDs (all 4), then add a fake individual.
        compute_all_mhi(adata, group_by="subject_ID", metrics=["D"])
        adata.obs.loc[adata.obs.index[0], "subject_ID"] = "ghost_subject"
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            save_mhi_to_obs(adata, nest_aggregate_by="subject_ID")
        assert any("NaN" in str(w.message) for w in caught)

    def test_two_level_saves_individual_level_values(self) -> None:
        """Two-level: saves per-individual MHI, not group-level values."""
        adata = _make_mhi_anndata()
        compute_all_mhi(
            adata, group_by="condition", nest_aggregate_by="subject_ID", metrics=["D"]
        )
        save_mhi_to_obs(adata, nest_aggregate_by="subject_ID")
        # All 4 subjects should have distinct MHI_D_median values in obs.
        unique_vals = adata.obs.groupby("subject_ID")["MHI_D_median"].first()
        assert unique_vals.nunique() == 4


# ─────────────────────────────────────────────────────────────────────────────
# Tests: run_mhi_statistics
# ─────────────────────────────────────────────────────────────────────────────


class TestRunMhiStatistics:
    """Tests for run_mhi_statistics()."""

    def _prepare(self) -> anndata.AnnData:
        adata = _make_mhi_anndata()
        compute_all_mhi(
            adata,
            group_by="condition",
            nest_aggregate_by="subject_ID",
            metrics=["D"],
        )
        return adata

    def test_returns_dict(self) -> None:
        adata = self._prepare()
        result = run_mhi_statistics(
            adata, group_by="condition", nest_aggregate_by="subject_ID",
            metric_cols=["MHI_D_median"], n_permutations=100,
        )
        assert isinstance(result, dict)

    def test_contains_expected_keys(self) -> None:
        adata = self._prepare()
        result = run_mhi_statistics(
            adata, group_by="condition", nest_aggregate_by="subject_ID",
            metric_cols=["MHI_D_median"], n_permutations=100,
        )
        assert "MHI_D_median" in result
        for key in ("kw_stat", "kw_pval", "perm_pval"):
            assert key in result["MHI_D_median"]

    def test_perm_pval_in_unit_interval(self) -> None:
        adata = self._prepare()
        result = run_mhi_statistics(
            adata, group_by="condition", nest_aggregate_by="subject_ID",
            metric_cols=["MHI_D_median"], n_permutations=200,
        )
        perm_pval = result["MHI_D_median"]["perm_pval"]
        assert 0.0 <= perm_pval <= 1.0

    def test_kw_pval_in_unit_interval(self) -> None:
        adata = self._prepare()
        result = run_mhi_statistics(
            adata, group_by="condition", nest_aggregate_by="subject_ID",
            metric_cols=["MHI_D_median"], n_permutations=100,
        )
        kw_pval = result["MHI_D_median"]["kw_pval"]
        assert 0.0 <= float(kw_pval) <= 1.0

    def test_stored_in_uns(self) -> None:
        adata = self._prepare()
        run_mhi_statistics(
            adata, group_by="condition", nest_aggregate_by="subject_ID",
            metric_cols=["MHI_D_median"], n_permutations=50,
        )
        assert "statistics" in adata.uns["mhi_results"]["condition"]

    def test_no_mhi_results_raises_value_error(self) -> None:
        adata = _make_mhi_anndata()
        with pytest.raises(ValueError, match="No MHI results"):
            run_mhi_statistics(
                adata, group_by="condition", nest_aggregate_by="subject_ID",
            )

    def test_auto_detects_metric_cols(self) -> None:
        adata = self._prepare()
        result = run_mhi_statistics(
            adata, group_by="condition", nest_aggregate_by="subject_ID",
            n_permutations=50,
        )
        assert "MHI_D_median" in result


# ─────────────────────────────────────────────────────────────────────────────
# Tests: run_mhi_age_tests
# ─────────────────────────────────────────────────────────────────────────────


class TestRunMhiAgeTests:
    """Tests for run_mhi_age_tests()."""

    def _prepare(self) -> anndata.AnnData:
        adata = _make_species_anndata()
        compute_all_mhi(
            adata,
            group_by="specie",
            nest_aggregate_by="subject_ID",
            metrics=["D"],
        )
        return adata

    def test_returns_dataframe(self) -> None:
        adata = self._prepare()
        result = run_mhi_age_tests(
            adata, species_col="specie", age_col="age_group",
            nest_aggregate_by="subject_ID",
            metric_cols=["MHI_D_median"], n_permutations=50,
        )
        assert isinstance(result, pd.DataFrame)

    def test_expected_columns_present(self) -> None:
        adata = self._prepare()
        result = run_mhi_age_tests(
            adata, species_col="specie", age_col="age_group",
            nest_aggregate_by="subject_ID",
            metric_cols=["MHI_D_median"], n_permutations=50,
        )
        for col in ("species", "metric_col", "n_young", "n_old",
                    "mwu_stat", "mwu_pval", "perm_pval"):
            assert col in result.columns, f"Missing column: {col}"

    def test_two_species_present(self) -> None:
        adata = self._prepare()
        result = run_mhi_age_tests(
            adata, species_col="specie", age_col="age_group",
            nest_aggregate_by="subject_ID",
            metric_cols=["MHI_D_median"], n_permutations=50,
        )
        assert set(result["species"].unique()) == {"SpeciesA", "SpeciesB"}

    def test_perm_pval_in_unit_interval(self) -> None:
        adata = self._prepare()
        result = run_mhi_age_tests(
            adata, species_col="specie", age_col="age_group",
            nest_aggregate_by="subject_ID",
            metric_cols=["MHI_D_median"], n_permutations=100,
        )
        assert ((result["perm_pval"] >= 0.0) & (result["perm_pval"] <= 1.0)).all()

    def test_missing_species_col_raises_value_error(self) -> None:
        adata = self._prepare()
        with pytest.raises(ValueError):
            run_mhi_age_tests(
                adata, species_col="nonexistent", age_col="age_group",
                nest_aggregate_by="subject_ID",
            )

    def test_stored_in_uns(self) -> None:
        adata = self._prepare()
        run_mhi_age_tests(
            adata, species_col="specie", age_col="age_group",
            nest_aggregate_by="subject_ID",
            metric_cols=["MHI_D_median"], n_permutations=50,
        )
        assert "age_tests" in adata.uns["mhi_results"]["specie"]


# ─────────────────────────────────────────────────────────────────────────────
# Tests: new plot functions (smoke tests)
# ─────────────────────────────────────────────────────────────────────────────


class TestPlotMhiBarplot:
    """Smoke tests for plot_mhi_barplot()."""

    def test_runs_single_level(self) -> None:
        adata = _make_mhi_anndata()
        plot_mhi_barplot(adata, group_by="subject_ID")

    def test_runs_two_level(self) -> None:
        adata = _make_mhi_anndata()
        compute_all_mhi(
            adata, group_by="condition", nest_aggregate_by="subject_ID", metrics=["D"]
        )
        plot_mhi_barplot(
            adata, group_by="condition", nest_aggregate_by="subject_ID",
            metric="MHI_D_median",
        )


class TestPlotMhiScatterDE:
    """Smoke tests for plot_mhi_scatter_de()."""

    def test_runs_single_level(self) -> None:
        adata = _make_mhi_anndata()
        plot_mhi_scatter_de(adata, group_by="subject_ID")

    def test_runs_two_level(self) -> None:
        adata = _make_mhi_anndata()
        plot_mhi_scatter_de(
            adata, group_by="condition", nest_aggregate_by="subject_ID"
        )


class TestPlotMhiStripplot:
    """Smoke tests for plot_mhi_stripplot()."""

    def test_runs_with_age_col(self) -> None:
        adata = _make_species_anndata()
        compute_all_mhi(
            adata, group_by="specie", nest_aggregate_by="subject_ID", metrics=["D"]
        )
        save_mhi_to_obs(adata, nest_aggregate_by="subject_ID")
        plot_mhi_stripplot(
            adata, group_by="specie", age_col="age_group",
            nest_aggregate_by="subject_ID", metrics=["MHI_D_median"],
        )

    def test_missing_age_col_raises_value_error(self) -> None:
        adata = _make_mhi_anndata()
        with pytest.raises(ValueError, match="age_col"):
            plot_mhi_stripplot(adata, group_by="subject_ID", age_col="nonexistent")


class TestPlotMhiSpeciesHeatmap:
    """Smoke tests for plot_mhi_species_heatmap()."""

    def test_runs_single_level(self) -> None:
        adata = _make_mhi_anndata()
        compute_all_mhi(adata, group_by="subject_ID", metrics=["D"])
        plot_mhi_species_heatmap(adata, group_by="subject_ID")

    def test_runs_two_level(self) -> None:
        adata = _make_mhi_anndata()
        compute_all_mhi(
            adata, group_by="condition", nest_aggregate_by="subject_ID", metrics=["D"]
        )
        plot_mhi_species_heatmap(
            adata, group_by="condition", nest_aggregate_by="subject_ID"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Tests: validate_all_chimera_pairs
# ─────────────────────────────────────────────────────────────────────────────


class TestValidateAllChimeraPairs:
    """Tests for validate_all_chimera_pairs()."""

    def test_returns_dataframe(self) -> None:
        adata = _make_species_anndata()
        result = validate_all_chimera_pairs(
            adata,
            species_col="specie",
            age_col="age_group",
            subject_col="subject_ID",
            ratios=[0.3, 0.5, 0.7],
            n_repetitions=2,
            metrics=["D"],
        )
        assert isinstance(result, pd.DataFrame)

    def test_species_and_pair_id_columns_present(self) -> None:
        adata = _make_species_anndata()
        result = validate_all_chimera_pairs(
            adata,
            species_col="specie",
            age_col="age_group",
            subject_col="subject_ID",
            ratios=[0.5],
            n_repetitions=1,
            metrics=["D"],
        )
        assert "species" in result.columns
        assert "pair_id" in result.columns

    def test_both_species_present_in_result(self) -> None:
        adata = _make_species_anndata()
        result = validate_all_chimera_pairs(
            adata,
            species_col="specie",
            age_col="age_group",
            subject_col="subject_ID",
            ratios=[0.5],
            n_repetitions=1,
            metrics=["D"],
        )
        assert "SpeciesA" in result["species"].values
        assert "SpeciesB" in result["species"].values

    def test_missing_species_col_raises(self) -> None:
        adata = _make_mhi_anndata()
        with pytest.raises(ValueError):
            validate_all_chimera_pairs(
                adata, species_col="nonexistent", age_col="condition",
                subject_col="subject_ID",
            )

    def test_ratio_column_present(self) -> None:
        adata = _make_species_anndata()
        result = validate_all_chimera_pairs(
            adata,
            species_col="specie",
            age_col="age_group",
            subject_col="subject_ID",
            ratios=[0.3, 0.7],
            n_repetitions=1,
            metrics=["D"],
        )
        assert "ratio" in result.columns
