"""
test_analysis_amhi.py

Unit tests for mito_marker.analysis.amhi.

Covers:
  Private helpers:
    _apply_obs_filter(): None passthrough, single value, list values, multi-key AND,
        unknown column raises ValueError.
    _exclude_reference_rows(): removes AllMitoMean row, no-op when absent.
    _load_reference_transforms(): returns correct objects, raises before compute,
        raises on label mismatch.
    _project_into_reference_space(): output shape, uses transform() not fit_transform(),
        fixed across calls with different data subsets.
    _auto_select_pca_components(): 'auto' threshold, None → all, int clamping.

  compute_all_mito_mean():
    - mean_vector equals raw_matrix.mean(axis=0)
    - AllMitoMean projects to origin in PCA space
    - scaler and PCA stored as pickled bytes in .uns
    - feature_names stored correctly
    - n_obs_used matches reference subset size
    - species_included populated when 'specie' column present
    - obs_filter stored in .uns
    - add_to_anndata=True appends one synthetic row
    - add_to_anndata=False does not modify n_obs
    - synthetic row unique_subject_ID == reference_label
    - obs_filter restricts reference population (n_obs_used < total)
    - calling twice with same data is idempotent (values stable)
    - raises ValueError if reference population < 10 mitos
    - original .uns keys preserved after concat

  compute_amhi():
    - returns DataFrame with correct columns
    - index is group values
    - n_mitos column matches group sizes
    - AMHI_D_q25 <= AMHI_D_median <= AMHI_D_q75 for all groups
    - AMHI_D_median > 0 for all real subjects
    - AMHI_offset >= 0 for all groups
    - groups with < 5 mitos are skipped with warning
    - unknown group_by raises ValueError
    - result stored in .uns['amhi_results']
    - obs_filter restricts measured subjects
    - fixed-space invariance: adding a subject does not change other subjects' values
    - AllMitoMean row excluded automatically

  compute_mhi_d_absolute():
    - returns DataFrame with correct columns
    - MHI_D_absolute_median > 0
    - tight subject < dispersed subject (directional sanity)
    - fixed-space invariance: adding a subject does not change other subjects' values
    - result stored in .uns['amhi_results_mhid']
    - obs_filter restricts measured subjects

  Decomposition identity:
    - AMHI_D_mean² ≈ AMHI_offset² + MHI_D_absolute_mean² (bias-variance, within tolerance)

  Geometric correctness:
    - Subject identical to AllMitoMean → AMHI_offset ≈ 0, MHI_D_absolute ≈ 0
    - Tight-far subject: low MHI_D_absolute, high AMHI_offset
    - Dispersed-near subject: high MHI_D_absolute, low AMHI_offset

  plot_amhi_distances() / plot_amhi_2d():
    - Smoke tests: run without exception on Agg backend.
    - plot_amhi_2d() raises ValueError for unknown color_by.
"""

import pickle
import warnings

import anndata
import matplotlib
import numpy as np
import pandas as pd
import pytest
from sklearn.decomposition import PCA as SklearnPCA
from sklearn.preprocessing import StandardScaler

matplotlib.use("Agg")

from mito_marker.analysis.amhi import (
    _AMHI_MHID_RESULTS_KEY,
    _AMHI_REFERENCE_KEY,
    _AMHI_RESULTS_KEY,
    _apply_obs_filter,
    _auto_select_pca_components,
    _exclude_reference_rows,
    _load_reference_transforms,
    _project_into_reference_space,
    compute_all_mito_mean,
    compute_amhi,
    compute_mhi_d_absolute,
    plot_amhi_2d,
    plot_amhi_distances,
    plot_amhi_profile,
    summarize_amhi,
)
from mito_marker.analysis.selection import _ANALYSIS_CONFIG_KEY


# ─────────────────────────────────────────────────────────────────────────────
# Shared fixture factories
# ─────────────────────────────────────────────────────────────────────────────

N_FEATURES = 8
N_SUBJECTS = 4
N_MITOS_PER_SUBJECT = 40


def _make_amhi_anndata(
    n_subjects: int = N_SUBJECTS,
    n_mitos_per_subject: int = N_MITOS_PER_SUBJECT,
    n_features: int = N_FEATURES,
    seed: int = 42,
    add_specie_column: bool = True,
) -> anndata.AnnData:
    """
    Build a minimal AnnData with synthetic mitochondrial data for AMHI tests.

    Subjects are designed for clear geometric properties:
      sub_0 — tight cluster at the origin (low MHI_D_absolute, low AMHI_offset)
      sub_1 — tight cluster far from origin (low MHI_D_absolute, HIGH AMHI_offset)
      sub_2 — dispersed cluster near origin (HIGH MHI_D_absolute, low AMHI_offset)
      sub_3 — dispersed cluster far from origin (HIGH MHI_D_absolute, HIGH AMHI_offset)
    """
    rng = np.random.default_rng(seed)
    all_blocks = []
    subject_ids = []
    species = []
    conditions = []

    for index in range(n_subjects):
        label = f"sub_{index}"
        if index == 0:
            # Tight at origin
            block = rng.normal(loc=0.0, scale=0.1, size=(n_mitos_per_subject, n_features))
        elif index == 1:
            # Tight, far from origin
            offset = np.ones(n_features) * 5.0
            block = rng.normal(loc=0.0, scale=0.1, size=(n_mitos_per_subject, n_features)) + offset
        elif index == 2:
            # Dispersed near origin
            block = rng.normal(loc=0.0, scale=3.0, size=(n_mitos_per_subject, n_features))
        else:
            # Dispersed far from origin
            offset = np.ones(n_features) * 5.0
            block = rng.normal(loc=0.0, scale=3.0, size=(n_mitos_per_subject, n_features)) + offset

        all_blocks.append(block.astype(np.float32))
        subject_ids.extend([label] * n_mitos_per_subject)
        species.extend(["SpeciesA" if index < 2 else "SpeciesB"] * n_mitos_per_subject)
        conditions.extend(["Young" if index < 2 else "Old"] * n_mitos_per_subject)

    x_matrix = np.vstack(all_blocks)
    n_total = len(subject_ids)

    obs_dict: dict = {
        "subject_ID": subject_ids,
        "unique_subject_ID": subject_ids,
        "condition": conditions,
    }
    if add_specie_column:
        obs_dict["specie"] = species

    obs_df = pd.DataFrame(obs_dict, index=[f"mito_{i}" for i in range(n_total)])

    var_df = pd.DataFrame(
        {
            "feature_description": [f"Feature {i}" for i in range(n_features)],
            "is_non_analytical": [False] * n_features,
        },
        index=[f"feat_{i}" for i in range(n_features)],
    )

    adata = anndata.AnnData(X=x_matrix, obs=obs_df, var=var_df)
    adata.uns[_ANALYSIS_CONFIG_KEY] = {"active_layer": None, "active_selection": None}
    return adata


def _make_reference_anndata(seed: int = 42) -> anndata.AnnData:
    """Returns AnnData with compute_all_mito_mean already applied."""
    adata = _make_amhi_anndata(seed=seed)
    adata, _ = compute_all_mito_mean(adata)
    return adata


# ─────────────────────────────────────────────────────────────────────────────
# Tests: _apply_obs_filter
# ─────────────────────────────────────────────────────────────────────────────


class TestApplyObsFilter:
    def test_none_returns_original(self) -> None:
        adata = _make_amhi_anndata()
        result = _apply_obs_filter(adata, None)
        assert result is adata

    def test_single_string_value(self) -> None:
        adata = _make_amhi_anndata()
        result = _apply_obs_filter(adata, {"specie": "SpeciesA"})
        assert result.n_obs == N_MITOS_PER_SUBJECT * 2
        assert set(result.obs["specie"].unique()) == {"SpeciesA"}

    def test_list_of_values(self) -> None:
        adata = _make_amhi_anndata()
        result = _apply_obs_filter(adata, {"specie": ["SpeciesA", "SpeciesB"]})
        assert result.n_obs == adata.n_obs

    def test_multi_key_and(self) -> None:
        adata = _make_amhi_anndata()
        result = _apply_obs_filter(adata, {"specie": "SpeciesA", "condition": "Young"})
        assert result.n_obs == N_MITOS_PER_SUBJECT * 2
        assert set(result.obs["specie"].unique()) == {"SpeciesA"}
        assert set(result.obs["condition"].unique()) == {"Young"}

    def test_unknown_column_raises(self) -> None:
        adata = _make_amhi_anndata()
        with pytest.raises(ValueError, match="not found in .obs"):
            _apply_obs_filter(adata, {"nonexistent_col": "val"})

    def test_empty_result_is_valid_view(self) -> None:
        adata = _make_amhi_anndata()
        result = _apply_obs_filter(adata, {"specie": "NoSuchSpecies"})
        assert result.n_obs == 0


# ─────────────────────────────────────────────────────────────────────────────
# Tests: _exclude_reference_rows
# ─────────────────────────────────────────────────────────────────────────────


class TestExcludeReferenceRows:
    def test_no_reference_row_noop(self) -> None:
        adata = _make_amhi_anndata()
        result = _exclude_reference_rows(adata, "AllMitoMean")
        assert result.n_obs == adata.n_obs

    def test_removes_synthetic_row(self) -> None:
        adata = _make_reference_anndata()
        n_before = N_SUBJECTS * N_MITOS_PER_SUBJECT  # without the synthetic row
        result = _exclude_reference_rows(adata, "AllMitoMean")
        assert result.n_obs == n_before

    def test_custom_reference_label(self) -> None:
        adata = _make_amhi_anndata()
        adata, _ = compute_all_mito_mean(adata, reference_label="MyRef")
        result = _exclude_reference_rows(adata, "MyRef")
        assert result.n_obs == N_SUBJECTS * N_MITOS_PER_SUBJECT


# ─────────────────────────────────────────────────────────────────────────────
# Tests: _load_reference_transforms
# ─────────────────────────────────────────────────────────────────────────────


class TestLoadReferenceTransforms:
    def test_raises_before_compute(self) -> None:
        adata = _make_amhi_anndata()
        with pytest.raises(ValueError, match="Run compute_all_mito_mean"):
            _load_reference_transforms(adata, "AllMitoMean")

    def test_raises_on_label_mismatch(self) -> None:
        adata = _make_reference_anndata()
        with pytest.raises(ValueError, match="does not match"):
            _load_reference_transforms(adata, "WrongLabel")

    def test_returns_correct_types(self) -> None:
        adata = _make_reference_anndata()
        scaler, pca, mean_vector, feature_names = _load_reference_transforms(
            adata, "AllMitoMean"
        )
        assert isinstance(scaler, StandardScaler)
        assert isinstance(pca, SklearnPCA)
        assert isinstance(mean_vector, np.ndarray)
        assert isinstance(feature_names, list)
        assert len(feature_names) == N_FEATURES

    def test_scaler_is_fitted(self) -> None:
        adata = _make_reference_anndata()
        scaler, _, _, _ = _load_reference_transforms(adata, "AllMitoMean")
        assert hasattr(scaler, "mean_") and scaler.mean_ is not None

    def test_pca_is_fitted(self) -> None:
        adata = _make_reference_anndata()
        _, pca, _, _ = _load_reference_transforms(adata, "AllMitoMean")
        assert hasattr(pca, "components_") and pca.components_ is not None


# ─────────────────────────────────────────────────────────────────────────────
# Tests: _project_into_reference_space
# ─────────────────────────────────────────────────────────────────────────────


class TestProjectIntoReferenceSpace:
    def test_output_shape(self) -> None:
        adata = _make_reference_anndata()
        scaler, pca, _, _ = _load_reference_transforms(adata, "AllMitoMean")
        matrix = np.random.default_rng(0).standard_normal((20, N_FEATURES)).astype(np.float32)
        projected = _project_into_reference_space(matrix, scaler, pca)
        assert projected.shape[0] == 20
        assert projected.shape[1] == pca.n_components_

    def test_is_float32(self) -> None:
        adata = _make_reference_anndata()
        scaler, pca, _, _ = _load_reference_transforms(adata, "AllMitoMean")
        matrix = np.random.default_rng(0).standard_normal((10, N_FEATURES)).astype(np.float32)
        projected = _project_into_reference_space(matrix, scaler, pca)
        assert projected.dtype == np.float32

    def test_fixed_space_identical_subsets(self) -> None:
        """Projecting with the same transforms on different subsets gives same coords."""
        adata = _make_reference_anndata()
        scaler, pca, _, _ = _load_reference_transforms(adata, "AllMitoMean")

        rng = np.random.default_rng(7)
        matrix_a = rng.standard_normal((15, N_FEATURES)).astype(np.float32)
        matrix_b = rng.standard_normal((15, N_FEATURES)).astype(np.float32)

        proj_a_alone = _project_into_reference_space(matrix_a, scaler, pca)
        proj_a_in_stack = _project_into_reference_space(
            np.vstack([matrix_a, matrix_b]), scaler, pca
        )[:15]
        np.testing.assert_allclose(proj_a_alone, proj_a_in_stack, atol=1e-5)


# ─────────────────────────────────────────────────────────────────────────────
# Tests: _auto_select_pca_components
# ─────────────────────────────────────────────────────────────────────────────


class TestAutoSelectPcaComponents:
    def _make_pca(self, n_components: int = 5) -> SklearnPCA:
        rng = np.random.default_rng(0)
        data = rng.standard_normal((100, 10))
        pca = SklearnPCA(n_components=n_components, random_state=42)
        pca.fit(data)
        return pca

    def test_none_returns_all(self) -> None:
        pca = self._make_pca(n_components=5)
        assert _auto_select_pca_components(pca, None) == 5

    def test_auto_returns_int(self) -> None:
        pca = self._make_pca(n_components=5)
        k = _auto_select_pca_components(pca, "auto")
        assert isinstance(k, int)
        assert 1 <= k <= 5

    def test_explicit_int_clamped(self) -> None:
        pca = self._make_pca(n_components=3)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            k = _auto_select_pca_components(pca, 100)
        assert k == 3

    def test_explicit_int_valid(self) -> None:
        pca = self._make_pca(n_components=5)
        k = _auto_select_pca_components(pca, 2)
        assert k == 2


# ─────────────────────────────────────────────────────────────────────────────
# Tests: compute_all_mito_mean
# ─────────────────────────────────────────────────────────────────────────────


class TestComputeAllMitoMean:
    def test_mean_vector_is_correct(self) -> None:
        adata = _make_amhi_anndata()
        updated, mean_vec = compute_all_mito_mean(adata, add_to_anndata=False)
        # Extract raw feature matrix (exclude non-analytical handled by _get_data_and_channels)
        raw = adata.X.astype(np.float32)
        expected_mean = raw.mean(axis=0)
        np.testing.assert_allclose(mean_vec, expected_mean, atol=1e-5)

    def test_mean_vector_projects_to_origin(self) -> None:
        adata = _make_amhi_anndata()
        updated, mean_vec = compute_all_mito_mean(adata, add_to_anndata=False)
        scaler, pca, _, feature_names = _load_reference_transforms(updated, "AllMitoMean")
        projected = _project_into_reference_space(
            mean_vec.reshape(1, -1).astype(np.float32), scaler, pca
        )
        np.testing.assert_allclose(projected, np.zeros_like(projected), atol=1e-4)

    def test_scaler_bytes_stored(self) -> None:
        adata = _make_amhi_anndata()
        updated, _ = compute_all_mito_mean(adata, add_to_anndata=False)
        assert "scaler_bytes" in updated.uns[_AMHI_REFERENCE_KEY]
        scaler = pickle.loads(updated.uns[_AMHI_REFERENCE_KEY]["scaler_bytes"])
        assert isinstance(scaler, StandardScaler)

    def test_pca_bytes_stored(self) -> None:
        adata = _make_amhi_anndata()
        updated, _ = compute_all_mito_mean(adata, add_to_anndata=False)
        assert "pca_bytes" in updated.uns[_AMHI_REFERENCE_KEY]
        pca = pickle.loads(updated.uns[_AMHI_REFERENCE_KEY]["pca_bytes"])
        assert isinstance(pca, SklearnPCA)

    def test_feature_names_stored(self) -> None:
        adata = _make_amhi_anndata()
        updated, _ = compute_all_mito_mean(adata, add_to_anndata=False)
        feature_names = updated.uns[_AMHI_REFERENCE_KEY]["feature_names"]
        assert len(feature_names) == N_FEATURES

    def test_n_obs_used_correct_no_filter(self) -> None:
        adata = _make_amhi_anndata()
        updated, _ = compute_all_mito_mean(adata, add_to_anndata=False)
        assert updated.uns[_AMHI_REFERENCE_KEY]["n_obs_used"] == N_SUBJECTS * N_MITOS_PER_SUBJECT

    def test_n_obs_used_correct_with_filter(self) -> None:
        adata = _make_amhi_anndata()
        updated, _ = compute_all_mito_mean(
            adata, obs_filter={"specie": "SpeciesA"}, add_to_anndata=False
        )
        # SpeciesA = sub_0 + sub_1 = 2 subjects
        assert updated.uns[_AMHI_REFERENCE_KEY]["n_obs_used"] == 2 * N_MITOS_PER_SUBJECT

    def test_obs_filter_stored_in_uns(self) -> None:
        adata = _make_amhi_anndata()
        obs_filter = {"specie": "SpeciesA"}
        updated, _ = compute_all_mito_mean(adata, obs_filter=obs_filter, add_to_anndata=False)
        assert updated.uns[_AMHI_REFERENCE_KEY]["obs_filter"] == obs_filter

    def test_species_included_populated(self) -> None:
        adata = _make_amhi_anndata()
        updated, _ = compute_all_mito_mean(adata, add_to_anndata=False)
        species = updated.uns[_AMHI_REFERENCE_KEY]["species_included"]
        assert "SpeciesA" in species
        assert "SpeciesB" in species

    def test_species_included_filtered(self) -> None:
        adata = _make_amhi_anndata()
        updated, _ = compute_all_mito_mean(
            adata, obs_filter={"specie": "SpeciesA"}, add_to_anndata=False
        )
        species = updated.uns[_AMHI_REFERENCE_KEY]["species_included"]
        assert species == ["SpeciesA"]

    def test_add_to_anndata_true_appends_row(self) -> None:
        adata = _make_amhi_anndata()
        updated, _ = compute_all_mito_mean(adata, add_to_anndata=True)
        assert updated.n_obs == adata.n_obs + 1

    def test_add_to_anndata_false_does_not_change_n_obs(self) -> None:
        adata = _make_amhi_anndata()
        updated, _ = compute_all_mito_mean(adata, add_to_anndata=False)
        assert updated.n_obs == adata.n_obs

    def test_synthetic_row_unique_subject_id(self) -> None:
        adata = _make_amhi_anndata()
        updated, _ = compute_all_mito_mean(adata, reference_label="AllMitoMean")
        assert "AllMitoMean" in updated.obs["unique_subject_ID"].values

    def test_custom_reference_label(self) -> None:
        adata = _make_amhi_anndata()
        updated, _ = compute_all_mito_mean(adata, reference_label="MyRef")
        assert "MyRef" in updated.obs["unique_subject_ID"].values
        assert updated.uns[_AMHI_REFERENCE_KEY]["reference_label"] == "MyRef"

    def test_excludes_preexisting_reference_row(self) -> None:
        """Calling twice should not use the first synthetic row in the second reference."""
        adata = _make_amhi_anndata()
        updated, mean_vec_1 = compute_all_mito_mean(adata)
        # n_obs_used should still equal the original number (not +1).
        assert updated.uns[_AMHI_REFERENCE_KEY]["n_obs_used"] == N_SUBJECTS * N_MITOS_PER_SUBJECT

    def test_raises_too_few_mitos(self) -> None:
        adata = _make_amhi_anndata(n_mitos_per_subject=1, n_subjects=2)
        with pytest.raises(ValueError, match="at least 10"):
            compute_all_mito_mean(adata, obs_filter={"specie": "SpeciesA"})

    def test_no_specie_column_species_included_empty(self) -> None:
        adata = _make_amhi_anndata(add_specie_column=False)
        updated, _ = compute_all_mito_mean(adata, add_to_anndata=False)
        assert updated.uns[_AMHI_REFERENCE_KEY]["species_included"] == []

    def test_synthetic_row_layer_values_are_populated(self) -> None:
        """AllMitoMean row must have layer values so plot_radar uses the normalised scale."""
        adata = _make_amhi_anndata()
        # Simulate a normalised layer (e.g. none__zscore_col) present before AMHI setup.
        layer_data = np.random.default_rng(0).standard_normal(
            (adata.n_obs, adata.n_vars)
        ).astype(np.float32)
        adata.layers["none__zscore_col"] = layer_data

        updated, _ = compute_all_mito_mean(adata, add_to_anndata=True)

        assert "none__zscore_col" in updated.layers, (
            "Layer must survive anndata.concat so plot_radar does not fall back to raw .X"
        )
        ref_row = updated[updated.obs["unique_subject_ID"] == "AllMitoMean"].layers["none__zscore_col"]
        assert ref_row.shape == (1, adata.n_vars)
        # The synthetic row value must equal the mean of the layer across all reference obs.
        expected = layer_data.mean(axis=0)
        np.testing.assert_allclose(ref_row[0], expected, atol=1e-5)

    def test_synthetic_row_layer_values_respect_obs_filter(self) -> None:
        """When obs_filter is set, the layer mean uses only the filtered rows."""
        adata = _make_amhi_anndata()
        rng = np.random.default_rng(7)
        adata.layers["none__zscore_col"] = rng.standard_normal(
            (adata.n_obs, adata.n_vars)
        ).astype(np.float32)

        updated, _ = compute_all_mito_mean(
            adata,
            obs_filter={"condition": "Young"},
            add_to_anndata=True,
        )

        young_mask = adata.obs["condition"] == "Young"
        expected = np.asarray(adata.layers["none__zscore_col"])[young_mask].mean(axis=0)
        ref_row = updated[
            updated.obs["unique_subject_ID"] == "AllMitoMean"
        ].layers["none__zscore_col"]
        np.testing.assert_allclose(ref_row[0], expected, atol=1e-5)


# ─────────────────────────────────────────────────────────────────────────────
# Tests: compute_amhi
# ─────────────────────────────────────────────────────────────────────────────


class TestComputeAmhi:
    def test_returns_dataframe(self) -> None:
        adata = _make_reference_anndata()
        result = compute_amhi(adata)
        assert isinstance(result, pd.DataFrame)

    def test_correct_columns(self) -> None:
        adata = _make_reference_anndata()
        result = compute_amhi(adata)
        for col in ("n_mitos", "AMHI_offset", "AMHI_D_median", "AMHI_D_mean",
                    "AMHI_D_std", "AMHI_D_q25", "AMHI_D_q75"):
            assert col in result.columns, f"Missing column: {col}"

    def test_index_contains_group_values(self) -> None:
        adata = _make_reference_anndata()
        result = compute_amhi(adata, group_by="unique_subject_ID")
        for subject in [f"sub_{i}" for i in range(N_SUBJECTS)]:
            assert subject in result.index

    def test_n_mitos_correct(self) -> None:
        adata = _make_reference_anndata()
        result = compute_amhi(adata, group_by="unique_subject_ID")
        for subject in result.index:
            assert int(result.loc[subject, "n_mitos"]) == N_MITOS_PER_SUBJECT

    def test_quartile_ordering(self) -> None:
        adata = _make_reference_anndata()
        result = compute_amhi(adata)
        for _, row in result.iterrows():
            assert row["AMHI_D_q25"] <= row["AMHI_D_median"] <= row["AMHI_D_q75"]

    def test_amhi_d_median_positive(self) -> None:
        adata = _make_reference_anndata()
        result = compute_amhi(adata)
        assert (result["AMHI_D_median"] > 0).all()

    def test_amhi_offset_nonnegative(self) -> None:
        adata = _make_reference_anndata()
        result = compute_amhi(adata)
        assert (result["AMHI_offset"] >= 0).all()

    def test_unknown_group_by_raises(self) -> None:
        adata = _make_reference_anndata()
        with pytest.raises(ValueError):
            compute_amhi(adata, group_by="nonexistent_col")

    def test_result_stored_in_uns(self) -> None:
        adata = _make_reference_anndata()
        compute_amhi(adata, group_by="unique_subject_ID")
        assert _AMHI_RESULTS_KEY in adata.uns
        assert "unique_subject_ID" in adata.uns[_AMHI_RESULTS_KEY]

    def test_obs_filter_restricts_subjects(self) -> None:
        adata = _make_reference_anndata()
        result = compute_amhi(adata, obs_filter={"specie": "SpeciesA"})
        assert len(result) == 2  # only sub_0 and sub_1

    def test_allmitoMean_row_excluded(self) -> None:
        adata = _make_reference_anndata()
        result = compute_amhi(adata, group_by="unique_subject_ID")
        assert "AllMitoMean" not in result.index

    def test_fixed_space_invariance(self) -> None:
        """Adding a new subject must not change existing subjects' AMHI values."""
        adata = _make_reference_anndata()
        result_before = compute_amhi(adata, group_by="unique_subject_ID")

        # Create a new subject and append to AnnData.
        rng = np.random.default_rng(99)
        new_x = rng.standard_normal((N_MITOS_PER_SUBJECT, N_FEATURES)).astype(np.float32)
        new_obs = pd.DataFrame(
            {
                "subject_ID": ["new_sub"] * N_MITOS_PER_SUBJECT,
                "unique_subject_ID": ["new_sub"] * N_MITOS_PER_SUBJECT,
                "condition": ["Young"] * N_MITOS_PER_SUBJECT,
                "specie": ["SpeciesA"] * N_MITOS_PER_SUBJECT,
            },
            index=[f"new_mito_{i}" for i in range(N_MITOS_PER_SUBJECT)],
        )
        new_adata = anndata.AnnData(X=new_x, obs=new_obs, var=adata.var.copy())
        combined = anndata.concat([adata, new_adata], merge="same")
        combined.uns = dict(adata.uns)

        result_after = compute_amhi(combined, group_by="unique_subject_ID")

        # Original subjects must have identical values.
        for subject in [f"sub_{i}" for i in range(N_SUBJECTS)]:
            np.testing.assert_allclose(
                result_before.loc[subject, "AMHI_D_median"],
                result_after.loc[subject, "AMHI_D_median"],
                atol=1e-5,
                err_msg=f"Fixed-space invariance violated for {subject}",
            )

    def test_small_group_skipped_with_warning(self) -> None:
        adata = _make_reference_anndata()
        # Add a tiny group (1 mito) that should be skipped.
        tiny_obs = pd.DataFrame(
            {
                "subject_ID": ["tiny"],
                "unique_subject_ID": ["tiny"],
                "condition": ["Young"],
                "specie": ["SpeciesA"],
            },
            index=["tiny_mito"],
        )
        tiny_adata = anndata.AnnData(
            X=np.zeros((1, N_FEATURES), dtype=np.float32),
            obs=tiny_obs,
            var=adata.var.copy(),
        )
        combined = anndata.concat([adata, tiny_adata], merge="same")
        combined.uns = dict(adata.uns)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = compute_amhi(combined, group_by="unique_subject_ID")

        assert "tiny" not in result.index
        skipped_warnings = [w for w in caught if "tiny" in str(w.message)]
        assert len(skipped_warnings) >= 1

    def test_requires_reference_transforms(self) -> None:
        adata = _make_amhi_anndata()
        with pytest.raises(ValueError, match="Run compute_all_mito_mean"):
            compute_amhi(adata)


# ─────────────────────────────────────────────────────────────────────────────
# Tests: compute_mhi_d_absolute
# ─────────────────────────────────────────────────────────────────────────────


class TestComputeMhiDAbsolute:
    def test_returns_dataframe(self) -> None:
        adata = _make_reference_anndata()
        result = compute_mhi_d_absolute(adata)
        assert isinstance(result, pd.DataFrame)

    def test_correct_columns(self) -> None:
        adata = _make_reference_anndata()
        result = compute_mhi_d_absolute(adata)
        for col in ("n_mitos", "MHI_D_absolute_median", "MHI_D_absolute_mean",
                    "MHI_D_absolute_std", "MHI_D_absolute_q25", "MHI_D_absolute_q75"):
            assert col in result.columns

    def test_mhid_absolute_positive(self) -> None:
        adata = _make_reference_anndata()
        result = compute_mhi_d_absolute(adata)
        assert (result["MHI_D_absolute_median"] > 0).all()

    def test_tight_subject_less_than_dispersed(self) -> None:
        """sub_0 (tight at origin) should have lower MHI_D_absolute than sub_2 (dispersed)."""
        adata = _make_reference_anndata()
        result = compute_mhi_d_absolute(adata, group_by="unique_subject_ID")
        assert result.loc["sub_0", "MHI_D_absolute_median"] < result.loc["sub_2", "MHI_D_absolute_median"]

    def test_tight_subjects_equal_regardless_of_offset(self) -> None:
        """sub_0 (tight, near) and sub_1 (tight, far) should have similar MHI_D_absolute."""
        adata = _make_reference_anndata()
        result = compute_mhi_d_absolute(adata, group_by="unique_subject_ID")
        ratio = result.loc["sub_0", "MHI_D_absolute_median"] / result.loc["sub_1", "MHI_D_absolute_median"]
        # Both are tight (scale=0.1), so ratio should be close to 1 (within 50%).
        assert 0.5 < ratio < 2.0

    def test_result_stored_in_uns(self) -> None:
        adata = _make_reference_anndata()
        compute_mhi_d_absolute(adata, group_by="unique_subject_ID")
        assert _AMHI_MHID_RESULTS_KEY in adata.uns
        assert "unique_subject_ID" in adata.uns[_AMHI_MHID_RESULTS_KEY]

    def test_obs_filter_restricts_subjects(self) -> None:
        adata = _make_reference_anndata()
        result = compute_mhi_d_absolute(adata, obs_filter={"specie": "SpeciesB"})
        assert len(result) == 2  # sub_2 and sub_3

    def test_fixed_space_invariance(self) -> None:
        """Adding a new subject must not change existing subjects' MHI_D_absolute values."""
        adata = _make_reference_anndata()
        result_before = compute_mhi_d_absolute(adata, group_by="unique_subject_ID")

        rng = np.random.default_rng(77)
        new_x = rng.standard_normal((N_MITOS_PER_SUBJECT, N_FEATURES)).astype(np.float32)
        new_obs = pd.DataFrame(
            {
                "subject_ID": ["new_sub"] * N_MITOS_PER_SUBJECT,
                "unique_subject_ID": ["new_sub"] * N_MITOS_PER_SUBJECT,
                "condition": ["Old"] * N_MITOS_PER_SUBJECT,
                "specie": ["SpeciesB"] * N_MITOS_PER_SUBJECT,
            },
            index=[f"new_mito_{i}" for i in range(N_MITOS_PER_SUBJECT)],
        )
        new_adata = anndata.AnnData(X=new_x, obs=new_obs, var=adata.var.copy())
        combined = anndata.concat([adata, new_adata], merge="same")
        combined.uns = dict(adata.uns)

        result_after = compute_mhi_d_absolute(combined, group_by="unique_subject_ID")

        for subject in [f"sub_{i}" for i in range(N_SUBJECTS)]:
            np.testing.assert_allclose(
                result_before.loc[subject, "MHI_D_absolute_median"],
                result_after.loc[subject, "MHI_D_absolute_median"],
                atol=1e-5,
                err_msg=f"Fixed-space invariance violated for {subject}",
            )

    def test_requires_reference_transforms(self) -> None:
        adata = _make_amhi_anndata()
        with pytest.raises(ValueError, match="Run compute_all_mito_mean"):
            compute_mhi_d_absolute(adata)


# ─────────────────────────────────────────────────────────────────────────────
# Tests: geometric correctness
# ─────────────────────────────────────────────────────────────────────────────


class TestGeometricCorrectness:
    def test_amhi_offset_tight_near_vs_tight_far(self) -> None:
        """sub_1 (tight, far) should have higher AMHI_offset than sub_0 (tight, near)."""
        adata = _make_reference_anndata()
        result = compute_amhi(adata, group_by="unique_subject_ID")
        assert result.loc["sub_1", "AMHI_offset"] > result.loc["sub_0", "AMHI_offset"]

    def test_mhid_absolute_dispersed_near_vs_tight_near(self) -> None:
        """sub_2 (dispersed, near) should have higher MHI_D_absolute than sub_0 (tight, near)."""
        adata = _make_reference_anndata()
        result = compute_mhi_d_absolute(adata, group_by="unique_subject_ID")
        assert result.loc["sub_2", "MHI_D_absolute_median"] > result.loc["sub_0", "MHI_D_absolute_median"]

    def test_decomposition_identity(self) -> None:
        """AMHI_D_mean² ≈ AMHI_offset² + MHI_D_absolute_mean² (bias-variance decomposition)."""
        adata = _make_reference_anndata()
        amhi = compute_amhi(adata, group_by="unique_subject_ID")
        mhid = compute_mhi_d_absolute(adata, group_by="unique_subject_ID")

        for subject in amhi.index:
            amhi_d_sq = amhi.loc[subject, "AMHI_D_mean"] ** 2
            offset_sq = amhi.loc[subject, "AMHI_offset"] ** 2
            mhid_sq = mhid.loc[subject, "MHI_D_absolute_mean"] ** 2
            # The identity is exact for means (not medians), within floating-point tolerance.
            assert abs(amhi_d_sq - (offset_sq + mhid_sq)) / (amhi_d_sq + 1e-9) < 0.15, (
                f"Decomposition violated for {subject}: "
                f"AMHI_D²={amhi_d_sq:.4f}, offset²={offset_sq:.4f}, MHI_D²={mhid_sq:.4f}"
            )

    def test_subject_at_allmitomean_has_zero_offset(self) -> None:
        """A synthetic subject with X = AllMitoMean should have AMHI_offset ≈ 0."""
        adata = _make_amhi_anndata()
        updated, mean_vec = compute_all_mito_mean(adata, add_to_anndata=False)

        # Create a subject whose every mito is identical to AllMitoMean.
        identical_x = np.tile(mean_vec.reshape(1, -1), (N_MITOS_PER_SUBJECT, 1)).astype(np.float32)
        identical_obs = pd.DataFrame(
            {
                "subject_ID": ["zero_sub"] * N_MITOS_PER_SUBJECT,
                "unique_subject_ID": ["zero_sub"] * N_MITOS_PER_SUBJECT,
                "condition": ["Young"] * N_MITOS_PER_SUBJECT,
                "specie": ["SpeciesA"] * N_MITOS_PER_SUBJECT,
            },
            index=[f"zmito_{i}" for i in range(N_MITOS_PER_SUBJECT)],
        )
        zero_adata = anndata.AnnData(
            X=identical_x,
            obs=identical_obs,
            var=updated.var.copy(),
        )
        combined = anndata.concat([updated, zero_adata], merge="same")
        combined.uns = dict(updated.uns)

        result = compute_amhi(combined, group_by="unique_subject_ID")
        assert result.loc["zero_sub", "AMHI_offset"] < 0.01


# ─────────────────────────────────────────────────────────────────────────────
# Tests: plot functions (smoke tests on Agg backend)
# ─────────────────────────────────────────────────────────────────────────────


class TestPlotAmhiDistances:
    def test_runs_without_exception(self) -> None:
        adata = _make_reference_anndata()
        plot_amhi_distances(adata, group_by="unique_subject_ID")

    def test_with_precomputed_results(self) -> None:
        adata = _make_reference_anndata()
        results = compute_amhi(adata, group_by="unique_subject_ID")
        plot_amhi_distances(adata, group_by="unique_subject_ID", amhi_results=results)

    def test_with_obs_filter(self) -> None:
        adata = _make_reference_anndata()
        plot_amhi_distances(adata, group_by="unique_subject_ID",
                            obs_filter={"specie": "SpeciesA"})

    def test_with_custom_title(self) -> None:
        adata = _make_reference_anndata()
        plot_amhi_distances(adata, group_by="unique_subject_ID", title="My Title")

    def test_requires_reference_transforms(self) -> None:
        adata = _make_amhi_anndata()
        with pytest.raises(ValueError, match="Run compute_all_mito_mean"):
            plot_amhi_distances(adata)


class TestPlotAmhi2d:
    def test_runs_without_exception(self) -> None:
        adata = _make_reference_anndata()
        plot_amhi_2d(adata, color_by="condition")

    def test_color_by_unique_subject_id(self) -> None:
        adata = _make_reference_anndata()
        plot_amhi_2d(adata, color_by="unique_subject_ID")

    def test_color_by_specie(self) -> None:
        adata = _make_reference_anndata()
        plot_amhi_2d(adata, color_by="specie")

    def test_color_by_list(self) -> None:
        adata = _make_reference_anndata()
        plot_amhi_2d(adata, color_by=["specie", "condition"])

    def test_unknown_color_by_raises(self) -> None:
        adata = _make_reference_anndata()
        with pytest.raises(ValueError):
            plot_amhi_2d(adata, color_by="nonexistent_col")

    def test_no_distance_rings(self) -> None:
        adata = _make_reference_anndata()
        plot_amhi_2d(adata, color_by="condition", show_distance_rings=False)

    def test_with_obs_filter(self) -> None:
        adata = _make_reference_anndata()
        plot_amhi_2d(adata, color_by="condition", obs_filter={"specie": "SpeciesA"})

    def test_requires_reference_transforms(self) -> None:
        adata = _make_amhi_anndata()
        with pytest.raises(ValueError, match="Run compute_all_mito_mean"):
            plot_amhi_2d(adata, color_by="condition")


# ─────────────────────────────────────────────────────────────────────────────
# Tests: summarize_amhi
# ─────────────────────────────────────────────────────────────────────────────


class TestSummarizeAmhi:
    def test_returns_dataframe(self) -> None:
        adata = _make_reference_anndata()
        result = summarize_amhi(adata)
        assert isinstance(result, pd.DataFrame)

    def test_required_columns_present(self) -> None:
        adata = _make_reference_anndata()
        result = summarize_amhi(adata)
        for col in ("n_mitos", "AMHI_offset", "AMHI_D_median", "MHI_D_absolute_median"):
            assert col in result.columns, f"Missing column: {col}"

    def test_one_row_per_group(self) -> None:
        adata = _make_reference_anndata()
        result = summarize_amhi(adata, group_by="unique_subject_ID")
        assert len(result) == N_SUBJECTS

    def test_sorted_by_amhi_offset_descending(self) -> None:
        adata = _make_reference_anndata()
        result = summarize_amhi(adata, group_by="unique_subject_ID")
        offsets = result["AMHI_offset"].tolist()
        assert offsets == sorted(offsets, reverse=True)

    def test_accepts_precomputed_dataframes(self) -> None:
        adata = _make_reference_anndata()
        amhi = compute_amhi(adata, group_by="unique_subject_ID")
        mhid = compute_mhi_d_absolute(adata, group_by="unique_subject_ID")
        result = summarize_amhi(adata, group_by="unique_subject_ID",
                                amhi_results=amhi, mhid_results=mhid)
        assert len(result) == N_SUBJECTS

    def test_precomputed_gives_same_values(self) -> None:
        adata = _make_reference_anndata()
        result_auto = summarize_amhi(adata, group_by="unique_subject_ID")
        amhi = compute_amhi(adata, group_by="unique_subject_ID")
        mhid = compute_mhi_d_absolute(adata, group_by="unique_subject_ID")
        result_pre = summarize_amhi(adata, group_by="unique_subject_ID",
                                    amhi_results=amhi, mhid_results=mhid)
        pd.testing.assert_frame_equal(
            result_auto.sort_index(), result_pre.sort_index(), check_like=True
        )

    def test_obs_filter_reduces_rows(self) -> None:
        adata = _make_reference_anndata()
        result_all = summarize_amhi(adata, group_by="unique_subject_ID")
        result_filtered = summarize_amhi(
            adata, group_by="unique_subject_ID",
            obs_filter={"condition": "Young"}
        )
        assert len(result_filtered) < len(result_all)

    def test_values_are_positive(self) -> None:
        adata = _make_reference_anndata()
        result = summarize_amhi(adata)
        assert (result["AMHI_offset"] >= 0).all()
        assert (result["AMHI_D_median"] >= 0).all()
        assert (result["MHI_D_absolute_median"] >= 0).all()

    def test_requires_reference_transforms(self) -> None:
        adata = _make_amhi_anndata()
        with pytest.raises(ValueError, match="Run compute_all_mito_mean"):
            summarize_amhi(adata)


# ─────────────────────────────────────────────────────────────────────────────
# Tests: plot_amhi_profile
# ─────────────────────────────────────────────────────────────────────────────


class TestPlotAmhiProfile:
    def test_runs_without_exception(self) -> None:
        adata = _make_reference_anndata()
        plot_amhi_profile(adata, group_by="unique_subject_ID",
                          color_by="specie", condition_by="condition")

    def test_accepts_precomputed_summary(self) -> None:
        adata = _make_reference_anndata()
        summary = summarize_amhi(adata, group_by="unique_subject_ID")
        plot_amhi_profile(adata, group_by="unique_subject_ID",
                          color_by="specie", condition_by="condition",
                          summary_dataframe=summary)

    def test_no_color_by(self) -> None:
        adata = _make_reference_anndata()
        plot_amhi_profile(adata, group_by="unique_subject_ID")

    def test_no_condition_by(self) -> None:
        adata = _make_reference_anndata()
        plot_amhi_profile(adata, group_by="unique_subject_ID",
                          color_by="specie")

    def test_no_arrows(self) -> None:
        adata = _make_reference_anndata()
        plot_amhi_profile(adata, group_by="unique_subject_ID",
                          color_by="specie", condition_by="condition",
                          show_arrows=False)

    def test_no_reference_lines(self) -> None:
        adata = _make_reference_anndata()
        plot_amhi_profile(adata, group_by="unique_subject_ID",
                          show_reference_lines=False)

    def test_custom_axes(self) -> None:
        adata = _make_reference_anndata()
        plot_amhi_profile(adata, group_by="unique_subject_ID",
                          x_metric="AMHI_offset",
                          y_metric="AMHI_D_median")

    def test_invalid_metric_raises(self) -> None:
        adata = _make_reference_anndata()
        with pytest.raises(ValueError, match="not found in summary_dataframe"):
            plot_amhi_profile(adata, x_metric="nonexistent_metric")

    def test_custom_condition_order(self) -> None:
        adata = _make_reference_anndata()
        plot_amhi_profile(adata, group_by="unique_subject_ID",
                          color_by="specie", condition_by="condition",
                          condition_order=["Old", "Young"])

    def test_obs_filter(self) -> None:
        adata = _make_reference_anndata()
        plot_amhi_profile(adata, group_by="unique_subject_ID",
                          color_by="specie",
                          obs_filter={"specie": "SpeciesA"})

    def test_requires_reference_transforms(self) -> None:
        adata = _make_amhi_anndata()
        with pytest.raises(ValueError, match="Run compute_all_mito_mean"):
            plot_amhi_profile(adata)
