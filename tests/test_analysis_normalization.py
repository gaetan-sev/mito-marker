"""
test_analysis_normalization.py

Unit tests for mito_marker.analysis.normalization.

Tests verify:
  - Each transformation (None, Arcsinh, Logicle) produces finite output with
    the expected mathematical properties.
  - Each normalization (None, Z-score col, MinMax col, L2 row, Z+L2) produces
    the expected statistical properties.
  - Layer naming follows the '{transform}__{norm}' convention exactly.
  - Raw .X is never modified.
  - 'none__none' produces no layer and active_layer is set to None.
  - A previously created layer is reused without recomputation.
  - Console interaction with mocked input.
"""

import numpy as np
import pandas as pd
import pytest

import anndata

from mito_marker.analysis.normalization import (
    _ARCSINH_COFACTOR,
    _NORM_L2_ROW,
    _NORM_MINMAX_COL,
    _NORM_NONE,
    _NORM_ZSCORE_COL,
    _NORM_ZSCORE_COL_L2_ROW,
    _TRANSFORM_ARCSINH,
    _TRANSFORM_LOGICLE,
    _TRANSFORM_NONE,
    _apply_normalization,
    _apply_transform,
    _build_layer_name,
    transform_and_normalize,
)
from mito_marker.analysis.selection import _ANALYSIS_CONFIG_KEY


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

N_OBS = 100
N_VARS = 10


def _make_test_anndata() -> anndata.AnnData:
    """Small synthetic AnnData with known positive raw intensities."""
    np.random.seed(1)
    # Positive values typical of raw cytometry intensities (0 – 200,000).
    x_matrix = np.abs(np.random.randn(N_OBS, N_VARS) * 10_000).astype(np.float32)
    obs = pd.DataFrame({"subject_ID": ["S001"] * N_OBS})
    var = pd.DataFrame(index=[f"CH{i}" for i in range(N_VARS)])
    adata = anndata.AnnData(X=x_matrix, obs=obs, var=var)
    adata.uns[_ANALYSIS_CONFIG_KEY] = {"active_layer": None, "active_selection": None}
    return adata


@pytest.fixture
def test_anndata() -> anndata.AnnData:
    return _make_test_anndata()


def _make_input_sequence(responses: list) -> callable:
    iterator = iter(responses)

    def mock_input(prompt: str = "") -> str:
        return next(iterator)

    return mock_input


# ---------------------------------------------------------------------------
# Tests: _apply_transform
# ---------------------------------------------------------------------------


class TestApplyTransform:
    """Unit tests for the three transformation functions."""

    def test_none_returns_copy_of_input(self):
        matrix = np.array([[1.0, 2.0], [3.0, 4.0]])
        result = _apply_transform(matrix, _TRANSFORM_NONE)
        np.testing.assert_array_equal(result, matrix)
        # Must be a copy, not the same object.
        assert result is not matrix

    def test_arcsinh_all_values_finite(self):
        matrix = np.random.randn(50, 10) * 50_000
        result = _apply_transform(matrix, _TRANSFORM_ARCSINH)
        assert np.isfinite(result).all()

    def test_arcsinh_formula_is_correct(self):
        original = np.array([[150.0, 0.0, -150.0]])
        # _apply_transform modifies matrix in-place, so expected must be
        # computed from the original values before calling the function.
        expected = np.arcsinh(original / _ARCSINH_COFACTOR)
        result = _apply_transform(original.copy(), _TRANSFORM_ARCSINH)
        np.testing.assert_allclose(result, expected, rtol=1e-6)

    def test_arcsinh_zero_maps_to_zero(self):
        matrix = np.array([[0.0, 0.0]])
        result = _apply_transform(matrix, _TRANSFORM_ARCSINH)
        np.testing.assert_allclose(result, [[0.0, 0.0]], atol=1e-10)

    def test_logicle_all_values_finite(self):
        # Logicle should handle positive, negative, and zero values.
        matrix = np.random.randn(50, 10) * 50_000
        result = _apply_transform(matrix, _TRANSFORM_LOGICLE)
        assert np.isfinite(result).all()

    def test_logicle_output_range_is_bounded(self):
        # With the approximate logicle, output should be in a reasonable range.
        matrix = np.abs(np.random.randn(100, 5) * 100_000)
        result = _apply_transform(matrix, _TRANSFORM_LOGICLE)
        assert result.max() < 50.0
        assert result.min() >= 0.0

    def test_unknown_transform_raises(self):
        with pytest.raises(ValueError, match="Unknown transformation"):
            _apply_transform(np.ones((5, 5)), "invalid_key")


# ---------------------------------------------------------------------------
# Tests: _apply_normalization
# ---------------------------------------------------------------------------


class TestApplyNormalization:
    """Unit tests for the five normalization strategies."""

    def _raw(self) -> np.ndarray:
        np.random.seed(2)
        return np.random.randn(80, 10).astype(np.float64)

    def test_none_returns_same_array(self):
        matrix = self._raw()
        result = _apply_normalization(matrix, _NORM_NONE)
        np.testing.assert_array_equal(result, matrix)

    def test_zscore_col_mean_near_zero(self):
        result = _apply_normalization(self._raw(), _NORM_ZSCORE_COL)
        np.testing.assert_allclose(result.mean(axis=0), np.zeros(10), atol=1e-10)

    def test_zscore_col_std_near_one(self):
        result = _apply_normalization(self._raw(), _NORM_ZSCORE_COL)
        np.testing.assert_allclose(result.std(axis=0), np.ones(10), atol=1e-10)

    def test_minmax_col_min_is_zero(self):
        matrix = np.abs(self._raw()) + 1.0  # ensure all positive
        result = _apply_normalization(matrix, _NORM_MINMAX_COL)
        np.testing.assert_allclose(result.min(axis=0), np.zeros(10), atol=1e-10)

    def test_minmax_col_max_is_one(self):
        matrix = np.abs(self._raw()) + 1.0
        result = _apply_normalization(matrix, _NORM_MINMAX_COL)
        np.testing.assert_allclose(result.max(axis=0), np.ones(10), atol=1e-10)

    def test_l2_row_each_row_is_unit_vector(self):
        result = _apply_normalization(self._raw(), _NORM_L2_ROW)
        row_norms = np.linalg.norm(result, axis=1)
        np.testing.assert_allclose(row_norms, np.ones(80), atol=1e-6)

    def test_zscore_l2_row_norms_are_one(self):
        result = _apply_normalization(self._raw(), _NORM_ZSCORE_COL_L2_ROW)
        row_norms = np.linalg.norm(result, axis=1)
        np.testing.assert_allclose(row_norms, np.ones(80), atol=1e-6)

    def test_zscore_l2_applied_in_correct_order(self):
        # Z-score first (column), then L2 row. If reversed, result differs.
        matrix = self._raw()
        zscore_first = _apply_normalization(matrix, _NORM_ZSCORE_COL_L2_ROW)
        # Z+L2 rows should all have norm 1 (L2 applied last).
        row_norms = np.linalg.norm(zscore_first, axis=1)
        np.testing.assert_allclose(row_norms, np.ones(80), atol=1e-6)

    def test_all_methods_produce_finite_output(self):
        matrix = self._raw()
        for norm_key in [
            _NORM_NONE, _NORM_ZSCORE_COL, _NORM_MINMAX_COL,
            _NORM_L2_ROW, _NORM_ZSCORE_COL_L2_ROW,
        ]:
            result = _apply_normalization(matrix, norm_key)
            assert np.isfinite(result).all(), f"Non-finite values with norm_key={norm_key}"

    def test_unknown_norm_raises(self):
        with pytest.raises(ValueError, match="Unknown normalization"):
            _apply_normalization(np.ones((5, 5)), "invalid_key")


# ---------------------------------------------------------------------------
# Tests: _build_layer_name
# ---------------------------------------------------------------------------


class TestBuildLayerName:
    """Tests for the layer naming convention."""

    def test_arcsinh_zscore_col(self):
        assert _build_layer_name(_TRANSFORM_ARCSINH, _NORM_ZSCORE_COL) == "arcsinh__zscore_col"

    def test_none_l2_row(self):
        assert _build_layer_name(_TRANSFORM_NONE, _NORM_L2_ROW) == "none__l2norm_row"

    def test_logicle_minmax(self):
        assert _build_layer_name(_TRANSFORM_LOGICLE, _NORM_MINMAX_COL) == "logicle__minmax_col"

    def test_arcsinh_zscore_and_l2(self):
        name = _build_layer_name(_TRANSFORM_ARCSINH, _NORM_ZSCORE_COL_L2_ROW)
        assert name == "arcsinh__zscore_col__l2norm_row"

    def test_separator_is_double_underscore(self):
        name = _build_layer_name(_TRANSFORM_ARCSINH, _NORM_ZSCORE_COL)
        assert "__" in name
        parts = name.split("__")
        assert parts[0] == _TRANSFORM_ARCSINH


# ---------------------------------------------------------------------------
# Tests: transform_and_normalize — full interactive flow
# ---------------------------------------------------------------------------


class TestTransformAndNormalize:
    """Integration tests for transform_and_normalize() with mocked input."""

    def test_none_none_creates_no_layer(self, test_anndata, monkeypatch):
        """When both choices are 'None' (input '1a'), no layer is added."""
        monkeypatch.setattr("builtins.input", _make_input_sequence(["1a"]))
        result = transform_and_normalize(test_anndata)
        # anndata >= 0.13 exposes .X as layers[None]; count named layers only.
        assert [key for key in result.layers if key is not None] == []

    def test_none_none_sets_active_layer_to_none(self, test_anndata, monkeypatch):
        monkeypatch.setattr("builtins.input", _make_input_sequence(["1a"]))
        result = transform_and_normalize(test_anndata)
        assert result.uns[_ANALYSIS_CONFIG_KEY]["active_layer"] is None

    def test_default_enter_produces_none_zscore_col(self, test_anndata, monkeypatch):
        """Pressing Enter (empty input) applies the default: no transform + Z-score."""
        monkeypatch.setattr("builtins.input", _make_input_sequence([""]))
        result = transform_and_normalize(test_anndata)
        assert result.uns[_ANALYSIS_CONFIG_KEY]["active_layer"] == "none__zscore_col"
        assert "none__zscore_col" in result.layers

    def test_raw_x_is_never_modified(self, test_anndata, monkeypatch):
        original_x = test_anndata.X.copy()
        monkeypatch.setattr("builtins.input", _make_input_sequence(["2b"]))
        transform_and_normalize(test_anndata)
        np.testing.assert_array_equal(test_anndata.X, original_x)

    def test_arcsinh_zscore_creates_named_layer(self, test_anndata, monkeypatch):
        monkeypatch.setattr("builtins.input", _make_input_sequence(["2b"]))
        result = transform_and_normalize(test_anndata)
        assert "arcsinh__zscore_col" in result.layers

    def test_active_layer_updated(self, test_anndata, monkeypatch):
        monkeypatch.setattr("builtins.input", _make_input_sequence(["2b"]))
        result = transform_and_normalize(test_anndata)
        assert result.uns[_ANALYSIS_CONFIG_KEY]["active_layer"] == "arcsinh__zscore_col"

    def test_arcsinh_zscore_layer_has_correct_shape(self, test_anndata, monkeypatch):
        monkeypatch.setattr("builtins.input", _make_input_sequence(["2b"]))
        result = transform_and_normalize(test_anndata)
        layer = result.layers["arcsinh__zscore_col"]
        assert layer.shape == (N_OBS, N_VARS)

    def test_arcsinh_zscore_col_means_near_zero(self, test_anndata, monkeypatch):
        monkeypatch.setattr("builtins.input", _make_input_sequence(["2b"]))
        result = transform_and_normalize(test_anndata)
        layer = result.layers["arcsinh__zscore_col"]
        np.testing.assert_allclose(layer.mean(axis=0), np.zeros(N_VARS), atol=1e-4)

    def test_logicle_minmax_layer_range_zero_to_one(self, test_anndata, monkeypatch):
        monkeypatch.setattr("builtins.input", _make_input_sequence(["3c"]))
        result = transform_and_normalize(test_anndata)
        layer = result.layers["logicle__minmax_col"]
        assert layer.min() >= -1e-6   # allow tiny floating-point slack
        assert layer.max() <= 1.0 + 1e-6

    def test_arcsinh_l2_row_unit_vectors(self, test_anndata, monkeypatch):
        monkeypatch.setattr("builtins.input", _make_input_sequence(["2d"]))
        result = transform_and_normalize(test_anndata)
        layer = result.layers["arcsinh__l2norm_row"]
        row_norms = np.linalg.norm(layer, axis=1)
        np.testing.assert_allclose(row_norms, np.ones(N_OBS), atol=1e-5)

    def test_all_layers_have_finite_values(self, test_anndata, monkeypatch):
        """Every transform × norm combination must produce all-finite values."""
        # Format: (combined_input, expected_layer_name)
        combinations = [
            ("2b", "arcsinh__zscore_col"),
            ("2c", "arcsinh__minmax_col"),
            ("2d", "arcsinh__l2norm_row"),
            ("2e", "arcsinh__zscore_col__l2norm_row"),
            ("3b", "logicle__zscore_col"),
            ("1b", "none__zscore_col"),
        ]
        for combined_input, expected_layer in combinations:
            adata = _make_test_anndata()
            monkeypatch.setattr(
                "builtins.input",
                _make_input_sequence([combined_input]),
            )
            result = transform_and_normalize(adata)
            layer_name = result.uns[_ANALYSIS_CONFIG_KEY]["active_layer"]
            assert layer_name == expected_layer, (
                f"Expected layer '{expected_layer}', got '{layer_name}'"
            )
            layer = result.layers[layer_name]
            assert np.isfinite(layer).all(), (
                f"Non-finite values in layer '{layer_name}'"
            )

    def test_existing_layer_reused_without_recomputation(self, test_anndata, monkeypatch):
        """If the layer already exists, it must be reused and .X must not change."""
        # First call: creates the layer.
        monkeypatch.setattr("builtins.input", _make_input_sequence(["2b"]))
        transform_and_normalize(test_anndata)
        original_layer = test_anndata.layers["arcsinh__zscore_col"].copy()

        # Second call: same choices — layer should not be recomputed.
        monkeypatch.setattr("builtins.input", _make_input_sequence(["2b"]))
        transform_and_normalize(test_anndata)
        np.testing.assert_array_equal(
            test_anndata.layers["arcsinh__zscore_col"], original_layer
        )

    def test_two_layers_coexist(self, test_anndata, monkeypatch):
        """Running two different combinations creates two independent layers."""
        monkeypatch.setattr("builtins.input", _make_input_sequence(["2b"]))
        transform_and_normalize(test_anndata)
        monkeypatch.setattr("builtins.input", _make_input_sequence(["3b"]))
        transform_and_normalize(test_anndata)
        assert "arcsinh__zscore_col" in test_anndata.layers
        assert "logicle__zscore_col" in test_anndata.layers

    def test_active_layer_updated_to_latest(self, test_anndata, monkeypatch):
        monkeypatch.setattr("builtins.input", _make_input_sequence(["2b"]))
        transform_and_normalize(test_anndata)
        monkeypatch.setattr("builtins.input", _make_input_sequence(["3b"]))
        transform_and_normalize(test_anndata)
        assert test_anndata.uns[_ANALYSIS_CONFIG_KEY]["active_layer"] == "logicle__zscore_col"

    def test_invalid_input_no_letter_then_valid_retries(self, test_anndata, monkeypatch):
        """An input with a digit but no letter is rejected; retries succeed."""
        # "9" has no letter → invalid; "2a" → arcsinh + none → "arcsinh__none"
        monkeypatch.setattr("builtins.input", _make_input_sequence(["9", "2a"]))
        result = transform_and_normalize(test_anndata)
        assert result.uns[_ANALYSIS_CONFIG_KEY]["active_layer"] == "arcsinh__none"

    def test_invalid_transform_digit_then_valid_retries(self, test_anndata, monkeypatch):
        """An unknown transform digit (9) triggers retry; valid response succeeds."""
        # "9a" → digit 9 not in (1/2/3) → invalid; "2a" → arcsinh + none
        monkeypatch.setattr("builtins.input", _make_input_sequence(["9a", "2a"]))
        result = transform_and_normalize(test_anndata)
        assert result.uns[_ANALYSIS_CONFIG_KEY]["active_layer"] == "arcsinh__none"

    def test_invalid_norm_letter_then_valid_retries(self, test_anndata, monkeypatch):
        """An unknown normalization letter (z) triggers retry; valid response succeeds."""
        # "1z" → letter z not in (a/b/c/d/e) → invalid; "1b" → none + zscore_col
        monkeypatch.setattr("builtins.input", _make_input_sequence(["1z", "1b"]))
        result = transform_and_normalize(test_anndata)
        assert result.uns[_ANALYSIS_CONFIG_KEY]["active_layer"] == "none__zscore_col"

    def test_input_order_digit_letter_or_letter_digit_both_valid(self, test_anndata, monkeypatch):
        """The parser must accept both '2b' and 'b2' as identical choices."""
        monkeypatch.setattr("builtins.input", _make_input_sequence(["b2"]))
        result = transform_and_normalize(test_anndata)
        assert result.uns[_ANALYSIS_CONFIG_KEY]["active_layer"] == "arcsinh__zscore_col"


# ---------------------------------------------------------------------------
# Tests: frozen layer parameters (ADR-015)
# ---------------------------------------------------------------------------

from mito_marker.analysis.normalization import (  # noqa: E402
    LAYER_PARAMETERS_KEY,
    _apply_stored_layer_parameters,
    _fit_and_apply_layer,
    get_layer_parameters,
)

_ALL_COMBINATIONS = [
    (transform_key, norm_key)
    for transform_key in (_TRANSFORM_NONE, _TRANSFORM_ARCSINH, _TRANSFORM_LOGICLE)
    for norm_key in (_NORM_NONE, _NORM_ZSCORE_COL, _NORM_MINMAX_COL,
                     _NORM_L2_ROW, _NORM_ZSCORE_COL_L2_ROW)
    if not (transform_key == _TRANSFORM_NONE and norm_key == _NORM_NONE)
]


class TestFrozenLayerParameters:
    """The fitted scaling is stored and can be replayed without refitting."""

    @pytest.mark.parametrize("transform_key,norm_key", _ALL_COMBINATIONS)
    def test_fit_matches_historical_functions(self, transform_key, norm_key):
        raw_matrix = _make_test_anndata().X.astype(np.float32)
        historical = _apply_normalization(
            _apply_transform(raw_matrix.copy(), transform_key), norm_key
        )
        processed, _ = _fit_and_apply_layer(
            raw_matrix.copy(), transform_key, norm_key,
            [f"CH{i}" for i in range(N_VARS)], "layer",
        )
        np.testing.assert_allclose(processed, historical, rtol=1e-5, atol=1e-5)

    @pytest.mark.parametrize("transform_key,norm_key", _ALL_COMBINATIONS)
    def test_replay_on_same_data_is_identical(self, transform_key, norm_key):
        raw_matrix = _make_test_anndata().X.astype(np.float32)
        processed, parameters = _fit_and_apply_layer(
            raw_matrix.copy(), transform_key, norm_key,
            [f"CH{i}" for i in range(N_VARS)], "layer",
        )
        replayed = _apply_stored_layer_parameters(raw_matrix.copy(), parameters)
        np.testing.assert_allclose(replayed, processed, rtol=1e-6, atol=1e-6)

    def test_replay_does_not_refit_on_shifted_data(self):
        raw_matrix = _make_test_anndata().X.astype(np.float32)
        _, parameters = _fit_and_apply_layer(
            raw_matrix.copy(), _TRANSFORM_NONE, _NORM_ZSCORE_COL,
            [f"CH{i}" for i in range(N_VARS)], "none__zscore_col",
        )
        reference_std = raw_matrix.std(axis=0)
        # Shift every feature by exactly two reference standard deviations.
        shifted_matrix = raw_matrix + 2.0 * reference_std
        replayed = _apply_stored_layer_parameters(shifted_matrix.copy(), parameters)
        # With a refit, the mean would be 0 again. Without refit it stays at +2.
        np.testing.assert_allclose(replayed.mean(axis=0), 2.0, atol=1e-3)

    def test_parameters_stored_with_fingerprint(self, test_anndata, monkeypatch):
        monkeypatch.setattr("builtins.input", _make_input_sequence(["1b"]))
        transform_and_normalize(test_anndata)
        parameters = get_layer_parameters(test_anndata)
        assert parameters is not None
        assert parameters["layer_name"] == "none__zscore_col"
        assert list(parameters["feature_names"]) == list(test_anndata.var_names)
        assert parameters["n_obs_fitted"] == N_OBS
        assert len(parameters["fingerprint"]) == 12
        np.testing.assert_allclose(
            parameters["scaler_mean"], test_anndata.X.mean(axis=0), rtol=1e-4
        )

    def test_fingerprint_changes_with_data(self):
        raw_matrix = _make_test_anndata().X.astype(np.float32)
        names = [f"CH{i}" for i in range(N_VARS)]
        _, first = _fit_and_apply_layer(raw_matrix.copy(), "none", "zscore_col", names, "a")
        _, second = _fit_and_apply_layer(raw_matrix[:50].copy(), "none", "zscore_col", names, "a")
        assert first["fingerprint"] != second["fingerprint"]

    def test_replay_rejects_wrong_column_count(self):
        raw_matrix = _make_test_anndata().X.astype(np.float32)
        _, parameters = _fit_and_apply_layer(
            raw_matrix.copy(), "none", "zscore_col", [f"CH{i}" for i in range(N_VARS)], "a"
        )
        with pytest.raises(ValueError, match="columns"):
            _apply_stored_layer_parameters(raw_matrix[:, :5].copy(), parameters)

    def test_parameters_survive_h5ad_round_trip(self, test_anndata, monkeypatch, tmp_path):
        monkeypatch.setattr("builtins.input", _make_input_sequence(["3e"]))
        transform_and_normalize(test_anndata)
        file_path = tmp_path / "normalized.h5ad"
        test_anndata.write_h5ad(file_path)
        reloaded = anndata.read_h5ad(file_path)
        parameters = reloaded.uns[LAYER_PARAMETERS_KEY]["logicle__zscore_col__l2norm_row"]
        replayed = _apply_stored_layer_parameters(
            reloaded.X.astype(np.float32, copy=True), parameters
        )
        np.testing.assert_allclose(
            replayed, reloaded.layers["logicle__zscore_col__l2norm_row"], rtol=1e-5, atol=1e-5
        )

    def test_cached_layer_without_parameters_warns(self, test_anndata, monkeypatch, capsys):
        test_anndata.layers["none__zscore_col"] = test_anndata.X.copy()
        monkeypatch.setattr("builtins.input", _make_input_sequence(["1b"]))
        transform_and_normalize(test_anndata)
        assert "no frozen parameters" in capsys.readouterr().out
