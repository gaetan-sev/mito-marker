"""
test_analysis_feature_selection.py

Unit tests for mito_marker.analysis.feature_selection.

Tests cover:
  - All four scoring methods (MIM, CMI, HighVariance, PCALoadings) on synthetic data.
  - .var column creation: score values and is_selected flags.
  - AnnData shape invariant: .n_vars never changes.
  - Multiple method runs coexist without overwriting each other's .var columns.
  - .uns['analysis_config']['active_selection'] is updated after each run.
  - get_selected_data_matrix() returns the right column subset.
  - Numerical target column is auto-binned.
  - Console interaction with mocked input().
"""

import numpy as np
import pandas as pd
import pytest

import anndata

from mito_marker.analysis.feature_selection import (
    _METHOD_CMI,
    _METHOD_CORR_FILTER,
    _METHOD_HIGH_VARIANCE,
    _METHOD_MIM,
    _METHOD_PCA_LOADINGS,
    _METHOD_SHAP_IMPORTANCE,
    _prepare_target_labels,
    _score_corr_filter,
    _score_high_variance,
    _score_mim,
    _score_pca_loadings,
    _score_shap_importance,
    _write_selection_to_var,
    get_selected_data_matrix,
    select_channels,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

N_OBS = 200
N_VARS = 20
TOP_N = 8


def _make_test_anndata(n_obs: int = N_OBS, n_vars: int = N_VARS) -> anndata.AnnData:
    """
    Synthetic AnnData with a structured X matrix and binary .obs labels.

    Channels 0-4 have high variance; channels 5-19 have low variance.
    This makes high-variance and MIM methods predictable in tests.
    """
    np.random.seed(0)

    # Channels 0-4: high-signal (large variance); 5-N_VARS: low-signal.
    x_high = np.random.randn(n_obs, 5) * 5.0
    x_low = np.random.randn(n_obs, n_vars - 5) * 0.1
    x_matrix = np.hstack([x_high, x_low]).astype(np.float32)

    # Binary label correlated with channel 0: useful for MIM / CMI.
    # Using a list comprehension avoids numpy fixed-width string truncation,
    # which silently corrupts labels when the replacement string is longer
    # than the original dtype width (e.g. "GroupA" > "False" → 5 chars).
    labels = np.array(
        ["GroupA" if x_matrix[i, 0] > 0 else "GroupB" for i in range(n_obs)]
    )

    obs = pd.DataFrame({
        "subject_ID": ["S001"] * (n_obs // 2) + ["S002"] * (n_obs // 2),
        "dilution": (["Diluted"] * (n_obs // 4) + ["Not_Diluted"] * (n_obs // 4)) * 2,
        "target_label": labels,
        "age": np.array([3.0] * (n_obs // 2) + [6.0] * (n_obs // 2), dtype=np.float32),
    })

    var = pd.DataFrame(
        {"channel_description": [f"Channel_{i}" for i in range(n_vars)]},
        index=[f"CH{i}" for i in range(n_vars)],
    )

    return anndata.AnnData(X=x_matrix, obs=obs, var=var)


@pytest.fixture
def test_anndata() -> anndata.AnnData:
    return _make_test_anndata()


def _make_input_sequence(responses: list) -> callable:
    iterator = iter(responses)

    def mock_input(prompt: str = "") -> str:
        return next(iterator)

    return mock_input


# ---------------------------------------------------------------------------
# Tests: scoring functions (no console interaction)
# ---------------------------------------------------------------------------


class TestScoringFunctions:
    """Unit tests for the four internal scoring functions."""

    def test_score_high_variance_returns_correct_shape(self, test_anndata):
        data = np.asarray(test_anndata.X)
        scores = _score_high_variance(data)
        assert scores.shape == (N_VARS,)

    def test_score_high_variance_high_signal_channels_score_highest(self, test_anndata):
        data = np.asarray(test_anndata.X)
        scores = _score_high_variance(data)
        top_5_indices = np.argsort(scores)[::-1][:5]
        # Channels 0-4 have variance ~25; channels 5-19 have variance ~0.01.
        assert set(top_5_indices) == {0, 1, 2, 3, 4}

    def test_score_mim_returns_correct_shape(self, test_anndata):
        data = np.asarray(test_anndata.X)
        scores = _score_mim(data, test_anndata.obs, "target_label")
        assert scores.shape == (N_VARS,)

    def test_score_mim_all_values_non_negative(self, test_anndata):
        data = np.asarray(test_anndata.X)
        scores = _score_mim(data, test_anndata.obs, "target_label")
        assert (scores >= 0).all()

    def test_score_mim_high_signal_channels_rank_higher(self, test_anndata):
        data = np.asarray(test_anndata.X)
        scores = _score_mim(data, test_anndata.obs, "target_label")
        # Channel 0 is directly correlated with the label → should have high MI.
        assert scores[0] > scores[-1]

    def test_score_pca_loadings_returns_correct_shape(self, test_anndata):
        data = np.asarray(test_anndata.X)
        scores = _score_pca_loadings(data, N_VARS)
        assert scores.shape == (N_VARS,)

    def test_score_pca_loadings_all_non_negative(self, test_anndata):
        data = np.asarray(test_anndata.X)
        scores = _score_pca_loadings(data, N_VARS)
        assert (scores >= 0).all()

    def test_score_pca_loadings_high_variance_channels_score_above_zero(self, test_anndata):
        data = np.asarray(test_anndata.X)
        scores = _score_pca_loadings(data, N_VARS)
        # High-signal channels (0-4) drive the top PCA components and receive
        # the largest weighted loading scores. Low-signal channels (5-19) should
        # all score lower than each high-signal channel.
        high_signal_scores = scores[:5]
        low_signal_scores = scores[5:]
        assert high_signal_scores.mean() > low_signal_scores.mean()


# ---------------------------------------------------------------------------
# Tests: _write_selection_to_var
# ---------------------------------------------------------------------------


class TestWriteSelectionToVar:
    """Unit tests for the .var column writing logic."""

    def test_score_column_created(self, test_anndata):
        scores = np.random.rand(N_VARS)
        _write_selection_to_var(test_anndata, _METHOD_MIM, scores, TOP_N)
        assert "MIM_score" in test_anndata.var.columns

    def test_is_selected_column_created(self, test_anndata):
        scores = np.random.rand(N_VARS)
        _write_selection_to_var(test_anndata, _METHOD_MIM, scores, TOP_N)
        assert "is_selected_MIM" in test_anndata.var.columns

    def test_exactly_top_n_channels_selected(self, test_anndata):
        scores = np.random.rand(N_VARS)
        _write_selection_to_var(test_anndata, _METHOD_MIM, scores, TOP_N)
        assert test_anndata.var["is_selected_MIM"].sum() == TOP_N

    def test_selected_channels_have_highest_scores(self, test_anndata):
        scores = np.arange(N_VARS, dtype=float)  # channel 19 has highest score
        _write_selection_to_var(test_anndata, _METHOD_HIGH_VARIANCE, scores, TOP_N)
        selected_mask = test_anndata.var[f"is_selected_{_METHOD_HIGH_VARIANCE}"]
        selected_scores = scores[selected_mask]
        unselected_scores = scores[~selected_mask]
        assert selected_scores.min() > unselected_scores.max()

    def test_two_methods_coexist_in_var(self, test_anndata):
        scores_mim = np.random.rand(N_VARS)
        scores_hv = np.random.rand(N_VARS)
        _write_selection_to_var(test_anndata, _METHOD_MIM, scores_mim, TOP_N)
        _write_selection_to_var(test_anndata, _METHOD_HIGH_VARIANCE, scores_hv, TOP_N)
        assert "MIM_score" in test_anndata.var.columns
        assert "is_selected_MIM" in test_anndata.var.columns
        assert f"{_METHOD_HIGH_VARIANCE}_score" in test_anndata.var.columns
        assert f"is_selected_{_METHOD_HIGH_VARIANCE}" in test_anndata.var.columns

    def test_score_column_dtype_is_float(self, test_anndata):
        scores = np.array([1.0] * N_VARS)
        _write_selection_to_var(test_anndata, _METHOD_MIM, scores, TOP_N)
        assert test_anndata.var["MIM_score"].dtype == np.float64

    def test_is_selected_column_dtype_is_bool(self, test_anndata):
        scores = np.random.rand(N_VARS)
        _write_selection_to_var(test_anndata, _METHOD_MIM, scores, TOP_N)
        assert test_anndata.var["is_selected_MIM"].dtype == bool


# ---------------------------------------------------------------------------
# Tests: _prepare_target_labels
# ---------------------------------------------------------------------------


class TestPrepareTargetLabels:
    """Unit tests for the target label preparation helper."""

    def test_categorical_column_returned_as_string_array(self, test_anndata):
        labels = _prepare_target_labels(test_anndata.obs, "dilution")
        assert labels.dtype.kind in ("U", "O")  # string or object
        assert set(labels) == {"Diluted", "Not_Diluted"}

    def test_float_column_is_binned_into_3_groups(self, test_anndata):
        labels = _prepare_target_labels(test_anndata.obs, "age")
        # age has 2 unique values but qcut with q=3 produces ≤3 groups
        assert len(set(labels)) <= 3

    def test_float_column_binned_labels_are_strings(self, test_anndata):
        labels = _prepare_target_labels(test_anndata.obs, "age")
        assert all(isinstance(label, str) for label in labels)


# ---------------------------------------------------------------------------
# Tests: get_selected_data_matrix
# ---------------------------------------------------------------------------


class TestGetSelectedDataMatrix:
    """Tests for the helper that returns only the selected channel columns."""

    def test_no_selection_returns_full_matrix(self, test_anndata):
        matrix = get_selected_data_matrix(test_anndata)
        assert matrix.shape == (N_OBS, N_VARS)

    def test_selection_reduces_columns(self, test_anndata):
        scores = np.random.rand(N_VARS)
        _write_selection_to_var(test_anndata, _METHOD_MIM, scores, TOP_N)
        test_anndata.uns[_ANALYSIS_CONFIG_KEY] = {"active_selection": _METHOD_MIM}
        matrix = get_selected_data_matrix(test_anndata)
        assert matrix.shape == (N_OBS, TOP_N)

    def test_selection_returns_correct_columns(self, test_anndata):
        # Give channel 0 the highest score so it is definitely selected.
        scores = np.zeros(N_VARS)
        scores[0] = 100.0
        _write_selection_to_var(test_anndata, _METHOD_MIM, scores, 1)
        test_anndata.uns[_ANALYSIS_CONFIG_KEY] = {"active_selection": _METHOD_MIM}
        matrix = get_selected_data_matrix(test_anndata)
        np.testing.assert_array_equal(matrix, np.asarray(test_anndata.X)[:, [0]])


# ---------------------------------------------------------------------------
# Tests: select_channels — full interactive flow
# ---------------------------------------------------------------------------

# Import needed for uns key constant check
from mito_marker.analysis.selection import _ANALYSIS_CONFIG_KEY


class TestSelectChannels:
    """Integration tests for select_channels() with mocked console input."""

    def _anndata_with_config(self) -> anndata.AnnData:
        """Build a test AnnData that already has analysis_config in .uns."""
        adata = _make_test_anndata()
        adata.uns[_ANALYSIS_CONFIG_KEY] = {
            "active_layer": None,
            "active_selection": None,
        }
        return adata

    def test_enter_skip_leaves_anndata_unchanged(self, monkeypatch):
        """Pressing Enter (empty input) keeps all channels and sets active_selection=None."""
        adata = self._anndata_with_config()
        monkeypatch.setattr("builtins.input", _make_input_sequence([""]))
        result = select_channels(adata)
        assert result.n_obs == N_OBS
        assert result.n_vars == N_VARS
        assert result.uns[_ANALYSIS_CONFIG_KEY]["active_selection"] is None

    def test_high_variance_does_not_change_shape(self, monkeypatch):
        adata = self._anndata_with_config()
        # method=4 (HighVariance), top_n=8
        monkeypatch.setattr(
            "builtins.input", _make_input_sequence(["4", str(TOP_N)])
        )
        result = select_channels(adata)
        assert result.n_obs == N_OBS
        assert result.n_vars == N_VARS

    def test_high_variance_writes_var_columns(self, monkeypatch):
        adata = self._anndata_with_config()
        monkeypatch.setattr(
            "builtins.input", _make_input_sequence(["4", str(TOP_N)])
        )
        result = select_channels(adata)
        assert f"{_METHOD_HIGH_VARIANCE}_score" in result.var.columns
        assert f"is_selected_{_METHOD_HIGH_VARIANCE}" in result.var.columns

    def test_high_variance_selects_exactly_top_n(self, monkeypatch):
        adata = self._anndata_with_config()
        monkeypatch.setattr(
            "builtins.input", _make_input_sequence(["4", str(TOP_N)])
        )
        result = select_channels(adata)
        assert result.var[f"is_selected_{_METHOD_HIGH_VARIANCE}"].sum() == TOP_N

    def test_high_variance_updates_active_selection(self, monkeypatch):
        adata = self._anndata_with_config()
        monkeypatch.setattr(
            "builtins.input", _make_input_sequence(["4", str(TOP_N)])
        )
        result = select_channels(adata)
        assert result.uns[_ANALYSIS_CONFIG_KEY]["active_selection"] == _METHOD_HIGH_VARIANCE

    def test_pca_loadings_writes_var_columns(self, monkeypatch):
        adata = self._anndata_with_config()
        monkeypatch.setattr(
            "builtins.input", _make_input_sequence(["5", str(TOP_N)])
        )
        result = select_channels(adata)
        assert f"{_METHOD_PCA_LOADINGS}_score" in result.var.columns
        assert f"is_selected_{_METHOD_PCA_LOADINGS}" in result.var.columns

    def test_mim_requires_target_and_writes_var_columns(self, monkeypatch):
        adata = self._anndata_with_config()
        # method=2 (MIM), target=1 (dilution — first filterable column present),
        # top_n=8, subsample=n
        monkeypatch.setattr(
            "builtins.input", _make_input_sequence(["2", "1", str(TOP_N), "n"])
        )
        result = select_channels(adata)
        assert f"{_METHOD_MIM}_score" in result.var.columns
        assert f"is_selected_{_METHOD_MIM}" in result.var.columns
        assert result.var[f"is_selected_{_METHOD_MIM}"].sum() == TOP_N

    def test_cmi_requires_target_and_writes_var_columns(self, monkeypatch):
        adata = self._anndata_with_config()
        # method=3 (CMI), target=1, top_n=8, subsample=n
        monkeypatch.setattr(
            "builtins.input", _make_input_sequence(["3", "1", str(TOP_N), "n"])
        )
        result = select_channels(adata)
        assert f"{_METHOD_CMI}_score" in result.var.columns
        assert f"is_selected_{_METHOD_CMI}" in result.var.columns
        assert result.var[f"is_selected_{_METHOD_CMI}"].sum() == TOP_N

    def test_two_sequential_methods_coexist_in_var(self, monkeypatch):
        adata = self._anndata_with_config()
        # First run: HighVariance
        monkeypatch.setattr(
            "builtins.input", _make_input_sequence(["4", str(TOP_N)])
        )
        select_channels(adata)
        # Second run: PCALoadings
        monkeypatch.setattr(
            "builtins.input", _make_input_sequence(["5", str(TOP_N)])
        )
        select_channels(adata)
        assert f"is_selected_{_METHOD_HIGH_VARIANCE}" in adata.var.columns
        assert f"is_selected_{_METHOD_PCA_LOADINGS}" in adata.var.columns

    def test_active_selection_updated_to_latest_method(self, monkeypatch):
        adata = self._anndata_with_config()
        monkeypatch.setattr(
            "builtins.input", _make_input_sequence(["4", str(TOP_N)])
        )
        select_channels(adata)
        monkeypatch.setattr(
            "builtins.input", _make_input_sequence(["5", str(TOP_N)])
        )
        select_channels(adata)
        assert adata.uns[_ANALYSIS_CONFIG_KEY]["active_selection"] == _METHOD_PCA_LOADINGS

    def test_invalid_method_then_valid_retries(self, monkeypatch):
        adata = self._anndata_with_config()
        # "9" is invalid, then "4" (HighVariance) is valid
        monkeypatch.setattr(
            "builtins.input", _make_input_sequence(["9", "4", str(TOP_N)])
        )
        result = select_channels(adata)
        assert f"is_selected_{_METHOD_HIGH_VARIANCE}" in result.var.columns

    def test_default_top_n_applied_when_empty_response(self, monkeypatch):
        adata = self._anndata_with_config()
        # Empty string for top_n triggers the default (min(40, n_vars)=20)
        monkeypatch.setattr(
            "builtins.input", _make_input_sequence(["4", ""])
        )
        result = select_channels(adata)
        expected_top_n = min(40, N_VARS)
        assert result.var[f"is_selected_{_METHOD_HIGH_VARIANCE}"].sum() == expected_top_n


# ---------------------------------------------------------------------------
# Tests: non-analytical channel exclusion (Time, FlowAI)
# ---------------------------------------------------------------------------


def _make_anndata_with_non_analytical_channels(
    n_obs: int = N_OBS,
    n_analytical: int = N_VARS,
) -> anndata.AnnData:
    """
    Synthetic AnnData that includes 'Time' and 'FlowAI' channels in .var.

    Channels 0-4 have high variance; channels 5-N_ANALYTICAL have low
    variance. Time and FlowAI are appended with artificially high values
    to verify they are never selected even when their raw score would be high.
    """
    np.random.seed(0)
    x_analytical = np.hstack([
        np.random.randn(n_obs, 5) * 5.0,
        np.random.randn(n_obs, n_analytical - 5) * 0.1,
    ])
    # Time and FlowAI columns: large values that would rank high if not excluded.
    x_non_analytical = np.column_stack([
        np.linspace(0, 10_000, n_obs),  # Time: monotonically increasing
        np.ones(n_obs) * 999.0,          # FlowAI: constant large value
    ])
    x_matrix = np.hstack([x_analytical, x_non_analytical]).astype(np.float32)

    labels = np.array(["GroupA" if x_matrix[i, 0] > 0 else "GroupB" for i in range(n_obs)])
    obs = pd.DataFrame({
        "subject_ID": ["S001"] * (n_obs // 2) + ["S002"] * (n_obs // 2),
        "dilution": (["Diluted"] * (n_obs // 4) + ["Not_Diluted"] * (n_obs // 4)) * 2,
        "target_label": labels,
    })
    var_names = [f"CH{i}" for i in range(n_analytical)] + ["Time", "FlowAI"]
    var = pd.DataFrame(
        {
            "channel_description": [f"Channel_{i}" for i in range(n_analytical)] + ["Time col", "FlowAI col"],
            # Mirror what anndata_builder stamps at ingestion time.
            "is_non_analytical": [False] * n_analytical + [True, True],
        },
        index=var_names,
    )
    adata = anndata.AnnData(X=x_matrix, obs=obs, var=var)
    adata.uns[_ANALYSIS_CONFIG_KEY] = {"active_layer": None, "active_selection": None}
    return adata


class TestNonAnalyticalChannelExclusion:
    """
    Tests that Time and FlowAI channels are excluded from feature selection
    scoring and are never marked as selected, regardless of their raw values.
    """

    def test_time_and_flowai_receive_zero_score(self, monkeypatch):
        adata = _make_anndata_with_non_analytical_channels()
        monkeypatch.setattr("builtins.input", _make_input_sequence(["4", str(N_VARS)]))
        select_channels(adata)
        assert adata.var.loc["Time", f"{_METHOD_HIGH_VARIANCE}_score"] == 0.0
        assert adata.var.loc["FlowAI", f"{_METHOD_HIGH_VARIANCE}_score"] == 0.0

    def test_time_and_flowai_are_never_selected(self, monkeypatch):
        adata = _make_anndata_with_non_analytical_channels()
        # Ask for all analytical channels — Time and FlowAI must still be excluded.
        monkeypatch.setattr("builtins.input", _make_input_sequence(["4", str(N_VARS)]))
        select_channels(adata)
        assert not adata.var.loc["Time", f"is_selected_{_METHOD_HIGH_VARIANCE}"]
        assert not adata.var.loc["FlowAI", f"is_selected_{_METHOD_HIGH_VARIANCE}"]

    def test_selected_count_counts_only_analytical_channels(self, monkeypatch):
        adata = _make_anndata_with_non_analytical_channels()
        top_n = 5
        monkeypatch.setattr("builtins.input", _make_input_sequence(["4", str(top_n)]))
        select_channels(adata)
        selected = adata.var[f"is_selected_{_METHOD_HIGH_VARIANCE}"]
        # Exactly top_n analytical channels selected, Time and FlowAI not counted.
        assert selected.sum() == top_n
        assert not selected["Time"]
        assert not selected["FlowAI"]

    def test_n_vars_unchanged_after_selection(self, monkeypatch):
        adata = _make_anndata_with_non_analytical_channels()
        n_vars_before = adata.n_vars
        monkeypatch.setattr("builtins.input", _make_input_sequence(["4", str(N_VARS)]))
        select_channels(adata)
        assert adata.n_vars == n_vars_before

    def test_get_selected_data_matrix_excludes_non_analytical_channels(self):
        """get_selected_data_matrix() must never include Time or FlowAI columns."""
        adata = _make_anndata_with_non_analytical_channels()
        n_analytical = N_VARS  # number of analytical channels (no Time/FlowAI)
        # Set high variance as active selection with all analytical channels selected.
        adata.var[f"is_selected_{_METHOD_HIGH_VARIANCE}"] = [True] * n_analytical + [False, False]
        adata.var[f"{_METHOD_HIGH_VARIANCE}_score"] = [1.0] * n_analytical + [0.0, 0.0]
        adata.uns[_ANALYSIS_CONFIG_KEY]["active_selection"] = _METHOD_HIGH_VARIANCE
        matrix = get_selected_data_matrix(adata)
        assert matrix.shape[1] == n_analytical

    def test_get_selected_data_matrix_excludes_non_analytical_when_no_selection(self):
        """With active_selection=None, Time and FlowAI are still excluded."""
        adata = _make_anndata_with_non_analytical_channels()
        # No feature selection applied — all channels would normally be returned.
        matrix = get_selected_data_matrix(adata)
        assert matrix.shape[1] == N_VARS  # only the analytical channels

    def test_tem_non_analytical_excluded_via_fallback_when_column_absent(self, monkeypatch):
        """
        TEM non-analytical features (Mito_CentroidX/Y, Mito_FeretX/Y) must be
        excluded from selection even when the is_non_analytical .var column is
        absent (fallback path — e.g. AnnData loaded from an older .h5ad file).
        """
        from mito_marker.controlled_vocabulary import TEM_FEATURE_COLUMNS, TEM_NON_ANALYTICAL_FEATURES

        non_analytical_set = set(TEM_NON_ANALYTICAL_FEATURES)
        # Build a TEM-like AnnData WITHOUT the is_non_analytical column.
        var = pd.DataFrame(
            {"feature_description": [""] * len(TEM_FEATURE_COLUMNS)},
            index=pd.Index(TEM_FEATURE_COLUMNS, name="feature_name"),
        )
        np.random.seed(0)
        n_obs = 100
        x_matrix = np.random.randn(n_obs, len(TEM_FEATURE_COLUMNS)).astype(np.float32)
        # Give CentroidX a very high variance so it would rank #1 if not excluded.
        x_matrix[:, TEM_FEATURE_COLUMNS.index("Mito_CentroidX")] *= 1000.0

        obs = pd.DataFrame({"age_group": ["Young"] * (n_obs // 2) + ["Old"] * (n_obs // 2)})
        adata = anndata.AnnData(X=x_matrix, obs=obs, var=var)
        adata.uns[_ANALYSIS_CONFIG_KEY] = {"active_layer": None, "active_selection": None}

        assert "is_non_analytical" not in adata.var.columns, "Precondition: column must be absent"

        top_n = 10
        monkeypatch.setattr("builtins.input", _make_input_sequence(["4", str(top_n)]))
        select_channels(adata)

        selected = adata.var_names[adata.var[f"is_selected_{_METHOD_HIGH_VARIANCE}"]].tolist()
        assert len(selected) == top_n
        for non_analytical_feature in TEM_NON_ANALYTICAL_FEATURES:
            assert non_analytical_feature not in selected, (
                f"TEM non-analytical feature '{non_analytical_feature}' must never be selected "
                f"even when is_non_analytical column is absent."
            )


# ---------------------------------------------------------------------------
# Tests: _prompt_target_column — derived columns (age_group, _tertiles, etc.)
# ---------------------------------------------------------------------------


def _make_anndata_with_derived_grouping() -> anndata.AnnData:
    """
    Synthetic AnnData where .obs contains a dynamically derived column
    (age_group) that is NOT in FILTERABLE_OBS_COLUMNS.

    This simulates the output of bin_obs_column() or split_obs_by_threshold().
    """
    from mito_marker.analysis.feature_selection import _ANALYSIS_CONFIG_KEY

    np.random.seed(7)
    n_obs = 60
    x_matrix = np.random.randn(n_obs, N_VARS).astype(np.float32)
    obs = pd.DataFrame({
        "subject_ID": ["S001"] * 30 + ["S002"] * 30,
        "dilution": (["Diluted"] * 15 + ["Not_Diluted"] * 15) * 2,
        # Dynamically derived column — mimics bin_obs_column() output.
        "age_group": (["young"] * 20 + ["old"] * 20 + ["young"] * 20)[:n_obs],
    })
    var = pd.DataFrame(index=[f"CH{i}" for i in range(N_VARS)])
    adata = anndata.AnnData(X=x_matrix, obs=obs, var=var)
    adata.uns[_ANALYSIS_CONFIG_KEY] = {"active_layer": None, "active_selection": None}
    return adata


def _get_target_candidates(adata: anndata.AnnData) -> list[str]:
    """
    Replicate the candidate-building logic from _prompt_target_column() so that
    tests can assert on the candidate list without triggering interactive input.
    """
    from mito_marker.controlled_vocabulary import FILTERABLE_OBS_COLUMNS
    from mito_marker.analysis.feature_selection import _NON_TARGET_OBS_COLUMNS

    seen: set[str] = set()
    candidates: list[str] = []
    for col in FILTERABLE_OBS_COLUMNS:
        if col in adata.obs.columns and col not in _NON_TARGET_OBS_COLUMNS:
            candidates.append(col)
            seen.add(col)
    for col in adata.obs.columns:
        if col not in seen and col not in _NON_TARGET_OBS_COLUMNS:
            if adata.obs[col].nunique() > 1:
                candidates.append(col)
    return candidates


class TestDerivedGroupingColumnsAsTarget:
    """
    Verify that columns produced by bin_obs_column() / split_obs_by_threshold()
    (e.g. age_group) appear in the supervised-target menu even though they are
    not declared in FILTERABLE_OBS_COLUMNS.
    """

    def test_derived_column_appears_in_candidate_list(self):
        """age_group must be offered as a supervised target."""
        adata = _make_anndata_with_derived_grouping()
        candidates = _get_target_candidates(adata)
        assert "age_group" in candidates, (
            "age_group must appear in the supervised target candidates "
            "even though it is not in FILTERABLE_OBS_COLUMNS."
        )

    def test_subject_id_excluded_from_candidates(self):
        """subject_ID must never appear as a supervised target."""
        adata = _make_anndata_with_derived_grouping()
        candidates = _get_target_candidates(adata)
        assert "subject_ID" not in candidates

    def test_filterable_columns_appear_before_derived_columns(self):
        """FILTERABLE_OBS_COLUMNS must come first in the menu for stable ordering."""
        adata = _make_anndata_with_derived_grouping()
        candidates = _get_target_candidates(adata)
        from mito_marker.controlled_vocabulary import FILTERABLE_OBS_COLUMNS
        from mito_marker.analysis.feature_selection import _NON_TARGET_OBS_COLUMNS

        filterable_in_obs = [
            c for c in FILTERABLE_OBS_COLUMNS
            if c in adata.obs.columns and c not in _NON_TARGET_OBS_COLUMNS
        ]
        # All declared filterable columns must appear before age_group.
        if "age_group" in candidates and filterable_in_obs:
            age_group_idx = candidates.index("age_group")
            for col in filterable_in_obs:
                assert candidates.index(col) < age_group_idx, (
                    f"Filterable column '{col}' must appear before derived 'age_group'."
                )

    def test_select_channels_mim_accepts_derived_column_as_target(self, monkeypatch):
        """End-to-end: MIM with age_group as target must complete without error."""
        adata = _make_anndata_with_derived_grouping()
        candidates = _get_target_candidates(adata)
        age_group_idx = candidates.index("age_group") + 1  # 1-based menu index

        # Inputs: method=MIM(2), target=age_group, top_n=TOP_N, no subsample(n)
        monkeypatch.setattr(
            "builtins.input",
            _make_input_sequence(["2", str(age_group_idx), str(TOP_N), "n"]),
        )
        result = select_channels(adata)
        assert f"is_selected_{_METHOD_MIM}" in result.var.columns
        assert result.var[f"is_selected_{_METHOD_MIM}"].sum() == TOP_N


# ---------------------------------------------------------------------------
# Tests: _score_corr_filter
# ---------------------------------------------------------------------------


class TestScoreCorrFilter:
    """Unit tests for the correlation filter scoring function."""

    def _make_correlated_matrix(self) -> tuple:
        """Build a matrix with known correlation structure.

        Channels 0 and 1 are perfectly correlated (r=1.0).
        Channel 2 is independent.
        Returns (matrix, var_names).
        """
        np.random.seed(5)
        n_obs = 100
        base = np.random.randn(n_obs)
        matrix = np.column_stack([
            base,          # CH0
            base,          # CH1 — identical to CH0, r=1.0
            np.random.randn(n_obs),  # CH2 — independent
        ]).astype(np.float32)
        var_names = ["CH0", "CH1", "CH2"]
        return matrix, var_names

    def test_returns_boolean_mask(self):
        matrix, var_names = self._make_correlated_matrix()
        mask = _score_corr_filter(matrix, var_names, threshold=0.95)
        assert mask.dtype == bool
        assert len(mask) == 3

    def test_drops_second_of_perfectly_correlated_pair(self):
        """CH0 is kept; CH1 (perfectly correlated with CH0) is dropped."""
        matrix, var_names = self._make_correlated_matrix()
        mask = _score_corr_filter(matrix, var_names, threshold=0.95)
        assert mask[0] is np.bool_(True)   # CH0 kept
        assert mask[1] is np.bool_(False)  # CH1 dropped
        assert mask[2] is np.bool_(True)   # CH2 kept

    def test_independent_channels_all_kept(self):
        """All channels with correlation < threshold are retained."""
        np.random.seed(99)
        matrix = np.random.randn(200, 5).astype(np.float32)
        var_names = [f"CH{i}" for i in range(5)]
        mask = _score_corr_filter(matrix, var_names, threshold=0.95)
        assert mask.all()

    def test_all_identical_columns_drops_all_but_one(self):
        """When all channels are identical, only the first is kept."""
        base = np.random.randn(50)
        matrix = np.tile(base[:, np.newaxis], (1, 4)).astype(np.float32)
        var_names = ["A", "B", "C", "D"]
        mask = _score_corr_filter(matrix, var_names, threshold=0.95)
        assert mask.sum() == 1
        assert mask[0]  # first column kept


# ---------------------------------------------------------------------------
# Tests: CorrFilter via select_channels()
# ---------------------------------------------------------------------------


class TestCorrFilterInSelectChannels:
    """Integration tests for CorrFilter method in select_channels()."""

    def _anndata_with_correlated_channels(self) -> anndata.AnnData:
        """AnnData with channels 0 and 1 perfectly correlated (r=1.0)."""
        np.random.seed(5)
        n_obs = 200
        base = np.random.randn(n_obs)
        x_high = np.random.randn(n_obs, 3) * 5.0
        x_matrix = np.column_stack([
            base,           # CH0
            base,           # CH1 — identical to CH0
            x_high,         # CH2-CH4: high variance, independent
            np.random.randn(n_obs, N_VARS - 5) * 0.1,  # CH5+: low variance
        ]).astype(np.float32)

        obs = pd.DataFrame({
            "subject_ID": ["S001"] * (n_obs // 2) + ["S002"] * (n_obs // 2),
        })
        var = pd.DataFrame(index=[f"CH{i}" for i in range(N_VARS)])
        adata = anndata.AnnData(X=x_matrix, obs=obs, var=var)
        adata.uns[_ANALYSIS_CONFIG_KEY] = {"active_layer": None, "active_selection": None}
        return adata

    def test_corr_filter_writes_var_columns(self, monkeypatch):
        """Running CorrFilter must create CorrFilter_score and is_selected_CorrFilter."""
        adata = self._anndata_with_correlated_channels()
        # method=1 (CorrFilter), threshold=95 (default via Enter)
        monkeypatch.setattr("builtins.input", _make_input_sequence(["1", ""]))
        result = select_channels(adata)
        assert "CorrFilter_score" in result.var.columns
        assert "is_selected_CorrFilter" in result.var.columns

    def test_corr_filter_removes_correlated_channel(self, monkeypatch):
        """CH1 (identical to CH0) must be marked as not selected after CorrFilter."""
        adata = self._anndata_with_correlated_channels()
        monkeypatch.setattr("builtins.input", _make_input_sequence(["1", ""]))
        result = select_channels(adata)
        # CH0 and CH1 are perfectly correlated → one must be dropped.
        selected = result.var["is_selected_CorrFilter"].values
        # Both CH0 and CH1 cannot both be selected.
        assert not (selected[0] and selected[1]), (
            "Both CH0 and CH1 are kept, but they are perfectly correlated. "
            "CorrFilter should drop one."
        )

    def test_corr_filter_active_selection_set(self, monkeypatch):
        """active_selection must be set to 'CorrFilter' after running it alone."""
        adata = self._anndata_with_correlated_channels()
        monkeypatch.setattr("builtins.input", _make_input_sequence(["1", ""]))
        result = select_channels(adata)
        assert result.uns[_ANALYSIS_CONFIG_KEY]["active_selection"] == _METHOD_CORR_FILTER

    def test_n_vars_unchanged_after_corr_filter(self, monkeypatch):
        """AnnData shape must not change when CorrFilter runs."""
        adata = self._anndata_with_correlated_channels()
        monkeypatch.setattr("builtins.input", _make_input_sequence(["1", ""]))
        result = select_channels(adata)
        assert result.n_vars == N_VARS


# ---------------------------------------------------------------------------
# Tests: multi-method feature selection pipeline
# ---------------------------------------------------------------------------


class TestMultiMethodSelection:
    """Tests for comma-separated multi-method selection in select_channels()."""

    def _anndata_with_config(self) -> anndata.AnnData:
        adata = _make_test_anndata()
        adata.uns[_ANALYSIS_CONFIG_KEY] = {
            "active_layer": None,
            "active_selection": None,
        }
        return adata

    def test_two_methods_both_var_columns_created(self, monkeypatch):
        """Selecting '1,4' (CorrFilter + HighVariance) creates both sets of .var columns."""
        adata = self._anndata_with_config()
        # method=1,4 (CorrFilter, HighVariance), threshold=Enter(95%), top_n=TOP_N
        monkeypatch.setattr(
            "builtins.input", _make_input_sequence(["1,4", "", str(TOP_N)])
        )
        result = select_channels(adata)
        assert "is_selected_CorrFilter" in result.var.columns
        assert f"is_selected_{_METHOD_HIGH_VARIANCE}" in result.var.columns

    def test_multi_method_pipeline_key_stored_as_active_selection(self, monkeypatch):
        """With two methods, active_selection must be a combined pipeline key."""
        adata = self._anndata_with_config()
        monkeypatch.setattr(
            "builtins.input", _make_input_sequence(["1,4", "", str(TOP_N)])
        )
        result = select_channels(adata)
        active = result.uns[_ANALYSIS_CONFIG_KEY]["active_selection"]
        # Pipeline key must include both method names.
        assert _METHOD_CORR_FILTER in active
        assert _METHOD_HIGH_VARIANCE in active

    def test_multi_method_intersection_stored_in_var(self, monkeypatch):
        """Combined pipeline mask must be stored in .var['is_selected_{pipeline_key}']."""
        adata = self._anndata_with_config()
        monkeypatch.setattr(
            "builtins.input", _make_input_sequence(["1,4", "", str(TOP_N)])
        )
        result = select_channels(adata)
        active = result.uns[_ANALYSIS_CONFIG_KEY]["active_selection"]
        assert f"is_selected_{active}" in result.var.columns

    def test_n_vars_unchanged_with_multi_method(self, monkeypatch):
        """AnnData shape must never change regardless of number of methods."""
        adata = self._anndata_with_config()
        monkeypatch.setattr(
            "builtins.input", _make_input_sequence(["4,5", str(TOP_N)])
        )
        result = select_channels(adata)
        assert result.n_vars == N_VARS

    def test_two_unsupervised_methods_intersection_at_most_top_n(self, monkeypatch):
        """Intersection of two methods selects at most TOP_N channels."""
        adata = self._anndata_with_config()
        monkeypatch.setattr(
            "builtins.input", _make_input_sequence(["4,5", str(TOP_N)])
        )
        result = select_channels(adata)
        active = result.uns[_ANALYSIS_CONFIG_KEY]["active_selection"]
        n_selected = result.var[f"is_selected_{active}"].sum()
        assert n_selected <= TOP_N


# ---------------------------------------------------------------------------
# Tests: _score_shap_importance
# ---------------------------------------------------------------------------


def _make_anndata_with_shap_values(n_obs: int = N_OBS, n_vars: int = N_VARS) -> anndata.AnnData:
    """
    Build a test AnnData with synthetic SHAP values stored in .uns['ml_results'].

    SHAP values are 2D (n_obs, n_vars) — binary / regression case.
    Channel 0 has the highest mean |SHAP|; channel n_vars-1 has the lowest.
    """
    np.random.seed(3)
    x_matrix = np.random.randn(n_obs, n_vars).astype(np.float32)
    obs = pd.DataFrame({
        "subject_ID": ["S001"] * (n_obs // 2) + ["S002"] * (n_obs // 2),
        "target_label": ["A"] * (n_obs // 2) + ["B"] * (n_obs // 2),
    })
    var = pd.DataFrame(index=[f"CH{i}" for i in range(n_vars)])
    adata = anndata.AnnData(X=x_matrix, obs=obs, var=var)
    adata.uns[_ANALYSIS_CONFIG_KEY] = {"active_layer": None, "active_selection": None}

    # Synthetic SHAP: channel 0 dominates with strictly decreasing importance.
    # Deterministic values (no randomness) so mean(|SHAP|) order is guaranteed.
    shap_matrix = np.zeros((n_obs, n_vars))
    for i in range(n_vars):
        shap_matrix[:, i] = float(n_vars - i)  # channel 0 = 20, channel 1 = 19, …

    adata.uns["ml_results"] = {
        "shap_values": shap_matrix,
        "shap_feature_names": [f"CH{i}" for i in range(n_vars)],
    }
    return adata


class TestScoreShapImportance:
    """Unit tests for the SHAP importance scoring function."""

    def test_returns_correct_shape(self):
        adata = _make_anndata_with_shap_values()
        scores = _score_shap_importance(adata)
        assert scores is not None
        assert scores.shape == (N_VARS,)

    def test_returns_none_when_no_ml_results(self):
        adata = _make_test_anndata()
        # No ml_results key at all.
        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            scores = _score_shap_importance(adata)
        assert scores is None
        assert len(w) >= 1

    def test_returns_none_when_shap_values_is_none(self):
        adata = _make_test_anndata()
        adata.uns["ml_results"] = {"shap_values": None, "shap_feature_names": None}
        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            scores = _score_shap_importance(adata)
        assert scores is None

    def test_channel_0_ranks_highest(self):
        """Channel 0 has the largest SHAP magnitude → must have the highest score."""
        adata = _make_anndata_with_shap_values()
        scores = _score_shap_importance(adata)
        assert scores[0] == scores.max()

    def test_all_scores_non_negative(self):
        """mean(|SHAP|) must always be ≥ 0."""
        adata = _make_anndata_with_shap_values()
        scores = _score_shap_importance(adata)
        assert (scores >= 0).all()

    def test_3d_shap_matrix_handled(self):
        """3D SHAP (multiclass) must return a 1D score array of length n_vars."""
        np.random.seed(11)
        n_obs, n_vars, n_classes = N_OBS, N_VARS, 3
        x_matrix = np.random.randn(n_obs, n_vars).astype(np.float32)
        obs = pd.DataFrame({"subject_ID": ["S001"] * n_obs})
        var = pd.DataFrame(index=[f"CH{i}" for i in range(n_vars)])
        adata = anndata.AnnData(X=x_matrix, obs=obs, var=var)
        adata.uns["ml_results"] = {
            "shap_values": np.random.randn(n_obs, n_vars, n_classes),
            "shap_feature_names": [f"CH{i}" for i in range(n_vars)],
        }
        scores = _score_shap_importance(adata)
        assert scores is not None
        assert scores.shape == (n_vars,)

    def test_unknown_feature_names_mapped_to_zero(self):
        """Features in shap_feature_names that are not in .var_names get score 0."""
        adata = _make_anndata_with_shap_values()
        # Replace one feature name with an unknown channel.
        adata.uns["ml_results"]["shap_feature_names"][0] = "UNKNOWN_CHANNEL"
        scores = _score_shap_importance(adata)
        # CH0 was mapped to UNKNOWN_CHANNEL, so its .var score should be 0.
        assert scores[0] == 0.0


class TestShapImportanceInSelectChannels:
    """Integration tests for SHAPImportance option in select_channels()."""

    def test_shap_importance_writes_var_columns(self, monkeypatch):
        """Selecting method [6] must create SHAPImportance_score and is_selected columns."""
        adata = _make_anndata_with_shap_values()
        # SHAP available, so [6] is shown. Input: "6", top_n=TOP_N
        monkeypatch.setattr(
            "builtins.input", _make_input_sequence(["6", str(TOP_N)])
        )
        result = select_channels(adata)
        assert f"{_METHOD_SHAP_IMPORTANCE}_score" in result.var.columns
        assert f"is_selected_{_METHOD_SHAP_IMPORTANCE}" in result.var.columns

    def test_shap_importance_selects_top_channels(self, monkeypatch):
        """SHAPImportance must select exactly TOP_N channels."""
        adata = _make_anndata_with_shap_values()
        monkeypatch.setattr(
            "builtins.input", _make_input_sequence(["6", str(TOP_N)])
        )
        result = select_channels(adata)
        assert result.var[f"is_selected_{_METHOD_SHAP_IMPORTANCE}"].sum() == TOP_N

    def test_shap_importance_active_selection_set(self, monkeypatch):
        """active_selection must be set to 'SHAPImportance' after running it alone."""
        adata = _make_anndata_with_shap_values()
        monkeypatch.setattr(
            "builtins.input", _make_input_sequence(["6", str(TOP_N)])
        )
        result = select_channels(adata)
        assert result.uns[_ANALYSIS_CONFIG_KEY]["active_selection"] == _METHOD_SHAP_IMPORTANCE

    def test_shap_option_not_shown_when_no_ml_results(self, monkeypatch):
        """Without ml_results, selecting [6] must be rejected as invalid."""
        adata = _make_test_anndata()
        adata.uns[_ANALYSIS_CONFIG_KEY] = {"active_layer": None, "active_selection": None}
        # [6] is invalid (not shown), second attempt is "" (skip).
        monkeypatch.setattr(
            "builtins.input", _make_input_sequence(["6", ""])
        )
        result = select_channels(adata)
        # Selection was skipped.
        assert result.uns[_ANALYSIS_CONFIG_KEY]["active_selection"] is None
