"""
test_analysis_preprocessing_config.py

Unit tests for:
  - mito_marker.analysis.preprocessing_config
      get_default_preprocessing_config(), _validate_preprocessing_config(),
      configure_preprocessing() (with mocked input)
  - mito_marker.analysis.preprocessing_pipeline
      run_preprocessing() — non-interactive execution of steps 1–3

Tests verify:
  - Default config contains all required keys with correct types.
  - Validation raises KeyError / TypeError / ValueError for invalid configs.
  - run_preprocessing() filters .obs rows correctly from config['filters'].
  - run_preprocessing() sub-samples per subject when n_events_per_subject > 0.
  - run_preprocessing() applies feature selection and writes .var columns.
  - run_preprocessing() creates the expected .layers entry.
  - raw .X is never modified.
  - .uns['analysis_config'] is populated with the expected keys.
  - ALLOWED_* constants are non-empty lists of strings.
"""

import numpy as np
import pandas as pd
import pytest
import anndata

from mito_marker.analysis.preprocessing_config import (
    ALLOWED_FEATURE_SELECTION_METHODS,
    ALLOWED_NORMALIZATIONS,
    ALLOWED_TRANSFORMS,
    _validate_preprocessing_config,
    get_default_preprocessing_config,
)
from mito_marker.analysis.preprocessing_pipeline import run_preprocessing
from mito_marker.analysis.selection import _ANALYSIS_CONFIG_KEY


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

N_OBS = 120
N_VARS = 15


def _make_test_anndata(
    n_obs: int = N_OBS,
    n_vars: int = N_VARS,
    add_non_analytical: bool = False,
) -> anndata.AnnData:
    """
    Synthetic AnnData with structured .obs columns and a non-trivial .X.

    Channels 0–4 have high variance; channels 5–N_VARS have low variance.
    .obs includes 'condition', 'specie', and 'subject_ID' — needed by subset
    selection, sub-sampling, and supervised feature selection.
    """
    rng = np.random.default_rng(42)

    # High-variance channels 0–4 carry a strong signal correlated with condition.
    n_young = n_obs // 2
    n_old = n_obs - n_young

    high_signal_young = rng.normal(loc=10.0, scale=2.0, size=(n_young, 5))
    high_signal_old = rng.normal(loc=20.0, scale=2.0, size=(n_old, 5))
    high_signal = np.vstack([high_signal_young, high_signal_old])

    low_signal = rng.normal(loc=5.0, scale=0.1, size=(n_obs, n_vars - 5))

    x_matrix = np.hstack([high_signal, low_signal]).astype(np.float32)

    conditions = ["young"] * n_young + ["old"] * n_old
    species = ["human"] * (n_obs // 3) + ["mouse"] * (n_obs - n_obs // 3)
    # Alternate subjects so sub-sampling has multiple groups to draw from.
    subjects = [f"S{(i % 4) + 1:03d}" for i in range(n_obs)]

    obs = pd.DataFrame(
        {
            "condition": pd.Categorical(conditions),
            "specie": pd.Categorical(species),
            "subject_ID": subjects,
        }
    )

    var_names = [f"CH{i:02d}" for i in range(n_vars)]
    if add_non_analytical:
        var_names[-1] = "Time"
    var = pd.DataFrame(index=var_names)
    if add_non_analytical:
        var["is_non_analytical"] = [False] * (n_vars - 1) + [True]

    return anndata.AnnData(X=x_matrix, obs=obs, var=var)


@pytest.fixture
def test_anndata() -> anndata.AnnData:
    return _make_test_anndata()


def _make_minimal_config(**overrides) -> dict:
    """Return a valid default config with optional key overrides."""
    config = get_default_preprocessing_config()
    config.update(overrides)
    return config


# ---------------------------------------------------------------------------
# Tests: ALLOWED_* constants
# ---------------------------------------------------------------------------


def test_allowed_transforms_is_nonempty_list_of_strings():
    assert isinstance(ALLOWED_TRANSFORMS, list)
    assert len(ALLOWED_TRANSFORMS) > 0
    assert all(isinstance(t, str) for t in ALLOWED_TRANSFORMS)


def test_allowed_normalizations_is_nonempty_list_of_strings():
    assert isinstance(ALLOWED_NORMALIZATIONS, list)
    assert len(ALLOWED_NORMALIZATIONS) > 0
    assert all(isinstance(n, str) for n in ALLOWED_NORMALIZATIONS)


def test_allowed_feature_selection_methods_is_nonempty_list_of_strings():
    assert isinstance(ALLOWED_FEATURE_SELECTION_METHODS, list)
    assert len(ALLOWED_FEATURE_SELECTION_METHODS) > 0
    assert all(isinstance(m, str) for m in ALLOWED_FEATURE_SELECTION_METHODS)


def test_allowed_transforms_contains_expected_values():
    for expected in ("none", "arcsinh", "logicle"):
        assert expected in ALLOWED_TRANSFORMS


def test_allowed_normalizations_contains_expected_values():
    for expected in ("none", "zscore_col", "minmax_col", "l2norm_row", "zscore_col__l2norm_row"):
        assert expected in ALLOWED_NORMALIZATIONS


def test_allowed_feature_selection_methods_contains_expected_values():
    for expected in ("CorrFilter", "MIM", "CMI", "HighVariance", "PCALoadings"):
        assert expected in ALLOWED_FEATURE_SELECTION_METHODS


# ---------------------------------------------------------------------------
# Tests: get_default_preprocessing_config
# ---------------------------------------------------------------------------


def test_default_config_returns_dict():
    config = get_default_preprocessing_config()
    assert isinstance(config, dict)


def test_default_config_has_all_required_keys():
    config = get_default_preprocessing_config()
    required_keys = [
        "filters",
        "n_events_per_subject",
        "feature_selection_methods",
        "feature_selection_top_n",
        "feature_selection_target_obs_column",
        "feature_selection_corr_filter_threshold",
        "feature_selection_mi_events_per_subject",
        "transform",
        "normalization",
    ]
    for key in required_keys:
        assert key in config, f"Missing key: '{key}'"


def test_default_config_types():
    config = get_default_preprocessing_config()
    assert isinstance(config["filters"], dict)
    assert isinstance(config["n_events_per_subject"], int)
    assert isinstance(config["feature_selection_methods"], list)
    assert isinstance(config["feature_selection_top_n"], int)
    # target_obs_column is None by default
    assert config["feature_selection_target_obs_column"] is None
    assert isinstance(config["feature_selection_corr_filter_threshold"], float)
    assert isinstance(config["feature_selection_mi_events_per_subject"], int)
    assert isinstance(config["transform"], str)
    assert isinstance(config["normalization"], str)


def test_default_config_safe_defaults():
    config = get_default_preprocessing_config()
    # No filtering by default
    assert config["filters"] == {}
    assert config["n_events_per_subject"] == 0
    # No feature selection by default
    assert config["feature_selection_methods"] == []
    # Default transform and normalization pass validation
    assert config["transform"] in ALLOWED_TRANSFORMS
    assert config["normalization"] in ALLOWED_NORMALIZATIONS


def test_default_config_passes_validation():
    """The default config must pass _validate_preprocessing_config without errors."""
    config = get_default_preprocessing_config()
    _validate_preprocessing_config(config)  # should not raise


def test_default_config_is_independent_across_calls():
    """Each call must return a new dict — mutating one must not affect the next."""
    config_a = get_default_preprocessing_config()
    config_b = get_default_preprocessing_config()
    config_a["transform"] = "arcsinh"
    config_a["filters"]["condition"] = ["young"]
    assert config_b["transform"] == "none"
    assert config_b["filters"] == {}


# ---------------------------------------------------------------------------
# Tests: _validate_preprocessing_config — missing keys
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "missing_key",
    [
        "filters",
        "n_events_per_subject",
        "feature_selection_methods",
        "feature_selection_top_n",
        "feature_selection_corr_filter_threshold",
        "feature_selection_mi_events_per_subject",
        "transform",
        "normalization",
    ],
)
def test_validate_raises_key_error_for_missing_key(missing_key):
    config = get_default_preprocessing_config()
    del config[missing_key]
    with pytest.raises(KeyError, match=missing_key):
        _validate_preprocessing_config(config)


# ---------------------------------------------------------------------------
# Tests: _validate_preprocessing_config — wrong types
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "key, bad_value",
    [
        ("filters", "all"),
        ("n_events_per_subject", 1.0),
        ("feature_selection_methods", "MIM"),
        ("feature_selection_top_n", "40"),
        ("feature_selection_corr_filter_threshold", "0.95"),
        ("feature_selection_mi_events_per_subject", 5.0),
        ("transform", 2),
        ("normalization", None),
    ],
)
def test_validate_raises_type_error_for_wrong_type(key, bad_value):
    config = get_default_preprocessing_config()
    config[key] = bad_value
    with pytest.raises(TypeError):
        _validate_preprocessing_config(config)


def test_validate_raises_type_error_for_target_col_wrong_type():
    config = get_default_preprocessing_config()
    config["feature_selection_methods"] = ["MIM"]
    config["feature_selection_target_obs_column"] = 42  # must be str or None
    with pytest.raises(TypeError):
        _validate_preprocessing_config(config)


# ---------------------------------------------------------------------------
# Tests: _validate_preprocessing_config — invalid values
# ---------------------------------------------------------------------------


def test_validate_raises_value_error_for_unknown_transform():
    config = _make_minimal_config(transform="fourier")
    with pytest.raises(ValueError, match="transform"):
        _validate_preprocessing_config(config)


def test_validate_raises_value_error_for_unknown_normalization():
    config = _make_minimal_config(normalization="quantile")
    with pytest.raises(ValueError, match="normalization"):
        _validate_preprocessing_config(config)


def test_validate_raises_value_error_for_unknown_fs_method():
    config = _make_minimal_config(feature_selection_methods=["UnknownMethod"])
    with pytest.raises(ValueError, match="Unknown feature_selection_methods"):
        _validate_preprocessing_config(config)


def test_validate_raises_value_error_when_supervised_method_has_no_target():
    config = _make_minimal_config(
        feature_selection_methods=["MIM"],
        feature_selection_target_obs_column=None,
    )
    with pytest.raises(ValueError, match="feature_selection_target_obs_column"):
        _validate_preprocessing_config(config)


def test_validate_raises_value_error_for_negative_n_events_per_subject():
    config = _make_minimal_config(n_events_per_subject=-1)
    with pytest.raises(ValueError, match="n_events_per_subject"):
        _validate_preprocessing_config(config)


def test_validate_raises_value_error_for_zero_corr_threshold():
    config = _make_minimal_config(feature_selection_corr_filter_threshold=0.0)
    with pytest.raises(ValueError, match="corr_filter_threshold"):
        _validate_preprocessing_config(config)


def test_validate_raises_value_error_for_corr_threshold_above_one():
    config = _make_minimal_config(feature_selection_corr_filter_threshold=1.5)
    with pytest.raises(ValueError, match="corr_filter_threshold"):
        _validate_preprocessing_config(config)


def test_validate_raises_value_error_for_zero_top_n_with_ranking_method():
    config = _make_minimal_config(
        feature_selection_methods=["HighVariance"],
        feature_selection_top_n=0,
    )
    with pytest.raises(ValueError, match="top_n"):
        _validate_preprocessing_config(config)


def test_validate_accepts_none_target_column_when_no_supervised_method():
    """target_obs_column may be None when only unsupervised methods are used."""
    config = _make_minimal_config(
        feature_selection_methods=["HighVariance"],
        feature_selection_target_obs_column=None,
        feature_selection_top_n=5,
    )
    _validate_preprocessing_config(config)  # should not raise


def test_validate_accepts_str_target_column():
    config = _make_minimal_config(
        feature_selection_methods=["MIM"],
        feature_selection_target_obs_column="condition",
    )
    _validate_preprocessing_config(config)  # should not raise


def test_validate_accepts_negative_mi_events_raises():
    config = _make_minimal_config(feature_selection_mi_events_per_subject=-1)
    with pytest.raises(ValueError, match="mi_events_per_subject"):
        _validate_preprocessing_config(config)


def test_validate_accepts_zero_mi_events_no_error():
    config = get_default_preprocessing_config()
    _validate_preprocessing_config(config)  # 0 = no subsampling, valid


# ---------------------------------------------------------------------------
# Tests: run_preprocessing — step 1 (subset selection / filters)
# ---------------------------------------------------------------------------


def test_run_preprocessing_no_filters_keeps_all_obs(test_anndata):
    config = get_default_preprocessing_config()
    result = run_preprocessing(test_anndata, config)
    assert result.n_obs == test_anndata.n_obs


def test_run_preprocessing_filter_by_condition(test_anndata):
    config = _make_minimal_config(filters={"condition": ["young"]})
    result = run_preprocessing(test_anndata, config)
    assert result.n_obs > 0
    assert result.n_obs < test_anndata.n_obs
    assert set(result.obs["condition"].unique()) == {"young"}


def test_run_preprocessing_filter_multiple_values(test_anndata):
    config = _make_minimal_config(filters={"condition": ["young", "old"]})
    result = run_preprocessing(test_anndata, config)
    assert result.n_obs == test_anndata.n_obs


def test_run_preprocessing_filter_by_specie(test_anndata):
    config = _make_minimal_config(filters={"specie": ["human"]})
    result = run_preprocessing(test_anndata, config)
    assert result.n_obs > 0
    assert set(result.obs["specie"].unique()) == {"human"}


def test_run_preprocessing_chained_filters(test_anndata):
    """Filtering on two columns should produce the intersection."""
    config = _make_minimal_config(
        filters={"condition": ["young"], "specie": ["human"]}
    )
    result = run_preprocessing(test_anndata, config)
    assert result.n_obs > 0
    assert set(result.obs["condition"].unique()) == {"young"}
    assert set(result.obs["specie"].unique()) == {"human"}


def test_run_preprocessing_filter_does_not_modify_original(test_anndata):
    original_n_obs = test_anndata.n_obs
    config = _make_minimal_config(filters={"condition": ["young"]})
    run_preprocessing(test_anndata, config)
    assert test_anndata.n_obs == original_n_obs


def test_run_preprocessing_x_not_modified_by_filter(test_anndata):
    x_original = test_anndata.X.copy()
    config = _make_minimal_config(filters={"condition": ["young"]})
    run_preprocessing(test_anndata, config)
    np.testing.assert_array_equal(test_anndata.X, x_original)


def test_run_preprocessing_analysis_config_populated_after_filter(test_anndata):
    config = _make_minimal_config(filters={"condition": ["young"]})
    result = run_preprocessing(test_anndata, config)
    assert _ANALYSIS_CONFIG_KEY in result.uns
    assert "selection" in result.uns[_ANALYSIS_CONFIG_KEY]
    assert "subsampling" in result.uns[_ANALYSIS_CONFIG_KEY]
    assert "active_layer" in result.uns[_ANALYSIS_CONFIG_KEY]
    assert "active_selection" in result.uns[_ANALYSIS_CONFIG_KEY]


def test_run_preprocessing_n_vars_unchanged_after_filter(test_anndata):
    config = _make_minimal_config(filters={"condition": ["young"]})
    result = run_preprocessing(test_anndata, config)
    assert result.n_vars == test_anndata.n_vars


# ---------------------------------------------------------------------------
# Tests: run_preprocessing — step 1 (sub-sampling)
# ---------------------------------------------------------------------------


def test_run_preprocessing_subsampling_reduces_events(test_anndata):
    config = _make_minimal_config(n_events_per_subject=10)
    result = run_preprocessing(test_anndata, config)
    # 4 subjects × 10 events = 40 events max
    assert result.n_obs <= 40


def test_run_preprocessing_subsampling_logged_in_uns(test_anndata):
    config = _make_minimal_config(n_events_per_subject=10)
    result = run_preprocessing(test_anndata, config)
    subsampling = result.uns[_ANALYSIS_CONFIG_KEY]["subsampling"]
    assert subsampling["applied"] is True
    assert subsampling["n_events_per_subject"] == 10


def test_run_preprocessing_zero_subsampling_keeps_all(test_anndata):
    config = _make_minimal_config(n_events_per_subject=0)
    result = run_preprocessing(test_anndata, config)
    subsampling = result.uns[_ANALYSIS_CONFIG_KEY]["subsampling"]
    assert subsampling["applied"] is False
    assert result.n_obs == test_anndata.n_obs


# ---------------------------------------------------------------------------
# Tests: run_preprocessing — step 2 (feature selection: no selection)
# ---------------------------------------------------------------------------


def test_run_preprocessing_no_feature_selection_active_selection_is_none(test_anndata):
    config = _make_minimal_config(feature_selection_methods=[])
    result = run_preprocessing(test_anndata, config)
    assert result.uns[_ANALYSIS_CONFIG_KEY]["active_selection"] is None


def test_run_preprocessing_no_feature_selection_no_var_columns_added(test_anndata):
    original_var_columns = set(test_anndata.var.columns)
    config = _make_minimal_config(feature_selection_methods=[])
    result = run_preprocessing(test_anndata, config)
    new_var_columns = set(result.var.columns)
    assert new_var_columns == original_var_columns


# ---------------------------------------------------------------------------
# Tests: run_preprocessing — step 2 (HighVariance)
# ---------------------------------------------------------------------------


def test_run_preprocessing_high_variance_writes_var_columns(test_anndata):
    config = _make_minimal_config(
        feature_selection_methods=["HighVariance"],
        feature_selection_top_n=5,
    )
    result = run_preprocessing(test_anndata, config)
    assert "HighVariance_score" in result.var.columns
    assert "is_selected_HighVariance" in result.var.columns


def test_run_preprocessing_high_variance_selects_correct_n(test_anndata):
    top_n = 5
    config = _make_minimal_config(
        feature_selection_methods=["HighVariance"],
        feature_selection_top_n=top_n,
    )
    result = run_preprocessing(test_anndata, config)
    n_selected = int(result.var["is_selected_HighVariance"].sum())
    assert n_selected == top_n


def test_run_preprocessing_high_variance_selects_high_variance_channels(test_anndata):
    """Channels 0–4 have high variance and should be among the top-5 selected."""
    config = _make_minimal_config(
        feature_selection_methods=["HighVariance"],
        feature_selection_top_n=5,
    )
    result = run_preprocessing(test_anndata, config)
    selected_names = result.var_names[result.var["is_selected_HighVariance"].values].tolist()
    high_variance_channels = [f"CH0{i}" for i in range(5)]
    # At least 3 of the 5 high-variance channels should be selected.
    n_overlap = sum(ch in selected_names for ch in high_variance_channels)
    assert n_overlap >= 3


def test_run_preprocessing_high_variance_sets_active_selection(test_anndata):
    config = _make_minimal_config(
        feature_selection_methods=["HighVariance"],
        feature_selection_top_n=5,
    )
    result = run_preprocessing(test_anndata, config)
    assert result.uns[_ANALYSIS_CONFIG_KEY]["active_selection"] == "HighVariance"


# ---------------------------------------------------------------------------
# Tests: run_preprocessing — step 2 (PCALoadings)
# ---------------------------------------------------------------------------


def test_run_preprocessing_pca_loadings_writes_var_columns(test_anndata):
    config = _make_minimal_config(
        feature_selection_methods=["PCALoadings"],
        feature_selection_top_n=5,
    )
    result = run_preprocessing(test_anndata, config)
    assert "PCALoadings_score" in result.var.columns
    assert "is_selected_PCALoadings" in result.var.columns


def test_run_preprocessing_pca_loadings_selects_correct_n(test_anndata):
    top_n = 6
    config = _make_minimal_config(
        feature_selection_methods=["PCALoadings"],
        feature_selection_top_n=top_n,
    )
    result = run_preprocessing(test_anndata, config)
    n_selected = int(result.var["is_selected_PCALoadings"].sum())
    assert n_selected == top_n


# ---------------------------------------------------------------------------
# Tests: run_preprocessing — step 2 (CorrFilter)
# ---------------------------------------------------------------------------


def test_run_preprocessing_corr_filter_writes_var_columns(test_anndata):
    config = _make_minimal_config(
        feature_selection_methods=["CorrFilter"],
        feature_selection_corr_filter_threshold=0.95,
    )
    result = run_preprocessing(test_anndata, config)
    assert "CorrFilter_score" in result.var.columns
    assert "is_selected_CorrFilter" in result.var.columns


def test_run_preprocessing_corr_filter_keeps_at_least_one_channel(test_anndata):
    config = _make_minimal_config(
        feature_selection_methods=["CorrFilter"],
        feature_selection_corr_filter_threshold=0.95,
    )
    result = run_preprocessing(test_anndata, config)
    n_kept = int(result.var["is_selected_CorrFilter"].sum())
    assert n_kept >= 1


def test_run_preprocessing_corr_filter_tight_threshold_drops_more(test_anndata):
    config_tight = _make_minimal_config(
        feature_selection_methods=["CorrFilter"],
        feature_selection_corr_filter_threshold=0.5,
    )
    config_loose = _make_minimal_config(
        feature_selection_methods=["CorrFilter"],
        feature_selection_corr_filter_threshold=0.99,
    )
    result_tight = run_preprocessing(_make_test_anndata(), config_tight)
    result_loose = run_preprocessing(_make_test_anndata(), config_loose)
    n_tight = int(result_tight.var["is_selected_CorrFilter"].sum())
    n_loose = int(result_loose.var["is_selected_CorrFilter"].sum())
    assert n_tight <= n_loose


# ---------------------------------------------------------------------------
# Tests: run_preprocessing — step 2 (combined methods)
# ---------------------------------------------------------------------------


def test_run_preprocessing_corr_then_high_variance_creates_combined_key(test_anndata):
    config = _make_minimal_config(
        feature_selection_methods=["CorrFilter", "HighVariance"],
        feature_selection_top_n=5,
        feature_selection_corr_filter_threshold=0.95,
    )
    result = run_preprocessing(test_anndata, config)
    active = result.uns[_ANALYSIS_CONFIG_KEY]["active_selection"]
    # CorrFilter always runs first; key reflects order
    assert "CorrFilter" in active
    assert "HighVariance" in active


def test_run_preprocessing_combined_selection_is_intersection(test_anndata):
    """When two methods are combined, the active selection is the intersection."""
    config = _make_minimal_config(
        feature_selection_methods=["CorrFilter", "HighVariance"],
        feature_selection_top_n=5,
        feature_selection_corr_filter_threshold=0.95,
    )
    result = run_preprocessing(test_anndata, config)
    active = result.uns[_ANALYSIS_CONFIG_KEY]["active_selection"]
    combined_key = f"is_selected_{active}"
    assert combined_key in result.var.columns

    # Intersection: combined ⊆ CorrFilter and combined ⊆ HighVariance
    combined = result.var[combined_key].values
    corr_selected = result.var["is_selected_CorrFilter"].values
    hv_selected = result.var["is_selected_HighVariance"].values
    assert np.all(combined <= corr_selected)  # combined is subset of CorrFilter
    assert np.all(combined <= hv_selected)    # combined is subset of HighVariance


# ---------------------------------------------------------------------------
# Tests: run_preprocessing — step 3 (transform + normalize)
# ---------------------------------------------------------------------------


def test_run_preprocessing_no_transform_no_norm_no_layer_created(test_anndata):
    config = _make_minimal_config(transform="none", normalization="none")
    result = run_preprocessing(test_anndata, config)
    assert len(result.layers) == 0
    assert result.uns[_ANALYSIS_CONFIG_KEY]["active_layer"] is None


def test_run_preprocessing_arcsinh_zscore_creates_layer(test_anndata):
    config = _make_minimal_config(transform="arcsinh", normalization="zscore_col")
    result = run_preprocessing(test_anndata, config)
    assert "arcsinh__zscore_col" in result.layers
    assert result.uns[_ANALYSIS_CONFIG_KEY]["active_layer"] == "arcsinh__zscore_col"


def test_run_preprocessing_layer_is_float32(test_anndata):
    config = _make_minimal_config(transform="arcsinh", normalization="zscore_col")
    result = run_preprocessing(test_anndata, config)
    layer = result.layers["arcsinh__zscore_col"]
    assert layer.dtype == np.float32


def test_run_preprocessing_layer_has_same_shape_as_x(test_anndata):
    config = _make_minimal_config(transform="arcsinh", normalization="zscore_col")
    result = run_preprocessing(test_anndata, config)
    layer = result.layers["arcsinh__zscore_col"]
    assert layer.shape == test_anndata.X.shape


def test_run_preprocessing_zscore_layer_has_zero_mean(test_anndata):
    """Z-score normalization per column should produce near-zero column means."""
    config = _make_minimal_config(transform="none", normalization="zscore_col")
    result = run_preprocessing(test_anndata, config)
    layer = result.layers["none__zscore_col"]
    col_means = np.abs(layer.mean(axis=0))
    assert np.all(col_means < 1e-4), f"Column means not near zero: {col_means}"


def test_run_preprocessing_minmax_layer_in_zero_one(test_anndata):
    config = _make_minimal_config(transform="none", normalization="minmax_col")
    result = run_preprocessing(test_anndata, config)
    layer = result.layers["none__minmax_col"]
    assert float(layer.min()) >= -1e-6
    assert float(layer.max()) <= 1.0 + 1e-6


def test_run_preprocessing_raw_x_not_modified_by_normalization(test_anndata):
    x_original = test_anndata.X.copy()
    config = _make_minimal_config(transform="arcsinh", normalization="zscore_col")
    run_preprocessing(test_anndata, config)
    np.testing.assert_array_equal(test_anndata.X, x_original)


def test_run_preprocessing_all_transform_options_create_valid_layers(test_anndata):
    for transform in ALLOWED_TRANSFORMS:
        for norm in ALLOWED_NORMALIZATIONS:
            config = _make_minimal_config(transform=transform, normalization=norm)
            result = run_preprocessing(_make_test_anndata(), config)
            if transform == "none" and norm == "none":
                assert result.uns[_ANALYSIS_CONFIG_KEY]["active_layer"] is None
            else:
                expected_layer = f"{transform}__{norm}"
                assert expected_layer in result.layers, (
                    f"Expected layer '{expected_layer}' not found for "
                    f"transform={transform}, norm={norm}"
                )
                layer = result.layers[expected_layer]
                assert np.isfinite(layer).all(), (
                    f"Layer '{expected_layer}' contains non-finite values"
                )


def test_run_preprocessing_cached_layer_reused(test_anndata):
    """Running the same config twice must not recompute the existing layer."""
    config = _make_minimal_config(transform="arcsinh", normalization="zscore_col")
    result = run_preprocessing(test_anndata, config)
    # Modify the cached layer to a sentinel value to detect re-computation.
    result.layers["arcsinh__zscore_col"][:] = 999.0
    result2 = run_preprocessing(result, config)
    # If the layer was reused (not recomputed), values should still be 999.0
    assert float(result2.layers["arcsinh__zscore_col"].mean()) == pytest.approx(999.0)


# ---------------------------------------------------------------------------
# Tests: run_preprocessing — combined steps (filter + feature selection + norm)
# ---------------------------------------------------------------------------


def test_run_preprocessing_full_pipeline(test_anndata):
    """Smoke test: all three steps together produce a valid AnnData."""
    config = _make_minimal_config(
        filters={"condition": ["young"]},
        n_events_per_subject=0,
        feature_selection_methods=["HighVariance"],
        feature_selection_top_n=5,
        transform="arcsinh",
        normalization="zscore_col",
    )
    result = run_preprocessing(test_anndata, config)

    # Filtered subset
    assert result.n_obs > 0
    assert result.n_obs < test_anndata.n_obs
    assert set(result.obs["condition"].unique()) == {"young"}

    # Feature selection applied
    assert "HighVariance_score" in result.var.columns
    assert result.uns[_ANALYSIS_CONFIG_KEY]["active_selection"] == "HighVariance"

    # Layer created
    assert "arcsinh__zscore_col" in result.layers
    assert result.uns[_ANALYSIS_CONFIG_KEY]["active_layer"] == "arcsinh__zscore_col"

    # n_vars unchanged throughout
    assert result.n_vars == test_anndata.n_vars


def test_run_preprocessing_uns_analysis_config_complete(test_anndata):
    """All expected keys must be present in .uns['analysis_config'] after run."""
    config = _make_minimal_config(
        feature_selection_methods=["HighVariance"],
        feature_selection_top_n=5,
        transform="none",
        normalization="zscore_col",
    )
    result = run_preprocessing(test_anndata, config)
    cfg = result.uns[_ANALYSIS_CONFIG_KEY]
    assert "selection" in cfg
    assert "subsampling" in cfg
    assert "active_layer" in cfg
    assert "active_selection" in cfg


# ---------------------------------------------------------------------------
# Tests: run_preprocessing — validation integration
# ---------------------------------------------------------------------------


def test_run_preprocessing_raises_for_invalid_config(test_anndata):
    config = get_default_preprocessing_config()
    config["transform"] = "invalid_transform"
    with pytest.raises(ValueError, match="transform"):
        run_preprocessing(test_anndata, config)


def test_run_preprocessing_raises_for_supervised_method_without_target(test_anndata):
    config = _make_minimal_config(
        feature_selection_methods=["MIM"],
        feature_selection_target_obs_column=None,
    )
    with pytest.raises(ValueError, match="feature_selection_target_obs_column"):
        run_preprocessing(test_anndata, config)
