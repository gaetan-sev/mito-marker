"""
normalization.py

Cytometry data transformation and normalization pipeline.

The cytometry standard is:
  1. TRANSFORM first (Arcsinh or Logicle) — compresses extreme intensity values
     and handles the bimodal distributions typical of flow cytometry data.
  2. NORMALIZE second (Z-score / MinMax / L2 Norm) — scales channels for
     comparability across experiments.

These are two independent choices presented in two sequential console menus.

Results are stored in .layers with an explicit double-underscore-separated name:
  .layers['arcsinh__zscore_col']          → Arcsinh then Z-score per column
  .layers['arcsinh__minmax_col']          → Arcsinh then Min-Max per column
  .layers['arcsinh__l2norm_row']          → Arcsinh then L2 norm per row
  .layers['arcsinh__zscore_col__l2norm_row']  → Arcsinh then Z-score then L2
  .layers['logicle__zscore_col']          → Logicle then Z-score per column
  .layers['none__zscore_col']             → Z-score without prior transformation
  .layers['arcsinh__none']                → Arcsinh only, no normalization

Special case: when both choices are "None", no layer is created and .X is used
directly. .uns['analysis_config']['active_layer'] is set to None.

Frozen parameters: every fitted normalization (z-score means/standard
deviations, min-max bounds, logicle channel scale) is also recorded in
.uns['layer_parameters'][<layer_name>] with a short fingerprint. Replaying
these parameters on another dataset (_apply_stored_layer_parameters) scales it
in the SAME space as the reference dataset — required to re-apply a fitted
cluster model (analysis/cluster_model.py, ADR-015).

Typical usage:
    from mito_marker.analysis import transform_and_normalize
    sfc_subset = transform_and_normalize(sfc_subset)
"""

from typing import Any, Dict, List, Optional, Tuple

import anndata
import numpy as np
from sklearn.preprocessing import MinMaxScaler, Normalizer, StandardScaler

from mito_marker.analysis._fingerprint import compute_fingerprint
from mito_marker.analysis.selection import _ANALYSIS_CONFIG_KEY

# .uns key holding the frozen parameters of every normalized layer:
# .uns['layer_parameters'][<layer_name>] = {...}. See _fit_and_apply_layer().
LAYER_PARAMETERS_KEY = "layer_parameters"

# Arcsinh cofactor standard for spectral flow cytometry.
# Dividing raw intensities by this value before arcsinh linearises the
# scale around zero without compressing low-intensity populations.
_ARCSINH_COFACTOR = 150.0

# Transformation and normalization key tokens used in layer names.
_TRANSFORM_NONE = "none"
_TRANSFORM_ARCSINH = "arcsinh"
_TRANSFORM_LOGICLE = "logicle"

_NORM_NONE = "none"
_NORM_ZSCORE_COL = "zscore_col"
_NORM_MINMAX_COL = "minmax_col"
_NORM_L2_ROW = "l2norm_row"
_NORM_ZSCORE_COL_L2_ROW = "zscore_col__l2norm_row"


def transform_and_normalize(anndata_object: anndata.AnnData) -> anndata.AnnData:
    """
    Interactively apply a transformation and normalization to an SFC AnnData.

    Presents a single combined console menu for both choices:
      Transform (1–3): None | Arcsinh | Logicle
      Norm (a–e):      None | Z-score col | MinMax col | L2 row | Z+L2
    Default (press Enter): no transform + Z-score per column.

    The processed matrix is stored in a named .layers entry whose key encodes
    both choices (e.g. 'arcsinh__zscore_col'). Raw .X is never modified.

    When both choices are "None", no layer is created and
    .uns['analysis_config']['active_layer'] is set to None.

    Arguments:
        anndata_object: SFC AnnData with .uns['analysis_config'] initialised by
                        select_sfc_subset().

    Returns:
        The same AnnData object with the new layer added and
        .uns['analysis_config']['active_layer'] updated.
    """
    print("=" * 60)
    print("TRANSFORMATION AND NORMALIZATION")
    print("=" * 60)
    print(
        f"Current dataset: {anndata_object.n_obs:,} events "
        f"× {anndata_object.n_vars} channels"
    )
    print()

    transform_key, norm_key = _prompt_transform_and_norm()

    if transform_key == _TRANSFORM_NONE and norm_key == _NORM_NONE:
        print("\n=> No transformation or normalization applied.")
        print("=> Downstream functions will read directly from .X.")
        _set_active_layer(anndata_object, None)
        print("=" * 60)
        return anndata_object

    layer_name = _build_layer_name(transform_key, norm_key)

    if layer_name in anndata_object.layers:
        print(f"\n=> Layer '{layer_name}' already exists — reusing cached result.")
        _set_active_layer(anndata_object, layer_name)
        _print_layer_summary(anndata_object.layers[layer_name], layer_name)
        _warn_if_layer_parameters_missing(anndata_object, layer_name)
        print("=" * 60)
        return anndata_object

    print(f"\nApplying '{layer_name}' — please wait…")

    raw_matrix = _get_raw_matrix(anndata_object)
    processed_matrix, layer_parameters = _fit_and_apply_layer(
        raw_matrix, transform_key, norm_key,
        anndata_object.var_names.tolist(), layer_name,
    )

    # copy=False avoids a redundant allocation when processed_matrix is
    # already float32 (the common case after _get_raw_matrix).
    anndata_object.layers[layer_name] = processed_matrix.astype(np.float32, copy=False)
    _store_layer_parameters(anndata_object, layer_parameters)
    _set_active_layer(anndata_object, layer_name)

    _print_layer_summary(anndata_object.layers[layer_name], layer_name)
    _print_layer_parameters(layer_parameters)
    print("=" * 60)

    return anndata_object


# ---------------------------------------------------------------------------
# Console interaction
# ---------------------------------------------------------------------------


def _prompt_transform_and_norm() -> Tuple[str, str]:
    """
    Display a single combined menu for transformation and normalization choices.

    The user selects one transform (number 1–3) and one normalization (letter a–e)
    in a single response, e.g. "2b" for Arcsinh + Z-score.
    Pressing Enter without input applies the default: no transform + Z-score per
    column ("1b"), the most common choice for spectral flow cytometry.

    Returns:
        Tuple of (transform_key, norm_key) from the _TRANSFORM_* and _NORM_*
        constants defined in this module.
    """
    transform_map = {
        "1": _TRANSFORM_NONE,
        "2": _TRANSFORM_ARCSINH,
        "3": _TRANSFORM_LOGICLE,
    }
    norm_map = {
        "a": _NORM_NONE,
        "b": _NORM_ZSCORE_COL,
        "c": _NORM_MINMAX_COL,
        "d": _NORM_L2_ROW,
        "e": _NORM_ZSCORE_COL_L2_ROW,
    }

    print("Transformation + Normalization:")
    print()
    print("  Transform (number):")
    print("    [1] None — raw intensities as-is")
    print(
        "    [2] Arcsinh (cofactor=150) — cytometry standard, "
        "compresses extreme values"
    )
    print(
        "    [3] Logicle — adaptive transformation, handles negative "
        "compensation values; best for well-calibrated panels"
    )
    print()
    print("  Normalization (letter):")
    print("    [a] None — no normalization")
    print(
        "    [b] Z-score per column (StandardScaler) — mean=0, std=1 per channel"
    )
    print(
        "    [c] Min-Max per column — scales each channel to [0, 1]"
    )
    print(
        "    [d] L2 Norm per row — each cell's profile becomes a unit vector "
        "(shape, not intensity)"
    )
    print("    [e] Z-score per column + L2 Norm per row")
    print()
    print(
        "  Default (press Enter): no transformation + Z-score per column [1b]"
    )

    while True:
        raw_response = input(
            "Select transform + norm (e.g. 1b, 2b, 3e): "
        ).strip().lower()

        if raw_response == "":
            return _TRANSFORM_NONE, _NORM_ZSCORE_COL

        # Extract exactly one digit and one letter from the input.
        digits = [ch for ch in raw_response if ch.isdigit()]
        letters = [ch for ch in raw_response if ch.isalpha()]

        if len(digits) != 1 or len(letters) != 1:
            print(
                f"  Invalid input '{raw_response}'. "
                "Enter one number (1–3) and one letter (a–e), e.g. '2b'."
            )
            continue

        digit, letter = digits[0], letters[0]

        if digit not in transform_map:
            print(f"  Unknown transform '{digit}'. Choose 1, 2, or 3.")
            continue
        if letter not in norm_map:
            print(f"  Unknown normalization '{letter}'. Choose a, b, c, d, or e.")
            continue

        return transform_map[digit], norm_map[letter]


# ---------------------------------------------------------------------------
# Transformation and normalization logic
# ---------------------------------------------------------------------------


def _apply_transform(matrix: np.ndarray, transform_key: str) -> np.ndarray:
    """
    Apply the selected transformation to a raw intensity matrix.

    Arguments:
        matrix: 2D float array (n_obs, n_vars) of raw or pre-processed values.
        transform_key: One of _TRANSFORM_NONE, _TRANSFORM_ARCSINH,
                       _TRANSFORM_LOGICLE.

    Returns:
        Transformed 2D float array (same shape).
    """
    if transform_key == _TRANSFORM_NONE:
        return matrix.copy()

    if transform_key == _TRANSFORM_ARCSINH:
        # Divide and arcsinh in-place to avoid allocating a temporary array
        # for the intermediate division result (saves one full copy of .X).
        np.divide(matrix, _ARCSINH_COFACTOR, out=matrix)
        np.arcsinh(matrix, out=matrix)
        return matrix

    if transform_key == _TRANSFORM_LOGICLE:
        return _logicle_transform(matrix)

    raise ValueError(f"Unknown transformation key: '{transform_key}'")


def _logicle_transform(matrix: np.ndarray) -> np.ndarray:
    """
    Apply a simplified Logicle-style transformation.

    The full Logicle transform (Parks et al. 2006) requires per-channel
    calibration parameters derived from instrument-specific negative controls.
    Without those parameters, we use a robust approximation:
        arcsinh(x / estimated_T * 5)
    where estimated_T is the 99th percentile of each channel, clamped to a
    minimum of 1 to avoid division by near-zero denominators.

    This produces similar visual results to the full Logicle for well-spread
    spectral cytometry data and avoids requiring user parameter input.

    Arguments:
        matrix: 2D float array (n_obs, n_vars).

    Returns:
        Transformed 2D float array (same shape).
    """
    # Estimate the top of the scale per channel from the 99th percentile.
    # channel_scale is a small (n_vars,) array — negligible memory.
    channel_scale = np.percentile(matrix, 99, axis=0)
    channel_scale = np.maximum(channel_scale, 1.0)  # avoid division by zero

    # All three steps are performed in-place on matrix to avoid allocating
    # a separate 'scaled' intermediate array (saves one full copy of .X).
    # Scale to approximately [−1, 1] before arcsinh so the result is
    # comparable in range to the Arcsinh cofactor approach.
    np.divide(matrix, channel_scale, out=matrix)
    np.multiply(matrix, 5.0, out=matrix)
    np.arcsinh(matrix, out=matrix)
    return matrix


def _apply_normalization(matrix: np.ndarray, norm_key: str) -> np.ndarray:
    """
    Apply the selected normalization strategy to a (possibly transformed) matrix.

    Arguments:
        matrix: 2D float array (n_obs, n_vars).
        norm_key: One of the _NORM_* constants.

    Returns:
        Normalized 2D float array (same shape).
    """
    if norm_key == _NORM_NONE:
        return matrix

    if norm_key == _NORM_ZSCORE_COL:
        # copy=False transforms in-place when the array dtype is compatible
        # (float32 or float64), avoiding a full-matrix allocation.
        return StandardScaler(copy=False).fit_transform(matrix)

    if norm_key == _NORM_MINMAX_COL:
        return MinMaxScaler(copy=False).fit_transform(matrix)

    if norm_key == _NORM_L2_ROW:
        return Normalizer(norm="l2", copy=False).fit_transform(matrix)

    if norm_key == _NORM_ZSCORE_COL_L2_ROW:
        # Both steps operate in-place on the same matrix, so no intermediate
        # 'z_scored' array is allocated between the two passes.
        StandardScaler(copy=False).fit_transform(matrix)
        Normalizer(norm="l2", copy=False).fit_transform(matrix)
        return matrix

    raise ValueError(f"Unknown normalization key: '{norm_key}'")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Reusable (frozen) layer parameters
#
# Every fitted normalization is recorded in .uns['layer_parameters'][layer]
# as plain numpy arrays (h5ad-compatible, no pickle). Replaying them with
# _apply_stored_layer_parameters() on ANOTHER dataset scales it with the
# reference dataset's means/standard deviations instead of its own — the
# only way to compare two datasets in one shared space (see ADR-015).
# ---------------------------------------------------------------------------


def _fit_and_apply_layer(
    raw_matrix: np.ndarray,
    transform_key: str,
    norm_key: str,
    feature_names: List[str],
    layer_name: str,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Fit a transformation + normalization on raw_matrix, apply it, and return
    both the processed matrix and every parameter needed to replay it.

    The matrix is modified in place (it must be a fresh copy, as returned by
    _get_raw_matrix). The processed values are identical to those of the
    historical _apply_transform() + _apply_normalization() pair.

    Arguments:
        raw_matrix: Fresh float32 copy of .X, shape (n_obs, n_vars).
        transform_key: One of the _TRANSFORM_* constants.
        norm_key: One of the _NORM_* constants.
        feature_names: The n_vars column names, in matrix order.
        layer_name: Name of the layer being created (stored for traceability).

    Returns:
        Tuple (processed_matrix, layer_parameters). layer_parameters holds:
          layer_name, feature_names, transform, arcsinh_cofactor,
          logicle_channel_scale, normalization, scaler_mean, scaler_scale,
          minmax_scale, minmax_min, n_obs_fitted, fingerprint.
    """
    empty_vector = np.zeros(0, dtype=np.float64)

    # Logicle is the only transform with a data-dependent parameter: the
    # per-channel 99th percentile. It is fitted HERE and frozen, so a second
    # dataset is compressed with the reference scale, not its own.
    logicle_channel_scale = empty_vector
    if transform_key == _TRANSFORM_LOGICLE:
        logicle_channel_scale = np.maximum(
            np.percentile(raw_matrix, 99, axis=0), 1.0
        ).astype(np.float64)
    elif transform_key not in (_TRANSFORM_NONE, _TRANSFORM_ARCSINH):
        raise ValueError(f"Unknown transformation key: '{transform_key}'")

    _transform_in_place(raw_matrix, transform_key, logicle_channel_scale)

    scaler_mean, scaler_scale = empty_vector, empty_vector
    minmax_scale, minmax_min = empty_vector, empty_vector
    if norm_key in (_NORM_ZSCORE_COL, _NORM_ZSCORE_COL_L2_ROW):
        # StandardScaler ignores NaN when fitting and sets the scale of a
        # constant column to 1, which keeps the replay free of divisions by 0.
        fitted_scaler = StandardScaler().fit(raw_matrix)
        scaler_mean = fitted_scaler.mean_.astype(np.float64)
        scaler_scale = fitted_scaler.scale_.astype(np.float64)
    elif norm_key == _NORM_MINMAX_COL:
        fitted_minmax = MinMaxScaler().fit(raw_matrix)
        minmax_scale = fitted_minmax.scale_.astype(np.float64)
        minmax_min = fitted_minmax.min_.astype(np.float64)
    elif norm_key not in (_NORM_NONE, _NORM_L2_ROW):
        raise ValueError(f"Unknown normalization key: '{norm_key}'")

    layer_parameters: Dict[str, Any] = {
        "layer_name": layer_name,
        "feature_names": [str(name) for name in feature_names],
        "transform": transform_key,
        "arcsinh_cofactor": float(_ARCSINH_COFACTOR),
        "logicle_channel_scale": logicle_channel_scale,
        "normalization": norm_key,
        "scaler_mean": scaler_mean,
        "scaler_scale": scaler_scale,
        "minmax_scale": minmax_scale,
        "minmax_min": minmax_min,
        "n_obs_fitted": int(raw_matrix.shape[0]),
    }
    layer_parameters["fingerprint"] = compute_fingerprint(
        layer_parameters["feature_names"], transform_key, float(_ARCSINH_COFACTOR),
        logicle_channel_scale, norm_key, scaler_mean, scaler_scale,
        minmax_scale, minmax_min,
    )

    _normalize_in_place(raw_matrix, layer_parameters)
    return raw_matrix, layer_parameters


def _apply_stored_layer_parameters(
    raw_matrix: np.ndarray,
    layer_parameters: Dict[str, Any],
) -> np.ndarray:
    """
    Replay a frozen transformation + normalization on a new raw matrix.

    Nothing is fitted: the reference dataset's parameters are used as-is.
    The columns of raw_matrix must already be in the order of
    layer_parameters['feature_names'].

    Arguments:
        raw_matrix: Fresh float32 copy of raw values, shape (n_obs, n_vars).
                    Modified in place.
        layer_parameters: Dict produced by _fit_and_apply_layer().

    Returns:
        The processed matrix (same object as raw_matrix).
    """
    n_expected = len(layer_parameters["feature_names"])
    if raw_matrix.shape[1] != n_expected:
        raise ValueError(
            f"Matrix has {raw_matrix.shape[1]} columns but the stored layer "
            f"parameters describe {n_expected} features."
        )
    _transform_in_place(
        raw_matrix,
        str(layer_parameters["transform"]),
        np.asarray(layer_parameters["logicle_channel_scale"], dtype=np.float64),
        float(layer_parameters["arcsinh_cofactor"]),
    )
    _normalize_in_place(raw_matrix, layer_parameters)
    return raw_matrix


def _transform_in_place(
    matrix: np.ndarray,
    transform_key: str,
    logicle_channel_scale: np.ndarray,
    arcsinh_cofactor: float = _ARCSINH_COFACTOR,
) -> None:
    """
    Apply a transformation in place with explicit (frozen) parameters.

    Arguments:
        matrix: 2D float array, modified in place.
        transform_key: One of the _TRANSFORM_* constants.
        logicle_channel_scale: Per-channel scale (logicle only, else empty).
        arcsinh_cofactor: Divisor applied before arcsinh.
    """
    if transform_key == _TRANSFORM_NONE:
        return
    if transform_key == _TRANSFORM_ARCSINH:
        np.divide(matrix, arcsinh_cofactor, out=matrix)
        np.arcsinh(matrix, out=matrix)
        return
    if transform_key == _TRANSFORM_LOGICLE:
        np.divide(matrix, logicle_channel_scale.astype(matrix.dtype), out=matrix)
        np.multiply(matrix, 5.0, out=matrix)
        np.arcsinh(matrix, out=matrix)
        return
    raise ValueError(f"Unknown transformation key: '{transform_key}'")


def _normalize_in_place(matrix: np.ndarray, layer_parameters: Dict[str, Any]) -> None:
    """
    Apply a normalization in place from stored parameters.

    Arguments:
        matrix: 2D float array (already transformed), modified in place.
        layer_parameters: Dict with 'normalization' and its parameter arrays.
    """
    norm_key = str(layer_parameters["normalization"])
    if norm_key in (_NORM_ZSCORE_COL, _NORM_ZSCORE_COL_L2_ROW):
        scaler_mean = np.asarray(layer_parameters["scaler_mean"], dtype=matrix.dtype)
        scaler_scale = np.asarray(layer_parameters["scaler_scale"], dtype=matrix.dtype)
        np.subtract(matrix, scaler_mean, out=matrix)
        np.divide(matrix, scaler_scale, out=matrix)
    elif norm_key == _NORM_MINMAX_COL:
        # MinMaxScaler.transform is X * scale_ + min_.
        np.multiply(matrix, np.asarray(layer_parameters["minmax_scale"], dtype=matrix.dtype), out=matrix)
        np.add(matrix, np.asarray(layer_parameters["minmax_min"], dtype=matrix.dtype), out=matrix)

    # The L2 row norm has no fitted parameter: each row is divided by its own
    # length, so it is applied identically on any dataset.
    if norm_key in (_NORM_L2_ROW, _NORM_ZSCORE_COL_L2_ROW):
        Normalizer(norm="l2", copy=False).fit_transform(matrix)


def _store_layer_parameters(
    anndata_object: anndata.AnnData,
    layer_parameters: Dict[str, Any],
) -> None:
    """
    Record frozen layer parameters in .uns['layer_parameters'][layer_name].

    Arguments:
        anndata_object: AnnData updated in place.
        layer_parameters: Dict produced by _fit_and_apply_layer().
    """
    stored = dict(anndata_object.uns.get(LAYER_PARAMETERS_KEY, {}))
    stored[layer_parameters["layer_name"]] = layer_parameters
    anndata_object.uns[LAYER_PARAMETERS_KEY] = stored


def get_layer_parameters(
    anndata_object: anndata.AnnData,
    layer_name: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """
    Return the frozen parameters of a normalized layer, or None if unknown.

    Layers created before parameters were recorded (older sessions or files)
    return None: their scaling cannot be replayed on another dataset.

    Arguments:
        anndata_object: AnnData to inspect.
        layer_name: Layer to look up; None means the active layer.

    Returns:
        The parameter dict, or None.
    """
    if layer_name is None:
        layer_name = anndata_object.uns.get(_ANALYSIS_CONFIG_KEY, {}).get("active_layer")
    if layer_name is None:
        return None
    return anndata_object.uns.get(LAYER_PARAMETERS_KEY, {}).get(layer_name)


def _warn_if_layer_parameters_missing(
    anndata_object: anndata.AnnData,
    layer_name: str,
) -> None:
    """
    Warn when a cached layer has no frozen parameters (created by an older
    version of the package): it works for plots, but its scaling cannot be
    replayed on another dataset, so it cannot serve as a reference space.

    Arguments:
        anndata_object: AnnData holding the layer.
        layer_name: Name of the cached layer.
    """
    if get_layer_parameters(anndata_object, layer_name) is None:
        print(
            f"WARNING: layer '{layer_name}' has no frozen parameters in "
            f".uns['{LAYER_PARAMETERS_KEY}'] (created by an older version). "
            "It cannot be replayed on another dataset. Delete the layer and "
            "re-run the normalization to record them."
        )


def _print_layer_parameters(layer_parameters: Dict[str, Any]) -> None:
    """
    Print the frozen parameters of a layer so a reader can see what was fitted.

    Arguments:
        layer_parameters: Dict produced by _fit_and_apply_layer().
    """
    print(
        f"=> Frozen parameters stored in .uns['{LAYER_PARAMETERS_KEY}']"
        f"['{layer_parameters['layer_name']}'] — fingerprint "
        f"{layer_parameters['fingerprint']} "
        f"(fitted on {layer_parameters['n_obs_fitted']:,} observations, "
        f"{len(layer_parameters['feature_names'])} features, "
        f"transform='{layer_parameters['transform']}', "
        f"normalization='{layer_parameters['normalization']}')"
    )


def _build_layer_name(transform_key: str, norm_key: str) -> str:
    """
    Build the .layers key string from the two user choices.

    Format: '{transform}__{norm}'
    The double underscore is the canonical separator between the two stages.

    Arguments:
        transform_key: Transformation identifier token.
        norm_key: Normalization identifier token.

    Returns:
        Layer key string, e.g. 'arcsinh__zscore_col'.
    """
    return f"{transform_key}__{norm_key}"


def _get_raw_matrix(anndata_object: anndata.AnnData) -> np.ndarray:
    """
    Return .X as a dense float32 numpy array, always as a new copy.

    The transformation and normalization are always applied to the raw .X,
    never to a previously normalized layer. This keeps the pipeline stages
    independent: different layer combinations can coexist without conflict.

    Using float32 (instead of float64) halves the memory footprint of every
    intermediate matrix produced during transformation and normalization.
    Cytometry intensities do not require double-precision arithmetic.

    The returned array is always a fresh copy so that in-place transforms
    downstream never mutate .X.

    Arguments:
        anndata_object: The AnnData to extract .X from.

    Returns:
        2D float32 numpy array (n_obs, n_vars), always a new allocation.
    """
    x_data = anndata_object.X
    if hasattr(x_data, "toarray"):
        # toarray() allocates a new dense array; pass dtype to avoid a
        # second allocation when the stored sparse dtype is not float32.
        return x_data.toarray(dtype=np.float32)
    # astype() with a differing dtype always allocates a new array.
    # With the same dtype, copy=True guarantees a fresh array so that
    # in-place ops below cannot accidentally mutate .X.
    return x_data.astype(np.float32, copy=True)


def _set_active_layer(
    anndata_object: anndata.AnnData,
    layer_name: Optional[str],
) -> None:
    """
    Update .uns['analysis_config']['active_layer'] to the given layer name.

    Initialises .uns['analysis_config'] if not already present.

    Arguments:
        anndata_object: AnnData to update in-place.
        layer_name: New active layer name, or None to indicate "use .X".
    """
    if _ANALYSIS_CONFIG_KEY not in anndata_object.uns:
        anndata_object.uns[_ANALYSIS_CONFIG_KEY] = {}
    anndata_object.uns[_ANALYSIS_CONFIG_KEY]["active_layer"] = layer_name


def _print_layer_summary(layer_matrix: np.ndarray, layer_name: str) -> None:
    """
    Print a QC summary for a newly created layer.

    Arguments:
        layer_matrix: The normalized/transformed matrix stored in .layers.
        layer_name: The layer key (for display only).
    """
    finite_values = layer_matrix[np.isfinite(layer_matrix)]
    nan_count = int(np.isnan(layer_matrix).sum())
    inf_count = int(np.isinf(layer_matrix).sum())

    print(f"\n=> Layer '.layers['{layer_name}']' created.")
    print(f"=> Active layer set to: '{layer_name}'")
    if finite_values.size > 0:
        print(
            f"=> Data summary: "
            f"min={finite_values.min():.4f}, "
            f"max={finite_values.max():.4f}, "
            f"mean={finite_values.mean():.4f}, "
            f"std={finite_values.std():.4f}"
        )
    print(f"=> NaN count: {nan_count} | Inf count: {inf_count}")
