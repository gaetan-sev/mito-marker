"""
GPU detection utilities for optional cuML (RAPIDS) acceleration.

cuML provides GPU-accelerated drop-in replacements for scikit-learn and umap-learn.
It is an optional dependency — this module makes GPU support transparent: if cuML
is absent or no CUDA device is present, callers fall back to CPU implementations.

Google Colab GPU runtimes already have cuML pre-installed — no manual installation needed.
Simply select a GPU runtime (Runtime → Change runtime type → GPU) and restart.
Do NOT manually install cuML: it downgrades pre-installed RAPIDS packages and breaks things.
"""


def is_cuml_available() -> bool:
    """
    Return True if cuML is importable and a CUDA device is present.

    cuML raises RuntimeError (not just ImportError) when the package is installed
    but no GPU is found. Both failure modes are caught here so callers never need
    their own try/except.

    Returns
    -------
    bool
        True if cuML can be used, False otherwise.
    """
    try:
        import cuml  # noqa: PLC0415, F401
        return True
    except Exception:
        return False
