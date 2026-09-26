"""
_fingerprint.py

Short, reproducible identifiers ("fingerprints") for fitted transformations.

A fingerprint is the first 12 hexadecimal characters of a SHA-256 hash of
every number and name that defines a fitted transformation (a scaler, a PCA,
a UMAP reducer). Two datasets whose layers carry the same fingerprint were
scaled with exactly the same parameters — this is how the cluster-model QC
proves that a "predict" dataset lives in the same reference space as the
"fit" dataset, without comparing large arrays by hand.

The fingerprint is computed ONCE, at fit time, and stored as a plain string
next to the parameters it describes. It is never recomputed from a file read
back from disk, so dtype changes introduced by .h5ad round-trips cannot
change it.
"""

import hashlib
from typing import Any

import numpy as np

# 12 hex characters = 48 bits: collisions between the handful of fitted
# transformations of one project are practically impossible, and the string
# stays short enough to print in every QC block.
_FINGERPRINT_LENGTH = 12


def compute_fingerprint(*parts: Any) -> str:
    """
    Hash an ordered sequence of values into a short hexadecimal identifier.

    Numeric arrays are hashed on their float64 bytes; strings, lists of
    strings and scalars are hashed on their text form. The order of parts
    matters.

    Arguments:
        *parts: Values that together define a fitted transformation
                (parameter arrays, feature names, method names, ...).

    Returns:
        12-character lowercase hexadecimal string.
    """
    hasher = hashlib.sha256()
    for part in parts:
        if isinstance(part, np.ndarray) and part.dtype.kind in "biuf":
            hasher.update(np.ascontiguousarray(part, dtype=np.float64).tobytes())
        elif isinstance(part, (list, tuple, np.ndarray)):
            hasher.update("|".join(str(item) for item in part).encode("utf-8"))
        elif isinstance(part, (bytes, bytearray)):
            hasher.update(bytes(part))
        else:
            hasher.update(str(part).encode("utf-8"))
        # Separator so that ("ab", "c") and ("a", "bc") hash differently.
        hasher.update(b"\x1f")
    return hasher.hexdigest()[:_FINGERPRINT_LENGTH]
