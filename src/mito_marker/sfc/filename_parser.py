"""
filename_parser.py

Parse a spectral flow cytometry FCS filename into a validated metadata dictionary
suitable for direct assignment to AnnData .obs columns.

Filename convention: tokens are separated by underscores ('_'). Each token is
inspected independently against the controlled vocabulary rules defined in
controlled_vocabulary.py. The order of tokens in the filename does not matter —
every token is examined for every possible field.

Example filename:
    good_events_MNMS_020_FlowAIGoodEvents_DeepRed.fcs
    → {'specie': 'MNMS', 'subject_ID': '020', 'age': None, 'diet': None,
       'dilution': 'Diluted', 'marker': 'MtDeepRed', 'FlowAI_Pass': True,
       'source_filename': 'good_events_MNMS_020_FlowAIGoodEvents_DeepRed.fcs'}
"""

import os
import re
from typing import Optional

from mito_marker.controlled_vocabulary import (
    AGE_GROUP_TOKEN_TO_CANONICAL_SFC,
    ALLOWED_DIET_SFC,
    ALLOWED_DILUTION_SFC,
    ALLOWED_MARKERS_SFC,
    ALLOWED_SPECIES_SFC,
    FLOWAI_PASS_TOKENS,
    MARKER_TOKEN_TO_CANONICAL_SFC,
    NUMERICAL_OBS_BOUNDS,
    SFC_MARKER_JOIN_ORDER,
    TREATMENT_TOKEN_TO_CANONICAL_SFC,
)


def parse_fcs_filename(filename: str) -> dict[str, str | int | bool | None]:
    """
    Parse an FCS filename into a validated metadata dictionary for AnnData .obs.

    Splits the filename (without extension) on '_', then inspects each token against
    the controlled vocabulary rules to extract specie, subject_ID, age, diet, dilution,
    marker, FlowAI_Pass, and source_filename.

    Arguments:
        filename: The base filename string, with or without the .fcs extension.
                  Example: 'good_events_MNMS_020_FlowAIGoodEvents_DeepRed.fcs'

    Returns:
        A dict with the following keys:
          - 'specie': str — canonical species token (e.g. 'MNMS')
          - 'subject_ID': str — 3-digit zero-padded ID string (e.g. '020')
          - 'age': int | None — numeric age value (e.g. 3 from 'D03', 6 from 'M06'), or None
          - 'diet': str | None — diet token (e.g. 'AL', 'IF'), or None
          - 'dilution': str — 'Diluted' or 'Not_Diluted'
          - 'marker': str — canonical marker value (e.g. 'MtDeepRed')
          - 'FlowAI_Pass': bool — True if FlowAI pass tokens found in filename
          - 'source_filename': str — the original filename (basename only)

    Raises:
        ValueError: If any required field (specie, subject_ID, marker) is absent
                    from the filename tokens.
    """
    # Keep the original basename for traceability, strip any directory prefix
    source_filename = os.path.basename(filename)

    # Remove .fcs extension before splitting (case-insensitive)
    stem = source_filename
    if stem.lower().endswith(".fcs"):
        stem = stem[:-4]

    tokens = stem.split("_")

    return {
        "specie": _extract_specie(tokens),
        "subject_ID": _extract_subject_id(tokens),
        "age": _extract_age(tokens),
        "diet": _extract_diet(tokens),
        "age_group": _extract_age_group(tokens),
        "treatment": _extract_treatment(tokens),
        "dilution": _extract_dilution(tokens),
        "marker": _extract_marker(tokens),
        "FlowAI_Pass": _detect_flowai_pass(tokens),
        "source_filename": source_filename,
    }


def _extract_specie(tokens: list[str]) -> str:
    """
    Find the first token that matches a key in ALLOWED_SPECIES_SFC.

    Arguments:
        tokens: List of filename tokens split on '_'.

    Returns:
        The matched specie token string (e.g. 'MNMS').

    Raises:
        ValueError: If no token matches any key in ALLOWED_SPECIES_SFC.
    """
    for token in tokens:
        if token in ALLOWED_SPECIES_SFC:
            return token

    allowed = list(ALLOWED_SPECIES_SFC.keys())
    raise ValueError(
        f"Could not find a valid specie token in filename tokens {tokens}. "
        f"Allowed specie values: {allowed}"
    )


def _extract_subject_id(tokens: list[str]) -> str:
    """
    Find the first token that matches the 3-digit subject ID pattern ('001'–'999').

    The leading zeros are preserved (e.g. '020' stays '020', not 20).
    The value '000' is excluded as it is not a valid subject ID.

    Arguments:
        tokens: List of filename tokens split on '_'.

    Returns:
        The matched subject ID string (e.g. '020').

    Raises:
        ValueError: If no token matches the 3-digit non-zero pattern.
    """
    for token in tokens:
        # fullmatch ensures the entire token is exactly 3 digits
        if re.fullmatch(r"[0-9]{3}", token) and token != "000":
            return token

    raise ValueError(
        f"Could not find a valid subject ID (3-digit number 001–999) in filename "
        f"tokens {tokens}."
    )


def _extract_age(tokens: list[str]) -> Optional[int]:
    """
    Find the first token matching an age pattern (D\\d{2}, M\\d{2}, or Y\\d{2}),
    extract the numeric part as an integer, and validate it against NUMERICAL_OBS_BOUNDS.

    D = days, M = months, Y = years.
    Examples: 'D03' → 3, 'M06' → 6, 'Y02' → 2.

    Arguments:
        tokens: List of filename tokens split on '_'.

    Returns:
        The integer age value, or None if no age token is present.

    Raises:
        AssertionError: If the extracted value falls outside the declared physiological bounds.
    """
    prefix_to_bounds_key = {
        "D": "age_days",
        "M": "age_months",
        "Y": "age_years",
    }
    for token in tokens:
        match = re.fullmatch(r"([DMY])(\d{2})", token)
        if match:
            prefix = match.group(1)
            age_value = int(match.group(2))
            bounds_key = prefix_to_bounds_key[prefix]
            bounds = NUMERICAL_OBS_BOUNDS[bounds_key]
            assert bounds["min"] <= age_value <= bounds["max"], (
                f"Age value {age_value} from token '{token}' is outside expected bounds "
                f"[{bounds['min']}, {bounds['max']}] {bounds['unit']} for '{bounds_key}'."
            )
            return age_value
    return None


def _extract_diet(tokens: list[str]) -> Optional[str]:
    """
    Find the first token that matches a key in ALLOWED_DIET_SFC.

    Arguments:
        tokens: List of filename tokens split on '_'.

    Returns:
        The matched diet token string (e.g. 'AL' or 'IF'), or None if absent.
    """
    for token in tokens:
        if token in ALLOWED_DIET_SFC:
            return token
    return None


def _extract_dilution(tokens: list[str]) -> str:
    """
    Determine dilution status from the token list.

    If the token 'ND' appears in the token list, the sample was not diluted.
    Otherwise, the default assumption is that the sample was diluted.
    Both canonical values must exist in ALLOWED_DILUTION_SFC.

    Arguments:
        tokens: List of filename tokens split on '_'.

    Returns:
        'Not_Diluted' if 'ND' is in tokens, otherwise 'Diluted'.
    """
    if "ND" in tokens:
        dilution_value = "Not_Diluted"
    else:
        dilution_value = "Diluted"

    # Validate against controlled vocabulary (defensive check)
    assert dilution_value in ALLOWED_DILUTION_SFC, (
        f"Internal error: computed dilution value '{dilution_value}' is not in "
        f"ALLOWED_DILUTION_SFC. This should never happen."
    )
    return dilution_value


def _extract_marker(tokens: list[str]) -> str:
    """
    Find all tokens matching a key in MARKER_TOKEN_TO_CANONICAL_SFC, resolve them to
    canonical values, and return a single string.

    When a single marker is present (e.g. 'DeepRed'), its canonical value is returned
    directly ('MtDeepRed'). When multiple markers are present (e.g. 'TMRM' and
    'DeepRed'), they are sorted by SFC_MARKER_JOIN_ORDER and joined with '+' to produce
    a deterministic combined value (e.g. 'TMRM+MtDeepRed').

    Arguments:
        tokens: List of filename tokens split on '_'.

    Returns:
        A canonical marker string (e.g. 'MtDeepRed', 'TMRM', 'TMRM+MtDeepRed').

    Raises:
        ValueError: If no token matches any key in MARKER_TOKEN_TO_CANONICAL_SFC.
    """
    found_canonical: list[str] = []
    for token in tokens:
        if token in MARKER_TOKEN_TO_CANONICAL_SFC:
            canonical_value = MARKER_TOKEN_TO_CANONICAL_SFC[token]
            assert canonical_value in ALLOWED_MARKERS_SFC, (
                f"Internal error: canonical marker value '{canonical_value}' resolved from "
                f"token '{token}' is not in ALLOWED_MARKERS_SFC."
            )
            if canonical_value not in found_canonical:
                found_canonical.append(canonical_value)

    if not found_canonical:
        allowed_tokens = list(MARKER_TOKEN_TO_CANONICAL_SFC.keys())
        raise ValueError(
            f"Could not find a valid marker token in filename tokens {tokens}. "
            f"Recognised marker tokens: {allowed_tokens}"
        )

    found_canonical.sort(key=lambda m: SFC_MARKER_JOIN_ORDER.index(m))
    return "+".join(found_canonical)


def _extract_age_group(tokens: list[str]) -> Optional[str]:
    """
    Find the first token matching a key in AGE_GROUP_TOKEN_TO_CANONICAL_SFC.

    Arguments:
        tokens: List of filename tokens split on '_'.

    Returns:
        The canonical age group string ('Young' or 'Old'), or None if absent.
    """
    for token in tokens:
        if token in AGE_GROUP_TOKEN_TO_CANONICAL_SFC:
            return AGE_GROUP_TOKEN_TO_CANONICAL_SFC[token]
    return None


def _extract_treatment(tokens: list[str]) -> Optional[str]:
    """
    Find the first token matching a key in TREATMENT_TOKEN_TO_CANONICAL_SFC.

    Arguments:
        tokens: List of filename tokens split on '_'.

    Returns:
        The canonical treatment string ('Vehicule', 'Arac15', or 'Arac30'), or None if absent.
    """
    for token in tokens:
        if token in TREATMENT_TOKEN_TO_CANONICAL_SFC:
            return TREATMENT_TOKEN_TO_CANONICAL_SFC[token]
    return None


def _detect_flowai_pass(tokens: list[str]) -> bool:
    """
    Return True if any token from FLOWAI_PASS_TOKENS appears in the token list.

    FlowAI is a quality control algorithm. Its presence in the filename (via tokens
    such as 'good', 'events', or 'FlowAIGoodEvents') indicates that the file has
    already been filtered to retain only good-quality events.

    Arguments:
        tokens: List of filename tokens split on '_'.

    Returns:
        True if the file is marked as FlowAI-passed, False otherwise.
    """
    # Use set intersection for efficient lookup across all tokens at once
    return bool(set(tokens) & FLOWAI_PASS_TOKENS)
