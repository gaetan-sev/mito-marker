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

The products added to the tube and the waiting times (the experimental
condition, ADR-019) are read separately by parse_condition_from_filename(),
because the product tokens are declared by the user in the notebook.
"""

import math
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
    SFC_ADDITION_TIME_TOKEN_PATTERN,
    SFC_ADDITION_TIME_UNIT_TO_MINUTES,
    SFC_CONDITION_WITHOUT_ADDITION,
    SFC_FILENAME_TOKENS_WITHOUT_METADATA,
    SFC_MARKER_JOIN_ORDER,
    SFC_PRODUCT_TOKEN_PATTERN,
    TREATMENT_TOKEN_TO_CANONICAL_SFC,
)

# Token marking a sample that was not diluted (see _extract_dilution).
_NOT_DILUTED_TOKEN = "ND"


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
    if _NOT_DILUTED_TOKEN in tokens:
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


def parse_condition_from_filename(
    filename: str,
    condition_tokens: dict[str, str],
) -> dict[str, str | int | float | list[str]]:
    """
    Read the products added to the tube, and the waiting times, from an FCS filename.

    The filename lists every product present in the tube, in order of addition,
    with each waiting time right after the product it refers to (ADR-019):

        WORM_001_D08_AL_DR                → condition 'No_addition'
        WORM_001_D08_AL_PGM_DR            → condition 'PGM'
        WORM_001_D08_AL_PGM_ADP_1min_DR   → condition 'PGM_ADP_1min'
        WORM_001_D08_AL_PGM_ADP_30min_DR  → condition 'PGM_ADP_30min'

    A token is a product only if it is a key of condition_tokens (declared in the
    notebook). A waiting time ('90s', '30min', '2h') is recognised by its form and
    converted to minutes. Tokens already read by parse_fcs_filename() (species,
    subject number, age, diet, marker…) and the tokens of
    SFC_FILENAME_TOKENS_WITHOUT_METADATA are skipped. Every other token is
    returned in 'unrecognised_tokens', so that the caller can warn the user.

    Arguments:
        filename: The base filename string, with or without the .fcs extension.
        condition_tokens: Declared products, {token as written in filenames:
                          plain-English description}, e.g. {"PGM": "Pyruvate +
                          glutamate + malate", "ADP": "Adenosine diphosphate"}.

    Returns:
        A dict with the following keys:
          - 'condition': str — product and time tokens joined with '_' in filename
            order (e.g. 'PGM_ADP_30min'), or 'No_addition' when no product is found
          - 'number_of_additions': int — number of product tokens (a product
            added twice counts twice)
          - 'minutes_after_last_addition': float — waiting time written after the
            last product, in minutes; NaN when no time follows the last product
          - 'products': list[str] — product tokens in filename order
          - 'unrecognised_tokens': list[str] — tokens that no rule recognises,
            including a waiting time written before any product

    Raises:
        TypeError, ValueError: If condition_tokens is not a valid declaration
                               (see _validate_condition_tokens).
        AssertionError: If a waiting time falls outside NUMERICAL_OBS_BOUNDS.
    """
    _validate_condition_tokens(condition_tokens)

    stem = os.path.basename(filename)
    if stem.lower().endswith(".fcs"):
        stem = stem[:-4]

    condition_parts: list[str] = []
    products: list[str] = []
    minutes_after_last_addition = math.nan
    unrecognised_tokens: list[str] = []

    for token in stem.split("_"):
        if token == "":
            continue  # double underscore in the filename

        if token in condition_tokens:
            products.append(token)
            condition_parts.append(token)
            # A new addition starts a new wait: an earlier time no longer applies.
            minutes_after_last_addition = math.nan
            continue

        time_match = re.fullmatch(SFC_ADDITION_TIME_TOKEN_PATTERN, token, flags=re.IGNORECASE)
        if time_match:
            if not products:
                # A waiting time must follow the product it refers to.
                unrecognised_tokens.append(token)
                continue
            time_value = int(time_match.group(1))
            time_unit = time_match.group(2).lower()
            minutes_after_last_addition = (
                time_value * SFC_ADDITION_TIME_UNIT_TO_MINUTES[time_unit]
            )
            bounds = NUMERICAL_OBS_BOUNDS["minutes_after_last_addition"]
            assert bounds["min"] <= minutes_after_last_addition <= bounds["max"], (
                f"Waiting time '{token}' in '{filename}' is outside the expected bounds "
                f"[{bounds['min']}, {bounds['max']}] {bounds['unit']}."
            )
            # Normalised spelling, so that '30MIN' and '30min' give one condition.
            condition_parts.append(f"{time_value}{time_unit}")
            continue

        if not _is_metadata_token(token):
            unrecognised_tokens.append(token)

    if products:
        condition = "_".join(condition_parts)
    else:
        condition = SFC_CONDITION_WITHOUT_ADDITION

    bounds = NUMERICAL_OBS_BOUNDS["number_of_additions"]
    assert bounds["min"] <= len(products) <= bounds["max"], (
        f"'{filename}' lists {len(products)} product additions, outside the expected "
        f"bounds [{bounds['min']}, {bounds['max']}]."
    )

    return {
        "condition": condition,
        "number_of_additions": len(products),
        "minutes_after_last_addition": minutes_after_last_addition,
        "products": products,
        "unrecognised_tokens": unrecognised_tokens,
    }


def _is_metadata_token(token: str) -> bool:
    """
    Return True if the token is read by parse_fcs_filename() or carries no metadata.

    Mirrors the rules of the _extract_* functions above: species, 3-digit subject
    number, age (D/M/Y + 2 digits), diet, age group, treatment, dilution, marker,
    FlowAI tags, plus the tokens of SFC_FILENAME_TOKENS_WITHOUT_METADATA.

    Arguments:
        token: One filename token.

    Returns:
        True if the token already has a meaning in SFC filenames, False otherwise.
    """
    known_tokens = (
        set(ALLOWED_SPECIES_SFC)
        | set(ALLOWED_DIET_SFC)
        | set(AGE_GROUP_TOKEN_TO_CANONICAL_SFC)
        | set(TREATMENT_TOKEN_TO_CANONICAL_SFC)
        | set(MARKER_TOKEN_TO_CANONICAL_SFC)
        | FLOWAI_PASS_TOKENS
        | set(SFC_FILENAME_TOKENS_WITHOUT_METADATA)
        | {_NOT_DILUTED_TOKEN}
    )
    if token in known_tokens:
        return True
    if re.fullmatch(r"[0-9]{3}", token):
        return True
    return re.fullmatch(r"[DMY]\d{2}", token) is not None


def _validate_condition_tokens(condition_tokens: dict[str, str]) -> None:
    """
    Check the products declared in the notebook before they are used.

    Each key must look like a product token (SFC_PRODUCT_TOKEN_PATTERN), must not
    already have a meaning in SFC filenames (e.g. 'AL' is the diet), and must come
    with a non-empty description.

    Arguments:
        condition_tokens: {product token: plain-English description}.

    Returns:
        None

    Raises:
        TypeError: If condition_tokens is not a dict.
        ValueError: If a token or a description is invalid; the message names it.
    """
    if not isinstance(condition_tokens, dict):
        raise TypeError(
            "condition_tokens must be a dict {product token: description}, e.g. "
            '{"PGM": "Pyruvate + glutamate + malate"}; '
            f"got {type(condition_tokens).__name__}."
        )
    for token, description in condition_tokens.items():
        if not isinstance(token, str) or not re.fullmatch(SFC_PRODUCT_TOKEN_PATTERN, token):
            raise ValueError(
                f"Invalid product token {token!r} in condition_tokens: a token starts "
                "with a letter and holds only letters, digits, '+' or '-' "
                "(no '_', which separates the words of a filename)."
            )
        if _is_metadata_token(token):
            raise ValueError(
                f"Product token '{token}' in condition_tokens already has a meaning in "
                "SFC filenames (species, subject number, age, diet, age group, treatment, "
                "dilution, marker, FlowAI or FlowJo tag). Choose another token."
            )
        if not isinstance(description, str) or not description.strip():
            raise ValueError(
                f"Product token '{token}' in condition_tokens needs a description, e.g. "
                f'"{token}": "<what this product is>".'
            )
