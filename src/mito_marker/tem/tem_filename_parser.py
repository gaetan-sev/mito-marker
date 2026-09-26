"""
tem_filename_parser.py

Parses a TEM measurement filename into structured .obs metadata:
  - specie     : canonical species name (from ALLOWED_SPECIES_TEM)
  - condition  : canonical condition name (from ALLOWED_CONDITIONS_TEM)
  - diet       : canonical diet value (from ALLOWED_DIET_TEM), or None if absent
  - subject_ID : numeric subject identifier as a string

Filename conventions are intentionally heterogeneous across experiments
(CamelCase, snake_case, abbreviated species names, French words).
The controlled_vocabulary dictionaries TEM_SPECIES_TOKEN_TO_CANONICAL,
TEM_CONDITION_TOKEN_TO_CANONICAL, and TEM_DIET_TOKEN_TO_CANONICAL are the
single place to register new tokens when a new naming convention is introduced.

Parsing strategy:
  1. Strip the fixed suffix "_MITO_measurements.txt" and the file extension.
  2. Cut the stem at the first occurrence of a known field-separator token
     ("grille", "longitudinalfield", …) — case-insensitive: only the prefix
     before it encodes biological metadata.
  3. Remove the magnification token (x<digits>) from the prefix.
  4. Scan for a species token (longest match first to avoid partial hits).
  5. Scan the remainder for a condition token.
  6. Scan the remainder for a diet token (IF / AL).
  7. Extract the subject_ID:
     - If a "<digits>fish" separator is present (KFish convention), take the
       digits that follow "fish".
     - Otherwise, collect all digit sequences and return the first one that is
       at least 3 characters long (to ignore single-digit counters like "J3").
  If any field cannot be resolved, a warning is emitted and None is returned
  for that field — the file is not rejected so that a full folder can still
  be ingested with partial metadata.
"""

import re
import warnings
from typing import Optional

from mito_marker.controlled_vocabulary import (
    ALLOWED_CONDITIONS_TEM,
    ALLOWED_DIET_TEM,
    ALLOWED_SPECIES_TEM,
    TEM_CONDITION_TOKEN_TO_CANONICAL,
    TEM_DIET_TOKEN_TO_CANONICAL,
    TEM_SPECIES_CONDITION_FROM_FILE,
    TEM_SPECIES_TOKEN_TO_CANONICAL,
)

# Fixed suffix that every TEM measurement file must end with (before .txt).
_MITO_MEASUREMENTS_SUFFIX: str = "_MITO_measurements"

# Tokens that mark the start of grid/field metadata in a TEM filename.
# Everything before the first of these tokens encodes biological metadata
# (species, condition, subject ID). Add new separators here when a new
# imaging convention is introduced.
_FIELD_SEPARATOR_TOKENS: list[str] = ["grille", "longitudinalfield"]


def parse_tem_filename(filename: str) -> dict[str, Optional[str]]:
    """
    Parse a TEM measurement filename into structured metadata fields.

    The filename may use any of the naming conventions registered in
    TEM_SPECIES_TOKEN_TO_CANONICAL and TEM_CONDITION_TOKEN_TO_CANONICAL.
    Unknown tokens produce a warning and return None for the affected field.

    Parameters
    ----------
    filename : str
        Basename of the TEM .txt file, with or without the .txt extension.
        Example: "KFishYoung1fish565x1200grille2Dfield2_MITO_measurements.txt"

    Returns
    -------
    dict[str, Optional[str]]
        Keys: "specie", "condition", "diet", "subject_ID".
        Values are canonical strings (as defined in controlled_vocabulary.py)
        or None when the field could not be resolved.
        "diet" is None when no diet token is present (not a warning condition).
    """
    stem = _strip_filename_to_stem(filename)
    prefix = _extract_biological_prefix(stem)
    prefix = _remove_magnification_token(prefix)

    specie, prefix = _extract_species_token(prefix, filename)
    condition, prefix = _extract_condition_token(prefix, filename, specie)
    diet, prefix = _extract_diet_token(prefix, filename)
    subject_id = _extract_subject_id(prefix, stem, filename)

    return {
        "specie": specie,
        "condition": condition,
        "diet": diet,
        "subject_ID": subject_id,
    }


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _strip_filename_to_stem(filename: str) -> str:
    """
    Remove the .txt extension and the '_MITO_measurements' suffix.

    Parameters
    ----------
    filename : str
        Raw filename, e.g. "KFishYoung1fish565x1200grille2Dfield2_MITO_measurements.txt"

    Returns
    -------
    str
        Stem with fixed suffix removed, e.g. "KFishYoung1fish565x1200grille2Dfield2"
    """
    stem = filename
    # Remove .txt extension if present
    if stem.lower().endswith(".txt"):
        stem = stem[:-4]
    # Remove the fixed measurements suffix if present
    if stem.endswith(_MITO_MEASUREMENTS_SUFFIX):
        stem = stem[: -len(_MITO_MEASUREMENTS_SUFFIX)]
    return stem


def _extract_biological_prefix(stem: str) -> str:
    """
    Cut the stem at the first occurrence of any known field-separator token
    (case-insensitive) and return everything before it.

    Field-separator tokens (e.g. 'grille', 'longitudinalfield') mark the start
    of grid/field metadata which is not needed for biological annotation.
    Everything before the earliest separator encodes species, condition, and
    subject ID.

    Parameters
    ----------
    stem : str
        Filename stem with the measurements suffix already removed.

    Returns
    -------
    str
        The biological prefix. If no separator token is found, the full stem is
        returned (the parser will still attempt to extract available fields).
    """
    lower_stem = stem.lower()
    # Find the earliest position among all known field-separator tokens.
    earliest_position = len(stem)
    for separator in _FIELD_SEPARATOR_TOKENS:
        position = lower_stem.find(separator)
        if position != -1 and position < earliest_position:
            earliest_position = position
    if earliest_position == len(stem):
        return stem
    return stem[:earliest_position]


def _remove_magnification_token(prefix: str) -> str:
    """
    Remove the magnification token (x<digits>, e.g. 'x1200') from the prefix.

    Parameters
    ----------
    prefix : str
        Biological prefix from _extract_biological_prefix.

    Returns
    -------
    str
        Prefix with the magnification pattern removed.
    """
    return re.sub(r"x\d+", "", prefix)


def _extract_species_token(
    prefix: str, filename: str
) -> tuple[Optional[str], str]:
    """
    Scan the prefix for a known species token and return the canonical species
    name plus the prefix with the matched token removed.

    Tokens are tried from longest to shortest to avoid partial matches
    (e.g. 'Zebrafish' must be tried before 'fish').

    Parameters
    ----------
    prefix : str
        Current working prefix (after suffix stripping and grille cut).
    filename : str
        Original filename, used only for warning messages.

    Returns
    -------
    tuple[Optional[str], str]
        (canonical_species_or_None, updated_prefix)
    """
    sorted_tokens = sorted(
        TEM_SPECIES_TOKEN_TO_CANONICAL.keys(), key=len, reverse=True
    )
    for token in sorted_tokens:
        position = prefix.find(token)
        if position != -1:
            canonical = TEM_SPECIES_TOKEN_TO_CANONICAL[token]
            assert canonical in ALLOWED_SPECIES_TEM, (
                f"Token '{token}' maps to '{canonical}' which is not in "
                f"ALLOWED_SPECIES_TEM. Update controlled_vocabulary.py."
            )
            updated_prefix = prefix[:position] + prefix[position + len(token):]
            return canonical, updated_prefix

    warnings.warn(
        f"No species token found in filename '{filename}'. "
        f"Known tokens: {sorted(TEM_SPECIES_TOKEN_TO_CANONICAL.keys())}. "
        f"Add the new token to TEM_SPECIES_TOKEN_TO_CANONICAL in controlled_vocabulary.py.",
        stacklevel=3,
    )
    return None, prefix


def _extract_condition_token(
    prefix: str, filename: str, specie: Optional[str] = None
) -> tuple[Optional[str], str]:
    """
    Scan the prefix for a known condition token and return the canonical
    condition name plus the prefix with the matched token removed.

    Tokens are tried from longest to shortest.

    For species listed in TEM_SPECIES_CONDITION_FROM_FILE (e.g. "Human"),
    the condition cannot be determined from the filename — it comes from the
    Condition_Name column inside the file.  In that case this function returns
    None silently (no warning), and the builder handles the fallback.

    Parameters
    ----------
    prefix : str
        Working prefix after species token removal.
    filename : str
        Original filename, used only for warning messages.
    specie : Optional[str]
        Canonical species name already extracted from the filename, used to
        decide whether a missing condition token is expected or an error.

    Returns
    -------
    tuple[Optional[str], str]
        (canonical_condition_or_None, updated_prefix)
    """
    sorted_tokens = sorted(
        TEM_CONDITION_TOKEN_TO_CANONICAL.keys(), key=len, reverse=True
    )
    for token in sorted_tokens:
        position = prefix.find(token)
        if position != -1:
            canonical = TEM_CONDITION_TOKEN_TO_CANONICAL[token]
            assert canonical in ALLOWED_CONDITIONS_TEM, (
                f"Token '{token}' maps to '{canonical}' which is not in "
                f"ALLOWED_CONDITIONS_TEM. Update controlled_vocabulary.py."
            )
            updated_prefix = prefix[:position] + prefix[position + len(token):]
            return canonical, updated_prefix

    # No condition token found. If the species is known to store its condition
    # inside the file (not in the filename), this is expected — return None silently.
    if specie in TEM_SPECIES_CONDITION_FROM_FILE:
        return None, prefix

    warnings.warn(
        f"No condition token found in filename '{filename}'. "
        f"Known tokens: {sorted(TEM_CONDITION_TOKEN_TO_CANONICAL.keys())}. "
        f"Add the new token to TEM_CONDITION_TOKEN_TO_CANONICAL in controlled_vocabulary.py.",
        stacklevel=3,
    )
    return None, prefix


def _extract_diet_token(
    prefix: str, filename: str
) -> tuple[Optional[str], str]:
    """
    Scan the prefix for a known diet token and return the canonical diet value
    plus the prefix with the matched token removed.

    Diet is an optional field — no warning is emitted when absent, because many
    experiments do not have a diet protocol (e.g. KFish Young/Old age-comparison).

    Parameters
    ----------
    prefix : str
        Working prefix after species and condition token removal.
    filename : str
        Original filename, kept for symmetry with other helpers (unused here
        since absence of a diet token is not an error).

    Returns
    -------
    tuple[Optional[str], str]
        (canonical_diet_or_None, updated_prefix)
    """
    sorted_tokens = sorted(
        TEM_DIET_TOKEN_TO_CANONICAL.keys(), key=len, reverse=True
    )
    for token in sorted_tokens:
        position = prefix.find(token)
        if position != -1:
            canonical = TEM_DIET_TOKEN_TO_CANONICAL[token]
            assert canonical in ALLOWED_DIET_TEM, (
                f"Token '{token}' maps to '{canonical}' which is not in "
                f"ALLOWED_DIET_TEM. Update controlled_vocabulary.py."
            )
            updated_prefix = prefix[:position] + prefix[position + len(token):]
            return canonical, updated_prefix
    return None, prefix


def _extract_subject_id(prefix: str, stem: str, filename: str) -> Optional[str]:
    """
    Extract the subject ID from the remaining prefix after species and condition
    tokens have been removed.

    Three strategies are applied in order:

    1. KFish convention — '<digits>fish<ID>': the subject ID is the digit
       sequence that follows the word 'fish'. Example: '1fish565' → '565'.
    2. Long-sequence convention — collect all digit sequences in the prefix and
       return the first one that is at least 2 characters long (ignoring
       single-digit counters like '3' in 'J3').
       This covers MNMS files ('051field26' → '051') and Drosophila files
       ('_03_' → '03').
    3. Grille-preceding convention — for files where the subject ID is the
       digit sequence immediately before 'grille' in the original stem.
       Used for mouse files where the ID is a single digit:
       'mouseold1grille...' → '1', 'mouseyoung3grille...' → '3'.

    Parameters
    ----------
    prefix : str
        Working prefix after both species and condition token removal.
    stem : str
        Full filename stem before any biological-prefix cutting, used by
        strategy 3 to locate the digit just before 'grille'.
    filename : str
        Original filename, used only for warning messages.

    Returns
    -------
    Optional[str]
        Subject ID as a string of digits, or None with a warning.
    """
    # Strategy 1: KFish '<n>fish<ID>' separator pattern
    fish_match = re.search(r"\d+[Ff]ish(\d+)", prefix)
    if fish_match:
        return fish_match.group(1)

    # Strategy 2: first digit sequence with at least 2 digits in the prefix.
    # The minimum is 2 (not 1) to skip single-digit counters (e.g. 'J3', 'G5').
    # Zero-padded 2-digit IDs like '03' and '12' (Drosophila convention) are
    # handled correctly because re.findall matches the full contiguous run.
    digit_sequences = re.findall(r"\d+", prefix)
    long_sequences = [seq for seq in digit_sequences if len(seq) >= 2]
    if long_sequences:
        return long_sequences[0]

    # Strategy 3: digit sequence immediately preceding 'grille' in the full stem.
    # Mouse files encode the subject number as a single digit just before the
    # grid reference: 'mouseold1grilleC5...' → subject_ID = '1'.
    grille_match = re.search(r"(\d+)[Gg]rille", stem)
    if grille_match:
        return grille_match.group(1)

    warnings.warn(
        f"No subject ID found in filename '{filename}'. "
        f"Remaining prefix after token removal: '{prefix}'. "
        f"Expected a digit sequence of at least 2 characters, or a digit "
        f"immediately before a field-separator token in the filename.",
        stacklevel=3,
    )
    return None
