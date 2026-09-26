"""
Shared helper to extract and format the current analysis run context for display.

Used by all plotting functions (radar, UMAP, PCA) to print a structured summary
before rendering a figure and to embed a compact footer annotation inside the figure.

The run context is derived from .uns["analysis_config"], which is populated by
select_sfc_subset() (selection log), transform_and_normalize() (active_layer),
and feature_selection steps (active_selection).
"""

from __future__ import annotations

from typing import Any

import anndata

_ANALYSIS_CONFIG_KEY = "analysis_config"

# Keys in the selection log that represent the scientifically meaningful condition axes.
# "specie" is shown separately as "Species"; only these two keys are shown under
# "Subset selected" to keep console output and figure footers concise.
_CONDITION_SELECTION_KEYS = ("diet", "condition")


def _resolve_obs_unique_values(anndata_object: anndata.AnnData, obs_column: str) -> str:
    """
    Return a string of the unique non-NaN values in .obs[obs_column].

    Used as a fallback when the selection log records "auto (only one value present)"
    instead of an explicit list — the actual value is still in .obs.

    Arguments:
        anndata_object: AnnData whose .obs is inspected.
        obs_column: Column name to look up.

    Returns:
        Unique values joined with "/" (e.g. "MNMS"), or "?" if the column is absent.
    """
    if obs_column not in anndata_object.obs.columns:
        return "?"
    unique_values = anndata_object.obs[obs_column].dropna().unique().tolist()
    return "/".join(str(v) for v in sorted(str(v) for v in unique_values))


def _format_selection_value(
    raw_value: Any, obs_column: str, anndata_object: anndata.AnnData
) -> str:
    """
    Convert a selection_log value to a display string.

    - list → join with ","
    - "all" → "all"
    - "auto (only one value present)" → read the actual value from .obs
    - any other string → return as-is

    Arguments:
        raw_value: The value stored in the selection log for this column.
        obs_column: The .obs column name (used to look up the real value when auto).
        anndata_object: AnnData to read .obs from.

    Returns:
        Human-readable string representation of the selection.
    """
    if isinstance(raw_value, list):
        return ",".join(str(v) for v in raw_value)
    if isinstance(raw_value, str) and "auto" in raw_value:
        # The column had only one unique value so select_sfc_subset skipped the prompt.
        # Look up the actual value from .obs so it is visible in the context output.
        return _resolve_obs_unique_values(anndata_object, obs_column)
    return str(raw_value)


def _extract_context_fields(anndata_object: anndata.AnnData) -> dict[str, str]:
    """
    Extract the four run-context fields from .uns["analysis_config"].

    Returns a dict with keys:
        "species"      — value of the "specie" key in the selection log
        "subset"       — formatted condition selection (diet or condition column)
        "layer"        — active normalization layer name
        "feature_sel"  — active feature selection method name

    All values fall back to "N/A" when the corresponding data is absent.
    "auto (only one value present)" entries are resolved to the actual .obs value.
    """
    analysis_config: dict[str, Any] = anndata_object.uns.get(_ANALYSIS_CONFIG_KEY, {})
    selection_log: dict[str, Any] = analysis_config.get("selection", {})

    # --- Species ---
    raw_species = selection_log.get("specie", None)
    if raw_species is None:
        species_text = "N/A"
    else:
        species_text = _format_selection_value(raw_species, "specie", anndata_object)

    # --- Condition subset (diet or condition key only) ---
    condition_parts: list[str] = []
    for key in _CONDITION_SELECTION_KEYS:
        raw_value = selection_log.get(key, None)
        if raw_value is None:
            continue
        # "all" means no filtering was applied — not informative, skip it.
        if isinstance(raw_value, str) and raw_value == "all":
            continue
        condition_parts.append(f"{key}={_format_selection_value(raw_value, key, anndata_object)}")

    subset_text = " | ".join(condition_parts) if condition_parts else "all"

    # --- Normalization layer ---
    active_layer = analysis_config.get("active_layer", None)
    layer_text = active_layer if active_layer else "raw .X"

    # --- Feature selection ---
    active_selection = analysis_config.get("active_selection", None)
    feature_sel_text = active_selection if active_selection else "None"

    return {
        "species": species_text,
        "subset": subset_text,
        "layer": layer_text,
        "feature_sel": feature_sel_text,
    }


def get_run_context_console_text(anndata_object: anndata.AnnData) -> str:
    """
    Build a multi-line console summary of the current analysis run context.

    Intended to be printed before any plot is rendered so the researcher can
    confirm exactly which data and transformations produced the figure.

    Arguments:
        anndata_object: AnnData carrying .uns["analysis_config"].

    Returns:
        A formatted multi-line string, e.g.:
            ============================================================
            PLOT CONTEXT
            ------------------------------------------------------------
              Species         : MNMS
              Subset selected : diet=IF
              Layer (norm)    : arcsinh__zscore_col
              Feature sel.    : PCALoadings
            ============================================================
    """
    fields = _extract_context_fields(anndata_object)
    separator = "=" * 60
    divider = "-" * 60
    lines = [
        separator,
        "PLOT CONTEXT",
        divider,
        f"  Species         : {fields['species']}",
        f"  Subset selected : {fields['subset']}",
        f"  Layer (norm)    : {fields['layer']}",
        f"  Feature sel.    : {fields['feature_sel']}",
        separator,
    ]
    return "\n".join(lines)


def get_species_label(anndata_object: anndata.AnnData) -> str:
    """
    Return the species string for use in auto-generated figure titles.

    Resolution order:
      1. Read ``.uns["analysis_config"]["selection"]["specie"]``.  If that
         value is a specific species name (not ``"N/A"`` or ``"all"``), use it.
      2. Otherwise fall back to reading the unique non-NaN values from
         ``.obs["specie"]`` directly (covers multi-species subsets and AnnData
         objects that were never passed through ``select_sfc_subset``).
      3. If ``.obs["specie"]`` is absent, return an empty string.

    Arguments:
        anndata_object: AnnData to inspect.

    Returns:
        - Single species: ``"MNMS"``
        - Multiple species found in ``.obs``: ``"KFish / MNMS"`` (sorted, joined with `` / ``)
        - No species information available: ``""``
    """
    species_from_config = _extract_context_fields(anndata_object)["species"]

    # Use the config value only when it is an explicit, non-generic species name.
    if species_from_config not in ("N/A", "all", ""):
        return species_from_config

    # Fall back: read unique values from .obs["specie"] directly.
    if "specie" not in anndata_object.obs.columns:
        return ""

    unique_values = sorted(
        str(v) for v in anndata_object.obs["specie"].dropna().unique()
    )
    if not unique_values:
        return ""
    return " / ".join(unique_values)


def get_run_context_footer_text(anndata_object: anndata.AnnData) -> str:
    """
    Build a compact single-line string for embedding as a figure footer.

    Intended to be placed at the bottom of matplotlib or Plotly figures so that
    exported images carry their data-provenance information.

    Arguments:
        anndata_object: AnnData carrying .uns["analysis_config"].

    Returns:
        A short string, e.g.:
            Species: MNMS | Subset: diet=IF | Layer: arcsinh__zscore_col | Feat.sel.: PCALoadings
    """
    fields = _extract_context_fields(anndata_object)
    return (
        f"Species: {fields['species']}"
        f" | Subset: {fields['subset']}"
        f" | Layer: {fields['layer']}"
        f" | Feat.sel.: {fields['feature_sel']}"
    )
