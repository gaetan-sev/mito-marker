"""
controlled_vocabulary.py

Single source of truth for all allowed values and physiological bounds for AnnData .obs
fields across all pipelines (TEM and SFC).

Rules:
- Every categorical .obs field must have its allowed values declared here as a
  dict[str, str] before being used anywhere in the codebase. Keys are the allowed
  string values; values are plain-English descriptions.
- Every numerical .obs field must have its expected unit and physiological bounds
  declared in NUMERICAL_OBS_BOUNDS.
- This file contains data only — no functions, no imports. It must remain trivially
  editable by a domain scientist who is not a programmer.
"""

# ---------------------------------------------------------------------------
# SHARED — used in both TEM and SFC contexts
# ---------------------------------------------------------------------------

ALLOWED_STAINING_VALUES: dict[str, str] = {
    "DeepRed": "MitoTracker Deep Red FM staining",
    "Unstained": "No staining applied — negative control",
}

# ---------------------------------------------------------------------------
# TEM PIPELINE
# ---------------------------------------------------------------------------


ALLOWED_CONDITIONS_TEM: dict[str, str] = {
    "Young": "Young age group",
    "Old": "Old age group",
    "Control": "Wild-type control group",
    "Double_KO_Mfn1_Mfn2": "Double knockout of Mitofusin 1 and Mitofusin 2",
    "Viking_D11": "Viking_D11",
    "Minoxidil_D11": "Minoxidil_D11",
    "Untreated_D11": "Untreated_D11",
    "Catagene_D00": "Catagene_D00",
    "Anagene_D00": "Anagene_D00",
}

# Canonical species names for TEM, in English.
# Keys are the values stored in .obs["specie"]; values are human-readable descriptions.
ALLOWED_SPECIES_TEM: dict[str, str] = {
    "KFish": "Nothobranchius furzeri — African turquoise killifish",
    "ZFish": "Danio rerio — zebrafish",
    "Mouse": "Mus musculus — house mouse",
    "Human": "Homo sapiens — human (MNMS cohort)",
    "Droso": "Drosophila melanogaster — fruit fly",
    "Worm": "c-Elegans",
}

# Maps every known filename token to the canonical species name stored in .obs.
# Keys are raw tokens as they appear in filenames (case-sensitive).
# Add new entries here when a new species or naming abbreviation is introduced.
# Sort order does not matter here — the parser sorts by key length at runtime.
TEM_SPECIES_TOKEN_TO_CANONICAL: dict[str, str] = {
    "Zebrafish": "ZFish",
    "zebrafish": "ZFish",
    "ZFish":     "ZFish",
    "KFish":     "KFish",
    "killi":     "KFish",
    "Killi":     "KFish",
    "Souris":    "Mouse",
    "souris":    "Mouse",
    "Mouse":     "Mouse",
    "mouse":     "Mouse",
    "MNMS":      "Human",
    "Droso":     "Droso",
    "Fly":       "Droso",
    "Worm":      "Worm",
}

# Species for which the condition cannot be inferred from the filename and must
# be read from the Condition_Name column inside the file.
# The parser returns condition=None (without warning) for these species;
# the builder then uses condition_name_from_file as the authoritative source.
TEM_SPECIES_CONDITION_FROM_FILE: frozenset[str] = frozenset({"Human"})

# Maps every known filename token to the canonical age group stored in .obs.
# Keys are raw tokens as they appear in filenames (case-sensitive).
# Add new entries here when a new condition naming convention is introduced.
TEM_CONDITION_TOKEN_TO_CANONICAL: dict[str, str] = {
    "Young": "Young",
    "young": "Young",
    "Old":   "Old",
    "old":   "Old",
}

# Allowed diet values for TEM .obs["diet"].
# Keys are canonical values stored in .obs; values are plain-English descriptions.
ALLOWED_DIET_TEM: dict[str, str] = {
    "AL": "Ad_Libitum — food available at all times",
    "IF": "Intermittent_Fasting — restricted feeding window",
}

# Maps every known filename token to the canonical diet value stored in .obs.
# Add new entries here when a new diet token convention is introduced.
TEM_DIET_TOKEN_TO_CANONICAL: dict[str, str] = {
    "IF": "IF",
    "if": "IF",
    "AL": "AL",
    "al": "AL",
}

# ---------------------------------------------------------------------------
# TEM PIPELINE — C. elegans gonad study (Mattout lab / EMito-Metrix, 2026)
#
# This cohort images 6 germline/somatic cell types under Control or Chemical
# Stress treatment. Species is always "Worm" (ALLOWED_SPECIES_TEM) — this
# study does not use the species axis. See
# mito-marker-lab, study 2026-celegans-gonad-mito-emitometrix (PROTOCOL.md) for the full
# data model. The raw-folder-name -> (cell_type, treatment_group) mapping is
# study-specific (one data drop's directory naming, not a reusable package
# concern) and lives in that study's own ingestion script, not here.
# ---------------------------------------------------------------------------

# Cell type — one of 6 tissues along C. elegans germline differentiation
# (Distal Gonad -> Loop -> Proximal Gonad -> Embryo -> Sperm) plus Muscle,
# the only somatic tissue used as an outgroup comparison to the germ cells.
ALLOWED_CELL_TYPES_TEM_GONAD_STUDY: dict[str, str] = {
    "Dis": "Distal Gonad — earliest germline differentiation stage",
    "Loop": "Loop — germline differentiation stage between Distal and Proximal Gonad",
    "Pro": "Proximal Gonad — most differentiated germline gonad stage",
    "Emb": "Embryo",
    "Sp": "Sperm",
    "Mu": "Muscle — somatic tissue, outgroup comparison to the germ cells above",
}

# Treatment group. "KD" (knockdown) files exist in the raw data but are
# deliberately NOT declared here — confirmed out of scope with Anna Mattout
# (2026-09-14). Add it only if a later figure set needs it.
ALLOWED_TREATMENT_GROUPS_TEM_GONAD_STUDY: dict[str, str] = {
    "Control": "Untreated control (called WT or non-endormi in some source folders)",
    "Chemical Stress": "Chemical stress via anesthesia (called Anesthesized/Anest/endormi in some source folders)",
}

# Display order confirmed with Anna Mattout (2026-09-14) — matches germ cell
# differentiation order (Dis -> Loop -> Pro), then Embryo, Sperm, and Muscle
# last (the only somatic/outgroup tissue). Use this order for every figure
# showing all cell types side by side (violin plots, density plots).
CELL_TYPE_ORDER_GONAD_STUDY: list[str] = ["Dis", "Loop", "Pro", "Emb", "Sp", "Mu"]

# Canonical species names ordered from most phylogenetically distant to closest
# relative of Homo sapiens.  Use this fixed order when displaying species side
# by side (e.g. heatmap columns, dendrogram) so that the layout reflects
# evolutionary distance rather than alphabetical or data-entry order.
TEM_SPECIES_PHYLOGENETIC_ORDER: list[str] = [
    "Worm",    # C-Elegans — Invertebrate
    "Droso",   # Drosophila melanogaster — invertebrate
    "ZFish",   # Danio rerio — teleost fish
    "KFish",   # Nothobranchius furzeri — teleost fish (closer to tetrapods than ZFish)
    "Mouse",   # Mus musculus — rodent mammal
    "Human",   # Homo sapiens — reference species
]

# ---------------------------------------------------------------------------
# TEM PIPELINE — species divergence times (reference phylogeny)
#
# Manually verified against timetree.org (Kumar et al., TimeTree 5) on
# 2026-08-26 — see git commit "manual verif and update of
# TEM_SPECIES_DIVERGENCE_TIME_MYA". TODO: the Methods section still needs a
# formal citation (query date, exact node IDs used) before publication —
# "values from timetree.org" is not yet a citable reference on its own.
#
# Symmetric matrix of pairwise divergence times in millions of years (Mya):
# TEM_SPECIES_DIVERGENCE_TIME_MYA[a][b] is the estimated time since the last
# common ancestor of species a and b. The diagonal is zero by definition.
#
# The tree these numbers encode:
#
#     ((Worm, Droso):572, ((ZFish, KFish):224, (Mouse, Human):87):429):708
#
#   Human  – Mouse                          87
#   ZFish  – KFish                         224
#   (Human, Mouse) – (ZFish, KFish)        429
#   Worm   – Droso                         572
#   (Worm, Droso) – all vertebrates        708
#
# The Ecdysozoa node (Worm–Droso, 572) and the bilaterian root
# (protostomes–deuterostomes, 708) are 136 Mya apart. That is a clear
# separation, not a near-polytomy — with the provisional placeholder values
# this project started from (682 / 685 Mya, ~3 Mya apart) that node pair WAS
# visually indistinguishable on a time-scaled tree, which is why
# build_species_phylogeny_linkage() offers branch_scale="cladogram" (topology
# only, equal branch lengths). With the sourced values above,
# branch_scale="time" already renders both nodes distinctly, so
# "cladogram" is now a readability/style option rather than a necessity —
# kept available for either purpose.
#
# Divergence times are ULTRAMETRIC (every leaf sits at the same distance
# from the root, because every living species has had the same amount of
# time to evolve). That property is what lets scipy's average-linkage
# (UPGMA) reproduce this tree exactly — topology AND node heights — from the
# matrix alone, with no phylogenetics dependency. If an edit breaks
# ultrametricity, the linkage silently produces a DIFFERENT tree, so the
# matrix is checked by assert_ultrametric_divergence_times() in
# analysis/phylogeny.py before every use.
# ---------------------------------------------------------------------------
TEM_SPECIES_DIVERGENCE_TIME_MYA: dict[str, dict[str, float]] = {
    "Worm":  {"Worm":   0.0, "Droso": 572.0, "ZFish": 708.0, "KFish": 708.0, "Mouse": 708.0, "Human": 708.0},
    "Droso": {"Worm": 572.0, "Droso":   0.0, "ZFish": 708.0, "KFish": 708.0, "Mouse": 708.0, "Human": 708.0},
    "ZFish": {"Worm": 708.0, "Droso": 708.0, "ZFish":   0.0, "KFish": 224.0, "Mouse": 429.0, "Human": 429.0},
    "KFish": {"Worm": 708.0, "Droso": 708.0, "ZFish": 224.0, "KFish":   0.0, "Mouse": 429.0, "Human": 429.0},
    "Mouse": {"Worm": 708.0, "Droso": 708.0, "ZFish": 429.0, "KFish": 429.0, "Mouse":   0.0, "Human":  87.0},
    "Human": {"Worm": 708.0, "Droso": 708.0, "ZFish": 429.0, "KFish": 429.0, "Mouse":  87.0, "Human":   0.0},
}

# The three clades the reference phylogeny above predicts. Used by
# compute_tanglegram_sensitivity() and compute_clade_support() to report,
# clade by clade, whether the data-derived morphology tree recovered them.
# Each entry is a frozenset so membership comparison is order-independent.
TEM_EXPECTED_PHYLOGENETIC_CLADES: dict[str, frozenset] = {
    "(Worm,Droso)": frozenset({"Worm", "Droso"}),
    "(ZFish,KFish)": frozenset({"ZFish", "KFish"}),
    "(Mouse,Human)": frozenset({"Mouse", "Human"}),
}

# All columns that must be present in every TEM .txt file (tab-separated).
# Mito_ID is a per-row identifier present in the file but NOT saved to .obs or .X.
# The 4 file-level metadata columns (Experiment_Name, Condition_Name, Image_Name, Mito_ID)
# are listed first, followed by the 28 morphological feature columns.
TEM_EXPECTED_FILE_COLUMNS: list[str] = [
    # File-level metadata — present in every row but describe the image, not the mitochondrion
    "Experiment_Name",
    "Condition_Name",
    "Image_Name",
    "Mito_ID",
    # 28 morphological features → .X
    "Mito_Area",
    "Mito_Perimeter",
    "AreaPerimeter_Ratio",
    "Mito_MeanInt",
    "Mito_MeanInt_CORR",
    "Mito_MedianInt",
    "Mito_MedianInt_CORR",
    "Mito_TotalInt",
    "Mito_TotalInt_CORR",
    "Intensity_SD",
    "Intensity_SD_CORR",
    "Intensity_SD_percent",
    "Intensity_SD_percent_CORR",
    "Mito_CentroidX",
    "Mito_CentroidY",
    "Mito_Circularity",
    "Mito_Roundness",
    "Mito_Solidity",
    "Mito_AR",
    "Mito_Feret_Diameter",
    "Mito_FeretX",
    "Mito_FeretY",
    "Skewness",
    "Kurtosis",
    "CristaeOrientation_Major",
    "CristaeOrientation_Minor",
    "CristaeOrientation_Angle",
    "CristaeOrientation_Area",
]

# The 28 morphological feature columns that are loaded into .X.
# Derived from TEM_EXPECTED_FILE_COLUMNS by excluding all metadata columns.
_TEM_METADATA_COLUMNS: set[str] = {
    "Experiment_Name", "Condition_Name", "Image_Name", "Mito_ID"
}
TEM_FEATURE_COLUMNS: list[str] = [
    col for col in TEM_EXPECTED_FILE_COLUMNS if col not in _TEM_METADATA_COLUMNS
]

# Features present in the raw TEM file that are stored in .X but carry no
# morphological information relevant for condition comparison:
#   - Mito_CentroidX/Y: spatial coordinates inside the image — depend on
#     where the mitochondrion happens to sit in the TEM field of view, not
#     on the biological condition.
#   - Mito_FeretX/Y: anchor coordinates of the Feret diameter measurement —
#     redundant with Mito_Feret_Diameter and equally location-dependent.
# These are stamped as is_non_analytical=True in .var at ingestion time so
# that all downstream analysis functions (PCA, UMAP, feature selection, radar)
# automatically exclude them without requiring per-function drop_cols logic.
TEM_NON_ANALYTICAL_FEATURES: list[str] = [
    "Mito_CentroidX",
    "Mito_CentroidY",
    "Mito_FeretX",
    "Mito_FeretY",
]

# ---------------------------------------------------------------------------
# TEM PIPELINE — biological feature categories (single source of truth)
#
# TEM_FEATURE_SUBSETS groups every ANALYTICAL TEM feature into one of four
# biological categories. This is THE canonical categorization of TEM features
# for the whole package — three different figures/analyses read it:
#   - run_feature_subset_challenge()  (analysis/ml_pipeline.py) — compares how
#     much predictive signal each category carries on its own.
#   - plot_radar()                    (analysis/radar_plot.py) — groups the
#     radar spokes into contiguous category blocks, colors the spoke labels
#     and the category legend by category, and draws a boundary line between
#     blocks.
#   - plot_phylo_tanglegram()         (analysis/phylo_tanglegram.py) — orders
#     the heatmap columns into category blocks and draws the colored category
#     strip above them.
#
# The four categories, ordered from the outside of the mitochondrion inwards:
#   - "Size"                — scale-dependent measurements (in pixels or
#     pixels²). Double the magnification and these values change.
#   - "Shape"               — dimensionless ratios of the outline. Unchanged
#     by magnification.
#   - "Intensity"           — electron density of the matrix and how it is
#     distributed. Corrected variants (_CORR) account for background signal.
#   - "Cristae Orientation" — orientation and extent of the inner-membrane
#     folds (the internal architecture of the mitochondrion).
#
# "Size" + "Shape" together are what used to be a single "Morphology" category.
# They are kept apart because in a cross-species comparison absolute size and
# outline shape are expected to behave very differently.
#
# Non-analytical spatial features (Mito_CentroidX/Y, Mito_FeretX/Y, listed in
# TEM_NON_ANALYTICAL_FEATURES) are deliberately ABSENT — they are image
# coordinates, not morphology, and must never be assigned a category.
#
# Typical usage:
#   from mito_marker.controlled_vocabulary import TEM_FEATURE_SUBSETS
#   results = run_feature_subset_challenge(tem_anndata, TEM_FEATURE_SUBSETS, ml_config=config)
TEM_FEATURE_SUBSETS: dict[str, list[str]] = {
    # Size — scale-dependent measurements. Double the magnification and these
    # values change.
    "Size": [
        "Mito_Area",
        "Mito_Perimeter",
        "AreaPerimeter_Ratio",
        "Mito_Feret_Diameter",
    ],
    # Shape — dimensionless ratios of the outline. Unchanged by magnification.
    "Shape": [
        "Mito_Circularity",
        "Mito_Roundness",
        "Mito_Solidity",
        "Mito_AR",
    ],
    # Signal intensity and its distribution within the mitochondrion.
    # Corrected variants (_CORR) account for background fluorescence.
    "Intensity": [
        "Mito_MeanInt",
        "Mito_MeanInt_CORR",
        "Mito_MedianInt",
        "Mito_MedianInt_CORR",
        "Mito_TotalInt",
        "Mito_TotalInt_CORR",
        "Intensity_SD",
        "Intensity_SD_CORR",
        "Intensity_SD_percent",
        "Intensity_SD_percent_CORR",
        "Skewness",
        "Kurtosis",
    ],
    # Cristae (inner mitochondrial membrane folds) orientation parameters.
    # These capture the internal architecture of the mitochondrion.
    "Cristae Orientation": [
        "CristaeOrientation_Major",
        "CristaeOrientation_Minor",
        "CristaeOrientation_Angle",
        "CristaeOrientation_Area",
    ],
}

# Display order of the categories wherever they appear side by side (radar
# spoke blocks + legend, tanglegram column strip). Derived from
# TEM_FEATURE_SUBSETS insertion order so it can never drift out of sync.
TEM_FEATURE_CATEGORY_ORDER: list[str] = list(TEM_FEATURE_SUBSETS)

# Inverse of TEM_FEATURE_SUBSETS: {feature_name: category}. Derived — edit
# TEM_FEATURE_SUBSETS, never this. Consumers look a feature up here and fall
# back to the string "Other" when it is missing (a feature added to
# TEM_FEATURE_COLUMNS but not to TEM_FEATURE_SUBSETS); "Other" is never an
# actual value in this dict.
TEM_FEATURE_CATEGORIES: dict[str, str] = {
    feature_name: category
    for category, feature_names in TEM_FEATURE_SUBSETS.items()
    for feature_name in feature_names
}

# One hex color per category — the single customization point for category
# colors in every figure that groups TEM features (radar spoke labels +
# category legend, tanglegram category strip). Same convention as
# PREFERRED_CONDITION_COLORS below (standard colors reference in that block).
# Keys MUST match TEM_FEATURE_SUBSETS; "Other" is the fallback for an
# uncategorised feature.
#
# To change one from a notebook, mutate this dict in place (do not reassign
# the name) — the plotting modules read it by reference at render time, so the
# change takes effect on the next call without a package reload:
#     from mito_marker.controlled_vocabulary import TEM_FEATURE_CATEGORY_COLORS
#     TEM_FEATURE_CATEGORY_COLORS["Size"] = "#123456"
TEM_FEATURE_CATEGORY_COLORS: dict[str, str] = {
    "Size": "#377eb8",                 # blue
    "Shape": "#4daf4a",                # green
    "Intensity": "#ff7f00",            # orange
    "Cristae Orientation": "#984ea3",  # purple
    "Other": "#999999",                # gray — fallback for an uncategorised feature
}

# ---------------------------------------------------------------------------
# TEM PIPELINE — radar plot spoke-label abbreviations (plot_radar(), TEM only)
#
# plot_radar() draws every analytical TEM feature as one spoke on a single
# radar figure. Spokes are grouped into contiguous TEM_FEATURE_CATEGORY_ORDER
# blocks and colored by TEM_FEATURE_CATEGORY_COLORS (both above). This dict
# only shortens the long raw feature names so the horizontal spoke labels
# (read left-to-right, not rotated along the spoke) stay readable at the ~24
# TEM feature count.
#
# The "Mito_" prefix is dropped everywhere (every TEM feature describes a
# mitochondrion, so the prefix carries no information) and long names are
# shortened further. A feature missing from this dict still renders — as its
# name with "Mito_" stripped — so keep this in sync with TEM_FEATURE_COLUMNS.
# ---------------------------------------------------------------------------
TEM_RADAR_FEATURE_ABBREVIATIONS: dict[str, str] = {
    "Mito_Area": "Area",
    "Mito_Perimeter": "Perimeter",
    "AreaPerimeter_Ratio": "Area/Perim.",
    "Mito_Circularity": "Circularity",
    "Mito_Roundness": "Roundness",
    "Mito_Solidity": "Solidity",
    "Mito_AR": "Aspect Ratio",
    "Mito_Feret_Diameter": "Feret Diam.",
    "Mito_MeanInt": "Mean Int.",
    "Mito_MeanInt_CORR": "Mean Int. (corr)",
    "Mito_MedianInt": "Median Int.",
    "Mito_MedianInt_CORR": "Median Int. (corr)",
    "Mito_TotalInt": "Total Int.",
    "Mito_TotalInt_CORR": "Total Int. (corr)",
    "Intensity_SD": "Int. SD",
    "Intensity_SD_CORR": "Int. SD (corr)",
    "Intensity_SD_percent": "Int. SD %",
    "Intensity_SD_percent_CORR": "Int. SD % (corr)",
    "Skewness": "Skewness",
    "Kurtosis": "Kurtosis",
    "CristaeOrientation_Major": "Cristae Major",
    "CristaeOrientation_Minor": "Cristae Minor",
    "CristaeOrientation_Angle": "Cristae Angle",
    "CristaeOrientation_Area": "Cristae Area",
}

# ---------------------------------------------------------------------------
# SFC PIPELINE — channel prefix patterns for feature subset challenge
# ---------------------------------------------------------------------------

# Prefixes that identify morphological scatter channels (cell size and granularity).
# Examples: FSC-A, FSC-H, FSC-W, SSC-A, SSC-B-A, SSC-H, SSC-W.
SFC_MORPHO_CHANNEL_PREFIXES: list[str] = ["FSC", "SSC"]

# Prefixes that identify spectral fluorescence channels.
# Examples: UV1-A, V1-A, B1-A, YG1-A, R1-A.
SFC_SPECTRAL_CHANNEL_PREFIXES: list[str] = ["UV", "V", "B", "YG", "R"]

# ---------------------------------------------------------------------------
# SFC PIPELINE — categorical .obs fields
# ---------------------------------------------------------------------------


ALLOWED_SPECIES_SFC: dict[str, str] = {
    "MNMS": "Human cohort MNMS",
    "WORM": "Caenorhabditis elegans",
    "WORMS": "Caenorhabditis elegans",  # alias for WORM — both tokens appear in filenames
    "FLY": "Drosophila melanogaster",
    "MOUSE": "Mus musculus — house mouse",
}

# Age group field (.obs["age_group"]) — orthogonal to treatment.
# Allows grouping all Young or all Old subjects regardless of treatment.
ALLOWED_AGE_GROUPS_SFC: dict[str, str] = {
    "Young": "Young age group",
    "Old": "Old age group",
}

# Treatment field (.obs["treatment"]) — orthogonal to age group.
# Allows grouping all subjects sharing a treatment regardless of age.
ALLOWED_TREATMENTS_SFC: dict[str, str] = {
    "Vehicule": "Vehicle control — no active compound",
    "Arac15": "Arachidonic acid 15 mg/kg treatment",
    "Arac30": "Arachidonic acid 30 mg/kg treatment",
    "Pilot": "Pilot experiment — no specific treatment protocol",
}

# Maps raw filename tokens (any case) to their canonical age_group value.
AGE_GROUP_TOKEN_TO_CANONICAL_SFC: dict[str, str] = {
    "Young": "Young",
    "young": "Young",
    "YOUNG": "Young",
    "Old":   "Old",
    "old":   "Old",
    "OLD":   "Old",
}

# Maps raw filename tokens (any case) to their canonical treatment value.
TREATMENT_TOKEN_TO_CANONICAL_SFC: dict[str, str] = {
    "Vehicule": "Vehicule",
    "vehicule": "Vehicule",
    "VEHICULE": "Vehicule",
    "Arac15":   "Arac15",
    "arac15":   "Arac15",
    "ARAC15":   "Arac15",
    "Arac30":   "Arac30",
    "arac30":   "Arac30",
    "ARAC30":   "Arac30",
    "Pilot":    "Pilot",
    "pilot":    "Pilot",
    "PILOT":    "Pilot",
}

ALLOWED_DIET_SFC: dict[str, str] = {
    "AL": "Ad_Libitum — food available at all times",
    "IF": "Intermittent_Fasting — restricted feeding window",
}

ALLOWED_DILUTION_SFC: dict[str, str] = {
    "Diluted": "Sample was diluted before acquisition (default when ND is absent from filename)",
    "Not_Diluted": "Sample was not diluted — ND token present in filename",
}

ALLOWED_MARKERS_SFC: dict[str, str] = {
    "MtDeepRed": "MitoTracker Deep Red FM — mitochondrial membrane potential dye",
    "TMRM": "Tetramethylrhodamine methyl ester — mitochondrial membrane potential dye",
    "Not_Marked": "No fluorescent marker applied — unstained or marker-free acquisition",
}

# Maps raw filename tokens to their canonical marker value stored in .obs.
#  'DR' and 'DeepRed' and 'DeepRed+' in a filename resolve to the canonical value 'MtDeepRed'.
# 'NM' resolves to 'Not_Marked' (no marker applied).
# Add new entries here when a new marker token convention is introduced.
MARKER_TOKEN_TO_CANONICAL_SFC: dict[str, str] = {
    "DR": "MtDeepRed",
    "DeepRed": "MtDeepRed",
    "DeepRed+": "MtDeepRed",
    "TMRM": "TMRM",
    "NM": "Not_Marked",
}

# Canonical order used when joining multiple marker tokens into a single string.
# Example: ["TMRM", "DeepRed"] → sorted to ["TMRM", "MtDeepRed"] → "TMRM+MtDeepRed".
SFC_MARKER_JOIN_ORDER: list[str] = ["TMRM", "MtDeepRed", "Not_Marked"]

# Filename tokens whose presence indicates that FlowAI quality control passed.
# FlowAI_Pass is set to True if any of these tokens appears in the filename tokens.
FLOWAI_PASS_TOKENS: set[str] = {"good", "events", "FlowAIGoodEvents"}

# ---------------------------------------------------------------------------
# SFC PIPELINE — cytometry channel lists (Cytek Aurora 5-laser instrument)
#
# The Cytek Aurora exports two sets of per-detector channels:
#   - Raw channels  : named by laser/detector position (UV1-A, B1-A, R1-A, …)
#   - FJComp channels: FlowJo-compensated duplicates, prefixed with 'FJComp-'
#
# Only the raw channels are used for analysis. FJComp channels are always excluded.
# ---------------------------------------------------------------------------

# Explicit list of channels to RETAIN, in canonical spectral order.
# This list serves two purposes:
#   1. Defines WHICH channels to keep (the complement of SFC_CHANNELS_TO_EXCLUDE).
#   2. Defines the COLUMN ORDER of .X in the AnnData — always FSC/SSC → UV → V → B → YG → R.
# The reordering in fcs_loader.py sorts retained channels to match their position here.
#
# Each detector exports -A (area) and -H (height) pulses. A small number of detectors
# also export -W (width): UV8, V8, B7, YG5, R4. These are determined by the instrument
# and cannot be changed without re-exporting from the acquisition software.
# Total: 3 FSC + 6 SSC + 33 UV + 33 V + 29 B + 21 YG + 17 R = 142 channels.
# Time and FlowAI are excluded entirely (see SFC_CHANNELS_TO_EXCLUDE).
SFC_CHANNELS_TO_KEEP: list[str] = [
    # Forward and side scatter (no laser)
    "FSC-A", "FSC-H", "FSC-W",
    "SSC-A",   "SSC-H",   "SSC-W",        # Main side scatter detector
    "SSC-B-A", "SSC-B-H", "SSC-B-W",      # Second side scatter detector (blue laser)
    # UV laser (355 nm) — 16 detectors; UV8 also exports -W
    "UV1-A",  "UV1-H",
    "UV2-A",  "UV2-H",
    "UV3-A",  "UV3-H",
    "UV4-A",  "UV4-H",
    "UV5-A",  "UV5-H",
    "UV6-A",  "UV6-H",
    "UV7-A",  "UV7-H",
    "UV8-A",  "UV8-H",  "UV8-W",
    "UV9-A",  "UV9-H",
    "UV10-A", "UV10-H",
    "UV11-A", "UV11-H",
    "UV12-A", "UV12-H",
    "UV13-A", "UV13-H",
    "UV14-A", "UV14-H",
    "UV15-A", "UV15-H",
    "UV16-A", "UV16-H",
    # Violet laser (405 nm) — 16 detectors; V8 also exports -W
    "V1-A",   "V1-H",
    "V2-A",   "V2-H",
    "V3-A",   "V3-H",
    "V4-A",   "V4-H",
    "V5-A",   "V5-H",
    "V6-A",   "V6-H",
    "V7-A",   "V7-H",
    "V8-A",   "V8-H",   "V8-W",
    "V9-A",   "V9-H",
    "V10-A",  "V10-H",
    "V11-A",  "V11-H",
    "V12-A",  "V12-H",
    "V13-A",  "V13-H",
    "V14-A",  "V14-H",
    "V15-A",  "V15-H",
    "V16-A",  "V16-H",
    # Blue laser (488 nm) — 14 detectors; B7 also exports -W
    "B1-A",   "B1-H",
    "B2-A",   "B2-H",
    "B3-A",   "B3-H",
    "B4-A",   "B4-H",
    "B5-A",   "B5-H",
    "B6-A",   "B6-H",
    "B7-A",   "B7-H",   "B7-W",
    "B8-A",   "B8-H",
    "B9-A",   "B9-H",
    "B10-A",  "B10-H",
    "B11-A",  "B11-H",
    "B12-A",  "B12-H",
    "B13-A",  "B13-H",
    "B14-A",  "B14-H",
    # Yellow-Green laser (561 nm) — 10 detectors; YG5 also exports -W
    "YG1-A",  "YG1-H",
    "YG2-A",  "YG2-H",
    "YG3-A",  "YG3-H",
    "YG4-A",  "YG4-H",
    "YG5-A",  "YG5-H",  "YG5-W",
    "YG6-A",  "YG6-H",
    "YG7-A",  "YG7-H",
    "YG8-A",  "YG8-H",
    "YG9-A",  "YG9-H",
    "YG10-A", "YG10-H",
    # Red laser (638 nm) — 8 detectors; R4 also exports -W
    "R1-A",   "R1-H",
    "R2-A",   "R2-H",
    "R3-A",   "R3-H",
    "R4-A",   "R4-H",   "R4-W",
    "R5-A",   "R5-H",
    "R6-A",   "R6-H",
    "R7-A",   "R7-H",
    "R8-A",   "R8-H",
]

# Channels that must NEVER be used in analytical computations (feature selection,
# PCA, UMAP, ML).  They may remain in .var for inspection purposes, but every
# analysis function must exclude them before scoring or fitting.
#
# - "Time"    : acquisition timestamp — reflects injection speed, not biology.
# - "FlowAI"  : quality score assigned by the FlowAI QC tool — not a cytometry
#               measurement; its numerical value is meaningless as a feature.
SFC_NON_ANALYTICAL_CHANNELS: list[str] = [
    "Time",
    "FlowAI",
]

# Channels to EXCLUDE 
# Never used in analysis and must be dropped from .X.
# — FlowJo spectral compensation outputs (FJComp-* prefix).
# These are redundant duplicates of the raw channels with compensation applied by FlowJo.
# Count: 16 UV + 16 V + 14 B + 10 YG + 8 R = 64 channels.
SFC_CHANNELS_TO_EXCLUDE: list[str] = [
    # UV laser FJComp channels (16)
    "FJComp-UV1-A",  "FJComp-UV2-A",  "FJComp-UV3-A",  "FJComp-UV4-A",
    "FJComp-UV5-A",  "FJComp-UV6-A",  "FJComp-UV7-A",  "FJComp-UV8-A",
    "FJComp-UV9-A",  "FJComp-UV10-A", "FJComp-UV11-A", "FJComp-UV12-A",
    "FJComp-UV13-A", "FJComp-UV14-A", "FJComp-UV15-A", "FJComp-UV16-A",
    # Violet laser FJComp channels (16)
    "FJComp-V1-A",   "FJComp-V2-A",   "FJComp-V3-A",   "FJComp-V4-A",
    "FJComp-V5-A",   "FJComp-V6-A",   "FJComp-V7-A",   "FJComp-V8-A",
    "FJComp-V9-A",   "FJComp-V10-A",  "FJComp-V11-A",  "FJComp-V12-A",
    "FJComp-V13-A",  "FJComp-V14-A",  "FJComp-V15-A",  "FJComp-V16-A",
    # Blue laser FJComp channels (14)
    "FJComp-B1-A",   "FJComp-B2-A",   "FJComp-B3-A",   "FJComp-B4-A",
    "FJComp-B5-A",   "FJComp-B6-A",   "FJComp-B7-A",   "FJComp-B8-A",
    "FJComp-B9-A",   "FJComp-B10-A",  "FJComp-B11-A",  "FJComp-B12-A",
    "FJComp-B13-A",  "FJComp-B14-A",
    # Yellow-Green laser FJComp channels (10)
    "FJComp-YG1-A",  "FJComp-YG2-A",  "FJComp-YG3-A",  "FJComp-YG4-A",
    "FJComp-YG5-A",  "FJComp-YG6-A",  "FJComp-YG7-A",  "FJComp-YG8-A",
    "FJComp-YG9-A",  "FJComp-YG10-A",
    # Red laser FJComp channels (8)
    "FJComp-R1-A",   "FJComp-R2-A",   "FJComp-R3-A",   "FJComp-R4-A",
    "FJComp-R5-A",   "FJComp-R6-A",   "FJComp-R7-A",   "FJComp-R8-A",
    # Other useless columns
    "Event #",
    # Time is a non-analytical acquisition timestamp — not a biological measurement.
    # Some cytometers omit it entirely, so excluding it here ensures a consistent
    # channel set across all files regardless of instrument configuration.
    "Time",
    # FlowAI is a quality score assigned by the FlowAI QC tool — not a cytometry
    # measurement. Its numerical value is meaningless as a biological feature.
    "FlowAI",
]

# Spectral group definitions in canonical display order.
# Maps each group label to the channel name prefix(es) that identify its members.
# Used by qc.py to count how many channels belong to each laser/detector group.
# The reordering in fcs_loader.py uses SFC_CHANNELS_TO_KEEP directly (not these prefixes).
# Note: prefixes are checked with str.startswith() — they are unambiguous for the
# Cytek Aurora naming convention (e.g. "V" only matches V1-A…V16-A, not "Viability-…").
SFC_SPECTRAL_GROUP_ORDER: dict[str, list[str]] = {
    "FSC_SSC": ["FSC-", "SSC-"],  # Scatter channels (no laser)
    "UV":      ["UV"],             # UV laser 355 nm, 16 detectors
    "V":       ["V"],              # Violet laser 405 nm, 16 detectors
    "B":       ["B"],              # Blue laser 488 nm, 14 detectors
    "YG":      ["YG"],             # Yellow-Green laser 561 nm, 10 detectors
    "R":       ["R"],              # Red laser 638 nm, 8 detectors
}

# ---------------------------------------------------------------------------
# SFC PIPELINE — FCS file metadata .obs fields
#
# These fields are extracted from the FCS TEXT segment (instrument metadata)
# and stored as .obs columns alongside the filename-parsed fields. They are
# broadcast to every event row since they describe the acquisition file, not
# individual events.
#
# Unlike categorical fields above, these are free-form strings sourced directly
# from the instrument — no allowed-values dictionary applies. They may be absent
# in some FCS files (e.g., older formats), in which case pd.NA is stored.
# ---------------------------------------------------------------------------

FCS_OBS_METADATA_FIELDS: dict[str, str] = {
    "fcs_acquisition_date": "Acquisition date of the FCS file, from TEXT field $DATE",
    "fcs_tube_name": "Sample tube name, from TEXT field TUBENAME (vendor extension, may be absent)",
    "fcs_volume_uL": "Volume of sample acquired in µL, from TEXT field $Vol (may be absent)",
    "fcs_total_events": "Total number of events recorded in the FCS file, from TEXT field $TOT",
}

# ---------------------------------------------------------------------------
# NUMERICAL .obs BOUNDS — both pipelines
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# MNMS CLINICAL CSV — metadata for enrich_with_clinical_data()
#
# These constants describe the format of the MNMS_Cleaned_Clinical_Dataset.csv
# file used to merge per-subject clinical variables into the SFC AnnData .obs.
# They are imported by sfc/clinical_data.py.
#
# MNMS_CLINICAL_CSV_SUBJECT_ID_COLUMN  : Name of the CSV index column.
# MNMS_CLINICAL_CSV_SUBJECT_ID_PATTERN : Regex to extract the numeric ID from
#     index values like "MNMS 005". The first capture group must yield the digits.
# MNMS_CLINICAL_CSV_MISSING_VALUE_CODE : Sentinel float used in the CSV to encode
#     missing measurements. Converted to NaN before merging into .obs.
# ---------------------------------------------------------------------------

MNMS_CLINICAL_CSV_SUBJECT_ID_COLUMN: str = "MNMS"
MNMS_CLINICAL_CSV_SUBJECT_ID_PATTERN: str = r"(\d+)"
MNMS_CLINICAL_CSV_MISSING_VALUE_CODE: float = -1.0

# ---------------------------------------------------------------------------
# ANALYSIS PIPELINE — interactive subset selection
#
# Lists the .obs columns offered to the user during select_sfc_subset().
# Order matters: subject_ID is always presented first.
# Only include columns that carry meaningful biological or batch-related
# information. Instrument metadata columns (fcs_volume_uL, fcs_total_events,
# source_filename) are intentionally excluded.
# ---------------------------------------------------------------------------

FILTERABLE_OBS_COLUMNS: list[str] = [
    "specie",               # biological species — shown first
    "unique_subject_ID",    # cross-species unique subject key (specie_subjectID) — shown second
    "condition",            # experimental condition (Young/Old for TEM, Control/KO, etc.)
    "marker",               # fluorescent marker applied (MtDeepRed / Not_Marked)
    "dilution",             # sample dilution status (Diluted / Not_Diluted)
    "diet",                 # diet regimen (AL / IF)
    "FlowAI_Pass",          # FlowAI quality control result (True / False)
    "age",                  # numerical age in the unit encoded in the filename
    "fcs_acquisition_date", # acquisition date — useful for detecting batch effects
]

# ---------------------------------------------------------------------------
# ANALYSIS PIPELINE — condition color palette overrides
#
# Pre-specify hex colors for condition values to override the automatic Set2
# palette assigned by assign_color_palette(). Leave all lines commented out
# to use the automatic palette.
#
# Standard colors reference:
#   Red          "#e41a1c"
#   Blue         "#377eb8"
#   Green        "#4daf4a"
#   Purple       "#984ea3"
#   Orange       "#ff7f00"
#   Yellow       "#ffff33"
#   Brown        "#a65628"
#   Pink         "#f781bf"
#   Gray         "#999999"
#   Teal         "#17becf"
# ---------------------------------------------------------------------------

PREFERRED_CONDITION_COLORS: dict[str, str] = {
    # --- Species colors (Wong 2011 colorblind-safe palette) ---
    # Assigned in phylogenetic order from most distant to closest relative of Homo sapiens.

    # JPP Palette :
    "Worm":   "#C1699B",  #
    "Droso":  "#58ACE0",  # 
    "ZFish":  "#009E73",  # 
    "KFish":  "#E69F00",  # 
    "Mouse":  "#0072B2",  # 
    "Human":  "#F0E442",  # 


    # --- Experimental condition colors ---
    "AL":     "#4daf4a",  # Ad Libitum
    "IF":     "#984ea3",  # Intermittent Fasting
    "Young":  "#2410c0",  # Young age group
    "Old":    "#a90e15",  # Old age group

    # --- C. elegans gonad study (Mattout lab, 2026) — CELL_TYPE_ORDER_GONAD_STUDY.
    # Proposed palette, not yet run through a colorblind simulator (see
    # PROTOCOL.md §2.1 before locking in for publication). Dis/Loop/Pro share
    # one magenta family, light -> dark = germ cell differentiation
    # progression (per Anna Mattout's request). Emb/Sp get distinct singleton
    # hues. Mu is near-black — deliberately the strongest possible contrast,
    # and the one choice whose distinctiveness relies on lightness rather
    # than hue, so it survives every common type of color vision deficiency.
    "Dis":  "#F2A6C6",
    "Loop": "#C94F91",
    "Pro":  "#7A1F5C",
    "Emb":  "#1B9AAA",
    "Sp":   "#D9A404",
    "Mu":   "#2B2B2B",
}

# ---------------------------------------------------------------------------
# NUMERICAL .obs BOUNDS — both pipelines
# ---------------------------------------------------------------------------

NUMERICAL_OBS_BOUNDS: dict[str, dict[str, float | str]] = {
    "insulin_level_uU_per_mL": {
        "unit": "µU/mL",
        "min": 0.0,
        "max": 300.0,
    },
    "steps_per_day": {
        "unit": "steps",
        "min": 0.0,
        "max": 100_000.0,
    },
    "age_days": {
        "unit": "days",
        "min": 0,
        "max": 1000,
    },
    "age_months": {
        "unit": "days",
        "min": 0,
        "max": 1000,
    },
    "age_years": {
        "unit": "days",
        "min": 0,
        "max": 200,
    },
    "body_weight_kilos": {
        "unit": "kilo",
        "min": 0,
        "max": 200,
    },
}

# ---------------------------------------------------------------------------
# ANALYSIS PIPELINE — feature-value subsets (analysis/feature_subset.py)
#
# Modes accepted by subset_by_feature_values(). Every mode keeps a subset of
# mitochondria based on the value of ONE feature (a .var column of raw .X, or
# a numeric .obs column).
# ---------------------------------------------------------------------------

ALLOWED_FEATURE_SUBSET_MODES: dict[str, str] = {
    "lowest_fraction": "Keep the given fraction of mitochondria with the LOWEST values (e.g. 25% smallest)",
    "highest_fraction": "Keep the given fraction of mitochondria with the HIGHEST values (e.g. 8% largest)",
    "between": "Keep mitochondria whose value lies between lower_value and upper_value",
    "below": "Keep mitochondria whose value is below upper_value",
    "above": "Keep mitochondria whose value is above lower_value",
}

# ---------------------------------------------------------------------------
# ANALYSIS PIPELINE — reusable cluster models (analysis/cluster_model.py)
#
# A cluster model is fitted once on a training dataset (fit) and re-applied,
# without refitting, to any other dataset projected in the same frozen
# reference space (predict). See ADR-015 in docs/DECISIONS.md.
# ---------------------------------------------------------------------------

ALLOWED_CLUSTERING_ALGORITHMS: dict[str, str] = {
    "kmeans": "K-means — hard assignment to the nearest centroid, k chosen by silhouette",
    "gmm": "Gaussian mixture model — soft assignment (membership probability), k chosen by BIC",
}

ALLOWED_CLUSTERING_SPACES: dict[str, str] = {
    "scaled": "Active normalized layer (e.g. z-score per feature), analytical features only",
    "pca": "First principal components of .obsm['X_pca'] (fitted on the reference dataset)",
    "umap": "Active UMAP coordinates (fitted on the reference dataset) — distorts distances, use with care",
}

# .obs column holding the cluster label of each mitochondrion: "cluster__<model_name>".
CLUSTER_OBS_COLUMN_PREFIX: str = "cluster__"
# .obs column holding the GMM membership probability of the assigned cluster.
CLUSTER_PROBABILITY_OBS_COLUMN_PREFIX: str = "cluster_probability__"
# Cluster label values: "Cluster_1", "Cluster_2", … ordered by decreasing training size.
CLUSTER_LABEL_PREFIX: str = "Cluster_"
# Model names end up in .obs column names and file names: keep them simple.
CLUSTER_MODEL_NAME_PATTERN: str = r"^[A-Za-z0-9_-]+$"
