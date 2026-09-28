# mito-marker

A Python package for loading, validating, and exploring mitochondrial morphology biomarker data from two experimental pipelines: **Spectral Flow Cytometry (SFC)** and **Transmission Electron Microscopy (TEM)**.

> **Note for AI assistants (Claude, Gemini, ChatGPT, etc.):** this README — especially [section 6](#6-module-and-function-reference) — is the reference for the public API of `mito-marker`. When advising a user on how to call a function, use only what is documented below; do not invent parameters or functions. Respect this project's conventions: never suggest the variable name `adata` (use `mitochondria_anndata_object`, `spectral_cytometry_anndata`, etc.), and never suggest writing a `.obs` value that is not declared in `controlled_vocabulary.py`. The API evolves under active development and past changes have included breaking renames with no backward-compatible alias, so if a suggestion doesn't work as documented, tell the user to check their installed version's actual docstring (`help(function_name)`) rather than assuming the README is stale.

---

## Table of Contents

1. [What This Project Does](#1-what-this-project-does)
2. [Who This Is For](#2-who-this-is-for)
3. [Quick Start](#3-quick-start)
4. [Project Architecture](#4-project-architecture)
5. [Full File Structure](#5-full-file-structure)
6. [Module and Function Reference](#6-module-and-function-reference)
7. [Data Flow](#7-data-flow)
8. [Controlled Vocabulary](#8-controlled-vocabulary)
9. [Running Tests](#9-running-tests)
10. [Environment Setup](#10-environment-setup)

---

## 1. What This Project Does

Mitochondria change shape depending on the health, age, diet, and disease state of a cell. This project turns raw mitochondrial measurements into structured, machine-learning-ready datasets that can be compared across subjects, conditions, and timepoints.

**Two input pipelines feed into the same downstream analysis stack:**

```
SFC pipeline:  .fcs files (flow cytometer output)
                    ↓
               Parse → Load → Build → Concatenate → Save
                    ↓
               AnnData (.h5ad file)

TEM pipeline:  .txt files (ImageJ/Fiji morphology measurements)
                    ↓
               Parse → Load → Build → Concatenate → Save
                    ↓
               AnnData (.h5ad file)
                    ↓
         Shared analysis pipeline:
         Preprocessing → Visualization → Dimensionality Reduction
         → Statistics → Heterogeneity Indices → Machine Learning → Report
```

The central data structure throughout is `AnnData` (from the [anndata](https://anndata.readthedocs.io/) library). Every dataset lives in an `.h5ad` file on disk. The package never modifies raw files.

---

## 2. Who This Is For

- **Codespaces (developers):** clone the repository, install dependencies, run tests, edit source code.
- **Google Colab (end users):** mount Google Drive, install the package, run notebooks.

Both environments use the same code — no environment-specific branches or patches.

---

## 3. Quick Start

### Install

```bash
pip install -r requirements.txt
```

### SFC ingestion + full analysis pipeline

```python
from mito_marker import (
    ingest_sfc_folder,
    inspect_anndata,
    configure_preprocessing,
    run_preprocessing,
    assign_color_palette,
    plot_radar,
    compute_umap, plot_umap,
    compute_pca, plot_pca_scatter, plot_pca_biplot,
    bin_obs_column,
    compute_all_mhi,
    plot_mhi_distances,
    run_ml_analysis,
)

# Ingest a folder of .fcs files
spectral_cytometry_anndata = ingest_sfc_folder(
    data_directory_path="data/raw/sfc/",
    output_file_path="data/processed/sfc.h5ad",
)

# Explore the dataset interactively
inspect_anndata(spectral_cytometry_anndata)

# Configure and run preprocessing (subset, feature selection, normalization)
preprocessing_config = configure_preprocessing(spectral_cytometry_anndata)
sfc_subset = run_preprocessing(spectral_cytometry_anndata, preprocessing_config)

# Assign consistent colors to conditions
sfc_subset = assign_color_palette(sfc_subset)

# Visualization
plot_radar(sfc_subset, group_by="subject_ID")
sfc_subset = compute_umap(sfc_subset)
plot_umap(sfc_subset, group_by="condition")
sfc_subset = compute_pca(sfc_subset)
plot_pca_scatter(sfc_subset, group_by="subject_ID")

# Stratify subjects into groups (e.g. by age tertiles)
sfc_subset = bin_obs_column(sfc_subset, obs_column="age")

# Compute mitochondrial heterogeneity
sfc_subset = compute_all_mhi(sfc_subset, group_by="condition")
plot_mhi_distances(sfc_subset, group_by="condition")

# Machine learning (classification or regression)
run_ml_analysis(sfc_subset, ml_config={"target_obs_column": "condition", "task_type": "classification"})
```

### TEM ingestion

```python
from mito_marker import ingest_tem_folder

tem_morphology_anndata = ingest_tem_folder(
    data_directory_path="data/raw/tem/",
    output_file_path="data/processed/tem.h5ad",
)
inspect_anndata(tem_morphology_anndata)
```

### Inspect an AnnData object

```python
from mito_marker import inspect_anndata
import anndata

spectral_cytometry_anndata = anndata.read_h5ad("data/processed/sfc.h5ad")
inspect_anndata(spectral_cytometry_anndata)
```

In a Jupyter or Colab notebook this opens a five-tab interactive widget. In a plain Python script it prints a formatted summary to the console.

---

## 4. Project Architecture

```
Raw data files  ──►  Pipeline module  ──►  AnnData (.h5ad)  ──►  Analysis
 (.fcs / .txt)        (sfc / tem)          [.X, .obs, .var]      (preprocessing → viz → stats → ML → report)
                           │
                    controlled_vocabulary.py
                    (validates all metadata)
```

Every `.obs` field (metadata column) — subject ID, condition, timepoint, staining — must be declared in `controlled_vocabulary.py` before it can be assigned. This prevents typos and inconsistent labels from silently entering the dataset.

---

## 5. Full File Structure

```
mito-marker/
│
├── src/
│   └── mito_marker/                       # Installable Python package
│       ├── __init__.py                    # Public API: all pipeline and analysis functions
│       ├── controlled_vocabulary.py       # All allowed .obs values, species, conditions, bounds
│       ├── inspector.py                   # inspect_anndata() interactive widget + console fallback
│       │
│       ├── sfc/                           # Spectral Flow Cytometry pipeline
│       │   ├── __init__.py
│       │   ├── ingestion.py               # ingest_sfc_folder() orchestrator
│       │   ├── filename_parser.py         # parse_fcs_filename() — metadata from filename tokens
│       │   ├── fcs_loader.py              # load_fcs_file() — reads .fcs binary with FlowIO
│       │   ├── anndata_builder.py         # build_sfc_anndata() — assembles AnnData
│       │   ├── clinical_data.py           # enrich_with_clinical_data() — links MNMS clinical CSV
│       │   └── qc.py                      # QC printing functions (no data modification)
│       │
│       ├── tem/                           # TEM pipeline (fully implemented)
│       │   ├── __init__.py
│       │   ├── tem_ingestion.py           # ingest_tem_folder() orchestrator
│       │   ├── tem_filename_parser.py     # parse_tem_filename() — species/condition/subject from filename
│       │   ├── tem_file_loader.py         # load_tem_file() — reads tab-separated .txt files
│       │   └── tem_anndata_builder.py     # build_tem_anndata() — assembles AnnData
│       │
│       └── analysis/                      # Shared downstream analysis
│           ├── __init__.py                # Exports all public analysis functions
│           │
│           ├── # ── PREPROCESSING ────────────────────────────────
│           ├── preprocessing_config.py    # configure_preprocessing(), get_default_preprocessing_config()
│           ├── preprocessing_pipeline.py  # run_preprocessing() — non-interactive steps 1–3
│           ├── selection.py               # select_sfc_subset() — interactive event filter
│           ├── feature_selection.py       # select_channels(), get_selected_data_matrix()
│           ├── normalization.py           # transform_and_normalize()
│           │
│           ├── # ── COLORS & VISUALIZATION ───────────────────────
│           ├── colors.py                  # assign_color_palette(), get_color_for_value(), get_subject_colors()
│           ├── radar_plot.py              # plot_radar() — polar with jitter scatter
│           ├── umap_plot.py               # compute_umap(), plot_umap()
│           ├── pca_plot.py                # compute_pca(), plot_pca_scatter/biplot/trajectory (2D + 3D)
│           ├── distribution_plots.py      # plot_histogram(), plot_density(), plot_violin()
│           ├── time_curve.py              # plot_time_curve() — longitudinal channel profiles
│           ├── channel_intensity_plot.py  # plot_channel_intensity_line/bar() — per-sample profile across channels
│           ├── age_plot.py                # plot_feature_vs_age() — regression scatter by subject
│           │
│           ├── # ── STATISTICS & STRATIFICATION ─────────────────
│           ├── stratification.py          # bin_obs_column(), split_obs_by_threshold(), map_obs_values()
│           ├── group_comparison.py        # compare_groups() — Mann-Whitney / Kruskal-Wallis + effect sizes
│           ├── channel_aggregation.py     # aggregate_sfc_channels_by_color()
│           │
│           ├── # ── HETEROGENEITY INDICES ────────────────────────
│           ├── mhi.py                     # compute_mhi_d/s/e(), compute_all_mhi(), chimera validation
│           ├── amhi.py                    # compute_all_mito_mean(), compute_amhi() — fixed-reference space
│           │
│           ├── # ── MACHINE LEARNING ─────────────────────────────
│           ├── ml_config.py               # configure_ml(), get_default_ml_config()
│           ├── ml_bagging.py              # create_bags() — per-subject statistical feature bags
│           ├── ml_pipeline.py             # run_ml_analysis(), run_feature_subset_challenge()
│           ├── ml_mil.py                  # run_mil_logo_cv(), get_mil_attention_dataframe(), plot_mil_attention()
│           ├── permutation_test.py        # run_permutation_test() — subject-level null distribution
│           ├── clustering.py              # run_hdbscan(), compute_umap_on_bags(), profile_clusters()
│           ├── shap_clustering.py         # cluster_shap_values(), plot_shap_cluster_heatmap()
│           │
│           ├── # ── REUSABLE CLUSTER MODELS (fit / predict) ────
│           ├── feature_subset.py          # subset_by_obs_values(), subset_by_feature_values(), get_subset_history()
│           ├── cluster_model.py           # MitoClusterModel, fit/predict_cluster_model(), save/load, project_to_reference_space()
│           ├── cluster_plots.py           # plot_cluster_radar/embedding/proportions(), compare_cluster_proportions()
│           │
│           ├── # ── FEATURE CLUSTERING / PHYLOGENY ───────────────
│           ├── clustermap_plot.py         # plot_feature_clustermap() — hierarchical clustering heatmap
│           ├── phylogeny.py               # build_species_phylogeny_linkage(), get_linkage_clades()
│           ├── phylo_tanglegram.py        # plot_phylo_tanglegram(), compute_tanglegram_sensitivity()
│           │
│           └── report.py                  # ReportBuilder — captures figures + console output → PDF
│
├── tests/
│   ├── fixtures/                          # Small committed sample files for pytest
│   │   ├── good_events_MNMS_020_FlowAIGoodEvents_DeepRed.fcs
│   │   ├── good_events_MNMS_031_ND_FlowAIGoodEvents_DeepRed.fcs
│   │   ├── M2124_J1_G5_0114_MITO_measurements.txt
│   │   ├── SourisJ3_Prot1533_Ech1755_Grille13E_x1500_field_19_MITO_measurements.txt
│   │   └── SourisV1_Prot1533_Ech1757_Grille14B_x1500_field_31_MITO_measurements.txt
│   │
│   ├── test_controlled_vocabulary.py
│   ├── test_inspector.py
│   ├── test_integration_pipeline.py
│   ├── test_sfc_filename_parser.py
│   ├── test_sfc_ingestion.py
│   ├── test_sfc_ingestion_mock.py
│   ├── test_sfc_clinical_data.py
│   ├── test_tem_filename_parser.py
│   ├── test_tem_file_loader.py
│   ├── test_tem_anndata_builder.py
│   ├── test_tem_ingestion.py
│   ├── test_analysis_preprocessing_config.py
│   ├── test_analysis_colors.py
│   ├── test_analysis_feature_selection.py
│   ├── test_analysis_normalization.py
│   ├── test_analysis_selection.py
│   ├── test_analysis_radar_plot.py
│   ├── test_analysis_umap_plot.py
│   ├── test_analysis_pca_plot.py
│   ├── test_analysis_plot_context.py
│   ├── test_analysis_stratification.py
│   ├── test_analysis_distribution_plots.py
│   ├── test_analysis_time_curve.py
│   ├── test_analysis_mhi.py
│   ├── test_analysis_amhi.py
│   ├── test_analysis_ml_config.py
│   ├── test_analysis_ml_bagging.py
│   ├── test_analysis_ml_pipeline.py
│   ├── test_analysis_ml_mil.py
│   ├── test_analysis_clustering.py
│   ├── test_analysis_shap_clustering.py
│   ├── test_analysis_channel_aggregation.py
│   ├── test_analysis_channel_intensity_plot.py
│   └── test_analysis_report.py
│
├── examples/                              # Demonstration notebooks
│   ├── sfc_analysis_demo.ipynb            # End-to-end SFC demo
│   ├── tem_ingestion_demo.ipynb           # TEM ingestion
│   ├── tem_cluster_model_demo.ipynb       # Reusable cluster models (fit once, predict)
│   ├── amhi_demo.ipynb                    # Absolute heterogeneity index
│   └── inspect_anndata_demo.ipynb         # Interactive inspector
│
├── docs/
│   ├── DECISIONS.md                       # Why the package works the way it does (ADRs)
│   └── GLOSSARY.md                        # Domain terms
│
├── .github/workflows/tests.yml            # Runs pytest + ruff on every push
├── CHANGELOG.md                           # What changed in each release
├── pyproject.toml                         # Package metadata and build configuration
├── requirements.txt                       # All Python dependencies
├── CLAUDE.md                              # AI assistant instructions and coding standards
└── README.md                              # This file
```

---

## 6. Module and Function Reference

### `src/mito_marker/__init__.py` — Package entry point

Exposes all public functions of the package.

---

### `src/mito_marker/controlled_vocabulary.py` — Allowed values registry

**No functions.** Contains module-level constants only. Every `.obs` field must be declared here before use anywhere in the codebase.

| Constant | Type | Description |
|---|---|---|
| `ALLOWED_STAINING_VALUES` | `dict[str, str]` | `"DeepRed"`, `"Unstained"` |
| `ALLOWED_CONDITIONS_TEM` | `dict[str, str]` | `"Young"`, `"Old"` |
| `ALLOWED_SPECIES_SFC` | `dict[str, str]` | `"MNMS"`, `"WORMS"`, `"FLY"`, etc. |
| `ALLOWED_DIET_SFC` | `dict[str, str]` | `"AL"` (ad libitum), `"IF"` (intermittent fasting) |
| `ALLOWED_DILUTION_SFC` | `dict[str, str]` | `"Diluted"`, `"Not_Diluted"` |
| `ALLOWED_MARKERS_SFC` | `dict[str, str]` | `"MtDeepRed"` |
| `TEM_SPECIES_TOKEN_TO_CANONICAL` | `dict[str, str]` | Maps TEM filename tokens → canonical species names |
| `TEM_CONDITION_TOKEN_TO_CANONICAL` | `dict[str, str]` | Maps TEM condition tokens → `"Young"` / `"Old"` |
| `TEM_FEATURE_COLUMNS` | `list[str]` | 28 morphological feature column names |
| `MARKER_TOKEN_TO_CANONICAL_SFC` | `dict[str, str]` | Maps raw filename tokens `"DR"`, `"DeepRed"` → `"MtDeepRed"` |
| `FLOWAI_PASS_TOKENS` | `set[str]` | Tokens that indicate a FlowAI-filtered file |
| `SFC_CHANNELS_TO_KEEP` | `list[str]` | 170 canonical channel names in spectral order |
| `SFC_CHANNELS_TO_EXCLUDE` | `list[str]` | 64 `FJComp-` compensation channels to always drop |
| `SFC_NON_ANALYTICAL_CHANNELS` | `list[str]` | Time and FlowAI channels excluded from analysis |
| `SFC_SPECTRAL_GROUP_ORDER` | `dict[str, list[str]]` | Laser → detector groupings for QC display |
| `FCS_OBS_METADATA_FIELDS` | `dict[str, str]` | FCS TEXT fields extracted as `.obs` columns |
| `FILTERABLE_OBS_COLUMNS` | `list[str]` | Columns shown in the interactive selection menu |
| `PREFERRED_CONDITION_COLORS` | `dict[str, str]` | Researcher-defined condition → hex color overrides |
| `NUMERICAL_OBS_BOUNDS` | `dict[str, dict]` | Physiological min/max bounds and units for numerical fields |
| `MNMS_CLINICAL_CSV_SUBJECT_ID_COLUMN` | `str` | Column name for subject ID in the MNMS clinical CSV |
| `MNMS_CLINICAL_CSV_MISSING_VALUE_CODE` | `float` | Missing value sentinel in the clinical CSV (`-1.0` → `NaN`) |

---

### `src/mito_marker/sfc/` — Spectral Flow Cytometry pipeline

#### `ingestion.py` — `ingest_sfc_folder`

```python
def ingest_sfc_folder(
    data_directory_path: str,
    output_file_path: str,
    channels_to_exclude: Optional[List[str]] = None,
    append_to_existing: bool = False,
) -> anndata.AnnData
```

Scans a folder of `.fcs` files, ingests each one, validates channel consistency across files, concatenates into a single AnnData, and saves to `.h5ad`.

---

#### `clinical_data.py` — `enrich_with_clinical_data`

```python
def enrich_with_clinical_data(
    anndata_object: anndata.AnnData,
    csv_path: str,
) -> anndata.AnnData
```

Links an MNMS clinical CSV (one row per subject) to an SFC AnnData via `subject_ID`. Column names are sanitized (unicode → ASCII, special chars → `_`, lowercase). Missing value code `-1.0` is replaced with `NaN`. The original ↔ sanitized name mapping is stored in `.uns["clinical_column_name_mapping"]`.

---

### `src/mito_marker/tem/` — TEM pipeline

#### `tem_ingestion.py` — `ingest_tem_folder`

```python
def ingest_tem_folder(
    data_directory_path: str,
    output_file_path: str,
) -> anndata.AnnData
```

Scans a folder for `*_MITO_measurements.txt` files, parses each filename for species/condition/subject metadata, loads the feature matrix, builds per-file AnnData objects, concatenates them, and saves to `.h5ad`.

**TEM AnnData structure:**

| Slot | Content |
|---|---|
| `.X` | `float32`, shape `(n_mitochondria, 28)` — morphological features, never modified |
| `.obs['specie']` | Canonical species name (from `TEM_SPECIES_TOKEN_TO_CANONICAL`) |
| `.obs['condition']` | `"Young"` or `"Old"` (from `TEM_CONDITION_TOKEN_TO_CANONICAL`) |
| `.obs['subject_ID']` | Digits after `"fish"` separator or first 3+ digit sequence |
| `.obs['Image_Name']` | Image identifier from the source file |
| `.obs['source_filename']` | Absolute path of the source `.txt` file |
| `.var` | 28 rows indexed by `TEM_FEATURE_COLUMNS` |

Missing tokens produce a `warnings.warn()` and the field is set to `None` — no hard failure.

---

### `src/mito_marker/inspector.py` — `inspect_anndata`

```python
def inspect_anndata(
    anndata_object: anndata.AnnData,
    max_display_rows: int = 15,
    max_display_columns: int = 12,
    group_by_column: str | None = None,
) -> None
```

Auto-detects the execution environment:
- **Jupyter / Colab:** five-tab `ipywidgets` widget (Overview, `.X` Matrix, `.obs`, `.var`, Reductions).
- **Plain Python / pytest:** formatted seven-section console summary.

---

### `src/mito_marker/analysis/` — Shared downstream analysis

The analysis pipeline writes results into well-defined AnnData slots:

| Slot | Written by | Content |
|---|---|---|
| `.layers['{transform}__{norm}']` | `transform_and_normalize` | Processed data matrix |
| `.var['{method}_score']` | `select_channels` | Per-channel feature selection score |
| `.var['is_selected_{method}']` | `select_channels` | Boolean selection flag |
| `.obsm['X_pca']` | `compute_pca` | PCA coordinates `(n_obs, n_components)` |
| `.obsm['X_umap_n{n}_d{d}']` | `compute_umap` | UMAP coordinates (NaN for unsampled events) |
| `.uns['analysis_config']` | Multiple steps | Active layer, active selection method |
| `.uns['color_palette']` | `assign_color_palette` | Condition value → hex color |
| `.uns['pca_loadings']` | `compute_pca` | Shape `(n_channels, n_components)` |
| `.uns['mhi_results'][col]['D'/'S'/'E']` | `compute_all_mhi` | MHI results per group |
| `.uns['amhi_reference']` | `compute_all_mito_mean` | Frozen scaler + PCA bytes for fixed-reference space |
| `.uns['amhi_results'][col]` | `compute_amhi` | AMHI results per group |
| `.uns['layer_parameters'][layer]` | `transform_and_normalize`, `run_preprocessing` | Frozen scaling (means / SDs / bounds) + fingerprint, replayable on another dataset |
| `.uns['pca_mean']`, `['pca_fingerprint']` | `compute_pca` | Centring vector + fingerprint: new data can be projected on the same axes |
| `.uns['umap_reducers'][key]` | `compute_umap(keep_reducer=True)` | Serialized UMAP reducer (opt-in) |
| `.uns['analysis_config']['subset_history']` | `subset_by_obs_values`, `subset_by_feature_values` | Every subset step: thresholds applied, counts before/after per subject |
| `.obs['cluster__<model>']` | `fit_cluster_model`, `predict_cluster_model` | Cluster label (`Cluster_1` = largest in training) |
| `.obs['cluster_probability__<model>']` | same (GMM only) | Membership probability of the assigned cluster |
| `.uns['cluster_models'][model]` | same | h5ad-safe summary of the model applied |

---

#### Preprocessing — `preprocessing_config.py` + `preprocessing_pipeline.py`

##### `configure_preprocessing`

```python
def configure_preprocessing(anndata_object: anndata.AnnData) -> dict
```

Interactive wizard that builds a preprocessing configuration dictionary covering: event subset filters, subsampling, feature selection method, transformation, and normalization.

##### `get_default_preprocessing_config`

```python
def get_default_preprocessing_config() -> dict
```

Returns a default configuration dict (no filters, no feature selection, `arcsinh` + `zscore_col`). Use as a starting point for non-interactive scripting.

##### `run_preprocessing`

```python
def run_preprocessing(
    anndata_object: anndata.AnnData,
    preprocessing_config: dict,
) -> anndata.AnnData
```

Runs steps 1–3 (subset selection, feature selection, normalization) non-interactively from a config dict. Replaces the older interactive `select_sfc_subset` → `select_channels` → `transform_and_normalize` call sequence when scripting.

`ALLOWED_TRANSFORMS`, `ALLOWED_NORMALIZATIONS`, and `ALLOWED_FEATURE_SELECTION_METHODS` (lists of strings) declare every valid value for the corresponding `preprocessing_config` keys — the same values `configure_preprocessing()`'s interactive wizard offers.

---

#### Feature selection — `feature_selection.py`

##### `select_channels`

```python
def select_channels(anndata_object: anndata.AnnData) -> anndata.AnnData
```

Interactive menu of five methods: None / MIM / CMI-mRMR / HighVariance / PCALoadings. Results written to `.var` as `{method}_score` and `is_selected_{method}`. The AnnData shape never changes.

##### `get_selected_data_matrix`

Returns the data matrix restricted to currently selected channels. Falls back to the full matrix if no selection is active. Non-analytical channels (`SFC_NON_ANALYTICAL_CHANNELS`) are always excluded.

---

#### Normalization — `normalization.py`

##### `transform_and_normalize`

Sequential menus: **Transform** (None / Arcsinh / Logicle) then **Normalize** (None / Z-score col / Min-Max col / L2 row / Z-score + L2). Processed matrix stored in `.layers['{transform}__{norm}']`. Raw `.X` never modified.

The fitted parameters (z-score means and SDs, min-max bounds, logicle channel scale) are frozen in `.uns['layer_parameters'][layer]` with a 12-character fingerprint, so the same scaling can be replayed on another dataset (used by cluster models, ADR-015).

---

#### Visualization

##### `assign_color_palette` / `get_color_for_value` / `get_subject_colors` — `colors.py`

```python
def assign_color_palette(anndata_object: anndata.AnnData, palette: str = "Set2") -> anndata.AnnData
def get_color_for_value(anndata_object: anndata.AnnData, condition_value: str, fallback_color: str = "#999999") -> str
def get_subject_colors(subject_ids: List[str], palette: str = "tab10") -> Dict[str, str]
```

Assigns a consistent hex color to every unique value of every filterable `.obs` column, stored in `.uns['color_palette']`. `PREFERRED_CONDITION_COLORS` overrides (declared in `controlled_vocabulary.py`) always win over the automatic palette; already-assigned values are preserved, so calling this twice is idempotent. `get_color_for_value` looks up one value's color from that stored palette. `get_subject_colors` generates a fresh per-subject color mapping at plot time instead — the subject list changes with every `select_sfc_subset()` call, so it is not persisted in `.uns`.

##### `plot_radar` — `radar_plot.py`

Polar spider plots, one figure per pulse-type suffix (`-A`, `-H`, `-W`). Shows jittered event scatter + mean line per group + global mean reference.

##### `compute_umap` / `plot_umap` — `umap_plot.py`

Fits UMAP on a stratified random sample. Key: `X_umap_n{n_neighbors}_d{min_dist}`. Cache: skip recompute if key already in `.obsm`.

##### `compute_pca` / `plot_pca_scatter` / `plot_pca_biplot` / `plot_pca_trajectory` — `pca_plot.py`

Fits PCA on all events. Stores loadings and explained variance in `.uns`. Includes 3D interactive variants: `plot_pca_3d_scatter`, `plot_pca_3d_biplot`, `plot_pca_3d_trajectory` (Plotly).

Every PCA plot shows a **loadings panel** in its right-hand column, below the color legend: for each PC its explained variance and its top features with their % contribution to the axis, `loading² × 100` (FactoMineR / factoextra convention, ADR-016; contributions sum to 100% on each PC; "+"/"-" gives the sign of the loading). The 2D plots (`plot_pca_scatter`, `plot_pca_biplot`, `plot_pca_trajectory`) take `show_loadings=True`, `loadings_top_n=5` and `loadings_n_components=3`; the 3D plots take `show_loadings=True` and `loadings_top_n=5` and list the three displayed axes. The console always prints the panel and how the loadings were computed (standard or weighted PCA, input layer, formula).

`get_top_loading_channels(anndata_object, pc_index=0, top_n=5)` returns the `top_n` channels with the highest absolute loading on one component, as `(channel_name, loading_value)` tuples. `plot_pca_loadings_bar(anndata_object, top_n=5, n_components=3)` prints and plots, per component, the top channels ranked by % contribution (`loading² × 100`), plus a cross-component summary of every channel that appears in any per-component top-N list, with its total contribution weighted by each PC's explained variance.

##### `plot_histogram` / `plot_density` / `plot_violin` — `distribution_plots.py`

Per-feature distribution plots, one subplot per channel, grouped by any `.obs` column.

##### `plot_time_curve` — `time_curve.py`

```python
def plot_time_curve(
    anndata_object: anndata.AnnData,
    x_obs_column: str,
    channels: Optional[List[str]] = None,
    group_by: Optional[str] = None,
    x_order: Optional[List[str]] = None,
) -> None
```

Line plot of mean channel values across a longitudinal `.obs` column (timepoint, age group, etc.). Optionally splits curves by a second `.obs` column.

##### `plot_channel_intensity_line` / `plot_channel_intensity_bar` — `channel_intensity_plot.py`

```python
def plot_channel_intensity_line(
    anndata_object: anndata.AnnData,
    channels: Optional[List[str]] = None,
    pulse_type: str = "A",
    sample_column: str = "source_filename",
    samples: Optional[List[str]] = None,
    statistic: str = "mean",
    show_error_bars: bool = False,
    log_scale: bool = False,
    title: str = "",
) -> None
```

Per-sample fluorescence intensity profile: **channels on the X-axis, one series per sample** (one SFC file). Each point is the mean (or median) over that sample's mitochondria. `show_error_bars=True` adds a ±1 std cap over the sample's events; it is off by default because the caps overlap and hide the profile shape once there are many channels or many samples. `plot_channel_intensity_bar` takes the identical signature and renders grouped bars instead.

`channels` accepts three token forms, so picking channels stays short:

| Token | Result |
|---|---|
| `"V1-A"` | that single channel |
| `"V-A"`, `"YG-H"` | **every** channel of that laser and pulse, as separate X ticks |
| `"V"` | every channel of that laser, using `pulse_type` |

Group tokens *expand* the axis — they never average channels together. Use `aggregate_sfc_channels_by_color()` when you do want laser-group means.

```python
from mito_marker import plot_channel_intensity_line, plot_channel_intensity_bar

plot_channel_intensity_line(sfc_anndata, channels=["V-A", "YG-A"])
plot_channel_intensity_bar(sfc_anndata, channels=["V1-A", "V2-A"], statistic="median")
plot_channel_intensity_line(sfc_anndata, channels=["V-H"], log_scale=True)
```

`log_scale=True` is worth using on raw `.X`, where intensities span several orders of magnitude; leave it off when an arcsinh / z-score layer is active.

##### `plot_feature_vs_age` — `age_plot.py`

Scatter plot of a feature value vs. a numerical age `.obs` column, aggregated per subject, with a regression band.

---

#### Statistics

##### `compare_groups` — `group_comparison.py`

```python
def compare_groups(
    anndata_object: anndata.AnnData,
    group_by: str,
    features: Optional[List[str]] = None,
) -> pd.DataFrame
```

Non-parametric group comparison for each feature. Two groups → Mann-Whitney U + rank-biserial correlation. Three+ groups → Kruskal-Wallis + eta-squared + Dunn post-hoc. Returns a DataFrame of p-values and effect sizes.

##### `bin_obs_column` — `stratification.py`

```python
def bin_obs_column(
    anndata_object: anndata.AnnData,
    obs_column: str,
    n_bins: int = 3,
    new_column_name: Optional[str] = None,
) -> anndata.AnnData
```

Equal-frequency quantile binning of a numerical `.obs` column. Auto column name: `_tertiles` / `_quartiles` / `_{n}bins`.

##### `split_obs_by_threshold` — `stratification.py`

Hard threshold split: `< threshold` vs. `>= threshold`.

##### `map_obs_values` — `stratification.py`

Remap `.obs` column values via a dictionary (e.g. `{"AL": "Control", "IF": "Fasting"}`).

##### `aggregate_sfc_channels_by_color` — `channel_aggregation.py`

Groups spectral channels by laser color and computes per-group summary statistics per event.

---

#### Reusable cluster models — `feature_subset.py`, `cluster_model.py`, `cluster_plots.py`

Fit a clustering once (e.g. on the 25% smallest Young mitochondria), save it, and re-apply it without refitting to any other dataset (Old, IF, another experiment). See ADR-015.

**Rule:** build the reference space ONCE on the full reference dataset — `run_preprocessing()` (z-score), `compute_pca()`, optionally `compute_umap(embed_all_events=True, keep_reducer=True)` — and only THEN take subsets. Subsets inherit the layer, the coordinates and the frozen parameters; the cluster model copies them, so its `.joblib` file is self-contained.

```python
tem_anndata = run_preprocessing(tem_anndata, get_default_preprocessing_config())
tem_anndata = compute_pca(tem_anndata, n_components=10)
young = subset_by_obs_values(tem_anndata, {"condition": ["Young"]})
old = subset_by_obs_values(tem_anndata, {"condition": ["Old"]})
young_small = subset_by_feature_values(young, "Mito_Area", "lowest_fraction", fraction=0.25,
                                       within_group="unique_subject_ID")
old_small = subset_by_feature_values(old, "Mito_Area", "lowest_fraction", fraction=0.25,
                                     within_group="unique_subject_ID")
young_small, model = fit_cluster_model(young_small, "young_small_kmeans", algorithm="kmeans",
                                       clustering_space="pca", max_clusters=8,
                                       weight_by=["unique_subject_ID"])
save_cluster_model(model, "/content/drive/MyDrive/mito-marker/models/young_small_kmeans.joblib")
old_small = predict_cluster_model(old_small, model)
plot_cluster_radar(young_small, model, nest_aggregate_by="unique_subject_ID")
proportions = compute_cluster_proportions({"Young": young_small, "Old": old_small}, model)
plot_cluster_proportions(proportions, model)
compare_cluster_proportions({"Young": young_small, "Old": old_small}, model)
```

| Function | Role |
|---|---|
| `subset_by_obs_values(anndata, {"col": [values]})` | Keep rows by `.obs` values (OR within a column, AND across columns) |
| `subset_by_feature_values(anndata, feature, mode, ...)` | `lowest_fraction` / `highest_fraction` (exact count, pooled or `within_group`), `between` / `below` / `above` (absolute values, raw units) |
| `get_subset_history(anndata)` | Ordered list of subset steps (thresholds applied, counts per subject) |
| `fit_cluster_model(...)` | KMeans (k by silhouette) or GMM (k by BIC), k = 2…`max_clusters`, on the `scaled`, `pca` or `umap` space; clusters named `Cluster_1…k` by decreasing size |
| `predict_cluster_model(anndata, model)` | Assign to the learned clusters; reuses stored coordinates when fingerprints match, otherwise projects from raw `.X`. QC: same variables, same scaler, drift, atypical share |
| `project_to_reference_space(anndata, model)` | Replay scaling → PCA → UMAP (transform only) on a separately loaded dataset |
| `save_cluster_model` / `load_cluster_model` / `describe_cluster_model` | `.joblib` persistence (load only your team's files) and readable summary |
| `plot_cluster_model_selection` / `plot_cluster_radar` / `plot_cluster_embedding` | Choice of k; mean profile per cluster; PCA / UMAP colored by cluster |
| `compute_cluster_proportions` | `pct_of_subset`, `pct_of_total_population` (denominator from the subset history), per-subject mean ± SD |
| `plot_cluster_proportions` | Stacked composition + per-subject points |
| `compare_cluster_proportions` | Per cluster: Mann-Whitney / Kruskal-Wallis on per-subject %, BH-adjusted, effect sizes; n = subjects (ADR-008) |

---

#### Heterogeneity Indices

##### MHI — Mitochondrial Heterogeneity Index (`mhi.py`)

Three metrics measuring **within-subject** spread of mitochondrial measurements:

| Function | Metric | Description |
|---|---|---|
| `compute_mhi_d` | MHI-D | Mean pairwise Euclidean distance in PCA space |
| `compute_mhi_s` | MHI-S | Bimodality score (dip test) in local PCA space |
| `compute_mhi_e` | MHI-E | Entropy of the per-mito distance distribution |
| `compute_all_mhi` | D + S + E | Computes all three in one call |

Key parameters:
- `group_by` — any `.obs` column (e.g. `"subject_ID"`, `"condition"`)
- `nest_aggregate_by` — enables two-level computation: compute per individual, then aggregate by group
- `n_pca_components` — `'auto'` (95% variance), `None` (raw features), or exact `int`
- `standardize_on` — `'all'` (global, inter-comparable) or `'group'` (per-individual)

Results stored in `.uns['mhi_results'][group_column]`.

Additional functions: `create_chimera`, `validate_mhi_with_chimeras`, `analyze_mhi_sensitivity`, `plot_mhi_distances`, `plot_mhi_pca`, `plot_mhi_barplot`, `plot_mhi_stripplot`, `run_mhi_statistics`, `save_mhi_to_obs`. Plus:

| Function | Description |
|---|---|
| `run_mhi_age_tests(anndata_object, species_col, age_col, ...)` | For each species and each `MHI_*` metric column, runs a Mann-Whitney U test and a label-shuffle permutation test comparing young vs. old individuals. Stored in `.uns['mhi_results'][species_col]['age_tests']`. |
| `validate_all_chimera_pairs(anndata_object, species_col="specie", age_col="age_group", ...)` | Enumerates every young/old individual pair within each species and calls `validate_mhi_with_chimeras()` on each, then displays a grid figure (one subplot per pair) to check MHI monotonicity across the whole cohort, not just one hand-picked pair. |
| `plot_mhi_scatter_de(anndata_object, group_by=None, ...)` | Scatter of MHI-D (dispersion) vs. MHI-E (KNN entropy) per individual, colored by group, with group centroids and inter-individual error crosses. |
| `plot_mhi_species_heatmap(anndata_object, group_by=None, ...)` | Annotated heatmap of pairwise Euclidean distances between groups' median MHI component vectors — which species/conditions are most similar or different in heterogeneity profile. |

---

##### AMHI — Absolute Mitochondrial Heterogeneity Index (`amhi.py`)

Measures heterogeneity in a **fixed reference space**, enabling comparison of new subjects against an existing population without refitting.

| Function | Description |
|---|---|
| `compute_all_mito_mean` | Fits StandardScaler + PCA on a reference population; serializes transforms into `.uns['amhi_reference']` |
| `compute_amhi` | Projects subjects into the fixed space; computes `AMHI_offset` (centroid distance to AllMitoMean) and `AMHI_D` (total per-mito distance) |
| `compute_mhi_d_absolute` | MHI-D computed in fixed space (internal spread only, no centroid offset) |
| `plot_amhi_distances` | Bar/strip plot of AMHI metrics per group |
| `plot_amhi_2d` | 2D scatter of subject centroids in reference PCA space |
| `plot_amhi_profile` | Radar profile of the AllMitoMean reference centroid |
| `plot_amhi_bar` | Horizontal bar chart ranking groups by a chosen AMHI metric (`AMHI_offset`, `AMHI_D_median`, or `MHI_D_absolute_median`), with optional IQR/std error bars and a population-median reference line |
| `plot_amhi_species_comparison` | Vertical grouped bar chart comparing AMHI across species (or any `group_by`), with individual subject dots overlaid and an optional `split_by` (e.g. condition) for side-by-side bars |
| `plot_amhi_umap` | Projects mitochondria into the frozen AMHI reference PCA space, then runs and displays a UMAP of those coordinates — colored by `color_by` (default `specie`) |
| `summarize_amhi` | Prints a formatted summary table |

The AllMitoMean is injected as a synthetic obs row so `plot_radar(group_by="unique_subject_ID")` includes it automatically.

---

#### Machine Learning

##### `create_bags` — `ml_bagging.py`

```python
def create_bags(
    anndata_object: anndata.AnnData,
    group_by: str,
    bag_features: Optional[List[str]] = None,
) -> pd.DataFrame
```

Aggregates per-mito measurements into per-subject statistical feature bags (mean, std, percentiles, skewness, kurtosis). Used as input for classical ML models.

##### `configure_ml` / `get_default_ml_config` — `ml_config.py`

Interactive wizard or default dict for ML configuration: model type, task type, cross-validation strategy, SHAP settings, target column. `DEFAULT_MODEL_PARAMS` (dict, keyed by model name) holds the expert-tuned hyperparameter presets used when a config doesn't override them.

##### `run_ml_analysis` — `ml_pipeline.py`

```python
def run_ml_analysis(
    anndata_object: anndata.AnnData,
    ml_config: dict,
) -> dict
```

Full ML pipeline: bagging → model training → LOGO or standard cross-validation → SHAP explainability → confusion matrix / ROC curve (classification) or R²/MAE/RMSE (regression). Supports `task_type="classification"` or `task_type="regression"`.

Models: Random Forest, XGBoost, SVM, Logistic Regression.

`run_feature_subset_challenge(anndata_object, feature_subsets_dict, ml_config=None)` re-runs `run_ml_analysis()` once per named feature subset (e.g. `{"Size": [...], "Shape": [...]}`) and returns a comparison DataFrame — used to decide which feature group carries the most predictive signal for a given target. `get_sfc_feature_subsets(anndata_object)` builds a ready-made `{"Morpho": [...], "Spectral": [...]}` subset dict for SFC data by matching channel-name prefixes. `plot_feature_vs_target(anndata_object, target_obs_column, feature_names=None, top_n=6)` is a raw-signal sanity check independent of any model: scatters per-subject mean(feature) against the target (e.g. age) with Pearson/Spearman annotations, defaulting to the top SHAP features from the last `run_ml_analysis()` call.

##### `run_permutation_test` — `permutation_test.py`

```python
def run_permutation_test(
    anndata_object: anndata.AnnData,
    ml_config: dict,
    max_permutations: Optional[int] = None,
    n_seeds_per_permutation: int = 1,
) -> dict
```

Subject-level permutation test for the model/strategy described by `ml_config`: re-runs `run_ml_analysis()` once per shuffled label assignment (SHAP and plots suppressed) to build a null distribution of the primary metric, then reports where the real-label score falls in it (`p_value`). Uses an exact test when the number of label assignments is small enough, otherwise a capped Monte Carlo sample (`max_permutations`). `n_seeds_per_permutation > 1` averages each permutation over several model seeds — recommended for models with real fit-to-fit randomness (Random Forest, XGBoost, MLP) on small cohorts, where a single seed can look misleadingly significant. Results are also stored in `.uns['permutation_test_results']`.

##### `run_mil_logo_cv` — `ml_mil.py`

```python
def run_mil_logo_cv(
    anndata_object: anndata.AnnData,
    target_obs_column: str,
    task_type: str = "classification",
    group_by: str = "subject_ID",
) -> dict
```

Multiple Instance Learning (ABMIL — Attention-Based MIL) with Leave-One-Group-Out cross-validation. Works directly on per-mito feature vectors (no bagging). GPU-accelerated via PyTorch when available.

Additional functions: `get_mil_attention_dataframe` (per-mito attention weights), `plot_mil_attention` (attention heatmap).

##### `run_hdbscan` / `compute_umap_on_bags` — `clustering.py`

Unsupervised clustering on subject bags: UMAP for dimensionality reduction, HDBSCAN for cluster assignment. `profile_clusters` computes per-cluster feature profiles. `plot_cluster_age_composition` visualizes age distribution per cluster. `evaluate_clustering(coordinates, cluster_labels)` computes quality metrics for one clustering result (`n_clusters`, `noise_ratio`, `silhouette_score`). `compare_bag_sizes(anndata_object, bag_sizes, ...)` runs the whole bag → UMAP → HDBSCAN → profile → evaluate sequence for each bag size in a list (e.g. `[1, 5, 30, 150]`) and stores per-size results in `.uns['bag_clustering_results']` — used to test whether an aging signal is a population-level effect that only becomes clear at larger bag sizes.

##### `cluster_shap_values` / `plot_shap_cluster_heatmap` — `shap_clustering.py`

Groups subjects by their SHAP value profiles (hierarchical clustering on the SHAP matrix). `plot_shap_cluster_heatmap` shows the clustered heatmap with group membership panel.

---

#### Feature Clustering — `clustermap_plot.py`

##### `plot_feature_clustermap`

```python
def plot_feature_clustermap(
    anndata_object: anndata.AnnData,
    group_by: Union[str, List[str]],
    nest_aggregate_by: Optional[Union[str, List[str]]] = None,
    standardize_group_means: bool = False,
    cluster_method: str = "average",
    cluster_metric: str = "correlation",
    rotate_group_nodes: Optional[List[int]] = None,
    rotate_feature_nodes: Optional[List[int]] = None,
) -> Dict[str, object]
```

Hierarchically-clustered heatmap of features (rows) x groups (columns): aggregates by group mean, then clusters both axes with scipy linkage + seaborn clustermap. Prints a merge-order diagnostic so a dendrogram node's two children can be swapped for display (`rotate_group_nodes` / `rotate_feature_nodes`) without changing which items belong under it. `nest_aggregate_by` applies the same nested-mean rule as `plot_radar()` (see `docs/DECISIONS.md`, ADR-004/ADR-011/ADR-013) so a group is never dominated by its most heavily-sampled subject. Returns a dict with the figure, final axis orders, the linkages actually used, and the plotted matrix.

---

#### Phylogeny vs Morphology (Tanglegram) — `phylogeny.py` + `phylo_tanglegram.py`

Compares the known species phylogeny (reconstructed from literature divergence times) against a tree derived from the measured TEM morphology, to test whether morphology recovers the phylogenetic signal (see `docs/DECISIONS.md` ADR-011/ADR-014 for the underlying weighting/aggregation conventions these functions build on).

| Function | Description |
|---|---|
| `build_species_phylogeny_linkage(species_list, branch_scale="time")` | Builds the known phylogeny as a scipy linkage matrix from `TEM_SPECIES_DIVERGENCE_TIME_MYA`. Average (UPGMA) linkage on an ultrametric divergence-time matrix reconstructs the true tree exactly. `branch_scale="cladogram"` keeps the topology but equalises branch lengths for readability. |
| `assert_ultrametric_divergence_times(divergence_time_matrix, ...)` | Validates a divergence-time table (zero diagonal, symmetry, three-point/ultrametric condition) before it is used to build a tree; raises identifying the exact violating species triple. |
| `linkage_to_newick_string(linkage_matrix, leaf_labels)` | Renders a linkage matrix's topology as a compact Newick-style string, for QC output and tree-vs-tree comparison. |
| `get_linkage_clades(linkage_matrix, leaf_labels, include_root=False)` | Lists the leaf set (clade) produced by every internal node — the unit used for bootstrap support and clade-recovery checks. |
| `plot_phylo_tanglegram(anndata_object, group_by="specie", nest_aggregate_by=None, ...)` | Draws the tanglegram: known phylogeny \| heatmap \| data-derived morphology tree, connected by lines showing where the two trees agree or cross. Returns `baker_gamma` (headline agreement statistic), its permutation p-value, cophenetic correlation, and crossing counts. |
| `compute_tanglegram_sensitivity(anndata_object, group_by="specie", ...)` | Re-runs the morphology tree under every `cluster_method` x `cluster_metric` x standardization combination and tabulates `baker_gamma` for each — a robustness sweep so a result is never reported from one cherry-picked combination. |
| `compute_clade_support(anndata_object, group_by="specie", nest_aggregate_by=..., n_bootstrap=1000)` | Bootstrap support for every clade, resampling **subjects** with replacement (never individual mitochondria, which would destroy subject structure and inflate apparent support). `nest_aggregate_by` is required. |

---

#### Reporting — `report.py`

##### `ReportBuilder`

```python
class ReportBuilder:
    def __enter__(self) -> "ReportBuilder": ...
    def __exit__(self, ...) -> None: ...
    def save(self, output_path: str) -> None: ...
```

Context manager that captures all `matplotlib` figures and console output produced inside the `with` block, then assembles them into a multi-page PDF.

```python
from mito_marker.analysis.report import ReportBuilder

with ReportBuilder() as report:
    plot_pca_scatter(sfc_subset, group_by="condition")
    plot_mhi_distances(sfc_subset, group_by="condition")
    run_ml_analysis(sfc_subset, ml_config)

report.save("results/analysis_report.pdf")
```

---

## 7. Data Flow

### SFC pipeline

```
data/raw/sfc/*.fcs
        │
        ▼ parse_fcs_filename()    → obs_metadata dict
        ▼ load_fcs_file()         → channel_names, event_matrix
        ▼ build_sfc_anndata()     → single-file AnnData
        ▼ _concatenate_anndata_objects() → combined AnnData
        ▼ .write_h5ad()           → data/processed/sfc.h5ad
```

### TEM pipeline

```
data/raw/tem/*_MITO_measurements.txt
        │
        ▼ parse_tem_filename()    → specie, condition, subject_ID
        ▼ load_tem_file()         → feature_matrix, image_name
        ▼ build_tem_anndata()     → single-file AnnData
        ▼ _concatenate_anndata_objects() → combined AnnData
        ▼ .write_h5ad()           → data/processed/tem.h5ad
```

### Analysis pipeline

```
sfc.h5ad or tem.h5ad
        │
        ▼ configure_preprocessing() + run_preprocessing()
        │  → subset filter, feature selection, normalization layer
        ▼ assign_color_palette()
        │  → .uns['color_palette']
        ▼ Visualization
        │  plot_radar / compute_umap+plot_umap / compute_pca+plot_pca_* /
        │  plot_histogram / plot_density / plot_violin / plot_time_curve
        ▼ Statistics
        │  compare_groups / bin_obs_column / split_obs_by_threshold
        ▼ Heterogeneity
        │  compute_all_mhi / compute_amhi
        ▼ Machine Learning
        │  run_ml_analysis / run_mil_logo_cv / run_hdbscan
        ▼ Report
           ReportBuilder → PDF
```

---

## 8. Controlled Vocabulary

All metadata values are validated before they enter the dataset. Adding a new allowed value requires editing `controlled_vocabulary.py` first.

**Categorical fields** — validate by asserting the value exists in the dict:

```python
from mito_marker.controlled_vocabulary import ALLOWED_STAINING_VALUES

assert staining_value in ALLOWED_STAINING_VALUES, f"Unknown staining: '{staining_value}'"
anndata_object.obs["staining"] = staining_value
```

**Numerical fields** — validate against declared physiological bounds:

```python
from mito_marker.controlled_vocabulary import NUMERICAL_OBS_BOUNDS

bounds = NUMERICAL_OBS_BOUNDS["insulin_level_uU_per_mL"]
assert (series >= bounds["min"]).all() and (series <= bounds["max"]).all(), (
    f"Insulin out of range [{bounds['min']}, {bounds['max']}] {bounds['unit']}"
)
anndata_object.obs["insulin_level_uU_per_mL"] = series
```

Never hardcode string values (like `"DeepRed"` or `"MNMS"`) outside of `controlled_vocabulary.py`.

---

## 9. Running Tests

```bash
# Run all tests (~1552 tests)
pytest tests/

# Run with verbose output
pytest tests/ -v

# Run a single test file
pytest tests/test_analysis_mhi.py -v

# Run linting
ruff check src/
```

**Test coverage summary (~1552 tests total):**

| File | What is tested |
|---|---|
| `test_controlled_vocabulary.py` | All constants, bounds, allowed values |
| `test_inspector.py` | Console output, helper functions, group detection |
| `test_integration_pipeline.py` | End-to-end chain from ingestion through PCA |
| `test_sfc_filename_parser.py` | All SFC filename token combinations and edge cases |
| `test_sfc_ingestion.py` | Single-file and multi-file ingestion, `.h5ad` round-trip, append mode |
| `test_sfc_ingestion_mock.py` | CI-safe ingestion tests (no real `.fcs` needed) |
| `test_sfc_clinical_data.py` | Clinical CSV enrichment, column sanitization, missing values |
| `test_tem_filename_parser.py` | TEM filename token extraction for all species and conditions |
| `test_tem_file_loader.py` | Tab-separated file loading, column validation, NaN checks |
| `test_tem_anndata_builder.py` | TEM AnnData assembly, `.obs` and `.var` structure |
| `test_tem_ingestion.py` | TEM folder ingestion, concatenation, no-filename-metadata mode |
| `test_analysis_preprocessing_config.py` | Config wizard, validation, default config |
| `test_analysis_colors.py` | Color palette assignment, preference overrides, subject colors |
| `test_analysis_feature_selection.py` | MIM, CMI, HighVariance, PCALoadings scoring and `.var` writing |
| `test_analysis_normalization.py` | All transform + normalization combinations, layer naming, caching |
| `test_analysis_selection.py` | Interactive subset selection, menu logic, `.obs` filtering |
| `test_analysis_radar_plot.py` | Radar plot rendering, pulse-type splitting, feature selection integration |
| `test_analysis_umap_plot.py` | UMAP computation, caching, stratified sampling, scatter rendering |
| `test_analysis_pca_plot.py` | PCA computation, caching, scatter, biplot, top loading channels |
| `test_analysis_plot_context.py` | Plot context helpers |
| `test_analysis_stratification.py` | Quantile binning, threshold split, value mapping |
| `test_analysis_distribution_plots.py` | Histogram, density, violin rendering |
| `test_analysis_time_curve.py` | Longitudinal curve plotting, groupby logic |
| `test_analysis_mhi.py` | MHI-D/S/E computation, chimera validation, two-level aggregation |
| `test_analysis_amhi.py` | AMHI fixed-reference pipeline, new-subject invariance |
| `test_analysis_ml_config.py` | ML config validation, defaults |
| `test_analysis_ml_bagging.py` | Bag feature computation per subject |
| `test_analysis_ml_pipeline.py` | Classification and regression end-to-end, SHAP |
| `test_analysis_ml_mil.py` | MIL attention model, LOGO-CV, regression mode |
| `test_analysis_clustering.py` | HDBSCAN clustering, UMAP on bags, cluster profiling |
| `test_analysis_shap_clustering.py` | SHAP value clustering, heatmap rendering |
| `test_analysis_channel_aggregation.py` | SFC channel aggregation by color |
| `test_analysis_channel_intensity_plot.py` | Per-sample channel intensity line and bar plots |
| `test_analysis_report.py` | ReportBuilder figure capture, PDF output |

Tests use small committed fixture files in `tests/fixtures/` — no real data needed.

---

## 10. Environment Setup

### GitHub Codespaces (development)

```bash
# Install dependencies, then the package itself in editable mode
pip install -r requirements.txt
pip install -e ".[dev]"

# Run all tests
pytest tests/

# Pull a sample of raw data (optional — tests use committed fixtures)
# gdown <google-drive-folder-url> -O data/raw/sfc/ --folder
```

### Google Colab (end users)

The repository is **public**: installing the package needs no GitHub account
or token.

Always install a release tag, never `main`: the package is under active
development and past changes have included breaking renames with no
backward-compatible alias (see `docs/DECISIONS.md`), so a notebook pinned to
a tag keeps working even after the package changes upstream. The available
tags and what changed in each are listed in `CHANGELOG.md`; changes marked
**Breaking** require updating your notebook before moving to that version.

Without `@vX.Y.Z`, pip installs the current state of `main`, including
unreleased changes — not the latest release. When the requested version
number differs from the installed one, pip replaces it; when it is the same
(e.g. installing `main` over the release it started from), pip reports
"already satisfied" unless you add `--force-reinstall`. A fresh Colab runtime
starts empty, so this only matters within a running session.

```python
# Mount Google Drive (where raw .fcs / .txt files live)
from google.colab import drive
drive.mount("/content/drive")

# Install a release tag (see CHANGELOG.md for the latest one)
!pip install git+https://github.com/gaetan-sev/mito-marker.git@v0.2.1

# SFC ingestion
from mito_marker import ingest_sfc_folder

spectral_cytometry_anndata = ingest_sfc_folder(
    data_directory_path="/content/drive/MyDrive/mito-marker/data/raw/sfc/",
    output_file_path="/content/drive/MyDrive/mito-marker/data/processed/sfc.h5ad",
)

# TEM ingestion
from mito_marker import ingest_tem_folder

tem_morphology_anndata = ingest_tem_folder(
    data_directory_path="/content/drive/MyDrive/mito-marker/data/raw/tem/",
    output_file_path="/content/drive/MyDrive/mito-marker/data/processed/tem.h5ad",
)
```

Never hardcode absolute paths inside loading functions. Always pass paths as arguments so the same code runs in both environments.
