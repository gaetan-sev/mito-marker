# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

---

## Repository scope

This repository contains only the `mito-marker` Python package: `src/`, `tests/`, `examples/` (demonstration notebooks), `docs/` (`DECISIONS.md`, `GLOSSARY.md`) and `CHANGELOG.md`. It is shared with colleagues, who install tagged releases from GitHub.

Research work that uses the package (experiments, studies, bibliography, exploratory notebooks, prototypes) lives in the private repository `mito-marker-lab`, checked out next to this one. Never add research material here. Code matures in the lab and is promoted here once it is stable, tested and useful to colleagues.

## Files to consult on session start

For non-trivial requests, read these files before responding:
- `docs/DECISIONS.md` — design and methodological choices already made (do not re-debate; supersede with a new ADR instead)
- `docs/GLOSSARY.md`  — domain terms
- `CHANGELOG.md`      — what changed in each release

---

## Project Context: Mitochondria Morphology Biomarker Research

This project uses mitochondrial morphology as a biomarker. Two distinct input pipelines feed into a shared analytical stack:

- **TEM pipeline**: Pre-extracted morphological feature files (tab-separated `.txt`, one row per mitochondrion) → AnnData. Image analysis is performed upstream by a separate tool (e.g. ImageJ/Fiji) and is outside the scope of this project. We only ingest its output.
- **SFC pipeline**: Spectral flow cytometry per-mito measurements → FlowAI / FlowJo Gating → AnnData

Both pipelines converge into the same downstream analysis: data visualization, dimensionality reduction (PCA, UMAP), statistics, machine learning, and explainability (SHAP).

**TEM input file format**: Tab-separated `.txt` files, one row per mitochondrion. Metadata columns (`Experiment_Name`, `Condition_Name`, `Image_Name`, `Mito_ID`) map to `.obs`. Morphological feature columns (`Mito_Area`, `Mito_Perimeter`, `Mito_Circularity`, `Mito_AR`, etc.) map to `.X`.

---

## 1. Environment & Commands

- **Language**: Python 3.10+
- **Development environment**: GitHub Codespaces (package/library development)
- **Execution environment**: Google Colab (end users run notebooks). All code, libraries, and visual outputs must be compatible with both.

**Install dependencies:**
```bash
pip install -r requirements.txt
```

**Run all tests:**
```bash
pytest tests/
```

**Run a single test file:**
```bash
pytest tests/test_<module_name>.py -v
```

**Run linting:**
```bash
ruff check src/
```

---

## 2. Project Structure

```
mito-marker/
├── src/
│   └── mito_marker/
│       ├── __init__.py
│       ├── controlled_vocabulary.py      # ALL allowed .obs values, species, conditions, bounds
│       ├── inspector.py                  # inspect_anndata() interactive widget
│       ├── tem/                          # TEM pipeline (fully implemented)
│       │   ├── tem_ingestion.py          # ingest_tem_folder() orchestrator
│       │   ├── tem_filename_parser.py    # parse_tem_filename()
│       │   ├── tem_file_loader.py        # load_tem_file()
│       │   └── tem_anndata_builder.py    # build_tem_anndata()
│       ├── sfc/                          # Spectral flow cytometry pipeline
│       │   ├── ingestion.py              # ingest_sfc_folder() orchestrator
│       │   ├── filename_parser.py        # parse_fcs_filename()
│       │   ├── fcs_loader.py             # load_fcs_file()
│       │   ├── anndata_builder.py        # build_sfc_anndata()
│       │   ├── clinical_data.py          # enrich_with_clinical_data()
│       │   └── qc.py                     # QC printing functions
│       └── analysis/                     # Shared downstream analysis
│           ├── preprocessing_config.py   # configure_preprocessing(), get_default_preprocessing_config()
│           ├── preprocessing_pipeline.py # run_preprocessing()
│           ├── selection.py              # select_sfc_subset()
│           ├── feature_selection.py      # select_channels(), get_selected_data_matrix()
│           ├── normalization.py          # transform_and_normalize()
│           ├── colors.py                 # assign_color_palette(), get_color_for_value()
│           ├── radar_plot.py             # plot_radar()
│           ├── umap_plot.py              # compute_umap(), plot_umap()
│           ├── pca_plot.py               # compute_pca(), plot_pca_scatter/biplot/trajectory (2D + 3D)
│           ├── stratification.py         # bin_obs_column(), split_obs_by_threshold(), map_obs_values()
│           ├── mhi.py                    # MHI — Mitochondrial Heterogeneity Index (D/S/E)
│           ├── amhi.py                   # AMHI — Absolute MHI in fixed-reference space
│           ├── distribution_plots.py     # plot_histogram(), plot_density(), plot_violin()
│           ├── time_curve.py             # plot_time_curve()
│           ├── group_comparison.py       # compare_groups() (Mann-Whitney / Kruskal-Wallis + effect sizes)
│           ├── age_plot.py               # plot_feature_vs_age()
│           ├── channel_aggregation.py    # aggregate_sfc_channels_by_color()
│           ├── channel_intensity_plot.py # plot_channel_intensity_line(), plot_channel_intensity_bar()
│           ├── clustering.py             # run_hdbscan(), compute_umap_on_bags()
│           ├── shap_clustering.py        # cluster_shap_values(), plot_shap_cluster_heatmap()
│           ├── ml_config.py              # configure_ml(), get_default_ml_config()
│           ├── ml_bagging.py             # create_bags()
│           ├── ml_pipeline.py            # run_ml_analysis(), run_feature_subset_challenge()
│           ├── ml_mil.py                 # run_mil_logo_cv(), get_mil_attention_dataframe(), plot_mil_attention()
│           └── report.py                 # ReportBuilder — PDF report generation
├── tests/
│   ├── fixtures/                         # Small committed sample files for pytest
│   └── test_*.py                         # ~1550 tests mirroring src/ structure
├── examples/                             # Demonstration notebooks (outputs stripped before commit)
├── docs/
│   ├── DECISIONS.md                      # ADRs — append-only
│   └── GLOSSARY.md                       # Domain terms
├── .github/workflows/tests.yml           # CI: pytest + ruff on every push
├── CHANGELOG.md
├── pyproject.toml
└── requirements.txt
```

---

## 3. Key Dependencies

| Library | Purpose |
|---|---|
| `anndata` | Primary data container |
| `pandas` | Tabular data handling before AnnData conversion |
| `numpy` | Numerical operations |
| `matplotlib`, `seaborn` | Visualization |
| `scikit-learn` | PCA, ML models, preprocessing |
| `umap-learn` | UMAP dimensionality reduction |
| `shap` | Model explainability |
| `torch` | MIL (Multiple Instance Learning) — GPU-accelerated attention model |
| `xgboost` | Gradient boosting classifier/regressor in ML pipeline |
| `scipy` | Statistical tests (Mann-Whitney, Kruskal-Wallis) |
| `diptest` | Bimodality detection for MHI |
| `plotly` | Interactive 3D PCA visualisations |
| `pypdf` | PDF report generation |
| `scikit-posthocs` | Post-hoc tests (Dunn) after non-parametric tests |
| `ipywidgets` | Interactive inspector widget in Jupyter/Colab |
| `pytest`, `ruff` | Testing and linting |
| `flowio` | SFC `.fcs` binary file parsing |
| `gdown` | Download raw data files from Google Drive in Codespaces |

---

## 4. Coding Style & Naming Conventions

- **PEP 8**: All code must follow PEP 8.
- **Type hints**: All function arguments and return values must have type annotations (`int`, `str`, `List`, `Optional`, etc.).
- **English only**: All code, variables, functions, comments, docstrings, visual output labels, and console output must be in English — even when user prompts are in French.
- **Simplicity first**: Write simple, readable code for a beginner audience. Prioritize clarity over optimization.
- **No generic abbreviations**: Avoid `df`, `tmp`, `calc`, `feat`, `var`. Use explicit, descriptive names.
  - Good: `mitochondria_dataframe`, `calculate_cell_area()`, `morphology_features_list`
  - Bad: `df`, `calc_area()`, `feat_list`, `tmp`
- **Domain acronyms are allowed**: Established scientific and technical terms (`TEM`, `SFC`, `PCA`, `UMAP`, `QC`, `NaN`, `SHAP`, `ML`) may be used as-is — they are vocabulary, not shortcuts.

---

## 5. Documentation & Comments

- **Docstrings**: Every function and class must have a docstring explaining its purpose, arguments, and return values.
- **Comments explain the *why***: Comment non-obvious logic, mathematical operations, and key decisions. Explain *why* something is done, not *what* the line does. Do not comment self-evident code.

---

## 6. Quality Control (QC) & Console Output

Every major pipeline step (data loading, parsing, transformation, normalization, modeling) must end with a dedicated QC block.

**Required in every QC block:**
- Print overall dimensions: `print(f"Observations: {anndata_object.n_obs}, Variables: {anndata_object.n_vars}")`
- For AnnData: print `.obs.head()`, `.var.head()`, and keys in `.obsm` and `.uns`
- Check `.X` for NaN and infinite values
- Print min, max, mean of `.X` to verify normalization/scaling
- Print the actual parameter values used in that step
- Implement sanity checks (e.g., morphological areas > 0, probabilities sum to 1)
- Memory management for Colab: delete large temporary variables with `del` and call `gc.collect()` after creating AnnData objects

**Console output standard**: A beginner must be able to follow the full execution and verify correctness without reading Python code. Print status messages before and after each major step, and print the content/shape of data structures after every transformation.

---

## 7. Core Data Structure: AnnData

Use `AnnData` as the primary data container for all datasets that combine measurements with metadata.

**Structure mapping:**
- `.X` — Quantitative measurements (morphological features or cytometry intensities). Use `.layers` for normalized/batch-corrected versions.
- `.obs` — Per-cell/per-mito/per-object metadata: subject ID, condition, timepoint (`J0`...`J20` or `D1`...`D18`), staining (`DeepRed`, `Unstained`), tube name, diagnostic/gating result, `FlowAI_Pass` (True/False).
- `.var` — Feature metadata: descriptions, `is_highly_variable`, `is_highly_correlated`.
- `.obsm` / `.uns` — Dimensionality reduction results (PCA, UMAP coordinates), ML outputs (SHAP values).

**Controlled vocabulary for `.obs` columns**: All `.obs` fields must be declared and validated using `src/mito_marker/controlled_vocabulary.py`. The validation pattern depends on the field type:

**Categorical fields** (staining, timepoint, condition, tube name, subject ID, diagnostic result): use a dictionary where keys are the allowed string values and values are plain-English descriptions.

```python
ALLOWED_STAINING_VALUES: dict[str, str] = {
    "DeepRed": "MitoTracker Deep Red FM staining",
    "Unstained": "No staining applied — negative control",
}

ALLOWED_TIMEPOINTS_SFC: dict[str, str] = {
    "J0": "Day 0 — baseline before treatment",
    "J7": "Day 7 — one week post-treatment",
    "J14": "Day 14",
    "J20": "Day 20 — end of follow-up",
}

ALLOWED_TIMEPOINTS_TEM: dict[str, str] = {
    "D1": "Day 1",
    "D18": "Day 18 — end of follow-up",
}
```

Validate by asserting the value exists in the dictionary before assigning:

```python
# Good
from mito_marker.controlled_vocabulary import ALLOWED_STAINING_VALUES
assert staining_value in ALLOWED_STAINING_VALUES, f"Unknown staining value: '{staining_value}'"
anndata_object.obs["staining"] = staining_value

# Bad — hardcoded string with no validation
anndata_object.obs["staining"] = "deepred"
```

**Numerical fields** (insulin level, steps/day, age, BMI, etc.): dictionaries of allowed values do not apply. Instead, declare the expected unit and physiological bounds in `NUMERICAL_OBS_BOUNDS`, and validate with a range assertion.

```python
NUMERICAL_OBS_BOUNDS: dict[str, dict[str, float | str]] = {
    "insulin_level_uU_per_mL": {"unit": "µU/mL", "min": 0.0, "max": 300.0},
    "steps_per_day":           {"unit": "steps",  "min": 0.0, "max": 100_000.0},
}
```

```python
from mito_marker.controlled_vocabulary import NUMERICAL_OBS_BOUNDS

bounds = NUMERICAL_OBS_BOUNDS["insulin_level_uU_per_mL"]
assert (insulin_series >= bounds["min"]).all() and (insulin_series <= bounds["max"]).all(), (
    f"Insulin values out of expected range [{bounds['min']}, {bounds['max']}] {bounds['unit']}"
)
anndata_object.obs["insulin_level_uU_per_mL"] = insulin_series
```

Any new `.obs` field — categorical or numerical — must be declared in `controlled_vocabulary.py` before being used anywhere in the codebase.

**Raw data access — never duplicate, never hardcode paths**: Raw data files live on Google Drive and must never be committed to the repository. All data loading functions must accept a `data_directory_path: str` argument so the same code works in both environments without modification:

```python
# In Google Colab (end users) — mount Drive, then pass the mounted path
from google.colab import drive
drive.mount("/content/drive")
load_sfc_data(data_directory_path="/content/drive/MyDrive/mito-marker/data/raw/sfc/")

# In GitHub Codespaces (development) — pull a small sample once with gdown, then pass the local path
load_sfc_data(data_directory_path="data/raw/sfc/")
```

Never use `os.getcwd()` or hardcoded absolute paths inside loading functions.

**Naming rule**: Never use `adata`. Use explicit names that reflect the data type:
- `mitochondria_anndata_object`
- `spectral_cytometry_anndata`
- `tem_morphology_anndata`

---

## 8. AI Interaction Rules

- **Reformulate first**: Start every response by reformulating the user's request to confirm understanding.
- **Explain before coding**: Describe the algorithmic strategy, mathematical logic, and methodology before writing any code.
- **Challenge when warranted**: Act as a bioinformatics research team. If a better analytical method or statistical validation exists in recent literature, raise it before implementing the user's approach.
- **Tone**: Direct, factual, and professional. No flattery.

## 9. General rules
- Never read more than 20 lines of a .csv or .txt or .fcs data file to understand its structure.

---

## 10. Releases, changelog and decisions

- **Changelog**: every user-visible change (new function, new parameter, changed default, bug fix) gets a line under `## [Unreleased]` in `CHANGELOG.md`, in the same commit as the change.
- **Release**: move the `Unreleased` lines under a new version heading, bump `version` in `pyproject.toml`, commit, then tag `vX.Y.Z` on that commit. Colleagues install tags, never `main`.
- **Breaking changes** (renamed or removed function or parameter, changed default): bump the minor version (before 1.0) and start the changelog line with **Breaking**, including the migration path.
- **Promotion from the lab**: code arriving from `mito-marker-lab` must meet sections 4–7 (naming, type hints, docstrings, QC block, controlled vocabulary) and come with tests before it is merged. Docstrings may cite a lab experiment as `mito-marker-lab, EXP-NNN`.
- **ADRs** (`docs/DECISIONS.md`): append-only. Never modify a past decision; supersede it with a new ADR that references it. Format: `## ADR-NNN — <title>`, then Date, Context, Decision, Alternatives considered, Consequences, Status (`accepted` | `superseded by ADR-MMM` | `deprecated`).
- **Commits**: conventional commits (`feat:`, `fix:`, `refactor:`, `test:`, `docs:`, `chore:`).
