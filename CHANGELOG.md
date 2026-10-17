# Changelog

All notable changes to `mito-marker` are listed here, newest first.
Install a release with `pip install git+https://github.com/gaetan-sev/mito-marker.git@vX.Y.Z`
(see README, section 10). Lines starting with **Breaking** require updating your notebooks.

## [Unreleased]

### Added
- `ml_config["mil_device"]` (default `"auto"`): compute device of the MIL model.
  `"auto"` picks an NVIDIA GPU (CUDA), then an Apple silicon GPU (MPS), then the CPU;
  `"cpu"`, `"cuda"` and `"mps"` force one device (ADR-019).

### Changed
- MIL training now uses the Apple silicon GPU (MPS) on a Mac. Results can differ from
  a CPU run in the last decimals; set `mil_device="cpu"` to reproduce a CPU result
  exactly (ADR-019).

## [0.3.0] — 2026-10-13

TEM cristae angle and grey-level columns excluded by default (ADR-018, **Breaking**);
random rows for LOGO KernelExplainer SHAP, and a console explanation of every SHAP figure.

### Added
- `refresh_tem_non_analytical_flags()` (exported from `mito_marker` and
  `mito_marker.tem`) re-stamps `.var['is_non_analytical']` on a TEM AnnData
  loaded from a `.h5ad` saved by an older version, so the current default
  exclusions apply. It prints the features newly excluded or included.
- `TEM_IMAGE_REFERENCED_FEATURES` and `TEM_UNVALIDATED_FEATURES` (with the
  reason for each feature) in `controlled_vocabulary.py`.

### Changed
- **Breaking** (changed default): the cristae angle and the 10 grey-level
  columns are excluded from analysis by default, like the centroid coordinates
  (ADR-018). That is `CristaeOrientation_Angle`, `Mito_MeanInt`,
  `Mito_MedianInt`, `Mito_TotalInt`, `Intensity_SD`, their four `_CORR`
  variants, and both `Intensity_SD_percent` columns.
  - `TEM_NON_ANALYTICAL_FEATURES` lists 15 features instead of 4, so 13 TEM
    features are analysed by default.
  - `TEM_FEATURE_SUBSETS` no longer lists them: Intensity keeps `Skewness` and
    `Kurtosis`, and Cristae Orientation keeps 3 features.
  - Migration: after `anndata.read_h5ad()`, call
    `refresh_tem_non_analytical_flags()`.
  - To analyse one of these features (for example, grey levels compared within
    one image), set `.var.loc[feature_name, "is_non_analytical"] = False`.
    Code that read them from `TEM_FEATURE_SUBSETS` must name them explicitly.
- SHAP figures are preceded by a console block stating what they show and how
  they are computed, for every model and both evaluation strategies: explainer
  and its algorithm, explained output and its unit, meaning and sign of a SHAP
  value, meaning of a class figure (output of that class, computed on samples
  of every true class), which samples are explained, and how folds are
  combined. Each figure adds how it is aggregated and how to read it.

### Fixed
- LOGO SHAP with KernelExplainer (MLP, SVM, GaussianNB, KNeighbors) explained
  the first 20 rows of each left-out fold, in AnnData order (often the
  acquisition order; in LAVO folds, the first subject only). It now explains
  20 rows drawn at random, seeded by `[random_state, fold index]`. SHAP
  figures of these models change; TreeExplainer and LinearExplainer, which
  explain every row, are unchanged.
- LOGO regression SHAP figures no longer show a class: the bar chart and the
  beeswarm were titled with the second target value (e.g. `Class: 1.0`) and
  the console described a binary classification.

## [0.2.1] — 2026-09-28

PCA loadings panel on every PCA plot and a single loading contribution formula (ADR-016).

### Added
- Every PCA plot shows a loadings panel in its right-hand column, below the color
  legend: for each PC (PC1–PC3 in 2D, the three displayed axes in 3D) its explained
  variance and its top 5 features with their % contribution to the axis
  (loading² × 100, ADR-016). Applies to `plot_pca_scatter()`, `plot_pca_biplot()`,
  `plot_pca_trajectory()`, `plot_pca_3d_scatter()`, `plot_pca_3d_biplot()` and
  `plot_pca_3d_trajectory()`. New parameters `show_loadings` (default `True`) and
  `loadings_top_n` (default 5), plus `loadings_n_components` (default 3) on the 2D plots.
- Every function that shows PCA loadings prints how they were computed (standard or
  weighted PCA, input layer, contribution formula). The `PCALoadings` feature selection
  prints its scoring formula.

### Changed
- **Changed values**: `plot_pca_loadings_bar()` and the AMHI console now report
  contributions as loading² × 100 instead of |loading| / Σ|loading| × 100, the standard
  of FactoMineR / factoextra (ADR-016). The ranking of features within a PC is unchanged;
  the percentages are not comparable with figures made before. The cross-component
  "Total" is now weighted by each PC's explained variance (sums to 100%).
- `plot_pca_trajectory()` places its legend in the right-hand column instead of inside
  the plot, to make room for the loadings panel.

### Fixed
- With more than 15 groups, the legend placed below `plot_pca_scatter()` and
  `plot_pca_biplot()` no longer covers the x-axis label.

## [0.2.0] — 2026-09-27

First release from this dedicated repository. The package code is the development
state of 2026-09-25; only its home changed.

### Changed
- The package now lives in its own repository. Research material (experiments,
  studies, bibliography, exploratory notebooks) moved to a separate private repository.
- Demonstration notebooks moved from `notebooks/` to `examples/`, with outputs stripped.
- Installation instructions pin a release tag instead of `main`.

### Added
- Continuous integration: `pytest` and `ruff` run on every push to `main` and on pull requests.
- This changelog.

### Fixed
No behaviour change: the full test suite passes with both anndata 0.12 and 0.13.
- Compatibility with anndata 0.13, which exposes `.X` as `layers[None]`. The AllMitoMean
  reference row (`compute_all_mito_mean()` and every AMHI function built on it) no
  longer fails, and the inspector, report and subset QC list named layers only.
- Lint: `ruff check src/` passes (import order, unused imports and variables,
  f-strings without placeholders, lambda assignments, type-only `torch` import).
- CI: `actions/checkout@v5` and `actions/setup-python@v6` (Node 20 deprecation).
