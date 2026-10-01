# Changelog

All notable changes to `mito-marker` are listed here, newest first.
Install a release with `pip install git+https://github.com/gaetan-sev/mito-marker.git@vX.Y.Z`
(see README, section 10). Lines starting with **Breaking** require updating your notebooks.

## [Unreleased]

### Added
- `controlled_vocabulary.TEM_NANOMETRES_PER_PIXEL`: physical size of one pixel of the
  raw TEM images (×1200, AMT NS12 camera), to convert pixel features into nanometres
  (ADR-017).

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
