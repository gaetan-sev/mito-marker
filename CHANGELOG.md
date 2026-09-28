# Changelog

All notable changes to `mito-marker` are listed here, newest first.
Install a release with `pip install git+https://github.com/gaetan-sev/mito-marker.git@vX.Y.Z`
(see README, section 10). Lines starting with **Breaking** require updating your notebooks.

## [Unreleased]

### Added
- `plot_pca_scatter()` shows a PCA loadings panel below the color legend: for PC1–PC3
  (with their explained variance), the top 5 features and their % contribution to the
  axis (squared loading × 100). New parameters `show_loadings` (default `True`),
  `loadings_top_n` (default 5) and `loadings_n_components` (default 3).

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
