# Changelog

All notable changes to `mito-marker` are listed here, newest first.
Install a release with `pip install git+https://github.com/gaetan-sev/mito-marker.git@vX.Y.Z`
(see README, section 10). Lines starting with **Breaking** require updating your notebooks.

## [Unreleased]

## [0.2.0] — 2026-09-26

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
