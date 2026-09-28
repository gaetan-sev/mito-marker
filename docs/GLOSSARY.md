---
title: Glossary
status: draft
owner: gaetan
created: 2026-04-28
updated: 2026-09-28
tags: [glossary, terminology, acronyms]
related: [docs/PROJECT.md, docs/DECISIONS.md]
priority: P2
---

# Glossary

Domain-specific terms, acronyms, and abbreviations used in the `mito-marker`
project. Entries are grouped by category and kept dense (3–5 lines each).

---

## Acquisition technologies

### TEM — Transmission Electron Microscopy

Gold-standard technique for imaging mitochondrial ultrastructure at nanometre
resolution. Produces 2D cross-section images from ultra-thin biological sections.
Slow throughput (~hours per sample), requires expert preparation. Image
segmentation and feature extraction are performed upstream by ImageJ/Fiji;
`mito-marker` only ingests the resulting `.txt` measurement files.

### SFC — Spectral Flow Cytometry

High-throughput single-cell technique that measures fluorescence emission spectra
across many detectors simultaneously, enabling unmixing of overlapping fluorophores.
In this project, used to measure per-mitochondrion morphological proxies (MitoTracker
Deep Red staining captures membrane potential and morphology). Orders of magnitude
faster than TEM, amenable to clinical cohort sizes. Produces `.fcs` binary files.
See: `src/mito_marker/sfc/`.

### FCS file format

Binary file format standardised by the International Society for Advancement of
Cytometry (ISAC). Stores per-event detector readings, instrument metadata, and
keyword-value pairs. Parsed by `flowio` in `sfc/fcs_loader.py`.

---

## Mitochondrial biology

### Mitochondria

Double-membrane organelles present in virtually all eukaryotic cells. Primary role:
ATP production via oxidative phosphorylation. Beyond bioenergetics: integrate
apoptotic signals, regulate calcium homeostasis, produce reactive oxygen species
(ROS), and communicate via fission/fusion dynamics. Mitochondrial morphology
(shape, size, cristae density) is a read-out of the functional and metabolic state
of the cell.

### MIPS — Mitochondrial Information Processing System

Framework proposed by Picard & Shirihai (2022, Nature Metabolism) arguing that
mitochondria function as an integrated signalling network, not merely as isolated
ATP factories. The morphological and functional heterogeneity within a cell is
considered biologically meaningful rather than noise. This hypothesis underpins the
rationale for using intra-individual mitochondrial heterogeneity as a health signal.

### Mitochondrial morphology

Quantitative descriptors of mitochondrial shape derived from imaging or cytometry:
area, perimeter, circularity, aspect ratio (AR), Feret diameter, etc. In TEM data,
these are measured directly from segmented images (see `TEM_FEATURE_COLUMNS` in
`controlled_vocabulary.py`). In SFC, morphological proxies are inferred from
light-scatter and fluorescence parameters (forward scatter, side scatter,
MitoTracker intensity).

### Cristae

Internal membrane folds within the inner mitochondrial membrane. Cristae density
and shape are associated with ATP synthase activity and respiratory efficiency.
Visible in TEM images; not directly measurable by SFC.

---

## Heterogeneity indices

### MHI — Mitochondrial Heterogeneity Index

Umbrella term for the family of metrics computed by `src/mito_marker/analysis/mhi.py`.
Quantifies the variability of mitochondrial morphology **within** an individual,
rather than the mean phenotype. Three variants: MHI-D, MHI-S, MHI-E (the last
currently deprecated). Key design decisions in ADR-004 and ADR-005.

### MHI-D — Dispersion

Median Euclidean distance of each mitochondrion from the centroid of that
individual's distribution, in PCA-reduced space. Higher MHI-D = more spread-out
population = greater morphological heterogeneity. Computed per individual, then
aggregated to group level (median ± SD). Stable metric above ~n=40–60 mitochondria.
See `compute_mhi_d()` in `mhi.py`.

### MHI-S — Structure

Multimodality score detecting whether an individual's mitochondrial population
contains distinct subpopulations (e.g. fragmented vs elongated). Two sub-scores:
- Hartigan dip test p-value (unimodality test on the first PCA axis)
- GMM-BIC score (Gaussian Mixture Model with Bayesian Information Criterion)
Uses LOCAL per-individual PCA (not the global AnnData PCA). See `compute_mhi_s()`.

### MHI-E — Entropy (deprecated)

Entropy of the mitochondrial distribution as an information-theoretic heterogeneity
measure. Deprecated: KNN estimator was unstable below n=100; binning estimator
compressed signal too strongly. Neither variant produced a usable signal. Removed
from the default pipeline (ADR-005). May be reintroduced if sample sizes grow or a
better estimator is identified.

### AMHI — Absolute Mitochondrial Heterogeneity Index

Fixed-reference variant of MHI that enables cross-study and cross-cohort
comparisons. A reference population (e.g. all mitochondria from young subjects)
defines a StandardScaler + PCA pipeline, serialised as pickle bytes in
`.uns['amhi_reference']`. All subsequent measurements are projected into this
fixed space — transforms are never refit. Three orthogonal metrics: MHI-D-absolute
(internal spread in fixed space), AMHI-offset (centroid distance to the reference
mean), AMHI-D (total per-mito distance to reference mean). See `amhi.py`.

---

## Machine learning components

### ElasticNet

L1 + L2 regularised linear regression / classification model from scikit-learn.
Used in `ml_pipeline.py` as the baseline interpretable ML model. Operates on
bags of mitochondria (see Bagging). Feature importance via SHAP values.

### Bagging

Strategy for aggregating single-mitochondrion measurements into subject-level
feature vectors. A "bag" is a random subsample of mitochondria from one subject;
multiple bags are drawn to create pseudo-replicates. Implemented in `ml_bagging.py`
via `create_bags()`. See ADR-008 for the rule that effective n = number of subjects,
not number of bags.

### LOGO-CV — Leave-One-Group-Out Cross-Validation

Cross-validation strategy where one subject is held out at a time and the model
is trained on all remaining subjects. The "group" is the subject ID, ensuring that
no mitochondria from the test subject leak into the training set via bagging.
Used as the standard CV strategy in `ml_pipeline.py` and `ml_mil.py`.

### Standard-CV

k-fold cross-validation without subject-level grouping. Used only as a baseline
comparison; LOGO-CV is the default because it respects subject-level independence.

### MIL — Multiple Instance Learning

Paradigm where the prediction unit (subject) is represented by a bag of instances
(mitochondria). The model learns to aggregate instance-level features into a
subject-level prediction, with an attention mechanism that weights each
mitochondrion's contribution. Implemented in `ml_mil.py` as an ABMIL
(Attention-Based MIL) model with a two-layer attention pooling head. Supports
both classification and regression (ADR-007).

### SHAP — SHapley Additive exPlanations

Game-theoretic framework for explaining ML model predictions by assigning each
feature a contribution score. Used in `ml_pipeline.py` and `shap_clustering.py`
to identify which mitochondrial morphology features drive predictions. TreeSHAP
for tree models (XGBoost); KernelSHAP or LinearSHAP for ElasticNet.

### HDBSCAN

Hierarchical Density-Based Spatial Clustering of Applications with Noise.
Unsupervised clustering algorithm that does not require specifying the number of
clusters. Used in `clustering.py` to identify subpopulations of mitochondria in
PCA/UMAP space.

---

### Cluster model (`MitoClusterModel`)

A clustering (KMeans or Gaussian mixture) fitted once on a training dataset by
`fit_cluster_model()`, together with the frozen reference space it lives in.
Saved as a `.joblib` file and re-applied without refitting by
`predict_cluster_model()`. Clusters are named `Cluster_1…k` by decreasing size in
the training set. See ADR-015.

### Reference space

The space (normalization → PCA → optional UMAP) fitted once on an explicit
reference dataset and afterwards only re-applied with `transform()`. Every
dataset compared with a cluster model must live in the same reference space.

### Fingerprint

12-character SHA-256 identifier of a fitted transformation (scaler, PCA, UMAP
reducer). Identical fingerprints prove two datasets were scaled / projected
with exactly the same parameters.

## Data structures

### AnnData

Primary data container from the `anndata` Python library. Stores:
- `.X` — measurement matrix (n_obs × n_vars), always raw, never modified
- `.obs` — per-observation metadata DataFrame (subjects, conditions, etc.)
- `.var` — per-variable metadata DataFrame (feature names, selection flags)
- `.obsm` — multi-dimensional embeddings (PCA, UMAP coordinates)
- `.uns` — unstructured metadata dict (config, ML outputs, colour palettes)
- `.layers` — alternative versions of `.X` (normalised data)
Persisted as `.h5ad` files on disk. See ADR-001 for adoption rationale.

### Controlled vocabulary

All `.obs` fields must be declared in `controlled_vocabulary.py` before use.
Categorical fields use string-keyed dicts; numerical fields use bounds dicts
in `NUMERICAL_OBS_BOUNDS`. Prevents silent label typos and makes the data
schema explicit. See ADR-002.

### Layer naming convention

Format: `{transform}__{norm}` (double underscore). Examples:
`arcsinh__zscore_col`, `logicle__minmax_col`. When transform = `none` and
norm = `none`, no layer is created and `active_layer = None` (downstream reads
`.X` directly).

---

## Cohorts and external resources

### MNMS cohort

Montréal Nutrition and Metabolism Study. 139 healthy men aged 20–93, recruited
in Montréal, Canada. Extensively phenotyped: in vivo activity tests, blood panels,
mitochondrial function assays, muscle biopsies, histology. Frozen plasma samples
used in the M2 Health Index study for SFC acquisition.

### INSPIRE-T cohort

Planned robustness cross-check cohort. Details to be confirmed. Not yet in use.

### RESTORE laboratory

Research on Targeted Therapies for Inflammation and Aging. INSERM U1301,
Université Toulouse III Paul Sabatier, Toulouse, France. Host laboratory for the
`mito-marker` project and the M2 Health Index internship.

---

## Aging biology concepts

### Intrinsic capacity

WHO Healthy Ageing Framework concept (World Health Organization, 2015): all physical and
mental capacities of an individual at any given point in time — locomotion, cognition,
sensory function, vitality, psychological wellbeing. It is the composite functional reserve
of the person.

**Critical methodological point**: intrinsic capacity is meaningless measured at the
population scale. Averaging it across individuals conflates people at fundamentally
different biological stages despite sharing the same chronological age. It must be
assessed and tracked at the individual level.

Vertebrate evidence (Bedbrook et al., *Science* 2026, killifish): chronologically
age-matched animals follow distinct individual aging trajectories from as early as
70 days (analogous to ~20–30 years in humans). Individual-level behavioral trajectory
predicted future lifespan; population-mean behavior did not.

**Implication for mito-marker**: MHI and AMHI are individual-level metrics by design.
Population-mean MHI is a useful group summary, but the clinically informative quantity
is the individual MHI trajectory over time. See also "Per-individual aggregation".

See: [2026-bedbrook-killifish-aging-trajectories.md](../biblio/notes/2026-bedbrook-killifish-aging-trajectories.md)

### Life stage (aging)

Discrete, stable phase of adult life characterised by a stereotyped biological state,
with abrupt transitions between stages rather than gradual continuous decline. First
formalised for vertebrate aging by Bedbrook et al. (2026): killifish progress through
6 ordered life stages with abrupt change-points, each stage stable for days to weeks.

Parallels staging in embryonic development — suggesting staged progression may be a
universal feature of biology throughout the entire lifespan. Cross-sectional studies
mixing individuals at different life stages generate noisy results; longitudinal
within-individual tracking is required to observe transitions.

---

## Statistical concepts specific to this project

### PCA loading contribution

Share of a principal component carried by one feature: loading² × 100, where
the loading is the feature's coefficient in the unit-norm eigenvector. Sums to
100% over all features on each PC. Over several PCs, contributions are averaged
weighted by each PC's explained variance. Shown in the loadings panel of every
PCA plot. See ADR-016.

### Per-individual aggregation

MHI is computed per individual first (each subject's mitochondria independently),
then group-level summaries (e.g. per species or per age group) are computed as
median ± SD of the per-individual values. This prevents conflating intra-individual
variance (the target signal) with inter-individual variance (noise). See ADR-004.

### Effective n

The statistically meaningful sample size for a given analysis. For ML and
statistical tests: effective n = number of subjects, regardless of how many
mitochondria or bags each subject contributes. See ADR-008.

### SNR — Signal-to-Noise Ratio framework

Used to assess whether a metric carries biological signal above measurement noise.
In this project: comparing group separation (e.g. young vs old) relative to
within-group variability.

### Stabilisation threshold

A metric is considered stable when its coefficient of variation (CV) drops below
5% as a function of n (number of mitochondria per individual). Empirically:
~n=40–60 for simpler species, ~n=60–80 for higher-complexity distributions.
Used in ADR-006 to define minimum acquisition requirements.
