---
title: Architecture Decision Records
status: draft
owner: gaetan
created: 2026-04-28
updated: 2026-09-28
tags: [decisions, ADR, architecture, methodology]
related: [docs/PROJECT.md, docs/GLOSSARY.md]
priority: P1
---

# Architecture Decision Records (DECISIONS.md)

Append-only log of architectural and methodological decisions.
Never modify past ADRs — supersede them with a new entry that references the older one.
Before re-debating any choice, check this file first.

---

## ADR-001 — AnnData as primary data container

**Date**: <to be confirmed — backfill with actual adoption date>
**Status**: accepted

**Context**: The project needed a structured container for measurement matrices
combined with per-observation metadata, supporting thousands to millions of
observations (mitochondria). The container also needed native support for
multiple versions of the data matrix (raw vs normalised) and dimensionality
reduction results.

**Decision**: Use `AnnData` (from the `anndata` library) as the single primary
data container for all datasets. Measurements go in `.X`, metadata in `.obs`/`.var`,
normalised versions in `.layers`, embeddings in `.obsm`, configuration and ML
outputs in `.uns`.

**Alternatives considered**:
- pandas DataFrames: no native support for multi-resolution data + metadata in
  a single object; no `.obsm` / `.uns` equivalent.
- Custom dataclass: would reinvent the wheel; no ecosystem support (no `.h5ad`
  serialisation, no scanpy compatibility).

**Consequences**: All code assumes AnnData structure. Switching would require
rewriting every module. The `.h5ad` format handles large datasets efficiently
with HDF5 compression.

---

## ADR-002 — Controlled vocabulary for `.obs` validation

**Date**: <to be confirmed>
**Status**: accepted

**Context**: Early versions used hardcoded strings for condition names, staining
labels, and timepoints. Silent typos (e.g. `"deepred"` vs `"DeepRed"`) caused
silent errors in downstream comparisons and grouping.

**Decision**: Every `.obs` field must be declared in `controlled_vocabulary.py`
before use. Categorical fields use string-keyed dicts (keys = allowed values,
values = descriptions). Numerical fields use `NUMERICAL_OBS_BOUNDS` with unit,
min, and max. Validation is done via `assert value in ALLOWED_X_VALUES` before
assignment.

**Alternatives considered**:
- Runtime type checking with pydantic: heavier dependency, harder to read for
  a beginner audience.
- No validation: accepted early on, abandoned after first silent bugs.

**Consequences**: Adding a new `.obs` field requires a one-line declaration in
`controlled_vocabulary.py` first. This creates a single source of truth for the
data schema.

---

## ADR-003 — Two pipelines, one shared analysis stack

**Date**: <to be confirmed>
**Status**: accepted

**Context**: The project handles two acquisition modalities (SFC and TEM) with
very different raw formats but overlapping analytical goals (heterogeneity,
ML, visualisation). Maintaining two separate analysis stacks would double
maintenance burden.

**Decision**: Both pipelines produce AnnData objects in the same structural
shape (same `.obs` column semantics, same `.X` convention, same `.layers`
naming). All downstream analysis modules operate on AnnData and are modality-agnostic.

**Alternatives considered**:
- Separate analysis stacks per modality: more flexibility but 2× code to maintain.
- Single unified ingestion: rejected because the raw formats are too different
  (binary FCS vs tab-separated TXT, different metadata structures).

**Consequences**: Any new analysis module is immediately available for both TEM
and SFC data. Ingestion modules own the complexity of translating modality-specific
formats into the common AnnData shape.

---

## ADR-004 — Per-individual aggregation before group-level summary (MHI)

**Date**: <to be confirmed — approximately 2025>
**Status**: accepted

**Context**: An early version of the MHI computation pooled all mitochondria from
all individuals of a species/group before computing the heterogeneity index.
This conflates intra-individual variance (the target signal, reflecting individual
health state) with inter-individual variance (noise from biological diversity
across subjects).

**Decision**: MHI is computed per individual first (each subject's mitochondria
independently), then group-level summaries are computed as median ± SD of the
per-individual values. The `individual_by` parameter in `compute_mhi_d()` /
`compute_all_mhi()` controls the two-level aggregation.

**Alternatives considered**:
- Pooling across individuals: simpler code, but statistically incorrect.
- Per-image aggregation (for TEM): rejected because images from the same subject
  should be treated as replicates, not independent units.

**Consequences**: Effective n for statistical tests = number of subjects, not
number of mitochondria. Group differences are more conservative but more valid.

---

## ADR-005 — Drop MHI-E (entropy) from the core MHI framework

**Date**: <to be confirmed>
**Status**: accepted (possible future revision)

**Context**: MHI-E aimed to quantify distributional entropy of mitochondrial
morphology. Two estimators were tried:
- KNN estimator: unstable below n=100 mitochondria (high variance, frequent
  numerical errors).
- Binning estimator: consistent but compresses signal too strongly — fails to
  distinguish genuinely heterogeneous from homogeneous populations.
Neither variant produced a usable signal across the tested sample sizes.

**Decision**: MHI = MHI-D + MHI-S only. MHI-E is removed from the default
pipeline (`compute_all_mhi()` does not compute it). The code is retained but not
exported.

**Alternatives considered**:
- Keep MHI-E as optional: creates confusion about which metrics to use in reports.
- Replace with mutual information estimator: not attempted yet.

**Consequences**: MHI reports and publications use only D and S components.
Reintroduction requires a new ADR referencing this one.

---

## ADR-006 — Stabilisation threshold: CV < 5% at realistic n

**Date**: <to be confirmed>
**Status**: accepted

**Context**: MHI-D and MHI-S have been shown to stabilise (i.e. converge to a
stable value) as the number of mitochondria per individual increases. Below a
minimum n, estimates are dominated by sampling noise. A principled threshold
was needed to define "enough mitochondria."

**Decision**: A metric is considered stable when its coefficient of variation
(CV = SD / mean across bootstrap resamples) drops below 5%. Empirically, this
corresponds to:
- ~n=40–60 mitochondria for simpler distributions (e.g. mouse)
- ~n=60–80 for higher-complexity distributions (e.g. KFish)

**Alternatives considered**:
- Fixed threshold (e.g. n=50): arbitrary.
- Bootstrap confidence interval width: equivalent but harder to communicate.

**Consequences**: Acquisition protocols must target at least 60–80 events per
subject. Subjects below threshold are flagged, not silently included.

---

## ADR-007 — MIL with attention model for subject-level prediction

**Date**: <to be confirmed>
**Status**: accepted

**Context**: The relevant prediction unit is the subject (e.g. biological age),
but the input is thousands of per-mitochondrion feature vectors. Naive averaging
(mean-aggregated bag features) loses the heterogeneity information that is the
core signal of this project.

**Decision**: Implement Multiple Instance Learning with an attention pooling layer
(ABMIL architecture). Each mitochondrion receives an attention weight; the
subject-level representation is the weighted sum. Implemented in `ml_mil.py`
alongside the standard ElasticNet + bags pipeline. Supports both classification
and regression tasks.

**Alternatives considered**:
- Mean-aggregated features only: loses heterogeneity signal.
- Per-mitochondrion prediction + vote: ignores subject-level label structure.
- Transformer-based pooling: more expressive but requires much larger n.

**Consequences**: MIL adds a PyTorch dependency (`torch`). Training requires
GPU for reasonable speed (GPU-accelerated attention). ABMIL attention weights
are interpretable (per-mitochondrion contribution to prediction).

---

## ADR-008 — Effective n = number of subjects, not number of bags

**Date**: <to be confirmed>
**Status**: accepted

**Context**: Bagging creates many pseudo-observations (bags) from a small number
of subjects (e.g. 139 subjects × 10 bags = 1390 "observations"). Reporting
cross-validation performance at the bag level inflates apparent statistical power
and is methodologically incorrect.

**Decision**: Cross-validation always uses LOGO (Leave-One-Group-Out) at the
subject level. Bag count is never reported as the effective sample size. Model
performance metrics are reported per subject (OOF predictions averaged over
bags of the same subject).

**Alternatives considered**:
- Standard k-fold on bags: computationally convenient but introduces data leakage
  (bags from the same subject can appear in both train and test).

**Consequences**: Model evaluation is more conservative (fewer folds = higher
variance of estimates). This is the correct approach given the subject-level
label structure.

---

## ADR-009 — Clinical data enrichment via separate CSV join

**Date**: <to be confirmed>
**Status**: accepted

**Context**: Clinical metadata (age, BMI, activity scores, lab values, etc.)
is available for MNMS subjects in a separate CSV. An earlier approach considered
embedding this information in `.fcs` filenames or in a companion sidecar file
per acquisition.

**Decision**: Clinical data is joined to AnnData post-ingestion via
`enrich_with_clinical_data(anndata, csv_path)`. The CSV is joined on `subject_ID`.
Column names are sanitised to ASCII snake_case. Missing values are encoded as
`np.nan`.

**Alternatives considered**:
- Encode clinical data in filenames: filenames freeze a snapshot; clinical data
  evolves independently (updated lab results, corrected values).
- Embed in FCS keywords: non-standard, instrument-dependent, hard to update.

**Consequences**: Clinical CSV must be maintained separately (lives on Google
Drive, git-ignored). The join is explicit and auditable. New clinical variables
can be added without modifying any acquisition files.

---

## ADR-010 — AMHI_D as the single composite heterogeneity scalar; no new composite index

**Date**: 2026-05-02
**Status**: accepted

**Context**: The AMHI framework produces three metrics per subject/group, each
capturing a distinct biological dimension:

- `AMHI_offset` — positional shift: distance of the group centroid from the
  pan-species mean (AllMitoMean). Captures inter-group divergence, independent
  of internal spread.
- `MHI_D_absolute` — within-group dispersion: median distance of individual
  mitochondria to their own group centroid, computed in the frozen reference
  space. Captures intra-subject heterogeneity, independent of where the group
  sits relative to AllMitoMean.
- `AMHI_D` — total deviation: median distance of individual mitochondria to the
  AllMitoMean origin. Combines both effects above.

The question was raised: should these three metrics be collapsed into a single
composite scalar index for simplicity and standardizability?

**Decision**: No new composite index is created. `AMHI_D_median` already serves
as the natural single-scalar composite. It integrates both AMHI_offset and
MHI_D_absolute in one geometrically principled Euclidean norm:

    AMHI_D² ≈ AMHI_offset² + MHI_D_absolute²

This is a Pythagorean decomposition analogous to total/explained/residual
variance in ANOVA. The relationship is approximate (not exact) because
aggregation uses the median of distances rather than the mean of squared
distances; the exact identity `E[‖x_i‖²] = ‖centroid‖² + E[‖x_i − centroid‖²]`
holds for means of squares, not medians.

Cross-experiment standardizability is guaranteed by the frozen reference space:
the StandardScaler and PCA fitted on AllMitoMean are stored in
`.uns['amhi_reference']` and applied via `transform()` only (never refitted)
for any new subject or dataset.

See `docs/notes/amhi-metrics-design.md` for full conceptual note.

**Alternatives considered**:
- Weighted sum of z-scored metrics (equal weights): rejected — arbitrary equal
  weighting is mathematically unprincipled for orthogonal dimensions (FDA 2017
  guidance on composite endpoints; Krzanowski 1988).
- PCA-derived composite (PC1 of the 4-metric matrix): valid in principle but
  data-dependent weights would need to be frozen separately from the AMHI
  reference; adds complexity without benefit given AMHI_D already exists.
- Rename MHI_D_absolute to MHI_D_frozen or MHI_D_ref: deferred — would be more
  precise (the "absolute" qualifier refers to the frozen space, not the distance
  reference point), but breaks existing API and notebooks. Current docstrings now
  explain the distinction explicitly.

**Consequences**: `AMHI_D_median` is the recommended primary summary metric when
a single number is needed. The three-metric decomposition (offset + internal
spread + total) must always be reported alongside it — the scalar alone
conceals which component drives the heterogeneity.

---

## ADR-011 — Handling unbalanced species/subject/event counts (multi-species Aging paper)

**Date**: 2026-07-23
**Status**: accepted

**Context**

The multi-species Aging paper compares mitochondrial morphology (TEM) across
6 species (Droso, Human, KFish, Mouse, Worm, ZFish). Two levels of imbalance
exist at once and neither is a data-quality problem — both simply reflect how
each species was sampled:

1. **Between species**: subject count ranges 6–12, total mitochondria events
   per species range from ~3,000 (Worm) to ~15,900 (Mouse).
2. **Between subjects within a species**: e.g. within Worm, individual
   subjects range from 99 events (Worm_04) to 423 events (Worm_02) — a >4×
   spread.

In plain terms: if you dump every mitochondrion from every subject of every
species into one pile and compute a statistic on the pile, the subjects (and
species) that happened to yield the most mitochondria will dominate that
statistic — not because they are more biologically important, but purely
because they contributed more rows of data. This is the same problem the
project already solved for MHI in [[ADR-004]] ("per-individual aggregation
before group-level summary") — this ADR extends that same principle to PCA
and to the radar plot, and adds guidance for UMAP.

Three different analyses were being debated (email thread "Pour le papier
Aging", 2026-07-15/23, Séverac/Foulon/Monsarrat/Pradère), and each one reacts
differently to this imbalance, so no single fix ("balance the dataset") works
for all three:

- **UMAP**: an early attempt duplicated events per subject ("remise au pot")
  to force every subject up to the same count (e.g. 99 → 1000). This broke
  the UMAP embedding (console warnings, points collapsing into an
  undifferentiated blob — see attached figure "UMPA - Worm — Young vs Old").
  Running UMAP on the raw, unbalanced data worked normally.
- **PCA**: computed across all species at once, so a species with more total
  events pulls the shared axes of variation toward itself, and the
  loadings — the very thing being used to compare species — become biased
  toward whichever species was sampled the most.
- **Radar plot**: currently draws the mean of every raw mitochondrion in a
  group. A subject with 1,612 events (Human_016) drowns out a subject with
  268 events (Human_027) in that mean, even though both subjects should count
  equally as one biological replicate.

**Decision**

Three different, purpose-matched rules — one per analysis. None of them
invent a new statistical idea; each is the standard fix used in the
single-cell / ecology literature for exactly this situation.

1. **UMAP → use the raw, unbalanced data. Do not resample.**

   *Why oversampling broke it*: UMAP works by asking, for every point, "who
   are my nearest neighbors?" and building a map that keeps neighbors close
   together. If a mitochondrion is copy-pasted 10 times to pad a
   under-sampled subject up to a target count, those 10 copies sit at
   *exactly* zero distance from each other. UMAP has no real neighborhood to
   learn from a stack of identical copies, which is exactly the "no
   neighbors found" warning and the collapsed cloud that were observed.
   Duplicating data does not add information — it just tells the algorithm a
   lie about how many mitochondria actually look a certain way.

   *Why raw data is fine*: UMAP is being used here to answer a qualitative
   question — "do Old and Young occupy different regions of the map, or do
   they overlap?" — not a quantitative one about cluster size or density.
   Reading cluster *size* or *density* as if it were a biological
   measurement is a well-documented trap in single-cell biology, discussed
   in Chari & Pachter 2023 ("The specious art of single-cell genomics",
   PLOS Comp. Biol.) — already flagged in the email thread. The fix is not a
   resampling trick, it's a sentence in the Methods/Figure legend: *"UMAP is
   computed on all available events per species/subject without
   resampling; group sizes are unequal (see Table X); only relative
   position, not cluster density or size, should be interpreted."*

   Non-goal: SMOTE / ADASYN / Tomek links (suggested in the same thread) are
   explicitly rejected for this pipeline, for any of the three analyses.
   Those tools were built to balance classes for a supervised classifier —
   here they would invent mitochondria that were never observed under a
   microscope, which is worse than the duplication problem above, not a fix
   for it. If a supervised classifier is trained later in this project and
   its training-set class balance becomes a real problem, that is a separate
   decision to make at that time, on its own merits.

2. **PCA → weighted covariance, not a fixed subsampling threshold.**

   Undersampling every subject down to a fixed number of events (the
   500-vs-100-vs-1000 question raised in the thread) is workable but has two
   downsides: the choice of threshold is arbitrary and hard to defend to a
   reviewer, and for species with 10,000+ events per subject it throws away
   almost all of the collected data.

   Instead, PCA is computed with a **weight per mitochondrion** so that:
   - every species contributes equally to the shared axes, regardless of how
     many total events it has, and
   - within a species, every subject contributes equally, regardless of how
     many events that particular subject has.

   Concretely, each mitochondrion's weight is:

       weight = 1 / (number of subjects in its species × number of events
                      recorded for its own subject)

   A worked example makes this concrete: Worm has 12 subjects; Worm_04 has
   99 events and Worm_02 has 423. Each of Worm_04's 99 mitochondria gets
   weight 1/(12 × 99); each of Worm_02's 423 mitochondria gets the smaller
   weight 1/(12 × 423). The two subjects end up contributing the *same total
   weight* to the Worm group, and Worm as a whole contributes the same total
   weight as any other species, no matter how many mitochondria were imaged.
   This is the "weighted covariance" approach to PCA (Delchambre 2015) — a
   standard, published technique, not an experimental one. It changes how
   the PCA axes are *fit*; every individual mitochondrion is still plotted
   as its own point afterwards, exactly as before.

   Threshold-sweeping subsampling (already planned: re-running PCA at n=100,
   500, 1000 events/subject) is kept, but repositioned as a **robustness
   check** ("do the loadings still tell the same story at different sample
   sizes?"), not as the primary method.

3. **Radar plot (and any other "one mean line per group" figure) → nested
   mean, mirroring [[ADR-004]].**

   Compute the mean profile for each *subject* first, then average those
   per-subject means together to get the group's line — instead of
   averaging every raw mitochondrion together directly. Every subject then
   counts as exactly one vote in the group average, whether it contributed
   99 mitochondria or 1,612. This is precisely the "per-individual
   aggregation before group-level summary" rule [[ADR-004]] already
   established for MHI, applied here to the radar plot for consistency
   across the paper. It also matches the "pseudobulk" convention recommended
   for single-cell RNA-seq (Squair et al. 2021, *Nature Communications*,
   "Confronting false discoveries in single-cell differential expression"):
   average within each biological replicate first, and only then compare
   replicate-level averages — pooling raw cells/events directly makes a
   study look like it has far more statistical power than it really does,
   because the true sample size is the number of subjects, not the number
   of mitochondria.

   Inverse-variance ("reliability") weighting of subjects — giving a subject
   with a more stable internal signal more say in the group mean, the way a
   meta-analysis weights studies — was considered and is **not** adopted as
   the default. With as few as 6–12 subjects per species and some subjects
   at only ~100 events, the reliability estimate for a given subject would
   itself be noisy, so this would trade one arbitrary choice (which events
   get more weight) for another (how reliability is defined) without a clear
   benefit. It remains available as an optional supplementary sensitivity
   check, not the headline method.

**Alternatives considered**

- Oversampling every subject to a common target count (SMOTE / ADASYN /
  duplication with replacement) for all three analyses: rejected — fabricates
  observations that were never measured, and actively breaks UMAP's
  neighbor-graph assumptions (see Decision §1).
- Fixed-threshold undersampling (drop down to a single n for every subject,
  e.g. 100 or 500) as the *primary* method for PCA: rejected as the primary
  method — arbitrary threshold choice, discards most of the data for
  data-rich species — but kept as a secondary robustness check.
- Simple mean/median of raw pooled events (current radar behavior): rejected
  as the default — lets whichever subject was sampled most heavily dominate
  the group profile, contradicting [[ADR-004]].
- Reliability/inverse-variance weighting of subjects in the radar mean:
  considered, not adopted as default — see Decision §3. Available as an
  optional sensitivity analysis.

**Consequences**

- `compute_pca()` gains an opt-in `weight_by` argument (list of `.obs`
  columns from coarsest to finest, e.g. `["species", "subject_ID"]`) that
  switches to the weighted-covariance PCA described above. Default behavior
  (`weight_by=None`) is unchanged — existing notebooks and reports are not
  affected unless a researcher explicitly opts in.
- `plot_radar()` gains an opt-in `individual_by` argument (name of the
  `.obs` column identifying each subject) that switches the group profile
  line, and the optional global mean/median reference lines, to the nested
  (per-subject-then-per-group) average described above. Default behavior
  (`individual_by=None`) is unchanged.
- The Aging paper's Methods section must state, for each figure, which mode
  was used (raw UMAP; weighted or subsampled PCA; pooled or nested-mean
  radar) and why — this ADR is the reference to cite.
- This is a general convention, not specific to the Aging paper: any future
  multi-species or multi-cohort comparison in this project with unequal
  subject/event counts should default to `weight_by` / `individual_by`
  rather than re-debating the imbalance question from scratch.

---

## ADR-012 — Rename `individual_by` to `nest_aggregate_by`

**Date**: 2026-07-26
**Status**: accepted

**Context**: [[ADR-004]] and [[ADR-011]] introduced the `individual_by` parameter
(`compute_mhi_d()` / `compute_all_mhi()` and friends in `mhi.py`, `plot_radar()`
in `radar_plot.py`, and the per-individual centroid mode of `plot_pca_scatter()`
in `pca_plot.py`) to name the `.obs` column used to nest an aggregate one level
deeper — compute per individual first, then aggregate across individuals —
instead of pooling raw events/mitochondria directly. The name `individual_by`
only states *which column* is used, following the `group_by`-style naming
idiom, but does not convey *why* the argument exists: that setting it switches
the aggregation strategy. This made the argument's purpose unclear from the
function signature alone, without reading the docstring.

**Decision**: Rename `individual_by` to `nest_aggregate_by` everywhere it
appears in the codebase (`mhi.py`, `pca_plot.py`, `radar_plot.py`, their
tests, README.md, and example notebooks). The renamed argument keeps the
exact same semantics, position, and default (`None`) described in
[[ADR-004]] and [[ADR-011]] — only the name changes, so that "nest the
aggregate by this column" is legible from the call site itself.

**Alternatives considered**:
- `subject_id_by` / `individual_id_by`: clearer that the column is an ID
  column, but still silent on the nesting mechanism, which was the actual
  source of confusion.
- Keeping `individual_by` and only clarifying it in docstrings: rejected —
  the ambiguity is at the call-site signature, which docstrings don't fix
  for a reader who doesn't open them.

**Consequences**: Any code written against the old name (`individual_by=...`)
raises `TypeError: unexpected keyword argument` after this change and must be
updated to `nest_aggregate_by=...` — there is no backward-compatible alias.
Historical text in [[ADR-004]] and [[ADR-011]] still refers to `individual_by`
by design (append-only log); read those entries as describing the parameter
under its pre-2026-07-26 name. While touching `mhi.py`'s
`_aggregate_individual_to_group()` for this rename, an unrelated pre-existing
`F821` undefined-name bug in a dead `... if False else ...` branch (line ~497)
was also fixed, since it referenced this exact identifier.

---

## ADR-013 — Hierarchical (multi-level) `nest_aggregate_by` in `plot_radar()`

**Date**: 2026-07-26
**Status**: accepted

**Context**: [[ADR-011]] fixed one imbalance in `plot_radar()`'s aggregate
profile line — a heavily-sampled *subject* outweighing a lightly-sampled one
— via single-level `nest_aggregate_by` (renamed in [[ADR-012]]). A residual,
one-level-coarser version of the same problem surfaces whenever `group_by`
cuts across a variable that `nest_aggregate_by` does not: e.g.
`group_by="condition"` pools subjects from every species together, and
`nest_aggregate_by="unique_subject_ID"` alone gives every *subject* one vote,
but a species with more sampled subjects (e.g. Worm, 6 individuals in the
"Old" group) still gets proportionally more say in the group profile than a
species with fewer (e.g. Droso or Mouse, 3 individuals) — the exact
imbalance ADR-011 already describes as the general rule, just one hierarchy
level up. This was found from a real console-log comparison on the SFC
multi-species Aging dataset: the "specie" grouping was visually unaffected
by nesting (individuals per species are sampled at comparable depth), while
the "condition" grouping changed substantially (species are sampled at very
unequal depth) — and even after nesting by subject, species with more
subjects remained overrepresented.

**Decision**: `plot_radar()`'s `nest_aggregate_by` argument accepts either a
single `.obs` column name (unchanged, single-level behavior) or a list of
column names ordered COARSEST to FINEST — e.g.
`nest_aggregate_by=["specie", "unique_subject_ID"]` — using the exact same
ordering convention already established for `compute_pca(weight_by=...)`
([[ADR-011]]). `_nested_group_aggregate()` (`radar_plot.py`) now averages at
the finest level first, then re-averages at each coarser level in turn, so
every group at every hierarchy level counts equally, regardless of how many
rows/children it has. `mhi.py` and `pca_plot.py`'s `nest_aggregate_by` are
NOT extended by this decision — they keep single-column semantics; only
`plot_radar()` had a concrete need for this today.

**Alternatives considered**:
- A weight-vector approach mirroring `_compute_nested_equal_weights()` in
  `pca_plot.py`: rejected for `plot_radar()` specifically because
  `group_aggregate` can be `"median"`, and medians don't compose through a
  per-row weight vector the way a weighted mean/covariance does — the
  step-by-step nested-means recursion generalizes to both `"mean"` and
  `"median"` without a separate code path per aggregate.
- Requiring the caller to pre-aggregate to one row per subject before
  calling `plot_radar()`: rejected — it would silently drop the raw-event
  scatter cloud, which `nest_aggregate_by` is documented to leave untouched.

**Consequences**: `nest_aggregate_by` is now `Optional[Union[str, List[str]]]`
in `plot_radar()`; existing single-string call sites are unaffected
(identical output, verified by the existing single-level test suite).
`_nested_group_aggregate()`'s second argument changed from a 1-D array to a
1-D-or-2-D array — an internal, non-breaking change since it is a private
(`_`-prefixed) helper.

---

## ADR-014 — One 4-way TEM feature categorization (`TEM_FEATURE_SUBSETS`) for every figure

**Date**: 2026-08-31
**Status**: accepted

**Context**: TEM morphological features were grouped three different ways, in
three separate constants in `controlled_vocabulary.py`, each maintained by
hand:

- `TEM_FEATURE_SUBSETS` — 3 categories (`Morphology`, `Intensity`, `Cristae`),
  used by `run_feature_subset_challenge()` to ask which feature group carries
  the most predictive signal.
- `TEM_RADAR_FEATURE_CATEGORIES` + `TEM_RADAR_CATEGORY_COLORS` — 2 categories
  (`Morphology`, `Ultrastructure`), used by `plot_radar()` for spoke-block
  grouping and spoke-label colour.
- `TEM_FEATURE_CATEGORIES` + `_ORDER` + `_COLORS` — 4 categories (`Size`,
  `Shape`, `Intensity`, `Cristae`), used by `plot_phylo_tanglegram()` for the
  heatmap column strip. This split separates `Morphology` into `Size`
  (scale-dependent, pixels/pixels²) and `Shape` (dimensionless outline ratios)
  because in a cross-species comparison absolute size and outline shape are
  expected to behave very differently.

The three groupings had drifted apart, could contradict each other, and a new
feature had to be added to all three. The 2-way radar split in particular was
coarser than the biology supports.

**Decision**: A single 4-way categorization is the source of truth for every
figure that groups TEM features:

- `Size` — Mito_Area, Mito_Perimeter, AreaPerimeter_Ratio, Mito_Feret_Diameter
- `Shape` — Mito_Circularity, Mito_Roundness, Mito_Solidity, Mito_AR
- `Intensity` — the 12 intensity/distribution features (unchanged)
- `Cristae Orientation` — the 4 CristaeOrientation_* features (renamed from
  `Cristae` for precision — the features measure orientation, not cristae
  count or density)

`TEM_FEATURE_SUBSETS` holds this mapping (category → feature list).
`TEM_FEATURE_CATEGORY_ORDER` (= `list(TEM_FEATURE_SUBSETS)`) and
`TEM_FEATURE_CATEGORIES` (the inverse feature → category map) are DERIVED from
it in `controlled_vocabulary.py`, so they cannot fall out of sync.
`TEM_FEATURE_CATEGORY_COLORS` carries one hex colour per category (+ an
`Other` fallback) and is the single place to recolour these blocks in any
figure. `TEM_RADAR_FEATURE_CATEGORIES` and `TEM_RADAR_CATEGORY_COLORS` are
deleted; `plot_radar()` now reads the shared constants and draws 4 coloured
spoke-blocks / a 4-entry category legend / up to 4 boundary lines instead of
2. `TEM_RADAR_FEATURE_ABBREVIATIONS` (spoke-label shortening only, orthogonal
to categorization) is kept.

**Alternatives considered**:
- Keep the 2-way radar split ("two categories are easier to read at a glance
  in a legend"): rejected — the reader loses the Size/Shape and
  Intensity/Cristae distinctions that the other figures make, and a
  publication using both figures would show inconsistent groupings.
- Keep `TEM_FEATURE_CATEGORIES` as a second hand-maintained dict: rejected —
  duplicated mappings are exactly what drifted before; deriving it removes the
  failure mode.
- Keep the name `Cristae`: rejected — `Cristae Orientation` states what the
  four features actually measure.

**Consequences**: `run_feature_subset_challenge(tem_anndata,
TEM_FEATURE_SUBSETS)` now compares 4 subsets, not 3 — any saved result table
or figure from the old 3-way call must be regenerated. Code importing
`TEM_RADAR_FEATURE_CATEGORIES` / `TEM_RADAR_CATEGORY_COLORS` breaks (there is
no alias) and must switch to `TEM_FEATURE_CATEGORIES` /
`TEM_FEATURE_CATEGORY_COLORS`. The Size/Shape split itself is not new — it was
already the tanglegram's grouping (unversioned); this ADR promotes it to the
package-wide convention and records it.

---

## ADR-015 — Reusable cluster models in one frozen reference space (fit once, predict without refit)

**Date**: 2026-09-23
**Status**: accepted

**Context**: The team needs to learn mitochondrial "types" on one population
(e.g. the 25% smallest mitochondria of Young animals), keep that model under a
name, and later re-apply it to another population (Old animals, IF vs AL, a new
experiment) to compare how the types are distributed. Until now nothing in the
package allowed it: `transform_and_normalize()` / `run_preprocessing()` refit a
`StandardScaler` on every dataset and discarded it, `compute_pca()` did not keep
its centring mean, `compute_umap()` discarded its reducer, and the existing
clustering functions (`run_hdbscan`, `cluster_shap_values`) returned labels only.
Z-scoring the Old dataset with its own means would erase precisely the Young/Old
difference the comparison is looking for.

**Decision**:
1. **One frozen reference space, built by the existing functions.** The
   normalization, the PCA and (optionally) the UMAP are fitted ONCE on an
   explicit reference dataset — by default the full dataset (Young + Old) BEFORE
   any subset — and are only ever re-applied with `transform()` afterwards. The
   functions now keep what they fit: `.uns['layer_parameters'][layer]`
   (means / SDs / bounds, h5ad-safe arrays), `.uns['pca_mean']` +
   `.uns['pca_fingerprint']`, and `compute_umap(keep_reducer=True,
   embed_all_events=True)`. Each fitted transformation carries a 12-character
   SHA-256 fingerprint, so the QC can prove two datasets share the same scaler.
2. **Subsets inherit the space.** `subset_by_obs_values()` /
   `subset_by_feature_values()` return copies that keep `.layers`, `.obsm` and
   `.uns`, and record every step (thresholds applied, counts per subject) in
   `.uns['analysis_config']['subset_history']`. The scaler is therefore fitted
   on the reference, NOT on the training subset: z-scoring the 25% smallest
   mitochondria on themselves would stretch the compressed `Mito_Area` range
   back to unit variance and make values incomparable with the full dataset.
3. **`fit_cluster_model()` computes no space.** It reads the scaled layer,
   `X_pca` or the UMAP coordinates, fits KMeans (k = 2…`max_clusters`, chosen
   by silhouette) or a Gaussian mixture (chosen by BIC), names clusters
   `Cluster_1…k` by decreasing training size, and copies the frozen chain into a
   `MitoClusterModel`, saved as a self-contained `.joblib` file.
4. **`predict_cluster_model()` never refits.** It reuses the stored coordinates
   when the dataset's fingerprints match the model (a subset of the reference),
   otherwise it projects from raw `.X` (`project_to_reference_space()`).
5. **Statistics on subjects.** `compare_cluster_proportions()` tests per-subject
   cluster proportions (Mann-Whitney / Kruskal-Wallis, BH across clusters),
   n = subjects (ADR-004, ADR-008).

**Alternatives considered**:
- Refit the scaler on each dataset (current behaviour of the normalization):
  rejected — it removes between-group location differences by construction.
- Fit the scaler on the training subset only: rejected as default (see Decision
  §2); still possible by building the space on the subset, and the fit QC then
  warns that the space was probably computed after subsetting.
- Let `fit_cluster_model()` compute its own PCA/UMAP: rejected — duplicates
  `compute_pca()` / `compute_umap()` and hides which space was used; the user
  asked to reuse the existing functions.
- HDBSCAN / agglomerative clustering: not in v1 — no native `predict()`;
  assignment of new points would be an approximation (nearest core point).
- Leiden / graph clustering (single-cell standard): rejected for this use —
  no out-of-sample assignment, extra dependencies.
- Weighted GMM: scikit-learn's `GaussianMixture` has no sample weights, and
  resampling subjects to equal size is rejected by ADR-011; `weight_by` is
  therefore KMeans-only and raises an explicit error with GMM.

**Consequences**:
- New modules `feature_subset.py`, `cluster_model.py`, `cluster_plots.py`.
- `transform_and_normalize()`, `run_preprocessing()`, `compute_pca()` and
  `compute_umap()` write extra `.uns` keys; default behaviour and outputs are
  unchanged. Layers / PCAs computed with an older version have no frozen
  parameters and cannot serve as a reference space: they must be recomputed
  (explicit error message).
- A model is only reusable on data projected into its reference space; the
  `.joblib` file contains that space. Loading a `.joblib` executes code — only
  load files produced by the team.
- Clustering in UMAP space is supported but warned against (distorted
  distances, Chari & Pachter 2023, cf. ADR-011); `pca` is the recommended space.
- The per-subject fraction option (`within_group`) makes
  `pct_of_total_population` redundant (= `pct_of_subset × fraction`); it is
  informative only for absolute-threshold subsets.

---

## ADR-016 — PCA loading contribution = squared loading (FactoMineR / factoextra convention)

**Date**: 2026-09-28
**Status**: accepted

**Context**: The package reported "% contribution of a feature to a PC" with
two different formulas. `plot_pca_loadings_bar()` and the AMHI console used
|loading| / Σ|loadings| × 100; `plot_pca_biplot()` and `plot_pca_trajectory()`
printed loading² × 100. For the same feature and the same PC the two numbers
differ widely (synthetic 11-feature example: 56.7% vs 31.9% for the top
feature of a PC; the top 5 features sum to 96% vs 80%). The ranking of
features within one PC is identical (both are increasing functions of
|loading|), but the values, and any ranking of totals across several PCs,
are not. The new loadings panel on every PCA plot needed one formula.

**Decision**:
1. **Contribution of feature j to PC k = loading_jk² × 100**, where the
   loadings are the unit-norm eigenvectors stored in `.uns['pca_loadings']`.
   On each PC the contributions of all features sum to 100%. This is the
   definition of FactoMineR (`PCA()$var$contrib`) and factoextra
   (`fviz_contrib()`), the most used PCA toolkits in the life sciences, and of
   Abdi & Williams (2010, *WIREs Comput Stat* 2:433): since the squared
   coefficients of a unit eigenvector sum to 1, each squared coefficient is
   the share of the axis carried by that variable. On standardized data it is
   also the share of the PC's variance attributable to the variable.
2. **Total contribution over several PCs is weighted by their variance**:
   Σ_k(contribution_jk × variance_k) / Σ_k variance_k — factoextra's
   `fviz_contrib(axes = 1:n)`. A PC that explains little variance weighs
   little, and totals still sum to 100%.
3. **Every function that shows loadings states the method in the console**:
   standard or weighted PCA (ADR-011), input layer, and this formula.
4. Implemented once in `pca_plot._compute_loading_contributions()` and
   `_compute_total_contributions()`; used by the loadings panel of all PCA
   plots, `plot_pca_loadings_bar()` and the AMHI console.

**Alternatives considered**:
- |loading| / Σ|loadings| × 100 (previous `plot_pca_loadings_bar()`): no
  variance or geometric interpretation; it compresses differences (a dominant
  feature looks smaller, negligible features look larger). Not used by any
  reference library.
- Raw signed loadings only (scanpy `pl.pca_loadings`, Bioconductor PCAtools
  `plotloadings`): keeps the sign but is not a percentage, which is what the
  panels are meant to show. The sign is kept in the panel as "+"/"-".
- Unweighted sum of contributions across PCs (previous cross-component
  "Total"): lets a low-variance PC count as much as PC1; totals range 0–n×100%.

**Consequences**:
- `plot_pca_loadings_bar()` and the AMHI console print different percentages
  than before; the ranking of features within a PC does not change, the
  cross-component total and its ordering may. Figures produced before this
  ADR are not comparable number-for-number.
- The `PCALoadings` feature-selection score (`_score_pca_loadings`,
  Σ_k variance_ratio_k × |loading_jk|) is a ranking score, not a percentage,
  and is left unchanged so that feature selections already run stay
  reproducible; the console now prints its formula. Aligning it on squared
  loadings would change which features are selected and needs its own decision.

---

## ADR-NNN — Template for future entries

Copy and fill in for each new decision:

```markdown
## ADR-NNN — <short title>

**Date**: YYYY-MM-DD
**Status**: accepted | superseded by ADR-MMM | deprecated

**Context**: <why did this decision need to be made? what was the situation?>

**Decision**: <what was decided?>

**Alternatives considered**:
- <alternative 1>: <why rejected>
- <alternative 2>: <why rejected>

**Consequences**: <what changes as a result? what are the trade-offs?>
```
