"""
mito_marker.analysis

Downstream analysis layer shared by the SFC and TEM pipelines. All functions
are designed to be driven interactively from Jupyter / Colab notebook cells —
non-programmer users navigate numbered menus answered with text input; no
Python knowledge is required to operate them.

Pipeline order:
    1. select_sfc_subset()       — filter events by subject, condition, age …
    2. select_channels()         — feature selection (MIM, CMI, HighVariance, PCALoadings)
    3. transform_and_normalize() — transformation (Arcsinh/Logicle) + normalization
    4. assign_color_palette()    — assign consistent hex colors to conditions
    4b. bin_obs_column()         — stratify population by quantile bins (tertiles, quartiles…)
    4b. split_obs_by_threshold() — split population into 2 groups by a fixed cut-off value
    4b. map_obs_values()         — merge categorical values into custom group labels
    5. plot_radar()              — radar plot with individual event distribution clouds
    5b. plot_channel_intensity_line() — per-sample intensity profile across SFC channels (lines)
        plot_channel_intensity_bar()  — same profile drawn as grouped bars
    6. compute_umap()            — fit UMAP on a stratified sample (cached)
       plot_umap()               — scatter colored by any .obs column
    7. compute_pca()             — fit PCA on all events (cached)
       plot_pca_scatter()        — PC1 vs PC2 scatter with centroids
       plot_pca_biplot()         — correlation circle with loading arrows
       plot_pca_trajectory()     — condition trajectories over time (centroid per condition × timepoint)
       plot_pca_loadings_bar()   — bar chart + console summary of top-N loading channels per component
    8. plot_histogram()          — overlapping density histograms per group, one subplot per feature
       plot_density()            — overlapping KDE density curves per group, one subplot per feature
       plot_violin()             — violin plots in "grouped" or "split" style
    9. configure_preprocessing() — interactive wizard to build a preprocessing_config dict
       get_default_preprocessing_config() — programmatic preprocessing config with safe defaults
       run_preprocessing()       — apply steps 1–3 non-interactively from a config dict
   10. configure_ml()            — interactive console wizard to build an ML config dict
       get_default_ml_config()  — programmatic ML config with safe defaults
       DEFAULT_MODEL_PARAMS     — expert-tuned hyperparameter presets for each model
       run_ml_analysis()        — train + evaluate one ML model on the AnnData
       run_feature_subset_challenge() — compare performance across named feature subsets
       get_sfc_feature_subsets() — build Morpho / Spectral subset dict for SFC AnnData
   14. build_species_phylogeny_linkage() / plot_phylo_tanglegram() /
       compute_tanglegram_sensitivity() / compute_clade_support() — dual-tree
       (tanglegram) comparison of the known species phylogeny against the
       data-derived morphology tree, with agreement statistics, a clustering
       sensitivity sweep and subject-level bootstrap clade support.
   13. plot_feature_clustermap() — hierarchical clustering heatmap of features by group,
       with optional dendrogram node rotation for either axis
   15. Reusable cluster models (fit once, predict on any dataset — ADR-015):
       subset_by_obs_values() / subset_by_feature_values() — non-interactive subsets
           (by .obs values; by the 25% lowest / 8% highest / range of one feature)
       get_subset_history()      — every subset step applied, with thresholds and counts
       fit_cluster_model()       — KMeans / GMM on the frozen scaled, PCA or UMAP space
       predict_cluster_model()   — assign another dataset to the learned clusters
       project_to_reference_space() — place a separately loaded dataset in the frozen space
       save_cluster_model() / load_cluster_model() / describe_cluster_model()
       plot_cluster_model_selection() / plot_cluster_radar() / plot_cluster_embedding()
       compute_cluster_proportions() / plot_cluster_proportions() / compare_cluster_proportions()
"""

from mito_marker.analysis.age_plot import plot_feature_vs_age
from mito_marker.analysis.amhi import (
    compute_all_mito_mean,
    compute_amhi,
    compute_mhi_d_absolute,
    plot_amhi_2d,
    plot_amhi_bar,
    plot_amhi_distances,
    plot_amhi_profile,
    plot_amhi_species_comparison,
    plot_amhi_umap,
    summarize_amhi,
)
from mito_marker.analysis.channel_aggregation import aggregate_sfc_channels_by_color
from mito_marker.analysis.channel_intensity_plot import (
    plot_channel_intensity_bar,
    plot_channel_intensity_line,
)
from mito_marker.analysis.cluster_model import (
    MitoClusterModel,
    describe_cluster_model,
    fit_cluster_model,
    load_cluster_model,
    predict_cluster_model,
    project_to_reference_space,
    save_cluster_model,
)
from mito_marker.analysis.cluster_plots import (
    compare_cluster_proportions,
    compute_cluster_proportions,
    plot_cluster_embedding,
    plot_cluster_model_selection,
    plot_cluster_proportions,
    plot_cluster_radar,
)
from mito_marker.analysis.clustering import (
    compare_bag_sizes,
    compute_umap_on_bags,
    evaluate_clustering,
    plot_cluster_age_composition,
    profile_clusters,
    run_hdbscan,
)
from mito_marker.analysis.clustermap_plot import plot_feature_clustermap
from mito_marker.analysis.colors import (
    assign_color_palette,
    get_color_for_value,
    get_subject_colors,
    sort_values_for_legend,
)
from mito_marker.analysis.distribution_plots import plot_density, plot_histogram, plot_violin
from mito_marker.analysis.feature_selection import get_selected_data_matrix, select_channels
from mito_marker.analysis.feature_subset import (
    get_subset_history,
    subset_by_feature_values,
    subset_by_obs_values,
)
from mito_marker.analysis.group_comparison import compare_groups
from mito_marker.analysis.mhi import (
    analyze_mhi_sensitivity,
    compute_all_mhi,
    compute_mhi_d,
    compute_mhi_e,
    compute_mhi_s,
    create_chimera,
    plot_mhi_barplot,
    plot_mhi_distances,
    plot_mhi_pca,
    plot_mhi_scatter_de,
    plot_mhi_species_heatmap,
    plot_mhi_stripplot,
    run_mhi_age_tests,
    run_mhi_statistics,
    save_mhi_to_obs,
    validate_all_chimera_pairs,
    validate_mhi_with_chimeras,
)
from mito_marker.analysis.ml_config import DEFAULT_MODEL_PARAMS, configure_ml, get_default_ml_config
from mito_marker.analysis.ml_mil import get_mil_attention_dataframe, plot_mil_attention
from mito_marker.analysis.ml_pipeline import (
    get_sfc_feature_subsets,
    plot_feature_vs_target,
    run_feature_subset_challenge,
    run_ml_analysis,
)
from mito_marker.analysis.normalization import transform_and_normalize
from mito_marker.analysis.pca_plot import (
    compute_pca,
    get_top_loading_channels,
    plot_pca_3d_biplot,
    plot_pca_3d_scatter,
    plot_pca_3d_trajectory,
    plot_pca_biplot,
    plot_pca_loadings_bar,
    plot_pca_scatter,
    plot_pca_trajectory,
)
from mito_marker.analysis.permutation_test import run_permutation_test
from mito_marker.analysis.phylo_tanglegram import (
    compute_clade_support,
    compute_tanglegram_sensitivity,
    plot_phylo_tanglegram,
)
from mito_marker.analysis.phylogeny import (
    assert_ultrametric_divergence_times,
    build_species_phylogeny_linkage,
    get_linkage_clades,
    linkage_to_newick_string,
)
from mito_marker.analysis.preprocessing_config import (
    ALLOWED_FEATURE_SELECTION_METHODS,
    ALLOWED_NORMALIZATIONS,
    ALLOWED_TRANSFORMS,
    configure_preprocessing,
    get_default_preprocessing_config,
)
from mito_marker.analysis.preprocessing_pipeline import run_preprocessing
from mito_marker.analysis.radar_plot import plot_radar
from mito_marker.analysis.report import ReportBuilder
from mito_marker.analysis.selection import select_sfc_subset
from mito_marker.analysis.shap_clustering import (
    cluster_shap_values,
    plot_shap_cluster_heatmap,
)
from mito_marker.analysis.stratification import (
    bin_obs_column,
    map_obs_values,
    split_obs_by_threshold,
)
from mito_marker.analysis.time_curve import plot_time_curve
from mito_marker.analysis.umap_plot import compute_umap, plot_umap

__all__ = [
    # Step 1 — subset selection
    "select_sfc_subset",
    # Step 2 — feature selection
    "select_channels",
    "get_selected_data_matrix",
    # Step 3 — normalization
    "transform_and_normalize",
    # Step 4 — colors
    "assign_color_palette",
    "get_color_for_value",
    "get_subject_colors",
    "sort_values_for_legend",
    # Step 4b — population stratification
    "bin_obs_column",
    "split_obs_by_threshold",
    "map_obs_values",
    # Step 4c — group statistical comparison
    "compare_groups",
    # Step 4d — feature vs age scatter plot
    "plot_feature_vs_age",
    # Step 5 — radar plot and time curves
    "plot_radar",
    "plot_time_curve",
    # Step 5b — per-sample SFC channel intensity profiles (line and bar)
    "plot_channel_intensity_line",
    "plot_channel_intensity_bar",
    # Channel aggregation (SFC: average channels by laser colour × pulse type)
    "aggregate_sfc_channels_by_color",
    # Step 6 — UMAP
    "compute_umap",
    "plot_umap",
    # Step 7 — PCA
    "compute_pca",
    "plot_pca_scatter",
    "plot_pca_biplot",
    "plot_pca_trajectory",
    "plot_pca_3d_scatter",
    "plot_pca_3d_biplot",
    "plot_pca_3d_trajectory",
    "get_top_loading_channels",
    "plot_pca_loadings_bar",
    # Step 8 — distribution plots
    "plot_histogram",
    "plot_density",
    "plot_violin",
    # Report
    "ReportBuilder",
    # Step 9 — Preprocessing config (non-interactive alternative to steps 1–3)
    "get_default_preprocessing_config",
    "configure_preprocessing",
    "run_preprocessing",
    "ALLOWED_TRANSFORMS",
    "ALLOWED_NORMALIZATIONS",
    "ALLOWED_FEATURE_SELECTION_METHODS",
    # Step 10 — ML analysis
    "configure_ml",
    "get_default_ml_config",
    "DEFAULT_MODEL_PARAMS",
    "run_ml_analysis",
    "run_feature_subset_challenge",
    "get_sfc_feature_subsets",
    "plot_feature_vs_target",
    "plot_mil_attention",
    "get_mil_attention_dataframe",
    "run_permutation_test",
    # Step 10 — Clustering analysis (bag sizes + SHAP endotypes)
    "compare_bag_sizes",
    "compute_umap_on_bags",
    "evaluate_clustering",
    "plot_cluster_age_composition",
    "profile_clusters",
    "run_hdbscan",
    "cluster_shap_values",
    "plot_shap_cluster_heatmap",
    # Step 15 — Reusable cluster models (fit / predict, frozen reference space — ADR-015)
    "subset_by_feature_values",
    "subset_by_obs_values",
    "get_subset_history",
    "MitoClusterModel",
    "fit_cluster_model",
    "predict_cluster_model",
    "project_to_reference_space",
    "save_cluster_model",
    "load_cluster_model",
    "describe_cluster_model",
    "plot_cluster_model_selection",
    "plot_cluster_radar",
    "plot_cluster_embedding",
    "compute_cluster_proportions",
    "plot_cluster_proportions",
    "compare_cluster_proportions",
    # Step 13 — Feature × group clustermap (hierarchical clustering + dendrogram rotation)
    "plot_feature_clustermap",
    # Step 14 — Phylogeny vs morphology tanglegram (dual tree + agreement statistics)
    "assert_ultrametric_divergence_times",
    "build_species_phylogeny_linkage",
    "get_linkage_clades",
    "linkage_to_newick_string",
    "plot_phylo_tanglegram",
    "compute_tanglegram_sensitivity",
    "compute_clade_support",
    # Step 11 — Absolute MHI (AMHI)
    "compute_all_mito_mean",
    "compute_amhi",
    "compute_mhi_d_absolute",
    "plot_amhi_distances",
    "plot_amhi_2d",
    "plot_amhi_umap",
    "summarize_amhi",
    "plot_amhi_profile",
    "plot_amhi_bar",
    "plot_amhi_species_comparison",
    # Step 12 — Mitochondrial Heterogeneity Index (MHI)
    "compute_mhi_d",
    "compute_mhi_s",
    "compute_mhi_e",
    "compute_all_mhi",
    "create_chimera",
    "validate_mhi_with_chimeras",
    "analyze_mhi_sensitivity",
    "plot_mhi_distances",
    "plot_mhi_pca",
    # MHI — persistence and statistics
    "save_mhi_to_obs",
    "run_mhi_statistics",
    "run_mhi_age_tests",
    # MHI — new visualizations
    "plot_mhi_barplot",
    "plot_mhi_scatter_de",
    "plot_mhi_stripplot",
    "plot_mhi_species_heatmap",
    # MHI — all-pairs chimera validation
    "validate_all_chimera_pairs",
]
