"""
mito_marker

Mitochondria morphology biomarker analysis package.

Pipelines:
    mito_marker.sfc       — Spectral flow cytometry ingestion
    mito_marker.tem       — TEM image morphology ingestion
    mito_marker.analysis  — Shared downstream analysis: PCA, UMAP, ML, SHAP

Utilities:
    mito_marker.inspector — Interactive AnnData / DataFrame explorer for non-code experts

Quick start (SFC pipeline):
    from mito_marker import ingest_sfc_folder, enrich_with_clinical_data, select_sfc_subset, compute_umap, plot_umap

    sfc_anndata = ingest_sfc_folder(
        data_directory_path="data/raw/sfc/",
        output_file_path="data/processed/sfc.h5ad",
        channels_to_exclude=["Time"],
    )
    sfc_anndata = enrich_with_clinical_data(sfc_anndata, "data/raw/clinical.csv")
    sfc_anndata = select_sfc_subset(sfc_anndata)
    sfc_anndata = compute_umap(sfc_anndata)
    plot_umap(sfc_anndata)

Quick start (inspector):
    from mito_marker import inspect_anndata

    inspect_anndata(sfc_anndata)   # displays a tabbed widget in Colab / Jupyter
"""

from importlib.metadata import version

__version__ = version("mito-marker")

from mito_marker.analysis import (
    ALLOWED_FEATURE_SELECTION_METHODS,
    ALLOWED_NORMALIZATIONS,
    ALLOWED_TRANSFORMS,
    DEFAULT_MODEL_PARAMS,
    # Reusable cluster models (fit / predict)
    MitoClusterModel,
    ReportBuilder,
    aggregate_sfc_channels_by_color,
    # MHI metrics
    analyze_mhi_sensitivity,
    # Phylogeny vs morphology tanglegram
    assert_ultrametric_divergence_times,
    assign_color_palette,
    bin_obs_column,
    build_species_phylogeny_linkage,
    cluster_shap_values,
    # Clustering & SHAP endotypes
    compare_bag_sizes,
    compare_cluster_proportions,
    compare_groups,
    compute_all_mhi,
    # AMHI metrics
    compute_all_mito_mean,
    compute_amhi,
    compute_clade_support,
    compute_cluster_proportions,
    compute_mhi_d,
    compute_mhi_d_absolute,
    compute_mhi_e,
    compute_mhi_s,
    compute_pca,
    compute_tanglegram_sensitivity,
    compute_umap,
    compute_umap_on_bags,
    configure_ml,
    configure_preprocessing,
    create_chimera,
    describe_cluster_model,
    evaluate_clustering,
    fit_cluster_model,
    get_color_for_value,
    get_default_ml_config,
    get_default_preprocessing_config,
    get_linkage_clades,
    get_mil_attention_dataframe,
    get_selected_data_matrix,
    get_sfc_feature_subsets,
    get_subject_colors,
    get_subset_history,
    get_top_loading_channels,
    linkage_to_newick_string,
    load_cluster_model,
    map_obs_values,
    plot_amhi_2d,
    plot_amhi_bar,
    plot_amhi_distances,
    plot_amhi_profile,
    plot_amhi_species_comparison,
    plot_amhi_umap,
    plot_channel_intensity_bar,
    plot_channel_intensity_line,
    plot_cluster_age_composition,
    plot_cluster_embedding,
    plot_cluster_model_selection,
    plot_cluster_proportions,
    plot_cluster_radar,
    plot_density,
    # Feature x group clustermap
    plot_feature_clustermap,
    plot_feature_vs_age,
    plot_feature_vs_target,
    plot_histogram,
    plot_mhi_barplot,
    plot_mhi_distances,
    plot_mhi_pca,
    plot_mhi_scatter_de,
    plot_mhi_species_heatmap,
    plot_mhi_stripplot,
    plot_mil_attention,
    plot_pca_3d_biplot,
    plot_pca_3d_scatter,
    plot_pca_3d_trajectory,
    plot_pca_biplot,
    plot_pca_loadings_bar,
    plot_pca_scatter,
    plot_pca_trajectory,
    plot_phylo_tanglegram,
    plot_radar,
    plot_shap_cluster_heatmap,
    plot_time_curve,
    plot_umap,
    plot_violin,
    predict_cluster_model,
    profile_clusters,
    project_to_reference_space,
    run_feature_subset_challenge,
    run_hdbscan,
    run_mhi_age_tests,
    run_mhi_statistics,
    run_ml_analysis,
    run_permutation_test,
    run_preprocessing,
    save_cluster_model,
    save_mhi_to_obs,
    select_channels,
    select_sfc_subset,
    split_obs_by_threshold,
    subset_by_feature_values,
    subset_by_obs_values,
    summarize_amhi,
    transform_and_normalize,
    validate_all_chimera_pairs,
    validate_mhi_with_chimeras,
)
from mito_marker.inspector import inspect_anndata
from mito_marker.sfc import enrich_with_clinical_data, ingest_sfc_folder
from mito_marker.tem import ingest_tem_folder

__all__ = [
    # TEM ingestion
    "ingest_tem_folder",
    # SFC ingestion
    "ingest_sfc_folder",
    "enrich_with_clinical_data",
    # Inspector
    "inspect_anndata",
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
    # Step 5 — radar plot
    "plot_radar",
    # Step 5b — per-sample SFC channel intensity profiles (line and bar)
    "plot_channel_intensity_line",
    "plot_channel_intensity_bar",
    # Step 6 — UMAP
    "compute_umap",
    "plot_umap",
    # Step 4b — population stratification
    "bin_obs_column",
    "split_obs_by_threshold",
    "map_obs_values",
    # Step 8 — distribution plots
    "plot_histogram",
    "plot_density",
    "plot_violin",
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
    # Report
    "ReportBuilder",
    # Preprocessing pipeline (non-interactive)
    "get_default_preprocessing_config",
    "configure_preprocessing",
    "run_preprocessing",
    # Step 9 — ML analysis
    "configure_ml",
    "get_default_ml_config",
    "DEFAULT_MODEL_PARAMS",
    "plot_feature_vs_target",
    "run_ml_analysis",
    "run_feature_subset_challenge",
    "get_sfc_feature_subsets",
    "plot_mil_attention",
    "get_mil_attention_dataframe",
    "run_permutation_test",
    # Preprocessing config constants
    "ALLOWED_TRANSFORMS",
    "ALLOWED_NORMALIZATIONS",
    "ALLOWED_FEATURE_SELECTION_METHODS",
    # Statistics & additional plots
    "aggregate_sfc_channels_by_color",
    "compare_groups",
    "plot_feature_vs_age",
    "plot_time_curve",
    # Clustering & SHAP endotypes
    "compare_bag_sizes",
    "compute_umap_on_bags",
    "evaluate_clustering",
    "plot_cluster_age_composition",
    "profile_clusters",
    "run_hdbscan",
    "cluster_shap_values",
    "plot_shap_cluster_heatmap",
    # Feature x group clustermap
    "plot_feature_clustermap",
    # Phylogeny vs morphology tanglegram
    "assert_ultrametric_divergence_times",
    "build_species_phylogeny_linkage",
    "get_linkage_clades",
    "linkage_to_newick_string",
    "plot_phylo_tanglegram",
    "compute_tanglegram_sensitivity",
    "compute_clade_support",
    # Step 10 — Absolute MHI (AMHI)
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
    # Step 12 — Reusable cluster models (fit / predict, frozen reference space — ADR-015)
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
    # Step 11 — Mitochondrial Heterogeneity Index (MHI)
    "compute_mhi_d",
    "compute_mhi_s",
    "compute_mhi_e",
    "compute_all_mhi",
    "create_chimera",
    "validate_mhi_with_chimeras",
    "analyze_mhi_sensitivity",
    "plot_mhi_distances",
    "plot_mhi_pca",
    "save_mhi_to_obs",
    "run_mhi_statistics",
    "run_mhi_age_tests",
    "plot_mhi_barplot",
    "plot_mhi_scatter_de",
    "plot_mhi_stripplot",
    "plot_mhi_species_heatmap",
    "validate_all_chimera_pairs",
]
