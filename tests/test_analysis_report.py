"""
Tests for mito_marker.analysis.report

Covers:
  - ReportBuilder.__enter__ / __exit__  — plt.show is patched and restored
  - _capture_and_show                   — figures captured as 300 DPI PNG buffers
  - add_plotly_figure                   — stores a buffer or placeholder
  - save_pdf                            — creates a non-empty PDF with the right page count
  - _human_readable_layer_name          — layer name token decoding
  - _human_readable_selection_method    — feature selection method label
  - _collect_metadata_lines             — correct text output from AnnData metadata
  - _build_metadata_figure              — returns a matplotlib Figure
  - _build_placeholder_figure           — returns a matplotlib Figure
  - Edge cases: AnnData with no uns['analysis_config'], empty report
"""

import io
import os
import tempfile

import anndata
import matplotlib
import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.use("Agg")  # Non-interactive backend — safe for CI and Colab

from mito_marker.analysis.report import (
    ReportBuilder,
    _build_metadata_figure,
    _build_placeholder_figure,
    _collect_metadata_lines,
    _human_readable_layer_name,
    _human_readable_selection_method,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_minimal_anndata(
    n_obs: int = 50,
    n_vars: int = 4,
    with_analysis_config: bool = True,
    with_pca: bool = False,
    with_umap_params: bool = False,
    with_unique_subject_id: bool = False,
) -> anndata.AnnData:
    """Build a minimal AnnData for testing report functions.

    Args:
        n_obs: Number of observations.
        n_vars: Number of variables.
        with_analysis_config: If True, populate uns['analysis_config'].
        with_pca: If True, add dummy pca_explained_variance_ratio to uns.
        with_umap_params: If True, add umap_params inside analysis_config.
        with_unique_subject_id: If True, add a unique_subject_ID column with
            many distinct values to test full-list display.

    Returns:
        A minimal AnnData object.
    """
    rng = np.random.default_rng(seed=0)
    x_matrix = rng.standard_normal((n_obs, n_vars)).astype(np.float32)

    obs_data: dict = {
        "specie": pd.Categorical(
            ["Mouse"] * (n_obs // 2) + ["Human"] * (n_obs - n_obs // 2)
        ),
        "age_group": pd.Categorical(
            ["Young"] * (n_obs // 2) + ["Old"] * (n_obs - n_obs // 2)
        ),
        "condition": pd.Categorical(["AL"] * n_obs),
        "subject_ID": [f"S{i:03d}" for i in range(n_obs)],
    }
    if with_unique_subject_id:
        obs_data["unique_subject_ID"] = [f"Mouse_S{i:03d}" for i in range(n_obs)]

    obs_dataframe = pd.DataFrame(obs_data, index=[f"obs_{i}" for i in range(n_obs)])

    var_dataframe = pd.DataFrame(
        {"feature_description": [f"Feature {i}" for i in range(n_vars)]},
        index=[f"CH_{i}" for i in range(n_vars)],
    )

    anndata_object = anndata.AnnData(X=x_matrix, obs=obs_dataframe, var=var_dataframe)

    if with_analysis_config:
        analysis_config: dict = {
            "active_layer": "arcsinh__zscore_col",
            "active_selection": "HighVariance",
            "selection": {"specie": ["Mouse", "Human"], "age_group": "all"},
        }
        anndata_object.var["is_selected_HighVariance"] = [True, True, False, False]

        if with_umap_params:
            analysis_config["umap_params"] = {
                "active_key": "X_umap_n15_d0.1",
                "n_neighbors": 15,
                "min_dist": 0.1,
                "random_state": 42,
                "n_events_sampled": 5000,
            }
        anndata_object.uns["analysis_config"] = analysis_config

    if with_pca:
        anndata_object.uns["pca_explained_variance_ratio"] = np.array(
            [0.35, 0.20, 0.12, 0.08], dtype=np.float32
        )

    return anndata_object


# ---------------------------------------------------------------------------
# ReportBuilder — context manager
# ---------------------------------------------------------------------------


class TestReportBuilderContextManager:
    """Tests for the __enter__ / __exit__ protocol."""

    def test_enter_patches_plt_show(self) -> None:
        """plt.show should be replaced with _capture_and_show inside the block."""
        original_show = plt.show
        with ReportBuilder():
            # Bound methods are not singletons — compare by underlying function
            assert getattr(plt.show, "__func__", None) is ReportBuilder._capture_and_show
        assert plt.show is original_show

    def test_exit_restores_plt_show(self) -> None:
        """plt.show is always restored, even if an exception is raised inside."""
        original_show = plt.show
        try:
            with ReportBuilder():
                raise RuntimeError("simulated error")
        except RuntimeError:
            pass
        assert plt.show is original_show

    def test_original_show_stored(self) -> None:
        """_original_show should reference the real plt.show during the block."""
        original_show = plt.show
        with ReportBuilder() as report:
            assert report._original_show is original_show
        assert report._original_show is None


# ---------------------------------------------------------------------------
# ReportBuilder — figure capture
# ---------------------------------------------------------------------------


class TestFigureCapture:
    """Tests for _capture_and_show: figures are captured as 300 DPI PNG buffers."""

    def test_single_figure_captured(self) -> None:
        """A figure created inside the with block should be captured."""
        with ReportBuilder() as report:
            fig, ax = plt.subplots()
            ax.plot([0, 1], [0, 1])
            plt.show()
        assert len(report._figure_buffers) == 1

    def test_multiple_figures_captured(self) -> None:
        """Each plt.show call should capture exactly the new figure at that moment."""
        with ReportBuilder() as report:
            fig1, ax1 = plt.subplots()
            ax1.plot([0], [0])
            plt.show()

            fig2, ax2 = plt.subplots()
            ax2.scatter([1], [1])
            plt.show()
        assert len(report._figure_buffers) == 2

    def test_no_figures_outside_block(self) -> None:
        """Figures created outside the with block should not be captured."""
        with ReportBuilder() as report:
            pass
        assert len(report._figure_buffers) == 0

    def test_buffer_is_valid_png(self) -> None:
        """Each captured buffer should be a valid PNG (starts with PNG signature)."""
        with ReportBuilder() as report:
            fig, ax = plt.subplots()
            ax.plot([0, 1, 2], [0, 1, 0])
            plt.show()

        buffer = report._figure_buffers[0]
        assert buffer.read(4) == b"\x89PNG"

    def test_buffer_is_readable_as_image(self) -> None:
        """PNG buffer should be readable as an image array via plt.imread."""
        with ReportBuilder() as report:
            fig, ax = plt.subplots()
            ax.scatter([0, 1], [0, 1])
            plt.show()

        buffer = report._figure_buffers[0]
        buffer.seek(0)
        image = plt.imread(buffer)
        assert image.ndim == 3  # height × width × RGBA channels


# ---------------------------------------------------------------------------
# ReportBuilder — add_plotly_figure
# ---------------------------------------------------------------------------


class TestAddPlotlyFigure:
    """Tests for add_plotly_figure: Plotly figures stored as buffer or placeholder."""

    def test_placeholder_stored_when_kaleido_missing(self) -> None:
        """A fake Plotly figure that raises on to_image should create a placeholder."""

        class FakePlotlyFigure:
            def to_image(self, **kwargs: object) -> bytes:
                raise ImportError("kaleido not available")

        report = ReportBuilder()
        report.add_plotly_figure(FakePlotlyFigure())

        assert len(report._plotly_placeholders) == 1
        assert len(report._plotly_buffers) == 0
        assert "kaleido" in report._plotly_placeholders[0].lower()

    def test_buffer_stored_when_to_image_succeeds(self) -> None:
        """A figure with a working to_image should produce a PNG buffer."""
        minimal_png = (
            b"\x89PNG\r\n\x1a\n"
            b"\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
            b"\x08\x02\x00\x00\x00\x90wS\xde"
            b"\x00\x00\x00\x0cIDATx\x9cc\xf8\xff\xff?\x00\x05\xfe\x02\xfe\xdc\xccY\xe7"
            b"\x00\x00\x00\x00IEND\xaeB`\x82"
        )

        class FakePlotlyFigure:
            def to_image(self, **kwargs: object) -> bytes:
                return minimal_png

        report = ReportBuilder()
        report.add_plotly_figure(FakePlotlyFigure())

        assert len(report._plotly_buffers) == 1
        assert len(report._plotly_placeholders) == 0


# ---------------------------------------------------------------------------
# save_pdf
# ---------------------------------------------------------------------------


class TestSavePdf:
    """Tests for save_pdf: PDF file creation."""

    def test_save_pdf_creates_file(self) -> None:
        """save_pdf should create a non-empty file at the given path."""
        anndata_object = _make_minimal_anndata()
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as temp_file:
            output_path = temp_file.name

        try:
            report = ReportBuilder()
            report.save_pdf(output_path, anndata_object=anndata_object, title="Test")
            assert os.path.exists(output_path)
            assert os.path.getsize(output_path) > 0
        finally:
            os.unlink(output_path)

    def test_save_pdf_with_captured_figures(self) -> None:
        """PDF with captured figures should be created without errors."""
        anndata_object = _make_minimal_anndata()
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as temp_file:
            output_path = temp_file.name

        try:
            with ReportBuilder() as report:
                fig, ax = plt.subplots()
                ax.plot([0, 1], [0, 1])
                plt.show()

            report.save_pdf(output_path, anndata_object=anndata_object, title="Test")
            assert os.path.getsize(output_path) > 1000
        finally:
            os.unlink(output_path)

    def test_save_pdf_empty_report(self) -> None:
        """PDF with no figures (metadata only) should still be valid."""
        anndata_object = _make_minimal_anndata(with_analysis_config=False)
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as temp_file:
            output_path = temp_file.name

        try:
            report = ReportBuilder()
            report.save_pdf(
                output_path, anndata_object=anndata_object, title="Empty Report"
            )
            assert os.path.getsize(output_path) > 0
        finally:
            os.unlink(output_path)

    def test_save_pdf_with_placeholder(self) -> None:
        """PDF should include placeholder pages for non-renderable Plotly figures."""
        anndata_object = _make_minimal_anndata()
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as temp_file:
            output_path = temp_file.name

        class FakePlotlyFigure:
            def to_image(self, **kwargs: object) -> bytes:
                raise ImportError("no kaleido")

        try:
            report = ReportBuilder()
            report.add_plotly_figure(FakePlotlyFigure())
            report.save_pdf(output_path, anndata_object=anndata_object, title="Test")
            assert os.path.getsize(output_path) > 0
        finally:
            os.unlink(output_path)


# ---------------------------------------------------------------------------
# _human_readable_layer_name
# ---------------------------------------------------------------------------


class TestHumanReadableLayerName:
    """Tests for the layer name → human-readable description conversion."""

    def test_none_returns_raw_description(self) -> None:
        """None active_layer should indicate raw .X is used."""
        result = _human_readable_layer_name(None)
        assert "raw" in result.lower() or "none" in result.lower()
        assert ".X" in result

    def test_arcsinh_zscore_col(self) -> None:
        """arcsinh__zscore_col → Arcsinh + Z-score per column."""
        result = _human_readable_layer_name("arcsinh__zscore_col")
        assert "Arcsinh" in result
        assert "Z-score" in result

    def test_logicle_minmax_col(self) -> None:
        """logicle__minmax_col → Logicle + Min-max scaling."""
        result = _human_readable_layer_name("logicle__minmax_col")
        assert "Logicle" in result
        assert "Min-max" in result

    def test_none_zscore_col(self) -> None:
        """none__zscore_col → Z-score only (no transform label)."""
        result = _human_readable_layer_name("none__zscore_col")
        assert "Z-score" in result
        assert "Arcsinh" not in result
        assert "Logicle" not in result

    def test_arcsinh_zscore_col_l2norm_row(self) -> None:
        """arcsinh__zscore_col__l2norm_row → Arcsinh + Z-score + L2."""
        result = _human_readable_layer_name("arcsinh__zscore_col__l2norm_row")
        assert "Arcsinh" in result
        assert "Z-score" in result
        assert "L2" in result

    def test_logicle_only(self) -> None:
        """logicle__none → only the transform, no norm label."""
        result = _human_readable_layer_name("logicle__none")
        assert "Logicle" in result

    def test_arcsinh_l2norm_row(self) -> None:
        """arcsinh__l2norm_row → Arcsinh + L2 normalization."""
        result = _human_readable_layer_name("arcsinh__l2norm_row")
        assert "Arcsinh" in result
        assert "L2" in result


# ---------------------------------------------------------------------------
# _human_readable_selection_method
# ---------------------------------------------------------------------------


class TestHumanReadableSelectionMethod:
    """Tests for the feature selection method → human-readable label conversion."""

    def test_none_returns_all_channels_description(self) -> None:
        """None method should indicate all channels are used."""
        result = _human_readable_selection_method(None)
        assert "all" in result.lower() or "none" in result.lower()

    def test_mim(self) -> None:
        result = _human_readable_selection_method("MIM")
        assert "Mutual Information" in result
        assert "MIM" in result

    def test_cmi(self) -> None:
        result = _human_readable_selection_method("CMI")
        assert "Conditional" in result
        assert "CMI" in result

    def test_high_variance(self) -> None:
        result = _human_readable_selection_method("HighVariance")
        assert "variance" in result.lower()
        assert "HighVariance" in result

    def test_pca_loadings(self) -> None:
        result = _human_readable_selection_method("PCALoadings")
        assert "PCA" in result
        assert "PCALoadings" in result

    def test_unknown_method_returns_method_name(self) -> None:
        """An unknown method code should be passed through unchanged."""
        result = _human_readable_selection_method("SomeUnknownMethod")
        assert "SomeUnknownMethod" in result


# ---------------------------------------------------------------------------
# _collect_metadata_lines
# ---------------------------------------------------------------------------


class TestCollectMetadataLines:
    """Tests for the metadata text assembly function."""

    def test_header_contains_title(self) -> None:
        anndata_object = _make_minimal_anndata()
        lines = _collect_metadata_lines(anndata_object, title="My Analysis")
        header_block = "\n".join(lines[:10])
        assert "My Analysis" in header_block

    def test_dimensions_present(self) -> None:
        anndata_object = _make_minimal_anndata(n_obs=50, n_vars=4)
        joined = "\n".join(_collect_metadata_lines(anndata_object, "Test"))
        assert "50" in joined
        assert "4" in joined

    def test_specie_and_age_group_shown(self) -> None:
        anndata_object = _make_minimal_anndata()
        joined = "\n".join(_collect_metadata_lines(anndata_object, "Test"))
        assert "specie" in joined
        assert "Mouse" in joined
        assert "age_group" in joined

    def test_subject_id_not_shown(self) -> None:
        """subject_ID should be excluded — too many unique values to be useful."""
        anndata_object = _make_minimal_anndata()
        joined = "\n".join(_collect_metadata_lines(anndata_object, "Test"))
        # The column header should not appear (individual S000, S001 values can appear
        # via other columns, so we check the section header specifically)
        assert "subject_ID:" not in joined

    def test_unique_subject_id_full_list(self) -> None:
        """unique_subject_ID should be shown in full with no truncation."""
        # 30 unique subjects — previously the top-20 cap would truncate this
        anndata_object = _make_minimal_anndata(
            n_obs=30, with_unique_subject_id=True
        )
        joined = "\n".join(_collect_metadata_lines(anndata_object, "Test"))
        # All 30 unique values should be present
        for i in range(30):
            assert f"Mouse_S{i:03d}" in joined

    def test_normalization_shows_human_readable_label(self) -> None:
        """Active layer should be shown as a human-readable description."""
        anndata_object = _make_minimal_anndata(with_analysis_config=True)
        joined = "\n".join(_collect_metadata_lines(anndata_object, "Test"))
        # The human-readable form should appear
        assert "Arcsinh" in joined
        assert "Z-score" in joined
        # The raw code should NOT appear verbatim
        assert "arcsinh__zscore_col" not in joined

    def test_feature_selection_shows_human_readable_label(self) -> None:
        """Active selection method should be shown as a human-readable description."""
        anndata_object = _make_minimal_anndata(with_analysis_config=True)
        joined = "\n".join(_collect_metadata_lines(anndata_object, "Test"))
        assert "variance" in joined.lower()

    def test_selection_filters_shown(self) -> None:
        """Subset filters from analysis_config['selection'] should be listed."""
        anndata_object = _make_minimal_anndata(with_analysis_config=True)
        joined = "\n".join(_collect_metadata_lines(anndata_object, "Test"))
        assert "specie" in joined
        assert "age_group" in joined

    def test_no_analysis_config_graceful(self) -> None:
        """AnnData without uns['analysis_config'] should not raise."""
        anndata_object = _make_minimal_anndata(with_analysis_config=False)
        lines = _collect_metadata_lines(anndata_object, "Bare AnnData")
        assert len(lines) > 5

    def test_pca_variance_shown(self) -> None:
        anndata_object = _make_minimal_anndata(with_pca=True)
        joined = "\n".join(_collect_metadata_lines(anndata_object, "Test"))
        assert "PC01" in joined

    def test_umap_params_shown(self) -> None:
        anndata_object = _make_minimal_anndata(with_umap_params=True)
        joined = "\n".join(_collect_metadata_lines(anndata_object, "Test"))
        assert "n_neighbors" in joined
        assert "15" in joined

    def test_selected_channels_shown(self) -> None:
        anndata_object = _make_minimal_anndata(with_analysis_config=True)
        joined = "\n".join(_collect_metadata_lines(anndata_object, "Test"))
        assert "CH_0" in joined
        assert "CH_1" in joined

    def test_layer_none_displayed_clearly(self) -> None:
        anndata_object = _make_minimal_anndata(with_analysis_config=True)
        anndata_object.uns["analysis_config"]["active_layer"] = None
        joined = "\n".join(_collect_metadata_lines(anndata_object, "Test"))
        assert "raw" in joined.lower() or "none" in joined.lower()


# ---------------------------------------------------------------------------
# _build_metadata_figure
# ---------------------------------------------------------------------------


class TestBuildMetadataFigure:
    """Tests for the metadata page figure builder."""

    def test_returns_figure(self) -> None:
        anndata_object = _make_minimal_anndata()
        figure = _build_metadata_figure(anndata_object, "Test Title")
        assert isinstance(figure, plt.Figure)
        plt.close(figure)

    def test_figure_has_text(self) -> None:
        anndata_object = _make_minimal_anndata()
        figure = _build_metadata_figure(anndata_object, "Test Title")
        assert len(figure.texts) >= 1
        plt.close(figure)


# ---------------------------------------------------------------------------
# _build_placeholder_figure
# ---------------------------------------------------------------------------


class TestBuildPlaceholderFigure:
    """Tests for the placeholder page figure builder."""

    def test_returns_figure(self) -> None:
        figure = _build_placeholder_figure("Placeholder text")
        assert isinstance(figure, plt.Figure)
        plt.close(figure)

    def test_placeholder_text_in_figure(self) -> None:
        figure = _build_placeholder_figure("kaleido not installed")
        text_content = " ".join(artist.get_text() for artist in figure.texts)
        assert "kaleido" in text_content
        plt.close(figure)


# ---------------------------------------------------------------------------
# Integration: full workflow
# ---------------------------------------------------------------------------


class TestFullWorkflow:
    """End-to-end test: capture figures, then save PDF."""

    def test_full_workflow_produces_pdf(self) -> None:
        """Capturing two figures and saving to PDF should produce a valid file."""
        anndata_object = _make_minimal_anndata(
            with_analysis_config=True, with_pca=True, with_umap_params=True
        )
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as temp_file:
            output_path = temp_file.name

        try:
            with ReportBuilder() as report:
                fig1, ax1 = plt.subplots()
                ax1.plot([0, 1, 2], [0, 1, 0], label="group A")
                ax1.set_title("Test Radar Proxy")
                plt.show()

                fig2, ax2 = plt.subplots()
                ax2.scatter([0, 1], [1, 0], c=["red", "blue"])
                ax2.set_title("Test UMAP Proxy")
                plt.show()

            report.save_pdf(output_path, anndata_object=anndata_object, title="Full Test")

            assert os.path.exists(output_path)
            assert os.path.getsize(output_path) > 3000
        finally:
            os.unlink(output_path)

    def test_plt_show_restored_after_full_workflow(self) -> None:
        original_show = plt.show
        with ReportBuilder():
            pass
        assert plt.show is original_show


# ---------------------------------------------------------------------------
# ReportBuilder — capture_console mode
# ---------------------------------------------------------------------------


from mito_marker.analysis.report import _build_console_text_figures  # noqa: E402


class TestCaptureConsole:
    """Tests for ReportBuilder(capture_console=True) and save_console_pdf."""

    def test_stdout_teed_to_both_streams(self) -> None:
        """Text printed inside the block must appear on real stdout AND be captured."""
        import io as _io
        import sys as _sys

        fake_stdout = _io.StringIO()
        real_stdout = _sys.stdout
        _sys.stdout = fake_stdout

        try:
            with ReportBuilder(capture_console=True) as report:
                print("captured line")
                fig, ax = plt.subplots()
                plt.show()
        finally:
            _sys.stdout = real_stdout

        # Text reaches the notebook (primary stream)
        assert "captured line" in fake_stdout.getvalue()
        # Text is also stored in _ordered_items for the PDF
        text_items = [c for kind, c in report._ordered_items if kind == "text"]
        assert any("captured line" in t for t in text_items)

    def test_ordered_items_contain_text_and_figures(self) -> None:
        """Interleaved text + figures must be stored in order."""
        with ReportBuilder(capture_console=True) as report:
            print("before figure")
            fig, ax = plt.subplots()
            ax.plot([0, 1])
            plt.show()
            print("after figure")

        # After __exit__, remaining text is flushed to _ordered_items
        kinds = [kind for kind, _ in report._ordered_items]
        # Expect: text ("before figure"), figure, text ("after figure")
        assert "text" in kinds
        assert "figure" in kinds
        # Text before figure must come first
        assert kinds.index("text") < kinds.index("figure")

    def test_text_content_is_captured(self) -> None:
        """The exact printed text must appear in the captured text item."""
        with ReportBuilder(capture_console=True) as report:
            print("hello world")
            fig, ax = plt.subplots()
            plt.show()

        text_items = [content for kind, content in report._ordered_items if kind == "text"]
        all_text = "\n".join(text_items)
        assert "hello world" in all_text

    def test_trailing_text_after_last_figure_captured(self) -> None:
        """Text printed after the last plt.show() must also be captured."""
        with ReportBuilder(capture_console=True) as report:
            fig, ax = plt.subplots()
            plt.show()
            print("trailing text")

        text_items = [content for kind, content in report._ordered_items if kind == "text"]
        all_text = "\n".join(text_items)
        assert "trailing text" in all_text

    def test_stdout_restored_after_exit(self) -> None:
        """sys.stdout must be restored to the original after __exit__."""
        import sys as _sys

        original = _sys.stdout
        with ReportBuilder(capture_console=True):
            pass
        assert _sys.stdout is original

    def test_stdout_restored_after_exception(self) -> None:
        """sys.stdout is restored even if an exception is raised inside."""
        import sys as _sys

        original = _sys.stdout
        try:
            with ReportBuilder(capture_console=True):
                raise ValueError("boom")
        except ValueError:
            pass
        assert _sys.stdout is original

    def test_save_console_pdf_creates_non_empty_file(self) -> None:
        """save_console_pdf must write a readable non-empty PDF."""
        import os
        import tempfile

        with ReportBuilder(capture_console=True) as report:
            print("Model accuracy: 0.91")
            fig, ax = plt.subplots()
            ax.bar(["A", "B"], [0.9, 0.7])
            plt.show()
            print("Done.")

        fd, output_path = tempfile.mkstemp(suffix=".pdf")
        os.close(fd)
        try:
            report.save_console_pdf(output_path)
            assert os.path.exists(output_path)
            assert os.path.getsize(output_path) > 1000
        finally:
            os.unlink(output_path)

    def test_save_console_pdf_empty_warns(self, capsys) -> None:
        """Calling save_console_pdf with no content must print a warning."""
        import tempfile, os

        report = ReportBuilder(capture_console=True)
        fd, output_path = tempfile.mkstemp(suffix=".pdf")
        os.close(fd)
        try:
            report.save_console_pdf(output_path)
            captured = capsys.readouterr()
            assert "Nothing to save" in captured.out
        finally:
            os.unlink(output_path)

    def test_default_mode_unaffected(self) -> None:
        """ReportBuilder() without flag must work exactly as before."""
        with ReportBuilder() as report:
            fig, ax = plt.subplots()
            ax.plot([1, 2])
            plt.show()

        assert len(report._figure_buffers) == 1
        assert report._ordered_items == []


class TestBuildConsoleTextFigures:
    """Tests for the _build_console_text_figures helper."""

    def test_returns_at_least_one_figure(self) -> None:
        figures = _build_console_text_figures("hello")
        assert len(figures) >= 1
        for fig in figures:
            plt.close(fig)

    def test_empty_text_returns_one_figure(self) -> None:
        figures = _build_console_text_figures("")
        assert len(figures) == 1
        plt.close(figures[0])

    def test_long_text_paginates(self) -> None:
        """More than 80 lines must produce multiple figures."""
        long_text = "\n".join([f"Line {i}" for i in range(200)])
        figures = _build_console_text_figures(long_text)
        assert len(figures) >= 3  # 200 lines / 80 lines_per_page = 3 pages
        for fig in figures:
            plt.close(fig)

    def test_long_line_wrapped(self) -> None:
        """A line longer than 120 chars must be broken across lines."""
        wide_line = "X" * 200
        figures = _build_console_text_figures(wide_line)
        assert len(figures) >= 1
        for fig in figures:
            plt.close(fig)
