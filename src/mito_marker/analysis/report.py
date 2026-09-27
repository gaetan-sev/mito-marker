"""
mito_marker.analysis.report
============================

Context-manager that silently captures every matplotlib figure produced
inside a ``with`` block and assembles them — together with a structured
dataset / pipeline metadata summary — into a multi-page PDF report.

Figures are embedded as **300 DPI PNG images**: publication quality (sharp
text, clear scatter dots), fast to render in any PDF viewer, and compact
enough for routine sharing (~500 KB–1.5 MB per scatter/UMAP figure).

The metadata page (text only) is written as vector PDF and is perfectly
sharp regardless of zoom.

Plotly 3-D interactive figures cannot be captured automatically (they do not
go through ``plt.show``).  Pass them explicitly via
``report.add_plotly_figure(fig)``; they are converted to static PNG pages
when kaleido is installed, or replaced by a text placeholder otherwise.

Typical usage
-------------
::

    import mito_marker

    with mito_marker.ReportBuilder() as report:
        mito_marker.plot_radar(sfc_subset, group_by="age_group", title="")
        mito_marker.plot_umap(sfc_subset, group_by="specie")
        mito_marker.plot_pca_scatter(sfc_subset, group_by=["specie", "age_group"])

        # 3-D Plotly figures: pass explicitly
        fig_3d = mito_marker.plot_pca_3d_scatter(sfc_subset, group_by="age_group")
        report.add_plotly_figure(fig_3d)
        fig_3d.show()

    report.save_pdf(
        "cross_species_analysis.pdf",
        anndata_object=sfc_subset,
        title="Cross-species mitochondria SFC analysis",
    )

Figures displayed in the notebook are NOT interrupted: the normal
``plt.show()`` call still happens; figures are simply captured right before.
"""

import datetime
import gc
import io
import sys
import textwrap
from typing import Any, Callable, List, Optional

import anndata
import matplotlib.figure
import matplotlib.pyplot as plt
from PIL import Image
from pypdf import PdfReader, PdfWriter

# ---------------------------------------------------------------------------
# Human-readable label dictionaries for pipeline configuration
# ---------------------------------------------------------------------------

# Maps internal transform tokens (from normalization.py) to display names.
# The empty string means "no transform" — it is omitted from the joined label.
_TRANSFORM_LABELS: dict[str, str] = {
    "arcsinh": "Arcsinh",
    "logicle": "Logicle",
    "none": "",
}

# Maps internal normalization tokens (from normalization.py) to display names.
_NORM_LABELS: dict[str, str] = {
    "zscore_col": "Z-score per column (StandardScaler)",
    "minmax_col": "Min-max scaling per column",
    "l2norm_row": "L2 normalization per row",
    "zscore_col__l2norm_row": "Z-score per column + L2 normalization per row",
    "none": "",
}

# Maps feature selection method codes (from feature_selection.py) to display names.
_SELECTION_METHOD_LABELS: dict[str, str] = {
    "MIM": "Mutual Information Maximization (MIM)",
    "CMI": "Conditional Mutual Information / mRMR (CMI)",
    "HighVariance": "Top N channels by variance (HighVariance)",
    "PCALoadings": "Top N channels by PCA loading magnitude (PCALoadings)",
}


# ---------------------------------------------------------------------------
# Internal helper — tee stream
# ---------------------------------------------------------------------------


class _TeeStream:
    """Write to two streams simultaneously.

    Used to keep print() output visible in the notebook while also
    capturing it into an in-memory buffer for the PDF report.
    """

    def __init__(self, primary: Any, secondary: io.StringIO) -> None:
        self._primary = primary
        self._secondary = secondary

    def write(self, text: str) -> int:
        self._primary.write(text)
        self._secondary.write(text)
        return len(text)

    def flush(self) -> None:
        self._primary.flush()
        self._secondary.flush()

    # Proxy all other attribute lookups to the primary stream so that
    # libraries that inspect sys.stdout (e.g. IPython) work normally.
    def __getattr__(self, name: str) -> Any:
        return getattr(self._primary, name)


# Sentinel attribute stamped directly on Figure objects to mark them as
# pre-existing (from before __enter__) or already captured.
# Using a custom attribute on the object avoids all id() and figure-number
# reuse problems: a brand-new Figure object never has these attributes.
_PRE_EXISTING_ATTR = "_mito_marker_pre_existing"
_CAPTURED_ATTR = "_mito_marker_captured"


# ---------------------------------------------------------------------------
# Public class
# ---------------------------------------------------------------------------


class ReportBuilder:
    """Capture matplotlib figures and assemble a 300 DPI PDF report.

    Use as a context manager.  All ``plt.show()`` calls that happen inside the
    ``with`` block are intercepted: each figure is saved as a 300 DPI PNG
    image so it can be embedded in the PDF at publication quality.

    Tracking is done by stamping custom attributes directly on each Figure
    object (``_mito_marker_pre_existing`` and ``_mito_marker_captured``).
    This avoids both Python id() reuse (freed object, same address) and
    matplotlib figure-number reuse (matplotlib always picks the smallest
    available integer, so closed figures' numbers are immediately recycled).

    After the ``with`` block, call :meth:`save_pdf` to write the report.
    """

    def __init__(self, capture_console: bool = False) -> None:
        """Initialise an empty ReportBuilder with no captured figures.

        Args:
            capture_console: When True, all text printed to stdout inside the
                ``with`` block is captured and stored in order with the figures.
                Call :meth:`save_console_pdf` after the block to write an
                interleaved text + figure PDF (one text page per section,
                one figure page per plot).  Normal :meth:`save_pdf` is not
                affected by this flag.
        """
        # 300 DPI PNG buffers for matplotlib figures captured via plt.show interception
        self._figure_buffers: List[io.BytesIO] = []
        # PNG buffers for Plotly figures added explicitly via add_plotly_figure
        self._plotly_buffers: List[io.BytesIO] = []
        # Text placeholders for Plotly figures that could not be rendered
        self._plotly_placeholders: List[str] = []
        # Reference to the real plt.show, saved on __enter__
        self._original_show: Optional[Callable[..., None]] = None
        # --- console capture ---
        self._capture_console: bool = capture_console
        # StringIO buffer that receives stdout while the with block runs
        self._console_buffer: io.StringIO = io.StringIO()
        # The real sys.stdout, saved on __enter__ so we can restore it
        self._original_stdout: Optional[Any] = None
        # Ordered list of ("text", str) and ("figure", BytesIO) items.
        # Populated only when capture_console=True; used by save_console_pdf().
        self._ordered_items: List[tuple[str, Any]] = []

    # ------------------------------------------------------------------
    # Context-manager protocol
    # ------------------------------------------------------------------

    def __enter__(self) -> "ReportBuilder":
        """Patch ``plt.show`` to start capturing figures.

        Records the Python object identities of figures that already exist at
        entry time so pre-existing figures from earlier notebook cells are never
        included in the report.

        Returns:
            The ReportBuilder instance, for use in ``with ... as`` syntax.
        """
        self._original_show = plt.show
        # Stamp every figure that already exists so _capture_and_show ignores them.
        # We use a custom attribute rather than tracking ids or figure numbers
        # because both can be reused after plt.close(): matplotlib always picks the
        # smallest available integer for new figure numbers, and CPython tends to
        # reuse freed object addresses.  A fresh Figure object never carries this
        # attribute, so it is guaranteed to be captured.
        for existing_num in plt.get_fignums():
            try:
                setattr(plt.figure(existing_num), _PRE_EXISTING_ATTR, True)
            except Exception:
                pass
        plt.show = self._capture_and_show  # type: ignore[assignment]
        # Tee stdout so print() appears in the notebook AND is captured
        if self._capture_console:
            self._console_buffer = io.StringIO()
            self._original_stdout = sys.stdout
            sys.stdout = _TeeStream(self._original_stdout, self._console_buffer)
        return self

    def __exit__(self, *args: Any) -> None:
        """Restore the original ``plt.show`` (and stdout when capture_console=True)."""
        if self._original_show is not None:
            plt.show = self._original_show  # type: ignore[assignment]
            self._original_show = None
        # Restore stdout and flush whatever text remained after the last figure
        if self._capture_console and self._original_stdout is not None:
            sys.stdout = self._original_stdout
            self._original_stdout = None
            remaining_text = self._console_buffer.getvalue()
            if remaining_text.strip():
                self._ordered_items.append(("text", remaining_text))
            self._console_buffer = io.StringIO()

    # ------------------------------------------------------------------
    # Internal capture helper
    # ------------------------------------------------------------------

    def _capture_and_show(self, *args: Any, **kwargs: Any) -> None:
        """Save new figures as 300 DPI PNG buffers, then display normally.

        Only captures figures that:
          (a) were not present before ``__enter__`` (no _PRE_EXISTING_ATTR), and
          (b) have not already been captured (no _CAPTURED_ATTR), and
          (c) have at least one axes (empty figures are skipped).

        Sentinel attributes stamped on the Figure object itself are immune to
        both id() reuse (same memory address, new object) and matplotlib figure-
        number reuse (matplotlib picks the smallest available integer).
        """
        for figure_number in plt.get_fignums():
            figure = plt.figure(figure_number)
            if getattr(figure, _PRE_EXISTING_ATTR, False):
                continue  # existed before __enter__, ignore
            if getattr(figure, _CAPTURED_ATTR, False):
                continue  # already captured in a prior plt.show() call
            if not figure.get_axes():
                continue  # empty figure (e.g. plt.figure() placeholder before SHAP)
            # Mark as captured before calling original_show() so re-entrant
            # calls (unusual but possible) cannot double-capture this figure.
            setattr(figure, _CAPTURED_ATTR, True)
            if self._capture_console:
                # Flush text printed since the last figure (or since __enter__)
                # as a text item before storing this figure.
                text_so_far = self._console_buffer.getvalue()
                if text_so_far.strip():
                    self._ordered_items.append(("text", text_so_far))
                # Reset the buffer and keep tee-ing stdout into it
                self._console_buffer = io.StringIO()
                sys.stdout = _TeeStream(self._original_stdout, self._console_buffer)
                # Capture figure into ordered items instead of figure_buffers
                buffer = io.BytesIO()
                figure.savefig(buffer, format="png", dpi=300, bbox_inches="tight")
                buffer.seek(0)
                self._ordered_items.append(("figure", buffer))
            else:
                buffer = io.BytesIO()
                figure.savefig(buffer, format="png", dpi=300, bbox_inches="tight")
                buffer.seek(0)
                self._figure_buffers.append(buffer)

        # Forward to the real plt.show so the user still sees figures in the notebook
        if self._original_show is not None:
            self._original_show(*args, **kwargs)

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

    def add_plotly_figure(self, figure: Any) -> None:
        """Add a Plotly interactive figure to the report.

        Because Plotly figures do not go through ``plt.show``, they must be
        passed here explicitly.  The figure is converted to a static PNG image
        if the ``kaleido`` package is installed; otherwise a text placeholder
        is inserted in the PDF.

        Args:
            figure: A Plotly ``go.Figure`` returned by ``plot_pca_3d_scatter``,
                ``plot_pca_3d_biplot``, or ``plot_pca_3d_trajectory``.
        """
        try:
            image_bytes = figure.to_image(format="png", scale=2)
            buffer = io.BytesIO(image_bytes)
            buffer.seek(0)
            self._plotly_buffers.append(buffer)
        except Exception:
            layout = getattr(figure, "layout", None)
            figure_title = (
                getattr(getattr(layout, "title", None), "text", None)
                or "Interactive 3-D figure"
            )
            self._plotly_placeholders.append(
                f"[Interactive figure — not embedded in PDF]\n"
                f"Title: {figure_title}\n\n"
                f"To embed 3-D figures as static images, install kaleido:\n"
                f"    pip install kaleido"
            )

    def save_pdf(
        self,
        output_path: str,
        anndata_object: anndata.AnnData,
        title: str = "Analysis Report",
    ) -> None:
        """Write the report to a multi-page PDF file.

        Page 1 contains a structured metadata summary of the AnnData object and
        its pipeline configuration (vector text, perfectly sharp at any zoom).
        Every subsequent page contains one captured figure at 300 DPI.

        Args:
            output_path: Destination path for the PDF file.  In Google Colab,
                use a path inside your mounted Drive, e.g.
                ``"/content/drive/MyDrive/report.pdf"``.
            anndata_object: The AnnData that was analysed.  Its ``.obs``,
                ``.var``, and ``.uns`` are read to produce the metadata summary.
            title: Report title shown at the top of the metadata page.
        """
        # Count captured figures, respecting which storage was used.
        if self._capture_console and self._ordered_items:
            n_captured = sum(1 for kind, _ in self._ordered_items if kind == "figure")
        else:
            n_captured = len(self._figure_buffers)
        total_figures = (
            n_captured
            + len(self._plotly_buffers)
            + len(self._plotly_placeholders)
        )
        print(f"Saving report: '{output_path}'  ({total_figures} figure(s)) ...")

        pdf_writer = PdfWriter()

        # --- Page 1: metadata summary (vector text, rendered once) ---
        metadata_fig = _build_metadata_figure(anndata_object, title)
        metadata_pdf_buffer = io.BytesIO()
        metadata_fig.savefig(metadata_pdf_buffer, format="pdf", bbox_inches="tight")
        plt.close(metadata_fig)
        del metadata_fig
        metadata_pdf_buffer.seek(0)
        pdf_writer.append(PdfReader(metadata_pdf_buffer))
        del metadata_pdf_buffer
        gc.collect()

        # Flush any console text printed after the last plt.show() (e.g. QC blocks
        # printed after the final figure) so it appears in the PDF when save_pdf
        # is called inside the `with` block before __exit__ runs.
        if self._capture_console and self._original_stdout is not None:
            remaining_text = self._console_buffer.getvalue()
            if remaining_text.strip():
                self._ordered_items.append(("text", remaining_text))
            # Reset the buffer so text printed by save_pdf itself is not double-captured.
            self._console_buffer = io.StringIO()
            sys.stdout = _TeeStream(self._original_stdout, self._console_buffer)

        # --- Pages 2+: figures (and optional interleaved text when capture_console=True) ---
        if self._capture_console and self._ordered_items:
            # Interleave console text pages with figure pages in the order produced.
            for item_type, item_content in self._ordered_items:
                if item_type == "text":
                    for text_figure in _build_console_text_figures(item_content):
                        text_pdf_buffer = io.BytesIO()
                        text_figure.savefig(
                            text_pdf_buffer, format="pdf", bbox_inches="tight"
                        )
                        plt.close(text_figure)
                        del text_figure
                        text_pdf_buffer.seek(0)
                        pdf_writer.append(PdfReader(text_pdf_buffer))
                        del text_pdf_buffer
                        gc.collect()
                elif item_type == "figure":
                    _embed_png_buffer_as_pdf_page(item_content, pdf_writer)
            self._ordered_items.clear()
        else:
            for buffer in self._figure_buffers:
                _embed_png_buffer_as_pdf_page(buffer, pdf_writer)
            self._figure_buffers.clear()

        # --- Pages: Plotly figures ---
        for buffer in self._plotly_buffers:
            _embed_png_buffer_as_pdf_page(buffer, pdf_writer)
        self._plotly_buffers.clear()

        # --- Pages: text placeholders for non-renderable Plotly figures ---
        for placeholder_text in self._plotly_placeholders:
            placeholder_fig = _build_placeholder_figure(placeholder_text)
            placeholder_pdf_buffer = io.BytesIO()
            placeholder_fig.savefig(
                placeholder_pdf_buffer, format="pdf", bbox_inches="tight"
            )
            plt.close(placeholder_fig)
            del placeholder_fig
            placeholder_pdf_buffer.seek(0)
            pdf_writer.append(PdfReader(placeholder_pdf_buffer))
            del placeholder_pdf_buffer

        with open(output_path, "wb") as output_file:
            pdf_writer.write(output_file)

        print(
            f"Report saved: {output_path}  "
            f"({total_figures} figure(s) + 1 metadata page)"
        )

    def save_console_pdf(self, output_path: str) -> None:
        """Write an interleaved console text + figure PDF.

        Only meaningful when the ReportBuilder was created with
        ``capture_console=True``.  Each block of text printed between two
        ``plt.show()`` calls becomes one (or more) monospace text pages; each
        figure becomes one 300 DPI image page.  Pages appear in the order they
        were produced — text first, then the figure it precedes.

        Args:
            output_path: Destination path for the PDF file.
        """
        # Flush any console text printed after the last plt.show() before counting.
        if self._capture_console and self._original_stdout is not None:
            remaining_text = self._console_buffer.getvalue()
            if remaining_text.strip():
                self._ordered_items.append(("text", remaining_text))
            self._console_buffer = io.StringIO()
            sys.stdout = _TeeStream(self._original_stdout, self._console_buffer)

        if not self._ordered_items:
            print(
                "Nothing to save — did you use ReportBuilder(capture_console=True) "
                "and run code inside the 'with' block?"
            )
            return

        n_figures = sum(1 for kind, _ in self._ordered_items if kind == "figure")
        n_text_sections = sum(1 for kind, _ in self._ordered_items if kind == "text")
        print(
            f"Saving console report: '{output_path}'  "
            f"({n_figures} figure(s), {n_text_sections} text section(s)) ..."
        )

        pdf_writer = PdfWriter()

        for item_type, item_content in self._ordered_items:
            if item_type == "text":
                # Render text as one or more monospace pages (paginates if long)
                for text_figure in _build_console_text_figures(item_content):
                    text_pdf_buffer = io.BytesIO()
                    text_figure.savefig(
                        text_pdf_buffer, format="pdf", bbox_inches="tight"
                    )
                    plt.close(text_figure)
                    del text_figure
                    text_pdf_buffer.seek(0)
                    pdf_writer.append(PdfReader(text_pdf_buffer))
                    del text_pdf_buffer
                    gc.collect()
            elif item_type == "figure":
                _embed_png_buffer_as_pdf_page(item_content, pdf_writer)

        self._ordered_items.clear()

        with open(output_path, "wb") as output_file:
            pdf_writer.write(output_file)

        print(f"Console report saved: {output_path}")


# ---------------------------------------------------------------------------
# Private helper — console text page renderer
# ---------------------------------------------------------------------------


def _build_console_text_figures(text: str) -> List[matplotlib.figure.Figure]:
    """Render a block of console text as one or more portrait-letter figures.

    Lines longer than 120 characters are wrapped.  The output paginates at
    80 lines per page so nothing is clipped.

    Args:
        text: Raw string captured from stdout (may contain newlines).

    Returns:
        A list of matplotlib Figure objects, one per page.
    """
    wrapped_lines: List[str] = []
    for raw_line in text.splitlines():
        if len(raw_line) > 120:
            # textwrap.wrap drops empty lines, so handle them explicitly
            wrapped = textwrap.wrap(raw_line, width=120)
            wrapped_lines.extend(wrapped if wrapped else [""])
        else:
            wrapped_lines.append(raw_line)

    lines_per_page = 80
    figures: List[matplotlib.figure.Figure] = []
    # Ensure at least one page even for empty text
    page_count = max(1, -(-len(wrapped_lines) // lines_per_page))  # ceiling division
    for page_index in range(page_count):
        start = page_index * lines_per_page
        page_lines = wrapped_lines[start : start + lines_per_page]
        page_text = "\n".join(page_lines)
        figure = plt.figure(figsize=(8.5, 11))
        figure.patch.set_facecolor("white")
        figure.text(
            0.04,
            0.97,
            page_text,
            transform=figure.transFigure,
            verticalalignment="top",
            fontfamily="monospace",
            fontsize=7.5,
            color="#1a1a1a",
        )
        figures.append(figure)

    return figures


# ---------------------------------------------------------------------------
# Private helper — memory-efficient PNG-to-PDF page writer
# ---------------------------------------------------------------------------


def _embed_png_buffer_as_pdf_page(buffer: io.BytesIO, pdf_writer: PdfWriter) -> None:
    """Embed one PNG buffer as a PDF page using PIL's built-in PDF writer.

    This replaces the previous matplotlib imshow + pdf.savefig approach, which
    forced matplotlib's ``_make_image()`` to create a float32 RGBA intermediate
    buffer (135 MB for a 300 DPI A4 page) on every figure — causing the RAM to
    spike to 20+ GB for a 51-figure report.

    The new pipeline:
      PNG bytes (3 MB) → PIL uint8 RGB (25 MB) → PIL PDF writer (libjpeg,
      no float32) → single-page PDF in memory (~4 MB) → pypdf appends page.

    Peak RAM per figure: ~35 MB instead of 135 MB.  With gc.collect() freeing
    each buffer before the next iteration, total overhead stays bounded.

    The ``resolution`` parameter tells PIL to embed the image at 300 DPI so
    the PDF page dimensions match the original figure size (e.g. 11×8.5 in).

    Args:
        buffer:     An in-memory BytesIO containing a 300 DPI PNG image.
        pdf_writer: The open PdfWriter to append the new page into.
    """
    buffer.seek(0)
    # Suppress PIL's decompression-bomb guard: our buffers come from matplotlib,
    # not untrusted user input, so large high-DPI figures are safe to open.
    Image.MAX_IMAGE_PIXELS = None
    pil_image = Image.open(buffer).convert("RGB")

    # Encode directly to a single-page PDF using PIL's built-in PDF plugin.
    # PIL uses libjpeg (uint8 path) to compress the image — no float32 arrays.
    # resolution=300 embeds the image at 300 DPI so the page size is correct.
    pdf_page_buffer = io.BytesIO()
    pil_image.save(pdf_page_buffer, format="PDF", resolution=300)
    pil_image.close()
    buffer.close()
    del pil_image

    # Append the single PDF page to the writer (pure byte-copy, no rendering).
    pdf_page_buffer.seek(0)
    pdf_writer.append(PdfReader(pdf_page_buffer))
    del pdf_page_buffer

    # Return mmap-backed memory to the OS before the next iteration.
    gc.collect()


# ---------------------------------------------------------------------------
# Private helpers — human-readable label conversion
# ---------------------------------------------------------------------------


def _human_readable_layer_name(active_layer: Optional[str]) -> str:
    """Convert an internal layer name to a human-readable description.

    Args:
        active_layer: Layer name from ``uns['analysis_config']['active_layer']``,
            e.g. ``"arcsinh__zscore_col"``, or ``None`` when raw ``.X`` is used.

    Returns:
        A human-readable string, e.g.
        ``"Arcsinh  →  Z-score per column (StandardScaler)"``.

    Examples:
        >>> _human_readable_layer_name(None)
        'None — raw .X values used directly'
        >>> _human_readable_layer_name("arcsinh__zscore_col")
        'Arcsinh  →  Z-score per column (StandardScaler)'
        >>> _human_readable_layer_name("none__minmax_col")
        'Min-max scaling per column'
        >>> _human_readable_layer_name("arcsinh__zscore_col__l2norm_row")
        'Arcsinh  →  Z-score per column + L2 normalization per row'
    """
    if active_layer is None:
        return "None — raw .X values used directly"

    # Layer names follow {transform}__{norm} with double underscore separator.
    # The norm part may itself contain __ (e.g. zscore_col__l2norm_row), so
    # split on the first __ only to separate transform from norm.
    parts = active_layer.split("__", maxsplit=1)
    transform_token = parts[0]
    norm_token = parts[1] if len(parts) > 1 else "none"

    transform_label = _TRANSFORM_LABELS.get(transform_token, transform_token)
    norm_label = _NORM_LABELS.get(norm_token, norm_token)

    # Combine non-empty parts with an arrow separator
    description_parts = [p for p in [transform_label, norm_label] if p]
    if not description_parts:
        return "None — raw .X values used directly"
    return "  →  ".join(description_parts)


def _human_readable_selection_method(method: Optional[str]) -> str:
    """Convert a feature selection method code to a human-readable description.

    Args:
        method: Method name from ``uns['analysis_config']['active_selection']``,
            e.g. ``"HighVariance"``, or ``None`` when all channels are used.

    Returns:
        A human-readable string, e.g.
        ``"Top N channels by variance (HighVariance)"``.
    """
    if method is None:
        return "None — all channels included"
    return _SELECTION_METHOD_LABELS.get(method, method)


# ---------------------------------------------------------------------------
# Private helpers — metadata page
# ---------------------------------------------------------------------------


def _build_metadata_figure(
    anndata_object: anndata.AnnData,
    title: str,
) -> matplotlib.figure.Figure:
    """Create a matplotlib Figure containing the dataset and pipeline metadata.

    Args:
        anndata_object: The AnnData to summarise.
        title: Report title string.

    Returns:
        A matplotlib Figure with all metadata rendered as vector monospace text.
    """
    lines = _collect_metadata_lines(anndata_object, title)
    text_content = "\n".join(lines)

    figure = plt.figure(figsize=(11, 8.5))
    figure.patch.set_facecolor("white")
    figure.text(
        0.04,
        0.97,
        text_content,
        transform=figure.transFigure,
        verticalalignment="top",
        fontfamily="monospace",
        fontsize=7.5,
        color="#1a1a1a",
    )
    return figure


def _collect_metadata_lines(
    anndata_object: anndata.AnnData,
    title: str,
) -> List[str]:
    """Assemble lines of metadata text from an AnnData object.

    Args:
        anndata_object: The AnnData to inspect.
        title: Report title string.

    Returns:
        A list of text lines to join with newlines.
    """
    lines: List[str] = []
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d  %H:%M")

    # ---- Header ----
    lines.append("=" * 90)
    lines.append(f"  {title}")
    lines.append(f"  Generated: {timestamp}")
    lines.append("=" * 90)
    lines.append("")

    # ---- Dataset dimensions ----
    lines.append("DATASET")
    lines.append("-" * 50)
    lines.append(f"  Observations (events / cells): {anndata_object.n_obs:>10,}")
    lines.append(f"  Variables (channels / features): {anndata_object.n_vars:>8}")
    # anndata >= 0.13 exposes .X as layers[None]; that key is not an error.
    raw_layer_keys = [key for key in anndata_object.layers.keys() if key is not None]
    # Layer keys are expected to be non-empty strings (see normalization.py).
    # A non-string/None key means something upstream did `anndata_object.layers[x] = ...`
    # with an unset variable — surface it instead of crashing the report.
    invalid_layer_keys = [key for key in raw_layer_keys if not isinstance(key, str) or not key]
    if invalid_layer_keys:
        print(
            f"[report] WARNING: {len(invalid_layer_keys)} invalid (non-string/empty) "
            f".layers key(s) found: {invalid_layer_keys!r} — check upstream code that "
            "assigns to .layers[...]."
        )
    layer_names = [key for key in raw_layer_keys if isinstance(key, str) and key]
    layer_display = ", ".join(layer_names) if layer_names else "none (raw .X only)"
    lines.append(f"  Available layers:  {layer_display}")
    lines.append("")

    # ---- Breakdown of key .obs categorical columns ----
    # subject_ID is omitted (too many unique values, not informative in aggregate).
    # unique_subject_ID is shown in full — no top-20 cap.
    obs_columns_to_show = ["specie", "age_group", "unique_subject_ID"]
    existing_obs_columns = [
        col for col in obs_columns_to_show if col in anndata_object.obs.columns
    ]

    if existing_obs_columns:
        lines.append("OBSERVATIONS (.obs  — value counts)")
        lines.append("-" * 50)
        for column_name in existing_obs_columns:
            value_counts = anndata_object.obs[column_name].value_counts()
            lines.append(f"  {column_name}:")
            for value, count in value_counts.items():
                percentage = 100.0 * count / anndata_object.n_obs
                lines.append(
                    f"    {str(value):<32s}  {count:>7,} events  ({percentage:5.1f}%)"
                )
        lines.append("")

    # ---- Pipeline configuration ----
    analysis_config = anndata_object.uns.get("analysis_config", {})
    lines.append("PIPELINE CONFIGURATION")
    lines.append("-" * 50)

    # Normalization layer — human-readable
    active_layer = analysis_config.get("active_layer", None)
    lines.append(
        f"  Transformation / normalization:  {_human_readable_layer_name(active_layer)}"
    )

    # Feature selection method — human-readable
    active_selection = analysis_config.get("active_selection", None)
    lines.append(
        f"  Feature selection:               {_human_readable_selection_method(active_selection)}"
    )

    # Subset filters applied during select_sfc_subset
    selection_filters = analysis_config.get("selection", {})
    if selection_filters:
        lines.append("  Subset filters:")
        for filter_key, filter_value in selection_filters.items():
            lines.append(f"    {filter_key:<20s}  {filter_value}")

    lines.append("")

    # ---- Selected channels ----
    active_selection_method = analysis_config.get("active_selection")
    if active_selection_method:
        selection_flag_column = f"is_selected_{active_selection_method}"
        if selection_flag_column in anndata_object.var.columns:
            selected_var = anndata_object.var[anndata_object.var[selection_flag_column]]
            score_column = f"{active_selection_method}_score"
            if score_column in selected_var.columns:
                # Sort descending so rank #1 = highest score.
                selected_var = selected_var.sort_values(score_column, ascending=False)
            selected_channels = selected_var.index.tolist()
            lines.append(
                f"SELECTED CHANNELS  ({len(selected_channels)} channels"
                f" — {_human_readable_selection_method(active_selection_method)})"
            )
            lines.append("-" * 50)
            for rank_start in range(0, len(selected_channels), 3):
                chunk = selected_channels[rank_start : rank_start + 3]
                row_parts = [
                    f"#{rank_start + i + 1:<2d} {ch}"
                    for i, ch in enumerate(chunk)
                ]
                lines.append("  " + "     ".join(row_parts))
            lines.append("")

    # ---- PCA summary ----
    pca_variance_ratio = anndata_object.uns.get("pca_explained_variance_ratio")
    if pca_variance_ratio is not None:
        lines.append("PCA  (explained variance per component)")
        lines.append("-" * 50)
        cumulative_variance = 0.0
        for pc_index, variance in enumerate(pca_variance_ratio[:10]):
            cumulative_variance += float(variance)
            lines.append(
                f"  PC{pc_index + 1:02d}:  {100.0 * variance:5.2f}%"
                f"   cumulative: {100.0 * cumulative_variance:6.2f}%"
            )
        lines.append("")

    # ---- UMAP parameters ----
    umap_params = analysis_config.get("umap_params", {})
    if umap_params:
        lines.append("UMAP PARAMETERS")
        lines.append("-" * 50)
        lines.append(f"  n_neighbors:    {umap_params.get('n_neighbors', 'N/A')}")
        lines.append(f"  min_dist:       {umap_params.get('min_dist', 'N/A')}")
        lines.append(f"  random_state:   {umap_params.get('random_state', 'N/A')}")
        n_events = umap_params.get("n_events_sampled", "N/A")
        n_events_display = f"{n_events:,}" if isinstance(n_events, int) else str(n_events)
        lines.append(f"  events sampled: {n_events_display}")
        lines.append("")

    return lines


def _build_placeholder_figure(placeholder_text: str) -> matplotlib.figure.Figure:
    """Create a simple figure with a text note for non-renderable figures.

    Args:
        placeholder_text: The message to display on the page.

    Returns:
        A matplotlib Figure with the placeholder text centred.
    """
    figure = plt.figure(figsize=(11, 8.5))
    figure.patch.set_facecolor("#f8f8f8")
    figure.text(
        0.5,
        0.5,
        placeholder_text,
        transform=figure.transFigure,
        horizontalalignment="center",
        verticalalignment="center",
        fontfamily="monospace",
        fontsize=10,
        color="#555555",
    )
    return figure
