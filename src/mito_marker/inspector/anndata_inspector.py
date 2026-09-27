"""
anndata_inspector.py

Interactive explorer for AnnData objects, designed for non-code experts.

In a Jupyter or Google Colab notebook, calling inspect_anndata() displays a
tabbed widget that mirrors the AnnData structure diagram:

    Tab 0 — Overview      : structural summary (all slots, shapes, memory)
    Tab 1 — .X matrix     : first N rows × M columns of the measurement matrix
    Tab 2 — .obs          : observation metadata table + column types
    Tab 3 — .var          : variable metadata table + column types
    Tab 4 — .obsm / .uns  : embedding shapes and unstructured metadata keys

In a plain Python console (e.g. pytest, terminal script), the same function
falls back to a formatted text summary that mirrors the QC style used
throughout this package.

ipywidgets is an optional dependency: if it is not installed, the console
fallback is used automatically.
"""

from typing import Dict, Union

import anndata
import numpy as np
import pandas as pd
import scipy.sparse

# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def inspect_anndata(
    anndata_object: anndata.AnnData,
    max_display_rows: int = 15,
    max_display_columns: int = 12,
    group_by_column: "str | None" = None,
) -> None:
    """
    Explore an AnnData object interactively.

    In a Jupyter or Google Colab notebook: displays a five-tab widget with
    sections for Overview, .X matrix, .obs metadata, .var metadata, and
    .obsm / .uns keys.  Click any tab to navigate.

    In a plain Python console: prints a formatted text summary.

    Arguments:
        anndata_object:      The AnnData object to inspect.
        max_display_rows:    Maximum number of rows shown in table previews
                             (default 15).
        max_display_columns: Maximum number of columns shown in the .X preview
                             (default 12).
        group_by_column:     Name of the .obs column used to build the subject
                             summary table in the Overview tab (e.g. 'subject_ID').
                             If None, the column is auto-detected by name.

    Returns:
        None
    """
    if _is_notebook_environment() and _ipywidgets_available():
        _display_notebook_widget(
            anndata_object, max_display_rows, max_display_columns, group_by_column
        )
    else:
        _print_console_summary(
            anndata_object, max_display_rows, max_display_columns, group_by_column
        )


# ---------------------------------------------------------------------------
# Environment detection helpers
# ---------------------------------------------------------------------------


def _is_notebook_environment() -> bool:
    """
    Return True if the code is running inside a Jupyter notebook or Google Colab.

    Relies on get_ipython(), which is injected into the global namespace by
    IPython.  Returns False in any plain Python interpreter or pytest context.
    """
    try:
        ipython_shell = get_ipython()  # type: ignore[name-defined]
        return ipython_shell is not None
    except NameError:
        return False


def _ipywidgets_available() -> bool:
    """
    Return True if ipywidgets is importable.

    ipywidgets is listed as an optional dependency.  When absent, the console
    fallback is used so the package still works in minimal environments.
    """
    try:
        import ipywidgets  # noqa: F401

        return True
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# Shared data-extraction helpers
# ---------------------------------------------------------------------------


def _format_bytes(byte_count: int) -> str:
    """
    Convert a byte count to a human-readable string (e.g. '4.2 MB').

    Arguments:
        byte_count: Size in bytes.

    Returns:
        Human-readable size string with one decimal place.
    """
    # Use float so fractional sizes (e.g. 1.5 MB) are preserved through division.
    size = float(byte_count)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024:
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"


def _x_matrix_info(anndata_object: anndata.AnnData) -> Dict[str, Union[str, bool, float]]:
    """
    Extract display metadata about the .X matrix.

    Computes dtype, storage format (sparse vs dense), fill density for sparse
    matrices, and an approximate memory size.

    Arguments:
        anndata_object: The AnnData object whose .X to inspect.

    Returns:
        Dict with keys:
            'dtype'           — string representation of the numpy dtype
            'is_sparse'       — True if .X is a scipy sparse matrix
            'density_percent' — percentage of non-zero elements (100.0 if dense)
            'size_human'      — human-readable memory size string
    """
    # For backed AnnData (stored on disk as .h5ad), accessing .X.nnz on a
    # backed sparse matrix reads the entire data array from disk — that would
    # defeat the purpose of backing.  Return lightweight placeholder metadata
    # instead; the shape and dtype are available as HDF5 dataset attributes
    # and do not require loading any data.
    if anndata_object.isbacked:
        x_matrix = anndata_object.X
        dtype = str(x_matrix.dtype) if hasattr(x_matrix, "dtype") else "unknown"
        shape = anndata_object.X.shape
        # Estimate size from shape and dtype without reading data.
        try:
            itemsize = np.dtype(dtype).itemsize
            estimated_bytes = shape[0] * shape[1] * itemsize
            size_human = _format_bytes(estimated_bytes) + " (est., backed)"
        except Exception:
            size_human = "on disk (backed)"
        return {
            "dtype": dtype,
            "is_sparse": False,
            "density_percent": 100.0,
            "size_human": size_human,
        }

    x_matrix = anndata_object.X
    is_sparse = scipy.sparse.issparse(x_matrix)
    dtype = str(x_matrix.dtype)

    if is_sparse:
        total_elements = x_matrix.shape[0] * x_matrix.shape[1]
        density_percent = (
            100.0 * x_matrix.nnz / total_elements if total_elements > 0 else 0.0
        )
        # Memory for CSR/CSC: data array + column indices + row/col pointers
        size_bytes = x_matrix.data.nbytes + x_matrix.indices.nbytes + x_matrix.indptr.nbytes
    else:
        density_percent = 100.0
        size_bytes = x_matrix.nbytes

    return {
        "dtype": dtype,
        "is_sparse": is_sparse,
        "density_percent": density_percent,
        "size_human": _format_bytes(size_bytes),
    }


def _x_matrix_slice_as_dataframe(
    anndata_object: anndata.AnnData,
    max_display_rows: int,
    max_display_columns: int,
) -> pd.DataFrame:
    """
    Return a dense pandas DataFrame slice of .X for display purposes.

    Converts a sparse slice to dense automatically so it can be rendered
    as an HTML table or printed as a string.

    Arguments:
        anndata_object:      Source AnnData object.
        max_display_rows:    Row limit for the preview.
        max_display_columns: Column limit for the preview.

    Returns:
        Dense DataFrame with obs_names as index and var_names as columns.
    """
    row_count = min(max_display_rows, anndata_object.n_obs)
    col_count = min(max_display_columns, anndata_object.n_vars)

    x_slice = anndata_object.X[:row_count, :col_count]
    if scipy.sparse.issparse(x_slice):
        x_slice = x_slice.toarray()

    return pd.DataFrame(
        x_slice,
        index=anndata_object.obs_names[:row_count],
        columns=anndata_object.var_names[:col_count],
    )


def _x_head_tail_dataframes(
    anndata_object: anndata.AnnData,
    n_rows: int,
) -> tuple:
    """
    Return the first and last n_rows of .X as dense DataFrames, plus the count
    of rows in between that are not shown.

    If the total number of observations fits within 2 × n_rows, all rows are
    returned in head_dataframe and tail_dataframe is None.

    Arguments:
        anndata_object: Source AnnData object.
        n_rows:         Number of rows to take from each end.

    Returns:
        Tuple (head_dataframe, tail_dataframe, n_hidden) where tail_dataframe
        is None when the full matrix fits within the display budget.
    """
    n_obs = anndata_object.n_obs

    def _to_dense_dataframe(x_slice: object, obs_index: object) -> pd.DataFrame:
        if scipy.sparse.issparse(x_slice):
            x_slice = x_slice.toarray()
        return pd.DataFrame(
            x_slice,
            index=obs_index,
            columns=anndata_object.var_names,
        )

    if n_obs <= 2 * n_rows:
        head_dataframe = _to_dense_dataframe(
            anndata_object.X, anndata_object.obs_names
        )
        return head_dataframe, None, 0

    head_dataframe = _to_dense_dataframe(
        anndata_object.X[:n_rows, :], anndata_object.obs_names[:n_rows]
    )
    tail_dataframe = _to_dense_dataframe(
        anndata_object.X[-n_rows:, :], anndata_object.obs_names[-n_rows:]
    )
    n_hidden = n_obs - 2 * n_rows
    return head_dataframe, tail_dataframe, n_hidden


def _head_tail_table_html(
    head_dataframe: pd.DataFrame,
    tail_dataframe: "pd.DataFrame | None",
    n_hidden: int,
) -> str:
    """
    Build an HTML table string that shows head rows, an optional separator,
    and tail rows.

    When tail_dataframe is None (dataset is small enough), the full
    head_dataframe is rendered without a separator.

    The separator row spans all columns and shows how many rows are hidden,
    making it clear to the user that only the extremes are displayed.

    Arguments:
        head_dataframe: First n rows to display.
        tail_dataframe: Last n rows to display, or None.
        n_hidden:       Number of rows between head and tail that are omitted.

    Returns:
        HTML string ready to embed in an f-string.
    """
    if tail_dataframe is None:
        return head_dataframe.round(4).to_html(border=0, classes="")

    # Number of columns including the index column that pandas renders.
    n_table_cols = len(head_dataframe.columns) + 1

    separator_row = (
        f'<tr><td colspan="{n_table_cols}" style="'
        f'text-align:center; color:#888; background:#f0f0f0; '
        f'font-style:italic; font-family:sans-serif; padding:6px;">'
        f'&#8942;&nbsp;&nbsp;{n_hidden:,} rows not shown&nbsp;&nbsp;&#8942;'
        f"</td></tr>"
    )

    # Parse pandas HTML output: extract everything up to the closing </tbody>
    # tag from the head table, then splice in the separator and the tail rows.
    # Pandas always emits well-structured HTML so this split is reliable.
    head_html = head_dataframe.round(4).to_html(border=0, classes="")
    tail_html = tail_dataframe.round(4).to_html(border=0, classes="")

    head_open = head_html[: head_html.rfind("</tbody>")]
    tail_body_start = tail_html.find("<tbody>") + len("<tbody>")
    tail_body_end = tail_html.find("</tbody>")
    tail_rows = tail_html[tail_body_start:tail_body_end]

    return f"{head_open}{separator_row}{tail_rows}</tbody></table>"


# ---------------------------------------------------------------------------
# Subject / group summary helpers
# ---------------------------------------------------------------------------


def _detect_subject_column(obs_dataframe: pd.DataFrame) -> "str | None":
    """
    Auto-detect the column in .obs that identifies individual subjects.

    Checks for common naming conventions in order of specificity.
    Returns None if no recognisable subject column is found.

    Arguments:
        obs_dataframe: The .obs DataFrame of an AnnData object.

    Returns:
        Column name string, or None.
    """
    candidate_names = [
        "subject_ID", "subject_id", "SubjectID",
        "patient_ID", "patient_id", "PatientID",
        "sample_ID", "sample_id", "SampleID",
        "subject", "patient", "sample",
    ]
    for name in candidate_names:
        if name in obs_dataframe.columns:
            return name
    return None


def _build_subject_summary_dataframe(
    obs_dataframe: pd.DataFrame,
    subject_column: str,
) -> pd.DataFrame:
    """
    Build a summary DataFrame with one row per unique subject.

    For each subject, every other metadata column shows the set of distinct
    values found across all observations for that subject:
      - If all observations share the same value  → that value is shown directly.
      - If multiple distinct values exist         → they are joined with  " / ".

    Arguments:
        obs_dataframe:  The .obs DataFrame to summarise.
        subject_column: Name of the column to use as the grouping key (rows).

    Returns:
        DataFrame indexed by subject_column with aggregated metadata columns.
    """

    def _join_unique(series: pd.Series) -> str:
        unique_values = series.dropna().astype(str).unique()
        if len(unique_values) == 0:
            return "—"
        return " / ".join(sorted(unique_values))

    # Cytometry .obs can have millions of identical per-event rows (same subject,
    # same condition, same timepoint for every cell in a tube).  Deduplicating
    # first collapses those millions of rows to a few dozen unique metadata
    # combinations before the groupby, making the aggregation near-instant.
    deduplicated_obs = obs_dataframe.drop_duplicates()
    return deduplicated_obs.groupby(subject_column, sort=True).agg(_join_unique)


# ---------------------------------------------------------------------------
# Notebook mode — ipywidgets Tab widget
# ---------------------------------------------------------------------------

# Inline CSS injected once per tab.  Colors match the AnnData schema SVG:
#   yellow (#efc41c)  = obs
#   blue   (#2c96c0)  = var / obsm
#   dark   (#194c61)  = headers
#   green  (#3c8b53)  = layers
#   purple (#965ba5)  = uns
_INSPECTOR_CSS = """
<style>
  .mito-inspector table { border-collapse: collapse; font-size: 12px;
                           font-family: monospace; }
  .mito-inspector th { background: #194c61; color: white;
                        padding: 5px 10px; text-align: left;
                        white-space: nowrap; }
  .mito-inspector td { padding: 3px 10px; border-bottom: 1px solid #e8e8e8;
                        white-space: nowrap; background: #ffffff !important;
                        color: #222 !important; }
  .mito-inspector tr:nth-child(even) td { background: #f7f7f7 !important;
                                           color: #222 !important; }
  .mito-inspector h4 { color: #194c61; margin-top: 14px; margin-bottom: 6px;
                        font-family: sans-serif; }
  .mito-inspector p  { font-family: sans-serif; font-size: 13px; color: #333; }
  .mito-inspector em { font-family: sans-serif; font-size: 12px; color: #888; }
  .mito-scroll      { overflow-x: auto; }
  .mito-subject td  { white-space: normal; max-width: 200px; word-break: break-word; }
  .mito-badge       { display: inline-block; padding: 2px 9px;
                       border-radius: 10px; font-size: 11px; font-weight: bold;
                       margin: 2px 3px; font-family: monospace; }
  .mito-badge-yellow  { background: #efc41c; color: #222; }
  .mito-badge-blue    { background: #2c96c0; color: white; }
  .mito-badge-green   { background: #3c8b53; color: white; }
  .mito-badge-purple  { background: #965ba5; color: white; }
  .mito-badge-grey    { background: #aaa;    color: white; }
</style>
"""


def _render_badges(keys: list, badge_class: str) -> str:
    """Return an HTML string of coloured badge pills for a list of key names."""
    if not keys:
        return "<em>none</em>"
    return " ".join(
        f'<span class="mito-badge {badge_class}">{key}</span>' for key in keys
    )


def _display_notebook_widget(
    anndata_object: anndata.AnnData,
    max_display_rows: int,
    max_display_columns: int,
    group_by_column: "str | None" = None,
) -> None:
    """
    Build and display a five-tab ipywidgets explorer for the AnnData object.

    The widget is displayed immediately with only the Overview tab rendered.
    All other tabs use lazy rendering: their content is built on first click
    so the initial display is instant regardless of dataset size.

    Arguments:
        anndata_object:      The AnnData object to explore.
        max_display_rows:    Row limit for table previews.
        max_display_columns: Column limit for .X preview.
        group_by_column:     .obs column used for the subject summary table.

    Returns:
        None
    """
    import ipywidgets as widgets
    from IPython.display import HTML, display

    tab_definitions = [
        (
            "Overview",
            lambda: _render_overview_tab(anndata_object, group_by_column),
        ),
        (
            ".X matrix",
            lambda: _render_x_tab(anndata_object, max_display_rows, max_display_columns),
        ),
        (
            ".obs",
            lambda: _render_obs_tab(anndata_object, max_display_rows),
        ),
        (
            ".var",
            lambda: _render_var_tab(anndata_object, max_display_rows),
        ),
        (
            ".obsm / .uns",
            lambda: _render_reductions_tab(anndata_object),
        ),
    ]

    # One Output widget per tab.
    output_widgets = [widgets.Output() for _ in tab_definitions]

    # Track which tabs have already been rendered so each is built only once.
    tabs_rendered = [False] * len(tab_definitions)

    def _render_tab(index: int) -> None:
        """Render a single tab by index, skipping it if already rendered."""
        if tabs_rendered[index]:
            return
        tabs_rendered[index] = True
        _, render_fn = tab_definitions[index]
        with output_widgets[index]:
            try:
                render_fn()
            except Exception as render_error:
                # Surface errors instead of leaving a blank tab.
                display(HTML(
                    f"{_INSPECTOR_CSS}"
                    f"<div class='mito-inspector'>"
                    f"<p style='color:red;'><b>Error rendering this tab:</b> "
                    f"{type(render_error).__name__}: {render_error}</p></div>"
                ))

    # Pre-render only the Overview tab (Tab 0) — it reads only metadata and
    # is always visible.  All other tabs are rendered lazily on first click.
    _render_tab(0)

    def _on_tab_selected(change: dict) -> None:
        """Observer callback: render the newly selected tab on first visit."""
        _render_tab(change["new"])

    tab_widget = widgets.Tab(children=output_widgets)
    for index, (title, _) in enumerate(tab_definitions):
        tab_widget.set_title(index, title)

    # Register the observer BEFORE display so it is active the moment the
    # widget appears.  The callback fires when the user clicks a tab, and
    # calling `with output_widget: display(...)` inside an observe callback
    # is reliable in both Jupyter and Colab — the widget comm protocol
    # guarantees the update is delivered even after the widget is displayed.
    tab_widget.observe(_on_tab_selected, names="selected_index")

    display(tab_widget)


def _render_overview_tab(
    anndata_object: anndata.AnnData,
    group_by_column: "str | None" = None,
) -> None:
    """Render the Overview tab: structural summary + subject/group summary table."""
    from IPython.display import HTML, display

    x_info = _x_matrix_info(anndata_object)
    matrix_type = "sparse" if x_info["is_sparse"] else "dense"
    density_note = (
        f", {x_info['density_percent']:.1f}% filled" if x_info["is_sparse"] else ""
    )

    # anndata >= 0.13 exposes .X as layers[None]; list named layers only.
    layers_keys = [key for key in anndata_object.layers.keys() if key is not None]
    obsm_keys = list(anndata_object.obsm.keys()) if anndata_object.obsm else []
    uns_keys = list(anndata_object.uns.keys()) if anndata_object.uns else []

    # Resolve the subject column: explicit argument beats auto-detection.
    effective_subject_column = group_by_column or _detect_subject_column(
        anndata_object.obs
    )

    subject_section_html = ""
    if effective_subject_column and effective_subject_column in anndata_object.obs.columns:
        summary_dataframe = _build_subject_summary_dataframe(
            anndata_object.obs, effective_subject_column
        )
        n_subjects = len(summary_dataframe)
        subject_section_html = f"""
        <h4>Subject / group summary
          &nbsp;<span style="font-weight:normal; font-size:12px; color:#888;">
            ({n_subjects} unique <em>{effective_subject_column}</em> values
            &nbsp;&mdash;&nbsp; one row per subject,
            multiple values within a subject joined with &ldquo; / &rdquo;)
          </span>
        </h4>
        <div class="mito-scroll">
          {summary_dataframe.to_html(border=0, classes="mito-subject")}
        </div>
        """

    html = f"""
    {_INSPECTOR_CSS}
    <div class="mito-inspector">
      <h4>AnnData — structural overview</h4>
      <table>
        <tr><th>Slot</th><th>Shape / content</th><th>Description</th></tr>
        <tr>
          <td><b>.obs</b></td>
          <td>{anndata_object.n_obs:,} rows &times; {anndata_object.obs.shape[1]} columns</td>
          <td>One row per observation (cell, event, mitochondrion)</td>
        </tr>
        <tr>
          <td><b>.var</b></td>
          <td>{anndata_object.n_vars:,} rows &times; {anndata_object.var.shape[1]} columns</td>
          <td>One row per feature (channel, morphological metric)</td>
        </tr>
        <tr>
          <td><b>.X</b></td>
          <td>{anndata_object.n_obs:,} &times; {anndata_object.n_vars:,}</td>
          <td>
            dtype <b>{x_info["dtype"]}</b> &nbsp;|&nbsp;
            {matrix_type}{density_note} &nbsp;|&nbsp;
            {x_info["size_human"]}
          </td>
        </tr>
        <tr>
          <td><b>.layers</b></td>
          <td colspan="2">{_render_badges(layers_keys, "mito-badge-green")}</td>
        </tr>
        <tr>
          <td><b>.obsm</b></td>
          <td colspan="2">{_render_badges(obsm_keys, "mito-badge-blue")}</td>
        </tr>
        <tr>
          <td><b>.uns</b></td>
          <td colspan="2">{_render_badges(uns_keys, "mito-badge-purple")}</td>
        </tr>
      </table>
      {subject_section_html}
    </div>
    """
    display(HTML(html))


def _render_x_tab(
    anndata_object: anndata.AnnData,
    max_display_rows: int,
    max_display_columns: int,
) -> None:
    """Render the .X matrix tab: metadata header + scrollable head/tail table.

    Shows the first and last max_display_rows rows with a separator in between.
    All columns are shown so the user can scroll right to inspect every channel.
    """
    from IPython.display import HTML, display

    x_info = _x_matrix_info(anndata_object)
    head_dataframe, tail_dataframe, n_hidden = _x_head_tail_dataframes(
        anndata_object, max_display_rows
    )

    row_note = ""
    if tail_dataframe is not None:
        row_note = (
            f"first and last {max_display_rows} rows shown"
            f" ({n_hidden:,} rows in between not shown)"
        )
    col_note = f"all {anndata_object.n_vars} columns — scroll right"

    display_note = f"<p><em>{row_note} &nbsp;|&nbsp; {col_note}</em></p>" if row_note else ""

    matrix_type = "sparse" if x_info["is_sparse"] else "dense"
    density_note = (
        f" ({x_info['density_percent']:.1f}% filled)" if x_info["is_sparse"] else ""
    )

    table_html = _head_tail_table_html(head_dataframe, tail_dataframe, n_hidden)

    html = f"""
    {_INSPECTOR_CSS}
    <div class="mito-inspector">
      <h4>.X &mdash; measurement matrix</h4>
      <p>
        Full shape: <b>{anndata_object.n_obs:,} &times; {anndata_object.n_vars}</b>
        &nbsp;|&nbsp; dtype: <b>{x_info["dtype"]}</b>
        &nbsp;|&nbsp; storage: <b>{matrix_type}{density_note}</b>
        &nbsp;|&nbsp; memory: <b>{x_info["size_human"]}</b>
      </p>
      {display_note}
      <div class="mito-scroll">
        {table_html}
      </div>
    </div>
    """
    display(HTML(html))


def _render_obs_tab(
    anndata_object: anndata.AnnData,
    max_display_rows: int,
) -> None:
    """Render the .obs tab: head/tail metadata table + column type summary."""
    from IPython.display import HTML, display

    obs_dataframe = anndata_object.obs
    n_total = obs_dataframe.shape[0]

    if n_total <= 2 * max_display_rows:
        head_df = obs_dataframe
        tail_df = None
        n_hidden = 0
    else:
        head_df = obs_dataframe.head(max_display_rows)
        tail_df = obs_dataframe.tail(max_display_rows)
        n_hidden = n_total - 2 * max_display_rows

    display_note = ""
    if tail_df is not None:
        display_note = (
            f"<p><em>First and last {max_display_rows} rows shown"
            f" ({n_hidden:,} rows in between not shown).</em></p>"
        )

    dtype_rows = "".join(
        f"<tr><td>{col}</td><td>{str(obs_dataframe[col].dtype)}</td>"
        f"<td>{obs_dataframe[col].nunique():,} unique values</td></tr>"
        for col in obs_dataframe.columns
    )
    dtype_table = f"""
    <h4>.obs column types</h4>
    <table>
      <tr><th>Column</th><th>dtype</th><th>Unique values</th></tr>
      {dtype_rows if dtype_rows else "<tr><td colspan='3'><em>No columns.</em></td></tr>"}
    </table>
    """

    html = f"""
    {_INSPECTOR_CSS}
    <div class="mito-inspector">
      <h4>.obs &mdash; observation metadata</h4>
      <p>Shape: <b>{n_total:,} rows &times; {obs_dataframe.shape[1]} columns</b></p>
      {display_note}
      <div class="mito-scroll">
        {_head_tail_table_html(head_df, tail_df, n_hidden)}
      </div>
      {dtype_table}
    </div>
    """
    display(HTML(html))


def _render_var_tab(
    anndata_object: anndata.AnnData,
    max_display_rows: int,
) -> None:
    """Render the .var tab: full variable list (no row truncation) + column types.

    All variables are shown in a vertically scrollable table so the user can
    browse the complete channel or feature list.
    """
    from IPython.display import HTML, display

    var_dataframe = anndata_object.var

    dtype_section = ""
    if var_dataframe.columns.size > 0:
        dtype_rows = "".join(
            f"<tr><td>{col}</td><td>{str(var_dataframe[col].dtype)}</td>"
            f"<td>{var_dataframe[col].nunique():,} unique values</td></tr>"
            for col in var_dataframe.columns
        )
        dtype_section = f"""
        <h4>.var column types</h4>
        <table>
          <tr><th>Column</th><th>dtype</th><th>Unique values</th></tr>
          {dtype_rows}
        </table>
        """

    var_table_html = (
        var_dataframe.to_html(border=0, classes="")
        if not var_dataframe.empty
        else "<em>No columns in .var.</em>"
    )

    html = f"""
    {_INSPECTOR_CSS}
    <div class="mito-inspector">
      <h4>.var &mdash; variable (feature) metadata</h4>
      <p>Shape: <b>{var_dataframe.shape[0]:,} variables</b></p>
      <div class="mito-scroll" style="max-height:400px; overflow-y:auto;">
        {var_table_html}
      </div>
      {dtype_section}
    </div>
    """
    display(HTML(html))


def _render_reductions_tab(anndata_object: anndata.AnnData) -> None:
    """Render the .obsm / .uns tab: embedding shapes and metadata key listing."""
    from IPython.display import HTML, display

    # .obsm — low-dimensional embeddings
    obsm_rows = ""
    if anndata_object.obsm:
        for key, array in anndata_object.obsm.items():
            shape_str = str(array.shape) if hasattr(array, "shape") else "unknown"
            dtype_str = str(array.dtype) if hasattr(array, "dtype") else "unknown"
            obsm_rows += (
                f"<tr><td><b>{key}</b></td><td>{shape_str}</td><td>{dtype_str}</td></tr>"
            )
    else:
        obsm_rows = "<tr><td colspan='3'><em>No embeddings stored yet.</em></td></tr>"

    # .uns — unstructured metadata (top-level keys only to avoid deep nesting)
    uns_rows = ""
    if anndata_object.uns:
        for key, value in anndata_object.uns.items():
            value_type = type(value).__name__
            if isinstance(value, dict):
                preview = f"dict with {len(value)} key(s): {list(value.keys())[:5]}"
            elif hasattr(value, "shape"):
                preview = f"array {value.shape}"
            else:
                # Truncate long string representations
                preview = str(value)[:100]
            uns_rows += (
                f"<tr><td><b>{key}</b></td><td>{value_type}</td><td>{preview}</td></tr>"
            )
    else:
        uns_rows = (
            "<tr><td colspan='3'><em>No unstructured metadata stored yet.</em></td></tr>"
        )

    html = f"""
    {_INSPECTOR_CSS}
    <div class="mito-inspector">
      <h4>.obsm &mdash; low-dimensional embeddings (PCA, UMAP, &hellip;)</h4>
      <table>
        <tr><th>Key</th><th>Shape</th><th>dtype</th></tr>
        {obsm_rows}
      </table>

      <h4>.uns &mdash; unstructured metadata (parameters, pipeline settings, &hellip;)</h4>
      <table>
        <tr><th>Key</th><th>Type</th><th>Preview</th></tr>
        {uns_rows}
      </table>
    </div>
    """
    display(HTML(html))


# ---------------------------------------------------------------------------
# Console fallback — formatted plain text
# ---------------------------------------------------------------------------


def _print_console_summary(
    anndata_object: anndata.AnnData,
    max_display_rows: int,
    max_display_columns: int,
    group_by_column: "str | None" = None,
) -> None:
    """
    Print a formatted text summary of an AnnData object.

    Used when ipywidgets is unavailable or when running outside a notebook.
    Uses the box-drawing style consistent with the SFC QC module.

    Arguments:
        anndata_object:      The AnnData object to summarise.
        max_display_rows:    How many rows to show in .obs and .var previews.
        max_display_columns: How many columns to show in the .X preview.
        group_by_column:     .obs column used for the subject summary table.

    Returns:
        None
    """
    separator = "=" * 70

    x_info = _x_matrix_info(anndata_object)
    matrix_type = "sparse" if x_info["is_sparse"] else "dense"
    density_note = (
        f", {x_info['density_percent']:.1f}% filled" if x_info["is_sparse"] else ""
    )

    # anndata >= 0.13 exposes .X as layers[None]; list named layers only.
    layers_keys = [key for key in anndata_object.layers.keys() if key is not None]
    obsm_keys = list(anndata_object.obsm.keys()) if anndata_object.obsm else []
    uns_keys = list(anndata_object.uns.keys()) if anndata_object.uns else []

    print(f"\n{separator}")
    print("ANNDATA INSPECTOR")
    print(separator)

    # [1] Structural overview
    print("\n[1] Structure")
    print(f"    Observations (.obs rows) : {anndata_object.n_obs:,}")
    print(f"    Variables (.var rows)    : {anndata_object.n_vars}")
    print(
        f"    .X matrix                : {anndata_object.n_obs:,} × {anndata_object.n_vars}"
        f"  ({x_info['dtype']}, {matrix_type}{density_note}, {x_info['size_human']})"
    )
    print(f"    .layers keys             : {layers_keys if layers_keys else 'none'}")
    print(f"    .obsm keys               : {obsm_keys if obsm_keys else 'none'}")
    print(f"    .uns keys                : {uns_keys if uns_keys else 'none'}")

    # [2] .obs preview
    rows_shown = min(max_display_rows, anndata_object.n_obs)
    print(f"\n[2] .obs metadata — first {rows_shown} rows")
    print(anndata_object.obs.head(rows_shown).to_string())
    print("\n    Column dtypes:")
    for column_name, column_dtype in anndata_object.obs.dtypes.items():
        n_unique = anndata_object.obs[column_name].nunique()
        print(f"      {column_name:<30} {str(column_dtype):<15} {n_unique:,} unique values")

    # [3] .var preview
    var_rows_shown = min(max_display_rows, anndata_object.n_vars)
    print(f"\n[3] .var metadata — first {var_rows_shown} rows")
    if not anndata_object.var.empty:
        print(anndata_object.var.head(var_rows_shown).to_string())
    else:
        print("    (no columns in .var)")

    var_names_preview = anndata_object.var_names[:20].tolist()
    print(f"\n    Variable names (first 20): {var_names_preview}")

    # [4] .X matrix preview
    x_slice_dataframe = _x_matrix_slice_as_dataframe(
        anndata_object, max_display_rows, max_display_columns
    )
    rows_shown_x = x_slice_dataframe.shape[0]
    cols_shown_x = x_slice_dataframe.shape[1]
    print(f"\n[4] .X matrix — first {rows_shown_x} rows × {cols_shown_x} columns")
    print(x_slice_dataframe.round(4).to_string())

    # [5] .obsm embeddings
    print("\n[5] .obsm — low-dimensional embeddings")
    if anndata_object.obsm:
        for key, array in anndata_object.obsm.items():
            shape_str = str(array.shape) if hasattr(array, "shape") else "unknown"
            dtype_str = str(array.dtype) if hasattr(array, "dtype") else "unknown"
            print(f"    {key:<25} shape {shape_str}  dtype {dtype_str}")
    else:
        print("    (none)")

    # [6] .uns metadata
    print("\n[6] .uns — unstructured metadata")
    if anndata_object.uns:
        for key, value in anndata_object.uns.items():
            value_type = type(value).__name__
            print(f"    {key:<25} {value_type}")
    else:
        print("    (none)")

    # [7] Subject / group summary
    effective_subject_column = group_by_column or _detect_subject_column(
        anndata_object.obs
    )
    if effective_subject_column and effective_subject_column in anndata_object.obs.columns:
        summary_dataframe = _build_subject_summary_dataframe(
            anndata_object.obs, effective_subject_column
        )
        print(f"\n[7] Subject / group summary — grouped by '{effective_subject_column}'")
        print(f"    {len(summary_dataframe)} unique subjects")
        print(summary_dataframe.to_string())

    print(separator + "\n")
