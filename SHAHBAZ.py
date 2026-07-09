"""
SHAHBAZ.PY
A simple, flexible Chart Generator + Dashboard Builder.
Supports two engines: "plotly" (interactive, default) and
"matplotlib" (uses matplotlib.pyplot + seaborn).

Supported chart_type values:
    bar, line, pie, scatter, hist, box, area, violin,
    density_heatmap, density_contour, sunburst, treemap,
    funnel, bubble, heatmap, strip

Note: sunburst, treemap, and funnel are Plotly-only (no
matplotlib/seaborn equivalent) — use engine="plotly" for those.

Themes:
    Plotly (`theme=`): "plotly_dark", "plotly_white", "ggplot2",
        "seaborn", "simple_white", "presentation", "none", ...
        (any built-in plotly.io template name)
    Matplotlib (`style=`): "darkgrid", "whitegrid", "dark", "white",
        "ticks" (seaborn styles), or any matplotlib style name such
        as "ggplot", "bmh", "fivethirtyeight", "seaborn-v0_8".

Bar options:
    orientation="v" | "h"          -> vertical / horizontal bars
    barmode="group" | "stack"      -> grouped / stacked bars
        (barmode only kicks in when `color` is set to a column
        name in df, which is used to split the bars into series)

Saving:
    save_path="out.html"  -> interactive HTML (plotly only)
    save_path="out.png"   -> static image (plotly needs `kaleido`
                              installed; matplotlib works natively)
    Works for both chart() and dashboard().

Usage:
    from shahbaz import chart, dashboard
    import pandas as pd

    df = pd.read_csv("your_data.csv")

    # Single chart (Plotly, default)
    chart(df, "bar", groupby_col="region", agg_func="count", title="Count by Region")

    # Single chart (matplotlib/seaborn), with a style + saved to PNG
    chart(df, "bar", x="region", y="sales", title="Sales by Region",
          engine="matplotlib", style="darkgrid", save_path="sales.png")

    # Horizontal, stacked bar, grouped by a "segment" column
    chart(df, "bar", x="region", y="sales", color="segment",
          orientation="h", barmode="stack", title="Sales by Region & Segment")

    # Dashboard with multiple charts
    fig = dashboard([
        {"df": df, "chart_type": "bar", "groupby_col": "region", "agg_func": "count", "title": "Count by Region"},
        {"df": df, "chart_type": "line", "x": "date", "y": "sales", "title": "Sales Over Time"},
        {"df": df, "chart_type": "box", "x": "category", "y": "price", "title": "Price by Category"},
        {"df": df, "chart_type": "bar", "groupby_col": "product", "agg_col": "revenue",
         "agg_func": "sum", "sort": True, "top_n": 5, "title": "Top 5 Products"},
    ])
    fig.show()

    # Same dashboard, matplotlib/seaborn engine, saved to a PNG
    fig = dashboard([...], engine="matplotlib", style="whitegrid", save_path="dashboard.png")
"""

import math
import os
import plotly.express as px
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns

# chart types that only exist in the Plotly engine (no matplotlib/seaborn equivalent)
PLOTLY_ONLY_TYPES = {"sunburst", "treemap", "funnel"}

# all recognized chart types, across both engines
ALL_CHART_TYPES = {
    "bar", "line", "pie", "scatter", "hist", "box", "area", "violin",
    "density_heatmap", "density_contour", "sunburst", "treemap",
    "funnel", "bubble", "heatmap", "strip",
}

# chart types that require an x column (checked up front for a clear error)
NEEDS_X = {
    "bar", "line", "pie", "scatter", "hist", "box", "area", "violin",
    "density_heatmap", "density_contour", "bubble", "funnel", "strip",
}

# chart types that require a y column (checked up front for a clear error)
NEEDS_Y = {
    "bar", "line", "pie", "scatter", "box", "area", "violin",
    "density_heatmap", "density_contour", "bubble", "funnel", "strip",
}

# chart types that are exempt from the generic x/y check because they
# validate their own required arguments further downstream
SELF_VALIDATING_TYPES = {"sunburst", "treemap", "heatmap"}

# seaborn's named styles, vs. everything else which is treated as a
# matplotlib style-sheet name (e.g. "ggplot", "bmh", "fivethirtyeight")
SEABORN_STYLES = {"darkgrid", "whitegrid", "dark", "white", "ticks"}


# ---------------------------------------------------------------------------
# SINGLE CHART GENERATOR
# ---------------------------------------------------------------------------
def chart(df, chart_type,
          x=None, y=None,
          title="Chart",
          theme="plotly_dark",
          style=None,
          color=None,
          groupby_col=None,
          agg_col=None,
          agg_func=None,
          sort=False,
          ascending=False,
          top_n=None,
          orientation="v",
          barmode="group",
          show=True,
          save_path=None,
          path=None,
          size=None,
          z=None,
          engine="plotly",
          ax=None):
    """
    Build a single chart, using either Plotly or matplotlib/seaborn.

    chart_type: "bar" | "line" | "pie" | "scatter" | "hist" | "box" | "area" | "violin"
              | "density_heatmap" | "sunburst" | "funnel" | "bubble"
              | "density_contour" | "treemap" | "heatmap" | "strip"

    engine: "plotly" (default, interactive) or "matplotlib" (static,
            drawn with matplotlib.pyplot + seaborn). "sunburst", "treemap",
            and "funnel" are only available with engine="plotly".

    theme: plotly template name, used when engine="plotly" (e.g.
           "plotly_dark", "plotly_white", "ggplot2", "seaborn", "simple_white").
    style: matplotlib/seaborn style name, used when engine="matplotlib"
           (e.g. "darkgrid", "whitegrid", "dark", "white", "ticks", "ggplot",
           "bmh", "fivethirtyeight"). Left as-is (no style change) if None.

    color: either
           - a color string / list of color strings (a custom palette), or
           - the name of a column in df, in which case bar/line/scatter/etc.
             charts are split into series by that column (this is what
             barmode="group"/"stack" applies to for bar charts).

    groupby_col + agg_func ("count"/"sum"/"mean"/"max"/"min") + agg_col:
        automatically groups & aggregates the data before plotting.
    sort / ascending / top_n: sort the aggregated data and keep only top_n rows.

    orientation: "v" (vertical, default) or "h" (horizontal) — applies to "bar".
    barmode: "group" (default) or "stack" — applies to "bar" when `color`
             is set to a grouping column.

    show: if True, displays the chart (fig.show() for plotly, plt.show() for
          matplotlib). Set False when building a dashboard.
    save_path: optional file path to save the chart to. ".html" saves an
               interactive Plotly file; ".png"/".jpg"/".svg"/".pdf" saves a
               static image (Plotly needs the `kaleido` package installed
               for static image export; matplotlib supports these natively).

    Extra params for special chart types:
      path : list of column names, required for "sunburst" and "treemap"
             (defines the hierarchy, e.g. ["region", "product"]). y (optional)
             is used as the "values" column for segment sizing.
      size : column name for bubble size, used by "bubble" (and optionally
             "scatter") to size the markers.
      z    : optional column name for "heatmap" (numeric values). If not
             given, "heatmap" plots a correlation matrix of numeric columns.
      ax   : (matplotlib engine only) an existing matplotlib Axes to draw
             into, used internally by dashboard(). Leave as None otherwise.

    Returns the chart object (Plotly Figure, or matplotlib Axes), or None on error.
    """

    data = df.copy()

    # --- validate chart type up front, for a clear error regardless of engine ---
    if chart_type not in ALL_CHART_TYPES:
        print(f"wrong chart type '{chart_type}'. choose from: {', '.join(sorted(ALL_CHART_TYPES))}")
        return None

    # --- normalize color argument: either a grouping column, or a palette ---
    color_col = None
    if isinstance(color, str) and color in data.columns:
        color_col = color
        color_seq = None
    elif color is None:
        color_seq = ["skyblue"]
    elif isinstance(color, str):
        color_seq = [color]
    else:
        color_seq = color

    # --- optional groupby / aggregation (shared by both engines) ---
    if groupby_col and agg_func:
        if groupby_col not in data.columns:
            print(f'groupby_col "{groupby_col}" not found in dataframe columns')
            return None

        if agg_func == "count":
            data = data[groupby_col].value_counts().reset_index()
            data.columns = [groupby_col, "Count"]
            x, y = groupby_col, "Count"

        elif agg_col:
            if agg_col not in data.columns:
                print(f'agg_col "{agg_col}" not found in dataframe columns')
                return None
            if agg_func not in ("sum", "mean", "max", "min"):
                print('wrong agg_func. choose from: "count", "sum", "mean", "max", "min"')
                return None
            grouped = data.groupby(groupby_col, as_index=False)[agg_col]
            data = getattr(grouped, agg_func)()
            x, y = groupby_col, agg_col

        else:
            print("agg_col is needed")
            return None

    elif groupby_col and not agg_func:
        print("agg_func is needed when groupby_col is given")
        return None

    # --- generic required-column checks (clear errors instead of a crash) ---
    if chart_type not in SELF_VALIDATING_TYPES:
        if chart_type in NEEDS_X and x is None:
            print(f'"{chart_type}" needs an x column, e.g. x="your_column"')
            return None
        if chart_type in NEEDS_Y and y is None:
            print(f'"{chart_type}" needs a y column, e.g. y="your_column"')
            return None

    if x is not None and x not in data.columns and chart_type not in ("sunburst", "treemap"):
        print(f'x column "{x}" not found in dataframe columns')
        return None
    if y is not None and y not in data.columns:
        print(f'y column "{y}" not found in dataframe columns')
        return None

    if orientation not in ("v", "h"):
        print('wrong orientation. choose "v" (vertical) or "h" (horizontal)')
        return None
    if barmode not in ("group", "stack"):
        print('wrong barmode. choose "group" or "stack"')
        return None

    # --- sort & limit rows ---
    if sort and y is not None:
        data = data.sort_values(y, ascending=ascending)

    if top_n is not None:
        data = data.head(top_n)

    if engine == "plotly":
        fig = _chart_plotly(data, chart_type, x, y, title, theme, color_col, color_seq,
                             path, size, z, orientation, barmode)
        return _finish_plotly(fig, show, save_path)
    elif engine in ("matplotlib", "seaborn"):
        result = _chart_matplotlib(data, chart_type, x, y, title, color_col, color_seq,
                                    path, size, z, ax, orientation, barmode, style)
        if result is None:
            return None
        drawn_ax, owns_fig = result
        return _finish_matplotlib(drawn_ax, owns_fig, show, save_path)
    else:
        print('wrong engine. choose "plotly" or "matplotlib"')
        return None


def _finish_plotly(fig, show, save_path):
    """Shared save/show logic for a single Plotly chart."""
    if fig is None:
        return None

    if save_path:
        _save_plotly(fig, save_path)

    if show:
        fig.show()
        # Don't also return fig here: in notebooks, returning a Figure as the
        # last expression of a cell makes Jupyter auto-display it AGAIN on
        # top of the fig.show() above, producing a duplicate chart.
        return None

    return fig


def _save_plotly(fig, save_path):
    ext = os.path.splitext(save_path)[1].lower()
    try:
        if ext == ".html":
            fig.write_html(save_path)
        elif ext in (".png", ".jpg", ".jpeg", ".svg", ".pdf", ".webp"):
            fig.write_image(save_path)
        else:
            print(f'unrecognized save_path extension "{ext}". use .html, .png, .jpg, .svg, or .pdf')
            return
        print(f"saved chart to {save_path}")
    except Exception as e:
        print(f"could not save chart to {save_path}: {e}")
        if ext != ".html":
            print('static image export needs the "kaleido" package (pip install kaleido)')


def _finish_matplotlib(ax, owns_fig, show, save_path):
    """Shared save/show logic for a single matplotlib chart."""
    fig = ax.get_figure()

    if owns_fig:
        plt.tight_layout()

    if save_path:
        try:
            fig.savefig(save_path, bbox_inches="tight")
            print(f"saved chart to {save_path}")
        except Exception as e:
            print(f"could not save chart to {save_path}: {e}")

    if owns_fig:
        if show:
            plt.show()
            return None
        return ax

    return ax


def _chart_plotly(data, chart_type, x, y, title, theme, color_col, color_seq,
                   path, size, z, orientation, barmode):
    """Plotly rendering path."""

    plot_fn = {
        "bar": px.bar,
        "line": px.line,
        "pie": px.pie,
        "scatter": px.scatter,
        "hist": px.histogram,
        "box": px.box,
        "area": px.area,
        "violin": px.violin,
        "density_heatmap": px.density_heatmap,
        "density_contour": px.density_contour,
        "sunburst": px.sunburst,
        "treemap": px.treemap,
        "funnel": px.funnel,
        "bubble": px.scatter,
        "heatmap": px.imshow,
        "strip": px.strip,
    }.get(chart_type)

    if chart_type == "pie":
        fig = plot_fn(data, names=x, values=y, title=title, color_discrete_sequence=color_seq)

    elif chart_type in ("sunburst", "treemap"):
        if not path:
            print(f'"{chart_type}" needs a path list, e.g. path=["region", "product"]')
            return None
        missing = [c for c in path if c not in data.columns]
        if missing:
            print(f'path column(s) not found in dataframe: {missing}')
            return None
        fig = plot_fn(data, path=path, values=y, title=title, color_discrete_sequence=color_seq)

    elif chart_type == "bubble":
        if not size:
            print('"bubble" needs a size column, e.g. size="revenue"')
            return None
        if size not in data.columns:
            print(f'size column "{size}" not found in dataframe columns')
            return None
        fig = plot_fn(data, x=x, y=y, size=size, color=color_col, title=title,
                      color_discrete_sequence=color_seq)

    elif chart_type == "bar":
        plot_x, plot_y = (y, x) if orientation == "h" else (x, y)
        fig = plot_fn(data, x=plot_x, y=plot_y, color=color_col, orientation=orientation,
                      barmode=barmode, title=title, color_discrete_sequence=color_seq)

    elif chart_type == "heatmap":
        if z is not None and x is not None and y is not None:
            if z not in data.columns:
                print(f'z column "{z}" not found in dataframe columns')
                return None
            pivot = data.pivot_table(index=y, columns=x, values=z, aggfunc="mean")
            fig = plot_fn(pivot, title=title, text_auto=True)
        else:
            numeric_df = data.select_dtypes(include="number")
            fig = plot_fn(numeric_df.corr(), title=title, text_auto=True)

    else:
        fig = plot_fn(data, x=x, y=y, color=color_col, title=title,
                      color_discrete_sequence=color_seq)

    fig.update_layout(template=theme)

    return fig


def _apply_matplotlib_style(style):
    """Apply a seaborn style or matplotlib style-sheet name, if given."""
    if not style:
        return
    if style in SEABORN_STYLES:
        sns.set_theme(style=style)
    else:
        try:
            plt.style.use(style)
        except (OSError, ValueError):
            print(f'unknown style "{style}", ignoring. try one of {sorted(SEABORN_STYLES)} '
                  f'or a matplotlib style like "ggplot", "bmh", "fivethirtyeight"')


def _chart_matplotlib(data, chart_type, x, y, title, color_col, color_seq,
                       path, size, z, ax, orientation, barmode, style):
    """matplotlib/seaborn rendering path. Returns (ax, owns_fig) or None on error."""

    if chart_type in PLOTLY_ONLY_TYPES:
        print(f'"{chart_type}" is only available with engine="plotly" '
              f'(no matplotlib/seaborn equivalent)')
        return None

    _apply_matplotlib_style(style)

    owns_fig = ax is None
    if owns_fig:
        fig, ax = plt.subplots()

    main_color = color_seq[0] if color_seq else None

    if chart_type == "bar":
        if color_col:
            pivot = data.pivot_table(index=x, columns=color_col, values=y, aggfunc="sum")
            kind = "barh" if orientation == "h" else "bar"
            stacked = barmode == "stack"
            pivot.plot(kind=kind, stacked=stacked, ax=ax, legend=True)
        elif orientation == "h":
            sns.barplot(data=data, x=y, y=x, color=main_color, ax=ax)
        else:
            sns.barplot(data=data, x=x, y=y, color=main_color, ax=ax)

    elif chart_type == "line":
        sns.lineplot(data=data, x=x, y=y, hue=color_col, color=main_color if not color_col else None,
                      ax=ax, marker="o")

    elif chart_type == "pie":
        ax.pie(data[y], labels=data[x], autopct="%1.1f%%")
        ax.axis("equal")

    elif chart_type == "scatter":
        sns.scatterplot(data=data, x=x, y=y, hue=color_col,
                         color=main_color if not color_col else None, ax=ax)

    elif chart_type == "hist":
        sns.histplot(data=data, x=x, hue=color_col,
                      color=main_color if not color_col else None, ax=ax)

    elif chart_type == "box":
        sns.boxplot(data=data, x=x, y=y, hue=color_col,
                     color=main_color if not color_col else None, ax=ax)

    elif chart_type == "area":
        sorted_data = data.sort_values(x) if x else data
        ax.fill_between(sorted_data[x], sorted_data[y], color=main_color, alpha=0.5)
        ax.plot(sorted_data[x], sorted_data[y], color=main_color)

    elif chart_type == "violin":
        sns.violinplot(data=data, x=x, y=y, hue=color_col,
                        color=main_color if not color_col else None, ax=ax)

    elif chart_type == "strip":
        sns.stripplot(data=data, x=x, y=y, hue=color_col,
                       color=main_color if not color_col else None, ax=ax)

    elif chart_type == "bubble":
        if not size:
            print('"bubble" needs a size column, e.g. size="revenue"')
            if owns_fig:
                plt.close(fig)
            return None
        if size not in data.columns:
            print(f'size column "{size}" not found in dataframe columns')
            if owns_fig:
                plt.close(fig)
            return None
        sizes = data[size] / data[size].max() * 500
        ax.scatter(data[x], data[y], s=sizes, color=main_color, alpha=0.6)

    elif chart_type == "density_heatmap":
        sns.histplot(data=data, x=x, y=y, bins=30, cbar=True, ax=ax)

    elif chart_type == "density_contour":
        sns.kdeplot(data=data, x=x, y=y, ax=ax, fill=False)

    elif chart_type == "heatmap":
        if z is not None and x is not None and y is not None:
            if z not in data.columns:
                print(f'z column "{z}" not found in dataframe columns')
                if owns_fig:
                    plt.close(fig)
                return None
            pivot = data.pivot_table(index=y, columns=x, values=z, aggfunc="mean")
            sns.heatmap(pivot, annot=True, ax=ax)
        else:
            numeric_df = data.select_dtypes(include="number")
            sns.heatmap(numeric_df.corr(), annot=True, ax=ax)

    ax.set_title(title)
    return ax, owns_fig


# ---------------------------------------------------------------------------
# DASHBOARD BUILDER
# ---------------------------------------------------------------------------
def dashboard(chart_specs, title="Dashboard", cols=2, theme="plotly_dark", style=None,
              height=None, width=1100, engine="plotly", figsize=None, show=True,
              save_path=None):
    """
    Build a dashboard (grid of subplots) from a list of chart specs.

    chart_specs: list of dicts. Each dict = kwargs you'd pass to chart(),
                 e.g. {"df": df, "chart_type": "bar", "x": "region", "y": "sales",
                       "title": "Sales by Region"}

    cols: number of columns in the grid (rows are computed automatically).
    engine: "plotly" (default, returns one interactive Figure) or
            "matplotlib" (returns one matplotlib Figure built with
            matplotlib.pyplot + seaborn subplots). "sunburst", "treemap",
            and "funnel" specs are skipped when engine="matplotlib" since
            they have no matplotlib/seaborn equivalent.
    theme: plotly template name (engine="plotly").
    style: matplotlib/seaborn style name, applied to the whole dashboard
           (engine="matplotlib").
    save_path: optional file path to save the whole dashboard to (".html"
               or ".png"/".jpg"/".svg"/".pdf", same rules as chart()).

    Plotly engine note: "pie", "sunburst", and "treemap" cells are
    automatically given a "domain" subplot type (instead of the default
    "xy") so they can render correctly inside the grid.

    Returns the combined Figure. Call fig.show() to display it (for the
    plotly engine; for matplotlib it's already shown if show=True).
    """

    if not chart_specs:
        print("chart_specs list is empty")
        return None

    n = len(chart_specs)
    rows = math.ceil(n / cols)

    if engine == "plotly":
        fig = _dashboard_plotly(chart_specs, title, cols, rows, theme, height, width)
        if fig is not None and save_path:
            _save_plotly(fig, save_path)
        return fig
    elif engine in ("matplotlib", "seaborn"):
        return _dashboard_matplotlib(chart_specs, title, cols, rows, figsize, show, style, save_path)
    else:
        print('wrong engine. choose "plotly" or "matplotlib"')
        return None


def _dashboard_plotly(chart_specs, title, cols, rows, theme, height, width):
    n = len(chart_specs)

    # These chart types need a "domain" subplot (not "xy") to render in a grid.
    DOMAIN_TYPES = {"pie", "sunburst", "treemap"}

    titles = [spec.get("title", f"Chart {i + 1}") for i, spec in enumerate(chart_specs)]

    # Build a per-cell "specs" grid so pie/sunburst/treemap get type="domain"
    # and everything else gets the normal type="xy".
    specs = []
    for r in range(rows):
        row_specs = []
        for c in range(cols):
            i = r * cols + c
            if i < n:
                ctype = chart_specs[i].get("chart_type")
                row_specs.append({"type": "domain"} if ctype in DOMAIN_TYPES else {"type": "xy"})
            else:
                row_specs.append(None)  # empty trailing cell in the grid
        specs.append(row_specs)

    fig = make_subplots(rows=rows, cols=cols, subplot_titles=titles, specs=specs)

    for i, spec in enumerate(chart_specs):
        ctype = spec.get("chart_type")

        spec = {**spec, "show": False, "engine": "plotly"}
        sub_fig = chart(**spec)
        if sub_fig is None:
            continue

        r, c = divmod(i, cols)
        for trace in sub_fig.data:
            fig.add_trace(trace, row=r + 1, col=c + 1)

        # axis titles only make sense for "xy" (cartesian) charts, not
        # domain charts like pie/sunburst/treemap
        if ctype not in DOMAIN_TYPES:
            x_label = spec.get("x")
            y_label = spec.get("y")
            if x_label:
                fig.update_xaxes(title_text=str(x_label), row=r + 1, col=c + 1)
            if y_label:
                fig.update_yaxes(title_text=str(y_label), row=r + 1, col=c + 1)

    # keep the legend if any pie/sunburst/treemap is present (needed to read
    # the slices) — otherwise hide it, since xy charts show their own labels
    has_domain_chart = any(s.get("chart_type") in DOMAIN_TYPES for s in chart_specs)

    fig.update_layout(
        title=title,
        template=theme,
        height=height or (400 * rows),
        width=width,
        showlegend=has_domain_chart,
    )

    return fig


def _dashboard_matplotlib(chart_specs, title, cols, rows, figsize, show, style, save_path):
    n = len(chart_specs)
    figsize = figsize or (6 * cols, 4.5 * rows)

    _apply_matplotlib_style(style)

    fig, axes = plt.subplots(rows, cols, figsize=figsize, squeeze=False)

    for i, spec in enumerate(chart_specs):
        r, c = divmod(i, cols)
        ax = axes[r][c]

        spec = {**spec, "show": False, "engine": "matplotlib", "ax": ax}
        result = chart(**spec)
        if result is None:
            ax.axis("off")

    # hide any unused trailing cells in the grid
    for i in range(n, rows * cols):
        r, c = divmod(i, cols)
        axes[r][c].axis("off")

    fig.suptitle(title)
    plt.tight_layout()

    if save_path:
        try:
            fig.savefig(save_path, bbox_inches="tight")
            print(f"saved dashboard to {save_path}")
        except Exception as e:
            print(f"could not save dashboard to {save_path}: {e}")

    if show:
        plt.show()

    return fig
