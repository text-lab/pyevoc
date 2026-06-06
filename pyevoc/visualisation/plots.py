"""PyEvoc plotting utilities.

This module consolidates the plotting routines originally developed in the
notebook workflow:

1. EVOC concentric maps;
2. EVOC semantic trees linking quadrant roots to collocation leaves;
3. emoji EVOC Plotly maps;
4. temporal EVOC Sankey diagrams.

The functions intentionally preserve the visual logic and default parameters of
the original notebook outputs while exposing a package-ready API.
"""

from __future__ import annotations

from pathlib import Path
import os
import re
import textwrap
import warnings

import numpy as np
import pandas as pd

# Optional plotting dependencies are imported lazily where possible.
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

try:
    from adjustText import adjust_text
except Exception:  # pragma: no cover
    adjust_text = None

try:
    import networkx as nx
except Exception:  # pragma: no cover
    nx = None

try:
    import plotly.graph_objects as go
except Exception:  # pragma: no cover
    go = None

try:
    from bs4 import BeautifulSoup
except Exception:  # pragma: no cover
    BeautifulSoup = None


OUTPUT_DIR = "evoc_outputs"

QUADRANT_ORDER = [
    "Central nucleus",
    "First periphery",
    "Contrast zone",
    "Peripheral system",
]

QUADRANT_COLOURS = {
    "Central nucleus": "#4f7fa6",
    "First periphery": "#4f9a55",
    "Contrast zone": "#b84848",
    "Peripheral system": "#8250a0",
}


# =============================================================================
# 1. EVOC CONCENTRIC MAP
# =============================================================================

QUADRANT_ANGLES = {
    "Central nucleus": (90, 180),
    "First periphery": (0, 90),
    "Contrast zone": (270, 360),
    "Peripheral system": (180, 270),
}

QUADRANT_ANGULAR_JITTER = {
    "Central nucleus": 1.5,
    "First periphery": 2.5,
    "Contrast zone": 18.0,
    "Peripheral system": 7.0,
}

QUADRANT_RADIAL_JITTER_SD = {
    "Central nucleus": 0.01,
    "First periphery": 0.03,
    "Contrast zone": 0.015,
    "Peripheral system": 0.01,
}

QUADRANT_RADIAL_JITTER_MEAN = {
    "Central nucleus": 0.00,
    "First periphery": 0.025,
    "Contrast zone": -0.05,
    "Peripheral system": 0.00,
}

CORE_COLOUR = "#f9dbe7"
PERIPHERY_COLOUR = "#e9f6fc"

RING_LINE_COLOUR = "#ffffff"
BORDER_COLOUR = "#cfcfcf"
LEADER_LINE_COLOUR = "#8a8a8a"

NODE_SIZE_MODE = "fixed"
FIXED_NODE_SIZE = 70
CONCRETENESS_SIZE_MIN = 90
CONCRETENESS_SIZE_MAX = 220

LABEL_OFFSET = 0.040
TERM_FONT_SIZE = 8

CENTRAL_LABEL_Y_OFFSET = 0.040
CONTRAST_LABEL_Y_OFFSET = -0.040

RANDOM_STATE = 123


def _ensure_output_dir(output_dir: str | Path = OUTPUT_DIR) -> str:
    output_dir = str(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def rescale_series(x, out_min, out_max):
    """Linearly rescale *x* to the interval ``[out_min, out_max]``.

    If all values are identical, every element is mapped to the midpoint
    ``(out_min + out_max) / 2``.

    Args:
        x: Array-like of numeric values.
        out_min: Lower bound of the output range.
        out_max: Upper bound of the output range.

    Returns:
        A :class:`numpy.ndarray` of rescaled float values.
    """
    x = pd.to_numeric(pd.Series(x), errors="coerce")
    xmin, xmax = x.min(), x.max()

    if xmin == xmax:
        return np.repeat((out_min + out_max) / 2, len(x))

    return np.asarray(
        out_min + (x - xmin) / (xmax - xmin) * (out_max - out_min),
        dtype=float,
    )


def select_terms_for_target(df, top_n_per_quadrant):
    """Select the top *n* terms per quadrant for the concentric map.

    Terms are ranked by quadrant-specific criteria: high-salience quadrants
    (Central nucleus, First periphery) prioritise relative diffusion; the
    Contrast zone prioritises rank; the Peripheral system prioritises
    diffusion then rank.

    Args:
        df: EVOC quadrants DataFrame containing ``relative_diffusion``,
            ``Rank``, ``frequency``, and ``quadrant`` columns.
        top_n_per_quadrant: Maximum number of terms to retain per quadrant.

    Returns:
        A concatenated :class:`~pandas.DataFrame` of selected terms.

    Raises:
        ValueError: If any required column is missing from *df*.
    """
    required_cols = {"relative_diffusion", "Rank", "frequency"}
    missing = required_cols.difference(df.columns)

    if missing:
        raise ValueError(
            f"evoc_quadrants_df is missing required columns for plotting: {missing}"
        )

    selected = []

    for q in QUADRANT_ORDER:
        sub = df[df["quadrant"].astype(str).eq(q)].copy()

        if q in {"Central nucleus", "First periphery"}:
            sub = sub.sort_values(
                ["relative_diffusion", "Rank", "frequency"],
                ascending=[False, True, False],
            )

        elif q == "Contrast zone":
            sub = sub.sort_values(
                ["Rank", "relative_diffusion", "frequency"],
                ascending=[False, False, False],
            )

        elif q == "Peripheral system":
            sub = sub.sort_values(
                ["relative_diffusion", "Rank", "frequency"],
                ascending=[False, False, False],
            )

        selected.append(sub.head(top_n_per_quadrant))

    return pd.concat(selected, ignore_index=True)


def get_aoe_threshold(pos_thresholds_round_df, upos):
    """Look up the AOE threshold for a given UPOS tag.

    Prefers the ``AOE_thr_round`` column when present, falling back to
    ``AOE_thr``.

    Args:
        pos_thresholds_round_df: DataFrame with at least ``upos`` and
            ``AOE_thr_round`` (or ``AOE_thr``) columns.
        upos: Universal POS tag string (e.g. ``"NOUN"``).

    Returns:
        The AOE threshold as a :class:`float`.

    Raises:
        ValueError: If no row matches *upos* or neither threshold column
            is found.
    """
    sub = pos_thresholds_round_df.loc[
        pos_thresholds_round_df["upos"].astype(str).eq(upos)
    ]

    if sub.empty:
        raise ValueError(f"No AOE threshold found for UPOS = {upos}")

    if "AOE_thr_round" in sub.columns:
        return float(sub["AOE_thr_round"].iloc[0])

    if "AOE_thr" in sub.columns:
        return float(sub["AOE_thr"].iloc[0])

    raise ValueError("Threshold table must contain 'AOE_thr_round' or 'AOE_thr'.")


def rank_to_radius(rank, rank_min, rank_max, out_min=0.15, out_max=0.92):
    """Map a single rank value linearly onto a radial distance.

    Args:
        rank: The rank value to convert.
        rank_min: Minimum rank in the dataset.
        rank_max: Maximum rank in the dataset.
        out_min: Innermost radius of the output range.
        out_max: Outermost radius of the output range.

    Returns:
        A :class:`float` radius in ``[out_min, out_max]``.
    """
    if rank_min == rank_max:
        return (out_min + out_max) / 2

    return out_min + (rank - rank_min) / (rank_max - rank_min) * (out_max - out_min)


def build_rank_reference_rings(
    plot_df,
    upos,
    pos_thresholds_round_df,
    out_min=0.15,
    out_max=0.92,
    sd_multiplier=1.0,
):
    """Compute the three reference ring radii for the concentric map.

    The rings correspond to ``AOE − sd``, ``AOE``, and ``AOE + sd`` where
    *sd* is the standard deviation of the Rank column scaled by
    *sd_multiplier*.

    Args:
        plot_df: DataFrame of selected terms with a numeric ``Rank`` column.
        upos: Universal POS tag used to look up the AOE threshold.
        pos_thresholds_round_df: Threshold table; see :func:`get_aoe_threshold`.
        out_min: Innermost radius bound passed to :func:`rank_to_radius`.
        out_max: Outermost radius bound passed to :func:`rank_to_radius`.
        sd_multiplier: Scales the rank standard deviation for the ring
            offset.

    Returns:
        A tuple ``(ring_radii, rank_values_clipped, rank_labels, aoe_thr,
        rank_sd)`` where *ring_radii* are the three display radii, and
        *rank_values_clipped* are the corresponding rank values clamped
        to the observed range.

    Raises:
        ValueError: If the ``Rank`` column is empty after dropping NaNs.
    """
    ranks = pd.to_numeric(plot_df["Rank"], errors="coerce").dropna()

    if ranks.empty:
        raise ValueError("Cannot compute rank rings because Rank is empty.")

    rank_min = float(ranks.min())
    rank_max = float(ranks.max())
    rank_sd = float(ranks.std(ddof=1))

    aoe_thr = get_aoe_threshold(pos_thresholds_round_df, upos)

    rank_values_raw = [
        aoe_thr - sd_multiplier * rank_sd,
        aoe_thr,
        aoe_thr + sd_multiplier * rank_sd,
    ]

    rank_values_clipped = [min(max(v, rank_min), rank_max) for v in rank_values_raw]

    ring_radii = [
        rank_to_radius(
            rank=r,
            rank_min=rank_min,
            rank_max=rank_max,
            out_min=out_min,
            out_max=out_max,
        )
        for r in rank_values_clipped
    ]

    rank_labels = [
        f"Rank = {rank_values_raw[0]:.2f}",
        f"AOE = {rank_values_raw[1]:.2f}",
        f"Rank = {rank_values_raw[2]:.2f}",
    ]

    return ring_radii, rank_values_clipped, rank_labels, aoe_thr, rank_sd


def compute_node_sizes(
    plot_df,
    mode=NODE_SIZE_MODE,
    fixed_size=FIXED_NODE_SIZE,
    size_min=CONCRETENESS_SIZE_MIN,
    size_max=CONCRETENESS_SIZE_MAX,
):
    """Compute scatter-plot marker sizes for each term node.

    Args:
        plot_df: DataFrame of selected terms.
        mode: ``"fixed"`` returns a constant size for every node;
            ``"concreteness"`` rescales sizes from the
            ``concreteness_score`` column.
        fixed_size: Marker size used in ``"fixed"`` mode.
        size_min: Minimum marker size in ``"concreteness"`` mode.
        size_max: Maximum marker size in ``"concreteness"`` mode.

    Returns:
        A :class:`numpy.ndarray` of marker sizes, one per row in *plot_df*.

    Raises:
        ValueError: If *mode* is unrecognised or ``concreteness_score`` is
            missing when *mode* is ``"concreteness"``.
    """
    if mode == "fixed":
        return np.repeat(fixed_size, len(plot_df))

    if mode != "concreteness":
        raise ValueError("node_size_mode must be either 'fixed' or 'concreteness'.")

    if "concreteness_score" not in plot_df.columns:
        raise ValueError(
            "node_size_mode='concreteness' requires a 'concreteness_score' column."
        )

    scores = pd.to_numeric(plot_df["concreteness_score"], errors="coerce")

    if scores.notna().sum() < 2:
        return np.repeat(fixed_size, len(plot_df))

    return rescale_series(
        scores.fillna(scores.median()),
        out_min=size_min,
        out_max=size_max,
    )


def assign_target_coordinates(
    plot_df,
    node_size_mode=NODE_SIZE_MODE,
    fixed_node_size=FIXED_NODE_SIZE,
    concreteness_size_min=CONCRETENESS_SIZE_MIN,
    concreteness_size_max=CONCRETENESS_SIZE_MAX,
    random_state=RANDOM_STATE,
):
    """Compute polar and Cartesian coordinates for every term in *plot_df*.

    Assigns each term a radius (from Rank), an angular position (from
    relative diffusion within its quadrant), and small random jitter.
    Adds ``radius``, ``theta``, ``x``, ``y``, and ``point_size`` columns
    to the returned DataFrame.

    Args:
        plot_df: DataFrame of selected terms with ``Rank``,
            ``relative_diffusion``, and ``quadrant`` columns.
        node_size_mode: Passed to :func:`compute_node_sizes`.
        fixed_node_size: Passed to :func:`compute_node_sizes`.
        concreteness_size_min: Passed to :func:`compute_node_sizes`.
        concreteness_size_max: Passed to :func:`compute_node_sizes`.
        random_state: Seed for the NumPy random generator used for jitter.

    Returns:
        A copy of *plot_df* with coordinate and size columns appended.
    """
    rng = np.random.default_rng(random_state)

    plot_df = plot_df.copy()

    plot_df["Rank"] = pd.to_numeric(plot_df["Rank"], errors="coerce")
    plot_df["frequency"] = pd.to_numeric(plot_df["frequency"], errors="coerce")
    plot_df["relative_diffusion"] = pd.to_numeric(
        plot_df["relative_diffusion"],
        errors="coerce",
    )

    plot_df = plot_df.dropna(subset=["Rank", "relative_diffusion"]).copy()

    plot_df["radius"] = rescale_series(
        plot_df["Rank"],
        out_min=0.15,
        out_max=0.92,
    )

    plot_df["point_size"] = compute_node_sizes(
        plot_df,
        mode=node_size_mode,
        fixed_size=fixed_node_size,
        size_min=concreteness_size_min,
        size_max=concreteness_size_max,
    )

    theta = np.zeros(len(plot_df))

    for q in QUADRANT_ORDER:
        idx = plot_df.index[plot_df["quadrant"].astype(str).eq(q)].to_list()

        if not idx:
            continue

        start, end = QUADRANT_ANGLES[q]
        sub = plot_df.loc[idx].copy()

        diff = pd.to_numeric(sub["relative_diffusion"], errors="coerce")

        if diff.nunique(dropna=True) <= 1:
            angles = np.repeat((start + end) / 2, len(sub))
        else:
            angles = rescale_series(diff, out_min=start + 15, out_max=end - 15)

        angular_jitter = QUADRANT_ANGULAR_JITTER.get(q, 1.5)

        if angular_jitter > 0:
            angles = angles + rng.normal(
                loc=0,
                scale=angular_jitter,
                size=len(sub),
            )

        angles = np.clip(angles, start + 8, end - 8)
        angle_map = dict(zip(sub.index, angles))

        for row_idx in sub.index:
            theta[row_idx] = np.deg2rad(angle_map[row_idx])

    plot_df["theta"] = theta

    radial_jitter = np.zeros(len(plot_df))

    for q in QUADRANT_ORDER:
        idx = plot_df.index[plot_df["quadrant"].astype(str).eq(q)].to_numpy()

        if len(idx) == 0:
            continue

        jitter_mean = QUADRANT_RADIAL_JITTER_MEAN.get(q, 0.0)
        jitter_sd = QUADRANT_RADIAL_JITTER_SD.get(q, 0.010)

        radial_jitter[idx] = rng.normal(
            loc=jitter_mean,
            scale=jitter_sd,
            size=len(idx),
        )

    plot_df["radius"] = (plot_df["radius"] + radial_jitter).clip(0.12, 0.94)

    plot_df["x"] = plot_df["radius"] * np.cos(plot_df["theta"])
    plot_df["y"] = plot_df["radius"] * np.sin(plot_df["theta"])

    return plot_df


def point_radius_data_units(ax, point_size):
    """Convert a scatter marker area (points²) to a radius in data coordinates.

    Forces a canvas draw to ensure the axis transform is up to date before
    converting pixel distances to data-space distances.

    Args:
        ax: The :class:`matplotlib.axes.Axes` containing the scatter plot.
        point_size: Marker area in points² (as passed to
            :func:`matplotlib.axes.Axes.scatter`).

    Returns:
        The marker radius expressed in data-coordinate units.
    """
    fig = ax.figure
    fig.canvas.draw()

    r_points = np.sqrt(point_size / np.pi)
    r_pixels = r_points * fig.dpi / 72.0

    x0, y0 = ax.transData.transform((0, 0))
    x1, _ = ax.transData.inverted().transform((x0 + r_pixels, y0))

    return abs(x1)


def label_direction_for_term(row, quadrant_position_index):
    """Return the preferred unit direction and y-offset for a term label.

    The direction alternates between left and right based on the term's
    position index within its quadrant, minimising overlap on crowded
    quadrant edges.

    Args:
        row: A :class:`pandas.Series` with at least ``quadrant``, ``x``,
            and ``y`` fields.
        quadrant_position_index: Zero-based rank of this term within its
            quadrant (after sorting).

    Returns:
        A tuple ``(ux, uy, y_offset)`` where *ux* and *uy* form a unit
        vector and *y_offset* is a small vertical displacement in data
        coordinates.
    """
    q = str(row["quadrant"])

    if q == "Central nucleus":
        ux, uy = (1.0, 0.0) if quadrant_position_index < 5 else (-1.0, 0.0)
        return ux, uy, CENTRAL_LABEL_Y_OFFSET

    if q == "Contrast zone":
        ux, uy = (-1.0, 0.0) if quadrant_position_index < 6 else (1.0, 0.0)
        return ux, uy, CONTRAST_LABEL_Y_OFFSET

    if q == "First periphery":
        ux, uy = (1.0, 0.0) if quadrant_position_index < 4 else (-1.0, 0.0)
        return ux, uy, CENTRAL_LABEL_Y_OFFSET

    if q == "Peripheral system":
        ux, uy = (-1.0, 0.0) if quadrant_position_index < 5 else (1.0, 0.0)
        return ux, uy, CONTRAST_LABEL_Y_OFFSET

    norm = np.hypot(row["x"], row["y"])
    ux, uy = (1.0, 0.0) if norm == 0 else (row["x"] / norm, row["y"] / norm)

    return ux, uy, 0.0


def build_evoc_target_plot(
    evoc_quadrants_df,
    upos,
    pos_thresholds_round_df,
    top_n_per_quadrant=20,
    output_dir=OUTPUT_DIR,
    figsize=(9.5, 8),
    dpi=600,
    node_size_mode=NODE_SIZE_MODE,
    fixed_node_size=FIXED_NODE_SIZE,
    concreteness_size_min=CONCRETENESS_SIZE_MIN,
    concreteness_size_max=CONCRETENESS_SIZE_MAX,
    sd_multiplier=1.0,
    random_state=RANDOM_STATE,
):
    """Build the original notebook-style EVOC concentric map."""

    if adjust_text is None:
        raise ImportError("Install adjustText to use build_evoc_target_plot().")

    output_dir = _ensure_output_dir(output_dir)

    required_cols = {
        "term",
        "upos",
        "quadrant",
        "Rank",
        "frequency",
        "relative_diffusion",
    }

    missing = required_cols.difference(evoc_quadrants_df.columns)

    if missing:
        raise ValueError(f"evoc_quadrants_df is missing required columns: {missing}")

    if (
        node_size_mode == "concreteness"
        and "concreteness_score" not in evoc_quadrants_df.columns
    ):
        raise ValueError(
            "node_size_mode='concreteness' requires 'concreteness_score'."
        )

    df = evoc_quadrants_df.loc[evoc_quadrants_df["upos"].astype(str).eq(upos)].copy()

    if df.empty:
        raise ValueError(f"No terms available for UPOS = {upos}")

    plot_df = select_terms_for_target(df, top_n_per_quadrant=top_n_per_quadrant)

    plot_df = assign_target_coordinates(
        plot_df,
        node_size_mode=node_size_mode,
        fixed_node_size=fixed_node_size,
        concreteness_size_min=concreteness_size_min,
        concreteness_size_max=concreteness_size_max,
        random_state=random_state,
    )

    ring_radii, rank_values, rank_labels, aoe_thr, rank_sd = build_rank_reference_rings(
        plot_df=plot_df,
        upos=upos,
        pos_thresholds_round_df=pos_thresholds_round_df,
        sd_multiplier=sd_multiplier,
    )

    core_limit = ring_radii[1]

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_aspect("equal")
    ax.axis("off")

    ax.add_patch(plt.Circle((0, 0), 1.0, color=PERIPHERY_COLOUR, zorder=0))
    ax.add_patch(plt.Circle((0, 0), core_limit, color=CORE_COLOUR, zorder=1))

    for r in ring_radii:
        ax.add_patch(
            plt.Circle(
                (0, 0),
                r,
                fill=False,
                edgecolor=RING_LINE_COLOUR,
                linewidth=1.6,
                zorder=2,
                alpha=0.98,
            )
        )

    ax.add_patch(
        plt.Circle(
            (0, 0),
            1.0,
            fill=False,
            edgecolor=BORDER_COLOUR,
            linewidth=1.0,
            zorder=3,
        )
    )

    for r, label in zip(ring_radii, rank_labels):
        ax.text(
            0.015 - r,
            0.0,
            label,
            ha="center",
            va="center",
            rotation=90,
            fontsize=7,
            fontweight="bold",
            color="#333333",
            bbox=dict(
                boxstyle="round,pad=0.20",
                facecolor="white",
                edgecolor="#c8c8c8",
                alpha=0.92,
            ),
            zorder=8,
        )

    texts = []
    text_rows = []

    for q in QUADRANT_ORDER:
        sub = plot_df[plot_df["quadrant"].astype(str).eq(q)].copy()

        if sub.empty:
            continue

        sub = sub.sort_values(
            ["Rank", "relative_diffusion", "frequency"],
            ascending=[True, False, False],
        )

        ax.scatter(
            sub["x"],
            sub["y"],
            s=sub["point_size"],
            facecolor=QUADRANT_COLOURS[q],
            edgecolor="white",
            linewidth=1.1,
            alpha=0.90,
            zorder=5,
        )

        for local_i, (idx, r) in enumerate(sub.iterrows()):
            label = str(r["term"])

            if len(label) > 18:
                label = label[:17] + "…"

            node_radius = point_radius_data_units(ax, r["point_size"])

            ux, uy, y_offset = label_direction_for_term(
                row=r,
                quadrant_position_index=local_i,
            )

            x_text = r["x"] + ux * (node_radius + LABEL_OFFSET)
            y_text = r["y"] + uy * (node_radius + LABEL_OFFSET) + y_offset

            txt = ax.text(
                x_text,
                y_text,
                label,
                fontsize=TERM_FONT_SIZE,
                fontstyle="italic",
                color="#222222",
                family="sans-serif",
                ha="left" if ux >= 0 else "right",
                va="center",
                zorder=7,
            )

            texts.append(txt)
            text_rows.append((idx, r, node_radius, ux, uy, y_offset))

    adjust_text(
        texts,
        x=plot_df["x"].values,
        y=plot_df["y"].values,
        force_text=(0.18, 0.30),
        force_points=(0.18, 0.30),
        expand_text=(1.03, 1.08),
        expand_points=(1.08, 1.15),
        only_move={
            "text": "xy",
            "static": "xy",
            "explode": "xy",
            "pull": "xy",
        },
        ax=ax,
    )

    fig.canvas.draw()

    for txt, (_, r, node_radius, ux, uy, y_offset) in zip(texts, text_rows):
        target_x, target_y = r["x"], r["y"]
        text_x, text_y = txt.get_position()

        dx = text_x - target_x
        dy = text_y - target_y
        dist = np.hypot(dx, dy)

        min_dist = node_radius + LABEL_OFFSET

        if dist < min_dist:
            text_x = target_x + ux * min_dist
            text_y = target_y + uy * min_dist + y_offset
            txt.set_position((text_x, text_y))

    fig.canvas.draw()

    for txt, (_, r, node_radius, ux, uy, y_offset) in zip(texts, text_rows):
        target_x, target_y = r["x"], r["y"]
        text_x, text_y = txt.get_position()

        dx = text_x - target_x
        dy = text_y - target_y
        dist = np.hypot(dx, dy)

        if dist <= node_radius + 0.01:
            continue

        x_edge = target_x + (dx / dist) * node_radius
        y_edge = target_y + (dy / dist) * node_radius

        ax.plot(
            [x_edge, text_x],
            [y_edge, text_y],
            linestyle="--",
            linewidth=0.65,
            color=LEADER_LINE_COLOUR,
            alpha=0.68,
            zorder=4,
        )

    legend_elements = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label=q,
            markerfacecolor=QUADRANT_COLOURS[q],
            markeredgecolor="white",
            markersize=7.8,
        )
        for q in QUADRANT_ORDER
    ]

    legend_elements.extend(
        [
            Patch(facecolor=CORE_COLOUR, edgecolor="black", label="Core zone"),
            Patch(facecolor=PERIPHERY_COLOUR, edgecolor="black", label="Periphery zone"),
        ]
    )

    ax.legend(
        handles=legend_elements,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=3,
        frameon=False,
        fontsize=8.2,
        title_fontproperties={"weight": "bold"},
        columnspacing=1.6,
        handletextpad=0.5,
        labelspacing=0.8,
        borderpad=0.4,
    )

    ax.set_xlim(-1.28, 1.18)
    ax.set_ylim(-1.24, 1.18)

    suffix = "fixed" if node_size_mode == "fixed" else "concreteness"

    output_file = os.path.join(
        output_dir,
        f"evoc_target_{upos.lower()}_concentric_{suffix}.png",
    )

    plt.savefig(output_file, dpi=dpi, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)

    print(f"Saved: {output_file}")

    return output_file


# =============================================================================
# 2. EVOC SEMANTIC TREES: TERMS -> COLLOCATION LEAVES
# =============================================================================

QUADRANT_ROW_ORDER = [
    "Central nucleus",
    "First periphery",
    "Contrast zone",
    "Peripheral system",
]

TOP_ROOTS_PER_QUADRANT = 4
MAX_LEAVES_PER_ROOT = 5
MIN_LEAF_FREQ = 5

ROOT_X_START = -3.4
ROOT_X_STEP = 2.25

ROW_HEIGHT = 2.35
ROW_TOP_PAD = 0.5
ROW_BOTTOM_PAD = 0.5

PERIPHERAL_EXTRA_ROOT_LINES = True
ROOTS_PER_LINE = 4
PERIPHERAL_LINE_GAP = 1.72

ROW_SEPARATOR_X_MIN = -4.70
ROW_SEPARATOR_X_MAX = 4.70
ROW_SEPARATOR_SHIFT = 0.00
ROW_SEPARATOR_COLOUR = "rgba(185,185,185,0.40)"
ROW_SEPARATOR_WIDTH = 1
ROW_SEPARATOR_RIGHT_PAD = 0.45

LEAF_VERTICAL_DROP = 0.835
LEAF_ROW_STAGGER = 0.275
LEAF_HORIZONTAL_SPREAD = 0.675

LEFT_LABEL_X = -4.95
LEFT_LINE_X = -4.80
LEFT_LINE_WIDTH = 3

ROOT_LABEL_MAX_CHARS = 22
LEAF_LABEL_MAX_CHARS = 34

ROOT_SIZE_MIN = 18
ROOT_SIZE_MAX = 38
LEAF_SIZE_MIN = 8
LEAF_SIZE_MAX = 22

EDGE_WIDTH_MIN = 0.6
EDGE_WIDTH_MAX = 4.0
EDGE_CURVATURE = 0.25

TREE_FIG_WIDTH = 1650
TREE_FIG_HEIGHT = 1050
PNG_SCALE = 4

LAYOUT_SEED = 123


def normalise_quadrant_name(x):
    """Normalise a snake_case quadrant identifier to its display form.

    Maps keys such as ``"central_nucleus"`` to ``"Central nucleus"``.
    Unknown values are returned as-is after stripping whitespace.

    Args:
        x: Raw quadrant name string.

    Returns:
        The normalised display string.
    """
    mapping = {
        "central_nucleus": "Central nucleus",
        "first_periphery": "First periphery",
        "contrast_zone": "Contrast zone",
        "peripheral_system": "Peripheral system",
    }
    return mapping.get(str(x).strip(), str(x).strip())


def shorten_label(x, max_chars):
    """Truncate *x* to *max_chars* characters, appending ``…`` if needed.

    Args:
        x: Label string.
        max_chars: Maximum number of characters (including the ellipsis).

    Returns:
        The original string if it fits, otherwise a truncated version.
    """
    x = str(x)
    return x if len(x) <= max_chars else x[: max_chars - 1] + "…"


def format_float(x, digits=2):
    """Format a numeric value as a fixed-point string, or ``"-"`` if NaN.

    Args:
        x: Numeric value.
        digits: Number of decimal places.

    Returns:
        Formatted string.
    """
    return "-" if pd.isna(x) else f"{float(x):.{digits}f}"


def format_int(x):
    """Format a numeric value as a comma-separated integer string, or ``"-"`` if NaN.

    Args:
        x: Numeric value.

    Returns:
        Formatted string.
    """
    return "-" if pd.isna(x) else f"{int(round(float(x))):,}"


def format_p_value(x):
    """Format a p-value for display, using scientific notation below 0.001.

    Args:
        x: p-value (float or NaN-like).

    Returns:
        Formatted string, or ``"-"`` for missing values.
    """
    if pd.isna(x):
        return "-"
    x = float(x)
    return f"{x:.2e}" if x < 0.001 else f"{x:.4f}"


def get_total_users(tokens_df):
    """Return the number of unique users in *tokens_df*.

    Args:
        tokens_df: Token-level DataFrame with a ``user_id`` column.

    Returns:
        Integer count of unique non-null user identifiers.

    Raises:
        ValueError: If *tokens_df* is ``None``, lacks a ``user_id``
            column, or contains no valid users.
    """
    if tokens_df is None or "user_id" not in tokens_df.columns:
        raise ValueError("tokens_df with a valid 'user_id' column is required.")

    n_users = tokens_df["user_id"].astype("string").nunique(dropna=True)

    if n_users <= 0:
        raise ValueError("No valid users found in tokens_df.")

    return n_users


def hex_to_rgb(hex_colour):
    """Convert a hex colour string to an ``(R, G, B)`` integer tuple.

    Args:
        hex_colour: Hex string with or without a leading ``#``
            (e.g. ``"#4f7fa6"`` or ``"4f7fa6"``).

    Returns:
        A tuple ``(r, g, b)`` of integers in ``[0, 255]``.
    """
    hex_colour = hex_colour.lstrip("#")
    return tuple(int(hex_colour[i : i + 2], 16) for i in (0, 2, 4))


def blend_with_white(hex_colour, amount):
    """Blend a hex colour toward white by *amount* (0 = original, 1 = white).

    Args:
        hex_colour: Base colour as a hex string.
        amount: Blend fraction in ``[0, 1]``.

    Returns:
        An ``"rgb(r,g,b)"`` CSS string.
    """
    r, g, b = hex_to_rgb(hex_colour)
    r = int(r + (255 - r) * amount)
    g = int(g + (255 - g) * amount)
    b = int(b + (255 - b) * amount)
    return f"rgb({r},{g},{b})"


def rescale(x, out_min, out_max):
    """Linearly rescale *x* to ``[out_min, out_max]``, handling NaN gracefully.

    If the input range is zero or contains only NaN values, every element is
    mapped to the midpoint ``(out_min + out_max) / 2``.

    Args:
        x: Array-like of numeric values.
        out_min: Lower bound of the output range.
        out_max: Upper bound of the output range.

    Returns:
        A :class:`numpy.ndarray` of rescaled float values.
    """
    x = pd.to_numeric(pd.Series(x), errors="coerce")
    xmin, xmax = x.min(), x.max()

    if pd.isna(xmin) or pd.isna(xmax) or xmin == xmax:
        return np.repeat((out_min + out_max) / 2, len(x))

    return out_min + (x - xmin) / (xmax - xmin) * (out_max - out_min)


def curved_edge_points(x0, y0, x1, y1, curvature=EDGE_CURVATURE, n=35):
    """Sample points along a quadratic Bézier curve between two nodes.

    The control point is placed at the horizontal midpoint and above the
    higher of the two endpoints by *curvature* units.

    Args:
        x0: X coordinate of the start node.
        y0: Y coordinate of the start node.
        x1: X coordinate of the end node.
        y1: Y coordinate of the end node.
        curvature: Vertical offset of the Bézier control point.
        n: Number of sample points along the curve.

    Returns:
        A tuple ``(x_points, y_points)`` of :class:`numpy.ndarray`.
    """
    t = np.linspace(0, 1, n)
    xm = (x0 + x1) / 2
    ym = max(y0, y1) + curvature

    x = (1 - t) ** 2 * x0 + 2 * (1 - t) * t * xm + t**2 * x1
    y = (1 - t) ** 2 * y0 + 2 * (1 - t) * t * ym + t**2 * y1

    return x, y


def effective_x_geometry(pos):
    """Derive axis and legend x-geometry from the node position dictionary.

    Computes separator x-coordinates, full axis range, and a normalised
    legend x-anchor that keeps the legend visually centred over the plot
    area.

    Args:
        pos: Dict mapping node identifiers to ``(x, y)`` tuples.  May be
            empty, in which case ``ROW_SEPARATOR_X_MAX`` is used as the
            right boundary.

    Returns:
        A :class:`dict` with keys ``max_x``, ``separator_x0``,
        ``separator_x1``, ``x_range_min``, ``x_range_max``, and
        ``legend_x``.
    """
    if not pos:
        max_x = ROW_SEPARATOR_X_MAX
    else:
        max_x = max(x for x, y in pos.values())

    separator_x0 = ROW_SEPARATOR_X_MIN
    separator_x1 = max_x + ROW_SEPARATOR_RIGHT_PAD

    x_range_min = LEFT_LABEL_X - 0.55
    x_range_max = separator_x1 + 0.35

    plot_center_x = (separator_x0 + separator_x1) / 2

    if x_range_max == x_range_min:
        legend_x = 0.5
    else:
        legend_x = (plot_center_x - x_range_min) / (x_range_max - x_range_min)
        legend_x = min(max(legend_x, 0.0), 1.0)

    return {
        "max_x": max_x,
        "separator_x0": separator_x0,
        "separator_x1": separator_x1,
        "x_range_min": x_range_min,
        "x_range_max": x_range_max,
        "legend_x": legend_x,
    }


def select_roots_for_upos(
    evoc_quadrants_df,
    tokens_df,
    upos,
    top_roots_per_quadrant=TOP_ROOTS_PER_QUADRANT,
):
    """Select the top root terms for a given UPOS tag across all quadrants.

    Terms are ranked by quadrant-specific criteria (diffusion, rank,
    frequency) and the top *top_roots_per_quadrant* are kept per quadrant.
    Relative diffusion is recomputed from ``frequency / n_users_total``.

    Args:
        evoc_quadrants_df: EVOC quadrants DataFrame.
        tokens_df: Token-level DataFrame used to derive the total user count.
        upos: Universal POS tag to filter by (e.g. ``"NOUN"``).
        top_roots_per_quadrant: Maximum roots to retain per quadrant.

    Returns:
        A concatenated :class:`~pandas.DataFrame` of selected root terms.

    Raises:
        ValueError: If required columns are missing or no roots are found.
    """
    required_cols = {
        "term",
        "upos",
        "quadrant",
        "Rank",
        "frequency",
        "relative_diffusion",
    }

    missing = required_cols.difference(evoc_quadrants_df.columns)
    if missing:
        raise ValueError(f"evoc_quadrants_df is missing required columns: {missing}")

    n_users_total = get_total_users(tokens_df)

    roots = evoc_quadrants_df.copy()
    roots["quadrant"] = roots["quadrant"].map(normalise_quadrant_name)
    roots["term_norm"] = roots["term"].astype(str).str.lower().str.strip()
    roots["upos"] = roots["upos"].astype(str)

    roots["Rank"] = pd.to_numeric(roots["Rank"], errors="coerce")
    roots["frequency"] = pd.to_numeric(roots["frequency"], errors="coerce")

    roots["relative_diffusion"] = roots["frequency"] / n_users_total

    roots = roots.loc[
        roots["upos"].eq(upos)
        & roots["quadrant"].isin(QUADRANT_ORDER)
        & roots["term_norm"].ne("")
        & roots["Rank"].notna()
        & roots["frequency"].notna()
    ].copy()

    selected = []

    for q in QUADRANT_ORDER:
        sub = roots.loc[roots["quadrant"].eq(q)].copy()

        if sub.empty:
            continue

        if q in {"Central nucleus", "First periphery"}:
            sub = sub.sort_values(
                ["relative_diffusion", "Rank", "frequency"],
                ascending=[False, True, False],
            )
        elif q == "Contrast zone":
            sub = sub.sort_values(
                ["Rank", "relative_diffusion", "frequency"],
                ascending=[False, False, False],
            )
        else:
            sub = sub.sort_values(
                ["relative_diffusion", "Rank", "frequency"],
                ascending=[False, False, False],
            )

        selected.append(sub.head(top_roots_per_quadrant))

    if not selected:
        raise ValueError(f"No root terms selected for UPOS = {upos}.")

    return pd.concat(selected, ignore_index=True)


def prepare_collocation_leaves(collocations_df, tokens_df, min_leaf_freq=MIN_LEAF_FREQ):
    """Clean and filter a collocation DataFrame for use as tree leaves.

    Normalises terms, computes relative diffusion, and removes low-frequency
    entries.  The result is sorted by association strength (G²) descending.

    Args:
        collocations_df: Raw collocation DataFrame with at least ``term``,
            ``freq``, ``n_users``, ``g2``, and ``p_value`` columns.
        tokens_df: Token-level DataFrame used to derive the total user count.
        min_leaf_freq: Minimum absolute frequency for a collocation to be
            retained.

    Returns:
        A filtered and sorted :class:`~pandas.DataFrame` of leaf candidates.

    Raises:
        ValueError: If *collocations_df* is empty or missing required
            columns.
    """
    if collocations_df is None or collocations_df.empty:
        raise ValueError("collocations_df is empty or missing.")

    required_cols = {"term", "freq", "n_users", "g2", "p_value"}
    missing = required_cols.difference(collocations_df.columns)

    if missing:
        raise ValueError(f"collocations_df is missing required columns: {missing}")

    n_users_total = get_total_users(tokens_df)

    leaves = collocations_df.copy()

    if "n_posts" not in leaves.columns and "n_docs" in leaves.columns:
        leaves["n_posts"] = leaves["n_docs"]

    if "n_posts" not in leaves.columns:
        leaves["n_posts"] = pd.NA

    leaves["term_norm"] = leaves["term"].astype(str).str.lower().str.strip()
    leaves["term_parts"] = leaves["term_norm"].str.split()

    for c in ["freq", "n_posts", "n_users", "g2", "p_value"]:
        leaves[c] = pd.to_numeric(leaves[c], errors="coerce")

    leaves["relative_diffusion"] = leaves["n_users"] / n_users_total

    leaves = leaves.loc[
        leaves["term_norm"].ne("")
        & leaves["freq"].ge(min_leaf_freq)
        & leaves["n_users"].notna()
    ].copy()

    return (
        leaves.sort_values(
            ["g2", "freq", "n_users", "n_posts"],
            ascending=[False, False, False, False],
        )
        .reset_index(drop=True)
    )


def match_collocations_containing_root(
    roots_df,
    leaves_df,
    max_leaves_per_root=MAX_LEAVES_PER_ROOT,
):
    """Match collocations that contain each root term as a constituent token.

    For every root the function scans ``leaves_df`` for collocations whose
    tokenised form includes the root (exact token match), then keeps the top
    *max_leaves_per_root* by G².

    Args:
        roots_df: Selected root terms with a ``term_norm`` column.
        leaves_df: Candidate collocation leaves prepared by
            :func:`prepare_collocation_leaves`.
        max_leaves_per_root: Maximum number of leaves to attach per root.

    Returns:
        An edge :class:`~pandas.DataFrame` with one row per root–leaf pair.

    Raises:
        ValueError: If no collocations contain any of the selected roots.
    """
    records = []

    for _, root in roots_df.iterrows():
        root_term = str(root["term_norm"]).strip()

        matched = leaves_df.loc[
            leaves_df["term_parts"].map(
                lambda parts: isinstance(parts, list) and root_term in parts
            )
        ].copy()

        if matched.empty:
            continue

        matched = (
            matched.sort_values(
                ["g2", "freq", "n_users", "n_posts"],
                ascending=[False, False, False, False],
            )
            .head(max_leaves_per_root)
        )

        for _, leaf in matched.iterrows():
            records.append(
                {
                    "root": root["term"],
                    "root_norm": root["term_norm"],
                    "root_upos": root["upos"],
                    "root_quadrant": root["quadrant"],
                    "root_rank": root["Rank"],
                    "root_abs_diffusion": root["frequency"],
                    "root_relative_diffusion": root["relative_diffusion"],
                    "leaf": leaf["term"],
                    "leaf_norm": leaf["term_norm"],
                    "leaf_abs_frequency": leaf["freq"],
                    "leaf_user_diffusion": leaf["n_users"],
                    "leaf_post_diffusion": leaf["n_posts"],
                    "leaf_relative_diffusion": leaf["relative_diffusion"],
                    "leaf_g2": leaf["g2"],
                    "leaf_p_value": leaf["p_value"],
                }
            )

    edges_df = pd.DataFrame(records)

    if edges_df.empty:
        raise ValueError("No collocations contain the selected roots exactly.")

    return edges_df


def build_graph(edges_df):
    """Build a :class:`networkx.Graph` from an edge DataFrame.

    Root nodes receive ``node_class="root"`` and leaf nodes receive
    ``node_class="collocation"``.  Edge weight is set to the leaf G²
    association strength.

    Args:
        edges_df: Edge DataFrame produced by
            :func:`match_collocations_containing_root`.

    Returns:
        An undirected :class:`networkx.Graph`.

    Raises:
        ImportError: If *networkx* is not installed.
    """
    if nx is None:
        raise ImportError("Install networkx to use semantic tree plots.")

    G = nx.Graph()

    for edge_idx, r in edges_df.reset_index(drop=True).iterrows():
        root_id = f"root::{r['root_norm']}"

        # Root-specific leaf identifier.
        # This duplicates the same collocation label when it is attached to
        # different roots. It is essential for NOUN_NOUN collocations, where
        # the same collocation may legitimately contain two selected roots.
        leaf_id = f"colloc::{r['root_norm']}::{r['leaf_norm']}::{edge_idx}"

        G.add_node(
            root_id,
            label=r["root"],
            node_class="root",
            upos=r["root_upos"],
            quadrant=r["root_quadrant"],
            rank=r["root_rank"],
            abs_diffusion=r["root_abs_diffusion"],
            relative_diffusion=r["root_relative_diffusion"],
        )

        G.add_node(
            leaf_id,
            label=r["leaf"],
            node_class="collocation",
            abs_frequency=r["leaf_abs_frequency"],
            user_diffusion=r["leaf_user_diffusion"],
            post_diffusion=r["leaf_post_diffusion"],
            relative_diffusion=r["leaf_relative_diffusion"],
            g2=r["leaf_g2"],
            p_value=r["leaf_p_value"],
            parent_root=r["root"],
            parent_root_norm=r["root_norm"],
        )

        G.add_edge(root_id, leaf_id, weight=r["leaf_g2"])

    return G

def _roots_per_quadrant(G):
    root_nodes = [n for n, d in G.nodes(data=True) if d.get("node_class") == "root"]

    return {
        q: [n for n in root_nodes if G.nodes[n].get("quadrant") == q]
        for q in QUADRANT_ROW_ORDER
    }


def compute_row_geometry(G, row_height=ROW_HEIGHT):
    """Compute vertical layout geometry for each quadrant row.

    Peripheral system rows may span multiple lines when there are more roots
    than ``ROOTS_PER_LINE``.  All other quadrants occupy a single row.

    Args:
        G: The semantic tree :class:`networkx.Graph`.
        row_height: Base height (in data units) for a single-line row.

    Returns:
        A tuple ``(quadrant_y, row_top, row_bottom, line_counts,
        row_heights)`` where each is a dict keyed by quadrant name.
    """
    roots_by_q = _roots_per_quadrant(G)

    line_counts = {}
    for q in QUADRANT_ROW_ORDER:
        n_roots = len(roots_by_q.get(q, []))
        if q == "Peripheral system" and PERIPHERAL_EXTRA_ROOT_LINES:
            line_counts[q] = max(1, int(np.ceil(n_roots / ROOTS_PER_LINE)))
        else:
            line_counts[q] = 1

    row_heights = {}
    for q in QUADRANT_ROW_ORDER:
        extra = max(0, line_counts[q] - 1) * PERIPHERAL_LINE_GAP
        row_heights[q] = row_height + extra

    row_top = {}
    row_bottom = {}
    quadrant_y = {}

    cursor_top = 0.0
    for q in QUADRANT_ROW_ORDER:
        row_top[q] = cursor_top
        row_bottom[q] = cursor_top - row_heights[q]
        quadrant_y[q] = row_top[q] - ROW_TOP_PAD
        cursor_top = row_bottom[q]

    return quadrant_y, row_top, row_bottom, line_counts, row_heights


def compute_fixed_quadrant_tree_layout(
    G,
    seed=LAYOUT_SEED,
    root_x_start=ROOT_X_START,
    root_x_step=ROOT_X_STEP,
    leaf_drop=LEAF_VERTICAL_DROP,
    leaf_row_stagger=LEAF_ROW_STAGGER,
    leaf_spread=LEAF_HORIZONTAL_SPREAD,
):
    """Assign fixed (x, y) positions to every node in the semantic tree.

    Roots are placed on a regular grid within their quadrant row.  Leaves
    are fanned out below their parent root with a small stagger to reduce
    overlap.  A seeded RNG adds tiny horizontal jitter to leaf positions.

    Args:
        G: The semantic tree :class:`networkx.Graph`.
        seed: Integer seed for reproducible jitter.
        root_x_start: X coordinate of the first root in each row.
        root_x_step: Horizontal step between consecutive roots.
        leaf_drop: Vertical distance from root to first leaf row.
        leaf_row_stagger: Additional vertical stagger between leaf rows.
        leaf_spread: Half-width of the horizontal fan of leaves.

    Returns:
        A tuple ``(pos, quadrant_y, row_top, row_bottom, line_counts,
        row_heights)`` where *pos* maps node IDs to ``(x, y)`` tuples.
    """
    rng = np.random.default_rng(seed)
    pos = {}

    root_nodes = [n for n, d in G.nodes(data=True) if d.get("node_class") == "root"]

    quadrant_y, row_top, row_bottom, line_counts, row_heights = compute_row_geometry(G)

    for q in QUADRANT_ROW_ORDER:
        roots_q = [n for n in root_nodes if G.nodes[n].get("quadrant") == q]

        if not roots_q:
            continue

        roots_q = sorted(
            roots_q,
            key=lambda n: (
                -float(G.nodes[n].get("relative_diffusion", 0) or 0),
                float(G.nodes[n].get("rank", 999) or 999),
            ),
        )

        if q == "Peripheral system" and PERIPHERAL_EXTRA_ROOT_LINES:
            roots_per_line = ROOTS_PER_LINE
        else:
            roots_per_line = len(roots_q)

        for j, root in enumerate(roots_q):
            line_id = j // roots_per_line
            col_id = j % roots_per_line

            x = root_x_start + col_id * root_x_step
            y_root = quadrant_y[q] - line_id * PERIPHERAL_LINE_GAP
            pos[root] = (x, y_root)

            leaves = [
                v for v in G.neighbors(root)
                if G.nodes[v].get("node_class") == "collocation"
            ]

            leaves = sorted(
                leaves,
                key=lambda v: float(G.nodes[v].get("g2", 0) or 0),
                reverse=True,
            )

            if len(leaves) == 1:
                leaf_xs = [x]
            else:
                leaf_xs = np.linspace(x - leaf_spread, x + leaf_spread, len(leaves))

            for i, leaf in enumerate(leaves):
                lx = leaf_xs[i] + rng.normal(0, 0.025)
                ly = y_root - leaf_drop - leaf_row_stagger * (i % 3)
                ly = max(ly, row_bottom[q] + ROW_BOTTOM_PAD)

                if leaf not in pos:
                    pos[leaf] = (lx, ly)
                else:
                    old_x, old_y = pos[leaf]
                    pos[leaf] = ((old_x + lx) / 2, (old_y + ly) / 2)

    return pos, quadrant_y, row_top, row_bottom, line_counts, row_heights


def add_constant_quadrant_legend_traces(fig):
    """Add one invisible legend scatter trace per quadrant to *fig*.

    Each trace uses ``x=[None], y=[None]`` so it contributes only a legend
    entry without any visible data points.

    Args:
        fig: A :class:`plotly.graph_objects.Figure` to modify in-place.
    """
    for q in QUADRANT_ROW_ORDER:
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                marker=dict(
                    size=14,
                    color=QUADRANT_COLOURS[q],
                    line=dict(width=1.0, color="white"),
                    opacity=0.96,
                ),
                name=q,
                showlegend=True,
                hoverinfo="skip",
            )
        )


def build_plotly_tree(
    G,
    pos,
    quadrant_y,
    row_top,
    row_bottom,
    output_html,
    output_png=None,
    width=TREE_FIG_WIDTH,
    height=TREE_FIG_HEIGHT,
    png_scale=PNG_SCALE,
):
    """Render the semantic tree as an interactive Plotly figure.

    Draws curved edges, root scatter markers (colour-coded and size-scaled
    by relative diffusion), collocation leaf markers, quadrant labels,
    row separators, and a legend.  Saves an HTML file and optionally a PNG
    (requires kaleido).

    Args:
        G: The semantic tree :class:`networkx.Graph`.
        pos: Node position dict from :func:`compute_fixed_quadrant_tree_layout`.
        quadrant_y: Quadrant label y-positions dict.
        row_top: Top y-coordinate per quadrant dict.
        row_bottom: Bottom y-coordinate per quadrant dict.
        output_html: Path for the output HTML file.
        output_png: Optional path for the output PNG file.
        width: Figure width in pixels.
        height: Figure height in pixels.
        png_scale: Resolution scale factor for PNG export.

    Returns:
        The :class:`plotly.graph_objects.Figure`.

    Raises:
        ImportError: If *plotly* is not installed.
    """
    if go is None:
        raise ImportError("Install plotly to use semantic tree plots.")

    fig = go.Figure()
    xgeo = effective_x_geometry(pos)

    edge_weights = [float(d.get("weight", 1) or 1) for _, _, d in G.edges(data=True)]
    w_min = min(edge_weights) if edge_weights else 1
    w_max = max(edge_weights) if edge_weights else 1

    for u, v, d in G.edges(data=True):
        x0, y0 = pos[u]
        x1, y1 = pos[v]

        if w_max > w_min:
            width_i = EDGE_WIDTH_MIN + (EDGE_WIDTH_MAX - EDGE_WIDTH_MIN) * (
                float(d.get("weight", w_min)) - w_min
            ) / (w_max - w_min)
        else:
            width_i = 1.2

        cx, cy = curved_edge_points(x0, y0, x1, y1)

        fig.add_trace(
            go.Scatter(
                x=cx,
                y=cy,
                mode="lines",
                line=dict(width=width_i, color="rgba(110,110,110,0.28)"),
                hoverinfo="skip",
                showlegend=False,
            )
        )

    root_nodes = [n for n, d in G.nodes(data=True) if d.get("node_class") == "root"]
    leaf_nodes = [
        n for n, d in G.nodes(data=True) if d.get("node_class") == "collocation"
    ]

    root_diff = [G.nodes[n].get("relative_diffusion", np.nan) for n in root_nodes]
    leaf_diff = [G.nodes[n].get("relative_diffusion", np.nan) for n in leaf_nodes]

    root_size_map = dict(zip(root_nodes, rescale(root_diff, ROOT_SIZE_MIN, ROOT_SIZE_MAX)))
    leaf_size_map = dict(zip(leaf_nodes, rescale(leaf_diff, LEAF_SIZE_MIN, LEAF_SIZE_MAX)))

    all_root_ranks = pd.Series(
        [G.nodes[n].get("rank", np.nan) for n in root_nodes],
        dtype="float",
    )
    rank_min = all_root_ranks.min()
    rank_max = all_root_ranks.max()

    for q in QUADRANT_ROW_ORDER:
        nodes = [n for n in root_nodes if G.nodes[n].get("quadrant") == q]

        if not nodes:
            continue

        colours = []
        for n in nodes:
            rank = float(G.nodes[n].get("rank", np.nan))

            if pd.isna(rank) or rank_min == rank_max:
                lighten = 0.25
            else:
                salience = 1 - (rank - rank_min) / (rank_max - rank_min)
                lighten = 0.55 - 0.45 * salience

            colours.append(blend_with_white(QUADRANT_COLOURS[q], lighten))

        customdata = [
            [
                G.nodes[n].get("upos"),
                G.nodes[n].get("quadrant"),
                format_float(G.nodes[n].get("rank"), 2),
                format_int(G.nodes[n].get("abs_diffusion")),
                format_float(G.nodes[n].get("relative_diffusion"), 4),
            ]
            for n in nodes
        ]

        fig.add_trace(
            go.Scatter(
                x=[pos[n][0] for n in nodes],
                y=[pos[n][1] for n in nodes],
                mode="markers+text",
                text=[
                    f"<b>{shorten_label(G.nodes[n]['label'], ROOT_LABEL_MAX_CHARS)}</b>"
                    for n in nodes
                ],
                textposition="top center",
                textfont=dict(size=12, color="#111111"),
                marker=dict(
                    size=[root_size_map[n] for n in nodes],
                    color=colours,
                    line=dict(width=1.0, color="white"),
                    opacity=0.96,
                ),
                name=q,
                showlegend=False,
                customdata=customdata,
                hovertemplate=(
                    "<b>%{text}</b><br>"
                    "UPOS: %{customdata[0]}<br>"
                    "Quadrant: %{customdata[1]}<br>"
                    "Rank: %{customdata[2]}<br>"
                    "Absolute user diffusion: %{customdata[3]}<br>"
                    "Relative user diffusion: %{customdata[4]}<br>"
                    "<extra></extra>"
                ),
            )
        )

    if leaf_nodes:
        fig.add_trace(
            go.Scatter(
                x=[pos[n][0] for n in leaf_nodes],
                y=[pos[n][1] for n in leaf_nodes],
                mode="markers+text",
                text=[
                    f"<i>{shorten_label(G.nodes[n]['label'], LEAF_LABEL_MAX_CHARS)}</i>"
                    for n in leaf_nodes
                ],
                textposition="bottom center",
                textfont=dict(size=14, color="#222222"),
                marker=dict(
                    size=[leaf_size_map[n] for n in leaf_nodes],
                    color="rgba(112,128,144,0.82)",
                    line=dict(width=0.7, color="white"),
                    opacity=0.90,
                ),
                showlegend=False,
                customdata=[
                    [
                        format_int(G.nodes[n].get("abs_frequency")),
                        format_int(G.nodes[n].get("user_diffusion")),
                        format_int(G.nodes[n].get("post_diffusion")),
                        format_float(G.nodes[n].get("relative_diffusion"), 4),
                        format_float(G.nodes[n].get("g2"), 2),
                        format_p_value(G.nodes[n].get("p_value")),
                    ]
                    for n in leaf_nodes
                ],
                hovertemplate=(
                    "<b>%{text}</b><br>"
                    "Absolute frequency: %{customdata[0]}<br>"
                    "User diffusion: %{customdata[1]}<br>"
                    "Post diffusion: %{customdata[2]}<br>"
                    "Relative user diffusion: %{customdata[3]}<br>"
                    "G²: %{customdata[4]}<br>"
                    "p-value: %{customdata[5]}<br>"
                    "<extra></extra>"
                ),
            )
        )

    for q in QUADRANT_ROW_ORDER:
        y_center = (row_top[q] + row_bottom[q]) / 2

        fig.add_annotation(
            x=LEFT_LABEL_X,
            y=y_center,
            text=f"<b>{q}</b>",
            textangle=-90,
            showarrow=False,
            xanchor="center",
            yanchor="middle",
            align="center",
            font=dict(size=18, color=QUADRANT_COLOURS[q]),
        )

        fig.add_shape(
            type="line",
            x0=LEFT_LINE_X,
            x1=LEFT_LINE_X,
            y0=row_top[q] - 0.18,
            y1=row_bottom[q] + 0.18,
            line=dict(color=QUADRANT_COLOURS[q], width=LEFT_LINE_WIDTH),
            layer="below",
        )

    for i in range(len(QUADRANT_ROW_ORDER) - 1):
        q_upper = QUADRANT_ROW_ORDER[i]
        y_sep = row_bottom[q_upper] + ROW_SEPARATOR_SHIFT

        fig.add_shape(
            type="line",
            x0=xgeo["separator_x0"],
            x1=xgeo["separator_x1"],
            y0=y_sep,
            y1=y_sep,
            line=dict(color=ROW_SEPARATOR_COLOUR, width=ROW_SEPARATOR_WIDTH),
            layer="below",
        )

    add_constant_quadrant_legend_traces(fig)

    fig.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode="markers",
            marker=dict(size=14, color="rgba(120,120,120,0.85)"),
            name="Node size = rel. diffusion",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode="markers",
            marker=dict(size=14, color="rgb(65,90,120)"),
            name="Node shade = salience",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=[None, None],
            y=[None, None],
            mode="lines",
            line=dict(width=3, color="rgba(110,110,110,0.50)"),
            name="Edge width = G² ass. strength",
        )
    )

    y_min = row_bottom[QUADRANT_ROW_ORDER[-1]] - 0.22
    y_max = row_top[QUADRANT_ROW_ORDER[0]] + 0.22

    fig.update_layout(
        width=width,
        height=height,
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=4, r=4, t=6, b=25),
        legend=dict(
            orientation="h",
            x=xgeo["legend_x"],
            y=0.005,
            xanchor="center",
            yanchor="top",
            font=dict(size=15),
            itemsizing="constant",
            bgcolor="rgba(255,255,255,0)",
        ),
        xaxis=dict(visible=False, range=[xgeo["x_range_min"], xgeo["x_range_max"]]),
        yaxis=dict(visible=False, range=[y_min, y_max]),
    )

    fig.write_html(output_html, include_plotlyjs="cdn")

    if output_png is not None:
        try:
            fig.write_image(output_png, width=width, height=height, scale=png_scale)
            print(f"Saved: {output_png}")
        except Exception as e:
            print("PNG export skipped. Install or update kaleido if static export is required.")
            print(f"Reason: {e}")

    print(f"Saved: {output_html}")

    return fig


def build_evoc_collocation_tree_for_upos(
    evoc_quadrants_df,
    collocations_df,
    tokens_df,
    upos,
    output_dir=OUTPUT_DIR,
    top_roots_per_quadrant=TOP_ROOTS_PER_QUADRANT,
    max_leaves_per_root=MAX_LEAVES_PER_ROOT,
    min_leaf_freq=MIN_LEAF_FREQ,
    seed=LAYOUT_SEED,
    width=TREE_FIG_WIDTH,
    height=TREE_FIG_HEIGHT,
    png_scale=PNG_SCALE,
    show_diagnostics=False,
):
    """Build and save the full EVOC semantic collocation tree for a UPOS tag.

    Orchestrates root selection, leaf preparation, graph construction,
    layout, and Plotly rendering.  Saves both HTML and PNG outputs to
    *output_dir*.

    Args:
        evoc_quadrants_df: EVOC quadrants DataFrame.
        collocations_df: Raw collocation DataFrame.
        tokens_df: Token-level DataFrame used for user counts.
        upos: Universal POS tag to visualise (e.g. ``"NOUN"``).
        output_dir: Directory for saved figures.
        top_roots_per_quadrant: Maximum root terms per quadrant.
        max_leaves_per_root: Maximum collocation leaves per root.
        min_leaf_freq: Minimum frequency threshold for leaf collocations.
        seed: Layout RNG seed.
        width: Figure width in pixels.
        height: Figure height in pixels.
        png_scale: Resolution scale factor for PNG export.
        show_diagnostics: If ``True``, display the diagnostics DataFrame
            in an interactive environment (uses ``display`` if available,
            otherwise ``print``).

    Returns:
        A :class:`dict` with keys ``fig``, ``graph``, ``positions``,
        ``quadrant_y``, ``row_top``, ``row_bottom``, ``line_counts``,
        ``row_heights``, ``roots_df``, ``leaves_df``, ``edges_df``,
        ``diagnostics_df``, ``html``, and ``png``.
    """
    output_dir = _ensure_output_dir(output_dir)

    roots_df = select_roots_for_upos(
        evoc_quadrants_df=evoc_quadrants_df,
        tokens_df=tokens_df,
        upos=upos,
        top_roots_per_quadrant=top_roots_per_quadrant,
    )

    leaves_df = prepare_collocation_leaves(
        collocations_df=collocations_df,
        tokens_df=tokens_df,
        min_leaf_freq=min_leaf_freq,
    )

    edges_df = match_collocations_containing_root(
        roots_df=roots_df,
        leaves_df=leaves_df,
        max_leaves_per_root=max_leaves_per_root,
    )

    G = build_graph(edges_df)

    pos, quadrant_y, row_top, row_bottom, line_counts, row_heights = (
        compute_fixed_quadrant_tree_layout(G, seed=seed)
    )

    output_html = os.path.join(output_dir, f"evoc_collocation_tree_{upos.lower()}_roots.html")
    output_png = os.path.join(output_dir, f"evoc_collocation_tree_{upos.lower()}_roots.png")

    fig = build_plotly_tree(
        G=G,
        pos=pos,
        quadrant_y=quadrant_y,
        row_top=row_top,
        row_bottom=row_bottom,
        output_html=output_html,
        output_png=output_png,
        width=width,
        height=height,
        png_scale=png_scale,
    )

    diagnostics_df = pd.DataFrame(
        {
            "object": [
                "upos",
                "selected_roots",
                "candidate_collocations",
                "matched_edges",
                "graph_nodes",
                "graph_edges",
                "html_output",
                "png_output",
            ],
            "value": [
                upos,
                len(roots_df),
                len(leaves_df),
                len(edges_df),
                G.number_of_nodes(),
                G.number_of_edges(),
                output_html,
                output_png,
            ],
        }
    )

    if show_diagnostics:
        try:
            display(diagnostics_df)  # type: ignore[name-defined]
        except NameError:
            print(diagnostics_df)

    return {
        "fig": fig,
        "graph": G,
        "positions": pos,
        "quadrant_y": quadrant_y,
        "row_top": row_top,
        "row_bottom": row_bottom,
        "line_counts": line_counts,
        "row_heights": row_heights,
        "roots_df": roots_df,
        "leaves_df": leaves_df,
        "edges_df": edges_df,
        "diagnostics_df": diagnostics_df,
        "html": output_html,
        "png": output_png,
    }


# =============================================================================
# 3. EMOJI EVOC PLOTLY MAP
# =============================================================================

EMOJI_HTML_FILE = "evoc_outputs/evoc_quadrants_emoji_compact.html"
EMOJI_OUTPUT_HTML = "emoji_evoc_plotly_map.html"
EMOJI_OUTPUT_PNG = "emoji_evoc_plotly_map.png"

EMOJI_FIG_WIDTH = 1050
EMOJI_FIG_HEIGHT = 760

EMOJI_FONT_SIZE = 16
LABEL_FONT_SIZE = 9
MAX_LABEL_CHARS = 30

TOP_N_PER_QUADRANT = 6

# Fallback ranges used only when dynamic ranges cannot be computed.
X_RANGE = [-0.15, 5.00]
Y_RANGE = [-0.15, 2.00]

EMOJI_AXIS_PAD_SALIENCE = 0.15
EMOJI_AXIS_PAD_DIFFUSION = 0.15


def parse_emoji_evoc_html(html_file):
    """Parse an EVOC compact HTML report and extract emoji quadrant data.

    Reads ``.quad-card`` elements from the HTML, extracting one row per
    emoji entry with relative diffusion, frequency, rank, description, and
    quadrant.

    Args:
        html_file: Path to the compact EVOC HTML report.

    Returns:
        A :class:`~pandas.DataFrame` with columns ``emoji``,
        ``relative_diffusion``, ``frequency``, ``rank``, ``description``,
        ``quadrant``, and ``salience``.

    Raises:
        ImportError: If *beautifulsoup4* is not installed.
        ValueError: If no emoji records are found in the file.
    """
    if BeautifulSoup is None:
        raise ImportError("Install beautifulsoup4 to parse emoji EVOC HTML reports.")

    with open(html_file, "r", encoding="utf-8") as f:
        soup = BeautifulSoup(f.read(), "html.parser")

    records = []

    for card in soup.select(".quad-card"):
        title_node = card.select_one(".quad-title div")

        if title_node is None:
            continue

        quadrant = title_node.get_text(strip=True)

        for row in card.select("tbody tr"):
            cells = row.select("td")

            if len(cells) < 5:
                continue

            try:
                rel_diff = float(cells[1].get_text(strip=True))
                frequency = int(cells[2].get_text(strip=True).replace(",", ""))
                rank = float(cells[3].get_text(strip=True))
            except ValueError:
                continue

            records.append(
                {
                    "emoji": cells[0].get_text(strip=True),
                    "relative_diffusion": rel_diff,
                    "frequency": frequency,
                    "rank": rank,
                    "description": cells[4].get_text(strip=True),
                    "quadrant": quadrant,
                }
            )

    df = pd.DataFrame(records)

    if df.empty:
        raise ValueError("No emoji records were extracted from the HTML file.")

    df["quadrant"] = pd.Categorical(df["quadrant"], categories=QUADRANT_ORDER, ordered=True)
    df["salience"] = df["rank"]

    return df


def extract_thresholds_from_html(html_file):
    """Extract AFE and AOE threshold values from an EVOC HTML report.

    Uses regex to locate the ``Rel. diff. ≥ X`` and ``AOE ≤ X`` patterns
    embedded in the report text.

    Args:
        html_file: Path to the EVOC HTML report.

    Returns:
        A tuple ``(afe, aoe)`` of :class:`float` or ``None`` values.
        A value is ``None`` when the corresponding pattern is not found.
    """
    with open(html_file, "r", encoding="utf-8") as f:
        txt = f.read()

    afe_match = re.search(r"Rel\. diff\. ≥ ([0-9.]+)", txt)
    aoe_match = re.search(r"AOE ≤ ([0-9.]+)", txt)

    afe = float(afe_match.group(1)) if afe_match else None
    aoe = float(aoe_match.group(1)) if aoe_match else None

    return afe, aoe



def compute_emoji_axis_ranges(
    df,
    *,
    salience_col="salience",
    diffusion_col="relative_diffusion",
    pad_salience=EMOJI_AXIS_PAD_SALIENCE,
    pad_diffusion=EMOJI_AXIS_PAD_DIFFUSION,
    fallback_x_range=X_RANGE,
    fallback_y_range=Y_RANGE,
):
    """Compute emoji-map axis ranges from parsed EVOC report values.

    The emoji map is built from the compact HTML report, not directly from the
    EVOC dataframe. After parsing, the available empirical variables are
    ``salience`` (currently the Rank/AOE proxy) and ``relative_diffusion``.
    The returned ranges are therefore ``min - pad`` and ``max + pad`` on these
    parsed values.
    """

    if df is None or df.empty:
        return list(fallback_x_range), list(fallback_y_range)

    salience = pd.to_numeric(df.get(salience_col), errors="coerce").dropna()
    diffusion = pd.to_numeric(df.get(diffusion_col), errors="coerce").dropna()

    if salience.empty or diffusion.empty:
        return list(fallback_x_range), list(fallback_y_range)

    sx_min = float(salience.min())
    sx_max = float(salience.max())
    dy_min = float(diffusion.min())
    dy_max = float(diffusion.max())

    if sx_min == sx_max:
        sx_min -= pad_salience
        sx_max += pad_salience
    else:
        sx_min -= pad_salience
        sx_max += pad_salience

    if dy_min == dy_max:
        dy_min -= pad_diffusion
        dy_max += pad_diffusion
    else:
        dy_min -= pad_diffusion
        dy_max += pad_diffusion

    return [sx_min, sx_max], [dy_min, dy_max]


def select_emojis_for_plot(df, top_n_per_quadrant=TOP_N_PER_QUADRANT):
    """Select the top *n* emojis per quadrant for the Plotly scatter map.

    Uses the same quadrant-specific ranking logic as
    :func:`select_terms_for_target` (diffusion / rank / frequency priority
    varies by quadrant).

    Args:
        df: Parsed emoji DataFrame from :func:`parse_emoji_evoc_html`.
        top_n_per_quadrant: Maximum emojis to keep per quadrant.
            Pass ``None`` to keep all.

    Returns:
        A concatenated :class:`~pandas.DataFrame` of selected emojis.

    Raises:
        ValueError: If no emojis are selected.
    """
    selected = []

    for q in QUADRANT_ORDER:
        sub = df[df["quadrant"].astype(str).eq(q)].copy()

        if sub.empty:
            continue

        if q in {"Central nucleus", "First periphery"}:
            sub = sub.sort_values(
                ["relative_diffusion", "rank", "frequency"],
                ascending=[False, True, False],
            )

        elif q == "Contrast zone":
            sub = sub.sort_values(
                ["rank", "relative_diffusion", "frequency"],
                ascending=[False, False, False],
            )

        else:
            sub = sub.sort_values(
                ["relative_diffusion", "rank", "frequency"],
                ascending=[False, False, False],
            )

        if top_n_per_quadrant is not None:
            sub = sub.head(top_n_per_quadrant)

        selected.append(sub)

    if not selected:
        raise ValueError("No emojis selected for plotting.")

    return pd.concat(selected, ignore_index=True)


def shorten_emoji_label(x, width=MAX_LABEL_CHARS):
    """Shorten an emoji description to *width* characters using word boundaries.

    Delegates to :func:`textwrap.shorten`, which breaks at word boundaries
    and appends ``…``.

    Args:
        x: Description string.
        width: Maximum character width.

    Returns:
        Shortened string.
    """
    x = str(x).strip()

    if len(x) <= width:
        return x

    return textwrap.shorten(x, width=width, placeholder="…")


def data_offset_from_pixels(
    dx_px,
    dy_px,
    *,
    x_range=None,
    y_range=None,
    width=EMOJI_FIG_WIDTH,
    height=EMOJI_FIG_HEIGHT,
):
    """Convert pixel offsets to data-coordinate offsets for the emoji map.

    Accounts for the fixed margins baked into the emoji figure layout.
    Note that *dy_px* is negated because screen y increases downward while
    data y increases upward.

    Args:
        dx_px: Horizontal pixel offset.
        dy_px: Vertical pixel offset (positive = upward in data space).
        x_range: ``[x_min, x_max]`` axis range; defaults to ``X_RANGE``.
        y_range: ``[y_min, y_max]`` axis range; defaults to ``Y_RANGE``.
        width: Total figure width in pixels.
        height: Total figure height in pixels.

    Returns:
        A tuple ``(dx_data, dy_data)`` in data coordinates.
    """
    x_range = list(x_range or X_RANGE)
    y_range = list(y_range or Y_RANGE)

    x_span = x_range[1] - x_range[0]
    y_span = y_range[1] - y_range[0]

    plot_width = max(width - 220, 1)
    plot_height = max(height - 170, 1)

    dx_data = dx_px / plot_width * x_span
    dy_data = -dy_px / plot_height * y_span

    return dx_data, dy_data


def label_offsets_for_quadrant(quadrant, i):
    """Return pixel offsets for placing a text label next to an emoji point.

    Offsets are quadrant-specific (left/right of the point, above/below)
    and are staggered by *i* to reduce vertical overlap between nearby
    labels.

    Args:
        quadrant: Quadrant name string.
        i: Zero-based position index of the emoji within its quadrant.

    Returns:
        A tuple ``(base_dx, base_dy)`` of pixel offsets.
    """
    if quadrant == "Central nucleus":
        base_dx = -70
        base_dy = -10

    elif quadrant == "First periphery":
        base_dx = 70
        base_dy = -10

    elif quadrant == "Contrast zone":
        base_dx = -70
        base_dy = 12

    else:
        base_dx = 70
        base_dy = 12

    stagger = [-24, -12, 0, 12, 24, 36, -36, 48, -48, 60]

    return base_dx, base_dy + stagger[i % len(stagger)]


def repel_labels_vertically(df, min_gap=0.045, y_min=None, y_max=None, y_range=None):
    """Push overlapping label y-positions apart within each quadrant side.

    Labels on the same side (left or right of their data point) are sorted
    by y, then shifted upward until each pair is at least *min_gap* apart.
    The block is then clamped to ``[y_min, y_max]``.

    Args:
        df: DataFrame with ``quadrant``, ``label_x``, ``label_y``, and
            ``salience`` columns.
        min_gap: Minimum vertical gap between adjacent labels in data units.
        y_min: Lower clamp bound; derived from *y_range* if not given.
        y_max: Upper clamp bound; derived from *y_range* if not given.
        y_range: ``[y_min, y_max]`` axis range used to set bounds when
            *y_min* / *y_max* are ``None``.

    Returns:
        A copy of *df* with adjusted ``label_y`` values.
    """
    out = df.copy()

    if y_range is not None:
        y_min = y_range[0] + 0.05 if y_min is None else y_min
        y_max = y_range[1] - 0.05 if y_max is None else y_max

    if y_min is None:
        y_min = -0.10

    if y_max is None:
        y_max = 0.96

    for q in QUADRANT_ORDER:
        sub_q = out[out["quadrant"].astype(str).eq(q)].copy()

        if sub_q.empty:
            continue

        sub_q["label_side"] = np.where(
            sub_q["label_x"] >= sub_q["salience"],
            "right",
            "left",
        )

        for side in ["left", "right"]:
            sub = sub_q[sub_q["label_side"].eq(side)].copy()

            if len(sub) <= 1:
                continue

            sub = sub.sort_values("label_y").copy()
            ys = sub["label_y"].to_numpy(dtype=float)

            for i in range(1, len(ys)):
                if ys[i] - ys[i - 1] < min_gap:
                    ys[i] = ys[i - 1] + min_gap

            if ys[-1] > y_max:
                ys = ys - (ys[-1] - y_max)

            if ys[0] < y_min:
                ys = ys + (y_min - ys[0])

            out.loc[sub.index, "label_y"] = ys

    return out


def compute_label_positions(
    df,
    *,
    x_range=None,
    y_range=None,
    width=EMOJI_FIG_WIDTH,
    height=EMOJI_FIG_HEIGHT,
):
    """Compute final label coordinates for every emoji in the map.

    For each quadrant the emojis are iterated in sort order; pixel offsets
    are converted to data-space via :func:`data_offset_from_pixels`,
    clamped to the axis range, and then vertically repelled by
    :func:`repel_labels_vertically`.

    Args:
        df: Selected emoji DataFrame with ``quadrant``, ``salience``,
            ``relative_diffusion``, and ``frequency`` columns.
        x_range: ``[x_min, x_max]`` axis range; defaults to ``X_RANGE``.
        y_range: ``[y_min, y_max]`` axis range; defaults to ``Y_RANGE``.
        width: Figure width in pixels.
        height: Figure height in pixels.

    Returns:
        A copy of *df* with ``label_x`` and ``label_y`` columns added.
    """
    x_range = list(x_range or X_RANGE)
    y_range = list(y_range or Y_RANGE)

    out = df.copy()
    out["label_x"] = np.nan
    out["label_y"] = np.nan

    for q in QUADRANT_ORDER:
        sub = out[out["quadrant"].astype(str).eq(q)].copy()

        if sub.empty:
            continue

        sub = sub.sort_values(
            ["relative_diffusion", "salience", "frequency"],
            ascending=[False, True, False],
        ).reset_index()

        for i, row in sub.iterrows():
            dx_px, dy_px = label_offsets_for_quadrant(q, i)

            dx_data, dy_data = data_offset_from_pixels(
                dx_px,
                dy_px,
                x_range=x_range,
                y_range=y_range,
                width=width,
                height=height,
            )

            x_lab = row["salience"] + dx_data
            y_lab = row["relative_diffusion"] + dy_data

            x_lab = min(max(x_lab, x_range[0] + 0.05), x_range[1] - 0.05)
            y_lab = min(max(y_lab, y_range[0] + 0.05), y_range[1] - 0.05)

            out.loc[row["index"], "label_x"] = x_lab
            out.loc[row["index"], "label_y"] = y_lab

    out = repel_labels_vertically(out, y_range=y_range)

    return out


def add_quadrant_labels(fig):
    """Add corner annotations naming each quadrant to *fig*.

    Labels are placed in paper coordinates so they stay at the corners
    regardless of axis zoom.  Each label is styled with the corresponding
    quadrant colour and a semi-transparent white background.

    Args:
        fig: A :class:`plotly.graph_objects.Figure` to modify in-place.
    """
    labels = [
        ("Central nucleus", 0.015, 0.985, "left", "top"),
        ("First periphery", 0.985, 0.985, "right", "top"),
        ("Contrast zone", 0.015, 0.035, "left", "bottom"),
        ("Peripheral system", 0.985, 0.035, "right", "bottom"),
    ]

    for label, x, y, xanchor, yanchor in labels:
        fig.add_annotation(
            x=x,
            y=y,
            xref="paper",
            yref="paper",
            text=f"<b>{label}</b>",
            showarrow=False,
            font=dict(size=10, color=QUADRANT_COLOURS[label]),
            xanchor=xanchor,
            yanchor=yanchor,
            align=xanchor,
            bgcolor="rgba(255,255,255,0.72)",
            bordercolor="rgba(210,210,210,0.60)",
            borderwidth=1,
            borderpad=4,
        )


def add_dashed_label_lines(fig, df):
    """Draw a dashed leader line from each emoji point to its text label.

    Args:
        fig: A :class:`plotly.graph_objects.Figure` to modify in-place.
        df: Emoji DataFrame with ``salience``, ``relative_diffusion``,
            ``label_x``, and ``label_y`` columns.
    """
    for _, r in df.iterrows():
        fig.add_shape(
            type="line",
            x0=r["salience"],
            y0=r["relative_diffusion"],
            x1=r["label_x"],
            y1=r["label_y"],
            xref="x",
            yref="y",
            line=dict(color="#888888", width=0.65, dash="dash"),
            layer="below",
        )


def add_italic_labels(fig, df):
    """Add italic description annotations at computed label positions.

    Each annotation is anchored left or right depending on whether the
    label sits to the right or left of its data point.

    Args:
        fig: A :class:`plotly.graph_objects.Figure` to modify in-place.
        df: Emoji DataFrame with ``label_x``, ``label_y``, ``salience``,
            and ``description`` columns.
    """
    for _, r in df.iterrows():
        xanchor = "left" if r["label_x"] >= r["salience"] else "right"

        fig.add_annotation(
            x=r["label_x"],
            y=r["label_y"],
            text=f"<i>{shorten_emoji_label(r['description'])}</i>",
            showarrow=False,
            font=dict(size=LABEL_FONT_SIZE, color="#222222"),
            xanchor=xanchor,
            yanchor="middle",
            align=xanchor,
            bgcolor="rgba(255,255,255,0)",
            bordercolor="rgba(255,255,255,0)",
            borderwidth=0,
            borderpad=0,
        )


def build_emoji_evoc_plot(
    html_file=EMOJI_HTML_FILE,
    output_dir=OUTPUT_DIR,
    output_html=EMOJI_OUTPUT_HTML,
    output_png=EMOJI_OUTPUT_PNG,
    top_n_per_quadrant=TOP_N_PER_QUADRANT,
    width=EMOJI_FIG_WIDTH,
    height=EMOJI_FIG_HEIGHT,
    pad_salience=EMOJI_AXIS_PAD_SALIENCE,
    pad_diffusion=EMOJI_AXIS_PAD_DIFFUSION,
    show=False,
):
    """Build and save the emoji EVOC Plotly scatter map.

    Parses an EVOC compact HTML report, selects the top emojis per
    quadrant, computes label positions, adds AFE/AOE threshold lines, and
    writes both an interactive HTML file and an optional PNG.

    Args:
        html_file: Path to the source EVOC compact HTML report.
        output_dir: Directory for saved figures.
        output_html: Filename (not path) for the output HTML file.
        output_png: Filename (not path) for the output PNG file, or
            ``None`` to skip PNG export.
        top_n_per_quadrant: Maximum emojis to display per quadrant.
        width: Figure width in pixels.
        height: Figure height in pixels.
        pad_salience: Axis padding added to the salience (x) range.
        pad_diffusion: Axis padding added to the diffusion (y) range.
        show: If ``True``, call ``fig.show()`` after saving.

    Returns:
        A tuple ``(fig, df, html_out)`` where *fig* is the Plotly figure,
        *df* is the plotted emoji DataFrame (with ``x_range`` and
        ``y_range`` stored in ``df.attrs``), and *html_out* is the path
        to the saved HTML file.

    Raises:
        ImportError: If *plotly* is not installed.
        ValueError: If AFE/AOE thresholds cannot be extracted from
            *html_file*.
    """
    if go is None:
        raise ImportError("Install plotly to use build_emoji_evoc_plot().")

    output_dir = _ensure_output_dir(output_dir)

    df_all = parse_emoji_evoc_html(html_file)
    df = select_emojis_for_plot(df_all, top_n_per_quadrant=top_n_per_quadrant)
    x_range, y_range = compute_emoji_axis_ranges(
        df,
        pad_salience=pad_salience,
        pad_diffusion=pad_diffusion,
    )

    df = compute_label_positions(
        df,
        x_range=x_range,
        y_range=y_range,
        width=width,
        height=height,
    )

    afe, aoe = extract_thresholds_from_html(html_file)

    if afe is None or aoe is None:
        raise ValueError("AFE/AOE thresholds could not be extracted from the HTML file.")

    salience_thr = aoe

    fig = go.Figure()

    add_dashed_label_lines(fig, df)

    for q in QUADRANT_ORDER:
        sub = df[df["quadrant"].astype(str).eq(q)].copy()

        if sub.empty:
            continue

        fig.add_trace(
            go.Scatter(
                x=sub["salience"],
                y=sub["relative_diffusion"],
                mode="text",
                text=sub["emoji"],
                textfont=dict(size=EMOJI_FONT_SIZE, color=QUADRANT_COLOURS[q]),
                name=q,
                showlegend=False,
                customdata=sub[["description", "frequency", "rank", "quadrant"]],
                hovertemplate=(
                    "<b>%{text}</b><br>"
                    "%{customdata[0]}<br>"
                    "Quadrant: %{customdata[3]}<br>"
                    "Rank: %{x:.2f}<br>"
                    "Relative diffusion: %{y:.2f}<br>"
                    "Abs. frequency: %{customdata[1]}"
                    "<extra></extra>"
                ),
            )
        )

    add_italic_labels(fig, df)

    fig.add_vline(x=salience_thr, line_width=1.2, line_dash="dash", line_color="#777777")
    fig.add_hline(y=afe, line_width=1.2, line_dash="dash", line_color="#777777")

    fig.add_annotation(
        x=salience_thr,
        xref="x",
        y=1.0,
        yref="paper",
        text=f"AOE = {aoe:.2f}",
        showarrow=False,
        yanchor="top",
        yshift=-6,
        xshift=8,
        font=dict(size=10, color="#555555"),
        bgcolor="rgba(255,255,255,0.85)",
    )

    fig.add_annotation(
        x=x_range[1] - 0.08,
        y=afe,
        text=f"AFE = {afe:.2f}",
        showarrow=False,
        xshift=-42,
        yshift=10,
        font=dict(size=10, color="#555555"),
        bgcolor="rgba(255,255,255,0.85)",
    )

    add_quadrant_labels(fig)

    fig.update_layout(
        width=width,
        height=height,
        title=None,
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=110, r=110, t=85, b=85),
        xaxis=dict(
            title="AOE (Rank)",
            range=x_range,
            zeroline=False,
            showline=True,
            linewidth=1,
            linecolor="#bbbbbb",
            mirror=True,
            gridcolor="#eeeeee",
            automargin=False,
        ),
        yaxis=dict(
            title="Relative diffusion",
            range=y_range,
            zeroline=False,
            showline=True,
            linewidth=1,
            linecolor="#bbbbbb",
            mirror=True,
            gridcolor="#eeeeee",
            automargin=False,
        ),
    )

    html_out = os.path.join(output_dir, output_html)
    png_out = os.path.join(output_dir, output_png) if output_png else None

    fig.write_html(html_out, include_plotlyjs="cdn")

    if png_out is not None:
        try:
            fig.write_image(png_out, scale=4, width=width, height=height)
            print(f"Saved: {png_out}")
        except Exception as e:
            print("PNG export skipped. Install kaleido if static export is required.")
            print(f"Reason: {e}")

    print(f"Saved: {html_out}")

    df.attrs["x_range"] = x_range
    df.attrs["y_range"] = y_range

    if show:
        fig.show()

    return fig, df, html_out


# =============================================================================
# 4. TEMPORAL SANKEY
# =============================================================================

QUADRANT_ORDER_WITH_ABSENT = [
    "Central nucleus",
    "First periphery",
    "Contrast zone",
    "Peripheral system",
    "Absent",
]

SANKEY_QUADRANT_COLOURS = {
    "Central nucleus": "rgba(79,127,166,0.90)",
    "First periphery": "rgba(79,154,85,0.90)",
    "Contrast zone": "rgba(184,72,72,0.90)",
    "Peripheral system": "rgba(130,80,160,0.90)",
    "Absent": "rgba(210,210,210,0.90)",
}

LINK_ALPHA = 0.32


def _rgba_with_alpha(rgba, alpha=LINK_ALPHA):
    if not isinstance(rgba, str) or not rgba.startswith("rgba"):
        return rgba
    parts = rgba.replace("rgba(", "").replace(")", "").split(",")
    if len(parts) < 3:
        return rgba
    return f"rgba({parts[0].strip()},{parts[1].strip()},{parts[2].strip()},{alpha})"


def _ordered_periods(df, period_col="period"):
    """Return periods in the order produced by temporal_stability.

    The temporal stability module writes ``period`` values by iterating over the
    categorical order created in ``build_time_periods``. This function therefore
    first honours an explicit categorical order and otherwise preserves first
    occurrence order. It does *not* alphabetically sort custom labels, so labels
    such as ``2018-2021``, ``2022-2023`` and ``2024-2026`` remain in their
    analytical order.
    """

    s = df[period_col]

    if isinstance(s.dtype, pd.CategoricalDtype):
        return [str(x) for x in s.cat.categories if pd.notna(x)]

    return pd.Series(s.dropna().astype(str)).drop_duplicates().tolist()


def _complete_period_quadrant_table(
    period_quadrants,
    period_col="period",
    term_col="term",
    quadrant_col="quadrant",
    quadrant_order=QUADRANT_ORDER_WITH_ABSENT,
):
    df = period_quadrants.copy()
    df[period_col] = df[period_col].astype(str)
    df[quadrant_col] = df[quadrant_col].astype(str)

    periods = _ordered_periods(period_quadrants, period_col=period_col)

    counts = (
        df.groupby([period_col, quadrant_col], observed=True)[term_col]
        .nunique()
        .reset_index(name="n_terms")
    )

    full_index = pd.MultiIndex.from_product(
        [periods, quadrant_order],
        names=[period_col, quadrant_col],
    )

    counts = (
        counts.set_index([period_col, quadrant_col])
        .reindex(full_index, fill_value=0)
        .reset_index()
    )

    return counts, periods


def _compute_slice_y_positions(
    node_counts,
    periods,
    quadrant_order=QUADRANT_ORDER_WITH_ABSENT,
    period_col="period",
    quadrant_col="quadrant",
    value_col="n_terms",
    top_margin=0.015,
    bottom_margin=0.015,
    min_gap=0.018,
    min_node_height=0.018,
):
    y_map = {}

    for p in periods:
        sub = node_counts.loc[node_counts[period_col].eq(p)].copy()
        sub[quadrant_col] = pd.Categorical(
            sub[quadrant_col],
            categories=quadrant_order,
            ordered=True,
        )
        sub = sub.sort_values(quadrant_col)

        values = sub[value_col].astype(float).to_numpy()
        total = values.sum()
        n_nodes = len(values)

        available = 1.0 - top_margin - bottom_margin - min_gap * (n_nodes - 1)
        available = max(available, 0.10)

        if total <= 0:
            heights = np.repeat(available / n_nodes, n_nodes)
        else:
            raw = values / total * available
            heights = np.maximum(raw, min_node_height)
            if heights.sum() > available:
                heights = heights / heights.sum() * available

        cursor = top_margin
        for q, h in zip(sub[quadrant_col].astype(str), heights):
            y_map[(p, q)] = float(min(max(cursor, 0.0), 0.98))
            cursor += h + min_gap

    return y_map


def build_temporal_sankey_ordered(
    period_quadrants,
    output_dir=OUTPUT_DIR,
    output_html="temporal_evoc_sankey_ordered.html",
    output_png="temporal_evoc_sankey_ordered.png",
    period_col="period",
    term_col="term",
    quadrant_col="quadrant",
    quadrant_order=QUADRANT_ORDER_WITH_ABSENT,
    quadrant_colours=SANKEY_QUADRANT_COLOURS,
    width=1700,
    height=900,
    node_thickness=22,
    node_pad=18,
    arrangement="fixed",
    link_alpha=LINK_ALPHA,
    show_zero_nodes=True,
    top_margin_y=0.015,
    bottom_margin_y=0.015,
    min_vertical_gap=0.018,
    min_node_height=0.018,
    export_png=True,
    png_scale=4,
    show=False,
):
    """Build a value-aware, order-preserving temporal EVOC Sankey.

    Period labels are taken directly from ``period_quadrants[period_col]``.
    If the column is categorical, its category order is used; otherwise first
    occurrence order is preserved. This is compatible with the labels produced
    by ``run_temporal_stability_analysis``.
    """

    if go is None:
        raise ImportError("Install plotly to use build_temporal_sankey_ordered().")

    output_dir = _ensure_output_dir(output_dir)

    df = period_quadrants.copy()
    required = {period_col, term_col, quadrant_col}
    missing = required.difference(df.columns)

    if missing:
        raise ValueError(f"period_quadrants is missing required columns: {missing}")

    periods = _ordered_periods(df, period_col=period_col)

    df[period_col] = df[period_col].astype(str)
    df[quadrant_col] = df[quadrant_col].astype(str)
    df.loc[~df[quadrant_col].isin(quadrant_order), quadrant_col] = "Absent"

    node_counts, periods = _complete_period_quadrant_table(
        df,
        period_col=period_col,
        term_col=term_col,
        quadrant_col=quadrant_col,
        quadrant_order=quadrant_order,
    )

    y_map = _compute_slice_y_positions(
        node_counts=node_counts,
        periods=periods,
        quadrant_order=quadrant_order,
        period_col=period_col,
        quadrant_col=quadrant_col,
        value_col="n_terms",
        top_margin=top_margin_y,
        bottom_margin=bottom_margin_y,
        min_gap=min_vertical_gap,
        min_node_height=min_node_height,
    )

    labels = []
    colours = []
    xs = []
    ys = []
    node_index = {}
    node_rows = []

    x_positions = {p: (i / max(1, len(periods) - 1)) for i, p in enumerate(periods)}

    idx = 0
    for p in periods:
        for q in quadrant_order:
            n_terms = int(
                node_counts.loc[
                    node_counts[period_col].eq(p)
                    & node_counts[quadrant_col].eq(q),
                    "n_terms",
                ].iloc[0]
            )

            if not show_zero_nodes and n_terms == 0:
                continue

            node_index[(p, q)] = idx
            labels.append(f"<b>{p}</b><br>{q}")
            colours.append(quadrant_colours.get(q, "rgba(180,180,180,0.85)"))
            xs.append(x_positions[p])
            ys.append(y_map[(p, q)])
            node_rows.append(
                {
                    "node_id": idx,
                    "period": p,
                    "quadrant": q,
                    "n_terms": n_terms,
                    "x": x_positions[p],
                    "y": y_map[(p, q)],
                }
            )
            idx += 1

    node_df = pd.DataFrame(node_rows)

    source = []
    target = []
    value = []
    link_colours = []
    link_rows = []

    for i in range(len(periods) - 1):
        p0 = periods[i]
        p1 = periods[i + 1]

        left = (
            df.loc[df[period_col].eq(p0), [term_col, quadrant_col]]
            .drop_duplicates()
            .rename(columns={quadrant_col: "q0"})
        )
        right = (
            df.loc[df[period_col].eq(p1), [term_col, quadrant_col]]
            .drop_duplicates()
            .rename(columns={quadrant_col: "q1"})
        )

        trans = left.merge(right, on=term_col, how="outer")
        trans["q0"] = trans["q0"].fillna("Absent")
        trans["q1"] = trans["q1"].fillna("Absent")

        trans_count = trans.groupby(["q0", "q1"], observed=True).size().reset_index(name="value")

        trans_count["q0"] = pd.Categorical(
            trans_count["q0"],
            categories=quadrant_order,
            ordered=True,
        )
        trans_count["q1"] = pd.Categorical(
            trans_count["q1"],
            categories=quadrant_order,
            ordered=True,
        )
        trans_count = trans_count.sort_values(["q0", "q1"]).reset_index(drop=True)

        for _, r in trans_count.iterrows():
            q0 = str(r["q0"])
            q1 = str(r["q1"])
            val = int(r["value"])

            if (p0, q0) not in node_index or (p1, q1) not in node_index:
                continue

            s = node_index[(p0, q0)]
            t = node_index[(p1, q1)]

            source.append(s)
            target.append(t)
            value.append(val)

            lc = _rgba_with_alpha(
                quadrant_colours.get(q0, "rgba(160,160,160,0.90)"),
                link_alpha,
            )
            link_colours.append(lc)

            link_rows.append(
                {
                    "period_from": p0,
                    "period_to": p1,
                    "source_quadrant": q0,
                    "target_quadrant": q1,
                    "value": val,
                }
            )

    link_df = pd.DataFrame(link_rows)

    fig = go.Figure(
        data=[
            go.Sankey(
                arrangement=arrangement,
                valueformat=",.0f",
                node=dict(
                    pad=node_pad,
                    thickness=node_thickness,
                    line=dict(color="rgba(255,255,255,0.95)", width=0.8),
                    label=labels,
                    color=colours,
                    x=xs,
                    y=ys,
                    hovertemplate="%{label}<br>Terms: %{value}<extra></extra>",
                ),
                link=dict(
                    source=source,
                    target=target,
                    value=value,
                    color=link_colours,
                    hovertemplate=(
                        "%{source.label}<br>"
                        "→ %{target.label}<br>"
                        "Terms: %{value}<extra></extra>"
                    ),
                ),
            )
        ]
    )

    fig.update_layout(
        width=width,
        height=height,
        paper_bgcolor="white",
        plot_bgcolor="white",
        margin=dict(l=20, r=20, t=20, b=20),
        font=dict(size=13, color="#111111"),
    )

    html_path = os.path.join(output_dir, output_html)
    png_path = os.path.join(output_dir, output_png) if output_png else None

    fig.write_html(html_path, include_plotlyjs="cdn")
    print(f"Saved: {html_path}")

    if export_png and png_path is not None:
        try:
            fig.write_image(png_path, width=width, height=height, scale=png_scale)
            print(f"Saved: {png_path}")
        except Exception as e:
            print("PNG export skipped. Install or update kaleido if static export is required.")
            print(f"Reason: {e}")
            png_path = None

    if show:
        fig.show()

    return {
        "fig": fig,
        "node_df": node_df,
        "link_df": link_df,
        "html": html_path,
        "png": png_path,
        "periods": periods,
    }


__all__ = [
    "build_evoc_target_plot",
    "build_evoc_collocation_tree_for_upos",
    "build_emoji_evoc_plot",
    "build_temporal_sankey_ordered",
    "select_terms_for_target",
    "assign_target_coordinates",
    "parse_emoji_evoc_html",
    "select_emojis_for_plot",
    "_ordered_periods",
]
