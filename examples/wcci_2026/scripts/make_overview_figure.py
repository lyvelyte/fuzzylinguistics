#!/usr/bin/env python3
"""
Generate a clean "big picture" overview figure for the FuzzyLinguistics paper.

- Single horizontal pipeline (6 blocks)
- Embedded inset placeholders inside:
    (2) Fuzzify  -> membership plot
    (5) Simplify -> DAG snapshot
- Optional: render real images into those insets (pass file paths)
- Saves both PNG (for quick viewing) and PDF (ideal for LaTeX)

Usage examples:
  python make_fuzzylinguistics_overview.py
  python make_fuzzylinguistics_overview.py --membership_img membership.png --dag_img dag.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless / CI-safe
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch


def _add_box(
    ax,
    x: float,
    y: float,
    w: float,
    h: float,
    title: str,
    body: str,
    face: str,
    edge: str,
    *,
    title_fs: float = 16,
    body_fs: float = 12,
):
    box = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.010,rounding_size=0.020",
        linewidth=2.2,
        edgecolor=edge,
        facecolor=face,
        zorder=1,
    )
    ax.add_patch(box)

    # Title near top
    ax.text(
        x + w / 2,
        y + h * 0.86,
        title,
        ha="center",
        va="center",
        fontsize=title_fs,
        weight="bold",
        color="#111",
        family="DejaVu Sans",
        zorder=2,
    )

    # Body
    ax.text(
        x + w / 2,
        y + h * 0.63,
        body,
        ha="center",
        va="center",
        fontsize=body_fs,
        color="#111",
        family="DejaVu Sans",
        zorder=2,
    )

    return box


def _add_inset_panel(
    ax,
    parent_box: FancyBboxPatch,
    edge: str,
    label: str,
    placeholder: str,
    *,
    img_path: str | None = None,
):
    """
    Adds a white inset "slot" inside a given parent box.
    If img_path is provided, draws the image in the slot; otherwise placeholder text.
    """
    x, y = parent_box.get_x(), parent_box.get_y()
    w, h = parent_box.get_width(), parent_box.get_height()

    # Inset geometry (bottom portion of the parent box)
    inset_x = x + w * 0.07
    inset_w = w * 0.86
    inset_y = y + h * 0.10
    inset_h = h * 0.36

    inset = FancyBboxPatch(
        (inset_x, inset_y),
        inset_w,
        inset_h,
        boxstyle="round,pad=0.006,rounding_size=0.015",
        linewidth=1.6,
        edgecolor=edge,
        facecolor="white",
        zorder=3,
    )
    ax.add_patch(inset)

    ax.text(
        inset_x + inset_w / 2,
        inset_y + inset_h * 0.80,
        label,
        ha="center",
        va="center",
        fontsize=10.2,
        weight="bold",
        color="#222",
        family="DejaVu Sans",
        zorder=4,
    )

    if img_path:
        img_path = str(img_path)
        try:
            img = plt.imread(img_path)
            # Draw image inside inset with a little padding
            pad_x = inset_w * 0.04
            pad_y = inset_h * 0.10
            ax.imshow(
                img,
                extent=(
                    inset_x + pad_x,
                    inset_x + inset_w - pad_x,
                    inset_y + pad_y,
                    inset_y + inset_h * 0.68,
                ),
                aspect="auto",
                zorder=4,
                clip_path=inset,
                clip_on=True,
            )
        except Exception:
            # Fallback to placeholder if image load fails
            ax.text(
                inset_x + inset_w / 2,
                inset_y + inset_h * 0.38,
                placeholder,
                ha="center",
                va="center",
                fontsize=10.0,
                color="#666",
                family="DejaVu Sans",
                zorder=4,
            )
    else:
        ax.text(
            inset_x + inset_w / 2,
            inset_y + inset_h * 0.38,
            placeholder,
            ha="center",
            va="center",
            fontsize=10.0,
            color="#666",
            family="DejaVu Sans",
            zorder=4,
        )


def _add_optional_pills(ax, parent_box: FancyBboxPatch):
    """
    Adds small dashed 'Optional' pills inside the Outputs block (bottom row),
    instead of separate external boxes.
    """
    x, y = parent_box.get_x(), parent_box.get_y()
    w, h = parent_box.get_width(), parent_box.get_height()

    pill_y = y + h * 0.12
    pill_h = h * 0.14
    gap = w * 0.04
    pill_w = (w - 3 * gap) / 2  # left gap, mid gap, right gap

    pills = [
        (x + gap, "Optional: Differential summaries"),
        (x + 2 * gap + pill_w, "Optional: LLM-grounded reasoning"),
    ]

    for px, txt in pills:
        pill = FancyBboxPatch(
            (px, pill_y),
            pill_w,
            pill_h,
            boxstyle="round,pad=0.006,rounding_size=0.020",
            linewidth=1.6,
            edgecolor="#666",
            facecolor="white",
            linestyle="--",
            zorder=5,
        )
        ax.add_patch(pill)
        ax.text(
            px + pill_w / 2,
            pill_y + pill_h / 2,
            txt,
            ha="center",
            va="center",
            fontsize=9.6,
            color="#333",
            family="DejaVu Sans",
            zorder=6,
        )


def make_figure(
    out_png: Path,
    out_pdf: Path,
    membership_img: str | None = None,
    dag_img: str | None = None,
    dpi: int = 300,
):
    # Figure sized to look good in IEEE 2-col (use PDF for LaTeX)
    fig = plt.figure(figsize=(13.5, 3.2), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Title
    fig.text(
        0.03,
        0.965,
        "FuzzyLinguistics overview (configuration-driven, deterministic)",
        ha="left",
        va="top",
        fontsize=18,
        weight="bold",
        family="DejaVu Sans",
        color="#111",
    )

    # Layout
    left, right = 0.03, 0.03
    gap = 0.012
    n = 6
    w = (1 - left - right - gap * (n - 1)) / n
    y = 0.20
    h = 0.64
    xs = [left + i * (w + gap) for i in range(n)]

    # Color palette (keeps your look, slightly softened fills)
    palette = [
        ("#E6F0FF", "#2B5CAA"),  # Inputs
        ("#E7F7EC", "#2E7D32"),  # Fuzzify
        ("#FFF2DD", "#B36B00"),  # Generate
        ("#EEE8FF", "#5A3DBD"),  # Score
        ("#FFE6EA", "#B00020"),  # Simplify
        ("#F2F2F2", "#333333"),  # Outputs
    ]

    titles = ["1. Inputs", "2. Fuzzify", "3. Generate", "4. Score", "5. Simplify", "6. Outputs"]
    bodies = [
        "Data (NumPy/Pandas)\nConfig (JSON)",
        "Evaluate μ(x)\n(vectorized)",
        "Protoforms\n(q, r, p)",
        "Truth / Focus /\nSimplicity → V(S)",
        "Redundancy DAG\n+ τ pruning\nGroup / compress",
        "Ranked + simplified\nCSV/TXT/LaTeX\n+ .gexf graphs",
    ]

    boxes: list[FancyBboxPatch] = []
    for i in range(n):
        face, edge = palette[i]
        b = _add_box(ax, xs[i], y, w, h, titles[i], bodies[i], face, edge)
        boxes.append(b)

    # Embedded inset slots (inside the pipeline blocks)
    _add_inset_panel(
        ax,
        boxes[1],  # Fuzzify
        palette[1][1],
        label="Example membership functions",
        placeholder="<insert membership plot here>",
        img_path=membership_img,
    )
    _add_inset_panel(
        ax,
        boxes[4],  # Simplify
        palette[4][1],
        label="Example redundancy DAG (.gexf)",
        placeholder="<insert DAG snapshot here>",
        img_path=dag_img,
    )

    # Optional pills inside Outputs (no extra external boxes)
    _add_optional_pills(ax, boxes[5])

    # Arrows between main blocks
    arrow_y = y + h * 0.56
    for i in range(n - 1):
        x1 = xs[i] + w
        x2 = xs[i + 1]
        arr = FancyArrowPatch(
            (x1 + gap * 0.12, arrow_y),
            (x2 - gap * 0.12, arrow_y),
            arrowstyle="-|>",
            mutation_scale=18,
            linewidth=2.0,
            color="#333",
            zorder=10,
        )
        ax.add_patch(arr)

    # Small footer (optional)
    fig.text(
        0.03,
        0.06,
        "Deterministic pipeline → auditable artifacts suitable for reporting and downstream LLM reasoning.",
        ha="left",
        va="bottom",
        fontsize=10.2,
        family="DejaVu Sans",
        color="#555",
    )

    # Save
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, bbox_inches="tight", pad_inches=0.10)
    fig.savefig(out_pdf, bbox_inches="tight", pad_inches=0.10)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default="fuzzylinguistics_overview", help="output base name (no extension)")
    ap.add_argument("--membership_img", type=str, default=None, help="path to a membership plot image to embed")
    ap.add_argument("--dag_img", type=str, default=None, help="path to a DAG snapshot image to embed")
    args = ap.parse_args()

    base = Path(args.out)
    make_figure(
        out_png=base.with_suffix(".png"),
        out_pdf=base.with_suffix(".pdf"),
        membership_img=args.membership_img,
        dag_img=args.dag_img,
    )


if __name__ == "__main__":
    main()
