"""Publication-grade (>=300 dpi) figure generator for DisEnergy."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
        }
    )


def plot_3d_manifolds(
    image: np.ndarray,
    manifolds: dict[str, np.ndarray],
    out_path: Path,
    title: str = "Multi-Scale 3D DisEnergy Manifolds",
) -> None:
    """Plot input image alongside Fine, Mid, Large, and Fused 3D elevation surfaces."""
    _style()
    names = list(manifolds.keys())
    n_panels = 1 + len(names)
    fig = plt.figure(figsize=(3.6 * n_panels, 3.6))

    ax0 = fig.add_subplot(1, n_panels, 1)
    disp = image[:, :, :3] if (image.ndim == 3 and image.shape[-1] >= 3) else np.squeeze(image)
    ax0.imshow(disp, cmap="gray" if disp.ndim == 2 else None)
    ax0.set_title("Input Image")
    ax0.axis("off")

    for idx, name in enumerate(names, start=2):
        m = np.squeeze(manifolds[name])
        h, w = m.shape
        yy, xx = np.mgrid[0:h:2, 0:w:2]
        zz = m[::2, ::2]
        ax = fig.add_subplot(1, n_panels, idx, projection="3d")
        ax.plot_surface(xx, yy, zz, cmap="viridis", linewidth=0, antialiased=True, rstride=1, cstride=1)
        ax.set_title(name)
        ax.set_zlim(0.0, 1.0)
        ax.view_init(elev=38, azim=-55)
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle(title, fontweight="bold", y=1.02)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)


def plot_critical_difference(
    mean_ranks: dict[str, float],
    cd: float,
    out_path: Path,
    title: str = "Friedman-Nemenyi Critical Difference Diagram",
) -> None:
    """Render a clean Critical Difference (CD) diagram for omnibus ranking."""
    _style()
    sorted_items = sorted(mean_ranks.items(), key=lambda kv: kv[1])
    k = len(sorted_items)
    fig, ax = plt.subplots(figsize=(8.0, 0.55 * k + 1.8))

    max_rank = max(k, int(np.ceil(max(mean_ranks.values()))))
    ax.hlines(k + 0.6, 1, max_rank, color="black", linewidth=1.2)
    for r in range(1, max_rank + 1):
        ax.vlines(r, k + 0.45, k + 0.75, color="black", linewidth=1.0)
        ax.text(r, k + 0.85, str(r), ha="center", va="bottom", fontsize=9)

    # CD bar
    ax.hlines(k + 1.35, 1, 1 + cd, color="crimson", linewidth=2.5)
    ax.text(1 + 0.5 * cd, k + 1.48, f"CD = {cd:.2f}", ha="center", va="bottom", color="crimson", fontweight="bold")

    for i, (name, rank) in enumerate(sorted_items):
        y = k - i - 0.3
        ax.plot([rank], [y], "o", color="#1f77b4", markersize=7)
        ax.text(rank + 0.12, y, f"{name} ({rank:.2f})", va="center", ha="left", fontsize=9)

    # Clique bars for pairs within CD
    clique_y = k + 0.15
    for i in range(k):
        j = i
        while j + 1 < k and (sorted_items[j + 1][1] - sorted_items[i][1]) <= cd:
            j += 1
        if j > i:
            ax.hlines(clique_y, sorted_items[i][1], sorted_items[j][1], color="black", linewidth=2.5)
            clique_y -= 0.15

    ax.set_xlim(0.7, max_rank + 2.2)
    ax.set_ylim(-0.5, k + 1.9)
    ax.axis("off")
    ax.set_title(title, fontweight="bold")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)


def plot_curves(
    x_vals: list[float],
    series: dict[str, tuple[list[float], list[float] | None]],
    xlabel: str,
    ylabel: str,
    title: str,
    out_path: Path,
) -> None:
    """Plot noise-stress or label-scarcity curves with optional shaded error bands."""
    _style()
    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    markers = ["o", "s", "^", "D", "v", "P", "X"]
    for idx, (label, (ys, errs)) in enumerate(series.items()):
        x_arr = np.asarray(x_vals, dtype=float)
        y_arr = np.asarray(ys, dtype=float)
        lw = 2.4 if "DisEnergy" in label else 1.5
        ax.plot(x_arr, y_arr, marker=markers[idx % len(markers)], linewidth=lw, label=label)
        if errs is not None:
            e_arr = np.asarray(errs, dtype=float)
            ax.fill_between(x_arr, y_arr - e_arr, y_arr + e_arr, alpha=0.15)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(frameon=True, fontsize=8)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)


def generate_graphical_abstract(
    sample_img: np.ndarray,
    fused_manifold: np.ndarray,
    pred_mask: np.ndarray,
    out_path: Path,
) -> None:
    """Generate Elsevier-compliant wide landscape Graphical Abstract (>= 1328 x 531 px @ 300 dpi)."""
    _style()
    fig = plt.figure(figsize=(10.0, 4.0), dpi=300)  # 3000 x 1200 px (> 1328 x 531 px)
    gs = fig.add_gridspec(1, 4, wspace=0.25)

    ax1 = fig.add_subplot(gs[0, 0])
    disp = sample_img[:, :, :3] if sample_img.ndim == 3 and sample_img.shape[-1] >= 3 else np.squeeze(sample_img)
    ax1.imshow(disp, cmap="gray" if disp.ndim == 2 else None)
    ax1.set_title("1. Raw Multi-Domain Input\n(0 Training Params)", fontsize=9, fontweight="bold")
    ax1.axis("off")

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.text(
        0.5,
        0.5,
        "2. DisEnergy Operator\n\n"
        r"$\mathcal{D}_s(k) \propto \sum \frac{|\Delta E_{\mathrm{loc}}| + \eta W_2}{\|c_k - c_j\|^\beta}$"
        "\n\n• Affine Shift Invariant\n• O(1/w_s) Noise Decay\n• Moran-Entropy Fusion",
        ha="center",
        va="center",
        fontsize=8.5,
        bbox={"boxstyle": "round,pad=0.6", "facecolor": "#eef5ff", "edgecolor": "#2b5b84"},
    )
    ax2.axis("off")

    ax3 = fig.add_subplot(gs[0, 2], projection="3d")
    m = np.squeeze(fused_manifold)
    h, w = m.shape
    yy, xx = np.mgrid[0:h:2, 0:w:2]
    ax3.plot_surface(xx, yy, m[::2, ::2], cmap="viridis", linewidth=0)
    ax3.set_title("3. 3D Fused Manifold\n(Box-Free Elevation)", fontsize=9, fontweight="bold")
    ax3.set_xticks([])
    ax3.set_yticks([])
    ax3.view_init(elev=36, azim=-55)

    ax4 = fig.add_subplot(gs[0, 3])
    ax4.imshow(disp, cmap="gray" if disp.ndim == 2 else None)
    ax4.contour(np.squeeze(pred_mask).astype(float), levels=[0.5], colors=["#ff3b30"], linewidths=1.8)
    ax4.set_title("4. Box-Free Mask & TDA\n(Watershed + H0/H1 PH)", fontsize=9, fontweight="bold")
    ax4.axis("off")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300)
    fig.savefig(out_path.with_suffix(".pdf"), dpi=300)
    plt.close(fig)
