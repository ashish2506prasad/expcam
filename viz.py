"""Figures specified in the revision plan. Grey-background convention throughout."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

try:
    from ._path import ensure_pkg_path
except ImportError:
    from _path import ensure_pkg_path

ensure_pkg_path()

from config import GREY_BG, ef1_grid_steps
from io_utils import ensure_dir
from metrics import denormalize, explanation_on_grey


def caption(p_pct: float, conf: float, top1: float) -> str:
    mark = "✓" if top1 >= 0.5 else "✗"
    return f"p={p_pct:.1f}%  conf={conf:.2f}  top1={mark}"


def _style_axis(ax, *, grid: bool = True) -> None:
    if grid:
        ax.grid(True, which="both", linestyle=":", linewidth=0.6, alpha=0.65)
        ax.set_axisbelow(True)


def _save(fig, path: Path) -> None:
    ensure_dir(path.parent)
    for ax in fig.get_axes():
        _style_axis(ax)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def ablation_bars(
    path: Path,
    labels: Sequence[str],
    means: Sequence[float],
    ylabel: str,
    title: str,
    stds: Optional[Sequence[float]] = None,
) -> None:
    """Bar chart for LOO / sequential ablations (optional error bars)."""
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(max(10, 1.45 * len(labels)), 4.5))
    if stds is not None:
        ax.bar(x, means, yerr=stds, capsize=3, alpha=0.85, color="#4C78A8")
    else:
        ax.bar(x, means, alpha=0.85, color="#4C78A8")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=28, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.tight_layout()
    _save(fig, path)


def sweep_metric_curve(
    path: Path,
    xs: Sequence[float],
    means: Sequence[float],
    stds: Sequence[float],
    xlabel: str,
    ylabel: str,
    title: str,
    xscale: str = "linear",
) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.errorbar(xs, means, yerr=stds, marker="o", capsize=3, lw=1.5)
    if xscale != "linear":
        ax.set_xscale(xscale)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.tight_layout()
    _save(fig, path)


def grouped_bars_dual(
    path: Path,
    labels: Sequence[str],
    left_mean: Sequence[float],
    left_std: Sequence[float],
    right_mean: Sequence[float],
    right_std: Sequence[float],
    left_ylabel: str,
    right_ylabel: str,
    title: str,
    left_color: str = "#4C78A8",
    right_color: str = "#F58518",
) -> None:
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(max(8, 1.1 * len(labels)), 4.5))
    ax.bar(x, left_mean, yerr=left_std, color=left_color, alpha=0.85, capsize=3, label=left_ylabel)
    ax.set_ylabel(left_ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax2 = ax.twinx()
    ax2.errorbar(x, right_mean, yerr=right_std, color=right_color, marker="o", lw=2, capsize=3, label=right_ylabel)
    ax2.set_ylabel(right_ylabel)
    ax.set_title(title)
    fig.tight_layout()
    _save(fig, path)


def small_multiples_bars(
    path: Path,
    labels: Sequence[str],
    series: Dict[str, Tuple[Sequence[float], Sequence[float]]],
    title: str,
) -> None:
    keys = list(series.keys())
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    x = np.arange(len(labels))
    for ax, key in zip(axes.ravel(), keys):
        mean, std = series[key]
        ax.bar(x, mean, yerr=std, capsize=3, color="#4C78A8")
        ax.set_title(key)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=25, ha="right")
    fig.suptitle(title)
    fig.tight_layout()
    _save(fig, path)


def training_curves(path: Path, curve: List[Dict[str, float]]) -> None:
    steps = [r["step"] for r in curve]
    am = ["act", "ce", "kl"]
    rest = ["area", "bin", "tv", "rob"]
    fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
    for ax, keys, title in (
        (axes[0], am, "L_AM components"),
        (axes[1], rest, "L_MIN / L_ROB components"),
    ):
        for k in keys:
            ax.plot(steps, [r[k] for r in curve], label=k)
        ax.plot(steps, [r["total"] for r in curve], color="k", lw=2, label="total")
        ax.set_ylabel("loss")
        ax.set_title(title)
        ax.legend(ncol=4, fontsize=8)
    axes[1].set_xlabel("iteration")
    fig.tight_layout()
    _save(fig, path)


def ef1_training_curves(
    path: Path,
    series: Dict[str, Tuple[Sequence[int], Sequence[float]]],
    title: str,
    hline: Optional[float] = None,
) -> None:
    """Plot EF1 vs training step for multiple images/classes (single combined figure)."""
    fig, ax = plt.subplots(figsize=(9, 5))
    for label, (steps, vals) in series.items():
        ax.plot(steps, vals, lw=1.5, label=label)
    ax.set_xlabel("steps")
    ax.set_ylabel("EF1 score")
    ax.set_ylim(0, 1.05)
    ax.set_title(title)
    _style_axis(ax)
    if hline is not None:
        ax.axhline(hline, color="crimson", ls="--", lw=1, label=f"threshold={hline:g}")
    ax.legend(fontsize=7, ncol=2, loc="best")
    fig.tight_layout()
    _save(fig, path)


def plot_ef1_section0_figures(
    fig_dir: Path,
    curves: Sequence[Dict],
    backbone: str,
    hline: Optional[float] = None,
) -> None:
    """Section 0: one EF1-vs-steps plot per image plus one all-classes overlay.

    ``curves`` entries: {class_name, image_id, steps, ef1}.
    Writes Fig0_EF1_{id}_{class}.png (one per image) and Fig0_EF1_all_classes.png.
    """
    fig_dir = ensure_dir(fig_dir)
    for row in curves:
        cid = row["image_id"]
        cname = row["class_name"].replace(" ", "_")
        _ef1_single_step_plot(
            fig_dir / f"Fig0_EF1_{cid}_{cname}.png",
            row["steps"],
            row["ef1"],
            f"EF1 vs steps — {row['class_name']} ({backbone}, img {cid})",
            hline=hline,
        )

    fig, ax = plt.subplots(figsize=(8, 5))
    for row in curves:
        ax.plot(row["steps"], row["ef1"], lw=1.2, label=row["class_name"])
    ax.set_xlabel("steps")
    ax.set_ylabel("EF1 score")
    ax.set_ylim(0, 1.05)
    ax.set_title(f"EF1 vs steps — all classes ({backbone})")
    _style_axis(ax)
    if hline is not None:
        ax.axhline(hline, color="crimson", ls="--", lw=1)
    ax.legend(fontsize=6, ncol=2, loc="best", framealpha=0.85)
    fig.tight_layout()
    _save(fig, fig_dir / "Fig0_EF1_all_classes.png")


def _load_snapshot_original_panel(snapshot_png: Path) -> np.ndarray:
    """Original image panel from triple snapshot (left third)."""
    img = np.asarray(Image.open(snapshot_png).convert("RGB"), dtype=np.float32) / 255.0
    h, w = img.shape[:2]
    panel = img[:, : w // 3, :3]
    return _trim_panel_content(panel)


def _load_snapshot_expl_panel(snapshot_png: Path) -> np.ndarray:
    """Explanation panel from snapshot; prefer standalone e.png, else crop triple."""
    e_only = snapshot_png.parent / "e.png"
    if e_only.exists():
        img = np.asarray(Image.open(e_only).convert("RGB"), dtype=np.float32) / 255.0
        return _trim_panel_content(img)
    img = np.asarray(Image.open(snapshot_png).convert("RGB"), dtype=np.float32) / 255.0
    h, w = img.shape[:2]
    panel = img[:, (2 * w) // 3 :, :3]
    return _crop_expl_from_triple_panel(panel)


def _crop_expl_from_triple_panel(panel: np.ndarray) -> np.ndarray:
    """Keep only the explanation image; drop suptitle bleed and 'e (explanation)' label."""
    arr = panel[..., :3] if panel.shape[-1] >= 3 else panel
    row_cov = [(arr[y] < 0.96).any(axis=1).mean() for y in range(arr.shape[0])]
    img_rows = [y for y, cov in enumerate(row_cov) if cov > 0.35]
    if not img_rows:
        return _trim_panel_content(arr)
    block = arr[img_rows[0] : img_rows[-1] + 1]
    col_mask = (block < 0.96).any(axis=0)
    cols = np.where(col_mask)[0]
    if len(cols) == 0:
        return block
    return block[:, cols[0] : cols[-1] + 1]


def _trim_panel_content(img: np.ndarray, white_thresh: float = 250 / 255.0) -> np.ndarray:
    """Drop titles and outer whitespace from a snapshot panel."""
    arr = img[..., :3] if img.shape[-1] >= 3 else img
    mask = (arr < white_thresh).any(axis=2)
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    if len(rows) == 0 or len(cols) == 0:
        return arr
    return arr[rows[0] : rows[-1] + 1, cols[0] : cols[-1] + 1]


def _metrics_at_step(curve_by_step: Dict[int, dict], step: int) -> Optional[dict]:
    return curve_by_step.get(step)


def _step_grid_caption(step: int, met: Optional[dict]) -> str:
    if not met:
        return f"step {step}"
    return (
        f"step {step}  EF1={met['ef1']:.2f}\n"
        f"p={100 * met['p_actual']:.1f}%  Δconf={met['delta_conf']:.2f}"
    )


def step_progression_grid(
    path: Path,
    rows: List[List[Tuple[np.ndarray, str]]],
    suptitle: str = "",
) -> None:
    """Compact grid for Section 0 step snapshots — minimal padding, large panels."""
    n_r, n_c = len(rows), max(len(r) for r in rows)
    fig, axes = plt.subplots(n_r, n_c, figsize=(2.05 * n_c, 2.15 * n_r))
    if n_r == 1:
        axes = np.array([axes])
    if n_c == 1:
        axes = axes.reshape(n_r, 1)
    for i in range(n_r):
        for j in range(n_c):
            ax = axes[i, j]
            ax.axis("off")
            if j >= len(rows[i]):
                continue
            img, cap = rows[i][j]
            if img.ndim == 2:
                ax.imshow(img, cmap="gray", vmin=0, vmax=1, aspect="equal")
            else:
                ax.imshow(np.clip(img, 0, 1), aspect="equal")
            ax.set_title(cap, fontsize=6, pad=2)
    if suptitle:
        fig.suptitle(suptitle, fontsize=10, y=0.995)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.94 if suptitle else 0.99, bottom=0.01, wspace=0.03, hspace=0.08)
    _save(fig, path)


def plot_section0_step_progression_grids(
    fig_dir: Path,
    snap_dir: Path,
    *,
    image_id: str,
    class_name: str,
    backbone: str,
    curve: Sequence[dict],
    grid_every: Sequence[int] = (40,),
    cols_per_row: int = 10,
    original_img: Optional[np.ndarray] = None,
) -> None:
    """Grid of explanation evolution across training steps at several sampling rates."""
    curve_by_step = {int(p["step"]): p for p in curve}
    cname = class_name.replace(" ", "_")
    for every in grid_every:
        steps = [s for s in ef1_grid_steps(every) if (snap_dir / f"step_{s:04d}" / "x_m_e.png").exists()]
        if not steps:
            continue
        rows: List[List[Tuple[np.ndarray, str]]] = []
        offset = 0
        for row_idx in range((len(steps) + cols_per_row - 1) // cols_per_row):
            chunk_size = cols_per_row - (1 if row_idx == 0 and original_img is not None else 0)
            chunk = steps[offset : offset + chunk_size]
            offset += len(chunk)
            row: List[Tuple[np.ndarray, str]] = []
            if row_idx == 0 and original_img is not None:
                row.append((original_img, "original"))
            for step in chunk:
                png = snap_dir / f"step_{step:04d}" / "x_m_e.png"
                row.append(
                    (
                        _load_snapshot_expl_panel(png),
                        _step_grid_caption(step, _metrics_at_step(curve_by_step, step)),
                    )
                )
            rows.append(row)
        step_progression_grid(
            fig_dir / f"Fig0_step_grid_every{every}_{image_id}_{cname}.png",
            rows,
        )


def _ef1_single_step_plot(
    path: Path,
    steps: Sequence[int],
    ef1: Sequence[float],
    title: str,
    hline: Optional[float] = None,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(steps, ef1, lw=1.5, color="#4C78A8")
    ax.set_xlabel("steps")
    ax.set_ylabel("EF1 score")
    ax.set_ylim(0, 1.05)
    ax.set_title(title)
    _style_axis(ax)
    if hline is not None:
        ax.axhline(hline, color="crimson", ls="--", lw=1)
    fig.tight_layout()
    _save(fig, path)


def fidelity_vs_budget(
    path: Path,
    p: Sequence[float],
    series: Dict[str, Tuple[Sequence[float], Sequence[float]]],
    p_actual_mean: Sequence[float],
    p_actual_std: Sequence[float],
    p_star: Optional[float],
    title: str,
    xscale: str = "linear",
    tau: float = 0.95,
) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    p = np.asarray(p) * 100
    colors = {"top1_agreement": "#4C78A8", "exp_minus_kl": "#54A24B", "one_minus_normalized_kl": "#54A24B"}
    labels = {"top1_agreement": "top-1 agreement", "exp_minus_kl": "exp(−KL)", "one_minus_normalized_kl": "1 − nKL (legacy)"}
    for name, (mean, std) in series.items():
        ax.plot(p, mean, marker="o", label=labels.get(name, name), color=colors.get(name))
        ax.fill_between(p, np.asarray(mean) - std, np.asarray(mean) + std, alpha=0.18, color=colors.get(name))
    if xscale != "linear":
        ax.set_xscale(xscale)
    ax.set_xlabel("mask budget p (% pixels)")
    ax.set_ylabel("Fidelity(p) components")
    ax.set_ylim(0, 1.05)
    ax2 = ax.twinx()
    ax2.plot(p, np.asarray(p_actual_mean) * 100, "k--", label="p_actual")
    ax2.plot(p, p, color="0.5", ls=":", label="p_actual = p_budget")
    ax2.set_ylabel("p_actual (%)")
    _style_axis(ax2)
    if p_star is not None:
        ax.axvline(p_star * 100, color="crimson", ls="--", label=f"p* (τ={tau:g}) = {p_star*100:.2f}%")
    ax.set_title(title)
    ax.legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    _save(fig, path)


def metric_vs_budget(
    path: Path,
    p_fractions: Sequence[float],
    values: Sequence[float],
    *,
    ylabel: str,
    title: str,
    p_model_pct: Optional[float] = None,
    xscale: str = "linear",
) -> None:
    """Single-metric curve vs mask budget p (% pixels); optional vertical line at model p."""
    xs = np.asarray(p_fractions, dtype=float) * 100.0
    ys = np.asarray(values, dtype=float)
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.plot(xs, ys, marker="o", lw=1.5, color="#4C78A8")
    if p_model_pct is not None:
        ax.axvline(p_model_pct, color="crimson", ls="--", lw=1.2, label=f"model p={p_model_pct:.1f}%")
        ax.legend(fontsize=8)
    if xscale != "linear":
        ax.set_xscale(xscale)
    ax.set_xlabel("mask budget p (% pixels)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    _style_axis(ax)
    fig.tight_layout()
    _save(fig, path)


def boxplot(path: Path, groups: Dict[str, Sequence[float]], ylabel: str, title: str, hline: Optional[float] = None) -> None:
    fig, ax = plt.subplots(figsize=(max(6, 0.9 * len(groups)), 4.5))
    labels = list(groups.keys())
    data = [list(groups[k]) for k in labels]
    try:
        ax.boxplot(data, tick_labels=labels, showmeans=True)
    except TypeError:
        ax.boxplot(data, labels=labels, showmeans=True)
    if hline is not None:
        ax.axhline(hline, color="crimson", ls="--")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    fig.tight_layout()
    _save(fig, path)


def scatter_parity(path: Path, xs: Sequence[float], ys: Sequence[float], xlabel: str, ylabel: str, title: str) -> None:
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(xs, ys, c="#4C78A8", alpha=0.75)
    lo = min(min(xs), min(ys))
    hi = max(max(xs), max(ys))
    ax.plot([lo, hi], [lo, hi], "k--", lw=1)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.tight_layout()
    _save(fig, path)


def line_auc(path: Path, curves: Dict[str, Tuple[np.ndarray, np.ndarray, float]], xlabel: str, ylabel: str, title: str) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for name, (xs, ys, auc) in curves.items():
        ax.plot(xs, ys, label=f"{name} (AUC={auc:.3f})")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=8)
    fig.tight_layout()
    _save(fig, path)


def grouped_bars(
    path: Path,
    labels: Sequence[str],
    groups: Dict[str, Tuple[Sequence[float], Sequence[float]]],
    ylabel: str,
    title: str,
) -> None:
    x = np.arange(len(labels))
    n = len(groups)
    width = 0.8 / max(n, 1)
    fig, ax = plt.subplots(figsize=(max(8, 1.2 * len(labels)), 4.5))
    for i, (name, (mean, std)) in enumerate(groups.items()):
        ax.bar(x + (i - (n - 1) / 2) * width, mean, width, yerr=std, capsize=3, label=name)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    _save(fig, path)


def triple_panel(
    path: Path,
    x_img: np.ndarray,
    m_img: np.ndarray,
    e_img: np.ndarray,
    suptitle: str = "",
    labels: Sequence[str] = ("x (original)", "m (mask)", "e (explanation)"),
) -> None:
    """Save x | mask | explanation side-by-side in one PNG."""
    fig, axes = plt.subplots(1, 3, figsize=(7.5, 2.8))
    for ax, img, lab in zip(axes, (x_img, m_img, e_img), labels):
        ax.axis("off")
        if img.ndim == 2:
            ax.imshow(img, cmap="gray", vmin=0, vmax=1)
        else:
            ax.imshow(np.clip(img, 0, 1))
        ax.set_title(lab, fontsize=8)
    if suptitle:
        fig.suptitle(suptitle, fontsize=9)
    fig.tight_layout()
    _save(fig, path)


def image_grid(
    path: Path,
    rows: List[List[Tuple[np.ndarray, str]]],
    suptitle: str = "",
) -> None:
    """rows[i][j] = (H,W,3 or H,W float image, caption). Grey bg already baked in."""
    n_r, n_c = len(rows), max(len(r) for r in rows)
    fig, axes = plt.subplots(n_r, n_c, figsize=(2.4 * n_c, 2.8 * n_r))
    if n_r == 1:
        axes = np.array([axes])
    if n_c == 1:
        axes = axes.reshape(n_r, 1)
    for i in range(n_r):
        for j in range(n_c):
            ax = axes[i, j]
            ax.axis("off")
            if j >= len(rows[i]):
                continue
            img, cap = rows[i][j]
            if img.ndim == 2:
                ax.imshow(img, cmap="gray", vmin=0, vmax=1)
            else:
                ax.imshow(np.clip(img, 0, 1))
            ax.set_title(cap, fontsize=7)
    if suptitle:
        fig.suptitle(suptitle, fontsize=11)
    fig.tight_layout()
    _save(fig, path)


def to_numpy_image(t) -> np.ndarray:
    import torch

    if isinstance(t, torch.Tensor):
        arr = t.detach().cpu()
        if arr.dim() == 4:
            arr = arr[0]
        if arr.shape[0] in (1, 3):
            arr = arr.permute(1, 2, 0)
        arr = arr.numpy()
        if arr.shape[-1] == 1:
            arr = arr[..., 0]
        return np.clip(arr, 0, 1)
    return np.asarray(t)


def panel_original(x) -> np.ndarray:
    return to_numpy_image(denormalize(x))


def panel_mask(mask) -> np.ndarray:
    return to_numpy_image(mask)


def panel_expl(x, mask, grey: float = GREY_BG) -> np.ndarray:
    return to_numpy_image(explanation_on_grey(x, mask, grey=grey))


def panel_cam_overlay(x, cam, alpha: float = 0.45) -> np.ndarray:
    """Red–blue CAM on the original image: red = important, blue = unimportant."""
    import torch

    base = to_numpy_image(denormalize(x))
    if isinstance(cam, torch.Tensor):
        h = cam.detach().float().cpu().numpy()
    else:
        h = np.asarray(cam, dtype=np.float32)
    while h.ndim > 2:
        h = h[0]
    h = np.clip(h, 0, None)
    h = h / (h.max() + 1e-12)
    cmap = plt.colormaps["RdBu_r"]
    colored = cmap(h)[..., :3]
    return np.clip((1.0 - alpha) * base + alpha * colored, 0, 1)


def save_explanation_png(
    path: Path,
    x,
    overlay,
    *,
    heatmap: bool = False,
    overlay_alpha: float = 0.45,
) -> None:
    """Save explanation panel: mask on grey, or red–blue CAM overlay on the image."""
    import torch

    if heatmap:
        cam = overlay.detach() if isinstance(overlay, torch.Tensor) else overlay
        img = panel_cam_overlay(x, cam, alpha=overlay_alpha)
    else:
        m = overlay.detach() if isinstance(overlay, torch.Tensor) else overlay
        if m.dim() == 2:
            m = m.view(1, 1, *m.shape)
        elif m.dim() == 3:
            m = m.unsqueeze(0)
        img = panel_expl(x, m)
    fig, ax = plt.subplots(figsize=(3.2, 3.2))
    ax.axis("off")
    ax.imshow(np.clip(img, 0, 1))
    fig.tight_layout(pad=0.05)
    _save(fig, path)
