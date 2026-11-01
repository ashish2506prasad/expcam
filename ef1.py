"""Section 0 — EF1 diagnostic training.

Trains the Full EXP-CAM PixelMask for config.STEPS (override with --steps).
"""

from __future__ import annotations

import argparse
from pathlib import Path

try:
    from ._path import ensure_pkg_path
except ImportError:
    from _path import ensure_pkg_path

ensure_pkg_path()

from cli import add_common_args, lambdas_from_args, resolve_run
from classifiers import load_classifier
from config import (
    SECTION0_PER_CLASS,
    figure_dir,
)
from io_utils import export_metric_csvs, load_json, rows_to_excel, save_json
from metrics import CAM_TABLE_KEYS
from runner import maybe_train_and_eval, run_dir
from train import TrainConfig, image_seed, log
from viz import plot_ef1_section0_figures


def _section_name(args) -> str:
    return "section0_new_score" if getattr(args, "ef1_new_version", False) else "section0"


def _curve_rows_from_runs(out_root: Path, section: str, backbone: str, subset) -> list:
    curves = []
    for rec in subset:
        out_dir = run_dir(out_root, section, backbone, "ef1_full", rec["id"])
        curve_path = out_dir / "ef1_curve.json"
        if not curve_path.exists():
            continue
        curve = load_json(curve_path)
        if not curve:
            continue
        curves.append(
            {
                "class_name": rec["class_name"],
                "image_id": rec["id"],
                "wnid": rec["wnid"],
                "steps": [p["step"] for p in curve],
                "ef1": [p["ef1"] for p in curve],
            }
        )
    return curves


def _plot_step_grids_for_backbone(out_root, section, backbone, subset, grid_every, eval_root) -> None:
    fig_dir = figure_dir(out_root, section, backbone)
    n = 0
    for rec in subset:
        out_dir = run_dir(out_root, section, backbone, "ef1_full", rec["id"])
        snap_dir = out_dir / "snapshots"
        curve_path = out_dir / "ef1_curve.json"
        if not snap_dir.exists() or not curve_path.exists():
            continue
        curve = load_json(curve_path)
        if not curve:
            continue
        pil = load_pil(rec, root=eval_root)
        import numpy as np

        original = np.asarray(pil.convert("RGB"), dtype=np.float32) / 255.0
        plot_section0_step_progression_grids(
            fig_dir,
            snap_dir,
            image_id=rec["id"],
            class_name=rec["class_name"],
            backbone=backbone,
            curve=curve,
            grid_every=grid_every,
            original_img=original,
        )
        n += 1
    if n:
        log(
            f"[figures] {backbone}: step grids every {list(grid_every)} "
            f"-> {fig_dir}  ({n} images)"
        )


def _plot_ef1_for_backbone(out_root, section, backbone, subset, hline) -> dict:
    curves = _curve_rows_from_runs(out_root, section, backbone, subset)
    if not curves:
        return {}
    fig_dir = figure_dir(out_root, section, backbone)
    plot_ef1_section0_figures(fig_dir, curves, backbone, hline=hline)
    log(f"[figures] {backbone} -> {fig_dir}  ({len(curves)} per-image + 1 all-classes)")
    series = {
        f"{c['class_name']} ({c['image_id']})": (c["steps"], c["ef1"]) for c in curves
    }
    if hline is not None:
        save_json(fig_dir / "ef1_threshold_stats.json", _threshold_stats(series, hline))
    return series


def _apply_subset_args(args) -> None:
    """--subset: 2 images/class, slots 0 and 1 (same as freq_expcam --subset)."""
    if getattr(args, "quick", False):
        args.section0_per_class = 1
        args.section0_slots = [0]
    elif getattr(args, "subset", False):
        args.section0_per_class = 2
        args.img_per_class = 2


def run_section0(args) -> None:
    eval_set, subset, device = resolve_run(args)
    out_root = args.out
    section = _section_name(args)
    steps = args.steps
    rows = []
    ef1_version = "new" if args.ef1_new_version else "legacy"
    eval_root = Path(eval_set["root"])
    lambdas = lambdas_from_args(args)

    log(
        f"Section 0  device={device}  images={len(subset)}  steps={steps}  "
        f"section={section}  lambdas={lambdas}"
    )

    for backbone in args.backbones:
        log(f"Loading classifier {backbone} ...")
        bundle = load_classifier(backbone, device)
        for rec in subset:
            out_dir = run_dir(out_root, section, backbone, "ef1_full", rec["id"])
            cfg = TrainConfig(
                lambdas=dict(lambdas),
                steps=steps,
                seed=image_seed(rec["id"], backbone, args.seed),
                log_every=args.log_every,
                no_ce=args.no_ce,
                record_ef1=True,
                keep_best_ef1=True,
                ef1_version=ef1_version,
            )
            row = maybe_train_and_eval(
                bundle,
                rec,
                eval_root,
                out_dir,
                cfg,
                skip_existing=args.skip_existing,
                device=device,
                extra_metrics={
                    "protocol": "ef1_new" if args.ef1_new_version else "ef1",
                    "config": "Full_no_ce",
                    "ef1_version": ef1_version,
                },
            )
            curve = load_json(out_dir / "ef1_curve.json") if (out_dir / "ef1_curve.json").exists() else []
            if curve:
                row["ef1_final"] = curve[-1]["ef1"]
                row["ef1_max"] = max(p["ef1"] for p in curve)
                row["ef1_mean"] = sum(p["ef1"] for p in curve) / len(curve)
            rows.append(row)

        _plot_ef1_for_backbone(out_root, section, backbone, subset, args.ef1_threshold)

    save_json(out_root / section / "ef1_runs.json", rows)
    rows_to_excel(out_root / section / "section0_results.xlsx", {"ef1": rows})
    cam_keys = [k for k in CAM_TABLE_KEYS if k != "insertion_auc"]
    export_metric_csvs(out_root / section, rows, "ddcam_metrics", cam_keys)
    log(f"EXP-CAM faithfulness CSV -> {out_root / section / 'ddcam_metrics_summary.csv'}")
    log(f"Section 0 complete under {out_root / section}")


def _threshold_stats(series, tau: float) -> dict:
    out = {}
    for label, (steps, vals) in series.items():
        hits = [s for s, v in zip(steps, vals) if v >= tau]
        out[label] = {
            "first_step_at_or_above": hits[0] if hits else None,
            "max_ef1": max(vals) if vals else None,
            "final_ef1": vals[-1] if vals else None,
        }
    return {"threshold": tau, "per_image": out}


def run_section0_figures_only(args) -> None:
    eval_set, subset, _device = resolve_run(args)
    out_root = args.out
    section = _section_name(args)
    js = out_root / section / "ef1_runs.json"
    if not js.exists():
        log(f"No {section}/ef1_runs.json — run Section 0 first.")
        return
    load_json(js)
    for backbone in args.backbones:
        _plot_ef1_for_backbone(out_root, section, backbone, subset, args.ef1_threshold)


def main(argv=None):
    p = add_common_args(argparse.ArgumentParser(description="Section 0: EF1 diagnostic training"))
    p.add_argument(
        "--ef1-new-version",
        action="store_true",
        help="Use ef1_new_version=HM(conf, 1−p); write to section0_new_score/ (plots still say EF1).",
    )
    p.add_argument(
        "--ef1-threshold",
        type=float,
        default=None,
        help="Optional horizontal line on EF1 plot to inspect a quality threshold.",
    )
    p.add_argument(
        "--subset",
        action="store_true",
        help="Use 2 images per class (overrides --img-per-class).",
    )
    p.add_argument(
        "--section0-per-class",
        type=int,
        default=SECTION0_PER_CLASS,
        help="Images per class for Section 0. Overridden by --subset.",
    )
    p.add_argument(
        "--figures-only",
        action="store_true",
        help="Regenerate EF1 plots from saved runs.",
    )
    args = p.parse_args(argv)
    _apply_subset_args(args)
    if args.figures_only:
        run_section0_figures_only(args)
    else:
        run_section0(args)


if __name__ == "__main__":
    main()
