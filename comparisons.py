"""Section 3 — Fair baseline comparison at size-matched sparsity.

EXP-CAM is trained here (800 steps by default) and the best-EF1 checkpoint is kept.
CAM baselines: GradCAM++, LayerCAM, ScoreCAM, EigenCAM, attention (ViT), RISE.
Explanation PNGs are saved per image × backbone × method.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import torch

try:
    from ._path import ensure_pkg_path
except ImportError:
    from _path import ensure_pkg_path

ensure_pkg_path()

from baselines import BASELINE_NAMES, compute_heatmap, otsu_binarize, topk_binarize
from cli import add_common_args, lambdas_from_args, resolve_run
from classifiers import load_classifier
from config import N_RISE_MASKS, SECTION3_EXP_STEPS, figure_dir
from data import load_tensor
from dd_cam import ddcam_heatmap
from io_utils import ensure_dir, export_metric_csvs, load_json, mean_std, rows_to_excel, save_json
from metrics import CAM_TABLE_KEYS, evaluate_explanation, insertion_curve
from runner import load_masks, run_dir
from train import TrainConfig, image_seed, log, save_mask_tensors, train_mask
from viz import (
    caption,
    grouped_bars,
    image_grid,
    line_auc,
    panel_cam_overlay,
    panel_expl,
    panel_original,
    save_explanation_png,
)


def _image_dir(out_root: Path, backbone: str, image_id: str) -> Path:
    return out_root / backbone / image_id


def _methods_for(args, bundle) -> list[str]:
    if args.quick:
        return ["gradcam++", "scorecam"]
    names = list(getattr(args, "compare", None) or BASELINE_NAMES)
    if "ddcam" not in names:
        names.append("ddcam")
    if not bundle.cam_is_vit:
        names = [n for n in names if n != "attention"]
    return list(dict.fromkeys(names))


def _section0_mask_dir(args, backbone: str, image_id: str) -> Path:
    candidates = []
    src = getattr(args, "section0_source", None)
    if src:
        candidates.append(src)
    candidates.extend(("section0", "section0_new_score"))
    seen = set()
    for name in candidates:
        if name in seen:
            continue
        seen.add(name)
        d = run_dir(args.out, name, backbone, "ef1_full", image_id)
        if (d / "masks.pt").exists():
            return d
    return run_dir(args.out, candidates[0], backbone, "ef1_full", image_id)


def _copy_section0_expcam(s0: Path, exp_dir: Path) -> None:
    import shutil

    ensure_dir(exp_dir)
    for name in ("masks.pt", "metrics.json", "ef1_curve.json", "train_meta.json"):
        src = s0 / name
        if src.exists():
            shutil.copy2(src, exp_dir / name)


def _expcam_from_section0(bundle, x, rec, dest: Path, args, device):
    """Reuse Section 0 mask; do not retrain. Returns None if missing."""
    s0 = _section0_mask_dir(args, bundle.name, rec["id"])
    if not (s0 / "masks.pt").exists():
        return None
    exp_dir = ensure_dir(dest / "expcam")
    _copy_section0_expcam(s0, exp_dir)
    m_hard, m_prob = load_masks(s0, device)
    met = evaluate_explanation(bundle, x, m_hard, mask_prob=m_prob)
    if (s0 / "metrics.json").exists():
        old = load_json(s0 / "metrics.json")
        for k in ("best_ef1", "best_ef1_step", "seconds", "lambdas"):
            if k in old:
                met[k] = old[k]
    xs, ys, auc = insertion_curve(bundle, x, m_prob[0, 0], binary_mask=False)
    met["insertion_auc"] = auc
    save_json(exp_dir / "metrics.json", met)
    return m_hard, m_prob, met, (xs, ys, auc)


def _metric_row(rec, backbone, method, mode, met, extra=None) -> dict:
    row = {
        "image_id": rec["id"],
        "wnid": rec["wnid"],
        "class_name": rec["class_name"],
        "backbone": backbone,
        "method": method,
        "mode": mode,
        "p_actual": met.get("p_actual"),
        "conf_e": met.get("conf_e"),
        "conf_x": met.get("conf_x"),
        "top1_agreement": met.get("top1_agreement"),
        "ad": met.get("ad"),
        "ic": met.get("ic"),
        "com": met.get("com"),
        "coh": met.get("coh"),
        "adcc": met.get("adcc"),
        "add": met.get("add"),
        "ef1": met.get("ef1"),
        "ef1_conf": met.get("ef1_conf"),
        "kl_divergence": met.get("kl_divergence"),
        "insertion_auc": met.get("insertion_auc"),
        "best_ef1": met.get("best_ef1"),
        "best_ef1_step": met.get("best_ef1_step"),
    }
    if extra:
        row.update(extra)
    return row


def _train_expcam(bundle, x, rec, exp_dir: Path, args) -> tuple:
    """Train EXP-CAM; return (mask_hard, mask_prob, metrics_dict, train_result)."""
    cfg = TrainConfig(
        lambdas=lambdas_from_args(args),
        steps=args.exp_steps,
        seed=image_seed(rec["id"], bundle.name, args.seed),
        log_every=args.log_every,
        no_ce=args.no_ce,
        record_ef1=True,
        keep_best_ef1=True,
    )
    log(
        f"  [EXP-CAM train] steps={cfg.steps}  no_ce={cfg.no_ce}  "
        f"best_ef1 checkpoint enabled"
    )
    result = train_mask(bundle, x, cfg)
    ensure_dir(exp_dir)
    save_mask_tensors(exp_dir, result)
    save_json(exp_dir / "ef1_curve.json", result.ef1_curve)
    save_json(
        exp_dir / "best_ef1.json",
        {
            "best_ef1": result.best_ef1,
            "best_ef1_step": result.best_ef1_step,
            "no_ce": cfg.no_ce,
            "steps": cfg.steps,
        },
    )
    met = evaluate_explanation(bundle, x, result.mask_hard, mask_prob=result.mask_prob)
    met["best_ef1"] = result.best_ef1
    met["best_ef1_step"] = result.best_ef1_step
    met["no_ce"] = cfg.no_ce
    save_json(exp_dir / "metrics.json", met)
    return result.mask_hard, result.mask_prob, met, result


def _save_explanation(out_dir: Path, method: str, x, overlay, *, heatmap: bool) -> None:
    expl_dir = ensure_dir(out_dir / "explanations")
    save_explanation_png(expl_dir / f"{_method_file_stem(method)}.png", x, overlay, heatmap=heatmap)


def run_section3(args) -> None:
    eval_set, images, device = resolve_run(args)
    out_root = args.out / "section3"
    rows = []
    curves = defaultdict(list)
    rise_masks = 64 if args.quick else args.rise_masks

    log(
        f"Section 3  device={device}  images={len(images)}  exp_steps={args.exp_steps}  "
        f"no_ce={args.no_ce}  backbones={list(args.backbones)}"
    )

    for backbone in args.backbones:
        log(f"Loading classifier {backbone} ...")
        bundle = load_classifier(backbone, device)
        methods = _methods_for(args, bundle)

        have = []
        for rec in images:
            s0 = _section0_mask_dir(args, backbone, rec["id"])
            if (s0 / "masks.pt").exists():
                have.append(rec)
        log(f"  {backbone}: {len(have)}/{len(images)} images have Section 0 masks")
        illustrate = bool(getattr(args, "image_illustration", False))

        for rec in have:
            dest = _image_dir(out_root, backbone, rec["id"])
            marker = dest / "complete.json"
            if args.skip_existing and marker.exists():
                log(f"[skip] {backbone} img={rec['id']} (complete)")
                continue

            x = load_tensor(rec, bundle.preprocess, device, root=Path(eval_set["root"]))
            log(f"[baselines] {backbone} img={rec['id']} {rec['class_name']}")

            train_exp = bool(getattr(args, "train_expcam", False))
            loaded = None if train_exp else _expcam_from_section0(bundle, x, rec, dest, args, device)
            if loaded is None and train_exp:
                exp_dir = ensure_dir(dest / "expcam")
                m_exp, p_exp, exp_metrics, _result = _train_expcam(bundle, x, rec, exp_dir, args)
                xs, ys, auc = insertion_curve(bundle, x, p_exp[0, 0], binary_mask=False)
                exp_metrics["insertion_auc"] = auc
                loaded = (m_exp, p_exp, exp_metrics, (xs, ys, auc))
            if loaded is None:
                log(f"[skip] no Section 0 mask for {backbone} img={rec['id']} — run section 0 first")
                continue

            m_exp, p_exp, exp_metrics, curve_e = loaded
            k = int(m_exp.sum().item())
            if illustrate:
                _save_explanation(dest, "EXP-CAM", x, m_exp, heatmap=False)
            rows.append(_metric_row(rec, backbone, "EXP-CAM", "size_matched", exp_metrics, {"no_ce": args.no_ce}))
            curves[("EXP-CAM", backbone)].append(curve_e)

            for name in methods:
                if name == "ddcam":
                    cam = ddcam_heatmap(bundle, x)
                    disp = "DD-CAM"
                else:
                    cam = compute_heatmap(name, bundle, x, rise_masks=rise_masks)
                    disp = name
                if cam is None:
                    continue
                log(f"  {disp} done")
                cam = cam.detach()
                matched = topk_binarize(cam, k).view(1, 1, *cam.shape[-2:])
                own = otsu_binarize(cam).view(1, 1, *cam.shape[-2:])
                if illustrate:
                    _save_explanation(dest, disp, x, cam, heatmap=True)

                for mode, mask in (("size_matched", matched), ("own_threshold", own)):
                    met = evaluate_explanation(bundle, x, mask, mask_prob=cam.view(1, 1, *cam.shape))
                    xs_i, ys_i, auc_i = insertion_curve(bundle, x, cam, binary_mask=False)
                    met["insertion_auc"] = auc_i
                    rows.append(_metric_row(rec, backbone, disp, mode, met, {"no_ce": args.no_ce}))
                    if mode == "size_matched":
                        curves[(disp, backbone)].append((xs_i, ys_i, auc_i))

                save_json(dest / f"{name}.json", {"k": k, "cam_max": float(cam.max().item())})
                torch.save(
                    {"cam": cam.cpu(), "matched": matched.cpu(), "own": own.cpu()},
                    dest / f"{name}.pt",
                )

            save_json(
                marker,
                {
                    "image_id": rec["id"],
                    "backbone": backbone,
                    "methods": ["EXP-CAM", *methods],
                    "explanations_dir": str(dest / "explanations"),
                    "expcam_from_section0": not train_exp,
                },
            )

    save_json(out_root / "all_runs.json", rows)
    rows_to_excel(out_root / "section3_results.xlsx", {"baselines": rows})
    cam_keys = [k for k in CAM_TABLE_KEYS if any(r.get(k) is not None for r in rows)]
    export_metric_csvs(out_root, rows, "comparison", cam_keys, extra_groups=("method", "mode"))
    log(f"Comparison CSV -> {out_root / 'comparison_summary.csv'}")
    if getattr(args, "image_illustration", False):
        _figures(eval_set, images, args, rows, curves, device)
    else:
        log("Skipping per-image CAM figures (--image-illustration not set)")


def _normalize_grid_ids(raw_ids) -> list[str]:
    """Accept '000', '0', 0 → zero-padded image id."""
    out = []
    for rid in raw_ids or []:
        s = str(rid).strip()
        if s.isdigit():
            out.append(f"{int(s):03d}")
        else:
            out.append(s)
    return out


def _fair_method_caption(bundle, x, device, blob) -> str:
    """Caption for size-matched baseline panel from saved mask + CAM."""
    mask = blob["matched"].to(device)
    cam = blob.get("cam")
    mask_prob = None
    if cam is not None:
        cam = cam.to(device)
        mask_prob = cam.view(1, 1, *cam.shape[-2:])
    met = evaluate_explanation(bundle, x, mask, mask_prob=mask_prob)
    return caption(met["p_actual_pct"], met["conf_e"], met["top1_agreement"])


def _comparison_row(
    eval_set,
    args,
    bundle,
    device,
    backbone,
    qid: str,
    *,
    fair: bool,
    all_rows: list,
) -> list[tuple] | None:
    """One row: original | EXP-CAM | baselines (fair=size-matched masks, else heatmaps)."""
    id_to_rec = {r["id"]: r for r in eval_set["images"]}
    rec = id_to_rec.get(qid)
    if rec is None:
        return None
    dest = _image_dir(args.out / "section3", backbone, qid)
    if not (dest / "expcam" / "metrics.json").exists():
        return None

    x = load_tensor(rec, bundle.preprocess, device, root=Path(eval_set["root"]))
    met = load_json(dest / "expcam" / "metrics.json")
    cap_exp = caption(met["p_actual_pct"], met["conf_e"], met["top1_agreement"])

    row = [(panel_original(x), f"{rec['class_name']}\noriginal")]

    from runner import load_masks

    m_exp, _ = load_masks(dest / "expcam", device)
    row.append((panel_expl(x, m_exp), f"EXP-CAM\n{cap_exp}"))

    for name in _method_cols(bundle)[1:]:
        stem = "ddcam" if name == "DD-CAM" else name
        pt = dest / f"{stem}.pt"
        if not pt.exists():
            continue
        blob = torch.load(pt, map_location=device)
        if fair:
            mask = blob["matched"].to(device)
            img = panel_expl(x, mask)
            cap = _fair_method_caption(bundle, x, device, blob)
            row.append((img, f"{name}\n{cap}"))
        else:
            panel = _load_cam_heatmap_panel(x, dest, stem, device)
            if panel is None:
                continue
            row.append((panel, name))

    return row if len(row) > 1 else None


def _export_per_image_comparisons(eval_set, images, args, device, backbone, fig_dir, all_rows) -> None:
    """Write comparisons_fair/comparison_XXX.png and comparisons_heatmap/comparison_XXX.png."""
    bundle = load_classifier(backbone, device)
    fair_dir = ensure_dir(fig_dir / "comparisons_fair")
    heat_dir = ensure_dir(fig_dir / "comparisons_heatmap")
    n = 0
    for rec in images:
        qid = rec["id"]
        row_fair = _comparison_row(
            eval_set, args, bundle, device, backbone, qid, fair=True, all_rows=all_rows
        )
        if row_fair:
            image_grid(
                fair_dir / f"comparison_{qid}.png",
                [row_fair],
                suptitle=f"{rec['class_name']} ({qid}) — size-matched fair ({backbone})",
            )
        row_hm = _comparison_row(
            eval_set, args, bundle, device, backbone, qid, fair=False, all_rows=all_rows
        )
        if row_hm:
            image_grid(
                heat_dir / f"comparison_{qid}.png",
                [row_hm],
                suptitle=f"{rec['class_name']} ({qid}) — CAM heatmaps ({backbone})",
            )
        if row_fair or row_hm:
            n += 1
    log(f"[figures] {backbone}: {n} per-image comparisons -> {fair_dir.name}/ & {heat_dir.name}/")


def _fig_custom_grid(eval_set, args, device, fig_dir, backbone, grid_ids, all_rows) -> None:
    """Multi-row grid from user-selected image ids."""
    if not grid_ids:
        return
    bundle = load_classifier(backbone, device)
    rows_fair, rows_hm = [], []
    for qid in grid_ids:
        rf = _comparison_row(
            eval_set, args, bundle, device, backbone, qid, fair=True, all_rows=all_rows
        )
        rh = _comparison_row(
            eval_set, args, bundle, device, backbone, qid, fair=False, all_rows=all_rows
        )
        if rf:
            rows_fair.append(rf)
        if rh:
            rows_hm.append(rh)
    ids_tag = "_".join(grid_ids[:6]) + ("_etc" if len(grid_ids) > 6 else "")
    if rows_fair:
        image_grid(
            fig_dir / f"FigC4_custom_fair_grid_{ids_tag}.png",
            rows_fair,
            suptitle=f"Custom fair comparison ({backbone}, n={len(rows_fair)})",
        )
    if rows_hm:
        image_grid(
            fig_dir / f"FigC5_custom_heatmap_grid_{ids_tag}.png",
            rows_hm,
            suptitle=f"Custom heatmap comparison ({backbone}, n={len(rows_hm)})",
        )


def run_section3_figures_only(args) -> None:
    """Regenerate Section 3 figures from saved all_runs.json (no re-computation)."""
    eval_set, images, device = resolve_run(args)
    out_root = args.out / "section3"
    grid_ids = _normalize_grid_ids(getattr(args, "grid_ids", None))

    if getattr(args, "comparisons_only", False):
        all_rows = load_json(out_root / "all_runs.json") if (out_root / "all_runs.json").exists() else []
        log(f"Comparisons-only: {len(images)} images, backbones={list(args.backbones)}")
        for backbone in args.backbones:
            fig_dir = figure_dir(args.out, "section3", backbone)
            _export_per_image_comparisons(eval_set, images, args, device, backbone, fig_dir, all_rows)
            if grid_ids:
                _fig_custom_grid(eval_set, args, device, fig_dir, backbone, grid_ids, all_rows)
        return

    if getattr(args, "heatmap_grid_only", False):
        all_rows = load_json(out_root / "all_runs.json") if (out_root / "all_runs.json").exists() else []
        log("Per-image heatmap comparisons from saved .pt CAM tensors")
        for backbone in args.backbones:
            fig_dir = figure_dir(args.out, "section3", backbone)
            _export_per_image_comparisons(eval_set, images, args, device, backbone, fig_dir, all_rows)
            if grid_ids:
                _fig_custom_grid(eval_set, args, device, fig_dir, backbone, grid_ids, all_rows)
        return

    js = out_root / "all_runs.json"
    if not js.exists():
        log("No section3/all_runs.json found — nothing to plot.")
        return
    rows = load_json(js)
    log(f"Figures-only: loaded {len(rows)} baseline rows from disk")
    _figures(eval_set, images, args, rows, defaultdict(list), device)


def _mean_curve(items):
    import numpy as np

    xs = items[0][0]
    ys = np.mean([it[1] for it in items], axis=0)
    auc = float(np.mean([it[2] for it in items]))
    return xs, ys, auc


def _figures(eval_set, images, args, rows, curves, device) -> None:
    backbones = list(args.backbones) or sorted({r["backbone"] for r in rows})
    for backbone in backbones:
        if not any(r["backbone"] == backbone for r in rows):
            log(f"[figures] skip {backbone}: no Section 3 runs")
            continue
        fig_dir = figure_dir(args.out, "section3", backbone)
        log(f"[figures] {backbone} -> {fig_dir}")
        _figures_for_backbone(eval_set, images, args, rows, curves, device, backbone, fig_dir)


def _figures_for_backbone(eval_set, images, args, rows, curves, device, backbone, fig_dir) -> None:
    matched = [r for r in rows if r["mode"] == "size_matched" and r["backbone"] == backbone]
    methods = []
    for r in matched:
        if r["method"] not in methods:
            methods.append(r["method"])
    by_cls = sorted({r["class_name"] for r in matched})
    groups = {}
    for cls in by_cls:
        means, stds = [], []
        for m in methods:
            vals = [r["insertion_auc"] for r in matched if r["method"] == m and r["class_name"] == cls]
            s = mean_std(vals)
            means.append(s["mean"] or 0)
            stds.append(s["std"] or 0)
        groups[cls] = (means, stds)
    if groups:
        grouped_bars(
            fig_dir / "FigC1_insertion_auc.png",
            methods,
            groups,
            "insertion AUC",
            f"Fig C1 — size-matched insertion AUC ({backbone})",
        )

    curve_plot = {}
    for (name, bb), items in curves.items():
        if bb != backbone or not items:
            continue
        curve_plot[name] = _mean_curve(items)
    if curve_plot:
        line_auc(
            fig_dir / "FigC2_insertion.png",
            curve_plot,
            "% pixels inserted",
            "normalized confidence of original class",
            f"Fig C2 — insertion curves ({backbone})",
        )

    labels = methods
    groups = {}
    for mode in ("own_threshold", "size_matched"):
        means, stds = [], []
        for m in labels:
            vals = [
                r["conf_e"]
                for r in rows
                if r["backbone"] == backbone and r["method"] == m and r["mode"] == mode
            ]
            s = mean_std(vals)
            means.append(s["mean"] or 0)
            stds.append(s["std"] or 0)
        groups[mode] = (means, stds)
    grouped_bars(
        fig_dir / "FigC3_own_vs_matched.png",
        labels,
        groups,
        "confidence preservation",
        f"Fig C3 — own-threshold vs size-matched ({backbone})",
    )

    for key, ylabel, fname in (
        ("ad", "Average Drop (↓)", "FigC6_AD.png"),
        ("ic", "Increase in Confidence (↑)", "FigC6_IC.png"),
        ("com", "Complexity / p_area (↓)", "FigC6_Com.png"),
        ("adcc", "ADCC (↑)", "FigC6_ADCC.png"),
        ("add", "ADD deletion drop (↑)", "FigC6_ADD.png"),
    ):
        means, stds = [], []
        for m in methods:
            vals = [
                r[key]
                for r in matched
                if r["method"] == m and r.get(key) is not None
            ]
            s = mean_std(vals)
            means.append(s["mean"] or 0)
            stds.append(s["std"] or 0)
        grouped_bars(
            fig_dir / fname,
            methods,
            {"size_matched": (means, stds)},
            ylabel,
            f"{ylabel} ({backbone})",
        )

    all_rows_list = list(rows)
    _export_per_image_comparisons(eval_set, images, args, device, backbone, fig_dir, all_rows_list)
    grid_ids = _normalize_grid_ids(getattr(args, "grid_ids", None))
    if grid_ids:
        _fig_custom_grid(eval_set, args, device, fig_dir, backbone, grid_ids, all_rows_list)


def _method_cols(bundle) -> list[str]:
    cols = ["EXP-CAM", "DD-CAM", "gradcam", "gradcam++", "layercam", "scorecam", "eigencam"]
    if bundle.cam_is_vit:
        cols.append("attention")
    cols.append("rise")
    return cols


def _method_file_stem(method: str) -> str:
    """Filesystem-safe stem: gradcam++ -> gradcampp, EXP-CAM -> EXP-CAM."""
    return method.replace("++", "pp")


def _expl_png_path(dest: Path, method: str) -> Path:
    return dest / "explanations" / f"{_method_file_stem(method)}.png"


def _load_cam_heatmap_panel(x, dest: Path, name: str, device):
    """Raw CAM heatmap on grey background from saved ``{name}.pt`` (``cam`` tensor)."""
    pt = dest / f"{name}.pt"
    if not pt.exists():
        return None
    blob = torch.load(pt, map_location=device)
    cam = blob["cam"].to(device)
    if cam.dim() == 2:
        cam = cam.unsqueeze(0).unsqueeze(0)
    elif cam.dim() == 3:
        cam = cam.unsqueeze(0)
    cam = cam.float()
    cam = cam / (cam.max() + 1e-12)
    return panel_cam_overlay(x, cam)


def main(argv=None):
    p = add_common_args(argparse.ArgumentParser(description="Section 3: fair baselines"))
    p.add_argument("--rise-masks", type=int, default=N_RISE_MASKS)
    p.add_argument(
        "--exp-steps",
        type=int,
        default=SECTION3_EXP_STEPS,
        help="EXP-CAM PixelMask steps when --train-expcam is set (default: 1000).",
    )
    p.add_argument(
        "--figures-only",
        action="store_true",
        help="Skip baseline computation; regenerate per-backbone figures from all_runs.json",
    )
    p.add_argument(
        "--heatmap-grid-only",
        action="store_true",
        help="With --figures-only: export per-image comparison PNGs only (alias for --comparisons-only).",
    )
    p.add_argument(
        "--comparisons-only",
        action="store_true",
        help="With --figures-only: export comparisons_fair/ & comparisons_heatmap/ for all images.",
    )
    p.add_argument(
        "--grid-ids",
        nargs="+",
        default=None,
        help="Image ids for a custom multi-row grid, e.g. --grid-ids 000 010 042",
    )
    p.add_argument(
        "--section0-source",
        default="section0",
        help="Section 0 folder under --out to reuse EXP-CAM masks (default: section0).",
    )
    p.add_argument(
        "--train-expcam",
        action="store_true",
        help="Retrain EXP-CAM here instead of reusing Section 0 masks.",
    )
    p.add_argument(
        "--image-illustration",
        action="store_true",
        help="Save per-image CAM figures. If omitted, only metric CSVs are written.",
    )
    args = p.parse_args(argv)
    if args.comparisons_only:
        args.figures_only = True
    if args.heatmap_grid_only:
        args.figures_only = True
        args.comparisons_only = True
    if args.figures_only:
        run_section3_figures_only(args)
    else:
        run_section3(args)


if __name__ == "__main__":
    main()
