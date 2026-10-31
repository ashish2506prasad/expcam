"""Section 2 — Minimality quantification.

2A reuses the Section 1 λ_area sweep.
2B trains with an explicit mask-budget constraint (projection by default).
p* is the smallest budget *found by the optimizer*, not a provable global min.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np

try:
    from ._path import ensure_pkg_path
except ImportError:
    from _path import ensure_pkg_path

ensure_pkg_path()

from cli import add_common_args, lambdas_from_args, resolve_run
from classifiers import load_classifier
from config import (
    AREA_SWEEP,
    BUDGETS,
    DEFAULT_LAMBDAS,
    FIDELITY_TAUS,
    KNAPSACK_BUDGETS,
    MASK_THRESH,
    MASK_THRESH_SWEEP,
    figure_dir,
)
from data import load_tensor, one_image_per_class, qualitative_image_ids
from io_utils import ensure_dir, load_json, mean_std, rows_to_excel, save_json, write_csv
from metrics import ef1_new_version, evaluate_explanation, fidelity_kl_component, min_sufficient_budget
from runner import load_masks, maybe_train_and_eval, run_dir
from baselines import topk_binarize
from train import TrainConfig, image_seed, log
from viz import (
    boxplot,
    caption,
    fidelity_vs_budget,
    image_grid,
    metric_vs_budget,
    panel_expl,
    panel_mask,
    panel_original,
    step_progression_grid,
    shaded_ef1_vs_p,
    triple_panel,
)

# Per-backbone gallery output folders under section2/
GALLERY_DIRS = {
    "resnet18": "gallery_resnet",
    "efficientnet_b0": "gallery_efficient",
    "vit_b_16": "gallery_vit",
}


def _figure_dir(out_root: Path, backbone: str) -> Path:
    return figure_dir(out_root, "section2", backbone)


def _gallery_dir(out_root: Path, backbone: str) -> Path:
    return out_root / "section2" / GALLERY_DIRS.get(backbone, f"gallery_{backbone}")


def _class_folder(rec: dict) -> str:
    name = rec["class_name"].replace(" ", "_")
    return f"{rec['wnid']}_{name}"


def run_section2(args) -> None:
    eval_set, images, device = resolve_run(args)
    out_root = args.out
    section = "section2"
    budgets = BUDGETS if not args.quick else (0.50, 0.10, 0.02)
    constraint = args.constraint
    rows_2b = []

    log(
        f"Section 2  device={device}  images={len(images)}  budgets={list(budgets)}  "
        f"constraint={constraint}  steps={args.steps}"
    )
    for backbone in args.backbones:
        log(f"Loading classifier {backbone} ...")
        bundle = load_classifier(backbone, device)
        for p in budgets:
            log(f"--- budget p={p:g} / {backbone} ---")
            for rec in images:
                cfg_name = f"budget_p{p:g}"
                out_dir = run_dir(out_root, section, backbone, cfg_name, rec["id"])
                lam = lambdas_from_args(args)
                lam["area"] = 0.0
                cfg = TrainConfig(
                    lambdas=lam,
                    steps=args.steps,
                    seed=image_seed(rec["id"], backbone, args.seed),
                    constraint=constraint,
                    p_target=p,
                    log_every=args.log_every,
                    no_ce=args.no_ce,
                    mask_model=getattr(args, "mask_model", None),
                )
                row = maybe_train_and_eval(
                    bundle,
                    rec,
                    Path(eval_set["root"]),
                    out_dir,
                    cfg,
                    skip_existing=args.skip_existing,
                    device=device,
                    extra_metrics={
                        "protocol": "budget",
                        "p_budget": p,
                        "p_gap": None,
                        "config": cfg_name,
                    },
                )
                row["p_budget"] = p
                row["p_gap"] = abs(p - row["p_actual"])
                save_json(out_dir / "metrics.json", row)
                rows_2b.append(row)

    # 2A (λ_area sweep) removed — Section 1 no longer runs sweeps.
    rows_2a = []

    save_json(out_root / section / "budget_runs.json", rows_2b)

    pstar_rows = []
    for backbone in args.backbones:
        for rec in images:
            subset = [r for r in rows_2b if r["backbone"] == backbone and r["image_id"] == rec["id"]]
            for tau in FIDELITY_TAUS:
                pstar = min_sufficient_budget(subset, tau)
                pstar_rows.append(
                    {
                        "image_id": rec["id"],
                        "wnid": rec["wnid"],
                        "class_name": rec["class_name"],
                        "backbone": backbone,
                        "tau": tau,
                        "p_star": pstar,
                        "note": "smallest mask budget at which the optimizer found a sufficiently faithful explanation; not a provable global minimum",
                    }
                )
    save_json(out_root / section / "p_star.json", pstar_rows)
    pstar_summary = []
    for tau in FIDELITY_TAUS:
        pstars_pct = [
            100.0 * r["p_star"]
            for r in pstar_rows
            if r["tau"] == tau and r["p_star"] is not None
        ]
        stats = mean_std(pstars_pct)
        pstar_summary.append(
            {
                "tau": tau,
                "mean_pct": stats["mean"],
                "std_pct": stats["std"],
                "n_images": len(pstars_pct),
                "definition": "p* = min{ p : top1_agreement(p) ≥ τ AND exp(−KL(p)) ≥ τ }",
                "note": (
                    "The smallest mask budget for which our optimization found a sufficiently "
                    "faithful explanation was p* — not a provable global minimum."
                ),
            }
        )
    save_json(out_root / section / "p_star_summary.json", pstar_summary)
    rows_to_excel(
        out_root / section / "section2_results.xlsx",
        {"budget": rows_2b, "p_star": pstar_rows, "p_star_summary": pstar_summary},
    )
    _figures(eval_set, images, args, rows_2b, pstar_rows, device, budgets)


def run_section2_figures_only(args) -> None:
    """Regenerate Section 2 figures from saved JSON (no retraining)."""
    eval_set, images, device = resolve_run(args)
    section = args.out / "section2"
    rows_2b = load_json(section / "budget_runs.json") if (section / "budget_runs.json").exists() else []
    pstar_rows = load_json(section / "p_star.json") if (section / "p_star.json").exists() else []
    if not rows_2b:
        log("No section2/budget_runs.json — run Section 2 training first.")
        return
    budgets = sorted({r["p_budget"] for r in rows_2b})
    log(f"Figures-only: {len(rows_2b)} budget runs, backbones={sorted({r['backbone'] for r in rows_2b})}")
    _figures(eval_set, images, args, rows_2b, pstar_rows, device, budgets)


def export_section2_gallery(args) -> None:
    """Export x | m | e side-by-side PNGs from existing Section 2 masks (no training)."""
    from io_utils import ensure_dir
    from runner import load_masks

    eval_set, images, device = resolve_run(args)
    section = args.out / "section2"
    rows_2b = load_json(section / "budget_runs.json") if (section / "budget_runs.json").exists() else []
    backbones = list(args.gallery_backbones or args.backbones)
    budget_filter = set(args.gallery_budgets) if args.gallery_budgets else None

    if rows_2b:
        available = sorted({r["backbone"] for r in rows_2b})
        if budget_filter is None:
            budget_filter = sorted({r["p_budget"] for r in rows_2b})
    else:
        available = [bb for bb in backbones if (section / bb).is_dir()]
        if budget_filter is None:
            budget_filter = sorted(
                {
                    float(d.name.replace("budget_p", ""))
                    for bb in available
                    for d in (section / bb).iterdir()
                    if d.is_dir() and d.name.startswith("budget_p")
                }
            )

    n_saved = 0
    n_skip = 0

    for backbone in backbones:
        if backbone not in available and not (section / backbone).is_dir():
            log(f"[gallery] skip {backbone}: no Section 2 runs")
            continue
        log(f"[gallery] {backbone} ...")
        bundle = load_classifier(backbone, device)
        gallery_root = _gallery_dir(args.out, backbone)

        for rec in images:
            x = load_tensor(rec, bundle.preprocess, device, root=Path(eval_set["root"]))
            class_dir = ensure_dir(gallery_root / _class_folder(rec))
            for p in budget_filter:
                out_dir = run_dir(args.out, "section2", backbone, f"budget_p{p:g}", rec["id"])
                if not (out_dir / "masks.pt").exists():
                    n_skip += 1
                    continue
                m_hard, _ = load_masks(out_dir, device)
                met = load_json(out_dir / "metrics.json")
                cap = caption(met["p_actual_pct"], met["conf_e"], met["top1_agreement"])
                fname = f"{rec['id']}_budget_p{p:g}.png"
                triple_panel(
                    class_dir / fname,
                    panel_original(x),
                    panel_mask(m_hard),
                    panel_expl(x, m_hard),
                    suptitle=f"{rec['class_name']}  img={rec['id']}  {backbone}  target p={p*100:g}%  {cap}",
                )
                n_saved += 1

    log(f"[gallery] saved {n_saved} triple-panels under section2/gallery_*  (skipped {n_skip} missing masks)")


def _series_for(rows, backbone, p_values, key, p_field="p_budget"):
    means, stds = [], []
    for p in p_values:
        subset = [r for r in rows if r["backbone"] == backbone and abs(r[p_field] - p) < 1e-9]
        if key == "exp_minus_kl":
            vals = [fidelity_kl_component(r) for r in subset]
        else:
            vals = [r[key] for r in subset]
        s = mean_std(vals)
        means.append(s["mean"] if s["mean"] is not None else 0.0)
        stds.append(s["std"] if s["std"] is not None else 0.0)
    return means, stds


def _figures(eval_set, images, args, rows_2b, pstar_rows, device, budgets) -> None:
    p_list = list(budgets)
    fid_keys = ("top1_agreement", "exp_minus_kl")
    backbones = list(args.backbones)
    if not backbones:
        backbones = sorted({r["backbone"] for r in rows_2b})

    for backbone in backbones:
        if not any(r["backbone"] == backbone for r in rows_2b):
            log(f"[figures] skip {backbone}: no Section 2 budget runs")
            continue
        fig_dir = _figure_dir(args.out, backbone)
        log(f"[figures] {backbone} -> {fig_dir}")
        series = {k: _series_for(rows_2b, backbone, p_list, k) for k in fid_keys}
        pa_m, pa_s = _series_for(rows_2b, backbone, p_list, "p_actual")
        for tau in FIDELITY_TAUS:
            pstars = [
                r["p_star"]
                for r in pstar_rows
                if r["backbone"] == backbone and r["tau"] == tau and r["p_star"] is not None
            ]
            p_star_mean = float(sum(pstars) / len(pstars)) if pstars else None
            fidelity_vs_budget(
                fig_dir / f"FigB1_minimality_curve_tau{tau:g}.png",
                p_list,
                series,
                pa_m,
                pa_s,
                p_star_mean,
                f"Fig B1 — Fidelity vs budget ({backbone}, τ={tau:g}). p* is optimizer-found, not a global min.",
                xscale="linear",
                tau=tau,
            )

        for tau in FIDELITY_TAUS:
            groups = defaultdict(list)
            for r in pstar_rows:
                if r["backbone"] == backbone and r["tau"] == tau and r["p_star"] is not None:
                    groups[r["class_name"]].append(100.0 * r["p_star"])
            if groups:
                boxplot(
                    fig_dir / f"FigB4_pstar_per_class_tau{tau:g}.png",
                    groups,
                    f"p* (% pixels) at τ={tau:g}",
                    f"Fig B4 — per-class p* ({backbone}, optimizer-found, τ={tau:g})",
                )

        _fig_b5(eval_set, args, device, fig_dir, budgets, backbone)


def _fig_b5(eval_set, args, device, fig_dir, budgets, backbone) -> None:
    from runner import load_masks

    qids = qualitative_image_ids(eval_set)
    id_to_rec = {r["id"]: r for r in eval_set["images"]}
    bundle = load_classifier(backbone, device)
    show_p = [p for p in (0.50, 0.20, 0.10, 0.05, 0.02, 0.01, 0.005) if p in set(budgets)]
    rows = []
    for qid in qids:
        rec = id_to_rec[qid]
        x = load_tensor(rec, bundle.preprocess, device, root=Path(eval_set["root"]))
        row = [(panel_original(x), f"{rec['class_name']}\noriginal")]
        for p in show_p:
            d = run_dir(args.out, "section2", backbone, f"budget_p{p:g}", qid)
            if not (d / "masks.pt").exists():
                continue
            m, _ = load_masks(d, device)
            met = load_json(d / "metrics.json")
            cap = caption(met["p_actual_pct"], met["conf_e"], met["top1_agreement"])
            row.append((panel_expl(x, m), f"p={p*100:g}%\n{cap}"))
        rows.append(row)
    if rows:
        image_grid(
            fig_dir / "FigB5_budget_strip.png",
            rows,
            suptitle=f"Fig B5 — explanation vs mask budget ({backbone})",
        )


def _section0_run_dir(out_root: Path, section0_source: str, backbone: str, image_id: str) -> Path:
    return out_root / section0_source / backbone / "ef1_full" / image_id


def _knapsack_mask_at_budget(mask_prob, p_budget: float):
    """Top-k binarize mask_prob to exactly p_budget fraction (0–1 knapsack / greedy top-k)."""
    scores = mask_prob[0, 0] if mask_prob.dim() == 4 else mask_prob.squeeze()
    n = scores.numel()
    k = max(1, int(round(float(p_budget) * n)))
    hard = topk_binarize(scores, k).view(1, 1, *scores.shape[-2:])
    return hard, k


def _eval_knapsack_point(bundle, x, mask_prob, p_budget: float, device) -> dict:
    mask_hard, k = _knapsack_mask_at_budget(mask_prob, p_budget)
    met = evaluate_explanation(bundle, x, mask_hard.to(device), mask_prob=mask_prob)
    ef1 = ef1_new_version(met["conf_e"], met["p_actual"])
    return {
        "p_budget": float(p_budget),
        "p_budget_pct": 100.0 * float(p_budget),
        "k_pixels": int(k),
        "p_actual": met["p_actual"],
        "p_actual_pct": met["p_actual_pct"],
        "abs_confidence_delta": met["abs_confidence_delta"],
        "kl_divergence": met["kl_divergence"],
        "ef1": ef1,
        "conf_e": met["conf_e"],
        "top1_agreement": met["top1_agreement"],
    }


def run_section2_knapsack(args) -> None:
    """Section 2 knapsack: sweep p on Section 0 best mask_prob (no retrain)."""
    eval_set, _images, device = resolve_run(args)
    out_root = args.out
    section0_source = args.section0_source
    budgets = tuple(args.knapsack_budgets) if args.knapsack_budgets else KNAPSACK_BUDGETS
    subset = one_image_per_class(eval_set)
    section = "section2"
    rows = []

    log(
        f"Section 2 knapsack  device={device}  section0={section0_source}  "
        f"images={len(subset)}  budgets={[f'{100*b:g}%' for b in budgets]}"
    )

    for backbone in args.backbones:
        bundle = load_classifier(backbone, device)
        for rec in subset:
            s0_dir = _section0_run_dir(out_root, section0_source, backbone, rec["id"])
            if not (s0_dir / "masks.pt").exists():
                log(f"[skip] {backbone} img={rec['id']}: no Section 0 masks at {s0_dir}")
                continue
            s0_met = load_json(s0_dir / "metrics.json") if (s0_dir / "metrics.json").exists() else {}
            p_model_pct = float(s0_met.get("p_actual_pct", 100.0 * s0_met.get("p_actual", 0.0)))
            _, mask_prob = load_masks(s0_dir, device)
            x = load_tensor(rec, bundle.preprocess, device, root=Path(eval_set["root"]))
            log(f"[knapsack] {backbone} img={rec['id']} {rec['class_name']}  model p={p_model_pct:.1f}%")
            for p in budgets:
                pt = _eval_knapsack_point(bundle, x, mask_prob, p, device)
                rows.append(
                    {
                        "image_id": rec["id"],
                        "wnid": rec["wnid"],
                        "class_name": rec["class_name"],
                        "backbone": backbone,
                        "section0_source": section0_source,
                        "p_model_pct": p_model_pct,
                        "best_ef1_step": s0_met.get("best_ef1_step"),
                        "protocol": "knapsack_section0",
                        **pt,
                    }
                )

    if not rows:
        log("No knapsack rows computed — run Section 0 with --ef1-new-version first.")
        return

    save_json(out_root / section / "knapsack_runs.json", rows)
    rows_to_excel(out_root / section / "section2_knapsack_results.xlsx", {"knapsack": rows})
    _figures_knapsack(args, eval_set, subset, rows, budgets, device)


def run_section2_knapsack_figures_only(args) -> None:
    eval_set, _, device = resolve_run(args)
    js = args.out / "section2" / "knapsack_runs.json"
    if not js.exists():
        log("No section2/knapsack_runs.json — run Section 2 knapsack first.")
        return
    rows = load_json(js)
    budgets = sorted({r["p_budget"] for r in rows})
    subset = one_image_per_class(eval_set)
    log(f"Knapsack figures-only: {len(rows)} rows")
    _figures_knapsack(args, eval_set, subset, rows, budgets, device)


def _figures_knapsack(args, eval_set, subset, rows, budgets, device) -> None:
    """Per-image plots: p vs |Δconf|, KL, EF1 (linear x-axis only)."""
    metrics = (
        ("abs_conf_delta", "abs_confidence_delta", "|Δconf|"),
        ("kl_divergence", "kl_divergence", "KL divergence"),
        ("ef1", "ef1", "EF1 score"),
    )
    backbones = list(args.backbones) or sorted({r["backbone"] for r in rows})

    for backbone in backbones:
        out_dir = ensure_dir(_figure_dir(args.out, backbone))
        n_plots = 0
        for rec in subset:
            qid = rec["id"]
            img_rows = sorted(
                [r for r in rows if r["backbone"] == backbone and r["image_id"] == qid],
                key=lambda r: r["p_budget"],
            )
            if not img_rows:
                continue
            p_model_pct = float(img_rows[0].get("p_model_pct", 0.0))
            p_fracs = [r["p_budget"] for r in img_rows]
            cname = rec["class_name"]
            for fname, key, ylabel in metrics:
                vals = [r[key] for r in img_rows]
                metric_vs_budget(
                    out_dir / f"{fname}_{qid}.png",
                    p_fracs,
                    vals,
                    ylabel=ylabel,
                    title=f"{ylabel} vs mask budget — {cname} ({backbone}, img {qid})",
                    p_model_pct=p_model_pct,
                    xscale="linear",
                )
                n_plots += 1
        log(f"[figures] {backbone}: {n_plots} knapsack plots -> {out_dir}")

    _figures_knapsack_mask_grids(args, eval_set, subset, rows, budgets, device)


def _chunk_panels(panels: list, cols_per_row: int) -> list:
    out = []
    for i in range(0, len(panels), cols_per_row):
        out.append(panels[i : i + cols_per_row])
    return out


def _figures_knapsack_mask_grids(args, eval_set, subset, rows, budgets, device) -> None:
    """One mask grid per image per backbone; panel caption is p= only."""
    section0_source = args.section0_source
    grid_backbones = [bb for bb in ("resnet18", "vit_b_16") if any(r["backbone"] == bb for r in rows)]
    if args.backbones:
        grid_backbones = [bb for bb in grid_backbones if bb in args.backbones]
    if not grid_backbones:
        grid_backbones = sorted({r["backbone"] for r in rows})

    cols_per_row = int(getattr(args, "knapsack_grid_cols", 6) or 6)
    bundles = {bb: load_classifier(bb, device) for bb in grid_backbones}
    n_saved = 0

    for backbone in grid_backbones:
        out_dir = ensure_dir(_figure_dir(args.out, backbone) / "mask_grids")
        bundle = bundles[backbone]
        for rec in subset:
            qid = rec["id"]
            cname = rec["class_name"].replace(" ", "_")
            s0_dir = _section0_run_dir(args.out, section0_source, backbone, qid)
            if not (s0_dir / "masks.pt").exists():
                continue
            _, mask_prob = load_masks(s0_dir, device)
            x = load_tensor(rec, bundle.preprocess, device, root=Path(eval_set["root"]))
            panels = []
            for p in budgets:
                mask_hard, _ = _knapsack_mask_at_budget(mask_prob, p)
                panels.append((panel_expl(x, mask_hard.to(device)), f"p={100 * p:g}%"))
            grid_rows = _chunk_panels(panels, cols_per_row)
            step_progression_grid(
                out_dir / f"mask_grid_{qid}_{cname}.png",
                grid_rows,
            )
            n_saved += 1

    log(f"[figures] mask grids: {n_saved} ({len(grid_backbones)} backbones × images) -> figures_*/mask_grids/")


def _default_area() -> float:
    return float(DEFAULT_LAMBDAS.get("area", 100.0))


def _section0_dir_for(args, backbone: str, image_id: str) -> Path:
    src = getattr(args, "section0_source", "section0")
    d = _section0_run_dir(args.out, src, backbone, image_id)
    if (d / "masks.pt").exists():
        return d
    for alt in ("section0", "section0_new_score"):
        d = _section0_run_dir(args.out, alt, backbone, image_id)
        if (d / "masks.pt").exists():
            return d
    return d


def _train_area_run(bundle, rec, eval_root, args, backbone, lam: float, device):
    """Train (or reuse Section 0) at a given λ_area. Works for unet / tiny_unet / none."""
    cfg_name = f"area_{lam:g}"
    out_dir = run_dir(args.out, "section2", backbone, cfg_name, rec["id"])
    reuse = abs(lam - _default_area()) < 1e-6
    s0 = _section0_dir_for(args, backbone, rec["id"]) if reuse else None
    if reuse and s0 is not None and (s0 / "masks.pt").exists() and (out_dir / "masks.pt").exists() is False:
        import shutil

        ensure_dir(out_dir)
        for name in ("masks.pt", "metrics.json", "train_meta.json"):
            if (s0 / name).exists():
                shutil.copy2(s0 / name, out_dir / name)
        log(f"[sensitivity] reused Section 0 for λ_area={lam:g}  img={rec['id']}")
    lam_map = lambdas_from_args(args)
    lam_map["area"] = float(lam)
    cfg = TrainConfig(
        lambdas=lam_map,
        steps=args.steps,
        seed=image_seed(rec["id"], backbone, args.seed),
        log_every=args.log_every,
        no_ce=args.no_ce,
        record_ef1=True,
        keep_best_ef1=True,
        mask_model=getattr(args, "mask_model", None),
        mask_thresh=MASK_THRESH,
    )
    row = maybe_train_and_eval(
        bundle,
        rec,
        eval_root,
        out_dir,
        cfg,
        skip_existing=args.skip_existing,
        device=device,
        extra_metrics={"protocol": "lambda_area_sweep", "lambda_area": lam, "config": cfg_name},
    )
    return out_dir, row


def run_sensitivity(args) -> None:
    """λ_area and mask_thresh sweeps: EF1 vs p_area, star at the trained operating point."""
    eval_set, images, device = resolve_run(args)
    eval_root = Path(eval_set["root"])
    area_vals = tuple(args.area_sweep) if getattr(args, "area_sweep", None) else AREA_SWEEP
    thresh_vals = tuple(args.thresh_sweep) if getattr(args, "thresh_sweep", None) else MASK_THRESH_SWEEP
    budgets = tuple(args.knapsack_budgets) if args.knapsack_budgets else KNAPSACK_BUDGETS
    if args.quick:
        area_vals = area_vals[:3]
        thresh_vals = thresh_vals[:3]
        budgets = (0.05, 0.10, 0.20, 0.50)
        images = images[: min(4, len(images))]
    rows = []
    log(
        f"Section 2 sensitivity  λ_area={list(area_vals)}  thresh={list(thresh_vals)}  "
        f"images={len(images)}  mask_model={getattr(args, 'mask_model', None)}"
    )
    for backbone in args.backbones:
        bundle = load_classifier(backbone, device)
        fig_dir = ensure_dir(_figure_dir(args.out, backbone) / "sensitivity")
        for rec in images:
            area_series = {}
            area_stars = {}
            default_prob = None
            x = load_tensor(rec, bundle.preprocess, device, root=eval_root)
            for lam in area_vals:
                out_dir, row = _train_area_run(bundle, rec, eval_root, args, backbone, lam, device)
                _, mask_prob = load_masks(out_dir, device)
                if abs(lam - _default_area()) < 1e-6:
                    default_prob = mask_prob
                knapsack = [_eval_knapsack_point(bundle, x, mask_prob, p, device) for p in budgets]
                lab = f"λ_area={lam:g}"
                area_series[lab] = ([k["p_budget"] for k in knapsack], [k["ef1"] for k in knapsack])
                area_stars[lab] = (float(row.get("p_actual", 0.0)), float(row.get("ef1", 0.0)))
                for pt in knapsack:
                    rows.append(
                        {
                            "image_id": rec["id"],
                            "class_name": rec["class_name"],
                            "backbone": backbone,
                            "sweep": "lambda_area",
                            "lambda_area": lam,
                            "mask_thresh": MASK_THRESH,
                            **pt,
                            "p_model": row.get("p_actual"),
                            "ef1_model": row.get("ef1"),
                        }
                    )
            shaded_ef1_vs_p(
                fig_dir / f"ef1_vs_p_lambda_area_{rec['id']}.png",
                area_series,
                stars=area_stars,
                title=f"EF1 vs p_area — λ_area sweep  {rec['class_name']} ({rec['id']}, {backbone})",
            )

            if default_prob is None:
                default_prob = mask_prob
            knapsack = [_eval_knapsack_point(bundle, x, default_prob, p, device) for p in budgets]
            thresh_series = {
                "top-k ranking": ([k["p_budget"] for k in knapsack], [k["ef1"] for k in knapsack])
            }
            thresh_stars = {}
            t_ps, t_ef = [], []
            for t in thresh_vals:
                hard = (default_prob > float(t)).float()
                if hard.dim() == 2:
                    hard = hard.view(1, 1, *hard.shape)
                met = evaluate_explanation(bundle, x, hard.to(device), mask_prob=default_prob)
                lab = f"t={t:g}"
                thresh_stars[lab] = (met["p_actual"], met["ef1"])
                t_ps.append(met["p_actual"])
                t_ef.append(met["ef1"])
                rows.append(
                    {
                        "image_id": rec["id"],
                        "class_name": rec["class_name"],
                        "backbone": backbone,
                        "sweep": "mask_thresh",
                        "lambda_area": _default_area(),
                        "mask_thresh": t,
                        "p_budget": met["p_actual"],
                        "ef1": met["ef1"],
                        "p_actual": met["p_actual"],
                        "conf_e": met["conf_e"],
                        "top1_agreement": met["top1_agreement"],
                    }
                )
            order = sorted(range(len(t_ps)), key=lambda i: t_ps[i])
            thresh_series["mask thresh"] = ([t_ps[i] for i in order], [t_ef[i] for i in order])
            for t in thresh_vals:
                lab = f"t={t:g}"
                px, py = thresh_stars[lab]
                thresh_series[lab] = ([px], [py])
            shaded_ef1_vs_p(
                fig_dir / f"ef1_vs_p_mask_thresh_{rec['id']}.png",
                thresh_series,
                stars=thresh_stars,
                title=f"EF1 vs p_area — mask_thresh sweep  {rec['class_name']} ({rec['id']}, {backbone})",
                base_rgb=(0.55, 0.28, 0.22),
            )

        _sensitivity_sweep_plots(args, rows, backbone, area_vals, thresh_vals, budgets)

    section = args.out / "section2"
    save_json(section / "sensitivity_runs.json", rows)
    write_csv(section / "sensitivity_runs.csv", rows)
    log(f"Sensitivity JSON/CSV -> {section / 'sensitivity_runs.csv'}")


def _sensitivity_sweep_plots(args, rows, backbone, area_vals, thresh_vals, budgets) -> None:
    fig_dir = ensure_dir(_figure_dir(args.out, backbone) / "sensitivity")
    sub = [r for r in rows if r["backbone"] == backbone]
    area_series, area_stars = {}, {}
    for lam in area_vals:
        lab = f"λ_area={lam:g}"
        xs, ys = [], []
        for p in budgets:
            vals = [
                r["ef1"]
                for r in sub
                if r.get("sweep") == "lambda_area"
                and abs(r.get("lambda_area", -1) - lam) < 1e-9
                and abs(r.get("p_budget", -1) - p) < 1e-9
            ]
            if vals:
                xs.append(p)
                ys.append(float(sum(vals) / len(vals)))
        if xs:
            area_series[lab] = (xs, ys)
        model_p, model_e = [], []
        for r in sub:
            if r.get("sweep") == "lambda_area" and abs(r.get("lambda_area", -1) - lam) < 1e-9:
                if r.get("p_model") is not None and r.get("ef1_model") is not None:
                    model_p.append(float(r["p_model"]))
                    model_e.append(float(r["ef1_model"]))
        if model_p:
            area_stars[lab] = (sum(model_p) / len(model_p), sum(model_e) / len(model_e))
    if area_series:
        shaded_ef1_vs_p(
            fig_dir / "ef1_vs_p_lambda_area_mean.png",
            area_series,
            stars=area_stars,
            title=f"EF1 vs p_area — λ_area sweep mean ({backbone})",
        )

    thresh_series, thresh_stars = {}, {}
    t_ps, t_ef = [], []
    for t in thresh_vals:
        lab = f"t={t:g}"
        vals_p, vals_e = [], []
        for r in sub:
            if r.get("sweep") == "mask_thresh" and abs(float(r.get("mask_thresh", -1)) - t) < 1e-9:
                vals_p.append(float(r["p_actual"]))
                vals_e.append(float(r["ef1"]))
        if vals_p:
            px, py = sum(vals_p) / len(vals_p), sum(vals_e) / len(vals_e)
            thresh_stars[lab] = (px, py)
            t_ps.append(px)
            t_ef.append(py)
            thresh_series[lab] = ([px], [py])
    if t_ps:
        order = sorted(range(len(t_ps)), key=lambda i: t_ps[i])
        thresh_series["mask thresh"] = ([t_ps[i] for i in order], [t_ef[i] for i in order])
        shaded_ef1_vs_p(
            fig_dir / "ef1_vs_p_mask_thresh_mean.png",
            thresh_series,
            stars=thresh_stars,
            title=f"EF1 vs p_area — mask_thresh sweep mean ({backbone})",
            base_rgb=(0.55, 0.28, 0.22),
        )


def run_sensitivity_figures_only(args) -> None:
    js = args.out / "section2" / "sensitivity_runs.json"
    if not js.exists():
        log("No section2/sensitivity_runs.json — run --sensitivity first.")
        return
    rows = load_json(js)
    area_vals = sorted({float(r["lambda_area"]) for r in rows if r.get("sweep") == "lambda_area"})
    thresh_vals = sorted({float(r["mask_thresh"]) for r in rows if r.get("sweep") == "mask_thresh"})
    budgets = sorted({float(r["p_budget"]) for r in rows if r.get("sweep") == "lambda_area"})
    for backbone in args.backbones or sorted({r["backbone"] for r in rows}):
        _sensitivity_sweep_plots(args, rows, backbone, area_vals, thresh_vals, budgets)
        log(f"[figures] sensitivity mean plots -> {_figure_dir(args.out, backbone) / 'sensitivity'}")


def main(argv=None):
    p = add_common_args(argparse.ArgumentParser(description="Section 2: minimality"))
    p.add_argument("--constraint", choices=("project", "lagrangian"), default="project")
    p.add_argument(
        "--figures-only",
        action="store_true",
        help="Skip training; regenerate per-backbone figures from budget_runs.json",
    )
    p.add_argument(
        "--export-gallery",
        action="store_true",
        help="Export x|m|e side-by-side PNGs into class folders from existing masks (no training)",
    )
    p.add_argument(
        "--gallery-backbones",
        nargs="+",
        default=None,
        help="Backbones for --export-gallery (default: same as --backbones)",
    )
    p.add_argument(
        "--gallery-budgets",
        nargs="+",
        type=float,
        default=None,
        help="Budget subset for gallery export (default: all budgets in budget_runs.json)",
    )
    p.add_argument(
        "--from-section0-knapsack",
        action="store_true",
        help="Knapsack sweep on Section 0 best mask_prob (no retrain); write knapsack plots.",
    )
    p.add_argument(
        "--section0-source",
        default="section0_new_score",
        help="Section 0 run folder under --out (default: section0_new_score).",
    )
    p.add_argument(
        "--knapsack-budgets",
        nargs="+",
        type=float,
        default=None,
        help="Mask budgets for knapsack sweep as fractions (default: 0.005…0.50).",
    )
    p.add_argument(
        "--sensitivity",
        action="store_true",
        help="Sweep λ_area and mask_thresh; plot EF1 vs p_area (stars = model p).",
    )
    p.add_argument(
        "--area-sweep",
        nargs="+",
        type=float,
        default=None,
        help="λ_area values (default: config.AREA_SWEEP).",
    )
    p.add_argument(
        "--thresh-sweep",
        nargs="+",
        type=float,
        default=None,
        help="mask_thresh values (default: config.MASK_THRESH_SWEEP).",
    )
    args = p.parse_args(argv)
    if args.sensitivity:
        if args.figures_only:
            run_sensitivity_figures_only(args)
        else:
            run_sensitivity(args)
    elif args.from_section0_knapsack:
        if args.figures_only:
            run_section2_knapsack_figures_only(args)
        else:
            run_section2_knapsack(args)
    elif args.export_gallery:
        export_section2_gallery(args)
    elif args.figures_only:
        run_section2_figures_only(args)
    else:
        run_section2(args)


if __name__ == "__main__":
    main()
