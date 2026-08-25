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

from cli import add_common_args, resolve_run
from classifiers import load_classifier
from config import BUDGETS, DEFAULT_LAMBDAS, FIDELITY_TAUS, figure_dir
from data import load_tensor, qualitative_image_ids
from io_utils import load_json, mean_std, rows_to_excel, save_json
from metrics import fidelity_kl_component, min_sufficient_budget
from runner import maybe_train_and_eval, run_dir
from train import TrainConfig, image_seed, log
from viz import (
    boxplot,
    caption,
    fidelity_vs_budget,
    image_grid,
    overlay_two_families,
    panel_expl,
    panel_mask,
    panel_original,
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
                lam = dict(DEFAULT_LAMBDAS)
                lam["area"] = 0.0
                cfg = TrainConfig(
                    lambdas=lam,
                    steps=args.steps,
                    seed=image_seed(rec["id"], backbone, args.seed),
                    constraint=constraint,
                    p_target=p,
                    log_every=args.log_every,
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

    # 2A: load λ_area sweep from section 1 if present
    rows_2a = []
    s1 = out_root / "section1" / "all_runs.json"
    if s1.exists():
        for r in load_json(s1):
            if r.get("protocol") == "sweep" and r.get("term") == "area":
                r = dict(r)
                r["p_budget"] = r["p_actual"]
                rows_2a.append(r)

    save_json(out_root / section / "budget_runs.json", rows_2b)
    save_json(out_root / section / "area_sweep_reuse.json", rows_2a)

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
        {"budget": rows_2b, "area_sweep_2A": rows_2a, "p_star": pstar_rows, "p_star_summary": pstar_summary},
    )
    _figures(eval_set, images, args, rows_2a, rows_2b, pstar_rows, device, budgets)


def run_section2_figures_only(args) -> None:
    """Regenerate Section 2 figures from saved JSON (no retraining)."""
    eval_set, images, device = resolve_run(args)
    section = args.out / "section2"
    rows_2b = load_json(section / "budget_runs.json") if (section / "budget_runs.json").exists() else []
    rows_2a = load_json(section / "area_sweep_reuse.json") if (section / "area_sweep_reuse.json").exists() else []
    pstar_rows = load_json(section / "p_star.json") if (section / "p_star.json").exists() else []
    if not rows_2b:
        log("No section2/budget_runs.json — run Section 2 training first.")
        return
    budgets = sorted({r["p_budget"] for r in rows_2b})
    log(f"Figures-only: {len(rows_2b)} budget runs, backbones={sorted({r['backbone'] for r in rows_2b})}")
    _figures(eval_set, images, args, rows_2a, rows_2b, pstar_rows, device, budgets)


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


def _figures(eval_set, images, args, rows_2a, rows_2b, pstar_rows, device, budgets) -> None:
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
            for suffix, xscale in (("_log", "log"), ("_linear", "linear")):
                fidelity_vs_budget(
                    fig_dir / f"FigB1_minimality_curve_tau{tau:g}{suffix}.png",
                    p_list,
                    series,
                    pa_m,
                    pa_s,
                    p_star_mean,
                    f"Fig B1 — Fidelity vs budget ({backbone}, τ={tau:g}). p* is optimizer-found, not a global min.",
                    xscale=xscale,
                    tau=tau,
                )

        rows_2a_bb = [r for r in rows_2a if r.get("backbone") == backbone]
        if rows_2a_bb:
            from collections import defaultdict as dd

            by_mult = dd(list)
            for r in rows_2a_bb:
                by_mult[float(r.get("mult", r.get("lambda_value", 0)))].append(r)
            fam_a = {k: ([], []) for k in fid_keys}
            p_a = []
            for mult in sorted(by_mult):
                grp = by_mult[mult]
                p_a.append(float(np.mean([g["p_actual"] for g in grp])))
                for k in fid_keys:
                    if k == "exp_minus_kl":
                        vals = [fidelity_kl_component(g) for g in grp]
                    else:
                        vals = [g[k] for g in grp]
                    s = mean_std(vals)
                    fam_a[k][0].append(s["mean"] or 0)
                    fam_a[k][1].append(s["std"] or 0)
            fam_b = {k: _series_for(rows_2b, backbone, p_list, k) for k in fid_keys}
            for suffix, xscale in (("_log", "log"), ("_linear", "linear")):
                overlay_two_families(
                    fig_dir / f"FigB2_2A_vs_2B{suffix}.png",
                    p_list,
                    fam_b,
                    fam_a,
                    "2B budget constraint",
                    "2A λ_area sweep (x = achieved p_actual)",
                    f"Fig B2 — 2A vs 2B ({backbone})"
                    + (" (log x)" if xscale != "linear" else " (linear x)"),
                    p_b=p_a,
                    xscale=xscale,
                )
            save_json(
                fig_dir / "figB2_2A_points.json",
                {
                    "backbone": backbone,
                    "p_actual_mean": p_a,
                    "family_2A": {k: {"mean": fam_a[k][0], "std": fam_a[k][1]} for k in fid_keys},
                },
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
    args = p.parse_args(argv)
    if args.export_gallery:
        export_section2_gallery(args)
    elif args.figures_only:
        run_section2_figures_only(args)
    else:
        run_section2(args)


if __name__ == "__main__":
    main()
