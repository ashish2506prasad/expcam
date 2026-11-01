"""Sequential loss ablation (Section 1).

Trains cumulative loss stages + Full model (300 steps, CE on, best-EF1 checkpoint).
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

try:
    from ._path import ensure_pkg_path
except ImportError:
    from _path import ensure_pkg_path

ensure_pkg_path()

from cli import add_common_args, lambdas_from_args, resolve_run
from classifiers import load_classifier
from config import SEQUENTIAL_PLOT_ORDER, figure_dir, sequential_configs
from data import load_tensor, qualitative_image_ids
from io_utils import load_json, rows_to_excel, save_json
from metrics import ef1_new_version
from runner import maybe_train_and_eval, run_dir
from train import TrainConfig, image_seed, log
from viz import ablation_bars, image_grid, panel_expl, panel_original

SEQ_DISPLAY = {
    "seq_act": "L_act",
    "seq_act_kl": "L_act + L_kl",
    "seq_act_kl_ce": "L_act + L_kl + L_ce",
    "seq_act_kl_ce_min": "L_act + L_kl + L_ce + L_area+bin+tv",
    "Full": "Full (L_act + L_kl + L_ce + L_area+bin+tv + L_rob)",
}

SEQ_QUALITATIVE_IDS = {
    "vit_b_16": ["000", "020", "021", "031", "051", "080", "091"],
    "resnet18": ["010", "011", "030", "051", "061", "080", "081"],
}

INDIVIDUAL_METRICS = (
    ("ef1", "EF1 score", 1.0),
    ("top1_agreement", "top-1 agreement %", 100.0),
)

SECTION = "section1"


def _row_ef1(row: dict) -> float:
    if "conf_e" in row and "p_actual" in row:
        return ef1_new_version(row["conf_e"], row["p_actual"])
    return float(row.get("ef1", 0.0))


def _row_metric(row: dict, key: str, scale: float = 1.0) -> float:
    if key == "ef1":
        return _row_ef1(row)
    return scale * float(row[key])


def _train_cfg(args, rec, backbone, **extra) -> TrainConfig:
    return TrainConfig(
        steps=args.steps,
        seed=image_seed(rec["id"], backbone, args.seed),
        log_every=args.log_every,
        no_ce=False,
        record_ef1=True,
        keep_best_ef1=True,
        **extra,
    )


def _collect_rows_from_disk(out_root: Path) -> list:
    rows = []
    for path in sorted((out_root / SECTION).rglob("metrics.json")):
        rel = path.relative_to(out_root / SECTION)
        if len(rel.parts) < 3:
            continue
        rows.append(load_json(path))
    return _filter_ablation_rows(rows)


def _filter_ablation_rows(rows: list) -> list:
    """Keep sequential runs; accept legacy Full from loo_Full as sequential Full."""
    out = []
    for r in rows:
        protocol = r.get("protocol")
        if protocol == "sequential":
            out.append(r)
        elif protocol == "loo" and r.get("config") == "Full":
            row = dict(r)
            row["protocol"] = "sequential"
            row["config"] = "Full"
            out.append(row)
    return out


def _full_cfg_name(out_root: Path, backbone: str, qid: str) -> str:
    for cfg in ("Full", "loo_Full"):
        d = run_dir(out_root, SECTION, backbone, cfg, qid)
        if (d / "masks.pt").exists():
            return cfg
    return "Full"


def run_ablation(args) -> None:
    eval_set, images, device = resolve_run(args)
    out_root = args.out
    all_rows = []
    sequential = sequential_configs(lambdas_from_args(args))
    if args.quick:
        sequential = {k: sequential[k] for k in ("seq_act", "seq_act_kl_ce_min", "Full") if k in sequential}

    log(
        f"Section 1  device={device}  images={len(images)}  backbones={list(args.backbones)}  "
        f"steps={args.steps}  sequential={len(sequential)}"
    )

    for backbone in args.backbones:
        log(f"Loading classifier {backbone} ...")
        bundle = load_classifier(backbone, device)
        for cfg_name, lambdas in sequential.items():
            log(f"--- {cfg_name} / {backbone} ---")
            for rec in images:
                out_dir = run_dir(out_root, SECTION, backbone, cfg_name, rec["id"])
                cfg = _train_cfg(args, rec, backbone, lambdas=lambdas)
                row = maybe_train_and_eval(
                    bundle,
                    rec,
                    Path(eval_set["root"]),
                    out_dir,
                    cfg,
                    skip_existing=args.skip_existing,
                    device=device,
                    extra_metrics={"protocol": "sequential", "config": cfg_name},
                )
                all_rows.append(row)

    save_json(out_root / SECTION / "all_runs.json", all_rows)
    rows_to_excel(out_root / SECTION / "section1_results.xlsx", {"sequential": all_rows, "all": all_rows})
    _make_figures(eval_set, args, all_rows, device)


def run_ablation_figures_only(args) -> None:
    eval_set, _, device = resolve_run(args)
    out_root = args.out
    all_rows = _collect_rows_from_disk(out_root)
    if not all_rows:
        log(f"No {SECTION} sequential metrics.json found — nothing to plot.")
        return
    log(f"Figures-only: loaded {len(all_rows)} sequential runs from disk")
    save_json(out_root / SECTION / "all_runs.json", all_rows)
    rows_to_excel(out_root / SECTION / "section1_results.xlsx", {"sequential": all_rows, "all": all_rows})
    _make_figures(eval_set, args, all_rows, device)


def _seq_qualitative_ids(args, backbone: str, eval_set) -> list[str]:
    if getattr(args, "seq_qualitative_ids", None):
        return list(args.seq_qualitative_ids)
    if backbone in SEQ_QUALITATIVE_IDS:
        return list(SEQ_QUALITATIVE_IDS[backbone])
    return qualitative_image_ids(eval_set)


def _make_figures(eval_set, args, all_rows, device) -> None:
    backbones = list(args.backbones) or sorted({r["backbone"] for r in all_rows})
    id_to_rec = {r["id"]: r for r in eval_set["images"]}
    for backbone in backbones:
        if not any(r["backbone"] == backbone for r in all_rows):
            continue
        fig_dir = figure_dir(args.out, SECTION, backbone)
        ind_dir = fig_dir / "individual"
        seq_qids = _seq_qualitative_ids(args, backbone, eval_set)
        log(f"[figures] {backbone} SEQ -> {fig_dir}  ({len(seq_qids)} images)")
        _figures_sequential_individual(all_rows, backbone, ind_dir, seq_qids, id_to_rec)
        _fig_qualitative_seq(eval_set, args, device, fig_dir, backbone, seq_qids)


def _figures_sequential_individual(all_rows, backbone, ind_dir, qids, id_to_rec) -> None:
    seq_rows = [r for r in all_rows if r.get("protocol") == "sequential" and r["backbone"] == backbone]
    by_cfg = defaultdict(list)
    for r in seq_rows:
        by_cfg[r["config"]].append(r)
    cfg_keys = [c for c in SEQUENTIAL_PLOT_ORDER if c in by_cfg]
    if not cfg_keys:
        return
    plot_labels = [SEQ_DISPLAY.get(c, c) for c in cfg_keys]
    ind_dir.mkdir(parents=True, exist_ok=True)
    for qid in qids:
        rec = id_to_rec.get(qid)
        if rec is None:
            continue
        cname = rec["class_name"]
        for key, ylabel, scale in INDIVIDUAL_METRICS:
            vals = []
            for cfg in cfg_keys:
                hit = next((r for r in by_cfg.get(cfg, []) if r["image_id"] == qid), None)
                vals.append(_row_metric(hit, key, scale) if hit else 0.0)
            ablation_bars(
                ind_dir / f"SEQ_{key}_{qid}.png",
                plot_labels,
                vals,
                ylabel,
                f"Sequential — {ylabel} ({cname}, img {qid}, {backbone})",
                stds=None,
            )


def _fig_qualitative_seq(eval_set, args, device, fig_dir, backbone, qids) -> None:
    from runner import load_masks

    id_to_rec = {r["id"]: r for r in eval_set["images"]}
    bundle = load_classifier(backbone, device)
    stage_cfgs = [(SEQ_DISPLAY[c], c) for c in SEQUENTIAL_PLOT_ORDER]
    rows = []
    for qid in qids:
        rec = id_to_rec.get(qid)
        if rec is None:
            continue
        x = load_tensor(rec, bundle.preprocess, device, root=Path(eval_set["root"]))
        row = [(panel_original(x), f"{rec['class_name']}\nimg {qid}\noriginal")]
        for label, cfg_name in stage_cfgs:
            run_cfg = _full_cfg_name(args.out, backbone, qid) if cfg_name == "Full" else cfg_name
            d = run_dir(args.out, SECTION, backbone, run_cfg, qid)
            if not (d / "masks.pt").exists():
                continue
            m_hard, _ = load_masks(d, device)
            met = load_json(d / "metrics.json")
            cap = (
                f"EF1={_row_ef1(met):.2f}\n"
                f"p={met['p_actual_pct']:.1f}%  "
                f"Δconf={met['confidence_delta']:.2f}"
            )
            row.append((panel_expl(x, m_hard), f"{label}\n{cap}"))
        if len(row) > 1:
            rows.append(row)
            out_sub = fig_dir / "seq_qualitative"
            out_sub.mkdir(parents=True, exist_ok=True)
            image_grid(
                out_sub / f"SEQ_qualitative_{qid}.png",
                [row],
                suptitle=f"Sequential — {rec['class_name']} img {qid} ({backbone})",
            )
    if rows:
        image_grid(
            fig_dir / f"FigA_SEQ_qualitative_{len(rows)}.png",
            rows,
            suptitle=f"Sequential qualitative — {len(rows)} images ({backbone}, best-EF1 @ {args.steps} steps)",
        )


def main(argv=None):
    p = add_common_args(argparse.ArgumentParser(description="Section 1: sequential loss ablation"))
    p.add_argument("--figures-only", action="store_true")
    p.add_argument(
        "--seq-qualitative-ids",
        nargs="+",
        default=None,
        help="Image IDs for SEQ qualitative/bar figures (overrides per-backbone defaults).",
    )
    args = p.parse_args(argv)
    if args.figures_only:
        run_ablation_figures_only(args)
    else:
        run_ablation(args)


if __name__ == "__main__":
    main()
