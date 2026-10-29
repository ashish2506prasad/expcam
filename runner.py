"""Train/eval one (image, backbone, config) with JSON + mask caching."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import torch

try:
    from ._path import ensure_pkg_path
except ImportError:
    from _path import ensure_pkg_path

ensure_pkg_path()

from classifiers import ClassifierBundle
from config import MASK_MODEL
from data import load_tensor
from io_utils import load_json, save_json
from metrics import evaluate_explanation
from train import TrainConfig, TrainResult, image_seed, log, save_mask_tensors, train_mask


def run_dir(root: Path, section: str, backbone: str, config_name: str, image_id: str) -> Path:
    return root / section / backbone / _safe(config_name) / image_id


def _safe(name: str) -> str:
    return name.replace("/", "_").replace(" ", "_")


def maybe_train_and_eval(
    bundle: ClassifierBundle,
    rec: dict,
    eval_root: Path,
    out_dir: Path,
    cfg: TrainConfig,
    skip_existing: bool,
    device: torch.device,
    extra_metrics: Optional[Dict] = None,
) -> Dict:
    metrics_path = out_dir / "metrics.json"
    tag = ""
    if extra_metrics:
        tag = extra_metrics.get("config") or extra_metrics.get("protocol") or ""
    who = f"{bundle.name}  img={rec['id']} {rec['class_name']}  {tag}".strip()
    if skip_existing and metrics_path.exists():
        log(f"[skip] {who}")
        payload = load_json(metrics_path)
        if not (out_dir / "x.png").exists() and (out_dir / "masks.pt").exists():
            x = load_tensor(rec, bundle.preprocess, device, root=eval_root)
            mask_hard, mask_prob = load_masks(out_dir, device)
            save_spatial_previews(out_dir, x, mask_hard, mask_prob)
        return payload

    log(f"[train] {who}  steps={cfg.steps}")
    x = load_tensor(rec, bundle.preprocess, device, root=eval_root)
    result: TrainResult = train_mask(bundle, x, cfg)
    metrics = evaluate_explanation(
        bundle,
        x,
        result.mask_hard,
        mask_prob=result.mask_prob,
    )
    payload = {
        "image_id": rec["id"],
        "relpath": rec["relpath"],
        "wnid": rec["wnid"],
        "class_name": rec["class_name"],
        "backbone": bundle.name,
        "seed": result.seed,
        "steps": result.steps,
        "lr": cfg.lr,
        "optimizer": "Adam",
        "seconds": result.seconds,
        "lambdas": result.lambdas,
        "constraint": cfg.constraint,
        "p_target": cfg.p_target,
        "mu_final": result.mu_final,
        "auto_balance": cfg.auto_balance,
        "p_actual": result.p_actual,
        "best_ef1": result.best_ef1,
        "best_ef1_step": result.best_ef1_step,
        **metrics,
        **(extra_metrics or {}),
    }
    save_json(metrics_path, payload)
    log(
        f"[done]  img={rec['id']}  {result.seconds:.1f}s  "
        f"p={metrics['p_actual_pct']:.1f}%  top1={int(metrics['top1_agreement'])}  "
        f"conf={metrics['conf_e']:.3f}  kl={metrics['kl_divergence']:.3f}"
    )
    if cfg.record_curve and result.loss_curve:
        save_json(out_dir / "loss_curve.json", result.loss_curve)
    if result.weight_curve:
        save_json(out_dir / "weight_curve.json", result.weight_curve)
    if result.ef1_curve:
        save_json(out_dir / "ef1_curve.json", result.ef1_curve)
    save_mask_tensors(out_dir, result)
    save_spatial_previews(out_dir, x, result.mask_hard, result.mask_prob)
    save_json(
        out_dir / "train_meta.json",
        {
            "seconds": result.seconds,
            "steps": result.steps,
            "seed": result.seed,
            "optimizer": "Adam",
            "lr": cfg.lr,
            "per_image": True,
            "architecture": {
                "none": "PixelMask",
                "pixel": "PixelMask",
                "direct": "PixelMask",
                "unet": "UNetCSAE",
                "unet_csae": "UNetCSAE",
                "tiny_unet": "TinyUnetCSAE",
                "tinyunet": "TinyUnetCSAE",
                "tiny_unet_csae": "TinyUnetCSAE",
                "tiny_csae": "TinyCSAE",
                "tinycsae": "TinyCSAE",
            }.get(str(cfg.mask_model or MASK_MODEL).strip().lower(), "UNetCSAE"),
            "mask_model": str(cfg.mask_model or MASK_MODEL),
        },
    )
    return payload


def load_masks(out_dir: Path, device: torch.device):
    blob = torch.load(out_dir / "masks.pt", map_location=device)
    return blob["mask_hard"].to(device), blob["mask_prob"].to(device)


def save_spatial_previews(out_dir: Path, x, mask_hard, mask_prob) -> None:
    """Write x / mask / explanation PNGs next to masks.pt."""
    import numpy as np
    from PIL import Image

    from io_utils import ensure_dir
    from viz import panel_expl, panel_mask, panel_original, triple_panel

    dest = ensure_dir(Path(out_dir))
    x_np = panel_original(x)
    m_hard = panel_mask(mask_hard)
    m_prob = panel_mask(mask_prob)
    e_np = panel_expl(x, mask_hard)
    Image.fromarray((np.clip(x_np, 0, 1) * 255).astype(np.uint8)).save(dest / "x.png")
    Image.fromarray((np.clip(m_hard, 0, 1) * 255).astype(np.uint8)).save(dest / "mask_hard.png")
    Image.fromarray((np.clip(m_prob, 0, 1) * 255).astype(np.uint8)).save(dest / "mask_prob.png")
    Image.fromarray((np.clip(e_np, 0, 1) * 255).astype(np.uint8)).save(dest / "e.png")
    triple_panel(dest / "x_m_e.png", x_np, m_hard, e_np, labels=("x", "m (hard)", "e = m⊙x"))


def export_spatial_pack(export_root: Path, backbone: str, rec: dict, src_dir: Path) -> Path:
    """Copy masks + previews into spatial_masks/{backbone}/{id}_{class}/."""
    from shutil import copy2

    from io_utils import ensure_dir, save_json

    cname = str(rec.get("class_name", "class")).replace(" ", "_")
    dest = ensure_dir(Path(export_root) / backbone / f"{rec['id']}_{cname}")
    for name in (
        "masks.pt",
        "metrics.json",
        "x.png",
        "mask_hard.png",
        "mask_prob.png",
        "e.png",
        "x_m_e.png",
    ):
        src = Path(src_dir) / name
        if src.exists():
            copy2(src, dest / name)
    save_json(
        dest / "image.json",
        {
            "id": rec["id"],
            "relpath": rec["relpath"],
            "wnid": rec["wnid"],
            "class_name": rec["class_name"],
            "backbone": backbone,
            "filename": rec.get("filename"),
        },
    )
    return dest
