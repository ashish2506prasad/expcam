"""Train/eval one (image, backbone, config) with JSON + mask caching."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence

import torch

try:
    from ._path import ensure_pkg_path
except ImportError:
    from _path import ensure_pkg_path

ensure_pkg_path()

from classifiers import ClassifierBundle
from config import BATCH_SIZE
from data import load_tensor
from io_utils import load_json, save_json
from metrics import CAM_FAITHFULNESS_KEYS, evaluate_explanation
from train import TrainConfig, TrainResult, log, save_mask_tensors, train_mask, train_mask_batch


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
        need_preview = not (out_dir / "x_m_e.png").exists() and (out_dir / "masks.pt").exists()
        need_cam = any(k not in payload for k in CAM_FAITHFULNESS_KEYS)
        if need_preview or need_cam:
            x = load_tensor(rec, bundle.preprocess, device, root=eval_root)
            mask_hard, mask_prob = load_masks(out_dir, device)
            if need_preview:
                save_spatial_previews(out_dir, bundle, x, mask_hard, mask_prob, payload)
            if need_cam:
                payload.update(evaluate_explanation(bundle, x, mask_hard, mask_prob=mask_prob))
                save_json(metrics_path, payload)
                log(f"[metrics] backfilled AD/IC/Com/ADCC/ADD  img={rec['id']}")
        return payload

    log(f"[train] {who}  steps={cfg.steps}")
    x = load_tensor(rec, bundle.preprocess, device, root=eval_root)
    result: TrainResult = train_mask(bundle, x, cfg)
    return _finalize_train(bundle, rec, x, out_dir, cfg, result, extra_metrics)


def _finalize_train(
    bundle: ClassifierBundle,
    rec: dict,
    x: torch.Tensor,
    out_dir: Path,
    cfg: TrainConfig,
    result: TrainResult,
    extra_metrics: Optional[Dict] = None,
) -> Dict:
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
    save_json(out_dir / "metrics.json", payload)
    log(
        f"[done]  img={rec['id']}  {result.seconds:.1f}s  "
        f"p={metrics['p_actual_pct']:.1f}%  top1={int(metrics['top1_agreement'])}  "
        f"conf={metrics['conf_e']:.3f}  kl={metrics['kl_divergence']:.3f}"
    )
    if result.ef1_curve:
        save_json(out_dir / "ef1_curve.json", result.ef1_curve)
    save_mask_tensors(out_dir, result)
    save_spatial_previews(out_dir, bundle, x, result.mask_hard, result.mask_prob, payload)
    return payload


def maybe_train_and_eval_many(
    bundle: ClassifierBundle,
    jobs: Sequence[dict],
    device: torch.device,
    batch_size: int = BATCH_SIZE,
) -> List[Dict]:
    """Train several images with batched classifier forwards; one mask per image.

    Each job: ``{rec, eval_root, out_dir, cfg, extra_metrics, skip_existing}``.
    Results stay in job order. Writes happen once per image after training.
    """
    n = len(jobs)
    results: List[Optional[Dict]] = [None] * n
    pending: List[tuple] = []
    bs = max(1, int(batch_size))

    def flush():
        if not pending:
            return
        xs = torch.cat([j["x"] for _, j in pending], dim=0)
        cfg = pending[0][1]["cfg"]
        log(f"[train-batch] {bundle.name}  n={len(pending)}  steps={cfg.steps}")
        trained = train_mask_batch(bundle, xs, cfg)
        for (idx, job), result in zip(pending, trained):
            results[idx] = _finalize_train(
                bundle, job["rec"], job["x"], job["out_dir"], job["cfg"], result, job.get("extra_metrics")
            )
        pending.clear()

    for i, job in enumerate(jobs):
        rec = job["rec"]
        out_dir = job["out_dir"]
        extra = job.get("extra_metrics")
        if job.get("skip_existing") and (out_dir / "metrics.json").exists():
            results[i] = maybe_train_and_eval(
                bundle, rec, job["eval_root"], out_dir, job["cfg"], True, device, extra
            )
            continue
        pending.append(
            (i, {**job, "x": load_tensor(rec, bundle.preprocess, device, root=job["eval_root"])})
        )
        if len(pending) >= bs:
            flush()
    flush()
    return [r for r in results if r is not None]


def load_masks(out_dir: Path, device: torch.device):
    blob = torch.load(out_dir / "masks.pt", map_location=device)
    return blob["mask_hard"].to(device), blob["mask_prob"].to(device)


def save_spatial_previews(out_dir: Path, bundle, x, mask_hard, mask_prob, metrics=None) -> None:
    """Write e.png and a spatial-direct-style x_m_e.png next to masks.pt."""
    import numpy as np
    import torch
    import torch.nn.functional as F
    from PIL import Image

    from classifiers import as_logits
    from config import IMAGENET_MEAN, IMAGENET_STD
    from io_utils import ensure_dir
    from viz import panel_original, save_xme_spatial_direct

    dest = ensure_dir(Path(out_dir))
    m = mask_hard.float()
    if m.dim() == 3:
        m = m.unsqueeze(0)
    e = m.expand_as(x) * x
    e_np = panel_original(e)
    Image.fromarray((np.clip(e_np, 0, 1) * 255).astype(np.uint8)).save(dest / "e.png")

    mean = x.new_tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    std = x.new_tensor(IMAGENET_STD).view(1, 3, 1, 1)
    mask_in = (m.expand(-1, 3, -1, -1) - mean) / std
    with torch.no_grad():
        logits_x = as_logits(bundle.model(x))
        logits_e = as_logits(bundle.model(e))
        logits_m = as_logits(bundle.model(mask_in))
    y = int(logits_x.argmax(dim=1)[0].item())
    px = F.softmax(logits_x, dim=1)
    pe = F.softmax(logits_e, dim=1)
    pm = F.softmax(logits_m, dim=1)
    ei = int(logits_e.argmax(dim=1)[0].item())
    mi = int(logits_m.argmax(dim=1)[0].item())
    labels = bundle.labels
    retained_pct = float(m.mean().item()) * 100.0
    met = metrics or {}
    ef1 = float(met.get("best_ef1", met.get("ef1", 0.0)))
    step = met.get("best_ef1_step")
    save_xme_spatial_direct(
        dest / "x_m_e.png",
        x,
        m,
        original_label=labels[y] if 0 <= y < len(labels) else str(y),
        original_conf=float(px[0, y].item()),
        mask_label=labels[mi] if 0 <= mi < len(labels) else str(mi),
        mask_target_conf=float(pm[0, y].item()),
        explanation_label=labels[ei] if 0 <= ei < len(labels) else str(ei),
        explanation_target_conf=float(pe[0, y].item()),
        retained_pct=retained_pct,
        ef1=ef1,
        step=0 if step is None else int(step),
    )


def export_spatial_pack(export_root: Path, backbone: str, rec: dict, src_dir: Path) -> Path:
    """Copy masks + previews into spatial_masks/{backbone}/{id}_{class}/."""
    from shutil import copy2

    from io_utils import ensure_dir, save_json

    cname = str(rec.get("class_name", "class")).replace(" ", "_")
    dest = ensure_dir(Path(export_root) / backbone / f"{rec['id']}_{cname}")
    for name in ("masks.pt", "ef1_curve.json", "e.png", "x_m_e.png"):
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
