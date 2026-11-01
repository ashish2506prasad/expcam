"""Per-image PixelMask training (spatial-direct). Optimizer, steps, and seed
are held fixed across ablations unless a section changes the loss weights
or adds a mask-budget constraint.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

try:
    from ._path import ensure_pkg_path
except ImportError:
    from _path import ensure_pkg_path

ensure_pkg_path()

from classifiers import ClassifierBundle, run_with_acts
from config import DEFAULT_LAMBDAS, LR, MASK_THRESH, ROB_K_BACKGROUNDS, SEED, STEPS, TEMPERATURE
from metrics import confidence_delta, ef1_new_version, explanation_f1
from losses import (
    abductive_loss,
    area_loss,
    binarization_loss,
    ce_loss,
    hard_st,
    kl_loss,
    lagrangian_area_penalty,
    layer_act_loss,
    project_topk_mask,
    sample_gaussian_backgrounds,
    tv8_loss,
)
from models import PixelMask


def log(msg: str) -> None:
    print(msg, flush=True)


def seed_all(seed: int) -> None:
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def image_seed(image_id: str, backbone: str, base: int = SEED) -> int:
    """Same init for every config of (image, backbone) — required for fair LOO."""
    return int(base + 17 * int(image_id) + 1009 * (sum(map(ord, backbone)) % 97))


@dataclass
class TrainConfig:
    lambdas: Dict[str, float] = field(default_factory=lambda: dict(DEFAULT_LAMBDAS))
    steps: int = STEPS
    lr: float = LR
    seed: int = SEED
    temperature: float = TEMPERATURE
    mask_thresh: float = MASK_THRESH
    constraint: Optional[str] = None  # None | "project" | "lagrangian"
    p_target: Optional[float] = None
    mu_init: float = 0.0
    mu_eta: float = 5.0
    auto_balance: Optional[str] = None  # None | "uncertainty"
    record_curve: bool = False
    curve_every: int = 1
    rob_k: int = ROB_K_BACKGROUNDS
    extra_backgrounds: Optional[Sequence[torch.Tensor]] = None
    log_every: int = 50  # 0 = silent; notebooks printed every 10 steps
    no_ce: bool = False  # when True, CE loss is omitted from the training objective
    record_ef1: bool = False
    keep_best_ef1: bool = False  # return mask from the step with highest EF1
    ef1_version: str = "legacy"  # "legacy" | "new" (ef1_new_version in metrics.py)
    save_expl_steps: Optional[Sequence[int]] = None
    save_expl_dir: Optional[str] = None


@dataclass
class TrainResult:
    mask_prob: torch.Tensor
    mask_hard: torch.Tensor
    e: torch.Tensor
    seconds: float
    steps: int
    lambdas: Dict[str, float]
    seed: int
    loss_curve: List[Dict[str, float]]
    weight_curve: List[Dict[str, float]]
    mu_final: float
    p_actual: float
    ef1_curve: List[Dict[str, float]] = field(default_factory=list)
    best_ef1: float = 0.0
    best_ef1_step: int = 0


def train_mask(
    bundle: ClassifierBundle,
    x: torch.Tensor,
    cfg: TrainConfig,
) -> TrainResult:
    device = x.device
    seed_all(cfg.seed)
    mask_net = PixelMask(int(x.shape[-2]), int(x.shape[-1])).to(device)
    arch = "PixelMask"
    mask_net.train()
    opt = torch.optim.Adam(mask_net.parameters(), lr=cfg.lr)

    s_params: Optional[nn.ParameterDict] = None
    if cfg.auto_balance == "uncertainty":
        # Kendall et al.: L = Σ exp(-s_i) L_i + s_i  (s_i = log-variance)
        keys = [k for k in cfg.lambdas if not (cfg.constraint is not None and k == "area")]
        if cfg.no_ce:
            keys = [k for k in keys if k != "ce"]
        s_params = nn.ParameterDict(
            {k: nn.Parameter(torch.zeros((), device=device)) for k in keys}
        )
        opt = torch.optim.Adam(list(mask_net.parameters()) + list(s_params.parameters()), lr=cfg.lr)

    acts_x, logits_x = run_with_acts(bundle, x, require_grad=False)
    shallow = set(list(bundle.tap_names)[:2])
    mu = float(cfg.mu_init)
    curve: List[Dict[str, float]] = []
    wcurve: List[Dict[str, float]] = []
    ef1_curve: List[Dict[str, float]] = []
    save_expl_set = set(cfg.save_expl_steps or ())
    track_ef1 = cfg.record_ef1 or cfg.keep_best_ef1
    best_ef1 = -1.0
    best_ef1_step = 0
    best_mask_prob: Optional[torch.Tensor] = None
    best_mask_hard: Optional[torch.Tensor] = None

    t0 = time.perf_counter()
    if cfg.log_every:
        log(
            f"  training {arch}  steps={cfg.steps}  lr={cfg.lr}  seed={cfg.seed}"
            + (f"  constraint={cfg.constraint} p={cfg.p_target}" if cfg.constraint else "")
        )
    for t in range(1, cfg.steps + 1):
        opt.zero_grad()
        mask_logits = mask_net(x)
        mask_prob = torch.sigmoid(mask_logits)
        if cfg.constraint == "project" and cfg.p_target is not None:
            mask_prob = project_topk_mask(mask_prob, cfg.p_target)
        mask_st = hard_st(mask_prob, cfg.mask_thresh)
        e = mask_st.expand_as(x) * x

        acts_e, logits_e = run_with_acts(bundle, e, require_grad=True)
        loss_ce = ce_loss(logits_x, logits_e) if not cfg.no_ce else x.new_zeros(())
        loss_kl = kl_loss(logits_x, logits_e, temperature=cfg.temperature)
        loss_act = x.new_zeros(())
        for name, w in bundle.layer_weights.items():
            if name not in acts_x or name not in acts_e:
                continue
            loss_act = loss_act + w * layer_act_loss(name, acts_x[name], acts_e[name], shallow)
        loss_area = area_loss(mask_prob)
        loss_bin = binarization_loss(mask_prob)
        loss_tv = tv8_loss(mask_prob)
        bgs = list(cfg.extra_backgrounds) if cfg.extra_backgrounds else sample_gaussian_backgrounds(x, cfg.rob_k)
        loss_rob = abductive_loss(bundle.model, x, mask_prob, bgs, temperature=cfg.temperature)

        components = {
            "act": loss_act,
            "ce": loss_ce,
            "kl": loss_kl,
            "area": loss_area,
            "bin": loss_bin,
            "tv": loss_tv,
            "rob": loss_rob,
        }

        if cfg.constraint in ("project", "lagrangian"):
            # Experiment 2B / 5a: do not minimize area — only constrain it.
            components_for_sum = {k: v for k, v in components.items() if k != "area"}
        else:
            components_for_sum = components

        if cfg.no_ce:
            components_for_sum = {k: v for k, v in components_for_sum.items() if k != "ce"}

        if s_params is not None:
            loss_total = x.new_zeros(())
            for k, L in components_for_sum.items():
                if k not in s_params:
                    loss_total = loss_total + cfg.lambdas.get(k, 0.0) * L
                    continue
                s = s_params[k]
                loss_total = loss_total + torch.exp(-s) * L + s
        else:
            loss_total = x.new_zeros(())
            for k, L in components_for_sum.items():
                loss_total = loss_total + cfg.lambdas.get(k, 0.0) * L

        if cfg.constraint == "lagrangian" and cfg.p_target is not None:
            p_now = mask_prob.mean()
            loss_total = loss_total + lagrangian_area_penalty(p_now, cfg.p_target, mu)

        loss_total.backward()
        opt.step()

        if cfg.constraint == "lagrangian" and cfg.p_target is not None:
            with torch.no_grad():
                viol = float(torch.relu(mask_prob.mean() - cfg.p_target).item())
            mu = max(0.0, mu + cfg.mu_eta * viol)

        if track_ef1 or save_expl_set:
            with torch.no_grad():
                p_hard = (mask_prob > cfg.mask_thresh).float()
                p_act = float(p_hard.mean().item())
                d_conf = confidence_delta(logits_x, logits_e)
                if cfg.ef1_version == "new":
                    y = logits_x.argmax(dim=1)
                    conf_e = float(
                        torch.nn.functional.softmax(logits_e, dim=1)[0, y[0]].item()
                    )
                    ef1 = ef1_new_version(conf_e, p_act)
                else:
                    ef1 = explanation_f1(d_conf, p_act)
                if track_ef1:
                    ef1_curve.append(
                        {
                            "step": t,
                            "ef1": ef1,
                            "delta_conf": d_conf,
                            "p_actual": p_act,
                        }
                    )
                if cfg.keep_best_ef1 and ef1 >= best_ef1:
                    best_ef1 = ef1
                    best_ef1_step = t
                    best_mask_prob = mask_prob.detach().clone()
                    best_mask_hard = p_hard.detach().clone()
                if t in save_expl_set and cfg.save_expl_dir:
                    _save_expl_snapshot(x, p_hard, cfg.save_expl_dir, t, ef1, p_act, d_conf)

        if cfg.record_curve and (t % cfg.curve_every == 0 or t == 1 or t == cfg.steps):
            row = {k: float(v.detach().item()) for k, v in components.items()}
            row["total"] = float(loss_total.detach().item())
            row["step"] = t
            row["p_actual"] = float(mask_prob.mean().item())
            row["mu"] = mu
            curve.append(row)
            if s_params is not None:
                wcurve.append(
                    {"step": t, **{k: float(torch.exp(-s_params[k]).detach().item()) for k in s_params}}
                )

        if cfg.log_every and (t == 1 or t % cfg.log_every == 0 or t == cfg.steps):
            with torch.no_grad():
                top1 = int(logits_e.argmax(dim=1).item() == logits_x.argmax(dim=1).item())
                p_now = float((mask_prob > cfg.mask_thresh).float().mean().item())
            log(
                f"  step {t:>4d}/{cfg.steps}  total={loss_total.item():.3f}  "
                f"act={loss_act.item():.3f}  "
                + ("" if cfg.no_ce else f"ce={loss_ce.item():.3f}  ")
                + f"kl={loss_kl.item():.3f}  "
                f"area={loss_area.item():.3f}  bin={loss_bin.item():.3f}  tv={loss_tv.item():.4f}  "
                f"rob={loss_rob.item():.3f}  p={100.0 * p_now:.1f}%  top1={'Y' if top1 else 'N'}"
            )

    seconds = time.perf_counter() - t0
    if cfg.keep_best_ef1 and best_mask_prob is not None and best_mask_hard is not None:
        mask_prob = best_mask_prob
        mask_hard = best_mask_hard
        e = mask_hard.expand_as(x) * x
        p_actual = float(mask_hard.mean().item())
        if cfg.log_every:
            log(
                f"  best EF1={best_ef1:.4f} at step {best_ef1_step}  "
                f"p_actual={100.0 * p_actual:.1f}%"
            )
    else:
        mask_net.eval()
        with torch.no_grad():
            mask_prob = torch.sigmoid(mask_net(x))
            if cfg.constraint == "project" and cfg.p_target is not None:
                mask_prob = project_topk_mask(mask_prob, cfg.p_target)
            mask_hard = (mask_prob > cfg.mask_thresh).float()
            e = mask_hard.expand_as(x) * x
            p_actual = float(mask_hard.mean().item())
        best_ef1 = max((row["ef1"] for row in ef1_curve), default=0.0)
        best_ef1_step = next((row["step"] for row in ef1_curve if row["ef1"] == best_ef1), 0)

    if cfg.log_every:
        log(f"  finished in {seconds:.1f}s  p_actual={100.0 * p_actual:.1f}%")

    return TrainResult(
        mask_prob=mask_prob.detach(),
        mask_hard=mask_hard.detach(),
        e=e.detach(),
        seconds=seconds,
        steps=cfg.steps,
        lambdas=dict(cfg.lambdas),
        seed=cfg.seed,
        loss_curve=curve,
        weight_curve=wcurve,
        mu_final=mu,
        p_actual=p_actual,
        ef1_curve=ef1_curve,
        best_ef1=float(best_ef1),
        best_ef1_step=int(best_ef1_step),
    )


def _save_expl_snapshot(x, mask_hard, out_dir: str, step: int, ef1: float, p_act: float, d_conf: float) -> None:
    from pathlib import Path

    import numpy as np
    from PIL import Image

    from io_utils import ensure_dir
    from viz import panel_expl, panel_mask, panel_original, triple_panel

    dest = ensure_dir(Path(out_dir) / f"step_{step:04d}")
    e_img = panel_expl(x, mask_hard)
    Image.fromarray((np.clip(e_img, 0, 1) * 255).astype(np.uint8)).save(dest / "e.png")
    triple_panel(
        dest / "x_m_e.png",
        panel_original(x),
        panel_mask(mask_hard),
        e_img,
        suptitle=f"step={step}  EF1={ef1:.3f}  p={100*p_act:.1f}%  Δconf={d_conf:.3f}",
    )


def save_mask_tensors(out_dir, result: TrainResult) -> None:
    from pathlib import Path
    from io_utils import ensure_dir

    out_dir = ensure_dir(Path(out_dir))
    torch.save(
        {
            "mask_hard": result.mask_hard.cpu(),
            "mask_prob": result.mask_prob.cpu(),
        },
        out_dir / "masks.pt",
    )
