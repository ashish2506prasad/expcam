"""Shared evaluation metrics. Every section must call these — do not reimplement.

Formulas match the EXP-CAM revision plan. Aggregates are always mean ± std
over the fixed image set, never a single-image statistic reported as typical.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

try:
    from ._path import ensure_pkg_path
except ImportError:
    from _path import ensure_pkg_path

ensure_pkg_path()

from classifiers import ClassifierBundle
from config import IMAGENET_MEAN, IMAGENET_STD, MASK_THRESH, N_INSERT_STEPS
from losses import binarization_loss, tv8_loss


def mask_size(mask: torch.Tensor) -> float:
    """p_actual = ||m||_1 / N."""
    m = mask.float()
    return float(m.mean().item())


def top1_agreement(logits_x: torch.Tensor, logits_e: torch.Tensor) -> float:
    return float((logits_e.argmax(dim=1) == logits_x.argmax(dim=1)).float().mean().item())


def confidence_delta(logits_x: torch.Tensor, logits_e: torch.Tensor) -> float:
    y = logits_x.argmax(dim=1)
    px = F.softmax(logits_x, dim=1)
    pe = F.softmax(logits_e, dim=1)
    return float((pe[range(len(y)), y] - px[range(len(y)), y]).mean().item())


def explanation_f1(delta_conf: float, p_actual: float) -> float:
    """EF1 = HM(1 - |Δconf|, 1 − p_actual).

    Harmonic mean of 1 minus absolute confidence delta and mask sparsity (1 − area).
    """
    a = max(0.0, 1.0 - abs(float(delta_conf)))
    b = max(0.0, 1.0 - float(p_actual))
    denom = a + b
    if denom <= 0.0:
        return 0.0
    return 2.0 * a * b / denom


def ef1_new_version(confidence: float, p_actual: float) -> float:
    """EF1 (new) = HM(confidence, 1 − p_actual)."""
    a = max(0.0, float(confidence))
    b = max(0.0, 1.0 - float(p_actual))
    denom = a + b
    if denom <= 0.0:
        return 0.0
    return 2.0 * a * b / denom


def class_confidence(logits: torch.Tensor, y: torch.Tensor) -> float:
    return float(F.softmax(logits, dim=1)[range(len(y)), y].mean().item())


def kl_divergence(logits_x: torch.Tensor, logits_e: torch.Tensor) -> float:
    """D_KL(softmax(f(x)) || softmax(f(e))) — evaluation KL, temperature = 1."""
    p_x = F.softmax(logits_x, dim=1)
    log_p_e = F.log_softmax(logits_e, dim=1)
    return float(F.kl_div(log_p_e, p_x, reduction="batchmean").item())


def exp_minus_kl(logits_x: torch.Tensor, logits_e: torch.Tensor) -> float:
    """Fidelity KL component in (0, 1]: exp(−D_KL). Equals 1 when distributions match."""
    return float(np.exp(-kl_divergence(logits_x, logits_e)))


def normalized_kl(logits_x: torch.Tensor, logits_e: torch.Tensor) -> float:
    """Legacy: KL / log(C) capped at 1. Prefer exp_minus_kl for fidelity reporting."""
    c = logits_x.shape[1]
    nkl = kl_divergence(logits_x, logits_e) / max(np.log(c), 1e-8)
    return float(min(nkl, 1.0))


def ce_on_original_label(logits_x: torch.Tensor, logits_e: torch.Tensor) -> float:
    y = logits_x.argmax(dim=1)
    return float(F.cross_entropy(logits_e, y).item())


def tv_energy_per_pixel(mask: torch.Tensor) -> float:
    tv = float(tv8_loss(mask.float()).item())
    area = max(float(mask.float().sum().item()), 1.0)
    return tv / area


def binarization_sharpness(mask_prob: torch.Tensor) -> float:
    return float(binarization_loss(mask_prob).item())


def robustness_accuracy(
    model: torch.nn.Module,
    x: torch.Tensor,
    mask: torch.Tensor,
    backgrounds: Sequence[torch.Tensor],
    y: Optional[torch.Tensor] = None,
) -> float:
    """P(argmax f(ẽ) = y) with ẽ = m⊙x + (1-m)⊙r, averaged over backgrounds."""
    if y is None:
        with torch.no_grad():
            y = model(x).argmax(dim=1)
    m = (mask.float() > MASK_THRESH).float()
    if m.shape[1] == 1:
        m = m.expand_as(x)
    hits = []
    with torch.no_grad():
        for r in backgrounds:
            e_r = m * x + (1.0 - m) * r.to(x.device)
            pred = model(e_r).argmax(dim=1)
            hits.append(float((pred == y).float().mean().item()))
    return float(np.mean(hits)) if hits else 0.0


def fidelity_bundle(logits_x: torch.Tensor, logits_e: torch.Tensor) -> Dict[str, float]:
    """Fidelity(p) as two reported components — not a weighted scalar."""
    return {
        "top1_agreement": top1_agreement(logits_x, logits_e),
        "exp_minus_kl": exp_minus_kl(logits_x, logits_e),
    }


def fidelity_kl_component(row: Dict) -> float:
    """Read KL fidelity from a metrics row (new or legacy runs)."""
    if "exp_minus_kl" in row:
        return float(row["exp_minus_kl"])
    return float(row.get("one_minus_normalized_kl", 0.0))


CAM_FAITHFULNESS_KEYS = ("ad", "ic", "com", "coh", "adcc", "add")
CAM_TABLE_KEYS = (
    "ad",
    "ic",
    "com",
    "coh",
    "adcc",
    "add",
    "ef1",
    "ef1_conf",
    "top1_agreement",
    "p_actual",
    "conf_e",
    "conf_x",
    "kl_divergence",
    "insertion_auc",
)


def cam_faithfulness(
    conf_x: float,
    conf_e: float,
    conf_del: float,
    complexity: float,
) -> Dict[str, float]:
    """DD-CAM / CAM-literature faithfulness on one image.

    AD  = max(0, Yc(x)−Yc(e)) / Yc(x)     (average drop; ↓)
    IC  = 1[Yc(e) > Yc(x)]                 (increase in confidence; ↑)
    Com = ||m||₁ / N                       (complexity / area; ↓)
    Coh = 1 − |Yc(x)−Yc(e)| / Yc(x)        (coherency; ↑)
    ADCC = harmonic mean of (1−AD, Coh, 1−Com)  (↑)
    ADD = max(0, Yc(x)−Yc(x⊙(1−m))) / Yc(x)  (drop under deletion; ↑)
    """
    eps = 1e-12
    cx = float(conf_x)
    ce = float(conf_e)
    cd = float(conf_del)
    com = float(max(0.0, min(1.0, complexity)))
    ad = max(0.0, cx - ce) / (cx + eps)
    ic = 1.0 if ce > cx else 0.0
    coh = max(0.0, 1.0 - abs(cx - ce) / (cx + eps))
    add = max(0.0, cx - cd) / (cx + eps)
    ad_p = max(1.0 - ad, eps)
    com_p = max(1.0 - com, eps)
    coh_c = max(coh, eps)
    adcc = 3.0 / (1.0 / ad_p + 1.0 / coh_c + 1.0 / com_p)
    return {
        "ad": float(ad),
        "ic": float(ic),
        "com": float(com),
        "coh": float(coh),
        "adcc": float(adcc),
        "add": float(add),
    }


def evaluate_explanation(
    bundle: ClassifierBundle,
    x: torch.Tensor,
    mask_hard: torch.Tensor,
    mask_prob: Optional[torch.Tensor] = None,
    backgrounds: Optional[Sequence[torch.Tensor]] = None,
) -> Dict[str, float]:
    """Compute the shared metric vector for one (image, mask) pair.

    Robustness accuracy is included only when `backgrounds` is passed
    (Section 4). It is not part of the default Section 1/2/3/5 vector.
    """
    m = mask_hard.float()
    if m.dim() == 3:
        m = m.unsqueeze(0)
    if m.shape[1] == 1:
        e = m.expand_as(x) * x
    else:
        e = m * x
    m1 = m
    if m1.shape[1] == 1:
        m1 = m1.expand_as(x)
    e_del = (1.0 - m1) * x
    with torch.no_grad():
        logits_x = bundle.model(x)
        logits_e = bundle.model(e)
        logits_del = bundle.model(e_del)
    y = logits_x.argmax(dim=1)
    px = F.softmax(logits_x, dim=1)
    pe = F.softmax(logits_e, dim=1)
    pdel = F.softmax(logits_del, dim=1)
    out: Dict[str, float] = {
        "p_actual": mask_size(m),
        "p_actual_pct": 100.0 * mask_size(m),
        "top1_agreement": top1_agreement(logits_x, logits_e),
        "confidence_delta": confidence_delta(logits_x, logits_e),
        "conf_x": float(px[0, y[0]].item()),
        "conf_e": float(pe[0, y[0]].item()),
        "pred_x": int(y[0].item()),
        "pred_e": int(logits_e.argmax(dim=1)[0].item()),
        "kl_divergence": kl_divergence(logits_x, logits_e),
        "exp_minus_kl": exp_minus_kl(logits_x, logits_e),
        "normalized_kl": normalized_kl(logits_x, logits_e),
        "one_minus_normalized_kl": 1.0 - normalized_kl(logits_x, logits_e),
        "ce": ce_on_original_label(logits_x, logits_e),
        "tv_energy_per_pixel": tv_energy_per_pixel(m),
        "binarization_sharpness": binarization_sharpness(mask_prob if mask_prob is not None else m),
    }
    if backgrounds is not None:
        out["robustness_accuracy"] = robustness_accuracy(bundle.model, x, m, backgrounds, y=y)
    out.update(fidelity_bundle(logits_x, logits_e))
    out["ef1"] = explanation_f1(out["confidence_delta"], out["p_actual"])
    out["ef1_conf"] = ef1_new_version(out["conf_e"], out["p_actual"])
    out["abs_confidence_delta"] = abs(out["confidence_delta"])
    conf_del = float(pdel[0, y[0]].item())
    out["conf_del"] = conf_del
    out.update(cam_faithfulness(out["conf_x"], out["conf_e"], conf_del, out["p_actual"]))
    return out


def match_abs_delta_topk(
    bundle: ClassifierBundle,
    x: torch.Tensor,
    scores: torch.Tensor,
    target_abs_delta: float,
) -> Dict[str, float]:
    """Smallest top-k on `scores` whose |Δconf| is ≤ target (EXP-CAM |Δconf|)."""
    from baselines import topk_binarize

    h, w = int(x.shape[-2]), int(x.shape[-1])
    n = h * w
    scores = scores.detach().float().reshape(h, w)
    target = float(abs(target_abs_delta))

    def eval_k(k: int) -> Dict[str, float]:
        k = int(max(1, min(n, k)))
        hard = topk_binarize(scores, k).view(1, 1, h, w)
        met = evaluate_explanation(bundle, x, hard.to(x.device), mask_prob=scores.view(1, 1, h, w))
        met["k_pixels"] = k
        met["abs_delta"] = abs(float(met["confidence_delta"]))
        return met

    lo, hi = 1, n
    best = eval_k(n)
    while lo < hi:
        mid = (lo + hi) // 2
        met = eval_k(mid)
        if met["abs_delta"] <= target + 1e-4:
            best = met
            hi = mid
        else:
            lo = mid + 1
    if lo != best.get("k_pixels"):
        best = eval_k(lo)
    return best


def min_sufficient_budget(
    rows_by_p: Sequence[Dict],
    tau: float,
    p_key: str = "p_budget",
) -> Optional[float]:
    """Smallest budget at which both fidelity components clear τ.

    Components: top1_agreement(p) ≥ τ AND exp(−KL(p)) ≥ τ.
    This is the smallest mask budget *found by the optimizer*, not a provable global min.
    """
    ordered = sorted(rows_by_p, key=lambda r: r[p_key])
    for row in ordered:
        if row["top1_agreement"] >= tau and fidelity_kl_component(row) >= tau:
            return float(row[p_key])
    return None


def insertion_curve(
    bundle: ClassifierBundle,
    x: torch.Tensor,
    scores: torch.Tensor,
    n_steps: int = N_INSERT_STEPS,
    n_seeds: int = 1,
    binary_mask: bool = False,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Insertion: start from blur, add pixels by importance; return xs, mean conf, AUC.

    For binary EXP-CAM masks with no intra-mask ranking, `binary_mask=True` inserts
    mask pixels in random order over `n_seeds` and averages.
    """
    model = bundle.model
    device = x.device
    y = model(x).argmax(dim=1)
    conf_full = class_confidence(model(x), y)
    blur = _gaussian_blur_nchw(x, kernel=21, sigma=5.0)
    h, w = x.shape[-2:]
    n = h * w
    steps = np.linspace(0, n, n_steps + 1, dtype=int)
    curves = []
    score_np = scores.detach().float().cpu().numpy().reshape(-1)
    mask_idx = np.flatnonzero(score_np > 0.5) if binary_mask else None
    rng = np.random.RandomState(0)
    for seed in range(n_seeds):
        if binary_mask:
            order = mask_idx.copy()
            rng.seed(seed)
            rng.shuffle(order)
            rest = np.setdiff1d(np.arange(n), order, assume_unique=False)
            rng.shuffle(rest)
            order = np.concatenate([order, rest])
        else:
            order = np.argsort(-score_np)
        confs = []
        with torch.no_grad():
            for k in steps:
                img = blur.clone()
                if k > 0:
                    sel = order[:k]
                    ys, xs = np.unravel_index(sel, (h, w))
                    img[:, :, ys, xs] = x[:, :, ys, xs]
                confs.append(class_confidence(model(img), y) / (conf_full + 1e-12))
        curves.append(confs)
    mean_curve = np.mean(np.asarray(curves), axis=0)
    xs = 100.0 * steps / n
    auc = float(np.trapz(mean_curve, xs) / 100.0)
    return xs, mean_curve, auc


def _gaussian_blur_nchw(x: torch.Tensor, kernel: int = 21, sigma: float = 5.0) -> torch.Tensor:
    import math

    k = kernel
    coords = torch.arange(k, device=x.device, dtype=x.dtype) - (k - 1) / 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    kernel2d = (g[:, None] * g[None, :]).view(1, 1, k, k)
    kernel2d = kernel2d.repeat(x.shape[1], 1, 1, 1)
    pad = k // 2
    return F.conv2d(x, kernel2d, padding=pad, groups=x.shape[1])


def denormalize(x: torch.Tensor) -> torch.Tensor:
    mean = x.new_tensor(IMAGENET_MEAN)[None, :, None, None]
    std = x.new_tensor(IMAGENET_STD)[None, :, None, None]
    return (x * std + mean).clamp(0, 1)


def explanation_on_grey(x: torch.Tensor, mask: torch.Tensor, grey: float = 0.5) -> torch.Tensor:
    vis = denormalize(x)
    m = mask.float()
    if m.dim() == 3:
        m = m.unsqueeze(0)
    if m.shape[1] == 1:
        m = m.expand_as(vis)
    return m * vis + (1.0 - m) * grey
