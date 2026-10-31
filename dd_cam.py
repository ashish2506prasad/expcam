"""DD-CAM baseline: 1-minimal last-layer units via delta debugging.

Khadka et al., arXiv:2602.19274. Gradient-free search over CNN feature maps
or ViT patch tokens; saliency is a weighted upsample of the kept units.
"""

from __future__ import annotations

from typing import List, Sequence

import torch
import torch.nn.functional as F

try:
    from ._path import ensure_pkg_path
except ImportError:
    from _path import ensure_pkg_path

ensure_pkg_path()

from classifiers import ClassifierBundle
from train import log


@torch.no_grad()
def ddcam_heatmap(bundle: ClassifierBundle, x: torch.Tensor, max_forwards: int = 800) -> torch.Tensor:
    """Return a [H, W] saliency map in [0, 1] for the original top-1 class."""
    if bundle.cam_is_vit:
        keep, acts, logits = _search_vit(bundle, x, max_forwards=max_forwards)
        cam = _vit_saliency(bundle, x, keep, acts, logits)
    else:
        keep, acts, logits = _search_cnn(bundle, x)
        cam = _cnn_saliency(bundle, x, keep, acts, logits)
    h, w = x.shape[-2:]
    if cam.shape[-2:] != (h, w):
        cam = F.interpolate(cam[None, None], size=(h, w), mode="bilinear", align_corners=False)[0, 0]
    cam = cam.clamp(min=0)
    return cam / (cam.max() + 1e-12)


def _pred_class(logits: torch.Tensor) -> int:
    return int(logits.argmax(dim=1)[0].item())


def _class_logit(logits: torch.Tensor, c: int) -> float:
    return float(logits[0, c].item())


def _search_cnn(bundle: ClassifierBundle, x: torch.Tensor) -> tuple:
    acts, logits = _cnn_forward(bundle, x, keep=None)
    c_hat = _pred_class(logits)
    k = int(acts.shape[1])
    energy = acts.pow(2).mean(dim=(0, 2, 3))
    order = torch.argsort(energy).tolist()
    keep = list(range(k))
    for i in order:
        trial = [u for u in keep if u != i]
        if not trial:
            continue
        _, logits_t = _cnn_forward(bundle, x, keep=trial)
        if _pred_class(logits_t) == c_hat:
            keep = trial
    log(f"  [DD-CAM] {bundle.name} kept {len(keep)}/{k} feature maps")
    return keep, acts, logits


def _search_vit(bundle: ClassifierBundle, x: torch.Tensor, max_forwards: int) -> tuple:
    n_patches = _n_vit_patches(bundle, x)
    acts, logits = _vit_forward(bundle, x, keep=list(range(n_patches)))
    c_hat = _pred_class(logits)
    calls = [0]
    keep = _dd(list(range(n_patches)), 2, bundle, x, c_hat, calls=calls, max_forwards=max_forwards)
    if not keep:
        keep = list(range(n_patches))
    log(f"  [DD-CAM] {bundle.name} kept {len(keep)}/{n_patches} patches  forwards={calls[0]}")
    return keep, acts, logits


def _dd(
    s: List[int],
    n: int,
    bundle: ClassifierBundle,
    x: torch.Tensor,
    c_hat: int,
    calls: List[int],
    max_forwards: int,
) -> List[int]:
    if len(s) <= 1 or calls[0] >= max_forwards:
        return list(s)
    n = min(max(2, n), len(s))
    parts = _split(s, n)
    for part in parts:
        stest = [u for u in s if u not in set(part)]
        if not stest:
            continue
        calls[0] += 1
        _, logits_t = _vit_forward(bundle, x, keep=stest)
        if _pred_class(logits_t) == c_hat:
            return _dd(stest, 2, bundle, x, c_hat, calls, max_forwards)
    n_next = min(2 * n, len(s))
    if n_next == n:
        return list(s)
    return _dd(s, n_next, bundle, x, c_hat, calls, max_forwards)


def _split(s: Sequence[int], n: int) -> List[List[int]]:
    s = list(s)
    n = min(n, len(s))
    sizes = [len(s) // n] * n
    for i in range(len(s) % n):
        sizes[i] += 1
    out, i = [], 0
    for sz in sizes:
        out.append(s[i : i + sz])
        i += sz
    return [p for p in out if p]


def _cnn_forward(bundle: ClassifierBundle, x: torch.Tensor, keep) -> tuple:
    acts = {}

    def hook(_m, _inp, out):
        if keep is None:
            acts["a"] = out
            return out
        m = torch.zeros_like(out)
        m[:, keep] = 1.0
        masked = out * m
        acts["a"] = masked
        return masked

    h = bundle.cam_layer.register_forward_hook(hook)
    try:
        logits = bundle.model(x)
    finally:
        h.remove()
    return acts["a"], logits


def _n_vit_patches(bundle: ClassifierBundle, x: torch.Tensor) -> int:
    n = x.shape[-1] // 16
    return n * n


def _vit_forward(bundle: ClassifierBundle, x: torch.Tensor, keep: Sequence[int]) -> tuple:
    keep_set = set(int(i) for i in keep)
    acts = {}

    def pre_hook(_m, inp):
        tokens = inp[0]
        out = tokens.clone()
        for i in range(1, tokens.shape[1]):
            if (i - 1) not in keep_set:
                out[:, i] = 0
        acts["a"] = out
        return (out, *inp[1:])

    layer = bundle.model.encoder.layers[-1]
    h = layer.register_forward_pre_hook(pre_hook)
    try:
        logits = bundle.model(x)
    finally:
        h.remove()
    return acts.get("a"), logits


def _cnn_saliency(bundle, x, keep, acts, logits) -> torch.Tensor:
    c_hat = _pred_class(logits)
    y0 = _class_logit(logits, c_hat)
    if not keep:
        keep = list(range(acts.shape[1]))
    drops = []
    maps = []
    for i in keep:
        trial = [u for u in keep if u != i]
        if not trial:
            drops.append(1.0)
        else:
            _, lg = _cnn_forward(bundle, x, keep=trial)
            drops.append(max(0.0, y0 - _class_logit(lg, c_hat)))
        maps.append(acts[0, i])
    w = torch.tensor(drops, device=acts.device, dtype=acts.dtype)
    if float(w.sum()) <= 0:
        w = torch.ones_like(w)
    w = w / w.sum()
    stacked = torch.stack(maps, dim=0)
    cam = (w[:, None, None] * stacked).sum(0)
    return cam


def _vit_saliency(bundle, x, keep, acts, logits) -> torch.Tensor:
    c_hat = _pred_class(logits)
    y0 = _class_logit(logits, c_hat)
    n = _n_vit_patches(bundle, x)
    g = int(round(n ** 0.5))
    if not keep:
        keep = list(range(n))
    drops = []
    for i in keep:
        trial = [u for u in keep if u != i]
        if not trial:
            drops.append(1.0)
            continue
        _, lg = _vit_forward(bundle, x, keep=trial)
        drops.append(max(0.0, y0 - _class_logit(lg, c_hat)))
    w = torch.tensor(drops, device=x.device, dtype=x.dtype)
    if float(w.sum()) <= 0:
        w = torch.ones_like(w)
    w = w / w.sum()
    grid = torch.zeros(g, g, device=x.device, dtype=x.dtype)
    for wi, i in zip(w, keep):
        grid[i // g, i % g] = wi
    return grid
