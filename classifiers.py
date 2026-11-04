"""Frozen ImageNet classifiers + activation hooks."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Tuple

import torch
import torch.nn as nn
import torchvision.models as models

try:
    from ._path import ensure_pkg_path
except ImportError:
    from _path import ensure_pkg_path

ensure_pkg_path()

LAYER_WEIGHTS_RESNET = {
    "relu": 1.0,
    "layer1": 2.0,
    "layer2": 2.0,
    "layer3": 4.0,
    "layer4": 8.0,
}


TapSpec = Dict[str, nn.Module]


@dataclass
class ClassifierBundle:
    name: str
    model: nn.Module
    preprocess: Callable
    labels: List[str]
    tap_names: Tuple[str, ...]
    layer_weights: Dict[str, float]
    cam_layer: nn.Module
    cam_is_vit: bool = False
    taps: Dict[str, nn.Module] = field(default_factory=dict)


def as_logits(out) -> torch.Tensor:
    if isinstance(out, (tuple, list)):
        return out[0]
    return getattr(out, "logits", out)


def _freeze(model: nn.Module) -> nn.Module:
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def _vgg_block_relus(features: nn.Sequential) -> Dict[str, nn.Module]:
    taps = {}
    last_relu = None
    n = 0
    for m in features:
        if isinstance(m, nn.ReLU):
            last_relu = m
        if isinstance(m, nn.MaxPool2d) and last_relu is not None:
            n += 1
            taps[f"relu{n}"] = last_relu
            last_relu = None
    return taps


def _vgg_layer_weights(taps: Dict[str, nn.Module]) -> Dict[str, float]:
    scale = (1.0, 2.0, 2.0, 4.0, 8.0)
    names = list(taps)
    return {names[i]: scale[min(i, len(scale) - 1)] for i in range(len(names))}


def _last_conv(module: nn.Module) -> nn.Module:
    last = None
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            last = m
    if last is None:
        raise RuntimeError("no Conv2d in module")
    return last


def _bundle(name, model, weights, taps, layer_weights, cam_layer, cam_is_vit=False) -> ClassifierBundle:
    return ClassifierBundle(
        name=name,
        model=_freeze(model),
        preprocess=weights.transforms(),
        labels=list(weights.meta["categories"]),
        tap_names=tuple(taps.keys()),
        layer_weights=dict(layer_weights),
        cam_layer=cam_layer,
        cam_is_vit=cam_is_vit,
        taps=taps,
    )


def _resnet(name: str, ctor, weights, device: torch.device) -> ClassifierBundle:
    model = ctor(weights=weights).to(device)
    taps = {
        "relu": model.relu,
        "layer1": model.layer1,
        "layer2": model.layer2,
        "layer3": model.layer3,
        "layer4": model.layer4,
    }
    return _bundle(name, model, weights, taps, LAYER_WEIGHTS_RESNET, model.layer4[-1])


def _vgg(name: str, ctor, weights, device: torch.device) -> ClassifierBundle:
    model = ctor(weights=weights).to(device)
    taps = _vgg_block_relus(model.features)
    return _bundle(name, model, weights, taps, _vgg_layer_weights(taps), _last_conv(model.features))


def _vit(name: str, ctor, weights, device: torch.device, taps_idx: Tuple[int, int, int]) -> ClassifierBundle:
    model = ctor(weights=weights).to(device)
    layers = model.encoder.layers
    i0, i1, i2 = taps_idx
    taps = {
        f"block{i0}": layers[i0],
        f"block{i1}": layers[i1],
        f"block{i2}": layers[i2],
    }
    lw = {f"block{i0}": 1.0, f"block{i1}": 2.0, f"block{i2}": 8.0}
    return _bundle(name, model, weights, taps, lw, layers[i2], cam_is_vit=True)


def _efficientnet_b0(device: torch.device) -> ClassifierBundle:
    weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1
    model = models.efficientnet_b0(weights=weights).to(device)
    taps = {
        "f1": model.features[1],
        "f3": model.features[3],
        "f5": model.features[5],
        "f7": model.features[7],
        "f8": model.features[8],
    }
    lw = {"f1": 1.0, "f3": 2.0, "f5": 2.0, "f7": 4.0, "f8": 8.0}
    return _bundle("efficientnet_b0", model, weights, taps, lw, model.features[-1])


def _inception_v3(device: torch.device) -> ClassifierBundle:
    weights = models.Inception_V3_Weights.IMAGENET1K_V1
    model = models.inception_v3(weights=weights).to(device)
    inner_forward = model.forward

    def _logits_forward(*args, **kwargs):
        return as_logits(inner_forward(*args, **kwargs))

    model.forward = _logits_forward
    taps = {
        "mixed5d": model.Mixed_5d,
        "mixed6e": model.Mixed_6e,
        "mixed7c": model.Mixed_7c,
    }
    lw = {"mixed5d": 1.0, "mixed6e": 2.0, "mixed7c": 8.0}
    return _bundle("inception_v3", model, weights, taps, lw, model.Mixed_7c)


_BUILDERS = {
    "vgg11": lambda d: _vgg("vgg11", models.vgg11, models.VGG11_Weights.IMAGENET1K_V1, d),
    "vgg16": lambda d: _vgg("vgg16", models.vgg16, models.VGG16_Weights.IMAGENET1K_V1, d),
    "vgg19": lambda d: _vgg("vgg19", models.vgg19, models.VGG19_Weights.IMAGENET1K_V1, d),
    "resnet18": lambda d: _resnet("resnet18", models.resnet18, models.ResNet18_Weights.IMAGENET1K_V1, d),
    "resnet50": lambda d: _resnet("resnet50", models.resnet50, models.ResNet50_Weights.IMAGENET1K_V1, d),
    "vit_b_16": lambda d: _vit("vit_b_16", models.vit_b_16, models.ViT_B_16_Weights.IMAGENET1K_V1, d, (3, 7, 11)),
    "vit_l_16": lambda d: _vit("vit_l_16", models.vit_l_16, models.ViT_L_16_Weights.IMAGENET1K_V1, d, (7, 15, 23)),
    "efficientnet_b0": _efficientnet_b0,
    "inception_v3": _inception_v3,
}


def load_classifier(name: str, device: torch.device) -> ClassifierBundle:
    key = name.strip().lower()
    if key not in _BUILDERS:
        raise ValueError(f"Unknown backbone {name!r}; choose from {sorted(_BUILDERS)}")
    return _BUILDERS[key](device)


def tap_modules(bundle: ClassifierBundle) -> TapSpec:
    if bundle.taps:
        return bundle.taps
    return {}


def run_with_acts(
    bundle: ClassifierBundle,
    x: torch.Tensor,
    require_grad: bool,
) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
    """Forward pass that returns named activations and logits."""
    model = bundle.model
    model.eval()
    acts: Dict[str, torch.Tensor] = {}
    handles = []

    def hook(name: str):
        def fn(_m, _inp, out):
            acts[name] = out if require_grad else out.detach()

        return fn

    for name, module in tap_modules(bundle).items():
        handles.append(module.register_forward_hook(hook(name)))
    with torch.set_grad_enabled(require_grad):
        logits = as_logits(model(x))
    for h in handles:
        h.remove()
    return acts, logits


def spatial_feat(feat: torch.Tensor) -> torch.Tensor:
    """Map ViT token maps (B, T, C) to (B, C, H, W) when possible; leave CNN maps alone."""
    if feat.dim() == 4:
        return feat
    if feat.dim() == 3:
        b, t, c = feat.shape
        grid = int(round((t - 1) ** 0.5))
        if grid * grid == t - 1:
            patches = feat[:, 1:, :].transpose(1, 2).reshape(b, c, grid, grid)
            return patches
        grid = int(round(t ** 0.5))
        if grid * grid == t:
            return feat.transpose(1, 2).reshape(b, c, grid, grid)
    return feat
