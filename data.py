"""Fixed evaluation set: sample once from trainx/, cache, reuse everywhere."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Sequence

try:
    from ._path import ensure_pkg_path
except ImportError:
    from _path import ensure_pkg_path

ensure_pkg_path()

import numpy as np
from PIL import Image

from config import (
    DEFAULT_BACKBONES,
    N_CLASSES,
    N_PER_CLASS,
    QUALITATIVE_PER_CLASS,
    SECTION0_IMAGE_SLOTS,
    SECTION0_PER_CLASS,
    SEED,
    TRAINX_CLASSES,
    add_where_args,
)
from io_utils import load_json, save_json


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".JPEG", ".JPG", ".PNG", ".bmp"}


def list_class_images(root: Path, wnid: str) -> List[Path]:
    folder = root / wnid
    if not folder.is_dir():
        raise FileNotFoundError(f"Missing class folder {folder}")
    files = [p for p in folder.iterdir() if p.suffix in IMAGE_EXTS]
    files.sort(key=lambda p: p.name)
    return files


def build_eval_set(
    root: Optional[Path] = None,
    n_per_class: int = N_PER_CLASS,
    seed: int = SEED,
    force: bool = False,
    path: Optional[Path] = None,
) -> Dict:
    import config as C

    if root is None:
        root = C.TRAINX_DIR
    if path is None:
        path = C.EVAL_SET_PATH
    root = Path(root)
    path = Path(path)

    if path.exists() and not force:
        data = load_json(path)
        data["root"] = str(root)
        print(f"Reusing cached eval set ({len(data['images'])} images) at {path}")
        return data

    rng = np.random.RandomState(seed)
    images = []
    classes_present = [wnid for wnid in TRAINX_CLASSES if (root / wnid).is_dir()]
    if len(classes_present) < N_CLASSES:
        raise RuntimeError(
            f"Need at least {N_CLASSES} classes in {root}; found {len(classes_present)}: {classes_present}"
        )
    classes_present = classes_present[:N_CLASSES]
    for class_i, wnid in enumerate(classes_present):
        files = list_class_images(root, wnid)
        if len(files) < n_per_class:
            raise RuntimeError(f"{wnid} has {len(files)} images, need {n_per_class}")
        pick = rng.choice(len(files), size=n_per_class, replace=False)
        pick.sort()
        meta = TRAINX_CLASSES[wnid]
        for local_i, file_i in enumerate(pick):
            p = files[int(file_i)]
            image_id = f"{len(images):03d}"
            images.append(
                {
                    "id": image_id,
                    "relpath": str(p.relative_to(root)).replace("\\", "/"),
                    "wnid": wnid,
                    "class_name": meta["name"],
                    "imagenet_idx": meta["imagenet_idx"],
                    "class_slot": class_i,
                    "filename": p.name,
                }
            )
    data = {
        "seed": seed,
        "root": str(root),
        "n_per_class": n_per_class,
        "n_images": len(images),
        "n_classes": len(classes_present),
        "classes": classes_present,
        "backbones": list(DEFAULT_BACKBONES),
        "qualitative_ids": qualitative_image_ids(
            {"images": images, "classes": classes_present},
            per_class=min(QUALITATIVE_PER_CLASS, n_per_class),
        ),
        "curve_image_id": images[0]["id"],
        "images": images,
        "note": (
            "This exact image list is reused in every revision section. "
            "Do not resample after seeing results."
        ),
    }
    save_json(path, data)
    print(f"Wrote eval set: {len(images)} images, {len(classes_present)} classes -> {path}")
    return data


def load_eval_set(path: Optional[Path] = None) -> Dict:
    import config as C

    if path is None:
        path = C.EVAL_SET_PATH
    path = Path(path)
    if not path.exists():
        return build_eval_set(path=path)
    data = load_json(path)
    expected_classes = [wnid for wnid in TRAINX_CLASSES if (C.TRAINX_DIR / wnid).is_dir()][
        : C.N_CLASSES
    ]
    if (
        data.get("n_per_class") != C.N_PER_CLASS
        or data.get("n_classes") != C.N_CLASSES
        or data.get("classes") != expected_classes
    ):
        print(
            f"Eval set has n_per_class={data.get('n_per_class')} n_classes={data.get('n_classes')} "
            f"classes={data.get('classes')} but config wants {C.N_PER_CLASS}/{C.N_CLASSES}/"
            f"{expected_classes}; rebuilding."
        )
        return build_eval_set(path=path, force=True)
    data["root"] = str(C.TRAINX_DIR)
    return data


def image_path(record: Dict, root: Optional[Path] = None) -> Path:
    import config as C

    root = Path(root) if root is not None else C.TRAINX_DIR
    return root / record["relpath"]


def load_pil(record: Dict, root: Optional[Path] = None) -> Image.Image:
    return Image.open(image_path(record, root)).convert("RGB")


def load_tensor(record: Dict, preprocess, device, root: Optional[Path] = None):
    import torch

    img = load_pil(record, root)
    return preprocess(img).unsqueeze(0).to(device)


def records_by_class(eval_set: Dict) -> Dict[str, List[Dict]]:
    out: Dict[str, List[Dict]] = {}
    for rec in eval_set["images"]:
        out.setdefault(rec["wnid"], []).append(rec)
    return out


def one_image_per_class(eval_set: Dict) -> List[Dict]:
    """First eval image from each class."""
    by_class = records_by_class(eval_set)
    class_order = eval_set.get("classes") or sorted(by_class.keys())
    out: List[Dict] = []
    for wnid in class_order:
        recs = by_class.get(wnid, [])
        if recs:
            out.append(recs[0])
    return out


def section0_image_records(
    eval_set: Dict,
    per_class: Optional[int] = None,
    slots: Optional[Sequence[int]] = None,
) -> List[Dict]:
    """Section 0 subset: fixed slots per class (default 3rd & 6th, not qualitative 1st/2nd)."""
    if per_class is None:
        per_class = SECTION0_PER_CLASS
    if slots is None:
        slots = SECTION0_IMAGE_SLOTS
    by_class = records_by_class(eval_set)
    class_order = eval_set.get("classes") or sorted(by_class.keys())
    out: List[Dict] = []
    for wnid in class_order:
        recs = by_class.get(wnid, [])
        for slot in slots[:per_class]:
            if slot < len(recs):
                out.append(recs[slot])
    return out


def qualitative_image_ids(eval_set: Dict, per_class: Optional[int] = None) -> List[str]:
    """Stable qualitative panel set: first `per_class` eval images from each class."""
    if per_class is None:
        per_class = QUALITATIVE_PER_CLASS
    by_class = records_by_class(eval_set)
    class_order = eval_set.get("classes") or sorted(by_class.keys())
    ids: List[str] = []
    for wnid in class_order:
        for rec in by_class.get(wnid, [])[:per_class]:
            ids.append(rec["id"])
    return ids


def other_class_records(eval_set: Dict, wnid: str) -> List[Dict]:
    return [r for r in eval_set["images"] if r["wnid"] != wnid]


def same_class_records(eval_set: Dict, rec: Dict) -> List[Dict]:
    return [r for r in eval_set["images"] if r["wnid"] == rec["wnid"] and r["id"] != rec["id"]]


def parse_eval_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build the cached EXP-CAM eval set from trainx/")
    add_where_args(p)
    p.add_argument("--root", type=Path, default=None, help="Alias for --data-root")
    p.add_argument("--n-per-class", type=int, default=N_PER_CLASS)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--force", action="store_true")
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    import config as C

    args = parse_eval_args(argv)
    data_root = args.data_root or args.root
    C.apply_where(args.where, data_root=data_root, runs_dir=args.out)
    build_eval_set(root=C.TRAINX_DIR, n_per_class=args.n_per_class, seed=args.seed, force=args.force)


if __name__ == "__main__":
    main()
