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
    IMG_PER_CLASS,
    N_CLASSES,
    N_PER_CLASS,
    QUALITATIVE_PER_CLASS,
    SECTION0_PER_CLASS,
    SEED,
    TRAINX_CLASSES,
    add_where_args,
)
from io_utils import load_json, save_json


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".JPEG", ".JPG", ".PNG", ".bmp", ".webp"}


def is_image_file(path: Path) -> bool:
    return Path(path).is_file() and Path(path).suffix in IMAGE_EXTS


def discover_class_folders(root: Path) -> List[str]:
    """Class subfolders that contain at least one image, sorted by folder name."""
    root = Path(root)
    names = []
    if not root.is_dir():
        return names
    for p in sorted(root.iterdir(), key=lambda x: x.name):
        if p.is_dir() and list_class_images(root, p.name):
            names.append(p.name)
    return names


def _class_meta(wnid: str) -> Dict:
    meta = TRAINX_CLASSES.get(wnid)
    if meta:
        return {"class_name": meta["name"], "imagenet_idx": meta["imagenet_idx"]}
    return {"class_name": wnid, "imagenet_idx": -1}


def select_records(
    root: Path,
    img_per_class: int = IMG_PER_CLASS,
    num_classes: Optional[int] = None,
) -> Dict:
    """Build an in-memory image list. Does not write eval_set.json.

    Folder: first ``num_classes`` synset dirs (sorted), first ``img_per_class``
    files in each (sorted names). File: a single-image run.
    """
    root = Path(root)
    images: List[Dict] = []
    if is_image_file(root):
        rec = {
            "id": "000",
            "relpath": root.name,
            "abspath": str(root.resolve()),
            "wnid": root.parent.name,
            "class_name": _class_meta(root.parent.name)["class_name"],
            "imagenet_idx": _class_meta(root.parent.name)["imagenet_idx"],
            "class_slot": 0,
            "filename": root.name,
        }
        images.append(rec)
        data_root = root.parent
        classes = [rec["wnid"]]
    else:
        data_root = root
        classes = discover_class_folders(root)
        if num_classes is not None:
            classes = classes[: max(0, int(num_classes))]
        n_img = max(1, int(img_per_class))
        for class_i, wnid in enumerate(classes):
            files = list_class_images(root, wnid)[:n_img]
            meta = _class_meta(wnid)
            for p in files:
                images.append(
                    {
                        "id": f"{len(images):03d}",
                        "relpath": str(p.relative_to(root)).replace("\\", "/"),
                        "wnid": wnid,
                        "class_name": meta["class_name"],
                        "imagenet_idx": meta["imagenet_idx"],
                        "class_slot": class_i,
                        "filename": p.name,
                    }
                )
    print(
        f"[data] {len(images)} images, {len(classes)} classes from {data_root} "
        f"(img-per-class={img_per_class}, num-classes={num_classes or 'all'})",
        flush=True,
    )
    return {
        "root": str(data_root),
        "n_per_class": int(img_per_class),
        "n_images": len(images),
        "n_classes": len(classes),
        "classes": classes,
        "images": images,
        "curve_image_id": images[0]["id"] if images else None,
        "cached": False,
    }


def eligible_trainx_classes(root: Path, n_per_class: int) -> List[str]:
    """Synsets in TRAINX_CLASSES that exist under root and have ≥ n_per_class images."""
    out = []
    for wnid in TRAINX_CLASSES:
        folder = root / wnid
        if not folder.is_dir():
            continue
        n = len(list_class_images(root, wnid))
        if n < n_per_class:
            continue
        out.append(wnid)
    return out


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
    classes_present = eligible_trainx_classes(root, n_per_class)
    skipped = [wnid for wnid in TRAINX_CLASSES if (root / wnid).is_dir() and wnid not in classes_present]
    for wnid in skipped:
        n = len(list_class_images(root, wnid))
        print(f"[eval-set] skip {wnid} ({TRAINX_CLASSES[wnid]['name']}): {n} images < {n_per_class}")
    n_want = N_CLASSES if N_CLASSES is not None else len(classes_present)
    if len(classes_present) < (N_CLASSES or 1):
        raise RuntimeError(
            f"Need at least {N_CLASSES or 1} class folder(s) in {root} with ≥{n_per_class} images; "
            f"found {len(classes_present)}: {classes_present}"
        )
    classes_present = classes_present[:n_want]
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
    data["root"] = str(C.TRAINX_DIR)
    print(f"Reusing eval set ({len(data.get('images', []))} images) at {path}", flush=True)
    return data


def image_path(record: Dict, root: Optional[Path] = None) -> Path:
    import config as C

    if record.get("abspath"):
        return Path(record["abspath"])
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
    """Section 0 subset: ``per_class`` images from each class.

    If ``slots`` is omitted, use the first ``per_class`` eval-set images
    (indices 0, 1, …). Passing ``slots`` picks those indices instead.
    """
    if per_class is None:
        per_class = SECTION0_PER_CLASS
    if slots is None:
        slots = tuple(range(int(per_class)))
    else:
        slots = tuple(int(s) for s in slots)
    by_class = records_by_class(eval_set)
    class_order = eval_set.get("classes") or sorted(by_class.keys())
    out: List[Dict] = []
    for wnid in class_order:
        recs = by_class.get(wnid, [])
        for slot in slots[: int(per_class)]:
            if 0 <= slot < len(recs):
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
    p = argparse.ArgumentParser(description="List images that would be used (no eval_set.json).")
    add_where_args(p)
    p.add_argument("--root", type=Path, default=None, help="Alias for --data-root")
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    import config as C

    args = parse_eval_args(argv)
    data_root = args.data_root or args.root
    C.apply_where(args.where, data_root=data_root, runs_dir=args.out, zip_out=False)
    recs = select_records(
        C.TRAINX_DIR,
        img_per_class=int(getattr(args, "img_per_class", None) or IMG_PER_CLASS),
        num_classes=getattr(args, "num_classes", None),
    )
    print(f"[data] listed {recs['n_images']} images; nothing written to disk", flush=True)


if __name__ == "__main__":
    main()
