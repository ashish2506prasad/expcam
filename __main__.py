"""python -m expcam <section|all> [args...]

Also works as a script on Kaggle:
    python __main__.py 0 --where kaggle
"""

from __future__ import annotations

import sys

try:
    from ._path import ensure_pkg_path
except ImportError:
    from _path import ensure_pkg_path

ensure_pkg_path()

USAGE = """EXP-CAM (spatial-direct PixelMask)

Images: first --img-per-class files per class (sorted names), first --num-classes
folders (sorted). No eval_set.json. --data-path / --data-root may be a folder or
a single image file. --steps is shared.

  python __main__.py 0 --where kaggle --data-root /kaggle/input/<slug>/trainX1_subset \\
      --out /kaggle/working/expcam_spatial --backbones vit_b_16 --img-per-class 1

  python __main__.py 2 --sensitivity --where kaggle --data-root /kaggle/input/<slug>/trainX1_subset \\
      --out /kaggle/working/expcam_spatial --backbones vit_b_16

  python __main__.py 3 --where kaggle --data-root /kaggle/input/<slug>/trainX1_subset \\
      --out /kaggle/working/expcam_spatial --backbones vit_b_16

Section 2 --sensitivity retrains PixelMask; it does not need Section 0.
Section 3 uses only images that already have Section 0 masks.pt.
Pass --image-illustration on Section 3 to write CAM figures; otherwise CSVs only.

Training length is config.STEPS for every section (override with --steps).
On Kaggle, --out is zipped when the run finishes.
"""


_SECTION_MODS = {
    "0": "ef1",
    "1": "ablation",
    "2": "minimality",
    "3": "comparisons",
    "4": "robustness",
    "5": "auto_weights",
    "6": "llm",
}


def _load(modname: str):
    return __import__(modname, fromlist=["main"])


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help"):
        print(USAGE)
        return
    cmd, rest = argv[0], argv[1:]
    if cmd in ("eval-set", "eval_set"):
        from data import main as data_main

        data_main(rest)
        return
    if cmd == "all":
        for key in ("0", "2", "3"):
            modname = _SECTION_MODS[key]
            print(f"\n===== Section {key} =====")
            getattr(_load(modname), "main")(rest)
        return
    if cmd not in _SECTION_MODS:
        print(USAGE)
        raise SystemExit(f"Unknown command: {cmd}")
    getattr(_load(_SECTION_MODS[cmd]), "main")(rest)


if __name__ == "__main__":
    main()
