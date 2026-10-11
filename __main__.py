"""python -m expcam <section|eval-set|all> [args...]

Also works as a script on Kaggle:
    python __main__.py 1 --where kaggle
    python ablation.py --where kaggle
"""

from __future__ import annotations

import sys

try:
    from ._path import ensure_pkg_path
except ImportError:
    from _path import ensure_pkg_path

ensure_pkg_path()

USAGE = """EXP-CAM experiments

Pixel Section 0 (spatial masks) on the same dataset as freq_expcam:

  python __main__.py eval-set --where kaggle --data-root /kaggle/input/<slug>/trainX1_subset \\
      --eval-set /kaggle/working/freq_expcam_fft/eval_set.json --out /kaggle/working/expcam_spatial

  python __main__.py 0 --subset --where kaggle \\
      --data-root /kaggle/input/<slug>/trainX1_subset \\
      --eval-set /kaggle/working/freq_expcam_fft/eval_set.json \\
      --out /kaggle/working/expcam_spatial \\
      --backbones resnet18 vit_b_16

Download /kaggle/working/expcam_spatial.zip
Masks: expcam_spatial/spatial_masks/resnet18/{id}_{class}/masks.pt
       expcam_spatial/spatial_masks/vit_b_16/{id}_{class}/masks.pt

If you omit --eval-set, a new eval_set.json is built (ids will not match freq runs).
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
        from data import main as data_main

        data_main(rest)
        for key in ("1", "2", "3"):
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
