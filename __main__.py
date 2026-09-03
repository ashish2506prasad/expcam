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

Local (from the folder that contains trainx/ and expcam/):
  python -m expcam eval-set --where local --force --out expcam_runs_wo_ce
  python -m expcam 1 --where local --out expcam_runs_wo_ce --backbones resnet18 vit_b_16
  python -m expcam 2 --from-section0-knapsack --where local --out expcam_runs_wo_ce
  python -m expcam 3 --where local --out expcam_runs_wo_ce --backbones resnet18 vit_b_16

See expcam/README.md for full Section 1–3 instructions.

Every experiment reuses eval_set.json under the runs directory.
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
