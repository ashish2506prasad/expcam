"""Canonical hyperparameters, paths, and experiment constants.

Values below are the paper's later full-model defaults, taken from the
AM4/AM5/AM6 training cells (not the earlier AM1/AM2 hand-tuning grid).
"""

from __future__ import annotations

from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_DIR.parent
TRAINX_DIR = PROJECT_ROOT / "trainx"
RUNS_DIR = PROJECT_ROOT / "expcam_runs"
EVAL_SET_PATH = RUNS_DIR / "eval_set.json"
WHERE = "local"

# Kaggle dataset layout (class folders such as n01440764 live under this root).
KAGGLE_TRAINX = Path("/kaggle/input/datasets/meashish2003/trainx/trainx")
KAGGLE_RUNS = Path("/kaggle/working/expcam_runs")
KAGGLE_WORKING = Path("/kaggle/working")

SEED = 0
N_CLASSES = 13
N_IMAGES = 91  # 7 per class × 13 trainx classes (bottleneck: n01749939 has 7)
N_PER_CLASS = 7
QUALITATIVE_PER_CLASS = 2  # images per class in Fig A11, B5, C4, E4, etc.
SECTION0_PER_CLASS = 2  # Section 0 trains 2 images per class
SECTION0_IMAGE_SLOTS = (2, 5)  # 3rd & 6th eval image (qualitative uses slots 0,1)
SECTION0_STEPS = 800
SECTION0_EF1_SNAPSHOT_START = 0
SECTION0_EF1_SNAPSHOT_EVERY = 40
SECTION3_EXP_STEPS = 800  # EXP-CAM training in Section 3; best-EF1 checkpoint saved
IMAGE_SIZE = 224

DEFAULT_BACKBONES = ("resnet18", "efficientnet_b0", "vit_b_16")

# Per-backbone figure output folders under section1/ … section5/.
FIGURE_DIRS = {
    "resnet18": "figures_resnet",
    "efficientnet_b0": "figures_efficient",
    "vit_b_16": "figures_vit",
}


def figure_dir(out_root: Path, section: str, backbone: str) -> Path:
    """Return e.g. section1/figures_resnet/ for a given backbone."""
    sub = FIGURE_DIRS.get(backbone, f"figures_{backbone}")
    return Path(out_root) / section / sub

# Per-image U-Net training (AM6 cell 15 / AM4–AM5 full-model cells)
STEPS = 700
SECTION1_STEPS = 300
LR = 5e-5
OPTIMIZER_NAME = "adam"
TEMPERATURE = 2.0
MASK_THRESH = 0.5
ROB_K_BACKGROUNDS = 3

DEFAULT_LAMBDAS = {
    "act": 10.0,
    "ce": 15.0,
    "kl": 5.0,
    "area": 100.0,
    "bin": 1.0,
    "tv": 500.0,
    "rob": 100.0,  # abductive / L_rob in the notebooks (lam_abd)
}

LOSS_TERMS = ("act", "ce", "kl", "area", "bin", "tv", "rob")

BUDGETS = (0.50, 0.30, 0.20, 0.10, 0.05, 0.02, 0.01, 0.005)
# Section 2 knapsack sweep on Section 0 best mask_prob (fraction of pixels kept)
KNAPSACK_BUDGETS = (0.005, 0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50)
FIDELITY_TAUS = (0.95, 0.90, 1.0)
P_TARGETS = (0.05, 0.10, 0.20)

LAYER_WEIGHTS_RESNET = {
    "relu": 1.0,
    "layer1": 2.0,
    "layer2": 2.0,
    "layer3": 4.0,
    "layer4": 8.0,
}

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
GREY_BG = 0.5

# ImageNet-1K synsets present in trainx/
TRAINX_CLASSES = {
    "n01484850": {"name": "great_white_shark", "imagenet_idx": 2},
    "n01494475": {"name": "hammerhead", "imagenet_idx": 4},
    "n01531178": {"name": "goldfinch", "imagenet_idx": 11},
    "n01632777": {"name": "axolotl", "imagenet_idx": 29},
    "n01665541": {"name": "leatherback_turtle", "imagenet_idx": 34},
    "n01687978": {"name": "agama", "imagenet_idx": 42},
    "n01695060": {"name": "komodo_dragon", "imagenet_idx": 48},
    "n01749939": {"name": "green_mamba", "imagenet_idx": 64},
    "n01818515": {"name": "macaw", "imagenet_idx": 88},
    "n01833805": {"name": "hummingbird", "imagenet_idx": 94},
    "n02012849": {"name": "crane", "imagenet_idx": 134},
    "n02018795": {"name": "bustard", "imagenet_idx": 138},
    "n02037110": {"name": "oystercatcher", "imagenet_idx": 143},
}


def ef1_snapshot_steps(
    start: int = SECTION0_EF1_SNAPSHOT_START,
    every: int = SECTION0_EF1_SNAPSHOT_EVERY,
    max_step: int = SECTION0_STEPS,
) -> tuple[int, ...]:
    """Steps at which to save explanation snapshots (40, 80, …, max_step)."""
    return tuple(range(start + every, max_step + 1, every))


def ef1_grid_steps(
    every: int,
    max_step: int = SECTION0_STEPS,
    first: int | None = None,
) -> tuple[int, ...]:
    """Steps for Section 0 progression grids (40, 80, …, max_step)."""
    if first is None:
        first = every
    return tuple(range(first, max_step + 1, every))

N_BG_DRAWS = 5
N_INSERT_STEPS = 50
N_RISE_MASKS = 800
N_INSERT_SEEDS = 5
AUG_ROT_DEG = 5.0
AUG_TRANSLATE_FRAC = 0.05
AUG_BRIGHTNESS = 0.10

# Retrospective count of distinct λ tuples in AM0–AM6 (Section 5 "tuning cost")
MANUAL_TRIAL_CONFIGS = 18

# Section 6 — local causal LM (no Hugging Face hub on air-gapped clusters)
DISTILGPT2_LOCAL = PROJECT_ROOT / "distilgpt2-local"


def detect_where() -> str:
    if Path("/kaggle/input").exists() or Path("/kaggle/working").exists():
        return "kaggle"
    return "local"


def apply_where(where: str = "auto", data_root=None, runs_dir=None) -> str:
    """Switch trainx/ and output roots for local vs Kaggle.

    local:  <repo>/trainx  and  <repo>/expcam_runs
    kaggle: /kaggle/input/datasets/meashish2003/trainx/trainx
            and /kaggle/working/expcam_runs
    """
    global PROJECT_ROOT, TRAINX_DIR, RUNS_DIR, EVAL_SET_PATH, WHERE
    if where in (None, "auto"):
        where = detect_where()
    WHERE = where
    if where == "kaggle":
        PROJECT_ROOT = KAGGLE_WORKING
        TRAINX_DIR = KAGGLE_TRAINX
        RUNS_DIR = KAGGLE_RUNS
    elif where == "local":
        PROJECT_ROOT = PACKAGE_DIR.parent
        TRAINX_DIR = PROJECT_ROOT / "trainx"
        RUNS_DIR = PROJECT_ROOT / "expcam_runs"
    else:
        raise ValueError(f"Unknown --where {where!r}; expected local, kaggle, or auto")
    if data_root is not None:
        TRAINX_DIR = Path(data_root)
    if runs_dir is not None:
        RUNS_DIR = Path(runs_dir)
    EVAL_SET_PATH = RUNS_DIR / "eval_set.json"
    print(f"[paths] where={WHERE} trainx={TRAINX_DIR} runs={RUNS_DIR}", flush=True)
    return WHERE


def add_where_args(p):
    p.add_argument(
        "--where",
        choices=("local", "kaggle", "auto"),
        default="auto",
        help=(
            "Path layout. local: <repo>/trainx and <repo>/expcam_runs. "
            "kaggle: /kaggle/input/datasets/meashish2003/trainx/trainx and "
            "/kaggle/working/expcam_runs. auto: kaggle if /kaggle exists."
        ),
    )
    p.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="Override the image folder (must contain n01440764 and the other synsets).",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Run directory. Default: expcam_runs (local) or /kaggle/working/expcam_runs (kaggle).",
    )
    return p


def lambdas_with_overrides(**overrides) -> dict:
    out = dict(DEFAULT_LAMBDAS)
    out.update(overrides)
    return out


def sequential_configs() -> dict:
    """Cumulative loss addition ending with the full model (+ L_rob)."""
    zero = {k: 0.0 for k in DEFAULT_LAMBDAS}
    cfgs = {}

    lam = dict(zero)
    lam["act"] = DEFAULT_LAMBDAS["act"]
    cfgs["seq_act"] = lam

    lam = dict(cfgs["seq_act"])
    lam["kl"] = DEFAULT_LAMBDAS["kl"]
    cfgs["seq_act_kl"] = lam

    lam = dict(cfgs["seq_act_kl"])
    lam["ce"] = DEFAULT_LAMBDAS["ce"]
    cfgs["seq_act_kl_ce"] = lam

    lam = dict(cfgs["seq_act_kl_ce"])
    lam["area"] = DEFAULT_LAMBDAS["area"]
    lam["bin"] = DEFAULT_LAMBDAS["bin"]
    lam["tv"] = DEFAULT_LAMBDAS["tv"]
    cfgs["seq_act_kl_ce_min"] = lam

    cfgs["Full"] = dict(DEFAULT_LAMBDAS)
    return cfgs


SEQUENTIAL_PLOT_ORDER = ["seq_act", "seq_act_kl", "seq_act_kl_ce", "seq_act_kl_ce_min", "Full"]
