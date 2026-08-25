#!/bin/bash
#SBATCH --job-name=expcam_local
#SBATCH --output=slurm_logs/expcam_local-%j.log
#SBATCH --error=slurm_logs/expcam_local-%j.err

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1

#SBATCH --partition=dgx
#SBATCH --qos=dgx
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=60:00:00


echo "=========================================="
echo "JOB ID: $SLURM_JOB_ID"
echo "NODE   : $(hostname)"
echo "GPUS   : $CUDA_VISIBLE_DEVICES"
echo "Start  : $(date)"
echo "=========================================="

source ~/miniconda3/etc/profile.d/conda.sh
conda activate vae

ROOT=/home/medal/21d180009
# Class folders (n01440764/tench and the other synsets) live under trainx/
DATA_ROOT=/home/medal/21d180009/trainx
OUT_DIR=/home/medal/21d180009/expcam_runs

cd "$ROOT"
mkdir -p slurm_logs "$OUT_DIR"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo "[paths] data-root=$DATA_ROOT"
echo "[paths] n01440764=$DATA_ROOT/n01440764"
echo "[paths] out=$OUT_DIR"
ls "$DATA_ROOT/n01440764" | wc -l

# Cached eval set (14 images/class from trainx/, including n01440764)
python -m expcam eval-set \
  --where local \
  --data-root "$DATA_ROOT" \
  --out "$OUT_DIR" \
  --force

# Full revision pipeline (sections 1–6). Re-submit is safe: existing runs are skipped.
# Smoke test instead:
#   python -m expcam 1 --where local --data-root "$DATA_ROOT" --out "$OUT_DIR" --quick
# python -m expcam 1 --where local --data-root "$DATA_ROOT" --out "$OUT_DIR" > expcam_section1_local.out 2>&1
# python -m expcam 2 --where local --data-root "$DATA_ROOT" --out "$OUT_DIR" > expcam_section2_local.out 2>&1
# python -m expcam 3 --where local --data-root "$DATA_ROOT" --out "$OUT_DIR" > expcam_section3_local.out 2>&1
# python -m expcam 4 --where local --data-root "$DATA_ROOT" --out "$OUT_DIR" > expcam_section4_local.out 2>&1
python -m expcam 5 --where local --data-root "$DATA_ROOT" --out "$OUT_DIR" > expcam_section5_local.out 2>&1
# python -m expcam 6 --where local --data-root "$DATA_ROOT" --out "$OUT_DIR" > expcam_section6_local.out 2>&1
python -m expcam 1 \
  --where local \
  --data-root /home/medal/21d180009/trainx \
  --out /home/medal/21d180009/expcam_runs \
  --backbones vit_b_16 \
  --steps 400 > expcam_section1_vit_b_16_local.out 2>&1

echo "Finished at $(date)"
