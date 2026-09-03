#!/bin/bash
#SBATCH --job-name=expcam_local
#SBATCH --output=slurm_logs/expcam_local-%j.log
#SBATCH --error=slurm_logs/expcam_local-%j.err

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1

#SBATCH --partition=a40
#SBATCH --qos=a40
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=48:00:00


echo "=========================================="
echo "JOB ID: $SLURM_JOB_ID"
echo "NODE   : $(hostname)"
echo "GPUS   : $CUDA_VISIBLE_DEVICES"
echo "Start  : $(date)"
echo "=========================================="

source ~/miniconda3/etc/profile.d/conda.sh
conda activate vae

ROOT=/home/medal/21d180009
DATA_ROOT=/home/medal/21d180009/trainx
OUT_DIR=/home/medal/21d180009/expcam_runs_wo_ce

cd "$ROOT"
mkdir -p slurm_logs "$OUT_DIR"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo "[paths] data-root=$DATA_ROOT"
echo "[paths] out=$OUT_DIR"

python -m expcam eval-set --where local --force --out "$OUT_DIR"

# python -m expcam 0 --ef1-new-version --no-ce --no-skip-existing --where local --out "$OUT_DIR" --backbones resnet18 vit_b_16
# python -m expcam 1 --where local --out "$OUT_DIR" --backbones resnet18 vit_b_16
# python -m expcam 2 --from-section0-knapsack --section0-source section0_new_score --where local --out "$OUT_DIR" --backbones resnet18 vit_b_16
# python -m expcam 3 --where local --out "$OUT_DIR" --backbones resnet18 vit_b_16

echo "Finished at $(date)"
