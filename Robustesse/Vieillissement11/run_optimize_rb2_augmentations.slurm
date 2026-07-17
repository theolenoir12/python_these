#!/bin/bash
#SBATCH --job-name=rb2_aug_v10
#SBATCH --cpus-per-task=24
#SBATCH --time=3-00:00:00
#SBATCH --mem=64G
#SBATCH --output=rb2_aug_v10_%j.out

set -euo pipefail
cd "$(dirname "$0")"
python optimize_rb2_augmentations.py --layer suite --workers "${SLURM_CPUS_PER_TASK}" --seeds 3
