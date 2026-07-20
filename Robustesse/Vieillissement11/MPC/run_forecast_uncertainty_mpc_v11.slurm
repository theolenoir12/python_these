#!/usr/bin/bash -l
# Sensibilite MPC H24 aux erreurs de prevision, 5 graines et sigma 0.5/1/1.5.

#SBATCH --job-name=mpc_v11_pred
#SBATCH --output=%x.%J.out
#SBATCH --error=%x.%J.out
#SBATCH --time=04:00:00
#SBATCH --partition=smp
#SBATCH --cpus-per-task=20
#SBATCH --mem=40G

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export MPLBACKEND=Agg
export MPLCONFIGDIR="$WORK/.mplconfig"
export GENIAL_DATA_DIR="$WORK/genial_data"
mkdir -p "$MPLCONFIGDIR"

module purge
module load anaconda3@2022.10/gcc-12.1.0

cd "$SLURM_SUBMIT_DIR" || exit 1
python -m unittest -v test_mpc_v11.py || exit 2
time python benchmark_forecast_uncertainty_mpc_v11.py \
  --years 1 --workers "$SLURM_CPUS_PER_TASK"
