#!/usr/bin/bash -l
# Tuning H24 sans SoH : 39 trajectoires de selection puis 48 de validation.

#SBATCH --job-name=mpc_v11_tune
#SBATCH --output=%x.%J.out
#SBATCH --error=%x.%J.out
#SBATCH --time=08:00:00
#SBATCH --partition=smp
#SBATCH --cpus-per-task=12
#SBATCH --mem=48G

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
python -c "import scipy; from scipy.optimize import milp; print('scipy', scipy.__version__, 'milp OK')" || exit 2
python -m unittest -v test_mpc_v11.py test_tuning_mpc_v11.py || exit 3
time python benchmark_tuning_mpc_v11.py \
  --phase all --years 1 --workers "$SLURM_CPUS_PER_TASK" \
  --baseline-run runs/forecast_uncertainty_1y_1acc8ef7e9d2
