#!/usr/bin/bash -l
# Confirmation 25 ans : baseline H24 vs combo_top3, bruit x1 et CRN réservées.

#SBATCH --job-name=mpc_v11_long
#SBATCH --output=%x.%J.out
#SBATCH --error=%x.%J.out
#SBATCH --time=24:00:00
#SBATCH --partition=smp
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

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
python -m unittest -v test_mpc_v11.py test_tuning_mpc_v11.py test_longrun_tuning_mpc_v11.py || exit 3
time python benchmark_longrun_tuning_mpc_v11.py \
  --years 25 --workers "$SLURM_CPUS_PER_TASK" \
  --decision-run runs/tune_validation_1y_9c728d3d847a
