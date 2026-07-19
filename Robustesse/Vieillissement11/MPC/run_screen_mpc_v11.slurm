#!/usr/bin/bash -l
# Screening d'un an du MPC deterministe V11-p=2 : H6/H24, sans SoH et beta=1.
# Soumettre depuis $WORK/genial/Robustesse/Vieillissement11/MPC.

#SBATCH --job-name=mpc_v11_screen
#SBATCH --output=%x.%J.out
#SBATCH --error=%x.%J.out
#SBATCH --time=03:00:00
#SBATCH --partition=smp
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G

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
python -m unittest -v test_mpc_v11.py || exit 3
time python benchmark_mpc_v11.py --years 1 --workers "$SLURM_CPUS_PER_TASK"
