#!/usr/bin/bash -l
# Front DP V11-p=2 d'un an, strictement comparable au screening MPC d'un an.

#SBATCH --job-name=dp_v11_mpc_ref
#SBATCH --output=%x.%J.out
#SBATCH --error=%x.%J.out
#SBATCH --time=08:00:00
#SBATCH --partition=smp
#SBATCH --cpus-per-task=19
#SBATCH --mem=64G

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export MPLBACKEND=Agg
export MPLCONFIGDIR="$WORK/.mplconfig"
export GENIAL_DATA_DIR="$WORK/genial_data"
export DP_PARETO_V2=1
mkdir -p "$MPLCONFIGDIR"

module purge
module load anaconda3@2022.10/gcc-12.1.0

cd "$SLURM_SUBMIT_DIR" || exit 1
time python run_dp_reference_for_mpc.py \
  --years 1 --n-iter 3 --workers "$SLURM_CPUS_PER_TASK"
