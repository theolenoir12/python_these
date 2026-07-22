#!/usr/bin/bash -l
# Complète la synthèse Pareto : recalcule au MÊME harnais un point PD central
# (DPPolicy rollout, eps=3) et la FLC-I0 réglée, les ajoute à la table de
# campagne, et régénère les figures de synthèse (plans LPSP/EENS vs dégradation +
# classement J3).
#
# Soumettre depuis $WORK/genial/Robustesse/Vieillissement11/FuzzyRules :
#   sbatch run_pareto_summary_v11.slurm
# Sorties : FuzzyRules/figures_synthese/{pareto_*.pdf/png, classement_j3.*,
#           pareto_points_merged.tsv}.

#SBATCH --job-name=pareto_summary_v11
#SBATCH --output=%x.%J.out
#SBATCH --error=%x.%J.out
#SBATCH --time=02:00:00
#SBATCH --partition=smp
#SBATCH --cpus-per-task=2
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

cd "$SLURM_SUBMIT_DIR/.." || exit 1

python -c "import numpy, matplotlib; print('numpy', numpy.__version__, '| mpl', matplotlib.__version__)" || exit 2

# Recalcul PD + FLC réglée au même harnais, puis régénération des figures.
time python -m FuzzyRules.run_pareto_summary_v11 --days 365 || exit 3

echo "=== TERMINE ==="
