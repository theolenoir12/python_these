#!/usr/bin/bash -l
# Direction B du rule-learning : DAgger (Dataset Aggregation) sur l'arbre I0
# distille. A chaque iteration, on deroule l'arbre courant 1 an, on re-etiquette
# les etats visites avec l'action de la PD (grille 1 an SoH=1 de dp_core), on
# agrege au jeu enseignant, on re-ajuste l'arbre. Question : corriger la derive
# de distribution rapproche-t-il l'arbre des references RB1/RB2 ?
#
# Soumettre depuis $WORK/genial/Robustesse/Vieillissement11/FuzzyRules :
#   sbatch run_dagger_rl_v11.slurm
# Resultats : FuzzyRules/runs/dagger_rl_<empreinte>/summary.json (+ la trace
# J3-par-iteration dans la sortie %x.%J.out).

#SBATCH --job-name=rl_dagger_v11
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

# Les modules FuzzyRules se lancent depuis Vieillissement11/ (python -m ...).
cd "$SLURM_SUBMIT_DIR/.." || exit 1

# 1) Garde d'environnement.
python -c "import numpy, sklearn; print('numpy', numpy.__version__, '| sklearn', sklearn.__version__)" || exit 2

# 2) Tests unitaires du pipeline DAgger (+ arbre/forecast/precharge).
python -m unittest -v \
  FuzzyRules.tests.test_rl_dagger_v11 \
  FuzzyRules.tests.test_rl_tree_policy_v11 || exit 3

# 3) DAgger 1 an, arbre I0 depth 4, 5 iterations.
echo "=== DAgger (depth 4, 5 iterations) ==="
time python -m FuzzyRules.run_dagger_rl_v11 \
  --days 365 --depth 4 --iterations 5 || exit 4

echo "=== TERMINE ==="
