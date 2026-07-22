#!/usr/bin/bash -l
# ANFIS Takagi-Sugeno ordre 1, deux branches, distillé de la PD (V11-p=2).
# Compare RB1/RB2, l'arbre I0 d4 (effet de famille) et l'ANFIS I0 (+ ablation IS),
# boucle fermée 1 an. Rapporte J3, nb de règles et temps de décision amorti.
#
# Soumettre depuis $WORK/genial/Robustesse/Vieillissement11/FuzzyRules :
#   sbatch run_anfis_v11.slurm
# Resultats : FuzzyRules/runs/smoke_anfis_<empreinte>/summary.json (+ %x.%J.out).

#SBATCH --job-name=anfis_v11
#SBATCH --output=%x.%J.out
#SBATCH --error=%x.%J.out
#SBATCH --time=01:00:00
#SBATCH --partition=smp
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G

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

# 2) Tests unitaires ANFIS (moteur + policy).
python -m unittest -v \
  FuzzyRules.tests.test_anfis_ts \
  FuzzyRules.tests.test_anfis_policy_v11 || exit 3

# 3) Smoke boucle fermée 1 an : RB1/RB2, arbre I0 d4, ANFIS I0 et IS.
echo "=== ANFIS I0 + IS (mf=3), vs RB1/RB2 et arbre d4 ==="
time python -m FuzzyRules.run_anfis_v11 \
  --days 365 --n-mf 3 --information-sets I0 IS --with-tree || exit 4

# 4) Sensibilité au nombre d'appartenances (mf=2 -> 8 règles/branche).
echo "=== ANFIS I0 (mf=2) ==="
time python -m FuzzyRules.run_anfis_v11 --days 365 --n-mf 2 || exit 5

echo "=== TERMINE ==="
