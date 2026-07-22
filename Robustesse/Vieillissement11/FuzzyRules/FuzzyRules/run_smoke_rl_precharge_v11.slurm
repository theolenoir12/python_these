#!/usr/bin/bash -l
# Direction A du rule-learning : arbre I0 distille + REGLE de precharge (le
# mecanisme -1,6 % de la FLC-IF), face a RB1/RB2, a l'arbre I0 nu et a la FLC-IF
# promue. Boucle fermee 1 an, V11-p=2, VoLL=3. Deux scenarios de prevision :
# oracle (borne haute) et gaussian_iid (erreur realiste, graine commune).
#
# Soumettre depuis $WORK/genial/Robustesse/Vieillissement11/FuzzyRules :
#   sbatch run_smoke_rl_precharge_v11.slurm
# Resultats : FuzzyRules/runs/smoke_rl_precharge_<empreinte>/summary.json
# (+ la sortie %x.%J.out contient les tableaux imprimes).

#SBATCH --job-name=rl_precharge_v11
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

# 2) Tests unitaires du rule-learning (60 tests : dataset, arbre, forecast, precharge).
python -m unittest -v \
  FuzzyRules.tests.test_rl_dataset_v11 \
  FuzzyRules.tests.test_rl_tree_policy_v11 \
  FuzzyRules.tests.test_rl_tree_forecast_policy_v11 \
  FuzzyRules.tests.test_rl_tree_precharge_policy_v11 || exit 3

# 3) Verification que la regle de precharge est strictement appariee a la FLC-IF
#    (meme signal sur une meme sequence) + test nul.
python -m FuzzyRules.rl_tree_precharge_policy_v11 || exit 4

# 4) Benchmark boucle fermee 1 an, prevision parfaite (borne haute).
echo "=== ORACLE (depth 4) ==="
time python -m FuzzyRules.run_smoke_rl_precharge_v11 \
  --days 365 --depth 4 --forecast-scenario oracle --with-flc-if || exit 5

# 5) Robustesse : meme comparaison sous erreur de prevision LSTM iid (graine 0).
echo "=== GAUSSIAN_IID seed 0 (depth 4) ==="
time python -m FuzzyRules.run_smoke_rl_precharge_v11 \
  --days 365 --depth 4 --forecast-scenario gaussian_iid --noise-seed 0 \
  --with-flc-if || exit 6

echo "=== TERMINE ==="
