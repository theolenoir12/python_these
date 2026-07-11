#!/usr/bin/bash -l
# ============================================================================
# run_dp_aging.slurm -- PD avec VIEILLISSEMENT 25 ans (EMS optimal) sur Helios
# ----------------------------------------------------------------------------
# Lance dp_aging.py en mode 'full' (horizon 25 ans, grille 51x51) : compare
# RB2 / PD BoL-figee / PD sequentielle sur le cout unifie (deg + VoLL*EENS).
# PROCESSUS UNIQUE (la PD sequentielle est intrinsequement sequentielle : chaque
# annee depend du vieillissement de la precedente) -> pas de ProcessPool ici.
#
#     sbatch run_dp_aging.slurm
#
# A soumettre DEPUIS  $WORK/genial/Robustesse/Vieillissement8/DP
# (fins de ligne Unix : si edite sous Windows, passer dos2unix dessus).
#
# Sorties : results/dp_aging_25y_51x51.txt (+ .npz) et dp_aging.<JOBID>.out
# ============================================================================

#SBATCH --job-name=dp_aging
#SBATCH --output=%x.%J.out        ## %x=nom du job, %J=id du job
#SBATCH --error=%x.%J.out
#SBATCH --time=08:00:00           ## walltime (run ~2-3 h, marge large)
#SBATCH --partition=smp           ## noeuds 32 coeurs / 96 Go
#SBATCH --cpus-per-task=4         ## process unique ; quelques coeurs pour numpy
#SBATCH --mem=16G                 ## << 96 Go (arrays 25 ans + tables PD)
# #SBATCH --mail-user=ton.adresse@exemple.fr
# #SBATCH --mail-type=END,FAIL

# --- Environnement ----------------------------------------------------------
# Process unique : on autorise numpy a utiliser les coeurs alloues.
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export MPLBACKEND=Agg
export MPLCONFIGDIR="$WORK/.mplconfig"
mkdir -p "$MPLCONFIGDIR"

# Dossier des donnees d'entree (profils PV/conso). Lu par Init via GENIAL_DATA_DIR.
export GENIAL_DATA_DIR="$WORK/genial_data"

module purge
module load anaconda3@2022.10/gcc-12.1.0
# conda activate genial   ## si environnement dedie

# --- Diagnostics ------------------------------------------------------------
echo "Job $SLURM_JOB_ID  noeud=$(hostname)  script=dp_aging.py full"
echo "cpus-per-task=$SLURM_CPUS_PER_TASK  debut=$(date)"
python -c "import numpy,scipy,sympy;print('numpy',numpy.__version__,'scipy',scipy.__version__)"

# --- Execution --------------------------------------------------------------
cd "$SLURM_SUBMIT_DIR" || exit 1
time python dp_aging.py full

echo "fin=$(date)"
