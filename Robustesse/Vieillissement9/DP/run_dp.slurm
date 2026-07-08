#!/usr/bin/bash -l
# ============================================================================
# run_dp.slurm -- soumission d'un calcul PD (EMS optimal) sur Helios (MesoFC)
# ----------------------------------------------------------------------------
# Noeud UNIQUE, parallelisme par processus (ProcessPoolExecutor du code Python).
# Script Python a lancer passe en ARGUMENT :
#
#     sbatch run_dp.slurm dp_gridcheck.py     ## controle de robustesse en grille
#
# A soumettre DEPUIS  $WORK/genial/Robustesse/Vieillissement8/DP
# (fins de ligne Unix : si edite sous Windows, passer dos2unix dessus).
# ============================================================================

#SBATCH --job-name=dp_genial
#SBATCH --output=%x.%J.out        ## %x=nom du job, %J=id du job
#SBATCH --error=%x.%J.out
#SBATCH --time=12:00:00           ## walltime hh:mm:ss
#SBATCH --partition=smp           ## noeuds 32 coeurs / 96 Go
#SBATCH --cpus-per-task=8         ## 1 tache, 8 coeurs (= 8 workers Python max)
#SBATCH --mem=80G                 ## <= 96 Go (RAM du noeud smp)
# #SBATCH --mail-user=ton.adresse@exemple.fr
# #SBATCH --mail-type=END,FAIL

# --- Script Python a executer (argument 1, defaut = dp_gridcheck.py) --------
SCRIPT="${1:-dp_gridcheck.py}"

# --- Environnement ----------------------------------------------------------
# 1 thread BLAS PAR PROCESSUS (on a deja N processus Python : pas de
# sur-souscription numpy/scipy).
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export MPLBACKEND=Agg
export MPLCONFIGDIR="$WORK/.mplconfig"
mkdir -p "$MPLCONFIGDIR"

# Dossier des donnees d'entree (profils PV/conso). Lu par Init via GENIAL_DATA_DIR.
export GENIAL_DATA_DIR="$WORK/genial_data"

module purge
module load anaconda3@2022.10/gcc-12.1.0
# conda activate genial   ## si environnement dedie

# --- Diagnostics ------------------------------------------------------------
echo "Job $SLURM_JOB_ID  noeud=$(hostname)  script=$SCRIPT"
echo "cpus-per-task=$SLURM_CPUS_PER_TASK  debut=$(date)"
python -c "import numpy,scipy,sympy;print('numpy',numpy.__version__,'scipy',scipy.__version__)"

# --- Execution --------------------------------------------------------------
cd "$SLURM_SUBMIT_DIR" || exit 1
time python "$SCRIPT"

echo "fin=$(date)"
