#!/usr/bin/bash -l
# ============================================================================
# run_sweep_rul.slurm -- BALAYAGE (RUL_ELY_REF, EXP_ELY) de RB2(RUL) sur Helios
# ----------------------------------------------------------------------------
# Noeud UNIQUE, parallelisme par processus (ProcessPoolExecutor du code Python).
# Lance sweep_rul.py : chaque couple de la grille = une simu 25 ans, distribuee
# sur les coeurs du noeud. Sorties sweep_rul.{csv,txt} dans le dossier Predictions.
#
#     sbatch run_sweep_rul.slurm
#
# A EDITER avant le premier lancement : --mail-user (optionnel) et, si besoin,
# --time. Fichier transfere depuis Windows : passer dos2unix dessus.
# ============================================================================

#SBATCH --job-name=sweep_rul
#SBATCH --output=%x.%J.out        ## %x=nom du job, %J=id du job
#SBATCH --error=%x.%J.out
#SBATCH --time=08:00:00           ## walltime hh:mm:ss (large ; ~30 couples x qq min)
#SBATCH --partition=smp           ## noeuds 32 coeurs / 96 Go
#SBATCH --cpus-per-task=32        ## 1 tache, 32 coeurs (= 32 workers Python)
#SBATCH --mem=80G                 ## <= 96 Go (RAM du noeud smp)
## Notifications mail DESACTIVEES par defaut. Pour les activer, decommente :
# #SBATCH --mail-user=PRENOM.NOM@univ-fcomte.fr
# #SBATCH --mail-type=END,FAIL

# --- Environnement ----------------------------------------------------------
# Une seule thread BLAS PAR PROCESSUS (deja N processus Python) -> pas de
# sur-souscription.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export MPLBACKEND=Agg
# $HOME en lecture seule sur les noeuds : cache matplotlib dans $WORK.
export MPLCONFIGDIR="$WORK/.mplconfig"
mkdir -p "$MPLCONFIGDIR"

# Profil PV/conso (gros CSV) : provisionne une fois dans $WORK (cf. procedure).
# Lu par Predictions/Common/Init_EMR_MG_v16_python.py via GENIAL_DATA_DIR.
export GENIAL_DATA_DIR="$WORK/genial_data"

module purge
module load anaconda3@2022.10/gcc-12.1.0
# Si environnement conda dedie : decommente
# conda activate genial

# --- Diagnostics ------------------------------------------------------------
echo "Job $SLURM_JOB_ID  noeud=$(hostname)"
echo "cpus-per-task=$SLURM_CPUS_PER_TASK  debut=$(date)"
python -c "import numpy,scipy,sympy;print('numpy',numpy.__version__,'scipy',scipy.__version__)"

# --- Execution --------------------------------------------------------------
# On se place dans le dossier de soumission (Predictions) : Common/ et les
# sorties (sweep_rul.{csv,txt}) y sont a cote du script.
cd "$SLURM_SUBMIT_DIR" || exit 1

# Pas de "srun" : UN seul processus Python qui voit ses 32 coeurs et lance
# lui-meme ses sous-processus (ProcessPoolExecutor, dimensionne via
# SLURM_CPUS_PER_TASK).
time python sweep_rul.py

echo "fin=$(date)"
