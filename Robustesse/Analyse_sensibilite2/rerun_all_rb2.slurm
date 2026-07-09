#!/usr/bin/bash -l
# ============================================================================
# rerun_all_rb2.slurm -- re-run CIBLE RB2/RB2(SoH) de TOUS les axes de
# sensibilite (splice + regeneration) sur Helios (MesoFC).
# ----------------------------------------------------------------------------
# Un seul noeud, parallelisme par processus (ProcessPoolExecutor du code Python).
# NE PREND AUCUN ARGUMENT : lance rerun_all_rb2.py (qui enchaine les axes).
#
#     sbatch rerun_all_rb2.slurm
#
# A EDITER si besoin : --time, --mail-user. Fichier transfere depuis Windows :
# passer dos2unix dessus (dos2unix rerun_all_rb2.slurm).
# ============================================================================

#SBATCH --job-name=rerun_rb2
#SBATCH --output=%x.%J.out        ## %x=nom du job, %J=id du job
#SBATCH --error=%x.%J.out
#SBATCH --time=06:00:00           ## walltime : ~1-1.5 h attendu, marge large
#SBATCH --partition=smp           ## noeuds 32 coeurs / 96 Go
#SBATCH --cpus-per-task=32        ## 1 tache, 32 coeurs (= 32 workers Python)
#SBATCH --mem=80G                 ## <= 96 Go (RAM du noeud smp)
## Notifications mail DESACTIVEES par defaut. Pour les activer, decommente :
# #SBATCH --mail-user=ton.adresse@exemple.fr
# #SBATCH --mail-type=END,FAIL

# --- Environnement ----------------------------------------------------------
# Une seule thread BLAS PAR PROCESSUS (32 processus Python -> pas de sur-souscription).
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export MPLBACKEND=Agg
# $HOME en lecture seule sur les noeuds : cache matplotlib dans $WORK.
export MPLCONFIGDIR="$WORK/.mplconfig"
mkdir -p "$MPLCONFIGDIR"

# Donnees d'entree (gros CSV profils PV/conso), provisionnees dans $WORK
# (cf. MESOCENTRE.md). Lu par Init via GENIAL_DATA_DIR.
export GENIAL_DATA_DIR="$WORK/genial_data"

module purge
module load anaconda3@2022.10/gcc-12.1.0
# conda activate genial   ## si environnement dedie

# --- Diagnostics ------------------------------------------------------------
echo "Job $SLURM_JOB_ID  noeud=$(hostname)  debut=$(date)"
echo "cpus-per-task=$SLURM_CPUS_PER_TASK"
python -c "import numpy,scipy,matplotlib,sympy;print('numpy',numpy.__version__,'scipy',scipy.__version__)"

# --- Execution --------------------------------------------------------------
# On se place dans Analyse_sensibilite : rerun_all_rb2.py y ecrit results_meso/.
cd "$SLURM_SUBMIT_DIR" || exit 1

# Pas de "srun" : UN processus Python qui voit ses 32 coeurs et lance ses
# sous-processus (ProcessPoolExecutor).
time python rerun_all_rb2.py

echo "fin=$(date)"
