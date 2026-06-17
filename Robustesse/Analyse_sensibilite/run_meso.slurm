#!/usr/bin/bash -l
# ============================================================================
# run_meso.slurm -- soumission d'UNE analyse de sensibilite sur Helios (MesoFC)
# ----------------------------------------------------------------------------
# Noeud UNIQUE, parallelisme par processus (ProcessPoolExecutor du code Python).
# Le script Python a lancer est passe en ARGUMENT :
#
#     sbatch run_meso.slurm sens_calendar.py
#     sbatch run_meso.slurm sens_eol.py
#     ... (sens_soh_estimation.py | sens_eol.py | sens_cweights.py |
#          sens_hthresholds.py | sens_sizing.py | sens_calendar.py)
#
# A EDITER avant le premier lancement : --mail-user et, si besoin, --time.
# Les fichiers .slurm transferes depuis Windows : passer dos2unix dessus.
# ============================================================================

#SBATCH --job-name=sens_genial
#SBATCH --output=%x.%J.out        ## %x=nom du job, %J=id du job
#SBATCH --error=%x.%J.out
#SBATCH --time=24:00:00           ## walltime hh:mm:ss (max 8 jours sur smp)
#SBATCH --partition=smp           ## noeuds 32 coeurs / 96 Go
#SBATCH --cpus-per-task=32        ## 1 tache, 32 coeurs (= 32 workers Python)
#SBATCH --mem=80G                 ## <= 96 Go (RAM du noeud smp)
## Notifications mail DESACTIVEES par defaut (pas d'adresse UFC). Pour les
## activer, decommente les 2 lignes ci-dessous et mets ton adresse :
# #SBATCH --mail-user=ton.adresse@exemple.fr
# #SBATCH --mail-type=END,FAIL

# --- Script Python a executer (argument 1, defaut = sens_calendar.py) -------
SCRIPT="${1:-sens_calendar.py}"

# --- Environnement ----------------------------------------------------------
# Une seule thread BLAS PAR PROCESSUS : on a deja 32 processus Python, il ne
# faut pas que numpy/scipy lance en plus 32 threads chacun (sur-souscription).
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export MPLBACKEND=Agg
# $HOME est en lecture seule sur les noeuds : on place le cache matplotlib
# dans $WORK (sinon avertissement + ralentissement avec 32 processus).
export MPLCONFIGDIR="$WORK/.mplconfig"
mkdir -p "$MPLCONFIGDIR"

# Dossier des donnees d'entree (gros CSV de profils PV/conso). A provisionner
# une fois dans $WORK (cf. MESOCENTRE.md). Lu par Init via GENIAL_DATA_DIR.
export GENIAL_DATA_DIR="$WORK/genial_data"

module purge
module load anaconda3@2022.10/gcc-12.1.0
# Si tu as cree un environnement conda dedie, decommente :
# conda activate genial

# --- Diagnostics ------------------------------------------------------------
echo "Job $SLURM_JOB_ID  noeud=$(hostname)  script=$SCRIPT"
echo "cpus-per-task=$SLURM_CPUS_PER_TASK  debut=$(date)"
python -c "import numpy,scipy,matplotlib,sympy;print('numpy',numpy.__version__,'scipy',scipy.__version__)"

# --- Execution --------------------------------------------------------------
# On se place dans le dossier de soumission (Analyse_sensibilite) : les sorties
# (results/) y sont ecrites a cote des scripts.
cd "$SLURM_SUBMIT_DIR" || exit 1

# Pas de "srun" ici : on veut UN seul processus Python qui voit ses 32 coeurs
# et lance lui-meme ses sous-processus (ProcessPoolExecutor).
time python "$SCRIPT"

echo "fin=$(date)"
