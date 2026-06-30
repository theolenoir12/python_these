#!/usr/bin/bash -l
# ============================================================================
# run_meso_pred.slurm -- sensibilite au BRUIT DE PREDICTION sur Helios (MesoFC)
# ----------------------------------------------------------------------------
# Noeud UNIQUE, parallelisme par processus (ProcessPoolExecutor du code Python).
# Lance sens_pred_noise.py avec un GRAND nombre de tirages Monte-Carlo pour des
# ellipses de Pareto lisses.
#
#     sbatch run_meso_pred.slurm                 # defaut : N=200 tirages, 25 ans
#     sbatch run_meso_pred.slurm 400             # N=400 tirages, 25 ans
#     sbatch run_meso_pred.slurm 200 25          # N et n_years explicites
#
# A EDITER avant le premier lancement : --mail-user (optionnel) et, si besoin,
# --time. Fichier transfere depuis Windows : passer dos2unix dessus.
# ============================================================================

#SBATCH --job-name=sens_pred
#SBATCH --output=%x.%J.out        ## %x=nom du job, %J=id du job
#SBATCH --error=%x.%J.out
#SBATCH --time=12:00:00           ## walltime hh:mm:ss (large : ~1h suffit a N=400)
#SBATCH --partition=smp           ## noeuds 32 coeurs / 96 Go
#SBATCH --cpus-per-task=32        ## 1 tache, 32 coeurs (= 32 workers Python)
#SBATCH --mem=80G                 ## <= 96 Go (RAM du noeud smp)
## Notifications mail DESACTIVEES par defaut. Pour les activer, decommente :
# #SBATCH --mail-user=PRENOM.NOM@univ-fcomte.fr
# #SBATCH --mail-type=END,FAIL

# --- Arguments : N tirages (defaut 200), n_years (defaut 25) -----------------
N_SAMPLES="${1:-200}"
N_YEARS="${2:-25}"

# --- Environnement ----------------------------------------------------------
# Une seule thread BLAS PAR PROCESSUS (32 processus Python deja) -> pas de
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
echo "Job $SLURM_JOB_ID  noeud=$(hostname)  N=$N_SAMPLES  n_years=$N_YEARS"
echo "cpus-per-task=$SLURM_CPUS_PER_TASK  debut=$(date)"
python -c "import numpy,scipy,matplotlib,sympy;print('numpy',numpy.__version__,'scipy',scipy.__version__)"

# --- Execution --------------------------------------------------------------
# On se place dans le dossier de soumission (Predictions) : Common/ et les
# sorties (sens_pred_noise.{pdf,png,txt}) y sont a cote du script.
cd "$SLURM_SUBMIT_DIR" || exit 1

# Pas de "srun" : UN seul processus Python qui voit ses 32 coeurs et lance
# lui-meme ses sous-processus (ProcessPoolExecutor, dimensionne via
# SLURM_CPUS_PER_TASK).
time python sens_pred_noise.py "$N_SAMPLES" "$N_YEARS"

echo "fin=$(date)"
