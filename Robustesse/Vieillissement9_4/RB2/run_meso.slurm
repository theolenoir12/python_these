#!/usr/bin/bash -l
# ============================================================================
# run_meso.slurm -- lanceur pour le sweep de setpoints RB2 sur Helios.
# ----------------------------------------------------------------------------
# Meme logique que Robustesse/Predictions/run_meso.slurm : le script Python est
# passe EN ARGUMENT. Noeud unique, parallelisme par processus (ProcessPoolExecutor,
# dimensionne via SLURM_CPUS_PER_TASK).
#
#     sbatch run_meso.slurm sweep_setpoints_rb2.py
#
# A soumettre DEPUIS le dossier Vieillissement8/RB2/ (le script y trouve Common/
# via son chemin, et ecrit ses sorties .txt/.pdf/.png a cote).
# Fichier transfere depuis Windows : passer  dos2unix run_meso.slurm  une fois.
# ============================================================================

#SBATCH --job-name=sweep_rb2
#SBATCH --output=%x.%J.out        ## %x=nom du job, %J=id du job
#SBATCH --error=%x.%J.out
#SBATCH --time=02:00:00           ## walltime hh:mm:ss (large ; le job dure ~qq min)
#SBATCH --partition=smp           ## noeuds 32 coeurs / 96 Go
#SBATCH --cpus-per-task=32        ## 1 tache, 32 coeurs (= workers Python max)
#SBATCH --mem=80G                 ## <= 96 Go (RAM du noeud smp)
## Notifications mail DESACTIVEES par defaut. Pour activer, decommente :
# #SBATCH --mail-user=theo.lenoir4@orange.fr
# #SBATCH --mail-type=END,FAIL

if [ -z "$1" ]; then
  echo "Usage : sbatch run_meso.slurm <script.py> [args...]" >&2
  exit 2
fi
SCRIPT="$1"; shift
SCRIPT_ARGS="$@"

# --- Environnement ----------------------------------------------------------
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export MPLBACKEND=Agg
export MPLCONFIGDIR="$WORK/.mplconfig"
mkdir -p "$MPLCONFIGDIR"

# Profil PV/conso (gros CSV) provisionne une fois dans $WORK (cf. procedure).
export GENIAL_DATA_DIR="$WORK/genial_data"

module purge
module load anaconda3@2022.10/gcc-12.1.0
# conda activate genial    # si environnement dedie

echo "Job $SLURM_JOB_ID  noeud=$(hostname)  script=$SCRIPT  args=$SCRIPT_ARGS"
echo "cpus-per-task=$SLURM_CPUS_PER_TASK  debut=$(date)"
python -c "import numpy,scipy,sympy;print('numpy',numpy.__version__,'scipy',scipy.__version__)"

# Dossier de soumission = Vieillissement8/RB2/ (Common/ trouve via le chemin du script).
cd "$SLURM_SUBMIT_DIR" || exit 1

time python "$SCRIPT" $SCRIPT_ARGS

echo "fin=$(date)"
