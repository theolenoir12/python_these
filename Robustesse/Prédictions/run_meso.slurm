#!/usr/bin/bash -l
# ============================================================================
# run_meso.slurm -- lanceur GENERIQUE pour les jobs Predictions/ sur Helios.
# ----------------------------------------------------------------------------
# Le script Python a lancer est passe EN ARGUMENT, comme dans la procedure
# Analyse_sensibilite (sbatch run_meso.slurm <script>.py [args...]).
# Noeud UNIQUE, parallelisme par processus (ProcessPoolExecutor du code Python,
# dimensionne via SLURM_CPUS_PER_TASK).
#
#     sbatch run_meso.slurm sweep_rul.py
#     sbatch run_meso.slurm mc_rul_uncertainty.py 200
#
# A EDITER au premier coup : --mail-user (optionnel), eventuellement --time.
# Fichier transfere depuis Windows : passer  dos2unix run_meso.slurm  une fois.
# ============================================================================

#SBATCH --job-name=pred_job
#SBATCH --output=%x.%J.out        ## %x=nom du job, %J=id du job
#SBATCH --error=%x.%J.out
#SBATCH --time=12:00:00           ## walltime hh:mm:ss (large)
#SBATCH --partition=smp           ## noeuds 32 coeurs / 96 Go
#SBATCH --cpus-per-task=32        ## 1 tache, 32 coeurs (= 32 workers Python)
#SBATCH --mem=80G                 ## <= 96 Go (RAM du noeud smp)
## Notifications mail DESACTIVEES par defaut. Pour activer, decommente :
# #SBATCH --mail-user=PRENOM.NOM@univ-fcomte.fr
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

# Dossier de soumission = Predictions/ (Common/ et sorties y sont a cote).
cd "$SLURM_SUBMIT_DIR" || exit 1

time python "$SCRIPT" $SCRIPT_ARGS

echo "fin=$(date)"
