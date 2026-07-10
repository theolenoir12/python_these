#!/usr/bin/bash -l
# ============================================================================
# run_meso_maintenance.slurm -- banc "valeur de la RUL / maintenance" sur Helios.
# ----------------------------------------------------------------------------
# Lance bench_maintenance.py (valeur de la RUL : fenetres de maintenance
# insulaires + cout fixe par intervention, politiques instant/corrective/
# calendar/rul). Noeud unique, parallelisme par processus (ProcessPoolExecutor,
# dimensionne via SLURM_CPUS_PER_TASK).
#
#     sbatch run_meso_maintenance.slurm                 # nominal : 25 ans, N=200, T=6 mois
#     sbatch run_meso_maintenance.slurm --tvisit 3      # autre periode (job separe)
#
# A soumettre DEPUIS le dossier du banc (le script y trouve Common/ et ecrit
# maintenance_25y_T<mois>m.{txt,pdf,png} a cote).
# Fichier transfere depuis Windows : passer  dos2unix  une fois.
# Cout indicatif : ~800 runs 25 ans ~20-30 min / 32 coeurs (40 s par run 25 ans,
# mesure job 214867). Reuse : avec le txt precedent, seuls les runs manquants
# sont relances. --fresh pour tout re-payer. Un txt par --tvisit/--margin.
# ============================================================================

#SBATCH --job-name=maintenance
#SBATCH --output=%x.%J.out        ## %x=nom du job, %J=id du job
#SBATCH --error=%x.%J.out
#SBATCH --time=24:00:00           ## walltime hh:mm:ss
#SBATCH --partition=smp           ## noeuds 32 coeurs / 96 Go
#SBATCH --cpus-per-task=32        ## 1 tache, 32 coeurs (= workers Python)
#SBATCH --mem=80G                 ## <= 96 Go (RAM du noeud smp)
## Notifications mail DESACTIVEES par defaut. Pour activer, decommente :
# #SBATCH --mail-user=theo.lenoir4@orange.fr
# #SBATCH --mail-type=END,FAIL

# --- Environnement ----------------------------------------------------------
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export MPLBACKEND=Agg
export MPLCONFIGDIR="$WORK/.mplconfig"
mkdir -p "$MPLCONFIGDIR"

# Profil PV/conso (gros CSV) provisionne une fois dans $WORK (cf. MESOCENTRE.md).
export GENIAL_DATA_DIR="$WORK/genial_data"

module purge
module load anaconda3@2022.10/gcc-12.1.0
# conda activate genial    # si environnement dedie

echo "Job $SLURM_JOB_ID  noeud=$(hostname)  args=$@"
echo "cpus-per-task=$SLURM_CPUS_PER_TASK  debut=$(date)"
python -c "import numpy,scipy,sympy;print('numpy',numpy.__version__,'scipy',scipy.__version__)"

# Dossier de soumission = Vieillissement9/ (Common/ trouve via le chemin du script).
cd "$SLURM_SUBMIT_DIR" || exit 1

time python bench_maintenance.py "$@"

echo "fin=$(date)"
