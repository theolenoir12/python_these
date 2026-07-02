#!/usr/bin/bash -l
# ============================================================================
# run_meso_fable.slurm -- banc d'essai des leviers Fable sur Helios (MesoFC)
# ----------------------------------------------------------------------------
# Noeud UNIQUE, parallelisme par processus (ProcessPoolExecutor du code Python).
#
#     sbatch run_meso_fable.slurm                    # bench nominal, N=200, 25 ans
#     sbatch run_meso_fable.slurm 200 25 --omni      # + bornes omniscientes
#     sbatch run_meso_fable.slurm 64 25 --sweep prop # balayage TAU
#     sbatch run_meso_fable.slurm 64 25 --sweep proba
#
# Fichier transfere depuis Windows : passer dos2unix dessus.
# ============================================================================

#SBATCH --job-name=fable
#SBATCH --output=%x.%J.out        ## %x=nom du job, %J=id du job
#SBATCH --error=%x.%J.out
#SBATCH --time=12:00:00           ## walltime hh:mm:ss
#SBATCH --partition=smp           ## noeuds 32 coeurs / 96 Go
#SBATCH --cpus-per-task=32        ## 1 tache, 32 coeurs (= 32 workers Python)
#SBATCH --mem=80G                 ## <= 96 Go (RAM du noeud smp)

# --- Arguments : N graines (defaut 200), n_years (defaut 25), options -------
N_SAMPLES="${1:-200}"
N_YEARS="${2:-25}"
shift 2 2>/dev/null
EXTRA_ARGS="$@"

# --- Environnement ----------------------------------------------------------
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export MPLBACKEND=Agg
export MPLCONFIGDIR="$WORK/.mplconfig"
mkdir -p "$MPLCONFIGDIR"

# Profil PV/conso (gros CSV) : provisionne une fois dans $WORK (cf. procedure).
export GENIAL_DATA_DIR="$WORK/genial_data"

module purge
module load anaconda3@2022.10/gcc-12.1.0

# --- Diagnostics ------------------------------------------------------------
echo "Job $SLURM_JOB_ID  noeud=$(hostname)  N=$N_SAMPLES  n_years=$N_YEARS  extra=$EXTRA_ARGS"
echo "cpus-per-task=$SLURM_CPUS_PER_TASK  debut=$(date)"
python -c "import numpy,scipy;print('numpy',numpy.__version__,'scipy',scipy.__version__)"

# --- Execution --------------------------------------------------------------
cd "$SLURM_SUBMIT_DIR" || exit 1
time python bench_fable.py "$N_SAMPLES" "$N_YEARS" $EXTRA_ARGS

echo "fin=$(date)"
