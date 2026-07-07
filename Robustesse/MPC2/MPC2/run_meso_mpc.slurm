#!/usr/bin/bash -l
# ============================================================================
# run_meso_mpc.slurm -- banc d'essai MPC(LP) vs RB (dossier MPC/) sur Helios (MesoFC)
# ----------------------------------------------------------------------------
# Noeud UNIQUE, parallelisme par processus (ProcessPoolExecutor du code Python).
# Le MPC resout un LP HiGHS (scipy.optimize.linprog) A CHAQUE PAS : compter
# ~35-45 min par run 25 ans en H=24 et ~80-95 min en H=48 (vs ~30 min pour un
# run RB). Dimensionnement par defaut : N=100 graines -> ~12-16 h sur 32 coeurs.
#
#     sbatch run_meso_mpc.slurm                    # bench nominal, N=100, 25 ans
#     sbatch run_meso_mpc.slurm 100 25 --omni      # + plafonds omniscients MPC
#     sbatch run_meso_mpc.slurm 32 25 --sweep h    # horizon 6/12/24/36/48
#     sbatch run_meso_mpc.slurm 32 25 --sweep pareto  # front MPC (VoLL interne)
#     sbatch run_meso_mpc.slurm 32 25 --sweep vh2  # valeur terminale H2
#
# Fichier transfere depuis Windows : passer dos2unix dessus.
# ============================================================================

#SBATCH --job-name=mpc
#SBATCH --output=%x.%J.out        ## %x=nom du job, %J=id du job
#SBATCH --error=%x.%J.out
#SBATCH --time=24:00:00           ## walltime hh:mm:ss
#SBATCH --partition=smp           ## noeuds 32 coeurs / 96 Go
#SBATCH --cpus-per-task=32        ## 1 tache, 32 coeurs (= 32 workers Python)
#SBATCH --mem=80G                 ## <= 96 Go (RAM du noeud smp)

# --- Arguments : N graines (defaut 100), n_years (defaut 25), options -------
N_SAMPLES="${1:-100}"
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
python -c "from scipy.optimize import linprog;print('linprog HiGHS OK')"

# --- Execution --------------------------------------------------------------
cd "$SLURM_SUBMIT_DIR" || exit 1
time python bench_mpc.py "$N_SAMPLES" "$N_YEARS" $EXTRA_ARGS

echo "fin=$(date)"
