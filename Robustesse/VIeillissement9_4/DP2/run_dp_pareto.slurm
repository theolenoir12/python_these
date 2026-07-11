#!/usr/bin/bash -l
# ============================================================================
# run_dp_pareto.slurm -- FRONTIERE DE PARETO deg <-> EENS (EMS optimal PD) Helios
# ----------------------------------------------------------------------------
# Lance dp_pareto.py 'full' : balaye 15 valeurs de epsilon (= poids fiabilite
# dans la resolution PD) et evalue pour chacune la PD sequentielle 25 ans sur
# la vraie boucle de vieillissement. Trace la frontiere (deg_kEUR, EENS_kWh).
#
# PARALLELISME : chaque resolution PD est MONO-THREAD -> on lance les epsilon en
# PARALLELE, 1 par coeur (ProcessPoolExecutor, max_workers = SLURM_CPUS_PER_TASK).
# => OMP/BLAS forces a 1 thread pour NE PAS sur-souscrire les coeurs.
#
#     sbatch run_dp_pareto.slurm
#
# A soumettre DEPUIS  $WORK/genial/Robustesse/Vieillissement8/DP
# (fins de ligne Unix : si edite sous Windows, passer dos2unix dessus).
#
# Sorties : results/dp_pareto_25y_51x51.txt (+ .npz + _traj.npz) et
#           dp_pareto.<JOBID>.out
# Budget  : ~15 eps x ~2.3 h MONO-THREAD, en parallele sur 16 coeurs -> ~2.5-3 h
#           de wall (le tableau .txt est re-ecrit a chaque eps fini = resultats
#           partiels surs meme si le walltime est atteint).
# ============================================================================

#SBATCH --job-name=dp_pareto
#SBATCH --output=%x.%J.out        ## %x=nom du job, %J=id du job
#SBATCH --error=%x.%J.out
#SBATCH --time=08:00:00           ## walltime (run ~3 h, marge large)
#SBATCH --partition=smp           ## noeuds 32 coeurs / 96 Go
#SBATCH --cpus-per-task=16        ## 16 workers paralleles (1 eps / coeur)
#SBATCH --mem=32G                 ## ~15 process * arrays 25 ans << 96 Go
# #SBATCH --mail-user=ton.adresse@exemple.fr
# #SBATCH --mail-type=END,FAIL

# --- Environnement ----------------------------------------------------------
# CHAQUE worker doit rester MONO-THREAD (sinon 16 process * N threads sursouscrit
# les coeurs et tout ralentit). Le parallelisme vient du ProcessPool, pas de BLAS.
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
echo "Job $SLURM_JOB_ID  noeud=$(hostname)  script=dp_pareto.py full"
echo "cpus-per-task=$SLURM_CPUS_PER_TASK (= workers paralleles)  debut=$(date)"
python -c "import numpy,scipy,sympy;print('numpy',numpy.__version__,'scipy',scipy.__version__)"

# --- Execution --------------------------------------------------------------
cd "$SLURM_SUBMIT_DIR" || exit 1
time python dp_pareto.py full

echo "fin=$(date)"
