#!/usr/bin/bash -l
# ============================================================================
# run_dp_pareto.slurm -- FRONTIERE DE PARETO deg <-> EENS (EMS optimal PD) Helios
# ----------------------------------------------------------------------------
# Lance dp_pareto.py 'full' : balaye 19 valeurs de epsilon (= poids fiabilite
# dans la resolution PD) et evalue pour chacune la PD sequentielle 25 ans sur
# la vraie boucle de vieillissement. Trace la frontiere (deg_kEUR, EENS_kWh).
#
# VARIANTE V2 (defaut) : projection du vieillissement + rollout exact
# (cf. note_dp_front_v2.txt). Corrige les ~10 kEUR de degradation FC "high"
# NON pricee du v1 (derive intra-annuelle du seuil 0.8*Pmax vieilli), qui
# laissait RB2(SoH_all)/RB2(SoH_all+Pred) passer sous le front. Sortie taggee
# *_v2 -> N'ECRASE PAS le front v1. Pour rejouer le v1 : DP_PARETO_V2=0.
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
# Sorties : results/dp_pareto_25y_51x51_v2.txt (+ .npz + _traj.npz) et
#           dp_pareto.<JOBID>.out
# Budget  : v1 mesurait ~6.7 h MONO-THREAD/eps ; v2 ~+30% (rollout) -> ~9 h.
#           19 eps sur 19 coeurs en parallele -> ~9 h de wall (walltime 14 h,
#           marge). Le tableau .txt est re-ecrit a CHAQUE eps fini = resultats
#           partiels surs meme si le walltime est atteint.
# ============================================================================

#SBATCH --job-name=dp_pareto
#SBATCH --output=%x.%J.out        ## %x=nom du job, %J=id du job
#SBATCH --error=%x.%J.out
#SBATCH --time=14:00:00           ## walltime (v2 ~9 h/eps mono-thread ; marge)
#SBATCH --partition=smp           ## noeuds 32 coeurs / 96 Go
#SBATCH --cpus-per-task=19        ## 19 workers paralleles (1 eps / coeur, 19 eps en v2)
#SBATCH --mem=48G                 ## ~19 process * (arrays 25 ans + policy + ckpt V)
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

# Variante DP : V2 (projection vieillissement + rollout) par defaut.
# Mettre 0 pour rejouer exactement le v1 (front legacy, sortie sans suffixe _v2).
export DP_PARETO_V2=1

module purge
module load anaconda3@2022.10/gcc-12.1.0
# conda activate genial   ## si environnement dedie

# --- Diagnostics ------------------------------------------------------------
echo "Job $SLURM_JOB_ID  noeud=$(hostname)  script=dp_pareto.py full  DP_PARETO_V2=$DP_PARETO_V2"
echo "cpus-per-task=$SLURM_CPUS_PER_TASK (= workers paralleles)  debut=$(date)"
python -c "import numpy,scipy,sympy;print('numpy',numpy.__version__,'scipy',scipy.__version__)"

# --- Execution --------------------------------------------------------------
cd "$SLURM_SUBMIT_DIR" || exit 1
time python dp_pareto.py full

echo "fin=$(date)"
