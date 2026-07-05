#!/usr/bin/bash -l
# ============================================================================
# run_gen_detailed_figs.slurm -- FIGURES DETAILLEES PD par epsilon sur Helios
# ----------------------------------------------------------------------------
# Pour quelques eps du front de Pareto PD, re-simule la politique PD-sequentielle
# (re-solve DP + boucle 25 ans EXACTE) et produit les MEMES figures que les
# autres strategies via Common.main_plot.run_main_plot -- notamment
# everything_combined_v2_2.pdf et all_aging_2.pdf.
#
# JOB ARRAY : chaque eps est INDEPENDANT (re-simulation complete, ~5-6 h chacun)
# -> 1 tache d'array par eps, lancees en parallele par l'ordonnanceur. Chaque
# tache est un PROCESS UNIQUE (la PD sequentielle est intrinsequement sequentielle
# annee apres annee) ; on autorise numpy a utiliser les coeurs alloues.
#
# eps choisis (3 regimes du front) :
#   index 0 -> eps=0.05 : MIN degradation (LPSP eleve)
#   index 1 -> eps=4    : COUDE (UNIF@3 minimal, meilleur compromis)
#   index 2 -> eps=50   : MIN LPSP (fiabilite max)
#
#     sbatch run_gen_detailed_figs.slurm
#
# A soumettre DEPUIS  $WORK/genial/Robustesse/Vieillissement8/DP
# (fins de ligne Unix : si edite sous Windows, passer dos2unix dessus).
#
# Sorties : figures_pareto_meso/eps_<val>/Figures/<n>h/*.pdf
#           gen_detailed_figs.<JOBID>_<ARRAYID>.out
# ============================================================================

#SBATCH --job-name=gen_detailed_figs
#SBATCH --output=%x.%A_%a.out      ## %x=nom du job, %A=id du job, %a=index array
#SBATCH --error=%x.%A_%a.out
#SBATCH --array=0-2                ## 3 eps -> 3 taches independantes
#SBATCH --time=10:00:00            ## walltime / tache (run ~5-6 h, marge large)
#SBATCH --partition=smp            ## noeuds 32 coeurs / 96 Go
#SBATCH --cpus-per-task=4          ## process unique ; quelques coeurs pour numpy
#SBATCH --mem=16G                  ## << 96 Go (arrays 25 ans + tables PD)
# #SBATCH --mail-user=ton.adresse@exemple.fr
# #SBATCH --mail-type=END,FAIL

# --- Environnement ----------------------------------------------------------
# Process unique par tache : on autorise numpy a utiliser les coeurs alloues.
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export MPLBACKEND=Agg
export MPLCONFIGDIR="$WORK/.mplconfig"
mkdir -p "$MPLCONFIGDIR"

# Dossier des donnees d'entree (profils PV/conso). Lu par Init via GENIAL_DATA_DIR.
# gen_detailed_figs.py fait un setdefault -> cette valeur exportee est conservee.
export GENIAL_DATA_DIR="$WORK/genial_data"

module purge
module load anaconda3@2022.10/gcc-12.1.0
# conda activate genial   ## si environnement dedie

# --- eps de cette tache d'array ---------------------------------------------
EPS=(0.05 4 50)
E=${EPS[$SLURM_ARRAY_TASK_ID]}

# --- Diagnostics ------------------------------------------------------------
echo "Job $SLURM_JOB_ID  tache $SLURM_ARRAY_TASK_ID  eps=$E  noeud=$(hostname)"
echo "cpus-per-task=$SLURM_CPUS_PER_TASK  debut=$(date)"
python -c "import numpy,scipy,sympy;print('numpy',numpy.__version__,'scipy',scipy.__version__)"

# --- Execution --------------------------------------------------------------
cd "$SLURM_SUBMIT_DIR" || exit 1
time python gen_detailed_figs.py "$E"

echo "fin=$(date)"
