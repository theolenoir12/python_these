#!/usr/bin/bash -l
# ============================================================================
# run_meso_dwell.slurm -- banc P4 "demarrages ELY / prevision" sur Helios.
# ----------------------------------------------------------------------------
# Lance bench_dwell_ely.py (P4 : filtre previsionnel de demarrage ELY,
# familles base/minoff/omni/noisy + diagnostic du gisement reversible).
# Noeud unique, parallelisme par processus (ProcessPoolExecutor,
# dimensionne via SLURM_CPUS_PER_TASK).
#
#     sbatch run_meso_dwell.slurm                       # nominal : 25 ans
#     sbatch run_meso_dwell.slurm --seeds 64            # variante
#
<<<<<<< HEAD
# A soumettre DEPUIS le dossier du banc (le script y trouve Common/ et ecrit
# dwell_ely_25y.{txt,pdf,png} a cote).
# Fichier transfere depuis Windows : passer  dos2unix  une fois.
# Cout indicatif : ~170 runs 25 ans ~5-10 min / 32 coeurs (40 s par run).
# Pas de cache : le banc re-execute tout (bon marche).
=======
# A soumettre DEPUIS le dossier du banc (ou avec GENIAL_V94_DIR). Les sorties
# vont dans runs/p4_dwell_<empreinte>/.
# Fichier transfere depuis Windows : passer  dos2unix  une fois.
# Cout indicatif : ~170 runs 25 ans ~5-10 min / 32 coeurs (40 s par run).
# Pas de reprise automatique : le banc re-execute tout (bon marche), avec
# verrou anti-double-ecriture et provenance finale.
>>>>>>> e8aeb969d592417a0e02eb6d96fc543e220d13ef
# ============================================================================

#SBATCH --job-name=dwell_ely
#SBATCH --output=%x.%J.out        ## %x=nom du job, %J=id du job
#SBATCH --error=%x.%J.out
#SBATCH --time=24:00:00           ## walltime hh:mm:ss
#SBATCH --partition=smp           ## noeuds 32 coeurs / 96 Go
#SBATCH --cpus-per-task=32        ## 1 tache, 32 coeurs (= workers Python)
#SBATCH --mem=80G                 ## <= 96 Go (RAM du noeud smp)
## Notifications mail DESACTIVEES par defaut. Pour activer, decommente :
# #SBATCH --mail-user=theo.lenoir4@orange.fr
# #SBATCH --mail-type=END,FAIL

<<<<<<< HEAD
# --- Environnement ----------------------------------------------------------
=======
set -euo pipefail

# --- Environnement ----------------------------------------------------------
SCRIPT_DIR="${GENIAL_V94_DIR:-${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)}}"
>>>>>>> e8aeb969d592417a0e02eb6d96fc543e220d13ef
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export MPLBACKEND=Agg
<<<<<<< HEAD
export MPLCONFIGDIR="$WORK/.mplconfig"
mkdir -p "$MPLCONFIGDIR"

# Profil PV/conso (gros CSV) provisionne une fois dans $WORK (cf. MESOCENTRE.md).
export GENIAL_DATA_DIR="$WORK/genial_data"
=======
export MPLCONFIGDIR="${WORK:?WORK non defini}/.mplconfig"
mkdir -p "$MPLCONFIGDIR"

# Profil PV/conso (gros CSV) provisionne une fois dans $WORK (cf. MESOCENTRE.md).
export GENIAL_DATA_DIR="${GENIAL_DATA_DIR:-$WORK/genial_data}"
>>>>>>> e8aeb969d592417a0e02eb6d96fc543e220d13ef

module purge
module load anaconda3@2022.10/gcc-12.1.0
# conda activate genial    # si environnement dedie

echo "Job $SLURM_JOB_ID  noeud=$(hostname)  args=$@"
echo "cpus-per-task=$SLURM_CPUS_PER_TASK  debut=$(date)"
python -c "import numpy,scipy,sympy;print('numpy',numpy.__version__,'scipy',scipy.__version__)"

<<<<<<< HEAD
# Dossier de soumission = Vieillissement9/ (Common/ trouve via le chemin du script).
cd "$SLURM_SUBMIT_DIR" || exit 1

time python bench_dwell_ely.py "$@"
=======
for required in \
    "$SCRIPT_DIR/bench_dwell_ely.py" \
    "$GENIAL_DATA_DIR/sidelec_roche_plate_csv.csv" \
    "$SCRIPT_DIR/Common/FC_efficiency_LU_table_power.csv" \
    "$SCRIPT_DIR/Common/ELY_efficiency_LU_table_power.csv"; do
    if [[ ! -r "$required" ]]; then
        echo "ERREUR staging incomplet : fichier absent $required" >&2
        exit 66
    fi
done

python -m py_compile "$SCRIPT_DIR/bench_dwell_ely.py" \
    "$SCRIPT_DIR/Common/main_init_and_loop.py" "$SCRIPT_DIR/Common/get_lol.py"
if [[ -r "$SCRIPT_DIR/../run_science_checks.py" ]]; then
    python -u "$SCRIPT_DIR/../run_science_checks.py"
else
    echo "AVERTISSEMENT : tests unitaires complets non stages ; soumettre ce job avec afterok sur l'invariance." >&2
fi

time python -u "$SCRIPT_DIR/bench_dwell_ely.py" --output-root "$SCRIPT_DIR" "$@"
>>>>>>> e8aeb969d592417a0e02eb6d96fc543e220d13ef

echo "fin=$(date)"
