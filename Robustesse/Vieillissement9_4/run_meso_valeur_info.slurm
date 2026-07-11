#!/usr/bin/bash -l
# P1 corrige : valeur informationnelle du SoH, N=200 mondes CRN.
# Soumettre depuis Vieillissement9_4 (ou definir GENIAL_V94_DIR) :
#   sbatch run_meso_valeur_info.slurm

#SBATCH --job-name=p1_soh_corr
#SBATCH --output=%x.%J.out
#SBATCH --error=%x.%J.out
#SBATCH --time=24:00:00
#SBATCH --partition=smp
#SBATCH --cpus-per-task=32
#SBATCH --mem=80G

set -euo pipefail

# Slurm copie le batch dans /var/spool : BASH_SOURCE ne designe alors plus le
# depot. Les jobs doivent etre soumis depuis ce dossier (ou GENIAL_V94_DIR fixe).
SCRIPT_DIR="${GENIAL_V94_DIR:-${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)}}"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export MPLBACKEND=Agg
export MPLCONFIGDIR="${WORK:?WORK non defini}/.mplconfig"
export GENIAL_DATA_DIR="${GENIAL_DATA_DIR:-$WORK/genial_data}"
mkdir -p "$MPLCONFIGDIR"

module purge
module load anaconda3@2022.10/gcc-12.1.0

for required in \
    "$SCRIPT_DIR/bench_valeur_info.py" \
    "$GENIAL_DATA_DIR/sidelec_roche_plate_csv.csv" \
    "$SCRIPT_DIR/Common/FC_efficiency_LU_table_power.csv" \
    "$SCRIPT_DIR/Common/ELY_efficiency_LU_table_power.csv"; do
    if [[ ! -r "$required" ]]; then
        echo "ERREUR staging incomplet : fichier absent $required" >&2
        exit 66
    fi
done

echo "job=$SLURM_JOB_ID host=$(hostname) cpus=$SLURM_CPUS_PER_TASK debut=$(date --iso-8601=seconds)"
echo "script=$SCRIPT_DIR/bench_valeur_info.py args=$*"
python -c "import numpy,scipy,sympy; print('numpy',numpy.__version__,'scipy',scipy.__version__,'sympy',sympy.__version__)"
python -m py_compile "$SCRIPT_DIR/bench_valeur_info.py" \
    "$SCRIPT_DIR/Common/main_init_and_loop.py" "$SCRIPT_DIR/Common/get_lol.py"
if [[ -r "$SCRIPT_DIR/../run_science_checks.py" ]]; then
    python -u "$SCRIPT_DIR/../run_science_checks.py"
else
    echo "AVERTISSEMENT : tests unitaires complets non stages ; le job d'invariance afterok reste obligatoire." >&2
fi
time python -u "$SCRIPT_DIR/bench_valeur_info.py" --output-root "$SCRIPT_DIR" "$@"
echo "fin=$(date --iso-8601=seconds)"
