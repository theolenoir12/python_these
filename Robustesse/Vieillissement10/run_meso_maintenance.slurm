#!/usr/bin/bash -l
# P3 corrige : politiques instant/corrective/calendar/rul, N=200 mondes CRN.
# Exemples :
#   sbatch run_meso_maintenance.slurm --tvisit 6 --margin 1
#   sbatch run_meso_maintenance.slurm --tvisit 12 --margin 1

#SBATCH --job-name=p3_rul_corr
#SBATCH --output=%x.%J.out
#SBATCH --error=%x.%J.out
#SBATCH --time=24:00:00
#SBATCH --partition=smp
#SBATCH --cpus-per-task=32
#SBATCH --mem=80G

set -euo pipefail

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
    "$SCRIPT_DIR/bench_maintenance.py" \
    "$GENIAL_DATA_DIR/sidelec_roche_plate_csv.csv" \
    "$SCRIPT_DIR/Common/FC_efficiency_LU_table_power.csv" \
    "$SCRIPT_DIR/Common/ELY_efficiency_LU_table_power.csv"; do
    if [[ ! -r "$required" ]]; then
        echo "ERREUR staging incomplet : fichier absent $required" >&2
        exit 66
    fi
done

echo "job=$SLURM_JOB_ID host=$(hostname) cpus=$SLURM_CPUS_PER_TASK debut=$(date --iso-8601=seconds)"
echo "script=$SCRIPT_DIR/bench_maintenance.py args=$*"
python -c "import numpy,scipy,sympy; print('numpy',numpy.__version__,'scipy',scipy.__version__,'sympy',sympy.__version__)"
python -m py_compile "$SCRIPT_DIR/bench_maintenance.py" \
    "$SCRIPT_DIR/Common/main_init_and_loop.py" \
    "$SCRIPT_DIR/Common/main_init_and_loop_maintenance.py" \
    "$SCRIPT_DIR/Common/get_lol.py"
if [[ -r "$SCRIPT_DIR/../run_science_checks.py" ]]; then
    python -u "$SCRIPT_DIR/../run_science_checks.py"
else
    echo "AVERTISSEMENT : tests unitaires complets non stages ; soumettre ce job avec afterok sur l'invariance." >&2
fi
time python -u "$SCRIPT_DIR/bench_maintenance.py" --output-root "$SCRIPT_DIR" "$@"
echo "fin=$(date --iso-8601=seconds)"
