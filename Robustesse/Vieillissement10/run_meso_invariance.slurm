#!/usr/bin/bash -l
# Validation 25 ans obligatoire avant P1/P3 : frontières, ledger, segments,
# test nul instant=base, gels correctifs et diagnostic legacy.
#   sbatch run_meso_invariance.slurm

#SBATCH --job-name=v94_invariance
#SBATCH --output=%x.%J.out
#SBATCH --error=%x.%J.out
#SBATCH --time=01:00:00
#SBATCH --partition=smp
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G

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
    "$SCRIPT_DIR/check_cost_reset_invariance.py" \
    "$GENIAL_DATA_DIR/sidelec_roche_plate_csv.csv" \
    "$SCRIPT_DIR/Common/FC_efficiency_LU_table_power.csv" \
    "$SCRIPT_DIR/Common/ELY_efficiency_LU_table_power.csv"; do
    if [[ ! -r "$required" ]]; then
        echo "ERREUR staging incomplet : fichier absent $required" >&2
        exit 66
    fi
done

echo "job=$SLURM_JOB_ID host=$(hostname) debut=$(date --iso-8601=seconds)"
python -c "import numpy,scipy,sympy; print('numpy',numpy.__version__,'scipy',scipy.__version__,'sympy',sympy.__version__)"
python -m py_compile "$SCRIPT_DIR/check_cost_reset_invariance.py" \
    "$SCRIPT_DIR/Common/main_init_and_loop.py" \
    "$SCRIPT_DIR/Common/main_init_and_loop_maintenance.py" \
    "$SCRIPT_DIR/Common/get_lol.py"
if [[ -r "$SCRIPT_DIR/../run_science_checks.py" ]]; then
    python -u "$SCRIPT_DIR/../run_science_checks.py"
else
    echo "AVERTISSEMENT : tests unitaires complets non stages ; validation d'invariance autonome poursuivie." >&2
fi
time python -u "$SCRIPT_DIR/check_cost_reset_invariance.py" --years 25 --require-replacements
echo "fin=$(date --iso-8601=seconds)"
