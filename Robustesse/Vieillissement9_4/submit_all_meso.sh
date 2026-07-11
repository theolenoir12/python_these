#!/usr/bin/env bash

# Soumet l'invariance, puis P1, P3 et P4 lorsque l'invariance a reussi.
# Utilisation : bash submit_all_meso.sh

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

USER_WORK_DIR="$(cd -- "$SCRIPT_DIR/../../.." && pwd)"
DATA_DIR="${GENIAL_DATA_DIR:-${WORK:-$USER_WORK_DIR}/genial_data}"

if [[ ! -r "$DATA_DIR/sidelec_roche_plate_csv.csv" ]]; then
    echo "ERREUR : fichier absent : $DATA_DIR/sidelec_roche_plate_csv.csv"
    exit 1
fi

if [[ ! -r "$SCRIPT_DIR/Common/FC_efficiency_LU_table_power.csv" ]]; then
    echo "ERREUR : fichier absent : $SCRIPT_DIR/Common/FC_efficiency_LU_table_power.csv"
    exit 1
fi

if [[ ! -r "$SCRIPT_DIR/Common/ELY_efficiency_LU_table_power.csv" ]]; then
    echo "ERREUR : fichier absent : $SCRIPT_DIR/Common/ELY_efficiency_LU_table_power.csv"
    exit 1
fi

if [[ ! -r "$SCRIPT_DIR/../reproducibility/provenance.py" ]]; then
    echo "ERREUR : fichier absent : $SCRIPT_DIR/../reproducibility/provenance.py"
    exit 1
fi

if [[ ! -r "$SCRIPT_DIR/../reproducibility/paired_stats.py" ]]; then
    echo "ERREUR : fichier absent : $SCRIPT_DIR/../reproducibility/paired_stats.py"
    exit 1
fi

export GENIAL_DATA_DIR="$DATA_DIR"

JOB_INV_RAW=$(sbatch --parsable run_meso_invariance.slurm)
JOB_INV=${JOB_INV_RAW%%;*}

JOB_P1_RAW=$(sbatch --parsable --dependency="afterok:$JOB_INV" run_meso_valeur_info.slurm)
JOB_P1=${JOB_P1_RAW%%;*}

JOB_P3_RAW=$(sbatch --parsable --dependency="afterok:$JOB_INV" run_meso_maintenance_matrix.slurm)
JOB_P3=${JOB_P3_RAW%%;*}

JOB_P4_RAW=$(sbatch --parsable --dependency="afterok:$JOB_INV" run_meso_dwell.slurm)
JOB_P4=${JOB_P4_RAW%%;*}

STATUS_FILE="$SCRIPT_DIR/DERNIERS_JOBS_SOUMIS.txt"
{
    echo "INVARIANCE=$JOB_INV"
    echo "P1=$JOB_P1"
    echo "P3=$JOB_P3"
    echo "P4=$JOB_P4"
} > "$STATUS_FILE"

echo
echo "SOUMISSION REUSSIE"
echo "INVARIANCE : $JOB_INV"
echo "P1         : $JOB_P1"
echo "P3         : $JOB_P3"
echo "P4         : $JOB_P4"
echo
echo "Les numeros sont aussi dans DERNIERS_JOBS_SOUMIS.txt"
echo "P1, P3 et P4 attendront automatiquement la reussite de l'invariance."
echo
squeue -j "$JOB_INV,$JOB_P1,$JOB_P3,$JOB_P4" || true
