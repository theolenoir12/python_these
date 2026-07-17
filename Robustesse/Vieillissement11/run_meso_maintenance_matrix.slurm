#!/usr/bin/bash -l
# Matrice P3 minimale reproduisant les cinq protocoles historiques avec le
# moteur corrige. Soumettre apres validation d'invariance :
#   sbatch --dependency=afterok:<JOB_INVARIANCE> run_meso_maintenance_matrix.slurm

#SBATCH --job-name=p3_matrix_corr
#SBATCH --output=%x.%A_%a.out
#SBATCH --error=%x.%A_%a.out
#SBATCH --time=24:00:00
#SBATCH --partition=smp
#SBATCH --cpus-per-task=32
#SBATCH --mem=80G
#SBATCH --array=0-4

set -euo pipefail

# Sous SLURM le script batch est copie dans le spool : BASH_SOURCE ne pointe
# plus vers le depot. On repart du repertoire de soumission.
SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)}"
if [[ ! -r "$SCRIPT_DIR/run_meso_maintenance.slurm" ]]; then
  echo "ERREUR : soumettre depuis Vieillissement9_4 ou definir un staging complet ($SCRIPT_DIR)." >&2
  exit 66
fi
case "$SLURM_ARRAY_TASK_ID" in
  0) PARAMS=(--tvisit 3  --margin 1) ;;
  1) PARAMS=(--tvisit 6  --margin 1) ;;
  2) PARAMS=(--tvisit 12 --margin 1) ;;
  3) PARAMS=(--tvisit 6  --margin 1.5) ;;
  4) PARAMS=(--tvisit 6  --margin 2) ;;
  *) echo "index de matrice invalide" >&2; exit 2 ;;
esac

exec /usr/bin/bash -l "$SCRIPT_DIR/run_meso_maintenance.slurm" "${PARAMS[@]}" "$@"
