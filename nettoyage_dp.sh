#!/usr/bin/bash
# Nettoyage DP : supprime les doublons non pertinents (recuperables via git).
# A lancer depuis la racine du depot, apres avoir applique dp_front_v2_COMPLET.patch.
set -e
cd "$(git rev-parse --show-toplevel)"
git rm -r -q "Robustesse/Vieillissement8/DP2/DP/DP"                 # imbrication accidentelle
git rm -r -q "Robustesse/Vieillissement8/.DP_meso/DP_CASSE_210435"  # snapshot casse
git commit -m "Nettoyage DP : suppression doublons non pertinents (imbrication + dossier casse)"
echo "Nettoyage minimal fait. Pour la consolidation COMPLETE (optionnelle) :"
echo "  git rm -r Robustesse/Vieillissement8/DP2 Robustesse/Vieillissement8/.DP_meso Robustesse/Vieillissement8/DP/results_meso2"
echo "  (le front v1+v2, le log 213210 et les trajectoires restent dans DP/results_meso/)"
