# -*- coding: utf-8 -*-
"""
compute_prono_sigma_horizons.py -- INCERTITUDE D'ESTIMATION DE LA RUL vs HORIZON.
=================================================================================
Variante de compute_prono_sigma.py qui caracterise comment l'ecart-type
d'estimation de la RUL depend de l'horizon (= a quelle distance se trouve la fin
de vie au moment ou on l'estime). On deplace l'instant d'estimation t_now le long
de la trajectoire de SoH (fractions FRACS de l'historique) ; pour chaque position,
inference MC-dropout (M passes) et statistiques du temps de franchissement du seuil.

A LANCER dans  Python/Pronostic SoH/  (modele .keras + SoH_ely_2.csv presents).
Leger (~quelques min). Sortie : tableau + sigma_rul_vs_horizon.csv.

Usage :  python compute_prono_sigma_horizons.py
"""
import os
import time
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

FILENAME   = 'SoH_ely_2.csv'
MODEL_PATH = 'resultats_mc_dropout/modele_mc_dropout_scaler2.keras'
WS         = 100
M          = 200          # passes MC (200 suffit pour un ecart-type ; 500 trop lourd ici)
THRESHOLD  = 0.90
MAX_STEPS  = 1600         # borne le projection ; au-dela on considere que EoL n'est pas atteinte
# Positions de t_now centrees sur la plage PERTINENTE pour RB2(RUL) : EoL de proche
# (~2 mois) a ~3 ans, soit RUL < RUL_ref=1000 j ou le levier s'active.
FRACS      = [0.70, 0.78, 0.85, 0.91, 0.96]
OUT_CSV    = 'sigma_rul_vs_horizon.csv'


def mc_forward(model, seed_norm, scaler, M, threshold, max_steps):
    windows = np.tile(seed_norm.flatten(), (M, 1))
    samples = []
    for _ in range(max_steps):
        pred = model(windows[:, :, np.newaxis], training=True).numpy().flatten()
        samples.append(pred)
        windows = np.roll(windows, -1, axis=1)
        windows[:, -1] = pred
        real = scaler.inverse_transform(pred.reshape(-1, 1)).flatten()
        if np.quantile(real, 0.975) <= threshold:
            break
    S = np.array(samples).T
    return np.array([scaler.inverse_transform(s.reshape(-1, 1)).flatten() for s in S])


def main():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(MODEL_PATH)
    if not os.path.exists(FILENAME):
        raise FileNotFoundError(FILENAME)
    model = tf.keras.models.load_model(MODEL_PATH)

    df = pd.read_csv(FILENAME, sep=';', header=None)
    datapro = df.iloc[:, 0].values.reshape(-1, 1)[::24]      # 1 pt / jour
    n_hist = len(datapro)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(datapro[:n_hist])

    rows = []
    print("=" * 78)
    print("INCERTITUDE D'ESTIMATION DE LA RUL EN FONCTION DE L'HORIZON (PEMWE, M=%d)" % M)
    print("-" * 78)
    print("  t_now[%%hist] ; RUL_moy[j] ; sigma_RUL[j] ; sigma_RUL_rel[%] ; sigma_SoH ; n_croisent", flush=True)
    for frac in FRACS:
        val_end = int(n_hist * frac)
        if val_end - WS < 0:
            continue
        seed_norm = scaler.transform(datapro[val_end - WS:val_end])
        t0 = time.time()
        S_real = mc_forward(model, seed_norm, scaler, M, THRESHOLD, MAX_STEPS)
        sigma_soh = float(np.std(S_real[:, 0]))
        t_eol = []
        for m in range(M):
            idx = np.where(S_real[m] <= THRESHOLD)[0]
            if len(idx):
                t_eol.append(int(idx[0]))
        t_eol = np.array(t_eol, dtype=float)
        n_cross = t_eol.size
        # PLATEAU : si peu/pas de trajectoires croisent le seuil dans MAX_STEPS, la
        # RUL n'est pas estimable a cet horizon -> on le signale au lieu de l'ignorer.
        if n_cross < 2:
            print("  %.2f : EoL NON atteinte (%d/%d trajectoires croisent en %d pas) "
                  "-> plateau, RUL non estimable a cet horizon  [%.0fs]"
                  % (frac, n_cross, M, MAX_STEPS, time.time() - t0), flush=True)
            continue
        rul_mean = float(t_eol.mean())
        sig_days = float(t_eol.std())
        sig_rel  = sig_days / rul_mean if rul_mean else float('nan')
        rows.append((frac, rul_mean, sig_days, sig_rel, sigma_soh, n_cross / M))
        print("  %9.2f ; %9.1f ; %11.1f ; %14.1f ; %.6f ; %4d/%d croisent  [%.0fs]"
              % (frac, rul_mean, sig_days, sig_rel * 100, sigma_soh, n_cross, M,
                 time.time() - t0), flush=True)
    print("=" * 78)

    with open(OUT_CSV, "w") as f:
        f.write("frac_tnow,RUL_mean_days,sigma_RUL_days,sigma_RUL_rel,sigma_SoH,frac_crossing\n")
        for r in rows:
            f.write("%.2f,%.2f,%.2f,%.5f,%.6f,%.3f\n" % r)
    print("-> %s" % OUT_CSV)
    print("Lecture : plus t_now est tot (fraction basse) -> EoL lointaine -> RUL_moy"
          " grande et sigma_RUL_rel plus eleve. C'est la dependance a l'horizon.")


if __name__ == "__main__":
    main()
