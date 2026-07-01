# -*- coding: utf-8 -*-
"""
diag_flattening.py -- DIAGNOSTIC de l'aplatissement du RUL aux horizons longs.
Pour chaque t_now (fraction de l'historique), on deroule M passes MC-dropout sur
MAX_STEPS et on mesure :
  - le PLANCHER de plateau : SoH final median et SoH final le plus bas (2.5%),
  - si/quand le quantile 97.5% croise EoL=0.90 (=> RUL estimable),
  - n_cross/M : combien de passes franchissent reellement le seuil.
But : montrer OU l'extrapolation echoue (plateau > 0.90 => RUL non defini).
Borne dure par MAX_STEPS (pas de boucle infinie). Leger : M reduit.
"""
import os, time
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

FILENAME   = 'SoH_ely_2.csv'
MODEL_PATH = 'resultats_mc_dropout/modele_mc_dropout_scaler2.keras'
WS         = 100
M          = 30
THRESHOLD  = 0.90
MAX_STEPS  = 900
FRACS      = [0.65, 0.55, 0.45, 0.35, 0.25]


def mc_rollout(model, seed_norm, scaler, M, max_steps):
    windows = np.tile(seed_norm.flatten(), (M, 1))
    samples = []
    for _ in range(max_steps):
        pred = model(windows[:, :, np.newaxis], training=True).numpy().flatten()
        samples.append(pred)
        windows = np.roll(windows, -1, axis=1)
        windows[:, -1] = pred
    S = np.array(samples).T                       # (M, steps) en normalise
    return np.array([scaler.inverse_transform(s.reshape(-1, 1)).flatten() for s in S])


def main():
    model = tf.keras.models.load_model(MODEL_PATH)
    df = pd.read_csv(FILENAME, sep=';', header=None)
    datapro = df.iloc[:, 0].values.reshape(-1, 1)[::24]
    n_hist = len(datapro)
    scaler = MinMaxScaler(feature_range=(0, 1)); scaler.fit(datapro[:n_hist])

    print("=" * 88)
    print("DIAGNOSTIC APLATISSEMENT (M=%d, MAX_STEPS=%d, EoL=%.2f)  n_daily=%d"
          % (M, MAX_STEPS, THRESHOLD, n_hist))
    print("-" * 88)
    print(" frac | t_now(j) SoH0   | plateau_med plateau_bas | n_cross | RUL_med(j) sig_rel | [s]", flush=True)
    for frac in FRACS:
        t = int(n_hist * frac)
        if t - WS < 0:
            continue
        seed = scaler.transform(datapro[t - WS:t])
        t0 = time.time()
        S = mc_rollout(model, seed, scaler, M, MAX_STEPS)     # (M, MAX_STEPS) reel
        soh0 = float(datapro[t, 0])
        plateau_med = float(np.median(S[:, -1]))              # SoH final median
        plateau_bas = float(np.quantile(S[:, -1], 0.025))     # SoH final le plus bas
        # croisement par passe
        rul = []
        for m in range(M):
            idx = np.where(S[m] <= THRESHOLD)[0]
            if len(idx):
                rul.append(int(idx[0]))
        n_cross = len(rul)
        if n_cross >= 2:
            rul = np.array(rul, float)
            rul_med = float(np.median(rul)); sig_rel = float(rul.std() / rul.mean())
            rr = "%9.0f  %5.1f%%" % (rul_med, sig_rel * 100)
        else:
            rr = "   --       --  "
        print("  %.2f | %6d  %.3f | %.4f      %.4f     | %3d/%d | %s | %4.0f"
              % (frac, t, soh0, plateau_med, plateau_bas, n_cross, M, rr, time.time() - t0),
              flush=True)
    print("=" * 88)
    print("Lecture : si plateau_med > 0.90 -> l'extrapolation MEDIANE n'atteint jamais EoL")
    print("          (aplatissement) -> RUL non estimable a cet horizon avec ce modele.")


if __name__ == "__main__":
    main()
