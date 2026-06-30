# -*- coding: utf-8 -*-
"""
compute_prono_sigma.py -- ECARTS-TYPES REELS d'estimation SoH / RUL (MC-dropout).
=================================================================================
A LANCER dans le dossier  Python/Pronostic SoH/  (la ou sont le modele .keras et
le CSV de SoH). Leger (~secondes), pas besoin du mesocentre.

Reprend la meme inference MC-dropout que plot_rul_soh.py (M passes stochastiques,
auto-alimentation de la fenetre), puis calcule :
  - SIGMA_SOH      : ecart-type des M predictions au PREMIER pas de prediction
                     (horizon ~0) = incertitude d'estimation du SoH "instantane"
                     telle que la verrait RB2(SoH). [unites de SoH]
  - SIGMA_RUL_DAYS : ecart-type, sur les M trajectoires, du temps de franchissement
                     du seuil EoL (t_EoL par trajectoire). [jours]
  - RUL_MEAN       : RUL moyenne (sur les trajectoires atteignant le seuil). [jours]
  - SIGMA_RUL_REL  : SIGMA_RUL_DAYS / RUL_MEAN = ecart-type RELATIF de la RUL.

Reporter SIGMA_RUL_REL et SIGMA_SOH dans mc_rul_uncertainty.py.

Usage :  python compute_prono_sigma.py
"""
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

FILENAME   = 'SoH_ely_2.csv'
MODEL_PATH = 'resultats_mc_dropout/modele_mc_dropout_scaler2.keras'
WS         = 100      # window size (cf plot_rul_soh.py)
M          = 500      # passes Monte-Carlo dropout
THRESHOLD  = 0.90     # SoH_EoL
MAX_STEPS  = 1500


def main():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(MODEL_PATH)
    if not os.path.exists(FILENAME):
        raise FileNotFoundError(FILENAME + " (CSV de SoH ely d'entree)")
    model = tf.keras.models.load_model(MODEL_PATH)

    df = pd.read_csv(FILENAME, sep=';', header=None)
    datapro = df.iloc[:, 0].values.reshape(-1, 1)[::24]      # 1 pt / jour (cf plot_rul_soh.py)
    n_hist = len(datapro)
    val_end = int(n_hist * 0.85)                            # t_now
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(datapro[:n_hist])
    seed_norm = scaler.transform(datapro[val_end - WS:val_end])

    # Inference MC-dropout (M trajectoires auto-alimentees jusqu'a MAX_STEPS)
    windows = np.tile(seed_norm.flatten(), (M, 1))
    samples_norm = []
    for _ in range(MAX_STEPS):
        x = windows[:, :, np.newaxis]
        pred = model(x, training=True).numpy().flatten()    # dropout actif
        samples_norm.append(pred)
        windows = np.roll(windows, -1, axis=1)
        windows[:, -1] = pred
        real = scaler.inverse_transform(pred.reshape(-1, 1)).flatten()
        if np.quantile(real, 0.975) <= THRESHOLD:           # toutes bornes ~ sous seuil
            break
    S = np.array(samples_norm).T                            # (M, n_pred) normalise
    S_real = np.array([scaler.inverse_transform(s.reshape(-1, 1)).flatten() for s in S])

    # SIGMA_SOH : dispersion des M predictions au 1er pas de prediction (horizon ~0)
    sigma_soh = float(np.std(S_real[:, 0]))

    # t_EoL par trajectoire = 1er franchissement du seuil ; RUL = t_EoL (en jours,
    # 1 pt = 1 jour ici car sous-echantillonnage [::24] sur des donnees horaires)
    t_eol = []
    for m in range(M):
        idx = np.where(S_real[m] <= THRESHOLD)[0]
        if len(idx):
            t_eol.append(int(idx[0]))
    t_eol = np.array(t_eol, dtype=float)
    rul_mean = float(t_eol.mean()) if t_eol.size else float("nan")
    sigma_rul_days = float(t_eol.std()) if t_eol.size else float("nan")
    sigma_rul_rel = sigma_rul_days / rul_mean if rul_mean and not np.isnan(rul_mean) else float("nan")

    print("=" * 64)
    print("ECARTS-TYPES D'ESTIMATION (MC-dropout, M=%d passes, PEMWE)" % M)
    print("-" * 64)
    print("  SIGMA_SOH        = %.5f   (unites de SoH, horizon ~0)" % sigma_soh)
    print("  RUL_MEAN         = %.1f   jours" % rul_mean)
    print("  SIGMA_RUL_DAYS   = %.1f   jours" % sigma_rul_days)
    print("  SIGMA_RUL_REL    = %.4f   (= %.1f %% de la RUL)" % (sigma_rul_rel, sigma_rul_rel * 100))
    print("=" * 64)
    print("-> reporter SIGMA_RUL_REL et SIGMA_SOH dans mc_rul_uncertainty.py")


if __name__ == "__main__":
    main()
