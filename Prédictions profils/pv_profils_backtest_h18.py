# -*- coding: utf-8 -*-
"""
Backtest multi-origines de l'energie NETTE au bus (conso - PV), aux horizons
d'INTEGRATION 18 h et 48 h.

Identique a pv_profils_backtest_net.py, mais on evalue l'energie cumulee a
plusieurs horizons. L'objectif est d'obtenir le bruit d'estimation (biais, sigma)
a l'horizon de DECISION reellement utilise par RB2(Pred) : H_PRE = 18 h
(la pre-charge n'integre P_tot_ref_future que sur les 18 premieres heures).

La prevision recursive va toujours a 48 h ; on integre simplement les h
premieres heures.
"""

import os, sys
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH  = os.path.join(SCRIPT_DIR, '..', 'sidelec_roche_plate_csv2.csv')
MODEL_DIR  = os.path.join(SCRIPT_DIR, 'resultats_mc_dropout')
OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'resultats_backtest')
os.makedirs(OUTPUT_DIR, exist_ok=True)

WS         = 96
HORIZON    = 48              # prevision recursive (h)
HORIZONS_E = [18, 48]        # horizons d'integration de l'energie cumulee (h)
STEP       = 12              # pas entre deux origines (h)

PROFILS = {'production': {'col': 1}, 'consommation': {'col': 2}}


def forecast_batch(model, seeds_norm, n_steps):
    windows = seeds_norm.copy()
    preds = np.zeros((windows.shape[0], n_steps), dtype=np.float32)
    for t in range(n_steps):
        x = windows[:, :, np.newaxis]
        p = model(x, training=False).numpy().flatten()
        preds[:, t] = p
        windows = np.roll(windows, -1, axis=1)
        windows[:, -1] = p
    return preds


df = pd.read_csv(DATA_PATH, sep=';', header=None)
n = len(df)
train_end, val_end = int(n * 0.70), int(n * 0.85)
origins = np.arange(val_end, n - HORIZON, STEP)
print(f"Origines : {len(origins)} (pas {STEP} h, test {val_end}..{n})")

series_real, series_pred = {}, {}
for profil, cfg in PROFILS.items():
    data = df.iloc[:, cfg['col']].values.reshape(-1, 1).astype(float)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(data[:train_end])
    data_norm = scaler.transform(data).flatten()
    model = tf.keras.models.load_model(os.path.join(MODEL_DIR, f'modele_mc_dropout_{profil}.keras'))
    seeds = np.stack([data_norm[t0 - WS:t0] for t0 in origins])
    preds_norm = forecast_batch(model, seeds, HORIZON)
    preds = scaler.inverse_transform(preds_norm.reshape(-1, 1)).reshape(preds_norm.shape)
    reals = np.stack([data[t0:t0 + HORIZON, 0] for t0 in origins])
    series_pred[profil] = preds
    series_real[profil] = reals
    print(f"{profil}: {preds.shape} pretes")

net_pred = series_pred['consommation'] - series_pred['production']
net_real = series_real['consommation'] - series_real['production']

print("\n" + "=" * 78)
print("BRUIT D'ESTIMATION SUR L'ENERGIE NETTE (conso - PV) PAR HORIZON")
print("Ecart = energie predite - energie reelle [kWh]")
print("=" * 78)

rows = []
for h in HORIZONS_E:
    e_pred = net_pred[:, :h].sum(axis=1) / 1000.0
    e_real = net_real[:, :h].sum(axis=1) / 1000.0
    err = e_pred - e_real
    rows.append(dict(
        horizon_h=h, n=len(err),
        e_real_moy=float(e_real.mean()),
        biais=float(err.mean()),
        std=float(err.std(ddof=1)),
        rmse=float(np.sqrt((err ** 2).mean())),
        mae=float(np.abs(err).mean()),
        p5=float(np.quantile(err, 0.05)),
        p95=float(np.quantile(err, 0.95)),
        err_rel_pct=float(100 * np.abs(err).mean() / np.abs(e_real).mean()),
    ))
    print(f"\nHorizon {h} h :")
    print(f"  E nette reelle moyenne : {e_real.mean():8.1f} kWh")
    print(f"  Biais                  : {rows[-1]['biais']:+8.2f} kWh")
    print(f"  Sigma (ecart-type)     : {rows[-1]['std']:8.2f} kWh")
    print(f"  RMSE / MAE             : {rows[-1]['rmse']:8.2f} / {rows[-1]['mae']:8.2f} kWh")
    print(f"  P5 / P95               : {rows[-1]['p5']:+.2f} / {rows[-1]['p95']:+.2f} kWh")

csv_path = os.path.join(OUTPUT_DIR, 'bruit_estimation_energie_net_h18_h48.csv')
pd.DataFrame(rows).to_csv(csv_path, index=False)
print(f"\nResume sauvegarde : {csv_path}")
print("Termine.")
