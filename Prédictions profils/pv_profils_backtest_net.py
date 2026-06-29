# -*- coding: utf-8 -*-
"""
Backtest multi-origines de l'énergie NETTE au bus (consommation − production PV).

Objectif : caractériser le bruit d'estimation sur l'énergie à amener/tirer au
bus, sur des horizons de 2 jours (48 h) et 4 jours (96 h).

Méthode :
    - on rejoue N prévisions récursives (déterministes) à 96 h, à différents
      instants (origines) du jeu de test ;
    - à chaque origine, prévision conso et PV -> net = conso − PV ;
    - on intègre l'énergie nette cumulée à 48 h et 96 h (somme des Wh horaires) ;
    - l'écart énergie prédite − réelle, sur l'ensemble des origines, fournit la
      DISTRIBUTION EMPIRIQUE de l'erreur d'estimation -> valeurs de bruit cohérentes.

Les modèles sont ceux entraînés par pv_profils_mcdropout.py (dropout=0.05).
La prévision ponctuelle utilise le dropout désactivé (inférence standard).
"""

import os, sys
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
try:
    sys.stdout.reconfigure(encoding='utf-8')   # éviter UnicodeEncodeError (cp1252)
except Exception:
    pass

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# ── Configuration ────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH  = os.path.join(SCRIPT_DIR, '..', 'sidelec_roche_plate_csv2.csv')
MODEL_DIR  = os.path.join(SCRIPT_DIR, 'resultats_mc_dropout')
OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'resultats_backtest')
os.makedirs(OUTPUT_DIR, exist_ok=True)

WS        = 96
HORIZON   = 48            # prévision à 2 jours (horizon de travail unique)
HORIZONS_E = [48]         # énergie cumulée évaluée : 2 j
STEP      = 12            # pas entre deux origines de prévision (h)

PROFILS = {
    'production':   {'col': 1},
    'consommation': {'col': 2},
}

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Latin Modern Roman", "Computer Modern Roman", "DejaVu Serif"],
    "text.usetex": False,
    "font.size": 16, "axes.titlesize": 22, "axes.labelsize": 20,
    "xtick.labelsize": 17, "ytick.labelsize": 17, "legend.fontsize": 16,
    "figure.dpi": 300, "savefig.dpi": 600,
    "axes.grid": True, "grid.alpha": 0.4, "grid.linestyle": '--',
})

# ── Prévision récursive déterministe, vectorisée sur les origines ────────────
def forecast_batch(model, seeds_norm, n_steps):
    """seeds_norm : (N, ws) -> prévisions (N, n_steps), dropout désactivé."""
    windows = seeds_norm.copy()
    preds = np.zeros((windows.shape[0], n_steps), dtype=np.float32)
    for t in range(n_steps):
        x = windows[:, :, np.newaxis]
        p = model(x, training=False).numpy().flatten()
        preds[:, t] = p
        windows = np.roll(windows, -1, axis=1)
        windows[:, -1] = p
    return preds

# ── Chargement données + modèles, construction des origines ──────────────────
df = pd.read_csv(DATA_PATH, sep=';', header=None)
n = len(df)
train_end, val_end = int(n * 0.70), int(n * 0.85)

# Origines : dans le jeu de test, en gardant 96 h disponibles devant.
origins = np.arange(val_end, n - HORIZON, STEP)
print(f"Nombre d'origines de prévision : {len(origins)} (pas {STEP} h, test {val_end}..{n})")

series_real = {}   # valeurs réelles futures (N, HORIZON) par profil
series_pred = {}   # prévisions (N, HORIZON) par profil

for profil, cfg in PROFILS.items():
    data = df.iloc[:, cfg['col']].values.reshape(-1, 1).astype(float)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(data[:train_end])
    data_norm = scaler.transform(data).flatten()

    model = tf.keras.models.load_model(os.path.join(MODEL_DIR, f'modele_mc_dropout_{profil}.keras'))

    seeds = np.stack([data_norm[t0 - WS:t0] for t0 in origins])      # (N, ws)
    preds_norm = forecast_batch(model, seeds, HORIZON)
    preds = scaler.inverse_transform(preds_norm.reshape(-1, 1)).reshape(preds_norm.shape)

    reals = np.stack([data[t0:t0 + HORIZON, 0] for t0 in origins])   # (N, HORIZON)

    series_pred[profil] = preds
    series_real[profil] = reals
    print(f"{profil}: prévisions {preds.shape} prêtes")

# ── Énergie nette (conso − production) ───────────────────────────────────────
net_pred = series_pred['consommation'] - series_pred['production']
net_real = series_real['consommation'] - series_real['production']

print("\n" + "=" * 78)
print("BRUIT D'ESTIMATION SUR L'ÉNERGIE NETTE AU BUS (conso − PV)")
print("Écart = énergie prédite − énergie réelle, sur les origines du test")
print("=" * 78)

summary_rows = []
err_by_h = {}
for h in HORIZONS_E:
    e_pred = net_pred[:, :h].sum(axis=1) / 1000.0   # kWh
    e_real = net_real[:, :h].sum(axis=1) / 1000.0   # kWh
    err = e_pred - e_real                            # kWh
    err_by_h[h] = err
    stats = dict(
        horizon_h=h, jours=h // 24, n=len(err),
        e_real_moy=float(e_real.mean()),
        biais=float(err.mean()),
        std=float(err.std(ddof=1)),
        rmse=float(np.sqrt((err ** 2).mean())),
        mae=float(np.abs(err).mean()),
        p5=float(np.quantile(err, 0.05)),
        p50=float(np.quantile(err, 0.50)),
        p95=float(np.quantile(err, 0.95)),
        err_rel_pct=float(100 * np.abs(err).mean() / np.abs(e_real).mean()),
    )
    summary_rows.append(stats)
    print(f"\nHorizon {h} h ({h//24} j) — {len(err)} origines :")
    print(f"  Énergie nette réelle moyenne : {e_real.mean():8.1f} kWh")
    print(f"  Biais (moy. écart)           : {stats['biais']:+8.2f} kWh")
    print(f"  Écart-type du bruit          : {stats['std']:8.2f} kWh")
    print(f"  RMSE / MAE                   : {stats['rmse']:8.2f} / {stats['mae']:8.2f} kWh")
    print(f"  Quantiles 5%/50%/95%         : {stats['p5']:+.2f} / {stats['p50']:+.2f} / {stats['p95']:+.2f} kWh")
    print(f"  Erreur relative moyenne      : {stats['err_rel_pct']:.1f} %")

df_summary = pd.DataFrame(summary_rows)
csv_path = os.path.join(OUTPUT_DIR, 'bruit_estimation_energie_net.csv')
df_summary.to_csv(csv_path, index=False)
print(f"\nRésumé sauvegardé : {csv_path}")

# ── Figure d'incertitude propre (horizon 2 j) ────────────────────────────────
h = HORIZONS_E[0]
err = err_by_h[h]
mu, sd = err.mean(), err.std(ddof=1)
p5, p95 = np.quantile(err, 0.05), np.quantile(err, 0.95)

fig, ax = plt.subplots(figsize=(10, 6.5))
ax.hist(err, bins=22, color='#3498DB', alpha=0.65, edgecolor='white', density=True,
        label='Distribution empirique')
# Gaussienne de référence N(biais, sigma) -> indique si un bruit gaussien suffit
xs = np.linspace(err.min(), err.max(), 400)
gauss = np.exp(-0.5 * ((xs - mu) / sd) ** 2) / (sd * np.sqrt(2 * np.pi))
ax.plot(xs, gauss, color='#2C3E50', lw=2.5, label=f'$\\mathcal{{N}}$({mu:.0f}, {sd:.0f}$^2$)')
ax.axvline(mu, color='#E74C3C', lw=2.5, label=f'Biais = {mu:+.1f} kWh')
ax.axvspan(mu - sd, mu + sd, color='#E74C3C', alpha=0.10, label=f'$\\pm\\sigma$ = {sd:.0f} kWh')
ax.axvline(p5,  color='#7F8C8D', lw=1.6, ls='--')
ax.axvline(p95, color='#7F8C8D', lw=1.6, ls='--', label=f'P5 / P95 = {p5:.0f} / {p95:.0f} kWh')

ax.set_title(f"Énergie nette au bus — horizon {h//24} j ({len(err)} origines)")
ax.set_xlabel("Écart énergie prédite − réelle [kWh]")
ax.set_ylabel("Densité")
ax.legend(loc='upper left', fontsize=14)
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
fig.tight_layout()
fig_path = os.path.join(OUTPUT_DIR, 'incertitude_energie_net_2j.pdf')
fig.savefig(fig_path)
fig.savefig(fig_path.replace('.pdf', '.png'))
plt.close(fig)
print(f"Figure sauvegardée : {fig_path}")
print("\nTerminé.")
