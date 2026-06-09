import numpy as np
import csv
import matplotlib.pyplot as plt
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
csv_path    = os.path.abspath(os.path.join(current_dir, '..', 'Data', 'sidelec_roche_plate_csv.csv'))

Ts = 60 * 10
_repeat = int(3600 // Ts)          # 6 si Ts=600

# --- Lecture (inchangée) ---
sidelec_PV    = []
sidelec_conso = []
with open(csv_path, 'r') as csvfile:
    reader = csv.reader(csvfile, delimiter=';')
    for row in reader:
        sidelec_PV.append(float(row[1]))
        sidelec_conso.append(float(row[2]))

# --- Bruitage intra-horaire ---
rng = np.random.default_rng(42)    # graine pour reproductibilité

def add_intrahour_noise(hourly_values, repeat, rel_sigma=0.05,
                        clip_min=0.0, preserve_mean=True, rng=rng):
    """
    Étend un profil horaire en sous-pas en ajoutant un bruit gaussien
    proportionnel autour de chaque moyenne horaire.

    hourly_values : profil échantillonné à l'heure (1D)
    repeat        : nb de sous-pas par heure (6 pour 10min)
    rel_sigma     : écart-type relatif du bruit (0.05 = 5% de la valeur horaire)
    clip_min      : plancher (0 pour interdire le négatif)
    preserve_mean : si True, recentre chaque bloc pour conserver la moyenne horaire
    """
    base = np.repeat(np.asarray(hourly_values, dtype=float), repeat)

    # Bruit proportionnel : sigma nul quand la valeur est nulle (ex. PV la nuit)
    sigma = rel_sigma * base
    noisy = base + rng.normal(0.0, 1.0, size=base.shape) * sigma

    if clip_min is not None:
        noisy = np.clip(noisy, clip_min, None)

    if preserve_mean:
        # Recentre chaque bloc de `repeat` points sur la moyenne d'origine
        blocks = noisy.reshape(-1, repeat)
        block_means = blocks.mean(axis=1, keepdims=True)
        target = base.reshape(-1, repeat)[:, :1]      # la valeur horaire (constante par bloc)
        # correction additive : on décale, puis on re-clippe
        blocks = blocks - block_means + target
        noisy = blocks.reshape(-1)
        if clip_min is not None:
            noisy = np.clip(noisy, clip_min, None)

    return noisy

# Profils répétés sur l'année (* 31) puis bruités
conso_year = np.asarray(sidelec_conso) * 31   # attention : voir note ci-dessous
pv_year    = np.asarray(sidelec_PV) * 31

LOAD = {
    'P_ref': add_intrahour_noise(conso_year, _repeat, rel_sigma=0.05),
    'Ts': Ts
}
PV = {
    'P': add_intrahour_noise(pv_year, _repeat, rel_sigma=0.05)
}

# --- Profils bruités (récupérés de l'étape précédente) ---
conso_noisy = LOAD['P_ref']
pv_noisy    = PV['P']

# Axe temporel en heures (Ts en secondes -> heures)
t = np.arange(len(conso_noisy)) * Ts / 3600

# --- PLOT ---
fig, ax = plt.subplots(2, 1, figsize=(14, 7), sharex=True)

ax[0].plot(t, pv_noisy, color='tab:orange', lw=0.7)
ax[0].set_ylabel('PV [W]')
ax[0].set_title(f'Profils échantillonnés à Ts = {Ts}s ({Ts/60:.0f} min)')
ax[0].grid(alpha=0.3)

ax[1].plot(t, conso_noisy, color='tab:blue', lw=0.7)
ax[1].set_ylabel('Conso [W]')
ax[1].set_xlabel('Temps [h]')
ax[1].grid(alpha=0.3)

plt.tight_layout()
plt.show()

# --- Zoom sur les premiers jours pour bien voir le bruit intra-horaire ---
n_zoom = int(3 * 24 * 3600 / Ts)   # 3 jours
fig2, ax2 = plt.subplots(figsize=(14, 4))
ax2.plot(t[:n_zoom], pv_noisy[:n_zoom],    color='tab:orange', lw=0.9, label='PV')
ax2.plot(t[:n_zoom], conso_noisy[:n_zoom], color='tab:blue',   lw=0.9, label='Conso')
ax2.set_xlabel('Temps [h]')
ax2.set_ylabel('Puissance [W]')
ax2.set_title('Zoom 3 jours — bruit intra-horaire')
ax2.legend()
ax2.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# --- ENREGISTREMENT CSV (même format que sidelec_roche_plate) ---
out_path = 'sidelec_roche_plate_10min.csv'
with open(out_path, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=';')
    for i in range(len(conso_noisy)):
        writer.writerow([i, pv_noisy[i], conso_noisy[i]])

print(f'Fichier écrit : {out_path}  ({len(conso_noisy)} lignes)')