# Lancer les analyses de sensibilite sur le mesocentre (MesoFC / Helios)

Guide pas-a-pas pour executer les scripts `sens_*.py` sur le centre de calcul
de Franche-Comte avec un nombre de tirages Monte-Carlo eleve. On vise un
**noeud unique** (parallelisme par processus, deja en place dans le code) sur
la partition `smp` (**32 coeurs** -> ~x4-5 par rapport aux 7 coeurs locaux).

> Le code n'a PAS besoin d'ecran : tous les scripts utilisent
> `matplotlib.use("Agg")`. Les chemins sont rendus portables (variable
> `GENIAL_DATA_DIR`) et le nombre de workers est detecte via
> `SLURM_CPUS_PER_TASK`. Aucune autre adaptation n'est requise.

---

## 0. Prerequis (a faire une fois)

- **Compte Helios** : l'acces necessite un compte ET une demande explicite
  (cf. doc "Access/Connect"). Si tu n'as qu'un compte sur **Lumiere** (cluster
  CPU historique, ressources par defaut), la demarche est identique, seuls les
  noms de noeud/partitions peuvent differer : verifie avec `sinfo`.
- **VPN** si tu es hors du reseau de l'universite (Helios n'est accessible que
  depuis le reseau interne).
- Un client SSH (OpenSSH inclus dans Windows 10/11, ou PuTTY) et un client de
  transfert (scp en ligne de commande, ou FileZilla en SFTP port 22).

Frontal : `mesohelios1.univ-fcomte.fr` (port 22). Il **redemarre chaque jour a
04h00** : ne pas y laisser de session/calcul actif.

---

## 1. Ou poser les fichiers : `$WORK`, pas `$HOME`

| Dossier | Variable | Quota | Sauvegarde | Remarque |
|---|---|---|---|---|
| `/Home/Users/<login>` | `$HOME` | 10 Go | oui | **lecture seule sur les noeuds de calcul** |
| `/Work/Users/<login>` | `$WORK` | 1 To  | non | dossier de travail : TOUT ici |

=> On installe le depot et on lance les jobs depuis **`$WORK`**.

---

## 2. Transferer le code et les donnees

Depuis ta machine Windows (PowerShell), en remplacant `<login>` :

```powershell
# Le dossier de code (depuis .../GENIAL/python_these)
scp -r .\Robustesse <login>@mesohelios1.univ-fcomte.fr:Work/genial/

# Les donnees d'entree (CSV + LUT) -- gitignorees, donc absentes d'un clone :
#   - le gros profil PV/conso
scp "C:\Users\tlenoi01\Doctorat\Data\sidelec_roche_plate_csv.csv" `
    <login>@mesohelios1.univ-fcomte.fr:Work/genial_data/
```

Les deux tables de rendement (`FC_efficiency_LU_table_power.csv`,
`ELY_efficiency_LU_table_power.csv`) vivent dans
`Robustesse/Vieillissement8/Common/` et sont donc transferees avec le code
ci-dessus (verifie qu'elles sont bien presentes localement avant le `scp -r`).

> **`scp` cree les sous-dossiers `Work/...` relativement a `$HOME`.** Une fois
> connecte, deplace si besoin vers `$WORK` (qui peut etre un autre chemin que
> `~/Work`) ; le plus simple est de viser directement `$WORK` :
> `scp ... <login>@...:/Work/Users/<login>/genial/`.

**Important (fichiers Windows)** : applique `dos2unix` au script SLURM apres
copie, sinon les fins de ligne CRLF cassent l'execution bash :

```bash
dos2unix run_meso.slurm
```

Arborescence attendue sur le cluster :

```
$WORK/genial/Robustesse/Analyse_sensibilite/   <- on lance ici
$WORK/genial/Robustesse/Vieillissement8/...     <- code de base (importe)
$WORK/genial_data/sidelec_roche_plate_csv.csv   <- pointe par GENIAL_DATA_DIR
```

Le script `run_meso.slurm` definit `GENIAL_DATA_DIR="$WORK/genial_data"`. Si tu
ranges le CSV ailleurs, edite cette ligne.

---

## 3. Environnement Python

L'archive Anaconda fournie contient deja numpy / scipy / matplotlib / sympy :

```bash
module load anaconda3@2022.10/gcc-12.1.0
python -c "import numpy,scipy,matplotlib,sympy;print('OK')"
```

Si tu preferes un environnement dedie (optionnel) :

```bash
cd $WORK
module load anaconda3@2022.10/gcc-12.1.0
conda create -y -n genial python numpy scipy matplotlib sympy
# puis decommente la ligne "conda activate genial" dans run_meso.slurm
```

---

## 4. Soumettre un job (une analyse = une soumission)

Depuis `$WORK/genial/Robustesse/Analyse_sensibilite/` :

```bash
sbatch run_meso.slurm sens_calendar.py
sbatch run_meso.slurm sens_eol.py
sbatch run_meso.slurm sens_hthresholds.py
sbatch run_meso.slurm sens_sizing.py
sbatch run_meso.slurm sens_cweights.py
sbatch run_meso.slurm sens_soh_estimation.py
```

Avant le premier lancement, edite dans `run_meso.slurm` :
`--mail-user=PRENOM.NOM@univ-fcomte.fr` (et `--time` si besoin).

Les 6 jobs sont independants : tu peux tout soumettre d'un coup, l'ordonnanceur
les fera tourner en parallele (chacun sur son noeud, dans la limite QoS).

---

## 5. Suivre et recuperer

```bash
squeue --me                 # jobs en attente / en cours
seff <jobid>                # efficacite CPU/RAM apres coup
scancel <jobid>             # annuler
cat sens_genial.<jobid>.out # log (stdout+stderr), avec le 'time' final
```

Les figures et .txt sont ecrits dans `Analyse_sensibilite/results/`. Pour les
rapatrier sur Windows :

```powershell
scp -r <login>@mesohelios1.univ-fcomte.fr:/Work/Users/<login>/genial/Robustesse/Analyse_sensibilite/results .\results_meso
```

---

## 6. Augmenter le nombre de tirages Monte-Carlo

C'est l'interet du cluster. Dans chaque script, augmente `N_MC` (ex. 15 -> 200).
Ordre de grandeur : ~70 s par simulation de 25 ans ; sur 32 coeurs, ~2 s par
tirage effectif. Exemple `sens_calendar.py` avec `N_MC=200` (~2000 runs) :
~70 min de calcul, tres en-dessous du walltime de 24 h. Tu peux donc viser
plusieurs centaines de tirages sans souci.

## 7. Limites a connaitre

- Partition `smp` : QoS par defaut **32 CPU/utilisateur** -> `--cpus-per-task`
  ne peut pas depasser 32 sans QoS speciale. C'est deja le max utile pour un
  job mono-noeud (les processus ne franchissent pas la frontiere du noeud).
- Memoire : noeud smp = 96 Go ; le script demande `--mem=80G` (32 workers,
  largement suffisant). Ne pas mettre `--mem` > 96G sur smp.
- Le parallelisme du code est **mono-noeud** (ProcessPoolExecutor). Passer
  multi-noeuds (ex. job array `--array` sur les 10 EMS) est possible mais
  demanderait de modifier les scripts ; inutile vu les temps ci-dessus.
