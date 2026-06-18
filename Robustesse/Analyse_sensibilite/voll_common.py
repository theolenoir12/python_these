"""
voll_common.py -- indicateur UNIFIE (degradation + cout financier du LPSP).
============================================================================
Toutes les analyses de sensibilite (results_meso/*.txt) donnent leurs resultats
selon DEUX axes : la degradation [kEUR] et la LPSP [%]. Ce module ajoute une
3e lecture : on VALORISE la LPSP en euros via un *Value Of Lost Load* (VOLL,
EUR/kWh), puis on resume tout par UN SEUL indicateur :

    cout_total [kEUR] = deg [kEUR] + cout_LPSP [kEUR]
    cout_LPSP  [kEUR] = VOLL(LPSP) [EUR/kWh] * energie_non_fournie [kWh] / 1000
    energie_non_fournie [kWh] = (LPSP% / 100) * E_REF_KWH

Plus bas = meilleur. Ce module ne fait QUE du post-traitement : il (re)lit les
.txt produits au mesocentre et n'execute aucune simulation.

SOURCE 100% ASCII (volontaire ; cf. sens_common.py) : robustesse a l'encodage.
"""
import os
import re

HERE     = os.path.dirname(os.path.abspath(__file__))
MESO_DIR = os.path.join(HERE, "results_meso")     # entrees (.txt) ET sorties (.pdf)

# =========================================================================
# 1) ENERGIE DE REFERENCE (denominateur de la LPSP)
# -------------------------------------------------------------------------
# La LPSP definie dans sens_common.metrics() vaut :
#     LPSP% = 100 * sum(clip(P_planned - P_real, 0)) / sum(clip(P_planned, 0))
# ou P_planned = (P_dc_load - P_dc_pv)/1000 [kW] (charge NETTE apres PV, au bus
# DC). Le denominateur sum(clip(P_planned,0)) * Ts/3600 est donc l'ENERGIE NETTE
# PLANIFIEE que le systeme (batterie + H2) doit fournir sur tout l'horizon. Cette
# energie est IDENTIQUE pour toutes les strategies (c'est la demande, pas le
# pilotage) -> une constante. On l'a calculee UNE fois en relancant une simu de
# base (RB2(SoH), 25 ans, Ts=3600 s) ; cf. tmp/compute_eref.py :
#
#     n_steps = 218999  (= 25.000 ans)   E_REF = 273380.731 kWh = 273.381 MWh
#
# energie_non_fournie [kWh] = (LPSP%/100) * E_REF_KWH est alors EXACTEMENT
# l'energie de charge non servie reconstituee a partir de la LPSP.
E_REF_KWH = 273380.731444          # kWh, energie nette planifiee sur 25 ans
HORIZON_Y = 25.0

# =========================================================================
# 2) VOLL PAR PALIERS (Value Of Lost Load)
# -------------------------------------------------------------------------
# Approche par paliers : plus la LPSP est elevee, plus la defaillance est jugee
# couteuse (penalite croissante). Les seuils sont en FRACTION de LPSP (0.05 = 5%).
# Reglage courant : 2 EUR/kWh si LPSP<5% ; 5 si 5-10% ; 10 si >=10%.
# (seuil_fraction_haut, VOLL EUR/kWh) -- trie par seuil croissant ; le dernier
# palier (seuil None) s'applique au-dela du dernier seuil.
VOLL_TIERS = [
    (0.05, 2),
    (0.1, 5),
    (None, 10)
]


def voll_eur_per_kwh(lpsp_pct):
    """VOLL [EUR/kWh] pour une LPSP donnee en POURCENT (palier)."""
    x = lpsp_pct / 100.0
    for thr, val in VOLL_TIERS:
        if thr is None or x < thr:
            return val
    return VOLL_TIERS[-1][1]


def lost_energy_kwh(lpsp_pct):
    """Energie de charge non fournie [kWh] reconstituee depuis la LPSP [%]."""
    return (lpsp_pct / 100.0) * E_REF_KWH


def cost_lpsp_keur(lpsp_pct):
    """Cout financier du LPSP [kEUR] = VOLL(LPSP) * energie_non_fournie."""
    return voll_eur_per_kwh(lpsp_pct) * lost_energy_kwh(lpsp_pct) / 1000.0


def total_cost_keur(lpsp_pct, deg_keur):
    """Indicateur unifie [kEUR] = degradation + cout financier du LPSP."""
    return deg_keur + cost_lpsp_keur(lpsp_pct)


# =========================================================================
# 3) STRATEGIES (ordre + couleurs partages avec les scripts sens_*.py)
# -------------------------------------------------------------------------
EMS_ORDER = ["0-100", "25-75", "50-50", "75-25", "100-0",
             "RB2", "RB2(SoH)", "RB1", "SoC1", "SoC06"]


def ems_colors():
    """{ems: couleur} via tab10, MEME mapping que sens_cweights/sens_sizing."""
    import numpy as np
    import matplotlib.pyplot as plt
    cols = plt.cm.tab10(np.linspace(0, 1, 10))
    return {e: cols[i] for i, e in enumerate(EMS_ORDER)}


# =========================================================================
# 4) PARSERS DES .txt (formats heterogenes, colonnes stables)
# -------------------------------------------------------------------------
def _path(name):
    return os.path.join(MESO_DIR, name)


def _data_lines(text):
    """Lignes utiles d'un bloc : non vides, hors commentaires (#) et entetes."""
    out = []
    for ln in text.splitlines():
        s = ln.strip()
        if not s or s.startswith("#"):
            continue
        out.append(s)
    return out


def parse_cweights():
    """results_meso/sens_cweights.txt -> {ems: dict(lpsp, deg, cout_lo95, cout_hi95)}.
    deg = cout_nominal ; lpsp INVARIANT aux poids. cols:
    strat;LPSP_%;cout_nominal;cost_bat;cost_fc;cost_ely;cout_mean;cout_std;cout_lo95;cout_hi95"""
    out = {}
    with open(_path("sens_cweights.txt"), encoding="utf-8") as f:
        for s in _data_lines(f.read()):
            if s.startswith("strat;"):
                continue
            p = s.split(";")
            if len(p) < 10:
                continue
            out[p[0]] = dict(lpsp=float(p[1]), deg=float(p[2]),
                             cout_lo95=float(p[8]), cout_hi95=float(p[9]))
    return out


def _parse_front_meanstd(fname):
    """sens_eol.txt / sens_hthresholds.txt : section '## Front ...', cols
    strat;LPSP_nom;deg_nom;LPSP_mean;LPSP_std;deg_mean;deg_std;...
    -> {ems: dict(lpsp_nom, deg_nom, lpsp_mean, deg_mean)} (1re section seulement)."""
    out = {}
    with open(_path(fname), encoding="utf-8") as f:
        txt = f.read()
    # on ne garde que la 1re section de front (avant l'eventuel bloc OAT)
    for s in _data_lines(txt):
        if s.startswith("strat;"):
            if out:                       # 2e entete -> on a fini le front
                break
            continue
        p = s.split(";")
        if len(p) < 6 or p[0] not in EMS_ORDER:
            continue
        out[p[0]] = dict(lpsp_nom=float(p[1]), deg_nom=float(p[2]),
                         lpsp_mean=float(p[3]), deg_mean=float(p[5]))
    return out


def parse_eol():
    return _parse_front_meanstd("sens_eol.txt")


def parse_hthresholds():
    return _parse_front_meanstd("sens_hthresholds.txt")


def parse_calendar():
    """results_meso/sens_calendar.txt -> {ems: dict(lpsp_off, deg_off, lpsp_cal, deg_cal)}.
    cols: ems;SoC_moy;LPSP_off;deg_off;LPSP_cal_mean;deg_cal_mean;..."""
    out = {}
    with open(_path("sens_calendar.txt"), encoding="utf-8") as f:
        for s in _data_lines(f.read()):
            if s.startswith("ems;"):
                if out:
                    break                 # fin de la section front
                continue
            p = s.split(";")
            if len(p) < 6 or p[0] not in EMS_ORDER:
                continue
            out[p[0]] = dict(soc=float(p[1]), lpsp_off=float(p[2]), deg_off=float(p[3]),
                             lpsp_cal=float(p[4]), deg_cal=float(p[5]))
    return out


def parse_sizing():
    """results_meso/sens_sizing.txt -> liste [(scenario_label, {ems: (lpsp, deg)})]
    dans l'ordre du fichier. cols de chaque section :
    ems;LPSP_%;deg_kEUR;rank_cout;non_domine"""
    with open(_path("sens_sizing.txt"), encoding="utf-8") as f:
        txt = f.read()
    blocks = []
    cur_label, cur = None, None
    for ln in txt.splitlines():
        s = ln.strip()
        m = re.match(r"##\s*Scenario\s*'(.*?)'", s)
        if m:
            if cur_label is not None:
                blocks.append((cur_label, cur))
            cur_label, cur = m.group(1), {}
            continue
        if cur is None or not s or s.startswith("#") or s.startswith("ems;"):
            continue
        p = s.split(";")
        if len(p) < 3 or p[0] not in EMS_ORDER:
            continue
        try:
            cur[p[0]] = (float(p[1]), float(p[2]))
        except ValueError:
            continue                      # ligne 'FAIL'
    if cur_label is not None:
        blocks.append((cur_label, cur))
    return blocks


def parse_soh():
    """results_meso/sens_soh.txt -> dict(baseline=(lpsp,deg),
    bias=[(bias,lpsp,deg)...], sigma=[(sigma,lpsp_mean,deg_mean)...]).
    Mono-strategie (RB2(SoH)) : sert d'annexe."""
    with open(_path("sens_soh.txt"), encoding="utf-8") as f:
        txt = f.read()
    base = None
    m = re.search(r"BASELINE;\s*LPSP=([\d.]+)%;\s*deg=([\d.]+)kEUR", txt)
    if m:
        base = (float(m.group(1)), float(m.group(2)))
    bias, sigma = [], []
    section = None
    for ln in txt.splitlines():
        s = ln.strip()
        if s.startswith("## Regime 1"):
            section = "bias"; continue
        if s.startswith("## Regime 2"):
            section = "sigma"; continue
        if not s or s.startswith("#") or s.startswith("bias;") or s.startswith("sigma;"):
            continue
        p = s.split(";")
        try:
            if section == "bias" and len(p) >= 3:
                bias.append((float(p[0]), float(p[1]), float(p[2])))
            elif section == "sigma" and len(p) >= 7:
                # sigma;N;LPSP_mean;LPSP_std;LPSP_min;LPSP_max;deg_mean;...
                sigma.append((float(p[0]), float(p[2]), float(p[6])))
        except ValueError:
            continue
    return dict(baseline=base, bias=bias, sigma=sigma)


# =========================================================================
# 5) CONSTRUCTION DES "CAS" (colonnes du resume transversal)
# -------------------------------------------------------------------------
def build_cases():
    """Assemble tous les cas (scenario/analyse) en une liste ordonnee :
        [(case_label, groupe, {ems: (lpsp_%, deg_kEUR)}), ...]
    Chaque cas porte (LPSP, deg) par strategie ; le cout total se calcule via
    total_cost_keur(). Le point 'Nominal' (commun a toutes les analyses) n'est
    compte qu'UNE fois ; les perturbations de chaque analyse forment les autres
    colonnes."""
    cw   = parse_cweights()
    eol  = parse_eol()
    hthr = parse_hthresholds()

    # Labels EN CLAIR (anglais, publication). Termes precis :
    #  - "EoL thresholds"            : seuils de fin de vie des composants.
    #  - "H2 degradation thresholds" : seuils des FONCTIONS DE DEGRADATION des
    #                                  composants H2 (PEMFC/PEMWE), pas un seuil
    #                                  "H2" generique.
    #  - "Replacement-cost weights"  : poids = couts de remplacement des composants
    #                                  (haut = pessimiste, bas = optimiste, IC95).
    cases = []
    # -- Nominal (reference commune ; on prend cweights : LPSP + cout_nominal) --
    cases.append(("Nominal", "base",
                  {e: (cw[e]["lpsp"], cw[e]["deg"]) for e in EMS_ORDER if e in cw}))
    # -- Stress seuils de fin de vie (moyenne MC) --
    cases.append(("EoL thresholds", "eol",
                  {e: (eol[e]["lpsp_mean"], eol[e]["deg_mean"]) for e in EMS_ORDER if e in eol}))
    # -- Stress seuils des fonctions de degradation H2 (moyenne MC) --
    cases.append(("H2 degradation thresholds", "hthr",
                  {e: (hthr[e]["lpsp_mean"], hthr[e]["deg_mean"]) for e in EMS_ORDER if e in hthr}))
    # -- Vieillissement calendaire batterie : VOLONTAIREMENT EXCLU pour l'instant
    #    (modele de vieillissement calendaire pas encore assez realiste). Le parser
    #    parse_calendar() reste disponible ; pour le reintegrer, decommenter :
    # cal = parse_calendar()
    # cases.append(("Calendar aging", "cal",
    #               {e: (cal[e]["lpsp_cal"], cal[e]["deg_cal"]) for e in EMS_ORDER if e in cal}))
    # -- Poids = couts de remplacement : haut (pessimiste) / bas (optimiste), IC95.
    #    LPSP invariant -> on garde la LPSP nominale, on remplace deg par la borne.
    cases.append(("Replacement-cost weights (high)", "cw",
                  {e: (cw[e]["lpsp"], cw[e]["cout_hi95"]) for e in EMS_ORDER if e in cw}))
    cases.append(("Replacement-cost weights (low)", "cw",
                  {e: (cw[e]["lpsp"], cw[e]["cout_lo95"]) for e in EMS_ORDER if e in cw}))
    # -- Dimensionnement : VOLONTAIREMENT EXCLU pour l'instant (traite plus tard).
    #    Le parser parse_sizing() reste disponible ; pour reintegrer ces cas,
    #    decommenter la boucle ci-dessous (cf. sens_sizing.txt) :
    # for label, d in parse_sizing():
    #     if label.lower() == "nominal":
    #         continue                      # == cas 'Nominal' deja present
    #     cases.append(("Dim. " + label, "sizing", dict(d)))
    return cases
