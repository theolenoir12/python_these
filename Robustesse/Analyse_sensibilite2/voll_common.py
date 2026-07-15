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
#     LPSP% = 100 * EENS / energie_totale_demandee.
# Le denominateur est la charge DC totale avant soustraction de la production
# PV. Cette convention unique est celle du manuscrit et de Vieillissement10.
#
#     n_steps = 218999  (= 25.000 ans)   E_REF = 523661.947 kWh = 523.662 MWh
#
# energie_non_fournie [kWh] = (LPSP%/100) * E_REF_KWH est alors EXACTEMENT
# l'energie de charge non servie reconstituee a partir de la LPSP.
E_REF_KWH = 523661.946666          # kWh, energie totale demandee sur 25 ans
HORIZON_Y = 25.0

# =========================================================================
# 2) VOLL (Value Of Lost Load)
# -------------------------------------------------------------------------
# VOLL CONSTANT (approche standard en fiabilite reseau / microgrid : cout de
# l'energie non servie = VOLL [EUR/kWh] * EENS [kWh], cf. ACER/London Economics,
# fourchette residentielle/mixte ~1-5 EUR/kWh). On retient 3 EUR/kWh (milieu de
# fourchette). Format conserve (seuil_fraction_haut, VOLL) avec un SEUL palier
# (seuil None) -> valeur constante quelle que soit la LPSP, pas d'effet de palier.
VOLL_TIERS = [
    (None, 3)
]

# Valorisation de l'energie non servie :
#   - False (defaut) : AGREGEE  -> cout_LPSP = VOLL(LPSP) * EENS (depuis la LPSP).
#     Avec un VOLL constant, c'est exactement VOLL * EENS. Mode retenu pour le
#     classement (coherent avec un VOLL constant ; la colonne 'clps' des .txt a
#     ete calculee PAS A PAS avec l'ANCIEN tiering et n'est donc plus utilisee).
#   - True : PAS A PAS -> on reutilise la colonne 'clps' stockee (sum_t VoLL(LPS_t)
#     * E_unserved_t). A ne reactiver que si 'clps' est recalcule avec ce VOLL.
USE_STEPWISE_LPS = False


def voll_eur_per_kwh(lpsp_pct):
    """VOLL [EUR/kWh] pour une LPSP (ou LPS instantanee) donnee en POURCENT."""
    x = lpsp_pct / 100.0
    for thr, val in VOLL_TIERS:
        if thr is None or x < thr:
            return val
    return VOLL_TIERS[-1][1]


def voll_eur_per_kwh_array(lpsp_pct):
    """Version VECTORISEE de voll_eur_per_kwh : lpsp_pct est un array [%].
    Sert a la valorisation PAS A PAS du LPS (cf. sens_common.lps_cost_keur) :
    le palier VoLL est reevalue a CHAQUE pas de temps a partir de la fraction non
    servie de ce pas, et non a partir de la LPSP agregee sur 25 ans."""
    import numpy as np
    x = np.asarray(lpsp_pct, dtype=float) / 100.0
    out = np.full(x.shape, float(VOLL_TIERS[-1][1]))
    # du palier le plus haut au plus bas : chaque condition x<thr ecrase la valeur,
    # ce qui reproduit exactement la logique scalaire ci-dessus.
    for thr, val in reversed(VOLL_TIERS):
        if thr is None:
            continue
        out[x < thr] = float(val)
    return out


def lost_energy_kwh(lpsp_pct):
    """Energie de charge non fournie [kWh] reconstituee depuis la LPSP [%]."""
    return (lpsp_pct / 100.0) * E_REF_KWH


def cost_lpsp_keur(lpsp_pct):
    """Cout financier du LPSP [kEUR] = VOLL(LPSP) * energie_non_fournie."""
    return voll_eur_per_kwh(lpsp_pct) * lost_energy_kwh(lpsp_pct) / 1000.0


def total_cost_keur(lpsp_pct, deg_keur, clps_keur=None):
    """Indicateur unifie [kEUR] = degradation + cout financier de l'energie non
    servie. Deux modes :
      - clps_keur fourni : on utilise le cout LPS evalue PAS A PAS pendant la
        simulation, sum_t VoLL(LPS(t))*E_unserved(t) (cf. sens_common.lps_cost_keur).
        C'est le mode courant (colonne 'clps' des .txt).
      - clps_keur None : repli sur l'ancienne valorisation AGREGEE
        VoLL(LPSP)*E_unserved (cost_lpsp_keur), pour rester compatible avec
        d'anciens .txt depourvus de la colonne 'clps'."""
    # Mode agrege par defaut (USE_STEPWISE_LPS=False) ou si 'clps' absent.
    if clps_keur is None or not USE_STEPWISE_LPS:
        return deg_keur + cost_lpsp_keur(lpsp_pct)
    return deg_keur + clps_keur


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
                             cb=float(p[3]), cf=float(p[4]), ce=float(p[5]),
                             mean=float(p[6]),
                             cout_lo95=float(p[8]), cout_hi95=float(p[9]),
                             clps=(float(p[10]) if len(p) > 10
                                   and p[10] not in ('', '-') else None))
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
                         lpsp_mean=float(p[3]), deg_mean=float(p[5]),
                         clps_nom=(float(p[10]) if len(p) > 10
                                   and p[10] not in ('', '-') else None),
                         clps_mean=(float(p[11]) if len(p) > 11
                                    and p[11] not in ('', '-') else None))
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
    """results_meso/sens_sizing.txt (format Monte-Carlo, comme sens_eol) ->
    {ems: dict(lpsp_nom, deg_nom, lpsp_mean, deg_mean)} (section '## Front ...').
    cols: strat;LPSP_nom;deg_nom;LPSP_mean;LPSP_std;deg_mean;deg_std;..."""
    return _parse_front_meanstd("sens_sizing.txt")


def parse_soh():
    """results_meso/sens_soh.txt -> dict(baseline=(lpsp,deg,clps),
    bias=[(bias,lpsp,deg,clps)...], sigma=[(sigma,lpsp_mean,deg_mean,clps_mean)...]).
    Mono-strategie (RB2(SoH)) : sert d'annexe. La 4e composante (clps = cout LPS
    pas-a-pas) vaut None si la colonne est absente (ancien .txt)."""
    def _opt(parts, idx):
        return (float(parts[idx]) if len(parts) > idx
                and parts[idx] not in ('', '-') else None)

    with open(_path("sens_soh.txt"), encoding="utf-8") as f:
        txt = f.read()
    base = None
    m = re.search(r"BASELINE;\s*LPSP=([\d.]+)%;\s*deg=([\d.]+)kEUR"
                  r"(?:;\s*clps=([\d.]+)kEUR)?", txt)
    if m:
        base = (float(m.group(1)), float(m.group(2)),
                float(m.group(3)) if m.group(3) else None)
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
                # bias;LPSP_%;deg_kEUR;dLPSP_pts;ddeg_%;clps_kEUR
                bias.append((float(p[0]), float(p[1]), float(p[2]), _opt(p, 5)))
            elif section == "sigma" and len(p) >= 7:
                # sigma;N;LPSP_mean;LPSP_std;LPSP_min;LPSP_max;deg_mean;...;clps_mean
                sigma.append((float(p[0]), float(p[2]), float(p[6]), _opt(p, 10)))
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
    siz  = parse_sizing()

    # Labels EN CLAIR (anglais, publication). Termes precis :
    #  - "EoL thresholds"            : seuils de fin de vie des composants.
    #  - "H2 degradation thresholds" : seuils des FONCTIONS DE DEGRADATION des
    #                                  composants H2 (PEMFC/PEMWE), pas un seuil
    #                                  "H2" generique.
    #  - "Replacement-cost weights"  : poids = couts de remplacement des composants.
    #  - "Component sizing"          : taille BAT/FC/ELY (+/-20%, Monte-Carlo).
    # TOUS les axes de stress sont desormais traites de facon UNIFIEE : moyenne du
    # Monte-Carlo (LPSP_mean, deg_mean), exactement comme l'EoL et l'H2.
    # Chaque strategie d'un cas est decrite par un TRIPLET (lpsp_%, deg_kEUR,
    # clps_kEUR) ou clps est le cout LPS evalue PAS A PAS (moyenne MC quand il y a
    # un Monte-Carlo). Si clps vaut None (ancien .txt), total_cost_keur retombe sur
    # la valorisation agregee a partir de la LPSP.
    cases = []
    # -- Nominal (reference commune ; on prend cweights : LPSP + cout_nominal) --
    cases.append(("Nominal", "base",
                  {e: (cw[e]["lpsp"], cw[e]["deg"], cw[e].get("clps")) for e in EMS_ORDER if e in cw}))
    # -- Stress seuils de fin de vie (moyenne MC) --
    cases.append(("EoL thresholds", "eol",
                  {e: (eol[e]["lpsp_mean"], eol[e]["deg_mean"], eol[e].get("clps_mean")) for e in EMS_ORDER if e in eol}))
    # -- Stress seuils des fonctions de degradation H2 (moyenne MC) --
    cases.append(("H2 degradation thresholds", "hthr",
                  {e: (hthr[e]["lpsp_mean"], hthr[e]["deg_mean"], hthr[e].get("clps_mean")) for e in EMS_ORDER if e in hthr}))
    # -- Poids = couts de remplacement (moyenne MC, +/-30%). La LPSP -- et donc le
    #    cout LPS pas-a-pas -- est invariante aux poids : seul le cout (deg) porte
    #    le MC, clps reste celui du run nominal. --
    cases.append(("Replacement-cost weights", "cw",
                  {e: (cw[e]["lpsp"], cw[e]["mean"], cw[e].get("clps")) for e in EMS_ORDER if e in cw}))
    # -- Dimensionnement (moyenne MC, +/-20%) -- ajoute seulement si les donnees MC
    #    sont presentes (sinon ancien fichier/format -> on saute proprement). --
    if siz:
        cases.append(("Component sizing", "sizing",
                      {e: (siz[e]["lpsp_mean"], siz[e]["deg_mean"], siz[e].get("clps_mean")) for e in EMS_ORDER if e in siz}))
    # -- Vieillissement calendaire batterie : VOLONTAIREMENT EXCLU pour l'instant
    #    (modele pas encore assez realiste). parse_calendar() reste disponible ;
    #    pour le reintegrer, decommenter :
    # cal = parse_calendar()
    # cases.append(("Calendar aging", "cal",
    #               {e: (cal[e]["lpsp_cal"], cal[e]["deg_cal"]) for e in EMS_ORDER if e in cal}))
    return cases
