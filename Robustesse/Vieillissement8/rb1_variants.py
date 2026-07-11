"""Noyau unique et parametre des variantes publiees de RB1.

Ce module contient uniquement la logique de repartition RB1. Les dossiers de
strategie nommes fixent chacun un couple ``(SOC_LOW, SOC_HIGH)`` et appellent
``run_rb1``. Cette separation evite qu'un nouveau recalage silencieux du
dossier historique ``RB1/`` change retroactivement le sens des resultats.

La fonction est volontairement sans dependance NumPy/SciPy : les tests de la
regle peuvent ainsi etre executes sans charger le moteur de simulation.
"""

VARIANTS = {
    "rb1_hist_020_060": {
        "soc_low": 0.20,
        "soc_high": 0.60,
        "role": "RB1 historique du chapitre 2 et reference du sweep defaillances",
    },
    "rb1_failopt_040_075": {
        "soc_low": 0.40,
        "soc_high": 0.75,
        "role": "optimum robustesse sous defaillance, socle de RB1(Pred)",
    },
    "rb1_costopt_v8_020_035": {
        "soc_low": 0.20,
        "soc_high": 0.35,
        "role": "optimum nominal en cout unifie obtenu avec le socle V8",
    },
}


def validate_thresholds(soc_low, soc_high):
    """Valide les deux genoux de la bande de melange RB1."""
    if not (0.0 < soc_low < soc_high < 1.0):
        raise ValueError(
            "RB1 exige 0 < SOC_LOW < SOC_HIGH < 1, recu %.6g/%.6g"
            % (soc_low, soc_high)
        )


def raw_rb1_action(soc_low, soc_high, soc, p_tot_ref, failures=()):
    """Calcule l'action RB1 avant passage dans le referee ``get_lol``.

    Le comportement reproduit strictement la regle historique, y compris le
    reroutage vers la batterie en cas de defaillance. Le referee applique
    ensuite les saturations physiques et annule la puissance du composant en
    panne.
    """
    validate_thresholds(soc_low, soc_high)

    if soc_low < soc < soc_high:
        if p_tot_ref <= 0:
            p_bat, p_fc, p_ely = p_tot_ref, 0.0, 0.0
        else:
            p_bat = p_tot_ref * (soc - soc_low) / (soc_high - soc_low)
            p_fc = p_tot_ref - p_bat
            p_ely = 0.0
    elif soc <= soc_low:
        if p_tot_ref > 0:
            p_bat, p_fc, p_ely = 0.0, p_tot_ref, 0.0
        else:
            p_bat, p_fc, p_ely = p_tot_ref, 0.0, 0.0
    else:
        if p_tot_ref >= 0:
            p_bat, p_fc, p_ely = p_tot_ref, 0.0, 0.0
        else:
            p_bat = p_tot_ref * (1.0 - soc) / (1.0 - soc_high)
            p_fc = 0.0
            p_ely = p_tot_ref - p_bat

    if "FC" in failures and p_tot_ref > 0:
        p_bat = p_tot_ref
    if "ELY" in failures and p_tot_ref < 0:
        p_bat = p_tot_ref

    return p_bat, p_fc, p_ely


def run_rb1(
    soc_low,
    soc_high,
    SoC_t,
    P_tot_ref_t,
    defaillances,
    lol_tab,
    alpha_fc_t,
    alpha_ely_t,
    SoH_bat_t,
    E_h2_t,
    E_h2_init,
    P_fc_max_t,
    P_ely_max_t,
    RUL_fc_t,
    RUL_ely_t,
    SoH_fc_t,
    SoH_ely_t,
    referee=None,
):
    """Execute une variante RB1 puis le referee physique commun.

    Les arguments de vieillissement inutilises restent dans la signature pour
    conserver l'interface commune des strategies du depot. ``referee`` sert
    aux tests unitaires ; en simulation, ``Common.get_lol.get_lol`` est charge
    paresseusement dans l'environnement du moteur selectionne.
    """
    del alpha_fc_t, alpha_ely_t, RUL_fc_t, RUL_ely_t, SoH_fc_t, SoH_ely_t
    action = raw_rb1_action(
        soc_low, soc_high, SoC_t, P_tot_ref_t, tuple(defaillances)
    )
    if referee is None:
        from Common.get_lol import get_lol as referee

    return referee(
        SoC_t,
        action,
        P_tot_ref_t,
        defaillances,
        E_h2_t,
        E_h2_init,
        P_fc_max_t,
        P_ely_max_t,
        SoH_bat_t,
    )
