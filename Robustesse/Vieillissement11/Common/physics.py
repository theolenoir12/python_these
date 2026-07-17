"""Compatibilite avec l'ancienne API, adossee au modele unifie."""

from .electrochemistry import fc_pmax, ely_pmax


def get_Pmax_fc(alpha_fc_t):
    return float(fc_pmax(alpha_fc_t))


def get_Pmax_ely(alpha_ely_t):
    return float(ely_pmax(alpha_ely_t))
