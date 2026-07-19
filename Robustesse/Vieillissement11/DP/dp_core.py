"""Noyau de programmation dynamique de Vieillissement11 (nominal ``p=2``).

L'etat discret est ``(SoC, E_h2, fc_on, ely_on)`` et le controle unique est la
puissance H2 au bus DC (positive pour la PEMFC, negative pour la PEMWE). Le
backward price uniquement le dommage *permanent* V11. Pour la PEMFC, l'etat de
stabilite n'est pas ajoute a la grille : le backward emploie une approximation
locale explicitement documentee, puis le rollout et la boucle physique utilisent
l'etat de stabilite courant. Les couts realises restent ceux du ledger V11.
"""
import os
import sys
import time
import numpy as np

_THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_THIS, '..')))   # -> Vieillissement8/
os.environ.setdefault('GENIAL_DATA_DIR',
                       '/home/theo/Documents/Doctorat/Data')

from Common.Init_EMR_MG_v16_python import LOAD, PV, FC, ELY, BAT, CONV, S
from Common.cost_fcn_total2 import (
    deg_cumul1, deg_cumul2,
    get_cost_bat,
)
from Common.degradation_v11 import (
    MODEL_ID, ELY_V11, FC_V11,
    advance_ely_power, advance_fc_power,
    new_ely_state, new_fc_state, state_cost_eur, voltage_reference,
)
from Common.electrochemistry import (
    ely_current_density, ely_power, fc_current_density,
)
from Common.get_lol import get_lol
from Common.reliability_metrics import compute_reliability_metrics
from Common.rb1_policy_v11 import make_rb1_policy_v11
from Common.rb2_policy import make_rb2_policy
from Common.simulate_transition import simulate_transition

# ---------------------------------------------------------------------------
# Constantes derivees
# ---------------------------------------------------------------------------
ETA      = CONV['eta']
EFF_BAT  = BAT['eff']
CAP_WH   = BAT['parallel_num'] * BAT['series_num'] * BAT['Q_bat'] * BAT['v_cell_nom']  # 51840 Wh (SoC 0->1)
V_SER    = BAT['v_cell_nom'] * BAT['series_num']   # 720 V (pour i_bat)
P_FC_MAX = FC['P_fc_max']      # ~1753 W
P_ELY_MAX = ELY['P_ely_max']   # ~18430 W
TS_H     = LOAD['Ts'] / 3600.0 # 1 h
VOLL     = 3.0                 # EUR/kWh (VOLL_TIERS = [(None,3)])

SOC_LO, SOC_HI = 0.2, 0.995
E_H2_INIT = 200.0              # kWh (capacite reservoir, = init de la boucle)

FC_LUT  = FC['lut']
ELY_LUT = ELY['lut']

N_YEAR = 8760                  # pas horaires sur 1 an

NOMINAL_ELY_STRESS_EXPONENT = 2.0
if not np.isclose(ELY_V11["stress_exponent"], NOMINAL_ELY_STRESS_EXPONENT):
    raise RuntimeError(
        "La PD V11 est attribuee au nominal p=2, mais degradation_v11 expose p=%r"
        % ELY_V11["stress_exponent"]
    )

RB1_REFERENCE_PARAMS = (0.20, 0.40)
RB2_REFERENCE_PARAMS = (0.574, 0.465)


# ---------------------------------------------------------------------------
# Profil net de reference [W] (cote DC) : P_tot_ref = P_dc_load - P_dc_pv
# ---------------------------------------------------------------------------
def net_reference(n=N_YEAR):
    P_dc_load = LOAD['P_ref'][:n] / ETA
    P_dc_pv   = PV['P'][:n]
    return P_dc_load - P_dc_pv, P_dc_load, P_dc_pv


def net_reference_window(start, n=N_YEAR):
    """Profil net [W] sur la FENETRE REELLE [start:start+n] des tableaux LOAD/PV.

    Indispensable pour la PD avec vieillissement : le profil 25 ans n'est PAS
    forcement 8760-periodique cale au pas 0 (la donnee mesocentre derive d'~1 h/an
    -> period. effective 8759). On cale donc le template PD sur la fenetre que la
    politique va effectivement gouverner (wrap au bout pour la derniere annee
    partielle)."""
    load = np.asarray(LOAD['P_ref']); pv = np.asarray(PV['P'])
    idx = (np.arange(start, start + n)) % len(load)
    return load[idx] / ETA - pv[idx]


# ---------------------------------------------------------------------------
# Grille de controle P_dc_h2  (>0 FC ; <0 ELY)
#   contrainte FC  : P_dc_fc / eta  <= P_fc_max   -> u <= P_fc_max*eta
#   contrainte ELY : |P_dc_ely|*eta <= P_ely_max  -> |u| <= P_ely_max/eta
#   extra_u : niveaux additionnels (memes conventions, W cote DC), notamment
#   les ancres PEMWE j=1 et j=2 du modele V11.
# ---------------------------------------------------------------------------
def control_grid(n_fc=10, n_ely=50, p_fc_max=P_FC_MAX, p_ely_max=P_ELY_MAX,
                 extra_u=()):
    u_fc  = np.linspace(0.0, p_fc_max * ETA, n_fc + 1)[1:]
    u_ely = -np.linspace(0.0, p_ely_max / ETA, n_ely + 1)[1:]
    u = np.concatenate(([0.0], u_fc, u_ely, np.asarray(extra_u, dtype=float)))
    u = u[(u <= p_fc_max * ETA + 1e-9) & (u >= -p_ely_max / ETA - 1e-9)]
    return np.unique(u)


def v11_control_anchors(alpha_ely=0.0, p_ely_max=P_ELY_MAX):
    """Puissances DC correspondant aux ancres PEMWE ``j=1`` et ``j=2``.

    Le coude de dommage est en densite de courant, pas en fraction de Pmax. Ces
    niveaux doivent donc etre recalcules lorsque ``alpha_ely`` evolue.
    """
    densities = np.array([1.0, 2.0])
    currents = densities * S * ELY['n_parallel']
    stack_powers = np.asarray(ely_power(currents, alpha_ely), dtype=float)
    dc_controls = -stack_powers / ETA
    return tuple(float(value) for value in dc_controls
                 if value >= -p_ely_max / ETA - 1e-9)


def permanent_uv_to_eur(component, permanent_uv):
    """Convertit un increment de perte permanente V11 [uV] en euros."""
    config = FC if component == "fc" else ELY
    return (np.asarray(permanent_uv, dtype=float) * 1e-6
            / voltage_reference(component)
            / (1.0 - config['SoH_EoL']) * config['cost'])


# ---------------------------------------------------------------------------
# Pre-calculs dependant du controle u et de l'etat electrochimique au rebuild
#   - P_h2[u]          : kW vers le reservoir (>0 stockage, <0 destockage)
#   - cur_fc_on[u]     : la FC est-elle active ?
#   - cur_ely_on[u]    : l'ELY est-il actif ?
#   - cost_fc[u, prev] : cout financier FC du pas (EUR), prev = etat on precedent
#   - cost_ely[u, prev]: cout financier ELY du pas (EUR)  (V_irr+idle+startstop)
# ---------------------------------------------------------------------------
def precompute_controls(u, p_fc_max=P_FC_MAX, p_ely_max=P_ELY_MAX,
                        alpha_fc=0.0, alpha_ely=0.0):
    """Precalcule dynamique H2 et cout permanent V11 de chaque controle.

    L'etat PD ne contient que les indicateurs marche/arret. Pour la PEMFC, le
    backward suppose donc ``steadiness=1`` et, si elle etait deja allumee,
    ``previous_j=current_j``. Un demarrage depuis l'arret emploie au contraire
    ``previous_j=0`` et la mise a jour V11 exacte. Le rollout corrige ensuite
    cette approximation avec la stabilite et le courant precedents realises.
    """
    u = np.asarray(u, dtype=float)
    P_dc_fc  = np.maximum(u, 0.0)
    P_dc_ely = np.minimum(u, 0.0)

    eff_fc  = np.interp(P_dc_fc / ETA / p_fc_max * 100.0, FC_LUT[0], FC_LUT[1]) / 100.0
    eff_ely = np.interp(np.abs(P_dc_ely * ETA / p_ely_max) * 100.0, ELY_LUT[0], ELY_LUT[1]) / 100.0

    with np.errstate(divide='ignore', invalid='ignore'):
        term_fc = np.where(P_dc_fc > 0, (P_dc_fc / ETA) / eff_fc, 0.0)
    P_h2 = (np.abs(P_dc_ely * ETA) * eff_ely - term_fc) / 1000.0   # kW

    # ----- PEMFC V11 : dommage permanent, stabilite locale approximee -----
    P_fc = P_dc_fc / ETA                       # cote stack
    j_fc = np.asarray(fc_current_density(P_fc, alpha_fc), dtype=float)
    cur_fc_on = j_fc > 1e-9
    cost_fc = np.zeros((len(u), 2))
    for prev in (0, 1):
        previous_j = j_fc if prev else np.zeros_like(j_fc)
        after_change = np.exp(
            -np.abs(j_fc - previous_j) / FC_V11['change_scale_a_cm2'])
        steadiness = 1.0 - (1.0 - after_change) * np.exp(
            -TS_H / FC_V11['steadiness_tau_h'])
        rate = (FC_V11['irr_dynamic_uvph']
                + steadiness * (FC_V11['irr_steady_uvph']
                                - FC_V11['irr_dynamic_uvph']))
        if FC_V11['current_exponent'] != 0.0:
            rate *= np.where(
                cur_fc_on,
                (j_fc / FC_V11['j_ref']) ** FC_V11['current_exponent'], 0.0)
        uv = np.where(cur_fc_on, rate * TS_H, 0.0)
        uv += ((prev == 0) & cur_fc_on) * FC_V11['start_uv']
        uv += ((j_fc > 0.0) & (j_fc <= 0.05)) * FC_V11['idle_uvph'] * TS_H
        cost_fc[:, prev] = permanent_uv_to_eur('fc', uv)

    # ----- PEMWE V11 p=2 : irreversible + quasi-idle + demarrage -----
    P_ely = np.abs(P_dc_ely * ETA)
    j_ely = np.asarray(ely_current_density(P_ely, alpha_ely), dtype=float)
    cur_ely_on = j_ely > 1e-9
    irr_rate = (ELY_V11['steady_2_uvph']
                * np.maximum(j_ely - 1.0, 0.0) ** NOMINAL_ELY_STRESS_EXPONENT)
    cost_ely = np.zeros((len(u), 2))
    for prev in (0, 1):
        uv = irr_rate * TS_H
        uv += ((j_ely > 0.0) & (j_ely <= 0.01)) * ELY_V11['idle_uvph'] * TS_H
        uv += ((prev == 0) & cur_ely_on) * ELY_V11['start_uv']
        cost_ely[:, prev] = permanent_uv_to_eur('ely', uv)

    return P_h2, cur_fc_on.astype(np.int64), cur_ely_on.astype(np.int64), cost_fc, cost_ely


# ---------------------------------------------------------------------------
# Cout batterie d'un pas, VECTORISE  (== get_cost_bat sur un pas, SoH=1)
#   entrees : soc (Ns,), P_bat (Nu,), soc_tp1 (Ns,Nu)   -> cout EUR (Ns,Nu)
# ---------------------------------------------------------------------------
def battery_cost_step(soc, P_bat, soc_tp1, soh_bat=1.0):
    cu_t   = np.interp(soc,     deg_cumul1, deg_cumul2)[:, None]   # (Ns,1)
    cu_tp1 = np.interp(soc_tp1, deg_cumul1, deg_cumul2)            # (Ns,Nu)
    deg_SoC = np.abs(cu_t - cu_tp1) * (BAT['Q_bat'] * soh_bat) * BAT['parallel_num'] / 2.15
    i_bat  = P_bat / V_SER                                         # (Nu,)
    C_rate = np.abs(i_bat) / (BAT['Q_bat'] * soh_bat * BAT['parallel_num'])
    scaling = np.where(C_rate > 1, 0.2956 * C_rate + (1 - 0.2956),
                       np.where(C_rate >= 0, 1.0, 0.0))            # (Nu,)
    cost_tot = deg_SoC * scaling[None, :]                          # micro-Ah-ish
    cost_bat = cost_tot * 1e-6 / ((1 - BAT['SoH_EoL']) * BAT['Q_bat'] * BAT['parallel_num'])
    return cost_bat * BAT['cost']                                  # EUR (Ns,Nu)


# ---------------------------------------------------------------------------
# Dynamique SoC + LPSP d'un pas, VECTORISE
#   Pour un P_tot_ref scalaire (l'heure t) et tous (soc, u) :
#     P_dc_bat = P_tot_ref - u   (force) ; P_bat = P_dc_bat / eta^sign
#     soc_tp1 brut ; clamp [0.2,0.995]
#       - plancher atteint EN DEFICIT  -> energie non servie -> cout VoLL
#       - plafond atteint EN EXCEDENT  -> curtailment (gratuit)
#   retourne soc_tp1_clamp (Ns,Nu), P_bat (Nu,), lpsp_eur (Ns,Nu)
# ---------------------------------------------------------------------------
def soc_step_and_lpsp(soc, u, P_tot_ref, cap_wh=CAP_WH):
    P_dc_bat = P_tot_ref - u                                       # (Nu,)
    P_bat = P_dc_bat / ETA ** np.sign(P_dc_bat)                    # (Nu,)
    fac = EFF_BAT ** np.sign(-P_bat)                               # (Nu,)
    soc_tp1 = soc[:, None] - (P_bat * fac / cap_wh)[None, :]       # (Ns,Nu)

    soc_clamp = np.clip(soc_tp1, SOC_LO, SOC_HI)

    lpsp_eur = np.zeros_like(soc_tp1)
    if P_tot_ref > 0:
        # plancher : batterie limitee a SOC_LO -> P_dc_bat realisable
        hit = soc_tp1 < SOC_LO
        # P_bat max de decharge pour atteindre SOC_LO (discharge: fac=1/eff_bat)
        P_bat_max = (soc[:, None] - SOC_LO) * cap_wh / EFF_BAT     # (Ns,1) broadcast
        P_dc_bat_real = P_bat_max * ETA                           # decharge -> *eta
        delivered = P_dc_bat_real + u[None, :]                    # + P_dc_h2
        unmet = np.clip(P_tot_ref - delivered, 0.0, None)         # W
        lpsp_eur = np.where(hit, unmet / 1000.0 * TS_H * VOLL, 0.0)
    return soc_clamp, P_bat, lpsp_eur


# ---------------------------------------------------------------------------
# Backward induction sur 1 an (SoH=1).  Periodicite : VT passe en argument.
#   retourne V0 (Ns,Nh,2,2) et, si store, la politique (T, Ns,Nh,2,2) int8
# ---------------------------------------------------------------------------
def backward(soc_grid, h2_grid, u, pre, P_ref, VT, store_policy=False,
             cap_wh=CAP_WH, soh_bat=1.0, ckpt_every=None, ckpt_out=None):
    """ckpt_every/ckpt_out : si fournis, ckpt_out (dict) recoit une COPIE de V
    aux pas t multiples de ckpt_every (+ le terminal T=len(P_ref)). Sert au
    rollout de dp_aging (recalcul des tranches V d'une journee a la volee,
    sans stocker les 8760 tranches)."""
    Ns, Nh, Nu = len(soc_grid), len(h2_grid), len(u)
    P_h2, cur_fc, cur_ely, cost_fc, cost_ely = pre
    INF = 1e18

    # E_h2 transitions (indep. du temps) : idx + poids bilineaires
    E_tp1 = h2_grid[:, None] + P_h2[None, :]                       # (Nh,Nu)
    feas_h2 = (E_tp1 >= -1e-9) & (E_tp1 <= E_H2_INIT + 1e-9)
    E_clip = np.clip(E_tp1, h2_grid[0], h2_grid[-1])
    jl = np.clip(np.searchsorted(h2_grid, E_clip) - 1, 0, Nh - 2)
    h2_lo = h2_grid[jl]; h2_hi = h2_grid[jl + 1]
    wj = (h2_hi - E_clip) / (h2_hi - h2_lo)                        # poids sur jl (Nh,Nu)

    nf = cur_fc                                                    # next fc_on (Nu,)
    ne = cur_ely                                                   # next ely_on(Nu,)

    V = VT
    policy = np.empty((len(P_ref), Ns, Nh, 2, 2), dtype=np.int8) if store_policy else None
    if ckpt_every and ckpt_out is not None:
        ckpt_out[len(P_ref)] = VT.copy()

    for t in range(len(P_ref) - 1, -1, -1):
        Ptot = P_ref[t]
        soc_clamp, P_bat, lpsp = soc_step_and_lpsp(soc_grid, u, Ptot, cap_wh)  # (Ns,Nu),(Nu,),(Ns,Nu)
        cbat = battery_cost_step(soc_grid, P_bat, soc_clamp, soh_bat)          # (Ns,Nu)
        cost_bl = cbat + lpsp                                          # (Ns,Nu)

        # idx/poids SoC
        il = np.clip(np.searchsorted(soc_grid, soc_clamp) - 1, 0, Ns - 2)  # (Ns,Nu)
        s_lo = soc_grid[il]; s_hi = soc_grid[il + 1]
        wl = (s_hi - soc_clamp) / (s_hi - s_lo)                        # (Ns,Nu)

        # future[s,h,u] = bilineaire de V[:,:,nf[u],ne[u]] en (soc_tp1, E_tp1)
        future = np.zeros((Ns, Nh, Nu))
        il_b = il[:, None, :]                                          # (Ns,1,Nu)
        jl_b = jl[None, :, :]                                          # (1,Nh,Nu)
        nf_b = nf[None, None, :]; ne_b = ne[None, None, :]
        for a in (0, 1):
            wa = (wl if a == 0 else 1 - wl)[:, None, :]                # (Ns,1,Nu)
            ia = il_b + a
            for b in (0, 1):
                wb = (wj if b == 0 else 1 - wj)[None, :, :]            # (1,Nh,Nu)
                jb = jl_b + b
                future += wa * wb * V[ia, jb, nf_b, ne_b]
        # H2 infaisable -> +inf
        future = future + np.where(feas_h2[None, :, :], 0.0, INF)

        base = future + cost_bl[:, None, :]                           # (Ns,Nh,Nu)
        Vnew = np.empty((Ns, Nh, 2, 2))
        for fc_on in (0, 1):
            for ely_on in (0, 1):
                cand = base + cost_fc[:, fc_on][None, None, :] \
                            + cost_ely[:, ely_on][None, None, :]
                Vnew[:, :, fc_on, ely_on] = cand.min(axis=2)
                if store_policy:
                    policy[t, :, :, fc_on, ely_on] = cand.argmin(axis=2).astype(np.int8)
        V = Vnew
        if ckpt_every and ckpt_out is not None and t % ckpt_every == 0:
            ckpt_out[t] = V.copy()
    return V, policy


def solve_cyclic(soc_grid, h2_grid, u, pre, P_ref, n_iter=3, verbose=True,
                 cap_wh=CAP_WH, soh_bat=1.0, ckpt_every=None):
    """Retourne (V0, policy) ; si ckpt_every est fourni, retourne
    (V0, policy, ckpt) avec ckpt = {t: V_t} captures a la DERNIERE iteration."""
    Ns, Nh = len(soc_grid), len(h2_grid)
    VT = np.zeros((Ns, Nh, 2, 2))
    ckpt = {} if ckpt_every else None
    for it in range(n_iter):
        store = (it == n_iter - 1)
        t0 = time.time()
        V0, policy = backward(soc_grid, h2_grid, u, pre, P_ref, VT, store_policy=store,
                              cap_wh=cap_wh, soh_bat=soh_bat,
                              ckpt_every=(ckpt_every if store else None),
                              ckpt_out=(ckpt if store else None))
        delta = np.max(np.abs(V0 - VT))
        if verbose:
            print(f"  [iter {it+1}/{n_iter}] max|V0-VT| = {delta:10.4f} EUR   "
                  f"({time.time()-t0:5.1f}s)")
        VT = V0
    if ckpt_every:
        return V0, policy, ckpt
    return V0, policy


# ---------------------------------------------------------------------------
# Forward sim EXACT (reutilise get_lol + simulate_transition)
#   policy_fn(SoC, E_h2, fc_on, ely_on, t) -> action (P_dc_bat, P_dc_fc, P_dc_ely)
#   retourne un dict minimal pour le reporting V11 du prototype
# ---------------------------------------------------------------------------
def forward_sim(policy_fn, n=N_YEAR, soc0=0.5, e_h2_0=E_H2_INIT):
    P_ref_net, P_dc_load, P_dc_pv = net_reference(n)
    SoC = np.zeros(n + 1); SoC[0] = soc0
    E_h2 = np.zeros(n + 1); E_h2[0] = e_h2_0
    P_bat = np.zeros(n); P_fc = np.zeros(n); P_ely = np.zeros(n)
    lol_tab = np.zeros(n)
    fc_on, ely_on = 0, 0
    for t in range(n):
        Ptot = P_ref_net[t]
        proposed = policy_fn(SoC[t], E_h2[t], fc_on, ely_on, t, Ptot)
        # Les references RB1/RB2 renvoient deja (action, lol). Recalculer lol a
        # partir de leur action deja clampee effacerait artificiellement l'EENS.
        if (isinstance(proposed, tuple) and len(proposed) == 2
                and hasattr(proposed[0], '__len__') and len(proposed[0]) == 3):
            action, lol = proposed
        else:
            action, lol = get_lol(
                SoC[t], proposed, Ptot, [], E_h2[t], E_H2_INIT,
                P_FC_MAX, P_ELY_MAX, 1.0)
        SoC_tp1, simOut = simulate_transition(SoC[t], action, Ptot, 0, lol,
                                              0.0, 0.0, 1.0, E_h2[t], E_H2_INIT,
                                              P_FC_MAX, P_ELY_MAX)
        if SoC_tp1 < 0:  # securite : action infaisable -> repli batterie pure
            action = (Ptot, 0.0, 0.0)
            action, lol = get_lol(SoC[t], action, Ptot, [], E_h2[t], E_H2_INIT,
                                  P_FC_MAX, P_ELY_MAX, 1.0)
            SoC_tp1, simOut = simulate_transition(SoC[t], action, Ptot, 0, lol,
                                                  0.0, 0.0, 1.0, E_h2[t], E_H2_INIT,
                                                  P_FC_MAX, P_ELY_MAX)
        P_bat[t] = simOut['P_bat']; P_fc[t] = simOut['P_fc']; P_ely[t] = simOut['P_ely']
        lol_tab[t] = lol
        E_h2[t + 1] = simOut['E_h2_tp1']; SoC[t + 1] = SoC_tp1
        fc_on  = int(P_fc[t]  >= 1.0)
        ely_on = int(abs(P_ely[t]) >= 0.0005 * P_ELY_MAX)
    return {
        "SoC": SoC, "E_h2": E_h2, "P_bat": P_bat, "P_fc": P_fc, "P_ely": P_ely,
        "P_dc_load": P_dc_load, "P_dc_pv": P_dc_pv, "lol_tab": lol_tab,
        "alpha_fc": np.zeros(n + 1), "alpha_ely": np.zeros(n + 1),
        "SoH_bat": np.ones(n + 1),
    }


# ---------------------------------------------------------------------------
# Metrique exacte V11 du prototype 1 an (sans remplacement)
# ---------------------------------------------------------------------------
def metrics(data):
    P_bat = np.asarray(data["P_bat"])
    P_fc = np.asarray(data["P_fc"])
    P_ely = np.asarray(data["P_ely"])
    SoC = np.asarray(data["SoC"])
    alpha_fc = np.asarray(data["alpha_fc"][:-1])
    alpha_ely = np.asarray(data["alpha_ely"][:-1])

    fc_state, ely_state = new_fc_state(), new_ely_state()
    previous_fc = previous_ely = 0.0
    for k in range(len(P_fc)):
        fc_state = advance_fc_power(
            fc_state, P_fc[k], previous_fc, alpha_fc[k], TS_H)
        ely_state = advance_ely_power(
            ely_state, P_ely[k], previous_ely, alpha_ely[k], TS_H)
        previous_fc, previous_ely = P_fc[k], P_ely[k]
    bat_eur = get_cost_bat(P_bat, SoC, np.ones(len(P_bat)))
    deg_keur = (
        bat_eur + state_cost_eur('fc', fc_state) + state_cost_eur('ely', ely_state)
    ) / 1000.0
    rel = compute_reliability_metrics(data)
    lps_keur = VOLL * rel['eens_kwh'] / 1000.0
    return dict(lpsp=rel['lpsp_pct'], eens_kwh=rel['eens_kwh'],
                deg_keur=deg_keur, lps_keur=lps_keur,
                unified_keur=deg_keur + lps_keur)


# ---------------------------------------------------------------------------
# Politiques
# ---------------------------------------------------------------------------
def make_dp_policy(soc_grid, h2_grid, u, policy):
    def pol(SoC, E_h2, fc_on, ely_on, t, Ptot):
        i = int(np.clip(np.searchsorted(soc_grid, SoC), 0, len(soc_grid) - 1))
        if i > 0 and (SoC - soc_grid[i - 1]) < (soc_grid[i] - SoC):
            i -= 1
        j = int(np.clip(np.searchsorted(h2_grid, E_h2), 0, len(h2_grid) - 1))
        if j > 0 and (E_h2 - h2_grid[j - 1]) < (h2_grid[j] - E_h2):
            j -= 1
        ui = policy[t, i, j, fc_on, ely_on]
        u_val = u[ui]
        P_dc_fc = max(u_val, 0.0); P_dc_ely = min(u_val, 0.0)
        P_dc_bat = Ptot - u_val
        return (P_dc_bat, P_dc_fc, P_dc_ely)
    return pol


def _reference_adapter(reference_policy):
    def policy(SoC, E_h2, fc_on, ely_on, t, Ptot):
        del fc_on, ely_on, t
        return reference_policy(
            SoC, Ptot, [], None, 0.0, 0.0, 1.0,
            E_h2, E_H2_INIT, P_FC_MAX, P_ELY_MAX,
            8000, 3000, 1.0, 1.0,
        )
    return policy


rb1_policy = _reference_adapter(make_rb1_policy_v11(*RB1_REFERENCE_PARAMS))
rb2_policy = _reference_adapter(make_rb2_policy(*RB2_REFERENCE_PARAMS))


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    print("=" * 70)
    print(" PROTOTYPE PD V11 p=2 -- 1 an, etat neuf (SoH=1)")
    print("=" * 70)
    Ns = Nh = 51
    soc_grid = np.linspace(SOC_LO, SOC_HI, Ns)
    h2_grid  = np.linspace(0.0, E_H2_INIT, Nh)
    u = control_grid(n_fc=10, n_ely=50, extra_u=v11_control_anchors())
    pre = precompute_controls(u)
    P_ref_net, _, _ = net_reference(N_YEAR)
    print(f" grille SoC={Ns}  E_h2={Nh}  controles={len(u)}  pas={N_YEAR}")

    # --- sanity check cout batterie vectorise vs get_cost_bat ---
    _check_battery_cost(soc_grid, u, P_ref_net)

    # --- references best-vs-best V11 p=2 ---
    references = {}
    for label, reference in (
        ("RB1(0.20,0.40)", rb1_policy),
        ("RB2(0.574,0.465)", rb2_policy),
    ):
        t0 = time.time()
        result = metrics(forward_sim(reference))
        references[label] = result
        print(f"\n {label:<18}: LPSP {result['lpsp']:.4f}%   "
              f"deg {result['deg_keur']:.3f} kEUR   LPS {result['lps_keur']:.3f} kEUR   "
              f"UNIFIE {result['unified_keur']:.3f} kEUR ({time.time()-t0:.1f}s)")

    # --- PD ---
    print("\n Resolution PD (periodicite annuelle) :")
    t0 = time.time()
    V0, policy = solve_cyclic(soc_grid, h2_grid, u, pre, P_ref_net, n_iter=3)
    print(f"  backward total : {time.time()-t0:.1f}s")
    data_dp = forward_sim(make_dp_policy(soc_grid, h2_grid, u, policy))
    m_dp = metrics(data_dp)
    print(f"\n PD    : LPSP {m_dp['lpsp']:.4f}%   deg {m_dp['deg_keur']:.3f} kEUR   "
          f"LPS {m_dp['lps_keur']:.3f} kEUR   UNIFIE {m_dp['unified_keur']:.3f} kEUR")

    # --- verdict ---
    print("\n" + "-" * 70)
    best_label, best_ref = min(
        references.items(), key=lambda item: item[1]['unified_keur'])
    gain = best_ref['unified_keur'] - m_dp['unified_keur']
    rel = gain / best_ref['unified_keur'] * 100 if best_ref['unified_keur'] else 0
    print(f" Cout unifie : {best_label} {best_ref['unified_keur']:.3f}  "
          f"->  PD {m_dp['unified_keur']:.3f} kEUR")
    print(f" Gain PD     : {gain:.3f} kEUR  ({rel:+.1f}%)   "
          f"{'OK' if m_dp['unified_keur'] <= best_ref['unified_keur'] + 1e-6 else 'A INVESTIGUER'}")
    print("-" * 70)


def _check_battery_cost(soc_grid, u, P_ref_net):
    """Verifie battery_cost_step contre get_cost_bat sur quelques pas."""
    Ptot = P_ref_net[12]
    soc_clamp, P_bat, _ = soc_step_and_lpsp(soc_grid, u, Ptot)
    cbat_vec = battery_cost_step(soc_grid, P_bat, soc_clamp)
    err = 0.0
    for si in (5, 20, 40):
        for ui in (0, 3, 30):
            ref = get_cost_bat(np.array([P_bat[ui]]),
                               np.array([soc_grid[si], soc_clamp[si, ui]]),
                               np.array([1.0]))
            err = max(err, abs(ref - cbat_vec[si, ui]))
    print(f" [check] max ecart cout_bat vectorise vs get_cost_bat : {err:.2e} EUR")


if __name__ == "__main__":
    main()
