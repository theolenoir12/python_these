"""
=============================================================================
dp_aging.py -- PD AVEC VIEILLISSEMENT (sequentielle, an par an)
=============================================================================

Etend le prototype PD (dp_core, 1 an BoL) a l'horizon 25 ans AVEC vieillissement.

Principe (cf. discussion) :
  - La PD reste resolue sur 1 an cyclique, MAIS au niveau de vieillissement
    courant (Pmax FC/ELY vieillis, capacite batterie vieillie). Comme le profil
    est annuel-periodique, la politique ne change d'une annee a l'autre QUE parce
    que les composants vieillissent.
  - La politique est branchee comme DROP-IN de get_optimal_action_RB dans la
    VRAIE boucle init_and_run_loop : le vieillissement (SoH, alpha, Pmax,
    remplacements), la faisabilite (get_lol) et les metriques sont EXACTS, ceux
    du modele de reference. => trajectoire garantie REALISTE.
  - Recompute de la table PD : a chaque debut d'annee + a chaque remplacement.
    Mode 'never' = politique figee a l'etat neuf (borne sup NON-adaptative).

Statut theorique : la PD sequentielle est une politique REALISABLE -> son cout
realise est une BORNE SUPERIEURE de l'optimum global 25 ans (qui, lui, exigerait
le SoH en variable d'etat -> intractable). Et c'est une borne plus serree que
RB2(SoH). Donc : optimum <= PD_seq <= RB2.

V2 (aging_proj + rollout, cf. DPPolicy) : le run v1 (meso 210452) etait battu
par RB2(SoH_all)/RB2(SoH_all+Pred) dans la zone LPSP 2.5-3%. Diagnostic
(decomposition get_cost_* des trajectoires 25 ans) : la PD payait ~10 kEUR de
degradation FC "haute puissance" NON PRICEE (57 000 h > 0.8*P_fc_max VIEILLI,
10 remplacements FC vs 1 pour RB2) + ~2 kEUR d'irreversible ELY au-dessus du
genou 30% vieilli. Cause : les seuils de degradation sont des fractions de
Pmax, qui DERIVE entre deux rebuilds annuels (FC ~-4%/an) ; un niveau de
controle "pile sous le seuil" au rebuild passe au-dessus en cours d'annee.
Remedes v2 :
  - aging_proj : seuils prices au Pmax PROJETE fin d'annee (pente observee
    entre rebuilds), capacite batterie du plan projetee a mi-annee, niveaux
    de controle "ride" juste sous les seuils projetes ;
  - rollout    : a chaque pas, controle choisi par minimisation du cout EXACT
    du pas au vieillissement COURANT (+ V(t+1) interpole depuis des
    checkpoints journaliers) au lieu du lookup plus-proche-voisin.

Lance : python dp_aging.py            (smoke test court)
        python dp_aging.py full       (25 ans, long)
=============================================================================
"""
import os
import sys
import time
import numpy as np

_THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_THIS, '..')))   # -> Vieillissement8/
os.environ.setdefault('GENIAL_DATA_DIR', '/home/theo/Documents/Doctorat/Data')

import dp_core as dp
from dp_core import (SOC_LO, SOC_HI, E_H2_INIT, CAP_WH, ETA, N_YEAR,
                     control_grid, precompute_controls, solve_cyclic,
                     net_reference, net_reference_window)
from Common.get_lol import get_lol
from Common.cost_fcn_total2 import FC_ALPHA_SHIFT

# Metrique UNIFIE officiel (== these) : sens_common.metrics + voll_common.
sys.path.insert(0, os.path.abspath(os.path.join(_THIS, '..', '..', 'Analyse_sensibilite')))
from sens_common import metrics as sens_metrics      # (LPSP %, deg kEUR)
import voll_common as voll                            # total_cost_keur, VOLL=3


# ---------------------------------------------------------------------------
# Politique PD branchable dans init_and_run_loop (signature get_optimal_action_RB)
# ---------------------------------------------------------------------------
class DPPolicy:
    def __init__(self, Ns=51, Nh=51, n_fc=10, n_ely=50, n_iter=3,
                 recompute='yearly', verbose=True, phase_align=True,
                 aging_proj=False, thr_factor=1.0, cap_factor=0.5,
                 rollout=False):
        self.Ns, self.Nh = Ns, Nh
        self.n_fc, self.n_ely, self.n_iter = n_fc, n_ely, n_iter
        self.recompute = recompute            # 'yearly' | 'never'
        self.verbose = verbose
        # phase_align : cale le template PD sur la FENETRE REELLE du profil que la
        # politique va gouverner (robuste a un profil non-8760-periodique). Si False
        # -> ancien comportement (template annee-0, lookup policy[j%8760]) ; ne sert
        # qu'a la validation/reproduction du bug de derive.
        self.phase_align = phase_align
        # aging_proj : entre deux rebuilds, Pmax et capacite DERIVENT (FC ~-4%/an,
        # bat ~-7%/an). Sans projection, un niveau plan "pile sous un seuil de
        # degradation" au rebuild passe au-dessus du seuil reel en cours d'annee
        # et paie FC_ALPHA_HIGH / a(f) ELY en silence (~10+2 kEUR/25 ans au run
        # meso eps=3). Avec aging_proj=True : les pentes de vieillissement sont
        # estimees sur l'intervalle entre rebuilds, les SEUILS sont prices au
        # Pmax projete en fin d'annee (thr_factor=1.0, conservatif car le cout
        # d'un depassement silencieux >> celui d'une marge), et la capacite
        # batterie du plan est projetee a mi-annee (cap_factor=0.5). L'annee 0
        # (aucune pente observee) reste au comportement historique.
        self.aging_proj = aging_proj
        self.thr_factor = thr_factor
        self.cap_factor = cap_factor
        # rollout : au lieu du lookup table plus-proche-voisin, chaque pas choisit
        # le controle en minimisant [cout EXACT du pas au vieillissement COURANT
        # (seuils FC/ELY vieillis, shift FC, start-stop, batterie, VOLL=eps)
        # + V(t+1) interpole]. V est reconstitue jour par jour depuis des
        # checkpoints captures au backward (24 pas re-derives par jour). Corrige
        # a la fois l'erreur de discretisation d'etat et la derive intra-annuelle
        # des seuils. Cout : ~ +1/3 de backward par an + ~0.1 ms/pas.
        self.rollout = rollout
        self.soc_grid = np.linspace(SOC_LO, SOC_HI, Ns)
        self.h2_grid  = np.linspace(0.0, E_H2_INIT, Nh)
        self.P_ref_net, _, _ = net_reference(N_YEAR)   # profil net annee-0 (fallback)
        self.j = 0                            # compteur de pas global
        self.j_anchor = 0                     # pas global au dernier rebuild (origine du template)
        self.policy = None
        self.u = None
        self.fc_on = 0
        self.ely_on = 0
        self._last_soh = None                 # (SoH_fc, SoH_ely, SoH_bat) du pas precedent
        self.n_rebuild = 0
        # pentes de vieillissement observees [fraction/an], estimees entre rebuilds
        self._drift = {'fc': 0.0, 'ely': 0.0, 'bat': 0.0}
        self._proj_ref = None                 # (j, Pfc, Pely, SoHbat) au rebuild precedent
        # etat rollout
        self.ckpt = None                      # {t: V_t} checkpoints journaliers
        self._pre = None
        self._P_ref_tpl = None
        self._cap_plan = None
        self._soh_plan = None
        self._Vday = None
        self._Vday_d = None
        self._P_fc_prev = 0.0                 # P_fc stack realise au pas precedent (shift)
        self._fc_to_eur  = dp.FC['cost']  / ((1 - dp.FC['SoH_EoL'])  * 100.0)
        self._ely_to_eur = dp.ELY['cost'] / ((1 - dp.ELY['SoH_EoL']) * 100.0)

    def _update_drift(self, p_fc_max, p_ely_max, soh_bat):
        """Estime les pentes de vieillissement [fraction/an] sur l'intervalle
        depuis le rebuild precedent. Un REMPLACEMENT dans l'intervalle fait
        REMONTER Pmax/SoH -> on garde alors l'estimation precedente."""
        if self._proj_ref is not None:
            j0, pfc0, pely0, sohb0 = self._proj_ref
            dt_y = (self.j - j0) / N_YEAR
            if dt_y >= 0.25:
                for key, cur, prev in (('fc', p_fc_max, pfc0),
                                       ('ely', p_ely_max, pely0),
                                       ('bat', soh_bat, sohb0)):
                    if cur <= prev + 1e-9 and prev > 0:
                        self._drift[key] = max(0.0, (1.0 - cur / prev) / dt_y)
        self._proj_ref = (self.j, p_fc_max, p_ely_max, soh_bat)

    def _rebuild(self, soh_bat, p_fc_max, p_ely_max):
        """Re-resout la PD 1 an au niveau de vieillissement courant, sur la fenetre
        de profil REELLE demarrant au pas courant (origine = self.j_anchor)."""
        t0 = time.time()
        thr_fc, thr_ely, soh_plan = p_fc_max, p_ely_max, soh_bat
        extra = ()
        if self.aging_proj:
            self._update_drift(p_fc_max, p_ely_max, soh_bat)
            thr_fc   = p_fc_max  * max(0.05, 1.0 - self.thr_factor * self._drift['fc'])
            thr_ely  = p_ely_max * max(0.05, 1.0 - self.thr_factor * self._drift['ely'])
            soh_plan = soh_bat   * max(0.05, 1.0 - self.cap_factor * self._drift['bat'])
            # niveaux "ride" juste SOUS les seuils projetes (la grille reguliere
            # ne tombe jamais pile dessous une fois le seuil decale)
            extra = (0.999 * dp.FC_FHIGH * thr_fc * ETA,
                     -0.999 * dp.ELY_F30 * thr_ely / ETA)
        u = control_grid(self.n_fc, self.n_ely, p_fc_max, p_ely_max, extra_u=extra)
        pre = precompute_controls(u, p_fc_max, p_ely_max,
                                  thr_fc_max=thr_fc, thr_ely_max=thr_ely)
        cap = CAP_WH * soh_plan
        if self.phase_align:
            P_ref = net_reference_window(self.j_anchor, N_YEAR)   # fenetre reelle
        else:
            P_ref = self.P_ref_net                                # template annee-0 (ancien)
        if self.rollout:
            _, policy, ckpt = solve_cyclic(self.soc_grid, self.h2_grid, u, pre, P_ref,
                                           n_iter=self.n_iter, verbose=False,
                                           cap_wh=cap, soh_bat=soh_plan, ckpt_every=24)
            self.ckpt = ckpt
        else:
            _, policy = solve_cyclic(self.soc_grid, self.h2_grid, u, pre, P_ref,
                                     n_iter=self.n_iter, verbose=False,
                                     cap_wh=cap, soh_bat=soh_plan)
        self.u = u
        self.policy = policy
        self._pre = pre
        self._P_ref_tpl = P_ref
        self._cap_plan = cap
        self._soh_plan = soh_plan
        self._Vday = None
        self._Vday_d = None
        self.n_rebuild += 1
        if self.verbose:
            yr = self.j // N_YEAR
            print(f"   [DP rebuild #{self.n_rebuild}] an {yr}  pas {self.j}  "
                  f"SoH_bat={soh_bat:.3f}  Pfc={p_fc_max:.0f}W  Pely={p_ely_max:.0f}W  "
                  f"({time.time()-t0:.0f}s)", flush=True)

    @staticmethod
    def _nearest(grid, x):
        i = int(np.clip(np.searchsorted(grid, x), 0, len(grid) - 1))
        if i > 0 and (x - grid[i - 1]) < (grid[i] - x):
            i -= 1
        return i

    # ------------------------------------------------------------------
    # Rollout : V(t) reconstitue a la demande depuis les checkpoints 24 h
    # ------------------------------------------------------------------
    def _V_at(self, tt1):
        """V au pas template tt1 (valeur d'etre dans un etat au DEBUT de tt1)."""
        T = len(self._P_ref_tpl)
        if tt1 > T:
            tt1 = T
        if tt1 in self.ckpt:
            return self.ckpt[tt1]
        d = tt1 // 24                          # tt1 non multiple de 24 -> jour d
        if self._Vday_d != d:
            s = d * 24
            e = min(s + 24, T)
            Vn = self.ckpt[e] if e in self.ckpt else self.ckpt[T]
            cache = {}
            for t in range(e - 1, s, -1):      # re-derive V_{e-1}..V_{s+1}
                Vt, _ = dp.backward(self.soc_grid, self.h2_grid, self.u, self._pre,
                                    self._P_ref_tpl[t:t + 1], Vn,
                                    cap_wh=self._cap_plan, soh_bat=self._soh_plan)
                cache[t] = Vt
                Vn = Vt
            self._Vday = cache
            self._Vday_d = d
        return self._Vday[tt1]

    def _rollout_action(self, tt, SoC, E_h2, Ptot, pfc_t, pely_t, soh_b, E_h2_init):
        """Choix du controle par minimisation [cout EXACT du pas au vieillissement
        COURANT + V(t+1) interpole]. Corrige la derive intra-annuelle des seuils
        (P_high FC, genou 30% ELY, capacite batterie) et l'erreur de lookup
        plus-proche-voisin. Le prix de l'energie non servie est dp.VOLL (= eps)."""
        u = self.u
        # candidats : grille du rebuild + niveaux "ride" au Pmax COURANT + 0
        cand = np.concatenate([u, [0.999 * dp.FC_FHIGH * pfc_t * ETA,
                                   -0.999 * dp.ELY_F30 * pely_t / ETA, 0.0]])
        cand = np.unique(cand[(cand <= pfc_t * ETA + 1e-9)
                              & (cand >= -pely_t / ETA - 1e-9)])
        P_dc_fc  = np.maximum(cand, 0.0)
        P_dc_ely = np.minimum(cand, 0.0)
        P_fc_st  = P_dc_fc / ETA                       # cote stack
        P_ely_st = np.abs(P_dc_ely) * ETA
        eff_fc  = np.interp(P_fc_st  / pfc_t  * 100.0, dp.FC_LUT[0],  dp.FC_LUT[1])  / 100.0
        eff_ely = np.interp(P_ely_st / pely_t * 100.0, dp.ELY_LUT[0], dp.ELY_LUT[1]) / 100.0
        with np.errstate(divide='ignore', invalid='ignore'):
            term_fc = np.where(P_dc_fc > 0, P_fc_st / eff_fc, 0.0)
        P_h2 = (P_ely_st * eff_ely - term_fc) / 1000.0            # kW
        E_tp1 = E_h2 + P_h2 * dp.TS_H
        feas = (E_tp1 >= -1e-9) & (E_tp1 <= E_h2_init + 1e-9)

        # batterie (capacite vieillie COURANTE) + energie non servie au plancher
        P_dc_bat = Ptot - cand
        P_bat = P_dc_bat / ETA ** np.sign(P_dc_bat)
        fac = dp.EFF_BAT ** np.sign(-P_bat)
        cap_wh_t = CAP_WH * soh_b
        soc_tp1 = SoC - P_bat * fac / cap_wh_t
        soc_cl = np.clip(soc_tp1, SOC_LO, SOC_HI)
        lpsp_eur = np.zeros_like(cand)
        if Ptot > 0:
            hit = soc_tp1 < SOC_LO
            P_bat_max = (SoC - SOC_LO) * cap_wh_t / dp.EFF_BAT
            unmet = np.clip(Ptot - (P_bat_max * ETA + cand), 0.0, None)
            lpsp_eur = np.where(hit, unmet / 1000.0 * dp.TS_H * dp.VOLL, 0.0)
        cbat = dp.battery_cost_step(np.array([SoC]), P_bat, soc_cl[None, :],
                                    soh_bat=soh_b)[0]

        # FC : seuils au Pmax vieilli COURANT + start-stop + shift (modele exact)
        P_high_t = dp.FC_FHIGH * pfc_t
        P_low_t  = dp.FC_FLOW  * pfc_t
        fc_on_n  = P_fc_st >= 1.0
        deg_fc = ((P_fc_st > P_high_t) * dp.FC_ALPHA_HIGH * dp.TS_H
                  + ((P_fc_st < P_low_t) & (P_fc_st > 1.0)) * dp.FC_ALPHA_LOW * dp.TS_H
                  + (fc_on_n & (self.fc_on == 0)) * 0.5 * dp.FC_ALPHA_ON_OFF
                  + FC_ALPHA_SHIFT * np.abs(P_fc_st - self._P_fc_prev)
                    / (P_high_t - P_low_t))
        cfc = deg_fc * self._fc_to_eur

        # ELY : rampe a(f), idle, start au Pmax vieilli COURANT (V_rev recuperable
        # -> ~0 EUR realise, ignore comme au backward)
        f = P_ely_st / pely_t
        a = np.where(f <= dp.ELY_F30, 0.0,
             np.where(f <= dp.ELY_F60,
                      dp.ELY_REC['a2'] * (f - dp.ELY_F30) / (dp.ELY_F60 - dp.ELY_F30),
                      dp.ELY_REC['a2']))
        ely_on_n = P_ely_st >= 0.0005 * pely_t
        deg_ely = (a * dp.TS_H * dp.UV_TO_PCT
                   + ((P_ely_st > 0) & (P_ely_st <= 0.01 * pely_t))
                     * dp.ELY_REC['idle'] * dp.TS_H * dp.UV_TO_PCT
                   + (ely_on_n & (self.ely_on == 0)) * (dp.ELY_REC['s'] * dp.UV_TO_PCT))
        cely = deg_ely * self._ely_to_eur

        # V(t+1) bilineaire en (SoC', E_h2'), etats marche/arret suivants
        V1 = self._V_at(tt + 1)
        sg, hg = self.soc_grid, self.h2_grid
        E_cl = np.clip(E_tp1, hg[0], hg[-1])
        il = np.clip(np.searchsorted(sg, soc_cl) - 1, 0, len(sg) - 2)
        wl = (sg[il + 1] - soc_cl) / (sg[il + 1] - sg[il])
        jl = np.clip(np.searchsorted(hg, E_cl) - 1, 0, len(hg) - 2)
        wj = (hg[jl + 1] - E_cl) / (hg[jl + 1] - hg[jl])
        nf = fc_on_n.astype(np.int64)
        ne = ely_on_n.astype(np.int64)
        fut = (wl * wj * V1[il, jl, nf, ne]
               + wl * (1 - wj) * V1[il, jl + 1, nf, ne]
               + (1 - wl) * wj * V1[il + 1, jl, nf, ne]
               + (1 - wl) * (1 - wj) * V1[il + 1, jl + 1, nf, ne])

        total = cbat + lpsp_eur + cfc + cely + fut + np.where(feas, 0.0, 1e18)
        return float(cand[int(np.argmin(total))])

    def __call__(self, SoC, P_tot_ref, defaillances, lol_tab, alpha_fc, alpha_ely,
                 SoH_bat, E_h2, E_h2_init, P_fc_max, P_ely_max, RUL_fc, RUL_ely,
                 SoH_fc, SoH_ely):
        hour = self.j % N_YEAR

        # parametres de vieillissement utilises pour le (re)build
        soh_b, pfc, pely = SoH_bat, P_fc_max, P_ely_max

        need = (self.policy is None)
        if self.recompute == 'yearly':
            if hour == 0:
                need = True
            # remplacement = SAUT de SoH vers ~1.0 (>0.05). NB : le SoH_ely
            # remonte legerement en continu (recuperation reversible PEMWE) ->
            # on NE doit PAS confondre avec un remplacement (seuil 0.05).
            if self._last_soh is not None:
                if (SoH_fc  > self._last_soh[0] + 0.05 or
                    SoH_ely > self._last_soh[1] + 0.05 or
                    SoH_bat > self._last_soh[2] + 0.05):
                    need = True
        elif self.recompute == 'never':
            # politique NON-adaptative au vieillissement : aging gele au BoL.
            # Mais avec phase_align on re-ancre la PHASE chaque annee (sinon le
            # template annee-0 desynchronise sur un profil derivant).
            soh_b, pfc, pely = 1.0, dp.P_FC_MAX, dp.P_ELY_MAX
            if self.phase_align and hour == 0:
                need = True
        if need:
            self.j_anchor = self.j            # origine du template = pas courant
            self._rebuild(soh_b, pfc, pely)
        self._last_soh = (SoH_fc, SoH_ely, SoH_bat)

        # index temporel dans le template PD :
        #  - phase_align : pas ecoule depuis le rebuild (template cale sur la fenetre
        #    reelle) -> reste synchrone au cycle jour/nuit meme si le profil derive.
        #  - sinon : heure-de-l'annee j%8760 (ancien, casse si profil non periodique).
        if self.phase_align:
            tt = min(self.j - self.j_anchor, self.policy.shape[0] - 1)
        else:
            tt = hour
        if self.rollout:
            # minimisation exacte au pas courant (etat continu, Pmax vieillis)
            u_val = self._rollout_action(tt, SoC, E_h2, P_tot_ref,
                                         P_fc_max, P_ely_max, SoH_bat, E_h2_init)
        else:
            # lookup table -> controle u
            i  = self._nearest(self.soc_grid, SoC)
            jx = self._nearest(self.h2_grid, E_h2)
            ui = self.policy[tt, i, jx, self.fc_on, self.ely_on]
            u_val = self.u[ui]
        P_dc_fc  = max(u_val, 0.0)
        P_dc_ely = min(u_val, 0.0)
        P_dc_bat = P_tot_ref - u_val
        action = (P_dc_bat, P_dc_fc, P_dc_ely)

        # faisabilite EXACTE (clamp batterie/Pmax/H2 + LPSP), comme RB2
        action, lol = get_lol(SoC, action, P_tot_ref, defaillances, E_h2, E_h2_init,
                              P_fc_max, P_ely_max, SoH_bat)

        # maj etat marche/arret a partir de l'action EFFECTIVE (post get_lol)
        self.fc_on  = int(action[1] / ETA >= 1.0)
        self.ely_on = int(abs(action[2]) * ETA >= 0.0005 * P_ely_max)
        self._P_fc_prev = action[1] / ETA          # pour le cout de shift FC
        self.j += 1
        return action, lol


# ---------------------------------------------------------------------------
# Resume realisme/trajectoire (independant du metrique de cout final)
# ---------------------------------------------------------------------------
def summarize(data, label):
    SoH_bat = data['SoH_bat']; SoH_fc = data['SoH_fc']; SoH_ely = data['SoH_ely']
    lol = data['lol_tab']
    # --- metrique OFFICIEL (these) ---
    lpsp, deg = sens_metrics(data)            # LPSP %, degradation kEUR
    total = voll.total_cost_keur(lpsp, deg)   # deg + VOLL(3)*EENS
    # remplacements = nombre de remontees de SoH (reset a 1)
    def n_repl(s):
        return int(np.sum(np.diff(s) > 0.05))
    print(f"\n--- {label} ---")
    print(f"  pas={len(lol)}  (~{len(lol)/N_YEAR:.1f} ans)")
    print(f"  SoH final : bat={SoH_bat[-1]:.3f}  fc={SoH_fc[-1]:.3f}  ely={SoH_ely[-1]:.3f}")
    print(f"  remplacements : bat={n_repl(SoH_bat)}  fc={n_repl(SoH_fc)}  ely={n_repl(SoH_ely)}")
    print(f"  SoH min   : bat={SoH_bat.min():.3f}  fc={SoH_fc.min():.3f}  ely={SoH_ely.min():.3f}")
    print(f"  LPSP {lpsp:.4f}%   deg {deg:.3f} kEUR   UNIFIE {total:.3f} kEUR")
    return dict(label=label, lpsp=lpsp, deg=deg, total=total,
                soh_bat=SoH_bat[-1], soh_fc=SoH_fc[-1], soh_ely=SoH_ely[-1])


def main():
    full = len(sys.argv) > 1 and sys.argv[1] == 'full'
    if full:
        n_years, Ns, Nh, n_fc, n_ely = 25, 51, 51, 10, 50
    else:
        n_years, Ns, Nh, n_fc, n_ely = 6, 25, 25, 7, 24   # smoke test rapide

    import inspect
    from Common.main_init_and_loop import init_and_run_loop
    sys.path.insert(0, os.path.abspath(os.path.join(_THIS, '..', 'RB2')))
    from get_optimal_action_RB import get_optimal_action_RB

    # La boucle de reference ne supporte 'n_years' que si Common a ete modifie
    # (test court). Pour l'horizon par defaut (25 ans) on l'appelle SANS l'argument
    # -> aucun besoin de toucher au Common partage sur le mesocentre.
    _has_ny = 'n_years' in inspect.signature(init_and_run_loop).parameters
    def run_loop(pol):
        if n_years == 25 or not _has_ny:
            return init_and_run_loop(pol)
        return init_and_run_loop(pol, n_years=n_years)

    print("=" * 70)
    print(f" PD VIEILLISSEMENT -- horizon {n_years} ans  grille {Ns}x{Nh}  "
          f"({'FULL' if full else 'SMOKE'})")
    print("=" * 70)

    res = []
    runs = {}
    # --- RB2 (reference) ---
    t0 = time.time()
    data_rb = run_loop(get_optimal_action_RB)
    res.append(summarize(data_rb, f"RB2"))
    runs['RB2'] = data_rb
    print(f"  (RB2 loop : {time.time()-t0:.0f}s)")

    # --- PD non-adaptative (politique BoL figee) = borne sup non-adaptative ---
    t0 = time.time()
    pol_bol = DPPolicy(Ns, Nh, n_fc, n_ely, recompute='never')
    data_bol = run_loop(pol_bol)
    res.append(summarize(data_bol, f"PD BoL-figee"))
    runs['PD_BoL'] = data_bol
    print(f"  (PD-BoL : {time.time()-t0:.0f}s, {pol_bol.n_rebuild} rebuild)")

    # --- PD sequentielle adaptative (recompute annuel) ---
    t0 = time.time()
    pol_seq = DPPolicy(Ns, Nh, n_fc, n_ely, recompute='yearly')
    data_seq = run_loop(pol_seq)
    res.append(summarize(data_seq, f"PD sequentielle"))
    runs['PD_seq'] = data_seq
    print(f"  (PD-seq : {time.time()-t0:.0f}s, {pol_seq.n_rebuild} rebuild)")

    # --- PD sequentielle v2 : projection du vieillissement + rollout exact ---
    t0 = time.time()
    pol_v2 = DPPolicy(Ns, Nh, n_fc, n_ely, recompute='yearly',
                      aging_proj=True, rollout=True)
    data_v2 = run_loop(pol_v2)
    res.append(summarize(data_v2, f"PD seq proj+rollout"))
    runs['PD_seq_v2'] = data_v2
    print(f"  (PD-seq-v2 : {time.time()-t0:.0f}s, {pol_v2.n_rebuild} rebuild)")

    # --- tableau comparatif + verdict borne ---
    rb_total = res[0]['total']
    header = f" {'strategie':<18} {'LPSP%':>8} {'deg_kE':>9} {'UNIF_kE':>9} {'gain%':>8}"
    lines = ["=" * 70,
             f" COMPARATIF UNIFIE ({n_years} ans, VoLL=3)  [borne sup de l'optimum global]",
             "=" * 70, header]
    for r in res:
        gain = (rb_total - r['total']) / rb_total * 100 if rb_total else 0.0
        lines.append(f" {r['label']:<18} {r['lpsp']:8.4f} {r['deg']:9.3f} "
                     f"{r['total']:9.3f} {gain:+8.2f}")
    lines += ["-" * 70,
              " optimum_global <= PD_seq <= RB2  ; gain(PD_seq) = borne INF du gain atteignable",
              f" grille {Ns}x{Nh}  n_fc={n_fc} n_ely={n_ely}  rebuilds PD-seq={pol_seq.n_rebuild}"]
    table = "\n".join(lines)
    print("\n" + table)

    # --- sauvegarde (txt + npz pour figures ulterieures) ---
    out_dir = os.path.join(_THIS, "results")
    os.makedirs(out_dir, exist_ok=True)
    tag = f"{n_years}y_{Ns}x{Nh}"
    with open(os.path.join(out_dir, f"dp_aging_{tag}.txt"), "w") as f:
        f.write(table + "\n")
    np.savez_compressed(
        os.path.join(out_dir, f"dp_aging_{tag}.npz"),
        **{f"{k}__{q}": v for k, data in runs.items()
           for q, v in data.items() if isinstance(v, np.ndarray)})
    print(f"\n Resultats -> {out_dir}/dp_aging_{tag}.txt (+ .npz)")


if __name__ == "__main__":
    main()
