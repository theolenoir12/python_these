"""PD sequentielle online sur le modele V11 nominal ``p=2``.

La table annuelle est reconstruite a l'etat de vieillissement courant. La
politique est ensuite branchee dans ``init_and_run_loop`` : faisabilite,
vieillissement, remplacements et cout realise proviennent donc du meme socle
V11 que RB1/RB2. Le rollout utilise en plus la stabilite PEMFC courante fournie
par ``aging_context``. Les anciens seuils de puissance de V8/V9 ne sont plus
utilises.
"""
import os
import sys
import time
import json
import numpy as np

_THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_THIS, '..')))   # -> Vieillissement8/
os.environ.setdefault('GENIAL_DATA_DIR', '/home/theo/Documents/Doctorat/Data')

import dp_core as dp
from dp_core import (SOC_LO, SOC_HI, E_H2_INIT, CAP_WH, ETA, N_YEAR,
                     control_grid, precompute_controls, solve_cyclic,
                     net_reference, net_reference_window,
                     v11_control_anchors)
from Common.get_lol import get_lol
from Common.reliability_metrics import compute_reliability_metrics
from Common.rb1_policy_v11 import make_rb1_policy_v11
from Common.rb2_policy import make_rb2_policy

VOLL_REF = 3.0


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
        # aging_proj est conserve pour les comparaisons A/B : sous V11 il ne
        # projette plus de seuil de puissance (le dommage PEMWE depend de j),
        # seulement la capacite batterie a l'interieur de l'annee.
        self.aging_proj = aging_proj
        self.thr_factor = thr_factor
        self.cap_factor = cap_factor
        # rollout : minimisation cout permanent V11 courant + V(t+1), avec
        # stabilite PEMFC et densites de courant courantes.
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
        self._drift_bat = 0.0                 # fraction/an entre rebuilds
        self._proj_ref = None                 # (j, SoHbat) au rebuild precedent
        # etat rollout
        self.ckpt = None                      # {t: V_t} checkpoints journaliers
        self._pre = None
        self._P_ref_tpl = None
        self._cap_plan = None
        self._soh_plan = None
        self._Vday = None
        self._Vday_d = None
        self._j_fc_prev = 0.0                 # densite realisee au pas precedent

    def reset(self):
        """Repart d'un etat vierge lorsque la boucle reutilise la politique."""
        self.j = 0
        self.j_anchor = 0
        self.policy = self.u = None
        self.fc_on = self.ely_on = 0
        self._last_soh = None
        self.n_rebuild = 0
        self._drift_bat = 0.0
        self._proj_ref = None
        self.ckpt = self._pre = self._P_ref_tpl = None
        self._cap_plan = self._soh_plan = None
        self._Vday = self._Vday_d = None
        self._j_fc_prev = 0.0

    def _update_drift(self, soh_bat):
        """Estime la derive annuelle de capacite entre deux rebuilds."""
        if self._proj_ref is not None:
            j0, sohb0 = self._proj_ref
            dt_y = (self.j - j0) / N_YEAR
            if dt_y >= 0.25 and soh_bat <= sohb0 + 1e-9 and sohb0 > 0:
                self._drift_bat = max(0.0, (1.0 - soh_bat / sohb0) / dt_y)
        self._proj_ref = (self.j, soh_bat)

    def _rebuild(self, soh_bat, p_fc_max, p_ely_max, alpha_fc, alpha_ely):
        """Re-resout la PD 1 an au niveau de vieillissement courant, sur la fenetre
        de profil REELLE demarrant au pas courant (origine = self.j_anchor)."""
        t0 = time.time()
        soh_plan = soh_bat
        if self.aging_proj:
            self._update_drift(soh_bat)
            soh_plan = soh_bat * max(0.05, 1.0 - self.cap_factor * self._drift_bat)
        extra = v11_control_anchors(alpha_ely, p_ely_max)
        u = control_grid(self.n_fc, self.n_ely, p_fc_max, p_ely_max, extra_u=extra)
        pre = precompute_controls(
            u, p_fc_max, p_ely_max, alpha_fc=alpha_fc, alpha_ely=alpha_ely)
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
                  f"alpha=({alpha_fc:.3f},{alpha_ely:.3f})  "
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

    def _rollout_action(self, tt, SoC, E_h2, Ptot, pfc_t, pely_t, soh_b,
                        E_h2_init, alpha_fc, alpha_ely, aging_context):
        """Minimise cout permanent V11 courant + valeur future interpolee."""
        u = self.u
        # Les ancres j=1 et j=2 suivent alpha_ely au pas courant.
        cand = np.concatenate([u, v11_control_anchors(alpha_ely, pely_t), [0.0]])
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

        # PEMFC : meme transition de stabilite que degradation_v11, avec l'etat
        # courant fourni par la boucle et la densite effectivement realisee au
        # pas precedent.
        j_fc = np.asarray(dp.fc_current_density(P_fc_st, alpha_fc), dtype=float)
        fc_on_n = j_fc > 1e-9
        old_steadiness = 1.0
        if aging_context is not None:
            old_steadiness = float(
                aging_context.get('fc', {}).get('steadiness', 1.0))
        after_change = old_steadiness * np.exp(
            -np.abs(j_fc - self._j_fc_prev) / dp.FC_V11['change_scale_a_cm2'])
        steadiness = 1.0 - (1.0 - after_change) * np.exp(
            -dp.TS_H / dp.FC_V11['steadiness_tau_h'])
        fc_rate = (dp.FC_V11['irr_dynamic_uvph']
                   + steadiness * (dp.FC_V11['irr_steady_uvph']
                                   - dp.FC_V11['irr_dynamic_uvph']))
        if dp.FC_V11['current_exponent'] != 0.0:
            fc_rate *= np.where(
                fc_on_n,
                (j_fc / dp.FC_V11['j_ref']) ** dp.FC_V11['current_exponent'],
                0.0)
        fc_uv = np.where(fc_on_n, fc_rate * dp.TS_H, 0.0)
        fc_uv += ((self.fc_on == 0) & fc_on_n) * dp.FC_V11['start_uv']
        fc_uv += ((j_fc > 0.0) & (j_fc <= 0.05)) * dp.FC_V11['idle_uvph'] * dp.TS_H
        cfc = dp.permanent_uv_to_eur('fc', fc_uv)

        # PEMWE : noyau irreversible quadratique nominal, plus termes hybrides.
        j_ely = np.asarray(dp.ely_current_density(P_ely_st, alpha_ely), dtype=float)
        ely_on_n = j_ely > 1e-9
        ely_uv = (dp.ELY_V11['steady_2_uvph']
                  * np.maximum(j_ely - 1.0, 0.0) ** dp.NOMINAL_ELY_STRESS_EXPONENT
                  * dp.TS_H)
        ely_uv += ((j_ely > 0.0) & (j_ely <= 0.01)) \
            * dp.ELY_V11['idle_uvph'] * dp.TS_H
        ely_uv += ((self.ely_on == 0) & ely_on_n) * dp.ELY_V11['start_uv']
        cely = dp.permanent_uv_to_eur('ely', ely_uv)

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
                 SoH_fc, SoH_ely, aging_context=None):
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
            alpha_fc_plan, alpha_ely_plan = 0.0, 0.0
            if self.phase_align and hour == 0:
                need = True
        else:
            alpha_fc_plan, alpha_ely_plan = alpha_fc, alpha_ely
        if self.recompute != 'never':
            alpha_fc_plan, alpha_ely_plan = alpha_fc, alpha_ely
        if need:
            self.j_anchor = self.j            # origine du template = pas courant
            self._rebuild(soh_b, pfc, pely, alpha_fc_plan, alpha_ely_plan)
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
                                         P_fc_max, P_ely_max, SoH_bat, E_h2_init,
                                         alpha_fc, alpha_ely, aging_context)
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
        self._j_fc_prev = float(dp.fc_current_density(action[1] / ETA, alpha_fc))
        self.fc_on = int(self._j_fc_prev > 1e-9)
        self.ely_on = int(
            dp.ely_current_density(abs(action[2]) * ETA, alpha_ely) > 1e-9)
        self.j += 1
        return action, lol


# ---------------------------------------------------------------------------
# Resume realisme/trajectoire (independant du metrique de cout final)
# ---------------------------------------------------------------------------
def realized_metrics(data, voll=VOLL_REF):
    """Metriques V11 attribuables : ledger corrige + fiabilite commune."""
    ledger = data.get('degradation_ledger')
    if ledger is None:
        raise ValueError("La trajectoire ne contient pas le degradation_ledger V11")
    degradation_keur = sum(ledger['total_eur'].values()) / 1000.0
    rel = compute_reliability_metrics(data)
    total_keur = degradation_keur + float(voll) * rel['eens_kwh'] / 1000.0
    return {
        'lpsp': rel['lpsp_pct'],
        'eens_kwh': rel['eens_kwh'],
        'demand_kwh': rel['load_energy_kwh'],
        'deg': degradation_keur,
        'total': total_keur,
    }


def replacement_counts(data):
    counts = {'bat': 0, 'fc': 0, 'ely': 0}
    for event in data['degradation_ledger']['events']:
        counts[event['component']] += 1
    return counts


def summarize(data, label):
    SoH_bat = data['SoH_bat']; SoH_fc = data['SoH_fc']; SoH_ely = data['SoH_ely']
    lol = data['lol_tab']
    metrics = realized_metrics(data)
    replacements = replacement_counts(data)
    print(f"\n--- {label} ---")
    print(f"  pas={len(lol)}  (~{len(lol)/N_YEAR:.1f} ans)")
    print(f"  SoH final : bat={SoH_bat[-1]:.3f}  fc={SoH_fc[-1]:.3f}  ely={SoH_ely[-1]:.3f}")
    print(f"  remplacements : bat={replacements['bat']}  fc={replacements['fc']}  "
          f"ely={replacements['ely']}")
    print(f"  SoH min   : bat={SoH_bat.min():.3f}  fc={SoH_fc.min():.3f}  ely={SoH_ely.min():.3f}")
    print(f"  LPSP {metrics['lpsp']:.4f}%   EENS {metrics['eens_kwh']:.1f} kWh  "
          f"deg {metrics['deg']:.3f} kEUR   UNIFIE {metrics['total']:.3f} kEUR")
    return dict(label=label, **metrics,
                soh_bat=SoH_bat[-1], soh_fc=SoH_fc[-1], soh_ely=SoH_ely[-1])


def main():
    full = len(sys.argv) > 1 and sys.argv[1] == 'full'
    if full:
        n_years, Ns, Nh, n_fc, n_ely, n_iter = 25, 51, 51, 10, 50, 3
    else:
        n_years, Ns, Nh, n_fc, n_ely, n_iter = 1, 7, 7, 3, 6, 1

    from Common.main_init_and_loop import init_and_run_loop

    def run_loop(pol):
        if n_years == 25:
            return init_and_run_loop(pol)
        return init_and_run_loop(pol, n_years=n_years)

    print("=" * 70)
    print(f" PD VIEILLISSEMENT -- horizon {n_years} ans  grille {Ns}x{Nh}  "
          f"({'FULL' if full else 'SMOKE'})")
    print("=" * 70)

    res = []
    runs = {}
    # --- references best-vs-best V11 p=2 ---
    for label, key, policy in (
        ("RB1(0.20,0.40)", "RB1_p2_tuned", make_rb1_policy_v11(0.20, 0.40)),
        ("RB2(0.574,0.465)", "RB2_p2_tuned", make_rb2_policy(0.574, 0.465)),
    ):
        t0 = time.time()
        data_ref = run_loop(policy)
        res.append(summarize(data_ref, label))
        runs[key] = data_ref
        print(f"  ({label} loop : {time.time()-t0:.0f}s)")

    # --- PD non-adaptative (politique BoL figee) = borne sup non-adaptative ---
    t0 = time.time()
    pol_bol = DPPolicy(Ns, Nh, n_fc, n_ely, n_iter=n_iter, recompute='never')
    data_bol = run_loop(pol_bol)
    res.append(summarize(data_bol, f"PD BoL-figee"))
    runs['PD_BoL'] = data_bol
    print(f"  (PD-BoL : {time.time()-t0:.0f}s, {pol_bol.n_rebuild} rebuild)")

    # --- PD sequentielle adaptative (recompute annuel) ---
    t0 = time.time()
    pol_seq = DPPolicy(Ns, Nh, n_fc, n_ely, n_iter=n_iter, recompute='yearly')
    data_seq = run_loop(pol_seq)
    res.append(summarize(data_seq, f"PD sequentielle"))
    runs['PD_seq'] = data_seq
    print(f"  (PD-seq : {time.time()-t0:.0f}s, {pol_seq.n_rebuild} rebuild)")

    # --- PD sequentielle v2 : projection du vieillissement + rollout exact ---
    t0 = time.time()
    pol_v2 = DPPolicy(Ns, Nh, n_fc, n_ely, n_iter=n_iter, recompute='yearly',
                      aging_proj=True, rollout=True)
    data_v2 = run_loop(pol_v2)
    res.append(summarize(data_v2, f"PD seq proj+rollout"))
    runs['PD_seq_v2'] = data_v2
    print(f"  (PD-seq-v2 : {time.time()-t0:.0f}s, {pol_v2.n_rebuild} rebuild)")

    # --- tableau comparatif + verdict borne ---
    best_reference = min(res[:2], key=lambda row: row['total'])
    reference_total = best_reference['total']
    header = f" {'strategie':<18} {'LPSP%':>8} {'deg_kE':>9} {'UNIF_kE':>9} {'gain%':>8}"
    lines = ["=" * 70,
             f" COMPARATIF UNIFIE V11 p=2 ({n_years} ans, VoLL=3)",
             "=" * 70, header]
    for r in res:
        gain = ((reference_total - r['total']) / reference_total * 100
                if reference_total else 0.0)
        lines.append(f" {r['label']:<18} {r['lpsp']:8.4f} {r['deg']:9.3f} "
                     f"{r['total']:9.3f} {gain:+8.2f}")
    lines += ["-" * 70,
              f" reference nominale = {best_reference['label']} ; p_ELY=2 ; ledger corrige",
              " La PD-seq est une politique realisable; aucune optimalite 25 ans n'est revendiquee.",
              f" grille {Ns}x{Nh}  n_fc={n_fc} n_ely={n_ely}  rebuilds PD-seq={pol_seq.n_rebuild}"]
    table = "\n".join(lines)
    print("\n" + table)

    # --- sauvegarde (txt + npz pour figures ulterieures) ---
    out_dir = os.path.join(_THIS, "runs")
    os.makedirs(out_dir, exist_ok=True)
    tag = f"v11_p2_{n_years}y_{Ns}x{Nh}"
    with open(os.path.join(out_dir, f"dp_aging_{tag}.txt"), "w") as f:
        f.write(table + "\n")
    np.savez_compressed(
        os.path.join(out_dir, f"dp_aging_{tag}.npz"),
        model_id=np.array(dp.MODEL_ID),
        ely_stress_exponent=np.array(dp.NOMINAL_ELY_STRESS_EXPONENT),
        rb1_parameters=np.array(dp.RB1_REFERENCE_PARAMS),
        rb2_parameters=np.array(dp.RB2_REFERENCE_PARAMS),
        **{f"{k}__{q}": v for k, data in runs.items()
           for q, v in data.items() if isinstance(v, np.ndarray)})
    with open(os.path.join(out_dir, f"dp_aging_{tag}_ledgers.json"), "w") as f:
        json.dump({
            'model_id': dp.MODEL_ID,
            'ely_stress_exponent': dp.NOMINAL_ELY_STRESS_EXPONENT,
            'rb1_parameters': dp.RB1_REFERENCE_PARAMS,
            'rb2_parameters': dp.RB2_REFERENCE_PARAMS,
            'runs': {key: data['degradation_ledger']
                     for key, data in runs.items()},
        }, f, indent=2)
    print(f"\n Resultats -> {out_dir}/dp_aging_{tag}.txt (+ .npz, + _ledgers.json)")


if __name__ == "__main__":
    main()
