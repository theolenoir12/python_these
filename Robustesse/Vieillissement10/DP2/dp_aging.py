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

# Metrique UNIFIE officiel (== these) : sens_common.metrics + voll_common.
sys.path.insert(0, os.path.abspath(os.path.join(_THIS, '..', '..', 'Analyse_sensibilite')))
from sens_common import metrics as sens_metrics      # (LPSP %, deg kEUR)
import voll_common as voll                            # total_cost_keur, VOLL=3


# ---------------------------------------------------------------------------
# Politique PD branchable dans init_and_run_loop (signature get_optimal_action_RB)
# ---------------------------------------------------------------------------
class DPPolicy:
    def __init__(self, Ns=51, Nh=51, n_fc=10, n_ely=50, n_iter=3,
                 recompute='yearly', verbose=True, phase_align=True):
        self.Ns, self.Nh = Ns, Nh
        self.n_fc, self.n_ely, self.n_iter = n_fc, n_ely, n_iter
        self.recompute = recompute            # 'yearly' | 'never'
        self.verbose = verbose
        # phase_align : cale le template PD sur la FENETRE REELLE du profil que la
        # politique va gouverner (robuste a un profil non-8760-periodique). Si False
        # -> ancien comportement (template annee-0, lookup policy[j%8760]) ; ne sert
        # qu'a la validation/reproduction du bug de derive.
        self.phase_align = phase_align
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

    def _rebuild(self, soh_bat, p_fc_max, p_ely_max):
        """Re-resout la PD 1 an au niveau de vieillissement courant, sur la fenetre
        de profil REELLE demarrant au pas courant (origine = self.j_anchor)."""
        t0 = time.time()
        u = control_grid(self.n_fc, self.n_ely, p_fc_max, p_ely_max)
        pre = precompute_controls(u, p_fc_max, p_ely_max)
        cap = CAP_WH * soh_bat
        if self.phase_align:
            P_ref = net_reference_window(self.j_anchor, N_YEAR)   # fenetre reelle
        else:
            P_ref = self.P_ref_net                                # template annee-0 (ancien)
        _, policy = solve_cyclic(self.soc_grid, self.h2_grid, u, pre, P_ref,
                                 n_iter=self.n_iter, verbose=False,
                                 cap_wh=cap, soh_bat=soh_bat)
        self.u = u
        self.policy = policy
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
