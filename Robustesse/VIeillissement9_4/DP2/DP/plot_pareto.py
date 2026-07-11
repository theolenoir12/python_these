"""
plot_pareto.py -- figure de la frontiere de Pareto degradation <-> fiabilite.

A lancer EN LOCAL apres rapatriement du .npz du mesocentre :
    python plot_pareto.py [chemin/vers/dp_pareto_25y_51x51.npz]
(par defaut : results/dp_pareto_25y_51x51.npz)

Produit dans le meme dossier que le .npz :
    pareto_deg_eens.pdf/.png   frontiere (deg_kEUR vs EENS_kWh), points colores par eps,
                               + point RB2 de reference.
    pareto_unif_vs_eps.pdf/.png  cout unifie @VoLL=3 en fonction de eps (localise le coude).
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    npz = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "results", "dp_pareto_25y_51x51.npz")
    d = np.load(npz)
    out_dir = os.path.dirname(os.path.abspath(npz))

    eps  = d['eps']; deg = d['deg_keur']; eens = d['eens_kwh']
    lpsp = d['lpsp']; unif3 = d['unif3_keur']
    rb_deg = float(d['RB2_deg_keur']); rb_eens = float(d['RB2_eens_kwh'])
    rb_unif = float(d['RB2_unif3_keur'])

    order = np.argsort(eps)
    eps, deg, eens, lpsp, unif3 = (a[order] for a in (eps, deg, eens, lpsp, unif3))

    # --- 1) Frontiere de Pareto deg <-> EENS --------------------------------
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    ax.plot(eens, deg, '-', color='0.6', zorder=1)
    sc = ax.scatter(eens, deg, c=np.log10(eps), cmap='viridis', s=55, zorder=2,
                    edgecolor='k', linewidth=0.4)
    for x, y, e in zip(eens, deg, eps):
        ax.annotate(f"{e:g}", (x, y), textcoords="offset points", xytext=(5, 4),
                    fontsize=7, color='0.3')
    ax.scatter([rb_eens], [rb_deg], marker='*', s=240, color='crimson',
               edgecolor='k', linewidth=0.6, zorder=3, label='RB2 (reference)')
    ax.set_xlabel("Energie non servie  EENS  [kWh / 25 ans]")
    ax.set_ylabel("Cout de degradation  [kEUR / 25 ans]")
    ax.set_title("Frontiere de Pareto  degradation <-> fiabilite  (EMS optimal PD)")
    cb = fig.colorbar(sc, ax=ax); cb.set_label(r"$\epsilon$ = poids fiabilite (VoLL) [EUR/kWh], echelle log")
    ax.grid(alpha=.3); ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "pareto_deg_eens.pdf"))
    fig.savefig(os.path.join(out_dir, "pareto_deg_eens.png"), dpi=120)

    # --- 2) Cout unifie @VoLL=3 vs eps (localise le coude / l'optimum) -------
    fig2, ax2 = plt.subplots(figsize=(7.2, 4.4))
    ax2.semilogx(eps, unif3, 'o-', color='#1f77b4', label='PD (par eps)')
    ax2.axhline(rb_unif, ls='--', color='crimson', label=f'RB2 = {rb_unif:.1f} kEUR')
    i_best = int(np.argmin(unif3))
    ax2.scatter([eps[i_best]], [unif3[i_best]], s=120, color='green', zorder=3,
                label=f'min @ eps={eps[i_best]:g} ({unif3[i_best]:.1f} kEUR)')
    ax2.set_xlabel(r"$\epsilon$ = poids fiabilite dans la resolution PD [EUR/kWh]")
    ax2.set_ylabel("Cout unifie @ VoLL=3  [kEUR / 25 ans]")
    ax2.set_title("Cout unifie (reference VoLL=3) selon le poids de resolution")
    ax2.grid(alpha=.3, which='both'); ax2.legend()
    fig2.tight_layout()
    fig2.savefig(os.path.join(out_dir, "pareto_unif_vs_eps.pdf"))
    fig2.savefig(os.path.join(out_dir, "pareto_unif_vs_eps.png"), dpi=120)

    print("Figures ecrites dans", out_dir)
    print(f"  Pareto deg<->EENS : pareto_deg_eens.{{pdf,png}}")
    print(f"  Unifie vs eps     : pareto_unif_vs_eps.{{pdf,png}}")
    print(f"  Meilleur eps (min unif@3) = {eps[i_best]:g}  -> {unif3[i_best]:.2f} kEUR "
          f"(RB2 {rb_unif:.2f}, gain {100*(rb_unif-unif3[i_best])/rb_unif:+.1f}%)")


if __name__ == "__main__":
    main()
