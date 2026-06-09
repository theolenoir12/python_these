import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

def create_refined_storage_plot():
    # Configuration globale pour une publication (Police augmentée)
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 14,
        "axes.labelweight": "bold",
        "axes.titlesize": 16
    })

    fig, ax = plt.subplots(figsize=(14, 9), dpi=100)
    ax.set_xscale('log')
    ax.set_yscale('log')
    
    # Limites du graphique [cite: 3, 33, 35, 39]
    x_min, x_max = 0.1, 1e8  # 0.1 kWh à 100 GWh
    y_min, y_max = 1, 1e6    # 1 kW à 1 GW

    # Données techniques recalibrées sur le document [cite: 1-41]
    # [P_min, P_max, t_min (h), t_max (h), couleur, label complet]
    tech_data = {
        'DLC':    [100, 1000, 0.5/3600, 5/3600, '#FDFD96', 'Double Layer Capacitor'], # [cite: 7, 25]
        'FES':    [10, 10000, 10/3600, 0.25, '#4682B4', 'Flywheel Energy Storage'], # [cite: 14, 25]
        'LA':     [1, 100, 1, 10, '#AEC6CF', 'Lead Acid Battery'], # [cite: 27, 32]
        'Li-ion': [1000, 100000, 0.25, 4, '#03C03C', 'Li Ion Battery'], # [cite: 10, 28]
        'NaS':    [500, 50000, 2, 10, '#FFF44F', 'Sodium Sulphur Battery'], # [cite: 11, 13, 29]
        'PHS':    [100000, 1000000, 4, 24, '#779ECB', 'Pumped Hydro Storage'], # 
        'H2':     [1000, 500000, 24, 720, '#FF6961', 'Hydrogen Storage'], # [cite: 21, 26]
        'SNG':    [1000, 100000, 720, 8760, '#90EE90', 'Synthetic Natural Gas'], # [cite: 22, 34]
        'CAES':   [50000, 500000, 2, 30, '#FF91A4', 'Compressed Air ES'], # [cite: 20, 24]

    }

    # Dessin des parallélogrammes basés sur le temps de décharge 
    for abbr, val in tech_data.items():
        p_min, p_max, t_min, t_max, col, full = val
        # Coordonnées (E, P) où E = P * t
        pts = np.array([
            [p_min * t_min, p_min], [p_max * t_min, p_max],
            [p_max * t_max, p_max], [p_min * t_max, p_min]
        ])
        poly = patches.Polygon(pts, closed=True, facecolor=col, alpha=0.85, 
                               edgecolor='black', linewidth=1.2, label=f"{abbr}: {full}")
        ax.add_patch(poly)
        
        # Label interne (centrage log)
        tx = np.sqrt(pts[0,0] * pts[2,0])
        ty = np.sqrt(pts[0,1] * pts[2,1])
        if abbr == 'DLC' :
            ax.text(tx*1.5, ty*1.3, abbr, ha='center', va='center', fontsize=16, fontweight='bold')
        elif abbr == 'PHS' :
            ax.text(tx*2.0, ty*2.0, abbr, ha='center', va='center', fontsize=16, fontweight='bold')
        else :
            ax.text(tx, ty, abbr, ha='center', va='center', fontsize=16, fontweight='bold')

    ax2 = ax.twiny()
    ax2.set_xscale('log')
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xlabel("Discharge time", labelpad=25,fontsize='18') # Label au-dessus 
    ax2.set_xticks([]) # On cache les ticks auto car ils représenteraient l'Energie

    # Isoclines de temps de décharge [cite: 2, 9, 18, 19, 41]
    times = [(1/3600, '1 sec'), (1/60, '1 min'), (1, '1 h'), (24, '1 d'), (720, '1 m')]
    for t_h, name in times:
        ax.plot([x_min, x_max], [x_min/t_h, x_max/t_h], color='black', ls='--', lw=1, alpha=0.3)
        # Positionnement des labels temporels aux limites du cadre
        if y_max * t_h < x_max:
            ax.text(y_max * t_h, y_max * 1.05, name, ha='center', va='bottom', fontsize='16', fontweight='normal', color='#444')
        else:
            ax.text(x_max * 1.05, x_max / t_h, name, va='center', ha='left', fontsize='16', fontweight='normal', color='#444')

    # Configuration des axes
    ax.set_xlabel('Energy',fontsize='18')
    ax.set_ylabel('Rated Power',fontsize='18')
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.tick_params(axis='both',labelsize=16)


    # Graduations [cite: 3, 4, 33, 35-39]
    ax.set_xticks([0.1, 1, 10, 100, 1000, 1e4, 1e5, 1e6, 1e7, 1e8])
    ax.set_xticklabels(['0.1 kWh', '1 kWh', '10 kWh', '100 kWh', '1 MWh', '10 MWh', '100 MWh', '1 GWh', '10 GWh', '100 GWh'], rotation=35)
    ax.set_yticks([1, 10, 100, 1000, 1e4, 1e5, 1e6])
    ax.set_yticklabels(['1 kW', '10 kW', '100 kW', '1 MW', '10 MW', '100 MW', '1 GW'])
    ax.grid(True, which="both", alpha=0.15)

    # Légende étirée verticalement et rapprochée du plot
    ax.legend(
            loc='upper left', 
            bbox_to_anchor=(1.07, 1.0), # Positionnée pile au bord haut-droit
            prop={'size': 18, 'weight': 'normal'}, 
            frameon=True,
            labelspacing=2.2, 
            handletextpad=1.0,
            borderpad=1.5,
            borderaxespad=0.
        )

    plt.subplots_adjust(right=0.72, left=0.08) # Ajustement de la marge droite pour la légende
    plt.savefig('ragone.pdf', format='pdf', bbox_inches='tight', dpi=300)
    plt.show()

create_refined_storage_plot()