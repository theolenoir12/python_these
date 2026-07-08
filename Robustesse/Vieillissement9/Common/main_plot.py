import numpy as np
import matplotlib.pyplot as plt
import os
from time import time as timer
from .get_rul import get_rul
from .cost_fcn_total2 import get_cost_total
from .physics import *
from .Init_EMR_MG_v16_python import *
import matplotlib.gridspec as gs

def run_main_plot(data, start_timer=0, strategy_name=None):
    # --- Configuration matplotlib ---
    plt.rcParams.update({
        "text.usetex": False,
        "mathtext.fontset": "cm",
        "font.family": "serif",
        "axes.labelsize": 18,
        "axes.titlesize": 20,
        "legend.fontsize": 15,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "lines.linewidth": 1.8,
        "grid.alpha": 0.7,
        "grid.linestyle": "--"
    })
    
    # --- 1. Préparation des données ---
    temps = data["temps"]
    n = len(temps)
    
    stride = 10 
    
    # Extraction et alignement des tailles (n)
    SoC = data["SoC"]
    E_h2 = data["E_h2"]
    P_bat = data["P_bat"]
    P_fc = data["P_fc"]
    P_ely = data["P_ely"]
    P_dc_load = data["P_dc_load"]
    P_dc_pv = data["P_dc_pv"]
    P_dc_bat = data["P_dc_bat"]
    P_dc_fc = data["P_dc_fc"]
    P_dc_ely = data["P_dc_ely"]
    lol_tab = data["lol_tab"]
    
    # Trimming des vecteurs d'état (qui font n+1)
    alpha_fc = data["alpha_fc"][:-1]
    alpha_ely = data["alpha_ely"][:-1]
    SoH_bat = data["SoH_bat"][:-1]
    SoH_fc = data["SoH_fc"][:-1]
    SoH_ely = data["SoH_ely"][:-1]
    deg_fc = data["deg_fc"]
    deg_ely = data["deg_ely"]

    i_fc_max = (-194.3950 * alpha_fc + 196.5598)
    P_fc_max = i_fc_max * FC['n_parallel'] * FC['n_series'] * (FC['E_0'] - FC['R'] * (1 + alpha_fc) * i_fc_max / FC['n_parallel'] 
                                            - A * FC['T'] * np.log((i_fc_max / S / FC['n_parallel'] + j_in) / FC['j_0'])
                                            - B * FC['T'] * np.log(1 - i_fc_max / S / FC['n_parallel'] / FC['j_L'] / (1 - alpha_fc)))
    i_ely_max = (-219.9 * alpha_ely + 219.9)
    P_ely_max = i_ely_max * ELY['n_parallel'] * ELY['n_series'] * (ELY['E_0'] + ELY['R'] * (1 + alpha_ely) * i_ely_max / ELY['n_parallel'] 
                                               + A * ELY['T'] * np.log((i_ely_max / S / ELY['n_parallel'] + j_in) / ELY['j_0'])
                                               + B * ELY['T'] * np.log(1 - i_ely_max / S / ELY['n_parallel'] / ELY['j_L'] / (1 - alpha_ely)))
    
    eff_fc  = np.interp((P_fc / CONV['eta'] / P_fc_max) * 100, *FC['lut']) / 100
    eff_ely = np.interp((P_ely * CONV['eta'] / -P_ely_max) * 100, *ELY['lut']) / 100

    # Métriques LoL / LoP
    lol_tab2 = lol_tab * ((np.array(P_dc_load) - np.array(P_dc_pv)) > 0)
    lop_tab2 = lol_tab * ((np.array(P_dc_load) - np.array(P_dc_pv)) < 0)

    for k in range(1,len(SoH_bat)) : 
        if SoH_bat[k] == 1 :
            SoH_bat[k-1] = np.nan
        if SoH_fc[k] == 1 : 
            SoH_fc[k-1] = np.nan
        if SoH_ely[k] == 1 :
            SoH_ely[k-1] = np.nan
        if deg_fc['total'][k] < 5e-3 and deg_fc['total'][k-1] > 1 :
            deg_fc['start-stop'][k-1]   = np.nan
            deg_fc['idling'][k-1]       = np.nan
            deg_fc['reversible'][k-1]   = np.nan
            deg_fc['irreversible'][k-1] = np.nan
            deg_fc['total'][k-1]        = np.nan
        if deg_ely['total'][k] < 5e-3 and deg_ely['total'][k-1] > 1 :
            deg_ely['start-stop'][k-1]    = np.nan
            deg_ely['maintaining'][k-1]   = np.nan
            deg_ely['reversible'][k-1]    = np.nan
            deg_ely['irreversible'][k-1]  = np.nan
            deg_ely['total'][k-1]         = np.nan
            
    # --- GESTION DU DOSSIER DE SAUVEGARDE (La modification est ici) ---
    if strategy_name:
        # Cas Batch : Figures/Nom_Strategie/8760h/
        savedir = os.path.join(strategy_name, "Figures", f"{n}h")    
    else:
        # Cas Local : Figures/8760h/
        savedir = os.path.join("Figures", f"{n}h")

    if not os.path.exists(savedir):
        os.makedirs(savedir, exist_ok=True)
    
    # --- Fonctions pour la création des graphiques ---
    def plot_and_save(x, y, filename, ylabel, title):
        plt.figure(figsize=(10,6))
        plt.plot(x/3600/24, y, lw=3)
        plt.title(title)
        plt.xlabel(r'$\mathbf{Time\ (days)}$')
        plt.ylabel(ylabel)
        plt.grid(True)
        ax = plt.gca()
        ax.tick_params(labelsize=15)
        plt.savefig(os.path.join(savedir, filename))
        plt.close() # Important pour ne pas saturer la RAM en batch
    
    # --- Figures de SoH ---
    def plot_SoH(temps, SoH, SoH_EoL, filename, title):
        plt.figure(figsize=(10,6))
        plt.plot(temps/3600/24, SoH, lw=3, label=r'$SoH$')
        plt.plot(temps/3600/24, SoH_EoL*np.ones(len(temps)), 'r--', label=r'$EoL\ criteria$')
        plt.title(title)
        plt.xlabel(r'$\mathbf{Time\ (days)}$')
        plt.ylabel(r'$\mathbf{{{SoH}}}$')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(savedir, filename), format='pdf', bbox_inches='tight')
        plt.close()
    
    plot_SoH(temps, SoH_bat, BAT['SoH_EoL'], 'SoH_bat.pdf', title=r'$\mathbf{SoH\ of\ the\ battery}$')
    plot_SoH(temps, SoH_fc, FC['SoH_EoL'], 'SoH_fc.pdf', title=r'$\mathbf{SoH\ of\ the\ fuel\ cell}$')
    plot_SoH(temps, SoH_ely, ELY['SoH_EoL'], 'SoH_ely.pdf', title=r'$\mathbf{SoH\ of\ the\ electrolyzer}$')
    
    # --- Figures de dégradations ---
    def plot_deg(temps, x1,x2,x3,x4,x5, filename, title,
                 labels=(r'$Start-stop$', r'$Idling$', r'$Transient$', r'$High\ power$')):
        plt.figure(figsize=(10,6))
        plt.plot(temps/3600/24, x1, lw=3, label=r'$Total$')
        plt.plot(temps/3600/24, x2, lw=3, label=labels[0])
        plt.plot(temps/3600/24, x3, lw=3, label=labels[1])
        plt.plot(temps/3600/24, x4, lw=3, label=labels[2])
        plt.plot(temps/3600/24, x5, lw=3, label=labels[3])
        plt.title(title)
        plt.xlabel(r'$\mathbf{Time\ (days)}$')
        plt.ylabel(r'$\mathbf{{{Degradations}}}$')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(savedir, filename), format='pdf', bbox_inches='tight')
        plt.close()
        
    plot_deg(temps,deg_fc['total'],deg_fc['start-stop'],deg_fc['idling'],deg_fc['reversible'],deg_fc['irreversible'],'deg_fc.pdf', title=r'$\mathbf{Degradations\ of\ the\ fuel\ cell}$',
             labels=(r'$Start-stop$', r'$Idling$', r'$Reversible$', r'$Irreversible$'))
    plot_deg(temps,deg_ely['total'],deg_ely['start-stop'],deg_ely['maintaining'],deg_ely['reversible'],deg_ely['irreversible'],'deg_ely.pdf', title=r'$\mathbf{Degradations\ of\ the\ electrolyzer}$',
             labels=(r'$Start-stop$', r'$Maintaining$', r'$Reversible$', r'$Irreversible$'))
            
    def plot_degradation_pie_charts(deg_fc, deg_ely, target_dir):
        def get_max_degradation_values(deg_dict, keys):
            total = np.array(deg_dict['total'])
            idx = np.argmax(total)
            vals = []
            for k in keys:
                vals.append(deg_dict[k][idx-1])
            return np.array(vals)
    
        keys_fc = ['start-stop', 'idling', 'reversible', 'irreversible']
        labels_fc = ['Start-stop', 'Idling', 'Reversible', 'Irreversible']
        values_fc = get_max_degradation_values(deg_fc, keys_fc)
    
        keys_ely = ['start-stop', 'maintaining', 'reversible', 'irreversible']
        labels_ely = ['Start-stop', 'Maintaining', 'Reversible', 'Irreversible']
        values_ely = get_max_degradation_values(deg_ely, keys_ely)
    
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        colors = ['#ff9999','#66b3ff','#99ff99','#ffcc99']
    
        def draw_pie(ax, values, labels, title):
            mask = values > 0
            v_filtered = values[mask]
            l_filtered = [l for i, l in enumerate(labels) if mask[i]]
            c_filtered = [colors[i] for i, color in enumerate(colors) if mask[i]]
            if len(v_filtered) > 0:
                ax.pie(v_filtered, labels=l_filtered, autopct='%1.1f%%', startangle=90, 
                       colors=c_filtered, pctdistance=0.75, explode=[0.05] * len(v_filtered),
                       textprops={'fontsize': 18, 'weight': 'bold'})
                for autotext in ax.texts: autotext.set_fontsize(20)
            ax.set_title(title, pad=30, fontsize=26, weight='bold')
    
        draw_pie(ax1, values_fc, labels_fc, r'$\mathbf{PEMFC\ Degradation}$')
        draw_pie(ax2, values_ely, labels_ely, r'$\mathbf{PEMWE\ Degradation}$')
        plt.tight_layout()
        plt.savefig(os.path.join(target_dir, 'degradation_pie_charts_final.pdf'), format='pdf', bbox_inches='tight')
        plt.close()
    
    plot_degradation_pie_charts(deg_fc, deg_ely, savedir)
            
    def plot_scientific_summary(temps, soh_d, deg_d, sdir):
        # --- Configuration Ultra-Lisible ---
        plt.rcParams.update({
            "font.family":"serif", "font.size":28, "axes.titlesize":36, 
            "axes.labelsize":34, "xtick.labelsize":30, "ytick.labelsize":30, "legend.fontsize":26
        })
        
        f = plt.figure(figsize=(40, 15))
        # hspace à 0.3 pour resserrer le tout verticalement
        g = gs.GridSpec(2, 6, figure=f, hspace=0.4, wspace=0.7, height_ratios=[1, 1.3])
        t_years, colors = temps/(3600*24*365), {'bat': '#1f77b4', 'fc': '#d62728', 'ely': '#2ca02c'}
        leg_locs = {'bat': 'upper right', 'fc': 'center left', 'ely': 'center left'}
        titles = {'bat': r'\mathbf{Battery}', 'fc': r'\mathbf{PEMFC}', 'ely': r'\mathbf{PEMWE}'}
    
        # --- LIGNE 1 : SoH (0-100% avec effet Luxe) ---
        for i, k in enumerate(['bat', 'fc', 'ely']):
            ax = f.add_subplot(g[0, i*2:(i+1)*2])
            
            # 1. Extraction des données stridées
            t_plot = t_years[::stride]
            v_plot = (soh_d[k]['val'] * 100)[::stride].copy() # .copy() est crucial pour ne pas modifier l'original
            e_pct = soh_d[k]['eol'] * 100
            
            # 2. Détection des sauts (remplacements)
            # Si le SoH augmente de plus de 5% entre deux points, c'est un remplacement
            diff = np.diff(v_plot)
            jump_indices = np.where(diff > 5)[0]
            
            # 3. Insertion de NaNs pour casser la ligne
            # On insère un point NaN juste après chaque indice de saut
            t_plot_clean = np.insert(t_plot.astype(float), jump_indices + 1, np.nan)
            v_plot_clean = np.insert(v_plot.astype(float), jump_indices + 1, np.nan)
            
            # 4. Tracé de la ligne (propre désormais)
            ax.plot(t_plot_clean, v_plot_clean, lw=6, color=colors[k], label=r'$SoH$', zorder=3)
            
            # Le fill_between peut rester sur les données originales stridées 
            # car il gère généralement mieux les zones discontinues
            ax.fill_between(t_plot, v_plot, e_pct, color=colors[k], alpha=0.15, zorder=2)
            
            ax.axhline(y=e_pct, color='r', ls='--', lw=3.5, label=r'$EoL\ limit$', zorder=4)
            
            ax.set_title(rf'${titles[k]}$', pad=25)
            ax.set_xlabel(r'$\mathbf{Time\ (years)}$')
            if i == 0: ax.set_ylabel(r'$SoH\ [\%]$')
            
            # Graduations forcées et nettes
            ax.set_yticks([int(e_pct), 100])
            ax.grid(True, alpha=0.3, ls=':')
            ax.legend(loc=leg_locs[k], frameon=True, framealpha=0.9, prop={'size': 26})
    
        # Ligne de séparation discrète
        f.add_artist(plt.Line2D([0.05, 0.95], [0.525, 0.525], color='black', lw=2, alpha=0.3, transform=f.transFigure))
    
        # --- LIGNE 2 : Pie Charts (Titres abaissés) ---
        cf_pie = [
            (deg_d['fc'], ['start-stop','idling','reversible','irreversible'], ['Start-stop','Idling','Reversible','Irreversible'], r'\mathbf{PEMFC\ Degradation}', g[1, 0:3]),
            (deg_d['ely'], ['start-stop','maintaining','reversible','irreversible'], ['Start-stop','Maintaining','Reversible','Irreversible'], r'\mathbf{PEMWE\ Degradation}', g[1, 3:6])
        ]
        
        colors_p = ['#ff9999','#66b3ff','#99ff99','#ffcc99']
        for d, k, l, t, pos in cf_pie:
            ax = f.add_subplot(pos)
            v = np.array([d[key][np.argmax(d['total'])-1] for key in k])
            m = v > 0
            if any(m):
                wedges, texts, autotexts = ax.pie(
                    v[m], autopct='%1.1f%%', startangle=140,
                    colors=np.array(colors_p)[m], pctdistance=0.75,
                    explode=[0.08]*sum(m))
                plt.setp(autotexts, fontsize=26, weight="bold")
                ax.legend(wedges, np.array(l)[m], loc='center left',
                          bbox_to_anchor=(0.95, 0.5), prop={'size': 26}, frameon=False)
            ax.set_title(rf'${t}$', pad=10)
    
        plt.savefig(os.path.join(sdir, 'all_aging_2.pdf'), format='pdf', bbox_inches='tight')
        plt.show()
    
    # Dictionnaires et appel
    soh_in = {
        'bat':{'val':SoH_bat,'eol':BAT['SoH_EoL']}, 
        'fc':{'val':SoH_fc, 'eol':FC['SoH_EoL']}, 
        'ely':{'val':SoH_ely,'eol':ELY['SoH_EoL']}
    }
    plot_scientific_summary(temps, soh_in, {'fc':deg_fc, 'ely':deg_ely}, savedir)
    
    plt.rcParams.update({
        "text.usetex": False,
        "mathtext.fontset": "cm",
        "font.family": "serif",
        "axes.labelsize": 18,
        "axes.titlesize": 20,
        "legend.fontsize": 15,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "lines.linewidth": 1.8,
        "grid.alpha": 0.7,
        "grid.linestyle": "--"
    })
    
    
    # --- Figures de RUL ---
    RUL_bat, true_RUL_bat = get_rul(SoH_bat,BAT['SoH_EoL']) 
    RUL_fc, true_RUL_fc   = get_rul(SoH_fc,FC['SoH_EoL']) 
    RUL_ely, true_RUL_ely = get_rul(SoH_ely,ELY['SoH_EoL'])
    
    def plot_RUL(temps, RUL_est, RUL_true, filename, title):
        plt.figure(figsize=(10,6))
        plt.plot(temps/3600/24, RUL_est, lw=3, label='Estimated')
        plt.plot(temps/3600/24, RUL_true, 'r--', label='True')
        plt.title(title)
        plt.xlabel(r'$\mathbf{Time\ (days)}$')
        plt.ylabel(r'$\mathbf{{{RUL}}}$')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(savedir, filename), format='pdf', bbox_inches='tight')
        plt.close()
    
    plot_RUL(temps, RUL_bat, true_RUL_bat, 'rul_bat.pdf', title=r'$\mathbf{RUL\ of\ the\ battery}$')
    plot_RUL(temps, RUL_fc, true_RUL_fc, 'rul_fc.pdf', title=r'$\mathbf{RUL\ of\ the\ fuel\ cell}$')
    plot_RUL(temps, RUL_ely, true_RUL_ely, 'rul_ely.pdf', title=r'$\mathbf{RUL\ of\ the\ electrolyzer}$')
        
    # --- Sous-figures multiples ---

    def plot_subplot(ax, x, y, color='b', label=None, title=None, ylabel=None):
        ax.margins(x=0.01)
        ax.plot(x, y, f'{color}.-', label=label,rasterized=True)
        if title: ax.set_title(title, fontsize=30)
        if ylabel: ax.set_ylabel(rf'${ylabel}$', fontsize=26)
        if label: ax.legend()
        ax.tick_params(labelsize=20)
        ax.grid(True)
        
    fig, axs = plt.subplots(7, 1, figsize=(15,12))
    P_planned = [(a-b)/1000 for a,b in zip(P_dc_load,P_dc_pv)]
    P_real    = [(a-b)*(1-c)/1000 for a,b,c in zip(P_dc_load,P_dc_pv,lol_tab)]
    
    plot_subplot(axs[0], temps/3600/24, P_planned, 'b', label='Planned', title=r'${P_{bus}}$', ylabel='P_{bus} (kW)')
    axs[0].plot(temps/3600/24, P_real, 'y.-', label='Real')
    plot_subplot(axs[1], temps/3600/24, P_dc_bat/1000, 'b', title=r'${P_{dc-bat}}$', ylabel='P_{dc-bat} (kW)')
    plot_subplot(axs[2], temps/3600/24, P_dc_fc/1000, 'r', title=r'${P_{dc-fc}}$', ylabel='P_{dc-fc} (kW)')
    plot_subplot(axs[3], temps/3600/24, P_dc_ely/1000, 'g', title=r'${P_{dc-ely}}$', ylabel='P_{dc-ely} (kW)')
    plot_subplot(axs[4], temps/3600/24, SoC[0:-1], 'b', title=r'${SoC_{bat}}$', ylabel='SoC_{bat}')
    plot_subplot(axs[5], temps/3600/24, lol_tab2, 'r', title=r'${LoL}$', ylabel='LoL')
    plot_subplot(axs[6], temps/3600/24, lop_tab2, 'g', title=r'${LoP}$', ylabel='LoP')
    axs[6].set_xlabel(r'$\mathbf{Time\ (days)}$')
    plt.tight_layout()
    plt.savefig(os.path.join(savedir, 'everything.pdf'), format='pdf', bbox_inches='tight')
    plt.close()

    # --- Grande Figure Combinée avec Zoom ---
    t_days = np.array(temps) / 3600 / 24
    P_planned = np.array(P_planned)
    P_real    = np.array(P_real)
    P_bat_k   = np.array(P_dc_bat)/1000
    P_fc_k    = np.array(P_dc_fc)/1000
    P_ely_k   = np.array(P_dc_ely)/1000
    SoC_p     = np.array(SoC[0:-1])*100
    E_h2_k    = np.array(E_h2[0:-1])
    LPS_p     = np.array(lol_tab2)*100
    
    zoom_start, zoom_end = 316, 323
    mask = (t_days >= zoom_start) & (t_days <= zoom_end)
    
    fig, axs = plt.subplots(7, 2, figsize=(41, 18), sharex='col')
    plt.subplots_adjust(wspace=0.1, hspace=0.4)
        
    def plot_row(row_idx, x, y, color='b', label=None, title=None, ylabel=None, y2=None, label2=None, color2='y', ymax_custom=None,annots=False, zoom_ploss=False):
        plt.rcParams.update({
            "text.usetex": False, "mathtext.fontset": "cm", "font.family": "serif",
            "axes.labelsize": 20, "axes.titlesize": 24, "legend.fontsize": 17,
            "xtick.labelsize": 18, "ytick.labelsize": 18, "lines.linewidth": 1.8,
            "grid.alpha": 0.7, "grid.linestyle": "--"
        })
        
        ax_left, ax_right = axs[row_idx, 0], axs[row_idx, 1]
        ax_left.plot(x[::stride]/365, y[::stride], color=color, label=label)
        if y2 is not None: 
            ax_left.plot(x[::stride]/365, y2[::stride], color=color2, linestyle='-', label=label2)
        
        ax_left.margins(x=0.01)
        if ylabel: ax_left.set_ylabel(rf'${ylabel}$', fontsize=28)
        if label: ax_left.legend(loc='upper right', fontsize=22)
        ax_left.grid(True)
        ax_left.tick_params(axis='both', labelsize=21)
        ymin, ymax = ax_left.get_ylim()
        if annots:
            for idx, off_x, off_y in annots:
                val = y[idx]
                # bbox_props avec alpha=1 pour masquer la grille derrière l'encadré
                bbox_p = dict(boxstyle="round,pad=0.3", fc="white", ec=color, lw=1.5, alpha=1.0)
                
                ax_left.annotate(f'{val:.2f}', 
                                 xy=(x[idx]/365, val), 
                                 xytext=(off_x, off_y),
                                 textcoords='offset points', 
                                 fontsize=23, fontweight='bold', color=color,
                                 bbox=bbox_p, 
                                 arrowprops=dict(arrowstyle="->", color='k', lw=1.8),
                                 ha='center', va='center')
        if ymax_custom is not None:
            ax_left.set_ylim(top=ymax_custom)
        ax_right.plot(x[mask], y[mask], color=color, marker='.', linestyle='-')
        if y2 is not None: 
            ax_right.plot(x[mask], y2[mask], color=color2, marker='.', linestyle='-')
        ax_right.grid(True)
        ax_right.set_xlim(zoom_start, zoom_end)
        
        if title: 
            ax_left.text(1.0, 1.02, title, transform=ax_left.transAxes, ha='center', va='bottom', fontsize=32)
        
        ymin, ymax = ax_left.get_ylim()
        ax_right.set_ylim(ymin, ymax)
        ax_right.tick_params(axis='both', labelleft=True, labelsize=21)

        # --- Annotation de la perte de puissance sur le zoom : puissance crete au 1er vs
        #     au dernier jour de la semaine, avec plus de decimales (variation faible) ---
        if zoom_ploss:
            idxs = np.flatnonzero(mask)
            if len(idxs):
                first = idxs[x[idxs] <= zoom_start + 1]
                last  = idxs[x[idxs] >= zoom_end   - 1]
                for seg, off_x, off_y in ((first, 55, 55), (last, -55, 55)):
                    if len(seg):
                        ii = seg[np.argmin(y[seg])]   # pic de puissance (plus negatif)
                        val = y[ii]
                        ax_right.annotate(f'{val:.4f}',
                                          xy=(x[ii], val),
                                          xytext=(off_x, off_y), textcoords='offset points',
                                          fontsize=23, fontweight='bold', color=color,
                                          bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=color, lw=1.5, alpha=1.0),
                                          arrowprops=dict(arrowstyle="->", color='k', lw=1.8),
                                          ha='center', va='center')

    # Appels avec les nouvelles limites
    plot_row(0, t_days, P_planned, 'b', label='Planned', title=r'$\mathbf{Power\ demand}$', ylabel=r'P_{\mathrm{bus}} [kW]', y2=P_real, label2='Real')
    plot_row(1, t_days, P_bat_k, 'b', title=r'$\mathbf{Battery\ power}$', ylabel=r'P_{\mathrm{bat}} [kW]')
    fc_replacements  = np.where((SoH_fc[1:]  == 1) & (SoH_fc[:-1]  != 1))[0]
    ely_replacements = np.where((SoH_ely[1:] == 1) & (SoH_ely[:-1] != 1))[0]
    def last_nonzero_before(replacements, power):
        if not len(replacements): return -1
        for k in range(int(replacements[-1]), -1, -1):
            if power[k] != 0: return k
        return -1
    idx_fc_last  = last_nonzero_before(fc_replacements,  P_fc_k)
    idx_ely_last = last_nonzero_before(ely_replacements, P_ely_k)
    plot_row(2, t_days, P_fc_k, 'r', title=r'$\mathbf{PEMFC\ power}$', ylabel=r'P_{\mathrm{FC}} [kW]')
    plot_row(3, t_days, P_ely_k, 'g', title=r'$\mathbf{PEMWE\ power}$', ylabel=r'P_{\mathrm{ELY}} [kW]')#, ymax_custom=1, annots=[(8, 60, 40), (idx_ely_last, -60, 40)], zoom_ploss=True)
    plot_row(4, t_days, SoC_p, 'b', title=r'$\mathbf{Battery\ state\ of\ charge}$', ylabel=r'SoC_{\mathrm{bat}} [\%]', ymax_custom=110)
    plot_row(5, t_days, E_h2_k, 'g', title=r'$\mathbf{Hydrogen\ energy\ stored}$', ylabel=r'E_{H2} [kWh]', ymax_custom=220)
    plot_row(6, t_days, LPS_p, 'r', title=r'$\mathbf{Loss\ of\ power\ supply}$', ylabel=r'LPS [\%]', ymax_custom=110)
    
    axs[6, 0].set_xlabel(r'$\mathbf{Time\ (years)}$', fontsize=26)
    axs[6, 1].set_xlabel(r'$\mathbf{Time\ (days)}$', fontsize=26)
    
    plt.savefig(os.path.join(savedir, 'everything_combined_v2_2.pdf'), format='pdf', bbox_inches='tight')
    plt.show()

    # --- Calculs finaux ---
    def get_LPSP(P_planned, P_real):
        p, r = np.clip(P_planned, 0, None), np.clip(P_real, 0, None)
        total_p = p.sum()
        return (np.clip(p-r, 0, None).sum() / total_p * 100) if total_p > 0 else 0.0
    
    print("LPSP :",get_LPSP(P_planned,P_real),'(%)')
    SoH_bat[np.isnan(SoH_bat)] = np.interp(
        np.flatnonzero(np.isnan(SoH_bat)), 
        np.flatnonzero(~np.isnan(SoH_bat)), 
        SoH_bat[~np.isnan(SoH_bat)])
    deg_eur = get_cost_total(alpha_fc, P_fc, alpha_ely, P_ely, P_bat, SoC, LOAD, BAT, FC, ELY, SoH_bat)
    print("Dégradations : ", deg_eur, '(EUR)')
    # --- Décomposition coût total de possession : BoP (installation, payé 1 fois) +
    #     dégradation actualisée (facteur d'annuité present-worth) = NPC ---
    _kwh_bat = BAT['series_num'] * BAT['parallel_num'] * BAT['Q_bat'] * BAT['v_cell_nom'] / 1000
    _bop = (BAT['CAPEX'] * _kwh_bat        - BAT['cost']) \
         + (FC['CAPEX']  * FC['P_fc_max']/1000  - FC['cost']) \
         + (ELY['CAPEX'] * ELY['P_ely_max']/1000 - ELY['cost'])
    _r = 0.05                                               # taux d'actualisation réel
    _N = len(temps) * LOAD['Ts'] / (3600 * 24 * 365)       # horizon (ans)
    _AF = 1.0 if _r <= 0 else (1 - (1 + _r) ** (-_N)) / (_r * _N)
    _npc = _bop + _AF * deg_eur
    print("  BoP (installation) : %.0f EUR  |  dégradation actualisée (r=%.0f%%, N=%.0f ans, AF=%.3f) : %.0f EUR"
          % (_bop, _r*100, _N, _AF, _AF * deg_eur))
    print("  NPC (coût total de possession) : %.0f EUR  =  %.1f k€" % (_npc, _npc / 1000))
    # --- Cout d'indisponibilite : VOLL (Value of Lost Load) x energie non fournie ---
    _Pp = np.clip(np.array(P_planned), 0, None); _Pr = np.clip(np.array(P_real), 0, None)
    _e_unserved = np.clip(_Pp - _Pr, 0, None).sum() * LOAD['Ts'] / 3600.0   # kWh sur l'horizon (P en kW)
    _VOLL = 5.0                                                              # EUR/kWh (socio-eco insulaire, a ajuster)
    _voll_cost = _AF * _VOLL * _e_unserved
    print("  Énergie non fournie : %.0f kWh  |  coût d'indisponibilité (VOLL=%.1f €/kWh, actualisé) : %.0f EUR"
          % (_e_unserved, _VOLL, _voll_cost))
    print("  COÛT TOTAL (NPC + indisponibilité) : %.0f EUR  =  %.1f k€" % (_npc + _voll_cost, (_npc + _voll_cost) / 1000))

    lpsp_percent = get_LPSP(P_planned, P_real)
    # Correction temporaire NaN pour le coût
    SoH_clean = np.copy(SoH_bat)
    if np.isnan(SoH_clean).any():
        idx = np.arange(len(SoH_clean))
        SoH_clean = np.interp(idx, idx[~np.isnan(SoH_clean)], SoH_clean[~np.isnan(SoH_clean)])
    
    cost_keur = get_cost_total(alpha_fc, P_fc, alpha_ely, P_ely, P_bat, SoC, LOAD, BAT, FC, ELY, SoH_clean) / 1000
        
   # --- Courant et tension en fonction du temps : PEMFC et PEMWE ---
    from scipy.optimize import brentq

    def solve_i_fc(P_target, alpha):
        if P_target <= 0:
            return 0.0
        i_max = -194.3950 * alpha + 196.5598
        def residual(i):
            V = FC['n_series'] * (FC['E_0'] - FC['R'] * (1 + alpha) * i / FC['n_parallel']
                - A * FC['T'] * np.log((i / S / FC['n_parallel'] + j_in) / FC['j_0'])
                - B * FC['T'] * np.log(1 - i / S / FC['n_parallel'] / FC['j_L'] / (1 - alpha)))
            return i * V - P_target
        try:
            return brentq(residual, 1e-6, i_max * 0.999)
        except Exception:
            return np.nan

    def solve_i_ely(P_target, alpha):
        if P_target >= 0:
            return 0.0
        P_abs = abs(P_target)
        i_max = -219.9 * alpha + 219.9
        def residual(i):
            V = ELY['n_series'] * (ELY['E_0'] + ELY['R'] * (1 + alpha) * i / ELY['n_parallel']
                + A * ELY['T'] * np.log((i / S / ELY['n_parallel'] + j_in) / ELY['j_0'])
                + B * ELY['T'] * np.log(1 - i / S / ELY['n_parallel'] / ELY['j_L'] / (1 - alpha)))
            return i * V - P_abs
        try:
            return brentq(residual, 1e-6, i_max * 0.999)
        except Exception:
            return np.nan

    i_fc_t  = np.array([solve_i_fc(p, a)  for p, a in zip(P_fc,  alpha_fc)])
    i_ely_t = np.array([solve_i_ely(p, a) for p, a in zip(P_ely, alpha_ely)])

    V_fc_t  = np.where(i_fc_t  > 1e-6, np.array(P_fc)          / (i_fc_t  + 1e-12), 0.0)
    V_ely_t = np.where(i_ely_t > 1e-6, np.abs(np.array(P_ely)) / (i_ely_t + 1e-12), 0.0)

    fig, (ax_i, ax_v) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    ax_i.plot(t_days[::stride], i_fc_t[::stride], 'r.-', markersize=1, lw=0.8)
    ax_i.set_ylabel(r'$i_{FC}\ \mathrm{[A]}$', fontsize=18)
    ax_i.set_title(r'$\mathbf{PEMFC\ —\ Current\ and\ Voltage\ vs\ Time}$', fontsize=20)
    ax_i.grid(True)
    ax_v.plot(t_days, V_fc_t, 'r.-', markersize=1, lw=0.8)
    ax_v.set_ylabel(r'$V_{FC}\ \mathrm{[V]}$', fontsize=18)
    ax_v.set_xlabel(r'$\mathbf{Time\ (days)}$', fontsize=16)
    ax_v.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(savedir, 'i_V_fc.pdf'), format='pdf', bbox_inches='tight')
    plt.close()

    fig, (ax_i, ax_v) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    ax_i.plot(t_days[::stride], i_ely_t[::stride], 'g.-', markersize=1, lw=0.8)
    ax_i.set_ylabel(r'$i_{ELY}\ \mathrm{[A]}$', fontsize=18)
    ax_i.set_title(r'$\mathbf{PEMWE\ —\ Current\ and\ Voltage\ vs\ Time}$', fontsize=20)
    ax_i.grid(True)
    ax_v.plot(t_days, V_ely_t, 'g.-', markersize=1, lw=0.8)
    ax_v.set_ylabel(r'$V_{ELY}\ \mathrm{[V]}$', fontsize=18)
    ax_v.set_xlabel(r'$\mathbf{Time\ (days)}$', fontsize=16)
    ax_v.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(savedir, 'i_V_ely.pdf'), format='pdf', bbox_inches='tight')
    plt.close()

    # --- Courbes de polarisation PEMFC (BoL, MoL, EoL) ---
    # alpha_fc varie de 0 (BoL) à sa valeur finale (EoL)
    alpha_fc_bol = 0.0
    alpha_fc_eol = float(np.nanmax(alpha_fc))
    alpha_fc_mol = (alpha_fc_bol + alpha_fc_eol) / 2.0

    # Vecteur courant FC : de 0 à i_fc_max(BoL) avec marge
    i_fc_bol_max = float(-194.3950 * alpha_fc_bol + 196.5598)
    i_fc_mol_max = float(-194.3950 * alpha_fc_mol + 196.5598)
    i_fc_eol_max = float(-194.3950 * alpha_fc_eol + 196.5598)
    i_fc_bol_sweep = np.linspace(0.01, i_fc_bol_max * 0.999, 500)
    i_fc_mol_sweep = np.linspace(0.01, i_fc_mol_max * 0.999, 500)
    i_fc_eol_sweep = np.linspace(0.01, i_fc_eol_max * 0.999, 500)

    def voltage_fc_sweep(alpha, i_vec):
        j_vec = i_vec / S / FC['n_parallel']
        return FC['n_series'] * (
            FC['E_0']
            - FC['R'] * (1 + alpha) * i_vec / FC['n_parallel']
            - A * FC['T'] * np.log((j_vec + j_in) / FC['j_0'])
            - B * FC['T'] * np.log(1 - j_vec / FC['j_L'] / (1 - alpha))
        )

    V_fc_bol = voltage_fc_sweep(alpha_fc_bol, i_fc_bol_sweep)
    V_fc_mol = voltage_fc_sweep(alpha_fc_mol, i_fc_mol_sweep)
    V_fc_eol = voltage_fc_sweep(alpha_fc_eol, i_fc_eol_sweep)

    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax2 = ax1.twinx()

    for i_fc_sweep, alpha, V, label, color in [
        (i_fc_bol_sweep, alpha_fc_bol, V_fc_bol, r'$BoL\ (\alpha=0)$',         '#1f77b4'),
        (i_fc_mol_sweep, alpha_fc_mol, V_fc_mol, rf'$MoL\ (\alpha={alpha_fc_mol:.3f})$', '#ff7f0e'),
        (i_fc_eol_sweep, alpha_fc_eol, V_fc_eol, rf'$EoL\ (\alpha={alpha_fc_eol:.3f})$', '#d62728'),
    ]:
        P_sweep = i_fc_sweep * V / 1000  # kW
        ax1.plot(i_fc_sweep, V,       color=color, lw=2.2,  label=label)
        ax2.plot(i_fc_sweep, P_sweep, color=color, lw=2.2,  linestyle='--')

    ax1.set_xlabel(r'$\mathbf{Current\ }i_{FC}\ \mathrm{[A]}$', fontsize=18)
    ax1.set_ylabel(r'$\mathbf{Voltage\ }V_{FC}\ \mathrm{[V]}$',  fontsize=18, color='k')
    ax2.set_ylabel(r'$\mathbf{Power\ }P_{FC}\ \mathrm{[kW]}$',   fontsize=18, color='gray')
    ax2.tick_params(axis='y', labelcolor='gray')
    ax1.set_title(r'$\mathbf{PEMFC\ polarization\ curve\ (BoL\ /\ MoL\ /\ EoL)}$', fontsize=20)
    ax1.legend(fontsize=14, loc='lower center')
    ax1.grid(True)
    # Ligne verticale indicative pour le courant nominal observé (~75% de i_max BoL)
    i_fc_nom_ref = 0.75 * i_fc_bol_max
    ax1.axvline(x=i_fc_nom_ref, color='k', linestyle=':', lw=1.5,
                label=rf'$i_{{nom,ref}} \approx {i_fc_nom_ref:.1f}\ A$')
    ax1.annotate(rf'$i_{{nom,ref}}$', xy=(i_fc_nom_ref, ax1.get_ylim()[0]),
                 xytext=(i_fc_nom_ref + i_fc_bol_max * 0.02, ax1.get_ylim()[0] + 5),
                 fontsize=13, color='k')
    plt.tight_layout()
    plt.savefig(os.path.join(savedir, 'polarization_fc.pdf'), format='pdf', bbox_inches='tight')
    plt.close()

    # --- Courbes de polarisation PEMWE (BoL, MoL, EoL) ---
    alpha_ely_bol = 0.0
    alpha_ely_eol = float(np.nanmax(alpha_ely))
    alpha_ely_mol = (alpha_ely_bol + alpha_ely_eol) / 2.0

    i_ely_bol_max   = float(-219.9 * alpha_ely_bol + 219.9)
    i_ely_mol_max   = float(-219.9 * alpha_ely_mol + 219.9)
    i_ely_eol_max   = float(-219.9 * alpha_ely_eol + 219.9)
    i_ely_bol_sweep = np.linspace(0.01, i_ely_bol_max * 0.999, 500)
    i_ely_mol_sweep = np.linspace(0.01, i_ely_mol_max * 0.999, 500)
    i_ely_eol_sweep = np.linspace(0.01, i_ely_eol_max * 0.999, 500)

    def voltage_ely_sweep(alpha, i_vec):
        j_vec = i_vec / S / ELY['n_parallel']
        return ELY['n_series'] * (
            ELY['E_0']
            + ELY['R'] * (1 + alpha) * i_vec / ELY['n_parallel']
            + A * ELY['T'] * np.log((j_vec + j_in) / ELY['j_0'])
            + B * ELY['T'] * np.log(1 - j_vec / ELY['j_L'] / (1 - alpha))
        )

    V_ely_bol = voltage_ely_sweep(alpha_ely_bol, i_ely_bol_sweep)
    V_ely_mol = voltage_ely_sweep(alpha_ely_mol, i_ely_mol_sweep)
    V_ely_eol = voltage_ely_sweep(alpha_ely_eol, i_ely_eol_sweep)

    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax2 = ax1.twinx()

    for i_ely_sweep, alpha, V, label, color in [
        (i_ely_bol_sweep, alpha_ely_bol, V_ely_bol, r'$BoL\ (\alpha=0)$',          '#2ca02c'),
        (i_ely_mol_sweep, alpha_ely_mol, V_ely_mol, rf'$MoL\ (\alpha={alpha_ely_mol:.3f})$', '#ff7f0e'),
        (i_ely_eol_sweep, alpha_ely_eol, V_ely_eol, rf'$EoL\ (\alpha={alpha_ely_eol:.3f})$', '#d62728'),
    ]:
        P_sweep = i_ely_sweep * V / 1000  # kW (puissance consommée)
        ax1.plot(i_ely_sweep, V,       color=color, lw=2.2,  label=label)
        ax2.plot(i_ely_sweep, P_sweep, color=color, lw=2.2,  linestyle='--')

    ax1.set_xlabel(r'$\mathbf{Current\ }i_{ELY}\ \mathrm{[A]}$', fontsize=18)
    ax1.set_ylabel(r'$\mathbf{Voltage\ }V_{ELY}\ \mathrm{[V]}$',  fontsize=18, color='k')
    ax2.set_ylabel(r'$\mathbf{Power\ }P_{ELY}\ \mathrm{[kW]}$',   fontsize=18, color='gray')
    ax2.tick_params(axis='y', labelcolor='gray')
    ax1.set_title(r'$\mathbf{PEMWE\ polarization\ curve\ (BoL\ /\ MoL\ /\ EoL)}$', fontsize=20)
    ax1.legend(fontsize=14, loc='upper left')
    ax1.grid(True)
    # Ligne verticale indicative pour le courant nominal observé (~75% de i_max BoL)
    i_ely_nom_ref = 0.75 * i_ely_bol_max
    ax1.axvline(x=i_ely_nom_ref, color='k', linestyle=':', lw=1.5)
    ax1.annotate(rf'$i_{{nom,ref}}$', xy=(i_ely_nom_ref, ax1.get_ylim()[0]),
                 xytext=(i_ely_nom_ref + i_ely_bol_max * 0.02, ax1.get_ylim()[0] + 0.5),
                 fontsize=13, color='k')
    plt.tight_layout()
    plt.savefig(os.path.join(savedir, 'polarization_ely.pdf'), format='pdf', bbox_inches='tight')
    plt.close()
    
    return lpsp_percent, cost_keur