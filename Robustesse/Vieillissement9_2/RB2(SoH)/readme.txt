Dossier RB2(SoH) -- strategie rule-based dont les 2 setpoints H2 sont modules
par le SoH :  P_fc_set = c_fc*Pmax_fc*SoH_fc^gamma_fc ;
              P_ely_set = c_ely*Pmax_ely*SoH_ely^gamma_ely.

Contenu (apres nettoyage V9) :
  get_optimal_action_RB.py  : la strategie (setpoints a re-caler apres sweep).
  main.py                   : run + plots d'une simulation unique.
  sweep_setpoints_rb2soh.py : *** sweep principal *** -> optimise (c_fc, gamma_fc,
                              c_ely, gamma_ely) par cout unifie (deg + VoLL*LPSP),
                              classe, reporte l'optimum + le gain attribuable au
                              SoH. Sortie : sweep_setpoints_rb2soh.txt/.pdf/.png.
  sweep_rb2soh_agedpmax.py  : VARIANTE de regle (P = c*Pmax_t, fraction de la
                              capacite VIEILLIE, inspiree du DP) ; ~ equivalente
                              au cas gamma~3 du sweep principal. Gardee comme test.
  run_meso.slurm            : lanceur generique mesocentre : sbatch run_meso.slurm
                              sweep_setpoints_rb2soh.py

Supprimes en V9 (redondants, absorbes par sweep_setpoints_rb2soh.py qui calcule
le cout unifie inline) : sweep_soh_attribution.py, rank_rb2soh_unified.py.

Workflow : lancer sweep_setpoints_rb2soh.py -> reporter l'optimum dans
get_optimal_action_RB.py -> relancer batch_pareto.py a la racine.
