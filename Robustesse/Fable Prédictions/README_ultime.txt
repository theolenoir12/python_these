================================================================================
FABLE PREDICTIONS : RB2 ULTIME (unifiee + pre-charge previsionnelle)
Dossier : Robustesse/Fable Predictions/       Cree : 2026-07-02
================================================================================

0. RESUME
--------------------------------------------------------------------------------
RB2(Ultime) = l'empilement des leviers VALIDES de la these, chacun atteste par
son sweep d'attribution :

    niveau 0  socle cost-min           0.440 / 0.310            (reopt_pred)
    niveau 1  setpoints H2 x SoH       SoH_fc^1, SoH_ely^2      (sweep_soh_attribution)
    niveau 2  plafond SoC x SoH_bat    g = 0.2                  (sweep_fable_socwin_fine)
    niveau 3  pre-charge previsionnelle hysteresis probabiliste
              +-1sigma (P_HI/P_LO = 0.84/0.16), gel 12 h        (sweep_fable_proba)

ADAPTATION niveau 2 x niveau 3 : la cible de pre-charge suit le PLAFOND VIEILLI
(SOC_TARGET_MODE = "ceiling" : cible = soc_max(t) - 0.005, lue dans
Common.get_lol). Avec une cible fixe 0.99, la pre-charge viserait une zone
interdite par le plafond abaisse -> ELY coupe pour rien. Le sweep "target"
teste cette hypothese contre des cibles fixes (0.90 / 0.95 / 0.99).

REPERES (25 ans, VoLL = 3) :
    RB2 socle ......................... 80.102   (ancrage)
    Unifiee (test nul, ENABLE=False) .. 78.336   (cf Fable/note_rb2_soh_unifiee.txt)
    RB2(SoH+Pred) reopt (g = 0) ....... 77.667   (reference actuelle de la these)
    RB2 ULTIME ........................ CIBLE < 77.67, attendu ~77.2-77.4
                                        (si la pre-charge conserve son -1.10
                                        de la base RB2(SoH), cf reopt_sohpred)

ATTRIBUTION EN CASCADE : chaque niveau se desactive proprement --
    ENABLE=False                        -> unifiee exacte (test nul prevision)
    _lol:SOC_MAX_AGED_GAIN = 0          -> RB2(SoH+Pred) (test nul plafond)
    GAMMA_* = 0 + gain 0 + ENABLE=False -> socle exact
Le bench inclut ces trois controles : toute derive se voit immediatement.

1. LANCER
--------------------------------------------------------------------------------
    python bench_ultime.py --quick               # fumee (1 an, N=2)
    sbatch run_meso_ultime.slurm 200 25 --omni   # bench complet + bornes omni
    sbatch run_meso_ultime.slurm 64 25 --sweep target  # cible pre-charge x gel
    sbatch run_meso_ultime.slurm 64 25 --sweep hpre    # horizon 12/18/24 h

Sorties : bench_ultime.txt (+ _cloud.csv), sweep_ultime_target.txt,
sweep_ultime_hpre.txt. Les jobs ecrivent des fichiers DISTINCTS par mode.

2. LECTURE DES RESULTATS
--------------------------------------------------------------------------------
  - "Unifiee (test nul)" doit redonner 78.336 pile (sinon : probleme d'env).
  - "RB2(SoH+Pred) (g=0)" doit tomber a ~77.6-77.7 : c'est la reproduction de
    la reference reopt_sohpred PAR CE MODULE (memes gammas, pre-charge +-1sigma
    equivalente M_SIGMA=1/gel 12) -- valide la comparabilite.
  - "RB2 ULTIME" est le point final : succes si < 77.67 (reference), ideal
    ~77.2-77.4. Verifier ELY_starts (pas d'inflation de demarrages) et sLPSP.
  - sweep target : si "ceiling" bat les cibles fixes, l'adaptation
    plafond-vieilli est validee (a consigner) ; sinon garder la meilleure fixe.
  - sweep hpre : H_PRE = 18 h attendu optimal (echelle diurne, comme partout).

3. GARDE-FOUS
--------------------------------------------------------------------------------
  - Ne PAS copier les .py du cluster vers le PC : seuls les resultats
    (.txt / .out) se rapatrient. Le code va toujours PC -> git -> cluster.
  - Le plafond SoC est applique par Common/get_lol via SOC_MAX_AGED_GAIN :
    le bench le fixe PAR TACHE ("_lol:SOC_MAX_AGED_GAIN") et le remet a 0
    ensuite ; en run unitaire, RB2(Ultime)/main.py le fixe explicitement.
  - La strategie LIT le plafond effectif dans Common.get_lol (jamais sa propre
    constante) : cible de pre-charge et contrainte restent coherentes.
================================================================================
