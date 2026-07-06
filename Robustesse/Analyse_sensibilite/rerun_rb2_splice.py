# -*- coding: utf-8 -*-
"""rerun_rb2_splice.py -- re-run CIBLE de RB2 / RB2(SoH) + SPLICE dans results_meso/.
=================================================================================
CONTEXTE. Les setpoints des strategies RB2 et RB2(SoH) ont ete re-deplaces
(cf. get_optimal_action_RB.py). SEULES ces deux strategies ont change : les 8
autres (0-100..SoC06, RB1) sont deterministes et inchangees. Comme les analyses
de sensibilite tirent les MEMES echantillons Monte-Carlo pour toutes les
strategies (common random numbers, seed fixe genere AVANT la boucle strategie),
on peut ne re-simuler que RB2 et RB2(SoH) et REMPLACER leurs lignes dans les
fichiers results_meso/sens_<axe>.txt deja produits, sans retoucher les autres.

CE QUE FAIT CE SCRIPT (sans MODIFIER les sens_*.py d'origine)
------------------------------------------------------------
Pour chaque axe demande :
  1. importe le module sens_<axe> ;
  2. monkeypatch sa liste de scenarios -> [RB2, RB2(SoH)] UNIQUEMENT ;
  3. redirige ses sorties (OUT_TXT + figures) vers un dossier TEMP -> les
     resultats d'origine dans results/ ne sont PAS ecrases ;
  4. appelle son main() : calcul IDENTIQUE a l'original (memes tirages MC, memes
     colonnes) mais seulement pour RB2 / RB2(SoH) (+ OAT, qui porte deja sur la
     reference RB2(SoH)) ;
  5. SPLICE : dans results_meso/sens_<axe>.txt, remplace les lignes "RB2;..." et
     "RB2(SoH);..." du front + le bloc "## OAT ..." par ceux fraichement calcules.
     Une sauvegarde .bak.<horodatage> est ecrite avant toute modification.

Les 8 autres strategies, l'entete et la structure du fichier sont conserves.

AXES COUVERTS (multi-strategies, consommes par ../Pareto/generate_ellipses.py) :
    eol  hthresholds  sizing  cweights  calendar
Axes deja mono/bi-strategie (RB2(SoH) seul, ou RB2 + RB2(SoH)) -> PAS de splice :
on re-genere le fichier ENTIER via la commande 'runfull' (sorties directes vers
results_meso/, et figure sens_soh_pareto.pdf recopiee dans figures_ellipses/) :
    soh       sens_soh_estimation.py   (RB2(SoH) : biais + bruit d'estimation SoH)
    timestep  sens_timestep.py         (RB2 + RB2(SoH) : pas de temps, bruit intra-h)

UTILISATION
-----------
  # 1) validation SANS simulation : verifie que le splice ne corrompt rien
  #    (re-injecte les lignes RB2/RB2(SoH) existantes -> doit etre identique)
  python rerun_rb2_splice.py selftest

  # 2) re-run + splice reel (setpoints actuels) :
  python rerun_rb2_splice.py run eol calendar            # axes grossierement stales
  python rerun_rb2_splice.py run eol hthresholds sizing cweights calendar   # tous
  python rerun_rb2_splice.py run cweights                # le moins couteux (analytique)

  # options :
  python rerun_rb2_splice.py run eol --nmc 5             # N_MC reduit (test mecanique)
  python rerun_rb2_splice.py run eol --dry               # calcule mais N'ECRIT PAS results_meso

  # 3) axes mono/bi-strategie (re-genere le fichier entier vers results_meso) :
  python rerun_rb2_splice.py runfull soh timestep

Puis regenerer les figures :  cd ../Pareto && python generate_ellipses.py

ATTENTION (a valider apres coup) : pour eol et calendar, les fichiers canoniques
actuels ont les 8 strategies de BASE a N=15 seulement, tandis que RB2/RB2(SoH)
seront re-calcules a N=200. Le nuage/ellipse en sera plus lisse pour RB2/RB2(SoH)
que pour les autres, et la colonne N_ok sera mixte (15 / 200). Les ellipses
tracees par generate_ellipses n'utilisent QUE (moyenne, ecart-type) -> l'aspect
visuel reste coherent, mais si tu veux une table homogene a N=200, relance ces
deux axes en COMPLET (les 10 strategies).
"""
import os
import sys
import shutil
import time
import importlib

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
RESULTS_MESO = os.path.join(HERE, "results_meso")
TMP_DIR = os.path.join(HERE, "_tmp_rb2_splice")

# strategie(s) re-simulee(s) : (dossier Vieillissement8, label)
ONLY = [("RB2", "RB2"), ("RB2(SoH)", "RB2(SoH)")]
RB2_KEYS = ("RB2", "RB2(SoH)")

# axe -> (module, fichier canonique dans results_meso, a-t-il un bloc ## OAT ?)
AXES = {
    "eol":         ("sens_eol",         "sens_eol.txt",         True),
    "hthresholds": ("sens_hthresholds", "sens_hthresholds.txt", True),
    "sizing":      ("sens_sizing",      "sens_sizing.txt",      True),
    "cweights":    ("sens_cweights",    "sens_cweights.txt",    False),
    "calendar":    ("sens_calendar",    "sens_calendar.txt",    True),
}

# Axes DEJA mono/bi-strategie (RB2 et/ou RB2(SoH) uniquement) : rien a splicer, on
# re-genere le FICHIER ENTIER en redirigeant les sorties vers results_meso/. Le
# 3e champ = figure a recopier aussi dans ../Pareto/figures_ellipses/ (ou None).
RUNFULL = {
    "soh":      ("sens_soh_estimation", "sens_soh.txt",      "sens_soh_pareto.pdf"),
    "timestep": ("sens_timestep",       "sens_timestep.txt", None),
}
FIG_ELLIPSES = os.path.join(HERE, "..", "Pareto", "figures_ellipses")


# --------------------------------------------------------------------------- #
#  SPLICE (pur texte, sans simulation) -- coeur du script, teste par selftest  #
# --------------------------------------------------------------------------- #
def _front_line(lines, key):
    """Renvoie (index, ligne) de l'UNIQUE ligne de front commencant par '<key>;'
    (avant un eventuel bloc ## OAT). Erreur si 0 ou >1 correspondances."""
    hits = [(i, ln) for i, ln in enumerate(lines)
            if ln.startswith(key + ";") and not ln.lstrip().startswith("#")]
    # on s'arrete au bloc OAT s'il existe
    oat = _oat_index(lines)
    if oat is not None:
        hits = [(i, ln) for i, ln in hits if i < oat]
    if len(hits) != 1:
        raise ValueError("attendu 1 ligne de front '%s;' , trouve %d" % (key, len(hits)))
    return hits[0]


def _oat_index(lines):
    """Index de la 1re ligne '## OAT ...' , ou None."""
    for i, ln in enumerate(lines):
        if ln.strip().startswith("## OAT"):
            return i
    return None


def splice(canonical_text, new_rb2, new_rb2soh, new_oat_block, has_oat):
    """Remplace, dans le texte canonique :
      - la ligne de front 'RB2;...'      par new_rb2
      - la ligne de front 'RB2(SoH);...' par new_rb2soh
      - (si has_oat) tout depuis '## OAT' jusqu'a la fin par new_oat_block
    Conserve entete, 8 autres strategies, ordre et structure. Renvoie le texte."""
    lines = canonical_text.splitlines(keepends=True)

    # 'head' = tout ce qui precede le bloc OAT (front + entete). Le remplacement
    # des lignes RB2/RB2(SoH) se fait UNIQUEMENT dans le front, jamais dans l'OAT.
    if has_oat:
        oat = _oat_index(lines)
        if oat is None:
            raise ValueError("bloc '## OAT' introuvable dans le fichier canonique")
        head = lines[:oat]
    else:
        head = list(lines)

    for key, repl in (("RB2", new_rb2), ("RB2(SoH)", new_rb2soh)):
        idx, old = _front_line(head, key)
        eol = old[len(old.rstrip("\r\n")):]     # preserve \n / \r\n d'origine
        head[idx] = repl.rstrip("\r\n") + eol

    if has_oat:
        block = new_oat_block if new_oat_block.endswith("\n") else new_oat_block + "\n"
        return "".join(head) + block
    return "".join(head)


def extract_new(temp_text, has_oat):
    """Depuis le txt fraichement produit (2 strategies), extrait :
      (ligne_front_RB2, ligne_front_RB2(SoH), bloc_OAT_ou_"")."""
    lines = temp_text.splitlines(keepends=True)
    _, rb2 = _front_line(lines, "RB2")
    _, rb2soh = _front_line(lines, "RB2(SoH)")
    oat_block = ""
    if has_oat:
        oat = _oat_index(lines)
        if oat is None:
            raise ValueError("bloc '## OAT' absent du txt temporaire")
        oat_block = "".join(lines[oat:])
    return rb2.rstrip("\r\n"), rb2soh.rstrip("\r\n"), oat_block


# --------------------------------------------------------------------------- #
#  SELFTEST : re-injecte les lignes existantes -> le fichier doit etre inchange #
# --------------------------------------------------------------------------- #
def selftest():
    ok = True
    for axis, (_mod, fname, has_oat) in AXES.items():
        path = os.path.join(RESULTS_MESO, fname)
        if not os.path.exists(path):
            print("  [%-11s] ABSENT -> %s" % (axis, path)); ok = False; continue
        with open(path, encoding="utf-8") as f:
            text = f.read()
        rb2, rb2soh, oat = extract_new(text, has_oat)
        out = splice(text, rb2, rb2soh, oat, has_oat)
        same = (out == text)
        print("  [%-11s] splice identite : %s" % (axis, "OK" if same else "DIFFERENT !"))
        if not same:
            ok = False
    print("SELFTEST:", "OK (splice non destructif)" if ok else "ECHEC")
    return ok


# --------------------------------------------------------------------------- #
#  RUN : monkeypatch + main() vers un temp, puis splice dans results_meso       #
# --------------------------------------------------------------------------- #
def run_axis(axis, nmc=None, dry=False):
    mod_name, fname, has_oat = AXES[axis]
    canonical = os.path.join(RESULTS_MESO, fname)
    if not os.path.exists(canonical):
        print("[%s] canonique introuvable -> %s (ignore)" % (axis, canonical)); return
    os.makedirs(TMP_DIR, exist_ok=True)

    mod = importlib.import_module(mod_name)

    # 1) restreindre a RB2 / RB2(SoH) : sens_calendar utilise EMS_LIST, les autres SCENARIOS
    if hasattr(mod, "SCENARIOS"):
        mod.SCENARIOS = list(ONLY)
    if hasattr(mod, "EMS_LIST"):
        mod.EMS_LIST = list(ONLY)
    # 2) rediriger les sorties vers le temp (txt + figures)
    tmp_txt = os.path.join(TMP_DIR, fname)
    mod.OUT_TXT = tmp_txt
    mod.RESULTS_DIR = TMP_DIR
    # 3) N_MC optionnel (test mecanique)
    if nmc is not None and hasattr(mod, "N_MC"):
        mod.N_MC = int(nmc)

    print("\n" + "=" * 78)
    print("[%s] re-run RB2 / RB2(SoH)  (N_MC=%s)  -> %s"
          % (axis, getattr(mod, "N_MC", "n/a"), tmp_txt))
    print("=" * 78, flush=True)
    t0 = time.time()
    mod.main()
    print("[%s] calcul termine en %.0fs" % (axis, time.time() - t0), flush=True)

    # 4) splice
    with open(tmp_txt, encoding="utf-8") as f:
        new_text = f.read()
    rb2, rb2soh, oat = extract_new(new_text, has_oat)
    with open(canonical, encoding="utf-8") as f:
        can_text = f.read()

    # apercu avant/apres
    old_rb2 = _front_line(can_text.splitlines(keepends=True), "RB2")[1].rstrip("\r\n")
    old_soh = _front_line(can_text.splitlines(keepends=True), "RB2(SoH)")[1].rstrip("\r\n")
    print("  RB2      : %s" % old_rb2)
    print("        -> : %s" % rb2)
    print("  RB2(SoH) : %s" % old_soh)
    print("        -> : %s" % rb2soh)

    spliced = splice(can_text, rb2, rb2soh, oat, has_oat)
    if dry:
        print("  [DRY] results_meso/%s NON modifie." % fname); return
    bak = canonical + ".bak." + time.strftime("%Y%m%d_%H%M%S")
    shutil.copy2(canonical, bak)
    with open(canonical, "w", encoding="utf-8") as f:
        f.write(spliced)
    print("  splice ecrit -> %s   (sauvegarde: %s)" % (canonical, os.path.basename(bak)))


def run_full(name, nmc=None, dry=False):
    """Axe mono/bi-strategie : re-genere le fichier ENTIER vers results_meso/.
    Redirige OUT_TXT / OUT_PDF / RESULTS_DIR (figures) vers results_meso, lance
    main(), et recopie la figure d'ellipses dans figures_ellipses/ si demandee."""
    mod_name, fname, fig_copy = RUNFULL[name]
    canonical = os.path.join(RESULTS_MESO, fname)
    os.makedirs(RESULTS_MESO, exist_ok=True)
    mod = importlib.import_module(mod_name)

    # rediriger toutes les sorties vers results_meso
    mod.RESULTS_DIR = RESULTS_MESO
    mod.OUT_TXT = canonical
    if hasattr(mod, "OUT_PDF"):
        mod.OUT_PDF = os.path.join(RESULTS_MESO, os.path.basename(mod.OUT_PDF))
    if nmc is not None and hasattr(mod, "N_MC"):
        mod.N_MC = int(nmc)

    print("\n" + "=" * 78)
    print("[%s] re-run COMPLET -> %s  (N_MC=%s)" % (name, canonical, getattr(mod, "N_MC", "n/a")))
    print("=" * 78, flush=True)
    if dry:
        print("  [DRY] rien lance."); return
    if os.path.exists(canonical):
        shutil.copy2(canonical, canonical + ".bak." + time.strftime("%Y%m%d_%H%M%S"))
    t0 = time.time()
    mod.main()
    print("[%s] termine en %.0fs -> %s" % (name, time.time() - t0, canonical), flush=True)

    if fig_copy:
        src = os.path.join(RESULTS_MESO, fig_copy)
        dst = os.path.join(FIG_ELLIPSES, fig_copy)
        if os.path.exists(src):
            if os.path.exists(dst):
                shutil.copy2(dst, dst + ".bak." + time.strftime("%Y%m%d_%H%M%S"))
            shutil.copy2(src, dst)
            print("  figure statique rafraichie -> %s" % dst)


def main(argv):
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__); return
    cmd = argv[0]
    if cmd == "selftest":
        sys.exit(0 if selftest() else 1)
    if cmd == "runfull":
        rest = argv[1:]
        nmc = None; dry = False; names = []
        i = 0
        while i < len(rest):
            a = rest[i]
            if a == "--nmc":
                nmc = rest[i + 1]; i += 2; continue
            if a == "--dry":
                dry = True; i += 1; continue
            names.append(a); i += 1
        names = names or list(RUNFULL)
        for name in names:
            if name not in RUNFULL:
                print("axe runfull inconnu :", name, "(", ", ".join(RUNFULL), ")"); continue
            run_full(name, nmc=nmc, dry=dry)
        print("\nTermine (runfull).")
        return
    if cmd == "run":
        rest = argv[1:]
        nmc = None
        dry = False
        axes = []
        i = 0
        while i < len(rest):
            a = rest[i]
            if a == "--nmc":
                nmc = rest[i + 1]; i += 2; continue
            if a == "--dry":
                dry = True; i += 1; continue
            axes.append(a); i += 1
        axes = axes or list(AXES)
        for axis in axes:
            if axis not in AXES:
                print("axe inconnu :", axis, "(", ", ".join(AXES), ")"); continue
            run_axis(axis, nmc=nmc, dry=dry)
        print("\nTermine. Regenere les figures : cd ../Pareto && python generate_ellipses.py")
        return
    print("commande inconnue :", cmd, "-- attendu 'selftest' ou 'run'")


if __name__ == "__main__":
    main(sys.argv[1:])
