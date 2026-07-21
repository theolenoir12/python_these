# Plan scientifique — logique floue, apprentissage de règles et ANFIS

Date : 21 juillet 2026.

Statut : protocole de développement. Aucun résultat de performance n'est acquis.
Le dossier `Vieillissement11/MPC/` est hors périmètre de ce chantier.

## 1. Question scientifique

Le chantier doit distinguer deux questions qui seraient confondues si une seule
EMS complexe était comparée directement à RB1/RB2 :

1. une structure de décision plus expressive qu'une règle déterministe simple
   rapproche-t-elle une EMS online de la référence PD offline ?
2. à structure et budget de réglage comparables, l'ajout du SoH ou d'une
   prévision apporte-t-il une information actionnable ?

Le premier effet est un effet de **famille de commande** ; le second est un
effet de **jeu d'information**. Ils doivent être mesurés séparément.

Le socle nominal reste V11 avec dommage PEMWE `p=2`, ledger corrigé, profil
canonique empreinté et VoLL de reporting égal à 3 EUR/kWh. La sensibilité
`p=3` ne sera rejouée que sur les stratégies finalistes.

## 2. Familles à comparer

### 2.1 FLC experte

Construire une logique floue explicite et auditée, sans apprentissage, comme
référence de l'état de l'art rule-based fuzzy. Une architecture hiérarchique à
deux branches évite l'explosion combinatoire :

- déficit : `P_net > 0`, décision de puissance PEMFC ;
- surplus : `P_net < 0`, décision de puissance PEMWE ;
- la batterie ferme le bilan de puissance ;
- `get_lol` et `simulate_transition` restent les seules autorités de
  faisabilité physique.

Entrées minimales de chaque branche : puissance nette courante normalisée,
SoC batterie et niveau H2 normalisé. Les fonctions d'appartenance sont
triangulaires ou trapézoïdales et la base de règles est exportée sous forme
lisible. Le premier contrôleur sera de type Mamdani afin de constituer une
référence floue distincte des modèles appris de type Sugeno.

La version SoH ne doit pas ajouter toutes les santés au produit cartésien des
règles. Une couche floue de correction de santé, inspirée des FLC « mutatives »,
modifie la consigne de la branche à partir de l'usure relative du composant H2
et de la batterie. Un coefficient nul doit reproduire exactement le parent
sans SoH.

### 2.2 Règles apprises depuis la PD

La stratégie principale de rule learning distille la commande signée de la
chaîne H2 produite par la PD : puissance PEMFC positive, puissance PEMWE
négative, batterie résiduelle. Le premier enseignant est le point central
`epsilon=3` de la PD V11-p=2.

La méthode de référence suit le schéma publié pour les FCHEV : simplification
ou stratification du jeu optimal, extraction d'une liste de règles
interprétables, puis régression locale des conséquences. Un arbre de régression
peu profond pourra servir de benchmark technique, mais ne devra pas être
présenté comme équivalent à une liste de règles de type RIPPER.

La profondeur, le nombre de feuilles/règles et le nombre de variables actives
font partie des résultats. La fidélité en MSE à la PD est un diagnostic ; le
critère de décision reste la performance en boucle fermée V11.

### 2.3 ANFIS

ANFIS est une extension conditionnelle, pas la première baseline. Il utilisera
un système Takagi--Sugeno de premier ordre, séparé en branches déficit et
surplus, initialisé à partir des partitions floues retenues. Les paramètres des
appartenances et des conséquences seront entraînés sur les mêmes données et
avec le même budget que les autres méthodes apprises.

ANFIS ne sera promu que si :

- le nombre de règles reste maîtrisé et publiable ;
- la validation hors apprentissage est meilleure que celle des règles plus
  simples ;
- le gain en boucle fermée ne vient pas d'une fuite d'information temporelle ;
- le temps de décision reste négligeable devant le pas horaire.

## 3. Jeux d'information

Chaque famille retenue doit être déclinée par paires attribuables.

### I0 — état online sans connaissance augmentée

- puissance nette courante `P_net(t)` ;
- SoC batterie ;
- énergie H2 normalisée ;
- éventuellement états marche/arrêt précédents si leur apport est testé par
  ablation.

Le mois, le jour ou l'heure de l'année sont exclus de I0 : sur un profil répété,
ils encoderaient implicitement une prévision/climatologie.

### IS — I0 + SoH

- usure normalisée batterie ;
- usure normalisée PEMFC dans la branche déficit ;
- usure normalisée PEMWE dans la branche surplus.

Les bornes physiques vieillies restent appliquées à toutes les stratégies pour
la sécurité du simulateur. Dans I0, elles ne doivent pas être réinjectées comme
variables de décision, car elles constitueraient un canal SoH caché.

### IF — I0 + prévision

La prévision sera ajoutée après stabilisation de I0/IS. Les variables candidates
sont des bilans énergétiques nets agrégés sur des horizons annoncés, et non la
fenêtre brute entière. H24 permet la comparaison directe au travail MPC ; des
horizons J+3/J+7 peuvent être testés séparément pour la réserve H2 saisonnière.
Information parfaite, bruit/biais et persistance doivent rester des scénarios
distincts.

### ISF — I0 + SoH + prévision

Cette combinaison n'est calculée qu'après les tests nuls et les effets simples
IS-I0 et IF-I0. Elle ne doit pas servir à masquer l'absence d'effet propre d'une
des deux informations.

## 4. Données enseignantes disponibles

Le cache canonique
`DP/runs/dp_aging_v11_p2_25y_51x51.npz` contient déjà, pour `PD_seq_v2`,
218 999 actions horaires ainsi que SoC, H2, puissances et SoH. Il permet de
construire le premier jeu d'apprentissage au point central sans relancer la PD.
L'audit canonique établit l'identité de ce point avec `epsilon=3` du front.

Le cache compact du front à 19 epsilon ne contient pas toutes les actions et
les états requis pour un apprentissage propre. Il ne doit pas être inversé ou
complété par des approximations silencieuses. Après sélection de l'architecture,
les enseignants supplémentaires du front seront produits seulement pour un
petit ensemble préannoncé de poids de fiabilité, avec sauvegarde explicite des
trajectoires d'action.

## 5. Construction des jeux d'apprentissage

Un découpage horaire aléatoire est interdit : les heures voisines et les années
répétées rendraient la validation artificiellement facile.

Le protocole initial doit :

1. séparer les blocs temporels d'apprentissage, validation et test ;
2. stratifier les observations par signe et amplitude de `P_net`, zones de SoC,
   zones de stock H2, proximité des contraintes et événements de délestage ;
3. conserver ou surpondérer les états rares critiques au lieu d'optimiser la
   seule erreur moyenne ;
4. produire deux matrices strictement appariées I0 et IS, qui diffèrent
   uniquement par les variables de santé ;
5. enregistrer l'empreinte du cache source, la liste des variables, les bornes
   de normalisation et les indices temporels de chaque partition.

Comme un seul profil météorologique canonique est disponible dans ce cache,
une performance sur les années tenues à l'écart n'établira pas à elle seule la
généralisation à d'autres profils.

## 6. Comparaison équitable

Pour chaque famille et chaque jeu d'information :

- même moteur V11-p=2 et même comptabilité des remplacements ;
- mêmes profils, états initiaux et scénarios d'incertitude ;
- même métrique de fiabilité et même VoLL de reporting ;
- même budget de réglage, compté en évaluations fermées du simulateur et en
  temps de calcul ;
- caches pleine précision dans `FuzzyRules/runs/<id>_<empreinte>/` ;
- identifiant immuable de stratégie, sans alias nu `Fuzzy`, `Rules` ou `ANFIS` ;
- temps de décision, nombre de règles et mémoire reportés avec les performances.

Les références minimales sont RB1 V11-p=2 `(0,20 ; 0,40)`, RB2 V11-p=2
`(0,574 ; 0,465)` et le point enseignant PD correspondant. Les parents doivent
être réévalués sur exactement le même profil que leurs variantes.

Tests nuls obligatoires :

- intensité de correction SoH nulle = contrôleur I0 bit-à-bit ;
- entrées SoH constantes à 1 = égalité fonctionnelle attendue au BoL ;
- désactivation de l'apprentissage = paramètres initiaux inchangés ;
- bilan de puissance fermé et même traitement de `get_lol` ;
- `reset()` restaure intégralement l'état interne d'une politique ;
- permutation d'une variable non utilisée sans effet sur la commande.

## 7. Critères de lecture

Les diagnostics d'imitation sont : MAE/RMSE de la commande H2, erreur par zone
d'état, accord sur marche/arrêt et fidélité aux événements rares. Ils ne
constituent pas le résultat scientifique principal.

La décision repose sur :

- EENS et LPSP ;
- coût de dégradation total et par composant ;
- `J = C_deg + 3 EENS` ;
- remplacements et premières durées de vie ;
- distance au front PD dans le plan EENS--dégradation ;
- écarts appariés sous bruit/biais SoH et profils hors apprentissage ;
- complexité interprétable et coût online.

Un gain d'au moins 1 % sur J central sert de seuil de criblage avant les calculs
longs. La revendication d'un apport matériel reste le seuil préannoncé de
quelques pourcents, supérieur à la sensibilité numérique et sans dégradation
rédhibitoire de la fiabilité.

## 8. Calcul progressif

### Étape A — préflight sans simulation longue

- auditer le cache enseignant central et créer son manifeste ;
- figer les conventions de signe, normalisation et saturation ;
- tester les moteurs flous sur des points synthétiques et aux frontières ;
- visualiser les surfaces de commande et vérifier leur cohérence physique.

### Étape B — screening central court

- FLC experte I0 ;
- règle apprise depuis PD I0 ;
- variantes IS strictement appariées ;
- smoke de quelques jours, puis un an seulement si les invariants passent.

### Étape C — validation aveugle

- blocs temporels réservés ;
- profils hors apprentissage disponibles ;
- bruit et biais SoH avec graines communes ;
- comparaison des niveaux et des écarts appariés.

### Étape D — ANFIS conditionnel

- même enseignant, mêmes partitions et même budget ;
- ablation I0/IS ;
- comparaison à complexité et temps online reportés.

### Étape E — front et horizon long

- promouvoir au plus les architectures qui passent le screening ;
- produire quelques enseignants PD supplémentaires préannoncés le long du
  front, plutôt que relancer aveuglément les 19 points ;
- construire le front réalisé de chaque famille ;
- rejouer 25 ans uniquement pour les finalistes ;
- appliquer ensuite la sensibilité V11 `p=3`.

## 9. Première implémentation recommandée

Commencer dans ce dossier par :

1. un chargeur audité du cache `PD_seq_v2` central ;
2. un moteur FLC Mamdani NumPy minimal, sans dépendance externe ;
3. une politique à deux branches branchable directement dans
   `Common.main_init_and_loop.init_and_run_loop` ;
4. les tests de faisabilité et de test nul ;
5. un benchmark un an I0 avant toute optimisation ou ANFIS.

Cette séquence fournit rapidement une baseline floue interprétable et prépare
le format commun nécessaire à la distillation et à ANFIS, sans toucher au code
MPC ni relancer la PD canonique.

## 10. Ancrage bibliographique initial

Ces références primaires motivent l'ordre méthodologique ; elles ne valident
pas par transfert les performances sur le micro-réseau GENIAL.

- Nivolianiti, Karnavas et Charpentier (2024),
  *Fuzzy Logic-Based Energy Management Strategy for Hybrid Fuel Cell Electric
  Ship Power and Propulsion System*, DOI `10.3390/jmse12101813` : exemple
  récent d'EMS Mamdani utilisant SoC et puissance demandée pour commander la
  PEMFC, avec appartenances triangulaires/trapézoïdales.
- Luca et al. (2022), *Comparative study of energy management systems for a
  hybrid fuel cell electric vehicle — A novel mutative fuzzy logic controller
  to prolong fuel cell lifetime*, DOI `10.1016/j.ijhydene.2022.05.192` :
  modification des appartenances à partir de la dégradation, qui motive une
  extension santé hiérarchique et un parent nul exact.
- Liu et al. (2020), *Online energy management strategy of fuel cell hybrid
  electric vehicles based on rule learning*, DOI
  `10.1016/j.jclepro.2020.121017` : trajectoires optimales hors ligne,
  clustering hiérarchique, extraction RIPPER et régression linéaire des règles.
- Zhou et al. (2023), *Dynamic programming improved online fuzzy power
  distribution in a demonstration fuel cell hybrid bus*, DOI
  `10.1016/j.energy.2023.128549` : construction d'une répartition floue online
  à partir des relations état--commande de la PD.
- Banagar et al. (2026), *Load-adaptive fuzzy energy management architecture
  for fuel cell hybrid electric heavy-duty trucks*, DOI
  `10.1016/j.ijhydene.2026.155828` : entraînement des appartenances floues pour
  reproduire une référence PD et validation explicite sur des cycles non vus,
  avec mise en évidence des limites de transfert hors calibration.
- Jang (1993), *ANFIS: Adaptive-Network-Based Fuzzy Inference System*, DOI
  `10.1109/21.256541` : formulation fondatrice du système flou adaptatif de type
  Sugeno et apprentissage hybride.
- Li et al. (2012), *Power management strategy based on adaptive neuro-fuzzy
  inference system for fuel cell-battery hybrid vehicle*, DOI
  `10.1063/1.3682057` : application ANFIS à la répartition de puissance d'un
  hybride PEMFC--batterie.

Avant insertion dans le manuscrit, les entrées BibTeX absentes devront être
créées depuis les notices éditeur et les affirmations quantitatives revérifiées
dans les articles complets.
