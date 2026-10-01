# Journal de bord

Prédiction du risque de réadmission hospitalière à 30 jours — sujet CISIA Santé.

Ce journal retrace la méthode et les choix, y compris les pistes écartées et les décisions révisées. Il est tenu au fil de l'eau, pas reconstitué après coup.

---

## 24/08/2026 — Séance 1 : cadrage et audit initial

### J1-01 · Lecture du sujet et du livrable attendu

Sujet : concevoir un système de prédiction du risque de réadmission à 30 jours pour un groupement hospitalier régional. Objectif double — distinguer les sorties à faible risque des situations à suivi renforcé, et proposer une solution intégrable au SIH et extensible à de nouvelles sources.

Trois axes sont explicitement désignés comme sensibles dans l'énoncé : la qualité des données issues de sources cliniques hétérogènes, les biais liés à l'âge / au territoire / au profil d'utilisation des soins, et les contraintes RGPD propres aux données de santé (article 9). L'énoncé pose une interdiction ferme : aucune donnée socio-économique individuelle non collectée en milieu hospitalier. La table INSEE agrégée est donc le seul proxy socio-économique autorisé.

Livrable : une production écrite accompagnée d'un journal de bord, le tout dans un notebook, présenté devant jury.

### J1-02 · Choix du périmètre de données

**Décision : exploiter les 11 tables via un pipeline modulaire — un bloc de features indépendant par source.**

Alternatives écartées :

- *Cœur SIH/PMSI + deux sources en démonstration.* Réduisait la charge d'environ 40 %, mais laissait de côté les pièges les plus démonstratifs (harmonisation des unités biologiques, séries temporelles de constantes) et exposait à la question « pourquoi le DPI est-il écarté ? ».
- *Profondeur sur un périmètre restreint* (patients + séjours + diagnostics). Notebook plus maîtrisable, mais l'axe intégration multi-sources — central dans l'énoncé — n'aurait pas été démontré.

Motif du choix : le sujet demande un système « capable d'évoluer avec l'ajout de nouvelles sources ». Une architecture en blocs branchables répond à cette exigence par construction plutôt que par argumentation. Ajouter une source revient à ajouter un bloc, sans toucher au reste du pipeline.

### J1-03 · Environnement de travail

venv Python 3.12.10 dédié. pandas 3.0.5, scikit-learn 1.9.0, numpy 2.5.2, matplotlib, seaborn, JupyterLab. Dépendances figées dans `requirements.txt` pour la reproductibilité. Les 11 tables se chargent sans erreur de parsing.

### J1-04 · Premier audit qualité — écarts constatés

Anomalies annoncées par l'énoncé et confirmées :

| Constat | Mesure |
|---|---|
| Durées de séjour négatives | minimum à -3 jours |
| Unités biologiques hétérogènes | Créatinine en mg/L (371) et µmol/L (364) ; Hémoglobine en g/L (360) et g/dL (333) |
| Valeurs biologiques aberrantes | CRP négative constatée dès la première ligne (-11,33 mg/L) |
| Capteurs de télésurveillance dégradés | 667 `CapteurDefaillant` + 505 `Gap` sur 6297 mesures, soit 18,6 % |

Manquants réels mesurés :

```
signes_vitaux   FrequenceRespiratoire 2021  Temperature 1673  TensionDiastolique 1471
                SpO2 1428  TensionSystolique 1275  FrequenceCardiaque 1245
medications     DateFin 777      (attendu : traitement en cours)
historique      DureeJours 326   (attendu : consultations)
biologies       Valeur 299
```

Sur `signes_vitaux`, les trous représentent 5 à 8 % par constante et se répartissent colonne par colonne : ce ne sont pas des lignes vides, mais des capteurs qui décrochent individuellement.

**Anomalies non annoncées par l'énoncé, découvertes à l'audit :**

1. *Manquants encodés en texte.* Dans `patients.csv`, l'absence est codée par la chaîne `"Non renseigné"` et non par une valeur nulle : Sexe (21), SituationFamiliale (67), RegimeAssurance. Un `isna()` renvoie zéro manquant sur cette table. Piège d'audit classique — le nettoyage doit normaliser ces sentinelles avant tout comptage.
2. *Intégrité référentielle cassée.* 73 lignes de `biologies` portent un `SejourID` absent de `sejours.csv`.
3. *Couverture partielle des actes.* `actes` ne référence que 649 séjours sur 878. 229 séjours sans acte, ce qui est cliniquement plausible : à traiter comme une absence informative, pas comme un manquant à imputer.

### J1-05 · Cadrage temporel — décision initiale

**Décision initiale : architecture à deux étages.**

- Modèle A au moment de la sortie (T0), sur les sources disponibles à cet instant, pour décider du plan de sortie.
- Modèle B à J+7, enrichi de la télésurveillance, pour déclencher une visite à domicile.

Motif : les objets connectés sont par nature des mesures *postérieures* à la sortie. Les injecter dans un modèle qui prédit à la sortie reviendrait à utiliser le futur — fuite temporelle. Le découpage en deux étages lève cette fuite tout en exploitant la source, et recouvre les deux usages nommés dans l'énoncé : « alerter les équipes en sortie » et « cibler les visites à domicile ».

### J1-06 · Vérification de faisabilité — et révision de la décision J1-05

Avant de construire sur ce cadrage, contrôle de la matière disponible pour le modèle B : combien de patients disposent réellement de télésurveillance après leur sortie ?

Le contrôle a invalidé l'hypothèse, et a révélé un problème plus large.

**Les horodatages des tables filles sont désynchronisés des dates de séjour.** Proportion de lignes tombant dans la fenêtre [admission, sortie] de leur propre séjour :

```
biologies        0,4 %
signes_vitaux    1,5 %
comptes_rendus   1,0 %
actes            0,2 %
```

Décalage médian entre le premier relevé de constantes et l'admission : **-236 jours**. Les dates filles ont été tirées indépendamment des dates de séjour. Cette anomalie n'est pas mentionnée dans l'énoncé.

Conséquence sur le modèle B — patients disposant d'au moins une mesure post-sortie :

| Fenêtre | Patients |
|---|---|
| J+0 → J+7 | 6 / 620 |
| J+0 → J+30 | 12 / 620 |
| J+0 → J+90 | 28 / 620 |

Sur 184 patients équipés, la médiane des mesures se situe 154 jours *avant* la sortie. Le flux couvre une année autour du séjour, sans relation avec lui. Un modèle entraîné sur 6 patients n'a aucune validité statistique.

**Décision révisée : conserver l'architecture à deux étages comme conception, implémenter intégralement le connecteur de l'étage B, mais ne pas l'entraîner.** Le bloc IoT est codé, branché, exécuté ; sa couverture réelle est mesurée et publiée. L'étage B est spécifié, avec le critère de volumétrie qu'il faudrait atteindre en production.

Alternative écartée : *resynchroniser les dates filles sur la fenêtre de leur séjour* pour rendre l'étage B entraînable. Techniquement faisable, méthodologiquement indéfendable — les performances mesureraient la règle de reconstruction, pas un phénomène clinique. Fabriquer la donnée qui manque n'est pas une correction de qualité.

Alternative écartée : *repli mono-étage avec l'IoT en simple audit qualité*. Plus court, mais l'extensibilité redeviendrait un argument écrit au lieu d'une propriété démontrée.

**Ce qui reste exploitable malgré la désynchronisation :**

- *Les liens `SejourID` sont valides.* Toute agrégation indifférente aux dates reste légitime : CRP moyenne du séjour, SpO2 minimale, nombre de prescriptions, comptages de diagnostics.
- *Le temps relatif intra-séjour est récupérable.* L'étendue temporelle des relevés par séjour (médiane 3,88 j) suit la durée de séjour (médiane 4,00 j), corrélation 0,354. Le bloc de relevés d'un séjour a la bonne durée, il est simplement mal positionné sur l'axe absolu. En recalant chaque séjour sur son propre premier relevé, on récupère les dynamiques intra-séjour — tendance de la SpO2, dégradation tensionnelle en fin de séjour — qui sont précisément les signaux prédictifs d'une décompensation après la sortie.

### J1-07 · Population d'étude et protocole d'évaluation

**Exclusion des séjours terminés par un décès.** `ModeSortie` compte 92 décès. Un patient décédé ne peut pas être réadmis : la cible devrait valoir 0 pour ces séjours. Les conserver reviendrait à apprendre au modèle à prédire le décès plutôt que la réadmission — fuite de cible. Exclusion documentée, avec mention de la population résiduelle.

> ⚠️ **Cette entrée a été révisée en séance 2.** Le raisonnement clinique était juste, mais l'hypothèse sur les données était fausse et n'avait pas été vérifiée. Voir J2-01.

**Découpage groupé par patient.** 620 patients pour 878 séjours : 218 patients ont 2 séjours, 20 en ont 3. Un découpage aléatoire classique placerait un séjour d'un patient en apprentissage et un autre en test, ce qui constitue une fuite. Le protocole retiendra un `GroupKFold` sur `PatientID`. Cette fuite est indolore en apparence — elle ne dégrade pas les métriques, elle les embellit — d'où la nécessité de la traiter explicitement.

**Métriques.** La cible est déséquilibrée : 158 réadmissions sur 878 séjours, soit 18,0 %. L'exactitude est disqualifiée (un modèle constant à 0 atteindrait 82 %). Le protocole s'appuiera sur le rappel, la précision, l'aire sous la courbe précision-rappel, et un choix de seuil explicitement argumenté par le coût asymétrique : un faux négatif est une réadmission non anticipée, un faux positif est une visite à domicile inutile.

### J1-08 · Contrôle de plausibilité du signal

Avant d'investir dans l'ingénierie de variables, vérification qu'un signal existe. Taux de réadmission par diagnostic principal (séjours ≥ 30) :

```
Hémorragie digestive              30,0 %   (n=30)
Fracture du col du fémur          29,0 %   (n=31)
Insuffisance cardiaque            25,5 %   (n=94)
Insuffisance rénale chronique     21,4 %   (n=103)
BPCO obstructive                  21,4 %   (n=103)
Diabète sucré de type 2           19,8 %   (n=91)
Hyperlipidémie                    17,3 %   (n=81)
Pneumopathie aiguë                13,3 %   (n=30)
Hypertension essentielle          13,3 %   (n=75)
Syndrome démentiel                10,5 %   (n=38)
Infection urinaire                 8,9 %   (n=45)
Arthrose de la hanche              7,5 %   (n=40)
Douleur thoracique                 3,1 %   (n=32)
```

Le gradient est cliniquement cohérent : pathologies chroniques décompensables en tête, motifs bénins ou chirurgie réglée en queue. Le jeu de données porte un signal apprenable, et ce classement est défendable sur le fond médical.

### J1-09 · Contrainte d'environnement — version de scipy figée

Incident rencontré : `scipy` 1.18.1 s'installe correctement mais refuse de s'importer, ses extensions compilées étant bloquées par Smart App Control (`VerifiedAndReputablePolicyState = 1`), la protection de Windows 11 qui rejette les binaires sans réputation établie. `scikit-learn` dépendant de scipy, toute la chaîne de modélisation était inutilisable.

Diagnostic : le blocage vise la version, non la bibliothèque. Les versions 1.16.2 et 1.15.3 s'importent sans difficulté — leur ancienneté leur a constitué une réputation.

**Décision : figer `scipy==1.16.2`** dans `requirements.txt`, ce qui contraint `numpy` à 2.4.6. Chaîne complète revalidée par un test fonctionnel — `LogisticRegression`, `HistGradientBoostingClassifier`, `GroupKFold`, `CalibratedClassifierCV` et les métriques s'exécutent correctement.

Alternative écartée : *désactiver Smart App Control*. Windows ne permet pas de le réactiver ensuite — la seule voie de retour est une réinstallation du système. Désactiver une protection irréversible pour contourner une contrainte de version n'est pas un arbitrage acceptable.

Alternative écartée : *basculer l'environnement sous WSL Debian*. Techniquement propre puisque la politique Windows n'y s'applique pas, mais la distribution ne dispose ni de `pip` ni de `venv`, et leur installation exige des privilèges administrateur. Coût disproportionné pour un problème réglé par une épingle de version.

Cette contrainte est notée ici parce qu'elle explique une version non courante dans `requirements.txt` : une reproduction du notebook sur une machine sans Smart App Control fonctionnerait aussi bien avec scipy récent.

### J1-10 · Bloc comptes-rendus — analyse du corpus et sélection des variables

**Le corpus n'est pas du texte libre.** Les 1144 documents suivent trois gabarits rigides, et chaque séjour dispose d'exactement un compte-rendu d'hospitalisation (878 CRH pour 878 séjours) ; 266 séjours portent en plus un courrier sortant ou une note soignante. L'analyse ne retient que les CRH, pour garantir un document par séjour.

Les trois gabarits portent des champs **disjoints** :

| Gabarit | Amorce | Champs disponibles | Séjours |
|---|---|---|---|
| A | « Patient admis pour… » | score de gravité, antécédents, évolution, destination | 296 |
| B | « Séjour de Xj. Motif… » | durée, complications, recommandations | 276 |
| C | « Hospitalisation de Xj pour… » | durée, comorbidités, modalité de suivi | 306 |

Conséquence structurelle : toute variable issue des comptes-rendus est absente pour environ deux tiers des séjours, et cette absence dépend du gabarit, non du hasard. Elle doit être traitée comme informative, avec indicateurs de présence. Cela plaide pour un modèle gérant nativement les valeurs manquantes (`HistGradientBoostingClassifier`) plutôt que pour une imputation généralisée.

**Test du lien à la cible avant construction des variables** (taux de base 18,0 %) :

| Variable candidate | n | Résultat | Retenue |
|---|---|---|---|
| Nombre de comorbidités | 306 | 1,43 (non réadmis) vs 2,24 (réadmis), p < 0,001 | **oui** |
| Antécédents déclarés | 296 | IRC 27,3 % → BPCO/tabagisme 7,5 %, gradient net | **oui** |
| Évolution favorable | 296 | 15,5 % vs 19,2 % | oui, à confirmer |
| Score de gravité /30 | 296 | 6,36 vs 7,20, p = 0,431 | non |
| Durée mentionnée | 582 | 5,60 vs 5,12, p = 0,633 | non (mais voir réparation) |
| Suivi par spécialiste | 59 | 11,9 % vs 18,4 %, effectif trop faible | non |
| Mention de complications | 878 | 18,6 % vs 18,2 % | non |
| Surveillance rapprochée | 878 | 17,8 % vs 18,1 % | non |

**Deux pièges évités grâce à ce test préalable.**

*La mention de complications est du remplissage de gabarit.* Le gabarit B contient systématiquement « Survenue de complications », parfois suivi d'un type. Une règle naïve `contient("complications")` aurait produit une variable qui n'encode rien d'autre que « ce séjour relève du gabarit B ». La mesure le confirme : 18,6 % contre 18,2 %, soit aucun écart.

*La destination de sortie mentionnée dans le texte est du bruit.* Le croisement avec `ModeSortie` ne montre aucune correspondance — la mention « Sortie autorisée vers domicile » apparaît notamment pour 13 séjours dont le mode de sortie est un décès. Ce champ a été généré indépendamment du reste ; il est écarté.

**Piste de réparation par le compte-rendu.** La durée mentionnée dans le texte concorde avec `DureeSejour` pour 95,8 % des 582 séjours concernés. Les 28 discordances correspondent exactement aux valeurs aberrantes signalées par l'énoncé, et le compte-rendu fournit dans ces cas une durée plausible. Réparation envisagée sur cette base.

### J1-11 · Triangulation des durées — la réparation retenue est plus large que prévu

Avant d'implémenter la réparation par le compte-rendu, contrôle de la cohérence interne de `sejours.csv` : la colonne `DureeSejour` concorde-t-elle avec l'écart entre ses propres dates d'admission et de sortie ?

Elle ne concorde qu'à 92,8 %. Surtout, **l'écart recalculé `DateSortie - DateAdmission` est valide sur la totalité des séjours** : il s'étend de 0,4 à 28 jours, sans aucune valeur négative, là où la colonne `DureeSejour` porte 20 valeurs négatives et des durées allant jusqu'à 99 jours.

Confrontation des trois sources sur les 582 séjours dont le compte-rendu mentionne une durée :

| Comparaison | Concordance |
|---|---|
| Compte-rendu vs colonne `DureeSejour` | 95,2 % |
| Compte-rendu vs écart recalculé sur les dates | **98,5 %** |

Et sur les cas litigieux, la convergence est totale : pour les 12 séjours à durée négative couverts par un compte-rendu, la valeur du texte égale l'écart recalculé dans 100 % des cas. Même chose sur les durées extrêmes — `DureeSejour = 99` face à un écart de 6 jours et un compte-rendu mentionnant 6 jours.

```
SEJ-000051   DureeSejour  -3   ecart recalcule  3   compte-rendu  3j
SEJ-000085   DureeSejour  99   ecart recalcule  6   compte-rendu  6j
SEJ-000499   DureeSejour  99   ecart recalcule  2   compte-rendu  2j
```

**Diagnostic : la colonne `DureeSejour` est la donnée corrompue ; les dates d'admission et de sortie sont saines.** Contrairement aux tables filles, `sejours.csv` est cohérent en interne.

**Décision révisée : recalculer systématiquement la durée de séjour à partir de `DateSortie - DateAdmission`, et écarter la colonne `DureeSejour`.** Cette réparation couvre les 878 séjours au lieu des 28 que permettait le compte-rendu, et le texte clinique sert de source de confirmation indépendante plutôt que de source de réparation.

Cette révision illustre un principe de méthode : avant de réparer une donnée depuis une source externe, vérifier si la source elle-même contient de quoi se corriger. La colonne dérivée était fausse, les données brutes dont elle dérive étaient justes.

### J1-12 · Bornes de référence biologiques — quatrième anomalie non annoncée

En préparant l'harmonisation des unités, constat sur les colonnes `ValeurReferenceBas` et `ValeurReferenceHaut` : elles annoncent **les mêmes bornes pour les deux unités d'un même panel**.

| Panel | Unité | Valeurs observées (p05–p95) | Bornes annoncées |
|---|---|---|---|
| Créatinine | mg/L | 2,3 – 16,6 | 60 – 110 |
| Créatinine | µmol/L | 40 – 999 | 60 – 110 |
| Hémoglobine | g/L | 90,7 – 151,9 | 12 – 16 |
| Hémoglobine | g/dL | 9,0 – 15,1 | 12 – 16 |

Les bornes correspondent à la plage µmol/L pour la créatinine et à la plage g/dL pour l'hémoglobine, quelle que soit l'unité de la ligne. Un indicateur « valeur hors bornes » construit sur ces colonnes classerait la totalité des créatinines en mg/L comme effondrées, et une bonne part des hémoglobines en g/L comme massivement élevées.

**Décision : ne pas utiliser les colonnes de bornes du fichier source.** Les plages de référence sont redéfinies par panel dans l'unité canonique, au sein de `config.py`. Les facteurs de conversion sont vérifiés sur les médianes observées plutôt que repris de mémoire : créatinine 10,25 mg/L × 8,84 donne 90,6 contre 93,9 µmol/L observés ; hémoglobine 11,84 g/dL × 10 donne 118,4 contre 121,5 g/L observés.

Unités canoniques retenues : µmol/L pour la créatinine, g/L pour l'hémoglobine — les unités majoritaires du jeu et les usages hospitaliers français courants.

### J1-13 · Construction du pipeline et contrôle de non-fuite

Pipeline implémenté en six modules : `config` (constantes justifiées), `loading` (chargement sans correction), `audit` (mesure sans correction), `clean` (corrections adossées à l'audit), `blocks` (registre de blocs) et `assemble` (assemblage filtré par étage).

**Choix de conception : l'étage est une propriété déclarée du bloc, pas une consigne.** Chaque bloc déclare s'il relève de l'étage A (disponible à la sortie) ou B (postérieur). L'assemblage filtre sur cette déclaration. L'étanchéité temporelle devient une propriété du code plutôt qu'une vigilance à exercer à chaque ajout de source — c'est précisément le type d'erreur qu'un pipeline extensible doit rendre impossible par construction.

Résultat de l'assemblage sur l'étage A : **786 séjours × 133 variables**, 141 réadmissions (17,9 %), 575 patients. L'étage B produit 11 variables et couvre 11 séjours, conforme à la mesure de J1-06.

**Contrôle de non-fuite.** Corrélation absolue de chaque variable numérique à la cible : maximum à 0,378 pour le nombre de comorbidités, suivi du nombre de pathologies chroniques (0,373) et de l'âge (0,357). Aucune valeur n'approche les niveaux qui trahiraient une fuite. Le classement est cliniquement plausible — comorbidités, chronicité, âge, polymédication — ce qui constitue un second contrôle, qualitatif.

**Modèle de contrôle** avant toute optimisation, découpage groupé par patient, cinq plis :

| Modèle | PR-AUC | ROC-AUC |
|---|---|---|
| Prédiction constante (taux de base) | 0,178 | 0,496 |
| `HistGradientBoostingClassifier` par défaut | 0,368 | 0,766 |

Le modèle double l'aire sous la courbe précision-rappel par rapport au taux de base. L'ordre de grandeur est cohérent avec la littérature sur la prédiction de réadmission, où les ROC-AUC publiés se situent typiquement entre 0,65 et 0,75. Un score nettement supérieur aurait dû éveiller le soupçon d'une fuite résiduelle plutôt que satisfaire.

Ce résultat n'est pas le modèle final : il établit que la chaîne fonctionne et que le signal survit à l'assemblage.

### J1-14 · Comparaison de modèles et calibration

Quatre modèles confrontés sur le même protocole — découpage groupé par patient, stratifié, cinq plis, prédictions hors échantillon.

| Modèle | PR-AUC | ROC-AUC | Brier |
|---|---|---|---|
| Gradient boosting | **0,428** | 0,772 | **0,126** |
| Forêt aléatoire | 0,414 | 0,785 | 0,155 |
| Régression logistique | 0,353 | 0,740 | 0,195 |
| Référence (taux de base) | 0,178 | 0,496 | 0,147 |

**Décision : retenir le gradient boosting.** Il gère nativement les valeurs manquantes, ce qui est déterminant ici — les blocs biologie et texte sont renseignés à 55 % et 51 % pour des raisons structurelles, et une imputation généralisée fabriquerait de la donnée sur près de la moitié du tableau.

Observation qui mérite d'être signalée : le score de Brier de la référence (0,147) est **meilleur** que celui de la forêt aléatoire (0,155) et de la régression logistique (0,195). La pondération `class_weight="balanced"` employée par ces deux modèles gonfle les probabilités prédites et détruit leur calibration. Un modèle peut donc mieux classer tout en produisant des probabilités moins fiables — distinction qui compte dès lors qu'on veut choisir un seuil sur ces probabilités.

**Calibration du modèle retenu**, par quintile de risque prédit :

| Risque prédit moyen | Réadmissions observées | Séjours |
|---|---|---|
| 0,022 | 0,044 | 158 |
| 0,053 | 0,076 | 157 |
| 0,092 | 0,115 | 157 |
| 0,187 | 0,223 | 157 |
| 0,434 | 0,439 | 157 |

L'accord est bon, avec une légère sous-confiance dans les tranches basses. Les probabilités sont directement interprétables comme des risques, ce qui permet de raisonner le seuil en termes cliniques. Aucune recalibration n'a été jugée nécessaire.

### J1-15 · Seuil de décision — un arbitrage métier, pas un hyperparamètre

Le seuil ne s'optimise pas : il se décide. Il dépend du coût relatif des deux erreurs — une réadmission non anticipée face à une alerte inutile.

Conséquences opérationnelles sur les 786 séjours (141 réadmissions réelles) :

| Rapport de coût | Seuil | Sorties signalées | Réadmissions rattrapées | Alertes inutiles |
|---|---|---|---|---|
| 2 | 0,23 | 193 (25 %) | 81 (57 %) | 112 |
| 3 | 0,18 | 234 (30 %) | 92 (65 %) | 142 |
| **5** | **0,11** | **347 (44 %)** | **112 (79 %)** | **235** |
| 8 | 0,08 | 426 (54 %) | 121 (86 %) | 305 |
| 12 et au-delà | 0,05 | 566 (72 %) | 133 (94 %) | 433 |

**Constat de méthode : l'argument économique seul mène à l'absurde.** Une réadmission représente un séjour hospitalier, soit plusieurs milliers d'euros ; une visite infirmière à domicile, une centaine. Le rapport économique réel avoisine 20 à 30, ce qui saturerait le seuil à 0,05 et conduirait à signaler 72 % des sorties. Aucun service ne peut renforcer trois sorties sur quatre.

La contrainte qui borne réellement le seuil n'est donc pas le coût mais la **capacité de suivi du service**. Le formuler ainsi change la nature de la question posée au groupement hospitalier : non pas « combien vaut une réadmission évitée », mais « combien de sorties pouvez-vous accompagner ».

**Décision : rapport de coût 5, seuil 0,11.** Le système fonctionne comme un outil de dépistage : on privilégie la sensibilité, le tri fin restant à l'évaluation clinique. 79 % des réadmissions sont anticipées, au prix de 235 alertes qui ne se concrétiseront pas. La valeur est inscrite dans `config.py` avec sa justification, et la table de sensibilité complète figure au notebook — le groupement peut déplacer le curseur en connaissance de cause.

### J1-16 · Apport marginal des sources — le résultat le plus important, et le moins flatteur

Le jury demandera ce que les onze sources apportent réellement. Mesure de l'apport marginal, à protocole identique :

| Périmètre | Variables | PR-AUC |
|---|---|---|
| Âge seul | 1 | 0,412 |
| Âge + nombre de pathologies | 2 | 0,414 |
| Bloc démographie seul | 15 | 0,399 |
| Administratif + codage (SIH + PMSI) | 38 | 0,380 |
| Pipeline complet (11 sources) | 133 | **0,428** |

**L'âge seul atteint 0,412. Le pipeline complet atteint 0,428.** Dix sources et 132 variables supplémentaires apportent +3,9 % en relatif. Deux périmètres intermédiaires font même *moins bien* que l'âge seul : les variables ajoutées apportent plus de bruit que de signal.

L'importance par permutation confirme : `demo_age` pèse 0,293, presque le double de la deuxième variable (`demo_nb_pathologies`, 0,156). **76 variables sur 133 ont une contribution nulle ou négative.**

> Ces trois valeurs sont issues d'un tirage par permutation, donc légèrement variables d'une exécution à l'autre — un essai à dix répétitions donnait 77 variables non contributives et un poids de 0,291. L'ordre de grandeur et la hiérarchie, eux, sont stables. Les chiffres cités ici sont ceux du notebook exécuté.

Test décisif chez les moins de 65 ans, population où l'âge ne discrimine plus (taux de base 4,4 %, n = 225) : l'âge seul obtient 0,045, le pipeline complet 0,057. Le modèle n'apprend pratiquement rien d'autre que l'âge.

**Interprétation.** Ce n'est pas un défaut d'architecture mais une conséquence directe de la corruption temporelle établie en J1-06. Ce qui, sur des données réelles, porterait le signal clinique — une CRP qui remonte la veille de la sortie, une saturation qui se dégrade, une tension qui décroche — a été détruit par la désynchronisation des horodatages. Il ne reste que ce qui ne dépend pas du temps : l'âge et la charge de comorbidités.

**Ce résultat n'est pas dissimulé, il est le cœur de la démonstration.** L'architecture est correcte et vérifiée, les blocs s'exécutent, l'extensibilité est prouvée. Ce que le pipeline ne peut pas faire, c'est extraire un signal que les données ne contiennent plus. La distinction entre « le système ne marche pas » et « la donnée ne porte pas le signal » est précisément ce qu'un concepteur doit être capable d'établir — et le même pipeline, sur des données cohérentes, laisserait parler les blocs DPI.

### J1-17 · Équité — trois disparités, et une bonne nouvelle

Performance par sous-population au seuil retenu.

**Disparité majeure liée à l'âge.** Le modèle signale 95,5 % des séjours de patients de 85 ans et plus, avec un rappel de 0,98. Chez les moins de 65 ans, le rappel tombe à 0,20 et la PR-AUC à 0,055 pour un taux de base de 0,044. Le dispositif est inopérant sur cette population : quatre réadmissions sur cinq n'y sont pas anticipées. Conséquence directe de J1-16.

**Les patients isolés sont moins bien servis alors qu'ils sont plus à risque.** Les patients vivant seuls présentent un taux de réadmission de 22,6 % contre 15,6 % pour les patients entourés, mais leur rappel est de 0,678 contre 0,878. Le modèle détecte moins bien la population la plus vulnérable — disparité inverse de celle qu'on souhaiterait.

**Les soins continus sont le service le moins bien couvert.** Taux de réadmission le plus élevé (25,2 %) et rappel le plus faible (0,667).

**Bonne nouvelle sur l'axe que l'énoncé désignait comme le plus sensible.** Aucune disparité territoriale : le rappel est de 0,788 en désert médical contre 0,796 en densité normale, et la précision y est même supérieure. Aucune disparité selon l'indice de défavorisation non plus (rappels de 0,81, 0,81 et 0,76 selon le tiers). Le choix de n'utiliser que des indicateurs agrégés à la commune, imposé par l'énoncé, n'a pas introduit de biais territorial.

L'écart selon le sexe est modéré (rappel 0,833 chez les femmes contre 0,724 chez les hommes) et mérite d'être surveillé sans constituer à ce stade un motif de rejet.

**Décision : documenter ces disparités comme des limites d'emploi opposables**, et non les corriger par une pondération. Rééquilibrer artificiellement le modèle sur les moins de 65 ans reviendrait à lui faire produire des alertes sans fondement dans les données. La conclusion honnête est que le dispositif ne doit pas être déployé seul sur cette population — c'est une restriction de périmètre, pas un défaut à masquer.

---

## 24/08/2026 — Séance 2 : figures, et une hypothèse invalidée

### J2-01 · Les décès ne sont pas une fuite de cible — cinquième anomalie

En construisant la figure destinée à illustrer la fuite de cible de J1-07, le graphique a contredit son propre titre. Le taux de réadmission des séjours terminés par un décès n'est pas nul.

| Mode de sortie | Séjours | Réadmissions | Taux |
|---|---:|---:|---:|
| Transfert | 144 | 28 | 19,4 % |
| Domicile | 480 | 89 | 18,5 % |
| **Décès** | **92** | **17** | **18,5 %** |
| HAD | 75 | 12 | 16,0 % |
| EHPAD | 87 | 12 | 13,8 % |

**17 des 92 séjours terminés par un décès sont marqués `Readmission30j = 1`.** Le taux des décès est indiscernable de celui des sorties à domicile.

Vérification complémentaire : **26 séjours débutent après la date de décès du même patient**, concernant 25 patients distincts.

**Ce que j'avais écrit était faux.** J'avais qualifié les décès de fuite de cible en raisonnant cliniquement — un patient décédé ne peut pas être réadmis, donc la cible vaut zéro par construction — sans jamais mesurer. Le raisonnement clinique est juste ; l'hypothèse sur les données ne l'était pas.

Une fuite de cible suppose que la variable **porte de l'information sur la cible**. Ici les décès portent exactement le taux de base : ils n'apprennent rien au modèle. Le retrait des 92 séjours ne déplace le taux global que de 18,0 % à 17,9 %, ce qui confirme l'absence de fuite — une vraie fuite aurait déplacé le taux nettement.

**C'est donc une cinquième anomalie de cohérence**, et la plus frappante cliniquement : le générateur n'a appliqué aucune contrainte de mortalité. Elle rejoint la désynchronisation des horodatages (J1-06) dans la même famille — le générateur a tiré les variables indépendamment les unes des autres, sans faire respecter les règles qui les lient.

**La décision d'exclure les décès est maintenue, avec une justification différente.** Ce n'est plus « éviter une fuite » mais **définir la population sur laquelle la question se pose** : « ce patient sera-t-il réadmis dans les trente jours » n'a pas de sens pour un patient qui vient de mourir. L'exclusion relève de la définition du périmètre, pas de la prévention d'une fuite.

> **La leçon de méthode, et elle vaut d'être dite au jury.** Un raisonnement clinique correct ne dispense pas de vérifier qu'il s'applique aux données qu'on a. J'ai posé une contrainte comme acquise au lieu de la mesurer — sur des données synthétiques, aucune contrainte métier n'est garantie tant qu'on ne l'a pas contrôlée. C'est précisément le type de contrôle que la figure a rendu visible en trois secondes, là où le tableau de chiffres était resté sous les yeux sans être lu.

### J2-02 · Neuf figures, et ce qu'elles ont changé

Le notebook était entièrement tabulaire. Neuf figures ajoutées, chacune portant **un** constat et lisible sans son paragraphe.

| Figure | Constat porté |
|---|---|
| Incohérence des décès | Le taux des décès est indiscernable des autres modalités |
| Désynchronisation | Moins de 2 % des lignes filles dans la fenêtre de leur séjour |
| Couverture de télésurveillance | La fenêtre J+0 à J+7 est un trait fin dans une année de mesures |
| Harmonisation des unités | Deux populations disjointes deviennent une distribution |
| Constantes impossibles | Les zones hors bornes physiologiques, et ce qu'elles contiennent |
| Gradient clinique | Le classement par diagnostic principal |
| Complétude des blocs | Biologie et texte partiellement renseignés, par construction |
| Âge et réadmission | 3 % chez les moins de 55 ans, 45 % chez les 85 ans et plus |
| Apport marginal | L'âge seul contre le pipeline complet |

**Choix de conception des figures.** Palette validée pour les daltonismes deutan et tritan avant d'être employée — bleu et orange en séries, rouge réservé aux zones défectueuses, gris pour les repères. Une seule échelle de valeur par graphique, jamais deux axes. Étiquettes directes plutôt que légendes quand une seule série est en jeu.

**Deux défauts corrigés après examen visuel des rendus.** Les histogrammes écrêtaient les valeurs extrêmes, ce qui les entassait dans la barre de bord et **créait un mode qui n'existe pas** — remplacé par une fenêtre d'affichage. Et les effectifs, posés sous les barres, percutaient les étiquettes d'axe — intégrés aux étiquettes elles-mêmes.

> Générer une figure ne suffit pas : il faut la regarder. Les deux défauts ci-dessus, comme l'erreur de J2-01, n'étaient visibles qu'à l'écran.

### J2-03 · Smart App Control — un motif, pas un incident

Second blocage de la même nature que celui de J1-09. Pendant l'exécution du notebook, Windows a signalé le blocage de `cd.cp312-win_amd64.pyd`, module compilé de `charset_normalizer`, rendant tout le paquet inimportable.

**Portée mesurée avant de corriger quoi que ce soit.** Aucune conséquence sur le livrable : le notebook s'exécute sans erreur, `requests` se rabat sur un avertissement, JupyterLab démarre. Le pipeline n'importe pas ce paquet — il n'arrive que comme dépendance transitive de `requests`, lui-même dépendance de l'infrastructure Jupyter.

Correctif identique à celui de scipy : **épingler `charset-normalizer==3.3.2`**, dont l'ancienneté a établi la réputation. Les versions 3.3.2 et 3.1.0 s'importent toutes deux ; 3.3.2 retenue comme la plus récente qui passe.

> **Ce qu'il faut en retenir pour la reproductibilité.** Deux épingles de version figurent dans `requirements.txt` — `scipy==1.16.2` et `charset-normalizer==3.3.2` — et ni l'une ni l'autre ne tient à un besoin fonctionnel. Elles contournent une protection Windows qui rejette les binaires trop récents pour avoir une réputation établie.
>
> Sur une machine sans Smart App Control, les versions courantes conviendraient. La conséquence pratique est qu'un `pip install --upgrade` sur ce poste **recassera l'environnement** : les épingles sont là pour ça, et leur motif est écrit ici pour qu'on ne les retire pas par inadvertance.

Le diagnostic vaut méthode : quand un binaire est bloqué, **mesurer d'abord ce qui casse réellement**. Ici le réflexe aurait été de réinstaller ou de désactiver la protection ; la mesure a montré qu'il n'y avait rien d'urgent à réparer, et qu'une épingle suffisait.

### J2-04 · Contre-vérification — les séjours post-mortem restent, et pourquoi

Une passe de contre-vérification indépendante — recalcul des affirmations depuis les CSV bruts, reproduction des modèles à graine fixée — a confirmé l'ensemble des chiffres publiés, et relevé deux faits que J2-01 n'avait pas épuisés.

**Premier fait : 24 des 26 séjours post-mortem sont dans la population d'étude.** Le filtre d'exclusion ne retire que les séjours dont `ModeSortie = Deces` ; les séjours qui *débutent après* le décès du patient (écart de 38 à 370 jours, médiane 124) n'en font pas partie. 24 sont donc restés, dont 4 marqués réadmis, dans 23 dossiers patients.

**Décision : les conserver, et le documenter.** L'argument est le suivant : sur un jeu de données dont la cohérence interne est cassée, rien ne permet de dire si c'est l'enregistrement du décès qui est erroné ou les séjours ultérieurs. Supprimer les seconds revient à décréter que le premier est vrai — un choix aussi arbitraire que l'inverse. Chaque séjour concerné reste, pris isolément, une observation valide de la question « cette sortie sera-t-elle suivie d'une réadmission ». L'impact a par ailleurs été mesuré avant de trancher : l'exclusion ramènerait la population à 762 séjours (137 réadmissions) et la PR-AUC à 0,412 — aucun changement de conclusion.

L'exclusion des 92 séjours `Deces` eux-mêmes reste en place : pour ceux-là, c'est la *ligne du séjour* qui déclare le décès, et la question posée à la sortie n'a pas de sens.

**Second fait : deux patients meurent deux fois.** `SEJ-000144`/`SEJ-000145` et `SEJ-000399`/`SEJ-000400` — deux séjours `Deces` chacun, et SEJ-000145 débute deux mois après le premier décès du même patient. Cela complète J2-01 : le générateur n'applique aucune contrainte de mortalité, pas même l'unicité du décès.

### J2-05 · Un caveat méthodologique sur le seuil

Relevé à la même contre-vérification : le seuil de 0,11 est sélectionné sur les prédictions hors-échantillon, puis ses conséquences (347 signalés, 112 rattrapés) sont rapportées **sur ces mêmes prédictions**. La sélection et l'évaluation partagent les données, ce qui introduit un optimisme léger — la procédure propre serait une sélection imbriquée, le seuil étant choisi dans chaque pli d'apprentissage et évalué sur le pli de test.

**Décision : documenter sans reprendre le calcul.** Avec 141 événements répartis sur cinq plis, une sélection imbriquée du seuil serait dominée par le bruit d'échantillonnage — le remède serait pire que le mal. Le biais est par ailleurs faible : le seuil retenu ne provient pas d'une optimisation fine mais d'un rapport de coût fixé a priori, et la table de sensibilité montre que les conséquences varient continûment autour de 0,11. En production, le seuil serait de toute façon réévalué sur les données courantes du groupement — c'est l'un des indicateurs de surveillance prévus.

### J2-06 · Quatre variables de la littérature, construites, mesurées — et écartées

Tentative d'amélioration du modèle par les indices standards de la littérature réadmission, implémentés comme quatre blocs d'extension branchés au pipeline sans toucher à l'existant (`extensions.py`) :

| Extension | Contenu | Source |
|---|---|---|
| `charlson` | indice de Charlson, 6 composantes présentes dans les codes CIM-10 (pondération Quan 2005) | diagnostics |
| `lace` | score LACE complet — durée, admission urgente, comorbidités, urgences à 6 mois | séjours + diagnostics + historique |
| `atc` | indicatrices des 12 classes thérapeutiques (diurétiques, antithrombotiques…) | medications |
| `bio_tendances` | pentes et variations des panels en temps relatif, même recalage que les constantes | biologies |

**Résultat au protocole standard (graine 42)** : référence 0,428 ; +charlson 0,438 ; +lace 0,426 ; +atc 0,422 ; +bio_tendances 0,419 ; les quatre ensemble 0,420. Seul Charlson semblait apporter.

**Contrôle de robustesse sur quatre graines** — le gain de Charlson ne survit pas :

| Graine | Référence | +Charlson | Écart |
|---:|---:|---:|---:|
| 42 | 0,428 | 0,438 | +0,009 |
| 7 | 0,386 | 0,386 | -0,000 |
| 123 | 0,369 | 0,369 | +0,000 |
| 2026 | 0,425 | 0,419 | -0,006 |

Écart moyen +0,001, gain positif sur une graine sur quatre. **Décision : aucune extension n'entre dans le modèle final.** Les quatre blocs restent dans le dépôt comme résultat négatif exécutable — et comme démonstration supplémentaire de l'extensibilité : quatre sources de variables ajoutées puis retirées sans toucher une ligne du pipeline.

Ce résultat **confirme** le diagnostic de J1-16 plutôt qu'il ne le contredit : Charlson et LACE raffinent le signal âge + comorbidités que le modèle capte déjà, et les tendances biologiques recalées ne trouvent rien parce que la dynamique a été détruite à la source (couverture de la pente CRP : 10 % des séjours). Le plafond est dans la donnée, pas dans le modèle.

**Constat annexe à assumer : la variabilité de partition.** La référence elle-même varie de 0,369 à 0,428 selon la graine du découpage — une incertitude de ±0,03, supérieure à tout effet d'extension mesuré. Le chiffre-titre de 0,428 se situe dans le haut de cette fourchette. La formulation honnête est : **PR-AUC de l'ordre de 0,37 à 0,43 selon la partition, soit environ le double du taux de base dans tous les cas** — et c'est cette formulation qui doit être servie au jury, pas le point haut seul.

### J2-07 · De l'architecture papier au système qui tourne

La note technique de Nicolas a reformulé la cible en une phrase — « à la fin d'un séjour, avec les données du patient, l'IA doit prévoir un meilleur suivi » — et pointé ce qui manquait au dossier : l'énoncé demande une **solution opérationnelle**, la section 12 n'offrait qu'un schéma. Elle a aussi enrichi l'étage B : la réévaluation à domicile ne vise pas qu'une visite, mais **l'adaptation de la médication et l'anticipation d'une réhospitalisation**.

Construit dans `api/` : le geste du module M0 (une API + une interface autour d'un modèle), appliqué au sujet d'examen.

| Pièce | Rôle |
|---|---|
| `entrainer.py` | fige le modèle final (mêmes hyperparamètres que le protocole), les données de démonstration et un `meta.json` de traçabilité — version, seuil, importance globale, note d'incertitude |
| `app.py` | FastAPI, quatre endpoints : info, liste de séjours, **score** (étage A), **télésurveillance** (étage B) |
| `index.html` | page de démonstration : jauge contre le seuil, facteurs du score, les deux phases affichées |

Trois choix de conception, dans la continuité du dossier :

- **Jamais un score nu.** La réponse de `/api/score` porte le risque calibré, le seuil et sa nature (un réglage de capacité, pas une propriété du modèle), les facteurs avec les valeurs du patient, la version du modèle — et un **avertissement automatique pour les moins de 65 ans**, la restriction de périmètre de la section 10 devenant un comportement du système plutôt qu'une ligne de documentation.
- **L'étage B répond au lieu de mentir.** L'endpoint existe, expose les décisions visées (médication, visite, réhospitalisation) et sa couverture réelle : 6 patients sur les 170 requis. Le système dit ce qui lui manque — c'est l'indicateur de surveillance de la section 12, rendu visible.
- **Le modèle servi est réentraîné sur toute la population**, mais les performances annoncées restent celles de la validation croisée, avec la fourchette de J2-06 (0,37–0,43) et non le seul point haut. Le `meta.json` l'écrit noir sur blanc.

Vérifié en conditions réelles : API démarrée, les quatre endpoints testés, l'avertissement jeune patient déclenché sur un séjour de 49 ans. Les artefacts (`modele.joblib`, parquets) sont régénérables par `entrainer.py` et exclus du dépôt.

### J2-08 · Le format de soutenance change tout — pitch, démo live, deck

La note de Nicolas précise le format réel de l'épreuve : **une société présente sa solution à un hôpital**. Le PowerPoint est le support principal, le notebook n'est là qu'en soutien, et le livrable est un projet complet — présentation, dossier, application fonctionnelle, code. La note demande aussi une démonstration vivante : un patient rentré chez lui, des capteurs qui émettent au fil du temps, et l'application qui lève une alerte à la réception.

**La démonstration live** (`api/simulation.py`) : un générateur de scénario produit le flux d'objets connectés d'un patient à domicile — fréquence cardiaque, SpO2, poids, activité — et une couche d'alerte l'évalue à chaque remontée. Le point de conception qui compte : le modèle appris de la phase 2 étant inentraînable (J1-06), la couche d'alerte v1 fonctionne par **règles cliniques explicites** — prise de poids ≥ 2 kg (rétention hydrique), SpO2 < 92 %, tachycardie de repos, effondrement d'activité — chacune portant sa justification clinique et sa préconisation. Deux signes sérieux simultanés déclenchent l'escalade : *anticiper une réhospitalisation en admission programmée, plutôt qu'un retour par les urgences*. C'est le fonctionnement réel des systèmes de télémédecine à leur lancement, et cela transforme la limite « étage B non entraînable » en trajectoire produit : règles d'abord, modèle appris quand la volumétrie existe.

Scénario « décompensation » vérifié en conditions réelles : deux jours calmes, désaturation au jour 5 (alerte téléconsultation), escalade au jour 6 (poids + désaturation), tableau complet au jour 7. Le scénario « convalescence normale » ne déclenche rien — le contraste fait la démonstration.

**Le deck** (`presentation/`) : douze diapositives générées par script (`python-pptx`), société fictive **Vigie Santé**, charte sobre reprenant la palette des figures. Structure : problème (18 % des sorties) → solution en deux phases → démo de l'application → démo de l'alerte en direct → preuves sans maquillage (8/10 anticipées, ×2 contre le hasard **sur toutes les partitions**, zéro écart territorial) → conformité → **« Ce que nous ne vous vendons pas »** — la diapositive des limites assumées, qui transpose au registre commercial l'honnêteté du dossier → déploiement → offre. Chaque affirmation chiffrée du deck provient du notebook.

Incident d'environnement en passant : troisième blocage Smart App Control (`lxml`, dépendance de python-pptx), résolu par la même épingle de version — `lxml==5.3.0`. Le motif de J2-03 se confirme.

### J2-09 · Le pipeline en deux notebooks — conformité au format attendu

Les notes de cadrage précisent la structure attendue du pipeline : **un notebook de nettoyage des données, un notebook d'entraînement avec répartition 80 % entraînement / 20 % test**, et un modèle classique — pas de LLM.

Livré : `01_nettoyage_donnees.ipynb` (chargement, audit résumé en quatre contrôles, nettoyage, assemblage, écriture de `donnees_propres.parquet`) et `02_entrainement_modele.ipynb` (chargement, répartition 80/20, entraînement, évaluation sur les 20 % jamais vus). Les deux sont **minces et branchés sur `src/`** — même code que le dossier technique, aucune duplication : le pipeline est une vue conforme du même système, pas une réécriture.

**Le point de méthode défendu : la répartition 80/20 est groupée par patient** (`GroupShuffleSplit`). Un 80/20 aléatoire simple aurait réintroduit la fuite par découpage documentée en J1-07 — 238 patients ont plusieurs séjours. Le notebook le vérifie par assertion : aucun patient des deux côtés.

**Résultat sur les 20 % jamais vus : PR-AUC 0,372** (taux de base 0,191, soit ×2,0), ROC-AUC 0,717. Au seuil 0,11 : 26 réadmissions rattrapées sur 29 (rappel 90 %), 94 alertes inutiles. Deux lectures à en tirer :

- le chiffre tombe **dans la fourchette annoncée en J2-06** (0,37–0,43 selon la partition) — la cohérence entre le protocole simple exigé et le protocole complet du dossier est vérifiée, pas affirmée ;
- avec seulement ~29 événements dans le jeu de test, un découpage unique est bruyant — c'est précisément pourquoi le dossier technique évalue en validation croisée. Le notebook d'entraînement porte cette mise en garde en clair, pour que le chiffre soit lu dans sa fourchette et non comme une mesure isolée.

### J2-10 · La démo ne doit pas jouer le même film pour tout le monde

Critique de Nicolas sur la première version de la simulation : tous les patients décompensaient à l'identique — même trajectoire, même alerte au jour 5. Exact : la graine et les valeurs de base étaient fixes, la démonstration sentait le scénario en boîte.

**Correction : la trajectoire est désormais propre à chaque patient.**

- **Graine dérivée de l'identifiant du séjour** — deux patients ne jouent jamais la même séquence, et chaque séquence reste rejouable à l'identique (indispensable pour répéter la démo).
- **Valeurs de repos issues du profil réel** — âge, sexe, pathologies : un BPCO part avec une SpO2 de repos abaissée (~2,6 points), un insuffisant cardiaque avec une fréquence de repos plus haute, l'activité quotidienne décroît avec l'âge et la charge de comorbidités.
- **Le type de décompensation suit les pathologies** — respiratoire pour un BPCO, cardiaque pour une insuffisance cardiaque, mixte sinon — et **le score de phase 1 module le début et la vitesse** : un patient à haut risque décompense plus tôt et plus vite. Les deux phases de l'application se répondent : le score de sortie n'est plus un chiffre isolé, il conditionne ce que la télésurveillance observera.

Vérifié sur trois profils : escalade au jour 5 pour une patiente de 97 ans à 82 % de risque (BPCO + insuffisance cardiaque, type cardiaque), au jour 11,75 pour une patiente à 6 %, au jour 10,25 en dégradation mixte pour un patient sans pathologie chronique. Le scénario « convalescence normale » reste muet sur 10 jours, y compris sur le profil le plus fragile — le contraste entre les deux boutons fait la démonstration.

### J2-11 · Le mur de supervision — la cohorte entière sous les yeux

Demande de Nicolas : remplacer la sélection patient par patient par une **vue d'ensemble** — toute la cohorte à domicile, chaque patient évoluant en direct, les alertes qui surgissent. C'est l'écran d'usage réel : un poste de soins, pas un formulaire.

Construit : `POST /api/cohorte/demarrer` met douze patients sous surveillance (un tiers haut risque, un tiers moyen, un tiers bas — sélectionnés sur le score de phase 1), `GET /api/cohorte/tick` avance toute la cohorte d'une remontée et renvoie états et événements nouveaux. Côté interface, une grille de tuiles à code couleur (calme / modéré / sérieux / escalade pulsante), un journal d'alertes horodaté en jours à domicile, et un clic sur une tuile ouvre la vue patient détaillée.

**Le point de conception qui fait la démonstration : qui décompensera n'est pas dévoilé.** Chaque patient tire son scénario avec une probabilité liée à son risque de phase 1 — on surveille sans savoir, comme en réalité. Résultat observé sur douze jours simulés : les quatre patients à 85 % de risque ont tous escaladé (J+4,25 à J+8,25), le BPCO à risque moyen est en alerte sérieuse, les quatre patients à 1 % sont restés calmes. Les deux phases de l'application se répondent à l'écran : le score de sortie annonce ce que la télésurveillance observera.

**Le garde-fou qui va avec, écrit dans l'interface même** : cette corrélation est simulée par construction. La démonstration illustre le fonctionnement opérationnel du dispositif — flux, règles, alertes, escalade — elle ne prouve pas la validité prédictive du score, qui est établie (et bornée) dans le dossier technique. Confondre les deux serait s'exposer à la première question adverse venue.

Une escalade fige la trajectoire du patient : l'écran reste sur l'état qui a déclenché la décision, comme le ferait un tableau de bord réel en attente de prise en charge.

### J2-12 · L'historique d'alertes devient un vrai journal

Deux demandes de Nicolas sur le mur de supervision : l'historique ne doit pas se perdre, et il faut un repère de position du jour dans le fil.

**L'historique vit désormais côté serveur** (`JOURNAL_ALERTES`), jamais purgé : il traverse les rafraîchissements de page et les redémarrages de cohorte — chaque nouvelle mise sous surveillance y dépose un séparateur, et les événements manqués entre deux sondages du navigateur ne sont plus perdus. L'interface le relit intégralement (`GET /api/cohorte/journal`) et le restaure au chargement de la page. Chaque entrée embarque l'identité du patient (âge, diagnostic) au moment des faits — l'historique reste lisible même sans la cohorte en mémoire.

**Le repère de jour** : à chaque jour franchi à domicile, un marqueur « Jour N à domicile » s'insère dans le fil, et le jour courant s'affiche en badge sur le mur. Le journal se lit comme une main courante de service : les jours qui passent, les alertes qui tombent, les escalades qui figent.

Vérifié : 26 entrées après deux cohortes successives — la première session complète (alertes de J+3,5 à J+6, deux escalades) reste intégralement lisible sous la seconde.

### J2-13 · Le curseur de position du jour — le présentateur pilote le temps

Retour de Nicolas sur la version précédente : le bouton « play » n'est pas pratique — il faut un **curseur de position du jour**. Le flux automatique imposait son rythme ; en démonstration, c'est le présentateur qui doit amener le jury au moment voulu.

**La mécanique change : la trajectoire des 30 jours est précalculée au chargement**, pour toute la cohorte (déterministe — les graines dérivent des identifiants). Le curseur navigue ensuite librement dedans, **en avant comme en arrière**, sans latence : `GET /api/cohorte/etat?pas=N` renvoie l'état exact du mur et le journal cumulé jusqu'à cette position. Vérifié : revenir au jour 5 après être allé au jour 30 rend un état identique au bit près.

L'interface : un curseur pleine largeur (J+0 → J+30), des boutons ‹ › pour avancer d'une remontée, la lecture automatique reléguée en option (elle ne fait qu'avancer le curseur), et « ↻ cohorte » pour retirer douze patients. La cohorte est prête dès l'ouverture de la page — plus aucun bouton à presser avant de démontrer.

Cette version **remplace** la mécanique de flux de J2-12 ; ce qu'elle en conserve : l'historique complet reste navigable (tirer le curseur, le journal suit, marqueurs de jour compris) et rien ne disparaît — il est précalculé. En démonstration : ouvrir la page, tirer le curseur au jour 5 devant le jury, montrer la première escalade, reculer, ré-avancer. Le temps appartient au présentateur.

### J2-14 · La fiche patient — l'état à la date du curseur, et la main courante des relevés

Demande de Nicolas : cliquer sur un patient doit ouvrir **sa fiche à la date en cours**, avec **un tableau des relevés de métriques**.

La vue patient est refondue en fiche contextuelle, synchronisée sur la position du curseur du mur :

- **l'identité** (âge, sexe, service, diagnostic, durée du séjour) ;
- **la phase 1** : score de sortie, jauge contre le seuil, préconisation, avertissement moins de 65 ans — les facteurs du score repliés dans un volet ;
- **la phase 2 à J+x** : badge d'état (rien à signaler / alerte modérée / signe sérieux / réhospitalisation à anticiper), les événements d'alerte du patient avec leurs préconisations, et **le tableau des remontées** — une ligne par relevé capteurs (FC, SpO2, poids, pas), les valeurs hors norme colorées selon les mêmes seuils que les règles d'alerte, la ligne de référence « repos » en pied de tableau.

Les boutons de simulation individuelle disparaissent : le mur pilote le temps, la fiche le lit. Un patient hors cohorte garde sa fiche de phase 1, avec la mention explicite qu'il n'est pas sous surveillance.

**Un défaut débusqué par le test avant d'être vu à l'écran** : après une escalade, l'état gelé du mur se dupliquait en lignes identiques dans le tableau. Corrigé avec la sémantique clinique juste — la main courante s'arrête à l'escalade, car le patient est pris en charge et sa télésurveillance s'interrompt ; la fiche l'écrit en toutes lettres. La tuile du mur, elle, reste figée sur l'état d'escalade, comme un tableau de bord en attente de prise en charge.

### J2-15 · Visualiser l'arbre de décision — sans mentir sur ce qu'il est

Question de Nicolas : peut-on visualiser l'arbre de décision pour un patient ? La réponse honnête d'abord : **« l'arbre » du modèle n'existe pas** — le gradient boosting agrège 300 arbres, chacun illisible isolément. En montrer un seul serait un mensonge visuel.

La réponse du métier : un **arbre de substitution** (*surrogate*) — un arbre unique de profondeur 4, entraîné à imiter les prédictions du modèle complet, livré **avec sa fidélité mesurée** : R² 0,541, accord de 78,5 % à la décision de seuil. L'arbre explique environ la moitié du comportement du modèle, et ce chiffre est affiché à côté de l'arbre — jamais l'arbre sans son degré d'approximation. La profondeur est choisie par mesure (4 retenue contre 3 uniquement parce que le gain de fidélité dépasse 0,04 de R²).

Dans la fiche patient, un volet dessine l'arbre en SVG avec **le chemin de ce patient surligné**, ses valeurs affichées à chaque nœud traversé, et les deux estimations côte à côte (arbre vs modèle) :

```
P-001 (97 ans)  : âge > 75,8 → 4 pathologies > 1,5 → 6 prescriptions > 5,5 → feuille 68 % (35 séjours) · modèle 82 %
P-002 (49 ans)  : âge ≤ 75,8 → pas de BPCO → 2 classes de médicaments ≤ 4,5 → feuille 5 % (303 séjours) · modèle 2 %
```

**Et la question qui a suivi — l'arbre évolue-t-il avec les 30 jours ? Non, par construction** : il explique le score de phase 1, calculé à la sortie sur des entrées figées. Ce qui vit avec les jours, c'est la logique de phase 2 — et elle est désormais visualisée aussi : un panneau « règles de décision évaluées à J+x » dans la fiche montre chaque règle avec son calcul en direct (« poids 63,1 − 60,5 = +2,6 kg ≥ 2 → DÉCLENCHÉE », compteur d'escalade 2/2), qui s'allume et s'éteint au fil du curseur. Les deux natures de décision du système — apprise et figée à la sortie, réglée et vivante à domicile — sont chacune visualisées pour ce qu'elles sont.

**Point de données acté au passage** : les quatre capteurs de la simulation (fréquence cardiaque, SpO2, poids, activité en pas) sont **exactement les quatre types** que contient la source `objets_connectes` du sujet — la démo est fidèle à la donnée fournie, et ajouter un capteur en production tient en une règle et une colonne.

### J2-16 · L'arbre en pleine page, le curseur sur la fiche

Deux retours d'usage de Nicolas, même séance.

**L'arbre était illisible dans le volet de la fiche** — seize feuilles compressées dans la largeur d'une carte. Il vit désormais dans une **page dédiée** (`/arbre?sejour=…`, ouverte dans un nouvel onglet) : plein écran, **zoom à la molette centré sur le pointeur, déplacement au glisser**, boutons ± / ajuster, et un bouton « suivre le chemin » qui cadre automatiquement la vue sur le parcours du patient à un zoom lisible — c'est d'ailleurs la vue d'ouverture. La géométrie est dessinée généreusement (la page est faite pour être zoomée), et la note de fidélité reste dans l'en-tête : l'arbre ne voyage jamais sans son degré d'approximation.

**Le curseur de position du jour arrive sur la fiche patient.** On peut faire évoluer l'état du patient jour par jour sans repasser par le mur : le badge J+x, l'état, les règles évaluées et la main courante des relevés se rafraîchissent au fil du curseur. Un choix de conception : **une seule position du temps, deux poignées** — le curseur de la fiche et celui du mur sont synchronisés, et revenir à l'onglet mur recale la grille sur la position où la fiche s'est arrêtée. Deux horloges désynchronisées auraient été un piège de démonstration.

### J2-17 · « 85 % au score, 42 % dans l'arbre ? » — la question qui devait arriver

Nicolas, devant l'arbre du patient P-393 : le score annonce 85 %, la feuille d'arrivée dit 42 %. Lequel croire ?

**Les deux — ils ne mesurent pas la même chose.** La feuille donne la **moyenne des 77 séjours** qui répondent identiquement aux quatre questions de l'arbre ; le score vient du modèle complet et de ses 133 variables. Dans ce groupe « âgé, multi-pathologies, polymédiqué », il y a des 20 % et des 85 % — l'arbre ne sait pas les départager, le modèle si : ce patient cumule ce que l'arbre ne regarde pas (97 ans bien au-delà du seuil de 75,8, la natrémie, le reste). L'écart entre feuille et score **est** le R² de 0,541 affiché — l'arbre explique la moitié du comportement du modèle, pas la totalité. Si quatre questions suffisaient, le modèle serait inutile.

**L'interface prêtait pourtant le flanc** : la feuille disait « risque 42 % » comme s'il s'agissait du patient. Corrigé pour rendre la confusion impossible — la feuille dit désormais « **moyenne** 42 % · des 77 séjours de ce chemin », une pastille distincte affiche « CE PATIENT : 85 % (modèle) » sous la feuille d'arrivée, et l'en-tête explicite la distinction en toutes lettres.

La formulation à servir au jury est actée : *l'arbre montre le chemin, le modèle donne le score ; l'écart entre les deux est le prix de la lisibilité, et il est mesuré.*

### J2-18 · Les courbes d'évolution — la dégradation se voit avant de se lire

Amélioration retenue par Nicolas dans la revue des suites possibles : des courbes d'évolution dans la fiche patient. Le tableau des relevés était exact mais demandait une lecture ligne à ligne ; la pente d'une SpO2 qui plonge vers l'escalade est une image, et c'est l'image que retient un jury.

**Quatre mini-graphiques** (SpO2, poids, fréquence cardiaque, activité), insérés entre l'état du patient et le panneau des règles, rafraîchis au fil du curseur comme le reste de la fiche. Les règles de représentation tenues :

- **une échelle par grandeur, jamais d'axe double** — quatre grandeurs aux unités incompatibles font quatre petits multiples, pas un graphique à deux ordonnées ;
- **le seuil de la règle d'alerte tracé sur chaque courbe** (tirets rouges : SpO2 92, FC 105, poids de repos + 2 kg, 35 % de l'activité de repos) et la **valeur de repos du patient** en pointillés discrets — le graphique montre exactement ce que les règles calculent ;
- **les franchissements en points rouges** sur la ligne, la valeur courante en étiquette directe au bout (rouge si la règle est déclenchée) ;
- ligne fine, repères effacés, pas de légende — une série par graphique, le titre suffit.

Vérification syntaxique du JavaScript des deux pages par analyse (sans exécution) avant livraison — le navigateur n'est pas le premier endroit où découvrir une parenthèse manquante.

### J2-19 · La vue patient ne montre que le patient en cours

Retour de Nicolas : l'onglet « Vue patient » affichait encore une liste de quarante séjours — dont des patients hors cohorte — héritée de la toute première version de l'application, d'avant le mur de supervision. Elle faisait double emploi et brouillait le parcours.

Retirée. Le parcours devient univoque : **le mur est la seule porte d'entrée** — on clique une tuile, la fiche s'ouvre en pleine largeur, et l'onglet porte le nom du patient en cours (« Vue patient — P-393 »). Sans sélection, l'onglet affiche simplement l'invitation à passer par le mur. Un écran, un rôle : le mur pour choisir et surveiller, la fiche pour comprendre un patient.

### J2-20 · Le cockpit clinique, et des cohortes qui se renouvellent

**La fiche patient est refondue en cockpit** — l'empilement vertical manquait de hiérarchie. Agencement retenu parmi trois proposés : un **bandeau d'en-tête** qui concentre tout l'essentiel au-dessus de la ligne de flottaison (identité, badge du score de sortie, badge d'état à J+x, curseur du jour, avertissement moins de 65 ans), puis deux colonnes — **à gauche le visuel** (courbes d'évolution, tableau des relevés), **à droite la décision** (jauge et préconisation de sortie, facteurs, accès à l'arbre, alertes, règles évaluées). C'est la disposition d'un dossier patient informatisé : l'œil trouve l'état en une seconde, la preuve à gauche, la conduite à tenir à droite.

**Et un vrai bug attrapé par l'usage** : le bouton « ↻ cohorte » redonnait toujours les douze mêmes patients. Cause : la sélection était entièrement déterministe — les k plus hauts risques, une tranche fixe du milieu, la queue du classement. Remplacée par un **tirage aléatoire stratifié** (un tiers tiré parmi les hauts risques, un tiers au milieu, un tiers en bas) dont la séquence est ancrée sur un compteur : chaque clic renouvelle la cohorte, mais le premier tirage est identique à chaque lancement — la démonstration reste répétable, la variété est au rendez-vous. Vérifié : trois clics, trois cohortes disjointes, chacune avec son tiers en alerte de sortie.

### J2-21 · La fenêtre d'attente d'admission — l'escalade ne coupe plus les capteurs

Question de Nicolas : une fois la réhospitalisation demandée, les capteurs cessent d'émettre — est-ce normal ? C'était le choix implémenté (J2-14), et sa question a mis le doigt sur la simplification : dans la réalité, entre la décision d'escalade et l'admission effective, le patient reste chez lui **sous surveillance** — c'est précisément la fenêtre où elle compte le plus.

**Le cycle de vie s'enrichit d'un état.** L'escalade n'interrompt plus rien : elle ouvre une **fenêtre d'attente d'admission** d'un jour simulé (quatre remontées) pendant laquelle les capteurs continuent d'émettre, la tuile reste rouge pulsante avec la mention « en attente d'admission — surveillance maintenue », et la main courante s'allonge. Au terme de la fenêtre, l'événement **« Admission réalisée — fin de la télésurveillance »** clôt l'épisode : le patient passe à l'état **« admis »** (tuile bleue apaisée, badge 🏥), et c'est l'admission — non l'alerte — qui arrête les remontées.

Refactorisation au passage : le statut est désormais calculé par la simulation elle-même (`calme / modéré / sérieux / escalade / admis`) et porté par chaque remontée, au lieu d'être reconstruit en aval — une seule source de vérité pour le mur, la fiche et le journal.

Vérifié de bout en bout sur un patient de cohorte : `calme ×16 → sérieux ×27 → escalade ×4 → admis`, les deux événements au journal, 47 relevés fenêtre incluse, et quatre patients à l'état « admis » sur le mur à J+30. Le cycle de vie affiché correspond maintenant à celui d'un vrai dispositif : surveiller, alerter, **accompagner jusqu'à la prise en charge**, clore.

### J2-22 · Reconstruire la télémétrie ? Testé dans les règles — et prouvé impossible

Question de Nicolas : les modules de formation enseignent la reconstruction de données manquantes — l'horodatage de la télémétrie est-il complètement perdu, ou peut-on le corréler aux hospitalisations ?

La reconstruction a deux conditions de légitimité, toutes deux testées par la mesure.

**Chercher un ancrage redondant.** Première découverte, qui corrige la lecture initiale : le flux de chaque patient n'est pas éparpillé sur un an — c'est **un épisode dense de ~13 jours cadencé à 6 heures** (médiane 34 mesures), le profil exact d'un programme de télésuivi post-sortie, simplement posé au mauvais endroit du calendrier. Mais aucun ancrage ne permet de le repositionner : le plus grand trou du flux ne ressemble pas à la durée du séjour (corrélation 0,111, concordance 12 % — le niveau du hasard), et neuf patients seulement montrent des mesures d'activité quasi nulles. **L'horodatage absolu est irrécupérable.**

**Tester si la structure relative est porteuse.** Même sans dates absolues, si le bloc était bien l'épisode post-sortie et que le générateur y avait mis du signal, son **contenu en temps relatif** devrait corréler avec la réadmission. Test sur les 178 patients équipés (35 réadmis) : pentes et niveaux des quatre grandeurs, comparaison des issues, modèle en validation croisée. Verdict sans appel — p de 0,40 à 0,97 sur les huit variables, PR-AUC 0,158 **sous** le taux de base (0,197), ROC-AUC 0,38. **Le contenu ne porte aucun signal d'issue.**

La règle des trois cas du dossier s'énonce désormais complète :

| Donnée | Redondance | Structure porteuse | Reconstruction |
|---|---|---|---|
| Durées de séjour | trois sources | — | ✅ triangulée |
| Constantes vitales | — | le temps relatif porte le signal clinique | ✅ recalées |
| Télémétrie | aucune | testée : rien | ❌ prouvé impossible |

Conséquence sur J1-06 : le refus de resynchroniser reposait sur un principe (« fabriquer la donnée qui manque n'est pas une correction ») ; il repose maintenant sur une **preuve empirique** — même resynchronisée à la perfection, cette donnée ne prédirait rien. La reconstruction n'est ni un réflexe ni un tabou : c'est une hypothèse qui se teste, et ici le test dit non.

### J2-23 · Le plan B de la démonstration — captures automatisées dans le deck

Toute démonstration live peut tomber le jour J pour une raison idiote — projecteur, port occupé, mise à jour système. Le deck embarque désormais son plan de secours : **trois diapositives annexes avec des captures réelles de l'application**, placées après la clôture — invisibles en déroulé normal, prêtes si la démo ne peut pas tourner. La même histoire se raconte alors sur images, sans casser le fil.

**Les captures sont automatisées** (`presentation/captures.py`) : Playwright pilote un navigateur sur l'application, amène le mur à J+5, sélectionne dynamiquement le patient le plus parlant à cette position (priorité escalade > sérieux), ouvre sa fiche puis son arbre, et enregistre trois PNG en haute densité. Si l'application évolue encore, une commande régénère captures et deck — le plan B ne peut pas se périmer.

Les trois images ont été **validées à l'œil avant intégration** : le mur montre deux patients en alerte sérieuse et le journal jalonné ; la fiche montre la SpO2 de P-347 plongeant sous le seuil (89 %) avec la règle « DÉCLENCHÉE » ; l'arbre montre le chemin surligné et la double lecture « moyenne 26 % des 29 séjours » / « CE PATIENT : 71 % (modèle) » — les trois moments-clés du pitch, figés en secours. Deck final : 15 diapositives (12 + 3 annexes).

### J2-24 · L'IA entre dans la décision de réhospitalisation — la bonne façon

Question de Nicolas : peut-on remplacer la décision « mathématique » de réhospitalisation par une décision IA — cela aurait-il plus d'impact en soutenance ?

Deux pistes écartées d'abord, parce qu'elles auraient détruit la crédibilité du dossier : un LLM décisionnaire (exclu par la consigne, et indéfendable en santé), et un modèle appris de phase 2 (**prouvé impossible en J2-22** — 2 cas positifs, contenu télémétrique sans signal ; le jury le démonterait en une question).

**La réponse retenue : le modèle appris de phase 1 entre dans la boucle de décision de phase 2.** Le seuil d'escalade n'est plus uniforme — il est **personnalisé par le score IA de sortie** : chez un patient dont le score calibré atteint 30 % (trois sur dix comme lui reviennent), *un seul* signe sérieux suffit à anticiper la réhospitalisation ; deux restent requis sinon. C'est l'état de l'art des programmes de télésuivi (la vigilance s'ajuste au risque), c'est cliniquement défendable, et cela n'invente rien : le score utilisé est réel, appris, validé.

L'effet est mesurable et visible : P-347 (BPCO, score 71 %) escalade désormais à J+4,25 sur sa seule désaturation — là où la règle uniforme attendait un second signe — et l'alerte porte son motif en toutes lettres : « 1 signe sérieux — seuil personnalisé à 1 par le score de sortie (vigilance renforcée) ». Le panneau des règles affiche le seuil du patient, la préconisation d'escalade a été neutralisée (le motif variable vit dans l'événement), et **les deux phases se répondent jusque dans la décision** — plus seulement à l'écran.

Captures du plan B et deck régénérés dans la foulée (une commande — l'automatisation de J2-23 a servi le jour même) et validés à l'œil : l'annexe fiche montre précisément l'escalade en vigilance renforcée.

### J2-25 · Y aurait-il un autre modèle pour le score ? — six familles, mesurées

Question de Nicolas : un autre modèle d'IA pourrait-il calculer le score de réadmission ? Réponse par la mesure, au protocole standard (découpage groupé par patient, cinq plis) — deux familles ajoutées aux quatre du dossier :

| Modèle | PR-AUC | ROC-AUC | Brier |
|---|---:|---:|---:|
| **Gradient boosting (retenu)** | **0,428** | 0,772 | **0,126** |
| Forêt aléatoire | 0,414 | 0,785 | 0,155 |
| LightGBM | 0,386 | 0,765 | 0,146 |
| Régression logistique | 0,353 | 0,740 | 0,195 |
| Réseau de neurones (MLP, 64-32) | 0,307 | 0,700 | 0,191 |
| Référence (taux de base) | 0,178 | 0,496 | 0,147 |

**LightGBM** — le standard industriel du boosting — est équivalent au modèle retenu à la variance de partition près (0,37–0,43) : même famille, mêmes forces ; aucun motif de changer, l'implémentation scikit-learn gérant en plus nativement catégorielles et manquants sans dépendance supplémentaire.

**Le réseau de neurones est mesurablement inférieur** : −0,12 de PR-AUC et une calibration dégradée (Brier 0,191). Attendu — 786 lignes tabulaires sont trop peu pour du deep learning — mais désormais démontré : la réponse « pourquoi pas un réseau de neurones ? » n'est plus un argument d'autorité, c'est une expérience du dossier.

Deux familles restent écartées sans mesure, pour cause d'infaisabilité ou de consigne : les **modèles de survie** (Cox, forêts de survie) modélisent le *délai* avant réadmission — or le jeu ne fournit que le oui/non à 30 jours, pas la date de réadmission ; et les **transformers tabulaires pré-entraînés** (TabPFN), état de la recherche pour les petits jeux, contreviennent à la consigne « modèle classique » — cités comme perspective, pas utilisés.

Le choix du gradient boosting sort de cette question renforcé : premier sur la métrique principale et la calibration, parmi six familles comparées au même protocole.

## 01/09/2026 — Séance 3 : les hyperparamètres, mesurés

### J3-01 · Optuna — la recherche a eu lieu, et elle est écartée par ses propres chiffres

Question de Nicolas : les hyperparamètres ont-ils été optimisés avec Optuna ?

Non. Vérification faite dans le code : ni Optuna, ni `GridSearchCV`, ni `RandomizedSearchCV` — aucune recherche. Le gradient boosting était réglé à la main (quinze feuilles, L2 à 1,0, pas d'apprentissage à 0,05, arrêt anticipé), avec des valeurs prudentes adaptées à un petit jeu. Défendable, mais le dossier ne l'expliquait nulle part, et « je n'ai pas cherché parce que ça n'aurait rien donné » est une affirmation non démontrée — exactement ce que ce travail refuse partout ailleurs. La question se mesure.

**Protocole — validation croisée imbriquée.** Optuna en échantillonnage TPE, cinquante essais par pli externe, chaque essai évalué par validation croisée à quatre plis internes : deux cent cinquante essais, un millier d'entraînements. La recherche ne voit que les plis internes ; l'évaluation se fait sur des plis externes qu'aucun essai n'a touchés. Sans cette imbrication, le score rapporté serait celui de la sélection elle-même. Les deux réglages sont confrontés sur les mêmes plis externes.

Premier signe que le protocole est juste : le réglage manuel y ressort à **0,4282** en mise en commun — exactement la valeur du tableau de la section 7. Le nouveau protocole reproduit le chiffre du dossier au lieu de le contredire.

**Le résultat, en deux agrégations qui se contredisent.**

| Agrégation | Manuel | Optuna |
|---|---:|---:|
| Mise en commun des plis externes | **0,4282** | 0,4189 |
| Moyenne des cinq plis | 0,4656 | 0,4815 |
| Écart-type entre plis | 0,085 | 0,111 |

Sur la métrique du dossier, la recherche perd neuf millièmes ; en moyennant les plis, elle en gagne seize, avec une dispersion plus large. Quand le sens du résultat dépend de la façon de compter, il n'y a pas de progrès à constater.

**Ce qui tranche vraiment, ce sont les réglages retenus.** Pas d'apprentissage de 0,012 à 0,105, régularisation L2 de 0,012 à 8,2 — facteur sept cents — itérations de 100 à 450. Cinq échantillons de six cents séjours désignent cinq optima incompatibles : la recherche épouse le bruit du pli qui l'a guidée. Le pli 5 le paie, sa configuration élue y perdant 0,059 face au réglage manuel. Et les scores atteints en interne (0,38 à 0,48) sont systématiquement plus flatteurs que ceux obtenus en externe — le surajustement de la sélection, rendu visible par l'imbrication.

**Décision : le réglage manuel est conservé** — non par défaut, mais parce que la mesure le justifie. Avec 786 séjours et 141 réadmissions, la variance d'échantillonnage (0,37 à 0,43 selon la partition) domine tout gain de réglage accessible. La conclusion est bornée à l'échelle du jeu, pas à la méthode : `src/tuning.py` reste dans le dépôt, paramétrable, réexécutable en une commande sur un volume supérieur.

Le dossier gagne au passage ce qui lui manquait objectivement : les hyperparamètres sont désormais décrits et leur choix justifié par une expérience, et non par une préférence.

## 07/09/2026 — Séance 4 : la relecture qui a trouvé une correction muette

### J3-02 · Une correction annoncée qui ne s'exécutait pas — et tout ce qui a bougé

Demande de Nicolas : relire le dossier entier, de zéro, avec un œil critique, sans rien modifier, et rendre un rapport. Méthode : tout lire — énoncé, grille d'évaluation, les trois notebooks, le journal, le code, l'API, le deck, le vault — puis **exécuter le code en lecture seule** là où la lecture ne tranchait pas. C'est l'exécution qui a trouvé le problème principal.

**La normalisation des sentinelles textuelles ne s'exécutait pas.** `clean.normalize_sentinels` et `audit.missing_report` reconnaissaient une colonne de texte par le test `dtype == object`. Sous pandas 3, le type par défaut des chaînes est `str`, et le test était toujours faux : la fonction parcourait les colonnes sans en toucher une. Les symptômes étaient sous les yeux depuis le 24/08, dans les sorties mêmes du notebook :

- la cellule 3.3 « colonnes dont les manquants sont invisibles à `isna()` » affichait **un tableau vide**, juste au-dessus de l'encart qui affirmait trois colonnes ;
- l'analyse d'équité par sexe comportait une ligne **« Non renseigné », 29 séjours**, jamais commentée ;
- trois variables dérivées étaient constantes : `demo_situation_inconnue` valait zéro partout, `demo_medecin_traitant_declare` et `demo_aidant_declare` valaient un partout — 65 et 105 patients « Non renseigné » comptés comme déclarés. L'argument RGPD « seule leur présence est encodée » reposait sur deux variables mortes ;
- la correction 1 du tableau de la section 4 n'était pas appliquée, et l'audit du notebook 01 n'affichait aucune ligne `patients`.

Les chiffres de J1-04 (Sexe 21, SituationFamiliale 67) avaient été établis par un comptage direct, pas par la fonction — c'est pour cela que le journal disait vrai et que le code ne faisait rien.

**Correctif** : une fonction `is_text_column` qui accepte `object` comme `str`, utilisée par les deux modules. L'audit trouve désormais **cinq colonnes** au lieu de trois : `PersonneAPrevenir` 105, `SituationFamiliale` 67, `MedecinTraitant` 65, `RegimeAssurance` 38, `Sexe` 21. Les variables reprennent vie : 81 situations inconnues, 700 médecins traitants déclarés sur 786 séjours, 655 aidants déclarés.

**Second défaut de méthode, corrigé dans la foulée** : l'importance par permutation était mesurée sur un seul pli — environ 157 séjours et 28 réadmissions. Elle est désormais moyennée sur les cinq plis, avec sa dispersion.

**Tout a été réexécuté** — les trois notebooks, la recherche Optuna, le contrôle sur quatre graines, les artefacts de l'API, les captures, le deck. Ce qui a bougé, et ce qui n'a pas bougé :

| Mesure | Avant | Après |
|---|---:|---:|
| Gradient boosting — PR-AUC / ROC-AUC / Brier | 0,428 / 0,772 / 0,126 | 0,428 / 0,774 / 0,126 |
| Forêt aléatoire | 0,414 / 0,785 / 0,155 | 0,415 / 0,777 / 0,156 |
| LightGBM | 0,386 / 0,765 / 0,146 | 0,373 / 0,764 / 0,150 |
| Réseau de neurones | 0,307 / 0,700 / 0,191 | 0,355 / 0,720 / 0,175 |
| Régression logistique | 0,353 / 0,740 / 0,195 | 0,350 / 0,735 / 0,199 |
| Seuil 0,11 — signalés / rattrapées / inutiles | 347 / 112 / 235 | 345 / 113 / 232 |
| Âge seul / pipeline complet | 0,412 / 0,428 | 0,412 / 0,428 |
| Fourchette sur quatre graines | 0,369 – 0,428 | 0,380 – 0,428 |
| Charlson, écart moyen sur quatre graines | +0,001 (1 graine sur 4 positive) | +0,006 (3 sur 4, jamais au-delà de +0,008) |
| Optuna vs manuel, mise en commun | −0,009 | −0,002 |
| Importance de l'âge / variables nulles | 0,293 / 76 sur 133 (un pli) | 0,175 / 47 sur 133 (cinq plis) |
| Arbre de substitution — R² / accord au seuil | 0,541 / 78,5 % | 0,560 / 75,6 % |
| Équité — rappel isolés, désert médical vs dense | 0,678 ; 0,788 vs 0,796 | 0,695 ; 0,788 vs 0,806 |
| Sans les 24 séjours post-mortem (J2-04) — PR-AUC | 0,412 | 0,403 |

**Pourquoi le modèle retenu n'a pas bougé.** Le gradient boosting traite une valeur manquante d'une variable catégorielle comme une modalité à part : pour lui, « Non renseigné » et une absence sont la même partition. Les deux indicateurs ressuscités pèsent peu. Les modèles qui imputent — réseau de neurones, régression logistique, forêt — ont bougé davantage, le réseau de neurones de cinq centièmes : l'indicateur de manquant que son imputation ajoute lui convient mieux qu'une modalité de plus. Le classement des six familles est inchangé en tête ; le réseau de neurones passe devant la régression logistique.

**Décision maintenue sur Charlson, mais assumée en connaissance de cause.** Le gain est réel et reproductible sur trois graines, et il ne dépasse jamais huit millièmes : dix fois sous la variance de partition. L'extension reste hors du modèle final ; la phrase « le gain ne survit pas » est remplacée par « le gain est réel et minuscule ».

**Ce que la relecture a corrigé d'autre, dans le même mouvement.**

- Une fausse alerte de la relecture, à consigner aussi : le rapport affirmait que les données de l'examen n'étaient pas dans le dépôt. Elles y sont depuis le 24/08. Le contrôle avait interrogé git avec un chemin dont l'accent de « Santé » était composé autrement que dans l'index (forme NFC contre forme NFD), et git n'avait rien trouvé. Le README dit désormais en clair que les données sont versionnées, et la leçon rejoint celle des sentinelles : une absence s'affirme après avoir vérifié le vérificateur.
- Le deck racontait une démo qui n'existait plus : la diapositive 5 renvoyait à la liste de séjours supprimée en J2-19, la diapositive 7 affirmait que « ce sont les patients que le score désignait qui s'aggravent » — faux sur le tirage de référence, où deux escalades sur cinq touchent des patients non signalés et où un patient signalé à 21 % reste calme. Les deux diapositives et la trame orale décrivent maintenant la démo telle qu'elle se joue, et assument le point : le score règle la vigilance, il ne désigne pas ; un dépistage à huit sur dix en laisse passer deux, la simulation aussi.
- Le « ×10 » de la diapositive 2 n'avait pas de source, et le journal J1-15 avançait « une centaine d'euros » pour une visite infirmière. Sources retenues : coût moyen d'un séjour de l'ordre de 2 100 € dans le public (ATIH, campagne tarifaire 2026 — la page ATIH refuse les requêtes depuis ce poste, le chiffre est repris d'un relais qui la cite), et nomenclature infirmière AMI 3,15 € plus déplacement 2,75 €. Le rapport économique est donc d'au moins 10 pour un accompagnement complet, et jusqu'à 100 pour une seule visite — ce qui renforce l'argument de J1-15 plutôt qu'il ne l'affaiblit. Le troisième chiffre de la diapositive, « 0 outil aujourd'hui », affirmait quelque chose sur le client sans preuve ; remplacé par le gradient d'âge, 45 % de réadmission après 85 ans contre 3 % avant 55.
- « Traçabilité complète pour chaque score émis » sur la diapositive 9 contredisait la section 11.2, qui la laisse à l'établissement, et l'API ne journalise rien. Reformulé : chaque score sort avec sa version, ses facteurs et son seuil, à journaliser dans le SIH.
- Le notebook 02 affichait 120 sorties signalées sur 152 — 79 % — sans un mot, là où le dossier en signale 44 % et l'application 34 %. Cause : arrêt anticipé à 14 itérations sur un jeu de validation d'une centaine de séjours, probabilités resserrées autour du taux de base. Une mise en garde l'explique : le seuil est un réglage de capacité, il se recale sur le modèle déployé.
- Les captures du plan B sont prises à J+7, au moment de la première escalade du tirage de référence, et non plus à J+5 où plus rien ne se passe.
- Deux comptages incohérents : la section 3 annonçait « quatre autres » anomalies pour cinq, la section 14 « deux décisions révisées » pour trois.

**La leçon vaut plus que le correctif.** Une correction annoncée se vérifie sur sa sortie, pas sur son code — et un test de non-régression sur les tables produites (nombre de modalités par catégorie, variables constantes, colonnes attendues) l'aurait attrapé le premier jour. C'est le manque objectif de ce dossier : aucun test automatisé, aucune intégration continue, là où les modules précédents de la formation en avaient. Noté pour la suite, pas traité ici.

## 08/09/2026 — Séance 5 : l'explication individuelle du score

### J3-03 · « Explicable et contestable » — l'application ne le prouvait pas, elle le dit maintenant variable par variable

Demande de Nicolas : améliorer l'interface de démonstration pour le pitch (raccourcis présentateur, synthèse de la cohorte, moins de texte à l'écran), puis, sur mon signalement, fermer un point faible : la fiche patient affichait sous « facteurs du score » l'**importance globale** du modèle — la même liste pour tous les patients, seules les valeurs changeaient. Le deck promettait une alerte « explicable et contestable » ; l'écran montrait une propriété du modèle, pas une explication du score de ce patient.

**Ce qui a été fait.** Chaque séjour reçoit désormais ses **contributions de Shapley** : pour chaque variable, ce qu'elle ajoute ou retire au score, en points de probabilité, avec la propriété d'additivité — séjour moyen (18,6 %) + somme des 133 contributions = score du patient, vérifié à 10⁻⁹ près sur les 786 séjours par le script qui les produit. Sur le tirage de référence, P-092 (69 %) : l'âge et le nombre de pathologies chroniques pèsent chacun une dizaine de points, la SpO2 moyenne et les antécédents d'hospitalisation cinq ; P-780 (58 ans, 8 %) : l'âge retire treize points. La fiche affiche les huit contributions les plus fortes en barres divergentes, le reste agrégé, et la phrase qui borne l'interprétation : ce que le modèle attribue à chaque variable pour ce séjour, pas une causalité.

**Un chemin écarté, et pourquoi.** L'explicateur d'arbres de la bibliothèque `shap` (TreeExplainer) est la voie naturelle pour un gradient boosting, exacte et instantanée. Sur ce modèle il répond sans erreur, et faux : la somme des contributions retombe à −2,5 en logit quand le score vaut +0,8, l'âge n'apparaît pas chez un patient de 78 ans, et les contributions sont quasi identiques d'un patient à l'autre. Cause : les coupures catégorielles natives de `HistGradientBoostingClassifier` (onze variables en `category`, gérées par masques de bits) ne sont pas lues par la conversion de shap. L'explicateur **par permutations**, indépendant du modèle, est retenu à la place : plus coûteux (0,3 s par séjour, huit permutations, fond de cent séjours), mais correct par construction, et le contrôle d'additivité est dans le script. Même leçon qu'en J3-02 : une sortie qui ne lève pas d'erreur n'est pas une sortie juste — se vérifier sur ce qu'on produit.

**Où ça vit.** `api/expliquer.py`, appelé en fin d'`entrainer.py` (l'étape 2 du README suffit toujours), produit `contributions.parquet` et `contributions.json` dans les artefacts, non versionnés comme les autres. La route `/api/score` renvoie l'explication à côté des facteurs globaux, qui restent servis en secours si les artefacts manquent. `shap`, `numba` et `llvmlite` rejoignent `requirements.txt`. Les captures du plan B et le deck sont régénérés ; deux formulations du deck sont ajustées (« la part de chaque variable dans le score », « contester une alerte sur un fait précis »).

**Ce que ça ne change pas.** Le modèle, ses performances, le seuil : rien n'est réentraîné, l'explication se calcule après coup sur le modèle figé. La question du jury « pourquoi pas la régression logistique, plus explicable ? » garde sa réponse (les manquants), et gagne un argument : l'explication individuelle est disponible sur le modèle retenu, sans sacrifier la gestion native des manquants.


## 21/09/2026 — Séance 6 : le troisième axe de vigilance, mesuré

### J3-04 · L'énoncé désignait trois axes de biais ; la section 10 en mesurait deux

Constat à la relecture du dossier contre le référentiel : la section 10 **cite** la phrase de l'énoncé — « l'âge, le territoire et le profil d'utilisation des soins » — puis analyse l'âge, le territoire, la défavorisation, l'isolement, le service et le sexe. Deux des trois axes demandés sont traités, deux axes non demandés sont ajoutés, et le troisième manque. Un jury qui a l'énoncé sous les yeux le voit en trente secondes, et le critère « biais potentiels identifiés » relève d'une compétence où la moitié des points se joue encore sur l'écrit et l'oral.

Le plus gênant est que la donnée était là depuis le premier jour : le bloc `historique` produit sept variables de recours antérieur aux soins, et il est le seul dont les dates soient cohérentes (J1-06). Rien n'empêchait de mesurer cet axe — il n'avait simplement pas été pensé comme un axe d'équité.

**Deux lectures plutôt qu'une.** « Profil d'utilisation des soins » ne se résume pas à un volume : l'intensité du recours et sa nature disent deux choses différentes. Les sous-populations ajoutées sont donc l'intensité du recours antérieur — aucun événement (256 séjours), un à deux (421), trois et plus (109), bornes posées sur la distribution observée — et le passage antérieur aux urgences, oui (243) ou non (543).

**Le résultat, et un contre-exemple à ma propre hypothèse.** Le rappel tombe à 0,629 chez les patients sans antécédent de recours, contre 0,866 et 0,833 dans les deux autres groupes : plus d'une réadmission sur trois y échappe au dépistage. J'ai d'abord supposé que c'était la disparité d'âge sous un autre nom — les patients sans historique étant les patients jeunes que le modèle ne sait pas traiter (9.2, 10.1). **Le contrôle dit non** : les trois groupes ont le même âge moyen (71,3, 72,2, 71,8 ans) et la même part de moins de 65 ans, environ 29 %. La disparité est propre à cet axe, et elle croise plutôt celle de l'isolement — le groupe sans antécédent compte 38,7 % de patients vivant seuls contre 23,9 % chez les usagers intensifs.

Vérifier une explication commode avant de l'écrire évite d'ajouter au dossier une affirmation fausse qui aurait sonné juste.

**Ce qui n'est pas un biais, et qu'il fallait aussi mesurer.** La crainte usuelle sur ce critère est inverse : qu'un modèle apprenne « consommateur de soins égale patient à signaler » et sur-alerte les patients les plus suivis. Ce n'est pas le cas ici — le taux signalé suit le taux observé (34,4 % pour 13,7 %, 48,0 % pour 19,5 %, 50,5 % pour 22,0 %) et la précision reste stable. Le passage antérieur aux urgences ne crée pas non plus de disparité (rappel 0,851 contre 0,777, PR-AUC 0,433 contre 0,429). Un axe d'équité se mesure dans les deux sens : l'absence de discrimination est un résultat, pas un silence.

**Portée.** Une seule cellule de code change — le dictionnaire des sous-populations —, et le contrôle de non-régression le confirme : sur les 51 cellules de code réexécutées, une seule voit sa sortie bouger. Aucun chiffre du dossier n'est affecté, le modèle n'est pas retouché. La section 10 gagne une sous-section 10.2, l'ancienne devenant 10.3.

**Ce que ça change pour les équipes.** Une conséquence opérationnelle s'ajoute à celle des moins de 65 ans : une absence d'antécédent de recours ne vaut pas un faible risque. C'est une restriction de lecture, à énoncer aux utilisateurs plutôt qu'à corriger par pondération — même raisonnement qu'en J1-17.


### J3-05 · Des seuils qu'on ne peut pas franchir n'en sont pas

La section 12.4 listait cinq indicateurs de surveillance avec, en regard, des formules du type « écart durable au taux d'apprentissage » ou « déplacement de la moyenne prédite ». Relu contre le critère — « des indicateurs de performance **et seuils associés** adaptés aux attendus sont définis » —, le tableau ne définissait rien : aucune de ces cinq lignes ne dit à partir de quand on agit. Une surveillance dont on ne peut pas dire si elle a déclenché est une intention.

**Le seuil ne se décrète pas, il se calcule.** Avant de poser des nombres, il fallait savoir ce que le volume d'un établissement permet de distinguer du hasard. Le calcul est ajouté au notebook : sur un mois de séjours, l'intervalle de confiance à 95 % du taux de réadmission vaut à lui seul **±9,3 points** ; il faut un trimestre pour descendre à ±5,3, et six mois pour ±3,8. Autrement dit, un tableau de bord mensuel qui alerterait sur deux points d'écart passerait son temps à crier au loup.

**D'où une distinction que le tableau ne faisait pas : la fréquence de relevé n'est pas la fenêtre d'évaluation.** On relève tous les mois — c'est le rythme de l'exploitation —, mais on juge sur une fenêtre glissante assez large pour que le seuil soit franchissable. Les bornes retenues pour le taux de réadmission, `[14,2 ; 21,7] %`, sont exactement l'intervalle de confiance à six mois autour des 17,9 % observés. Elles ne sont pas choisies, elles sont déduites.

**Un décalage structurel, à énoncer plutôt qu'à subir.** La cible n'existe qu'à trente jours, et le codage du séjour suivant prend encore quelques semaines : un séjour de janvier n'est évaluable qu'au début mars. Tout indicateur de *performance* traîne donc environ deux mois de retard, sans qu'aucune organisation puisse y changer quoi que ce soit. C'est ce qui justifie de surveiller en priorité les indicateurs d'*entrée* — distribution des scores, remplissage des blocs — qui sont disponibles immédiatement et bougent avant la performance. La référence de remplissage par bloc est mesurée à la mise en service (biologie 55 %, comptes-rendus 51 %, historique 95 %, le reste complet) ; une baisse de dix points y signale une source amont qui se dégrade bien avant que le modèle ne s'en ressente.

**Le réentraînement devient un événement, pas une date.** Cinq déclencheurs, dont quatre mesurables — deux relevés consécutifs hors bornes, un rappel sous 0,70 sur douze mois, un indice de stabilité au-delà de 0,25, l'ajout ou la réparation d'une source — et un cinquième de sécurité : une revue annuelle même en l'absence d'alerte, pour que « rien n'a sonné » ne se confonde pas avec « rien n'a vieilli ». Les indicateurs eux-mêmes sont réinterrogés une fois par an, et la revue pose une question inconfortable : un indicateur qui n'a jamais alerté est-il rassurant, ou ne mesure-t-il rien ?

**Ce que cette section ne prétend pas être.** Les six indicateurs sont définis, leurs seuils calculés, leurs actions associées — mais rien n'est automatisé : ni tâche planifiée, ni tableau de bord alimenté, ni évaluation continue rattachée au dépôt. C'est une spécification de surveillance, et le notebook le dit désormais explicitement plutôt que de laisser croire à un dispositif en fonctionnement. La première brique d'industrialisation à prévoir est l'exécution périodique de ce calcul sur les séjours clos, avec l'historisation de ses résultats — avant tout enrichissement du modèle.


### J3-06 · Une architecture sans coût ni interlocuteur n'est pas une architecture

Troisième écart relevé en confrontant le dossier au référentiel. La section 12 décrivait **une** architecture, correctement — où le score se calcule, ce qu'il restitue, comment une source s'ajoute — mais laissait trois critères sans réponse : les contraintes économiques des **différents scénarios**, la consultation des **acteurs** pour préciser les contraintes de généralisation, et, en santé, l'obligation d'hébergement certifié. Le mot HDS n'apparaissait nulle part dans un dossier qui traite exclusivement de données de santé.

**Mesurer avant de dimensionner.** Le réflexe aurait été d'écrire des ordres de grandeur plausibles. J'ai préféré mesurer, et les chiffres changent la conclusion : le modèle pèse 232 Ko, s'entraîne en une seconde et demie, et score un séjour en quelques millisecondes. Pour cinquante mille séjours annuels — l'ordre de grandeur d'un groupement régional — le calcul cumulé représente **une minute et demie de processeur par an**. Un serveur dédié allumé en continu consommerait près de quatre cent mille fois l'énergie du calcul qu'il servirait.

Ce résultat tranche deux questions d'un coup. L'éco-conception, d'abord : elle ne se joue pas sur le choix de l'algorithme mais sur la décision de **mutualiser le calcul plutôt que de dédier une machine** — un gradient boosting plus frugal ne changerait rien, une infrastructure dédiée changerait tout. L'architecture ensuite : aucune charge n'impose une infrastructure particulière, donc le choix se décide sur la conformité et l'exploitation, pas sur la puissance.

**Trois scénarios, séparés par une frontière juridique et non technique.** L'article L. 1111-8 du code de la santé publique impose la certification HDS pour l'hébergement de données de santé **pour le compte d'un tiers** ; un établissement qui héberge ses propres données n'y est pas soumis. Cette ligne sépare le scénario sur site mutualisé, l'hébergement chez un prestataire certifié, et l'hybride où seuls des agrégats de surveillance sortent. La charge mesurée désigne le premier, et c'est un résultat plutôt qu'une préférence.

**Le coût n'est pas là où on le cherchait.** Au réglage retenu, 44 % des sorties sont signalées : pour cinquante mille séjours, environ vingt-deux mille accompagnements par an, à cent cinquante ou deux cent cinquante euros pièce. Le poste soignant dépasse l'infrastructure de plusieurs ordres de grandeur. **Le dimensionnement de ce système est une décision de ressources humaines**, ce qui rejoint par un autre chemin la conclusion de J1-15 : la contrainte qui borne le seuil n'est pas le coût d'une réadmission, c'est la capacité de suivi du service. Deux raisonnements indépendants qui convergent valent mieux qu'un seul répété.

**Et les acteurs qu'on ne peut pas interroger.** Le critère demande que les acteurs métiers, techniques et le commanditaire soient consultés. Ils ne l'ont pas été, et ils ne pouvaient pas l'être : le cas d'usage repose sur un jeu synthétique, sans établissement réel derrière. Plutôt que de faire silence ou de simuler des entretiens, la section 12.7 pose les six questions qui restent ouvertes, nomme leur destinataire — DSI, DPO, cadre de santé, médecin DIM, direction financière — et dit ce que chaque réponse déciderait. Le délai de codage PMSI décale l'instant de calcul ; la capacité hebdomadaire du service fixe le seuil ; le taux de réadmission réel de l'établissement valide ou invalide les bornes de surveillance calculées en J3-05.

Une question ouverte, nommée et adressée, vaut mieux qu'une hypothèse tacite. C'est aussi ce qui permet de dire au jury ce qui relève du dossier et ce qui relève de l'établissement.

**Et une erreur commise en chemin, de la même famille que J3-02.** Les premiers ordres de grandeur avaient été mesurés hors du notebook, dans un script jetable : entraînement 1,5 s, 90 secondes de calcul annuel. Le texte de la section a été écrit sur ces valeurs. À la première réexécution dans le notebook, les mêmes mesures donnaient 0,2 s et une seconde — un facteur sept sur l'entraînement, cent sur l'inférence : mes mesures externes payaient l'initialisation des bibliothèques et le chargement des données, celles du notebook tournent à chaud. **Le texte affirmait donc des chiffres que la cellule juste au-dessus contredisait.**

Deux corrections en découlent. La première est de fond : l'extrapolation utilisait le temps de traitement **en lot** (18 µs par séjour), alors qu'en exploitation le score se calcule **séjour par séjour à la clôture** — c'est l'appel unitaire, quelque cent fois plus coûteux à cause du coût fixe de chaque appel, qui donne la charge réelle. La seconde est d'écriture : les durées d'exécution varient d'un facteur sept selon l'état de la machine, du cache et des fils d'exécution. Les citer au dixième de seconde dans un dossier dont le principe est que chaque affirmation corresponde à une sortie, c'est se condamner à mentir à la réexécution suivante. Le texte dit désormais « moins d'une seconde », « de l'ordre du quart d'heure par an », « quatre à cinq ordres de grandeur » — vrai sur n'importe quelle machine.

Troisième occurrence de la même leçon, sous une forme nouvelle : en J3-02 une correction ne s'exécutait pas, en J3-03 une bibliothèque répondait faux sans erreur, ici c'est le **commentaire qui s'est désolidarisé de sa propre sortie**. Un chiffre écrit à la main à côté d'un chiffre calculé finit toujours par diverger ; la parade est de ne mettre dans le texte que ce qui reste vrai quand le calcul change.


### J3-07 · Le dossier traitait le droit et appelait ça de l'éthique

Quatrième écart du même audit, et le plus embarrassant : la section 11 s'intitulait « cadre réglementaire et gouvernance » et traitait exclusivement du **droit** — article 9, article 22, secret médical, minimisation. Le critère demande autre chose : que les **chartes éthiques, européennes et françaises, soient connues et appliquées**. Aucune n'était citée. Un dossier peut être irréprochable au regard du RGPD et n'avoir jamais posé la question de ce qui est souhaitable.

**Le référentiel retenu, et pourquoi celui-là.** Les lignes directrices pour une IA digne de confiance, publiées en 2019 par le groupe d'experts mandaté par la Commission européenne, posent sept exigences que la plupart des chartes nationales ont reprises depuis. Elles ont l'avantage d'être **vérifiables une à une** plutôt que déclaratives : on peut dire, pour chacune, ce que le dossier oppose et ce qui lui manque. Trois autres corpus sont mentionnés sans être déroulés, parce qu'ils n'ajoutent ici aucune exigence nouvelle : les six principes de l'OMS pour la gouvernance de l'IA en santé, la grille d'analyse de la HAS, et les recommandations de la CNIL.

**L'exercice a produit quatre manques que je n'avais pas vus.** Confronter le dossier à une grille écrite par d'autres fait apparaître ce qu'une relecture de soi-même ne trouve pas :

1. l'explication du score est disponible pour le clinicien, mais **sa restitution au patient n'est spécifiée nulle part** — or c'est le patient que la décision concerne ;
2. aucune **procédure de recours** n'existe pour un patient qui contesterait son orientation, alors que le dossier revendique une alerte « contestable » ;
3. aucun **audit externe** n'est prévu, alors que l'exigence de responsabilité le demande explicitement ;
4. l'effet du dispositif sur la **charge de travail des équipes** n'est pas évalué — signaler 44 % des sorties n'est pas neutre pour un service, et c'est un impact sociétal au sens de la sixième exigence.

Deux autres manques étaient déjà connus et se retrouvent ici sous un autre angle : l'absence de tests automatisés au titre de la robustesse, et l'absence de cycle de vie documenté du jeu de données au titre de la gouvernance. Une grille éthique ne crée pas ces trous, elle les éclaire depuis un autre côté.

**Et deux qualifications que le dossier n'avait pas posées.** Le règlement européen sur l'intelligence artificielle classe à haut risque certains systèmes évaluant l'éligibilité à des services essentiels, de santé compris. La qualification de celui-ci n'est ni évidente — il n'attribue ni ne retire aucun droit — ni à écarter, puisque le score conditionne en fait l'accès à un accompagnement. Le règlement sur les dispositifs médicaux pose la question la plus déterminante : tout dépend de la **destination revendiquée**. Présenté comme un outil d'organisation des sorties, le système reste un outil de gestion ; présenté comme une aide à la décision clinique, il bascule dans le marquage CE médical. Le dossier retient délibérément la première formulation, mais la frontière est étroite — une plaquette commerciale imprudente suffirait à la franchir, et un système qualifié de dispositif médical devient de ce seul fait à haut risque au sens du règlement sur l'IA.

Ces deux questions ne se tranchent pas par un concepteur seul. Elles rejoignent la liste de 12.7 : à poser au juriste de l'établissement, au moment du cadrage contractuel, et pas après la mise en service.


### J3-08 · Le pipeline écrivait du parquet sans jamais dire pourquoi

Dernier écart de rédaction du même audit. Le critère demande que le **modèle de stockage soit justifié** au regard des types de données et des usages, et que le **cycle de vie du jeu de données soit documenté** puis soumis aux parties prenantes. Le pipeline écrivait `donnees_propres.parquet` depuis la première séance sans que rien n'explique ce choix, et aucune durée de conservation n'était écrite nulle part.

**Justifier après coup oblige à vérifier ce qu'on croyait savoir, et je m'y suis repris à deux fois.** Première mesure, menée sur le fichier écrit par le pipeline : aucune colonne catégorielle. J'en ai conclu que l'argument attendu — le format colonnaire préserve les types catégoriels — était faux ici, et j'ai rédigé en conséquence. Seconde mesure, cette fois **dans le notebook** : quarante et une colonnes sur cent trente-trois altérées par un aller-retour CSV, dont les onze catégorielles qui redeviennent du texte. Les deux mesures sont exactes ; elles ne portaient simplement pas sur le même objet.

L'explication est instructive. `01_nettoyage_donnees` écrit les onze variables en texte et dépose à côté la liste `colonnes_categorielles.txt` ; `02_entrainement_modele` relit et reconvertit explicitement depuis cette liste. Le typage catégoriel vit **hors** du fichier de données, dans une métadonnée versionnée avec le code — parce qu'une variable catégorielle est une décision de modélisation, révisable, et non une propriété de la donnée. Ce notebook-ci construit sa table en mémoire et conserve les catégories directement. Deux chemins, le même modèle, des contraintes différentes : rien n'était cassé, mais rien ne l'expliquait non plus.

Ce que la mesure établit au final : le format colonnaire protège deux choses, les **catégories** — que le modèle traite nativement, et dont la perte changerait son comportement et non sa seule consommation mémoire — et les **entiers nullables**, rabattus par le CSV sur des entiers simples. Cette seconde perte est sans effet aujourd'hui, ces colonnes ne portant aucun manquant ; la première valeur absente les ferait relire en flottant, et le modèle recevrait un type inconnu de lui. Une régression qui ne lèverait aucune erreur. Le volume, moitié moindre, n'est qu'un bénéfice accessoire.

**Une conclusion de méthode, la deuxième de la journée.** Mesurer hors du notebook pour écrire dans le notebook m'a fait écrire faux deux fois de suite — sur les temps d'exécution en J3-06, sur les types ici. Un script jetable ne travaille ni sur les mêmes objets ni dans le même état que le document. La règle qui s'impose : ce que le dossier affirme se mesure dans le dossier, et nulle part ailleurs.

**Trois natures de données, trois stockages.** L'erreur classique est de tout mettre en base, ou tout laisser en fichiers. Les données sources restent dans le SIH et ne sont pas recopiées — une copie serait un second traitement de données de santé à déclarer, sécuriser et héberger en HDS. Les tables intermédiaires dérivent entièrement des sources par un code versionné à graine fixe : les perdre ne coûte qu'un recalcul, elles n'ont donc ni à vivre en base ni à être sauvegardées. Les scores émis, eux, justifient une base relationnelle : interrogés à l'unité, joints au dossier, soumis à traçabilité longue. Les artefacts du modèle sont versionnés et immuables, parce qu'un score passé ne se défend qu'en rejouant le modèle qui l'a produit.

**Le cycle de vie se déduit des usages futurs, pas d'une règle générale.** Trois usages commandent ce qui doit survivre : le réentraînement suppose de conserver le code et les métadonnées, pas les tables ; l'audit suppose de pouvoir rejouer un score ancien, donc la version du modèle et le seuil alors en vigueur ; la contestation d'une orientation suppose que la probabilité, ses facteurs et la version aient été **enregistrés au moment de l'émission**, car ils ne se recalculent pas fidèlement après coup. Ce dernier point est le plus facile à négliger et le plus coûteux à rattraper.

Deux durées restent hors de portée du concepteur — la conservation des scores, selon qu'ils sont rattachés au dossier médical ou traités comme donnée de gestion, et la rétention des versions successives du modèle, qui en découle. Elles rejoignent la liste de 12.7, adressées au délégué à la protection des données et au médecin responsable de l'information médicale.


### J3-09 · Trois incidents valaient mieux qu'un argument : la suite de tests

Dernier écart de l'audit, et le seul qui demandait du code. Le critère C6 exige un processus de livraison et de déploiement continu ; le critère C9, un système d'évaluation automatisé du modèle intégré à cette chaîne. Le dépôt n'avait ni test, ni chaîne, ni évaluation automatique — et le journal, lui, racontait déjà trois incidents que des tests auraient arrêtés.

**L'argument n'est pas la bonne pratique, ce sont les trois occurrences.** En J3-02, la normalisation des sentinelles avait cessé de s'appliquer sous pandas 3, silencieusement, pendant deux semaines. En J3-03, l'explicateur d'arbres répondait sans erreur et faux. En J3-06, un commentaire affirmait des chiffres que la cellule juste au-dessus contredisait. Aucun de ces trois n'aurait été détecté par une relecture ; deux l'ont été par hasard.

**Dix-neuf contrôles, en trois familles.** Huit sur la préparation — chaque correction de la section 4 y est gardée par un test, à commencer par celui qui aurait vu J3-02. Six sur l'assemblage, dont l'étanchéité entre les deux étages : elle est annoncée comme une propriété du code depuis la section 1.5, elle est maintenant maintenue par un test plutôt que par la vigilance. Cinq sur le modèle, qui rejouent le protocole complet — validation croisée groupée, prédictions hors échantillon — et comparent la performance aux bornes documentées.

**Un test symétrique, qui est la vraie leçon de J3-02.** Vérifier qu'il ne reste plus de sentinelle après nettoyage ne prouve rien : une fonction devenue inerte passe ce contrôle sans broncher, puisqu'un jeu sans correction et un jeu parfaitement corrigé peuvent afficher le même résultat si l'on ne regarde que l'absence d'erreur. Le test jumeau vérifie donc que la normalisation **a effectivement converti quelque chose**. C'est lui qui aurait crié il y a deux semaines.

**Les bornes de l'évaluation sont fermées des deux côtés.** La PR-AUC doit tomber dans `[0,35 ; 0,50]` : trop basse, elle signale une régression ; **trop haute, une fuite**. Un test qui ne surveille que le plancher laisse passer le pire des défauts, celui qui embellit les métriques — et J1-07 rappelait déjà qu'une fuite n'abîme pas les chiffres, elle les flatte. Les bornes sont larges parce que la performance varie de 0,38 à 0,43 selon la partition : un test plus serré échouerait sur du bruit, serait désactivé au bout de trois fois, et un test qu'on finit par ignorer est pire qu'un test absent.

**Les tests ont eux-mêmes été testés.** Un contrôle qui ne peut pas échouer ne protège rien. Trois pannes ont été introduites volontairement pour vérifier que la suite les voit : normalisation rendue inerte, filtre des décès désactivé, conversion d'unités oubliée. Les trois sont détectées — la dernière fait ressortir un rapport de 9,2 entre les deux unités de créatinine et de 10,3 pour l'hémoglobine, soit exactement les facteurs de conversion qu'on avait oublié d'appliquer.

**La chaîne elle-même.** Un workflow se déclenche sur toute modification du dossier de soutenance, et sur elle seule : une chaîne qui se lance pour un fichier sans rapport finit par être ignorée. Elle sépare délibérément les contrôles du pipeline de l'évaluation du modèle, parce que ce ne sont pas les mêmes objets. Elle installe ses dépendances depuis un fichier dédié : le `requirements.txt` du notebook est un relevé complet de l'environnement de développement, avec Jupyter, Playwright et un paquet spécifique à Windows qui aurait fait échouer un agent Linux sur la première ligne.

**Ce que cela ne fait pas, et que la section 12.4 disait mal.** Ce paragraphe affirmait qu'aucune chaîne d'évaluation n'existait ; c'est devenu faux en une heure, et il est corrigé. La distinction à tenir est ailleurs : l'évaluation du **modèle** est désormais automatique, la surveillance de la **production** ne l'est pas et ne peut pas l'être ici, faute de données de production. Les six indicateurs de 12.4 restent une spécification, adossée maintenant à une chaîne réelle plutôt qu'à une intention.

## 28/09/2026 — Séance 7 : les onze tables dessinées

### J3-10 · Une deuxième clé fantôme, que l'audit comptait sans la nommer

En dessinant les onze tables et leurs jointures pour la présentation, un chiffre ne tombait pas juste : `objets_connectes` référence 184 patients, dont un qui n'existe pas. 125 mesures portent le `PatientID` `PAT-999999`, absent de `patients.csv`.

**Le contrôle l'avait vu, le texte ne l'a jamais dit.** La fonction d'intégrité référentielle vérifie cette clé depuis la première séance, et le tableau du §3.4 affiche bien 125 orphelins. Mais J1-04 ne relevait que les 73 biologies, et le commentaire du §3.4 a repris J1-04 plutôt que le tableau au-dessus de lui. Même famille que l'incident de J3-06 : un texte écrit à côté d'un calcul, qui s'en est détaché — cette fois par omission plutôt que par contradiction.

**C'est le même motif que les biologies.** Vérifié : les 73 biologies orphelines portent toutes `SEJ-999999`. Un identifiant au format valide, répété, sans correspondant — une valeur sentinelle qui tient lieu de clé inconnue. Elle passe tout contrôle de format ; seule la jointure la trahit.

**Sans effet sur le modèle, et c'est vérifié plutôt que supposé.** Les mesures connectées n'entrent que par deux chemins, et tous deux passent par les sorties : l'audit de couverture rattache chaque mesure à la dernière sortie du patient — `PAT-999999` n'en a aucune, ses mesures tombent hors de toute fenêtre ; le bloc `telesurveillance` part de la population et n'y trouve pas ce patient. Les couvertures de J1-06 (6, 12 et 28 patients sur 620) restent exactes.

**Deux chiffres publiés le comptaient pourtant.** Les « 184 patients équipés » de J1-06 sont 183. Et les 18,6 % de mesures dégradées de J1-04 incluent 31 mesures du patient fantôme (18 `CapteurDefaillant`, 13 `Gap`) : hors fantôme, 1 141 sur 6 172, soit 18,5 %. Aucune conclusion ne bouge.

**Ce qui change.** Le commentaire du §3.4 cite désormais les deux clés et leur valeur. Les 125 lignes ne sont pas retirées au nettoyage : aucune jointure ne les retient, les supprimer ne changerait aucun résultat. J1-04 et J1-06 restent tels qu'écrits — le journal trace ce que je savais à la date, la correction vit ici.


### J3-11 · Le régime d'assurance : permis, inutile, retiré

En préparant la slide sur l'éthique, j'ai relu table par table ce que le modèle reçoit. `RegimeAssurance` y entrait comme une catégorie, et ses valeurs comprennent l'**AME**, l'aide médicale d'État, accordée sous condition de ressources : 20 séjours de la population. C'est un marqueur individuel de précarité. Or le §11.1 affirmait « aucune variable individuelle de cette nature ».

**J'ai d'abord mal lu l'énoncé.** J'ai cru la variable interdite. Relu mot à mot, le sujet n'écarte que les « données socio-économiques individuelles **non collectées en milieu hospitalier** » ; le régime de couverture est collecté à l'admission, et le sujet le liste lui-même parmi les colonnes fournies, AME comprise. L'utiliser était permis. L'inexactitude n'était pas dans le modèle, elle était dans la phrase du §11.1, trop large.

**La décision s'est prise sur la mesure, pas sur la règle.** Avant de retirer quoi que ce soit, même protocole que le dossier :

| | PR-AUC | ROC-AUC |
|---|---:|---:|
| avec `demo_regime` | 0,428 | 0,774 |
| sans `demo_regime` | 0,435 | 0,778 |

À la permutation, la variable était **dernière des 133**, à −0,007 ± 0,015. Le gain apparent au retrait est dans le bruit — la fourchette de partition fait sept fois cet écart. Une donnée permise, sensible et inutile n'a pas à être traitée : c'est la minimisation au sens strict, et c'est un meilleur argument qu'une interdiction qui n'existait pas. Retrait décidé avec Nicolas.

**Tout a été réexécuté** : les trois notebooks, la recherche d'hyperparamètres imbriquée, et le contrôle Charlson sur quatre graines. Ce dernier n'avait jamais été versionné — fait à la main en J2-06 et en J3-02. Il vit désormais dans `robustesse_graines.py` ; avant de lui faire confiance, je lui ai fait reproduire l'ancien tableau en remettant la variable : les quatre lignes retombent au millième près. Ce qui a bougé :

| Grandeur | Avant (133 variables) | Après (132 variables) |
|---|---|---|
| PR-AUC du gradient boosting | 0,428 | 0,435 |
| Fourchette selon la partition | 0,38 – 0,43 | 0,39 – 0,44 |
| Au seuil 0,11 : signalés / rattrapées / alertes inutiles | 345 / 113 / 232 | 354 / 114 / 240 |
| Importance de l'âge | 0,175 | 0,173 |
| Variables sans contribution | 47 sur 133 | 69 sur 132 |
| Rappel désert médical / densité normale | 0,788 / 0,806 | 0,758 / 0,824 |
| Charlson sur quatre graines | +0,006, positif 3 fois sur 4 | −0,001, positif 1 fois sur 4 |
| Optuna contre réglage manuel | −0,002 | −0,007 |

**Trois de ces chiffres méritent d'être lus pour ce qu'ils disent de la mesure, pas du modèle.**

1. **Le décompte des variables nulles passe de 47 à 69** pour une seule variable retirée, elle-même sans effet. Ce nombre n'est pas une propriété du modèle, c'est un seuil posé sur un bruit : le dossier ne l'avance plus que comme « environ la moitié ».
2. **L'écart territorial passe de deux à six points.** Sur 33 réadmissions en désert médical, six points font deux patients. Le constat reste « pas de disparité démontrable », mais le §10.3 dit désormais pourquoi l'écart bouge — et pourquoi la surveillance (12.4) alerte sur son creusement dans la durée plutôt que sur sa valeur initiale.
3. **Charlson ne fait plus rien du tout.** L'extension était déjà écartée ; elle l'est maintenant sans même le « gain minuscule » qu'on lui concédait.

**Deux chiffres faux trouvés en chemin, sans rapport avec le retrait.** Le deck citait encore 0,307 pour le réseau de neurones, valeur d'avant le correctif J3-02. Et le temps de score unitaire, mesuré à 12,6 ms le 21/09, ressort aujourd'hui entre 45 et 56 ms — avec ou sans la variable, j'ai vérifié les deux : c'est l'état de la machine qui a changé, pas le modèle. Le §12.5 donne désormais des ordres de grandeur (quelques dizaines de millisecondes, moins d'une heure de processeur par an) plutôt qu'un chiffre qu'une exécution suivante contredirait. Même leçon qu'en J3-06, dans une autre variante : un chiffre écrit à la main à côté d'une mesure finit par la contredire. Le titre de la figure d'apport marginal, qui écrivait « moins de 4 % » en dur, est maintenant calculé sur les valeurs qu'il commente — il affiche 6 %.

Les entrées antérieures restent telles qu'écrites : elles disent ce qui était mesuré à leur date.

### J3-12 · L'inventaire des colonnes : huit écartées, dont trois sans raison écrite

En préparant la soutenance, une question simple : où le dossier dit-il ce que sont devenues les colonnes qu'il n'utilise pas, le nom et le prénom en tête ? La réponse tenait en deux phrases — une au §1.4, une ligne du tableau de minimisation au §11.1 — et n'allait pas plus loin. Aucun endroit ne listait les 80 colonnes livrées avec leur devenir.

**L'inventaire, dressé colonne par colonne, a trouvé trois exclusions que rien ne justifiait.** Le nom et le régime d'assurance étaient écartés par minimisation, les bornes de référence biologiques depuis J1-12. Mais `LibelleActe`, `LibelleMedicament` et `Posologie` n'étaient ni utilisées ni mentionnées nulle part. Les raisons ont été vérifiées avant d'être écrites :

- **Les deux libellés sont redondants** : un libellé par code, sans exception — 12 codes CCAM pour 12 libellés, 12 codes ATC pour 12 libellés, aucun manquant. Les codes portent déjà l'information, sous forme normalisée.
- **La posologie n'est pas du texte libre**, comme je l'ai d'abord supposé : sept valeurs normalisées. Mais elle ne dépend ni du médicament (test d'indépendance, p = 0,99) ni de la voie (p = 0,08), au point de la contredire : « 1 IV x3/j » apparaît 366 fois avec une voie orale, « 1 patch/72h » 345 fois. Elle a été générée indépendamment du reste de la prescription — même cas que le champ de compte-rendu écarté au §5.4.

**Deux constats de plus, en chemin.** La date de début de prescription est désynchronisée du séjour comme les autres dates filles : 0,5 % des prescriptions commencent dans la fenêtre de leur séjour, décalage médian −224 jours. J1-06 avait mesuré quatre tables filles et oublié celle-ci ; la colonne n'était de toute façon pas utilisée, elle est maintenant déclarée écartée. Et **la voie d'administration, elle, est utilisée**, sous la forme de la part de prescriptions intraveineuses, alors qu'elle ne dépend pas non plus du médicament (p = 0,29).

**La voie est mesurée, puis conservée.** Importance par permutation sur les cinq plis, même protocole que `permutation_scores` : +0,0028 de PR-AUC en moyenne (écart-type 0,0025), à comparer à 0,175 pour l'âge et à une fourchette de partition de 0,39 à 0,44. Elle ne pèse rien. Elle n'est pas retirée pour autant : contrairement au régime d'assurance (J3-11), ce n'est pas une donnée sensible, l'argument de minimisation ne s'applique pas ; et J3-11 a montré qu'un retrait sans effet oblige à tout réexécuter et déplace des chiffres partout, pour ne rien gagner. Elle figure à l'inventaire avec sa mesure.

**Le mécanisme.** L'inventaire est déclaré à la main dans `src/inventaire.py`, parce que la raison d'un devenir ne se déduit pas du code — un contrôle automatique aurait classé « jamais citée » une colonne écartée par décision comme une colonne oubliée. Le test `tests/test_inventaire.py` le confronte aux tables livrées : une colonne non déclarée, ou déclarée sans exister, fait échouer la chaîne d'intégration. C'est la leçon de J3-06 et J3-11 appliquée d'avance : un texte écrit à côté des données doit être vérifié contre elles, sans quoi il s'en détache. Le §11.1 affiche l'inventaire et sa synthèse ; le §12.8 compte désormais vingt-trois contrôles en quatre familles. Aucune variable du modèle ne change.
