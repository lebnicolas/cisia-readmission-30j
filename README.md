# Réadmission à 30 jours — certification CISIA

Nicolas Lebon · sujet d'examen **CISIA Santé** · soutenance du 13 octobre 2026.

La commande : prédire, à la sortie d'hospitalisation, le risque qu'un patient soit
réadmis dans les 30 jours, et proposer une solution intégrable au système
d'information hospitalier.

## Ce que contient ce dossier

| Pièce | Où | Rôle |
|---|---|---|
| Le cahier électronique | `notebook/soutenance_cisia.ipynb` | La production sur les données, sections 0 à 13, et le journal de bord, section 14 |
| Le support de présentation | `Nicolas LEBON - Soutenance CISIA — Réadmission à 30 jours.pdf` | La démarche en dix étapes, 15 pages |
| L'application | `api/` | Le score expliqué à la sortie et le suivi à domicile simulé, démontrés en direct |
| Le code du pipeline | `notebook/src/` | Chargement, audit, nettoyage, assemblage, modélisation, figures |
| Les tests | `tests/` et `.github/workflows/soutenance-cisia.yml` | 23 contrôles automatiques, rejoués par l'intégration continue |
| Les données | `CISIA - Sujet Santé/` | Les onze CSV synthétiques de l'épreuve et l'énoncé |

Le journal de bord existe aussi en texte dans `notebook/JOURNAL.md` : c'est la
source que la section 14 du cahier reprend à l'identique.

Le cahier se lit sans rien installer : toutes ses cellules sont exécutées, les
sorties et les figures sont enregistrées dedans.

---

## Démarrage rapide

Prérequis : **Python 3.12** (le projet est développé et figé sur 3.12.10). Les
commandes sont en PowerShell, depuis la racine de ce dossier ; sur un autre
système, remplacer `.venv\Scripts\` par `.venv/bin/`.

```powershell
# 1. Environnement
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r notebook\requirements.txt

# 2. Application
cd api
..\.venv\Scripts\python.exe -m uvicorn app:app --port 8077
```

Puis ouvrir **http://127.0.0.1:8077**.

Les artefacts du modèle sont fournis dans `api/artefacts/` : l'application démarre
sans réentraînement. Pour les reconstruire depuis les CSV :

```powershell
# depuis api/
..\.venv\Scripts\python.exe entrainer.py
```

`entrainer.py` réentraîne le modèle sur toute la population d'étude, écrit
`modele.joblib`, `arbre.joblib`, `arbre.json`, `donnees.parquet`,
`contexte.parquet` et `meta.json`, puis appelle `expliquer.py` qui calcule la
contribution de chaque variable au score de chaque séjour (`contributions.parquet`,
`contributions.json`, quelques minutes). La graine est fixée : le résultat est
reproductible à l'identique.

---

## Ce que montre l'application

**À la sortie de l'hôpital — le score.** Le modèle évalue chaque séjour au moment
de la sortie et rend un risque calibré, expliqué à deux niveaux. Dans la fiche, **la
contribution de chaque variable à ce score** (valeurs de Shapley, en points de
probabilité, additives : séjour moyen + contributions = score du patient) — c'est
ce qui rend une alerte contestable sur un fait précis. Et un arbre de substitution
navigable (`/arbre?sejour=...`) : on suit le chemin réel du patient, et on lit côte
à côte la moyenne de la feuille et le score du modèle.

**Au domicile — le suivi.** Un mur de supervision affiche la cohorte à code
couleur, avec un curseur de position dans le temps et un journal des alertes. La
fiche patient montre les courbes avec leurs seuils, les règles évaluées en direct
et la main courante des relevés. Les mesures des capteurs sont **simulées** pour la
démonstration : la télésurveillance fournie ne couvre que 6 patients sur les 170
qu'il faudrait pour entraîner cet étage (section 5.5 du cahier).

**Les deux se répondent.** Le seuil d'escalade n'est pas uniforme : il est
personnalisé par le score de sortie. Chez un patient à risque élevé, *un seul*
signe sérieux suffit à déclencher l'anticipation — deux restent requis sinon.

**Parcours de démonstration** (premier tirage de cohorte après démarrage de l'API,
déterministe) : ouvrir la fiche de **P-092** — 78 ans, insuffisance rénale chronique,
score de sortie 68 % — puis amener le mur à **J+7** : c'est sa première escalade, en
vigilance renforcée. Sur ce tirage, cinq patients escaladent en trente jours, trois
signalés à la sortie et deux non signalés : la simulation tire le scénario de chaque
patient avec une probabilité qui suit son score, elle ne le désigne pas. Le bouton
« ↻ cohorte » tire douze autres patients ; redémarrer l'API ramène au tirage de
référence.

---

## Les notebooks

L'environnement virtuel créé au démarrage rapide suffit : Jupyter est installé
par `notebook\requirements.txt`, rien n'est à ajouter.

| Composant | Version |
|---|---|
| Python | 3.12.10 |
| JupyterLab | 4.6.3 |
| ipykernel | 7.3.0 |
| IPython | 9.16.1 |

### Avec JupyterLab

```powershell
cd notebook
..\.venv\Scripts\python.exe -m jupyter lab
```

Le noyau à utiliser est **Python 3 (ipykernel)** : c'est celui du `.venv`.

- Seul **JupyterLab** est installé : `jupyter notebook` (l'interface classique)
  ne fonctionne pas.
- Lancer Jupyter **depuis `notebook/`**, pas depuis la racine. Les notebooks
  ajoutent le dossier courant au `sys.path` pour importer `src/`.

### Avec VS Code

Ouvrir un `.ipynb`, cliquer sur **Select Kernel** en haut à droite, puis
**Python Environments** et choisir `.venv (3.12.10)`. Par défaut, VS Code exécute
le notebook depuis son propre dossier : les imports de `src/` fonctionnent sans
réglage.

### Ordre d'exécution

| Notebook | Contenu | Lit | Écrit |
|---|---|---|---|
| `01_nettoyage_donnees.ipynb` | Chargement des 11 tables, audit qualité, construction de la table propre | les CSV de `CISIA - Sujet Santé/` | `outputs/donnees_propres.parquet`, `outputs/colonnes_categorielles.txt` |
| `02_entrainement_modele.ipynb` | Entraînement et évaluation au découpage 80/20 | les deux fichiers écrits par `01` | — |
| `soutenance_cisia.ipynb` | Le cahier complet : audit, méthode, comparatif des modèles, architecture, journal de bord | les CSV et `outputs/hyperparametres.json` | — |

`02` dépend de `01`, mais les fichiers de `outputs/` sont fournis : chaque
notebook peut s'exécuter seul. `soutenance_cisia.ipynb` est autonome : il repart
des CSV bruts et relit seulement le résultat figé de la recherche
d'hyperparamètres. Pour tout réexécuter d'un coup : **Run → Run All Cells**
(JupyterLab) ou **Run All** (VS Code).

Le code du pipeline vit dans `notebook/src/` — les notebooks l'appellent, ils ne
le dupliquent pas.

La recherche d'hyperparamètres (`src/tuning.py`) demande plusieurs minutes : elle
est exécutée à part et son résultat figé, que le cahier relit.

```powershell
# depuis notebook/
..\.venv\Scripts\python.exe recherche_hyperparametres.py 50   # → outputs/hyperparametres.json
```

`sync_journal.py` reporte `JOURNAL.md` dans la section 14 du cahier, et
`sync_sommaire.py` régénère son sommaire ; ils ne touchent qu'aux cellules
Markdown, aucune sortie de calcul n'est perdue.

---

## Tests et intégration continue

La suite couvre l'inventaire des colonnes, le nettoyage, l'assemblage et le modèle
(23 tests). pytest n'est pas dans `notebook\requirements.txt` : on l'ajoute avec les
dépendances minimales des tests, dont les versions sont les mêmes.

```powershell
# depuis la racine
.\.venv\Scripts\python.exe -m pip install -r tests\requirements-tests.txt   # une seule fois
.\.venv\Scripts\python.exe -m pytest tests -q
```

`tests\requirements-tests.txt` suffit aussi seul, sans Jupyter, sur une machine
Linux : c'est ce qu'utilise la chaîne d'intégration continue
(`.github/workflows/soutenance-cisia.yml`). À chaque modification, elle rejoue les
contrôles du pipeline, puis l'évaluation du modèle sur le protocole du dossier —
validation croisée groupée par patient — et refuse une performance sortie des
bornes documentées, dans un sens comme dans l'autre : trop basse signale une
régression, trop haute une fuite.
